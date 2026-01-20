import argparse
import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml


class PairwiseRanker(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int):
        super().__init__()
        # 编码器 f_theta，用于把输入样本映射到特征向量 z。
        self.encoder = encoder
        # 线性打分向量 w，比较 logit 为 w^T (z_i - z_j)。
        self.w = nn.Linear(embed_dim, 1, bias=False)

    def encode(self, x) -> torch.Tensor:
        # 支持任意形状输入，若 encoder 输出为高维特征图则拉平成向量。
        z = self.encoder(x)
        if z.dim() > 2:
            z = z.flatten(1)
        return z

    def forward(self, xi, xj) -> Tuple[torch.Tensor, torch.Tensor]:
        # 前向：得到两个样本的特征向量 z_i、z_j。
        zi = self.encode(xi)
        zj = self.encode(xj)
        # 比较 logit：Δ_ij = w^T (z_i - z_j)。
        delta = self.w(zi - zj).squeeze(-1)
        # 预测：\hat{y}_{ij} = 1[w^T z_i > w^T z_j]。
        y_hat = (self.w(zi) > self.w(zj)).long().squeeze(-1)
        return delta, y_hat

    @staticmethod
    def rank_loss(delta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 排序损失：log(1 + exp(-(2y-1) * Δ_ij))，用 softplus 保持数值稳定。
        y_sign = (2 * y.float() - 1.0)
        return F.softplus(-y_sign * delta).mean()


class RandomPairDataset(Dataset):
    def __init__(self, num_samples: int, input_shape: Tuple[int, ...], seed: int = 42):
        self.num_samples = num_samples
        self.input_shape = input_shape
        # 随机生成样本对与标签，仅用于冒烟测试训练流程。
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(num_samples * 2, *input_shape, generator=g)
        self.y = torch.randint(0, 2, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        xi = self.x[2 * idx]
        xj = self.x[2 * idx + 1]
        y = self.y[idx]
        return xi, xj, y


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        # 作为示例的简单编码器：两层 MLP。
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.net(x)


class FrameBboxEncoder(nn.Module):
    def __init__(self, image_encoder: nn.Module, embed_dim: int, bbox_dim: int = 6):
        super().__init__()
        self.image_encoder = image_encoder
        self.bbox_mlp = nn.Sequential(
            nn.Linear(bbox_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )
        self.proj = nn.Linear(embed_dim + 64, embed_dim)

    def forward(self, inputs):
        # inputs: (image, bbox_feat)
        image, bbox_feat = inputs
        img_feat = self.image_encoder(image)
        if img_feat.dim() > 2:
            img_feat = img_feat.flatten(1)
        bbox_feat = self.bbox_mlp(bbox_feat)
        fused = torch.cat([img_feat, bbox_feat], dim=1)
        return self.proj(fused)


class ResNetEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(backbone.fc.in_features, embed_dim)
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持输入为 (N, C, H, W) 或 (N, T, C, H, W)。
        if x.dim() == 5:
            n, t, c, h, w = x.shape
            x = x.reshape(n * t, c, h, w)
            feat = self.backbone(x)
            feat = feat.reshape(n, t, -1).mean(dim=1)
        else:
            feat = self.backbone(x)
        return self.proj(feat)


class ORViTEncoder(nn.Module):
    def __init__(
        self,
        cfg_path: str,
        frame_size: int,
        num_frames: int = 1,
        num_objects: int = 1,
        checkpoint_path: Optional[str] = None,
        orvit_root: Optional[str] = None,
    ):
        super().__init__()
        if not cfg_path:
            raise ValueError("orvit encoder requires --orvit-cfg")
        if frame_size <= 0:
            raise ValueError("frame_size must be positive for orvit encoder")

        if orvit_root is None:
            orvit_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ORViT"))
        if orvit_root not in sys.path:
            sys.path.insert(0, orvit_root)

        from slowfast.config.defaults import get_cfg
        from slowfast.models import build_model
        from slowfast.utils.checkpoint import load_checkpoint

        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.NUM_GPUS = 0
        cfg.MODEL.NUM_CLASSES = 0
        cfg.DATA.NUM_FRAMES = num_frames
        cfg.DATA.TRAIN_CROP_SIZE = frame_size
        cfg.DATA.TEST_CROP_SIZE = frame_size
        cfg.MF.PATCH_SIZE_TEMP = 1
        cfg.MF.TEMPORAL_RESOLUTION = num_frames
        cfg.MF.VIDEO_INPUT = True
        cfg.ORVIT.ENABLE = True
        cfg.ORVIT.O = num_objects

        self.cfg = cfg
        self.model = build_model(cfg)
        if checkpoint_path:
            load_checkpoint(checkpoint_path, self.model, data_parallel=False, inflation=False)

        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            raise RuntimeError("Failed to infer ORViT embed_dim from model")

    @staticmethod
    def _xyxy_to_cxcywh_norm(bbox_xyxy: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x1, y1, x2, y2 = bbox_xyxy.unbind(-1)
        cx = (x1 + x2) * 0.5 / max(1.0, float(width))
        cy = (y1 + y2) * 0.5 / max(1.0, float(height))
        w = (x2 - x1) / max(1.0, float(width))
        h = (y2 - y1) / max(1.0, float(height))
        return torch.stack([cx, cy, w, h], dim=-1).clamp(0.0, 1.0)

    def forward(self, inputs) -> torch.Tensor:
        image, bbox_xyxy = inputs
        if image.dim() == 4:
            image = image.unsqueeze(2)
        if bbox_xyxy.dim() == 2:
            bbox_xyxy = bbox_xyxy.unsqueeze(1)
        height = int(image.shape[-2])
        width = int(image.shape[-1])
        bbox_norm = self._xyxy_to_cxcywh_norm(bbox_xyxy, height, width)
        bbox_norm = bbox_norm.unsqueeze(1)
        metadata = {"orvit_bboxes": bbox_norm}
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features([image], metadata)
        return self.model([image], metadata)


def build_encoder(
    name: str,
    embed_dim: int,
    input_dim: Optional[int] = None,
    orvit_cfg: Optional[str] = None,
    orvit_checkpoint: Optional[str] = None,
    orvit_frame_size: Optional[int] = None,
    orvit_num_frames: int = 1,
    orvit_num_objects: int = 1,
) -> Tuple[nn.Module, int]:
    if name == "deepgaitv2":
        module = importlib.import_module("deepgaitv2")
        if hasattr(module, "DeepGaitV2"):
            return module.DeepGaitV2(embed_dim=embed_dim), embed_dim
        raise ImportError("deepgaitv2 module does not expose DeepGaitV2")
    if name in {"resnet18", "resnet34", "resnet50"}:
        # 通过 torchvision 构建 ResNet 编码器，输入为裁剪后的行人框帧/序列。
        tv = importlib.import_module("torchvision.models")
        weights = None
        if name == "resnet18":
            backbone = tv.resnet18(weights=weights)
        elif name == "resnet34":
            backbone = tv.resnet34(weights=weights)
        else:
            backbone = tv.resnet50(weights=weights)
        return ResNetEncoder(backbone=backbone, embed_dim=embed_dim), embed_dim
    if name == "orvit":
        if orvit_frame_size is None:
            raise ValueError("orvit_frame_size is required for orvit encoder")
        encoder = ORViTEncoder(
            cfg_path=orvit_cfg,
            frame_size=orvit_frame_size,
            num_frames=orvit_num_frames,
            num_objects=orvit_num_objects,
            checkpoint_path=orvit_checkpoint,
        )
        return encoder, encoder.embed_dim
    if name == "mlp":
        if input_dim is None:
            raise ValueError("input_dim is required for MLP encoder")
        return MLPEncoder(input_dim=input_dim, embed_dim=embed_dim), embed_dim
    raise ValueError(f"Unknown encoder: {name}")


class FrameBboxPairDataset(Dataset):
    """
    读取 frame + bbox 的成对数据。
    tracks_json: 记录列表，每条包含
      - person_id: 人物ID（需与 pairs_json 中一致）
      - video: 视频路径
      - frame: 帧号（从 1 开始）
      - bbox: [x, y, w, h] 或 [x1, y1, x2, y2]
    pairs_json: 列表，包含 {"id_i": "...", "id_j": "...", "y": 0/1}
    return_orvit_bbox: 若为 True，返回 (crop, bbox_xyxy_resized) 以供 ORViT 使用
    """

    def __init__(
        self,
        tracks_json: str,
        pairs_json: str,
        frame_size: Tuple[int, int],
        bbox_format: str,
        seed: int,
        split: str = "all",
        split_mode: str = "id",
        train_ratio: float = 0.8,
        leave_one_id: Optional[str] = None,
        context_ratio: float = 0.0,
        full_frame: bool = False,
        return_orvit_bbox: bool = False,
    ):
        with open(tracks_json, "r", encoding="utf-8") as f:
            track_records = json.load(f)
        with open(pairs_json, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

        self.frame_w, self.frame_h = frame_size
        self.bbox_format = bbox_format
        self.rng = random.Random(seed)
        self.context_ratio = context_ratio
        self.full_frame = full_frame
        self.return_orvit_bbox = return_orvit_bbox
        self.leave_one_id = leave_one_id
        self.split = split
        self.split_mode = split_mode

        self.person_to_samples = {}
        for rec in track_records:
            # 优先使用 global_id（与 pairwise 标签一致）
            pid = rec.get("global_id") or rec.get("person_id") or rec.get("id")
            if pid is None:
                continue
            self.person_to_samples.setdefault(pid, []).append(rec)

        if not self.person_to_samples:
            raise ValueError("tracks_json 中未找到有效的 person_id/global_id/id 字段。")

        valid_ids = set(self.person_to_samples.keys())

        if split_mode == "id" and split in {"train", "test"}:
            all_ids = sorted(valid_ids)
            self.rng.shuffle(all_ids)
            split_idx = int(len(all_ids) * train_ratio)
            if split == "train":
                valid_ids = set(all_ids[:split_idx])
            else:
                valid_ids = set(all_ids[split_idx:])

            # 只保留当前 split 的样本
            self.person_to_samples = {
                pid: recs for pid, recs in self.person_to_samples.items() if pid in valid_ids
            }
        elif split_mode == "leave_one" and split in {"train", "test"}:
            if not leave_one_id:
                raise ValueError("leave_one 模式需要提供 leave_one_id")
            if leave_one_id not in valid_ids:
                raise ValueError(f"leave_one_id 未在 tracks 中找到: {leave_one_id}")
            if split == "train":
                valid_ids = set(valid_ids) - {leave_one_id}
            else:
                valid_ids = {leave_one_id} | (set(valid_ids) - {leave_one_id})

            self.person_to_samples = {
                pid: recs for pid, recs in self.person_to_samples.items() if pid in valid_ids
            }

        self.valid_ids = set(valid_ids)

        # 过滤掉在轨迹中不存在的 pair，避免 KeyError。
        before = len(self.pairs)
        if split_mode == "leave_one" and split in {"train", "test"}:
            if split == "train":
                self.pairs = [
                    p for p in self.pairs
                    if p["id_i"] in valid_ids and p["id_j"] in valid_ids
                ]
            else:
                # test: 只保留与 leave_one_id 相关的 pair（1 * N）
                self.pairs = [
                    p for p in self.pairs
                    if (p["id_i"] == leave_one_id and p["id_j"] in valid_ids)
                    or (p["id_j"] == leave_one_id and p["id_i"] in valid_ids)
                ]
        else:
            self.pairs = [p for p in self.pairs if p["id_i"] in valid_ids and p["id_j"] in valid_ids]
        after = len(self.pairs)
        if after < before:
            print(f"[FrameBboxPairDataset] 过滤无效 pair: {before} -> {after}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_frame(self, video_path: str, frame_idx: int):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"读取帧失败: {video_path} frame={frame_idx}")
        return frame

    def _bbox_to_xyxy(self, bbox):
        if self.bbox_format == "xywh":
            x, y, w, h = bbox
            return x, y, x + w, y + h
        return bbox[0], bbox[1], bbox[2], bbox[3]

    def _bbox_to_resized_xyxy(self, bbox_xyxy, frame_size):
        frame_w, frame_h = frame_size
        x1, y1, x2, y2 = bbox_xyxy
        if self.context_ratio > 0:
            bw = x2 - x1
            bh = y2 - y1
            pad_w = bw * self.context_ratio
            pad_h = bh * self.context_ratio
            x1 -= pad_w
            x2 += pad_w
            y1 -= pad_h
            y2 += pad_h
        x1 = max(0.0, min(frame_w - 1.0, float(x1)))
        y1 = max(0.0, min(frame_h - 1.0, float(y1)))
        x2 = max(0.0, min(frame_w, float(x2)))
        y2 = max(0.0, min(frame_h, float(y2)))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0.0, 0.0, float(frame_w), float(frame_h)
        scale_x = self.frame_w / max(1.0, float(frame_w))
        scale_y = self.frame_h / max(1.0, float(frame_h))
        return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

    def _crop_and_resize(self, frame, bbox_xyxy):
        h, w = frame.shape[:2]
        if self.full_frame:
            resized = cv2.resize(frame, (self.frame_w, self.frame_h))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            resized = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            return resized, (w, h)
        x1, y1, x2, y2 = bbox_xyxy
        # 扩展 bbox 以引入背景上下文
        if self.context_ratio > 0:
            bw = x2 - x1
            bh = y2 - y1
            pad_w = bw * self.context_ratio
            pad_h = bh * self.context_ratio
            x1 -= pad_w
            x2 += pad_w
            y1 -= pad_h
            y2 += pad_h
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w, h
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.frame_w, self.frame_h))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        return crop, (w, h)

    def _bbox_feat(self, bbox_xyxy, frame_size):
        frame_w, frame_h = frame_size
        x1, y1, x2, y2 = bbox_xyxy
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0
        area = w * h
        aspect = w / h
        feat = [
            xc / frame_w,
            yc / frame_h,
            w / frame_w,
            h / frame_h,
            area / (frame_w * frame_h),
            aspect,
        ]
        return torch.tensor(feat, dtype=torch.float32)

    def _sample_for_person(self, person_id):
        samples = self.person_to_samples.get(person_id, [])
        if not samples:
            raise KeyError(f"未找到 person_id={person_id} 的样本。")
        return self.rng.choice(samples)

    def _load_sample(self, person_id):
        rec = self._sample_for_person(person_id)
        video_path = rec.get("video") or rec.get("source")
        if not video_path:
            raise KeyError("tracks_json 记录缺少 video/source 字段。")
        frame = self._load_frame(video_path, int(rec["frame"]))
        bbox_xyxy = self._bbox_to_xyxy(rec["bbox"])
        crop, frame_size = self._crop_and_resize(frame, bbox_xyxy)
        if self.return_orvit_bbox:
            if self.full_frame:
                bbox_resized = self._bbox_to_resized_xyxy(bbox_xyxy, frame_size)
            else:
                bbox_resized = (0.0, 0.0, float(self.frame_w), float(self.frame_h))
            bbox_resized = torch.tensor(bbox_resized, dtype=torch.float32)
            return crop, bbox_resized
        bbox_feat = self._bbox_feat(bbox_xyxy, frame_size)
        return crop, bbox_feat

    def sample_for_id(self, person_id):
        return self._load_sample(person_id)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        id_i = pair["id_i"]
        id_j = pair["id_j"]
        y = pair["y"]
        xi = self._load_sample(id_i)
        xj = self._load_sample(id_j)
        return xi, xj, torch.tensor(y, dtype=torch.long)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model: PairwiseRanker, loader: DataLoader, cfg: TrainConfig, optimizer) -> Tuple[float, float]:
    # 单个 epoch 的训练循环。
    model.train()

    def move_to_device(x, device):
        if isinstance(x, (tuple, list)):
            return tuple(move_to_device(v, device) for v in x)
        return x.to(device)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for xi, xj, y in loader:
        # 将输入与标签移动到设备上。
        xi = move_to_device(xi, cfg.device)
        xj = move_to_device(xj, cfg.device)
        y = y.to(cfg.device)

        # 计算对比 logit 与损失。
        delta, y_hat = model(xi, xj)
        loss = model.rank_loss(delta, y)

        # 标准优化步骤。
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_correct += (y_hat == y).sum().item()
        total_count += y.numel()

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


def evaluate(model: PairwiseRanker, loader: DataLoader, cfg: TrainConfig) -> Tuple[float, float]:
    model.eval()

    def move_to_device(x, device):
        if isinstance(x, (tuple, list)):
            return tuple(move_to_device(v, device) for v in x)
        return x.to(device)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for xi, xj, y in loader:
            xi = move_to_device(xi, cfg.device)
            xj = move_to_device(xj, cfg.device)
            y = y.to(cfg.device)
            delta, y_hat = model(xi, xj)
            loss = model.rank_loss(delta, y)
            total_loss += loss.item() * y.numel()
            total_correct += (y_hat == y).sum().item()
            total_count += y.numel()

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


def build_pair_label_map(pairs_path: str) -> dict:
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    label_map = {}
    for p in pairs:
        label_map[(p["id_i"], p["id_j"])] = int(p["y"])
    return label_map


def evaluate_leave_one_ranking(
    model: PairwiseRanker,
    train_dataset: FrameBboxPairDataset,
    test_dataset: FrameBboxPairDataset,
    label_map: dict,
    cfg: TrainConfig,
    samples_per_id: int,
    output_path: Optional[str] = None,
) -> Tuple[float, int]:
    test_id = test_dataset.leave_one_id
    if not test_id:
        return 0.0, 0

    def move_to_device(x, device):
        if isinstance(x, (tuple, list)):
            return tuple(move_to_device(v, device) for v in x)
        return x.to(device)
    def add_batch(x):
        if isinstance(x, (tuple, list)):
            return tuple(add_batch(v) for v in x)
        return x.unsqueeze(0)

    correct = 0
    total = 0
    test_vs_train = []
    for train_id in sorted(train_dataset.valid_ids):
        deltas = []
        for _ in range(samples_per_id):
            xi = test_dataset.sample_for_id(test_id)
            xj = train_dataset.sample_for_id(train_id)
            xi = add_batch(move_to_device(xi, cfg.device))
            xj = add_batch(move_to_device(xj, cfg.device))
            delta, _ = model(xi, xj)
            deltas.append(float(delta.item()))
        avg_delta = sum(deltas) / max(1, len(deltas))
        test_vs_train.append((train_id, avg_delta))

        if (test_id, train_id) in label_map:
            y = label_map[(test_id, train_id)]
            pred = 1 if avg_delta > 0 else 0
        elif (train_id, test_id) in label_map:
            y_rev = label_map[(train_id, test_id)]
            y = 1 - y_rev
            pred = 1 if avg_delta > 0 else 0
        else:
            continue
        if pred == y:
            correct += 1
        total += 1

    higher_count = sum(1 for _, d in test_vs_train if d < 0)
    rank = higher_count + 1
    print(f"[rank] test_id={test_id} rank={rank}/{len(test_vs_train)+1}")
    if output_path:
        ranking = sorted(test_vs_train, key=lambda x: x[1], reverse=True)
        payload = {
            "test_id": test_id,
            "ranking": [tid for tid, _ in ranking],
            "scores": {tid: float(score) for tid, score in ranking},
            "rank_of_test": rank,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
    return (correct / total) if total else 0.0, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise ranking training with DeepGaitV2 encoder")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--profile", type=str, default=None, help="YAML 中的配置分组名")
    parser.add_argument(
        "--encoder",
        type=str,
        default="deepgaitv2",
        choices=["deepgaitv2", "resnet18", "resnet34", "resnet50", "mlp", "orvit"],
    )
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-dim", type=int, default=512, help="Used for mlp encoder")
    parser.add_argument("--num-samples", type=int, default=1024, help="Used for random demo dataset")
    parser.add_argument("--input-shape", type=str, default="128,64", help="Used for random demo dataset")
    parser.add_argument("--tracks-json", type=str, default=None, help="Frame+bbox 轨迹文件 (JSON)")
    parser.add_argument("--pairs-json", type=str, default=None, help="Pairwise 标签文件 (JSON)")
    parser.add_argument("--frame-size", type=str, default="128,64", help="裁剪后帧大小 (W,H)")
    parser.add_argument("--bbox-format", type=str, default="xywh", choices=["xywh", "xyxy"])
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "test"])
    parser.add_argument("--split-mode", type=str, default="id", choices=["id", "leave_one"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--leave-one-id", type=str, default=None)
    parser.add_argument("--context-ratio", type=float, default=0.0, help="bbox 上下文扩展比例（例如 0.2）")
    parser.add_argument("--full-frame", action="store_true", help="使用整帧图像作为视觉输入")
    parser.add_argument("--orvit-cfg", type=str, default=None, help="ORViT 配置文件路径")
    parser.add_argument("--orvit-checkpoint", type=str, default=None, help="ORViT 权重路径")
    parser.add_argument("--orvit-num-frames", type=int, default=1, help="ORViT 使用的帧数")
    parser.add_argument("--orvit-num-objects", type=int, default=1, help="ORViT 每帧对象数")
    parser.add_argument("--val", action="store_true", help="启用训练过程中的验证集评估")
    parser.add_argument("--val-split", type=str, default="test", choices=["test", "train", "all"])
    parser.add_argument("--val-split-mode", type=str, default=None, choices=["id", "leave_one"])
    parser.add_argument("--val-ratio", type=float, default=0.8)
    parser.add_argument("--val-split-seed", type=int, default=42)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--val-tracks-json", type=str, default=None)
    parser.add_argument("--val-pairs-json", type=str, default=None)
    parser.add_argument("--val-frame-size", type=str, default=None)
    parser.add_argument("--val-bbox-format", type=str, default=None, choices=["xywh", "xyxy"])
    parser.add_argument("--val-leave-one-id", type=str, default=None)
    parser.add_argument("--val-context-ratio", type=float, default=None)
    parser.add_argument("--val-full-frame", action="store_true", help="验证集使用整帧图像")
    parser.add_argument("--rank-eval", action="store_true", help="leave_one 测试ID与所有训练ID比较并输出准确度")
    parser.add_argument("--rank-samples", type=int, default=3, help="每个ID对采样帧次数")
    parser.add_argument("--rank-output", type=str, default=None, help="保存测试集整体排名的 JSON 路径")
    return parser.parse_args()


def apply_config(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if args.profile:
            cfg = cfg.get(args.profile, {})
        args = apply_config(args, cfg)
    # 解析输入形状，用于生成随机样本或计算 MLP 输入维度。
    input_shape = tuple(int(x.strip()) for x in args.input_shape.split(",") if x.strip())
    input_dim = math.prod(input_shape)

    # 构建编码器与排序模型。
    frame_size = tuple(int(x.strip()) for x in args.frame_size.split(",") if x.strip())
    if args.encoder == "orvit":
        if len(frame_size) != 2 or frame_size[0] != frame_size[1]:
            raise ValueError("ORViT 仅支持正方形裁剪尺寸，例如 224,224")
        if not args.full_frame:
            raise ValueError("ORViT 需要 full_frame=True 以使用完整帧与 bbox")
        if args.orvit_num_frames != 1:
            raise ValueError("当前数据集为单帧输入，ORViT 仅支持 --orvit-num-frames=1")
    encoder, encoder_dim = build_encoder(
        args.encoder,
        embed_dim=args.embed_dim,
        input_dim=input_dim,
        orvit_cfg=args.orvit_cfg,
        orvit_checkpoint=args.orvit_checkpoint,
        orvit_frame_size=frame_size[0] if frame_size else None,
        orvit_num_frames=args.orvit_num_frames,
        orvit_num_objects=args.orvit_num_objects,
    )
    if args.tracks_json and args.pairs_json and args.encoder != "orvit":
        encoder = FrameBboxEncoder(encoder, embed_dim=encoder_dim)
    model = PairwiseRanker(encoder, embed_dim=encoder_dim).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.tracks_json and args.pairs_json:
        train_dataset = FrameBboxPairDataset(
            tracks_json=args.tracks_json,
            pairs_json=args.pairs_json,
            frame_size=frame_size,
            bbox_format=args.bbox_format,
            seed=args.split_seed,
            split=args.split,
            split_mode=args.split_mode,
            train_ratio=args.train_ratio,
            leave_one_id=args.leave_one_id,
            context_ratio=args.context_ratio,
            full_frame=args.full_frame,
            return_orvit_bbox=(args.encoder == "orvit"),
        )
    else:
        # 仅示例：使用随机数据作为训练集。
        train_dataset = RandomPairDataset(num_samples=args.num_samples, input_shape=input_shape)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 构建验证集（可选）
    val_loader = None
    val_dataset = None
    if args.val:
        val_tracks = args.val_tracks_json or args.tracks_json
        val_pairs = args.val_pairs_json or args.pairs_json
        if val_tracks and val_pairs:
            val_frame_size_str = args.val_frame_size or args.frame_size
            val_frame_size = tuple(int(x.strip()) for x in val_frame_size_str.split(",") if x.strip())
            if args.encoder == "orvit":
                if len(val_frame_size) != 2 or val_frame_size[0] != val_frame_size[1]:
                    raise ValueError("ORViT 验证集也需要正方形裁剪尺寸，例如 224,224")
            val_bbox_format = args.val_bbox_format or args.bbox_format
            val_context_ratio = args.val_context_ratio if args.val_context_ratio is not None else args.context_ratio
            val_full_frame = args.val_full_frame or args.full_frame
            if args.encoder == "orvit" and not val_full_frame:
                raise ValueError("ORViT 验证集需要 full_frame=True 以使用完整帧与 bbox")
            val_split_mode = args.val_split_mode or args.split_mode
            val_dataset = FrameBboxPairDataset(
                tracks_json=val_tracks,
                pairs_json=val_pairs,
                frame_size=val_frame_size,
                bbox_format=val_bbox_format,
                seed=args.val_split_seed,
                split=args.val_split,
                split_mode=val_split_mode,
                train_ratio=args.val_ratio,
                leave_one_id=args.val_leave_one_id or args.leave_one_id,
                context_ratio=val_context_ratio,
                full_frame=val_full_frame,
                return_orvit_bbox=(args.encoder == "orvit"),
            )
            val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    # 启动训练（带可选验证）。
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=model.w.weight.device.type)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    label_map = None
    if args.rank_eval and args.pairs_json:
        label_map = build_pair_label_map(args.pairs_json)
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, loader, cfg, optimizer)
        print(f"epoch={epoch+1} train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, cfg)
            print(f"epoch={epoch+1} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if (
                args.rank_eval
                and isinstance(train_dataset, FrameBboxPairDataset)
                and isinstance(val_dataset, FrameBboxPairDataset)
                and label_map is not None
            ):
                rank_acc, rank_total = evaluate_leave_one_ranking(
                    model,
                    train_dataset,
                    val_dataset,
                    label_map,
                    cfg,
                    samples_per_id=args.rank_samples,
                    output_path=args.rank_output,
                )
                print(f"epoch={epoch+1} rank_acc={rank_acc:.4f} rank_total={rank_total}")


if __name__ == "__main__":
    main()
