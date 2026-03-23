import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pandas.errors import EmptyDataError

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


CAMERA_PATTERN = re.compile(r"(300|350|400)cm_(inside|outside|slantside)")


def _load_rank_maps(rank_dir: str) -> Dict[str, Dict[str, float]]:
    rank_maps: Dict[str, Dict[str, float]] = {}
    if not os.path.isdir(rank_dir):
        return rank_maps
    for name in os.listdir(rank_dir):
        if not name.endswith("_rank.json"):
            continue
        path = os.path.join(rank_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranking = data.get("ranking", [])
        if not ranking:
            continue
        camera = data.get("camera")
        if not camera:
            camera = name.replace("camera_", "").replace("_rank.json", "")
        n = len(ranking)
        # label 越大越高：ranking[0] 最高，映射成 n
        rank_maps[camera] = {pid: float(n - idx) for idx, pid in enumerate(ranking)}
    return rank_maps


def _load_video_filter(list_path: Optional[str]) -> Optional[Set[str]]:
    if not list_path:
        return None
    if not os.path.isfile(list_path):
        return None
    with open(list_path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    return ids if ids else None


def _token_variants(token: str) -> Set[str]:
    token = token.strip()
    if not token:
        return set()
    token = token.replace("\\", "/")
    base = os.path.basename(token)
    stem = os.path.splitext(base)[0]
    parent = os.path.basename(os.path.dirname(token))
    variants = {token, token.lower(), base, base.lower(), stem, stem.lower()}
    if parent:
        variants.add(parent)
        variants.add(parent.lower())
        variants.add(f"{parent}/{base}")
        variants.add(f"{parent}/{stem}")
    return variants


def _matches_filter(filter_ids: Optional[Set[str]], person_id: str, stem: str) -> bool:
    if not filter_ids:
        return True
    keys = {
        person_id,
        person_id.lower(),
        stem,
        stem.lower(),
        f"{person_id}/{stem}",
        f"{person_id}/{stem}.mp4",
        f"tmp_test_videos/{person_id}/{stem}.mp4",
    }
    for token in filter_ids:
        if not token:
            continue
        vars_token = _token_variants(token)
        if keys.intersection(vars_token):
            return True
    return False


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def _build_video_index(video_dir: Optional[str]) -> Dict[str, Dict[str, str]]:
    """Index video_dir/<person>/<video>.<ext> -> {person: {video_stem: video_path}}."""
    index: Dict[str, Dict[str, str]] = {}
    if not video_dir or not os.path.isdir(video_dir):
        return index
    for person in os.listdir(video_dir):
        pdir = os.path.join(video_dir, person)
        if not os.path.isdir(pdir):
            continue
        stem_map: Dict[str, str] = {}
        for fn in os.listdir(pdir):
            fp = os.path.join(pdir, fn)
            if os.path.isfile(fp) and os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                stem_map[os.path.splitext(fn)[0]] = fp
        if stem_map:
            index[person] = stem_map
    return index


def _read_bboxes(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        arr = np.asarray([x.get("bbox", []) for x in frames], dtype=np.float32)
        frame_ids = np.asarray(
            [
                x.get("frame_idx", x.get("frame_id", x.get("frame", idx + 1)))
                for idx, x in enumerate(frames)
            ],
            dtype=np.int64,
        )
    else:
        # txt/csv: 支持 MOT 风格 [frame,id,x,y,w,h,...] 或直接 [x,y,w,h]
        sep = "," if ext in (".txt", ".csv") else None
        try:
            df = pd.read_csv(path, header=None, sep=sep, engine="python")
        except EmptyDataError:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        arr_all = df.values.astype(np.float32)
        if arr_all.shape[1] >= 6:
            arr = arr_all[:, 2:6]
            frame_ids = arr_all[:, 0].astype(np.int64)
        elif arr_all.shape[1] >= 4:
            arr = arr_all[:, :4]
            frame_ids = np.arange(1, arr.shape[0] + 1, dtype=np.int64)
        else:
            raise ValueError(f"bbox file columns < 4: {path}")
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"invalid bbox shape in {path}: {arr.shape}")
    # 清理异常值，避免后续出现 NaN/Inf loss
    arr = arr.astype(np.float32, copy=False)
    finite_mask = np.isfinite(arr).all(axis=1)
    arr = arr[finite_mask]
    frame_ids = frame_ids[finite_mask]
    if arr.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    # w/h 必须为正；过大值裁剪，避免数值爆炸
    wh_valid = (arr[:, 2] > 0.0) & (arr[:, 3] > 0.0)
    arr = arr[wh_valid]
    frame_ids = frame_ids[wh_valid]
    if arr.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    arr = np.clip(arr, a_min=-1e6, a_max=1e6)
    return arr, frame_ids


def _sample_or_pad(bboxes: np.ndarray, frame_ids: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(bboxes.shape[0])
    if n >= T:
        indices = np.linspace(0, n - 1, T).round().astype(np.int64)
        out = bboxes[indices]
        out_frame_ids = frame_ids[indices]
        mask = np.ones(T, dtype=np.float32)
    else:
        pad = np.zeros((T - n, 4), dtype=np.float32)
        out = np.concatenate([bboxes, pad], axis=0)
        frame_pad = np.zeros((T - n,), dtype=np.int64)
        out_frame_ids = np.concatenate([frame_ids, frame_pad], axis=0)
        mask = np.zeros(T, dtype=np.float32)
        mask[:n] = 1.0
    out[:, 2:4] = np.clip(out[:, 2:4], a_min=1e-6, a_max=None)
    return out, out_frame_ids, mask


def _read_video_frames(video_path: str, frame_ids: np.ndarray, mask: np.ndarray, frame_size: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video loading but is not available.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = np.zeros((len(frame_ids), 3, frame_size, frame_size), dtype=np.float32)
    try:
        for idx, (frame_id, is_valid) in enumerate(zip(frame_ids.tolist(), mask.tolist())):
            if not is_valid:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_id) - 1))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
            frames[idx] = np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))
    finally:
        cap.release()
    return frames


class HeightRankDataset(Dataset):
    def __init__(
        self,
        video_dir: Optional[str],
        bbox_dir: str,
        rank_dir: str,
        T: int = 32,
        list_path: Optional[str] = None,
        use_video: bool = False,
        video_frame_size: int = 112,
    ) -> None:
        super().__init__()
        self.T = T
        self.bbox_dir = bbox_dir
        self.rank_dir = rank_dir
        self.video_dir = video_dir
        self.use_video = use_video
        self.video_frame_size = video_frame_size
        self._rank_maps = _load_rank_maps(rank_dir)
        self._video_filter = _load_video_filter(list_path)
        self._video_index = _build_video_index(video_dir)
        if self.use_video and not self._video_index:
            raise RuntimeError(
                f"Video branch enabled but no videos indexed from video_dir={video_dir}"
            )
        self.samples = self._discover_samples()

    def _discover_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        if not os.path.isdir(self.bbox_dir):
            return samples

        for root, _, files in os.walk(self.bbox_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in (".txt", ".csv", ".json"):
                    continue

                bbox_path = os.path.join(root, name)
                if os.path.getsize(bbox_path) == 0:
                    continue
                stem = os.path.splitext(name)[0]
                person_id = os.path.basename(root) if root != self.bbox_dir else stem

                if not _matches_filter(self._video_filter, person_id, stem):
                    continue

                # 若提供了 video_dir，则优先要求轨迹文件与视频名对齐
                # 兼容：当 person 目录不存在时不过滤，避免误删可用样本。
                video_path = None
                if self._video_index and person_id in self._video_index:
                    video_path = self._video_index[person_id].get(stem)
                    if video_path is None:
                        continue
                elif self.use_video:
                    continue

                # 1) 优先单样本 rank json: rank_dir/{sample_id}.json
                rank_label = None
                single_rank_path = os.path.join(self.rank_dir, f"{stem}.json")
                if os.path.isfile(single_rank_path):
                    with open(single_rank_path, "r", encoding="utf-8") as f:
                        rank_data = json.load(f)
                    if "height_rank" in rank_data:
                        rank_label = float(rank_data["height_rank"])

                # 2) 从 camera 排名文件推导 rank_label
                if rank_label is None:
                    text = f"{stem}_{name}"
                    m = CAMERA_PATTERN.search(text)
                    if m:
                        camera = f"{m.group(1)}cm_{m.group(2)}"
                        rank_map = self._rank_maps.get(camera, {})
                        if person_id in rank_map:
                            rank_label = rank_map[person_id]

                if rank_label is None:
                    continue

                samples.append(
                    {
                        "sample_id": stem,
                        "person_id": person_id,
                        "bbox_path": bbox_path,
                        "rank_label": rank_label,
                        "video_path": video_path,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 容错：遇到空文件/坏文件时，向后尝试其他样本，避免训练中断。
        n = len(self.samples)
        for offset in range(n):
            i = (idx + offset) % n
            sample = self.samples[i]
            try:
                bboxes, frame_ids = _read_bboxes(sample["bbox_path"])
            except Exception:
                continue
            if bboxes.shape[0] == 0:
                continue
            bboxes, frame_ids, mask = _sample_or_pad(bboxes, frame_ids, self.T)
            if mask.sum() <= 0:
                continue
            item = {
                "sample_id": sample["sample_id"],
                "person_id": sample["person_id"],
                "bboxes": torch.tensor(bboxes, dtype=torch.float32),
                "mask": torch.tensor(mask, dtype=torch.float32),
                "rank_label": torch.tensor(sample["rank_label"], dtype=torch.float32),
            }
            if self.use_video:
                if not sample.get("video_path"):
                    continue
                try:
                    frames = _read_video_frames(
                        video_path=sample["video_path"],
                        frame_ids=frame_ids,
                        mask=mask,
                        frame_size=self.video_frame_size,
                    )
                except Exception:
                    continue
                item["video_frames"] = torch.tensor(frames, dtype=torch.float32)
            return item
        raise RuntimeError("All candidate samples are empty or unreadable.")


def get_dataloader(
    video_dir: Optional[str],
    bbox_dir: str,
    rank_dir: str,
    batch_size: int,
    T: int,
    shuffle: bool = True,
    list_path: Optional[str] = None,
    num_workers: int = 0,
    use_video: bool = False,
    video_frame_size: int = 112,
) -> DataLoader:
    dataset = HeightRankDataset(
        video_dir=video_dir,
        bbox_dir=bbox_dir,
        rank_dir=rank_dir,
        T=T,
        list_path=list_path,
        use_video=use_video,
        video_frame_size=video_frame_size,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No valid samples found for dataset. "
            f"video_dir={video_dir}, bbox_dir={bbox_dir}, rank_dir={rank_dir}, list_path={list_path}"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
