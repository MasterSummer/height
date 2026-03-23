import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import HeightRankDataset
from models import HeightRankModel

try:
    from scipy.stats import kendalltau, spearmanr
except Exception:  # pragma: no cover
    kendalltau = None
    spearmanr = None


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pairwise_accuracy(scores: np.ndarray, ranks: np.ndarray) -> float:
    n = len(scores)
    correct, total = 0, 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ranks[i] > ranks[j]:
                total += 1
                if scores[i] > scores[j]:
                    correct += 1
    return float(correct / total) if total > 0 else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def spearman_corr(scores: np.ndarray, ranks: np.ndarray) -> float:
    if spearmanr is not None:
        return float(spearmanr(scores, ranks).correlation)
    return _pearson(_rankdata(scores), _rankdata(ranks))


def kendall_corr(scores: np.ndarray, ranks: np.ndarray) -> float:
    if kendalltau is not None:
        return float(kendalltau(scores, ranks).correlation)
    n = len(scores)
    concordant, discordant = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a = np.sign(scores[i] - scores[j])
            b = np.sign(ranks[i] - ranks[j])
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total > 0 else 0.0


def build_eval_loader(cfg: Dict) -> DataLoader:
    test_list = cfg["data"].get("test_list")
    ds = HeightRankDataset(
        video_dir=cfg["data"].get("video_dir"),
        bbox_dir=cfg["data"]["bbox_dir"],
        rank_dir=cfg["data"]["rank_dir"],
        T=cfg["model"]["T"],
        list_path=test_list,
        use_video=cfg["video"].get("enabled", False),
        video_frame_size=cfg["video"].get("frame_size", 112),
    )
    if len(ds) == 0:
        print(
            "[WARN] Eval split from test_list is empty. "
            "Fallback to full dataset for evaluation."
        )
        ds = HeightRankDataset(
            video_dir=cfg["data"].get("video_dir"),
            bbox_dir=cfg["data"]["bbox_dir"],
            rank_dir=cfg["data"]["rank_dir"],
            T=cfg["model"]["T"],
            list_path=None,
            use_video=cfg["video"].get("enabled", False),
            video_frame_size=cfg["video"].get("frame_size", 112),
        )
    if len(ds) == 0:
        raise RuntimeError(
            "Eval dataset is empty. "
            f"bbox_dir={cfg['data']['bbox_dir']}, rank_dir={cfg['data']['rank_dir']}, test_list={test_list}"
        )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        drop_last=False,
    )


@torch.no_grad()
def evaluate(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = build_eval_loader(cfg)

    model_cfg = {**cfg["model"], **cfg["train"], "use_video": cfg["video"].get("enabled", False), "video_dim": cfg["video"].get("video_dim", cfg["model"]["d_model"])}
    model = HeightRankModel(model_cfg).to(device)
    ckpt_path = os.path.join(cfg["train"]["save_dir"], "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    all_scores = []
    all_ranks = []
    stab_std = []
    stab_diff = []
    recon_errors = []

    for batch in dataloader:
        bboxes = batch["bboxes"].to(device)
        mask = batch["mask"].to(device)
        rank_label = batch["rank_label"].to(device)
        video_frames = batch.get("video_frames")
        if video_frames is not None:
            video_frames = video_frames.to(device)
        out = model(bboxes, mask, video_frames=video_frames)

        s_v = out["s_v"]
        s_t = out["s_t"]
        z_t_H = out["z_t_H"]
        z_v_H = out["z_v_H"]
        pred_log_wh = out["recon_log_wh"]
        target_log_wh = torch.log(bboxes[..., 2:4].clamp_min(1e-6))

        all_scores.append(s_v.detach().cpu().numpy())
        all_ranks.append(rank_label.detach().cpu().numpy())

        for bi in range(bboxes.shape[0]):
            valid = mask[bi] > 0
            if valid.sum() == 0:
                continue
            s_t_valid = s_t[bi][valid]
            z_t_valid = z_t_H[bi][valid]
            zv = z_v_H[bi]

            stab_std.append(float(torch.std(s_t_valid).item()))
            stab_diff.append(float(torch.norm(z_t_valid - zv.unsqueeze(0), dim=-1).mean().item()))
            recon_errors.append(float(torch.abs(pred_log_wh[bi][valid] - target_log_wh[bi][valid]).mean().item()))

    scores = np.concatenate(all_scores, axis=0)
    ranks = np.concatenate(all_ranks, axis=0)

    pair_acc = pairwise_accuracy(scores, ranks)
    spearman = spearman_corr(scores, ranks)
    kendall = kendall_corr(scores, ranks)
    frame_std = float(np.mean(stab_std)) if stab_std else 0.0
    frame_diff = float(np.mean(stab_diff)) if stab_diff else 0.0
    recon_err = float(np.mean(recon_errors)) if recon_errors else 0.0

    print(f"Pairwise Accuracy: {pair_acc:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(f"Kendall: {kendall:.4f}")
    print(f"Frame Stability (std s_t): {frame_std:.4f}")
    print(f"Frame Stability (mean ||z_t-z_v||): {frame_diff:.4f}")
    print(f"Reconstruction Error (L1 log wh): {recon_err:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)
