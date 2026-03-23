from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.metrics import pairwise_accuracy_from_ranked_lists, rmse
from heightnet.model import HeightNetTiny
from heightnet.utils import ensure_dir


def _load_rank_labels(rank_dir: str) -> dict:
    rank_map = {}
    for name in os.listdir(rank_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(rank_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        cam = obj.get("camera")
        ranking = obj.get("ranking", [])
        if cam and isinstance(ranking, list):
            rank_map[cam] = ranking
    return rank_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rank-dir", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ds = HeightDataset(
        cfg.paths.test_manifest,
        tuple(cfg.data.image_size),
        normalize_rgb=cfg.data.normalize_rgb,
        use_pair_consistency=False,
        max_matches=cfg.data.max_matches,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers)

    model = HeightNetTiny(base_channels=cfg.model.base_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meters = {"rmse": 0.0, "n": 0}
    person_scores_by_camera = {}
    vis_dir = os.path.join(cfg.paths.output_dir, "vis")
    if cfg.eval.save_visualizations:
        ensure_dir(vis_dir)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            image = batch["image"].to(device)
            target = batch["height"].to(device)
            mask = batch["mask"].to(device)

            pred = model(image)
            meters["rmse"] += rmse(pred, target, mask).item()
            meters["n"] += 1

            if args.rank_dir:
                cam = batch.get("camera_id", ["unknown_camera"])[0]
                pid = batch.get("person_id", ["unknown_person"])[0]
                score = (pred * mask).sum() / torch.clamp(mask.sum(), min=1.0)
                person_scores_by_camera.setdefault(cam, {}).setdefault(pid, []).append(float(score.item()))

            if cfg.eval.save_visualizations and i < cfg.eval.vis_limit:
                rgb = image[0].detach().cpu().numpy().transpose(1, 2, 0)
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
                gt = target[0, 0].detach().cpu().numpy()
                pd = pred[0, 0].detach().cpu().numpy()

                fig, ax = plt.subplots(1, 3, figsize=(14, 4))
                ax[0].imshow(rgb)
                ax[0].set_title("RGB")
                ax[1].imshow(gt, cmap="viridis")
                ax[1].set_title("GT Height")
                ax[2].imshow(pd, cmap="viridis")
                ax[2].set_title("Pred Height")
                for a in ax:
                    a.axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(vis_dir, f"{i:05d}.png"), dpi=120)
                plt.close(fig)

    meters["rmse"] /= max(meters["n"], 1)

    print(f"[TEST] rmse={meters['rmse']:.4f}")

    if args.rank_dir:
        rank_map = _load_rank_labels(args.rank_dir)
        accs = []
        for cam, person_scores in person_scores_by_camera.items():
            if cam not in rank_map:
                continue
            avg_scores = {pid: sum(v) / max(len(v), 1) for pid, v in person_scores.items()}
            acc = pairwise_accuracy_from_ranked_lists(avg_scores, rank_map[cam])
            accs.append(acc)
            print(f"[PAIRWISE] camera={cam} acc={acc:.4f} n_person={len(avg_scores)}")

        if accs:
            print(f"[PAIRWISE] mean_acc={sum(accs)/len(accs):.4f}")
        else:
            print("[PAIRWISE] no matched camera/person ids found for evaluation")


if __name__ == "__main__":
    main()
