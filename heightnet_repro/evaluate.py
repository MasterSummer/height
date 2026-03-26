from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.metrics import pairwise_accuracy_from_ranked_lists, rmse
from heightnet.model import HeightNetTiny
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir


def _require_file(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found: {path}. "
            "Please generate manifests with tools/build_manifest.py or fix paths in config."
        )


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
    parser.add_argument(
        "--pairwise-out",
        type=str,
        default="",
        help="Export predicted camera-wise pairwise comparison JSON. "
        "Default: <output_dir>/pairwise_pred.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    _require_file(cfg.paths.test_manifest, "test_manifest")
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
    segmenter = None
    if cfg.runtime_seg.enabled:
        segmenter = PersonSegmenter(
            model_path=cfg.runtime_seg.model_path,
            conf=cfg.runtime_seg.conf,
            iou=cfg.runtime_seg.iou,
            imgsz=cfg.runtime_seg.imgsz,
            strict_native=cfg.runtime_seg.strict_native,
        )

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
            if segmenter is not None:
                if "image_raw" not in batch:
                    raise RuntimeError("image_raw is required for runtime segmentation in evaluation.")
                person_mask = segmenter.infer_batch(batch["image_raw"], device)
            else:
                person_mask = batch.get("person_mask")
                if person_mask is None:
                    raise RuntimeError("person_mask is required for evaluation when runtime_seg is disabled.")
                person_mask = person_mask.to(device)

            pred = model(image)
            meters["rmse"] += rmse(pred, target, mask).item()
            meters["n"] += 1

            if args.rank_dir:
                cam = batch.get("camera_id", ["unknown_camera"])[0]
                pid = batch.get("person_id", ["unknown_person"])[0]
                valid = person_mask > 0.5
                if valid.any():
                    score = torch.max(pred[valid])
                else:
                    raise RuntimeError(f"empty person_mask region for camera={cam}, person={pid}")
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

    # Always export predicted person-level ranking/pairwise results by camera.
    pairwise_payload = {"cameras": {}}
    for cam, person_scores in person_scores_by_camera.items():
        avg_scores = {pid: sum(v) / max(len(v), 1) for pid, v in person_scores.items()}
        ranking = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        pairwise = []
        for (pid_i, score_i), (pid_j, score_j) in combinations(ranking, 2):
            pairwise.append(
                {
                    "id_i": pid_i,
                    "id_j": pid_j,
                    "score_i": float(score_i),
                    "score_j": float(score_j),
                    "pred": 1 if score_i > score_j else 0,  # 1 means i taller than j
                }
            )
        pairwise_payload["cameras"][cam] = {
            "n_person": len(avg_scores),
            "ranking": [{"person_id": pid, "score": float(s)} for pid, s in ranking],
            "pairwise": pairwise,
        }
        print(f"[PRED] camera={cam} n_person={len(avg_scores)} n_pairwise={len(pairwise)}")

    pairwise_out = args.pairwise_out or os.path.join(cfg.paths.output_dir, "pairwise_pred.json")
    ensure_dir(os.path.dirname(pairwise_out))
    with open(pairwise_out, "w", encoding="utf-8") as f:
        json.dump(pairwise_payload, f, ensure_ascii=False, indent=2)
    print(f"[PRED] pairwise exported: {pairwise_out}")

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
