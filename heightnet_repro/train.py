from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.losses import HeightNetLoss
from heightnet.metrics import rmse
from heightnet.model import HeightNetTiny
from heightnet.utils import ensure_dir, set_seed


def collate_fn(batch):
    out: Dict[str, torch.Tensor] = {}
    for key in ["image", "height", "mask", "image_pair"]:
        if key in batch[0]:
            out[key] = torch.stack([x[key] for x in batch], dim=0)

    if "matches" in batch[0]:
        max_n = max(x["matches"].shape[0] for x in batch)
        matches = []
        for x in batch:
            cur = x["matches"]
            if cur.shape[0] < max_n:
                pad = torch.zeros((max_n - cur.shape[0], 4), dtype=cur.dtype)
                cur = torch.cat([cur, pad], dim=0)
            matches.append(cur)
        out["matches"] = torch.stack(matches, dim=0)

    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    meters = {"rmse": 0.0, "n": 0}

    for batch in loader:
        image = batch["image"].to(device)
        target = batch["height"].to(device)
        mask = batch["mask"].to(device)
        pred = model(image)

        meters["rmse"] += rmse(pred, target, mask).item()
        meters["n"] += 1

    meters["rmse"] /= max(meters["n"], 1)
    return meters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg.paths.output_dir)

    train_ds = HeightDataset(
        cfg.paths.train_manifest,
        tuple(cfg.data.image_size),
        normalize_rgb=cfg.data.normalize_rgb,
        use_pair_consistency=cfg.data.use_pair_consistency,
        max_matches=cfg.data.max_matches,
    )
    val_ds = HeightDataset(
        cfg.paths.val_manifest,
        tuple(cfg.data.image_size),
        normalize_rgb=cfg.data.normalize_rgb,
        use_pair_consistency=False,
        max_matches=cfg.data.max_matches,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = HeightNetTiny(base_channels=cfg.model.base_channels).to(device)
    criterion = HeightNetLoss(cfg.loss.silog_lambda, cfg.loss.consistency_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp and device.type == "cuda")

    best_rmse = float("inf")
    for epoch in range(cfg.train.epochs):
        model.train()
        running = {"total": 0.0, "silog": 0.0, "consistency": 0.0, "n": 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        enable_cons = cfg.data.use_pair_consistency and epoch >= cfg.loss.consistency_warmup_epochs

        for batch in pbar:
            image = batch["image"].to(device)
            target = batch["height"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.train.use_amp and device.type == "cuda"):
                pred = model(image)
                pred_pair = None
                matches = None
                if enable_cons and "image_pair" in batch and "matches" in batch:
                    image_pair = batch["image_pair"].to(device)
                    pred_pair = model(image_pair)
                    matches = batch["matches"].to(device)

                losses = criterion(
                    pred=pred,
                    target=target,
                    mask=mask,
                    pred_pair=pred_pair,
                    matches=matches,
                    enable_consistency=enable_cons,
                )

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running["total"] += losses["total"].item()
            running["silog"] += losses["silog"].item()
            running["consistency"] += losses["consistency"].item()
            running["n"] += 1
            pbar.set_postfix(
                total=running["total"] / running["n"],
                silog=running["silog"] / running["n"],
                cons=running["consistency"] / running["n"],
            )

        metrics = evaluate(model, val_loader, device)
        print(f"[VAL] epoch={epoch + 1} rmse={metrics['rmse']:.4f}")

        ckpt_last = os.path.join(cfg.paths.output_dir, "checkpoint_last.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "cfg": cfg}, ckpt_last)

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            ckpt_best = os.path.join(cfg.paths.output_dir, "checkpoint_best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "cfg": cfg}, ckpt_best)


if __name__ == "__main__":
    main()
