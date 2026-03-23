import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split

from dataset import HeightRankDataset
from losses import frame_consistency_loss, pairwise_ranking_loss, recon_loss
from models import HeightRankModel

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    bbox_dir = cfg["data"]["bbox_dir"]
    rank_dir = cfg["data"]["rank_dir"]
    train_list = cfg["data"].get("train_list")
    test_list = cfg["data"].get("test_list")

    def build_full_split():
        full_ds = HeightRankDataset(
            video_dir=cfg["data"].get("video_dir"),
            bbox_dir=bbox_dir,
            rank_dir=rank_dir,
            T=cfg["model"]["T"],
            list_path=None,
            use_video=cfg["video"].get("enabled", False),
            video_frame_size=cfg["video"].get("frame_size", 112),
        )
        n_all = len(full_ds)
        if n_all == 0:
            raise RuntimeError(
                "No samples found. "
                f"bbox_dir={bbox_dir}, rank_dir={rank_dir}, "
                f"train_list={train_list}, test_list={test_list}"
            )
        n_val = max(1, int(n_all * cfg["train"]["val_ratio"]))
        n_train = max(1, n_all - n_val)
        if n_train + n_val > n_all:
            n_val = n_all - n_train
        return random_split(
            full_ds,
            lengths=[n_train, n_val],
            generator=torch.Generator().manual_seed(cfg["train"]["seed"]),
        )

    if train_list or test_list:
        train_ds = HeightRankDataset(
            video_dir=cfg["data"].get("video_dir"),
            bbox_dir=bbox_dir,
            rank_dir=rank_dir,
            T=cfg["model"]["T"],
            list_path=train_list,
            use_video=cfg["video"].get("enabled", False),
            video_frame_size=cfg["video"].get("frame_size", 112),
        )
        val_ds = HeightRankDataset(
            video_dir=cfg["data"].get("video_dir"),
            bbox_dir=bbox_dir,
            rank_dir=rank_dir,
            T=cfg["model"]["T"],
            list_path=test_list,
            use_video=cfg["video"].get("enabled", False),
            video_frame_size=cfg["video"].get("frame_size", 112),
        )
        # 列表存在但匹配失败时，自动回退为全量划分，避免直接中断。
        if len(train_ds) == 0 or len(val_ds) == 0:
            print(
                "[WARN] train/test list matched empty split. "
                f"train={len(train_ds)}, val={len(val_ds)}. "
                "Fallback to random split from full dataset."
            )
            train_ds, val_ds = build_full_split()
    else:
        train_ds, val_ds = build_full_split()

    if len(train_ds) == 0:
        raise RuntimeError("Training split is empty. Please check list files or data paths.")
    if len(val_ds) == 0:
        raise RuntimeError("Validation split is empty. Please check list files or data paths.")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        drop_last=False,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate_pairwise_acc(model: HeightRankModel, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_scores = []
    all_ranks = []
    for batch in dataloader:
        bboxes = batch["bboxes"].to(device)
        mask = batch["mask"].to(device)
        rank_label = batch["rank_label"].to(device)
        video_frames = batch.get("video_frames")
        if video_frames is not None:
            video_frames = video_frames.to(device)
        out = model(bboxes, mask, video_frames=video_frames)
        all_scores.append(out["s_v"].detach().cpu().numpy())
        all_ranks.append(rank_label.detach().cpu().numpy())
    scores = np.concatenate(all_scores, axis=0)
    ranks = np.concatenate(all_ranks, axis=0)
    return pairwise_accuracy(scores, ranks)


def train(cfg: Dict):
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)
    model_cfg = {**cfg["model"], **cfg["train"], "use_video": cfg["video"].get("enabled", False), "video_dim": cfg["video"].get("video_dim", cfg["model"]["d_model"])}
    model = HeightRankModel(model_cfg).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=cfg["train"]["log_dir"])
    best_acc = -1.0
    global_step = 0

    stage_a_epochs = cfg["train"]["stage_a_epochs"]
    total_epochs = cfg["train"]["num_epochs"]
    stage_b_lr_mult = float(cfg["train"].get("stage_b_lr_mult", 0.2))

    for epoch in range(total_epochs):
        if epoch == stage_a_epochs:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * stage_b_lr_mult
            print(f"[INFO] Enter stage B, lr scaled by {stage_b_lr_mult}.")

        model.train()
        epoch_losses = {"total": 0.0, "frame": 0.0, "recon": 0.0, "rank": 0.0}
        use_rank = epoch >= stage_a_epochs
        skipped_batches = 0

        for step, batch in enumerate(train_loader):
            bboxes = batch["bboxes"].to(device)
            mask = batch["mask"].to(device)
            rank_label = batch["rank_label"].to(device)
            video_frames = batch.get("video_frames")
            if video_frames is not None:
                video_frames = video_frames.to(device)
            # 二次防护：消除 NaN/Inf
            bboxes = torch.nan_to_num(bboxes, nan=0.0, posinf=1e6, neginf=-1e6)
            bboxes[..., 2:4] = bboxes[..., 2:4].clamp_min(1e-6)

            out = model(bboxes, mask, video_frames=video_frames)
            target_log_wh = torch.log(bboxes[..., 2:4].clamp_min(1e-6))

            loss_frame = frame_consistency_loss(out["z_t_H"], mask)
            loss_recon = recon_loss(out["recon_log_wh"], target_log_wh, mask)
            loss = (
                cfg["loss"]["w_frame"] * loss_frame
                + cfg["loss"]["w_recon"] * loss_recon
            )

            loss_rank = torch.zeros((), device=device)
            if use_rank:
                loss_rank = pairwise_ranking_loss(out["s_v"], rank_label, cfg["pair_sampling"])
                loss = loss + cfg["loss"]["w_rank"] * loss_rank

            if not torch.isfinite(loss):
                skipped_batches += 1
                if step % cfg["train"]["log_interval"] == 0:
                    print(
                        f"[WARN] Skip non-finite batch: epoch={epoch}, step={step}, "
                        f"frame={loss_frame.item()}, recon={loss_recon.item()}, rank={loss_rank.item()}"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_grad"])
            optimizer.step()

            epoch_losses["total"] += loss.item()
            epoch_losses["frame"] += loss_frame.item()
            epoch_losses["recon"] += loss_recon.item()
            epoch_losses["rank"] += loss_rank.item()
            global_step += 1

            if step % cfg["train"]["log_interval"] == 0:
                writer.add_scalar("train/loss_total", loss.item(), global_step)
                writer.add_scalar("train/loss_frame", loss_frame.item(), global_step)
                writer.add_scalar("train/loss_recon", loss_recon.item(), global_step)
                if use_rank:
                    writer.add_scalar("train/loss_rank", loss_rank.item(), global_step)

        num_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        val_acc = evaluate_pairwise_acc(model, val_loader, device)
        writer.add_scalar("val/pairwise_acc", val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": cfg,
            }
            torch.save(ckpt, os.path.join(cfg["train"]["save_dir"], "best.pt"))

        stage_name = "A(recon+frame)" if not use_rank else "B(+rank)"
        print(
            f"Epoch {epoch:03d} [{stage_name}] "
            f"loss={epoch_losses['total']:.4f} "
            f"frame={epoch_losses['frame']:.4f} "
            f"recon={epoch_losses['recon']:.4f} "
            f"rank={epoch_losses['rank']:.4f} "
            f"val_pair_acc={val_acc:.4f} "
            f"skipped={skipped_batches}"
        )

    writer.close()
    print(f"Training done. Best val pairwise accuracy: {best_acc:.4f}")
    print(f"Best checkpoint: {os.path.join(cfg['train']['save_dir'], 'best.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
