from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from heightnet.config import load_config
from heightnet.datasets import HeightDataset
from heightnet.losses import HeightNetLoss
from heightnet.metrics import rmse
from heightnet.model import HeightNetTiny
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir, set_seed


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_main_process() -> bool:
    return _get_rank() == 0


def _setup_distributed(cfg_device: str) -> tuple[bool, int, int, torch.device]:
    use_distributed = _is_distributed()
    if not use_distributed:
        device = torch.device(cfg_device if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA devices.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = _get_rank()
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    return True, rank, world_size, device


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _require_file(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found: {path}. "
            "Please generate manifests with tools/build_manifest.py or fix paths in config."
        )


def collate_fn(batch):
    out: Dict[str, torch.Tensor | list[str]] = {}
    for key in [
        "image",
        "image_raw",
        "height",
        "mask",
        "person_mask",
        "image_pair",
        "image_pair_raw",
        "person_mask_pair",
    ]:
        if key in batch[0]:
            out[key] = torch.stack([x[key] for x in batch], dim=0)

    for key in ["person_id", "camera_id", "sequence_id"]:
        if key in batch[0]:
            out[key] = [x[key] for x in batch]

    return out


def _denormalize_image_batch(image: torch.Tensor, normalize_rgb: bool) -> torch.Tensor:
    if not normalize_rgb:
        return torch.clamp(image, 0.0, 1.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    return torch.clamp(image * std + mean, 0.0, 1.0)


def _norm_map(x: torch.Tensor) -> torch.Tensor:
    x_min = torch.amin(x, dim=(2, 3), keepdim=True)
    x_max = torch.amax(x, dim=(2, 3), keepdim=True)
    return torch.clamp((x - x_min) / (x_max - x_min + 1e-6), 0.0, 1.0)


@torch.no_grad()
def evaluate(model, loader, device, distributed: bool = False):
    model.eval()
    meters = {"rmse": 0.0, "n": 0}

    for batch in loader:
        image = batch["image"].to(device)
        target = batch["height"].to(device)
        mask = batch["mask"].to(device)
        pred = model(image)

        meters["rmse"] += rmse(pred, target, mask).item()
        meters["n"] += 1

    if distributed:
        rmse_sum = torch.tensor([meters["rmse"]], dtype=torch.float64, device=device)
        n_sum = torch.tensor([meters["n"]], dtype=torch.float64, device=device)
        dist.all_reduce(rmse_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_sum, op=dist.ReduceOp.SUM)
        meters["rmse"] = (rmse_sum / torch.clamp(n_sum, min=1.0)).item()
        meters["n"] = int(n_sum.item())
    else:
        meters["rmse"] /= max(meters["n"], 1)
    return meters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    distributed = False
    writer = None
    try:
        cfg = load_config(args.config)
        if os.environ.get("HEIGHTNET_ANOMALY", "0") == "1":
            torch.autograd.set_detect_anomaly(True)
        distributed, rank, world_size, device = _setup_distributed(cfg.device)
        set_seed(cfg.seed + rank)
        _require_file(cfg.paths.train_manifest, "train_manifest")
        _require_file(cfg.paths.val_manifest, "val_manifest")
        if cfg.loss.pairwise_enabled:
            _require_file(cfg.loss.pairwise_json, "pairwise_json")

        if _is_main_process():
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

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        model = HeightNetTiny(base_channels=cfg.model.base_channels).to(device)
        if distributed:
            model = DDP(model, device_ids=[device.index], output_device=device.index)
        segmenter = None
        if cfg.runtime_seg.enabled:
            segmenter = PersonSegmenter(
                model_path=cfg.runtime_seg.model_path,
                conf=cfg.runtime_seg.conf,
                iou=cfg.runtime_seg.iou,
                imgsz=cfg.runtime_seg.imgsz,
                strict_native=cfg.runtime_seg.strict_native,
            )
        criterion = HeightNetLoss(
            silog_lambda=cfg.loss.silog_lambda,
            consistency_weight=cfg.loss.consistency_weight,
            pairwise_weight=cfg.loss.pairwise_weight,
            pairwise_json=cfg.loss.pairwise_json,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp and device.type == "cuda")
        if cfg.train.use_tensorboard and _is_main_process():
            tb_dir = os.path.join(cfg.paths.output_dir, "tb")
            ensure_dir(tb_dir)
            writer = SummaryWriter(log_dir=tb_dir)

        if _is_main_process():
            mode = "ddp" if distributed else "single"
            print(
                f"[train] mode={mode}, world_size={world_size}, "
                f"batch_per_gpu={cfg.train.batch_size}, effective_batch={cfg.train.batch_size * world_size}"
            )

        best_rmse = float("inf")
        global_step = 0
        start_epoch = 0

        if args.resume:
            _require_file(args.resume, "resume checkpoint")
            ckpt = torch.load(args.resume, map_location=device)
            model_state = ckpt.get("model")
            if model_state is None:
                raise RuntimeError(f"Invalid checkpoint (missing 'model'): {args.resume}")
            if distributed:
                model.module.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            best_rmse = float(ckpt.get("best_rmse", float("inf")))
            if _is_main_process():
                print(
                    f"[resume] loaded={args.resume}, start_epoch={start_epoch + 1}, "
                    f"global_step={global_step}, best_rmse={best_rmse:.4f}"
                )

        for epoch in range(start_epoch, cfg.train.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            model.train()
            running = {"total": 0.0, "silog": 0.0, "consistency": 0.0, "pairwise": 0.0, "n": 0}
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.train.epochs}",
                disable=not _is_main_process(),
            )
            enable_cons = cfg.data.use_pair_consistency and epoch >= cfg.loss.consistency_warmup_epochs
            enable_pairwise = cfg.loss.pairwise_enabled and epoch >= cfg.loss.pairwise_warmup_epochs
            vis_logged = False

            for batch in pbar:
                image = batch["image"].to(device)
                target = batch["height"].to(device)
                mask = batch["mask"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg.train.use_amp and device.type == "cuda"):
                    pred = model(image)
                    pred_pair = None
                    if segmenter is not None:
                        if "image_raw" not in batch:
                            raise RuntimeError("image_raw is required for runtime segmentation.")
                        person_mask = segmenter.infer_batch(batch["image_raw"], device)
                        person_mask_pair = None
                        if enable_cons:
                            if "image_pair_raw" not in batch:
                                raise RuntimeError("image_pair_raw is required when pair consistency is enabled.")
                            person_mask_pair = segmenter.infer_batch(batch["image_pair_raw"], device)
                    else:
                        person_mask = batch.get("person_mask")
                        person_mask_pair = batch.get("person_mask_pair")
                        if person_mask is not None:
                            person_mask = person_mask.to(device)
                        if person_mask_pair is not None:
                            person_mask_pair = person_mask_pair.to(device)
                        if person_mask is None:
                            raise RuntimeError("person_mask is required for training when runtime_seg is disabled.")

                    if enable_cons and "image_pair" in batch:
                        if person_mask_pair is None:
                            raise RuntimeError("person_mask_pair is required when pair consistency is enabled.")
                        image_pair = batch["image_pair"].to(device)
                        pred_pair = model(image_pair)

                    losses = criterion(
                        pred=pred,
                        target=target,
                        mask=mask,
                        pred_pair=pred_pair,
                        person_mask=person_mask,
                        person_mask_pair=person_mask_pair,
                        person_ids=batch.get("person_id"),
                        camera_ids=batch.get("camera_id"),
                        enable_pairwise=enable_pairwise,
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
                running["pairwise"] += losses["pairwise"].item()
                running["n"] += 1
                global_step += 1
                if _is_main_process():
                    pbar.set_postfix(
                        total=running["total"] / running["n"],
                        silog=running["silog"] / running["n"],
                        cons=running["consistency"] / running["n"],
                        rank=running["pairwise"] / running["n"],
                    )

                if writer is not None and global_step % max(cfg.train.log_interval, 1) == 0:
                    writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
                    writer.add_scalar("train/loss_silog", losses["silog"].item(), global_step)
                    writer.add_scalar("train/loss_consistency", losses["consistency"].item(), global_step)
                    writer.add_scalar("train/loss_pairwise", losses["pairwise"].item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                if writer is not None and (not vis_logged):
                    n_vis = min(cfg.train.tb_num_images, image.shape[0])
                    rgb = _denormalize_image_batch(image[:n_vis], cfg.data.normalize_rgb).detach().cpu()
                    gt = _norm_map(target[:n_vis]).repeat(1, 3, 1, 1).detach().cpu()
                    pd = _norm_map(pred[:n_vis]).repeat(1, 3, 1, 1).detach().cpu()
                    writer.add_images("train_vis/rgb", rgb, epoch + 1)
                    writer.add_images("train_vis/gt_height", gt, epoch + 1)
                    writer.add_images("train_vis/pred_height", pd, epoch + 1)
                    vis_logged = True

            metrics = evaluate(model, val_loader, device, distributed=distributed)
            if _is_main_process():
                print(f"[VAL] epoch={epoch + 1} rmse={metrics['rmse']:.4f}")
            if writer is not None:
                writer.add_scalar("val/rmse", metrics["rmse"], epoch + 1)
                writer.add_scalar("epoch/loss_total", running["total"] / max(running["n"], 1), epoch + 1)
                writer.add_scalar("epoch/loss_silog", running["silog"] / max(running["n"], 1), epoch + 1)
                writer.add_scalar("epoch/loss_consistency", running["consistency"] / max(running["n"], 1), epoch + 1)
                writer.add_scalar("epoch/loss_pairwise", running["pairwise"] / max(running["n"], 1), epoch + 1)

            if _is_main_process():
                ckpt_last = os.path.join(cfg.paths.output_dir, "checkpoint_last.pt")
                state_dict = model.module.state_dict() if distributed else model.state_dict()
                torch.save(
                    {
                        "model": state_dict,
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "best_rmse": best_rmse,
                        "cfg": cfg,
                    },
                    ckpt_last,
                )

                if metrics["rmse"] < best_rmse:
                    best_rmse = metrics["rmse"]
                    ckpt_best = os.path.join(cfg.paths.output_dir, "checkpoint_best.pt")
                    torch.save(
                        {
                            "model": state_dict,
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "best_rmse": best_rmse,
                            "cfg": cfg,
                        },
                        ckpt_best,
                    )

    finally:
        if writer is not None:
            writer.close()
        if distributed:
            _cleanup_distributed()


if __name__ == "__main__":
    main()
