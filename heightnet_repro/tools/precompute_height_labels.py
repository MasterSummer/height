from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np


def _estimate_bg_depth_from_seg(depth: np.ndarray, seg: np.ndarray, ground_class_id: int) -> np.ndarray:
    h, w = depth.shape
    ground_mask = (seg == ground_class_id).astype(np.uint8)
    ground_depth = np.where(ground_mask > 0, depth, np.nan)
    col_bg = np.nanmedian(ground_depth, axis=0)
    global_bg = np.nanmedian(ground_depth)
    col_bg = np.where(np.isnan(col_bg), global_bg, col_bg)
    return np.tile(col_bg[None, :], (h, 1)).astype(np.float32)


def _estimate_bg_depth_without_seg(depth: np.ndarray, bottom_band_ratio: float) -> np.ndarray:
    h, w = depth.shape
    band_h = max(1, int(h * bottom_band_ratio))
    bottom = depth[h - band_h :, :]
    col_bg = np.nanmedian(bottom, axis=0)
    global_bg = np.nanmedian(bottom)
    col_bg = np.where(np.isnan(col_bg), global_bg, col_bg)
    return np.tile(col_bg[None, :], (h, 1)).astype(np.float32)


def compute_height_map(depth: np.ndarray, d_bg: np.ndarray, camera_height_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrected geometric target:
      h = C_h * (D_b - D_f) / D_b
    D_f is foreground/pixel depth, D_b is local background depth.
    """
    d_f = depth.astype(np.float32)
    d_b = d_bg.astype(np.float32)
    valid = np.isfinite(d_f) & np.isfinite(d_b) & (np.abs(d_b) > 1e-6)
    height = np.zeros_like(d_f, dtype=np.float32)
    height[valid] = camera_height_m * (d_b[valid] - d_f[valid]) / d_b[valid]
    height = np.clip(height, 0.0, camera_height_m * 3.0)
    return height, valid.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--camera-height-m", type=float, required=True)
    parser.add_argument("--ground-class-id", type=int, default=0)
    parser.add_argument("--bottom-band-ratio", type=float, default=0.15)
    args = parser.parse_args()

    split_root = os.path.join(args.data_root, args.split)
    depth_files = sorted(glob.glob(os.path.join(split_root, "depth", "*", "*.npy")))

    for depth_path in depth_files:
        rel = os.path.relpath(depth_path, os.path.join(split_root, "depth"))
        sequence_id = rel.split(os.sep)[0]
        stem = os.path.splitext(os.path.basename(depth_path))[0]
        depth = np.load(depth_path).astype(np.float32)
        seg_path = os.path.join(split_root, "seg", sequence_id, f"{stem}.png")

        if os.path.exists(seg_path):
            seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            if seg is not None:
                d_bg = _estimate_bg_depth_from_seg(depth, seg, args.ground_class_id)
            else:
                d_bg = _estimate_bg_depth_without_seg(depth, args.bottom_band_ratio)
        else:
            d_bg = _estimate_bg_depth_without_seg(depth, args.bottom_band_ratio)

        height, valid = compute_height_map(depth, d_bg, args.camera_height_m)

        out_h = os.path.join(split_root, "height", sequence_id)
        out_m = os.path.join(split_root, "valid_mask", sequence_id)
        os.makedirs(out_h, exist_ok=True)
        os.makedirs(out_m, exist_ok=True)
        np.save(os.path.join(out_h, f"{stem}.npy"), height)
        np.save(os.path.join(out_m, f"{stem}.npy"), valid)

    print("Height labels generated.")


if __name__ == "__main__":
    main()
