from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class RuntimeDepthConfig:
    enabled: bool
    depthanything_root: str
    encoder: str
    checkpoint: str
    input_size: int
    assume_inverse: bool = False


class RuntimeDepthEstimator:
    def __init__(
        self,
        depthanything_root: str,
        encoder: str,
        checkpoint: str,
        input_size: int = 518,
    ) -> None:
        if not depthanything_root or not os.path.exists(depthanything_root):
            raise FileNotFoundError(f"depthanything root not found: {depthanything_root}")
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError(f"depth checkpoint not found: {checkpoint}")

        self.depthanything_root = depthanything_root
        self.encoder = encoder
        self.checkpoint = checkpoint
        self.input_size = int(input_size)

        if depthanything_root not in sys.path:
            sys.path.insert(0, depthanything_root)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        if encoder not in model_configs:
            raise ValueError(f"unsupported encoder: {encoder}")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model = DepthAnythingV2(**model_configs[encoder])
        state_dict = _torch_load_compat(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)

    def to(self, device: torch.device) -> "RuntimeDepthEstimator":
        self.model = self.model.to(device).eval()
        return self

    @torch.no_grad()
    def infer_batch(self, images_raw: torch.Tensor) -> torch.Tensor:
        """
        images_raw: uint8 tensor [B,3,H,W], RGB.
        returns: depth tensor [B,1,H,W], float32.
        """
        if images_raw.ndim != 4:
            raise ValueError(f"images_raw should be [B,3,H,W], got {tuple(images_raw.shape)}")

        b, _, h, w = images_raw.shape
        outs = []
        for i in range(b):
            img = images_raw[i].permute(1, 2, 0).detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            depth = self.model.infer_image(img_bgr, self.input_size).astype(np.float32)
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            outs.append(torch.from_numpy(depth).unsqueeze(0))
        return torch.stack(outs, dim=0).to(images_raw.device)


def depth_to_height(
    depth: torch.Tensor,
    bg_depth: torch.Tensor,
    camera_height_m: torch.Tensor,
    eps: float = 1e-6,
    assume_inverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    depth/bg_depth: [B,1,H,W], camera_height_m: [B,1,1,1]
    returns (height, valid_mask)
    """
    depth_work = depth
    bg_work = bg_depth
    if assume_inverse:
        depth_work = torch.where(
            torch.isfinite(depth_work),
            1.0 / torch.clamp(depth_work, min=eps),
            depth_work,
        )
        bg_work = torch.where(
            torch.isfinite(bg_work),
            1.0 / torch.clamp(bg_work, min=eps),
            bg_work,
        )

    valid = torch.isfinite(depth_work) & torch.isfinite(bg_work) & (bg_work.abs() > eps)
    height = torch.zeros_like(depth)
    height[valid] = camera_height_m.expand_as(depth)[valid] * (bg_work[valid] - depth_work[valid]) / bg_work[valid]
    upper = camera_height_m.expand_as(depth) * 3.0
    height = torch.clamp(height, min=0.0)
    height = torch.minimum(height, upper)
    return height, valid.float()


def _torch_load_compat(path: str, map_location: str | torch.device, weights_only: bool) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
