from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch


def select_largest_person_mask(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    pm = np.zeros((h, w), dtype=np.uint8)
    if arr.size == 0:
        return pm

    best_area = -1
    for m in arr:
        mu8 = (m > 0.5).astype(np.uint8)
        area = int(mu8.sum())
        if area > best_area:
            best_area = area
            pm = mu8
    return pm


class PersonSegmenter:
    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        strict_native: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("ultralytics is required for runtime segmentation") from e

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"segmentation model not found: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.strict_native = strict_native

    @torch.no_grad()
    def infer_batch(self, images_raw: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        images_raw: uint8 tensor, shape [B, 3, H, W], RGB.
        returns: float tensor [B, 1, H, W], {0,1}
        """
        if images_raw.ndim != 4:
            raise ValueError(f"images_raw should be [B,3,H,W], got {tuple(images_raw.shape)}")

        b, _, h, w = images_raw.shape
        imgs = [images_raw[i].permute(1, 2, 0).cpu().numpy() for i in range(b)]
        if device.type == "cuda":
            infer_device = str(device.index if device.index is not None else 0)
        else:
            infer_device = device.type

        results = self.model.predict(
            source=imgs,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=infer_device,
            classes=[0],  # person
            retina_masks=True,
            verbose=False,
        )

        masks = []
        for i in range(b):
            r = results[i]
            pm = np.zeros((h, w), dtype=np.uint8)
            if getattr(r, "masks", None) is not None and r.masks.data is not None:
                arr = r.masks.data.detach().cpu().numpy()
                resized = []
                for m in arr:
                    mu8 = (m > 0.5).astype(np.uint8)
                    if mu8.shape != (h, w):
                        if self.strict_native:
                            raise RuntimeError(
                                f"runtime seg shape mismatch at batch[{i}]: pred={mu8.shape}, image={(h, w)}"
                            )
                        import cv2

                        mu8 = cv2.resize(mu8, (w, h), interpolation=cv2.INTER_NEAREST)
                    resized.append(mu8)
                if resized:
                    pm = select_largest_person_mask(np.stack(resized, axis=0), h, w)
            masks.append(torch.from_numpy(pm.astype(np.float32)).unsqueeze(0))
        return torch.stack(masks, dim=0).to(device)
