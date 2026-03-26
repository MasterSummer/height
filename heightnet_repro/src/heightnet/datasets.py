from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SampleRow:
    rgb_path: str
    height_path: str
    valid_mask_path: str
    sequence_id: str
    frame_idx: int
    person_id: str
    camera_id: str
    person_mask_path: str


class HeightDataset(Dataset):
    """Dataset for HeightNet-style supervision with optional pair-consistency data."""

    def __init__(
        self,
        manifest_path: str,
        image_size: Tuple[int, int],
        normalize_rgb: bool = True,
        use_pair_consistency: bool = True,
        max_matches: int = 512,
    ) -> None:
        self.manifest_path = manifest_path
        self.image_h, self.image_w = image_size
        self.normalize_rgb = normalize_rgb
        self.use_pair_consistency = use_pair_consistency
        self.max_matches = max_matches

        frame = pd.read_csv(manifest_path)
        required_cols = {
            "rgb_path",
            "height_path",
            "valid_mask_path",
            "sequence_id",
            "frame_idx",
        }
        missing = required_cols - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        self.rows: List[SampleRow] = []
        for _, row in frame.iterrows():
            self.rows.append(
                SampleRow(
                    rgb_path=str(row["rgb_path"]),
                    height_path=str(row["height_path"]),
                    valid_mask_path=str(row["valid_mask_path"]),
                    sequence_id=str(row["sequence_id"]),
                    frame_idx=int(row["frame_idx"]),
                    person_id=str(row["person_id"]) if "person_id" in frame.columns else "unknown_person",
                    camera_id=str(row["camera_id"]) if "camera_id" in frame.columns else "unknown_camera",
                    person_mask_path=str(row["person_mask_path"]) if "person_mask_path" in frame.columns else "",
                )
            )

        self.seq_to_indices: Dict[str, List[int]] = {}
        for idx, row in enumerate(self.rows):
            self.seq_to_indices.setdefault(row.sequence_id, []).append(idx)

        for seq in self.seq_to_indices:
            self.seq_to_indices[seq].sort(key=lambda i: self.rows[i].frame_idx)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_rgb_pair(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
        raw = img.copy()
        img = img.astype(np.float32) / 255.0
        if self.normalize_rgb:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        raw = np.transpose(raw, (2, 0, 1))
        return img, raw

    def _load_height(self, path: str) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        arr = cv2.resize(arr, (self.image_w, self.image_h), interpolation=cv2.INTER_NEAREST)
        return arr

    def _load_mask(self, path: str) -> np.ndarray:
        if path.lower().endswith(".npy"):
            arr = np.load(path).astype(np.float32)
        else:
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise FileNotFoundError(path)
            arr = arr.astype(np.float32) / 255.0
        arr = cv2.resize(arr, (self.image_w, self.image_h), interpolation=cv2.INTER_NEAREST)
        return (arr > 0.5).astype(np.float32)

    def _get_next_index(self, idx: int) -> int:
        row = self.rows[idx]
        indices = self.seq_to_indices[row.sequence_id]
        pos = indices.index(idx)
        if pos + 1 < len(indices):
            return indices[pos + 1]
        if pos > 0:
            return indices[pos - 1]
        return idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        rgb, rgb_raw = self._load_rgb_pair(row.rgb_path)
        height = self._load_height(row.height_path)
        mask = self._load_mask(row.valid_mask_path)
        person_mask = None
        if row.person_mask_path and os.path.exists(row.person_mask_path):
            person_mask = self._load_mask(row.person_mask_path)

        sample = {
            "image": torch.from_numpy(rgb),
            "image_raw": torch.from_numpy(rgb_raw.copy()),
            "height": torch.from_numpy(height).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "person_id": row.person_id,
            "camera_id": row.camera_id,
        }
        if person_mask is not None:
            sample["person_mask"] = torch.from_numpy(person_mask).unsqueeze(0)

        if self.use_pair_consistency:
            idx_pair = self._get_next_index(idx)
            row_pair = self.rows[idx_pair]
            rgb_pair, rgb_pair_raw = self._load_rgb_pair(row_pair.rgb_path)
            sample["image_pair"] = torch.from_numpy(rgb_pair)
            sample["image_pair_raw"] = torch.from_numpy(rgb_pair_raw.copy())
            if row_pair.person_mask_path and os.path.exists(row_pair.person_mask_path):
                person_mask_pair = self._load_mask(row_pair.person_mask_path)
                sample["person_mask_pair"] = torch.from_numpy(person_mask_pair).unsqueeze(0)

        return sample
