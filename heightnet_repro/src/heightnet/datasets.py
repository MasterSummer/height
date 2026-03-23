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
            "matches_root",
        }
        missing = required_cols - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        self.rows: List[SampleRow] = []
        self.matches_root: List[str] = []
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
                )
            )
            self.matches_root.append(str(row["matches_root"]))

        self.seq_to_indices: Dict[str, List[int]] = {}
        for idx, row in enumerate(self.rows):
            self.seq_to_indices.setdefault(row.sequence_id, []).append(idx)

        for seq in self.seq_to_indices:
            self.seq_to_indices[seq].sort(key=lambda i: self.rows[i].frame_idx)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_rgb(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        if self.normalize_rgb:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        return img

    def _load_height(self, path: str) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        arr = cv2.resize(arr, (self.image_w, self.image_h), interpolation=cv2.INTER_NEAREST)
        return arr

    def _load_mask(self, path: str) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
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

    def _load_matches(self, idx0: int, idx1: int) -> np.ndarray:
        row0 = self.rows[idx0]
        row1 = self.rows[idx1]
        matches_root = self.matches_root[idx0]
        if idx0 == idx1:
            return np.zeros((0, 4), dtype=np.float32)

        file_name = f"{row0.frame_idx:06d}_{row1.frame_idx:06d}.npz"
        path = os.path.join(matches_root, row0.sequence_id, file_name)
        if not os.path.exists(path):
            return np.zeros((0, 4), dtype=np.float32)

        data = np.load(path)
        pts0 = data["pts0"].astype(np.float32)
        pts1 = data["pts1"].astype(np.float32)
        n = min(len(pts0), len(pts1), self.max_matches)
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32)
        pts0 = pts0[:n]
        pts1 = pts1[:n]

        sx = self.image_w / float(data.get("orig_w", self.image_w))
        sy = self.image_h / float(data.get("orig_h", self.image_h))
        pts0[:, 0] *= sx
        pts0[:, 1] *= sy
        pts1[:, 0] *= sx
        pts1[:, 1] *= sy
        return np.concatenate([pts0, pts1], axis=1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        rgb = self._load_rgb(row.rgb_path)
        height = self._load_height(row.height_path)
        mask = self._load_mask(row.valid_mask_path)

        sample = {
            "image": torch.from_numpy(rgb),
            "height": torch.from_numpy(height).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "person_id": row.person_id,
            "camera_id": row.camera_id,
        }

        if self.use_pair_consistency:
            idx_pair = self._get_next_index(idx)
            rgb_pair = self._load_rgb(self.rows[idx_pair].rgb_path)
            sample["image_pair"] = torch.from_numpy(rgb_pair)
            sample["matches"] = torch.from_numpy(self._load_matches(idx, idx_pair))

        return sample
