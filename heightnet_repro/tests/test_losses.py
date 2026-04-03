from __future__ import annotations

import unittest

import torch

from src.heightnet.losses import build_rank_pairs


class LossTests(unittest.TestCase):
    def test_build_rank_pairs_skips_empty_or_tiny_person_masks(self) -> None:
        person_ids = ["p1", "p2", "p3"]
        camera_ids = ["cam", "cam", "cam"]
        masks = torch.zeros((3, 1, 8, 8), dtype=torch.float32)
        masks[0, 0, :4, :4] = 1.0
        masks[1, 0, 0, 0] = 1.0
        masks[2, 0, :4, :4] = 1.0
        pairwise_labels = {"cam": {("p1", "p2"): 1, ("p2", "p1"): 0, ("p1", "p3"): 1, ("p3", "p1"): 0}}

        idx_i, idx_j, y = build_rank_pairs(
            person_ids=person_ids,
            camera_ids=camera_ids,
            person_masks=masks,
            pairwise_labels=pairwise_labels,
            min_valid_pixels=4,
            min_valid_ratio=0.05,
            device=torch.device("cpu"),
        )

        self.assertEqual(idx_i.tolist(), [0])
        self.assertEqual(idx_j.tolist(), [2])
        self.assertEqual(y.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
