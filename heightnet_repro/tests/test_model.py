from __future__ import annotations

import unittest

import torch

from src.heightnet.model import HeightNetTiny


class ModelTests(unittest.TestCase):
    def test_pair_heads_support_all_configured_types(self) -> None:
        image = torch.rand((2, 3, 64, 64), dtype=torch.float32)
        mask = torch.ones((2, 1, 64, 64), dtype=torch.float32)
        for comparator_type in ("conv", "resnet", "vit"):
            model = HeightNetTiny(
                base_channels=8,
                comparator_channels=8,
                comparator_type=comparator_type,
                comparator_layers=2,
                comparator_num_heads=2,
                comparator_patch_size=8,
            )
            pred = model(image)["pred_height_map"]
            feat = model.encode_person(pred, mask)
            logit = model.compare_encoded(feat[:1], feat[1:])

            self.assertEqual(tuple(pred.shape), (2, 1, 64, 64))
            self.assertEqual(tuple(feat.shape), (2, 8))
            self.assertEqual(tuple(logit.shape), (1,))


if __name__ == "__main__":
    unittest.main()
