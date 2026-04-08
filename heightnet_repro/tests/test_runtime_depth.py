from __future__ import annotations

import unittest

import torch

from src.heightnet.runtime_depth import depth_to_height


class RuntimeDepthTests(unittest.TestCase):
    def test_depth_to_height_supports_inverse_depth(self) -> None:
        depth = torch.tensor([[[[2.0]]]], dtype=torch.float32)
        bg_depth = torch.tensor([[[[4.0]]]], dtype=torch.float32)
        cam_h = torch.tensor([[[[3.0]]]], dtype=torch.float32)

        direct_height, direct_mask = depth_to_height(
            depth=depth,
            bg_depth=bg_depth,
            camera_height_m=cam_h,
            eps=1e-6,
            assume_inverse=False,
        )
        inverse_height, inverse_mask = depth_to_height(
            depth=depth,
            bg_depth=bg_depth,
            camera_height_m=cam_h,
            eps=1e-6,
            assume_inverse=True,
        )

        self.assertTrue(bool(direct_mask.item()))
        self.assertTrue(bool(inverse_mask.item()))
        self.assertAlmostEqual(float(direct_height.item()), 1.5, places=5)
        self.assertAlmostEqual(float(inverse_height.item()), 0.0, places=5)

    def test_inverse_depth_flips_order_before_height_formula(self) -> None:
        depth = torch.tensor([[[[0.25]]]], dtype=torch.float32)
        bg_depth = torch.tensor([[[[0.5]]]], dtype=torch.float32)
        cam_h = torch.tensor([[[[4.0]]]], dtype=torch.float32)

        direct_height, _ = depth_to_height(
            depth=depth,
            bg_depth=bg_depth,
            camera_height_m=cam_h,
            eps=1e-6,
            assume_inverse=False,
        )
        inverse_height, _ = depth_to_height(
            depth=depth,
            bg_depth=bg_depth,
            camera_height_m=cam_h,
            eps=1e-6,
            assume_inverse=True,
        )

        self.assertAlmostEqual(float(direct_height.item()), 2.0, places=5)
        self.assertAlmostEqual(float(inverse_height.item()), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
