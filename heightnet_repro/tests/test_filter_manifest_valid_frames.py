from __future__ import annotations

import unittest

import numpy as np

from tools.filter_manifest_valid_frames import person_frame_is_valid


class FilterManifestValidFramesTests(unittest.TestCase):
    def test_person_frame_is_valid_requires_area_and_center(self) -> None:
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[2:8, 3:7] = 1.0
        self.assertTrue(
            person_frame_is_valid(
                mask,
                min_person_pixels=20,
                min_person_area_ratio=0.1,
                center_margin_ratio=0.1,
            )
        )

    def test_person_frame_is_valid_rejects_edge_person(self) -> None:
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[1:9, 0:3] = 1.0
        self.assertFalse(
            person_frame_is_valid(
                mask,
                min_person_pixels=10,
                min_person_area_ratio=0.05,
                center_margin_ratio=0.15,
            )
        )


if __name__ == "__main__":
    unittest.main()
