from __future__ import annotations

import types
import unittest

import numpy as np

from src.heightnet.datasets import HeightDataset, VideoRow


class DatasetTests(unittest.TestCase):
    def test_getitem_falls_back_to_nearby_frame_when_decode_fails(self) -> None:
        ds = HeightDataset.__new__(HeightDataset)
        ds.image_h = 4
        ds.image_w = 4
        ds.normalize_rgb = False
        ds.use_pair_consistency = False
        ds.train_mode = True
        ds._cache_arrays = {}
        ds.rows = [
            VideoRow(
                video_path="dummy.mp4",
                frame_path="",
                sequence_id="seq",
                person_id="p1",
                camera_id="cam",
                frame_start=0,
                frame_end=3,
                fps=30.0,
                valid_frames_path="",
                height_cache_path="",
                valid_mask_cache_path="",
                depth_cache_path="",
                bg_depth_path="bg.npy",
                camera_height_m=4.0,
            )
        ]
        ds._pick_frame = types.MethodType(lambda self, start, end: 1, ds)

        seen_frames: list[int] = []

        def _load_video_frame(self: HeightDataset, video_path: str, frame_idx: int) -> np.ndarray:
            seen_frames.append(frame_idx)
            if frame_idx == 1:
                raise RuntimeError(f"failed to decode frame={frame_idx} from {video_path}")
            return np.full((4, 4, 3), frame_idx, dtype=np.uint8)

        def _load_height_and_mask(
            self: HeightDataset, row: VideoRow, frame_idx: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
            height = np.full((4, 4), frame_idx, dtype=np.float32)
            valid = np.ones((4, 4), dtype=np.float32)
            bg = np.ones((4, 4), dtype=np.float32)
            return height, valid, bg, row.camera_height_m, 0.0

        ds._load_video_frame = types.MethodType(_load_video_frame, ds)
        ds._load_height_and_mask = types.MethodType(_load_height_and_mask, ds)

        sample = ds[0]

        self.assertEqual(seen_frames, [1, 2])
        self.assertEqual(sample["frame_idx"], 2)
        self.assertEqual(float(sample["height"][0, 0, 0]), 2.0)
        self.assertEqual(int(sample["image_raw"][0, 0, 0]), 2)

    def test_pick_frame_for_row_prefers_valid_frames_list(self) -> None:
        ds = HeightDataset.__new__(HeightDataset)
        ds.train_mode = False
        ds._cache_arrays = {"/tmp/valid.npy": np.asarray([5, 9, 12], dtype=np.int32)}
        ds._sequence_sampled_frames = {}
        row = VideoRow(
            video_path="dummy.mp4",
            frame_path="",
            sequence_id="seq",
            person_id="p1",
            camera_id="cam",
            frame_start=0,
            frame_end=20,
            fps=30.0,
            valid_frames_path="/tmp/valid.npy",
            height_cache_path="",
            valid_mask_cache_path="",
            depth_cache_path="",
            bg_depth_path="bg.npy",
            camera_height_m=4.0,
        )

        frame_idx = ds._pick_frame_for_row(row)
        adj_idx = ds._pick_adjacent_for_row(row, 9)

        self.assertEqual(frame_idx, 5)
        self.assertEqual(adj_idx, 12)


if __name__ == "__main__":
    unittest.main()
