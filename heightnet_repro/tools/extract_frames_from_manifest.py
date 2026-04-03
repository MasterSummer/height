from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def extract_frame(video_path: str, frame_idx: int, out_path: Path) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to decode frame={frame_idx} from {video_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f"failed to write frame image: {out_path}")
    finally:
        cap.release()


def process_manifest(manifest_path: Path, out_manifest: Path, frames_root: Path) -> dict:
    frame = pd.read_csv(manifest_path)
    rows = []
    for row in frame.to_dict(orient="records"):
        person_id = str(row["person_id"])
        sequence_id = str(row["sequence_id"])
        frame_idx = int(row["frame_idx"])
        filename = f"{sequence_id}__frame{frame_idx:06d}.jpg"
        frame_path = frames_root / person_id / filename
        extract_frame(str(row["video_path"]), frame_idx, frame_path)
        row["frame_path"] = str(frame_path.resolve())
        rows.append(row)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    return {
        "input_manifest": str(manifest_path.resolve()),
        "output_manifest": str(out_manifest.resolve()),
        "num_frames": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RGB frame images from a frame-level manifest and write frame_path back into a new manifest.")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    frames_root = out_dir / "frames"
    for manifest in args.manifest:
        manifest_path = Path(manifest).resolve()
        out_manifest = out_dir / manifest_path.name
        summary = process_manifest(manifest_path, out_manifest, frames_root)
        print(f"[EXTRACT] {manifest_path.name}: frames={summary['num_frames']}")


if __name__ == "__main__":
    main()
