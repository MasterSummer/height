from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd


def _infer_person_camera(sequence_id: str, rel_rgb_path: str) -> tuple[str, str]:
    person_pat = re.compile(r"\d{4}_(?:man|woman)\d+", re.IGNORECASE)
    camera_pat = re.compile(r"\d+cm_(?:inside|outside|slantside|side|front|back)", re.IGNORECASE)

    person_id = "unknown_person"
    camera_id = "unknown_camera"

    full_text = f"{sequence_id} {rel_rgb_path}"
    m_person = person_pat.search(full_text)
    m_camera = camera_pat.search(full_text)
    if m_person:
        person_id = m_person.group(0)
    if m_camera:
        camera_id = m_camera.group(0).lower()

    return person_id, camera_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--matches-root", type=str, required=True)
    args = parser.parse_args()

    split_root = os.path.join(args.data_root, args.split)
    rgb_files = sorted(glob.glob(os.path.join(split_root, "rgb", "*", "*.png")))

    rows = []
    for rgb in rgb_files:
        rel = os.path.relpath(rgb, os.path.join(split_root, "rgb"))
        sequence_id = rel.split(os.sep)[0]
        stem = os.path.splitext(os.path.basename(rgb))[0]
        person_id, camera_id = _infer_person_camera(sequence_id, rel)
        height_path = os.path.join(split_root, "height", sequence_id, f"{stem}.npy")
        mask_path = os.path.join(split_root, "valid_mask", sequence_id, f"{stem}.npy")
        if not (os.path.exists(height_path) and os.path.exists(mask_path)):
            continue
        rows.append(
            {
                "rgb_path": os.path.abspath(rgb),
                "height_path": os.path.abspath(height_path),
                "valid_mask_path": os.path.abspath(mask_path),
                "sequence_id": sequence_id,
                "frame_idx": int(stem),
                "matches_root": os.path.abspath(args.matches_root),
                "person_id": person_id,
                "camera_id": camera_id,
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["sequence_id", "frame_idx"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Saved {len(frame)} rows to {args.out}")


if __name__ == "__main__":
    main()
