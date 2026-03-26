from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd


def _infer_person_camera(sequence_id: str, rel_rgb_path: str) -> tuple[str, str]:
    person_pat = re.compile(r"\d{4}_(?:man|woman)\d+", re.IGNORECASE)
    camera_pat = re.compile(r"\d+cm_(?:inside|outside|slantside)", re.IGNORECASE)

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


def _build_one_split(data_root: str, split: str, out_csv: str, matches_root: str, allow_empty_split: bool = False) -> int:
    split_root = os.path.join(data_root, split)
    rgb_files = sorted(glob.glob(os.path.join(split_root, "rgb", "*", "*.png")))

    rows = []
    missing_person_mask = 0
    for rgb in rgb_files:
        rel = os.path.relpath(rgb, os.path.join(split_root, "rgb"))
        sequence_id = rel.split(os.sep)[0]
        stem = os.path.splitext(os.path.basename(rgb))[0]
        person_id, camera_id = _infer_person_camera(sequence_id, rel)
        height_path = os.path.join(split_root, "height", sequence_id, f"{stem}.npy")
        mask_path = os.path.join(split_root, "valid_mask", sequence_id, f"{stem}.npy")
        person_mask_npy = os.path.join(split_root, "person_mask", sequence_id, f"{stem}.npy")
        person_mask_png = os.path.join(split_root, "person_mask", sequence_id, f"{stem}.png")
        person_mask_path = ""
        if os.path.exists(person_mask_npy):
            person_mask_path = person_mask_npy
        elif os.path.exists(person_mask_png):
            person_mask_path = person_mask_png
        if not (os.path.exists(height_path) and os.path.exists(mask_path)):
            continue
        if not person_mask_path:
            missing_person_mask += 1
        rows.append(
            {
                "rgb_path": os.path.abspath(rgb),
                "height_path": os.path.abspath(height_path),
                "valid_mask_path": os.path.abspath(mask_path),
                "sequence_id": sequence_id,
                "frame_idx": int(stem),
                "matches_root": os.path.abspath(matches_root),
                "person_id": person_id,
                "camera_id": camera_id,
                "person_mask_path": os.path.abspath(person_mask_path) if person_mask_path else "",
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["sequence_id", "frame_idx"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    frame.to_csv(out_csv, index=False)
    print(f"[{split}] Saved {len(frame)} rows to {out_csv} (rows_without_person_mask={missing_person_mask})")
    if len(frame) == 0:
        if allow_empty_split:
            print(f"[skip] [{split}] no valid rows for manifest.")
            return 0
        raise RuntimeError(f"[{split}] no valid rows for manifest. Please ensure height/valid_mask are generated.")
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--out", type=str, default="", help="Output csv path when split is train/val/test.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory when split=all. It will generate train/val/test manifests in this folder.",
    )
    parser.add_argument("--matches-root", type=str, required=True)
    parser.add_argument(
        "--allow-empty-split",
        action="store_true",
        help="Allow empty train/val/test splits; generate empty manifest CSV and continue.",
    )
    args = parser.parse_args()

    if args.split == "all":
        out_dir = args.out_dir or os.path.join(os.getcwd(), "data")
        counts = {}
        for split in ["train", "val", "test"]:
            out_csv = os.path.join(out_dir, f"{split}_manifest.csv")
            counts[split] = _build_one_split(
                args.data_root,
                split,
                out_csv,
                args.matches_root,
                allow_empty_split=args.allow_empty_split,
            )
        if counts["train"] + counts["val"] + counts["test"] == 0:
            raise RuntimeError("All splits are empty after manifest build. Please check data-root/depth generation.")
        print(
            "Done. "
            f"train={counts['train']}, val={counts['val']}, test={counts['test']}, out_dir={os.path.abspath(out_dir)}"
        )
        return

    if not args.out:
        raise ValueError("--out is required when --split is train/val/test")
    _build_one_split(
        args.data_root,
        args.split,
        args.out,
        args.matches_root,
        allow_empty_split=args.allow_empty_split,
    )


if __name__ == "__main__":
    main()
