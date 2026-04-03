from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def subsample_manifest(manifest_path: Path, out_path: Path, max_videos_per_person: int) -> dict:
    frame = pd.read_csv(manifest_path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        grouped[str(row["person_id"])].append(row)

    out_rows = []
    for person_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: str(r["sequence_id"]))
        out_rows.extend(rows[:max_videos_per_person])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    return {
        "input_manifest": str(manifest_path.resolve()),
        "output_manifest": str(out_path.resolve()),
        "num_rows_in": int(len(frame)),
        "num_rows_out": int(len(out_rows)),
        "num_people": int(len(grouped)),
        "max_videos_per_person": int(max_videos_per_person),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cap each person_id to at most K videos in one or more manifests.")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-videos-per-person", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for manifest in args.manifest:
        manifest_path = Path(manifest).resolve()
        out_path = out_dir / manifest_path.name
        summary = subsample_manifest(
            manifest_path=manifest_path,
            out_path=out_path,
            max_videos_per_person=max(args.max_videos_per_person, 1),
        )
        print(
            f"[SUBSAMPLE] {manifest_path.name}: kept={summary['num_rows_out']}/{summary['num_rows_in']} "
            f"people={summary['num_people']} max_per_person={summary['max_videos_per_person']}"
        )


if __name__ == "__main__":
    main()
