from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare mid-run dataset from videos in layout: "
            "<video_root>/<person_id>/*.mp4"
        )
    )
    parser.add_argument("--video-root", type=str, required=True, help="Root dir of raw videos.")
    parser.add_argument("--data-root", type=str, required=True, help="Output dataroot for heightnet.")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="data",
        help="Where to save sampled video lists and split metadata.",
    )
    parser.add_argument("--sample-size", type=int, default=20, help="Number of videos to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps", type=float, default=3.0, help="Frame extraction fps.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=240,
        help="Max frames per video after fps sampling.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Val split ratio. Test ratio = 1 - train - val.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".mp4,.avi,.mov,.mkv",
        help="Comma-separated video extensions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frames for the same sequence_id.",
    )
    parser.add_argument(
        "--max-videos-per-person-camera",
        type=int,
        default=3,
        help=(
            "Maximum number of videos kept for each (person_id, camera_id) pair "
            "before global sampling. Use <=0 to disable."
        ),
    )
    return parser.parse_args()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH.")


def collect_videos(video_root: Path, exts: Sequence[str]) -> List[Path]:
    exts_lower = {e.lower() for e in exts}
    videos: List[Path] = []
    for person_dir in sorted(video_root.iterdir()):
        if not person_dir.is_dir():
            continue
        for p in sorted(person_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts_lower:
                videos.append(p)
    return videos


def infer_camera_id_from_name(video_name: str) -> str:
    m = re.search(r"\d+cm_(?:inside|outside|slantside|side|front|back)", video_name, flags=re.IGNORECASE)
    if m:
        return m.group(0).lower()
    return "unknown_camera"


def limit_by_person_camera(
    videos: List[Path],
    max_per_person_camera: int,
    seed: int,
) -> Tuple[List[Path], Dict[str, int]]:
    if max_per_person_camera <= 0:
        return videos, {"groups": 0, "filtered_out": 0}

    groups: Dict[Tuple[str, str], List[Path]] = {}
    for v in videos:
        person_id = v.parent.name
        camera_id = infer_camera_id_from_name(v.name)
        groups.setdefault((person_id, camera_id), []).append(v)

    rnd = random.Random(seed)
    kept: List[Path] = []
    filtered_out = 0
    for _, items in groups.items():
        items = sorted(items)
        if len(items) <= max_per_person_camera:
            kept.extend(items)
            continue
        picked = rnd.sample(items, max_per_person_camera)
        kept.extend(sorted(picked))
        filtered_out += len(items) - max_per_person_camera

    kept = sorted(kept)
    return kept, {"groups": len(groups), "filtered_out": filtered_out}


def split_videos(videos: List[Path], train_ratio: float, val_ratio: float) -> dict:
    n = len(videos)
    n_train = int(math.floor(n * train_ratio))
    n_val = int(math.floor(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test <= 0:
        # Keep at least one test item.
        n_test = 1
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
    return {
        "train": videos[:n_train],
        "val": videos[n_train : n_train + n_val],
        "test": videos[n_train + n_val :],
    }


def sequence_id_for_video(video_path: Path) -> str:
    person = video_path.parent.name
    stem = video_path.stem
    return f"{person}__{stem}"


def write_list(paths: Sequence[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p.resolve()) + "\n")


def render_progress(processed: int, total: int, ok: int, skipped: int, failed: int) -> None:
    if total <= 0:
        return
    width = 30
    ratio = min(max(processed / total, 0.0), 1.0)
    fill = int(round(width * ratio))
    bar = "#" * fill + "-" * (width - fill)
    msg = (
        f"\rProgress [{bar}] {processed}/{total} "
        f"({ratio * 100:5.1f}%) | ok={ok} skip={skipped} fail={failed}"
    )
    print(msg, end="", flush=True)


def extract_frames(video: Path, out_dir: Path, fps: float, max_frames: int, overwrite: bool) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    first_frame = out_dir / "000001.png"
    if first_frame.exists() and not overwrite:
        return {"status": "skipped_existing", "error": ""}

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-vframes",
        str(max_frames),
        str(out_dir / "%06d.png"),
    ]
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        if not err:
            err = f"ffmpeg failed with return code {result.returncode}"
        return {"status": "failed", "error": err}
    return {"status": "ok", "error": ""}


def main() -> None:
    args = parse_args()
    ensure_ffmpeg()

    video_root = Path(args.video_root).resolve()
    data_root = Path(args.data_root).resolve()
    work_dir = Path(args.work_dir).resolve()

    if not video_root.exists():
        raise FileNotFoundError(f"video_root not found: {video_root}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be > 0")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1")

    exts = [x.strip() for x in args.exts.split(",") if x.strip()]
    videos_all = collect_videos(video_root, exts)
    videos_pool, limit_stats = limit_by_person_camera(
        videos=videos_all,
        max_per_person_camera=args.max_videos_per_person_camera,
        seed=args.seed,
    )
    if len(videos_pool) < args.sample_size:
        raise RuntimeError(
            "Not enough videos after person-camera filtering: "
            f"found={len(videos_pool)}, requested={args.sample_size}. "
            "Consider reducing --sample-size or increasing --max-videos-per-person-camera."
        )

    rnd = random.Random(args.seed)
    sampled = rnd.sample(videos_pool, args.sample_size)
    rnd.shuffle(sampled)

    splits = split_videos(sampled, args.train_ratio, args.val_ratio)

    failed_items: List[Dict[str, str]] = []
    skipped_existing = 0
    ok_count = 0
    total_jobs = sum(len(v) for v in splits.values())
    processed_jobs = 0

    for split_name in ("train", "val", "test"):
        split_rgb_root = data_root / split_name / "rgb"
        split_rgb_root.mkdir(parents=True, exist_ok=True)
        for video in splits[split_name]:
            sid = sequence_id_for_video(video)
            out_dir = split_rgb_root / sid
            status = extract_frames(
                video=video,
                out_dir=out_dir,
                fps=args.fps,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
            if status["status"] == "ok":
                ok_count += 1
            elif status["status"] == "skipped_existing":
                skipped_existing += 1
            else:
                failed_items.append(
                    {
                        "split": split_name,
                        "video_path": str(video.resolve()),
                        "sequence_id": sid,
                        "error": status["error"],
                    }
                )
                print(f"[WARN] ffmpeg failed: {video}")
                if status["error"]:
                    print(f"[WARN] {status['error'][-400:]}")
            processed_jobs += 1
            render_progress(
                processed=processed_jobs,
                total=total_jobs,
                ok=ok_count,
                skipped=skipped_existing,
                failed=len(failed_items),
            )
    if total_jobs > 0:
        print()

    work_dir.mkdir(parents=True, exist_ok=True)
    write_list(sampled, work_dir / "midrun_20.txt")
    write_list(splits["train"], work_dir / "midrun_train.txt")
    write_list(splits["val"], work_dir / "midrun_val.txt")
    write_list(splits["test"], work_dir / "midrun_test.txt")
    failed_txt = work_dir / "midrun_failed_videos.txt"
    failed_json = work_dir / "midrun_failed_videos.json"
    with failed_txt.open("w", encoding="utf-8") as f:
        for item in failed_items:
            f.write(item["video_path"] + "\n")
    with failed_json.open("w", encoding="utf-8") as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2)

    summary = {
        "video_root": str(video_root),
        "data_root": str(data_root),
        "sample_size": args.sample_size,
        "videos_found_before_filter": len(videos_all),
        "videos_pool_after_filter": len(videos_pool),
        "max_videos_per_person_camera": args.max_videos_per_person_camera,
        "person_camera_groups": limit_stats["groups"],
        "videos_filtered_out_by_person_camera_cap": limit_stats["filtered_out"],
        "seed": args.seed,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "counts": {k: len(v) for k, v in splits.items()},
        "extract_stats": {
            "ok": ok_count,
            "skipped_existing": skipped_existing,
            "failed": len(failed_items),
        },
        "split_lists": {
            "all": str((work_dir / "midrun_20.txt").resolve()),
            "train": str((work_dir / "midrun_train.txt").resolve()),
            "val": str((work_dir / "midrun_val.txt").resolve()),
            "test": str((work_dir / "midrun_test.txt").resolve()),
        },
        "failed_lists": {
            "txt": str(failed_txt.resolve()),
            "json": str(failed_json.resolve()),
        },
    }
    with (work_dir / "midrun_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "Prepared mid-run dataset. "
        f"train={summary['counts']['train']}, "
        f"val={summary['counts']['val']}, "
        f"test={summary['counts']['test']}, "
        f"ok={ok_count}, failed={len(failed_items)}, skipped_existing={skipped_existing}"
    )


if __name__ == "__main__":
    main()
