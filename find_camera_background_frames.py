#!/usr/bin/env python3
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}
POSITION_TOKENS = {"inside", "outside", "slantside"}


@dataclass
class FrameCandidate:
    video_path: Path
    video_name: str
    camera_id: str
    timestamp_sec: float
    score: float
    frame_bgr: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find no-person background frames per camera from a video directory."
    )
    parser.add_argument("--input", required=True, type=str, help="Root directory containing videos")
    parser.add_argument("--out", required=True, type=str, help="Output directory")
    parser.add_argument("--model", default="yolov8n.pt", type=str, help="YOLO model path or name")
    parser.add_argument("--sample-every-sec", default=2.0, type=float, help="Sampling stride in seconds")
    parser.add_argument("--frame-width", default=640, type=int, help="Resize width for detector input")
    parser.add_argument("--person-conf-threshold", default=0.25, type=float, help="Person confidence threshold")
    parser.add_argument("--max-per-video", default=1, type=int, help="Maximum frames to keep per video")
    parser.add_argument("--max-per-camera", default=20, type=int, help="Maximum frames to keep per camera")
    parser.add_argument(
        "--camera-id-regex",
        default="",
        type=str,
        help="Optional regex with one capture group to derive camera_id from filename stem",
    )
    parser.add_argument(
        "--allow-fallback-camera-id",
        action="store_true",
        help="If parsing fails, fallback to parent directory name as camera_id",
    )
    return parser.parse_args()


def find_videos(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def parse_camera_id(video_name: str, regex: str) -> Tuple[Optional[str], Optional[str]]:
    stem = Path(video_name).stem

    if regex:
        m = re.search(regex, stem)
        if m and m.groups():
            return m.group(1), None
        return None, f"camera-id regex not matched: {regex}"

    tokens = stem.split("_")
    for i in range(len(tokens) - 1):
        if re.fullmatch(r"\d+cm", tokens[i]) and tokens[i + 1].lower() in POSITION_TOKENS:
            return f"{tokens[i]}_{tokens[i + 1].lower()}", None

    # NVR style, e.g. NVR_ch11_main_...
    for token in tokens:
        if re.fullmatch(r"ch\d+", token.lower()):
            return token.lower(), None

    return None, "unable to infer camera_id from filename"


def resized(frame: np.ndarray, frame_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if frame_width <= 0 or w <= frame_width:
        return frame
    scale = frame_width / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (frame_width, nh), interpolation=cv2.INTER_AREA)


def frame_quality_score(frame: np.ndarray) -> float:
    # Prefer sharper frames as background candidates.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_frame_indices(total_frames: int, fps: float, sample_every_sec: float) -> List[int]:
    if total_frames <= 0:
        return []
    if fps <= 0:
        fps = 25.0
    stride = max(1, int(round(sample_every_sec * fps)))
    indices = list(range(0, total_frames, stride))
    if (total_frames - 1) not in indices:
        indices.append(total_frames - 1)
    return indices


def find_video_candidate(
    model,
    video_path: Path,
    camera_id: str,
    frame_width: int,
    person_conf_threshold: float,
    sample_every_sec: float,
) -> Tuple[Optional[FrameCandidate], Optional[str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "cannot open video"
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        indices = sample_frame_indices(total_frames, fps, sample_every_sec)
        best: Optional[FrameCandidate] = None

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            det_frame = resized(frame, frame_width)
            pred = model.predict(
                source=det_frame,
                classes=[0],  # person
                conf=person_conf_threshold,
                verbose=False,
            )[0]
            has_person = pred.boxes is not None and len(pred.boxes) > 0
            if has_person:
                continue

            score = frame_quality_score(frame)
            ts = idx / max(fps, 1e-6)
            cand = FrameCandidate(
                video_path=video_path,
                video_name=video_path.name,
                camera_id=camera_id,
                timestamp_sec=float(ts),
                score=float(score),
                frame_bgr=frame,
            )
            if best is None or cand.score > best.score:
                best = cand

        if best is None:
            return None, "no no-person frame found"
        return best, None
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    # Keep ultralytics config/cache inside writable output dir.
    import os

    ultra_dir = out_dir / ".ultralytics"
    ultra_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(ultra_dir))

    from ultralytics import YOLO

    model = YOLO(args.model)
    videos = find_videos(input_dir)

    parse_failures: List[Dict] = []
    per_camera_all: Dict[str, List[FrameCandidate]] = {}
    results: List[Dict] = []

    for video_path in videos:
        camera_id, parse_error = parse_camera_id(video_path.name, args.camera_id_regex)
        if camera_id is None and args.allow_fallback_camera_id:
            camera_id = video_path.parent.name
            parse_error = None
        if camera_id is None:
            parse_failures.append({"video_path": str(video_path), "error": parse_error})
            continue

        candidate, err = find_video_candidate(
            model=model,
            video_path=video_path,
            camera_id=camera_id,
            frame_width=args.frame_width,
            person_conf_threshold=args.person_conf_threshold,
            sample_every_sec=args.sample_every_sec,
        )
        if candidate is None:
            results.append(
                {
                    "video_path": str(video_path),
                    "video_name": video_path.name,
                    "camera_id": camera_id,
                    "timestamp_sec": 0.0,
                    "score": float("-inf"),
                    "output_path": None,
                    "found_background_frame": False,
                    "detector": "ultralytics",
                    "person_conf_threshold": args.person_conf_threshold,
                    "skip_reason": err,
                }
            )
            continue

        per_camera_all.setdefault(camera_id, []).append(candidate)

    kept_records: List[Dict] = []
    camera_stats: Dict[str, Dict] = {}

    for camera_id, candidates in per_camera_all.items():
        candidates.sort(key=lambda x: x.score, reverse=True)
        kept = candidates[: max(0, args.max_per_camera)]
        dropped = max(0, len(candidates) - len(kept))
        cam_dir = out_dir / camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        for cand in kept:
            out_name = f"{Path(cand.video_name).stem}_bg.jpg"
            out_path = cam_dir / out_name
            cv2.imwrite(str(out_path), cand.frame_bgr)
            kept_records.append(
                {
                    "video_path": str(cand.video_path),
                    "video_name": cand.video_name,
                    "camera_id": cand.camera_id,
                    "timestamp_sec": round(cand.timestamp_sec, 3),
                    "score": cand.score,
                    "output_path": str(out_path),
                    "found_background_frame": True,
                    "detector": "ultralytics",
                    "person_conf_threshold": args.person_conf_threshold,
                    "skip_reason": None,
                }
            )

        camera_stats[camera_id] = {
            "eligible_videos": len(candidates),
            "found_background_frame": len(candidates),
            "kept": len(kept),
            "missing_to_20": max(0, args.max_per_camera - len(kept)),
            "no_background_frame_found": 0,
            "camera_cap_dropped": dropped,
            "other_failures": 0,
        }

    for rec in results:
        if rec["camera_id"] not in camera_stats:
            camera_stats[rec["camera_id"]] = {
                "eligible_videos": 0,
                "found_background_frame": 0,
                "kept": 0,
                "missing_to_20": args.max_per_camera,
                "no_background_frame_found": 0,
                "camera_cap_dropped": 0,
                "other_failures": 0,
            }
        camera_stats[rec["camera_id"]]["eligible_videos"] += 1
        if rec["skip_reason"] == "no no-person frame found":
            camera_stats[rec["camera_id"]]["no_background_frame_found"] += 1
        else:
            camera_stats[rec["camera_id"]]["other_failures"] += 1

    for cam, stat in camera_stats.items():
        stat["eligible_videos"] = int(stat["eligible_videos"])
        stat["found_background_frame"] = int(stat["found_background_frame"])
        stat["kept"] = int(stat["kept"])
        stat["missing_to_20"] = max(0, args.max_per_camera - stat["kept"])

    all_results = kept_records + results
    all_results.sort(key=lambda x: (x["camera_id"], x["video_name"]))

    manifest = {
        "config": {
            "input": str(input_dir),
            "out": str(out_dir),
            "sample_every_sec": args.sample_every_sec,
            "max_per_camera": args.max_per_camera,
            "max_per_video": args.max_per_video,
            "person_conf_threshold": args.person_conf_threshold,
            "frame_width": args.frame_width,
            "detector": "ultralytics",
            "model": args.model,
            "camera_id_regex": args.camera_id_regex,
            "allow_fallback_camera_id": args.allow_fallback_camera_id,
        },
        "parse_failures": parse_failures,
        "results": all_results,
    }

    summary = {
        "totals": {
            "cameras": len(camera_stats),
            "eligible_videos": sum(v["eligible_videos"] for v in camera_stats.values()),
            "kept_images": len(kept_records),
            "parse_failures": len(parse_failures),
            "no_background_frame_found": sum(v["no_background_frame_found"] for v in camera_stats.values()),
            "camera_cap_dropped": sum(v["camera_cap_dropped"] for v in camera_stats.values()),
        },
        "camera_stats": camera_stats,
    }

    manifest_path = out_dir / "manifest.json"
    summary_path = out_dir / "summary.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written: {manifest_path}")
    print(f"written: {summary_path}")


if __name__ == "__main__":
    main()
