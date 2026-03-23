import os
import json
import pandas as pd
from loguru import logger

from rank_core import (
    add_grid_columns,
    filter_jump_frames,
    load_correction_factors,
    save_correction_factors,
    solve_global_heights_and_factors,
)
from rank_dataset import parse_camera_signature


def _parse_label_line(line):
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 6:
        return None
    try:
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
    except ValueError:
        return None
    return frame, track_id, x, y, w, h


def _load_label_file(label_path, person_id, video_width, video_height):
    records = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_label_line(line)
            if parsed is None:
                continue
            frame, track_id, x, y, w, h = parsed
            records.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "person_id": person_id,
                    "global_id": person_id,
                    "x_center": x + w / 2.0,
                    "y_center": y + h / 2.0,
                    "width": w,
                    "height": h,
                    "bbox": [x, y, w, h],
                    "video_width": video_width,
                    "video_height": video_height,
                    "source": label_path,
                }
            )
    return records


def run_labels_rank(args):
    labels_root = args.labels_root
    if not labels_root or not os.path.isdir(labels_root):
        logger.error(f"labels_root 不存在或不是目录: {labels_root}")
        return

    exclude_people = set()
    if getattr(args, "exclude_people", None):
        exclude_people = {p.strip() for p in args.exclude_people.split(",") if p.strip()}

    if args.labels_video_width is None or args.labels_video_height is None:
        raise ValueError("labels_rank 模式需要提供 --labels_video_width 和 --labels_video_height。")

    video_width = float(args.labels_video_width)
    video_height = float(args.labels_video_height)

    camera_to_records = {}
    for person_name in os.listdir(labels_root):
        person_dir = os.path.join(labels_root, person_name)
        if not os.path.isdir(person_dir):
            continue
        if exclude_people and person_name not in exclude_people:
            continue
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(".txt"):
                continue
            label_path = os.path.join(person_dir, filename)
            camera_key = parse_camera_signature(filename)
            if args.camera_filter and camera_key != args.camera_filter:
                continue
            records = _load_label_file(label_path, person_name, video_width, video_height)
            if not records:
                continue
            camera_to_records.setdefault(camera_key, []).extend(records)

    if not camera_to_records:
        logger.error("未找到任何可用的 labels 记录。")
        return

    output_dir = os.path.join(args.output_base_dir, "labels_rank")
    os.makedirs(output_dir, exist_ok=True)

    for camera_key, records in camera_to_records.items():
        logger.info("=" * 50)
        logger.info(f"开始处理 labels 摄像头组: {camera_key} (记录数={len(records)})")
        df_raw = pd.DataFrame(records)
        df_raw = filter_jump_frames(
            df=df_raw,
            jump_ratio=args.jump_ratio,
            jump_window=args.jump_window,
            jump_cum_ratio=args.jump_cum_ratio,
        )
        df_raw = add_grid_columns(df_raw, grid_size=args.grid_size)

        cg_dir = args.cg_dir or output_dir
        cg_path = os.path.join(cg_dir, f"camera_{camera_key}_cg.json")
        history_cg = None
        if args.use_cg:
            history_cg = load_correction_factors(cg_path)
            if history_cg is not None and len(history_cg) == 0:
                history_cg = None

        final_heights, final_cg = solve_global_heights_and_factors(df_raw, history_cg=history_cg)
        if not final_heights:
            logger.warning(f"{camera_key} 无法得到有效身高结果，跳过。")
            continue

        ranked = sorted(final_heights.items(), key=lambda item: (-item[1], str(item[0])))
        ranking = [gid for gid, _ in ranked]

        out_path = os.path.join(output_dir, f"camera_{camera_key}_rank.json")
        payload = {
            "camera": camera_key,
            "ranking": ranking,
            "scores": {gid: float(score) for gid, score in ranked},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        logger.success(f"已保存 labels 排名结果: {out_path}")

        if args.export_cg and final_cg:
            os.makedirs(cg_dir, exist_ok=True)
            save_correction_factors(final_cg, cg_path)
            logger.success(f"已保存 C_g: {cg_path}")
