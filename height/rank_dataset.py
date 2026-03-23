import os
from collections import defaultdict
import json
import re
import pandas as pd
from loguru import logger

from rank_core import (
    add_grid_columns,
    filter_jump_frames,
    load_correction_factors,
    save_correction_factors,
    solve_global_heights_and_factors,
)
from rank_video import collect_video_records


def parse_camera_signature(filename):
    """
    从文件名中解析摄像头签名：高度(cm) + 方向。
    例: UpChange_phone_400cm_slantside_male_day_yhjz100.mp4 -> 400cm_slantside
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    tokens = base.split("_")
    height_token = None
    direction_token = None
    directions = {"slantside", "inside", "outside", "side", "front", "back"}
    for token in tokens:
        if re.fullmatch(r"\d+cm", token):
            height_token = token
        if token in directions:
            direction_token = token
    if height_token is None:
        height_token = "unknowncm"
    if direction_token is None:
        direction_token = "unknown"
    return f"{height_token}_{direction_token}"


def run_dataset_rank(args, model):
    """
    遍历数据集，按“相同高度+方向”的摄像头分组，输出单摄像头全序排名。
    """
    dataset_root = args.dataset_root
    if not os.path.isdir(dataset_root):
        logger.error(f"数据集目录不存在: {dataset_root}")
        return

    exclude_people = set()
    if args.exclude_people:
        exclude_people = {p.strip() for p in args.exclude_people.split(",") if p.strip()}
    exclude_keywords = set()
    if args.exclude_name_keyword:
        exclude_keywords = {k.strip() for k in args.exclude_name_keyword.split(",") if k.strip()}

    person_names = [p for p in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, p))]
    person_names = sorted(person_names)
    if args.person_shard_count < 1:
        logger.error("person_shard_count 必须 >= 1。")
        return
    if args.person_shard_id < 0 or args.person_shard_id >= args.person_shard_count:
        logger.error("person_shard_id 超出范围。")
        return
    if args.person_shard_count > 1:
        person_names = [
            name for idx, name in enumerate(person_names)
            if idx % args.person_shard_count == args.person_shard_id
        ]
        logger.info(f"分片模式: shard {args.person_shard_id}/{args.person_shard_count} 人数={len(person_names)}")

    camera_to_videos = defaultdict(list)
    per_person_counts = defaultdict(int)
    for person_name in person_names:
        person_dir = os.path.join(dataset_root, person_name)
        if exclude_people and person_name in exclude_people:
            continue
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(".mp4"):
                continue
            if exclude_keywords and any(k in filename for k in exclude_keywords):
                continue
            camera_key = parse_camera_signature(filename)
            if args.max_videos_per_person_per_camera:
                count_key = (person_name, camera_key)
                if per_person_counts[count_key] >= args.max_videos_per_person_per_camera:
                    continue
                per_person_counts[count_key] += 1
            video_path = os.path.join(person_dir, filename)
            camera_to_videos[camera_key].append((video_path, person_name))

    if not camera_to_videos:
        logger.error("未找到任何视频文件。")
        return

    output_dir = os.path.join(args.output_base_dir, "dataset_rank")
    os.makedirs(output_dir, exist_ok=True)
    video_output_dir = os.path.join(output_dir, "videos")
    if args.save_video:
        os.makedirs(video_output_dir, exist_ok=True)

    for camera_key, items in camera_to_videos.items():
        if args.camera_filter and camera_key != args.camera_filter:
            continue
        logger.info("=" * 50)
        logger.info(f"开始处理摄像头组: {camera_key} (视频数={len(items)})")
        all_records = []
        for video_path, person_id in items:
            logger.info(f"  处理视频: {video_path} (person={person_id})")
            video_output_path = None
            if args.save_video:
                base = os.path.splitext(os.path.basename(video_path))[0]
                video_output_path = os.path.join(video_output_dir, f"{base}_tracked.mp4")
            records = collect_video_records(
                model,
                video_path,
                person_id,
                args.min_frames,
                save_video=args.save_video,
                video_output_path=video_output_path,
                primary_track_only=args.primary_track_only,
            )
            all_records.extend(records)

        if not all_records:
            logger.warning(f"{camera_key} 没有有效追踪记录，跳过。")
            continue

        if args.export_tracks or args.tracks_only:
            shard_suffix = ""
            if args.person_shard_count > 1:
                shard_suffix = f"_shard{args.person_shard_id}"
            tracks_path = os.path.join(output_dir, f"camera_{camera_key}_tracks{shard_suffix}.json")
            with open(tracks_path, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=4, ensure_ascii=False)
            logger.success(f"已保存 bbox 轨迹: {tracks_path}")
        if args.tracks_only:
            logger.info(f"{camera_key} 仅导出 tracks，跳过排名。")
            continue

        df_raw = pd.DataFrame(all_records)
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
        logger.success(f"已保存排名结果: {out_path}")

        if args.export_cg and final_cg:
            os.makedirs(cg_dir, exist_ok=True)
            save_correction_factors(final_cg, cg_path)
            logger.success(f"已保存 C_g: {cg_path}")

        if args.export_pairs:
            pairs = []
            for i in range(len(ranking)):
                for j in range(i + 1, len(ranking)):
                    pairs.append({"id_i": ranking[i], "id_j": ranking[j], "y": 1})
            pairs_path = os.path.join(output_dir, f"camera_{camera_key}_pairs.json")
            with open(pairs_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=4, ensure_ascii=False)
            logger.success(f"已保存 pairwise 标签: {pairs_path}")
