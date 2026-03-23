import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd
from loguru import logger
from ultralytics import YOLO

_MODULE_DIR = os.path.dirname(__file__)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from rank_core import add_grid_columns, filter_jump_frames, solve_global_heights_and_factors  # noqa: E402
from rank_dataset import parse_camera_signature  # noqa: E402
from rank_video import collect_video_records  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Label height zones by bbox comparison within camera groups")
    parser.add_argument("--train_root", type=str, required=True, help="训练集视频根目录")
    parser.add_argument("--test_labels_root", type=str, required=True, help="测试集 processed_labels 根目录")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型权重路径")
    parser.add_argument("--output_train", type=str, required=True, help="训练集标签输出 JSON")
    parser.add_argument("--output_test", type=str, required=True, help="测试集标签输出 JSON")
    parser.add_argument("--labels_video_width", type=float, required=True, help="测试集视频宽度")
    parser.add_argument("--labels_video_height", type=float, required=True, help="测试集视频高度")
    parser.add_argument("--grid_size", type=int, default=80, help="网格大小")
    parser.add_argument("--min_frames", type=int, default=20, help="最少帧数")
    parser.add_argument(
        "--max_videos_per_person",
        type=int,
        default=5,
        help="训练集每个人最多使用的视频数量",
    )
    parser.add_argument(
        "--boundary_person",
        type=str,
        default=None,
        help="指定分界人（只用该人的视频进行对比）",
    )
    parser.add_argument("--camera_filter", type=str, default=None, help="仅处理指定摄像头组")
    parser.add_argument("--jump_ratio", type=float, default=0.15, help="跳变过滤阈值")
    parser.add_argument("--jump_window", type=int, default=5, help="跳变窗口大小")
    parser.add_argument("--jump_cum_ratio", type=float, default=0.6, help="累计跳变阈值")
    return parser.parse_args()


def extract_person_id(key):
    tokens = key.split("_")
    if len(tokens) >= 2 and tokens[0].isdigit():
        return f"{tokens[0]}_{tokens[1]}"
    return tokens[0] if tokens else key


def load_label_records(labels_root, video_width, video_height, camera_filter):
    records_by_camera = defaultdict(list)
    for person_name in os.listdir(labels_root):
        person_dir = os.path.join(labels_root, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(".txt"):
                continue
            camera_key = parse_camera_signature(filename)
            if camera_filter and camera_key != camera_filter:
                continue
            label_path = os.path.join(person_dir, filename)
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) < 6:
                        continue
                    try:
                        frame = int(float(parts[0]))
                        track_id = int(float(parts[1]))
                        x = float(parts[2])
                        y = float(parts[3])
                        w = float(parts[4])
                        h = float(parts[5])
                    except ValueError:
                        continue
                    records_by_camera[camera_key].append(
                        {
                            "frame": frame,
                            "track_id": track_id,
                            "person_id": person_name,
                            "global_id": person_name,
                            "x_center": x + w / 2.0,
                            "y_center": y + h / 2.0,
                            "width": w,
                            "height": h,
                            "bbox": [x, y, w, h],
                            "video_width": float(video_width),
                            "video_height": float(video_height),
                            "source": label_path,
                        }
                    )
    return records_by_camera


def load_train_records(train_root, model, min_frames, camera_filter, max_videos_per_person, person_filter):
    records_by_camera = defaultdict(list)
    per_person_counts = defaultdict(int)
    if not os.path.isdir(train_root):
        raise ValueError(f"训练集目录不存在: {train_root}")
    for person_name in sorted(os.listdir(train_root)):
        person_dir = os.path.join(train_root, person_name)
        if not os.path.isdir(person_dir):
            continue
        if person_filter and person_name != person_filter:
            continue
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith(".mp4"):
                continue
            if max_videos_per_person is not None and per_person_counts[person_name] >= max_videos_per_person:
                continue
            camera_key = parse_camera_signature(filename)
            if camera_filter and camera_key != camera_filter:
                continue
            video_path = os.path.join(person_dir, filename)
            try:
                records = collect_video_records(
                    model,
                    video_path,
                    person_name,
                    min_frames,
                    save_video=False,
                    video_output_path=None,
                    primary_track_only=False,
                )
            except FileNotFoundError:
                logger.warning(f"训练集视频不存在，已跳过: {video_path}")
                continue
            if records:
                records_by_camera[camera_key].extend(records)
                per_person_counts[person_name] += 1
    return records_by_camera


def compute_heights(records, grid_size, jump_ratio, jump_window, jump_cum_ratio, history_cg=None):
    if not records:
        return {}, {}
    df_raw = pd.DataFrame(records)
    df_raw = filter_jump_frames(df_raw, jump_ratio, jump_window, jump_cum_ratio)
    df_raw = add_grid_columns(df_raw, grid_size=grid_size)
    heights, cg = solve_global_heights_and_factors(df_raw, history_cg=history_cg)
    if heights is None:
        return {}, {}
    return heights, cg or {}


def compute_person_means(heights):
    buckets = defaultdict(list)
    for gid, score in heights.items():
        pid = extract_person_id(str(gid))
        buckets[pid].append(float(score))
    return {pid: sum(vals) / len(vals) for pid, vals in buckets.items() if vals}


def select_boundary_person(means):
    if not means:
        return None, None
    ordered = sorted(means.items(), key=lambda item: item[1], reverse=True)
    split = len(ordered) // 2
    boundary_pid, boundary_score = ordered[split - 1] if split > 0 else ordered[0]
    return boundary_pid, boundary_score


def compute_global_means(train_means_by_camera):
    buckets = defaultdict(list)
    for people in train_means_by_camera.values():
        for pid, score in people.items():
            buckets[pid].append(float(score))
    return {pid: sum(vals) / len(vals) for pid, vals in buckets.items() if vals}


def label_people(means, boundary_score):
    labels = {}
    for pid, score in means.items():
        labels[pid] = {
            "mean_height": score,
            "label": "high" if score >= boundary_score else "low",
        }
    return labels


def merge_votes(labels_by_camera):
    merged = {}
    for pid, per_cam in labels_by_camera.items():
        votes = [1 if v["label"] == "high" else 0 for v in per_cam.values()]
        mean_vote = sum(votes) / len(votes) if votes else None
        merged[pid] = {
            "per_camera": per_cam,
            "mean_vote": mean_vote,
            "label": "high" if mean_vote is not None and mean_vote >= 0.5 else "low",
        }
    return merged


def main():
    args = parse_args()
    model = YOLO(args.model)

    train_records = load_train_records(
        args.train_root,
        model,
        args.min_frames,
        args.camera_filter,
        args.max_videos_per_person,
        args.boundary_person,
    )
    test_records = load_label_records(
        args.test_labels_root,
        args.labels_video_width,
        args.labels_video_height,
        args.camera_filter,
    )

    boundaries = {}
    train_labels_by_cam = defaultdict(dict)
    test_labels_by_cam = defaultdict(dict)
    train_means_by_camera = {}
    train_cg_by_camera = {}

    all_cameras = set(train_records) | set(test_records)
    for camera_key in sorted(all_cameras):
        if args.camera_filter and camera_key != args.camera_filter:
            continue
        train_heights, train_cg = compute_heights(
            train_records.get(camera_key, []),
            args.grid_size,
            args.jump_ratio,
            args.jump_window,
            args.jump_cum_ratio,
            history_cg=None,
        )
        if not train_heights:
            logger.warning(f"{camera_key} 训练集无有效身高结果，跳过。")
            continue

        train_means = compute_person_means(train_heights)
        train_means_by_camera[camera_key] = train_means
        train_cg_by_camera[camera_key] = train_cg

    if args.boundary_person:
        global_boundary_pid = args.boundary_person
        boundaries["global_boundary_person"] = global_boundary_pid
        boundaries["per_camera"] = {}
    else:
        global_means = compute_global_means(train_means_by_camera)
        global_boundary_pid, global_boundary_score = select_boundary_person(global_means)
        if global_boundary_pid is None:
            raise SystemExit("训练集无法确定全局分界人。")
        boundaries["global_boundary_person"] = global_boundary_pid
        boundaries["global_boundary_height"] = global_boundary_score
        boundaries["per_camera"] = {}

    for camera_key in sorted(train_means_by_camera.keys()):
        if args.camera_filter and camera_key != args.camera_filter:
            continue
        train_means = train_means_by_camera[camera_key]
        if global_boundary_pid not in train_means:
            logger.warning(f"{camera_key} 无全局分界人数据，跳过该摄像头组。")
            continue
        boundary_score = train_means[global_boundary_pid]
        boundaries["per_camera"][camera_key] = {
            "boundary_person": global_boundary_pid,
            "boundary_height": boundary_score,
            "population": len(train_means),
        }

        train_labels = label_people(train_means, boundary_score)
        for pid, payload in train_labels.items():
            train_labels_by_cam[pid][camera_key] = payload

        test_heights, _ = compute_heights(
            test_records.get(camera_key, []),
            args.grid_size,
            args.jump_ratio,
            args.jump_window,
            args.jump_cum_ratio,
            history_cg=train_cg_by_camera.get(camera_key),
        )
        if not test_heights:
            logger.warning(f"{camera_key} 测试集无有效身高结果，跳过。")
            continue
        test_means = compute_person_means(test_heights)
        test_labels = label_people(test_means, boundary_score)
        for pid, payload in test_labels.items():
            test_labels_by_cam[pid][camera_key] = payload

    train_output = {"boundaries": boundaries, "labels": merge_votes(train_labels_by_cam)}
    test_output = {"boundaries": boundaries, "labels": merge_votes(test_labels_by_cam)}

    with open(args.output_train, "w", encoding="utf-8") as f:
        json.dump(train_output, f, ensure_ascii=False, indent=4)
    with open(args.output_test, "w", encoding="utf-8") as f:
        json.dump(test_output, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
