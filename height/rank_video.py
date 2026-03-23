import os
from collections import defaultdict
from datetime import datetime
import json
import numpy as np
import pandas as pd
import cv2
from loguru import logger

from rank_core import (
    add_grid_columns,
    apply_correction_from_file,
    load_correction_factors,
    load_heights_from_file,
    merge_cg,
    merge_heights,
    save_correction_factors,
    save_heights_to_file,
    solve_global_heights_and_factors,
)


def collect_video_records(
    model,
    video_path,
    person_id,
    min_frames_threshold,
    save_video=False,
    video_output_path=None,
    primary_track_only=False,
):
    """
    对单个视频进行追踪，收集 bbox 记录，使用 person_id 作为全局身份。
    """
    records = []
    track_counts = defaultdict(int)
    track_spans = {}
    video_width = None
    video_height = None

    results_generator = model.track(source=video_path, tracker="botsort.yaml", stream=True, persist=True, verbose=False)
    video_writer = None
    for frame_id, r in enumerate(results_generator):
        if video_width is None:
            video_height, video_width = r.orig_shape
        if r.boxes.id is None:
            if save_video and video_output_path:
                if video_writer is None and video_width is not None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_output_path, fourcc, 30, (video_width, video_height))
                if video_writer is not None:
                    video_writer.write(r.orig_img)
            continue
        track_ids = r.boxes.id.int().cpu().tolist()
        xywh_pixel = r.boxes.xywh.cpu()
        xyxy_pixel = r.boxes.xyxy.cpu().numpy()
        for i, track_id in enumerate(track_ids):
            class_id = int(r.boxes.cls[i].item())
            if class_id != 0:
                continue
            track_counts[track_id] += 1
            if track_id not in track_spans:
                track_spans[track_id] = [frame_id, frame_id]
            else:
                track_spans[track_id][1] = frame_id
            x1, y1, x2, y2 = xyxy_pixel[i]
            w = x2 - x1
            h = y2 - y1
            video_base = os.path.splitext(os.path.basename(video_path))[0]
            folder_name = os.path.basename(os.path.dirname(video_path))
            record = {
                "frame": frame_id + 1,
                "track_id": track_id,
                "person_id": person_id,
                "global_id": f"{folder_name}_{video_base}_id{track_id}",
                "x_center": xywh_pixel[i][0].item(),
                "y_center": xywh_pixel[i][1].item(),
                "width": xywh_pixel[i][2].item(),
                "height": xywh_pixel[i][3].item(),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "video_width": video_width,
                "video_height": video_height,
                "source": video_path,
            }
            records.append(record)

            if save_video and video_output_path:
                cv2.rectangle(r.orig_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"ID:{track_id}"
                cv2.putText(
                    r.orig_img,
                    label,
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

        if save_video and video_output_path:
            if video_writer is None and video_width is not None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_output_path, fourcc, 30, (video_width, video_height))
            if video_writer is not None:
                video_writer.write(r.orig_img)

    if min_frames_threshold > 1 and track_counts:
        valid_ids = {tid for tid, cnt in track_counts.items() if cnt >= min_frames_threshold}
        records = [r for r in records if r["track_id"] in valid_ids]

    if primary_track_only and track_counts:
        def _score(tid):
            start, end = track_spans.get(tid, (0, 0))
            span = end - start
            return (span, track_counts.get(tid, 0))

        primary_id = max(track_counts.keys(), key=_score)
        records = [r for r in records if r["track_id"] == primary_id]

    if video_writer is not None:
        video_writer.release()
    return records


def run_video_flow(args, model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{args.mode}"
    if args.highlight_id is not None:
        run_name += f"_highlight_id_{args.highlight_id}"

    output_dir = os.path.join(args.output_base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"本次运行的所有结果将保存在: {output_dir}")

    show_video = True
    save_video = False
    font_scale, font_thickness, box_thickness = 0.5, 1, 2
    highlight_color, highlight_box_thickness, highlight_font_scale = (0, 255, 255), 4, 0.7
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
        (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128),
        (255, 165, 0), (255, 215, 0), (184, 134, 11), (255, 20, 147)
    ]

    all_tracking_data = []
    video_writer, video_info, is_paused = None, {}, False
    results_generator = model.track(source=args.source, tracker="botsort.yaml", stream=True, persist=True, verbose=False)

    logger.info("="*30)
    logger.info("开始交互式追踪...")
    if args.highlight_id is not None:
        logger.info(f"*** 高亮模式激活：追踪 ID: {args.highlight_id} ***")
    logger.info("  - 按 'p' 键: 暂停/继续")
    logger.info("  - 按 's' 键: (暂停时) 单步前进")
    logger.info("  - 按 'q' 键: 退出")
    logger.info("="*30)

    if show_video:
        cv2.namedWindow("Interactive Tracking")
        cv2.createTrackbar('Speed (ms delay)', 'Interactive Tracking', 25, 1000, lambda x: None)

    for frame_id, r in enumerate(results_generator):
        frame = r.orig_img
        if video_info.get('width') is None:
            video_height, video_width = r.orig_shape
            video_info = {'width': video_width, 'height': video_height}

        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
            xyxy_pixel = r.boxes.xyxy.cpu().numpy()
            xywh_pixel = r.boxes.xywh.cpu()

            for i, track_id in enumerate(track_ids):
                is_target = (args.highlight_id is not None and track_id == args.highlight_id)
                if is_target:
                    color, b_thickness, f_scale, label = highlight_color, highlight_box_thickness, highlight_font_scale, f"*** ID:{track_id} ***"
                else:
                    color, b_thickness, f_scale, label = colors[track_id % len(colors)], box_thickness, font_scale, f"ID:{track_id}"

                x1, y1, x2, y2 = xyxy_pixel[i]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, b_thickness)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, f_scale, font_thickness)
                label_x, label_y = int(x1), int(y1) - 10
                cv2.rectangle(frame, (label_x, label_y - h), (label_x + w, label_y), color, -1)
                cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 0, 0), font_thickness)

                class_id = int(r.boxes.cls[i].item())
                if class_id == 0:
                    record = {
                        'frame': frame_id + 1,
                        'track_id': track_id,
                        'x_center': xywh_pixel[i][0].item(),
                        'y_center': xywh_pixel[i][1].item(),
                        'width': xywh_pixel[i][2].item(),
                        'height': xywh_pixel[i][3].item(),
                    }
                    all_tracking_data.append(record)

        if show_video:
            cv2.imshow("Interactive Tracking", frame)
            key = cv2.waitKey(1 if not is_paused else 0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                is_paused = not is_paused
            elif key == ord('s') and is_paused:
                pass
            while is_paused:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('p'):
                    is_paused = False
                    break
                if key == ord('s'):
                    break
                if key == ord('q'):
                    break
            if key == ord('q'):
                break

        if save_video:
            if video_writer is None:
                video_name = os.path.basename(args.source)
                output_video_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_tracked.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (video_info['width'], video_info['height']))
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    logger.info(f"========== 追踪循环结束，共收集到 {len(all_tracking_data)} 条记录 ==========")
    if not all_tracking_data:
        logger.warning("在视频中没有追踪到任何 'person' 目标，分析中止。")
        return

    df_raw = pd.DataFrame(all_tracking_data)
    df_raw = add_grid_columns(
        df=df_raw,
        grid_size=50,
        video_width=video_info['width'],
        video_height=video_info['height'],
    )
    video_name = os.path.basename(args.source)
    df_raw['global_id'] = df_raw['track_id'].apply(lambda tid: f"{video_name}_{tid}")

    final_heights = None
    correction_factors = None

    camera_rank_file = os.path.join(args.output_base_dir, f"{args.cam}_heights.json")
    camera_cg_file = os.path.join(args.output_base_dir, f"{args.cam}_cg.json")
    history_cg = load_correction_factors(camera_cg_file)
    if history_cg is not None and len(history_cg) == 0:
        history_cg = None

    if args.mode == 'inference':
        logger.info("运行在 'inference' 模式: 加载地图并进行快速排名...")
        final_heights = apply_correction_from_file(df_raw, args.perspective_model)
    else:
        final_heights, new_cg = solve_global_heights_and_factors(df_raw, history_cg=history_cg)
        if final_heights is None or new_cg is None:
            logger.error("全局优化未能得到有效的结果，流程中止。")
            return

        merged_cg = new_cg if history_cg is None else merge_cg(history_cg, new_cg, alpha=0.5)
        save_correction_factors(merged_cg, camera_cg_file)
        correction_factors = new_cg

        if args.mode == 'full':
            logger.info("运行在 'full' 模式: 优化并排名当前视频并更新历史...")
            old_heights = load_heights_from_file(camera_rank_file)
            merged_heights = merge_heights(old_heights, final_heights)
            save_heights_to_file(merged_heights, camera_rank_file)
            logger.success(f"已更新并保存摄像头历史身高排名到: {camera_rank_file}")
            final_heights = merged_heights
        elif args.mode == 'calibrate':
            logger.info("运行在 'calibrate' 模式: 学习透视地图并保存...")
            with open(args.perspective_model, 'w') as f:
                json.dump({str(k): v for k, v in merged_cg.items()}, f, indent=4)
            logger.success(f"场景标定成功！透视地图已保存到: {args.perspective_model}")

    if final_heights:
        if args.mode == 'inference':
            avg_cg_per_person = df_raw.groupby('track_id')['correction_factor'].mean().to_dict()
        else:
            person_grid_visits = df_raw.groupby('global_id')['grid_cell'].unique().to_dict()
            avg_cg_per_person = {}
            if correction_factors is None:
                logger.warning("缺少本次求解的校正因子，平均校正因子将显示为 'N/A'")
            else:
                for global_id, visited_grids in person_grid_visits.items():
                    cg_values = [correction_factors.get(grid, 1.0) for grid in visited_grids]
                    avg_cg_per_person[global_id] = np.mean(cg_values) if cg_values else 1.0

        sorted_ranking = sorted(final_heights.items(), key=lambda item: item[1], reverse=True)
        logger.info("\n" + "="*30 + " 全局身高排名 " + "="*30)
        logger.info(f"{'排名':<5} | {'GlobalID':<20} | {'标准化身高 (H_p)':<20} | {'平均校正因子 (Avg C_g)':<25} | {'百分位':<10}")
        logger.info("-" * 100)

        total_people = len(sorted_ranking)
        for rank, (global_id, height) in enumerate(sorted_ranking):
            percentile = (total_people - rank - 1) / total_people * 100
            avg_cg = avg_cg_per_person.get(global_id, None)
            avg_cg_str = f"{avg_cg:.4f}" if isinstance(avg_cg, (int, float)) else "N/A"
            rank_str, height_str, percentile_str = f"{rank + 1}/{total_people}", f"{height:.2f}", f"{percentile:.2f}%"
            logger.info(f"{rank_str:<5} | {global_id:<20} | {height_str:<20} | {avg_cg_str:<25} | {percentile_str:<10}")

        try:
            video_name = os.path.basename(args.source)
            heights_filename = f"{os.path.splitext(video_name)[0]}_heights.json"
            heights_filepath = os.path.join(output_dir, heights_filename)
            heights_to_save = {str(k): v for k, v in final_heights.items()}
            with open(heights_filepath, 'w', encoding='utf-8') as f:
                json.dump(heights_to_save, f, indent=4, ensure_ascii=False)
            logger.success(f"已将标准化身高 (H_p) 结果保存到: {heights_filepath}")
        except Exception as exc:
            logger.error(f"保存身高结果文件时发生错误: {exc}")
