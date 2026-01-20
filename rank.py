import argparse
from ultralytics import YOLO

from rank_core import pre_analyze_ids
from rank_dataset import run_dataset_rank
from rank_link import run_manual_link_mode
from rank_labels import run_labels_rank
from rank_tracks import run_tracks_rank
from rank_video import run_video_flow


def parse_args():
    parser = argparse.ArgumentParser("Advanced Tracking, Analysis, and Debugging Tool")
    parser.add_argument("--source", type=str, default="2503_videos/0312_man3/DnChange1_shoulder_400cm_slantside_male_day_yhjz100.mp4", help="Path to the video file.")
    parser.add_argument("--model", type=str, default="model/v8_best_3.pt", help="Path to your YOLOv8 model.")
    parser.add_argument("--output_base_dir", type=str, default="analysis_runs", help="Base directory to save all run outputs.")
    parser.add_argument("--highlight_id", type=int, default=None, help="Specify a Track ID to highlight for debugging.")
    parser.add_argument("--min_frames", type=int, default=20, help="Minimum number of frames an ID must appear to be considered.")
    parser.add_argument("--cam", type=str, required=False, help="摄像头唯一标识名（如 cam01）")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=['pre_analyze', 'full', 'calibrate', 'inference', 'link', 'dataset_rank', 'tracks_rank', 'labels_rank'],
        help=(
            "'pre_analyze' -> 仅生成ID报告。 "
            "'full' -> 对单个视频完整执行优化和排名。 "
            "'calibrate' -> 从标定视频中学习并保存透视地图。 "
            "'inference' -> 加载已有的透视地图对新视频进行快速排名。"
            "'link' -> 基于手动标注将不同摄像头的历史排名文件合并。"
            "'dataset_rank' -> 按摄像头分组，对整个数据集生成全序排名。"
            "'tracks_rank' -> 直接基于 bbox 轨迹 JSON 进行身高排名。"
            "'labels_rank' -> 基于 processed_labels 中的 bbox 文本进行身高排名。"
        )
    )
    parser.add_argument(
        "--perspective_model",
        type=str,
        default="perspective_map.json",
        help="用于保存(calibrate模式)或加载(inference模式)的透视校正地图文件路径 (.json)。"
    )
    parser.add_argument(
        "--link_files",
        nargs='+',
        help="link 模式下要合并的摄像头排名 JSON 文件列表。"
    )
    parser.add_argument(
        "--link_map",
        type=str,
        help="link 模式下的人工标注文件，描述哪些 global_id 属于同一个真实人物。"
    )
    parser.add_argument(
        "--link_output",
        type=str,
        default="analysis_runs/linked_heights.json",
        help="link 模式导出的全局排名 JSON 文件路径。"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="2503_videos",
        help="dataset_rank 模式下的数据集根目录（子文件夹为不同人）。"
    )
    parser.add_argument(
        "--export_pairs",
        action="store_true",
        help="dataset_rank 模式下额外导出 pairwise 标签 JSON。"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="dataset_rank 模式下保存带检测框的视频。"
    )
    parser.add_argument(
        "--export_tracks",
        action="store_true",
        help="dataset_rank 模式下导出 bbox 轨迹 JSON。"
    )
    parser.add_argument(
        "--camera_filter",
        type=str,
        default=None,
        help="dataset_rank 模式下仅处理指定摄像头组（如 300cm_inside）。"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=80,
        help="dataset_rank 模式下网格划分大小（值越大网格越细）。"
    )
    parser.add_argument(
        "--jump_ratio",
        type=float,
        default=0.15,
        help="dataset_rank / tracks_rank 模式下帧间高度跳变过滤阈值（相对比例）。"
    )
    parser.add_argument(
        "--jump_window",
        type=int,
        default=5,
        help="dataset_rank / tracks_rank 模式下连续变化检测窗口大小（基于已保留帧）。"
    )
    parser.add_argument(
        "--jump_cum_ratio",
        type=float,
        default=0.6,
        help="dataset_rank / tracks_rank 模式下连续变化累计幅度阈值（相对比例）。"
    )
    parser.add_argument(
        "--primary_track_only",
        action="store_true",
        help="dataset_rank 模式下每个视频只保留跨越最久且帧数最多的 track_id。",
    )
    parser.add_argument(
        "--tracks_json",
        type=str,
        default=None,
        help="tracks_rank 模式下的 bbox 轨迹 JSON 文件。"
    )
    parser.add_argument(
        "--tracks_output",
        type=str,
        default="analysis_runs/tracks_rank.json",
        help="tracks_rank 模式输出的排名 JSON 路径。"
    )
    parser.add_argument(
        "--tracks_grid_size",
        type=int,
        default=50,
        help="tracks_rank 模式网格大小。"
    )
    parser.add_argument(
        "--tracks_bbox_format",
        type=str,
        default="xywh",
        choices=["xywh", "xyxy"],
        help="tracks_rank 模式下 bbox 格式。"
    )
    parser.add_argument(
        "--tracks_video_width",
        type=float,
        default=None,
        help="tracks_rank 模式下缺失 video_width 时的默认值。"
    )
    parser.add_argument(
        "--tracks_video_height",
        type=float,
        default=None,
        help="tracks_rank 模式下缺失 video_height 时的默认值。"
    )
    parser.add_argument(
        "--tracks_global_id_prefix",
        type=str,
        default="track",
        help="tracks_rank 模式下缺失 global_id 时的前缀。"
    )
    parser.add_argument(
        "--labels_root",
        type=str,
        default="/Users/yiding/Downloads/processed_labels",
        help="labels_rank 模式下的 processed_labels 根目录。"
    )
    parser.add_argument(
        "--labels_video_width",
        type=float,
        default=None,
        help="labels_rank 模式下视频宽度（像素）。"
    )
    parser.add_argument(
        "--labels_video_height",
        type=float,
        default=None,
        help="labels_rank 模式下视频高度（像素）。"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'link':
        run_manual_link_mode(args.link_files, args.link_map, args.link_output)
        return
    if args.mode == 'dataset_rank':
        model = YOLO(args.model)
        run_dataset_rank(args, model)
        return
    if args.mode == 'tracks_rank':
        run_tracks_rank(
            tracks_json=args.tracks_json,
            output_path=args.tracks_output,
            grid_size=args.tracks_grid_size,
            jump_ratio=args.jump_ratio,
            jump_window=args.jump_window,
            jump_cum_ratio=args.jump_cum_ratio,
            bbox_format=args.tracks_bbox_format,
            default_width=args.tracks_video_width,
            default_height=args.tracks_video_height,
            global_id_prefix=args.tracks_global_id_prefix,
        )
        return
    if args.mode == 'labels_rank':
        run_labels_rank(args)
        return

    if not args.cam:
        raise SystemExit("--cam 参数在非 link/dataset_rank/tracks_rank 模式下是必须的。")

    model = YOLO(args.model)
    if args.mode == 'pre_analyze':
        pre_analyze_ids(model, args.source, args.min_frames)
        return

    run_video_flow(args, model)


if __name__ == "__main__":
    main()
