import os
import ast
import json
import numpy as np
from loguru import logger
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsmr


def derive_camera_id(file_path):
    """
    根据排名 JSON 文件名推断摄像头 ID。
    规则：取 basename 去掉扩展名，如果以 '_heights' 结尾则再次去掉该后缀。
    """
    base = os.path.splitext(os.path.basename(file_path))[0]
    suffix = '_heights'
    if base.endswith(suffix):
        base = base[:-len(suffix)]
    return base


def apply_correction_from_file(df, perspective_model_path):
    """
    加载一个已保存的透视校正地图，并用它来计算标准化身高和排名。
    """
    logger.info(f"正在从文件加载透视模型: {perspective_model_path}")
    try:
        with open(perspective_model_path, 'r') as f:
            correction_factors = json.load(f)
            correction_factors = {ast.literal_eval(k): v for k, v in correction_factors.items()}
    except FileNotFoundError:
        logger.error(f"错误: 透视模型文件未找到: {perspective_model_path}")
        return None

    logger.info("应用加载的透视模型进行身高标准化...")
    df['correction_factor'] = df['grid_cell'].apply(
        lambda grid: correction_factors.get(grid, 1.0)
    )
    df['normalized_height'] = df['height'] / df['correction_factor']
    final_scores = df.groupby('track_id')['normalized_height'].mean().sort_values(ascending=False)
    return final_scores.to_dict()


def solve_global_heights_and_factors(df, history_cg=None):
    """
    通过最小二乘求解所有人的标准化身高与网格校正因子。
    """
    logger.info("开始进行全局最小二乘法优化 (第三版统一尺度系统)...")

    df_filtered = df[df['height'] > 0].copy()
    if df_filtered.empty:
        logger.error("有效数据为空，无法进行计算。")
        return None, None

    df_filtered['grids_visited_count'] = df_filtered.groupby('track_id')['grid_cell'].transform('nunique')
    df_filtered['people_in_grid_count'] = df_filtered.groupby('grid_cell')['track_id'].transform('nunique')
    is_isolated = (df_filtered['grids_visited_count'] == 1) & (df_filtered['people_in_grid_count'] == 1)
    df_filtered = df_filtered[~is_isolated]

    if df_filtered.empty:
        logger.error("过滤后无数据，无法优化。")
        return None, None

    person_ids = df_filtered['global_id'].unique()
    grid_cells = df_filtered['grid_cell'].unique()
    person_map = {pid: i for i, pid in enumerate(person_ids)}
    grid_map = {g: i for i, g in enumerate(grid_cells)}

    num_persons = len(person_ids)
    num_grids = len(grid_cells)
    observations = len(df_filtered)
    num_constraints = len(history_cg) if history_cg else 0
    total_rows = observations + (1 if history_cg is None else 0) + num_constraints

    A = lil_matrix((total_rows, num_persons + num_grids), dtype=float)
    b = np.zeros(total_rows)

    df_filtered['log_h'] = np.log(df_filtered['height'])
    for i, (_, row) in enumerate(df_filtered.iterrows()):
        pid = row['global_id']
        grid = row['grid_cell']
        A[i, person_map[pid]] = 1.0
        A[i, num_persons + grid_map[grid]] = -1.0
        b[i] = row['log_h']

    row_ptr = observations
    if history_cg:
        logger.info("发现历史 C_g：将其作为硬约束加入系统。")
        for grid, cg_val in history_cg.items():
            if grid not in grid_map:
                continue
            A[row_ptr, num_persons + grid_map[grid]] = 1.0
            b[row_ptr] = np.log(cg_val)
            row_ptr += 1
        logger.info("已使用历史尺度系统，无需设锚点。")
    else:
        logger.info("没有历史 C_g，第一次运行 → 自动选择锚点。")
        ref_grid = df_filtered['grid_cell'].value_counts().idxmax()
        logger.info(f"第一次运行，将 {ref_grid} 作为尺度锚点 (C_g=1.0)")
        A[row_ptr, num_persons + grid_map[ref_grid]] = 1.0
        b[row_ptr] = 0.0
        row_ptr += 1

    logger.info("开始最小二乘求解...")
    sol = lsmr(A.tocsr(), b, damp=1e-3)[0]
    log_H = sol[:num_persons]
    log_C = sol[num_persons:]
    height_results = {pid: np.exp(log_H[person_map[pid]]) for pid in person_ids}
    factor_results = {g: np.exp(log_C[grid_map[g]]) for g in grid_cells}
    logger.info("优化完成，返回 height 和 C_g。")
    return height_results, factor_results


def pre_analyze_ids(model, source_video, min_frames_threshold):
    """
    对视频进行快速静默追踪，生成包含帧数、起止时间的详细ID报告。
    """
    logger.info("=" * 50)
    logger.info("正在进行预分析，生成详细 Track ID 报告...")
    results_generator = model.track(source=source_video, tracker="botsort.yaml", persist=True, verbose=False)

    id_info = {}
    for frame_id, r in enumerate(results_generator):
        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
            for tid in track_ids:
                if tid not in id_info:
                    id_info[tid] = {'count': 1, 'start_frame': frame_id, 'end_frame': frame_id}
                else:
                    id_info[tid]['count'] += 1
                    id_info[tid]['end_frame'] = frame_id

    if not id_info:
        logger.warning("预分析未能追踪到任何 IDs。")
        return None

    filtered_ids = {tid: info for tid, info in id_info.items() if info['count'] >= min_frames_threshold}
    sorted_ids = sorted(filtered_ids.items(), key=lambda item: item[1]['count'], reverse=True)
    logger.info("--- 预分析报告 (已过滤) ---")
    logger.info(f"过滤条件: 只显示出现 >= {min_frames_threshold} 帧的 IDs。")
    logger.info(f"视频中共有 {len(id_info)} 个原始ID，过滤后剩下 {len(sorted_ids)} 个重要ID。")
    logger.info("详细信息 (按出现次数排序):")
    for tid, info in sorted_ids:
        logger.info(
            f"  - ID: {tid: <4} | 出现帧数: {info['count']: <5} | 活跃时段: [帧 {info['start_frame']} -> 帧 {info['end_frame']}]"
        )
    logger.info("=" * 50)
    return filtered_ids


def save_correction_factors(cg_dict, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cg_str = {str(k): v for k, v in cg_dict.items()}
    with open(filepath, 'w') as f:
        json.dump(cg_str, f, indent=4)


def merge_heights(old, new):
    merged = old.copy()
    for k, v in new.items():
        merged[k] = (merged[k] + v) / 2 if k in merged else v
    return merged


def save_heights_to_file(heights, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    heights_str = {str(k): v for k, v in heights.items()}
    with open(filepath, 'w') as f:
        json.dump(heights_str, f, indent=4)


def load_heights_from_file(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning(f"历史排名文件损坏或格式错误，已忽略: {filepath}，错误信息: {exc}")
        return {}


def load_correction_factors(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r') as f:
            raw = json.load(f)
            return {ast.literal_eval(k): v for k, v in raw.items()}
    except Exception as exc:
        logger.warning(f"透视地图文件损坏或格式错误，已忽略: {filepath}，错误信息: {exc}")
        return {}


def merge_cg(old, new, alpha=0.5):
    merged = old.copy()
    for k, v in new.items():
        merged[k] = alpha * v + (1 - alpha) * merged[k] if k in merged else v
    return merged


def ensure_global_id(df, prefix="track"):
    if 'global_id' in df.columns:
        missing = df['global_id'].isna()
        if missing.any() and 'track_id' in df.columns:
            df.loc[missing, 'global_id'] = df.loc[missing, 'track_id'].apply(lambda tid: f"{prefix}_{tid}")
        return df
    if 'track_id' in df.columns:
        df['global_id'] = df['track_id'].apply(lambda tid: f"{prefix}_{tid}")
    else:
        df['global_id'] = df.index.astype(str)
    return df


def filter_jump_frames(df, jump_ratio, jump_window, jump_cum_ratio):
    if df.empty:
        return df

    def _filter_jumps(group):
        group = group.sort_values("frame")
        kept = []
        recent = []
        last_h = None
        for _, row in group.iterrows():
            h = float(row["height"])
            if last_h is None:
                kept.append(True)
                last_h = h
                recent.append(h)
                continue

            step_ratio = abs(h - last_h) / max(last_h, 1e-6)
            recent_median = np.median(recent) if recent else last_h
            window_ratio = (max(recent + [h]) - min(recent + [h])) / max(recent_median, 1e-6)

            if step_ratio <= jump_ratio and window_ratio <= jump_cum_ratio:
                kept.append(True)
                last_h = h
                recent.append(h)
                if len(recent) > jump_window:
                    recent.pop(0)
            else:
                kept.append(False)
        return group[kept]

    return df.groupby("global_id", group_keys=False).apply(_filter_jumps)


def add_grid_columns(df, grid_size, video_width=None, video_height=None):
    df['anchor_x'] = df['x_center']
    df['anchor_y'] = df['y_center'] + df['height'] / 2

    if video_width is None:
        if 'video_width' not in df.columns:
            raise ValueError("缺少 video_width，无法计算网格。")
        width = df['video_width'].astype(float)
    else:
        width = float(video_width)

    if video_height is None:
        if 'video_height' not in df.columns:
            raise ValueError("缺少 video_height，无法计算网格。")
        height = df['video_height'].astype(float)
    else:
        height = float(video_height)

    df['grid_x'] = (df['anchor_x'] / width * grid_size).astype(int).clip(0, grid_size - 1)
    df['grid_y'] = (df['anchor_y'] / height * grid_size).astype(int).clip(0, grid_size - 1)
    df['grid_cell'] = list(zip(df['grid_x'], df['grid_y']))
    return df
