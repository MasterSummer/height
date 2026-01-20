import os
import json
import pandas as pd
from loguru import logger

from rank_core import add_grid_columns, ensure_global_id, filter_jump_frames, solve_global_heights_and_factors


def _apply_bbox_fallback(df, bbox_format):
    if 'bbox' not in df.columns:
        return df
    if 'x_center' in df.columns and 'y_center' in df.columns and 'width' in df.columns and 'height' in df.columns:
        return df

    def _from_bbox(b):
        if b is None or len(b) != 4:
            return None
        if bbox_format == "xyxy":
            x1, y1, x2, y2 = b
            w = x2 - x1
            h = y2 - y1
            xc = x1 + w / 2.0
            yc = y1 + h / 2.0
        else:
            x1, y1, w, h = b
            xc = x1 + w / 2.0
            yc = y1 + h / 2.0
        return xc, yc, w, h

    xc, yc, w, h = [], [], [], []
    for bbox in df['bbox'].tolist():
        result = _from_bbox(bbox)
        if result is None:
            xc.append(None)
            yc.append(None)
            w.append(None)
            h.append(None)
        else:
            _xc, _yc, _w, _h = result
            xc.append(_xc)
            yc.append(_yc)
            w.append(_w)
            h.append(_h)
    df['x_center'] = df.get('x_center', pd.Series(xc))
    df['y_center'] = df.get('y_center', pd.Series(yc))
    df['width'] = df.get('width', pd.Series(w))
    df['height'] = df.get('height', pd.Series(h))
    return df


def run_tracks_rank(tracks_json, output_path, grid_size, jump_ratio, jump_window, jump_cum_ratio, bbox_format, default_width, default_height, global_id_prefix):
    if not tracks_json:
        logger.error("tracks_rank 模式需要提供 --tracks_json")
        return
    if not os.path.exists(tracks_json):
        logger.error(f"tracks_json 文件不存在: {tracks_json}")
        return

    with open(tracks_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        logger.error("tracks_json 中没有任何记录。")
        return

    df_raw = pd.DataFrame(records)
    df_raw = _apply_bbox_fallback(df_raw, bbox_format=bbox_format)
    if 'video_width' not in df_raw.columns:
        if default_width is None:
            raise ValueError("tracks_json 缺少 video_width，需通过 --tracks_video_width 提供。")
        df_raw['video_width'] = float(default_width)
    if 'video_height' not in df_raw.columns:
        if default_height is None:
            raise ValueError("tracks_json 缺少 video_height，需通过 --tracks_video_height 提供。")
        df_raw['video_height'] = float(default_height)

    if 'frame' not in df_raw.columns:
        df_raw['frame'] = 0
    if 'track_id' not in df_raw.columns and 'person_id' in df_raw.columns:
        df_raw['track_id'] = df_raw['person_id']

    df_raw = ensure_global_id(df_raw, prefix=global_id_prefix)
    df_raw = filter_jump_frames(df_raw, jump_ratio, jump_window, jump_cum_ratio)
    df_raw = add_grid_columns(df_raw, grid_size=grid_size)

    final_heights, _ = solve_global_heights_and_factors(df_raw, history_cg=None)
    if not final_heights:
        logger.error("tracks_rank 无法得到有效身高结果。")
        return

    ranked = sorted(final_heights.items(), key=lambda item: (-item[1], str(item[0])))
    ranking = [gid for gid, _ in ranked]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "ranking": ranking,
        "scores": {gid: float(score) for gid, score in ranked},
        "source_tracks": tracks_json,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    logger.success(f"已保存轨迹排名结果: {output_path}")
