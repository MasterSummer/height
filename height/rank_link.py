import os
from collections import defaultdict
from itertools import combinations
import json
import numpy as np
from loguru import logger

from rank_core import derive_camera_id, load_heights_from_file


def compute_camera_scales(manual_groups, id_to_camera, height_map):
    """
    根据人工标注组中跨摄像头的同一人约束，求出每个摄像头的全局缩放系数。
    返回 (scales, reference_camera, used_pairs)
    """
    cameras = set(id_to_camera.values())
    if len(cameras) <= 1:
        return {cam: 1.0 for cam in cameras}, None, []

    equations = []
    used_pairs = []
    for label, members in manual_groups.items():
        entries = []
        for gid in members:
            cam = id_to_camera.get(gid)
            height_val = height_map.get(gid)
            if cam is None or height_val is None or height_val <= 0:
                continue
            entries.append((cam, float(height_val), gid))
        if len(entries) < 2:
            continue
        for (cam_a, h_a, gid_a), (cam_b, h_b, gid_b) in combinations(entries, 2):
            if cam_a == cam_b:
                continue
            rhs = np.log(h_b) - np.log(h_a)
            equations.append((cam_a, cam_b, rhs))
            used_pairs.append({
                "label": label,
                "cam_a": cam_a,
                "cam_b": cam_b,
                "gid_a": gid_a,
                "gid_b": gid_b,
                "ratio": h_a / h_b
            })

    constrained_cams = sorted({cam for cam_a, cam_b, _ in equations for cam in (cam_a, cam_b)})
    if len(constrained_cams) <= 1:
        logger.warning("人工标注未提供跨摄像头配对或仅涉及单个摄像头，无法估计缩放系数，将默认所有摄像头系数为1.0。")
        return {cam: 1.0 for cam in cameras}, None, used_pairs

    cam_index = {cam: idx for idx, cam in enumerate(constrained_cams)}
    num_eqs = len(equations)
    A = np.zeros((num_eqs + 1, len(constrained_cams)))
    b = np.zeros(num_eqs + 1)

    for row, (cam_a, cam_b, rhs) in enumerate(equations):
        A[row, cam_index[cam_a]] = 1.0
        A[row, cam_index[cam_b]] = -1.0
        b[row] = rhs

    ref_camera = constrained_cams[0]
    A[num_eqs, cam_index[ref_camera]] = 1.0
    b[num_eqs] = 0.0

    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    scales = {cam: float(np.exp(solution[cam_index[cam]])) for cam in constrained_cams}

    for cam in cameras:
        if cam not in scales:
            scales[cam] = 1.0

    return scales, ref_camera, used_pairs


def load_manual_links(filepath):
    """
    允许三种格式的人工标注文件：
    1) { "videoA_1": "Person001", "videoB_5": "Person001" }
    2) { "Person001": ["videoA_1", "videoB_5"] }
    3) [ {"label": "Person001", "members": ["videoA_1", ...]} ]
    返回 {label: set(global_ids)}
    """
    if not filepath:
        return {}
    if not os.path.exists(filepath):
        logger.error(f"指定的人工标注文件不存在: {filepath}")
        return {}
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as exc:
        logger.error(f"读取人工标注文件失败: {filepath}，错误: {exc}")
        return {}

    link_groups = defaultdict(set)

    def register(label, members):
        if not label:
            return
        for m in members:
            if isinstance(m, str):
                link_groups[label].add(m)
            else:
                logger.warning(f"忽略无效的成员标注: {m}")

    if isinstance(data, dict):
        if data and all(isinstance(v, str) for v in data.values()):
            for gid, label in data.items():
                register(label, [gid])
        elif data and all(isinstance(v, (list, tuple)) for v in data.values()):
            for label, members in data.items():
                register(label, members)
        else:
            logger.error("无法解析人工标注文件格式，请参照文档提供正确格式。")
            return {}
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                logger.warning(f"忽略非字典的标注项: {item}")
                continue
            label = item.get('label') or item.get('name')
            members = item.get('members') or item.get('ids') or []
            register(label, members)
    else:
        logger.error("人工标注文件必须是 JSON 对象或数组。")
        return {}

    return {label: sorted(members) for label, members in link_groups.items()}


def aggregate_linked_profiles(all_heights, manual_groups):
    """
    根据人工标注的同一人集合聚合身高，返回 (linked_profiles, unlinked_entries, missing_ids)
    """
    linked_profiles = {}
    linked_ids = set()
    missing_ids = []

    for label, members in manual_groups.items():
        member_heights = {}
        values = []
        for gid in members:
            height_val = all_heights.get(gid)
            member_heights[gid] = height_val
            linked_ids.add(gid)
            if height_val is None:
                missing_ids.append(gid)
            else:
                values.append(height_val)
        mean_height = float(np.mean(values)) if values else None
        linked_profiles[label] = {
            "mean_height": mean_height,
            "member_count": len(members),
            "members": member_heights
        }

    unlinked_entries = {gid: h for gid, h in all_heights.items() if gid not in linked_ids}
    return linked_profiles, unlinked_entries, missing_ids


def run_manual_link_mode(link_files, link_map_path, output_path):
    if not link_files:
        logger.error("link 模式需要通过 '--link_files' 指定至少一个摄像头排名文件。")
        return
    if not link_map_path:
        logger.error("link 模式需要提供 '--link_map' 人工标注文件。")
        return

    logger.info("开始读取各摄像头的排名 JSON...")
    combined_heights = {}
    id_to_camera = {}
    for file_path in link_files:
        data = load_heights_from_file(file_path)
        if not data:
            logger.warning(f"文件 {file_path} 中没有可用的排名数据。")
            continue
        logger.info(f"读取 {file_path}: {len(data)} 条记录。")
        cam_id = derive_camera_id(file_path)
        for gid, height in data.items():
            if gid in combined_heights and combined_heights[gid] != height:
                logger.warning(f"重复的 global_id {gid} 检测到不同数值，将采用最新值。")
            combined_heights[gid] = height
            id_to_camera[gid] = cam_id

    if not combined_heights:
        logger.error("没有任何可供链接的排名数据，link 模式结束。")
        return

    manual_groups = load_manual_links(link_map_path)
    if not manual_groups:
        logger.error("未能从人工标注文件中读取到有效的链接关系。")
        return

    camera_scales, ref_camera, used_pairs = compute_camera_scales(manual_groups, id_to_camera, combined_heights)
    if ref_camera:
        logger.info(f"以 '{ref_camera}' 作为尺度参考摄像头。")

    logger.info("摄像头缩放系数：")
    for cam, scale in sorted(camera_scales.items()):
        logger.info(f"  - {cam}: x {scale:.4f}")

    adjusted_heights = {}
    for gid, height in combined_heights.items():
        cam = id_to_camera.get(gid)
        scale = camera_scales.get(cam, 1.0)
        adjusted_heights[gid] = height * scale

    linked_profiles, unlinked_entries, missing_ids = aggregate_linked_profiles(adjusted_heights, manual_groups)

    if missing_ids:
        logger.warning(f"以下 global_id 在排名数据中未找到，将在结果中以 null 显示: {missing_ids}")

    ranked_profiles = sorted(
        linked_profiles.items(),
        key=lambda item: (item[1]['mean_height'] if item[1]['mean_height'] is not None else -np.inf),
        reverse=True
    )

    logger.info("=" * 30 + " 手动链接后的全局排名 " + "=" * 30)
    logger.info(f"{'排名':<5} | {'标签':<20} | {'平均身高':<12} | {'成员数':<8}")
    logger.info("-" * 70)
    for idx, (label, profile) in enumerate(ranked_profiles, start=1):
        mean_height = profile['mean_height']
        mean_str = f"{mean_height:.2f}" if mean_height is not None else "N/A"
        logger.info(f"{idx:<5} | {label:<20} | {mean_str:<12} | {profile['member_count']:<8}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_payload = {
        "linked_profiles": linked_profiles,
        "unlinked_entries": unlinked_entries,
        "missing_ids": missing_ids,
        "camera_scales": camera_scales,
        "scale_reference_camera": ref_camera,
        "scale_pairs_used": used_pairs
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_payload, f, indent=4, ensure_ascii=False)
    logger.success(f"手动链接后的结果已保存到: {output_path}")
