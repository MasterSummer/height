import os
import sys
import json
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. 配置区
VIDEO_ROOT = "/data2/zengxh/6python9/6data9/video_2503/"
EXCLUDE_PEOPLE = set([
    "0313_man2","0313_man3","0318_woman3","0319_woman1","0319_woman2","0320_woman3","0320_woman4",
    "0321_woman2","0324_man1","0324_woman3","0324_woman4","0326_woman1","0326_woman2","0326_woman4",
    "0327_woman1","0327_woman2"
])
MAX_VIDEOS_PER_CAMERA = 5
DEIMV2_CONFIG = "DEIMv2/configs/deimv2/deimv2_dinov3_s_coco.yml"
DEIMV2_MODEL = "DEIMv2/deimv2_best.pth"
LOG_PATH = "deimv2_train.log"
CG_PLOT_PATH = "deimv2_cg.png"
TRACKS_JSON = "deimv2_bbox_tracks.json"

# 2. 加载DEIMv2模型
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../DEIMv2")))
from engine.core import YAMLConfig
cfg = YAMLConfig(DEIMV2_CONFIG)
model = cfg.model
state_dict = torch.load(DEIMV2_MODEL, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# 3. 遍历训练集
person_dirs = [d for d in os.listdir(VIDEO_ROOT) if os.path.isdir(os.path.join(VIDEO_ROOT, d)) and d not in EXCLUDE_PEOPLE]

tracks_by_camera = defaultdict(list)
log_lines = []

for person in tqdm(person_dirs, desc="People"):
    person_dir = os.path.join(VIDEO_ROOT, person)
    for cam in sorted(os.listdir(person_dir)):
        cam_dir = os.path.join(person_dir, cam)
        if not os.path.isdir(cam_dir):
            continue
        videos = [f for f in sorted(os.listdir(cam_dir)) if f.lower().endswith('.mp4')]
        for video in videos[:MAX_VIDEOS_PER_CAMERA]:
            video_path = os.path.join(cam_dir, video)
            # 4. 推理，收集bbox轨迹
            # 这里只做伪代码，需根据DEIMv2实际推理接口调整
            try:
                # 假设有一个推理函数deimv2_infer返回bbox列表
                # bboxes = deimv2_infer(model, video_path)
                bboxes = []  # TODO: 替换为实际推理
                tracks_by_camera[cam].append({
                    "person": person,
                    "video": video,
                    "bboxes": bboxes
                })
                log_lines.append(f"OK: {person}/{cam}/{video}")
            except Exception as e:
                log_lines.append(f"FAIL: {person}/{cam}/{video} {e}")

# 5. 保存bbox轨迹
with open(TRACKS_JSON, 'w') as f:
    json.dump(tracks_by_camera, f, indent=2)

# 6. 计算Cg和排名（调用你现有的rank/height相关代码）
# 这里只做伪代码，需根据你的rank_core/rank_dataset实际接口调整
# from rank_core import solve_global_heights_and_factors
# heights, cg = solve_global_heights_and_factors(...)
# ...
# plt.savefig(CG_PLOT_PATH)

# 7. 保存日志
with open(LOG_PATH, 'w') as f:
    for line in log_lines:
        f.write(line + '\n')

print(f"已保存: {TRACKS_JSON}, {LOG_PATH}, {CG_PLOT_PATH} (Cg图需在6步补全)")
