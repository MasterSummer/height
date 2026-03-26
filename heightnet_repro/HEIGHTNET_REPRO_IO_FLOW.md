# HeightNet Repro: 输入输出与数据流说明

## 1. 项目目标

本项目复现的是单目高度估计（HeightNet 思路）：  
输入单帧 RGB，输出每像素高度图（相对地面高度）。

当前版本的一致性约束采用：  
同一人 `person_mask` 区域内的**最高预测高度分数**跨帧保持稳定（不使用 ORB 匹配点）。  
`person_mask` 默认在训练/推理时由分割模型在线生成。
另外，`pairwise_rank` 已接入训练主损失，用于直接监督“谁更高”。

---

## 2. 总体输入与输出

### 2.1 训练输入

- RGB 帧：`{split}/rgb/<sequence_id>/<frame>.png`
- 深度图（由 Depth Anything V2 生成）：`{split}/depth/<sequence_id>/<frame>.npy`
- 人体掩码：默认**在线生成**（可选离线缓存到 `person_mask/`）
- 摄像头背景平均深度图（必需）：  
  `<bg_depth_root>/<camera_id>/<camera_id>_avg_depth.npy`  
  例如：`2503_test_bg_depthmap/300cm_inside/300cm_inside_avg_depth.npy`

### 2.2 训练监督标签（预处理生成）

- 高度标签：`{split}/height/<sequence_id>/<frame>.npy`
- 有效像素掩码：`{split}/valid_mask/<sequence_id>/<frame>.npy`

### 2.3 核心输出

- 模型权重：`runs/<exp>/checkpoint_last.pt`、`checkpoint_best.pt`
- 测试可视化：`runs/<exp>/vis/*.png`
- 评估日志：终端打印 `RMSE` 与 `pairwise ranking accuracy`

---

## 3. 目录结构（期望）

```text
data_root/
  train/
    rgb/<sequence_id>/*.png
    depth/<sequence_id>/*.npy
    person_mask/<sequence_id>/*.npy        # 可选（在线分割模式下可不提供）
    height/<sequence_id>/*.npy             # 预处理生成
    valid_mask/<sequence_id>/*.npy         # 预处理生成
  val/
    ...
  test/
    ...
```

`heightnet_repro/data/` 下生成：

- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`

---

## 4. 关键脚本的输入输出

## 4.1 `tools/prepare_midrun_dataset.py`

输入：
- 视频目录：`<video_root>/<person_id>/*.mp4`

输出：
- 抽帧后的 `rgb` 数据集（train/val/test）
- 视频抽样列表：`midrun_20.txt`、`midrun_train.txt`、`midrun_val.txt`、`midrun_test.txt`

## 4.2 `tools/precompute_height_labels.py`

输入：
- `rgb/depth`（必须）
- `bg_depth_root`（必须）
- `camera_height_m`

输出：
- `height/*.npy`
- `valid_mask/*.npy`

公式（当前实现）：
- `h = C_h * (D_b - D_f) / D_b`
- `D_f`：像素深度；`D_b`：背景深度（来自 camera 平均背景深度图）

## 4.3 `tools/build_manifest.py`

输入：
- `data_root`（含 `rgb/height/valid_mask`）

输出：
- `*_manifest.csv`（DataLoader 索引表）

每行主要字段：
- `rgb_path`
- `height_path`
- `valid_mask_path`
- `person_mask_path`（可为空，在线分割时不使用）
- `sequence_id`
- `frame_idx`
- `person_id`
- `camera_id`

## 4.3.1 `tools/prepare_end2end_data.py`

输入：
- `data_root`
- `bg_depth_root`
- `camera_height_m` 或 `camera_height_mode`
- `person-mask-source`（`depth` 或 `seg`）

输出：
- `height/*.npy`、`valid_mask/*.npy`
- `*_manifest.csv`
- 当 `person-mask-source=seg` 时，还会离线生成 `person_mask/*.npy`

## 4.4 `train.py`

输入：
- `train_manifest.csv`、`val_manifest.csv`
- `configs/default.yaml`

输出：
- `checkpoint_last.pt`、`checkpoint_best.pt`

损失：
- 主损失：`SiLog(height_pred, height_gt)`
- 一致性损失（可开关）：  
  相邻帧中同一人 `person_mask` 区域最高高度分数的 L1 差值
- 排序损失（可开关）：
  使用 `data/pairwise_rank/all_pairs.json` 中的 `(camera, id_i, id_j, y)` 监督 batch 内人物高度分数排序

在线分割：
- 当 `configs/default.yaml` 中 `runtime_seg.enabled=true` 时，`train.py` 会调用分割模型在线生成 `person_mask`。
- 需要配置 `runtime_seg.model_path` 为分割模型（如 `yolov8n-seg.pt`）。

## 4.5 `evaluate.py`

输入：
- `test_manifest.csv`
- 训练好的 checkpoint
- `--rank-dir`（如 `2503_test_rank`）

输出：
- `RMSE`
- 每摄像头及平均 `pairwise ranking accuracy`
- 可视化图片（可选）

---

## 5. 端到端数据流

1. 视频抽帧  
`raw videos -> train/val/test/rgb`

2. 深度估计  
`rgb -> depth (Depth Anything V2)`

3. 高度标签生成  
`depth + bg_depth_map -> height + valid_mask`

4. 清单构建  
`rgb + height + valid_mask -> manifest.csv`

5. 模型训练  
`image -> model -> pred_height`  
监督：`SiLog + person-mask max consistency + pairwise rank loss`  
其中 `person_mask` 由在线分割模型从 `image_raw` 实时生成

6. 评估  
`pred_height + gt_height -> RMSE`  
`person-level score + rank json -> pairwise ranking accuracy`

---

## 6. 当前一致性约束说明（与你的需求对齐）

- 不再依赖 ORB 点匹配。
- 使用同一 sequence 的相邻帧。
- 每帧从 `person_mask` 区域提取一个标量分数（最高预测高度）。
- 最小化相邻帧分数差值，约束“同一人身高估计稳定”。

---

## 7. 常见问题

1. `manifest` 全是 0 行：  
通常是 `height/valid_mask` 未生成，或目录层级不符合预期。

2. 在线分割报错找不到模型：  
检查 `runtime_seg.model_path` 是否指向有效的 `*-seg.pt`。

3. `pairwise` 没结果：  
检查 `manifest` 里的 `person_id/camera_id` 是否能匹配 `rank-dir` 里的 JSON。
