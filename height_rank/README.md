# 单摄像头行人身高排名学习（仅 bbox 轨迹）

本工程实现：
1. bbox 分支：`bbox -> BBoxEncoder`。
2. 可选视频分支：`video frame sequence -> VideoEncoder`。
3. 两路特征拼接后送入 `TemporalEncoder`，得到时序特征 `h_t`。
4. `z_t^H = g_H(h_t)` 的帧级身高表征，视频级 `z_v^H = masked mean(z_t^H)`。
5. 结构化解码器重构 `log(w,h)`（学习透视/位置导致的尺度变化）。
6. 使用 pairwise logistic ranking loss 做监督排名。
7. 两阶段训练：阶段 A（`recon + frame consistency`），阶段 B（联合 `+ ranking`）。

## 数据组织

默认路径（见 `configs/default.yaml`）：
- bbox 轨迹：`height/processed_labels`
- 排名标签：`height/labels_rank`
- 视频目录：`data.video_dir`

支持两种标签格式：
1. 单样本标签：`height/labels_rank/{sample_id}.json`，包含 `height_rank`。
2. 相机级标签：`height/labels_rank/camera_*_rank.json`，包含 `ranking` 列表。  
   对应 bbox 文件名中应包含如 `300cm_inside/350cm_outside/400cm_slantside` 等 camera 字段。

`processed_labels` 支持：
- `json`：`{"frames":[{"bbox":[x,y,w,h]}, ...]}`
- `txt/csv`：支持 MOT 风格 `[frame,id,x,y,w,h,...]` 或直接 `[x,y,w,h]`

视频分支：
- 当 `video.enabled=false` 时，仅使用 bbox 分支。
- 当 `video.enabled=true` 时，数据集会按 `video_dir/<person_id>/<sample_id>.(mp4|mov|avi|mkv)` 查找对应视频，并按采样后的 `frame_id` 读取视频帧。

## 训练

```bash
cd height_rank
python train.py --config configs/default.yaml
```

训练输出：
- TensorBoard 日志：`train.log_dir`
- 最优模型：`train.save_dir/best.pt`

## 评估

```bash
cd height_rank
python eval.py --config configs/default.yaml
```

评估指标：
- Pairwise Accuracy
- Spearman
- Kendall
- 帧稳定性：`std(s_t)` 与 `mean ||z_t^H - z_v^H||`
- 重构误差：`L1(log(w,h))`

## 已修复问题（num_samples=0）

原错误 `ValueError: num_samples=0` 由 DataLoader 在空数据集上启用 `shuffle=True` 触发。  
现已修复为：
1. 数据集构建逻辑支持当前 `processed_labels/labels_rank` 实际格式。
2. 在构造 DataLoader 前显式检查样本数，并给出清晰报错路径信息。
3. 当 `train_list/test_list` 为空时自动按 `val_ratio` 划分训练/验证集。

## 快速复现实验建议

1. 先用默认配置跑通：`python train.py --config configs/default.yaml`
2. 若使用自定义列表，确保 `train_list/test_list` 里的 ID 与 `person_id` 或样本名匹配。
3. 若评估集为空，先检查 `test_list` 与 `camera_*_rank.json` 是否对应当前 bbox 文件名。
