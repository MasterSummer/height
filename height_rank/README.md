# 单摄像头行人身高排名学习

## 项目简介
本项目实现基于单摄像头视频的行人身高排名学习，输入为每帧 bbox，输出身高表征并进行排名。

## 数据格式
- 视频数据集：`tmp_test_videos/`，每个文件夹为一个视频，文件夹名为视频ID。
- bbox标注：`height/processed_labels/{video_id}.json`，格式如下：
  ```json
  {
    "frames": [
      {"frame_idx": 0, "bbox": [x, y, w, h]},
      {"frame_idx": 1, "bbox": [x, y, w, h]},
      ...
    ]
  }
  ```
- 排名label：`height/labels_rank/{video_id}.json`，格式如下：
  ```json
  {
    "height_rank": 3  # 数字越大表示身高越高
  }
  ```

## 训练
```bash
python train.py --config configs/default.yaml
```

## 评估
```bash
ython eval.py --config configs/default.yaml
```

## 主要模块
- `dataset.py`：数据加载与处理
- `models.py`：模型结构
- `losses.py`：损失函数
- `train.py`：训练流程
- `eval.py`：评估指标

## 复现实验
1. 准备数据，确保 `tmp_test_videos/`、`height/processed_labels/`、`height/labels_rank/` 按上述格式组织。
2. 配置超参（可修改 `configs/default.yaml`）。
3. 运行训练与评估脚本。

## 依赖
- Python >= 3.8
- PyTorch >= 1.10
- numpy, pandas, scipy

## 结果输出
训练完成后会保存最优 checkpoint，并打印各评估指标（Pairwise Accuracy、Spearman、Kendall、帧稳定性、重构误差）。
