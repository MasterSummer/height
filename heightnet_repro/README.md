# HeightNet Reproduction Skeleton

This folder provides a minimal, runnable baseline for reproducing the HeightNet workflow:

1. Generate height/valid_mask from depth + camera background depth.
2. Build train/val/test manifests.
3. Train/evaluate with runtime segmentation (person mask generated on the fly).
4. Optimize not only dense height regression, but also person-level pairwise ranking.

## Project Structure

- `configs/default.yaml`: default training config.
- `src/heightnet/`: dataset, model, losses, metrics, utility code.
- `tools/precompute_height_labels.py`: create `height/*.npy` and `valid_mask/*.npy`.
- `tools/prepare_end2end_data.py`: one-command data prep for all splits.
- `tools/build_manifest.py`: build csv manifests.
- `tools/generate_pairwise_rank_from_2503.py`: generate pairwise labels from `2503_test_rank`.
- `train.py`: training entrypoint.
- `evaluate.py`: testing + visualization.

## Expected Data Layout

```text
data_root/
  train/
    rgb/<sequence_id>/<frame>.png
    depth/<sequence_id>/<frame>.npy  # from Depth Anything V2
    person_mask/<sequence_id>/<frame>.npy   # optional if runtime_seg.enabled=true
  val/
    ...
  test/
    ...
```

Default label-generation behavior:

- Corrected formula: `h = C_h * (D_b - D_f) / D_b`.
- `D_b` must come from camera background depth map (`--bg-depth-root`).
- `person_mask` can be generated at runtime by segmentation model (recommended).
- `pairwise_rank` is used in training via `loss.pairwise_json` and in evaluation via `--rank-dir`.

## Quick Start

```bash
cd /Users/yiding/code/gait/height/heightnet_repro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) End-to-end data prep (height + valid_mask + manifests)
python tools/prepare_end2end_data.py \
  --data-root /path/to/data_root \
  --bg-depth-root /path/to/2503_test_bg_depthmap \
  --person-mask-source depth \
  --camera-height-m 6.4 \
  --out-dir data \
  --matches-root data/matches

# 2) Train
# set runtime_seg.model_path in configs/default.yaml first (e.g. yolov8n-seg.pt)
# pairwise supervision defaults to data/pairwise_rank/all_pairs.json
python train.py --config configs/default.yaml

# 3) Evaluate
python evaluate.py --config configs/default.yaml --checkpoint runs/default/checkpoint_best.pt --rank-dir /Users/yiding/code/gait/2503_test_rank

# 4) Optional: export explicit pairwise labels from 2503_test_rank
python tools/generate_pairwise_rank_from_2503.py --rank-dir /Users/yiding/code/gait/2503_test_rank --out-dir data/pairwise_rank
```

## One-Click Pipeline (videos -> depth -> height -> manifest)

If you want a single command from raw videos to manifests, use:

```bash
python tools/prepare_videos_to_manifest.py \
  --video-root /path/to/video_root \
  --data-root /path/to/data_root \
  --bg-depth-root /path/to/bg_depth_root \
  --camera-height-m 6.4 \
  --depthanything-root /path/to/Depth-Anything-V2 \
  --depth-encoder vitl \
  --depth-checkpoint /path/to/depth_anything_v2_vitl.pth \
  --person-mask-source seg \
  --person-seg-model /path/to/yolov8n-seg.pt \
  --out-dir /path/to/heightnet_repro/data \
  --matches-root /path/to/heightnet_repro/data/matches
```

This command orchestrates:
1) `prepare_midrun_dataset.py` (video -> rgb)  
2) Depth Anything V2 inference (rgb -> depth .npy)  
3) `prepare_end2end_data.py` (depth -> height/valid_mask + manifests)

## One-Click From Existing RGB (rgb -> depth -> labels -> manifests -> train)

If you already have `data_root/{train,val,test}/rgb` and camera background depth maps:

```bash
python tools/from_rgb_one_click.py \
  --data-root /path/to/heightnet_data_midrun \
  --bg-depth-root /path/to/2503_test_bg_depthmap \
  --camera-height-m 6.4 \
  --depthanything-root /path/to/Depth-Anything-V2 \
  --depth-encoder vitl \
  --depth-checkpoint /path/to/depth_anything_v2_vitl.pth \
  --person-mask-source depth \
  --run-name auto_from_rgb
```

Notes:
- `person-mask-source depth` only affects offline data prep.
- Training/evaluation `person_mask` is generated online when `runtime_seg.enabled=true`.
- Use `--prepare-only` if you want to stop before `train.py`.
- Add `--launcher torchrun --nproc-per-node 4` for 4-GPU training.

### 4-GPU Training (A100 x4)

Directly train with DDP:

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/auto_from_rgb.yaml
```

Or one-click from RGB:

```bash
python tools/from_rgb_one_click.py \
  --data-root /path/to/heightnet_data_midrun \
  --bg-depth-root /path/to/2503_test_bg_depthmap \
  --camera-height-m 6.4 \
  --depthanything-root /path/to/Depth-Anything-V2 \
  --depth-encoder vitl \
  --depth-checkpoint /path/to/depth_anything_v2_vitl.pth \
  --person-mask-source depth \
  --run-name auto_from_rgb_4gpu \
  --launcher torchrun \
  --nproc-per-node 4
```

## Notes

- This is a pragmatic scaffold, not the exact original HeightNet implementation.
- Paper details that are unclear are implemented with conservative defaults:
  - `SiLog lambda = 0.85`
  - `consistency_weight = 0.1`
  - `pairwise_weight = 0.5`
  - warmup 5 epochs before consistency loss
  - person-mask max-score consistency across adjacent frames
- To align closer to paper-reported numbers, you should replace `HeightNetTiny` with the LapDepth-style encoder-decoder and tune data generation details.
