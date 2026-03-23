# HeightNet Reproduction Skeleton

This folder provides a minimal, runnable baseline for reproducing the HeightNet workflow:

1. Generate height labels from Depth Anything V2 depth maps (optionally with segmentation).
2. Precompute cross-frame keypoint matches for consistency loss.
3. Build train/val/test manifests.
4. Train a monocular height estimator.
5. Evaluate RMSE + pairwise ranking accuracy.

## Project Structure

- `configs/default.yaml`: default training config.
- `src/heightnet/`: dataset, model, losses, metrics, utility code.
- `tools/precompute_height_labels.py`: create `height/*.npy` and `valid_mask/*.npy`.
- `tools/precompute_matches.py`: create ORB+RANSAC pair match cache.
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
    seg/<sequence_id>/<frame>.png
  val/
    ...
  test/
    ...
```

Default label-generation behavior:

- Corrected formula: `h = C_h * (D_b - D_f) / D_b`.
- If segmentation exists: `seg == --ground-class-id` is treated as ground class.
- If segmentation does not exist: `D_b` is estimated from bottom image band depth statistics.

## Quick Start

```bash
cd /Users/yiding/code/gait/heightnet_repro
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Build labels
python tools/precompute_height_labels.py --data-root /path/to/data_root --split train --camera-height-m 6.4
python tools/precompute_height_labels.py --data-root /path/to/data_root --split val --camera-height-m 6.4
python tools/precompute_height_labels.py --data-root /path/to/data_root --split test --camera-height-m 6.4

# 2) Build pair matches
python tools/precompute_matches.py --data-root /path/to/data_root --split train --out-root /path/to/matches

# 3) Build manifests
python tools/build_manifest.py --data-root /path/to/data_root --split train --matches-root /path/to/matches --out data/train_manifest.csv
python tools/build_manifest.py --data-root /path/to/data_root --split val --matches-root /path/to/matches --out data/val_manifest.csv
python tools/build_manifest.py --data-root /path/to/data_root --split test --matches-root /path/to/matches --out data/test_manifest.csv

# 4) Train
python train.py --config configs/default.yaml

# 5) Evaluate
python evaluate.py --config configs/default.yaml --checkpoint runs/default/checkpoint_best.pt --rank-dir /Users/yiding/code/gait/2503_test_rank

# 6) Optional: export explicit pairwise labels from 2503_test_rank
python tools/generate_pairwise_rank_from_2503.py --rank-dir /Users/yiding/code/gait/2503_test_rank --out-dir data/pairwise_rank
```

## Notes

- This is a pragmatic scaffold, not the exact original HeightNet implementation.
- Paper details that are unclear are implemented with conservative defaults:
  - `SiLog lambda = 0.85`
  - `consistency_weight = 0.1`
  - warmup 5 epochs before consistency loss
  - ORB + BFMatcher + RANSAC for cross-frame correspondences
- To align closer to paper-reported numbers, you should replace `HeightNetTiny` with the LapDepth-style encoder-decoder and tune data generation details.
