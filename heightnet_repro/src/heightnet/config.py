from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class PathsConfig:
    train_manifest: str
    val_manifest: str
    test_manifest: str
    output_dir: str


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    use_amp: bool


@dataclass
class LossConfig:
    silog_lambda: float
    consistency_weight: float
    consistency_warmup_epochs: int


@dataclass
class DataConfig:
    image_size: List[int]
    normalize_rgb: bool
    use_pair_consistency: bool
    max_matches: int


@dataclass
class ModelConfig:
    name: str
    base_channels: int


@dataclass
class EvalConfig:
    save_visualizations: bool
    vis_limit: int


@dataclass
class Config:
    seed: int
    device: str
    paths: PathsConfig
    train: TrainConfig
    loss: LossConfig
    data: DataConfig
    model: ModelConfig
    eval: EvalConfig


def _to_abs(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(path))
    workspace = os.path.dirname(base_dir)

    paths = PathsConfig(
        train_manifest=_to_abs(workspace, raw["paths"]["train_manifest"]),
        val_manifest=_to_abs(workspace, raw["paths"]["val_manifest"]),
        test_manifest=_to_abs(workspace, raw["paths"]["test_manifest"]),
        output_dir=_to_abs(workspace, raw["paths"]["output_dir"]),
    )

    return Config(
        seed=raw["seed"],
        device=raw["device"],
        paths=paths,
        train=TrainConfig(**raw["train"]),
        loss=LossConfig(**raw["loss"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        eval=EvalConfig(**raw["eval"]),
    )
