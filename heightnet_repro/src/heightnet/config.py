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
    use_tensorboard: bool = True
    log_interval: int = 20
    tb_num_images: int = 4


@dataclass
class LossConfig:
    silog_lambda: float
    consistency_weight: float
    consistency_warmup_epochs: int
    pairwise_weight: float = 0.0
    pairwise_enabled: bool = False
    pairwise_json: str = ""
    pairwise_warmup_epochs: int = 0


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
class RuntimeSegConfig:
    enabled: bool
    model_path: str
    conf: float
    iou: float
    imgsz: int
    strict_native: bool


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
    runtime_seg: RuntimeSegConfig


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

    loss_raw = dict(raw["loss"])
    if loss_raw.get("pairwise_json"):
        loss_raw["pairwise_json"] = _to_abs(workspace, loss_raw["pairwise_json"])

    return Config(
        seed=raw["seed"],
        device=raw["device"],
        paths=paths,
        train=TrainConfig(**raw["train"]),
        loss=LossConfig(**loss_raw),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        eval=EvalConfig(**raw["eval"]),
        runtime_seg=RuntimeSegConfig(**raw.get("runtime_seg", {
            "enabled": False,
            "model_path": "",
            "conf": 0.25,
            "iou": 0.7,
            "imgsz": 640,
            "strict_native": True,
        })),
    )
