import argparse
import json
import os
import sys

import cv2
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser("Preview DEIMv2 single-frame output")
    parser.add_argument("--deimv2_root", type=str, default="/data1/zhaoyd/DEIMv2", help="DEIMv2 代码目录")
    parser.add_argument("--config", type=str, required=True, help="DEIMv2 config (yml/json)")
    parser.add_argument("--weights", type=str, required=True, help="DEIMv2 weights (safetensors)")
    parser.add_argument("--video", type=str, required=True, help="视频路径")
    parser.add_argument("--frame", type=int, default=1, help="帧号（从 1 开始）")
    parser.add_argument("--resize_to", type=str, default=None, help="强制 resize 到 W,H（如 640,640）")
    return parser.parse_args()


def load_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    return frame


def resize_to_multiple_of_32(frame):
    h, w = frame.shape[:2]
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    if new_w <= 0 or new_h <= 0 or (new_w == w and new_h == h):
        return frame, (w, h)
    resized = cv2.resize(frame, (new_w, new_h))
    return resized, (w, h)


def build_model(config):
    from safetensors.torch import load_file as safetensors_load
    from engine.deim import HybridEncoder, DEIMTransformer
    from engine.deim.postprocessor import PostProcessor
    from engine.backbone import DINOv3STAs
    import torch.nn as nn

    class DEIMv2(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.backbone = DINOv3STAs(**cfg["DINOv3STAs"])
            self.encoder = HybridEncoder(**cfg["HybridEncoder"])
            self.decoder = DEIMTransformer(**cfg["DEIMTransformer"])
            self.postprocessor = PostProcessor(**cfg["PostProcessor"])

        def forward(self, x, orig_target_sizes):
            x = self.backbone(x)
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.postprocessor(x, orig_target_sizes)
            return x

    model = DEIMv2(config)
    state_dict = safetensors_load(args.weights)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("missing keys:", missing)
    if unexpected:
        print("unexpected keys:", unexpected)
    model.eval()
    return model


def describe_output(output):
    if isinstance(output, dict):
        print("output keys:", list(output.keys()))
        for k, v in output.items():
            if torch.is_tensor(v):
                print(k, v.shape, v.dtype)
            else:
                print(k, type(v))
        return
    if isinstance(output, (list, tuple)):
        print("output list len:", len(output))
        if output:
            describe_output(output[0])
        return
    print("output type:", type(output))


if __name__ == "__main__":
    args = parse_args()
    sys.path.insert(0, args.deimv2_root)

    with open(args.config, "r", encoding="utf-8") as f:
        if args.config.lower().endswith((".yml", ".yaml")):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)

    frame = load_frame(args.video, args.frame)
    orig_h, orig_w = frame.shape[:2]
    if args.resize_to:
        w, h = (int(x) for x in args.resize_to.split(","))
        frame = cv2.resize(frame, (w, h))
    else:
        frame, _ = resize_to_multiple_of_32(frame)
        h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)
    sizes = torch.tensor([[h, w]], device=device)

    with torch.no_grad():
        output = model(img, sizes)
    describe_output(output)
