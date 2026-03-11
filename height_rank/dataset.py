import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class HeightRankDataset(Dataset):
    def __init__(self, video_dir, bbox_dir, rank_dir, T=32, list_path=None):
        if list_path is not None:
            with open(list_path, 'r') as f:
                video_ids = [line.strip() for line in f if line.strip()]
        else:
            video_ids = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
        self.video_ids = video_ids
        self.bbox_dir = bbox_dir
        self.rank_dir = rank_dir
        self.T = T
        self.samples = []
        for vid in self.video_ids:
            bbox_path = os.path.join(bbox_dir, f"{vid}.json")
            rank_path = os.path.join(rank_dir, f"{vid}.json")
            if os.path.exists(bbox_path) and os.path.exists(rank_path):
                self.samples.append((vid, bbox_path, rank_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, bbox_path, rank_path = self.samples[idx]
        with open(bbox_path, 'r') as f:
            bbox_data = json.load(f)
        bboxes = [f['bbox'] for f in bbox_data['frames']]
        bboxes = np.array(bboxes)
        if len(bboxes) < self.T:
            pad = np.zeros((self.T - len(bboxes), 4))
            bboxes = np.concatenate([bboxes, pad], axis=0)
        else:
            bboxes = bboxes[:self.T]
        mask = np.zeros(self.T)
        mask[:min(len(bboxes), self.T)] = 1
        with open(rank_path, 'r') as f:
            rank_data = json.load(f)
        rank_label = rank_data['height_rank']
        return {
            'video_id': vid,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'rank_label': torch.tensor(rank_label, dtype=torch.float32)
        }

def get_dataloader(video_dir, bbox_dir, rank_dir, batch_size, T, shuffle=True, list_path=None):
    dataset = HeightRankDataset(video_dir, bbox_dir, rank_dir, T, list_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class CameraHeightRankDataset(Dataset):
    def __init__(self, camera, processed_dir, labels_rank_dir, T=32):
        # 读取排名 json
        rank_path = os.path.join(labels_rank_dir, f"{camera}_rank.json")
        with open(rank_path, 'r') as f:
            rank_data = json.load(f)
        self.person_ids = rank_data['ranking']
        self.scores = rank_data['scores']
        self.camera = camera
        self.processed_dir = os.path.join(processed_dir, camera)
        self.T = T
        self.samples = []
        for pid in self.person_ids:
            bbox_path = os.path.join(self.processed_dir, f"{pid}.txt")
            if os.path.exists(bbox_path):
                self.samples.append((pid, bbox_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, bbox_path = self.samples[idx]
        # 读取 txt 格式 bbox
        df = pd.read_csv(bbox_path, sep=' ', header=None)
        bboxes = df.values
        if len(bboxes) < self.T:
            pad = np.zeros((self.T - len(bboxes), 4))
            bboxes = np.concatenate([bboxes, pad], axis=0)
        else:
            bboxes = bboxes[:self.T]
        mask = np.zeros(self.T)
        mask[:min(len(df), self.T)] = 1
        # 排名 label
        rank_label = self.person_ids.index(pid) + 1  # 排名越大身高越高
        return {
            'person_id': pid,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'rank_label': torch.tensor(rank_label, dtype=torch.float32)
        }

def get_camera_dataloader(camera, processed_dir, labels_rank_dir, batch_size, T, shuffle=True):
    dataset = CameraHeightRankDataset(camera, processed_dir, labels_rank_dir, T)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
