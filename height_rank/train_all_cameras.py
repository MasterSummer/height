import os
import yaml
import torch
from dataset import get_camera_dataloader
from models import HeightRankModel
from losses import frame_consistency_loss, recon_loss, pairwise_ranking_loss

import numpy as np

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def train_camera(cfg, camera):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_camera_dataloader(
        camera=camera,
        processed_dir='height/processed_labels',
        labels_rank_dir='height/labels_rank',
        batch_size=cfg['batch_size'],
        T=cfg['T'],
        shuffle=True
    )
    model = HeightRankModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    best_acc = 0.0
    save_dir = os.path.join(cfg['save_dir'], camera)
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(cfg['num_epochs']):
        model.train()
        for step, batch in enumerate(dataloader):
            bboxes = batch['bboxes'].to(device)
            mask = batch['mask'].to(device)
            rank_label = batch['rank_label'].to(device)
            out = model(bboxes, mask)
            loss_frame = frame_consistency_loss(out['z_t_H'], mask)
            loss_recon = recon_loss(out['z_t_P'], torch.log(bboxes[...,2:]), mask)
            loss = cfg['loss_weight']['frame_consistency'] * loss_frame + cfg['loss_weight']['recon'] * loss_recon
            if epoch > cfg['num_epochs']//2:
                loss_rank = pairwise_ranking_loss(out['s_v'], rank_label, cfg['pair_sampling'])
                loss += cfg['loss_weight']['ranking'] * loss_rank
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            optimizer.step()
        acc = eval_pairwise_acc(model, dataloader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
        print(f"Camera {camera} Epoch {epoch}: loss={loss.item():.4f}, pairwise_acc={acc:.4f}")

def eval_pairwise_acc(model, dataloader, device):
    model.eval()
    scores, ranks = [], []
    with torch.no_grad():
        for batch in dataloader:
            bboxes = batch['bboxes'].to(device)
            mask = batch['mask'].to(device)
            rank_label = batch['rank_label'].to(device)
            out = model(bboxes, mask)
            scores.append(out['s_v'].cpu().numpy())
            ranks.append(rank_label.cpu().numpy())
    scores = np.concatenate(scores)
    ranks = np.concatenate(ranks)
    correct, total = 0, 0
    for i in range(len(scores)):
        for j in range(len(scores)):
            if i == j: continue
            if ranks[i] > ranks[j]:
                total += 1
                if scores[i] > scores[j]:
                    correct += 1
    return correct / total if total > 0 else 0.0

def main():
    cfg = load_config('height_rank/configs/default.yaml')
    cameras = [f.split('_rank.json')[0] for f in os.listdir('height/labels_rank') if f.endswith('_rank.json')]
    for camera in cameras:
        print(f"==== Training camera: {camera} ====")
        train_camera(cfg, camera)

if __name__ == '__main__':
    main()
