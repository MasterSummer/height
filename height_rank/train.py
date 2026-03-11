import os
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloader
from models import HeightRankModel
from losses import frame_consistency_loss, recon_loss, pairwise_ranking_loss

import numpy as np

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader(
        video_dir='tmp_test_videos',
        bbox_dir='height/processed_labels',
        rank_dir='height/labels_rank',
        batch_size=cfg['batch_size'],
        T=cfg['T'],
        shuffle=True
    )
    model = HeightRankModel(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    writer = SummaryWriter()
    best_acc = 0.0
    os.makedirs(cfg['save_dir'], exist_ok=True)
    for epoch in range(cfg['num_epochs']):
        model.train()
        for step, batch in enumerate(dataloader):
            bboxes = batch['bboxes'].to(device)
            mask = batch['mask'].to(device)
            rank_label = batch['rank_label'].to(device)
            out = model(bboxes, mask)
            # 阶段A：重构+帧一致性
            loss_frame = frame_consistency_loss(out['z_t_H'], mask)
            loss_recon = recon_loss(out['z_t_P'], torch.log(bboxes[...,2:]), mask) # 只重构 log(w,h)
            loss = cfg['loss_weight']['frame_consistency'] * loss_frame + cfg['loss_weight']['recon'] * loss_recon
            # 阶段B：加排名
            if epoch > cfg['num_epochs']//2:
                loss_rank = pairwise_ranking_loss(out['s_v'], rank_label, cfg['pair_sampling'])
                loss += cfg['loss_weight']['ranking'] * loss_rank
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            optimizer.step()
            if step % cfg['log_interval'] == 0:
                writer.add_scalar('loss/total', loss.item(), epoch*len(dataloader)+step)
        # 简单评估
        acc = eval_pairwise_acc(model, dataloader, device)
        writer.add_scalar('eval/pairwise_acc', acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], 'best.pt'))
        print(f"Epoch {epoch}: loss={loss.item():.4f}, pairwise_acc={acc:.4f}")
    writer.close()

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
    # 计算 pairwise accuracy
    correct, total = 0, 0
    for i in range(len(scores)):
        for j in range(len(scores)):
            if i == j: continue
            if ranks[i] > ranks[j]:
                total += 1
                if scores[i] > scores[j]:
                    correct += 1
    return correct / total if total > 0 else 0.0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
