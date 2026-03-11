import yaml
import torch
import numpy as np
from dataset import get_dataloader
from models import HeightRankModel
from scipy.stats import spearmanr, kendalltau

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def evaluate(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader(
        video_dir='tmp_test_videos',
        bbox_dir='height/processed_labels',
        rank_dir='height/labels_rank',
        batch_size=cfg['batch_size'],
        T=cfg['T'],
        shuffle=False
    )
    model = HeightRankModel(cfg).to(device)
    model.load_state_dict(torch.load(cfg['save_dir']+'/best.pt', map_location=device))
    model.eval()
    scores, ranks, frame_stds, recon_errs = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            bboxes = batch['bboxes'].to(device)
            mask = batch['mask'].to(device)
            rank_label = batch['rank_label'].to(device)
            out = model(bboxes, mask)
            scores.append(out['s_v'].cpu().numpy())
            ranks.append(rank_label.cpu().numpy())
            # 帧稳定性
            z_t_H = out['z_t_H'].cpu().numpy()
            z_v_H = out['z_v_H'].cpu().numpy()
            std = np.std(z_t_H, axis=1)
            mean_diff = np.mean(np.linalg.norm(z_t_H - z_v_H[:,None], axis=-1), axis=1)
            frame_stds.extend(std)
            frame_stds.extend(mean_diff)
            # 重构误差
            recon_err = np.mean(np.abs(out['z_t_P'].cpu().numpy() - np.log(bboxes[...,2:].cpu().numpy())))
            recon_errs.append(recon_err)
    scores = np.concatenate(scores)
    ranks = np.concatenate(ranks)
    # Pairwise Accuracy
    correct, total = 0, 0
    for i in range(len(scores)):
        for j in range(len(scores)):
            if i == j: continue
            if ranks[i] > ranks[j]:
                total += 1
                if scores[i] > scores[j]:
                    correct += 1
    pairwise_acc = correct / total if total > 0 else 0.0
    # Spearman/Kendall
    spearman = spearmanr(scores, ranks).correlation
    kendall = kendalltau(scores, ranks).correlation
    # 帧稳定性
    frame_std = np.mean(frame_stds)
    # 重构误差
    recon_err = np.mean(recon_errs)
    print(f"Pairwise Accuracy: {pairwise_acc:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(f"Kendall: {kendall:.4f}")
    print(f"Frame Stability: {frame_std:.4f}")
    print(f"Reconstruction Error: {recon_err:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)
