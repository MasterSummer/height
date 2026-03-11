import torch
import torch.nn.functional as F

def frame_consistency_loss(z_t_H, mask):
    # 视频内帧一致性：std 或 mean ||z_t-z_v||
    z_v_H = (z_t_H * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    diff = (z_t_H - z_v_H.unsqueeze(1)) * mask.unsqueeze(-1)
    loss = diff.norm(dim=-1).mean()
    return loss

def recon_loss(pred, target, mask):
    # 重构 bbox（log 尺度）
    loss = F.l1_loss(pred * mask.unsqueeze(-1), target * mask.unsqueeze(-1), reduction='sum')
    loss = loss / mask.sum()
    return loss

def pairwise_ranking_loss(scores, ranks, pair_sampling='hard'):
    # batch 内构造 pairs
    batch_size = scores.shape[0]
    loss = 0.0
    num_pairs = 0
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            if ranks[i] > ranks[j]:
                if pair_sampling == 'hard' and abs(ranks[i] - ranks[j]) > 1:
                    continue
                s_i, s_j = scores[i], scores[j]
                loss += torch.log(1 + torch.exp(-(s_i - s_j)))
                num_pairs += 1
    if num_pairs == 0:
        return torch.tensor(0.0, device=scores.device)
    return loss / num_pairs
