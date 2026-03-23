import torch
import torch.nn.functional as F


def frame_consistency_loss(z_t_H, mask):
    """mean_t ||z_t^H - z_v^H|| over valid frames."""
    valid = mask.unsqueeze(-1)
    z_v_H = (z_t_H * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    diff = (z_t_H - z_v_H).norm(dim=-1) * mask
    return diff.sum() / mask.sum().clamp(min=1.0)


def recon_loss(pred_log_wh, target_log_wh, mask):
    valid = mask.unsqueeze(-1)
    loss = F.l1_loss(pred_log_wh * valid, target_log_wh * valid, reduction="sum")
    return loss / valid.sum().clamp(min=1.0)


def build_pair_indices(ranks, sampling="all", hard_margin=1.0, max_pairs=None):
    """
    Build ordered pairs (i, j) where rank_i > rank_j.
    sampling='hard': keep only |rank_i-rank_j| <= hard_margin.
    """
    ranks = ranks.view(-1)
    n = ranks.shape[0]
    if n <= 1:
        return None, None

    i_idx = []
    j_idx = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gap = torch.abs(ranks[i] - ranks[j]).item()
            if ranks[i] > ranks[j]:
                if sampling == "hard" and gap > hard_margin:
                    continue
                i_idx.append(i)
                j_idx.append(j)

    if not i_idx:
        return None, None

    if max_pairs is not None and len(i_idx) > max_pairs:
        perm = torch.randperm(len(i_idx))[:max_pairs]
        i_idx = [i_idx[k] for k in perm.tolist()]
        j_idx = [j_idx[k] for k in perm.tolist()]

    return torch.tensor(i_idx, dtype=torch.long), torch.tensor(j_idx, dtype=torch.long)


def pairwise_ranking_loss(scores, ranks, pair_cfg=None):
    """
    For any pair(i,j) with rank_i > rank_j:
      loss = log(1 + exp(-(s_i - s_j))) = softplus(-(s_i - s_j)).
    """
    pair_cfg = pair_cfg or {}
    sampling = pair_cfg.get("sampling", "all")
    hard_margin = float(pair_cfg.get("hard_margin", 1.0))
    max_pairs = pair_cfg.get("max_pairs", None)

    pair_i, pair_j = build_pair_indices(
        ranks=ranks,
        sampling=sampling,
        hard_margin=hard_margin,
        max_pairs=max_pairs,
    )
    if pair_i is None:
        return scores.new_zeros(())

    pair_i = pair_i.to(scores.device)
    pair_j = pair_j.to(scores.device)
    margin = scores[pair_i] - scores[pair_j]
    return F.softplus(-margin).mean()
