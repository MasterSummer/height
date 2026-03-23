from __future__ import annotations

from typing import Dict, List

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = mask > 0.5
    if m.sum() == 0:
        return pred.new_tensor(0.0)
    err = pred[m] - target[m]
    return torch.sqrt(torch.mean(err * err) + eps)


def delta_threshold(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, thresh: float) -> torch.Tensor:
    m = mask > 0.5
    if m.sum() == 0:
        return pred.new_tensor(0.0)
    p = torch.clamp(pred[m], min=1e-6)
    t = torch.clamp(target[m], min=1e-6)
    ratio = torch.maximum(p / t, t / p)
    return (ratio < thresh).float().mean()


def pairwise_accuracy_from_ranked_lists(
    pred_scores: Dict[str, float],
    gt_ranking: List[str],
) -> float:
    """
    Compute pairwise ranking accuracy from a global ranking list.
    For any i<j in gt_ranking, prediction is correct iff score(i) > score(j).
    """
    ids = [x for x in gt_ranking if x in pred_scores]
    if len(ids) < 2:
        return 0.0
    correct = 0
    total = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            total += 1
            if pred_scores[ids[i]] > pred_scores[ids[j]]:
                correct += 1
    return float(correct / total) if total > 0 else 0.0
