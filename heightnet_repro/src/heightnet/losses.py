from __future__ import annotations

import torch
import torch.nn as nn


def silog_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    lambd: float = 0.85,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-invariant log loss, adapted for positive-valued height maps."""
    mask = valid_mask > 0.5
    pred_valid = torch.clamp(pred[mask], min=eps)
    target_valid = torch.clamp(target[mask], min=eps)

    if pred_valid.numel() == 0:
        return pred.new_tensor(0.0)

    d = torch.log(pred_valid) - torch.log(target_valid)
    return torch.sqrt(torch.mean(d * d) - lambd * torch.mean(d) * torch.mean(d) + eps)


def _sample_points(pred: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    """Bilinear sampling for (x, y) points on a single prediction map."""
    _, _, h, w = pred.shape
    if points_xy.numel() == 0:
        return pred.new_zeros((0,))

    grid = points_xy.clone()
    grid[:, 0] = (grid[:, 0] / max(w - 1, 1)) * 2.0 - 1.0
    grid[:, 1] = (grid[:, 1] / max(h - 1, 1)) * 2.0 - 1.0
    grid = grid.view(1, -1, 1, 2)

    sampled = torch.nn.functional.grid_sample(
        pred,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(-1)


def consistency_loss(
    pred0: torch.Tensor,
    pred1: torch.Tensor,
    matches: torch.Tensor,
) -> torch.Tensor:
    """L1 consistency on matched points between adjacent frames."""
    if matches.numel() == 0:
        return pred0.new_tensor(0.0)

    pts0 = matches[:, :2]
    pts1 = matches[:, 2:]
    v0 = _sample_points(pred0, pts0)
    v1 = _sample_points(pred1, pts1)
    if v0.numel() == 0:
        return pred0.new_tensor(0.0)
    return torch.mean(torch.abs(v0 - v1))


class HeightNetLoss(nn.Module):
    def __init__(self, silog_lambda: float = 0.85, consistency_weight: float = 0.1) -> None:
        super().__init__()
        self.silog_lambda = silog_lambda
        self.consistency_weight = consistency_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        pred_pair: torch.Tensor | None = None,
        matches: torch.Tensor | None = None,
        enable_consistency: bool = False,
    ) -> dict:
        l_silog = silog_loss(pred, target, mask, lambd=self.silog_lambda)
        l_cons = pred.new_tensor(0.0)

        if enable_consistency and pred_pair is not None and matches is not None:
            batch_losses = []
            for i in range(pred.shape[0]):
                batch_losses.append(consistency_loss(pred[i : i + 1], pred_pair[i : i + 1], matches[i]))
            if batch_losses:
                l_cons = torch.stack(batch_losses).mean()

        total = l_silog + self.consistency_weight * l_cons
        return {"total": total, "silog": l_silog, "consistency": l_cons}
