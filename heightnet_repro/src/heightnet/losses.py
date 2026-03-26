from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _masked_max_score(
    pred0: torch.Tensor,
    person_mask0: torch.Tensor,
) -> torch.Tensor:
    m0 = person_mask0 > 0.5
    if m0.sum() == 0:
        return pred0.new_tensor(0.0)
    v0 = pred0[m0]
    return torch.max(v0)


def consistency_loss(
    pred0: torch.Tensor,
    pred1: torch.Tensor,
    person_mask0: torch.Tensor,
    person_mask1: torch.Tensor,
) -> torch.Tensor:
    """L1 consistency on masked max height score across adjacent frames."""
    s0 = _masked_max_score(pred0, person_mask0)
    s1 = _masked_max_score(pred1, person_mask1)
    return torch.abs(s0 - s1)


class HeightNetLoss(nn.Module):
    def __init__(
        self,
        silog_lambda: float = 0.85,
        consistency_weight: float = 0.1,
        pairwise_weight: float = 0.0,
        pairwise_json: str = "",
    ) -> None:
        super().__init__()
        self.silog_lambda = silog_lambda
        self.consistency_weight = consistency_weight
        self.pairwise_weight = pairwise_weight
        self.pairwise_labels = self._load_pairwise_labels(pairwise_json) if pairwise_json else {}

    @staticmethod
    def _load_pairwise_labels(path: str) -> Dict[str, Dict[Tuple[str, str], int]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels: Dict[str, Dict[Tuple[str, str], int]] = defaultdict(dict)
        for item in data:
            cam = str(item["camera"])
            i = str(item["id_i"])
            j = str(item["id_j"])
            y = int(item["y"])
            labels[cam][(i, j)] = y
            labels[cam][(j, i)] = 1 - y
        return dict(labels)

    @staticmethod
    def masked_max_scores(pred: torch.Tensor, person_mask: torch.Tensor) -> torch.Tensor:
        scores: List[torch.Tensor] = []
        for i in range(pred.shape[0]):
            scores.append(_masked_max_score(pred[i : i + 1], person_mask[i : i + 1]))
        return torch.stack(scores, dim=0)

    def pairwise_loss(
        self,
        scores: torch.Tensor,
        person_ids: List[str],
        camera_ids: List[str],
    ) -> torch.Tensor:
        terms: List[torch.Tensor] = []
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                if camera_ids[i] != camera_ids[j]:
                    continue
                if person_ids[i] == person_ids[j]:
                    continue
                label_map = self.pairwise_labels.get(camera_ids[i], {})
                key = (person_ids[i], person_ids[j])
                if key not in label_map:
                    continue
                y = float(label_map[key])
                delta = scores[i] - scores[j]
                target = scores.new_tensor(1.0 if y > 0.5 else -1.0)
                terms.append(F.softplus(-target * delta))
        if not terms:
            return scores.new_tensor(0.0)
        return torch.stack(terms).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        pred_pair: torch.Tensor | None = None,
        person_mask: torch.Tensor | None = None,
        person_mask_pair: torch.Tensor | None = None,
        person_ids: List[str] | None = None,
        camera_ids: List[str] | None = None,
        enable_pairwise: bool = False,
        enable_consistency: bool = False,
    ) -> dict:
        l_silog = silog_loss(pred, target, mask, lambd=self.silog_lambda)
        l_cons = pred.new_tensor(0.0)
        l_pair = pred.new_tensor(0.0)

        if (
            enable_consistency
            and pred_pair is not None
            and person_mask is not None
            and person_mask_pair is not None
        ):
            batch_losses = []
            for i in range(pred.shape[0]):
                batch_losses.append(
                    consistency_loss(
                        pred[i : i + 1],
                        pred_pair[i : i + 1],
                        person_mask[i : i + 1],
                        person_mask_pair[i : i + 1],
                    )
                )
            if batch_losses:
                l_cons = torch.stack(batch_losses).mean()

        if (
            enable_pairwise
            and person_mask is not None
            and person_ids is not None
            and camera_ids is not None
            and self.pairwise_weight > 0.0
            and self.pairwise_labels
        ):
            scores = self.masked_max_scores(pred, person_mask)
            l_pair = self.pairwise_loss(scores, person_ids, camera_ids)

        total = l_silog + self.consistency_weight * l_cons + self.pairwise_weight * l_pair
        return {"total": total, "silog": l_silog, "consistency": l_cons, "pairwise": l_pair}
