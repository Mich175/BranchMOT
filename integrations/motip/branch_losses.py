"""Differentiable BranchMOT losses for a MOTIP training environment.

This module intentionally lives outside the installable core: it expects the
PyTorch version already required by upstream MOTIP.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class BranchLossWeights:
    id_prediction: float = 1.0
    branch_ranking: float = 0.5
    risk_calibration: float = 0.2
    compute_budget: float = 0.01
    delay_budget: float = 0.01


def branchmot_training_loss(
    *,
    oracle_id_logits: torch.Tensor,
    id_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    branch_scores: torch.Tensor,
    oracle_branch: torch.Tensor,
    risk_logits: torch.Tensor,
    immediate_correct: torch.Tensor,
    expected_branches: torch.Tensor,
    expected_delay: torch.Tensor,
    weights: BranchLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Combine ID prediction, oracle-branch ranking, calibration, and budget.

    Shapes are ``oracle_id_logits=[B,T,N,V]``, ``id_targets=[B,T,N]``,
    ``valid_mask=[B,T,N]``, ``branch_scores=[B,K]``, and all remaining
    supervision tensors are batch-aligned. ``oracle_branch`` is constructed
    from GT continuity during training only; it is never used at inference.
    """

    loss_weights = weights or BranchLossWeights()
    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be boolean")
    if not torch.any(valid_mask):
        raise ValueError("a training batch needs at least one valid ID target")

    id_loss = functional.cross_entropy(
        oracle_id_logits[valid_mask], id_targets[valid_mask]
    )
    ranking_loss = functional.cross_entropy(branch_scores, oracle_branch)
    error_target = (~immediate_correct.bool()).to(risk_logits.dtype)
    calibration_loss = functional.mse_loss(
        torch.sigmoid(risk_logits), error_target
    )
    compute_loss = expected_branches.float().mean()
    delay_loss = expected_delay.float().mean()
    total = (
        loss_weights.id_prediction * id_loss
        + loss_weights.branch_ranking * ranking_loss
        + loss_weights.risk_calibration * calibration_loss
        + loss_weights.compute_budget * compute_loss
        + loss_weights.delay_budget * delay_loss
    )
    return {
        "loss": total,
        "id_prediction_loss": id_loss,
        "branch_ranking_loss": ranking_loss,
        "risk_calibration_loss": calibration_loss,
        "compute_budget_loss": compute_loss,
        "delay_budget_loss": delay_loss,
    }
