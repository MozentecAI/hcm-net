"""Class-balanced focal and regional-to-global consistency losses."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def effective_number_weights(
    class_counts: list[int],
    beta: float,
) -> Tensor:
    """Compute normalized class weights from the effective sample count."""
    counts = torch.tensor(class_counts, dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError(f"Every class needs training samples, got {class_counts}.")
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    return weights / weights.mean()


def focal_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    alpha: Tensor,
    gamma: float,
) -> Tensor:
    """Multiclass focal loss with one weight per target class."""
    cross_entropy = F.cross_entropy(logits, targets, reduction="none")
    probabilities = logits.softmax(dim=-1)
    target_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_factor = (1.0 - target_probabilities).pow(gamma)
    sample_weights = alpha.to(logits.device)[targets]
    return (sample_weights * focal_factor * cross_entropy).mean()


def regional_to_global_consistency(
    logits: dict[str, Tensor],
) -> Tensor:
    """KL divergence from soft regional OR logic to the overall coarse head.

    v2 clinical definition: the overall status aggregates LVOT (outflow) and
    midventricular (middle) obstruction only; the mid-apical phenotype is
    nested within the MVO spectrum and is not an independent source.
    """
    regional_obstruction = [
        logits[f"{region}_coarse"].float().softmax(dim=-1)[:, 1]
        for region in ("outflow", "middle")
    ]
    derived_obstruction = 1.0 - torch.stack(
        [1.0 - probability for probability in regional_obstruction],
        dim=1,
    ).prod(dim=1)
    derived = torch.stack(
        [1.0 - derived_obstruction, derived_obstruction],
        dim=1,
    )
    overall = logits["overall_coarse"].float().softmax(dim=-1)
    epsilon = torch.finfo(overall.dtype).eps
    derived = derived.clamp_min(epsilon)
    overall = overall.clamp_min(epsilon)
    return (derived * (derived.log() - overall.log())).sum(dim=1).mean()
