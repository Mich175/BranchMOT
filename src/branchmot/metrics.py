"""Calibration metrics for association predictions."""

from __future__ import annotations

import numpy as np


def association_calibration(
    probabilities: np.ndarray, targets: np.ndarray, *, n_bins: int = 10
) -> dict[str, float]:
    """Compute ECE, multiclass Brier score, NLL, and accuracy."""

    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2 or labels.shape != (probs.shape[0],):
        raise ValueError("targets must provide one class index per probability row")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("target index is outside the probability matrix")
    row_sums = probs.sum(axis=1)
    if np.any(probs < 0) or not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("probability rows must be non-negative and sum to one")

    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predictions == labels
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (confidence > lower) & (confidence <= upper)
        if index == 0:
            in_bin |= confidence == 0.0
        if np.any(in_bin):
            gap = abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))
            ece += float(in_bin.mean()) * gap

    one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    target_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    nll = float(-np.log(target_probs).mean())
    return {
        "accuracy": float(correct.mean()),
        "ece": ece,
        "brier": brier,
        "nll": nll,
    }
