"""Bounded multi-hypothesis association for an online MOT prototype.

This module intentionally operates on association probabilities rather than
detector features. It can therefore sit on top of MOTIP, LA-MOTR, or a
tracking-by-detection baseline during the first validation stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np


@dataclass(frozen=True)
class AssociationConfig:
    """Controls when the tracker branches and when it commits."""

    beam_size: int = 4
    max_delay: int = 4
    entropy_threshold: float = 0.55
    margin_threshold: float = 0.20
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.beam_size < 1:
            raise ValueError("beam_size must be positive")
        if self.max_delay < 0:
            raise ValueError("max_delay must be non-negative")


@dataclass(frozen=True)
class Hypothesis:
    """One association history retained in the beam."""

    assignments: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    log_score: float = 0.0

    def extend(self, assignment: tuple[int, ...], score: float) -> Hypothesis:
        return Hypothesis(self.assignments + (assignment,), self.log_score + score)


class BranchingAssociator:
    """Maintains a small beam of persistent identity assignments.

    Rows of ``probabilities`` correspond to detections and columns correspond
    to active track identities. During a delayed episode, row positions must
    represent the same short-term observation chains across frames (for
    example, mask-propagated instances). Each hypothesis therefore keeps one
    fixed chain-to-identity mapping while later evidence accumulates.

    This MVP assumes an equal number of observation chains and tracks inside
    an already selected ambiguous subgraph.
    """

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self._beam = [Hypothesis()]
        self._age = 0

    @property
    def hypotheses(self) -> tuple[Hypothesis, ...]:
        return tuple(self._beam)

    def reset(self) -> None:
        self._beam = [Hypothesis()]
        self._age = 0

    def is_ambiguous(self, probabilities: np.ndarray) -> bool:
        probs = self._validate(probabilities)
        row_entropy = -np.sum(probs * np.log(probs + self.config.eps), axis=1)
        if probs.shape[1] > 1:
            row_entropy /= np.log(probs.shape[1])
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        return bool(
            np.any(row_entropy > self.config.entropy_threshold)
            or np.any(margins < self.config.margin_threshold)
        )

    def step(self, probabilities: np.ndarray) -> tuple[int, ...] | None:
        """Update the beam and optionally emit a committed assignment.

        Returns a detection-to-track assignment when evidence is clear or the
        bounded delay expires. Returns ``None`` while multiple plausible paths
        are intentionally retained.
        """

        probs = self._validate(probabilities)
        candidates: list[Hypothesis] = []
        if self._age == 0:
            for assignment in permutations(range(probs.shape[1]), probs.shape[0]):
                likelihood = probs[np.arange(probs.shape[0]), assignment]
                score = float(np.log(likelihood + self.config.eps).sum())
                candidates.append(self._beam[0].extend(tuple(assignment), score))
        else:
            for hypothesis in self._beam:
                assignment = hypothesis.assignments[0]
                likelihood = probs[np.arange(probs.shape[0]), assignment]
                score = float(np.log(likelihood + self.config.eps).sum())
                candidates.append(hypothesis.extend(assignment, score))

        candidates.sort(key=lambda item: item.log_score, reverse=True)
        self._beam = candidates[: self.config.beam_size]
        self._age += 1

        should_commit = not self.is_ambiguous(probs) or self._age > self.config.max_delay
        if not should_commit:
            return None

        winner = self._beam[0]
        assignment = winner.assignments[-1]
        self.reset()
        return assignment

    def best_path(self) -> tuple[tuple[int, ...], ...]:
        """Return the current maximum-posterior path without committing it."""

        return self._beam[0].assignments

    def _validate(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[0] == 0:
            raise ValueError("probabilities must be a non-empty 2-D matrix")
        if probs.shape[0] != probs.shape[1]:
            raise ValueError("MVP requires a square ambiguous association subgraph")
        if np.any(probs < 0):
            raise ValueError("probabilities must be non-negative")
        row_sums = probs.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("each detection must have positive probability mass")
        return probs / row_sums
