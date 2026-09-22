"""Hypothesis-conditioned memory branching for reversible ID prediction."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np

DecodeFn = Callable[[Any, Any], np.ndarray]
UpdateFn = Callable[[Any, Any, tuple[int, ...]], Any]


@dataclass(frozen=True)
class ConditionalMemoryConfig:
    """Risk, latency, and compute budget for conditional memory inference."""

    max_branches: int = 4
    max_delay: int = 4
    retained_posterior_mass: float = 0.95
    commit_posterior: float = 0.90
    entropy_threshold: float = 0.55
    margin_threshold: float = 0.20
    max_conflict_size: int = 6
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.max_branches < 2:
            raise ValueError("max_branches must be at least two")
        if self.max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        for name, value in (
            ("retained_posterior_mass", self.retained_posterior_mass),
            ("commit_posterior", self.commit_posterior),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.max_conflict_size < 2:
            raise ValueError("max_conflict_size must be at least two")


@dataclass(frozen=True)
class ConditionalMemoryBranch:
    """One persistent chain-to-ID mapping and its private tracker memory."""

    assignment: tuple[int, ...]
    memory: Any
    log_score: float
    age: int


@dataclass(frozen=True)
class ConditionalMemoryDecision:
    assignment: tuple[int, ...]
    memory: Any
    delayed_frames: int
    posterior: float


class HypothesisConditionedAssociator:
    """Fork tracker memory and decode future evidence under every hypothesis.

    ``decode(memory, observation)`` must return a square row-to-ID probability
    matrix. ``update(memory, observation, assignment)`` must return the memory
    after applying that assignment. Inputs are deep-copied before updates, so
    an unresolved branch cannot contaminate canonical tracker memory.
    """

    def __init__(self, config: ConditionalMemoryConfig | None = None) -> None:
        self.config = config or ConditionalMemoryConfig()
        self._branches: list[ConditionalMemoryBranch] = []

    @property
    def branches(self) -> tuple[ConditionalMemoryBranch, ...]:
        return tuple(self._branches)

    def reset(self) -> None:
        self._branches.clear()

    def step(
        self,
        canonical_memory: Any,
        observation: Any,
        *,
        decode: DecodeFn,
        update: UpdateFn,
    ) -> ConditionalMemoryDecision | None:
        """Start or advance a conflict episode and optionally commit a branch."""

        if not self._branches:
            probabilities = self._validate(decode(canonical_memory, observation))
            assignments = tuple(permutations(range(probabilities.shape[1])))
            scored = [
                ConditionalMemoryBranch(
                    assignment=assignment,
                    memory=update(deepcopy(canonical_memory), observation, assignment),
                    log_score=self._score(probabilities, assignment),
                    age=1,
                )
                for assignment in assignments
            ]
            scored.sort(key=lambda branch: branch.log_score, reverse=True)
            if not self._is_ambiguous(probabilities) or self.config.max_delay == 0:
                return self._decision(scored[0], scored, delayed_frames=0)
            self._branches = self._retain(scored)
            return None

        candidates: list[ConditionalMemoryBranch] = []
        for branch in self._branches:
            probabilities = self._validate(decode(branch.memory, observation))
            candidates.append(
                ConditionalMemoryBranch(
                    assignment=branch.assignment,
                    memory=update(
                        deepcopy(branch.memory), observation, branch.assignment
                    ),
                    log_score=branch.log_score
                    + self._score(probabilities, branch.assignment),
                    age=branch.age + 1,
                )
            )
        candidates.sort(key=lambda branch: branch.log_score, reverse=True)
        retained = self._retain(candidates)
        posterior = self._posterior(retained)[0]
        winner = retained[0]
        if (
            posterior >= self.config.commit_posterior
            or winner.age > self.config.max_delay
        ):
            decision = ConditionalMemoryDecision(
                winner.assignment,
                winner.memory,
                winner.age - 1,
                float(posterior),
            )
            self.reset()
            return decision
        self._branches = retained
        return None

    def _retain(
        self, branches: list[ConditionalMemoryBranch]
    ) -> list[ConditionalMemoryBranch]:
        posterior = self._posterior(branches)
        retained_count = min(2, len(branches))
        while (
            retained_count < min(self.config.max_branches, len(branches))
            and posterior[:retained_count].sum()
            < self.config.retained_posterior_mass
        ):
            retained_count += 1
        return branches[:retained_count]

    def _decision(
        self,
        winner: ConditionalMemoryBranch,
        candidates: list[ConditionalMemoryBranch],
        *,
        delayed_frames: int,
    ) -> ConditionalMemoryDecision:
        posterior = float(self._posterior(candidates)[0])
        self.reset()
        return ConditionalMemoryDecision(
            winner.assignment, winner.memory, delayed_frames, posterior
        )

    def _is_ambiguous(self, probabilities: np.ndarray) -> bool:
        entropy = -np.sum(
            probabilities * np.log(probabilities + self.config.eps), axis=1
        )
        if probabilities.shape[1] > 1:
            entropy /= np.log(probabilities.shape[1])
        ordered = np.sort(probabilities, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
        return bool(
            np.any(entropy > self.config.entropy_threshold)
            or np.any(margins < self.config.margin_threshold)
        )

    def _score(
        self, probabilities: np.ndarray, assignment: tuple[int, ...]
    ) -> float:
        likelihood = probabilities[np.arange(len(assignment)), assignment]
        return float(np.log(likelihood + self.config.eps).sum())

    @staticmethod
    def _posterior(branches: list[ConditionalMemoryBranch]) -> np.ndarray:
        scores = np.asarray([branch.log_score for branch in branches])
        weights = np.exp(scores - scores.max())
        return weights / weights.sum()

    def _validate(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("probabilities must be a non-empty matrix")
        if values.shape[0] != values.shape[1]:
            raise ValueError("conditional memory currently requires square conflicts")
        if values.shape[0] > self.config.max_conflict_size:
            raise ValueError("conflict exceeds the configured exact-search limit")
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("probabilities must be finite and non-negative")
        mass = values.sum(axis=1, keepdims=True)
        if np.any(mass <= 0):
            raise ValueError("each row needs positive probability mass")
        return values / mass
