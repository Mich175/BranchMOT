"""Delayed association with explicit newborn and missed-track hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .association import AssociationConfig

Identity = int | None


@dataclass(frozen=True)
class LifecycleHypothesis:
    """A persistent observation-chain to track/newborn mapping."""

    assignment: tuple[Identity, ...]
    log_score: float
    age: int = 1


def enumerate_lifecycle_assignments(
    n_detections: int, track_ids: tuple[int, ...]
) -> tuple[tuple[Identity, ...], ...]:
    """Enumerate injective track assignments with a per-detection newborn option."""

    assignments: list[tuple[Identity, ...]] = []

    def visit(index: int, used_tracks: set[int], current: list[Identity]) -> None:
        if index == n_detections:
            assignments.append(tuple(current))
            return
        current.append(None)
        visit(index + 1, used_tracks, current)
        current.pop()
        for track_id in track_ids:
            if track_id in used_tracks:
                continue
            used_tracks.add(track_id)
            current.append(track_id)
            visit(index + 1, used_tracks, current)
            current.pop()
            used_tracks.remove(track_id)

    visit(0, set(), [])
    return tuple(assignments)


class LifecycleAssociator:
    """Beam search over persistent mappings, births, and missed tracks."""

    def __init__(
        self,
        config: AssociationConfig | None = None,
        *,
        missed_track_probability: float = 0.95,
    ) -> None:
        self.config = config or AssociationConfig()
        if not 0.0 < missed_track_probability <= 1.0:
            raise ValueError("missed_track_probability must be in (0, 1]")
        self.missed_track_probability = missed_track_probability
        self._beam: list[LifecycleHypothesis] = []
        self._track_ids: tuple[int, ...] = ()

    @property
    def hypotheses(self) -> tuple[LifecycleHypothesis, ...]:
        return tuple(self._beam)

    def reset(self) -> None:
        self._beam.clear()
        self._track_ids = ()

    def step(
        self,
        track_probabilities: np.ndarray,
        newborn_probabilities: np.ndarray,
        track_ids: list[int] | tuple[int, ...],
    ) -> tuple[Identity, ...] | None:
        tracks, newborn, identities = self._validate(
            track_probabilities, newborn_probabilities, track_ids
        )
        if not self._beam:
            self._track_ids = identities
            assignments = enumerate_lifecycle_assignments(len(tracks), identities)
            candidates = [
                LifecycleHypothesis(
                    assignment,
                    self._score(assignment, tracks, newborn, identities),
                )
                for assignment in assignments
            ]
        else:
            if identities != self._track_ids:
                raise ValueError("track IDs must remain aligned during a delayed episode")
            candidates = [
                LifecycleHypothesis(
                    hypothesis.assignment,
                    hypothesis.log_score
                    + self._score(hypothesis.assignment, tracks, newborn, identities),
                    hypothesis.age + 1,
                )
                for hypothesis in self._beam
            ]

        candidates.sort(key=lambda item: item.log_score, reverse=True)
        self._beam = candidates[: self.config.beam_size]
        best = self._beam[0]
        if self._is_ambiguous(tracks, newborn) and best.age <= self.config.max_delay:
            return None
        decision = best.assignment
        self.reset()
        return decision

    def _score(
        self,
        assignment: tuple[Identity, ...],
        track_probabilities: np.ndarray,
        newborn_probabilities: np.ndarray,
        track_ids: tuple[int, ...],
    ) -> float:
        track_columns = {track_id: index for index, track_id in enumerate(track_ids)}
        score = 0.0
        used_tracks: set[int] = set()
        for detection, identity in enumerate(assignment):
            probability = (
                newborn_probabilities[detection]
                if identity is None
                else track_probabilities[detection, track_columns[identity]]
            )
            score += float(np.log(probability + self.config.eps))
            if identity is not None:
                used_tracks.add(identity)
        score += (len(track_ids) - len(used_tracks)) * float(
            np.log(self.missed_track_probability)
        )
        return score

    def _is_ambiguous(
        self, track_probabilities: np.ndarray, newborn_probabilities: np.ndarray
    ) -> bool:
        combined = np.column_stack([track_probabilities, newborn_probabilities])
        combined /= combined.sum(axis=1, keepdims=True)
        entropy = -np.sum(combined * np.log(combined + self.config.eps), axis=1)
        if combined.shape[1] > 1:
            entropy /= np.log(combined.shape[1])
        ordered = np.sort(combined, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
        return bool(
            np.any(entropy > self.config.entropy_threshold)
            or np.any(margins < self.config.margin_threshold)
        )

    @staticmethod
    def _validate(
        track_probabilities: np.ndarray,
        newborn_probabilities: np.ndarray,
        track_ids: list[int] | tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        tracks = np.asarray(track_probabilities, dtype=np.float64)
        newborn = np.asarray(newborn_probabilities, dtype=np.float64)
        identities = tuple(track_ids)
        if tracks.ndim != 2 or tracks.shape[0] == 0:
            raise ValueError("track probabilities must be a non-empty 2-D matrix")
        if tracks.shape[1] != len(identities):
            raise ValueError("track IDs must align with probability columns")
        if newborn.shape != (tracks.shape[0],):
            raise ValueError("newborn probabilities must align with detections")
        if len(set(identities)) != len(identities):
            raise ValueError("track IDs must be unique")
        combined = np.column_stack([tracks, newborn])
        if np.any(~np.isfinite(combined)) or np.any(combined < 0):
            raise ValueError("probabilities must be finite and non-negative")
        mass = combined.sum(axis=1)
        if np.any(mass <= 0):
            raise ValueError("each detection needs positive probability mass")
        tracks = tracks / mass[:, None]
        newborn = newborn / mass
        return tracks, newborn, identities

