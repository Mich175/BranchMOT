"""Offline replay utilities for falsifying the delayed-commitment hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .association import AssociationConfig, BranchingAssociator
from .cache import AssociationFrame


@dataclass(frozen=True)
class ReplayResult:
    immediate_correct: int
    delayed_correct: int
    total: int
    committed_episodes: int
    delayed_frames: int

    @property
    def immediate_accuracy(self) -> float:
        return self.immediate_correct / self.total if self.total else 0.0

    @property
    def delayed_accuracy(self) -> float:
        return self.delayed_correct / self.total if self.total else 0.0


def best_one_to_one(probabilities: np.ndarray) -> tuple[int, ...]:
    """Return the maximum-likelihood assignment for a small square matrix."""

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != probs.shape[1]:
        raise ValueError("one-to-one replay requires a square matrix")
    return max(
        permutations(range(probs.shape[1])),
        key=lambda assignment: float(
            np.log(probs[np.arange(probs.shape[0]), assignment] + 1e-9).sum()
        ),
    )


def replay_aligned_episode(
    frames: list[AssociationFrame], config: AssociationConfig | None = None
) -> ReplayResult:
    """Compare immediate and delayed assignment on one aligned episode.

    All records must share ordered detection-chain IDs and ordered track IDs.
    Birth/death episodes are intentionally rejected until the lifecycle solver
    is enabled, preventing optimistic evaluation on unsupported cases.
    """

    if not frames:
        return ReplayResult(0, 0, 0, 0, 0)
    reference_detections = frames[0].detection_ids
    reference_tracks = frames[0].track_ids
    if len(reference_detections) != len(reference_tracks):
        raise ValueError("aligned replay currently requires square episodes")

    tracker = BranchingAssociator(config)
    pending_targets: list[tuple[int, ...]] = []
    immediate_correct = delayed_correct = total = episodes = delayed_frames = 0

    for frame in frames:
        frame.validate()
        if frame.detection_ids != reference_detections or frame.track_ids != reference_tracks:
            raise ValueError("episode rows and columns must remain aligned")
        if frame.ground_truth_track_ids is None:
            raise ValueError("ground-truth track IDs are required for replay")

        probs = np.asarray(frame.probabilities, dtype=np.float64)
        row_mass = probs.sum(axis=1, keepdims=True)
        if np.any(row_mass <= 0):
            raise ValueError("every replay row needs active-track probability mass")
        probs = probs / row_mass
        target = tuple(reference_tracks.index(track) for track in frame.ground_truth_track_ids)
        immediate = best_one_to_one(probs)
        immediate_correct += sum(a == b for a, b in zip(immediate, target))
        total += len(target)

        pending_targets.append(target)
        decision = tracker.step(probs)
        if decision is None:
            delayed_frames += 1
            continue
        episodes += 1
        for pending_target in pending_targets:
            delayed_correct += sum(a == b for a, b in zip(decision, pending_target))
        pending_targets.clear()

    if pending_targets:
        decision = tracker.hypotheses[0].assignments[0]
        episodes += 1
        for pending_target in pending_targets:
            delayed_correct += sum(a == b for a, b in zip(decision, pending_target))

    return ReplayResult(
        immediate_correct,
        delayed_correct,
        total,
        episodes,
        delayed_frames,
    )
