"""Sparse bipartite conflict decomposition for association matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConflictComponent:
    """Connected detection/track candidates that must be solved jointly."""

    detection_indices: tuple[int, ...]
    track_indices: tuple[int, ...]
    ambiguous: bool

    @property
    def branchable(self) -> bool:
        """Whether the current square-beam prototype can process this component."""

        return self.ambiguous and len(self.detection_indices) == len(self.track_indices)

    def submatrix(self, probabilities: np.ndarray) -> np.ndarray:
        matrix = np.asarray(probabilities)
        return matrix[np.ix_(self.detection_indices, self.track_indices)]


def decompose_conflicts(
    probabilities: np.ndarray, *, candidate_threshold: float
) -> tuple[ConflictComponent, ...]:
    """Decompose a gated association matrix into bipartite components.

    Every detection is returned, including detections with no candidate track;
    the latter become birth-only components. Track-only components are omitted
    because no current detection can update them.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError("probabilities must be a non-empty 2-D matrix")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not 0.0 <= candidate_threshold <= 1.0:
        raise ValueError("candidate_threshold must be in [0, 1]")

    adjacency = probs >= candidate_threshold
    n_detections, _ = probs.shape
    visited_detections: set[int] = set()
    components: list[ConflictComponent] = []

    for seed in range(n_detections):
        if seed in visited_detections:
            continue
        pending_detections = [seed]
        component_detections: set[int] = set()
        component_tracks: set[int] = set()

        while pending_detections:
            detection = pending_detections.pop()
            if detection in component_detections:
                continue
            component_detections.add(detection)
            visited_detections.add(detection)
            tracks = set(np.flatnonzero(adjacency[detection]).tolist())
            new_tracks = tracks - component_tracks
            component_tracks.update(tracks)
            for track in new_tracks:
                linked_detections = np.flatnonzero(adjacency[:, track]).tolist()
                pending_detections.extend(linked_detections)

        detection_indices = tuple(sorted(component_detections))
        track_indices = tuple(sorted(component_tracks))
        detection_degrees = adjacency[list(detection_indices)].sum(axis=1)
        track_degrees = (
            adjacency[:, list(track_indices)].sum(axis=0)
            if track_indices
            else np.array([], dtype=np.int64)
        )
        ambiguous = bool(
            np.any(detection_degrees > 1) or np.any(track_degrees > 1)
        )
        components.append(
            ConflictComponent(detection_indices, track_indices, ambiguous)
        )

    return tuple(components)

