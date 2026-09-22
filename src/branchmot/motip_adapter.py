"""Adapter at MOTIP's ID-score/assignment boundary.

MOTIP produces scores over a fixed ID vocabulary plus a final newborn class.
This adapter selects the columns belonging to currently active trajectories
and preserves the newborn mass separately for later birth handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import AssociationFrame


@dataclass(frozen=True)
class MOTIPProjection:
    track_probabilities: np.ndarray
    newborn_probabilities: np.ndarray


def project_motip_scores(
    id_scores: np.ndarray,
    active_id_labels: list[int],
    *,
    newborn_label: int,
) -> MOTIPProjection:
    """Project MOTIP vocabulary scores onto active tracks plus newborn mass."""

    scores = np.asarray(id_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("id_scores must have shape [detections, vocabulary]")
    if newborn_label < 0 or newborn_label >= scores.shape[1]:
        raise ValueError("newborn_label is outside the score vocabulary")
    if len(set(active_id_labels)) != len(active_id_labels):
        raise ValueError("active ID labels must be unique")
    if any(label < 0 or label >= newborn_label for label in active_id_labels):
        raise ValueError("active ID labels must refer to non-newborn vocabulary entries")
    if np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("id_scores must be finite and non-negative probabilities")

    selected = scores[:, active_id_labels]
    newborn = scores[:, newborn_label]
    mass = selected.sum(axis=1) + newborn
    if np.any(mass <= 0):
        raise ValueError("each detection needs active-track or newborn probability mass")
    return MOTIPProjection(selected / mass[:, None], newborn / mass)


def make_cache_frame(
    *,
    frame_index: int,
    detection_ids: list[int],
    active_id_labels: list[int],
    id_scores: np.ndarray,
    newborn_label: int,
    ground_truth_track_ids: list[int] | None = None,
) -> AssociationFrame:
    """Create a validated cache record directly from MOTIP runtime tensors."""

    projection = project_motip_scores(
        id_scores, active_id_labels, newborn_label=newborn_label
    )
    frame = AssociationFrame(
        frame_index=frame_index,
        detection_ids=detection_ids,
        track_ids=active_id_labels,
        probabilities=projection.track_probabilities.tolist(),
        ground_truth_track_ids=ground_truth_track_ids,
        newborn_probabilities=projection.newborn_probabilities.tolist(),
    )
    frame.validate()
    return frame

