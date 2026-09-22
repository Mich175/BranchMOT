"""Leakage-free identity targets derived from stable tracker output IDs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from .cache import AssociationFrame


@dataclass(frozen=True)
class IdentityTargetStats:
    """Diagnostics for the causal stable-output-ID to GT-ID mapping."""

    seeded_output_ids: int
    evaluated_detections: int
    conflicting_observations: int
    ambiguous_targets: int


@dataclass(frozen=True)
class IdentityAssignmentMetrics:
    """Immediate assignment accuracy on causally evaluable detections."""

    correct: int
    evaluated: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.evaluated if self.evaluated else 0.0


def attach_internal_identity_targets(
    frames: Iterable[AssociationFrame],
) -> tuple[list[AssociationFrame], IdentityTargetStats]:
    """Translate dataset GT IDs into MOTIP vocabulary labels without leakage.

    A stable MOTIP output ID is associated with a dataset GT ID only *after*
    the frame's targets have been computed. Therefore the target for frame t
    depends exclusively on identity evidence observed before frame t.
    """

    output_to_ground_truth: dict[int, int] = {}
    annotated: list[AssociationFrame] = []
    seeded = evaluated = conflicts = ambiguous = 0

    for frame in frames:
        frame.validate()
        internal_targets: list[int | None] | None = None
        if frame.ground_truth_track_ids is not None:
            internal_targets = []
            active_outputs = frame.active_output_ids or []
            for ground_truth_id in frame.ground_truth_track_ids:
                if ground_truth_id is None:
                    internal_targets.append(None)
                    continue
                candidates = [
                    index
                    for index, output_id in enumerate(active_outputs)
                    if output_to_ground_truth.get(output_id) == ground_truth_id
                ]
                if len(candidates) == 1:
                    internal_targets.append(frame.track_ids[candidates[0]])
                    evaluated += 1
                else:
                    internal_targets.append(None)
                    ambiguous += int(len(candidates) > 1)

        annotated_frame = replace(
            frame, ground_truth_internal_ids=internal_targets
        )
        annotated_frame.validate()
        annotated.append(annotated_frame)

        if (
            frame.assigned_output_ids is None
            or frame.ground_truth_track_ids is None
        ):
            continue
        for output_id, ground_truth_id in zip(
            frame.assigned_output_ids, frame.ground_truth_track_ids, strict=True
        ):
            if ground_truth_id is None:
                continue
            previous = output_to_ground_truth.get(output_id)
            if previous is None:
                output_to_ground_truth[output_id] = ground_truth_id
                seeded += 1
            elif previous != ground_truth_id:
                conflicts += 1

    return annotated, IdentityTargetStats(
        seeded_output_ids=seeded,
        evaluated_detections=evaluated,
        conflicting_observations=conflicts,
        ambiguous_targets=ambiguous,
    )


def identity_assignment_metrics(
    frames: Iterable[AssociationFrame],
) -> IdentityAssignmentMetrics:
    """Score final internal-label decisions against attached causal targets."""

    correct = evaluated = 0
    for frame in frames:
        if (
            frame.assigned_track_ids is None
            or frame.ground_truth_internal_ids is None
        ):
            continue
        for assigned, target in zip(
            frame.assigned_track_ids, frame.ground_truth_internal_ids, strict=True
        ):
            if target is not None:
                evaluated += 1
                correct += int(assigned == target)
    return IdentityAssignmentMetrics(correct=correct, evaluated=evaluated)
