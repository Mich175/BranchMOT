"""Bridge private MOTIP runtime memories to conditional branch decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .motip_state import MOTIPRuntimeState, MOTIPStateAdapter
from .motip_tap import _to_numpy, _unwrap_model, probabilities_from_logits


@dataclass(frozen=True)
class MOTIPBranchObservation:
    """Activated detections and local conflict coordinates for one frame."""

    boxes: Any
    output_embeds: Any
    conflict_detection_indices: tuple[int, ...]
    candidate_id_labels: tuple[int, ...]
    fixed_id_labels: tuple[int | None, ...] | None = None
    scores: Any | None = None
    categories: Any | None = None


class MOTIPBranchDecoder:
    """Decode one local conflict under a branch-private MOTIP memory."""

    def __init__(self, state_adapter: MOTIPStateAdapter) -> None:
        self.state_adapter = state_adapter

    def decode(
        self, state: MOTIPRuntimeState, observation: MOTIPBranchObservation
    ) -> np.ndarray:
        """Return normalized conflict-row by candidate-ID probabilities."""

        self._validate_observation(state, observation)
        runtime = self.state_adapter.runtime_tracker
        model = _unwrap_model(runtime.model)
        decoder = getattr(model, "id_decoder", None)
        if decoder is None or not hasattr(decoder, "register_forward_hook"):
            raise ValueError("could not locate a hookable MOTIP id_decoder")

        captured: list[np.ndarray] = []

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            raw_logits = output[0] if isinstance(output, (tuple, list)) else output
            logits = _to_numpy(raw_logits)
            if logits.ndim != 5 or logits.shape[:3] != (1, 1, 1):
                raise ValueError(
                    "unexpected MOTIP ID-logit shape; expected [1, 1, 1, N, V]"
                )
            captured.append(logits[0, 0, 0])

        handle = decoder.register_forward_hook(capture)
        try:
            self.state_adapter.transact(
                state,
                lambda active: active._get_id_pred_labels(
                    boxes=observation.boxes,
                    output_embeds=observation.output_embeds,
                ),
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("MOTIP branch decode must invoke id_decoder exactly once")

        scores = probabilities_from_logits(
            captured[0], use_sigmoid=bool(runtime.use_sigmoid)
        )
        rows = np.asarray(observation.conflict_detection_indices, dtype=np.int64)
        columns = np.asarray(observation.candidate_id_labels, dtype=np.int64)
        if np.any(rows >= scores.shape[0]) or np.any(columns >= scores.shape[1] - 1):
            raise ValueError("MOTIP conflict indices exceed decoder logits")
        local = scores[np.ix_(rows, columns)]
        mass = local.sum(axis=1, keepdims=True)
        if np.any(mass <= 0):
            raise ValueError("candidate identities have zero probability mass")
        return local / mass

    def update(
        self,
        state: MOTIPRuntimeState,
        observation: MOTIPBranchObservation,
        assignment: tuple[int, ...],
    ) -> MOTIPRuntimeState:
        """Apply one local assignment to a complete private MOTIP frame state."""

        self._validate_observation(state, observation)
        if observation.fixed_id_labels is None:
            raise ValueError("fixed full-frame ID labels are required for branch update")
        if observation.scores is None or observation.categories is None:
            raise ValueError("scores and categories are required for branch results")
        if sorted(assignment) != list(range(len(observation.candidate_id_labels))):
            raise ValueError("branch assignment must permute the candidate ID columns")

        fixed = list(observation.fixed_id_labels)
        detection_count = len(_to_numpy(observation.boxes))
        if len(fixed) != detection_count:
            raise ValueError("fixed ID labels must align with activated detections")
        conflict_rows = set(observation.conflict_detection_indices)
        if any(
            (index in conflict_rows) != (label is None)
            for index, label in enumerate(fixed)
        ):
            raise ValueError("only conflict rows may have unresolved fixed ID labels")
        for row, candidate_column in zip(
            observation.conflict_detection_indices, assignment, strict=True
        ):
            fixed[row] = observation.candidate_id_labels[candidate_column]

        _, successor = self.state_adapter.transact(
            state,
            lambda active: self._apply_full_frame(active, observation, fixed),
        )
        return successor

    @staticmethod
    def _apply_full_frame(
        runtime: Any,
        observation: MOTIPBranchObservation,
        resolved_labels: list[int | None],
    ) -> None:
        if any(label is None for label in resolved_labels):
            raise ValueError("all branch labels must be resolved before update")
        label_values = [int(label) for label in resolved_labels if label is not None]
        template = runtime.trajectory_id_labels
        if hasattr(template, "new_tensor"):
            pred_id_labels = template.new_tensor(label_values)
        else:
            pred_id_labels = np.asarray(label_values, dtype=np.int64)

        newborn_label = int(runtime.num_id_vocabulary)
        active_labels = [label for label in label_values if label != newborn_label]
        if len(set(active_labels)) != len(active_labels):
            raise ValueError("a full-frame branch assigns an active ID more than once")
        for label in active_labels:
            runtime.id_queue.add(label)
        newborn_count = sum(label == newborn_label for label in label_values)
        remaining = len(runtime.id_queue) - len(active_labels)
        if newborn_count > remaining:
            raise ValueError("branch requires more newborn labels than MOTIP has free")

        id_labels = runtime._assign_newborn_id_labels(
            pred_id_labels=pred_id_labels
        )
        final_labels = [int(label) for label in id_labels.tolist()]
        if len(set(final_labels)) != len(final_labels):
            raise ValueError("MOTIP branch produced duplicate final ID labels")
        for label in final_labels:
            runtime.id_queue.add(label)

        boxes_xywh = observation.boxes.clone() if hasattr(
            observation.boxes, "clone"
        ) else np.array(observation.boxes, copy=True)
        boxes_xywh[:, :2] -= boxes_xywh[:, 2:] / 2
        boxes_xywh *= runtime.bbox_unnorm
        if hasattr(id_labels, "clone"):
            stable_ids = id_labels.clone()
            for index, label in enumerate(final_labels):
                stable_ids[index] = runtime.id_label_to_id[label]
        else:
            stable_ids = np.asarray(
                [runtime.id_label_to_id[label] for label in final_labels],
                dtype=np.int64,
            )
        runtime.current_track_results = {
            "score": observation.scores,
            "category": observation.categories,
            "bbox": boxes_xywh,
            "id": stable_ids,
        }
        runtime._update_trajectory_infos(
            boxes=observation.boxes,
            output_embeds=observation.output_embeds,
            id_labels=id_labels,
        )
        runtime._filter_out_inactive_tracks()

    @staticmethod
    def _validate_observation(
        state: MOTIPRuntimeState, observation: MOTIPBranchObservation
    ) -> None:
        rows = observation.conflict_detection_indices
        labels = observation.candidate_id_labels
        if not rows or len(rows) != len(labels):
            raise ValueError("MOTIP branch conflicts must be non-empty and square")
        if len(set(rows)) != len(rows) or min(rows) < 0:
            raise ValueError("conflict detection indices must be unique and non-negative")
        if len(set(labels)) != len(labels):
            raise ValueError("candidate ID labels must be unique")
        if any(label < 0 or label >= state.vocabulary_size for label in labels):
            raise ValueError("candidate ID label is outside the MOTIP vocabulary")
        active = set(_to_numpy(state.trajectory_id_labels[0]).astype(int).tolist())
        if not set(labels).issubset(active):
            raise ValueError("candidate ID labels must be active in branch memory")
