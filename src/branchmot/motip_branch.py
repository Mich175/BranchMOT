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
