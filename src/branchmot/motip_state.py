"""Safe snapshot transactions for MOTIP's mutable online runtime state."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeVar

MOTIP_RUNTIME_COMMIT = "ffc0e905ac196a603027eca8d18fb0dff48c8bcc"

_TRAJECTORY_FIELDS = (
    "trajectory_features",
    "trajectory_boxes",
    "trajectory_id_labels",
    "trajectory_times",
    "trajectory_masks",
)

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class MOTIPRuntimeState:
    """All assignment-dependent state owned by upstream ``RuntimeTracker``."""

    vocabulary_size: int
    next_id: int
    id_label_to_id: dict[int, int]
    id_queue_values: tuple[int, ...]
    id_queue_type: type[Any]
    trajectory_features: Any
    trajectory_boxes: Any
    trajectory_id_labels: Any
    trajectory_times: Any
    trajectory_masks: Any
    current_track_results: dict[str, Any]

    def fork(self) -> MOTIPRuntimeState:
        """Return an independent branch state with cloned tensor storage."""

        return _clone_state(self)


def capture_motip_state(runtime_tracker: Any) -> MOTIPRuntimeState:
    """Capture every field mutated by MOTIP's update/ID-recycling path."""

    _require_runtime_fields(runtime_tracker)
    state = MOTIPRuntimeState(
        vocabulary_size=int(runtime_tracker.num_id_vocabulary),
        next_id=int(runtime_tracker.next_id),
        id_label_to_id=dict(runtime_tracker.id_label_to_id),
        id_queue_values=tuple(int(value) for value in runtime_tracker.id_queue),
        id_queue_type=type(runtime_tracker.id_queue),
        trajectory_features=_clone_value(runtime_tracker.trajectory_features),
        trajectory_boxes=_clone_value(runtime_tracker.trajectory_boxes),
        trajectory_id_labels=_clone_value(runtime_tracker.trajectory_id_labels),
        trajectory_times=_clone_value(runtime_tracker.trajectory_times),
        trajectory_masks=_clone_value(runtime_tracker.trajectory_masks),
        current_track_results=_clone_value(runtime_tracker.current_track_results),
    )
    _validate_state(state)
    return state


def restore_motip_state(runtime_tracker: Any, state: MOTIPRuntimeState) -> None:
    """Replace runtime state from a branch without sharing mutable tensors."""

    _require_runtime_fields(runtime_tracker)
    _validate_state(state)
    if int(runtime_tracker.num_id_vocabulary) != state.vocabulary_size:
        raise ValueError("MOTIP state vocabulary does not match the runtime")

    runtime_tracker.next_id = state.next_id
    runtime_tracker.id_label_to_id = dict(state.id_label_to_id)
    queue = state.id_queue_type()
    if not hasattr(queue, "add"):
        raise ValueError("MOTIP id_queue type must provide add()")
    for value in state.id_queue_values:
        queue.add(value)
    runtime_tracker.id_queue = queue
    for field in _TRAJECTORY_FIELDS:
        setattr(runtime_tracker, field, _clone_value(getattr(state, field)))
    runtime_tracker.current_track_results = _clone_value(
        state.current_track_results
    )


class MOTIPStateAdapter:
    """Run branch operations transactionally against one MOTIP tracker object."""

    def __init__(self, runtime_tracker: Any) -> None:
        self.runtime_tracker = runtime_tracker
        _require_runtime_fields(runtime_tracker)

    def capture(self) -> MOTIPRuntimeState:
        return capture_motip_state(self.runtime_tracker)

    def commit(self, state: MOTIPRuntimeState) -> None:
        restore_motip_state(self.runtime_tracker, state)

    def transact(
        self,
        state: MOTIPRuntimeState,
        operation: Callable[[Any], ResultT],
    ) -> tuple[ResultT, MOTIPRuntimeState]:
        """Execute on private state, return its successor, restore canonical state."""

        canonical = self.capture()
        try:
            restore_motip_state(self.runtime_tracker, state)
            result = operation(self.runtime_tracker)
            successor = self.capture()
        finally:
            restore_motip_state(self.runtime_tracker, canonical)
        return result, successor


def _clone_state(state: MOTIPRuntimeState) -> MOTIPRuntimeState:
    return MOTIPRuntimeState(
        vocabulary_size=state.vocabulary_size,
        next_id=state.next_id,
        id_label_to_id=dict(state.id_label_to_id),
        id_queue_values=tuple(state.id_queue_values),
        id_queue_type=state.id_queue_type,
        trajectory_features=_clone_value(state.trajectory_features),
        trajectory_boxes=_clone_value(state.trajectory_boxes),
        trajectory_id_labels=_clone_value(state.trajectory_id_labels),
        trajectory_times=_clone_value(state.trajectory_times),
        trajectory_masks=_clone_value(state.trajectory_masks),
        current_track_results=_clone_value(state.current_track_results),
    )


def _clone_value(value: Any) -> Any:
    if hasattr(value, "clone") and callable(value.clone):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if hasattr(value, "copy") and callable(value.copy):
        return value.copy()
    return deepcopy(value)


def _require_runtime_fields(runtime_tracker: Any) -> None:
    required = {
        "num_id_vocabulary",
        "next_id",
        "id_label_to_id",
        "id_queue",
        "current_track_results",
        *_TRAJECTORY_FIELDS,
    }
    missing = sorted(field for field in required if not hasattr(runtime_tracker, field))
    if missing:
        raise ValueError(f"incompatible MOTIP runtime; missing fields: {missing}")


def _validate_state(state: MOTIPRuntimeState) -> None:
    if state.vocabulary_size < 1 or state.next_id < 0:
        raise ValueError("invalid MOTIP vocabulary or stable ID counter")
    if len(set(state.id_queue_values)) != len(state.id_queue_values):
        raise ValueError("MOTIP id_queue snapshot contains duplicate labels")
    if any(
        label < 0 or label >= state.vocabulary_size
        for label in state.id_queue_values
    ):
        raise ValueError("MOTIP id_queue contains an out-of-vocabulary label")
    shapes = [tuple(getattr(state, field).shape[:2]) for field in _TRAJECTORY_FIELDS]
    if len(set(shapes)) != 1:
        raise ValueError("MOTIP trajectory tensors do not share [time, track] axes")
