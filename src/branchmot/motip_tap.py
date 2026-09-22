"""Non-invasive score capture for an installed MOTIP runtime tracker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from .cache import AssociationFrame, write_jsonl
from .motip_adapter import make_cache_frame


def probabilities_from_logits(logits: np.ndarray, *, use_sigmoid: bool) -> np.ndarray:
    """Match MOTIP's runtime softmax/sigmoid conversion using NumPy."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must have shape [detections, vocabulary]")
    if values.shape[0] == 0:
        return np.empty_like(values)
    if use_sigmoid:
        positive = values >= 0
        probabilities = np.empty_like(values)
        probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        probabilities[~positive] = exp_values / (1.0 + exp_values)
        return probabilities
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _to_numpy(value: Any) -> np.ndarray:
    for method in ("detach", "float", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _unwrap_model(model: Any) -> Any:
    """Unwrap common distributed wrappers without importing MOTIP."""

    seen: set[int] = set()
    current = model
    while hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        current = current.module
    return current


@dataclass
class _FrameContext:
    frame_index: int
    detection_ids: list[int] | None
    ground_truth_track_ids: list[int | None] | None


class MOTIPScoreTap:
    """Capture ID logits with a PyTorch forward hook.

    Call :meth:`set_frame_context` immediately before each
    ``runtime_tracker.update(image)``. The first video frame normally produces
    no ID-decoder call because MOTIP has no trajectory memory yet.
    """

    def __init__(self, runtime_tracker: Any) -> None:
        self.runtime_tracker = runtime_tracker
        model = _unwrap_model(runtime_tracker.model)
        decoder = getattr(model, "id_decoder", None)
        if decoder is None or not hasattr(decoder, "register_forward_hook"):
            raise ValueError("could not locate a hookable MOTIP id_decoder")
        self._handle = decoder.register_forward_hook(self._capture)
        self._context: _FrameContext | None = None
        self._frames: list[AssociationFrame] = []
        self._fallback_frame_index = 0
        self._pending_boxes_xyxy: list[list[float]] | None = None
        self._pending_detection_scores: list[float] | None = None
        self._pending_id_scores: np.ndarray | None = None
        self._pending_pred_labels: np.ndarray | None = None
        self._pending_active_labels: list[int] | None = None
        self._pending_active_output_ids: list[int] | None = None
        self._original_activate = getattr(
            runtime_tracker, "_get_activate_detections", None
        )
        if self._original_activate is not None:
            runtime_tracker._get_activate_detections = self._capture_detections
        self._original_predict = getattr(runtime_tracker, "_get_id_pred_labels", None)
        if self._original_predict is not None:
            runtime_tracker._get_id_pred_labels = self._capture_predictions
        self._original_assign = getattr(
            runtime_tracker, "_assign_newborn_id_labels", None
        )
        if self._original_assign is not None:
            runtime_tracker._assign_newborn_id_labels = self._capture_assignment

    @property
    def frames(self) -> tuple[AssociationFrame, ...]:
        return tuple(self._frames)

    def set_frame_context(
        self,
        frame_index: int,
        *,
        detection_ids: list[int] | None = None,
        ground_truth_track_ids: list[int | None] | None = None,
    ) -> None:
        """Attach stable observation-chain IDs and optional evaluation labels."""

        self._context = _FrameContext(
            frame_index, detection_ids, ground_truth_track_ids
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._original_activate is not None:
            self.runtime_tracker._get_activate_detections = self._original_activate
            self._original_activate = None
        if self._original_predict is not None:
            self.runtime_tracker._get_id_pred_labels = self._original_predict
            self._original_predict = None
        if self._original_assign is not None:
            self.runtime_tracker._assign_newborn_id_labels = self._original_assign
            self._original_assign = None

    def write(self, path: str | Path) -> None:
        write_jsonl(path, self._frames)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> None:
        raw_logits = output[0] if isinstance(output, (tuple, list)) else output
        logits = _to_numpy(raw_logits)
        if logits.ndim != 5 or logits.shape[:3] != (1, 1, 1):
            raise ValueError(
                "unexpected MOTIP ID-logit shape; expected [1, 1, 1, N, V]"
            )
        logits = logits[0, 0, 0]
        scores = probabilities_from_logits(
            logits, use_sigmoid=bool(self.runtime_tracker.use_sigmoid)
        )

        trajectory_labels = _to_numpy(self.runtime_tracker.trajectory_id_labels[0])
        active_id_labels = trajectory_labels.astype(np.int64).tolist()
        self._pending_id_scores = scores
        self._pending_active_labels = active_id_labels
        self._pending_active_output_ids = self._stable_output_ids(active_id_labels)

    def _capture_predictions(self, *args: Any, **kwargs: Any) -> Any:
        """Retain pre-threshold labels so later filtering can be reproduced."""

        result = self._original_predict(*args, **kwargs)
        self._pending_pred_labels = _to_numpy(result).astype(np.int64).reshape(-1)
        return result

    def _capture_assignment(self, pred_id_labels: Any, *args: Any, **kwargs: Any) -> Any:
        """Finalize a cache row after MOTIP filters and allocates newborn IDs."""

        labels_before = _to_numpy(pred_id_labels).astype(np.int64).reshape(-1).copy()
        result = self._original_assign(pred_id_labels, *args, **kwargs)
        labels_after = _to_numpy(
            pred_id_labels if result is None else result
        ).astype(np.int64).reshape(-1)
        self._finalize_frame(labels_before, labels_after)
        return result

    def _finalize_frame(
        self, labels_before: np.ndarray, labels_after: np.ndarray
    ) -> None:
        if labels_before.shape != labels_after.shape:
            raise ValueError("MOTIP newborn assignment changed the detection count")

        source_labels = self._pending_pred_labels
        if source_labels is None:
            source_labels = labels_before
        kept_indices = _subsequence_indices(source_labels, labels_before)
        active_labels = self._pending_active_labels or []
        newborn_label = int(self.runtime_tracker.num_id_vocabulary)
        id_scores = self._pending_id_scores
        if id_scores is None:
            id_scores = np.zeros((len(source_labels), newborn_label + 1))
            id_scores[:, newborn_label] = 1.0
        if id_scores.shape[0] != len(source_labels):
            raise ValueError("MOTIP logits and predicted labels have different row counts")

        context = self._context
        frame_index = (
            context.frame_index if context is not None else self._fallback_frame_index
        )
        all_detection_ids = (
            context.detection_ids
            if context is not None and context.detection_ids is not None
            else list(range(len(source_labels)))
        )
        all_ground_truth = (
            context.ground_truth_track_ids if context is not None else None
        )
        detection_ids = _select(all_detection_ids, kept_indices, "detection IDs")
        ground_truth = (
            _select(all_ground_truth, kept_indices, "ground-truth IDs")
            if all_ground_truth is not None
            else None
        )
        boxes = (
            _select(self._pending_boxes_xyxy, kept_indices, "boxes")
            if self._pending_boxes_xyxy is not None
            else None
        )
        detection_scores = (
            _select(self._pending_detection_scores, kept_indices, "detection scores")
            if self._pending_detection_scores is not None
            else None
        )
        frame = make_cache_frame(
            frame_index=frame_index,
            detection_ids=detection_ids,
            active_id_labels=active_labels,
            id_scores=id_scores[kept_indices],
            newborn_label=newborn_label,
            ground_truth_track_ids=ground_truth,
            boxes_xyxy=boxes,
            detection_scores=detection_scores,
            assigned_track_ids=labels_after.tolist(),
            active_output_ids=self._pending_active_output_ids,
            assigned_output_ids=self._stable_output_ids(labels_after.tolist()),
        )
        self._frames.append(frame)
        self._context = None
        self._fallback_frame_index = frame_index + 1
        self._pending_boxes_xyxy = None
        self._pending_detection_scores = None
        self._pending_id_scores = None
        self._pending_pred_labels = None
        self._pending_active_labels = None
        self._pending_active_output_ids = None

    def _stable_output_ids(self, labels: list[int]) -> list[int] | None:
        mapping = getattr(self.runtime_tracker, "id_label_to_id", None)
        if mapping is None or any(label not in mapping for label in labels):
            return None
        return [int(mapping[label]) for label in labels]

    def _capture_detections(self, *args: Any, **kwargs: Any) -> Any:
        """Call MOTIP's original selector and retain its exact activated boxes."""

        result = self._original_activate(*args, **kwargs)
        detection_scores = _to_numpy(result[0]).astype(np.float64).reshape(-1)
        boxes_cxcywh = _to_numpy(result[2]).astype(np.float64)
        if detection_scores.shape != (len(boxes_cxcywh),):
            raise ValueError("MOTIP detection scores must align with activated boxes")
        boxes_xyxy = np.empty_like(boxes_cxcywh)
        boxes_xyxy[:, :2] = boxes_cxcywh[:, :2] - boxes_cxcywh[:, 2:] / 2.0
        boxes_xyxy[:, 2:] = boxes_cxcywh[:, :2] + boxes_cxcywh[:, 2:] / 2.0
        if hasattr(self.runtime_tracker, "bbox_unnorm"):
            scale = _to_numpy(self.runtime_tracker.bbox_unnorm).astype(np.float64)
            if scale.shape != (4,):
                raise ValueError("MOTIP bbox_unnorm must have four coordinates")
            boxes_xyxy *= scale
        self._pending_boxes_xyxy = boxes_xyxy.tolist()
        self._pending_detection_scores = detection_scores.tolist()
        return result


def _subsequence_indices(source: np.ndarray, target: np.ndarray) -> list[int]:
    """Locate an order-preserving filtered tensor, including repeated labels."""

    indices: list[int] = []
    cursor = 0
    for value in target.tolist():
        while cursor < len(source) and int(source[cursor]) != int(value):
            cursor += 1
        if cursor == len(source):
            raise ValueError("filtered MOTIP labels are not a subsequence of predictions")
        indices.append(cursor)
        cursor += 1
    return indices


def _select(values: list[Any], indices: list[int], name: str) -> list[Any]:
    if len(values) < (max(indices) + 1 if indices else 0):
        raise ValueError(f"{name} do not align with activated detections")
    return [values[index] for index in indices]
