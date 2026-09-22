"""One-frame equivalence gate against unmodified upstream MOTIP inference."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from .motip_branch import MOTIPBranchDecoder
from .motip_state import MOTIPRuntimeState, MOTIPStateAdapter
from .motip_tap import _to_numpy


@dataclass(frozen=True)
class MOTIPEquivalenceReport:
    equivalent: bool
    mismatched_fields: tuple[str, ...]


def compare_motip_states(
    reference: MOTIPRuntimeState,
    candidate: MOTIPRuntimeState,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> MOTIPEquivalenceReport:
    """Compare complete runtime states, including nested result tensors."""

    mismatches: list[str] = []
    for field in fields(MOTIPRuntimeState):
        name = field.name
        if name == "id_queue_type":
            continue
        if not _values_equal(
            getattr(reference, name), getattr(candidate, name), atol=atol, rtol=rtol
        ):
            mismatches.append(name)
    return MOTIPEquivalenceReport(not mismatches, tuple(mismatches))


def verify_one_frame_equivalence(
    runtime_tracker: Any,
    image: Any,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> MOTIPEquivalenceReport:
    """Run official update, replay its decisions, and compare every state field.

    The tracker is left in the official post-frame state whether equivalence
    succeeds or raises. This makes the check usable while iterating a sequence.
    """

    adapter = MOTIPStateAdapter(runtime_tracker)
    decoder = MOTIPBranchDecoder(adapter)
    initial = adapter.capture()
    captured: dict[str, Any] = {}
    original_assign = runtime_tracker._assign_newborn_id_labels
    original_update = runtime_tracker._update_trajectory_infos

    def capture_assign(*args: Any, **kwargs: Any) -> Any:
        labels = kwargs.get("pred_id_labels", args[0] if args else None)
        captured["pred_id_labels"] = _clone(labels)
        return original_assign(*args, **kwargs)

    def capture_update(*args: Any, **kwargs: Any) -> Any:
        captured["boxes"] = _clone(kwargs.get("boxes", args[0] if args else None))
        captured["output_embeds"] = _clone(
            kwargs.get("output_embeds", args[1] if len(args) > 1 else None)
        )
        return original_update(*args, **kwargs)

    runtime_tracker._assign_newborn_id_labels = capture_assign
    runtime_tracker._update_trajectory_infos = capture_update
    try:
        runtime_tracker.update(image)
    finally:
        runtime_tracker._assign_newborn_id_labels = original_assign
        runtime_tracker._update_trajectory_infos = original_update
    official = adapter.capture()

    try:
        required = {"pred_id_labels", "boxes", "output_embeds"}
        if missing := sorted(required - captured.keys()):
            raise ValueError(f"official MOTIP update did not expose: {missing}")
        results = official.current_track_results
        replayed = decoder.apply_resolved(
            initial,
            boxes=captured["boxes"],
            output_embeds=captured["output_embeds"],
            scores=results["score"],
            categories=results["category"],
            pred_id_labels=_to_numpy(captured["pred_id_labels"])
            .astype(np.int64)
            .tolist(),
        )
        return compare_motip_states(official, replayed, atol=atol, rtol=rtol)
    finally:
        adapter.commit(official)


def assert_one_frame_equivalent(
    runtime_tracker: Any, image: Any, *, atol: float = 1e-7, rtol: float = 1e-6
) -> None:
    report = verify_one_frame_equivalence(
        runtime_tracker, image, atol=atol, rtol=rtol
    )
    if not report.equivalent:
        raise AssertionError(
            "BranchMOT replay differs from official MOTIP fields: "
            + ", ".join(report.mismatched_fields)
        )


def _clone(value: Any) -> Any:
    if value is None:
        raise ValueError("MOTIP equivalence hook received a missing tensor")
    if hasattr(value, "clone"):
        return value.clone()
    return np.array(value, copy=True)


def _values_equal(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key], atol=atol, rtol=rtol)
            for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _values_equal(a, b, atol=atol, rtol=rtol)
            for a, b in zip(left, right, strict=True)
        )
    if hasattr(left, "shape") or hasattr(right, "shape"):
        try:
            return bool(
                np.allclose(
                    _to_numpy(left),
                    _to_numpy(right),
                    atol=atol,
                    rtol=rtol,
                    equal_nan=True,
                )
            )
        except (TypeError, ValueError):
            return False
    return bool(left == right)
