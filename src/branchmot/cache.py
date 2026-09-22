"""Versioned interchange format for detector-to-track association evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AssociationFrame:
    """Association evidence for one frame or one local conflict subgraph."""

    frame_index: int
    detection_ids: list[int]
    track_ids: list[int]
    probabilities: list[list[float]]
    ground_truth_track_ids: list[int] | None = None
    newborn_probabilities: list[float] | None = None
    boxes_xyxy: list[list[float]] | None = None

    def validate(self) -> None:
        probs = np.asarray(self.probabilities, dtype=np.float64)
        expected = (len(self.detection_ids), len(self.track_ids))
        if probs.shape != expected:
            raise ValueError(f"probability shape {probs.shape} does not match {expected}")
        if len(set(self.detection_ids)) != len(self.detection_ids):
            raise ValueError("detection_ids must be unique within a frame")
        if len(set(self.track_ids)) != len(self.track_ids):
            raise ValueError("track_ids must be unique within a frame")
        if np.any(~np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if probs.size and np.any(probs.sum(axis=1) > 1.0 + 1e-6):
            raise ValueError("track probabilities cannot sum to more than one")
        if self.ground_truth_track_ids is not None and len(
            self.ground_truth_track_ids
        ) != len(self.detection_ids):
            raise ValueError("ground-truth IDs must align with detections")
        if self.newborn_probabilities is not None:
            newborn = np.asarray(self.newborn_probabilities, dtype=np.float64)
            if newborn.shape != (len(self.detection_ids),):
                raise ValueError("newborn probabilities must align with detections")
            if np.any(~np.isfinite(newborn)) or np.any(newborn < 0):
                raise ValueError("newborn probabilities must be finite and non-negative")
            total = probs.sum(axis=1) + newborn
            if np.any(np.abs(total - 1.0) > 1e-5):
                raise ValueError("track and newborn probabilities must sum to one")
        if self.boxes_xyxy is not None:
            boxes = np.asarray(self.boxes_xyxy, dtype=np.float64)
            if boxes.shape != (len(self.detection_ids), 4):
                raise ValueError("boxes must have shape [detections, 4]")
            if np.any(~np.isfinite(boxes)) or np.any(boxes[:, 2:] < boxes[:, :2]):
                raise ValueError("boxes must be finite, valid xyxy coordinates")


def write_jsonl(path: str | Path, frames: Iterable[AssociationFrame]) -> None:
    """Write validated frames as portable, streamable JSON Lines."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema_version": SCHEMA_VERSION}) + "\n")
        for frame in frames:
            frame.validate()
            handle.write(json.dumps(asdict(frame), separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> Iterator[AssociationFrame]:
    """Read and validate association frames without loading the full sequence."""

    with Path(path).open(encoding="utf-8") as handle:
        try:
            header = json.loads(next(handle))
        except StopIteration as error:
            raise ValueError("association cache is empty") from error
        if header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported association schema: {header}")
        previous_frame = -1
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            try:
                frame = AssociationFrame(**json.loads(line))
                frame.validate()
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid cache record at line {line_number}") from error
            if frame.frame_index <= previous_frame:
                raise ValueError("frame indices must be strictly increasing")
            previous_frame = frame.frame_index
            yield frame
