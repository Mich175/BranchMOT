"""MOTChallenge/DanceTrack ground-truth parsing and cache annotation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from .cache import AssociationFrame
from .linking import box_iou_xyxy, linear_sum_assignment


@dataclass(frozen=True)
class MOTObject:
    frame_index: int
    track_id: int
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int | None = None
    visibility: float | None = None


@dataclass(frozen=True)
class OcclusionEvent:
    track_id: int
    last_visible_frame: int
    reappearance_frame: int
    missing_frames: int


def read_mot_ground_truth(path: str | Path) -> dict[int, tuple[MOTObject, ...]]:
    """Read MOT rows: frame,id,x,y,width,height,confidence,class,visibility."""

    grouped: dict[int, list[MOTObject]] = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = [field.strip() for field in stripped.replace(" ", ",").split(",")]
            fields = [field for field in fields if field]
            if len(fields) < 6:
                raise ValueError(f"invalid MOT row at line {line_number}")
            try:
                frame_index, track_id = int(float(fields[0])), int(float(fields[1]))
                x, y, width, height = map(float, fields[2:6])
                confidence = float(fields[6]) if len(fields) > 6 else 1.0
                class_id = int(float(fields[7])) if len(fields) > 7 else None
                visibility = float(fields[8]) if len(fields) > 8 else None
            except ValueError as error:
                raise ValueError(f"invalid MOT value at line {line_number}") from error
            if width < 0 or height < 0:
                raise ValueError(f"negative box size at line {line_number}")
            if confidence <= 0:
                continue
            grouped[frame_index].append(
                MOTObject(
                    frame_index,
                    track_id,
                    (x, y, x + width, y + height),
                    confidence,
                    class_id,
                    visibility,
                )
            )
    return {
        frame: tuple(sorted(objects, key=lambda item: item.track_id))
        for frame, objects in sorted(grouped.items())
    }


def annotate_cache_with_ground_truth(
    frames: list[AssociationFrame],
    ground_truth: dict[int, tuple[MOTObject, ...]],
    *,
    frame_offset: int = 1,
    min_iou: float = 0.5,
) -> tuple[AssociationFrame, ...]:
    """Attach a GT identity (or ``None``) to every cached detection."""

    if not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be in [0, 1]")
    annotated: list[AssociationFrame] = []
    for frame in frames:
        frame.validate()
        if frame.boxes_xyxy is None:
            raise ValueError("cached boxes are required for GT annotation")
        objects = ground_truth.get(frame.frame_index + frame_offset, ())
        labels: list[int | None] = [None] * len(frame.detection_ids)
        if objects and frame.detection_ids:
            detection_boxes = np.asarray(frame.boxes_xyxy, dtype=np.float64)
            target_boxes = np.asarray([item.box_xyxy for item in objects])
            iou = box_iou_xyxy(detection_boxes, target_boxes)
            rows, columns = linear_sum_assignment(1.0 - iou)
            for detection, target in zip(rows.tolist(), columns.tolist()):
                if iou[detection, target] >= min_iou:
                    labels[detection] = objects[target].track_id
        annotated.append(replace(frame, ground_truth_track_ids=labels))
    return tuple(annotated)


def find_occlusion_events(
    ground_truth: dict[int, tuple[MOTObject, ...]],
) -> tuple[OcclusionEvent, ...]:
    """Find gaps between visible annotations of the same GT identity."""

    track_frames: dict[int, list[int]] = defaultdict(list)
    for frame_index, objects in ground_truth.items():
        for item in objects:
            track_frames[item.track_id].append(frame_index)
    events: list[OcclusionEvent] = []
    for track_id, frames in track_frames.items():
        ordered = sorted(set(frames))
        for previous, current in pairwise(ordered):
            if current - previous > 1:
                events.append(
                    OcclusionEvent(track_id, previous, current, current - previous - 1)
                )
    return tuple(
        sorted(events, key=lambda event: (event.reappearance_frame, event.track_id))
    )
