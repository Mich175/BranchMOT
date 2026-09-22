"""Command-line evaluation of MOTIP evidence caches against MOT ground truth."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .cache import read_jsonl, write_jsonl
from .identity_targets import (
    attach_internal_identity_targets,
    identity_assignment_metrics,
)
from .mot_ground_truth import annotate_cache_with_ground_truth, read_mot_ground_truth


@dataclass(frozen=True)
class CacheEvaluation:
    frames: int
    detections: int
    ground_truth_matches: int
    evaluable_detections: int
    correct_assignments: int
    assignment_accuracy: float
    causal_coverage: float
    seeded_output_ids: int
    conflicting_observations: int
    ambiguous_targets: int


def evaluate_cache(
    cache_path: str | Path,
    ground_truth_path: str | Path,
    *,
    frame_offset: int = 1,
    min_iou: float = 0.5,
    annotated_cache_path: str | Path | None = None,
) -> CacheEvaluation:
    """Evaluate immediate MOTIP identity decisions with causal GT targets."""

    frames = list(read_jsonl(cache_path))
    ground_truth = read_mot_ground_truth(ground_truth_path)
    matched = annotate_cache_with_ground_truth(
        frames, ground_truth, frame_offset=frame_offset, min_iou=min_iou
    )
    annotated, target_stats = attach_internal_identity_targets(matched)
    metrics = identity_assignment_metrics(annotated)
    if annotated_cache_path is not None:
        write_jsonl(annotated_cache_path, annotated)

    detections = sum(len(frame.detection_ids) for frame in annotated)
    ground_truth_matches = sum(
        ground_truth_id is not None
        for frame in annotated
        for ground_truth_id in (frame.ground_truth_track_ids or [])
    )
    coverage = (
        metrics.evaluated / ground_truth_matches if ground_truth_matches else 0.0
    )
    return CacheEvaluation(
        frames=len(annotated),
        detections=detections,
        ground_truth_matches=ground_truth_matches,
        evaluable_detections=metrics.evaluated,
        correct_assignments=metrics.correct,
        assignment_accuracy=metrics.accuracy,
        causal_coverage=coverage,
        seeded_output_ids=target_stats.seeded_output_ids,
        conflicting_observations=target_stats.conflicting_observations,
        ambiguous_targets=target_stats.ambiguous_targets,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a BranchMOT/MOTIP cache against MOT-format GT"
    )
    parser.add_argument("cache", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--frame-offset", type=int, default=1)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--annotated-cache", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = evaluate_cache(
        args.cache,
        args.ground_truth,
        frame_offset=args.frame_offset,
        min_iou=args.min_iou,
        annotated_cache_path=args.annotated_cache,
    )
    rendered = json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
