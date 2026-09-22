import json
from pathlib import Path

from branchmot import AssociationFrame, evaluate_cache, read_jsonl, write_jsonl
from branchmot.evaluate import main


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    frames = [
        AssociationFrame(
            frame_index=0,
            detection_ids=[0],
            track_ids=[],
            probabilities=[[]],
            newborn_probabilities=[1.0],
            boxes_xyxy=[[0.0, 0.0, 10.0, 10.0]],
            assigned_track_ids=[0],
            active_output_ids=[],
            assigned_output_ids=[10],
        ),
        AssociationFrame(
            frame_index=1,
            detection_ids=[0],
            track_ids=[0],
            probabilities=[[1.0]],
            newborn_probabilities=[0.0],
            boxes_xyxy=[[1.0, 0.0, 11.0, 10.0]],
            assigned_track_ids=[0],
            active_output_ids=[10],
            assigned_output_ids=[10],
        ),
    ]
    cache = tmp_path / "cache.jsonl"
    write_jsonl(cache, frames)
    ground_truth = tmp_path / "gt.txt"
    ground_truth.write_text(
        "1,7,0,0,10,10,1,1,1\n2,7,1,0,10,10,1,1,1\n",
        encoding="utf-8",
    )
    return cache, ground_truth


def test_evaluate_cache_reports_causal_coverage(tmp_path: Path) -> None:
    cache, ground_truth = _write_inputs(tmp_path)
    annotated = tmp_path / "annotated.jsonl"
    result = evaluate_cache(cache, ground_truth, annotated_cache_path=annotated)
    assert result.frames == 2
    assert result.ground_truth_matches == 2
    assert result.evaluable_detections == 1
    assert result.correct_assignments == 1
    assert result.assignment_accuracy == 1.0
    assert result.causal_coverage == 0.5
    assert list(read_jsonl(annotated))[1].ground_truth_internal_ids == [0]


def test_evaluate_cli_writes_json_report(tmp_path: Path) -> None:
    cache, ground_truth = _write_inputs(tmp_path)
    output = tmp_path / "report.json"
    main([str(cache), str(ground_truth), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["assignment_accuracy"] == 1.0
    assert report["seeded_output_ids"] == 1
