from pathlib import Path

import numpy as np
import pytest

from branchmot import AssociationFrame, read_jsonl, write_jsonl


def test_cache_round_trip(tmp_path: Path) -> None:
    frame = AssociationFrame(
        frame_index=7,
        detection_ids=[10, 11],
        track_ids=[3, 8],
        probabilities=[[0.7, 0.2], [0.1, 0.8]],
        ground_truth_track_ids=[3, 8],
        newborn_probabilities=[0.1, 0.1],
    )
    path = tmp_path / "sequence.jsonl"
    write_jsonl(path, [frame])
    assert list(read_jsonl(path)) == [frame]


def test_reader_accepts_version_one_cache(tmp_path: Path) -> None:
    path = tmp_path / "old.jsonl"
    path.write_text(
        '{"schema_version":1}\n'
        '{"frame_index":0,"detection_ids":[],"track_ids":[],"probabilities":[]}\n',
        encoding="utf-8",
    )
    assert next(read_jsonl(path)).frame_index == 0


def test_cache_rejects_probability_mass_above_one() -> None:
    frame = AssociationFrame(
        frame_index=0,
        detection_ids=[0],
        track_ids=[1, 2],
        probabilities=[[0.8, 0.5]],
    )
    with pytest.raises(ValueError, match="sum"):
        frame.validate()


def test_cache_rejects_non_finite_values() -> None:
    frame = AssociationFrame(
        frame_index=0,
        detection_ids=[0],
        track_ids=[1],
        probabilities=[[np.nan]],
    )
    with pytest.raises(ValueError, match="finite"):
        frame.validate()


def test_cache_validates_optional_boxes() -> None:
    frame = AssociationFrame(
        frame_index=0,
        detection_ids=[0],
        track_ids=[1],
        probabilities=[[1.0]],
        boxes_xyxy=[[5.0, 5.0, 4.0, 8.0]],
    )
    with pytest.raises(ValueError, match="boxes"):
        frame.validate()
