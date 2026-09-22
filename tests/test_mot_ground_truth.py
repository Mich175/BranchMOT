from pathlib import Path

from branchmot import AssociationFrame
from branchmot.mot_ground_truth import (
    annotate_cache_with_ground_truth,
    find_occlusion_events,
    read_mot_ground_truth,
)


def test_parser_annotation_and_occlusion_events(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(
        "1,7,10,20,30,40,1,1,0.9\n"
        "1,8,100,20,20,40,1,1,0.8\n"
        "3,7,14,20,30,40,1,1,0.9\n"
        "3,9,200,20,20,40,0,1,0.7\n",
        encoding="utf-8",
    )
    ground_truth = read_mot_ground_truth(gt_path)
    assert [item.track_id for item in ground_truth[1]] == [7, 8]
    assert 9 not in [item.track_id for item in ground_truth[3]]

    frame = AssociationFrame(
        frame_index=0,
        detection_ids=[0, 1, 2],
        track_ids=[],
        probabilities=[[], [], []],
        newborn_probabilities=[1.0, 1.0, 1.0],
        boxes_xyxy=[
            [10.0, 20.0, 40.0, 60.0],
            [100.0, 20.0, 120.0, 60.0],
            [300.0, 300.0, 320.0, 320.0],
        ],
    )
    annotated = annotate_cache_with_ground_truth([frame], ground_truth)
    assert annotated[0].ground_truth_track_ids == [7, 8, None]

    events = find_occlusion_events(ground_truth)
    assert len(events) == 1
    assert events[0].track_id == 7
    assert events[0].missing_frames == 1


def test_parser_rejects_negative_box_size(tmp_path: Path) -> None:
    path = tmp_path / "gt.txt"
    path.write_text("1,1,0,0,-1,5,1\n", encoding="utf-8")
    try:
        read_mot_ground_truth(path)
    except ValueError as error:
        assert "negative" in str(error)
    else:
        raise AssertionError("negative box size should fail closed")
