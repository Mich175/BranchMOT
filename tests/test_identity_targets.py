from branchmot import (
    AssociationFrame,
    attach_internal_identity_targets,
    identity_assignment_metrics,
)


def _frame(
    frame_index: int,
    *,
    track_ids: list[int],
    active_output_ids: list[int],
    assigned_track_ids: list[int],
    assigned_output_ids: list[int],
    ground_truth_ids: list[int],
) -> AssociationFrame:
    detections = len(ground_truth_ids)
    return AssociationFrame(
        frame_index=frame_index,
        detection_ids=list(range(detections)),
        track_ids=track_ids,
        probabilities=[
            [1.0 / len(track_ids)] * len(track_ids) if track_ids else []
            for _ in range(detections)
        ],
        newborn_probabilities=[0.0 if track_ids else 1.0] * detections,
        active_output_ids=active_output_ids,
        assigned_track_ids=assigned_track_ids,
        assigned_output_ids=assigned_output_ids,
        ground_truth_track_ids=ground_truth_ids,
    )


def test_targets_use_only_identity_mapping_from_previous_frames() -> None:
    birth = _frame(
        0,
        track_ids=[],
        active_output_ids=[],
        assigned_track_ids=[0, 1],
        assigned_output_ids=[10, 11],
        ground_truth_ids=[7, 8],
    )
    continuation = _frame(
        1,
        track_ids=[0, 1],
        active_output_ids=[10, 11],
        assigned_track_ids=[1, 0],
        assigned_output_ids=[11, 10],
        ground_truth_ids=[7, 8],
    )

    annotated, stats = attach_internal_identity_targets([birth, continuation])

    assert annotated[0].ground_truth_internal_ids == [None, None]
    assert annotated[1].ground_truth_internal_ids == [0, 1]
    assert stats.seeded_output_ids == 2
    assert stats.evaluated_detections == 2
    metrics = identity_assignment_metrics(annotated)
    assert metrics.correct == 0
    assert metrics.evaluated == 2
    assert metrics.accuracy == 0.0


def test_mapping_is_immutable_when_output_id_switches_ground_truth() -> None:
    frames = [
        _frame(
            0,
            track_ids=[],
            active_output_ids=[],
            assigned_track_ids=[0],
            assigned_output_ids=[10],
            ground_truth_ids=[7],
        ),
        _frame(
            1,
            track_ids=[0],
            active_output_ids=[10],
            assigned_track_ids=[0],
            assigned_output_ids=[10],
            ground_truth_ids=[8],
        ),
    ]
    annotated, stats = attach_internal_identity_targets(frames)
    assert annotated[1].ground_truth_internal_ids == [None]
    assert stats.conflicting_observations == 1


def test_fragmented_output_id_can_seed_same_ground_truth_for_future_frames() -> None:
    frames = [
        _frame(
            0,
            track_ids=[],
            active_output_ids=[],
            assigned_track_ids=[0],
            assigned_output_ids=[10],
            ground_truth_ids=[7],
        ),
        _frame(
            1,
            track_ids=[],
            active_output_ids=[],
            assigned_track_ids=[1],
            assigned_output_ids=[20],
            ground_truth_ids=[7],
        ),
        _frame(
            2,
            track_ids=[1],
            active_output_ids=[20],
            assigned_track_ids=[1],
            assigned_output_ids=[20],
            ground_truth_ids=[7],
        ),
    ]
    annotated, _ = attach_internal_identity_targets(frames)
    assert annotated[2].ground_truth_internal_ids == [1]
