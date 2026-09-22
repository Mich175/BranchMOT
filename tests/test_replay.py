from branchmot import AssociationConfig, AssociationFrame, replay_aligned_episode


def _frame(index: int, probabilities: list[list[float]]) -> AssociationFrame:
    return AssociationFrame(
        frame_index=index,
        detection_ids=[100, 101],
        track_ids=[3, 8],
        probabilities=probabilities,
        ground_truth_track_ids=[3, 8],
    )


def test_delayed_replay_corrects_an_ambiguous_wrong_frame() -> None:
    result = replay_aligned_episode(
        [
            _frame(0, [[0.45, 0.55], [0.55, 0.45]]),
            _frame(1, [[0.95, 0.05], [0.05, 0.95]]),
        ],
        AssociationConfig(beam_size=2, max_delay=3, entropy_threshold=0.5),
    )
    assert result.immediate_correct == 2
    assert result.delayed_correct == 4
    assert result.total == 4
    assert result.delayed_frames == 1


def test_replay_rejects_unaligned_detection_chains() -> None:
    first = _frame(0, [[0.5, 0.5], [0.5, 0.5]])
    second = AssociationFrame(
        frame_index=1,
        detection_ids=[101, 100],
        track_ids=[3, 8],
        probabilities=[[0.9, 0.1], [0.1, 0.9]],
        ground_truth_track_ids=[8, 3],
    )
    try:
        replay_aligned_episode([first, second])
    except ValueError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("unaligned replay should fail closed")
