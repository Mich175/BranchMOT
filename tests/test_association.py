import numpy as np
import pytest

from branchmot import AssociationConfig, BranchingAssociator


def test_clear_frame_commits_immediately() -> None:
    tracker = BranchingAssociator()
    assignment = tracker.step(np.array([[0.95, 0.05], [0.04, 0.96]]))
    assert assignment == (0, 1)
    assert tracker.best_path() == ()


def test_ambiguous_crossing_delays_then_uses_future_evidence() -> None:
    tracker = BranchingAssociator(
        AssociationConfig(beam_size=4, max_delay=3, entropy_threshold=0.5)
    )

    assert tracker.step(np.array([[0.51, 0.49], [0.49, 0.51]])) is None
    assert len(tracker.hypotheses) > 1

    assignment = tracker.step(np.array([[0.90, 0.10], [0.10, 0.90]]))
    assert assignment == (0, 1)


def test_max_delay_forces_bounded_commitment() -> None:
    tracker = BranchingAssociator(
        AssociationConfig(beam_size=2, max_delay=1, entropy_threshold=0.1)
    )
    ambiguous = np.array([[0.5, 0.5], [0.5, 0.5]])

    assert tracker.step(ambiguous) is None
    assert tracker.step(ambiguous) in {(0, 1), (1, 0)}


def test_invalid_matrix_is_rejected() -> None:
    tracker = BranchingAssociator()
    with pytest.raises(ValueError, match="square"):
        tracker.step(np.ones((2, 3)))

