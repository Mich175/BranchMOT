import numpy as np

from branchmot import (
    AssociationConfig,
    LifecycleAssociator,
    enumerate_lifecycle_assignments,
)


def test_enumeration_allows_multiple_births_but_unique_tracks() -> None:
    assignments = enumerate_lifecycle_assignments(2, (7,))
    assert assignments == ((None, None), (None, 7), (7, None))


def test_rectangular_many_to_one_selects_track_and_birth() -> None:
    tracker = LifecycleAssociator(
        AssociationConfig(entropy_threshold=1.1, margin_threshold=0.0)
    )
    decision = tracker.step(
        np.array([[0.9], [0.2]]),
        np.array([0.1, 0.8]),
        [7],
    )
    assert decision == (7, None)


def test_unmatched_track_is_allowed() -> None:
    tracker = LifecycleAssociator(
        AssociationConfig(entropy_threshold=1.1, margin_threshold=0.0),
        missed_track_probability=0.9,
    )
    decision = tracker.step(
        np.array([[0.1, 0.8]]),
        np.array([0.1]),
        [3, 8],
    )
    assert decision == (8,)


def test_future_evidence_resolves_track_versus_birth() -> None:
    tracker = LifecycleAssociator(
        AssociationConfig(beam_size=3, max_delay=3, entropy_threshold=0.5)
    )
    assert tracker.step(np.array([[0.45]]), np.array([0.55]), [3]) is None
    decision = tracker.step(np.array([[0.95]]), np.array([0.05]), [3])
    assert decision == (3,)


def test_track_set_change_fails_closed_during_delay() -> None:
    tracker = LifecycleAssociator(
        AssociationConfig(beam_size=2, max_delay=3, entropy_threshold=0.2)
    )
    assert tracker.step(np.array([[0.5]]), np.array([0.5]), [3]) is None
    try:
        tracker.step(np.array([[0.5]]), np.array([0.5]), [8])
    except ValueError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("changed track set should fail closed")
