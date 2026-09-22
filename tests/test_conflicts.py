import numpy as np
import pytest

from branchmot import decompose_conflicts


def test_independent_matches_remain_independent() -> None:
    components = decompose_conflicts(
        np.array([[0.9, 0.0], [0.0, 0.8]]), candidate_threshold=0.2
    )
    assert [component.detection_indices for component in components] == [(0,), (1,)]
    assert [component.track_indices for component in components] == [(0,), (1,)]
    assert not any(component.ambiguous for component in components)


def test_crossing_targets_form_one_branchable_component() -> None:
    probabilities = np.array(
        [
            [0.49, 0.48, 0.01],
            [0.47, 0.50, 0.01],
            [0.01, 0.01, 0.95],
        ]
    )
    components = decompose_conflicts(probabilities, candidate_threshold=0.2)
    assert components[0].detection_indices == (0, 1)
    assert components[0].track_indices == (0, 1)
    assert components[0].branchable
    np.testing.assert_array_equal(
        components[0].submatrix(probabilities), probabilities[:2, :2]
    )
    assert components[1].detection_indices == (2,)
    assert not components[1].ambiguous


def test_detection_without_track_becomes_birth_component() -> None:
    components = decompose_conflicts(
        np.array([[0.05, 0.02], [0.01, 0.01]]), candidate_threshold=0.1
    )
    assert len(components) == 2
    assert all(component.track_indices == () for component in components)
    assert all(not component.branchable for component in components)


def test_many_to_one_component_is_not_yet_branchable() -> None:
    components = decompose_conflicts(
        np.array([[0.8], [0.7]]), candidate_threshold=0.2
    )
    assert components[0].ambiguous
    assert not components[0].branchable


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold"):
        decompose_conflicts(np.ones((1, 1)), candidate_threshold=1.1)
