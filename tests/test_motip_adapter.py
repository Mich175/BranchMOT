import numpy as np
import pytest

from branchmot import make_cache_frame, project_motip_scores


def test_projection_selects_active_ids_and_preserves_newborn_mass() -> None:
    scores = np.array(
        [
            [0.60, 0.05, 0.10, 0.25],
            [0.10, 0.10, 0.70, 0.10],
        ]
    )
    projection = project_motip_scores(scores, [0, 2], newborn_label=3)
    np.testing.assert_allclose(
        projection.track_probabilities,
        [[0.60 / 0.95, 0.10 / 0.95], [0.10 / 0.90, 0.70 / 0.90]],
    )
    np.testing.assert_allclose(
        projection.newborn_probabilities, [0.25 / 0.95, 0.10 / 0.90]
    )


def test_cache_frame_matches_motip_runtime_contract() -> None:
    frame = make_cache_frame(
        frame_index=4,
        detection_ids=[20],
        active_id_labels=[2],
        id_scores=np.array([[0.1, 0.1, 0.7, 0.1]]),
        newborn_label=3,
        ground_truth_track_ids=[2],
    )
    assert frame.track_ids == [2]
    assert frame.probabilities == [[0.875]]
    assert frame.newborn_probabilities == pytest.approx([0.125])


def test_duplicate_active_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        project_motip_scores(np.ones((1, 3)), [0, 0], newborn_label=2)

