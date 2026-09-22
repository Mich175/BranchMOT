import numpy as np

from branchmot import (
    AssociationFrame,
    ObservationChainLinker,
    box_iou_xyxy,
    linear_sum_assignment,
    link_cache_frames,
)


def test_iou_and_hungarian_rectangular_assignment() -> None:
    boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=float)
    np.testing.assert_allclose(box_iou_xyxy(boxes, boxes), np.eye(2))
    rows, columns = linear_sum_assignment(
        np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0]])
    )
    assert tuple(rows) == (0, 1)
    assert tuple(columns) == (1, 0)


def test_linker_preserves_chains_under_constant_motion() -> None:
    linker = ObservationChainLinker(min_iou=0.0, max_age=1, velocity_momentum=0.0)
    first = linker.update(0, np.array([[0, 0, 10, 10], [30, 0, 40, 10]]))
    second = linker.update(1, np.array([[8, 0, 18, 10], [22, 0, 32, 10]]))
    third = linker.update(2, np.array([[16, 0, 26, 10], [14, 0, 24, 10]]))
    assert first == (0, 1)
    assert second == (0, 1)
    assert third == (0, 1)


def test_unmatched_detection_starts_new_chain() -> None:
    linker = ObservationChainLinker(
        min_iou=0.5, max_normalized_center_distance=0.5, max_age=0
    )
    assert linker.update(0, np.array([[0, 0, 10, 10]])) == (0,)
    assert linker.update(1, np.array([[100, 100, 110, 110]])) == (1,)


def test_expired_chain_is_not_reused() -> None:
    linker = ObservationChainLinker(max_age=0)
    assert linker.update(0, np.array([[0, 0, 10, 10]])) == (0,)
    assert linker.update(2, np.array([[0, 0, 10, 10]])) == (1,)


def test_cached_frames_receive_stable_chain_ids() -> None:
    frames = [
        AssociationFrame(
            frame_index=index,
            detection_ids=[0],
            track_ids=[3],
            probabilities=[[1.0]],
            boxes_xyxy=[[float(index), 0.0, float(index + 10), 10.0]],
        )
        for index in range(2)
    ]
    linked = link_cache_frames(frames)
    assert linked[0].detection_ids == [0]
    assert linked[1].detection_ids == [0]
