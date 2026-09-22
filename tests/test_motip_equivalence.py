from types import SimpleNamespace

import numpy as np

from branchmot import (
    MOTIPStateAdapter,
    compare_motip_states,
    verify_one_frame_equivalence,
)


class _OrderedSet:
    def __init__(self) -> None:
        self.values = [0, 1, 2]

    def add(self, value):
        if value in self.values:
            self.values.remove(value)
        self.values.append(value)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


def _runtime():
    runtime = SimpleNamespace(
        num_id_vocabulary=3,
        next_id=2,
        id_label_to_id={0: 10, 2: 11},
        id_queue=_OrderedSet(),
        trajectory_features=np.ones((1, 2, 3)),
        trajectory_boxes=np.ones((1, 2, 4)),
        trajectory_id_labels=np.array([[0, 2]]),
        trajectory_times=np.zeros((1, 2), dtype=np.int64),
        trajectory_masks=np.zeros((1, 2), dtype=bool),
        current_track_results={},
        bbox_unnorm=np.array([100.0, 100.0, 100.0, 100.0]),
    )

    def assign(*, pred_id_labels):
        return pred_id_labels

    def update_trajectory(*, boxes, output_embeds, id_labels):
        runtime.trajectory_features = np.asarray(output_embeds)[None, ...].copy()
        runtime.trajectory_boxes = np.asarray(boxes)[None, ...].copy()
        runtime.trajectory_id_labels = np.asarray(id_labels)[None, ...].copy()
        runtime.trajectory_times = np.ones((1, len(id_labels)), dtype=np.int64)
        runtime.trajectory_masks = np.zeros((1, len(id_labels)), dtype=bool)

    def update(_image):
        scores = np.array([0.9, 0.8])
        categories = np.array([1, 1])
        boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.3, 0.4, 0.2, 0.2]])
        embeds = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        predicted = np.array([0, 2], dtype=np.int64)
        for label in predicted.tolist():
            runtime.id_queue.add(label)
        labels = runtime._assign_newborn_id_labels(pred_id_labels=predicted)
        for label in labels.tolist():
            runtime.id_queue.add(label)
        boxes_xywh = boxes.copy()
        boxes_xywh[:, :2] -= boxes_xywh[:, 2:] / 2
        boxes_xywh *= runtime.bbox_unnorm
        runtime.current_track_results = {
            "score": scores,
            "category": categories,
            "bbox": boxes_xywh,
            "id": np.asarray(
                [runtime.id_label_to_id[label] for label in labels.tolist()]
            ),
        }
        runtime._update_trajectory_infos(
            boxes=boxes, output_embeds=embeds, id_labels=labels
        )
        runtime._filter_out_inactive_tracks()

    runtime._assign_newborn_id_labels = assign
    runtime._update_trajectory_infos = update_trajectory
    runtime._filter_out_inactive_tracks = lambda: None
    runtime.update = update
    return runtime


def test_official_update_and_branch_replay_are_equivalent() -> None:
    runtime = _runtime()
    report = verify_one_frame_equivalence(runtime, image=object())
    assert report.equivalent
    assert report.mismatched_fields == ()
    np.testing.assert_array_equal(runtime.current_track_results["id"], [10, 11])


def test_state_comparison_reports_exact_field_names() -> None:
    runtime = _runtime()
    adapter = MOTIPStateAdapter(runtime)
    reference = adapter.capture()
    runtime.next_id = 99
    runtime.trajectory_boxes[0, 0, 0] = 123
    candidate = adapter.capture()
    report = compare_motip_states(reference, candidate)
    assert not report.equivalent
    assert report.mismatched_fields == ("next_id", "trajectory_boxes")
