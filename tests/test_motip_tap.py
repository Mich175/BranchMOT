from pathlib import Path
from types import SimpleNamespace

import numpy as np

from branchmot import MOTIPScoreTap, probabilities_from_logits, read_jsonl


class _HookHandle:
    def __init__(self, decoder: "_Decoder") -> None:
        self.decoder = decoder

    def remove(self) -> None:
        self.decoder.hook = None


class _Decoder:
    def __init__(self) -> None:
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return _HookHandle(self)

    def emit(self, logits: np.ndarray) -> None:
        assert self.hook is not None
        self.hook(self, (), (logits, None, None))


def _tracker(*, first_frame: bool = False) -> SimpleNamespace:
    decoder = _Decoder()
    tracker = SimpleNamespace(
        model=SimpleNamespace(id_decoder=decoder),
        use_sigmoid=False,
        num_id_vocabulary=3,
        trajectory_id_labels=(
            np.empty((1, 0), dtype=np.int64)
            if first_frame
            else np.array([[0, 2]])
        ),
        id_label_to_id={} if first_frame else {0: 10, 2: 12},
        bbox_unnorm=np.array([100.0, 100.0, 100.0, 100.0]),
    )

    def activate(_detr_out):
        return (
            np.array([0.9, 0.8]),
            np.array([1, 1]),
            np.array([[0.5, 0.5, 0.2, 0.4], [0.3, 0.4, 0.2, 0.2]]),
            None,
        )

    def predict(*_args):
        if first_frame:
            return np.array([3, 3])
        decoder.emit(
            np.array([[[[[3.0, -2.0, 0.5, -1.0], [-1.0, -2.0, 3.0, 0.0]]]]])
        )
        return np.array([0, 2])

    def assign(labels):
        for index in np.flatnonzero(labels == 3):
            new_label = int(index)
            labels[index] = new_label
            tracker.id_label_to_id[new_label] = 20 + new_label
        return labels

    tracker._get_activate_detections = activate
    tracker._get_id_pred_labels = predict
    tracker._assign_newborn_id_labels = assign
    return tracker


def _run_update_boundary(tracker: SimpleNamespace) -> None:
    tracker._get_activate_detections(None)
    labels = tracker._get_id_pred_labels(None)
    tracker._assign_newborn_id_labels(labels)


def test_stable_softmax_handles_large_logits() -> None:
    probabilities = probabilities_from_logits(
        np.array([[1000.0, 999.0]]), use_sigmoid=False
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0])
    assert probabilities[0, 0] > probabilities[0, 1]


def test_tap_captures_final_runtime_identity_state(tmp_path: Path) -> None:
    tracker = _tracker()
    with MOTIPScoreTap(tracker) as tap:
        tap.set_frame_context(
            12, detection_ids=[50, 51], ground_truth_track_ids=[7, 8]
        )
        _run_update_boundary(tracker)
        assert len(tap.frames) == 1
        frame = tap.frames[0]
        assert frame.track_ids == [0, 2]
        assert frame.active_output_ids == [10, 12]
        assert frame.assigned_track_ids == [0, 2]
        assert frame.assigned_output_ids == [10, 12]
        assert frame.detection_ids == [50, 51]
        assert frame.detection_scores == [0.9, 0.8]
        np.testing.assert_allclose(frame.boxes_xyxy[0], [40.0, 30.0, 60.0, 70.0])
        output = tmp_path / "scores.jsonl"
        tap.write(output)

    assert tracker.model.id_decoder.hook is None
    assert list(read_jsonl(output)) == list(tap.frames)


def test_tap_records_first_frame_births_without_decoder_call() -> None:
    tracker = _tracker(first_frame=True)
    with MOTIPScoreTap(tracker) as tap:
        tap.set_frame_context(0, detection_ids=[0, 1], ground_truth_track_ids=[7, 8])
        _run_update_boundary(tracker)

    frame = tap.frames[0]
    assert frame.track_ids == []
    assert frame.probabilities == [[], []]
    assert frame.newborn_probabilities == [1.0, 1.0]
    assert frame.assigned_track_ids == [0, 1]
    assert frame.assigned_output_ids == [20, 21]


def test_tap_rejects_unknown_decoder_shape() -> None:
    tracker = _tracker()
    with MOTIPScoreTap(tracker):
        try:
            tracker.model.id_decoder.emit(np.zeros((2, 4)))
        except ValueError as error:
            assert "shape" in str(error)
        else:
            raise AssertionError("unexpected decoder shape should fail closed")


def test_tap_aligns_rows_after_runtime_filtering() -> None:
    tracker = _tracker()

    def predict(*_args):
        tracker.model.id_decoder.emit(
            np.array([[[[[3.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 3.0]]]]])
        )
        return np.array([0, 3])

    tracker._get_id_pred_labels = predict
    with MOTIPScoreTap(tracker) as tap:
        tap.set_frame_context(2, detection_ids=[40, 41], ground_truth_track_ids=[7, 9])
        tracker._get_activate_detections(None)
        labels = tracker._get_id_pred_labels(None)
        tracker._assign_newborn_id_labels(labels[:1])

    assert tap.frames[0].detection_ids == [40]
    assert tap.frames[0].ground_truth_track_ids == [7]
    assert tap.frames[0].detection_scores == [0.9]


def test_tap_restores_wrapped_runtime_methods() -> None:
    tracker = _tracker()
    originals = (
        tracker._get_activate_detections,
        tracker._get_id_pred_labels,
        tracker._assign_newborn_id_labels,
    )
    tap = MOTIPScoreTap(tracker)
    tap.close()
    assert tracker._get_activate_detections is originals[0]
    assert tracker._get_id_pred_labels is originals[1]
    assert tracker._assign_newborn_id_labels is originals[2]
