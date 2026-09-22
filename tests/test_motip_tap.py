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


def _tracker() -> SimpleNamespace:
    decoder = _Decoder()
    def activate(_detr_out):
        return None, None, np.array([[0.5, 0.5, 0.2, 0.4]]), None

    return SimpleNamespace(
        model=SimpleNamespace(id_decoder=decoder),
        use_sigmoid=False,
        num_id_vocabulary=3,
        trajectory_id_labels=np.array([[0, 2]]),
        _get_activate_detections=activate,
    )


def test_stable_softmax_handles_large_logits() -> None:
    probabilities = probabilities_from_logits(
        np.array([[1000.0, 999.0]]), use_sigmoid=False
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0])
    assert probabilities[0, 0] > probabilities[0, 1]


def test_tap_captures_and_writes_runtime_scores(tmp_path: Path) -> None:
    tracker = _tracker()
    with MOTIPScoreTap(tracker) as tap:
        tap.set_frame_context(
            12, detection_ids=[50, 51], ground_truth_track_ids=[0, 2]
        )
        tracker.model.id_decoder.emit(
            np.array([[[[[3.0, -2.0, 0.5, -1.0], [-1.0, -2.0, 3.0, 0.0]]]]])
        )
        assert len(tap.frames) == 1
        assert tap.frames[0].track_ids == [0, 2]
        assert tap.frames[0].detection_ids == [50, 51]
        output = tmp_path / "scores.jsonl"
        tap.write(output)

    assert tracker.model.id_decoder.hook is None
    assert list(read_jsonl(output)) == list(tap.frames)


def test_tap_rejects_unknown_decoder_shape() -> None:
    tracker = _tracker()
    with MOTIPScoreTap(tracker):
        try:
            tracker.model.id_decoder.emit(np.zeros((2, 4)))
        except ValueError as error:
            assert "shape" in str(error)
        else:
            raise AssertionError("unexpected decoder shape should fail closed")


def test_tap_captures_activated_boxes_and_restores_method() -> None:
    tracker = _tracker()
    original = tracker._get_activate_detections
    with MOTIPScoreTap(tracker) as tap:
        tracker._get_activate_detections(None)
        tap.set_frame_context(1, detection_ids=[5, 6])
        # Two logits rows require two boxes, so duplicate the captured box.
        tap._pending_boxes_xyxy *= 2
        tracker.model.id_decoder.emit(np.zeros((1, 1, 1, 2, 4)))
        np.testing.assert_allclose(
            tap.frames[0].boxes_xyxy[0], [0.4, 0.3, 0.6, 0.7]
        )
    assert tracker._get_activate_detections is original
