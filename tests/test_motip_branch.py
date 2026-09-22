from types import SimpleNamespace

import numpy as np

from branchmot import (
    MOTIPBranchDecoder,
    MOTIPBranchObservation,
    MOTIPStateAdapter,
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


class _Handle:
    def __init__(self, decoder):
        self.decoder = decoder

    def remove(self):
        self.decoder.hook = None


class _Decoder:
    def __init__(self):
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return _Handle(self)

    def emit(self, logits):
        self.hook(self, (), (logits, None, None))


def _runtime():
    decoder = _Decoder()
    runtime = SimpleNamespace(
        model=SimpleNamespace(id_decoder=decoder),
        use_sigmoid=False,
        num_id_vocabulary=3,
        next_id=2,
        id_label_to_id={0: 10, 2: 11},
        id_queue=_OrderedSet(),
        trajectory_features=np.ones((1, 2, 3)),
        trajectory_boxes=np.ones((1, 2, 4)),
        trajectory_id_labels=np.array([[0, 2]]),
        trajectory_times=np.zeros((1, 2)),
        trajectory_masks=np.zeros((1, 2), dtype=bool),
        current_track_results={},
        bbox_unnorm=np.array([100.0, 100.0, 100.0, 100.0]),
    )

    def predict(*, boxes, output_embeds):
        del boxes, output_embeds
        decoder.emit(
            np.array([[[[[3.0, -2.0, 0.0, -1.0], [0.0, -2.0, 3.0, -1.0]]]]])
        )
        return np.array([0, 2])

    runtime._get_id_pred_labels = predict

    def assign_newborn(*, pred_id_labels):
        return pred_id_labels

    def update_trajectory(*, boxes, output_embeds, id_labels):
        runtime.last_update = (
            np.array(boxes),
            np.array(output_embeds),
            np.array(id_labels),
        )
        runtime.trajectory_masks = np.zeros_like(runtime.trajectory_masks)

    runtime._assign_newborn_id_labels = assign_newborn
    runtime._update_trajectory_infos = update_trajectory
    runtime._filter_out_inactive_tracks = lambda: None
    return runtime


def test_branch_decoder_projects_private_runtime_logits() -> None:
    runtime = _runtime()
    adapter = MOTIPStateAdapter(runtime)
    decoder = MOTIPBranchDecoder(adapter)
    state = adapter.capture()
    observation = MOTIPBranchObservation(
        boxes=np.zeros((2, 4)),
        output_embeds=np.zeros((2, 3)),
        conflict_detection_indices=(0, 1),
        candidate_id_labels=(0, 2),
    )
    probabilities = decoder.decode(state, observation)
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]
    assert runtime.model.id_decoder.hook is None
    assert runtime.next_id == 2


def test_branch_decoder_rejects_inactive_candidate_label() -> None:
    runtime = _runtime()
    adapter = MOTIPStateAdapter(runtime)
    decoder = MOTIPBranchDecoder(adapter)
    observation = MOTIPBranchObservation(
        boxes=np.zeros((2, 4)),
        output_embeds=np.zeros((2, 3)),
        conflict_detection_indices=(0, 1),
        candidate_id_labels=(0, 1),
    )
    try:
        decoder.decode(adapter.capture(), observation)
    except ValueError as error:
        assert "active" in str(error)
    else:
        raise AssertionError("inactive candidate label should fail closed")


def test_branch_update_applies_full_frame_labels_only_to_private_state() -> None:
    runtime = _runtime()
    adapter = MOTIPStateAdapter(runtime)
    decoder = MOTIPBranchDecoder(adapter)
    observation = MOTIPBranchObservation(
        boxes=np.array([[0.5, 0.5, 0.2, 0.4], [0.3, 0.4, 0.2, 0.2]]),
        output_embeds=np.zeros((2, 3)),
        conflict_detection_indices=(0, 1),
        candidate_id_labels=(0, 2),
        fixed_id_labels=(None, None),
        scores=np.array([0.9, 0.8]),
        categories=np.array([1, 1]),
    )
    successor = decoder.update(adapter.capture(), observation, (1, 0))

    np.testing.assert_array_equal(successor.current_track_results["id"], [11, 10])
    np.testing.assert_allclose(
        successor.current_track_results["bbox"][0], [40.0, 30.0, 20.0, 40.0]
    )
    assert list(runtime.current_track_results) == []

    adapter.commit(successor)
    np.testing.assert_array_equal(runtime.current_track_results["id"], [11, 10])


def test_branch_update_rejects_duplicate_fixed_identity() -> None:
    runtime = _runtime()
    decoder = MOTIPBranchDecoder(MOTIPStateAdapter(runtime))
    observation = MOTIPBranchObservation(
        boxes=np.zeros((3, 4)),
        output_embeds=np.zeros((3, 3)),
        conflict_detection_indices=(0, 1),
        candidate_id_labels=(0, 2),
        fixed_id_labels=(None, None, 0),
        scores=np.ones(3),
        categories=np.ones(3),
    )
    try:
        decoder.update(decoder.state_adapter.capture(), observation, (0, 1))
    except ValueError as error:
        assert "more than once" in str(error)
    else:
        raise AssertionError("duplicate full-frame identity should fail closed")
