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
    )

    def predict(*, boxes, output_embeds):
        del boxes, output_embeds
        decoder.emit(
            np.array([[[[[3.0, -2.0, 0.0, -1.0], [0.0, -2.0, 3.0, -1.0]]]]])
        )
        return np.array([0, 2])

    runtime._get_id_pred_labels = predict
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
