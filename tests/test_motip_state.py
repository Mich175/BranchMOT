from types import SimpleNamespace

import numpy as np
import pytest

from branchmot import MOTIPStateAdapter, capture_motip_state, restore_motip_state


class _OrderedSet:
    def __init__(self) -> None:
        self.values: list[int] = []

    def add(self, value: int) -> None:
        if value in self.values:
            self.values.remove(value)
        self.values.append(value)

    def __iter__(self):
        return iter(self.values)


def _runtime():
    queue = _OrderedSet()
    queue.add(2)
    queue.add(0)
    return SimpleNamespace(
        num_id_vocabulary=3,
        next_id=12,
        id_label_to_id={0: 10, 2: 11},
        id_queue=queue,
        trajectory_features=np.ones((2, 2, 3)),
        trajectory_boxes=np.ones((2, 2, 4)),
        trajectory_id_labels=np.array([[0, 2], [0, 2]]),
        trajectory_times=np.array([[0, 0], [1, 1]]),
        trajectory_masks=np.zeros((2, 2), dtype=bool),
        current_track_results={"id": np.array([10, 11])},
    )


def test_snapshot_and_restore_do_not_share_mutable_storage() -> None:
    runtime = _runtime()
    state = capture_motip_state(runtime)
    runtime.trajectory_features[0, 0, 0] = 99
    runtime.id_label_to_id[0] = 999
    assert state.trajectory_features[0, 0, 0] == 1
    assert state.id_label_to_id[0] == 10

    restore_motip_state(runtime, state)
    runtime.trajectory_features[0, 0, 0] = 55
    assert state.trajectory_features[0, 0, 0] == 1
    assert list(runtime.id_queue) == [2, 0]


def test_transaction_returns_branch_successor_and_restores_canonical() -> None:
    runtime = _runtime()
    adapter = MOTIPStateAdapter(runtime)
    branch = adapter.capture().fork()

    def update_branch(active_runtime):
        active_runtime.next_id = 20
        active_runtime.id_label_to_id[1] = 19
        active_runtime.trajectory_masks[1, 0] = True
        return "branch-result"

    result, successor = adapter.transact(branch, update_branch)
    assert result == "branch-result"
    assert successor.next_id == 20
    assert successor.id_label_to_id[1] == 19
    assert runtime.next_id == 12
    assert 1 not in runtime.id_label_to_id
    assert not runtime.trajectory_masks[1, 0]

    adapter.commit(successor)
    assert runtime.next_id == 20
    assert runtime.trajectory_masks[1, 0]


def test_transaction_restores_canonical_after_failure() -> None:
    runtime = _runtime()
    adapter = MOTIPStateAdapter(runtime)

    def fail(active_runtime):
        active_runtime.next_id = 99
        raise RuntimeError("decoder failed")

    with pytest.raises(RuntimeError, match="decoder failed"):
        adapter.transact(adapter.capture(), fail)
    assert runtime.next_id == 12


def test_restore_rejects_wrong_vocabulary() -> None:
    source = _runtime()
    state = capture_motip_state(source)
    target = _runtime()
    target.num_id_vocabulary = 4
    with pytest.raises(ValueError, match="vocabulary"):
        restore_motip_state(target, state)
