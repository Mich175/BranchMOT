import numpy as np

from branchmot import ConditionalMemoryConfig, HypothesisConditionedAssociator


def _update(memory, observation, assignment):
    memory["history"].append((observation, assignment))
    return memory


def test_branches_own_private_memory_and_future_decode_is_conditional() -> None:
    canonical = {"history": []}
    tracker = HypothesisConditionedAssociator(
        ConditionalMemoryConfig(max_branches=2, max_delay=3, commit_posterior=0.8)
    )

    def decode(memory, observation):
        if observation == "ambiguous":
            return np.array([[0.49, 0.51], [0.51, 0.49]])
        previous_assignment = memory["history"][-1][1]
        if previous_assignment == (0, 1):
            return np.array([[0.95, 0.05], [0.05, 0.95]])
        return np.array([[0.60, 0.40], [0.40, 0.60]])

    assert (
        tracker.step(canonical, "ambiguous", decode=decode, update=_update) is None
    )
    assert canonical == {"history": []}
    assert {branch.assignment for branch in tracker.branches} == {(0, 1), (1, 0)}
    assert tracker.branches[0].memory is not tracker.branches[1].memory

    decision = tracker.step(canonical, "future", decode=decode, update=_update)
    assert decision is not None
    assert decision.assignment == (0, 1)
    assert decision.delayed_frames == 1
    assert decision.posterior > 0.8
    assert len(decision.memory["history"]) == 2
    assert tracker.branches == ()


def test_clear_evidence_commits_without_forking_canonical_memory() -> None:
    canonical = {"history": []}
    tracker = HypothesisConditionedAssociator()

    def decode(_memory, _observation):
        return np.array([[0.99, 0.01], [0.01, 0.99]])

    decision = tracker.step(canonical, 0, decode=decode, update=_update)
    assert decision is not None
    assert decision.assignment == (0, 1)
    assert decision.delayed_frames == 0
    assert canonical["history"] == []
    assert decision.memory["history"] == [(0, (0, 1))]


def test_budget_retains_only_needed_high_probability_branches() -> None:
    tracker = HypothesisConditionedAssociator(
        ConditionalMemoryConfig(
            max_branches=6,
            max_delay=3,
            retained_posterior_mass=0.75,
            entropy_threshold=0.0,
        )
    )

    def decode(_memory, _observation):
        return np.array(
            [[0.70, 0.20, 0.10], [0.20, 0.70, 0.10], [0.10, 0.20, 0.70]]
        )

    assert tracker.step({}, 0, decode=decode, update=lambda m, _o, _a: m) is None
    assert 2 <= len(tracker.branches) < 6
