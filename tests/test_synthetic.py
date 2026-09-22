from pathlib import Path

from branchmot.synthetic import (
    generate_crossing_episode,
    run_synthetic_sweep,
    write_results,
)


def test_episode_is_deterministic_for_seed() -> None:
    import numpy as np

    first = generate_crossing_episode(
        occlusion_length=2, rng=np.random.default_rng(7)
    )
    second = generate_crossing_episode(
        occlusion_length=2, rng=np.random.default_rng(7)
    )
    assert first == second


def test_delayed_commitment_improves_crossing_stress_case() -> None:
    result = run_synthetic_sweep(
        occlusion_lengths=[2], episodes=20, seed=3, max_delays=[4]
    )[0]
    assert result.delayed_accuracy > result.immediate_accuracy
    assert result.mean_delayed_frames == 2.0


def test_short_delay_exposes_the_latency_accuracy_tradeoff() -> None:
    results = run_synthetic_sweep(
        occlusion_lengths=[8], episodes=20, seed=3, max_delays=[2, 16]
    )
    assert results[1].delayed_accuracy > results[0].delayed_accuracy
    assert results[1].mean_delayed_frames > results[0].mean_delayed_frames


def test_results_are_written_as_csv_and_json(tmp_path: Path) -> None:
    results = run_synthetic_sweep(occlusion_lengths=[1], episodes=2)
    csv_path = tmp_path / "result.csv"
    json_path = tmp_path / "result.json"
    write_results(results, csv_path, json_path)
    assert csv_path.read_text(encoding="utf-8").startswith("occlusion_length,")
    assert '"occlusion_length": 1' in json_path.read_text(encoding="utf-8")
