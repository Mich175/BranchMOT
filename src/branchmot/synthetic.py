"""Deterministic stress benchmark for delayed identity commitment."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .association import AssociationConfig
from .cache import AssociationFrame
from .replay import replay_aligned_episode


@dataclass(frozen=True)
class SyntheticResult:
    occlusion_length: int
    max_delay: int
    episodes: int
    immediate_accuracy: float
    delayed_accuracy: float
    accuracy_gain: float
    mean_delayed_frames: float


def generate_crossing_episode(
    *,
    occlusion_length: int,
    rng: np.random.Generator,
    wrong_logit_bias: float = 0.2,
    recovery_logit: float = 3.0,
    logit_noise: float = 0.15,
) -> list[AssociationFrame]:
    """Generate two stable chains that become confusing during a crossing."""

    if occlusion_length < 1:
        raise ValueError("occlusion_length must be positive")
    if recovery_logit <= 0 or wrong_logit_bias < 0 or logit_noise < 0:
        raise ValueError("synthetic evidence parameters must be non-negative")

    frames: list[AssociationFrame] = []
    total_frames = occlusion_length + 1
    for frame_index in range(total_frames):
        if frame_index < occlusion_length:
            correct_logit = rng.normal(-wrong_logit_bias, logit_noise)
        else:
            correct_logit = rng.normal(recovery_logit, logit_noise)
        correct_probability = float(1.0 / (1.0 + np.exp(-correct_logit)))
        probabilities = [
            [correct_probability, 1.0 - correct_probability],
            [1.0 - correct_probability, correct_probability],
        ]
        progress = frame_index / max(total_frames - 1, 1)
        left = 0.7 * progress
        right = 0.7 * (1.0 - progress)
        frames.append(
            AssociationFrame(
                frame_index=frame_index,
                detection_ids=[100, 101],
                track_ids=[0, 1],
                probabilities=probabilities,
                ground_truth_track_ids=[0, 1],
                boxes_xyxy=[
                    [left, 0.2, left + 0.2, 0.8],
                    [right, 0.2, right + 0.2, 0.8],
                ],
            )
        )
    return frames


def run_synthetic_sweep(
    *,
    occlusion_lengths: list[int],
    max_delays: list[int] | None = None,
    episodes: int = 200,
    seed: int = 42,
    beam_size: int = 2,
    wrong_logit_bias: float = 0.2,
    recovery_logit: float = 3.0,
    logit_noise: float = 0.15,
) -> list[SyntheticResult]:
    """Run reproducible Monte Carlo episodes over several occlusion lengths."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    delays = max_delays or [16]
    if any(delay < 0 for delay in delays):
        raise ValueError("max delays must be non-negative")
    settings = [
        (occlusion_length, max_delay)
        for occlusion_length in occlusion_lengths
        for max_delay in delays
    ]
    seed_sequence = np.random.SeedSequence(seed)
    setting_seeds = seed_sequence.spawn(len(settings))
    results: list[SyntheticResult] = []
    for (occlusion_length, max_delay), setting_seed in zip(settings, setting_seeds):
        rng = np.random.default_rng(setting_seed)
        immediate_correct = delayed_correct = total = delayed_frames = 0
        for _ in range(episodes):
            frames = generate_crossing_episode(
                occlusion_length=occlusion_length,
                rng=rng,
                wrong_logit_bias=wrong_logit_bias,
                recovery_logit=recovery_logit,
                logit_noise=logit_noise,
            )
            replay = replay_aligned_episode(
                frames,
                AssociationConfig(
                    beam_size=beam_size,
                    max_delay=max_delay,
                    entropy_threshold=0.5,
                    margin_threshold=0.2,
                ),
            )
            immediate_correct += replay.immediate_correct
            delayed_correct += replay.delayed_correct
            total += replay.total
            delayed_frames += replay.delayed_frames
        immediate_accuracy = immediate_correct / total
        delayed_accuracy = delayed_correct / total
        results.append(
            SyntheticResult(
                occlusion_length=occlusion_length,
                max_delay=max_delay,
                episodes=episodes,
                immediate_accuracy=immediate_accuracy,
                delayed_accuracy=delayed_accuracy,
                accuracy_gain=delayed_accuracy - immediate_accuracy,
                mean_delayed_frames=delayed_frames / episodes,
            )
        )
    return results


def write_results(
    results: list[SyntheticResult], csv_path: str | Path, json_path: str | Path
) -> None:
    """Write machine-readable benchmark results with stable column ordering."""

    rows = [asdict(result) for result in results]
    csv_destination = Path(csv_path)
    json_destination = Path(json_path)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    with csv_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SyntheticResult.__annotations__))
        writer.writeheader()
        writer.writerows(rows)
    with json_destination.open("w", encoding="utf-8") as handle:
        json.dump({"results": rows}, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occlusion-lengths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-delays", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    results = run_synthetic_sweep(
        occlusion_lengths=args.occlusion_lengths,
        max_delays=args.max_delays,
        episodes=args.episodes,
        seed=args.seed,
        beam_size=args.beam_size,
    )
    destination = Path(args.output_dir)
    write_results(
        results,
        destination / "synthetic_benchmark.csv",
        destination / "synthetic_benchmark.json",
    )
    for result in results:
        print(
            f"L={result.occlusion_length:2d} "
            f"D={result.max_delay:2d} "
            f"immediate={result.immediate_accuracy:.3f} "
            f"delayed={result.delayed_accuracy:.3f} "
            f"gain={result.accuracy_gain:+.3f} "
            f"delay={result.mean_delayed_frames:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
