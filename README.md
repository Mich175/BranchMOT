# BranchMOT

BranchMOT is a research codebase for **calibrated delayed-commitment identity
reasoning** in online multi-object tracking (MOT).

Instead of committing every detection to a single identity immediately,
BranchMOT keeps a small beam of plausible local association hypotheses when
the evidence is ambiguous. Later observations rescore the beam before the
identity decision is committed, reducing post-occlusion identity switches and
preventing uncertain observations from contaminating long-term track memory.

The publication-oriented path goes further than score post-processing:
`HypothesisConditionedAssociator` forks the ID-prompt/trajectory memory for
each local hypothesis, decodes future evidence under each private history, and
atomically commits only the winning memory. See [`docs/TRAINING.md`](docs/TRAINING.md)
for the staged MOTIP fine-tuning recipe.

`MOTIPStateAdapter` provides the correctness-first runtime boundary: complete
ID recycling and trajectory state is restored transactionally for each branch,
and decoder failures cannot leak partial state into the online tracker.

## Research hypothesis

Modern online trackers often make an irreversible identity decision from one
ambiguous frame. The central hypothesis of this project is that a bounded,
uncertainty-triggered delay can improve association accuracy while preserving
practical online latency.

## MVP

The initial implementation provides:

- entropy- and margin-based ambiguity detection;
- top-K beam association with one-to-one assignment constraints;
- temporal path rescoring and bounded decision delay;
- causal mapping from stable tracker IDs to dataset identity targets;
- post-filter MOTIP capture of boxes, scores, internal labels, and output IDs;
- deterministic unit tests for crossing and unambiguous tracks;
- a DanceTrack-oriented experiment configuration.

```bash
python -m pip install -e ".[dev]"
pytest
```

Before a real MOTIP/DanceTrack export, validate the external inputs:

```bash
branchmot-preflight \
  --motip-root /path/to/MOTIP \
  --data-root /path/to/datasets \
  --checkpoint /path/to/r50_deformable_detr_motip_dancetrack.pth
```

Run the deterministic crossing/occlusion stress benchmark:

```bash
branchmot-synthetic --occlusion-lengths 1 2 4 8 --max-delays 2 4 8 16 \
  --episodes 200 --seed 42
```

This writes CSV and JSON results under `results/` for regression testing and
later comparison with DanceTrack slices.

Evaluate an exported MOTIP cache against a DanceTrack sequence:

```bash
branchmot-evaluate outputs/branchmot_cache/sequence.jsonl \
  /path/to/DanceTrack/val/sequence/gt/gt.txt \
  --annotated-cache outputs/branchmot_cache/sequence.annotated.jsonl \
  --output results/sequence.identity.json
```

The report separates GT matching coverage from causally evaluable identity
coverage, so the first appearance of a target is never counted using leaked
current-frame identity information.

The checked-in [reference run](benchmarks/reference/synthetic_seed42.csv) is a
mechanism sanity check only; see
[`docs/synthetic_benchmark.md`](docs/synthetic_benchmark.md) for limitations.

## Planned evaluation

Primary benchmarks: DanceTrack, SportsMOT, MOT17, and MOT20.

Primary metrics: HOTA, AssA, IDF1, identity switches, post-occlusion recovery,
association ECE/Brier score, decision latency, and runtime.

See [`docs/research_plan.md`](docs/research_plan.md) for the staged research
plan and ablation matrix.

## Status

Early research prototype. APIs and experiment formats may change.
