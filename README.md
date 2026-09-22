# BranchMOT

BranchMOT is a research codebase for **calibrated delayed-commitment identity
reasoning** in online multi-object tracking (MOT).

Instead of committing every detection to a single identity immediately,
BranchMOT keeps a small beam of plausible local association hypotheses when
the evidence is ambiguous. Later observations rescore the beam before the
identity decision is committed, reducing post-occlusion identity switches and
preventing uncertain observations from contaminating long-term track memory.

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

## Planned evaluation

Primary benchmarks: DanceTrack, SportsMOT, MOT17, and MOT20.

Primary metrics: HOTA, AssA, IDF1, identity switches, post-occlusion recovery,
association ECE/Brier score, decision latency, and runtime.

See [`docs/research_plan.md`](docs/research_plan.md) for the staged research
plan and ablation matrix.

## Status

Early research prototype. APIs and experiment formats may change.
