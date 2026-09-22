# Synthetic crossing benchmark

This benchmark is a deterministic mechanism test, not evidence of real-world
MOT superiority. It creates two observation chains whose ID evidence is
slightly wrong during an occlusion and strongly correct after reappearance.

Reference settings: 200 episodes per cell, seed 42, beam size 2, ambiguous
correct-ID logit mean -0.2, recovery logit mean 3.0, logit noise 0.15.

| Occlusion | Max delay | Immediate acc. | Delayed acc. | Gain |
|---:|---:|---:|---:|---:|
| 4 | 2 | 0.286 | 0.412 | +0.126 |
| 4 | 4 | 0.273 | 1.000 | +0.727 |
| 8 | 2 | 0.204 | 0.338 | +0.134 |
| 8 | 4 | 0.191 | 0.444 | +0.253 |
| 8 | 8 | 0.189 | 1.000 | +0.811 |

The intended sanity-check result is visible: when the delay budget is shorter
than the ambiguous interval, performance is bounded; when it covers the
interval, post-occlusion evidence can resolve the persistent hypothesis.

The complete table is stored in
[`benchmarks/reference/synthetic_seed42.csv`](../benchmarks/reference/synthetic_seed42.csv).
All paper claims must be based on real benchmark caches and include chain-link
errors, detector errors, runtime, and calibration.
