# BranchMOT research plan

## Question

Can an online tracker reduce post-occlusion identity switches by delaying only
uncertain local associations and using future evidence, without incurring the
cost and latency of offline global optimization?

## Stage 1 — Falsify the core hypothesis

1. Cache per-frame association probabilities from a reproducible baseline.
2. Compare immediate Hungarian assignment against the BranchMOT beam.
3. Evaluate on DanceTrack validation with identical detections.
4. Stratify association accuracy by occlusion length and target similarity.

Delayed episodes require short-term observation-chain alignment across frames;
without this constraint, framewise hypotheses factorize and future evidence
cannot revise an earlier identity preference.

The initial linker uses motion-predicted boxes, IoU/center-distance gating, and
global one-to-one matching. It is a controlled baseline for later mask- or
flow-propagated chains, not a final contribution claim.

The integration contract and cache insertion point are documented in
[`MOTIP_INTEGRATION.md`](MOTIP_INTEGRATION.md).

**Go criterion:** a repeatable AssA/IDSW improvement concentrated in ambiguous
and post-occlusion slices, with mean delay below eight frames.

## Stage 2 — Learn calibrated association uncertainty

- Compare temperature scaling, Brier-trained logits, and evidential outputs.
- Report association ECE, NLL, Brier score, and risk-coverage curves.
- Gate long-term memory writes using calibrated confidence.

## Stage 3 — End-to-end integration

- Integrate with an ID-prediction baseline.
- Replace exhaustive local permutations with sparse conflict-subgraph search.
- Treat rectangular conflict components explicitly as birth/death cases.
- Score newborn assignments and missed tracks inside the same delayed beam.
- Add appearance, motion, mask, and group-relation path scores.

## Required ablations

| Factor | Values |
|---|---|
| Beam size | 1, 2, 4, 8 |
| Maximum delay | 0, 2, 4, 8, 16 frames |
| Branch trigger | entropy, margin, calibrated risk |
| Memory update | always, hard gate, soft gate |
| Path cues | appearance, motion, mask, relations |
| Search scope | global, local conflict subgraph |

## Publication claims to avoid until demonstrated

- Do not claim the first use of multiple hypotheses in tracking.
- Do not call the method real-time until measured end to end.
- Do not attribute gains to uncertainty calibration without reliability tests.
- Do not compare methods using different detections without a separate label.
