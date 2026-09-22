# Training hypothesis-conditioned BranchMOT

## What is trained

BranchMOT starts from a released MOTIP checkpoint. The detector remains frozen
for the first paper iteration. We train three components:

1. a risk head predicting whether immediate ID assignment will be wrong within
   a short future horizon;
2. a branch scorer ranking hypothesis-conditioned trajectory memories;
3. MOTIP's ID decoder and memory updater, lightly fine-tuned so future evidence
   separates correct and incorrect memory branches.

The core inference engine is `HypothesisConditionedAssociator`. Its `decode`
callback runs MOTIP's ID decoder against one private trajectory-memory state;
its `update` callback writes the proposed assignment only to that branch. The
canonical online state is replaced only after a branch is committed.

## Training example construction

Use 12-frame DanceTrack clips by default: four warm-up frames establish
trajectories, one frame opens a conflict, and up to four future frames provide
resolution evidence. Remaining frames provide temporal context.

Mini-batches use a fixed mixture:

- 50% hard conflicts: small top-2 ID margin, competing detections, or a wrong
  immediate assignment;
- 25% occlusion recoveries sampled around GT disappearance/reappearance;
- 25% ordinary clips to preserve normal tracking and calibrate false alarms.

At the conflict frame, enumerate valid local one-to-one assignments and always
insert the GT-continuous assignment during training. Each candidate receives a
copy-on-write memory containing trajectory embeddings, ID labels, boxes,
disappearance counters, and stable-ID mappings. Unroll every retained branch
over the same future detections. Ground truth selects the oracle branch; it is
never an inference input.

## Objective

For branch `k`, let `S_k` be its accumulated learned score over the future
horizon and `k*` the GT-continuous branch. Optimize

```text
L = L_id
  + 0.5 L_rank
  + 0.2 L_risk
  + 0.01 E[number of branches]
  + 0.01 E[decision delay]

L_rank = -log exp(S_k*) / sum_k exp(S_k)
L_risk = (sigmoid(r) - 1[immediate assignment is wrong])²
```

`L_id` is MOTIP's ordinary ID cross-entropy evaluated on the oracle memory
branch and on ordinary clips. The differentiable implementation is
[`integrations/motip/branch_losses.py`](../integrations/motip/branch_losses.py).
Ranking supervision teaches the model which history is physically consistent;
the Brier term makes the branch trigger a calibrated error probability rather
than an arbitrary entropy threshold. Budget terms discourage accuracy gained
only through excessive branches or latency.

## Three-stage schedule

### Stage 1: risk calibration

Freeze MOTIP. Cache logits and train only the risk head for five epochs. Input
features should include top-2 margin, entropy, newborn probability, conflict
size, trajectory age, missed-frame count, box overlap, and optional appearance
similarity. Select the threshold using a risk-coverage curve on validation,
not the test set.

### Stage 2: branch discrimination

Freeze the detector. Train the branch scorer, memory updater, and ID decoder
for 12 epochs with GT-injected candidate sets. Use gradient accumulation because
each clip contains several private memories. Stop gradients through branch
selection itself; gradients flow through the selected branch scores and
unrolled ID decoders.

### Stage 3: budget-aware joint fine-tuning

Fine-tune the new heads and ID decoder together for six epochs. Gradually
increase compute and delay penalties until validation AssA no longer improves
at the allowed latency. Keep the detector frozen unless the branch mechanism
has already shown a repeatable association gain with identical detections.

The starting configuration is
[`configs/dancetrack_branch_training.yaml`](../configs/dancetrack_branch_training.yaml).

## MOTIP state boundary

The adapter must snapshot and restore at least:

- trajectory object features and masks;
- `trajectory_id_labels`;
- trajectory boxes and disappearance counters;
- `id_label_to_id`, free/recycled labels, and the next stable output ID;
- any decoder cache whose contents depend on previous assignments.

Tensor storage should be shared until a branch writes to it; copying the full
video state for every hypothesis would erase the runtime advantage. Only the
small conflict subgraph needs branch-private tensors.

The current `MOTIPStateAdapter` intentionally begins with correctness-first
full tensor clones and transactional restoration. This is the reference
implementation for equivalence tests. Conflict-column copy-on-write is the
subsequent optimization and must reproduce the reference branch scores and
committed state exactly.

`MOTIPBranchDecoder.decode` already supplies the inference engine's `decode`
callback using real upstream trajectory modeling and ID-decoder logits. The
matching `update` callback now resolves the local permutation into fixed
full-frame decisions, applies ID recycling and trajectory updates, and builds
stable-ID results in private state. End-to-end equivalence against unmodified
MOTIP update remains required before real benchmark claims.

## Leakage and evaluation rules

- Split by video sequence before mining clips.
- GT may construct oracle branches and losses only during training.
- Validation thresholds must not be tuned on DanceTrack test.
- Compare methods with identical detections and report detector changes
  separately.
- Report mean and P95 delay, mean/maximum live branches, FPS, memory, AssA,
  HOTA, IDF1, IDSW, and occlusion-length slices.
- Run at least three seeds for learned heads and report variation.
