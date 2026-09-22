# MOTIP integration

BranchMOT integrates at MOTIP's probability-to-assignment boundary. No upstream
source is vendored into this repository.

## Verified upstream contract

At upstream `MCG-NJU/MOTIP` main commit
`ffc0e905ac196a603027eca8d18fb0dff48c8bcc`:

1. `models/motip/id_decoder.py` returns `id_logits` over the fixed ID
   vocabulary plus one empty/newborn token.
2. `models/runtime_tracker.py::_get_id_pred_labels` converts those logits with
   softmax or sigmoid into `id_scores`.
3. The runtime filters low-confidence newborns, allocates reusable internal ID
   labels, maps those labels to stable output IDs, and updates trajectories.

The score hook runs after step 2, while the cache row is finalized only after
step 3. This preserves the exact subset and ordering that survives MOTIP's
runtime filters. The exporter records:

- `id_scores.detach().float().cpu().numpy()`;
- active ID labels from the current trajectory state;
- the final vocabulary index as `newborn_label`;
- stable per-frame detection indices;
- detector confidence and selected boxes;
- final internal labels and stable output IDs;
- ground-truth IDs only during evaluation.

`branchmot.motip_adapter.make_cache_frame` projects the full vocabulary onto
the active tracks and the newborn class. Records are written with
`branchmot.cache.write_jsonl`.

## Non-invasive exporter

`branchmot.MOTIPScoreTap` registers a forward hook on MOTIP's `id_decoder` and
captures its logits without modifying the tracker. A minimal runner is in
[`integrations/motip/export_example.py`](../integrations/motip/export_example.py).

```python
with MOTIPScoreTap(runtime_tracker) as tap:
    for frame_index, image in frames:
        tap.set_frame_context(frame_index)
        runtime_tracker.update(image=image)
    tap.write("outputs/branchmot_cache/sequence.jsonl")
```

Default per-frame row indices are suitable for score inspection only. Delayed
replay requires `detection_ids` to represent stable short-term observation
chains produced by mask propagation, optical flow, or another linker.

The tap wraps MOTIP's existing `_get_activate_detections`,
`_get_id_pred_labels`, and `_assign_newborn_id_labels` calls. It stores the
exact activated detections, then aligns them to any rows removed by runtime
thresholding. All original methods are restored when the tap closes.
`branchmot.link_cache_frames` converts captured boxes into short-term chain IDs
using the motion/IoU linker.

## Branch-private runtime state

At the pinned upstream commit, an assignment can mutate more than trajectory
features. `MOTIPStateAdapter` therefore snapshots and restores the complete
assignment-dependent boundary:

- `next_id`, `id_label_to_id`, and the recency-ordered `id_queue`;
- `trajectory_features`, `trajectory_boxes`, `trajectory_id_labels`;
- `trajectory_times`, `trajectory_masks`, and `current_track_results`.

`adapter.transact(branch_state, operation)` temporarily activates one branch,
runs a decoder/update operation, captures its successor, and restores the
canonical runtime in a `finally` block. `adapter.commit(winner)` is the only
operation that replaces canonical state. Tensor storage is cloned on both
capture and restore, including nested tensors in current results, preventing
in-place writes in upstream `_update_trajectory_infos` from crossing branch
boundaries.

The verified source contract is commit
`ffc0e905ac196a603027eca8d18fb0dff48c8bcc`. The adapter fails closed when
required fields or the ID-vocabulary size differ. A later performance pass can
replace full safe clones with conflict-column copy-on-write storage without
changing the inference API.

`MOTIPBranchDecoder` is the score-side bridge. Given activated boxes/features,
local detection rows, and candidate internal ID labels, it transactionally
activates each branch state and calls upstream `_get_id_pred_labels`. A
temporary decoder hook captures the branch-conditioned logits, projects them
onto the local candidate labels, normalizes the local mass, removes the hook,
and restores canonical state. Candidate labels must be active in that branch;
shape, vocabulary, and decoder-call mismatches fail closed.

`MOTIPBranchDecoder.update` supplies the matching inference-engine `update`
callback. It combines the local permutation with fixed decisions for every
non-conflict detection, checks one-to-one IDs and vocabulary capacity, runs
MOTIP newborn allocation, updates the recency queue and trajectory tensors,
filters inactive tracks, and constructs stable-ID results inside the private
transaction. The returned successor can be retained as a branch or committed
atomically through `MOTIPStateAdapter`.

`MOTIPConditionalTracker` closes the loop: one `step(observation)` call runs
the hypothesis-conditioned associator with the decoder/update callbacks. While
the decision is unresolved, canonical MOTIP state remains byte-for-byte
logically unchanged; once the posterior or latency rule fires, the winning
successor is committed atomically. `reset()` discards pending branches without
touching canonical state.

## Causal identity targets

MOTIP's vocabulary labels are recyclable and therefore cannot be compared
directly with DanceTrack identity numbers. Call
`attach_internal_identity_targets` on GT-annotated cache frames. It learns the
mapping from each stable MOTIP output ID to GT identity only after scoring the
current frame. Thus a frame's target uses prior evidence and never its own
answer. `identity_assignment_metrics` then measures immediate assignments on
the causally evaluable subset.

The cache schema is version 2. Readers remain compatible with version 1;
new identity fields are optional so synthetic and adapter-only data remain
valid.

## Why JSONL first

The MVP format is deliberately transparent and streamable. It makes schema
errors visible before large experiments and avoids pickle-based loading. Once
the interface is stable, a columnar or compressed binary backend can be added
without changing `AssociationFrame`.

## Licensing boundary

MOTIP is Apache-2.0 licensed. BranchMOT currently links through an adapter and
does not copy or modify upstream source. Any future maintained fork must retain
MOTIP's copyright and Apache-2.0 notices.
