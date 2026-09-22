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
3. The runtime immediately dispatches `id_scores` to a hard assignment
   protocol.

The exporter should be called immediately after step 2. It must pass:

- `id_scores.detach().float().cpu().numpy()`;
- active ID labels from the current trajectory state;
- the final vocabulary index as `newborn_label`;
- stable per-frame detection indices;
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

## Why JSONL first

The MVP format is deliberately transparent and streamable. It makes schema
errors visible before large experiments and avoids pickle-based loading. Once
the interface is stable, a columnar or compressed binary backend can be added
without changing `AssociationFrame`.

## Licensing boundary

MOTIP is Apache-2.0 licensed. BranchMOT currently links through an adapter and
does not copy or modify upstream source. Any future maintained fork must retain
MOTIP's copyright and Apache-2.0 notices.
