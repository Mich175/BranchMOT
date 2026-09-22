"""Minimal integration example to be copied into a MOTIP evaluation runner.

This file is illustrative and is not imported by the BranchMOT package because
MOTIP owns dataset construction and frame loading.
"""

from collections.abc import Iterable
from pathlib import Path

from branchmot import MOTIPScoreTap


def export_sequence(runtime_tracker, indexed_images: Iterable[tuple[int, object]]) -> None:
    """Run the unmodified MOTIP tracker while recording ID probabilities."""

    with MOTIPScoreTap(runtime_tracker) as tap:
        for frame_index, image in indexed_images:
            # Replace these row IDs with mask/flow-linked observation-chain IDs
            # before using delayed replay across frames.
            tap.set_frame_context(frame_index)
            runtime_tracker.update(image=image)
        tap.write(Path("outputs/branchmot_cache/sequence.jsonl"))
