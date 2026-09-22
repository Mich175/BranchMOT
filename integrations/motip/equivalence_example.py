"""Fail-closed equivalence gate to run inside a pinned MOTIP evaluation job."""

from collections.abc import Iterable
from typing import Any

from branchmot import assert_one_frame_equivalent


def verify_sequence_prefix(
    runtime_tracker: Any,
    images: Iterable[Any],
    *,
    maximum_frames: int = 50,
) -> None:
    """Require exact official/replayed state equality on a sequence prefix."""

    for frame_index, image in enumerate(images):
        if frame_index >= maximum_frames:
            break
        try:
            assert_one_frame_equivalent(runtime_tracker, image)
        except Exception as error:
            raise RuntimeError(
                f"MOTIP equivalence failed at zero-based frame {frame_index}"
            ) from error
