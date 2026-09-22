"""BranchMOT research primitives."""

from .association import AssociationConfig, BranchingAssociator, Hypothesis
from .cache import AssociationFrame, read_jsonl, write_jsonl
from .conflicts import ConflictComponent, decompose_conflicts
from .lifecycle import (
    LifecycleAssociator,
    LifecycleHypothesis,
    enumerate_lifecycle_assignments,
)
from .metrics import association_calibration
from .motip_adapter import MOTIPProjection, make_cache_frame, project_motip_scores
from .replay import ReplayResult, best_one_to_one, replay_aligned_episode

__all__ = [
    "AssociationConfig",
    "AssociationFrame",
    "BranchingAssociator",
    "ConflictComponent",
    "Hypothesis",
    "LifecycleAssociator",
    "LifecycleHypothesis",
    "MOTIPProjection",
    "ReplayResult",
    "association_calibration",
    "best_one_to_one",
    "decompose_conflicts",
    "enumerate_lifecycle_assignments",
    "make_cache_frame",
    "project_motip_scores",
    "read_jsonl",
    "replay_aligned_episode",
    "write_jsonl",
]
