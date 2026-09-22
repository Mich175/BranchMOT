"""BranchMOT research primitives."""

from .association import AssociationConfig, BranchingAssociator, Hypothesis
from .cache import AssociationFrame, read_jsonl, write_jsonl
from .metrics import association_calibration
from .motip_adapter import MOTIPProjection, make_cache_frame, project_motip_scores

__all__ = [
    "AssociationConfig",
    "AssociationFrame",
    "BranchingAssociator",
    "Hypothesis",
    "MOTIPProjection",
    "association_calibration",
    "make_cache_frame",
    "project_motip_scores",
    "read_jsonl",
    "write_jsonl",
]
