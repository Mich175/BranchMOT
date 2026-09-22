"""BranchMOT research primitives."""

from .association import AssociationConfig, BranchingAssociator, Hypothesis
from .cache import AssociationFrame, read_jsonl, write_jsonl
from .conflicts import ConflictComponent, decompose_conflicts
from .lifecycle import (
    LifecycleAssociator,
    LifecycleHypothesis,
    enumerate_lifecycle_assignments,
)
from .linking import (
    ObservationChainLinker,
    box_iou_xyxy,
    linear_sum_assignment,
    link_cache_frames,
)
from .metrics import association_calibration
from .motip_adapter import MOTIPProjection, make_cache_frame, project_motip_scores
from .motip_tap import MOTIPScoreTap, probabilities_from_logits
from .preflight import PreflightReport, check_experiment
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
    "MOTIPScoreTap",
    "ObservationChainLinker",
    "PreflightReport",
    "ReplayResult",
    "association_calibration",
    "best_one_to_one",
    "box_iou_xyxy",
    "check_experiment",
    "decompose_conflicts",
    "enumerate_lifecycle_assignments",
    "linear_sum_assignment",
    "link_cache_frames",
    "make_cache_frame",
    "probabilities_from_logits",
    "project_motip_scores",
    "read_jsonl",
    "replay_aligned_episode",
    "write_jsonl",
]
