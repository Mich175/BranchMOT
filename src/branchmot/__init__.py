"""BranchMOT research primitives."""

from .association import AssociationConfig, BranchingAssociator, Hypothesis
from .cache import AssociationFrame, read_jsonl, write_jsonl
from .conflicts import ConflictComponent, decompose_conflicts
from .evaluate import CacheEvaluation, evaluate_cache
from .identity_targets import (
    IdentityAssignmentMetrics,
    IdentityTargetStats,
    attach_internal_identity_targets,
    identity_assignment_metrics,
)
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
from .mot_ground_truth import (
    MOTObject,
    OcclusionEvent,
    annotate_cache_with_ground_truth,
    find_occlusion_events,
    read_mot_ground_truth,
)
from .motip_adapter import MOTIPProjection, make_cache_frame, project_motip_scores
from .motip_tap import MOTIPScoreTap, probabilities_from_logits
from .replay import ReplayResult, best_one_to_one, replay_aligned_episode

__all__ = [
    "AssociationConfig",
    "AssociationFrame",
    "BranchingAssociator",
    "CacheEvaluation",
    "ConflictComponent",
    "Hypothesis",
    "IdentityAssignmentMetrics",
    "IdentityTargetStats",
    "LifecycleAssociator",
    "LifecycleHypothesis",
    "MOTIPProjection",
    "MOTIPScoreTap",
    "MOTObject",
    "ObservationChainLinker",
    "OcclusionEvent",
    "ReplayResult",
    "annotate_cache_with_ground_truth",
    "association_calibration",
    "attach_internal_identity_targets",
    "best_one_to_one",
    "box_iou_xyxy",
    "decompose_conflicts",
    "enumerate_lifecycle_assignments",
    "evaluate_cache",
    "find_occlusion_events",
    "identity_assignment_metrics",
    "linear_sum_assignment",
    "link_cache_frames",
    "make_cache_frame",
    "probabilities_from_logits",
    "project_motip_scores",
    "read_jsonl",
    "read_mot_ground_truth",
    "replay_aligned_episode",
    "write_jsonl",
]
