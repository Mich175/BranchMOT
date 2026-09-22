"""Short-term observation-chain linking for delayed identity reasoning."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .cache import AssociationFrame


def box_iou_xyxy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pairwise IoU for two arrays of ``[x1, y1, x2, y2]`` boxes."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1:] != (4,) or b.shape[1:] != (4,):
        raise ValueError("boxes must have shape [N, 4]")
    if np.any(a[:, 2:] < a[:, :2]) or np.any(b[:, 2:] < b[:, :2]):
        raise ValueError("box maxima must not be smaller than minima")
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular Hungarian assignment with no SciPy dependency."""

    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost matrix must be 2-D")
    if np.any(~np.isfinite(cost)):
        raise ValueError("cost matrix must be finite")
    if 0 in cost.shape:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n_rows, n_columns = cost.shape
    row_potential = np.zeros(n_rows + 1)
    column_potential = np.zeros(n_columns + 1)
    column_to_row = np.zeros(n_columns + 1, dtype=np.int64)
    predecessor = np.zeros(n_columns + 1, dtype=np.int64)

    for row in range(1, n_rows + 1):
        column_to_row[0] = row
        minimum = np.full(n_columns + 1, np.inf)
        used = np.zeros(n_columns + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = column_to_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, n_columns + 1):
                if used[candidate]:
                    continue
                reduced = (
                    cost[active_row - 1, candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(n_columns + 1):
                if used[candidate]:
                    row_potential[column_to_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if column_to_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            column_to_row[column] = column_to_row[previous]
            column = previous
            if column == 0:
                break

    row_indices = []
    column_indices = []
    for column in range(1, n_columns + 1):
        if column_to_row[column] != 0:
            row_indices.append(column_to_row[column] - 1)
            column_indices.append(column - 1)
    rows = np.asarray(row_indices, dtype=np.int64)
    columns = np.asarray(column_indices, dtype=np.int64)
    order = np.argsort(rows)
    rows, columns = rows[order], columns[order]
    return (columns, rows) if transposed else (rows, columns)


@dataclass
class _ChainState:
    chain_id: int
    box: np.ndarray
    velocity: np.ndarray
    last_frame: int


class ObservationChainLinker:
    """Link detections across a short window using motion-predicted boxes."""

    def __init__(
        self,
        *,
        min_iou: float = 0.1,
        max_normalized_center_distance: float = 2.0,
        iou_weight: float = 0.7,
        unmatched_cost: float = 0.8,
        max_age: int = 2,
        velocity_momentum: float = 0.5,
    ) -> None:
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou must be in [0, 1]")
        if not 0.0 <= iou_weight <= 1.0:
            raise ValueError("iou_weight must be in [0, 1]")
        if max_normalized_center_distance <= 0 or unmatched_cost < 0 or max_age < 0:
            raise ValueError("distance, unmatched cost, and age must be valid")
        if not 0.0 <= velocity_momentum <= 1.0:
            raise ValueError("velocity_momentum must be in [0, 1]")
        self.min_iou = min_iou
        self.max_distance = max_normalized_center_distance
        self.iou_weight = iou_weight
        self.unmatched_cost = unmatched_cost
        self.max_age = max_age
        self.velocity_momentum = velocity_momentum
        self._chains: dict[int, _ChainState] = {}
        self._next_chain_id = 0
        self._last_frame = -1

    def update(self, frame_index: int, boxes_xyxy: np.ndarray) -> tuple[int, ...]:
        boxes = np.asarray(boxes_xyxy, dtype=np.float64)
        box_iou_xyxy(boxes, boxes)
        if frame_index <= self._last_frame:
            raise ValueError("frame indices must be strictly increasing")
        self._last_frame = frame_index
        self._chains = {
            key: chain
            for key, chain in self._chains.items()
            if frame_index - chain.last_frame <= self.max_age + 1
        }
        active = sorted(self._chains.values(), key=lambda chain: chain.chain_id)
        if not active:
            return self._start_new_chains(frame_index, boxes)

        predicted = np.stack(
            [
                chain.box + chain.velocity * (frame_index - chain.last_frame)
                for chain in active
            ]
        )
        iou = box_iou_xyxy(boxes, predicted)
        centers = (boxes[:, None, :2] + boxes[:, None, 2:]) / 2.0
        predicted_centers = (predicted[None, :, :2] + predicted[None, :, 2:]) / 2.0
        center_distance = np.linalg.norm(centers - predicted_centers, axis=2)
        diagonals = np.linalg.norm(predicted[:, 2:] - predicted[:, :2], axis=1)
        normalized_distance = center_distance / np.maximum(diagonals[None, :], 1e-6)
        allowed = (iou >= self.min_iou) | (normalized_distance <= self.max_distance)
        similarity = self.iou_weight * iou + (1.0 - self.iou_weight) * np.exp(
            -normalized_distance
        )

        n_detections = len(boxes)
        real_cost = np.where(allowed, 1.0 - similarity, 1e6)
        dummy_cost = np.full((n_detections, n_detections), self.unmatched_cost)
        rows, columns = linear_sum_assignment(np.concatenate([real_cost, dummy_cost], axis=1))
        chain_ids = [-1] * n_detections
        for detection, column in zip(rows.tolist(), columns.tolist()):
            if column >= len(active) or not allowed[detection, column]:
                continue
            chain = active[column]
            delta = max(frame_index - chain.last_frame, 1)
            observed_velocity = (boxes[detection] - chain.box) / delta
            chain.velocity = (
                self.velocity_momentum * chain.velocity
                + (1.0 - self.velocity_momentum) * observed_velocity
            )
            chain.box = boxes[detection].copy()
            chain.last_frame = frame_index
            chain_ids[detection] = chain.chain_id

        for detection, chain_id in enumerate(chain_ids):
            if chain_id < 0:
                chain_ids[detection] = self._start_chain(frame_index, boxes[detection])
        return tuple(chain_ids)

    def _start_new_chains(self, frame_index: int, boxes: np.ndarray) -> tuple[int, ...]:
        return tuple(self._start_chain(frame_index, box) for box in boxes)

    def _start_chain(self, frame_index: int, box: np.ndarray) -> int:
        chain_id = self._next_chain_id
        self._next_chain_id += 1
        self._chains[chain_id] = _ChainState(
            chain_id, box.copy(), np.zeros(4, dtype=np.float64), frame_index
        )
        return chain_id


def link_cache_frames(
    frames: list[AssociationFrame],
    linker: ObservationChainLinker | None = None,
) -> tuple[AssociationFrame, ...]:
    """Replace per-frame detection indices with stable observation-chain IDs."""

    chain_linker = linker or ObservationChainLinker()
    linked: list[AssociationFrame] = []
    for frame in frames:
        frame.validate()
        if frame.boxes_xyxy is None:
            raise ValueError("cached boxes are required for observation linking")
        chain_ids = chain_linker.update(
            frame.frame_index, np.asarray(frame.boxes_xyxy, dtype=np.float64)
        )
        linked.append(replace(frame, detection_ids=list(chain_ids)))
    return tuple(linked)
