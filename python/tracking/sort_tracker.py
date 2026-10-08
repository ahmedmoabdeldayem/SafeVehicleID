"""
SORT: Simple Online and Real-time Tracking

Algorithm:
  1. Predict all existing tracks forward using Kalman filter
  2. Compute IoU cost matrix between predictions and new detections
  3. Run Hungarian algorithm to find minimum-cost assignment
  4. Update matched tracks, create new ones for unmatched detections
  5. Delete tracks that have been unmatched for too many consecutive frames

Reference: Bewley et al., "Simple Online and Realtime Tracking" (2016)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter


class TrackState(Enum):
    TENTATIVE = "tentative"   # newly created, not yet confirmed
    CONFIRMED = "confirmed"   # seen enough consecutive times
    DELETED = "deleted"       # marked for removal


@dataclass
class Track:
    """
    A single tracked vehicle.

    Attributes
    ----------
    track_id : int
        Globally unique ID assigned when track is created.
    state : TrackState
        Lifecycle state of this track.
    hits : int
        Number of frames this track has been matched to a detection.
    consecutive_misses : int
        Frames since last successful match.
    """
    track_id: int
    kalman: KalmanFilter
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    consecutive_misses: int = 0

    # Vehicle identity (assigned by identification layer)
    vehicle_id: str | None = None
    identification_confidence: float = 0.0

    # History for anomaly detection
    bbox_history: list[np.ndarray] = field(default_factory=list)

    def predict(self) -> np.ndarray:
        return self.kalman.predict()

    def update(self, bbox: np.ndarray) -> None:
        self.kalman.update(bbox)
        self.hits += 1
        self.consecutive_misses = 0
        self.bbox_history.append(bbox.copy())
        if len(self.bbox_history) > 100:
            self.bbox_history.pop(0)

    def mark_missed(self) -> None:
        self.consecutive_misses += 1

    def get_velocity(self) -> tuple[float, float]:
        return self.kalman.get_velocity()

    def get_bbox(self) -> np.ndarray:
        return self.kalman.get_state_bbox()


class SORTTracker:
    """
    SORT tracker with configurable confirmation and deletion thresholds.

    Parameters
    ----------
    max_disappeared : int
        Number of consecutive missed frames before a track is deleted.
    min_hits_to_confirm : int
        Minimum consecutive matches before a track is reported externally.
    iou_threshold : float
        Minimum IoU for a detection-track match to be accepted.
    process_noise : float
        Kalman filter process noise (Q scale).
    measurement_noise : float
        Kalman filter measurement noise (R scale).
    """

    def __init__(
        self,
        max_disappeared: int = 15,
        min_hits_to_confirm: int = 3,
        iou_threshold: float = 0.3,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        on_track_deleted: "Callable[[int], None] | None" = None,
    ):
        self.max_disappeared = max_disappeared
        self.min_hits_to_confirm = min_hits_to_confirm
        self.iou_threshold = iou_threshold
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.on_track_deleted = on_track_deleted

        self.tracks: list[Track] = []
        self._next_id: int = 1

    def update(self, detections: np.ndarray) -> list[Track]:
        """
        Process one frame of detections.

        Parameters
        ----------
        detections : ndarray of shape (N, 4) or (N, 5)
            Each row: [x1, y1, x2, y2] or [x1, y1, x2, y2, score]

        Returns
        -------
        List of confirmed active tracks (state == CONFIRMED).
        """
        if len(detections) == 0:
            detections = np.empty((0, 4))
        else:
            detections = np.asarray(detections)[:, :4]

        # Step 1: predict all existing tracks
        predictions = np.array([t.predict() for t in self.tracks]) if self.tracks else np.empty((0, 4))

        # Step 2: match predictions to detections
        matched, unmatched_dets, unmatched_tracks = self._match(predictions, detections)

        # Step 3: update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
            if self.tracks[track_idx].hits >= self.min_hits_to_confirm:
                self.tracks[track_idx].state = TrackState.CONFIRMED

        # Step 4: mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            if self.tracks[track_idx].consecutive_misses > self.max_disappeared:
                self.tracks[track_idx].state = TrackState.DELETED

        # Step 5: create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        # Remove deleted tracks, firing callback for each one
        if self.on_track_deleted is not None:
            for t in self.tracks:
                if t.state == TrackState.DELETED:
                    self.on_track_deleted(t.track_id)
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def _match(
        self,
        predictions: np.ndarray,
        detections: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Match track predictions to detections using IoU + Hungarian algorithm.

        Returns (matched_pairs, unmatched_detection_indices, unmatched_track_indices)
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # IoU cost matrix: rows = tracks, cols = detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
        for t_idx, pred_bbox in enumerate(predictions):
            for d_idx, det_bbox in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(pred_bbox, det_bbox)

        # Hungarian algorithm minimizes cost — invert IoU to get cost
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: list[tuple[int, int]] = []
        unmatched_tracks: list[int] = list(range(len(self.tracks)))
        unmatched_dets: list[int] = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return matched, unmatched_dets, unmatched_tracks

    def _create_track(self, bbox: np.ndarray) -> Track:
        kf = KalmanFilter(
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )
        kf.initialize(bbox)
        track = Track(track_id=self._next_id, kalman=kf)
        track.bbox_history.append(bbox.copy())
        self._next_id += 1
        self.tracks.append(track)
        return track

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union of two bounding boxes.

    Parameters
    ----------
    box1, box2 : array [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)
