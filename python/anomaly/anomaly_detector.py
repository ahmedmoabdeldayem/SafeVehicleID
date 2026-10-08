"""
Anomaly detection for vehicle tracks.

Three-layer approach:
  Layer 1 — Kinematic rules: velocity, direction, dwell time
  Layer 2 — Statistical: Mahalanobis distance from learned normal distribution
  Layer 3 — Identity: unknown vehicle in a restricted zone

This layered design is intentional. Rules catch clear, well-defined violations
(going too fast). Statistics catch subtle anomalies that are hard to enumerate
as rules. Identity checks enforce access control.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import deque


class AnomalyType(Enum):
    OVERSPEED = "overspeed"
    WRONG_DIRECTION = "wrong_direction"
    STALLED = "stalled"
    SUDDEN_STOP = "sudden_stop"
    STATISTICAL = "statistical_outlier"
    UNKNOWN_VEHICLE = "unknown_vehicle"
    TRAJECTORY_DEVIATION = "trajectory_deviation"


@dataclass
class AnomalyEvent:
    track_id: int
    anomaly_type: AnomalyType
    severity: float             # 0.0 = low, 1.0 = critical
    description: str
    frame_number: int
    position: tuple[float, float]   # (cx, cy) of vehicle center


class KinematicRules:
    """
    Rule-based anomaly detection on per-track kinematic features.

    Parameters
    ----------
    max_speed_pixels_per_frame : float
        Speed limit in pixels/frame. At 30fps and typical parking camera:
        ~5 pixels/frame ≈ 5 km/h. Tune per zone.
    max_dwell_frames : int
        Max frames a vehicle can stay in a moving zone before flagged as stalled.
    sudden_stop_threshold : float
        Velocity drop ratio in one frame to flag as sudden stop.
    """

    def __init__(
        self,
        max_speed_pixels_per_frame: float = 25.0,
        max_dwell_frames: int = 150,  # 5 seconds at 30fps
        sudden_stop_threshold: float = 0.8,
    ):
        self.max_speed = max_speed_pixels_per_frame
        self.max_dwell_frames = max_dwell_frames
        self.sudden_stop_threshold = sudden_stop_threshold

        # Per-track state: {track_id: deque of (vx, vy)}
        self._velocity_history: dict[int, deque] = {}
        self._dwell_counters: dict[int, int] = {}

    def check(
        self,
        track_id: int,
        vx: float,
        vy: float,
        cx: float,
        cy: float,
        frame_number: int,
        allowed_direction: tuple[float, float] | None = None,
    ) -> list[AnomalyEvent]:
        """
        Check a track's current state against kinematic rules.

        Parameters
        ----------
        vx, vy : current velocity in pixels/frame
        cx, cy : current position
        allowed_direction : unit vector (dx, dy) of permitted travel direction.
                           None = no direction constraint.
        """
        events = []
        speed = np.sqrt(vx**2 + vy**2)

        if track_id not in self._velocity_history:
            self._velocity_history[track_id] = deque(maxlen=10)
            self._dwell_counters[track_id] = 0

        prev_velocities = self._velocity_history[track_id]

        # Rule 1: Overspeed
        if speed > self.max_speed:
            events.append(AnomalyEvent(
                track_id=track_id,
                anomaly_type=AnomalyType.OVERSPEED,
                severity=min(1.0, speed / (self.max_speed * 2)),
                description=f"Speed {speed:.1f} px/frame exceeds limit {self.max_speed}",
                frame_number=frame_number,
                position=(cx, cy),
            ))

        # Rule 2: Wrong direction
        if allowed_direction is not None and speed > 2.0:
            allowed_dx, allowed_dy = allowed_direction
            velocity_norm = np.array([vx, vy]) / (speed + 1e-6)
            dot = velocity_norm[0] * allowed_dx + velocity_norm[1] * allowed_dy
            if dot < -0.5:  # moving more than 120° from allowed direction
                events.append(AnomalyEvent(
                    track_id=track_id,
                    anomaly_type=AnomalyType.WRONG_DIRECTION,
                    severity=0.9,
                    description=f"Vehicle moving against allowed direction (dot={dot:.2f})",
                    frame_number=frame_number,
                    position=(cx, cy),
                ))

        # Rule 3: Stalled vehicle
        if speed < 1.0:
            self._dwell_counters[track_id] += 1
        else:
            self._dwell_counters[track_id] = 0

        if self._dwell_counters[track_id] > self.max_dwell_frames:
            events.append(AnomalyEvent(
                track_id=track_id,
                anomaly_type=AnomalyType.STALLED,
                severity=0.7,
                description=f"Vehicle stationary for {self._dwell_counters[track_id]} frames",
                frame_number=frame_number,
                position=(cx, cy),
            ))

        # Rule 4: Sudden stop (velocity drop > threshold in one frame)
        if prev_velocities:
            prev_speed = np.sqrt(prev_velocities[-1][0]**2 + prev_velocities[-1][1]**2)
            if prev_speed > 5.0 and speed < prev_speed * (1 - self.sudden_stop_threshold):
                events.append(AnomalyEvent(
                    track_id=track_id,
                    anomaly_type=AnomalyType.SUDDEN_STOP,
                    severity=0.6,
                    description=f"Speed dropped from {prev_speed:.1f} to {speed:.1f} in one frame",
                    frame_number=frame_number,
                    position=(cx, cy),
                ))

        self._velocity_history[track_id].append((vx, vy))
        return events

    def remove_track(self, track_id: int) -> None:
        self._velocity_history.pop(track_id, None)
        self._dwell_counters.pop(track_id, None)


class StatisticalAnomalyDetector:
    """
    Mahalanobis distance-based anomaly detection.

    Learns the normal joint distribution of [speed, acceleration, path_curvature]
    from training frames. Flags observations where Mahalanobis distance exceeds
    a threshold corresponding to the desired false positive rate.

    Must call fit() with normal trajectory data before detect().
    """

    def __init__(self, threshold_percentile: float = 99.0):
        self.threshold_percentile = threshold_percentile
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._threshold: float | None = None
        self._training_distances: list[float] = []

    def fit(self, normal_feature_vectors: np.ndarray) -> None:
        """
        Fit the normal distribution from training data.

        Parameters
        ----------
        normal_feature_vectors : shape (N, 3)
            Each row: [speed, acceleration, curvature] from normal vehicle behavior.
        """
        self._mean = np.mean(normal_feature_vectors, axis=0)
        cov = np.cov(normal_feature_vectors.T)
        self._inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)

        # Compute distances for all training samples to set the threshold
        distances = [
            self._mahalanobis(v, self._mean, self._inv_cov)
            for v in normal_feature_vectors
        ]
        self._threshold = np.percentile(distances, self.threshold_percentile)

    def detect(
        self,
        track_id: int,
        speed: float,
        acceleration: float,
        curvature: float,
        frame_number: int,
        position: tuple[float, float],
    ) -> AnomalyEvent | None:
        """
        Check if current feature vector is an outlier from the learned normal.

        Returns AnomalyEvent if anomalous, None otherwise.
        """
        if self._mean is None:
            return None  # not fitted yet

        features = np.array([speed, acceleration, curvature])
        d = self._mahalanobis(features, self._mean, self._inv_cov)

        if d > self._threshold:
            severity = min(1.0, (d - self._threshold) / self._threshold)
            return AnomalyEvent(
                track_id=track_id,
                anomaly_type=AnomalyType.STATISTICAL,
                severity=severity,
                description=f"Mahalanobis distance {d:.2f} > threshold {self._threshold:.2f}",
                frame_number=frame_number,
                position=position,
            )
        return None

    @staticmethod
    def _mahalanobis(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
        diff = x - mean
        return float(np.sqrt(diff @ inv_cov @ diff))

    def is_fitted(self) -> bool:
        return self._mean is not None


class AnomalyDetector:
    """
    Unified anomaly detector combining all layers.

    Usage:
        detector = AnomalyDetector()
        # Optionally fit statistical model with normal data
        detector.fit_statistical(normal_features)
        # Each frame:
        events = detector.check_track(track_id, vx, vy, cx, cy, frame_number)
    """

    def __init__(
        self,
        max_speed_pixels_per_frame: float = 25.0,
        max_dwell_frames: int = 150,
        similarity_threshold: float = 0.80,
    ):
        self.rules = KinematicRules(
            max_speed_pixels_per_frame=max_speed_pixels_per_frame,
            max_dwell_frames=max_dwell_frames,
        )
        self.statistical = StatisticalAnomalyDetector()
        self._velocity_windows: dict[int, deque] = {}
        self._prev_velocity: dict[int, tuple[float, float]] = {}

    def fit_statistical(self, normal_feature_vectors: np.ndarray) -> None:
        """Fit the statistical model on normal behavior data."""
        self.statistical.fit(normal_feature_vectors)

    def check_track(
        self,
        track_id: int,
        vx: float,
        vy: float,
        cx: float,
        cy: float,
        frame_number: int,
        vehicle_id: str | None = None,
        identification_confidence: float = 1.0,
        allowed_direction: tuple[float, float] | None = None,
        restricted_zone: bool = False,
    ) -> list[AnomalyEvent]:
        """
        Run all anomaly checks for a single track update.

        Parameters
        ----------
        vehicle_id : None if vehicle could not be identified
        identification_confidence : confidence of identification
        restricted_zone : if True, unidentified vehicles are flagged
        """
        events = []
        speed = np.sqrt(vx**2 + vy**2)

        # Layer 1: kinematic rules
        events.extend(self.rules.check(
            track_id, vx, vy, cx, cy, frame_number, allowed_direction
        ))

        # Layer 2: statistical (if model fitted)
        if self.statistical.is_fitted():
            if track_id not in self._velocity_windows:
                self._velocity_windows[track_id] = deque(maxlen=5)
            win = self._velocity_windows[track_id]
            win.append(speed)

            if len(win) >= 2:
                acceleration = abs(win[-1] - win[-2])
                # Curvature approximation: angular change in heading from previous frame
                curvature = 0.0
                if track_id in self._prev_velocity:
                    prev_vx, prev_vy = self._prev_velocity[track_id]
                    speed_prev = math.sqrt(prev_vx ** 2 + prev_vy ** 2)
                    if speed > 0 and speed_prev > 0:
                        delta = math.atan2(vy, vx) - math.atan2(prev_vy, prev_vx)
                        # Normalize to [-π, π]
                        delta = (delta + math.pi) % (2 * math.pi) - math.pi
                        curvature = abs(delta)

                event = self.statistical.detect(
                    track_id, speed, acceleration, curvature,
                    frame_number, (cx, cy)
                )
                if event:
                    events.append(event)

            self._prev_velocity[track_id] = (vx, vy)

        # Layer 3: identity check in restricted zone
        if restricted_zone and (vehicle_id is None or identification_confidence < 0.70):
            events.append(AnomalyEvent(
                track_id=track_id,
                anomaly_type=AnomalyType.UNKNOWN_VEHICLE,
                severity=0.95,
                description=f"Unidentified vehicle in restricted zone "
                            f"(confidence={identification_confidence:.2f})",
                frame_number=frame_number,
                position=(cx, cy),
            ))

        return events

    def remove_track(self, track_id: int) -> None:
        self.rules.remove_track(track_id)
        self._velocity_windows.pop(track_id, None)
        self._prev_velocity.pop(track_id, None)
