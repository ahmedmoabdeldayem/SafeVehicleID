"""
Kalman filter for 2D vehicle tracking.

State vector: [cx, cy, vx, vy, w, h]
              center(x,y), velocity(x,y), width, height

Measurement vector: [cx, cy, w, h]
                    (what the detector gives us — no velocity)

Motion model: constant velocity
  cx_new = cx + vx * dt
  cy_new = cy + vy * dt
  vx_new = vx  (constant)
  vy_new = vy  (constant)
"""

import numpy as np


class KalmanFilter:
    """
    6-state Kalman filter for bounding box tracking.

    Parameters
    ----------
    process_noise : float
        Q matrix scale — how much the vehicle can deviate from constant velocity.
        Higher value = trusts measurements more, reacts faster but noisier.
    measurement_noise : float
        R matrix scale — confidence in detector output.
        Higher value = trusts motion model more, smoother but slower to react.
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.n_state = 6       # [cx, cy, vx, vy, w, h]
        self.n_measure = 4     # [cx, cy, w, h]

        # State transition matrix F: x_new = F * x
        # Assumes dt=1 (one frame), constant velocity model
        self.F = np.eye(self.n_state, dtype=np.float64)
        self.F[0, 2] = 1.0  # cx += vx
        self.F[1, 3] = 1.0  # cy += vy

        # Measurement matrix H: z = H * x
        # We observe cx, cy, w, h — not velocities
        self.H = np.zeros((self.n_measure, self.n_state), dtype=np.float64)
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy
        self.H[2, 4] = 1.0  # w
        self.H[3, 5] = 1.0  # h

        # Process noise covariance Q
        self.Q = np.eye(self.n_state, dtype=np.float64) * process_noise
        # Velocity uncertainty is higher — vehicles can accelerate/brake
        self.Q[2, 2] *= 10.0
        self.Q[3, 3] *= 10.0

        # Measurement noise covariance R
        self.R = np.eye(self.n_measure, dtype=np.float64) * measurement_noise

        # State estimate and covariance — initialized on first measurement
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def initialize(self, bbox: np.ndarray) -> None:
        """
        Initialize state from first detection.

        Parameters
        ----------
        bbox : array [x1, y1, x2, y2]
        """
        cx, cy, w, h = self._bbox_to_center(bbox)
        self.x = np.array([cx, cy, 0.0, 0.0, w, h], dtype=np.float64)
        # High initial uncertainty on velocity
        self.P = np.eye(self.n_state, dtype=np.float64)
        self.P[2, 2] = 100.0
        self.P[3, 3] = 100.0

    def predict(self) -> np.ndarray:
        """
        Advance state estimate one step forward using motion model.

        Returns predicted bounding box [x1, y1, x2, y2].
        """
        if self.x is None:
            raise RuntimeError("Kalman filter not initialized — call initialize() first")

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self._center_to_bbox(self.x)

    def update(self, bbox: np.ndarray) -> np.ndarray:
        """
        Correct state estimate with a new detection measurement.

        Parameters
        ----------
        bbox : array [x1, y1, x2, y2] from detector

        Returns corrected bounding box [x1, y1, x2, y2].
        """
        cx, cy, w, h = self._bbox_to_center(bbox)
        z = np.array([cx, cy, w, h], dtype=np.float64)

        # Innovation: difference between measurement and prediction
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: how much to trust the measurement vs the model
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self._center_to_bbox(self.x)

    def get_state_bbox(self) -> np.ndarray:
        """Return current state as bounding box [x1, y1, x2, y2]."""
        return self._center_to_bbox(self.x)

    def get_velocity(self) -> tuple[float, float]:
        """Return current velocity estimate (vx, vy) in pixels/frame."""
        return float(self.x[2]), float(self.x[3])

    @staticmethod
    def _bbox_to_center(bbox: np.ndarray) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return cx, cy, w, h

    @staticmethod
    def _center_to_bbox(x: np.ndarray) -> np.ndarray:
        cx, cy, _, _, w, h = x
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float64)
