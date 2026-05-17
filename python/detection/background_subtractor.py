"""
Background subtraction-based vehicle detector.

Best suited for static infrastructure cameras (parking lots, logistics yards)
where the background is constant and vehicles are foreground objects.

Uses MOG2 (Mixture of Gaussians v2) which:
- Models each pixel as a mixture of Gaussians
- Adapts to slow illumination changes (shadows, day/night transitions)
- Handles multiple background modes (e.g., waving foliage, water reflections)
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    bbox: np.ndarray    # [x1, y1, x2, y2]
    confidence: float   # 1.0 for background subtraction (no score)
    area: float


class BackgroundSubtractorDetector:
    """
    Detects vehicles using MOG2 background subtraction.

    Pipeline:
      frame → grayscale → MOG2 → threshold → morphological cleanup
            → contour extraction → size filter → bounding boxes

    Parameters
    ----------
    history : int
        Number of frames used to build the background model.
    var_threshold : float
        Mahalanobis distance threshold for foreground decision.
        Higher = fewer pixels marked as foreground.
    min_area : int
        Minimum contour area in pixels to count as a vehicle detection.
        Filters out noise, pedestrians, small debris.
    max_area : int
        Maximum area — prevents entire frame being flagged as foreground
        during lighting changes.
    learning_rate : float
        Background model update rate (0 = frozen, 1 = full update each frame).
        -1 = automatic selection by OpenCV.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        min_area: int = 3000,
        max_area: int = 200000,
        learning_rate: float = -1,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.learning_rate = learning_rate

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,   # shadows detected as gray (127), not white (255)
        )

        # Morphological kernels
        # Closing: fills small holes inside vehicle blobs
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        # Opening: removes small noise blobs
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect vehicles in a single frame.

        Parameters
        ----------
        frame : BGR image (H, W, 3)

        Returns
        -------
        List of Detection objects sorted by area (largest first).
        """
        fg_mask = self._get_foreground_mask(frame)
        contours = self._extract_contours(fg_mask)
        detections = self._contours_to_detections(contours, frame.shape)
        return sorted(detections, key=lambda d: d.area, reverse=True)

    def _get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Apply MOG2 and clean up the foreground mask."""
        # Convert to grayscale — reduces noise, faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Slight blur to reduce pixel-level noise before subtraction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply background subtractor
        # Returns: 255 = foreground, 127 = shadow, 0 = background
        raw_mask = self._subtractor.apply(blurred, learningRate=self.learning_rate)

        # Remove shadows (127) — only keep full foreground (255)
        _, binary = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        # 1. Opening removes small isolated noise
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._open_kernel)
        # 2. Closing fills holes in vehicle blobs (windows, roof features)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self._close_kernel)

        return cleaned

    def _extract_contours(self, mask: np.ndarray) -> list:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _contours_to_detections(
        self, contours: list, frame_shape: tuple
    ) -> list[Detection]:
        h, w = frame_shape[:2]
        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            # Clamp to frame boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)

            # Basic aspect ratio filter — vehicles are wider than tall or square-ish
            aspect = bw / (bh + 1e-6)
            if aspect < 0.3 or aspect > 4.0:
                continue

            detections.append(Detection(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                confidence=1.0,
                area=float(area),
            ))

        return detections

    def freeze_background(self) -> None:
        """Stop updating the background model (e.g., during an event)."""
        self.learning_rate = 0.0

    def resume_learning(self) -> None:
        """Resume automatic background model updates."""
        self.learning_rate = -1

    def visualize(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw detection bounding boxes on frame for debugging."""
        viz = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                viz, f"area:{int(det.area)}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        return viz
