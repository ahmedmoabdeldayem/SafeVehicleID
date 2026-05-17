"""
License plate recognition (ANPR) pipeline.

Pipeline stages:
  1. Crop vehicle bounding box from frame
  2. Detect plate region within the crop (contour-based or cascade classifier)
  3. Perspective correction to get a front-facing plate view
  4. OCR to extract plate text
  5. Normalize and validate plate string

For production: replace the contour-based plate locator with a dedicated
YOLO plate detector trained on real parking facility data.
"""

from __future__ import annotations

import re
import cv2
import numpy as np
from dataclasses import dataclass


# Common European plate aspect ratio range
PLATE_ASPECT_MIN = 2.5
PLATE_ASPECT_MAX = 6.0
PLATE_MIN_WIDTH = 60
PLATE_MIN_HEIGHT = 15


@dataclass
class PlateResult:
    text: str                   # Raw OCR text
    normalized: str             # Cleaned plate string (uppercase, spaces removed)
    confidence: float           # OCR confidence [0, 1]
    plate_bbox: np.ndarray      # [x1, y1, x2, y2] within vehicle crop
    plate_image: np.ndarray     # Perspective-corrected plate crop


class LicensePlateRecognizer:
    """
    ANPR pipeline using contour-based plate detection and EasyOCR.

    Parameters
    ----------
    min_confidence : float
        Minimum OCR confidence to accept a result.
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self._reader = None  # lazy init — EasyOCR loads a model on creation

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
                # gpu=False for CPU inference; set True if CUDA available
                self._reader = easyocr.Reader(["en", "de"], gpu=False, verbose=False)
            except ImportError:
                raise ImportError("easyocr required: pip install easyocr")
        return self._reader

    def recognize(self, frame: np.ndarray, vehicle_bbox: np.ndarray) -> PlateResult | None:
        """
        Attempt to recognize the license plate in a vehicle crop.

        Parameters
        ----------
        frame : full BGR frame
        vehicle_bbox : [x1, y1, x2, y2] of vehicle in frame

        Returns
        -------
        PlateResult if a plate is found and OCR confidence is sufficient, else None.
        """
        vehicle_crop = self._crop(frame, vehicle_bbox)
        if vehicle_crop is None:
            return None

        plate_contour, plate_bbox = self._detect_plate_region(vehicle_crop)
        if plate_contour is None:
            return None

        plate_img = self._perspective_correct(vehicle_crop, plate_contour)
        if plate_img is None:
            return None

        result = self._run_ocr(plate_img)
        if result is None:
            return None

        text, confidence = result
        normalized = self._normalize_plate(text)

        if confidence < self.min_confidence or len(normalized) < 4:
            return None

        return PlateResult(
            text=text,
            normalized=normalized,
            confidence=confidence,
            plate_bbox=plate_bbox,
            plate_image=plate_img,
        )

    def _crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _detect_plate_region(
        self, vehicle_crop: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Locate the license plate region using edge detection and contour analysis.
        Looks for rectangles with plate-like aspect ratio in the lower half of the vehicle.
        """
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        # Focus on lower 60% of vehicle crop — plate is usually on front/rear bumper
        h, w = gray.shape
        roi = gray[int(h * 0.4):, :]

        # Edge detection
        blurred = cv2.bilateralFilter(roi, 11, 17, 17)
        edges = cv2.Canny(blurred, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        y_offset = int(h * 0.4)
        best_contour = None
        best_bbox = np.zeros(4)

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) != 4:
                continue

            bx, by, bw, bh = cv2.boundingRect(approx)
            if bw < PLATE_MIN_WIDTH or bh < PLATE_MIN_HEIGHT:
                continue

            aspect = bw / (bh + 1e-6)
            if not (PLATE_ASPECT_MIN <= aspect <= PLATE_ASPECT_MAX):
                continue

            best_contour = approx
            # Adjust bbox back to full vehicle crop coordinates
            best_bbox = np.array([bx, by + y_offset, bx + bw, by + bh + y_offset])
            break

        return best_contour, best_bbox

    def _perspective_correct(
        self, vehicle_crop: np.ndarray, contour: np.ndarray
    ) -> np.ndarray | None:
        """Warp the detected plate quadrilateral to a flat rectangle."""
        pts = contour.reshape(4, 2).astype(np.float32)

        # Order points: top-left, top-right, bottom-right, bottom-left
        ordered = self._order_points(pts)

        # Compute output dimensions
        width_top = np.linalg.norm(ordered[1] - ordered[0])
        width_bot = np.linalg.norm(ordered[2] - ordered[3])
        height_left = np.linalg.norm(ordered[3] - ordered[0])
        height_right = np.linalg.norm(ordered[2] - ordered[1])

        out_w = int(max(width_top, width_bot))
        out_h = int(max(height_left, height_right))

        if out_w < PLATE_MIN_WIDTH or out_h < PLATE_MIN_HEIGHT:
            return None

        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(vehicle_crop, M, (out_w, out_h))

    def _run_ocr(self, plate_img: np.ndarray) -> tuple[str, float] | None:
        """Run EasyOCR on the plate image. Returns (text, confidence) or None."""
        reader = self._get_reader()
        results = reader.readtext(plate_img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -")
        if not results:
            return None
        # Take highest-confidence result
        best = max(results, key=lambda r: r[2])
        return best[1], float(best[2])

    @staticmethod
    def _normalize_plate(text: str) -> str:
        """Remove spaces, hyphens, lowercase → standard plate format."""
        normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
        return normalized

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]    # top-left: smallest sum
        ordered[2] = pts[np.argmax(s)]    # bottom-right: largest sum
        diff = np.diff(pts, axis=1)
        ordered[1] = pts[np.argmin(diff)] # top-right: smallest diff
        ordered[3] = pts[np.argmax(diff)] # bottom-left: largest diff
        return ordered
