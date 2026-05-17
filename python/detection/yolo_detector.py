"""
YOLOv8-based vehicle detector.

Wraps the Ultralytics YOLOv8 inference pipeline with:
- Vehicle class filtering (car, truck, bus, motorcycle)
- Confidence and NMS threshold configuration
- Batch inference support for multi-camera scaling
- Optional GPU / CPU device selection
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    bbox: np.ndarray        # [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """
    YOLOv8 vehicle detector with class and confidence filtering.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier. Options:
        'yolov8n' (nano, fastest), 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
        For production on Jetson AGX Orin: yolov8s or yolov8m hits 30fps.
    confidence_threshold : float
        Minimum detection confidence to report.
    nms_threshold : float
        IoU threshold for Non-Maximum Suppression.
    device : str
        'cuda', 'cpu', or 'cuda:0' for specific GPU.
    vehicle_classes_only : bool
        If True, only return car/truck/bus/motorcycle detections.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cpu",
        vehicle_classes_only: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.vehicle_classes_only = vehicle_classes_only
        self._model = None
        self._model_name = model_name
        self._device = device

    def _load_model(self):
        """Lazy-load model on first inference call."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_name)
            self._model.to(self._device)
        except ImportError:
            raise ImportError(
                "ultralytics package required for YOLO detector. "
                "Install with: pip install ultralytics"
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        frame : ndarray (H, W, 3) BGR

        Returns
        -------
        List of Detection objects, sorted by confidence descending.
        """
        if self._model is None:
            self._load_model()

        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False,
            device=self._device,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if self.vehicle_classes_only and cls_id not in VEHICLE_CLASSES:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append(Detection(
                    bbox=xyxy.astype(np.float32),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=VEHICLE_CLASSES.get(cls_id, str(cls_id)),
                ))

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run detection on a batch of frames in one forward pass.
        More efficient than calling detect() in a loop when using GPU.

        Parameters
        ----------
        frames : list of BGR frames

        Returns
        -------
        List of detection lists, one per input frame.
        """
        if self._model is None:
            self._load_model()

        results = self._model.predict(
            source=frames,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False,
            device=self._device,
        )

        batch_detections = []
        for result in results:
            frame_dets = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if self.vehicle_classes_only and cls_id not in VEHICLE_CLASSES:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    frame_dets.append(Detection(
                        bbox=xyxy.astype(np.float32),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=VEHICLE_CLASSES.get(cls_id, str(cls_id)),
                    ))
            batch_detections.append(
                sorted(frame_dets, key=lambda d: d.confidence, reverse=True)
            )

        return batch_detections

    def visualize(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw detections on frame for debugging."""
        import cv2
        viz = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                viz, label,
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1
            )
        return viz
