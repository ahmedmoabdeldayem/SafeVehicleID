"""
Single-camera perception pipeline.

Orchestrates: detection → tracking → identification → anomaly detection

Designed to run as a standalone process per camera, publishing track events
to a central state service via callback. This separation allows horizontal
scaling: 50 cameras = 50 pipeline processes.

The pipeline uses a producer-consumer model with a frame queue, decoupling
capture (I/O-bound) from processing (compute-bound).
"""

from __future__ import annotations

import time
import queue
import threading
import logging
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import yaml

from ..detection.background_subtractor import BackgroundSubtractorDetector
from ..detection.yolo_detector import YOLODetector
from ..tracking.sort_tracker import SORTTracker, Track
from ..identification.license_plate import LicensePlateRecognizer
from ..identification.vehicle_reid import VehicleReID
from ..anomaly.anomaly_detector import AnomalyDetector, AnomalyEvent

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    camera_id: int = 1
    detection_method: str = "background_subtraction"  # or "yolo"
    yolo_model: str = "yolov8n"
    detection_confidence: float = 0.5
    max_disappeared_frames: int = 15
    min_hits_to_confirm: int = 3
    iou_threshold: float = 0.3
    identification_confidence_threshold: float = 0.85
    max_identification_age_frames: int = 30
    anomaly_max_speed: float = 25.0
    frame_queue_size: int = 16
    display: bool = False
    restricted_zone: bool = False

    @classmethod
    def from_yaml(cls, path: str, camera_id: int = 1) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls(camera_id=camera_id)
        det = raw.get("detection", {})
        trk = raw.get("tracking", {})
        sft = raw.get("safety", {})
        anom = raw.get("anomaly", {})

        cfg.detection_method = det.get("method", cfg.detection_method)
        cfg.yolo_model = det.get("yolo_model", cfg.yolo_model)
        cfg.detection_confidence = det.get("confidence_threshold", cfg.detection_confidence)
        cfg.max_disappeared_frames = trk.get("max_disappeared_frames", cfg.max_disappeared_frames)
        cfg.min_hits_to_confirm = trk.get("min_hits_to_confirm", cfg.min_hits_to_confirm)
        cfg.iou_threshold = trk.get("iou_threshold", cfg.iou_threshold)
        cfg.identification_confidence_threshold = sft.get(
            "identification_confidence_threshold", cfg.identification_confidence_threshold
        )
        cfg.anomaly_max_speed = anom.get("max_speed_pixels_per_frame", cfg.anomaly_max_speed)
        return cfg


@dataclass
class TrackEvent:
    """Published output for each confirmed track update."""
    camera_id: int
    frame_number: int
    timestamp: float
    track_id: int
    bbox: np.ndarray
    velocity: tuple[float, float]
    vehicle_id: str | None
    identification_confidence: float
    anomalies: list[AnomalyEvent] = field(default_factory=list)

    @property
    def is_safe_to_act(self) -> bool:
        """True if identification confidence is above safety threshold."""
        return (
            self.vehicle_id is not None
            and self.identification_confidence >= 0.85
            and len(self.anomalies) == 0
        )


class CameraPipeline:
    """
    Full perception pipeline for a single camera.

    Parameters
    ----------
    source : video source — file path, camera index (int), or RTSP URL
    config : PipelineConfig
    on_track_event : callback(TrackEvent) called for each confirmed track update
    on_anomaly : callback(AnomalyEvent) called for each anomaly detected
    """

    def __init__(
        self,
        source,
        config: PipelineConfig | None = None,
        on_track_event: Callable[[TrackEvent], None] | None = None,
        on_anomaly: Callable[[AnomalyEvent], None] | None = None,
    ):
        self.source = source
        self.config = config or PipelineConfig()
        self.on_track_event = on_track_event or (lambda e: None)
        self.on_anomaly = on_anomaly or (lambda e: None)

        # Frame queue decouples capture from processing
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.config.frame_queue_size)
        self._running = threading.Event()
        self._frame_number = 0

        # Pipeline components
        self._detector = self._build_detector()
        self._tracker = SORTTracker(
            max_disappeared=self.config.max_disappeared_frames,
            min_hits_to_confirm=self.config.min_hits_to_confirm,
            iou_threshold=self.config.iou_threshold,
        )
        self._plate_recognizer = LicensePlateRecognizer(
            min_confidence=self.config.identification_confidence_threshold
        )
        self._reid = VehicleReID(
            similarity_threshold=self.config.identification_confidence_threshold
        )
        self._anomaly_detector = AnomalyDetector(
            max_speed_pixels_per_frame=self.config.anomaly_max_speed,
        )

        # Track identification cache: track_id → (vehicle_id, confidence, age)
        self._id_cache: dict[int, tuple[str, float, int]] = {}

    def _build_detector(self):
        if self.config.detection_method == "yolo":
            return YOLODetector(
                model_name=self.config.yolo_model,
                confidence_threshold=self.config.detection_confidence,
            )
        return BackgroundSubtractorDetector()

    def run(self) -> None:
        """Start the pipeline. Blocks until stop() is called or source ends."""
        self._running.set()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        # Start capture thread
        capture_thread = threading.Thread(
            target=self._capture_loop, args=(cap,), daemon=True
        )
        capture_thread.start()

        logger.info(f"Camera {self.config.camera_id} pipeline started | source={self.source}")

        try:
            self._processing_loop()
        finally:
            self._running.clear()
            capture_thread.join(timeout=2.0)
            cap.release()
            if self.config.display:
                cv2.destroyAllWindows()
            logger.info(f"Camera {self.config.camera_id} pipeline stopped")

    def stop(self) -> None:
        self._running.clear()

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        """Read frames from camera and push to queue. Runs in a dedicated thread."""
        while self._running.is_set():
            ret, frame = cap.read()
            if not ret:
                self._running.clear()
                break
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                # Drop oldest frame if queue is full — processing can't keep up
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass

    def _processing_loop(self) -> None:
        """Main processing loop: detect → track → identify → check anomalies."""
        while self._running.is_set():
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._frame_number += 1
            t_start = time.monotonic()

            # 1. Detect
            detections = self._detector.detect(frame)
            det_bboxes = np.array([d.bbox for d in detections]) if detections else np.empty((0, 4))

            # 2. Track
            confirmed_tracks = self._tracker.update(det_bboxes)

            # 3. Identify + anomaly check for each confirmed track
            for track in confirmed_tracks:
                vehicle_id, id_confidence = self._get_identification(frame, track)
                vx, vy = track.get_velocity()
                bbox = track.get_bbox()
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                # Anomaly check
                anomalies = self._anomaly_detector.check_track(
                    track_id=track.track_id,
                    vx=vx, vy=vy, cx=cx, cy=cy,
                    frame_number=self._frame_number,
                    vehicle_id=vehicle_id,
                    identification_confidence=id_confidence,
                    restricted_zone=self.config.restricted_zone,
                )

                event = TrackEvent(
                    camera_id=self.config.camera_id,
                    frame_number=self._frame_number,
                    timestamp=time.time(),
                    track_id=track.track_id,
                    bbox=bbox,
                    velocity=(vx, vy),
                    vehicle_id=vehicle_id,
                    identification_confidence=id_confidence,
                    anomalies=anomalies,
                )

                self.on_track_event(event)

                for anomaly in anomalies:
                    self.on_anomaly(anomaly)
                    logger.warning(
                        f"CAM {self.config.camera_id} | ANOMALY [{anomaly.anomaly_type.value}] "
                        f"track={track.track_id} severity={anomaly.severity:.2f}"
                    )

            # 4. Optional visualization
            if self.config.display:
                self._display(frame, confirmed_tracks)

            dt = (time.monotonic() - t_start) * 1000
            logger.debug(
                f"CAM {self.config.camera_id} | frame={self._frame_number} "
                f"tracks={len(confirmed_tracks)} dt={dt:.1f}ms"
            )

    def _get_identification(
        self, frame: np.ndarray, track: Track
    ) -> tuple[str | None, float]:
        """
        Try to identify a vehicle. Uses cache to avoid re-running OCR every frame.
        Returns (vehicle_id, confidence).
        """
        # Use cached result if it's fresh
        if track.track_id in self._id_cache:
            cached_id, cached_conf, age = self._id_cache[track.track_id]
            if age < self.config.max_identification_age_frames:
                self._id_cache[track.track_id] = (cached_id, cached_conf, age + 1)
                return cached_id, cached_conf

        bbox = track.get_bbox()

        # Try ANPR first (more reliable identity)
        plate_result = self._plate_recognizer.recognize(frame, bbox)
        if plate_result is not None:
            self._id_cache[track.track_id] = (
                plate_result.normalized, plate_result.confidence, 0
            )
            return plate_result.normalized, plate_result.confidence

        # Fall back to Re-ID
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                try:
                    reid_result = self._reid.query(crop)
                    if not reid_result.is_new_vehicle:
                        self._id_cache[track.track_id] = (
                            reid_result.vehicle_id, reid_result.similarity, 0
                        )
                        return reid_result.vehicle_id, reid_result.similarity
                except Exception:
                    pass

        return None, 0.0

    def _display(self, frame: np.ndarray, tracks: list[Track]) -> None:
        viz = frame.copy()
        for track in tracks:
            bbox = track.get_bbox().astype(int)
            cached = self._id_cache.get(track.track_id)
            label = f"T{track.track_id}"
            if cached:
                vid, conf, _ = cached
                label = f"{vid} ({conf:.2f})"

            cv2.rectangle(viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(viz, label, (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(viz, f"CAM {self.config.camera_id} | frame {self._frame_number}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.imshow(f"Camera {self.config.camera_id}", viz)
        cv2.waitKey(1)
