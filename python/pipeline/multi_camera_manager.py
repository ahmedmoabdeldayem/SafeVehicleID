"""
Multi-camera manager.

Runs one CameraPipeline per camera source, all in parallel threads.
Aggregates track events from all cameras into a global vehicle state map.
Handles cross-camera vehicle linking (same vehicle seen by multiple cameras).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

from .camera_pipeline import CameraPipeline, PipelineConfig, TrackEvent
from ..anomaly.anomaly_detector import AnomalyEvent

logger = logging.getLogger(__name__)


@dataclass
class GlobalVehicleState:
    """Aggregated state of a vehicle across all cameras."""
    vehicle_id: str
    last_seen_camera: int
    last_seen_frame: int
    last_position: tuple[float, float]
    last_velocity: tuple[float, float]
    last_seen_timestamp: float
    active_anomalies: list[AnomalyEvent] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # position history

    @property
    def is_stale(self) -> bool:
        """True if not seen for more than 5 seconds."""
        return (time.time() - self.last_seen_timestamp) > 5.0


class MultiCameraManager:
    """
    Manages multiple camera pipelines and maintains global vehicle state.

    Parameters
    ----------
    config_path : path to pipeline_config.yaml
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self._raw_config = yaml.safe_load(f)

        self._pipelines: list[CameraPipeline] = []
        self._threads: list[threading.Thread] = []
        self._global_state: dict[str, GlobalVehicleState] = {}
        self._state_lock = threading.Lock()
        self._running = False

        # Statistics
        self._total_events = 0
        self._total_anomalies = 0

    def start(self) -> None:
        """Start all camera pipelines in parallel threads."""
        cameras = self._raw_config.get("cameras", [])
        if not cameras:
            raise ValueError("No cameras defined in config file under 'cameras:' key")

        self._running = True

        for cam_cfg in cameras:
            cam_id = cam_cfg["id"]
            source = cam_cfg["source"]

            config = PipelineConfig.from_yaml(self._raw_config_path_for(cam_id), cam_id)
            config.camera_id = cam_id
            config.restricted_zone = cam_cfg.get("restricted_zone", False)

            pipeline = CameraPipeline(
                source=source,
                config=config,
                on_track_event=self._handle_track_event,
                on_anomaly=self._handle_anomaly,
            )
            self._pipelines.append(pipeline)

            t = threading.Thread(
                target=pipeline.run,
                name=f"camera-{cam_id}",
                daemon=True,
            )
            self._threads.append(t)
            t.start()
            logger.info(f"Started camera {cam_id} pipeline | source={source}")

        logger.info(f"MultiCameraManager: {len(self._pipelines)} cameras running")

    def stop(self) -> None:
        """Stop all pipelines gracefully."""
        self._running = False
        for pipeline in self._pipelines:
            pipeline.stop()
        for t in self._threads:
            t.join(timeout=5.0)
        logger.info("All camera pipelines stopped")

    def get_all_vehicles(self) -> dict[str, GlobalVehicleState]:
        """Return snapshot of current global vehicle state."""
        with self._state_lock:
            return {k: v for k, v in self._global_state.items() if not v.is_stale}

    def get_vehicle(self, vehicle_id: str) -> GlobalVehicleState | None:
        with self._state_lock:
            return self._global_state.get(vehicle_id)

    def print_status(self) -> None:
        """Print current state summary."""
        vehicles = self.get_all_vehicles()
        print(f"\n{'='*60}")
        print(f"Active vehicles: {len(vehicles)} | "
              f"Events: {self._total_events} | "
              f"Anomalies: {self._total_anomalies}")
        for vid, state in vehicles.items():
            pos = state.last_position
            vel = state.last_velocity
            speed = (vel[0]**2 + vel[1]**2) ** 0.5
            anom_str = f" ⚠ {len(state.active_anomalies)} anomalies" if state.active_anomalies else ""
            print(f"  {vid:15s} | cam={state.last_seen_camera} | "
                  f"pos=({pos[0]:.0f},{pos[1]:.0f}) | "
                  f"speed={speed:.1f}px/f{anom_str}")
        print('='*60)

    def _handle_track_event(self, event: TrackEvent) -> None:
        """Update global state from a track event. Thread-safe."""
        if event.vehicle_id is None:
            return

        self._total_events += 1
        cx = (event.bbox[0] + event.bbox[2]) / 2
        cy = (event.bbox[1] + event.bbox[3]) / 2

        with self._state_lock:
            if event.vehicle_id not in self._global_state:
                self._global_state[event.vehicle_id] = GlobalVehicleState(
                    vehicle_id=event.vehicle_id,
                    last_seen_camera=event.camera_id,
                    last_seen_frame=event.frame_number,
                    last_position=(cx, cy),
                    last_velocity=event.velocity,
                    last_seen_timestamp=event.timestamp,
                )
                logger.info(
                    f"New vehicle registered: {event.vehicle_id} on camera {event.camera_id}"
                )
            else:
                state = self._global_state[event.vehicle_id]
                state.last_seen_camera = event.camera_id
                state.last_seen_frame = event.frame_number
                state.last_position = (cx, cy)
                state.last_velocity = event.velocity
                state.last_seen_timestamp = event.timestamp
                state.history.append({
                    "camera": event.camera_id,
                    "frame": event.frame_number,
                    "pos": (cx, cy),
                    "ts": event.timestamp,
                })
                if len(state.history) > 200:
                    state.history.pop(0)

    def _handle_anomaly(self, event: AnomalyEvent) -> None:
        self._total_anomalies += 1
        logger.warning(
            f"ANOMALY | type={event.anomaly_type.value} "
            f"track={event.track_id} severity={event.severity:.2f} | {event.description}"
        )

    def _raw_config_path_for(self, cam_id: int) -> str:
        # In this simplified version, all cameras share the same config file path
        # In production, cameras would have per-zone config overrides
        import os
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "pipeline_config.yaml"
        )
