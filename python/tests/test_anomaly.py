"""
Unit tests for the anomaly detector.

Tests verify:
- Overspeed rule triggers above threshold
- Stalled vehicle rule triggers after max dwell frames
- Wrong direction rule triggers for reverse travel
- Unknown vehicle in restricted zone triggers anomaly
- No anomaly for normal behavior
"""

import numpy as np
import pytest
from anomaly.anomaly_detector import AnomalyDetector, AnomalyType, KinematicRules


def test_no_anomaly_for_normal_speed():
    rules = KinematicRules(max_speed_pixels_per_frame=25.0)
    # Speed = 10 px/frame — well under limit
    events = rules.check(1, vx=8.0, vy=6.0, cx=200, cy=200, frame_number=1)
    overspeed_events = [e for e in events if e.anomaly_type == AnomalyType.OVERSPEED]
    assert len(overspeed_events) == 0


def test_overspeed_rule_triggers():
    rules = KinematicRules(max_speed_pixels_per_frame=25.0)
    # Speed = sqrt(30^2 + 0^2) = 30 > 25
    events = rules.check(1, vx=30.0, vy=0.0, cx=200, cy=200, frame_number=1)
    types = [e.anomaly_type for e in events]
    assert AnomalyType.OVERSPEED in types


def test_overspeed_severity_scales_with_speed():
    rules = KinematicRules(max_speed_pixels_per_frame=25.0)
    events_low = rules.check(1, vx=26.0, vy=0.0, cx=0, cy=0, frame_number=1)
    events_high = rules.check(2, vx=60.0, vy=0.0, cx=0, cy=0, frame_number=1)

    low_sev = next(e.severity for e in events_low if e.anomaly_type == AnomalyType.OVERSPEED)
    high_sev = next(e.severity for e in events_high if e.anomaly_type == AnomalyType.OVERSPEED)
    assert high_sev > low_sev


def test_stalled_vehicle_rule():
    rules = KinematicRules(max_dwell_frames=5)
    # Speed = 0 for 6 frames > max_dwell_frames=5
    for i in range(6):
        events = rules.check(1, vx=0.0, vy=0.0, cx=100, cy=100, frame_number=i)
    stalled = [e for e in events if e.anomaly_type == AnomalyType.STALLED]
    assert len(stalled) > 0


def test_stalled_resets_after_movement():
    rules = KinematicRules(max_dwell_frames=5)
    # Stall for 4 frames
    for i in range(4):
        rules.check(1, vx=0.0, vy=0.0, cx=100, cy=100, frame_number=i)
    # Move
    rules.check(1, vx=10.0, vy=0.0, cx=110, cy=100, frame_number=5)
    # Stall again — counter should have reset, so need another 5 frames
    events = rules.check(1, vx=0.0, vy=0.0, cx=110, cy=100, frame_number=6)
    stalled = [e for e in events if e.anomaly_type == AnomalyType.STALLED]
    assert len(stalled) == 0


def test_wrong_direction_rule():
    rules = KinematicRules()
    # Allowed direction: moving right (+x). Vehicle moving left (-x).
    allowed_direction = (1.0, 0.0)  # unit vector pointing right
    events = rules.check(
        1, vx=-15.0, vy=0.0, cx=200, cy=200, frame_number=1,
        allowed_direction=allowed_direction
    )
    wrong_dir = [e for e in events if e.anomaly_type == AnomalyType.WRONG_DIRECTION]
    assert len(wrong_dir) > 0


def test_correct_direction_no_anomaly():
    rules = KinematicRules()
    allowed_direction = (1.0, 0.0)
    events = rules.check(
        1, vx=15.0, vy=0.0, cx=200, cy=200, frame_number=1,
        allowed_direction=allowed_direction
    )
    wrong_dir = [e for e in events if e.anomaly_type == AnomalyType.WRONG_DIRECTION]
    assert len(wrong_dir) == 0


def test_unknown_vehicle_in_restricted_zone():
    detector = AnomalyDetector()
    events = detector.check_track(
        track_id=1,
        vx=5.0, vy=0.0, cx=200, cy=200,
        frame_number=1,
        vehicle_id=None,
        identification_confidence=0.0,
        restricted_zone=True,
    )
    unknown = [e for e in events if e.anomaly_type == AnomalyType.UNKNOWN_VEHICLE]
    assert len(unknown) > 0


def test_identified_vehicle_no_identity_anomaly():
    detector = AnomalyDetector()
    events = detector.check_track(
        track_id=1,
        vx=5.0, vy=0.0, cx=200, cy=200,
        frame_number=1,
        vehicle_id="VEH-00001",
        identification_confidence=0.95,
        restricted_zone=True,
    )
    unknown = [e for e in events if e.anomaly_type == AnomalyType.UNKNOWN_VEHICLE]
    assert len(unknown) == 0


def test_statistical_detector_flags_outlier():
    from anomaly.anomaly_detector import StatisticalAnomalyDetector

    # Normal data: speed ~10, accel ~1, curvature ~0.1
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=[10.0, 1.0, 0.1], scale=[2.0, 0.3, 0.05], size=(500, 3))
    normal = np.abs(normal)

    detector = StatisticalAnomalyDetector(threshold_percentile=99.0)
    detector.fit(normal)

    assert detector.is_fitted()

    # Normal observation — should NOT be flagged
    result = detector.detect(1, speed=10.0, acceleration=1.0, curvature=0.1,
                              frame_number=1, position=(0, 0))
    assert result is None

    # Extreme outlier — should be flagged
    result = detector.detect(1, speed=100.0, acceleration=50.0, curvature=5.0,
                              frame_number=2, position=(0, 0))
    assert result is not None
    assert result.anomaly_type == AnomalyType.STATISTICAL
    assert result.severity > 0.0
