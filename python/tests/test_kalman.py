"""
Unit tests for the Kalman filter implementation.

Tests verify:
- Initialization from a bounding box
- Predict step propagates position by velocity
- Update step corrects state toward measurement
- Covariance evolves correctly
"""

import numpy as np
import pytest
from tracking.kalman_filter import KalmanFilter


def test_initialization():
    kf = KalmanFilter()
    bbox = np.array([100.0, 200.0, 200.0, 300.0])  # 100x100 box centered at (150, 250)
    kf.initialize(bbox)

    assert kf.x is not None
    assert kf.P is not None

    # Check center and size
    np.testing.assert_allclose(kf.x[0], 150.0)   # cx
    np.testing.assert_allclose(kf.x[1], 250.0)   # cy
    np.testing.assert_allclose(kf.x[4], 100.0)   # w
    np.testing.assert_allclose(kf.x[5], 100.0)   # h
    # Initial velocity should be zero
    np.testing.assert_allclose(kf.x[2], 0.0)     # vx
    np.testing.assert_allclose(kf.x[3], 0.0)     # vy


def test_predict_stationary_vehicle():
    """A vehicle with zero initial velocity should stay in place after predict."""
    kf = KalmanFilter()
    bbox = np.array([100.0, 100.0, 200.0, 200.0])
    kf.initialize(bbox)

    predicted = kf.predict()

    # With zero velocity, center should not move
    np.testing.assert_allclose(predicted[0], 100.0, atol=1.0)  # x1
    np.testing.assert_allclose(predicted[1], 100.0, atol=1.0)  # y1


def test_predict_propagates_velocity():
    """
    After updating with a displaced measurement, the Kalman filter should
    estimate a non-zero velocity and predict accordingly.
    """
    kf = KalmanFilter(measurement_noise=0.01)  # trust measurements heavily
    kf.initialize(np.array([100.0, 100.0, 200.0, 200.0]))

    # Simulate vehicle moving right by 20px per frame
    for _ in range(10):
        kf.update(np.array([120.0, 100.0, 220.0, 200.0]))
        kf.predict()

    vx, vy = kf.get_velocity()
    # After 10 consistent measurements, velocity should converge toward 20 px/frame
    assert vx > 5.0, f"Expected positive vx, got {vx:.2f}"
    assert abs(vy) < 5.0, f"Expected near-zero vy, got {vy:.2f}"


def test_update_corrects_toward_measurement():
    """Update should move state estimate toward the measurement."""
    kf = KalmanFilter()
    kf.initialize(np.array([100.0, 100.0, 200.0, 200.0]))

    # Predict moves to ~(150, 150) center (no velocity change)
    kf.predict()
    state_before = kf.x.copy()

    # Measurement says vehicle is actually at (175, 150) center
    kf.update(np.array([125.0, 100.0, 225.0, 200.0]))
    state_after = kf.x.copy()

    # cx should have moved toward 175
    assert state_after[0] > state_before[0], "cx should have increased toward measurement"


def test_predict_requires_initialization():
    """Predict without initialization should raise RuntimeError."""
    kf = KalmanFilter()
    with pytest.raises(RuntimeError, match="not initialized"):
        kf.predict()


def test_bbox_roundtrip():
    """Converting bbox → center → bbox should be lossless."""
    kf = KalmanFilter()
    original = np.array([50.0, 75.0, 150.0, 175.0])
    kf.initialize(original)
    result = kf.get_state_bbox()
    np.testing.assert_allclose(result, original, atol=1e-6)


def test_covariance_grows_during_prediction():
    """Covariance should increase during prediction (uncertainty grows without measurement)."""
    kf = KalmanFilter()
    kf.initialize(np.array([100.0, 100.0, 200.0, 200.0]))

    initial_trace = np.trace(kf.P)
    kf.predict()
    after_predict_trace = np.trace(kf.P)

    assert after_predict_trace > initial_trace, "Covariance should increase during prediction"


def test_covariance_decreases_after_update():
    """Covariance should decrease after an update (uncertainty reduced by measurement)."""
    kf = KalmanFilter()
    kf.initialize(np.array([100.0, 100.0, 200.0, 200.0]))
    kf.predict()
    before_update_trace = np.trace(kf.P)

    kf.update(np.array([100.0, 100.0, 200.0, 200.0]))
    after_update_trace = np.trace(kf.P)

    assert after_update_trace < before_update_trace, "Covariance should decrease after update"
