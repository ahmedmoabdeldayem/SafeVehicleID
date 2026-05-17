"""
Unit tests for the SORT tracker.

Tests verify:
- New detection creates a tentative track
- Track becomes confirmed after min_hits
- Track is deleted after max_disappeared frames
- Hungarian matching correctly assigns detections to tracks
- IoU below threshold results in no match
"""

import numpy as np
import pytest
from tracking.sort_tracker import SORTTracker, TrackState, compute_iou


# --- IoU tests ---

def test_iou_identical_boxes():
    box = np.array([0.0, 0.0, 100.0, 100.0])
    assert compute_iou(box, box) == pytest.approx(1.0, abs=1e-5)


def test_iou_no_overlap():
    box1 = np.array([0.0, 0.0, 50.0, 50.0])
    box2 = np.array([100.0, 100.0, 150.0, 150.0])
    assert compute_iou(box1, box2) == pytest.approx(0.0, abs=1e-5)


def test_iou_partial_overlap():
    box1 = np.array([0.0, 0.0, 100.0, 100.0])
    box2 = np.array([50.0, 0.0, 150.0, 100.0])
    # Intersection: 50x100 = 5000
    # Union: 10000 + 10000 - 5000 = 15000
    expected = 5000.0 / 15000.0
    assert compute_iou(box1, box2) == pytest.approx(expected, abs=1e-4)


# --- SORT tracker tests ---

def make_bbox(cx: float, cy: float, w: float = 100, h: float = 60) -> np.ndarray:
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float64)


def test_new_detection_creates_tentative_track():
    tracker = SORTTracker(min_hits_to_confirm=3)
    detections = np.array([make_bbox(200, 200)])
    confirmed = tracker.update(detections)
    assert len(confirmed) == 0, "Track should not be confirmed on first detection"
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].state == TrackState.TENTATIVE


def test_track_confirmed_after_min_hits():
    tracker = SORTTracker(min_hits_to_confirm=3)
    det = np.array([make_bbox(200, 200)])
    for _ in range(3):
        confirmed = tracker.update(det)
    assert len(confirmed) == 1
    assert confirmed[0].state == TrackState.CONFIRMED


def test_track_deleted_after_max_disappeared():
    tracker = SORTTracker(min_hits_to_confirm=1, max_disappeared=3)
    det = np.array([make_bbox(200, 200)])
    tracker.update(det)  # creates + confirms

    # Now stop sending detections
    for _ in range(4):  # one more than max_disappeared
        confirmed = tracker.update(np.empty((0, 4)))

    assert len(tracker.tracks) == 0, "Track should be deleted after max_disappeared"


def test_unique_track_ids():
    tracker = SORTTracker(min_hits_to_confirm=1)
    # Two separate vehicles
    det = np.array([make_bbox(100, 100), make_bbox(500, 100)])
    tracker.update(det)
    ids = [t.track_id for t in tracker.tracks]
    assert len(set(ids)) == 2, "Each track should have a unique ID"


def test_matching_assigns_closest_detection():
    """
    Two vehicles at known positions. New detections close to both.
    Verify that each vehicle gets the correct detection assigned.
    """
    tracker = SORTTracker(min_hits_to_confirm=1, iou_threshold=0.1)

    # Initial positions
    tracker.update(np.array([make_bbox(100, 200), make_bbox(400, 200)]))

    # Vehicles move slightly to the right
    detections = np.array([make_bbox(110, 200), make_bbox(410, 200)])
    confirmed = tracker.update(detections)

    assert len(confirmed) == 2

    # Track IDs should persist (not create 2 new tracks)
    ids = {t.track_id for t in confirmed}
    assert ids == {1, 2}, f"Expected track IDs 1 and 2, got {ids}"


def test_no_match_below_iou_threshold():
    """Detection far from all tracks should create a new track, not match existing."""
    tracker = SORTTracker(min_hits_to_confirm=1, iou_threshold=0.3)

    tracker.update(np.array([make_bbox(100, 100)]))  # track at (100, 100)

    # New detection completely elsewhere
    tracker.update(np.array([make_bbox(800, 800)]))

    assert len(tracker.tracks) == 2, "Should create a new track for non-matching detection"


def test_empty_frame_does_not_crash():
    tracker = SORTTracker()
    result = tracker.update(np.empty((0, 4)))
    assert result == []


def test_reset_clears_all_state():
    tracker = SORTTracker(min_hits_to_confirm=1)
    tracker.update(np.array([make_bbox(100, 100)]))
    tracker.reset()
    assert len(tracker.tracks) == 0
    assert tracker._next_id == 1


def test_track_velocity_increases_with_motion():
    """After several frames of motion, track velocity should be non-zero."""
    tracker = SORTTracker(min_hits_to_confirm=1)
    cx = 200.0
    for _ in range(15):
        cx += 10.0
        tracker.update(np.array([make_bbox(cx, 200)]))

    confirmed = [t for t in tracker.tracks if t.state == TrackState.CONFIRMED]
    assert len(confirmed) == 1
    vx, vy = confirmed[0].get_velocity()
    assert vx > 2.0, f"Expected positive vx after motion, got {vx:.2f}"
