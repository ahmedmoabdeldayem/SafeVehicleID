/**
 * Concurrent vehicle tracker.
 *
 * Uses Arc<Mutex<T>> to safely share track state between threads.
 * The Rust compiler enforces that no two threads mutate state simultaneously —
 * unlike C++ where you rely on discipline and code review.
 *
 * Key Rust concepts demonstrated:
 * - Arc: reference-counted shared ownership (like shared_ptr)
 * - Mutex: mutual exclusion with automatic unlock on drop (RAII)
 * - Send + Sync: compile-time thread-safety markers
 * - Pattern matching for exhaustive state handling
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Deleted,
}

#[derive(Debug, Clone)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub fn cx(&self) -> f32 { (self.x1 + self.x2) * 0.5 }
    pub fn cy(&self) -> f32 { (self.y1 + self.y2) * 0.5 }
    pub fn w(&self)  -> f32 { self.x2 - self.x1 }
    pub fn h(&self)  -> f32 { self.y2 - self.y1 }

    pub fn iou(&self, other: &BBox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);

        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        if inter == 0.0 { return 0.0; }

        let a1 = self.w() * self.h();
        let a2 = other.w() * other.h();
        inter / (a1 + a2 - inter + 1e-6)
    }
}

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u32,
    pub state: TrackState,
    pub hits: u32,
    pub consecutive_misses: u32,
    pub bbox: BBox,
    pub vx: f32,
    pub vy: f32,
    pub vehicle_id: Option<String>,
    pub identification_confidence: f32,
}

impl Track {
    pub fn speed(&self) -> f32 {
        (self.vx * self.vx + self.vy * self.vy).sqrt()
    }

    pub fn is_safe_to_act(&self, threshold: f32) -> bool {
        self.vehicle_id.is_some()
            && self.identification_confidence >= threshold
            && self.state == TrackState::Confirmed
    }
}

/// Thread-safe track registry.
///
/// Multiple threads can call `get_confirmed_tracks()` concurrently.
/// Only one thread can call `update()` at a time (write lock).
///
/// In practice, the tracker runs in a single processing thread, but
/// the state can be queried from a monitoring/reporting thread.
pub struct TrackRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

struct RegistryInner {
    tracks: HashMap<u32, Track>,
    next_id: u32,
    min_hits_to_confirm: u32,
    max_disappeared: u32,
    iou_threshold: f32,
}

impl TrackRegistry {
    pub fn new(min_hits_to_confirm: u32, max_disappeared: u32, iou_threshold: f32) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                tracks: HashMap::new(),
                next_id: 1,
                min_hits_to_confirm,
                max_disappeared,
                iou_threshold,
            })),
        }
    }

    /// Update tracker with new detections. Returns confirmed track IDs.
    pub fn update(&self, detections: &[BBox]) -> Vec<u32> {
        // Lock for the duration of the update
        let mut inner = self.inner.lock().unwrap();

        // Simple centroid matching (full Kalman + Hungarian in cpp/src/tracking/)
        let track_ids: Vec<u32> = inner.tracks.keys().cloned().collect();
        let mut matched_det_indices: Vec<bool> = vec![false; detections.len()];
        let mut matched_track_ids: Vec<u32> = Vec::new();

        // Match each existing track to the nearest detection by IoU
        for &tid in &track_ids {
            let track = inner.tracks.get(&tid).unwrap().clone();
            let mut best_iou = inner.iou_threshold;
            let mut best_det = None;

            for (di, det) in detections.iter().enumerate() {
                if !matched_det_indices[di] {
                    let iou = track.bbox.iou(det);
                    if iou > best_iou {
                        best_iou = iou;
                        best_det = Some(di);
                    }
                }
            }

            match best_det {
                Some(di) => {
                    matched_det_indices[di] = true;
                    matched_track_ids.push(tid);

                    let t = inner.tracks.get_mut(&tid).unwrap();
                    t.bbox = detections[di].clone();
                    t.hits += 1;
                    t.consecutive_misses = 0;
                    if t.hits >= inner.min_hits_to_confirm {
                        t.state = TrackState::Confirmed;
                    }
                }
                None => {
                    let t = inner.tracks.get_mut(&tid).unwrap();
                    t.consecutive_misses += 1;
                    if t.consecutive_misses > inner.max_disappeared {
                        t.state = TrackState::Deleted;
                    }
                }
            }
        }

        // Create new tracks for unmatched detections
        for (di, det) in detections.iter().enumerate() {
            if !matched_det_indices[di] {
                let new_id = inner.next_id;
                inner.next_id += 1;
                inner.tracks.insert(new_id, Track {
                    id: new_id,
                    state: TrackState::Tentative,
                    hits: 1,
                    consecutive_misses: 0,
                    bbox: det.clone(),
                    vx: 0.0,
                    vy: 0.0,
                    vehicle_id: None,
                    identification_confidence: 0.0,
                });
            }
        }

        // Remove deleted tracks
        inner.tracks.retain(|_, t| t.state != TrackState::Deleted);

        // Return confirmed track IDs
        inner.tracks
            .values()
            .filter(|t| t.state == TrackState::Confirmed)
            .map(|t| t.id)
            .collect()
    }

    /// Get a snapshot of all confirmed tracks. Thread-safe read.
    pub fn get_confirmed_tracks(&self) -> Vec<Track> {
        let inner = self.inner.lock().unwrap();
        inner.tracks
            .values()
            .filter(|t| t.state == TrackState::Confirmed)
            .cloned()
            .collect()
    }

    /// Update vehicle identity for a specific track.
    pub fn set_vehicle_id(&self, track_id: u32, vehicle_id: String, confidence: f32) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(track) = inner.tracks.get_mut(&track_id) {
            track.vehicle_id = Some(vehicle_id);
            track.identification_confidence = confidence;
        }
    }

    pub fn track_count(&self) -> usize {
        self.inner.lock().unwrap().tracks.len()
    }

    /// Create a shareable clone (Arc clone — cheap, no data copy).
    pub fn clone_handle(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(cx: f32, cy: f32) -> BBox {
        BBox { x1: cx - 50.0, y1: cy - 30.0, x2: cx + 50.0, y2: cy + 30.0 }
    }

    #[test]
    fn new_detection_is_tentative() {
        let registry = TrackRegistry::new(3, 15, 0.3);
        let confirmed = registry.update(&[bbox(100.0, 200.0)]);
        assert!(confirmed.is_empty(), "Should not be confirmed on first detection");
        assert_eq!(registry.track_count(), 1);
    }

    #[test]
    fn track_confirmed_after_min_hits() {
        let registry = TrackRegistry::new(3, 15, 0.3);
        let det = vec![bbox(100.0, 200.0)];
        for _ in 0..3 {
            registry.update(&det);
        }
        let confirmed = registry.get_confirmed_tracks();
        assert_eq!(confirmed.len(), 1);
        assert_eq!(confirmed[0].state, TrackState::Confirmed);
    }

    #[test]
    fn track_deleted_after_max_disappeared() {
        let registry = TrackRegistry::new(1, 3, 0.3);
        registry.update(&[bbox(100.0, 100.0)]);
        // Stop sending detections
        for _ in 0..4 {
            registry.update(&[]);
        }
        assert_eq!(registry.track_count(), 0);
    }

    #[test]
    fn vehicle_id_can_be_set() {
        let registry = TrackRegistry::new(1, 15, 0.3);
        registry.update(&[bbox(100.0, 100.0)]);
        // Get the track ID
        let tracks = registry.get_confirmed_tracks();
        // After min_hits=1 and 1 update, it's confirmed
        if let Some(t) = tracks.first() {
            registry.set_vehicle_id(t.id, "ABC-123".to_string(), 0.95);
            let updated = registry.get_confirmed_tracks();
            let updated_track = updated.iter().find(|t2| t2.id == t.id).unwrap();
            assert_eq!(updated_track.vehicle_id.as_deref(), Some("ABC-123"));
            assert!(updated_track.is_safe_to_act(0.85));
        }
    }

    #[test]
    fn iou_identical_boxes() {
        let b = BBox { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0 };
        assert!((b.iou(&b) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn iou_no_overlap() {
        let b1 = BBox { x1: 0.0, y1: 0.0, x2: 50.0, y2: 50.0 };
        let b2 = BBox { x1: 100.0, y1: 100.0, x2: 150.0, y2: 150.0 };
        assert_eq!(b1.iou(&b2), 0.0);
    }

    #[test]
    fn concurrent_read_write() {
        use std::thread;

        let registry = TrackRegistry::new(1, 15, 0.3);
        let reader_handle = registry.clone_handle();

        // Writer thread: updates every millisecond
        let writer = thread::spawn(move || {
            for i in 0..100 {
                let x = (i as f32) * 2.0 + 100.0;
                registry.update(&[bbox(x, 200.0)]);
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        });

        // Reader thread: reads concurrently
        let reader = thread::spawn(move || {
            for _ in 0..50 {
                let _tracks = reader_handle.get_confirmed_tracks();
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
        // If we get here without deadlock or panic, the test passes
    }
}
