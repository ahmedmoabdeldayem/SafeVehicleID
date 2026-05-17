/**
 * Parallel frame processor using Rayon.
 *
 * When running multiple cameras on one server, we can process all camera
 * frames in parallel using Rayon's thread pool. This is the Rust equivalent
 * of running N threads manually, but with automatic work stealing.
 *
 * Rayon parallelism is data parallelism: the same operation (detect+track)
 * runs on each camera frame independently. Safe because each camera has its
 * own independent state.
 */

use rayon::prelude::*;
use crate::tracker::BBox;

/// Represents one camera's frame at a point in time.
#[derive(Debug, Clone)]
pub struct CameraFrame {
    pub camera_id: usize,
    pub frame_number: u64,
    pub width: u32,
    pub height: u32,
    /// Flattened pixel buffer (grayscale for demo — production uses BGR/YUV)
    pub pixels: Vec<u8>,
}

/// Result of processing one camera frame.
#[derive(Debug, Clone)]
pub struct FrameDetections {
    pub camera_id: usize,
    pub frame_number: u64,
    pub detections: Vec<BBox>,
    pub processing_time_us: u64,
}

/// Process a batch of frames from multiple cameras in parallel.
///
/// Each frame is processed independently — no shared mutable state.
/// Rayon distributes work across CPU cores automatically.
///
/// # Example
/// ```rust
/// let frames: Vec<CameraFrame> = cameras.iter()
///     .map(|cam| cam.capture_frame())
///     .collect();
/// let results = process_frames_parallel(&frames);
/// ```
pub fn process_frames_parallel(frames: &[CameraFrame]) -> Vec<FrameDetections> {
    frames
        .par_iter()
        .map(|frame| process_single_frame(frame))
        .collect()
}

fn process_single_frame(frame: &CameraFrame) -> FrameDetections {
    let start = std::time::Instant::now();

    // Simple motion detection via row-variance analysis
    // In production: call into C++ detection via FFI or use an OpenCV Rust binding
    let detections = detect_motion_regions(frame);

    let elapsed_us = start.elapsed().as_micros() as u64;

    FrameDetections {
        camera_id: frame.camera_id,
        frame_number: frame.frame_number,
        detections,
        processing_time_us: elapsed_us,
    }
}

/// Detect motion regions by finding high-variance rectangular regions.
/// This is a simplified placeholder for the real background subtraction.
fn detect_motion_regions(frame: &CameraFrame) -> Vec<BBox> {
    let w = frame.width as usize;
    let h = frame.height as usize;

    if frame.pixels.len() < w * h {
        return vec![];
    }

    let mut detections = Vec::new();
    let block_size = 32usize;

    for row in (0..h.saturating_sub(block_size)).step_by(block_size) {
        for col in (0..w.saturating_sub(block_size)).step_by(block_size) {
            let variance = compute_block_variance(&frame.pixels, w, row, col, block_size);
            if variance > 800.0 {
                detections.push(BBox {
                    x1: col as f32,
                    y1: row as f32,
                    x2: (col + block_size) as f32,
                    y2: (row + block_size) as f32,
                });
            }
        }
    }

    detections
}

fn compute_block_variance(pixels: &[u8], width: usize, row: usize, col: usize, size: usize) -> f32 {
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    let mut count = 0u64;

    for r in row..(row + size) {
        for c in col..(col + size) {
            let idx = r * width + c;
            if idx < pixels.len() {
                let v = pixels[idx] as u64;
                sum += v;
                sum_sq += v * v;
                count += 1;
            }
        }
    }

    if count == 0 { return 0.0; }

    let mean = sum as f64 / count as f64;
    let variance = (sum_sq as f64 / count as f64) - mean * mean;
    variance as f32
}

/// Generate synthetic camera frames for testing (no hardware needed).
pub fn generate_test_frames(num_cameras: usize, frame_number: u64) -> Vec<CameraFrame> {
    (0..num_cameras)
        .map(|cam_id| {
            let width = 640u32;
            let height = 480u32;
            // Simulate motion: a moving "vehicle" blob
            let mut pixels = vec![50u8; (width * height) as usize];
            let vehicle_x = ((frame_number * 3 + cam_id as u64 * 50) % width as u64) as usize;
            let vehicle_y = (height / 3) as usize;
            for dy in 0..60 {
                for dx in 0..100 {
                    let px = vehicle_x + dx;
                    let py = vehicle_y + dy;
                    if px < width as usize && py < height as usize {
                        pixels[py * width as usize + px] = 200;
                    }
                }
            }
            CameraFrame {
                camera_id: cam_id,
                frame_number,
                width,
                height,
                pixels,
            }
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_processing_returns_one_result_per_frame() {
        let frames = generate_test_frames(4, 1);
        let results = process_frames_parallel(&frames);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn camera_ids_preserved() {
        let frames = generate_test_frames(3, 1);
        let mut results = process_frames_parallel(&frames);
        results.sort_by_key(|r| r.camera_id);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.camera_id, i);
        }
    }

    #[test]
    fn synthetic_frames_have_detectable_motion() {
        let frames = generate_test_frames(1, 1);
        let results = process_frames_parallel(&frames);
        // The synthetic vehicle blob should produce at least one detection
        assert!(
            !results[0].detections.is_empty(),
            "Expected motion detection in synthetic frame"
        );
    }

    #[test]
    fn parallel_vs_sequential_same_results() {
        let frames = generate_test_frames(8, 42);

        // Sequential
        let sequential: Vec<FrameDetections> = frames.iter()
            .map(|f| process_single_frame(f))
            .collect();

        // Parallel
        let parallel = process_frames_parallel(&frames);

        assert_eq!(sequential.len(), parallel.len());
        for (seq, par) in sequential.iter().zip(parallel.iter()) {
            assert_eq!(seq.camera_id, par.camera_id);
            assert_eq!(seq.detections.len(), par.detections.len(),
                "Camera {} had different detection counts: seq={} par={}",
                seq.camera_id, seq.detections.len(), par.detections.len());
        }
    }

    #[test]
    fn empty_frame_list_returns_empty_results() {
        let results = process_frames_parallel(&[]);
        assert!(results.is_empty());
    }
}
