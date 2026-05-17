/**
 * SafeVehicleID — Rust entry point.
 *
 * Demonstrates:
 * - Parallel frame processing across multiple cameras using Rayon
 * - Lock-free SPSC ring buffer for frame passing
 * - Thread-safe vehicle track registry
 * - Ownership model preventing data races at compile time
 *
 * Run:
 *   cargo run --release -- --cameras 4 --frames 1000
 */

mod ring_buffer;
mod tracker;
mod frame_processor;

use clap::Parser;
use frame_processor::{generate_test_frames, process_frames_parallel};
use ring_buffer::ring_buffer;
use tracker::TrackRegistry;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "safe_vehicle_id", about = "Multi-camera vehicle tracking (Rust)")]
struct Args {
    /// Number of simulated cameras
    #[arg(long, default_value_t = 4)]
    cameras: usize,

    /// Number of frames to process
    #[arg(long, default_value_t = 500)]
    frames: u64,

    /// Print stats every N frames
    #[arg(long, default_value_t = 100)]
    print_interval: u64,
}

fn main() {
    let args = Args::parse();

    println!("SafeVehicleID Rust Processor");
    println!("  Cameras : {}", args.cameras);
    println!("  Frames  : {}", args.frames);
    println!();

    // One registry per camera (independent state — safe to parallelize)
    let registries: Vec<TrackRegistry> = (0..args.cameras)
        .map(|_| TrackRegistry::new(3, 15, 0.3))
        .collect();

    // Ring buffer: frame generator → processor
    let (mut tx, mut rx) = ring_buffer::<Vec<frame_processor::CameraFrame>, 16>();

    let cameras = args.cameras;
    let total_frames = args.frames;

    // Producer thread: generates synthetic frames
    let producer = thread::spawn(move || {
        for frame_num in 0..total_frames {
            let frames = generate_test_frames(cameras, frame_num);
            // Retry on full buffer
            loop {
                match tx.push(frames.clone()) {
                    Ok(_) => break,
                    Err(_) => thread::yield_now(),
                }
            }
        }
    });

    // Consumer (main thread): process frames
    let start = Instant::now();
    let mut processed = 0u64;
    let mut total_detections = 0usize;

    while processed < args.frames {
        match rx.pop() {
            Some(frames) => {
                // Process all cameras in parallel
                let results = process_frames_parallel(&frames);

                // Update trackers (sequentially per camera — each has its own state)
                for result in &results {
                    let registry = &registries[result.camera_id];
                    let confirmed = registry.update(&result.detections);
                    total_detections += result.detections.len();

                    // Simulate identification for confirmed tracks
                    for track_id in confirmed {
                        // In real system: run ANPR/Re-ID here
                        // For demo: assign synthetic ID after confidence threshold
                        registry.set_vehicle_id(
                            track_id,
                            format!("VEH-{:05}", track_id),
                            0.92,
                        );
                    }
                }

                processed += 1;

                if processed % args.print_interval == 0 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let fps = processed as f64 / elapsed;
                    print_status(processed, args.frames, fps, &registries, total_detections);
                }
            }
            None => thread::yield_now(),
        }
    }

    producer.join().unwrap();

    let elapsed = start.elapsed();
    println!("\n--- Final Summary ---");
    println!("Frames processed : {}", processed);
    println!("Total detections : {}", total_detections);
    println!("Total time       : {:.2}s", elapsed.as_secs_f64());
    println!("Throughput       : {:.1} frames/sec", processed as f64 / elapsed.as_secs_f64());

    for (i, registry) in registries.iter().enumerate() {
        let tracks = registry.get_confirmed_tracks();
        let safe_count = tracks.iter().filter(|t| t.is_safe_to_act(0.85)).count();
        println!("Camera {i} : {} confirmed tracks, {} safe-to-act", tracks.len(), safe_count);
    }
}

fn print_status(
    processed: u64,
    total: u64,
    fps: f64,
    registries: &[TrackRegistry],
    total_detections: usize,
) {
    let total_tracks: usize = registries.iter()
        .map(|r| r.get_confirmed_tracks().len())
        .sum();

    println!(
        "[{:>5}/{:>5}] {:.1} fps | {} confirmed tracks | {} total detections",
        processed, total, fps, total_tracks, total_detections
    );
}
