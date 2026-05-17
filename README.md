# SafeVehicleID

Infrastructure-based multi-camera vehicle identification and tracking system for autonomous parking and logistics environments.

Built to demonstrate the architecture behind systems like Copernicus ACS — where intelligence lives in the infrastructure (cameras + servers), not in the vehicle.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PARKING FACILITY                              │
│                                                                      │
│   [CAM 1]──┐                                                        │
│   [CAM 2]──┤                                                        │
│   [CAM 3]──┼──► Edge Server ──► [Detection] ──► [Tracking]         │
│   [CAM N]──┘        │                                ▼              │
│                      │                     [Identification]         │
│                      │                     (ANPR + Re-ID)           │
│                      │                          │                   │
└──────────────────────┼──────────────────────────┼───────────────────┘
                        │                          │
                        ▼                          ▼
               ┌─────────────────┐      ┌──────────────────────┐
               │  Message Broker  │      │   Central State DB   │
               │    (Kafka)       │      │  (vehicle positions) │
               └────────┬────────┘      └──────────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Parking Control  │
               │ System / Gates   │
               └─────────────────┘
```

---

## Components

| Layer | Language | Purpose |
|---|---|---|
| `python/detection` | Python | Background subtraction + YOLO detector wrappers |
| `python/tracking` | Python | Kalman filter + SORT tracker (algorithm prototyping) |
| `python/identification` | Python | License plate OCR + vehicle Re-ID embeddings |
| `python/anomaly` | Python | Rule-based + statistical anomaly detection |
| `python/pipeline` | Python | Per-camera pipeline orchestration + multi-camera manager |
| `cpp/` | C++17 | Production real-time pipeline: ring buffer, thread pool, SORT tracker |
| `rust/` | Rust | Lock-free ring buffer + concurrent frame processor |
| `proto/` | Protobuf | gRPC service definitions for inter-component messaging |
| `config/` | YAML | Pipeline configuration (thresholds, camera params, safety limits) |

---

## Architecture Decisions

### Why infrastructure-based?
Vehicle-side autonomy requires each car to carry a full sensor suite (LiDAR, radar, cameras). Infrastructure-based autonomy uses fixed cameras at known positions with known geometry — simpler calibration, cheaper per-vehicle cost, deterministic coverage.

### Pipeline design: Python → C++
Algorithms are prototyped in Python (fast iteration, rich CV/ML libraries). Once validated, the hot path is implemented in C++ with:
- Pre-allocated frame buffers (no heap allocation in the frame loop)
- Lock-free ring buffers between pipeline stages
- Each stage runs in its own thread

### Safety model (SOTIF)
This is ISO 21448 territory, not ISO 26262. Every identification decision carries a confidence score. Actions (gate open, move command) are only triggered above a configurable threshold. Below threshold: hold state, alert operator.

---

## Quick Start

### Python pipeline

```bash
cd python
pip install -r requirements.txt

# Run single-camera pipeline on a video file
python -m pipeline.camera_pipeline --source ./docs/sample.mp4 --camera-id 1

# Run multi-camera manager (reads from config)
python -m pipeline.multi_camera_manager --config ../config/pipeline_config.yaml
```

### C++ pipeline

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run on video source
./safe_vehicle_id --source /dev/video0 --camera-id 1
```

### Rust ring buffer + processor

```bash
cd rust
cargo build --release
cargo test
cargo run --release -- --frames 1000
```

### Docker (full stack)

```bash
docker-compose up
```

---

## Configuration

All tunable parameters live in `config/pipeline_config.yaml`:

```yaml
safety:
  identification_confidence_threshold: 0.85  # below this: hold, do not act
  max_identification_age_frames: 30

tracking:
  max_disappeared_frames: 15
  iou_threshold: 0.3
  kalman:
    process_noise: 0.01
    measurement_noise: 0.1

detection:
  method: yolo          # yolo | background_subtraction
  yolo_model: yolov8n
  confidence_threshold: 0.5
  nms_threshold: 0.4
```

---

## Testing

```bash
# Python
cd python
pytest tests/ -v

# C++
cd cpp/build
ctest --output-on-failure

# Rust
cd rust
cargo test
```

---

## Project Structure

```
SafeVehicleID/
├── python/
│   ├── detection/
│   │   ├── background_subtractor.py    # MOG2-based static camera detector
│   │   └── yolo_detector.py            # YOLOv8 wrapper with confidence filtering
│   ├── tracking/
│   │   ├── kalman_filter.py            # Kalman filter from scratch
│   │   └── sort_tracker.py             # SORT: Kalman + Hungarian matching
│   ├── identification/
│   │   ├── license_plate.py            # ANPR pipeline (EasyOCR)
│   │   └── vehicle_reid.py             # Embedding-based Re-ID
│   ├── anomaly/
│   │   └── anomaly_detector.py         # Rule-based + Mahalanobis anomaly detection
│   ├── pipeline/
│   │   ├── camera_pipeline.py          # Single-camera pipeline
│   │   └── multi_camera_manager.py     # Multi-camera orchestration
│   └── tests/
│       ├── test_kalman.py
│       ├── test_sort.py
│       └── test_anomaly.py
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── utils/
│   │   │   ├── ring_buffer.hpp         # Lock-free ring buffer
│   │   │   └── thread_pool.hpp         # Fixed-size thread pool
│   │   ├── tracking/
│   │   │   ├── kalman_tracker.hpp
│   │   │   └── sort_tracker.hpp
│   │   ├── detection/
│   │   │   └── detector.hpp            # Abstract detector interface
│   │   └── pipeline/
│   │       └── frame_pipeline.hpp      # Multi-stage threaded pipeline
│   └── src/
│       ├── tracking/
│       │   ├── kalman_tracker.cpp
│       │   └── sort_tracker.cpp
│       ├── pipeline/
│       │   └── frame_pipeline.cpp
│       └── main.cpp
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── ring_buffer.rs              # SPSC lock-free ring buffer
│       ├── tracker.rs                  # Concurrent vehicle tracker
│       └── frame_processor.rs         # Parallel frame processing with Rayon
├── proto/
│   └── tracking.proto                  # gRPC service definitions
├── config/
│   └── pipeline_config.yaml
├── docs/
│   └── ARCHITECTURE.md
├── docker/
│   ├── Dockerfile.python
│   └── Dockerfile.cpp
└── docker-compose.yml
```

---

## Key Concepts Implemented

- **Kalman Filter**: full predict/update cycle with configurable process and measurement noise
- **SORT tracker**: Hungarian algorithm matching on IoU cost matrix, Kalman prediction for occluded tracks
- **Background subtraction**: MOG2 with morphological cleanup and contour-based bounding boxes
- **ANPR pipeline**: plate region detection → perspective correction → OCR
- **Vehicle Re-ID**: CNN embedding extraction + cosine similarity matching across cameras
- **Anomaly detection**: velocity rules + Mahalanobis distance from learned normal trajectory distribution
- **Lock-free ring buffer**: both C++ (atomic) and Rust (ownership-safe) implementations
- **Thread pool**: fixed worker threads with task queue, zero allocation in hot path
- **Safety thresholds**: all identification decisions gated by confidence, no action below threshold
