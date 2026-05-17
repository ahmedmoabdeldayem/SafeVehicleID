# System Architecture

## Overview

SafeVehicleID is structured as a layered perception pipeline. The Python layer is the algorithm development environment — fast iteration, rich libraries, easy visualization. The C++ layer is the production runtime — deterministic timing, pre-allocated memory, zero-overhead abstractions. The Rust layer provides safety-critical data structures with compile-time concurrency guarantees.

---

## Pipeline Stages

### 1. Video Ingestion

Each camera feeds frames into the pipeline at a configurable frame rate (default 30 fps → 33ms per frame budget). The C++ implementation uses a lock-free ring buffer between the capture thread and the processing thread to avoid blocking the camera reader.

```
Camera HW → GStreamer/V4L2 → RingBuffer<Frame> → Detection Thread
```

Latency budget:
- Capture: ~0ms (hardware timestamp)
- Queue: < 1ms (lock-free push)
- Detection: 10–30ms (YOLO on GPU, or ~2ms for background subtraction)
- Tracking: ~1ms
- Identification: ~5ms (plate OCR)
- Total: target < 33ms for real-time at 30fps

---

### 2. Detection

Two strategies selectable via config:

**Background Subtraction (MOG2)**
Best for: static infrastructure cameras, controlled lighting, low compute budget.

```
frame → MOG2 → fg_mask → morphological_cleanup → contour_extraction → bounding_boxes
```

MOG2 models each pixel as a mixture of Gaussians. It adapts to slow illumination changes (shadow, day/night) but can confuse slow-moving vehicles as background.

**YOLO Detection**
Best for: variable scenes, outdoor environments, higher accuracy requirement.

```
frame → resize_to_640x640 → YOLOv8_inference → NMS_filtering → class_filter(vehicle) → bounding_boxes
```

NMS (Non-Maximum Suppression) removes duplicate detections for the same vehicle. IoU threshold of 0.4 is a typical starting point.

---

### 3. Tracking

SORT (Simple Online and Real-time Tracking) is implemented in both Python (reference) and C++ (production).

**Core algorithm:**

```
For each new frame:
  1. Predict: advance each existing track's Kalman filter one step forward
  2. Match: compute IoU cost matrix between predictions and new detections
  3. Assign: Hungarian algorithm finds minimum-cost matching
  4. Update: matched tracks update their Kalman state with the detection measurement
  5. Create: unmatched detections spawn new tracks
  6. Delete: tracks with no match for > N consecutive frames are removed
```

**Track state machine:**

```
[tentative] --matched N times--> [confirmed] --unmatched M times--> [deleted]
```

Tentative tracks are not reported externally until confirmed. This prevents spurious tracks from detector noise.

**Kalman state vector:**

```
x = [cx, cy, vx, vy, w, h]
     center  velocity  size
```

Constant velocity motion model. Works well for slow parking-speed vehicles. For faster logistics yards, can extend to constant acceleration model.

---

### 4. Identification

Two complementary mechanisms:

**License Plate Recognition (ANPR)**
Primary identity signal. Resistant to camera handoff issues.

```
vehicle_crop → plate_detector → perspective_warp → ocr → plate_string → identity_db_lookup
```

**Vehicle Re-ID**
Used when plate is not visible (occlusion, angle, distance). An embedding CNN produces a 256-dim feature vector. Same vehicle from different angles → high cosine similarity.

```
vehicle_crop → resize(128x256) → resnet_backbone → l2_normalized_embedding
                                                          ↓
                                              cosine_similarity vs gallery
```

Gallery = set of embeddings for known vehicles in the system.

**Fusion:**
If ANPR succeeds with high confidence → use that identity.
If ANPR fails → fall back to Re-ID if similarity > threshold.
If both fail → mark as UNKNOWN, do not act.

---

### 5. Anomaly Detection

Three rule layers, applied in order:

**Layer 1: Kinematic rules**
- Velocity too high for zone (speeding)
- Dwell time too long in moving zone (stalled)
- Direction violation (one-way lane)
- Sudden velocity reversal

**Layer 2: Statistical (Mahalanobis distance)**
A Gaussian model is fitted to normal trajectory features (velocity, acceleration, path curvature) from historical data. At inference, Mahalanobis distance from the mean signals unusual behavior.

```python
d_M = sqrt((x - mu)^T * Sigma^-1 * (x - mu))
```

Threshold tuned to 99th percentile of normal distribution → ~1% false positive rate.

**Layer 3: Unknown vehicle**
Any vehicle that cannot be identified after K frames in an area requiring identification → anomaly.

---

### 6. Multi-Camera Management

Each camera runs an independent pipeline (separate threads/processes). Cross-camera coordination handles:

**Zone transitions**: when a vehicle exits one camera's view and enters another's, the Re-ID embedding is used to link the identity.

**Global track IDs**: each camera assigns local track IDs. The multi-camera manager maps local IDs to global vehicle IDs using identification.

**Conflict resolution**: if two cameras claim the same vehicle is in different locations simultaneously → flag as identification conflict, hold actions.

---

## Safety Architecture

```
Detection result
      │
      ▼
Tracking update (confidence: float 0-1)
      │
      ▼
Identification (confidence: float 0-1)
      │
      ├── confidence >= HIGH_THRESHOLD (0.90) ──► SAFE_ACTION (gate open, move command)
      │
      ├── confidence >= LOW_THRESHOLD (0.70)  ──► MONITOR_ONLY (log, display, no action)
      │
      └── confidence < LOW_THRESHOLD          ──► HOLD + ALERT (operator notification)
```

No action is taken on a vehicle that cannot be confidently identified. This is the core SOTIF mitigation: the system's failure mode is conservative (hold/wait), not permissive (guess and act).

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│ EDGE SERVER (per camera zone)                                         │
│                                                                       │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Camera   │───►│  Detector    │───►│   Tracker    │               │
│  │  Reader   │    │ (BG / YOLO)  │    │    (SORT)    │               │
│  └──────────┘    └──────────────┘    └──────┬───────┘               │
│                                             │                         │
│                                    ┌────────▼───────┐               │
│                                    │  Identifier     │               │
│                                    │  ANPR + Re-ID   │               │
│                                    └────────┬───────┘               │
│                                             │                         │
│                                    ┌────────▼───────┐               │
│                                    │ Anomaly Checker │               │
│                                    └────────┬───────┘               │
│                                             │                         │
└─────────────────────────────────────────────┼────────────────────────┘
                                              │ TrackEvent (gRPC/Kafka)
                                              ▼
                               ┌──────────────────────────┐
                               │   Central State Service   │
                               │  - Global vehicle map     │
                               │  - Cross-camera linking   │
                               │  - Action authorization   │
                               └──────────────────────────┘
```

---

## Thread Model (C++ Production)

```
Thread 0: Camera reader
  └── pushes frames to RingBuffer<Frame>

Thread 1: Detection worker
  └── pops from RingBuffer<Frame>
  └── runs detector
  └── pushes detections to RingBuffer<DetectionResult>

Thread 2: Tracking + Identification worker
  └── pops from RingBuffer<DetectionResult>
  └── updates SORT tracker
  └── runs ANPR on vehicle crops
  └── publishes TrackEvent via gRPC

Thread 3: Anomaly monitor
  └── subscribes to track updates
  └── evaluates anomaly rules
  └── publishes AnomalyEvent if triggered
```

Ring buffers are sized to 16 frames — enough buffer for processing spikes without unbounded memory growth.

---

## Deployment

### Edge (per parking zone)
- Hardware: NVIDIA Jetson AGX Orin or similar edge GPU
- OS: Ubuntu 22.04 with PREEMPT_RT kernel patch for real-time guarantees
- Process: C++ pipeline binary, one instance per camera
- Communication: gRPC to central service

### Central service
- Kubernetes deployment (2+ replicas for redundancy)
- State stored in Redis (low-latency) + PostgreSQL (persistent log)
- Kafka for event streaming from all edge nodes

### Cloud (optional, for model training + data storage)
- AWS S3: video clip storage for anomaly events
- AWS SageMaker: re-training Re-ID and detection models on new data
- AWS IoT Greengrass: OTA model deployment to edge devices
