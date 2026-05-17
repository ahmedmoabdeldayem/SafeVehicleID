#pragma once

/**
 * Multi-stage threaded frame processing pipeline.
 *
 * Three threads, two ring buffers:
 *
 *   [CaptureThread] ──RingBuffer<Frame>──► [DetectionThread]
 *                                                │
 *                                     RingBuffer<DetectionResult>
 *                                                │
 *                                                ▼
 *                                       [TrackingThread]
 *                                                │
 *                                           on_track_event callback
 *
 * No heap allocation after initialization in the steady-state frame loop.
 * Frame buffers are reused via move semantics.
 */

#include "../utils/ring_buffer.hpp"
#include "../tracking/sort_tracker.hpp"
#include <opencv2/opencv.hpp>
#include <functional>
#include <thread>
#include <atomic>
#include <string>
#include <vector>

namespace safevid {

struct TrackEvent {
    int         camera_id;
    int         frame_number;
    double      timestamp;
    int         track_id;
    BBox        bbox;
    float       vx;
    float       vy;
    std::string vehicle_id;
    float       identification_confidence;
    bool        is_safe_to_act;
};

struct DetectionResult {
    int                  frame_number;
    double               timestamp;
    std::vector<BBox>    detections;
    cv::Mat              frame;          // kept for identification stage
};

struct PipelineCfg {
    int         camera_id{1};
    int         frame_queue_size{16};
    int         detection_queue_size{8};
    float       identification_threshold{0.85f};
    SORTConfig  sort_config;
    bool        display{false};
};

class FramePipeline {
public:
    using TrackCallback = std::function<void(const TrackEvent&)>;

    explicit FramePipeline(const PipelineCfg& config, TrackCallback on_event);
    ~FramePipeline();

    FramePipeline(const FramePipeline&) = delete;
    FramePipeline& operator=(const FramePipeline&) = delete;

    /**
     * Open source and start all pipeline threads.
     * source: camera index (int as string), file path, or RTSP URL.
     */
    void start(const std::string& source);

    /**
     * Signal all threads to stop and wait for them to finish.
     */
    void stop();

    bool is_running() const { return running_.load(); }

    int frames_processed() const { return frames_processed_.load(); }
    int frames_dropped() const   { return frames_dropped_.load(); }

private:
    PipelineCfg   config_;
    TrackCallback on_event_;

    // Ring buffers between stages (size must be power of 2)
    RingBuffer<cv::Mat, 16>           frame_buffer_;
    RingBuffer<DetectionResult, 8>    detection_buffer_;

    std::thread capture_thread_;
    std::thread detection_thread_;
    std::thread tracking_thread_;

    std::atomic<bool> running_{false};
    std::atomic<int>  frames_processed_{0};
    std::atomic<int>  frames_dropped_{0};

    SORTTracker tracker_;

    void capture_loop(const std::string& source);
    void detection_loop();
    void tracking_loop();

    // Simple background subtraction (no external ML dependency)
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor_;
    std::vector<BBox> run_detection(const cv::Mat& frame);
};

} // namespace safevid
