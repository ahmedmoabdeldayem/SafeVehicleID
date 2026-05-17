#include "../../include/pipeline/frame_pipeline.hpp"
#include <chrono>
#include <iostream>
#include <thread>

namespace safevid {

FramePipeline::FramePipeline(const PipelineCfg& config, TrackCallback on_event)
    : config_(config)
    , on_event_(std::move(on_event))
    , tracker_(config.sort_config)
{
    bg_subtractor_ = cv::createBackgroundSubtractorMOG2(500, 16, true);
}

FramePipeline::~FramePipeline() {
    stop();
}

void FramePipeline::start(const std::string& source) {
    if (running_.load()) return;
    running_.store(true);

    capture_thread_   = std::thread(&FramePipeline::capture_loop,   this, source);
    detection_thread_ = std::thread(&FramePipeline::detection_loop, this);
    tracking_thread_  = std::thread(&FramePipeline::tracking_loop,  this);
}

void FramePipeline::stop() {
    running_.store(false);
    if (capture_thread_.joinable())   capture_thread_.join();
    if (detection_thread_.joinable()) detection_thread_.join();
    if (tracking_thread_.joinable())  tracking_thread_.join();
}

// ─── Thread 0: Capture ────────────────────────────────────────────────────────

void FramePipeline::capture_loop(const std::string& source) {
    cv::VideoCapture cap;

    // Support integer camera index or file/URL string
    bool is_index = !source.empty() && std::all_of(source.begin(), source.end(), ::isdigit);
    if (is_index) {
        cap.open(std::stoi(source));
    } else {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        std::cerr << "[FramePipeline] Failed to open source: " << source << "\n";
        running_.store(false);
        return;
    }

    while (running_.load()) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            running_.store(false);
            break;
        }

        if (!frame_buffer_.push(std::move(frame))) {
            // Buffer full — drop oldest and retry
            frame_buffer_.pop();
            frame_buffer_.push(std::move(frame));
            frames_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// ─── Thread 1: Detection ──────────────────────────────────────────────────────

void FramePipeline::detection_loop() {
    int frame_num = 0;

    while (running_.load() || !frame_buffer_.empty()) {
        auto opt = frame_buffer_.pop();
        if (!opt.has_value()) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        cv::Mat frame = std::move(*opt);
        ++frame_num;

        auto detections = run_detection(frame);
        double ts = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();

        DetectionResult result;
        result.frame_number = frame_num;
        result.timestamp    = ts;
        result.detections   = std::move(detections);
        result.frame        = frame.clone();  // clone for identification stage

        if (!detection_buffer_.push(std::move(result))) {
            detection_buffer_.pop();
            detection_buffer_.push(std::move(result));
        }
    }
}

// ─── Thread 2: Tracking ───────────────────────────────────────────────────────

void FramePipeline::tracking_loop() {
    while (running_.load() || !detection_buffer_.empty()) {
        auto opt = detection_buffer_.pop();
        if (!opt.has_value()) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        DetectionResult result = std::move(*opt);
        auto confirmed = tracker_.update(result.detections);
        frames_processed_.fetch_add(1, std::memory_order_relaxed);

        for (const Track* track : confirmed) {
            BBox bbox = track->get_bbox();
            TrackEvent event;
            event.camera_id                  = config_.camera_id;
            event.frame_number               = result.frame_number;
            event.timestamp                  = result.timestamp;
            event.track_id                   = track->track_id;
            event.bbox                       = bbox;
            event.vx                         = track->kalman.vx();
            event.vy                         = track->kalman.vy();
            event.vehicle_id                 = track->vehicle_id;
            event.identification_confidence  = track->identification_confidence;
            event.is_safe_to_act             =
                !track->vehicle_id.empty() &&
                track->identification_confidence >= config_.identification_threshold;

            on_event_(event);
        }

        if (config_.display) {
            cv::Mat viz = result.frame.clone();
            for (const Track* track : confirmed) {
                BBox b = track->get_bbox();
                cv::rectangle(viz,
                    cv::Point(static_cast<int>(b.x1), static_cast<int>(b.y1)),
                    cv::Point(static_cast<int>(b.x2), static_cast<int>(b.y2)),
                    cv::Scalar(0, 255, 0), 2);
                std::string label = "T" + std::to_string(track->track_id);
                if (!track->vehicle_id.empty())
                    label += " " + track->vehicle_id;
                cv::putText(viz, label,
                    cv::Point(static_cast<int>(b.x1), static_cast<int>(b.y1) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            cv::putText(viz,
                "CAM " + std::to_string(config_.camera_id) +
                " | frame " + std::to_string(result.frame_number),
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 0), 1);
            cv::imshow("Camera " + std::to_string(config_.camera_id), viz);
            cv::waitKey(1);
        }
    }
}

// ─── Detection helper ─────────────────────────────────────────────────────────

std::vector<BBox> FramePipeline::run_detection(const cv::Mat& frame) {
    cv::Mat gray, blurred, mask;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    bg_subtractor_->apply(blurred, mask);

    // Remove shadows (127), keep foreground (255)
    cv::threshold(mask, mask, 200, 255, cv::THRESH_BINARY);

    // Morphological cleanup
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::Mat kernel_open  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel_open);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_close);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<BBox> detections;
    const int frame_w = frame.cols;
    const int frame_h = frame.rows;
    constexpr int MIN_AREA = 3000;
    constexpr int MAX_AREA = 200000;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < MIN_AREA || area > MAX_AREA) continue;

        cv::Rect r = cv::boundingRect(contour);
        float aspect = static_cast<float>(r.width) / (r.height + 1e-6f);
        if (aspect < 0.3f || aspect > 4.0f) continue;

        BBox bbox;
        bbox.x1 = static_cast<float>(std::max(0, r.x));
        bbox.y1 = static_cast<float>(std::max(0, r.y));
        bbox.x2 = static_cast<float>(std::min(frame_w, r.x + r.width));
        bbox.y2 = static_cast<float>(std::min(frame_h, r.y + r.height));
        detections.push_back(bbox);
    }

    return detections;
}

} // namespace safevid
