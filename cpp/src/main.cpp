/**
 * SafeVehicleID — C++ pipeline entry point.
 *
 * Usage:
 *   ./safe_vehicle_id --source <video_path|camera_index|rtsp_url>
 *                     --camera-id <int>
 *                     [--display]
 *                     [--threshold <float>]
 *
 * Example:
 *   ./safe_vehicle_id --source /dev/video0 --camera-id 1 --display
 *   ./safe_vehicle_id --source parking_lot.mp4 --camera-id 2
 */

#include "../include/pipeline/frame_pipeline.hpp"
#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <iomanip>

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int) {
    g_shutdown.store(true);
}

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& default_val = "") {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == key) return argv[i + 1];
    }
    return default_val;
}

static bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == key) return true;
    return false;
}

int main(int argc, char** argv) {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string source      = get_arg(argc, argv, "--source", "0");
    int         camera_id   = std::stoi(get_arg(argc, argv, "--camera-id", "1"));
    float       threshold   = std::stof(get_arg(argc, argv, "--threshold", "0.85"));
    bool        display     = has_flag(argc, argv, "--display");

    std::cout << "SafeVehicleID C++ Pipeline\n"
              << "  Source    : " << source << "\n"
              << "  Camera ID : " << camera_id << "\n"
              << "  Threshold : " << threshold << "\n"
              << "  Display   : " << (display ? "on" : "off") << "\n\n";

    safevid::PipelineCfg cfg;
    cfg.camera_id                  = camera_id;
    cfg.identification_threshold   = threshold;
    cfg.display                    = display;
    cfg.sort_config.max_disappeared   = 15;
    cfg.sort_config.min_hits_to_confirm = 3;
    cfg.sort_config.iou_threshold     = 0.3f;

    int event_count = 0;
    int safe_action_count = 0;

    auto on_event = [&](const safevid::TrackEvent& e) {
        ++event_count;
        if (e.is_safe_to_act) ++safe_action_count;

        // Log every 30 frames to avoid flooding the terminal
        if (e.frame_number % 30 == 0) {
            float speed = std::sqrt(e.vx * e.vx + e.vy * e.vy);
            std::cout << std::fixed << std::setprecision(1)
                      << "[CAM " << e.camera_id << " | f" << e.frame_number << "] "
                      << "Track " << e.track_id
                      << " @ (" << (e.bbox.x1 + e.bbox.x2) / 2 << ", "
                                << (e.bbox.y1 + e.bbox.y2) / 2 << ")"
                      << " speed=" << speed << "px/f"
                      << " id=" << (e.vehicle_id.empty() ? "UNKNOWN" : e.vehicle_id)
                      << " conf=" << std::setprecision(2) << e.identification_confidence
                      << (e.is_safe_to_act ? " [SAFE]" : " [HOLD]")
                      << "\n";
        }
    };

    safevid::FramePipeline pipeline(cfg, on_event);

    try {
        pipeline.start(source);
    } catch (const std::exception& ex) {
        std::cerr << "Pipeline error: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "Pipeline running. Press Ctrl+C to stop.\n";

    while (!g_shutdown.load() && pipeline.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    pipeline.stop();

    std::cout << "\n--- Summary ---\n"
              << "Frames processed : " << pipeline.frames_processed() << "\n"
              << "Frames dropped   : " << pipeline.frames_dropped() << "\n"
              << "Track events     : " << event_count << "\n"
              << "Safe-to-act      : " << safe_action_count << "\n";

    return 0;
}
