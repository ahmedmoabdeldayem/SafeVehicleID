#pragma once

/**
 * SORT tracker (C++ production implementation).
 *
 * Same algorithm as python/tracking/sort_tracker.py:
 *   1. Predict all tracks with Kalman filter
 *   2. Build IoU cost matrix
 *   3. Hungarian algorithm for assignment
 *   4. Update matched, increment missed on unmatched, create new tracks
 *
 * Key differences from Python version:
 *   - Pre-allocated track vector (no heap growth in steady state)
 *   - Inline IoU computation (no Python function call overhead)
 *   - Lock-free — designed to run in a single processing thread
 */

#include "kalman_tracker.hpp"
#include <vector>
#include <cstdint>
#include <string>

namespace safevid {

enum class TrackState : uint8_t {
    Tentative = 0,
    Confirmed = 1,
    Deleted   = 2,
};

struct Track {
    int           track_id{-1};
    TrackState    state{TrackState::Tentative};
    int           hits{0};
    int           consecutive_misses{0};
    KalmanTracker kalman;
    std::string   vehicle_id;
    float         identification_confidence{0.0f};

    Track() = default;
    explicit Track(int id, float process_noise, float measurement_noise)
        : track_id(id)
        , kalman(process_noise, measurement_noise)
    {}

    BBox get_bbox() const { return kalman.get_state_bbox(); }
    BBox predict()        { return kalman.predict(); }
    void update(const BBox& bbox) { kalman.update(bbox); }
    float speed() const   { return kalman.speed(); }
};

struct SORTConfig {
    int   max_disappeared{15};
    int   min_hits_to_confirm{3};
    float iou_threshold{0.3f};
    float process_noise{0.01f};
    float measurement_noise{0.1f};
};

class SORTTracker {
public:
    explicit SORTTracker(const SORTConfig& config = {});

    /**
     * Update tracker with detections from the current frame.
     *
     * @param detections  Vector of bounding boxes (one per detected vehicle)
     * @return            References to all currently confirmed tracks
     */
    std::vector<const Track*> update(const std::vector<BBox>& detections);

    /**
     * All tracks regardless of state (for debugging/visualization).
     */
    const std::vector<Track>& all_tracks() const { return tracks_; }

    void reset();

private:
    SORTConfig config_;
    std::vector<Track> tracks_;
    int next_id_{1};

    /**
     * Solve the assignment problem using the Hungarian algorithm.
     * Returns matched pairs (track_idx, det_idx).
     * Fills unmatched_tracks and unmatched_dets vectors.
     */
    void match(
        const std::vector<BBox>& predictions,
        const std::vector<BBox>& detections,
        std::vector<std::pair<int,int>>& matched,
        std::vector<int>& unmatched_tracks,
        std::vector<int>& unmatched_dets
    );

    /**
     * Minimal Hungarian algorithm implementation.
     * cost_matrix: rows = tracks, cols = detections
     * Returns assignment: assignment[row] = col (-1 if unassigned)
     */
    static std::vector<int> hungarian(const std::vector<std::vector<float>>& cost_matrix);
};

} // namespace safevid
