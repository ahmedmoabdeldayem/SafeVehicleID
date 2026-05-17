#include "../../include/tracking/sort_tracker.hpp"
#include <algorithm>
#include <numeric>
#include <limits>

namespace safevid {

SORTTracker::SORTTracker(const SORTConfig& config) : config_(config) {
    tracks_.reserve(64);  // pre-allocate — avoids reallocation in steady state
}

std::vector<const Track*> SORTTracker::update(const std::vector<BBox>& detections) {
    // Step 1: predict all existing tracks
    std::vector<BBox> predictions;
    predictions.reserve(tracks_.size());
    for (auto& track : tracks_) {
        predictions.push_back(track.predict());
    }

    // Step 2: solve assignment
    std::vector<std::pair<int,int>> matched;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_dets;

    match(predictions, detections, matched, unmatched_tracks, unmatched_dets);

    // Step 3: update matched tracks
    for (auto [ti, di] : matched) {
        tracks_[ti].update(detections[di]);
        tracks_[ti].hits++;
        tracks_[ti].consecutive_misses = 0;
        if (tracks_[ti].hits >= config_.min_hits_to_confirm) {
            tracks_[ti].state = TrackState::Confirmed;
        }
    }

    // Step 4: handle unmatched tracks
    for (int ti : unmatched_tracks) {
        tracks_[ti].consecutive_misses++;
        if (tracks_[ti].consecutive_misses > config_.max_disappeared) {
            tracks_[ti].state = TrackState::Deleted;
        }
    }

    // Step 5: create new tracks for unmatched detections
    for (int di : unmatched_dets) {
        Track& t = tracks_.emplace_back(
            next_id_++,
            config_.process_noise,
            config_.measurement_noise
        );
        t.kalman.initialize(detections[di]);
        t.hits = 1;
    }

    // Remove deleted tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [](const Track& t) { return t.state == TrackState::Deleted; }),
        tracks_.end()
    );

    // Return pointers to confirmed tracks only
    std::vector<const Track*> confirmed;
    for (const auto& t : tracks_) {
        if (t.state == TrackState::Confirmed) {
            confirmed.push_back(&t);
        }
    }
    return confirmed;
}

void SORTTracker::match(
    const std::vector<BBox>& predictions,
    const std::vector<BBox>& detections,
    std::vector<std::pair<int,int>>& matched,
    std::vector<int>& unmatched_tracks,
    std::vector<int>& unmatched_dets
) {
    const int n_tracks = static_cast<int>(predictions.size());
    const int n_dets   = static_cast<int>(detections.size());

    // All unmatched by default
    for (int i = 0; i < n_tracks; ++i) unmatched_tracks.push_back(i);
    for (int i = 0; i < n_dets;   ++i) unmatched_dets.push_back(i);

    if (n_tracks == 0 || n_dets == 0) return;

    // Build IoU cost matrix (cost = 1 - IoU)
    std::vector<std::vector<float>> cost(n_tracks, std::vector<float>(n_dets));
    for (int ti = 0; ti < n_tracks; ++ti)
        for (int di = 0; di < n_dets; ++di)
            cost[ti][di] = 1.0f - predictions[ti].iou(detections[di]);

    // Hungarian algorithm
    std::vector<int> assignment = hungarian(cost);

    // Process assignment results
    for (int ti = 0; ti < n_tracks; ++ti) {
        int di = assignment[ti];
        if (di < 0) continue;  // unassigned

        float iou_val = 1.0f - cost[ti][di];
        if (iou_val >= config_.iou_threshold) {
            matched.push_back({ti, di});
            unmatched_tracks.erase(
                std::remove(unmatched_tracks.begin(), unmatched_tracks.end(), ti),
                unmatched_tracks.end()
            );
            unmatched_dets.erase(
                std::remove(unmatched_dets.begin(), unmatched_dets.end(), di),
                unmatched_dets.end()
            );
        }
    }
}

std::vector<int> SORTTracker::hungarian(const std::vector<std::vector<float>>& cost_matrix) {
    const int n_rows = static_cast<int>(cost_matrix.size());
    if (n_rows == 0) return {};
    const int n_cols = static_cast<int>(cost_matrix[0].size());

    // Munkres (Hungarian) algorithm
    // Works on square matrix — pad if needed
    const int n = std::max(n_rows, n_cols);
    constexpr float INF = std::numeric_limits<float>::max() / 2.0f;

    std::vector<std::vector<float>> C(n, std::vector<float>(n, INF));
    for (int i = 0; i < n_rows; ++i)
        for (int j = 0; j < n_cols; ++j)
            C[i][j] = cost_matrix[i][j];

    std::vector<float> u(n + 1, 0.0f), v(n + 1, 0.0f);
    std::vector<int>  p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minVal(n + 1, INF);
        std::vector<bool>  used(n + 1, false);

        do {
            used[j0] = true;
            int i0 = p[j0], j1 = -1;
            float delta = INF;

            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    float cur = C[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minVal[j]) {
                        minVal[j] = cur;
                        way[j] = j0;
                    }
                    if (minVal[j] < delta) {
                        delta = minVal[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          { minVal[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // Extract assignment: result[row] = col (0-indexed), -1 if no valid assignment
    std::vector<int> result(n_rows, -1);
    for (int j = 1; j <= n; ++j) {
        int row = p[j] - 1;
        int col = j - 1;
        if (row >= 0 && row < n_rows && col < n_cols) {
            if (C[row][col] < INF) {
                result[row] = col;
            }
        }
    }
    return result;
}

void SORTTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

} // namespace safevid
