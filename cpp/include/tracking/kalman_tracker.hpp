#pragma once

/**
 * Kalman filter tracker for 2D bounding box tracking.
 *
 * State vector: [cx, cy, vx, vy, w, h]
 * Measurement:  [cx, cy, w, h]
 *
 * Same math as the Python implementation — reference that for derivation.
 * This C++ version uses Eigen for matrix operations.
 */

#include <array>
#include <stdexcept>
#include <cmath>

// Forward declaration — Eigen used in .cpp to avoid header bloat
// Defined here as raw arrays for header-only compilation option

namespace safevid {

struct BBox {
    float x1, y1, x2, y2;

    float cx() const { return (x1 + x2) * 0.5f; }
    float cy() const { return (y1 + y2) * 0.5f; }
    float w()  const { return x2 - x1; }
    float h()  const { return y2 - y1; }

    static BBox from_center(float cx, float cy, float w, float h) {
        return { cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f };
    }

    float iou(const BBox& other) const {
        float ix1 = std::max(x1, other.x1);
        float iy1 = std::max(y1, other.y1);
        float ix2 = std::min(x2, other.x2);
        float iy2 = std::min(y2, other.y2);

        float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
        if (inter == 0.0f) return 0.0f;

        float a1 = w() * h();
        float a2 = other.w() * other.h();
        return inter / (a1 + a2 - inter + 1e-6f);
    }
};

/**
 * 6-state Kalman filter operating on 4x4 float matrices.
 * Implements predict/update cycle with configurable noise covariances.
 */
class KalmanTracker {
public:
    static constexpr int STATE_DIM = 6;
    static constexpr int MEAS_DIM  = 4;

    using StateVec  = std::array<float, STATE_DIM>;
    using StateMat  = std::array<std::array<float, STATE_DIM>, STATE_DIM>;
    using MeasVec   = std::array<float, MEAS_DIM>;
    using MeasMat   = std::array<std::array<float, MEAS_DIM>, MEAS_DIM>;

    explicit KalmanTracker(float process_noise = 0.01f, float measurement_noise = 0.1f);

    void initialize(const BBox& bbox);
    BBox predict();
    BBox update(const BBox& measurement);
    BBox get_state_bbox() const;

    float vx() const { return x_[2]; }
    float vy() const { return x_[3]; }
    float speed() const { return std::sqrt(x_[2]*x_[2] + x_[3]*x_[3]); }

    bool is_initialized() const { return initialized_; }

private:
    bool initialized_{false};

    StateVec x_{};     // state estimate
    StateMat P_{};     // state covariance
    StateMat F_{};     // state transition (motion model)
    StateMat Q_{};     // process noise covariance

    // Measurement model (H: 4x6 matrix stored as [row][col])
    std::array<std::array<float, STATE_DIM>, MEAS_DIM> H_{};
    MeasMat R_{};      // measurement noise covariance

    // Matrix operation helpers
    static StateMat mat_mul_66(const StateMat& A, const StateMat& B);
    static StateMat mat_add_66(const StateMat& A, const StateMat& B);
    static StateMat mat_transpose_66(const StateMat& A);
    static MeasMat  invert_44(const MeasMat& M);

    static BBox state_to_bbox(const StateVec& x);
    static StateVec bbox_to_state(const BBox& bbox);
};

} // namespace safevid
