#include "../../include/tracking/kalman_tracker.hpp"
#include <cstring>
#include <algorithm>

namespace safevid {

KalmanTracker::KalmanTracker(float process_noise, float measurement_noise) {
    // State transition matrix F (constant velocity model)
    for (auto& row : F_) row.fill(0.0f);
    for (int i = 0; i < STATE_DIM; ++i) F_[i][i] = 1.0f;
    F_[0][2] = 1.0f;  // cx += vx
    F_[1][3] = 1.0f;  // cy += vy

    // Measurement matrix H (4x6): observe cx, cy, w, h — not velocity
    for (auto& row : H_) row.fill(0.0f);
    H_[0][0] = 1.0f;  // cx
    H_[1][1] = 1.0f;  // cy
    H_[2][4] = 1.0f;  // w
    H_[3][5] = 1.0f;  // h

    // Process noise Q
    for (auto& row : Q_) row.fill(0.0f);
    for (int i = 0; i < STATE_DIM; ++i) Q_[i][i] = process_noise;
    Q_[2][2] *= 10.0f;  // velocity uncertainty higher
    Q_[3][3] *= 10.0f;

    // Measurement noise R
    for (auto& row : R_) row.fill(0.0f);
    for (int i = 0; i < MEAS_DIM; ++i) R_[i][i] = measurement_noise;
}

void KalmanTracker::initialize(const BBox& bbox) {
    x_ = bbox_to_state(bbox);

    for (auto& row : P_) row.fill(0.0f);
    for (int i = 0; i < STATE_DIM; ++i) P_[i][i] = 1.0f;
    P_[2][2] = 100.0f;  // high initial velocity uncertainty
    P_[3][3] = 100.0f;

    initialized_ = true;
}

BBox KalmanTracker::predict() {
    if (!initialized_) {
        return {0, 0, 0, 0};
    }

    // x = F * x
    StateVec new_x{};
    for (int i = 0; i < STATE_DIM; ++i) {
        new_x[i] = 0.0f;
        for (int j = 0; j < STATE_DIM; ++j) {
            new_x[i] += F_[i][j] * x_[j];
        }
    }
    x_ = new_x;

    // P = F * P * F^T + Q
    auto FP  = mat_mul_66(F_, P_);
    auto FT  = mat_transpose_66(F_);
    auto FPFT = mat_mul_66(FP, FT);
    P_ = mat_add_66(FPFT, Q_);

    return state_to_bbox(x_);
}

BBox KalmanTracker::update(const BBox& measurement) {
    MeasVec z = {measurement.cx(), measurement.cy(), measurement.w(), measurement.h()};

    // Innovation y = z - H * x
    MeasVec y{};
    for (int i = 0; i < MEAS_DIM; ++i) {
        y[i] = z[i];
        for (int j = 0; j < STATE_DIM; ++j) {
            y[i] -= H_[i][j] * x_[j];
        }
    }

    // S = H * P * H^T + R
    // H*P: (4x6) * (6x6) = 4x6
    std::array<std::array<float, STATE_DIM>, MEAS_DIM> HP{};
    for (int i = 0; i < MEAS_DIM; ++i)
        for (int j = 0; j < STATE_DIM; ++j)
            for (int k = 0; k < STATE_DIM; ++k)
                HP[i][j] += H_[i][k] * P_[k][j];

    // HP * H^T: (4x6) * (6x4) = 4x4
    MeasMat S{};
    for (int i = 0; i < MEAS_DIM; ++i)
        for (int j = 0; j < MEAS_DIM; ++j)
            for (int k = 0; k < STATE_DIM; ++k)
                S[i][j] += HP[i][k] * H_[j][k];  // H^T[k][j] = H[j][k]

    for (int i = 0; i < MEAS_DIM; ++i) S[i][i] += R_[i][i];

    // K = P * H^T * S^-1
    // P * H^T: (6x6) * (6x4) = 6x4
    std::array<std::array<float, MEAS_DIM>, STATE_DIM> PHT{};
    for (int i = 0; i < STATE_DIM; ++i)
        for (int j = 0; j < MEAS_DIM; ++j)
            for (int k = 0; k < STATE_DIM; ++k)
                PHT[i][j] += P_[i][k] * H_[j][k];  // H^T[k][j] = H[j][k]

    MeasMat S_inv = invert_44(S);

    // K = PHT * S_inv: (6x4) * (4x4) = 6x4
    std::array<std::array<float, MEAS_DIM>, STATE_DIM> K{};
    for (int i = 0; i < STATE_DIM; ++i)
        for (int j = 0; j < MEAS_DIM; ++j)
            for (int k = 0; k < MEAS_DIM; ++k)
                K[i][j] += PHT[i][k] * S_inv[k][j];

    // x = x + K * y
    for (int i = 0; i < STATE_DIM; ++i)
        for (int j = 0; j < MEAS_DIM; ++j)
            x_[i] += K[i][j] * y[j];

    // P = (I - K*H) * P (simplified)
    // K*H: (6x4) * (4x6) = 6x6
    StateMat KH{};
    for (int i = 0; i < STATE_DIM; ++i)
        for (int j = 0; j < STATE_DIM; ++j)
            for (int k = 0; k < MEAS_DIM; ++k)
                KH[i][j] += K[i][k] * H_[k][j];

    StateMat IKH{};
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            IKH[i][j] = (i == j ? 1.0f : 0.0f) - KH[i][j];
        }
    }
    P_ = mat_mul_66(IKH, P_);

    return state_to_bbox(x_);
}

BBox KalmanTracker::get_state_bbox() const {
    return state_to_bbox(x_);
}

// --- static helpers ---

KalmanTracker::StateMat KalmanTracker::mat_mul_66(const StateMat& A, const StateMat& B) {
    StateMat C{};
    for (auto& row : C) row.fill(0.0f);
    for (int i = 0; i < STATE_DIM; ++i)
        for (int k = 0; k < STATE_DIM; ++k)
            if (A[i][k] != 0.0f)
                for (int j = 0; j < STATE_DIM; ++j)
                    C[i][j] += A[i][k] * B[k][j];
    return C;
}

KalmanTracker::StateMat KalmanTracker::mat_add_66(const StateMat& A, const StateMat& B) {
    StateMat C;
    for (int i = 0; i < STATE_DIM; ++i)
        for (int j = 0; j < STATE_DIM; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

KalmanTracker::StateMat KalmanTracker::mat_transpose_66(const StateMat& A) {
    StateMat T;
    for (int i = 0; i < STATE_DIM; ++i)
        for (int j = 0; j < STATE_DIM; ++j)
            T[i][j] = A[j][i];
    return T;
}

KalmanTracker::MeasMat KalmanTracker::invert_44(const MeasMat& M) {
    // Gauss-Jordan elimination for 4x4 matrix inversion
    float aug[4][8]{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) aug[i][j] = M[i][j];
        aug[i][4 + i] = 1.0f;
    }
    for (int col = 0; col < 4; ++col) {
        // Find pivot
        int pivot = col;
        for (int row = col + 1; row < 4; ++row)
            if (std::abs(aug[row][col]) > std::abs(aug[pivot][col]))
                pivot = row;
        if (pivot != col) std::swap(aug[col], aug[pivot]);

        float div = aug[col][col];
        if (std::abs(div) < 1e-10f) div = 1e-10f;  // avoid division by zero
        for (int j = 0; j < 8; ++j) aug[col][j] /= div;

        for (int row = 0; row < 4; ++row) {
            if (row == col) continue;
            float factor = aug[row][col];
            for (int j = 0; j < 8; ++j)
                aug[row][j] -= factor * aug[col][j];
        }
    }
    MeasMat inv{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            inv[i][j] = aug[i][4 + j];
    return inv;
}

BBox KalmanTracker::state_to_bbox(const StateVec& x) {
    return BBox::from_center(x[0], x[1], x[4], x[5]);
}

KalmanTracker::StateVec KalmanTracker::bbox_to_state(const BBox& b) {
    return { b.cx(), b.cy(), 0.0f, 0.0f, b.w(), b.h() };
}

} // namespace safevid
