#include <gtest/gtest.h>
#include "tracking/kalman_tracker.hpp"

using namespace safevid;

static BBox make_bbox(float x1, float y1, float x2, float y2) {
    return BBox{x1, y1, x2, y2};
}

TEST(KalmanTracker, InitializationSetsCorrectCenter) {
    KalmanTracker kf;
    BBox bbox = make_bbox(100.f, 200.f, 200.f, 300.f);
    kf.initialize(bbox);

    EXPECT_TRUE(kf.is_initialized());
    BBox state = kf.get_state_bbox();
    EXPECT_NEAR(state.cx(), 150.f, 1.f);
    EXPECT_NEAR(state.cy(), 250.f, 1.f);
}

TEST(KalmanTracker, InitialVelocityIsZero) {
    KalmanTracker kf;
    kf.initialize(make_bbox(0.f, 0.f, 100.f, 100.f));
    EXPECT_NEAR(kf.vx(), 0.f, 1e-4f);
    EXPECT_NEAR(kf.vy(), 0.f, 1e-4f);
}

TEST(KalmanTracker, PredictStationaryVehicleStaysInPlace) {
    KalmanTracker kf;
    kf.initialize(make_bbox(100.f, 100.f, 200.f, 200.f));
    BBox predicted = kf.predict();

    EXPECT_NEAR(predicted.cx(), 150.f, 2.f);
    EXPECT_NEAR(predicted.cy(), 150.f, 2.f);
}

TEST(KalmanTracker, UpdateCorrectsBboxTowardMeasurement) {
    KalmanTracker kf;
    kf.initialize(make_bbox(0.f, 0.f, 100.f, 100.f));
    kf.predict();

    BBox before = kf.get_state_bbox();
    // Measurement shifted right by 50px
    kf.update(make_bbox(50.f, 0.f, 150.f, 100.f));
    BBox after = kf.get_state_bbox();

    EXPECT_GT(after.cx(), before.cx());
}

TEST(KalmanTracker, IsInitializedFalseByDefault) {
    KalmanTracker kf;
    EXPECT_FALSE(kf.is_initialized());
}

TEST(KalmanTracker, VelocityConvergesAfterRepeatedUpdates) {
    KalmanTracker kf(0.01f, 0.01f);  // trust measurements heavily
    kf.initialize(make_bbox(100.f, 100.f, 200.f, 200.f));

    for (int i = 0; i < 15; ++i) {
        float offset = 20.f * (i + 1);
        kf.update(make_bbox(100.f + offset, 100.f, 200.f + offset, 200.f));
        kf.predict();
    }

    EXPECT_GT(kf.vx(), 5.f);
    EXPECT_NEAR(kf.vy(), 0.f, 5.f);
}

TEST(KalmanTracker, BBoxRoundtrip) {
    KalmanTracker kf;
    BBox original = make_bbox(50.f, 75.f, 150.f, 175.f);
    kf.initialize(original);
    BBox result = kf.get_state_bbox();

    EXPECT_NEAR(result.x1, original.x1, 0.01f);
    EXPECT_NEAR(result.y1, original.y1, 0.01f);
    EXPECT_NEAR(result.x2, original.x2, 0.01f);
    EXPECT_NEAR(result.y2, original.y2, 0.01f);
}

TEST(BBox, IoUIdenticalBoxes) {
    BBox b = make_bbox(0.f, 0.f, 100.f, 100.f);
    EXPECT_NEAR(b.iou(b), 1.f, 1e-5f);
}

TEST(BBox, IoUNoOverlap) {
    BBox b1 = make_bbox(0.f,   0.f,  50.f,  50.f);
    BBox b2 = make_bbox(100.f, 100.f, 150.f, 150.f);
    EXPECT_NEAR(b1.iou(b2), 0.f, 1e-5f);
}

TEST(BBox, IoUPartialOverlap) {
    BBox b1 = make_bbox(0.f, 0.f, 100.f, 100.f);
    BBox b2 = make_bbox(50.f, 0.f, 150.f, 100.f);
    float expected = 5000.f / 15000.f;
    EXPECT_NEAR(b1.iou(b2), expected, 1e-4f);
}
