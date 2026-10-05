#include <gtest/gtest.h>
#include <set>
#include "tracking/sort_tracker.hpp"

using namespace safevid;

static BBox make_bbox(float cx, float cy, float w = 100.f, float h = 60.f) {
    return BBox{cx - w/2, cy - h/2, cx + w/2, cy + h/2};
}

TEST(SORTTracker, NewDetectionCreatesTentativeTrack) {
    SORTConfig cfg;
    cfg.min_hits_to_confirm = 3;
    SORTTracker tracker(cfg);

    auto confirmed = tracker.update({make_bbox(200.f, 200.f)});
    EXPECT_EQ(confirmed.size(), 0u);
    EXPECT_EQ(tracker.all_tracks().size(), 1u);
    EXPECT_EQ(tracker.all_tracks()[0].state, TrackState::Tentative);
}

TEST(SORTTracker, TrackConfirmedAfterMinHits) {
    SORTConfig cfg;
    cfg.min_hits_to_confirm = 3;
    SORTTracker tracker(cfg);

    std::vector<const Track*> confirmed;
    for (int i = 0; i < 3; ++i)
        confirmed = tracker.update({make_bbox(200.f, 200.f)});

    EXPECT_EQ(confirmed.size(), 1u);
    EXPECT_EQ(confirmed[0]->state, TrackState::Confirmed);
}

TEST(SORTTracker, TrackDeletedAfterMaxDisappeared) {
    SORTConfig cfg;
    cfg.min_hits_to_confirm = 1;
    cfg.max_disappeared = 3;
    SORTTracker tracker(cfg);

    tracker.update({make_bbox(200.f, 200.f)});

    for (int i = 0; i < 4; ++i)
        tracker.update({});

    EXPECT_EQ(tracker.all_tracks().size(), 0u);
}

TEST(SORTTracker, UniqueTrackIds) {
    SORTConfig cfg;
    cfg.min_hits_to_confirm = 1;
    SORTTracker tracker(cfg);

    tracker.update({make_bbox(100.f, 100.f), make_bbox(500.f, 100.f)});
    auto& tracks = tracker.all_tracks();
    ASSERT_EQ(tracks.size(), 2u);
    EXPECT_NE(tracks[0].track_id, tracks[1].track_id);
}

TEST(SORTTracker, MatchingPersistsTrackIds) {
    SORTConfig cfg;
    cfg.min_hits_to_confirm = 1;
    cfg.iou_threshold = 0.1f;
    SORTTracker tracker(cfg);

    tracker.update({make_bbox(100.f, 200.f), make_bbox(400.f, 200.f)});
    auto confirmed = tracker.update({make_bbox(110.f, 200.f), make_bbox(410.f, 200.f)});

    ASSERT_EQ(confirmed.size(), 2u);
    std::set<int> ids = {confirmed[0]->track_id, confirmed[1]->track_id};
    EXPECT_EQ(ids, (std::set<int>{1, 2}));
}

TEST(SORTTracker, NoMatchBelowIoUThresholdCreatesNewTrack) {
    SORTConfig cfg;
    cfg.min_hits_to_confirm = 1;
    cfg.iou_threshold = 0.3f;
    SORTTracker tracker(cfg);

    tracker.update({make_bbox(100.f, 100.f)});
    tracker.update({make_bbox(800.f, 800.f)});

    EXPECT_EQ(tracker.all_tracks().size(), 2u);
}

TEST(SORTTracker, EmptyFrameDoesNotCrash) {
    SORTTracker tracker;
    auto result = tracker.update({});
    EXPECT_EQ(result.size(), 0u);
}

TEST(SORTTracker, ResetClearsAllState) {
    SORTConfig cfg;
    cfg.min_hits_to_confirm = 1;
    SORTTracker tracker(cfg);

    tracker.update({make_bbox(100.f, 100.f)});
    tracker.reset();

    EXPECT_EQ(tracker.all_tracks().size(), 0u);
}
