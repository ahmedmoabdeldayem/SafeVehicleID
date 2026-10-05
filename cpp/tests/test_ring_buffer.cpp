#include <gtest/gtest.h>
#include "utils/ring_buffer.hpp"

TEST(RingBuffer, PushAndPop) {
    RingBuffer<int, 4> buf;
    EXPECT_TRUE(buf.push(42));
    auto val = buf.pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
}

TEST(RingBuffer, EmptyReturnsNullopt) {
    RingBuffer<int, 4> buf;
    EXPECT_FALSE(buf.pop().has_value());
}

TEST(RingBuffer, FullReturnsFalse) {
    RingBuffer<int, 4> buf;
    // N=4 holds N-1=3 items
    EXPECT_TRUE(buf.push(1));
    EXPECT_TRUE(buf.push(2));
    EXPECT_TRUE(buf.push(3));
    EXPECT_FALSE(buf.push(4));
}

TEST(RingBuffer, FifoOrdering) {
    RingBuffer<int, 8> buf;
    for (int i = 0; i < 5; ++i) buf.push(i);
    for (int i = 0; i < 5; ++i) {
        auto v = buf.pop();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, i);
    }
}

TEST(RingBuffer, EmptyAfterPushPop) {
    RingBuffer<int, 4> buf;
    buf.push(1);
    buf.pop();
    EXPECT_TRUE(buf.empty());
}

TEST(RingBuffer, SizeTracksElements) {
    RingBuffer<int, 8> buf;
    EXPECT_EQ(buf.size(), 0u);
    buf.push(1);
    EXPECT_EQ(buf.size(), 1u);
    buf.push(2);
    EXPECT_EQ(buf.size(), 2u);
    buf.pop();
    EXPECT_EQ(buf.size(), 1u);
}

TEST(RingBuffer, WrapAround) {
    RingBuffer<int, 4> buf;
    for (int cycle = 0; cycle < 20; ++cycle) {
        EXPECT_TRUE(buf.push(cycle));
        auto v = buf.pop();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, cycle);
    }
}

TEST(RingBuffer, CapacityIsConstant) {
    RingBuffer<int, 16> buf;
    EXPECT_EQ(buf.capacity(), 16u);
}
