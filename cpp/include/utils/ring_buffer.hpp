#pragma once

/**
 * Lock-free single-producer single-consumer ring buffer.
 *
 * Designed for the camera-capture → detection-processing pipeline:
 *   - Camera thread (producer): pushes frames without blocking
 *   - Detection thread (consumer): pops frames without blocking
 *
 * Uses std::atomic for head/tail indices — no mutex required.
 * Cache-line aligned to prevent false sharing between producer and consumer.
 *
 * Template parameters:
 *   T    — element type (e.g., cv::Mat for video frames)
 *   N    — capacity (must be power of 2 for efficient modulo)
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>

template <typename T, std::size_t N>
class RingBuffer {
    static_assert((N & (N - 1)) == 0, "RingBuffer capacity N must be a power of 2");

public:
    RingBuffer() : head_(0), tail_(0) {}

    /**
     * Push an element. Returns false if the buffer is full.
     * Called from the producer thread only.
     */
    bool push(T item) {
        const std::size_t current_head = head_.load(std::memory_order_relaxed);
        const std::size_t next_head = (current_head + 1) & MASK;

        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;  // buffer full
        }

        buffer_[current_head] = std::move(item);
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    /**
     * Pop an element. Returns std::nullopt if the buffer is empty.
     * Called from the consumer thread only.
     */
    std::optional<T> pop() {
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);

        if (current_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // buffer empty
        }

        T item = std::move(buffer_[current_tail]);
        tail_.store((current_tail + 1) & MASK, std::memory_order_release);
        return item;
    }

    /**
     * True if no elements are available to pop.
     * Note: result can be stale by the time caller acts on it.
     */
    bool empty() const {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_acquire);
    }

    /**
     * Number of elements currently in the buffer.
     */
    std::size_t size() const {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        return (h - t) & MASK;
    }

    static constexpr std::size_t capacity() { return N; }

private:
    static constexpr std::size_t MASK = N - 1;

    std::array<T, N> buffer_;

    // Align to separate cache lines to prevent false sharing
    alignas(64) std::atomic<std::size_t> head_;
    alignas(64) std::atomic<std::size_t> tail_;
};
