/**
 * SPSC (Single-Producer Single-Consumer) lock-free ring buffer.
 *
 * Rust's ownership model makes the safety guarantees here explicit and
 * compile-time enforced — the borrow checker prevents us from having
 * both a mutable producer handle and a mutable consumer handle on the
 * same buffer simultaneously.
 *
 * In C++ this is convention. In Rust it is a type error.
 *
 * This is the key interview point: Rust's ownership system doesn't just
 * make code safer — it makes the *correct* concurrent usage pattern
 * impossible to violate accidentally.
 */

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Internal shared state between producer and consumer.
struct Inner<T, const N: usize> {
    buffer: [UnsafeCell<MaybeUninit<T>>; N],
    head: AtomicUsize,  // producer writes here
    tail: AtomicUsize,  // consumer reads here
}

// Safety: we enforce SPSC discipline at the type level (separate Producer/Consumer handles).
// Only one writer (head) and one reader (tail) ever exist at a time.
unsafe impl<T: Send, const N: usize> Send for Inner<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for Inner<T, N> {}

/// Producer handle — can only be held by one thread.
/// Ownership enforces: only one producer can exist at a time.
pub struct Producer<T, const N: usize> {
    inner: Arc<Inner<T, N>>,
}

/// Consumer handle — can only be held by one thread.
/// Ownership enforces: only one consumer can exist at a time.
pub struct Consumer<T, const N: usize> {
    inner: Arc<Inner<T, N>>,
}

const MASK_FOR<const N: usize>: usize = N - 1;

/// Create a matched producer/consumer pair for a ring buffer of capacity N.
///
/// N must be a power of 2. This is checked at compile time via the const assertion.
///
/// # Example
/// ```rust
/// let (mut tx, mut rx) = ring_buffer::<u32, 16>();
/// tx.push(42);
/// assert_eq!(rx.pop(), Some(42));
/// ```
pub fn ring_buffer<T, const N: usize>() -> (Producer<T, N>, Consumer<T, N>) {
    // Compile-time check: N must be power of 2
    const { assert!(N > 0 && (N & (N - 1)) == 0, "RingBuffer capacity must be a power of 2") };

    // Safety: MaybeUninit arrays are valid in any bit pattern.
    let inner = Arc::new(Inner {
        buffer: unsafe {
            MaybeUninit::<[UnsafeCell<MaybeUninit<T>>; N]>::uninit().assume_init()
        },
        head: AtomicUsize::new(0),
        tail: AtomicUsize::new(0),
    });

    let producer = Producer { inner: Arc::clone(&inner) };
    let consumer = Consumer { inner };
    (producer, consumer)
}

impl<T, const N: usize> Producer<T, N> {
    /// Push an item. Returns `Err(item)` if the buffer is full.
    pub fn push(&mut self, item: T) -> Result<(), T> {
        let head = self.inner.head.load(Ordering::Relaxed);
        let next_head = (head + 1) & (N - 1);

        if next_head == self.inner.tail.load(Ordering::Acquire) {
            return Err(item);  // buffer full
        }

        // Safety: head is only written by this producer.
        // No other thread touches buffer[head] until tail advances past it.
        unsafe {
            (*self.inner.buffer[head].get()).write(item);
        }

        self.inner.head.store(next_head, Ordering::Release);
        Ok(())
    }

    pub fn is_full(&self) -> bool {
        let head = self.inner.head.load(Ordering::Relaxed);
        let next_head = (head + 1) & (N - 1);
        next_head == self.inner.tail.load(Ordering::Acquire)
    }
}

impl<T, const N: usize> Consumer<T, N> {
    /// Pop an item. Returns `None` if the buffer is empty.
    pub fn pop(&mut self) -> Option<T> {
        let tail = self.inner.tail.load(Ordering::Relaxed);

        if tail == self.inner.head.load(Ordering::Acquire) {
            return None;  // buffer empty
        }

        // Safety: tail is only written by this consumer.
        // The producer has already written and released buffer[tail].
        let item = unsafe {
            (*self.inner.buffer[tail].get()).assume_init_read()
        };

        self.inner.tail.store((tail + 1) & (N - 1), Ordering::Release);
        Some(item)
    }

    pub fn is_empty(&self) -> bool {
        self.inner.tail.load(Ordering::Relaxed) == self.inner.head.load(Ordering::Acquire)
    }

    pub fn len(&self) -> usize {
        let h = self.inner.head.load(Ordering::Acquire);
        let t = self.inner.tail.load(Ordering::Relaxed);
        (h.wrapping_sub(t)) & (N - 1)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn push_and_pop_single_item() {
        let (mut tx, mut rx) = ring_buffer::<u32, 4>();
        assert!(tx.push(42).is_ok());
        assert_eq!(rx.pop(), Some(42));
    }

    #[test]
    fn empty_buffer_returns_none() {
        let (_tx, mut rx) = ring_buffer::<u32, 4>();
        assert_eq!(rx.pop(), None);
    }

    #[test]
    fn full_buffer_returns_err() {
        let (mut tx, _rx) = ring_buffer::<u32, 4>();
        // Capacity N=4, but ring buffer holds N-1=3 items max
        assert!(tx.push(1).is_ok());
        assert!(tx.push(2).is_ok());
        assert!(tx.push(3).is_ok());
        assert!(tx.push(4).is_err());  // full
    }

    #[test]
    fn fifo_ordering() {
        let (mut tx, mut rx) = ring_buffer::<u32, 8>();
        for i in 0u32..5 {
            tx.push(i).unwrap();
        }
        for i in 0u32..5 {
            assert_eq!(rx.pop(), Some(i));
        }
    }

    #[test]
    fn concurrent_producer_consumer() {
        // Verify correctness under actual SPSC threading
        let (mut tx, mut rx) = ring_buffer::<u64, 64>();
        const ITEMS: u64 = 10_000;

        let producer = thread::spawn(move || {
            let mut sent = 0u64;
            while sent < ITEMS {
                if tx.push(sent).is_ok() {
                    sent += 1;
                } else {
                    thread::yield_now();
                }
            }
        });

        let consumer = thread::spawn(move || {
            let mut received = 0u64;
            let mut sum = 0u64;
            while received < ITEMS {
                if let Some(val) = rx.pop() {
                    assert_eq!(val, received, "Expected {received} but got {val} — ordering violated");
                    sum += val;
                    received += 1;
                } else {
                    thread::yield_now();
                }
            }
            sum
        });

        producer.join().unwrap();
        let sum = consumer.join().unwrap();
        let expected_sum: u64 = (0..ITEMS).sum();
        assert_eq!(sum, expected_sum);
    }

    #[test]
    fn wrap_around_works() {
        // Push and pop in a cycle to force index wrap-around
        let (mut tx, mut rx) = ring_buffer::<u32, 4>();
        for cycle in 0..100u32 {
            tx.push(cycle).unwrap();
            assert_eq!(rx.pop(), Some(cycle));
        }
    }
}
