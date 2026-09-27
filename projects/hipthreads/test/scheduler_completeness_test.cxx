// scheduler_completeness_test.cxx -- Liveness and exactly-once execution.
//
// Verifies that every task the scheduler picks up from the work queue actually
// runs, runs exactly once, and that the width parameter path in the wrapper()
// function in worknode.h (where `if (threadIdx.x < width)` controls which
// lanes execute the callable) is exercised correctly for full-warp tasks.

#include "hip/thread"
#include "hip/hip_runtime.h"
#include <cassert>
#include <iostream>
#include <vector>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

// Creates 8192 hip::threads from the host, each doing atomicAdd(d_counter, 1).
// After all joined, a verifier thread checks d_counter == 8192. Proves
// liveness: every single task the scheduler picked up from the work queue
// actually ran and completed. If threading_main had a bug where getWork()
// sometimes returned null when work existed, or invokeNext() silently dropped
// a work node, the counter would be less than 8192.
static void test_fanout() {
    constexpr unsigned int N = 8192;
    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    {
        ::std::vector<hip::thread> threads(N);
        for (unsigned int i = 0; i < N; ++i) {
            threads[i] = hip::thread([d_counter] __device__() {
                atomicAdd(d_counter, 1);
            });
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    hip::thread([d_counter] __device__() {
        assert(atomicAdd(d_counter, 0) == 8192);
    }).join();

    ::std::cerr << "test_fanout passed\n";
}

// Same 8192 threads, but each writes to its own unique slot: thread i does
// atomicAdd(&d_slots[i], 1). The verifier checks every slot equals exactly 1.
// Proves exactly-once execution: no task was skipped (slot would be 0) and no
// task ran twice (slot would be 2). Catches a different class of bug than the
// counter test -- if WorkQueue::tryPop had a race where two vcores both popped
// the same work node, the counter test might still pass (counter still gets
// incremented) but this test would fail (one slot would be 2, another 0).
static void test_unique_ids() {
    constexpr unsigned int N = 8192;
    uint32_t *d_slots;
    CHECK(hipMalloc(&d_slots, N * sizeof(uint32_t)));
    CHECK(hipMemset(d_slots, 0, N * sizeof(uint32_t)));

    {
        ::std::vector<hip::thread> threads(N);
        for (unsigned int i = 0; i < N; ++i) {
            threads[i] = hip::thread([d_slots, i] __device__() {
                atomicAdd(&d_slots[i], 1);
            });
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    hip::thread([d_slots] __device__() {
        for (unsigned int i = 0; i < 8192; ++i) {
            assert(d_slots[i] == 1);
        }
    }).join();

    ::std::cerr << "test_unique_ids passed\n";
}

// 4096 threads created with hip::thread::max_width() (width=32, full warp).
// Inside each, only fiber 0 (get_fiber_id() == 0) increments the counter.
// Verifier checks d_counter == 4096. Proves the scheduler correctly handles
// full-warp tasks, not just single-lane tasks. Exercises the width parameter
// path in the wrapper() function in worknode.h where `if (threadIdx.x < width)`
// controls which lanes execute the callable.
static void test_varying_widths() {
    constexpr unsigned int N = 4096;
    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    {
        ::std::vector<hip::thread> threads(N);
        for (unsigned int i = 0; i < N; ++i) {
            threads[i] = hip::thread(hip::thread::max_width(), [d_counter] __device__() {
                if (hip::this_thread::get_fiber_id() == 0)
                    atomicAdd(d_counter, 1);
            });
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    hip::thread([d_counter] __device__() {
        assert(atomicAdd(d_counter, 0) == 4096);
    }).join();

    ::std::cerr << "test_varying_widths passed\n";
}

int main() {
    test_fanout();
    test_unique_ids();
    test_varying_widths();
    return 0;
}
