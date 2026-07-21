// pseudo_yield_test.cxx -- Cooperative yield correctness.
//
// Verifies that hip::this_thread::pseudo_yield() preserves the caller's local
// state across the save/restore path in invokeNext(true) -> makeCurrent(yielding)
// -> ... -> waiting->makeCurrent(false), that yielding actually enables forward
// progress, and that yielding inside lock-spin loops (used by pseudo_mutex)
// does not interfere with the lock's atomics.

#include "hip/thread"
#include "hip/pseudo_mutex"
#include "hip/mutex"
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

// Sets two local variables, calls pseudo_yield(), checks they're still the
// same. Today this is mostly a "compiler keeps locals live across a function
// call" check rather than a real scheduler check: invokeNext(true) just calls
// another worknode's wrapper_fn synchronously on the same wave, so there is
// no save/restore to break -- the caller's locals stay live in registers
// like across any normal call.
//
// Kept as a cheap regression guard for if pseudo_yield is ever switched to a
// real context-switch implementation (e.g. __syncthreads-based, or the
// wider-width).
__device__ void test_yield_basic_device() {
    int local_val = 42;
    volatile int sentinel = 0xDEAD;

    hip::this_thread::pseudo_yield();

    assert(local_val == 42);
    assert(sentinel == 0xDEAD);
}

static void test_yield_basic() {
    hip::thread([] __device__() { test_yield_basic_device(); }).join();
    ::std::cerr << "test_yield_basic passed\n";
}

// A parent thread spawns a child that sets flag = 1, then the parent
// spin-loops on flag calling pseudo_yield() each iteration. Proves that
// pseudo_yield() actually enables forward progress -- the child task
// eventually runs (either during a yield on the parent's vcore, or on a
// different vcore) and the parent sees the flag change. If pseudo_yield
// didn't pick up available work and no other vcore grabbed it either, the
// parent would deadlock.
__device__ void test_yield_forward_progress_device() {
    static __device__ int flag = 0;
    atomicExch(&flag, 0);

    auto child = hip::thread([] __device__() {
        atomicExch(&flag, 1);
    });

    while (atomicAdd(&flag, 0) == 0) {
        hip::this_thread::pseudo_yield();
    }

    child.join();
    assert(atomicAdd(&flag, 0) == 1);
}

static void test_yield_forward_progress() {
    hip::thread([] __device__() { test_yield_forward_progress_device(); }).join();
    ::std::cerr << "test_yield_forward_progress passed\n";
}

__device__ void contention_work(int *d_counter) {
    static __device__ hip::pseudo_mutex mtx;
    hip::unique_lock guard(mtx);
    int cur = atomicAdd(d_counter, 0);
    atomicExch(d_counter, cur + 1);
}

// 4096 threads all contend on a single pseudo_mutex. Inside the critical
// section, each thread does a read-then-write on the counter (deliberately
// NOT a single atomicAdd -- it reads, then writes cur + 1). If the mutex
// isn't providing mutual exclusion, two threads could read the same value
// and both write cur + 1, losing an increment.
//
// The key connection to pseudo_yield: pseudo_mutex::lock() calls
// hip::this_thread::pseudo_yield() after every 65536 failed lock attempts.
// With 4096 threads contending, pseudo_yield gets called hundreds of times
// under real contention. If yielding during a lock spin corrupted state, or
// if the yield/resume path interfered with the lock's atomics, the final
// counter wouldn't be 4096.
static void test_pseudo_mutex_contention() {
    constexpr unsigned int N = 1 << 12;
    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    {
        ::std::vector<hip::thread> threads(N);
        for (unsigned int i = 0; i < N; ++i) {
            threads[i] = hip::thread([d_counter] __device__() {
                contention_work(d_counter);
            });
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    hip::thread([d_counter] __device__() {
        assert(atomicAdd(d_counter, 0) == (1 << 12));
    }).join();

    ::std::cerr << "test_pseudo_mutex_contention passed\n";
}

int main() {
    test_yield_basic();
    test_yield_forward_progress();
    test_pseudo_mutex_contention();
    return 0;
}
