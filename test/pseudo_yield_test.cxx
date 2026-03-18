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
