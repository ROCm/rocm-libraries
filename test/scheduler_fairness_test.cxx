// scheduler_fairness_test.cxx -- Work distribution across vcores.
//
// Verifies that the scheduler spreads work across many vcores rather than
// funneling everything through a small number of them, and that multiple
// vcores execute work concurrently rather than being serialized through a
// single vcore by the activeVcoreCount gating in threading_main.

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

// Submits 4 * hardware_concurrency() tasks. Each task records which vcore it
// ran on via atomicExch(&d_seen[blockIdx.x], 1) and then spins for 10000
// iterations to stay alive long enough for other vcores to pick up work.
// After all tasks complete, counts how many distinct vcores were used and
// asserts > 50%.
//
// Proves work spreads across the GPU and isn't funneling through a small
// number of vcores. If getWork() had a bias where only a few blocks ever
// successfully popped from the queue, the count would be much lower. Not all
// vcores are expected to be used (tasks complete quickly, so some vcores
// never get a chance), but the majority should participate.
static void test_vcore_distribution() {
    const unsigned int hwc = hip::thread::hardware_concurrency();
    const unsigned int N = hwc * 4;

    uint32_t *d_seen;
    CHECK(hipMalloc(&d_seen, hwc * sizeof(uint32_t)));
    CHECK(hipMemset(d_seen, 0, hwc * sizeof(uint32_t)));

    {
        ::std::vector<hip::thread> threads(N);
        for (unsigned int i = 0; i < N; ++i) {
            threads[i] = hip::thread([d_seen] __device__() {
                atomicExch(&d_seen[blockIdx.x], 1);
                for (volatile int spin = 0; spin < 10000; ++spin) {}
            });
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    uint32_t *d_unique_count;
    CHECK(hipMalloc(&d_unique_count, sizeof(uint32_t)));
    CHECK(hipMemset(d_unique_count, 0, sizeof(uint32_t)));

    hip::thread([d_seen, d_unique_count, hwc] __device__() {
        uint32_t count = 0;
        for (unsigned int i = 0; i < hwc; ++i) {
            if (d_seen[i] != 0)
                ++count;
        }
        atomicExch(d_unique_count, count);

        assert(count > hwc / 2 && "Less than 50% of vcores were utilized");
    }).join();

    uint32_t h_unique_count = 0;
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(&h_unique_count, d_unique_count, sizeof(uint32_t), hipMemcpyDeviceToHost));
    ::std::cerr << "test_vcore_distribution: " << h_unique_count << " / " << hwc
                << " vcores utilized\n";
}

// 256 tasks, each atomically increments an active counter on entry, updates a
// high-water mark via atomicCAS, spins for 10000 iterations, then decrements
// the active counter on exit. Asserts that the peak active count is > 1 and
// that the final active count is 0.
//
// Proves the scheduler runs work in parallel across multiple vcores
// simultaneously, not serializing everything through one. If threading_main
// had a bug where activeVcoreCount gating prevented concurrent execution, the
// peak would be 1. The final-count-is-0 check proves every task that
// incremented also decremented -- no task was abandoned mid-execution.
static void test_multiple_vcores_active() {
    int *d_max_active;
    CHECK(hipMalloc(&d_max_active, sizeof(int)));
    CHECK(hipMemset(d_max_active, 0, sizeof(int)));

    int *d_active;
    CHECK(hipMalloc(&d_active, sizeof(int)));
    CHECK(hipMemset(d_active, 0, sizeof(int)));

    constexpr unsigned int N = 256;

    {
        ::std::vector<hip::thread> threads(N);
        for (unsigned int i = 0; i < N; ++i) {
            threads[i] = hip::thread([d_active, d_max_active] __device__() {
                int cur = atomicAdd(d_active, 1) + 1;

                int prev_max = atomicAdd(d_max_active, 0);
                while (cur > prev_max) {
                    if (atomicCAS(d_max_active, prev_max, cur) == prev_max)
                        break;
                    prev_max = atomicAdd(d_max_active, 0);
                }

                for (volatile int spin = 0; spin < 10000; ++spin) {}

                atomicSub(d_active, 1);
            });
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    hip::thread([d_max_active, d_active] __device__() {
        assert(atomicAdd(d_active, 0) == 0);
        assert(atomicAdd(d_max_active, 0) > 1 && "Only 1 vcore was ever active");
    }).join();

    int h_max_active = 0;
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(&h_max_active, d_max_active, sizeof(int), hipMemcpyDeviceToHost));
    ::std::cerr << "test_multiple_vcores_active: peak " << h_max_active
                << " vcores simultaneously active\n";
}

int main() {
    test_vcore_distribution();
    test_multiple_vcores_active();
    return 0;
}
