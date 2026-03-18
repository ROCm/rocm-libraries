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
