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

static void test_maximum_throughput() {
    constexpr unsigned int N = 1 << 14;
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
        assert(atomicAdd(d_counter, 0) == (1 << 14));
    }).join();

    ::std::cerr << "test_maximum_throughput passed (" << N << " tasks)\n";
}

static void test_mixed_storm() {
    constexpr unsigned int HOST_TASKS = 1024;
    constexpr unsigned int DEVICE_SPAWNERS = 256;
    constexpr unsigned int CHILDREN_PER_SPAWNER = 4;
    constexpr unsigned int EXPECTED =
        HOST_TASKS + DEVICE_SPAWNERS + DEVICE_SPAWNERS * CHILDREN_PER_SPAWNER;

    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    {
        ::std::vector<hip::thread> threads(HOST_TASKS + DEVICE_SPAWNERS);

        for (unsigned int i = 0; i < HOST_TASKS; ++i) {
            threads[i] = hip::thread([d_counter] __device__() {
                atomicAdd(d_counter, 1);
            });
        }

        for (unsigned int i = 0; i < DEVICE_SPAWNERS; ++i) {
            threads[HOST_TASKS + i] = hip::thread([d_counter] __device__() {
                atomicAdd(d_counter, 1);
                hip::thread children[4];
                for (auto &c : children) {
                    c = hip::thread([d_counter] __device__() {
                        atomicAdd(d_counter, 1);
                    });
                }
                for (auto &c : children) {
                    c.join();
                }
            });
        }

        for (auto &t : threads) {
            t.join();
        }
    }

    hip::thread([d_counter, EXPECTED] __device__() {
        assert(static_cast<unsigned int>(atomicAdd(d_counter, 0)) == EXPECTED);
    }).join();

    ::std::cerr << "test_mixed_storm passed (" << EXPECTED << " total tasks)\n";
}

static void test_rapid_single_thread_cycles() {
    constexpr int ITERATIONS = 2000;
    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    for (int i = 0; i < ITERATIONS; ++i) {
        hip::thread([d_counter] __device__() {
            atomicAdd(d_counter, 1);
        }).join();
    }

    hip::thread([d_counter] __device__() {
        assert(atomicAdd(d_counter, 0) == 2000);
    }).join();

    ::std::cerr << "test_rapid_single_thread_cycles passed (" << ITERATIONS << " cycles)\n";
}

int main() {
    test_maximum_throughput();
    test_mixed_storm();
    test_rapid_single_thread_cycles();
    return 0;
}
