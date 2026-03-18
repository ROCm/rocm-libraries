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

static void test_burst_submission() {
    constexpr unsigned int N = 4096;
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
        assert(atomicAdd(d_counter, 0) == 4096);
    }).join();

    ::std::cerr << "test_burst_submission passed\n";
}

static void test_overflow_and_drain() {
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

    ::std::cerr << "test_overflow_and_drain passed\n";
}

static void test_mixed_host_device_queues() {
    constexpr unsigned int HOST_TASKS = 256;
    constexpr unsigned int DEVICE_CHILDREN_PER_PARENT = 16;
    constexpr unsigned int DEVICE_PARENTS = 64;
    constexpr unsigned int EXPECTED =
        HOST_TASKS + DEVICE_PARENTS * DEVICE_CHILDREN_PER_PARENT;

    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    {
        ::std::vector<hip::thread> host_threads(HOST_TASKS + DEVICE_PARENTS);

        for (unsigned int i = 0; i < HOST_TASKS; ++i) {
            host_threads[i] = hip::thread([d_counter] __device__() {
                atomicAdd(d_counter, 1);
            });
        }

        for (unsigned int i = 0; i < DEVICE_PARENTS; ++i) {
            host_threads[HOST_TASKS + i] = hip::thread([d_counter] __device__() {
                hip::thread children[16];
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

        for (auto &t : host_threads) {
            t.join();
        }
    }

    hip::thread([d_counter, EXPECTED] __device__() {
        assert(static_cast<unsigned int>(atomicAdd(d_counter, 0)) == EXPECTED);
    }).join();

    ::std::cerr << "test_mixed_host_device_queues passed\n";
}

int main() {
    test_burst_submission();
    test_overflow_and_drain();
    test_mixed_host_device_queues();
    return 0;
}
