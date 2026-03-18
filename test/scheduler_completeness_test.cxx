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
