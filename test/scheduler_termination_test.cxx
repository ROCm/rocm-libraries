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

static void test_single_task_terminates() {
    int *d_flag;
    CHECK(hipMalloc(&d_flag, sizeof(int)));
    CHECK(hipMemset(d_flag, 0, sizeof(int)));

    hip::thread([d_flag] __device__() {
        atomicExch(d_flag, 1);
    }).join();

    CHECK(hipDeviceSynchronize());

    int h_flag = 0;
    CHECK(hipMemcpy(&h_flag, d_flag, sizeof(int), hipMemcpyDeviceToHost));
    assert(h_flag == 1);

    ::std::cerr << "test_single_task_terminates passed\n";
}

static void test_multiple_rounds() {
    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    constexpr int NUM_ROUNDS = 5;
    constexpr int TASKS_PER_ROUND = 512;

    for (int round = 0; round < NUM_ROUNDS; ++round) {
        {
            ::std::vector<hip::thread> threads(TASKS_PER_ROUND);
            for (int i = 0; i < TASKS_PER_ROUND; ++i) {
                threads[static_cast<size_t>(i)] = hip::thread([d_counter] __device__() {
                    atomicAdd(d_counter, 1);
                });
            }
            for (auto &t : threads) {
                t.join();
            }
        }
    }

    constexpr int EXPECTED = NUM_ROUNDS * TASKS_PER_ROUND;

    hip::thread([d_counter] __device__() {
        assert(atomicAdd(d_counter, 0) == 2560);
    }).join();

    CHECK(hipDeviceSynchronize());

    int h_counter = 0;
    CHECK(hipMemcpy(&h_counter, d_counter, sizeof(int), hipMemcpyDeviceToHost));
    assert(h_counter == EXPECTED);

    ::std::cerr << "test_multiple_rounds passed (" << NUM_ROUNDS << " rounds x "
                << TASKS_PER_ROUND << " tasks = " << h_counter << ")\n";
}

static void test_rapid_create_join() {
    constexpr int ITERATIONS = 1000;
    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    for (int i = 0; i < ITERATIONS; ++i) {
        hip::thread([d_counter] __device__() {
            atomicAdd(d_counter, 1);
        }).join();
    }

    hip::thread([d_counter] __device__() {
        assert(atomicAdd(d_counter, 0) == 1000);
    }).join();

    ::std::cerr << "test_rapid_create_join passed (" << ITERATIONS << " cycles)\n";
}

int main() {
    test_single_task_terminates();
    test_multiple_rounds();
    test_rapid_create_join();
    return 0;
}
