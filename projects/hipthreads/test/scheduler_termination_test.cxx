// scheduler_termination_test.cxx -- Persistent kernel shutdown and relaunch.
//
// Verifies the full lifecycle of the persistent scheduler kernel: the kernel
// exits cleanly when shouldKeepPollingForWork() observes no more work, and a
// subsequent hip::thread on the host correctly re-runs prepDeviceForWork()
// to relaunch it without losing or duplicating work across stop/start cycles.

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

// Creates one hip::thread, joins it, then does hipDeviceSynchronize()
// followed by hipMemcpy to read the result from the host. The fact that
// hipDeviceSynchronize() returns at all proves the persistent kernel exited.
// If shouldKeepPollingForWork() had a bug where it kept returning true after
// all work was done, hipDeviceSynchronize() would hang forever. Tests the
// simplest termination path: gpuThreadFromHost_counter goes 0 -> 1 (create),
// thread joins, destructor runs, counter goes 1 -> 0,
// notifyDeviceThereMightNotBeAnyMoreWork() fires, kernel sees no more work,
// exits.
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

// 5 separate batches of 512 tasks. Each batch creates threads in a scoped
// block, joins them all, then the block ends and all destructors run.
// Between rounds, the full shutdown/relaunch cycle happens:
//   - Round ends: last destructor decrements gpuThreadFromHost_counter to 0,
//     calls notifyDeviceThereMightNotBeAnyMoreWork().
//   - Kernel sees cpuWorkQueue.pushCount updated, no active vcores, no
//     pending work -- exits.
//   - Next round starts: first hip::thread constructor calls
//     prepDeviceForWork(), counter goes 0 -> 1, kernel is relaunched.
// If the relaunch path had a bug (e.g., cpuWorkQueue.pushCount wasn't reset
// to -1U correctly, or stale state from the previous round confused the new
// kernel instance), the counter would be wrong. Getting 2560 (5 * 512)
// proves all 5 stop/start cycles worked cleanly.
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

// 1000 iterations of: create one thread, immediately join it. Each iteration
// is a full lifecycle: prepDeviceForWork -> task runs -> join -> destructor
// -> possibly notifyDeviceThereMightNotBeAnyMoreWork -> kernel might stop
// -> next iteration relaunches. Whether the kernel actually stops between
// iterations depends on timing -- sometimes the next constructor runs before
// the kernel has exited, so gpuThreadFromHost_counter never hits 0. Other
// times it does hit 0 and the kernel must be relaunched. Both paths are
// exercised across 1000 iterations. Getting counter == 1000 proves neither
// path loses work.
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
