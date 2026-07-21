// workqueue_integrity_test.cxx -- Queue correctness under contention.
//
// Drives the work queues at and beyond capacity to verify that the circular
// buffers, the host-side backpressure loop in waitForSpaceInCPUQueue, and the
// priority logic in getWork() (mainWorkQueue wins over cpuWorkQueue, with CPU
// work re-pushed) all behave correctly when multiple producers and consumers
// race on the same slots.

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

// Submits exactly 4096 tasks from the host. That number matches
// CPU_WORK_QUEUE_SIZE. Every host-created thread goes through sendToGPU()
// which pushes into cpuWorkQueue. By submitting exactly the queue capacity in
// a burst, this tests whether the circular buffer handles being completely
// full. If waitForSpaceInCPUQueue had a bug where it didn't properly wait for
// the GPU to drain slots, work nodes would overwrite each other.
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

// Submits 8192 tasks (2x the queue size). The queue only has 4096 slots, so
// the host MUST wait for the GPU to pop and process some work nodes before it
// can push more. Exercises the backpressure loop in waitForSpaceInCPUQueue
// which polls popCount from the GPU. If the wraparound logic had an off-by-one
// or a race, tasks would be lost or the deadlock assertion in waitForSpace
// would fire.
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

// Submits 256 simple tasks from the host (into cpuWorkQueue) AND 64 parent
// tasks that each spawn 16 children from the device (children go into
// mainWorkQueue). Both queues are used simultaneously. Expected total:
// 256 + (64 * 16) = 1280. Exercises the priority logic in getWork() where
// mainWorkQueue wins over cpuWorkQueue and CPU work gets re-pushed. If the
// re-push logic lost work nodes, or if tryPop_cpuSafe had a race with the
// host's hipStreamWriteValue64 writes, the counter would come up short.
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
