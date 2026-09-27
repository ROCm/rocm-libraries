// scheduler_stress_test.cxx -- Race condition hunting under extreme scale.
//
// Pushes the scheduler past the work queue's capacity, mixes host- and
// device-spawned tasks across both queues simultaneously, and rapidly cycles
// the persistent kernel start/stop path. Any rare ordering bug in tryPop /
// tryPop_cpuSafe, the cpuWorkQueue/mainWorkQueue priority logic, or the
// notify/prep handshake would surface here as a counter mismatch.

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

// 16384 tasks (4x the queue size), all from the host, each doing one
// atomicAdd. The host-side waitForSpaceInCPUQueue loop has to cycle at least
// 4 times, waiting for the GPU to drain slots before pushing more. With
// 16384 tasks all racing through getWork() and invokeNext(), this maximizes
// contention on popCount and pushCount atomics. Any rare race condition in
// tryPop or tryPop_cpuSafe -- like two vcores both succeeding on the same
// slot, or a work node getting overwritten before it's read -- would show up
// as a counter mismatch.
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

// The most chaotic test across all 7 files. Simultaneously submits 1024
// simple host tasks (into cpuWorkQueue) AND 256 spawner tasks that each
// create 4 children from the device (into mainWorkQueue). Spawner tasks also
// increment the counter themselves. Expected total:
// 1024 + 256 + (256 * 4) = 2304. Combines everything: host-to-device path,
// device-to-device path, queue priority logic in getWork(), device-side
// join() spin-waiting, and the cpuWorkQueue/mainWorkQueue interaction all
// happening concurrently. If any of these paths had a subtle ordering bug
// that only manifests under heavy concurrent access, this is where it would
// show up.
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

// 2000 iterations of create-one-thread-and-immediately-join. Each iteration
// hammers the host-side path: prepDeviceForWork (increment counter, possibly
// relaunch kernel), sendToGPU (allocate work node, memcpy to device,
// hipStreamWriteValue64 into queue slot), join (poll link_to_self via
// hipMemcpyAsync), destructor (decrement counter, possibly notify). Doing
// this 2000 times in a tight loop stresses the interaction between the
// enqueuing stream, the main stream, and the persistent kernel's
// startup/shutdown timing. If there were a race between
// notifyDeviceThereMightNotBeAnyMoreWork and the next prepDeviceForWork, it
// would eventually surface over 2000 repetitions.
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
