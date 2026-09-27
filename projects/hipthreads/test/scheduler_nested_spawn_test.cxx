// scheduler_nested_spawn_test.cxx -- Device-to-device thread creation.
//
// Verifies that hip::thread(lambda) called from device code correctly
// allocates a WorkNode via device malloc, constructs it with make_worknode,
// pushes it into mainWorkQueue via insertIntoMainQueue(), and that
// device-side join() works (the parent spins on isSchedulerDoneWith() until
// each child finishes). Also covers serial dependency chains and many
// parents simultaneously blocked in device-side join().

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

// 64 parent threads from the host. Each parent, running on the GPU, creates
// 16 children (also on the GPU) and joins them all. Expected counter:
// 64 * 16 = 1024. Proves that hip::thread(lambda) called from device code
// correctly allocates a WorkNode via device malloc, constructs it with
// make_worknode, and pushes it into mainWorkQueue via insertIntoMainQueue().
// Also proves device-side join() works (the parent spins on
// isSchedulerDoneWith() until each child finishes).
static void test_two_level_fanout() {
    constexpr unsigned int N_PARENTS = 64;
    constexpr unsigned int M_CHILDREN = 16;
    constexpr unsigned int EXPECTED = N_PARENTS * M_CHILDREN;

    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    {
        ::std::vector<hip::thread> parents(N_PARENTS);
        for (unsigned int i = 0; i < N_PARENTS; ++i) {
            parents[i] = hip::thread([d_counter] __device__() {
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
        for (auto &t : parents) {
            t.join();
        }
    }

    hip::thread([d_counter, EXPECTED] __device__() {
        assert(static_cast<unsigned int>(atomicAdd(d_counter, 0)) == EXPECTED);
    }).join();

    ::std::cerr << "test_two_level_fanout passed\n";
}

__device__ void chain_link(uint32_t *seq, unsigned int level, unsigned int max_depth) {
    atomicExch(&seq[level], level + 1);
    if (level + 1 < max_depth) {
        auto child = hip::thread([seq, level, max_depth] __device__() {
            chain_link(seq, level + 1, max_depth);
        });
        child.join();
    }
}

// A single thread spawns a child, that child spawns another child, and so on,
// 8 levels deep. Each level writes level + 1 to d_sequence[level]. The
// verifier checks d_sequence = {1, 2, 3, 4, 5, 6, 7, 8}. This is a serial
// dependency chain -- level 3 can't start until level 2 finishes spawning it.
// Tests that the scheduler doesn't deadlock when a thread on one vcore joins
// a child that might need to run on the same vcore (since the parent is
// blocked in join() spinning, the scheduler needs some other vcore to pick up
// the child). Also tests that 8 sequential round-trips through
// insertIntoMainQueue -> getWork -> invokeNext -> join all work correctly.
// Implemented as a __device__ function chain_link to avoid reference captures
// across vcores.
static void test_chain_spawn() {
    constexpr unsigned int DEPTH = 8;

    uint32_t *d_sequence;
    CHECK(hipMalloc(&d_sequence, DEPTH * sizeof(uint32_t)));
    CHECK(hipMemset(d_sequence, 0, DEPTH * sizeof(uint32_t)));

    hip::thread([d_sequence] __device__() {
        chain_link(d_sequence, 0, 8);
    }).join();

    hip::thread([d_sequence] __device__() {
        for (unsigned int i = 0; i < 8; ++i) {
            assert(d_sequence[i] == i + 1);
        }
    }).join();

    ::std::cerr << "test_chain_spawn passed\n";
}

// 128 parents from the host, each forks 8 children on the device and joins
// all 8. Expected counter: 128 * 8 = 1024. Similar to test_two_level_fanout
// but with more parents (128 vs 64) and fewer children each (8 vs 16). The
// higher parent count means more concurrent contention on mainWorkQueue's
// pushCount atomic, and up to 128 parents simultaneously in device-side
// join() spin-waiting, testing that activeVcoreCount bookkeeping handles the
// case where many vcores are "active" but actually blocked waiting for
// children.
static void test_fork_join() {
    constexpr unsigned int N_PARENTS = 128;
    constexpr unsigned int K_CHILDREN = 8;
    constexpr unsigned int EXPECTED = N_PARENTS * K_CHILDREN;

    int *d_counter;
    CHECK(hipMalloc(&d_counter, sizeof(int)));
    CHECK(hipMemset(d_counter, 0, sizeof(int)));

    {
        ::std::vector<hip::thread> parents(N_PARENTS);
        for (unsigned int i = 0; i < N_PARENTS; ++i) {
            parents[i] = hip::thread([d_counter] __device__() {
                hip::thread children[8];
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
        for (auto &t : parents) {
            t.join();
        }
    }

    hip::thread([d_counter, EXPECTED] __device__() {
        assert(static_cast<unsigned int>(atomicAdd(d_counter, 0)) == EXPECTED);
    }).join();

    ::std::cerr << "test_fork_join passed\n";
}

int main() {
    test_two_level_fanout();
    test_chain_spawn();
    test_fork_join();
    return 0;
}
