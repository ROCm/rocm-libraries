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

marsavic@daniel-ubuntu-desktop:~/gpulib$ ./build/bin/scheduler_nested_spawn_test 
test_two_level_fanout passed
test_chain_spawn passed
test_fork_join passed
marsavic@daniel-ubuntu-desktop:~/gpulib$ ./build/bin/scheduler_nested_spawn_test 
test_two_level_fanout passed
test_chain_spawn passed
test_fork_join passed
marsavic@daniel-ubuntu-desktop:~/gpulib$ ./build/bin/scheduler_nested_spawn_test 
test_two_level_fanout passed
test_chain_spawn passed
test_fork_join passed
marsavic@daniel-ubuntu-desktop:~/gpulib$ ./build/bin/scheduler_nested_spawn_test 
test_two_level_fanout passed
test_chain_spawn passed
test_fork_join passed
marsavic@daniel-ubuntu-desktop:~/gpulib$ ./build/bin/scheduler_nested_spawn_test 
test_two_level_fanout passed
test_chain_spawn passed
test_fork_join passedstatic void test_chain_spawn() {
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
