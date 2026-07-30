#include "hip/mutex"
#include "hip/hip_runtime.h"
#include <iostream>
#include <cassert>
#include <mutex>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

__global__ void gmain() {
    hip::spin_mutex m, m2;
    m.lock();
    m.unlock();
    m.try_lock();
    m.unlock();
    {
        hip::lock_guard guard(m);
    }
    {
        m.lock();
        hip::lock_guard guard(m, ::std::adopt_lock);
    }
    {
        hip::unique_lock guard(m);
        hip::unique_lock guard2(m2);
        assert(guard.owns_lock());
        assert(guard2.owns_lock());
        guard.unlock();
        assert(!guard.owns_lock());
        assert(guard2.owns_lock());
        guard.lock();
        assert(guard.owns_lock());
        assert(guard2.owns_lock());
        guard2.unlock();
        assert(guard.owns_lock());
        assert(!guard2.owns_lock());
    }
    {
        hip::unique_lock guard(m, ::std::defer_lock);
        assert(!guard.owns_lock());
        guard.lock();
        assert(guard.owns_lock());
    }
    {
        hip::unique_lock guard(m, ::std::try_to_lock);
        assert(guard.owns_lock());
    }
    {
        m.lock();
        hip::unique_lock guard(m, ::std::try_to_lock);
        assert(!guard.owns_lock());
        m.unlock();
        assert(!guard.owns_lock());
        guard.lock();
        assert(guard.owns_lock());
    }
    {
        hip::lock(m, m2);
        m.unlock();
        m2.unlock();
    }
}

__global__ void block_sync_test() {
    static __device__ hip::spin_mutex m;
    static __device__ volatile int count = 0;
    {
        hip::unique_lock guard(m);
        for (int i = 0; i < 32; ++i)
        {
            assert(count++ == i);
        }
        for (int i = 32; i > 0; --i)
        {
            assert(count-- == i);
        }
    }
}

// TODO: Test using multiple threads per block
#if 0
[[clang::optnone]] __device__ bool critical_section(const hip::unique_lock<hip::spin_mutex> &guard, volatile int &count) {
    if (guard.owns_lock()) {
        int threadId = threadIdx.x;
        printf("Thread %u owns lock\n", threadId);
        for (int i = 0; i < 2; ++i)
        {
            assert(count++ == i);
        }
        for (int i = 2; i > 0; --i)
        {
            assert(count-- == i);
        }
        return true;
    }
    return false;
}

__global__ void thread_test() {
    static __device__ hip::spin_mutex m [[maybe_unused]];
    static __device__ volatile int count [[maybe_unused]] = 0;
    int attempt [[maybe_unused]] = 0;
    int threadId = threadIdx.x;
    printf("Thread %u starting\n", threadId);
    for (bool success = false; !success;) {
        if (attempt++ == 50)
            printf("Attempt #50 for thread %u\n", threadId);

        hip::unique_lock guard(m, ::std::try_to_lock);
        // TODO: if (guard.owns_lock()) { critical_section(count); break; /* OR */ success = true; }
        // results in critical section getting hoiseted out to AFTER the loop, breaking this code
        success = critical_section(guard, count);
    }
}
#endif // 0

int main() {
    hipLaunchKernelGGL(gmain, dim3(1), dim3(1), 0, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(block_sync_test, dim3(1<<16), dim3(1), 0, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    return 0;
}
