#include "gpu/mutex"
#include "gpu/pseudo_mutex" // From gpu::thread library
#include "hip/hip_runtime.h"
#include <iostream>
#include <cassert>
#include <mutex>
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

__device__ void gmain() {
    gpu::pseudo_mutex m, m2;
    m.lock();
    m.unlock();
    m.try_lock();
    m.unlock();
    {
        gpu::lock_guard guard(m);
    }
    {
        m.lock();
        gpu::lock_guard guard(m, std::adopt_lock);
    }
    {
        gpu::unique_lock guard(m);
        gpu::unique_lock guard2(m2);
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
        gpu::unique_lock guard(m, std::defer_lock);
        assert(!guard.owns_lock());
        guard.lock();
        assert(guard.owns_lock());
    }
    {
        gpu::unique_lock guard(m, std::try_to_lock);
        assert(guard.owns_lock());
    }
    {
        m.lock();
        gpu::unique_lock guard(m, std::try_to_lock);
        assert(!guard.owns_lock());
        m.unlock();
        assert(!guard.owns_lock());
        guard.lock();
        assert(guard.owns_lock());
    }
    {
        gpu::lock(m, m2);
        m.unlock();
        m2.unlock();
    }
}

__device__ void block_sync_test() {
    static __device__ gpu::pseudo_mutex m;
    static __device__ volatile int count = 0;
    {
        gpu::unique_lock guard(m);
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
[[clang::optnone]] __device__ bool critical_section(const gpu::unique_lock<gpu::pseudo_mutex> &guard, volatile int &count) {
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
    static __device__ gpu::pseudo_mutex m [[maybe_unused]];
    static __device__ volatile int count [[maybe_unused]] = 0;
    int attempt [[maybe_unused]] = 0;
    int threadId = threadIdx.x;
    printf("Thread %u starting\n", threadId);
    for (bool success = false; !success;) {
        if (attempt++ == 50)
            printf("Attempt #50 for thread %u\n", threadId);

        gpu::unique_lock guard(m, std::try_to_lock);
        // TODO: if (guard.owns_lock()) { critical_section(count); break; /* OR */ success = true; }
        // results in critical section getting hoiseted out to AFTER the loop, breaking this code
        success = critical_section(guard, count);
    }
}
#endif // 0

int main() {
    gpu::start();
    gpu::thread([] __device__(){gmain();}).join();

    std::vector<gpu::thread> threads(1<<16);
    for (unsigned int i = 0; i < threads.size(); ++i) {
        threads[i] = gpu::thread([] __device__(){block_sync_test();});
        assert(threads[i].joinable());
    }
    for (unsigned int i = 0; i < threads.size(); ++i) {
        try {
            threads[i].join();
        } catch (...) {
            printf("Exception when joining thread %u\n", i);
            printf("threads[%u].get_id() = %d\n", i, threads[i].get_id());
            throw;
        }
    }
    gpu::finish();
    return 0;
}
