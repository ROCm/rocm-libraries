//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// ADDITIONAL_COMPILE_FLAGS: -DTEST_USE_GPU_THREADS

// <thread>

// class thread

// void detach();

#include <gpu/thread>
#include <hip/std/atomic>
#include <system_error>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"
#include "force_include_hip.h"

__device__ hip::std::atomic_bool done(false);

class G
{
    int alive_;
    bool done_;
public:
    static __device__ int n_alive;
    static __device__ bool op_run;

    __device__ G() : alive_(1), done_(false)
    {
        ++n_alive;
    }

    __device__ G(const G& g) : alive_(g.alive_), done_(false)
    {
        ++n_alive;
    }
    __device__ ~G()
    {
        alive_ = 0;
        --n_alive;
        if (done_) done = true;
    }

    __device__ void operator()()
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
        op_run = true;
        done_ = true;
    }
};

__device__ int G::n_alive = 0;
__device__ bool G::op_run = false;

__device__ void foo() { done = true; }

__global__ void wait_for_done_kern() { while(!done) {} }
__host__ __device__ void wait_for_done() {
#ifdef __HIP_DEVICE_COMPILE__
    while (!done) {}
#else
    hipLaunchKernelGGL(wait_for_done_kern, dim3(1), dim3(1), 0, gpu::internal::getEnqueingStream());
    __LIBGPU_HIP_CHECK__(hipStreamSynchronize(gpu::internal::getEnqueingStream()));
#endif
}

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    {
        G g;
        gpu::thread t0 = support::make_test_thread(g);
        assert(t0.joinable());
        t0.detach();
        assert(!t0.joinable());
        wait_for_done();
        assert(G::op_run);
        assert(G::n_alive == 1);
    }
    assert(G::n_alive == 0);
    done = false;
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
#ifdef __HIP_DEVICE_COMPILE__
        gpu::thread t0 = support::make_test_thread(foo);
#else
        gpu::thread t0 = support::make_test_thread([]__device__(){foo();});
#endif
        assert(t0.joinable());
        t0.detach();
        assert(!t0.joinable());
#ifndef __HIP_DEVICE_COMPILE__
        try {
            t0.detach();
        } catch (std::system_error const&) {
        }
#endif
        // Wait to make sure that the detached thread has started up.
        // Without this, we could exit main and start destructing global
        // resources that are needed when the thread starts up, while the
        // detached thread would start up only later.
        wait_for_done();
    }
#endif

  return 0;
}
