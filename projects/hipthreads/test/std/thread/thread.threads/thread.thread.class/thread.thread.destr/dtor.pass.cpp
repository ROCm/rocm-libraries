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

// ~thread();

#include <cassert>
#include <cstdlib>
#include <exception>
#include <new>
#include <gpu/thread>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

class G
{
    int alive_;
public:
    static __device__ int n_alive;
    static __device__ bool op_run;

    __device__ G() : alive_(1) {++n_alive;}
    __device__ G(const G& g) : alive_(g.alive_) {++n_alive;}
    __device__ ~G() {alive_ = 0; --n_alive;}

    __device__ void operator()()
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
        op_run = true;
    }
};

__device__ int G::n_alive = 0;
__device__ bool G::op_run = false;

void f1()
{
    std::_Exit(0);
}

int main(int, char**)
{
#ifndef __HIP_DEVICE_COMPILE__
    std::set_terminate(f1);
#else
    {
        assert(G::n_alive == 0);
        assert(!G::op_run);
        G g;
        {
          gpu::thread t = support::make_test_thread(g);
          gpu::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
        }
    }
    assert(false);
#endif

  return 0;
}
