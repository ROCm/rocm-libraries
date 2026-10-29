//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <thread>

// class wthread

// void swap(wthread& t);

#include <hip/thread>
#include <new>
#include <cstdlib>
#include <cassert>

#include "force_include_hip.h"

#include "make_test_thread.h"
#include "test_macros.h"

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

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    {
        G g;
        hip::wthread t0 = support::make_test_thread(g);
        hip::wthread::id id0 = t0.get_id();
        hip::wthread t1;
        hip::wthread::id id1 = t1.get_id();
        t0.swap(t1);
        assert(t0.get_id() == id1);
        assert(t1.get_id() == id0);
        t1.join();
    }
#endif

  return 0;
}
