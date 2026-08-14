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

// wthread(wthread&& t);

#include <hip/thread>
#include <cassert>
#include <cstdlib>
#include <utility>

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

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    assert(G::n_alive == 0);
    assert(!G::op_run);
    {
        G g;
        assert(G::n_alive == 1);
        assert(!G::op_run);

        hip::wthread t0 = support::make_test_thread(g);
        hip::wthread::id id = t0.get_id();

        hip::wthread t1 = ::std::move(t0);
        assert(t1.get_id() == id);
        assert(t0.get_id() == hip::wthread::id());

        t1.join();
        assert(G::n_alive == 1);
        assert(G::op_run);
    }
    assert(G::n_alive == 0);
    assert(G::op_run);
#endif

    return 0;
}
