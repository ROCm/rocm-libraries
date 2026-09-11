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

// ~wthread();

#include <cassert>
#include <cstdlib>
#include <exception>
#include <new>
#include <hip/thread>
#include <hip/__support/misuse.h>

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

// The C++ standard says destroying a joinable thread calls std::terminate().
// On the GPU there is no host terminate handler, and a device abort surfaces as
// an HSA hardware exception whose host exit code is not stable across runtimes,
// so the libc++ death-test form (set_terminate -> _Exit(0)) cannot be verified
// by exit code here. Instead we put the library into nonfatal mode and verify
// that the misuse is *detected* (see hip/__thread/thread.h).
int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    {
        hip::__hipthreads_misuse_nonfatal = true;

        assert(G::n_alive == 0);
        assert(!G::op_run);
        G g;

        unsigned int before = hip::__hipthreads_misuse_count;
        {
          hip::wthread t = support::make_test_thread(g);
          hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
          // t is still joinable here; its destructor runs at the closing brace
          // and must be detected as misuse.
        }
        assert(hip::__hipthreads_misuse_count == before + 1);
    }
#endif

  return 0;
}
