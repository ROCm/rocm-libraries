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

// id get_id() const;

#include <hip/thread>
#include <new>
#include <cstdlib>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"
#include "force_include_hip.h"

class G
{
    int alive_ = 1;
public:
    __device__ void operator()()
    {
        assert(alive_ == 1);
    }
};

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    {
        G g;
        hip::wthread t0 = support::make_test_thread(g);
        hip::wthread::id id0 = t0.get_id();
        hip::wthread t1;
        hip::wthread::id id1 = t1.get_id();
        assert(t0.get_id() == id0);
        assert(id0 != id1);
        assert(t1.get_id() == hip::wthread::id());
        t0.join();
    }
#endif

  return 0;
}
