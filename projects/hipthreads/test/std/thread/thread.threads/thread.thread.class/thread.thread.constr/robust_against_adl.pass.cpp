//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// <thread>

// class wthread

// template <class F, class ...Args> wthread(F&& f, Args&&... args);


#include <hip/thread>

#include "test_macros.h"

#include "force_include_hip.h"

struct Incomplete;
template<class T> struct Holder { T t; };

__device__ void f(Holder<Incomplete> *) { }

int main(int, char **)
{
    // Since f is a device function, we can't reference it in host code, and our usual trick of wrapping the function
    // call in an extended lambda causes an ADL lookup.
#ifdef __HIP_DEVICE_COMPILE__ 
    Holder<Incomplete> *p = nullptr;
    hip::wthread t(f, p);
    t.join();
#endif
    return 0;
}
