//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <thread>

// class wthread
//     template <class _Fp, class ..._Args,
//         explicit wthread(_Fp&& __f, _Args&&... __args);
//  This constructor shall not participate in overload resolution
//       if decay<F>::type is the same type as hip::wthread.


#include <hip/thread>

#include "force_include_hip.h"

int main(int, char**)
{
    volatile hip::wthread t1;
    hip::wthread t2 ( t1, 1, 2.0 );
    return 0;
}
