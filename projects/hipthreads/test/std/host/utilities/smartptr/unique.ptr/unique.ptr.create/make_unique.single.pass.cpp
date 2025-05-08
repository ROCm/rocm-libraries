//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
#include "gpu/memory"
//#include <string>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX23 bool test() {
    {
        gpu::unique_ptr_h<int> p1 = gpu::make_unique<int>(1);
        assert(*p1 == 1);
        p1 = gpu::make_unique<int>();
        // For performance reasons, gpu::make_unique<T>() doesn't perform any initialization.
        // Users must use gpu::make_unique<T>(T()) if they want initialization.
        //assert(*p1 == 0);
    }

    // TODO: uncomment once we implement gpu::string
    // {
    //     gpu::unique_ptr_h<gpu::string> p2 = gpu::make_unique<gpu::string>("Meow!");
    //     assert(*p2 == "Meow!");
    //     p2 = gpu::make_unique<gpu::string>();
    //     assert(*p2 == "");
    //     p2 = gpu::make_unique<gpu::string>(6, 'z');
    //     assert(*p2 == "zzzzzz");
    // }

    return true;
}

int main(int, char**) {
    test();
#if TEST_STD_VER >= 23
    static_assert(test());
#endif

    return 0;
}
