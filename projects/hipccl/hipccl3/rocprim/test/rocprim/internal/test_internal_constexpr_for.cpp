// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../common_test_header.hpp"

#include <rocprim/detail/various.hpp>

TEST(RocprimInternalCosntexprForTests, Basic)
{
    // lt
    rocprim::detail::constexpr_for_lt<1, 6, 2>(
        [](auto i)
        {
            static_assert(1 <= i && i <= 5);
            static_assert(i == 1 || i == 3 || i == 5);
        });
    rocprim::detail::constexpr_for_lt<1, 7, 2>(
        [](auto i)
        {
            static_assert(1 <= i && i <= 5);
            static_assert(i == 1 || i == 3 || i == 5);
        });

    // lte
    rocprim::detail::constexpr_for_lte<2, 8, 3>(
        [](auto i)
        {
            static_assert(2 <= i && i <= 8);
            static_assert(i == 2 || i == 5 || i == 8);
        });
    rocprim::detail::constexpr_for_lte<2, 9, 3>(
        [](auto i)
        {
            static_assert(2 <= i && i <= 8);
            static_assert(i == 2 || i == 5 || i == 8);
        });

    // gt
    rocprim::detail::constexpr_for_gt<12, 3, -4>(
        [](auto i)
        {
            static_assert(12 >= i && i >= 4);
            static_assert(i == 12 || i == 8 || i == 4);
        });
    rocprim::detail::constexpr_for_gt<12, 2, -4>(
        [](auto i)
        {
            static_assert(12 >= i && i >= 4);
            static_assert(i == 12 || i == 8 || i == 4);
        });

    // gte
    rocprim::detail::constexpr_for_gte<15, 5, -5>(
        [](auto i)
        {
            static_assert(15 >= i && i >= 5);
            static_assert(i == 15 || i == 10 || i == 5);
        });
    rocprim::detail::constexpr_for_gte<15, 4, -5>(
        [](auto i)
        {
            static_assert(15 >= i && i >= 5);
            static_assert(i == 15 || i == 10 || i == 5);
        });
}
