// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../common_test_header.hpp"

#include "../../common/utils_device_ptr.hpp"

#include "rocprim/device/device_transform.hpp"
#include "rocprim/types/tuple.hpp"

#include <rocprim/test_utils_custom_test_types.hpp>
#include <type_traits>
#include <utility>
#include <vector>

// helper functions to use with ::rocprim::apply (or std::apply)
constexpr auto f_sum   = [](auto&&... v) { return (v + ...); };
constexpr auto f_sum_x = [](auto&&... v) { return (v.x + ...); };

TEST(RocprimTupleTests, HostApply)
{
    // Check if basic apply works
    ASSERT_EQ(rocprim::apply(f_sum, rocprim::make_tuple(int{1}, int{2}, int{3})), 6);

    // And also check with mixed types
    ASSERT_EQ(rocprim::apply(f_sum, rocprim::make_tuple(double{1.0}, float{2.0}, int{3})), 6);

    // Check if values passed can be passed by reference
    auto tuple_a = rocprim::make_tuple(1, 2);
    ASSERT_TRUE(rocprim::apply([](auto&&... v)
                               { return (std::is_rvalue_reference_v<decltype(v)> && ...); },
                               std::move(tuple_a)));

    // Check if passing value by const ref works
    auto        tuple_b      = rocprim::make_tuple(1, 2);
    const auto& tuple_b_cref = tuple_b;
    ASSERT_EQ(rocprim::apply(f_sum, tuple_b_cref), 3);

    // and also normal ref
    auto& tuple_b_ref = tuple_b;
    ASSERT_EQ(rocprim::apply(f_sum, tuple_b_ref), 3);

    // non copyable (so has to be moved or referenced)
    using nc = test_utils::custom_non_copyable_type<int>;
    ASSERT_EQ(rocprim::apply(f_sum_x, rocprim::make_tuple(nc{4}, nc{2}, nc{3})), 9);
    ASSERT_EQ(std::apply(f_sum_x, std::make_tuple(nc{1}, nc{0}, nc{3})), 4);

    // non default
    using nd = test_utils::custom_non_default_type<int>;
    ASSERT_EQ(rocprim::apply(f_sum_x, rocprim::make_tuple(nd{1}, nd{3}, nd{1})), 5);
}

TEST(RocprimTupleTests, DeviceApply)
{
    // we ignore the weird type related tests here since we already checked them on host

    auto h_tuple  = ::rocprim::make_tuple(1, 2, 3);
    auto d_tuples = common::device_ptr<decltype(h_tuple)>(std::vector{h_tuple});
    auto d_result = common::device_ptr<int>(1);

    HIP_CHECK(rocprim::transform(d_tuples.get(),
                                 d_result.get(),
                                 d_result.size(),
                                 [](auto t) { return rocprim::apply(f_sum, t); }));
    ASSERT_EQ(d_result.load().front(), 6);
}
