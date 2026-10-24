// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "data_descriptor_test_access.h"
#include "plan.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // One of two bricks splitting a 1D field of length 8 (batch 1) in halves
    rocfft_brick_t half_brick(size_t half, rocfft_precision precision, rocfft_array_type array_type)
    {
        return {data_descriptor_t(precision, array_type, {4 * half, 0}, {4 * half + 4, 1}, {1, 4}),
                rocfft_location_t(0, static_cast<int>(half))};
    }

    // Same as above, without array type nor precision (like user-added bricks)
    rocfft_brick_t untyped_half_brick(size_t half)
    {
        return {data_descriptor_t({4 * half, 0}, {4 * half + 4, 1}, {1, 4}),
                rocfft_location_t(0, static_cast<int>(half))};
    }

    // Expects the field constructor to reject `bricks` because of their array types
    // or precisions (other validations, e.g., tessellation, may reject them too).
    void expect_rejected_for_types(const std::vector<rocfft_brick_t>& bricks)
    {
        try
        {
            rocfft_field_t field(bricks);
            ADD_FAILURE() << "field constructor accepted bricks with invalid types";
        }
        catch(const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find("consistent array types and precisions"),
                      std::string::npos)
                << e.what();
        }
    }
}

TEST(field_test, exposes_array_type_and_precision)
{
    const rocfft_field_t field({half_brick(0, rocfft_precision_double, rocfft_array_type_real),
                                half_brick(1, rocfft_precision_double, rocfft_array_type_real)});
    EXPECT_EQ(field.get_array_type(), rocfft_array_type_real);
    EXPECT_EQ(field.get_precision(), rocfft_precision_double);
}

TEST(field_test, rejects_untyped_bricks)
{
    expect_rejected_for_types({untyped_half_brick(0), untyped_half_brick(1)});
}

TEST(field_test, rejects_brick_with_unset_array_type)
{
    const auto valid      = half_brick(0, rocfft_precision_single, rocfft_array_type_real);
    const auto unset_type = half_brick(1, rocfft_precision_single, rocfft_array_type_unset);
    ASSERT_TRUE(unset_type.layout.has_precision());
    EXPECT_NO_THROW(
        rocfft_field_t({valid, half_brick(1, rocfft_precision_single, rocfft_array_type_real)}));
    expect_rejected_for_types({valid, unset_type});
    expect_rejected_for_types({unset_type, valid});
    expect_rejected_for_types({half_brick(0, rocfft_precision_single, rocfft_array_type_unset)});
}

TEST(field_test, rejects_brick_with_unset_precision)
{
    const auto valid           = half_brick(0, rocfft_precision_single, rocfft_array_type_real);
    auto       unset_precision = untyped_half_brick(1);
    data_descriptor_test_access::set_array_type(unset_precision.layout, rocfft_array_type_real);
    ASSERT_FALSE(unset_precision.layout.has_precision());
    expect_rejected_for_types({valid, unset_precision});
    expect_rejected_for_types({unset_precision, valid});
    auto lone_unset_precision = untyped_half_brick(0);
    data_descriptor_test_access::set_array_type(lone_unset_precision.layout,
                                                rocfft_array_type_real);
    expect_rejected_for_types({lone_unset_precision});
}

TEST(field_test, rejects_inconsistent_array_types)
{
    expect_rejected_for_types(
        {half_brick(0, rocfft_precision_single, rocfft_array_type_complex_interleaved),
         half_brick(1, rocfft_precision_single, rocfft_array_type_hermitian_interleaved)});
}

TEST(field_test, rejects_inconsistent_precisions)
{
    expect_rejected_for_types(
        {half_brick(0, rocfft_precision_single, rocfft_array_type_complex_interleaved),
         half_brick(1, rocfft_precision_double, rocfft_array_type_complex_interleaved)});
}

TEST(field_test, reinterpret_as_relabels_all_bricks)
{
    rocfft_field_t field(
        {half_brick(0, rocfft_precision_half, rocfft_array_type_hermitian_interleaved),
         half_brick(1, rocfft_precision_half, rocfft_array_type_hermitian_interleaved)});
    field.reinterpret_as(rocfft_array_type_complex_interleaved);
    EXPECT_EQ(field.get_array_type(), rocfft_array_type_complex_interleaved);
    EXPECT_EQ(field.get_precision(), rocfft_precision_half);
    for(size_t b = 0; b < 2; b++)
        EXPECT_EQ(field.get_brick(b),
                  half_brick(b, rocfft_precision_half, rocfft_array_type_complex_interleaved));
}

TEST(field_test, failing_reinterpret_as_leaves_field_unchanged)
{
    rocfft_field_t field({half_brick(0, rocfft_precision_single, rocfft_array_type_real),
                          half_brick(1, rocfft_precision_single, rocfft_array_type_real)});
    const auto     original = field;
    EXPECT_THROW(field.reinterpret_as(rocfft_array_type_complex_interleaved), std::logic_error);
    EXPECT_EQ(field, original);
    EXPECT_EQ(field.get_array_type(), rocfft_array_type_real);
}
