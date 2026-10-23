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
#include <set>
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

    // One of two bricks splitting a 3D field of lengths {4, 6, 8} (batch 1) along its
    // outermost length axis, at index `split`
    rocfft_brick_t split_brick(size_t            half,
                               size_t            split,
                               rocfft_precision  precision,
                               rocfft_array_type array_type)
    {
        const size_t lower = half ? split : 0;
        const size_t upper = half ? 8 : split;
        return {data_descriptor_t(precision,
                                  array_type,
                                  {0, 0, lower, 0},
                                  {4, 6, upper, 1},
                                  {1, 4, 24, 24 * (upper - lower)}),
                rocfft_location_t(0, static_cast<int>(half))};
    }

    rocfft_field_t
        field_3d(rocfft_precision precision, rocfft_array_type array_type, size_t split = 4)
    {
        return rocfft_field_t({split_brick(0, split, precision, array_type),
                               split_brick(1, split, precision, array_type)});
    }

    // Expects `sub` to be `field.get_field_for_lengths(len_dims)` with bricks of array type
    // `array_type` (and the original precision and locations)
    void expect_field_for_lengths(const rocfft_field_t&   field,
                                  const std::set<size_t>& len_dims,
                                  rocfft_array_type       array_type)
    {
        const auto sub = field.get_field_for_lengths(len_dims);
        EXPECT_EQ(sub.get_array_type(), array_type);
        EXPECT_EQ(sub.get_precision(), field.get_precision());
        EXPECT_EQ(sub.get_global_data_range().get_len_rank(), len_dims.size());
        ASSERT_EQ(sub.num_bricks(), field.num_bricks());
        for(size_t b = 0; b < field.num_bricks(); b++)
        {
            const auto& brick = field.get_brick(b);
            EXPECT_EQ(sub.get_brick(b),
                      rocfft_brick_t(
                          brick.layout.get_layout_for_len_axes(len_dims).reinterpret_as(array_type),
                          brick.location));
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

TEST(field_test, field_for_lengths_passes_array_type_and_precision_through)
{
    for(const auto array_type : {rocfft_array_type_real, rocfft_array_type_complex_interleaved})
    {
        const auto field = field_3d(rocfft_precision_double, array_type);
        for(const auto& len_dims :
            std::vector<std::set<size_t>>{{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}})
            expect_field_for_lengths(field, len_dims, array_type);
        EXPECT_THROW(field.get_field_for_lengths({}), std::invalid_argument);
        EXPECT_THROW(field.get_field_for_lengths({3}), std::invalid_argument);
    }
}

TEST(field_test, field_for_lengths_keeps_hermitian_when_axis_0_is_selected)
{
    const auto field = field_3d(rocfft_precision_single, rocfft_array_type_hermitian_interleaved);
    for(const auto& len_dims : std::vector<std::set<size_t>>{{0}, {0, 1}, {0, 2}})
        expect_field_for_lengths(field, len_dims, rocfft_array_type_hermitian_interleaved);
}

TEST(field_test, field_for_lengths_relabels_hermitian_when_axis_0_is_not_selected)
{
    const auto field = field_3d(rocfft_precision_single, rocfft_array_type_hermitian_interleaved);
    for(const auto& len_dims : std::vector<std::set<size_t>>{{1}, {2}, {1, 2}})
        expect_field_for_lengths(field, len_dims, rocfft_array_type_complex_interleaved);
}

TEST(field_test, field_for_all_lengths_keeps_hermitian)
{
    const auto field = field_3d(rocfft_precision_single, rocfft_array_type_hermitian_interleaved);
    EXPECT_EQ(field.get_field_for_lengths({0, 1, 2}), field);
}

TEST(field_test, embedding_field_relabels_hermitian_and_restores_lengths)
{
    const auto hermitian = field_3d(rocfft_precision_half, rocfft_array_type_hermitian_interleaved);
    const auto complex   = field_3d(rocfft_precision_half, rocfft_array_type_complex_interleaved);
    for(const auto& len_dims : std::vector<std::set<size_t>>{{0}, {0, 1}, {1, 2}, {2}})
    {
        EXPECT_EQ(hermitian.get_field_for_lengths(len_dims).get_embedding_field(), complex);
        EXPECT_EQ(complex.get_field_for_lengths(len_dims).get_embedding_field(), complex);
    }
    const auto real = field_3d(rocfft_precision_half, rocfft_array_type_real);
    EXPECT_EQ(real.get_field_for_lengths({1}).get_embedding_field(), real);
}

TEST(field_test, embedding_field_of_non_embedded_field_is_identical)
{
    const auto field = field_3d(rocfft_precision_half, rocfft_array_type_hermitian_interleaved);
    EXPECT_EQ(field.get_embedding_field(), field);
    EXPECT_EQ(field.get_embedding_field().get_array_type(),
              rocfft_array_type_hermitian_interleaved);
}

TEST(field_test, is_same_up_to_hermitian_relabel)
{
    const auto hermitian
        = field_3d(rocfft_precision_double, rocfft_array_type_hermitian_interleaved);
    const auto complex = field_3d(rocfft_precision_double, rocfft_array_type_complex_interleaved);
    EXPECT_TRUE(hermitian.is_same_up_to_hermitian_relabel(complex));
    EXPECT_TRUE(complex.is_same_up_to_hermitian_relabel(hermitian));
    EXPECT_TRUE(hermitian.is_same_up_to_hermitian_relabel(hermitian));
    EXPECT_TRUE(field_3d(rocfft_precision_double, rocfft_array_type_hermitian_planar)
                    .is_same_up_to_hermitian_relabel(
                        field_3d(rocfft_precision_double, rocfft_array_type_complex_planar)));

    // real vs complex, interleaved vs planar, precision and bounds still matter
    EXPECT_FALSE(field_3d(rocfft_precision_double, rocfft_array_type_real)
                     .is_same_up_to_hermitian_relabel(complex));
    EXPECT_FALSE(hermitian.is_same_up_to_hermitian_relabel(
        field_3d(rocfft_precision_double, rocfft_array_type_complex_planar)));
    EXPECT_FALSE(hermitian.is_same_up_to_hermitian_relabel(
        field_3d(rocfft_precision_single, rocfft_array_type_complex_interleaved)));
    EXPECT_FALSE(hermitian.is_same_up_to_hermitian_relabel(
        field_3d(rocfft_precision_double, rocfft_array_type_complex_interleaved, 2)));
}
