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

#include "data_layout.h"
#include <gtest/gtest.h>
#include <stdexcept>

TEST(rocfft_internal, data_layout_constructor_rejects_bad_input)
{
    // no batch axis at all
    EXPECT_THROW(data_layout_t({0, 0}, {4, 8}, {1, 4}, 0, false), std::invalid_argument);

    // lower, upper and strides must be the same size
    EXPECT_THROW(data_layout_t({0, 0, 0}, {4, 8}, {1, 4, 32}, 1, false), std::invalid_argument);

    // an axis whose lower bound is past its upper bound
    EXPECT_THROW(data_layout_t({5, 0, 0}, {4, 8, 1}, {1, 4, 32}, 1, true), std::invalid_argument);

    // a layout covering a full range cannot start anywhere but 0
    EXPECT_THROW(data_layout_t({1, 0, 0}, {4, 8, 1}, {1, 4, 32}, 1, false), std::invalid_argument);

    // the same thing marked partial is fine
    EXPECT_NO_THROW(data_layout_t({1, 0, 0}, {4, 8, 1}, {1, 4, 32}, 1, true));
}

TEST(rocfft_internal, data_layout_single_batch_axis_accessors)
{
    auto one = data_layout_t::default_full_layout({4, 8}, 3);
    EXPECT_EQ(one.batch(), 3u);
    EXPECT_EQ(one.distance(), 32u);

    // two batch axes: length 4, batches 8 and 3
    data_layout_t two({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 2, false);
    EXPECT_EQ(two.get_len_rank(), 1u);
    EXPECT_EQ(two.get_batch_rank(), 2u);
    EXPECT_THROW(two.batch(), std::logic_error);
    EXPECT_THROW(two.distance(), std::logic_error);
}

TEST(rocfft_internal, data_layout_default_strides_are_packed)
{
    auto packed = data_layout_t::default_full_layout({4, 8}, 3);

    EXPECT_EQ(packed.lengths(), (std::vector<size_t>{4, 8}));
    EXPECT_EQ(packed.strides_and_distances(), (std::vector<size_t>{1, 4, 32}));

    // spelling out the same strides by hand gives the same object
    EXPECT_TRUE(packed == data_layout_t::full_layout({4, 8}, {1, 4}, 3, 32));
}

TEST(rocfft_internal, data_layout_counts_packed_vs_padded)
{
    auto packed = data_layout_t::default_full_layout({4, 8}, 3);
    EXPECT_EQ(packed.logical_count(), 4u * 8u * 3u);
    EXPECT_EQ(packed.buffer_element_count(), 4u * 8u * 3u);
    EXPECT_TRUE(packed.is_contiguous());

    // real in-place transforms pad the second axis to 2*(n/2+1) reals
    auto padded = data_layout_t::default_full_layout({4, 8}, 3, true);
    EXPECT_EQ(padded.strides_and_distances(), (std::vector<size_t>{1, 6, 48}));
    EXPECT_EQ(padded.logical_count(), 4u * 8u * 3u);
    EXPECT_GT(padded.buffer_element_count(), padded.logical_count());
    EXPECT_FALSE(padded.is_contiguous());
}

TEST(rocfft_internal, data_layout_contiguous_strides)
{
    auto padded = data_layout_t::default_full_layout({4, 8}, 3, true);
    EXPECT_EQ(padded.contiguous_strides_and_distances(), (std::vector<size_t>{1, 4, 32}));

    // 4x8 with strides 8 and 1 still covers every slot exactly once
    auto swapped = data_layout_t::full_layout({4, 8}, {8, 1}, 1, 32);
    EXPECT_TRUE(swapped.is_contiguous());
    EXPECT_EQ(swapped.contiguous_strides_and_distances(), (std::vector<size_t>{8, 1, 32}));
}

TEST(rocfft_internal, data_layout_offset_in)
{
    auto outer = data_layout_t::full_layout({8, 8}, {1, 8}, 1, 64);

    // sub-region starting at logical coordinate (1, 2)
    data_layout_t inner({1, 2, 0}, {3, 5, 1}, {1, 8, 64}, 1, true);
    EXPECT_EQ(inner.offset_in(outer), 1u * 1u + 2u * 8u);

    // a sub-region whose first element is outside the outer range is rejected
    data_layout_t outside({9, 0, 0}, {10, 8, 1}, {1, 8, 64}, 1, true);
    EXPECT_THROW(outside.offset_in(outer), std::invalid_argument);

    // so is a region of a different rank
    auto other_rank = data_layout_t::full_layout({8}, {1}, 1, 8);
    EXPECT_THROW(other_rank.offset_in(outer), std::invalid_argument);
}

TEST(rocfft_internal, data_layout_continuous_chunk)
{
    data_layout_t whole({0, 0, 0}, {4, 8, 1}, {1, 4, 32}, 1, true);

    // rows 2 through 4 of the slowest axis: one unbroken run
    data_layout_t slice_of_slow_axis({0, 2, 0}, {4, 5, 1}, {1, 4, 32}, 1, true);
    EXPECT_TRUE(slice_of_slow_axis.is_continuous_in(whole));

    // a narrower slice of the fastest axis is not
    data_layout_t slice_of_fast_axis({1, 0, 0}, {3, 8, 1}, {1, 4, 32}, 1, true);
    EXPECT_FALSE(slice_of_fast_axis.is_continuous_in(whole));
}

TEST(rocfft_internal, data_layout_intersection)
{
    auto          first = data_layout_t::full_layout({8, 8}, {1, 8}, 1, 64);
    data_layout_t second({2, 3, 0}, {6, 7, 1}, {1, 8, 64}, 1, true);

    auto overlap = data_layout_t::make_contiguous_intersection_of(first, second);
    EXPECT_EQ(overlap.lower(), (std::vector<size_t>{2, 3, 0}));
    EXPECT_EQ(overlap.upper(), (std::vector<size_t>{6, 7, 1}));
    EXPECT_EQ(overlap.logical_count(), 4u * 4u);
    // the result is always given packed strides
    EXPECT_EQ(overlap.strides_and_distances(), (std::vector<size_t>{1, 4, 16}));

    // regions that do not touch produce an empty result rather than throwing
    data_layout_t far({6, 0, 0}, {8, 8, 1}, {1, 8, 64}, 1, true);
    data_layout_t near({0, 0, 0}, {2, 8, 1}, {1, 8, 64}, 1, true);
    EXPECT_TRUE(data_layout_t::make_contiguous_intersection_of(far, near).is_empty());
}

TEST(rocfft_internal, data_layout_inplace_complex_is_unchanged)
{
    auto in  = data_layout_t::default_full_layout({16}, 4);
    auto out = in.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                               rocfft_transform_type_complex_forward);
    ASSERT_TRUE(out.has_value());
    EXPECT_TRUE(*out == in);
}

TEST(rocfft_internal, data_layout_inplace_real_derives_matching_shape)
{
    // 8 real values per batch, padded to 2*(8/2+1) = 10 reals
    auto real_side = data_layout_t::default_full_layout({8}, 3, true);
    ASSERT_EQ(real_side.distance(), 10u);

    auto herm_side = real_side.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                            rocfft_transform_type_real_forward);
    ASSERT_TRUE(herm_side.has_value());
    // 8 reals become 8/2+1 = 5 complex values, and the batch distance halves
    EXPECT_EQ(herm_side->lengths(), (std::vector<size_t>{5}));
    EXPECT_EQ(herm_side->batch(), 3u);
    EXPECT_EQ(herm_side->distance(), 5u);

    // and going back the other way returns what we started with
    auto back = herm_side->get_other_inplace_layout_for(io_data_label::INPUT,
                                                        rocfft_transform_type_real_forward);
    ASSERT_TRUE(back.has_value());
    EXPECT_TRUE(*back == real_side);
}

TEST(rocfft_internal, data_layout_inplace_real_refuses_impossible_layouts)
{
    // batch distance is odd, so it cannot also be a whole number of complex
    // values.  9 is too small to hold the half spectrum as well, so use 11 to
    // check the odd-ness rule on its own.
    for(size_t odd_distance : {size_t{9}, size_t{11}})
    {
        auto odd = data_layout_t::full_layout({8}, {1}, 3, odd_distance);
        EXPECT_FALSE(odd.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                      rocfft_transform_type_real_forward)
                         .has_value())
            << "distance " << odd_distance;
    }

    // batch distance is even but too small to hold the half spectrum
    auto too_short = data_layout_t::full_layout({8}, {1}, 3, 8);
    EXPECT_FALSE(
        too_short
            .get_other_inplace_layout_for(io_data_label::OUTPUT, rocfft_transform_type_real_forward)
            .has_value());

    // the real side must be packed along its innermost axis
    auto strided = data_layout_t::full_layout({8}, {2}, 3, 20);
    EXPECT_FALSE(
        strided
            .get_other_inplace_layout_for(io_data_label::OUTPUT, rocfft_transform_type_real_forward)
            .has_value());
}

TEST(rocfft_internal, data_layout_inplace_real_odd_length)
{
    // 5 complex values came from either 8 or 9 real values
    auto herm = data_layout_t::full_layout({5}, {1}, 2, 5);

    auto even = herm.get_other_inplace_layout_for(
        io_data_label::INPUT, rocfft_transform_type_real_forward, false);
    ASSERT_TRUE(even.has_value());
    EXPECT_EQ(even->lengths(), (std::vector<size_t>{8}));

    auto odd = herm.get_other_inplace_layout_for(
        io_data_label::INPUT, rocfft_transform_type_real_forward, true);
    ASSERT_TRUE(odd.has_value());
    EXPECT_EQ(odd->lengths(), (std::vector<size_t>{9}));
}

TEST(rocfft_internal, data_layout_inplace_query_rejects_bad_arguments)
{
    auto layout = data_layout_t::default_full_layout({8}, 1);
    EXPECT_THROW(layout.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                     static_cast<rocfft_transform_type>(99)),
                 std::invalid_argument);

    // a layout covering nothing has no counterpart
    data_layout_t empty({0, 0}, {0, 1}, {1, 1}, 1, true);
    EXPECT_THROW(empty.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                    rocfft_transform_type_real_forward),
                 std::logic_error);
}

TEST(rocfft_internal, data_layout_axes_by_increasing_stride)
{
    auto layout = data_layout_t::full_layout({4, 8, 2}, {8, 1, 32}, 1, 64);

    EXPECT_EQ(layout.length_axes_by_increasing_strides(false), (std::vector<size_t>{1, 0, 2}));
    EXPECT_EQ(layout.length_axes_by_increasing_strides(true), (std::vector<size_t>{0, 1, 2}));
}

TEST(rocfft_internal, data_layout_subset_of_axes)
{
    auto layout = data_layout_t::default_full_layout({4, 8, 2}, 3);

    auto innermost_only = layout.get_layout_for_len_axes({0});
    EXPECT_EQ(innermost_only.get_len_rank(), 1u);
    EXPECT_EQ(innermost_only.lengths(), (std::vector<size_t>{4}));
    EXPECT_EQ(innermost_only.get_batch_rank(), 3u);

    // asking for every axis is just a copy
    EXPECT_TRUE(layout.get_layout_for_len_axes({0, 1, 2}) == layout);

    EXPECT_THROW(layout.get_layout_for_len_axes({}), std::invalid_argument);
    EXPECT_THROW(layout.get_layout_for_len_axes({0, 7}), std::invalid_argument);
}
