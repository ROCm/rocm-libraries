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

#include "../../../shared/precision_type.h"
#include "client_data_layout_helpers.h"
#include "data_set.h"
#include "enum_printer.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// internal_test_main.cpp
extern size_t random_seed;
extern size_t num_random_cases;

struct data_set_test_access
{
    static void reorder_length_axes(data_set_t& data_set, const std::vector<size_t>& order)
    {
        data_set.reorder_length_axes(order);
    }
    static void clear(data_set_t& data_set)
    {
        data_set.clear();
    }
    static void
        full_range_reset(data_set_t& data_set, const std::vector<size_t>& lengths, size_t batch)
    {
        data_set.full_range_reset(lengths, {}, {batch}, {}, false);
    }
    // Copy of a data set (embedding included) given an array type and a precision
    // (left unset if std::nullopt) the way friends set them, e.g., upon finalization.
    static data_set_t with_type(data_set_t                      data_set,
                                rocfft_array_type               array_type,
                                std::optional<rocfft_precision> precision)
    {
        data_set.array_type = array_type;
        if(precision)
            data_set.set_precision(*precision);
        else
            data_set.precision = data_set_t::precision_unset;
        return data_set;
    }
};

namespace
{
    std::vector<size_t> reversed(std::vector<size_t> v)
    {
        std::reverse(v.begin(), v.end());
        return v;
    }

    std::string to_str(const std::vector<size_t>& v)
    {
        std::string ret = "{";
        for(auto x : v)
            ret += " " + std::to_string(x);
        return ret + " }";
    }

    // all array types, unset included; the tables below follow this order
    const rocfft_array_type all_array_types[] = {rocfft_array_type_unset,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_array_type_complex_planar,
                                                 rocfft_array_type_real,
                                                 rocfft_array_type_hermitian_interleaved,
                                                 rocfft_array_type_hermitian_planar};
    const rocfft_precision  all_precisions[]
        = {rocfft_precision_half, rocfft_precision_single, rocfft_precision_double};

    const auto with_type = data_set_test_access::with_type;

    rocfft_array_type type_of(const data_set_t& data_set)
    {
        return data_set.has_array_type() ? data_set.get_array_type() : rocfft_array_type_unset;
    }
    std::optional<rocfft_precision> precision_of(const data_set_t& data_set)
    {
        return data_set.has_precision() ? std::make_optional(data_set.get_precision())
                                        : std::nullopt;
    }
    std::string to_str(std::optional<rocfft_precision> precision)
    {
        return precision ? PrintPrecision(*precision) : "unset";
    }
    rocfft_array_type complex_counterpart(rocfft_array_type t)
    {
        return t == rocfft_array_type_hermitian_interleaved ? rocfft_array_type_complex_interleaved
               : t == rocfft_array_type_hermitian_planar    ? rocfft_array_type_complex_planar
                                                            : t;
    }

    // outcome of a call: accepted, std::invalid_argument, std::logic_error reporting
    // something not implemented yet, or any other std::logic_error
    enum outcome
    {
        Y,
        I,
        N,
        L
    };
    template <typename F>
    outcome outcome_of(F&& f)
    {
        try
        {
            f();
            return Y;
        }
        catch(const std::invalid_argument&)
        {
            return I;
        }
        catch(const std::logic_error& e)
        {
            return std::string(e.what()).find("not implemented yet") != std::string::npos ? N : L;
        }
    }
}

// Every test restarts its generator from random_seed, so a failing case replays
// with --seed <value> --gtest_filter=<test>, and at least as many --num_random_cases.
class data_set_test : public ::testing::Test
{
protected:
    std::mt19937_64        rng{random_seed};
    ::testing::ScopedTrace seed_trace{
        __FILE__, __LINE__, "random_seed = " + std::to_string(random_seed)};

    size_t uniform(size_t lo, size_t hi)
    {
        return std::uniform_int_distribution<size_t>(lo, hi)(rng);
    }
    std::vector<size_t> random_vec(size_t size, size_t lo, size_t hi)
    {
        std::vector<size_t> ret(size);
        for(auto& x : ret)
            x = uniform(lo, hi);
        return ret;
    }
    std::vector<size_t> random_permutation(size_t size)
    {
        std::vector<size_t> ret(size);
        std::iota(ret.begin(), ret.end(), 0);
        std::shuffle(ret.begin(), ret.end(), rng);
        return ret;
    }
    rocfft_array_type random_array_type()
    {
        return all_array_types[uniform(0, std::size(all_array_types) - 1)];
    }
    // unset in about a quarter of the cases
    std::optional<rocfft_precision> random_precision()
    {
        const auto idx = uniform(0, std::size(all_precisions));
        return idx < std::size(all_precisions) ? std::make_optional(all_precisions[idx])
                                               : std::nullopt;
    }
    // Full-range data set of random ranks/spans whose axes are laid out in random order,
    // with random padding between consecutive axes (non-aliasing, not necessarily packed).
    data_set_t random_full_range()
    {
        const auto          len_rank   = uniform(1, 4);
        const auto          batch_rank = uniform(1, 2);
        const auto          spans      = random_vec(len_rank + batch_rank, 1, 7);
        std::vector<size_t> strides(spans.size());
        size_t              next = uniform(1, 2);
        for(auto axis : random_permutation(spans.size()))
        {
            strides[axis] = next;
            next          = next * spans[axis] + uniform(0, 3);
        }
        return data_set_t(std::vector<size_t>(spans.size(), 0), spans, strides, batch_rank);
    }
};

TEST_F(data_set_test, constructor_rejects_bad_input)
{
    // no batch axis at all
    EXPECT_THROW(data_set_t({0, 0}, {4, 8}, {1, 4}, 0), std::invalid_argument);
    // no length axis left once batch axes are taken
    EXPECT_THROW(data_set_t({0, 0}, {4, 8}, {1, 4}, 2), std::invalid_argument);
    // lower, upper and strides must be the same size
    EXPECT_THROW(data_set_t({0, 0, 0}, {4, 8}, {1, 4, 32}), std::invalid_argument);
    // an axis whose lower bound is past its upper bound
    EXPECT_THROW(data_set_t({5, 0, 0}, {4, 8, 1}, {1, 4, 32}), std::invalid_argument);
}

TEST_F(data_set_test, single_batch_axis_accessors)
{
    auto one = data_set_t::default_full_range({4, 8}, 3);
    EXPECT_EQ(one.batch(), 3u);
    EXPECT_EQ(one.distance(), 32u);

    // two batch axes: length 4, batches 8 and 3
    data_set_t two({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 2);
    EXPECT_EQ(two.get_len_rank(), 1u);
    EXPECT_EQ(two.get_batch_rank(), 2u);
    EXPECT_THROW(two.batch(), std::logic_error);
    EXPECT_THROW(two.distance(), std::logic_error);
}

// The client helpers are row-major (last index fastest) while data_set_t lists
// the fastest axis first, so lengths, batches and results are reversed.
TEST_F(data_set_test, default_layouts_match_client_helpers)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L = random_vec(uniform(1, 4), 1, 16);
        const auto B = random_vec(uniform(1, 3), 1, 5);
        SCOPED_TRACE("lengths " + to_str(L) + " batches " + to_str(B));
        const auto rL = reversed(L);
        const auto rB = reversed(B);

        const auto cplx = data_set_t::default_full_range(L, B);
        EXPECT_EQ(cplx.strides(),
                  reversed(default_strides(
                      fft_transform_type_complex_forward, fft_placement_inplace, fft_io_in, rL)));
        EXPECT_EQ(
            cplx.distances(),
            reversed(default_distances(
                fft_transform_type_complex_forward, fft_placement_inplace, fft_io_in, rL, rB)));

        const auto real = data_set_t::default_full_range(L, B, true);
        EXPECT_EQ(real.strides(),
                  reversed(default_strides(
                      fft_transform_type_real_forward, fft_placement_inplace, fft_io_in, rL)));
        EXPECT_EQ(real.distances(),
                  reversed(default_distances(
                      fft_transform_type_real_forward, fft_placement_inplace, fft_io_in, rL, rB)));

        const auto herm = real.get_other_inplace_data_set_for(io_data_label::OUTPUT,
                                                              rocfft_transform_type_real_forward);
        ASSERT_TRUE(herm.has_value());
        auto herm_lengths = L;
        herm_lengths[0]   = L[0] / 2 + 1;
        EXPECT_EQ(herm->lengths(), herm_lengths);
        EXPECT_EQ(herm->batches(), B);
        EXPECT_EQ(herm->strides(),
                  reversed(default_strides(
                      fft_transform_type_real_forward, fft_placement_inplace, fft_io_out, rL)));
        EXPECT_EQ(herm->distances(),
                  reversed(default_distances(
                      fft_transform_type_real_forward, fft_placement_inplace, fft_io_out, rL, rB)));
    }
}

// Padding applies to the innermost axis only: each innermost row of n reals
// occupies P = 2 * (n / 2 + 1) slots, and buffer_element_count() stops at the
// last real element, so the trailing P - n slots of the last row are excluded.
TEST_F(data_set_test, padded_buffer_element_count)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L = random_vec(uniform(1, 4), 1, 16);
        const auto B = random_vec(uniform(1, 3), 1, 5);
        SCOPED_TRACE("lengths " + to_str(L) + " batches " + to_str(B));

        const auto packed = data_set_t::default_full_range(L, B);
        const auto padded = data_set_t::default_full_range(L, B, true);
        const auto n      = L[0];
        const auto P      = 2 * (n / 2 + 1);
        const auto N      = packed.buffer_element_count();
        EXPECT_EQ(N, packed.logical_count());
        EXPECT_EQ(padded.logical_count(), packed.logical_count());
        EXPECT_EQ(padded.buffer_element_count(), P * (N / n) - (P - n));
    }
}

// A data set is contiguous iff its elements fill a buffer without gaps: taken by increasing
// stride, each non-unit-span axis must be strided by the number of elements spanned by the
// axes before it (1 for the first one), whatever the axis order.
// contiguous_strides_and_distances() returns the data set's own strides if it is contiguous,
// and packed strides in natural axis order otherwise.
TEST_F(data_set_test, contiguous_strides)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // random spans (some of them 1) laid out in random order: packed in about half the
        // cases, otherwise with a possibly strided first axis and random padding between
        // axes (which may still happen to be packed)
        const auto          len_rank   = uniform(1, 4);
        const auto          batch_rank = uniform(1, 2);
        const auto          rank       = len_rank + batch_rank;
        const auto          spans      = random_vec(rank, 1, 7);
        const bool          packed     = uniform(0, 1);
        std::vector<size_t> strides(rank);
        size_t              next = packed ? 1 : uniform(1, 2);
        for(auto axis : random_permutation(rank))
        {
            strides[axis] = next;
            next          = next * spans[axis] + (packed ? 0 : uniform(0, 3));
        }
        const data_set_t data_set(std::vector<size_t>(rank, 0), spans, strides, batch_rank);
        SCOPED_TRACE("data set " + data_set.str());

        std::vector<size_t> non_unit_axes;
        for(size_t d = 0; d < rank; d++)
            if(spans[d] > 1)
                non_unit_axes.push_back(d);
        std::sort(non_unit_axes.begin(), non_unit_axes.end(), [&](size_t a, size_t b) {
            return strides[a] < strides[b];
        });
        bool   contiguous = true;
        size_t count      = 1;
        for(auto d : non_unit_axes)
        {
            contiguous &= strides[d] == count;
            count *= spans[d];
        }

        std::vector<size_t> packed_strides(rank, 1);
        for(size_t d = 1; d < rank; d++)
            packed_strides[d] = packed_strides[d - 1] * spans[d - 1];

        EXPECT_EQ(data_set.is_contiguous(), contiguous);
        EXPECT_EQ(data_set.contiguous_strides_and_distances(),
                  contiguous ? strides : packed_strides);
    }
}

TEST_F(data_set_test, offset_in)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // outer: random sub-range, along every axis (batch axes included), of a random
        // full-range data set, so its lower bounds are generally not 0; inner: random
        // sub-range of outer, with arbitrary strides since offset_in only uses outer's strides
        const auto          full         = random_full_range();
        const auto          rank         = full.get_full_rank();
        const auto          full_upper   = full.upper();
        const auto          outer_stride = full.strides_and_distances();
        std::vector<size_t> outer_lo(rank), outer_hi(rank), inner_lo(rank), inner_hi(rank);
        for(size_t d = 0; d < rank; d++)
        {
            outer_lo[d] = uniform(0, full_upper[d] - 1);
            outer_hi[d] = uniform(outer_lo[d] + 1, full_upper[d]);
            inner_lo[d] = uniform(outer_lo[d], outer_hi[d] - 1);
            inner_hi[d] = uniform(inner_lo[d] + 1, outer_hi[d]);
        }
        const data_set_t outer(outer_lo, outer_hi, outer_stride, full.get_batch_rank());
        const data_set_t inner(inner_lo, inner_hi, random_vec(rank, 1, 100), full.get_batch_rank());
        SCOPED_TRACE("outer " + outer.str() + " / inner " + inner.str());

        size_t expected = 0;
        for(size_t d = 0; d < rank; d++)
            expected += outer_stride[d] * (inner_lo[d] - outer_lo[d]);
        EXPECT_EQ(inner.offset_in(outer), expected);

        // first element outside outer along one random axis (below or above)
        auto       bad_lo = inner_lo;
        auto       bad_hi = inner_hi;
        const auto d      = uniform(0, rank - 1);
        bad_lo[d]         = (outer_lo[d] > 0 && uniform(0, 1)) ? outer_lo[d] - 1 : outer_hi[d];
        bad_hi[d]         = bad_lo[d] + 1;
        const data_set_t outside(bad_lo, bad_hi, outer_stride, full.get_batch_rank());
        EXPECT_THROW(outside.offset_in(outer), std::invalid_argument);
    }

    // data sets must have the same numbers of length and batch axes
    const auto data_set_2d = data_set_t::full_range({8, 8}, {1, 8}, 1, 64);
    const auto data_set_1d = data_set_t::full_range({8}, {1}, 1, 8);
    EXPECT_THROW(data_set_1d.offset_in(data_set_2d), std::invalid_argument);

    // even with the same total number of axes
    const data_set_t data_set_1d_two_batches({0, 0, 0}, {8, 8, 1}, {1, 8, 64}, 2);
    EXPECT_THROW(data_set_1d_two_batches.offset_in(data_set_2d), std::invalid_argument);
}

// Any sub-range along the slowest-varying axis of a data set is one unbroken run of it.
// Unit-length axes do not vary, so they are never the slowest-varying axis, whatever
// their stride (e.g. a trailing batch axis of length 1).
TEST_F(data_set_test, continuous_chunk)
{
    // 4x8 transform, batch 1: rows 2-4 of the length-8 axis are one unbroken run
    const auto       batch_of_one = data_set_t::default_full_range({4, 8}, 1);
    const data_set_t rows({0, 2, 0}, {4, 5, 1}, {1, 4, 32});
    EXPECT_TRUE(rows.is_continuous_in(batch_of_one));

    for(size_t i = 0; i < num_random_cases; i++)
    {
        // random spans (some of them 1) laid out in random order, with random padding;
        // the axis of largest stride is made unit-length in about half the cases
        const auto len_rank   = uniform(1, 4);
        const auto batch_rank = uniform(1, 2);
        const auto rank       = len_rank + batch_rank;
        auto       spans      = random_vec(rank, 1, 7);
        const auto order      = random_permutation(rank);
        if(uniform(0, 1))
            spans[order.back()] = 1;
        std::vector<size_t> strides(rank);
        size_t              next = uniform(1, 2);
        for(auto axis : order)
        {
            strides[axis] = next;
            next          = next * spans[axis] + uniform(0, 3);
        }
        const data_set_t whole(std::vector<size_t>(rank, 0), spans, strides, batch_rank);
        SCOPED_TRACE("whole " + whole.str());

        // slowest-varying axis: the last non-unit axis laid out above (axis 0 if none)
        size_t slow = 0;
        for(auto axis : order)
            if(spans[axis] > 1)
                slow = axis;

        auto lo  = whole.lower();
        auto hi  = whole.upper();
        lo[slow] = uniform(0, hi[slow] - 1);
        hi[slow] = uniform(lo[slow] + 1, hi[slow]);
        const data_set_t chunk(lo, hi, strides, batch_rank);
        EXPECT_TRUE(chunk.is_continuous_in(whole)) << chunk.str();

        // a chunk running past the end of the slowest-varying axis is not continuous
        auto past_hi  = hi;
        past_hi[slow] = spans[slow] + uniform(1, 3);
        const data_set_t overflow(lo, past_hi, strides, batch_rank);
        EXPECT_FALSE(overflow.is_continuous_in(whole)) << overflow.str();

        // a different stride along the slowest axis only matters if the chunk spans it
        auto slow_restrided = strides;
        slow_restrided[slow] += 1;
        const data_set_t slow_chunk(lo, hi, slow_restrided, batch_rank);
        EXPECT_EQ(slow_chunk.is_continuous_in(whole), hi[slow] - lo[slow] == 1) << slow_chunk.str();

        // strides along unit-length axes are irrelevant
        auto unit_restrided = strides;
        for(size_t d = 0; d < rank; d++)
            if(spans[d] == 1)
                unit_restrided[d] += uniform(1, 5);
        const data_set_t unit_chunk(lo, hi, unit_restrided, batch_rank);
        EXPECT_TRUE(unit_chunk.is_continuous_in(whole)) << unit_chunk.str();

        // narrowing or restriding any other non-unit axis breaks continuity
        std::vector<size_t> others;
        for(size_t d = 0; d < rank; d++)
            if(d != slow && spans[d] > 1)
                others.push_back(d);
        if(others.empty())
            continue;
        const auto d = others[uniform(0, others.size() - 1)];

        auto gap_hi = hi;
        gap_hi[d] -= 1;
        const data_set_t gap(lo, gap_hi, strides, batch_rank);
        EXPECT_FALSE(gap.is_continuous_in(whole)) << gap.str();

        auto other_restrided = strides;
        other_restrided[d] += 1;
        const data_set_t other_chunk(lo, hi, other_restrided, batch_rank);
        EXPECT_FALSE(other_chunk.is_continuous_in(whole)) << other_chunk.str();
    }
}

TEST_F(data_set_test, same_logical_range)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto          full       = random_full_range();
        const auto          rank       = full.get_full_rank();
        const auto          batch_rank = full.get_batch_rank();
        const auto          upper      = full.upper();
        std::vector<size_t> lo(rank), hi(rank);
        for(size_t d = 0; d < rank; d++)
        {
            lo[d] = uniform(0, upper[d] - 1);
            hi[d] = uniform(lo[d] + 1, upper[d]);
        }
        const data_set_t a(lo, hi, full.strides_and_distances(), batch_rank);
        SCOPED_TRACE("a " + a.str());

        // strides are irrelevant
        const data_set_t b(lo, hi, random_vec(rank, 1, 100), batch_rank);
        EXPECT_TRUE(a.has_same_logical_range_as(b)) << b.str();
        EXPECT_TRUE(b.has_same_logical_range_as(a)) << b.str();

        // moving either bound of any one axis, length or batch, makes the ranges differ
        const auto d = uniform(0, rank - 1);
        if(uniform(0, 1))
            lo[d] += 1;
        else
            hi[d] += uniform(1, 3);
        const data_set_t c(lo, hi, full.strides_and_distances(), batch_rank);
        EXPECT_FALSE(a.has_same_logical_range_as(c)) << c.str();
        EXPECT_FALSE(c.has_same_logical_range_as(a)) << c.str();
    }

    // same full rank, but a different split between length and batch axes
    const data_set_t one_batch({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 1);
    const data_set_t two_batches({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 2);
    EXPECT_THROW(one_batch.has_same_logical_range_as(two_batches), std::invalid_argument);
}

TEST_F(data_set_test, intersection)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto          full  = random_full_range();
        const auto          rank  = full.get_full_rank();
        const auto          upper = full.upper();
        std::vector<size_t> lo[2], hi[2];
        for(size_t k = 0; k < 2; k++)
        {
            lo[k].resize(rank);
            hi[k].resize(rank);
            for(size_t d = 0; d < rank; d++)
            {
                lo[k][d] = uniform(0, upper[d]);
                hi[k][d] = uniform(lo[k][d], upper[d]);
            }
        }
        const data_set_t first(lo[0], hi[0], full.strides_and_distances(), full.get_batch_rank());
        const data_set_t second(lo[1], hi[1], random_vec(rank, 1, 100), full.get_batch_rank());
        SCOPED_TRACE("first " + first.str() + " / second " + second.str());

        std::vector<size_t> exp_lo(rank), exp_hi(rank), exp_stride(rank);
        size_t              count = 1;
        for(size_t d = 0; d < rank; d++)
        {
            exp_lo[d]     = std::max(lo[0][d], lo[1][d]);
            exp_hi[d]     = std::max(std::min(hi[0][d], hi[1][d]), exp_lo[d]);
            exp_stride[d] = count;
            count *= exp_hi[d] - exp_lo[d];
        }
        const auto overlap = data_set_t::make_contiguous_intersection_of(first, second);
        EXPECT_EQ(overlap.lower(), exp_lo);
        EXPECT_EQ(overlap.upper(), exp_hi);
        EXPECT_EQ(overlap.logical_count(), count);
        EXPECT_EQ(overlap.is_empty(), count == 0);
        EXPECT_EQ(overlap.strides_and_distances(), exp_stride);
    }

    // arguments must have the same length/batch split...
    const auto       full_3d = data_set_t::default_full_range({4, 8, 2}, 3);
    const data_set_t two_batches({0, 0, 0, 0}, {4, 8, 2, 3}, {1, 4, 32, 64}, 2);
    EXPECT_THROW(data_set_t::make_contiguous_intersection_of(full_3d, two_batches),
                 std::invalid_argument);

    // ...and the same embedding (if any): views of different length axes, or an embedded
    // view and a plain data set with the same axes, are rejected
    const auto       innermost = full_3d.get_data_set_for_len_axes({0});
    const auto       middle    = full_3d.get_data_set_for_len_axes({1});
    const data_set_t plain_middle(
        middle.lower(), middle.upper(), middle.strides_and_distances(), middle.get_batch_rank());
    EXPECT_THROW(data_set_t::make_contiguous_intersection_of(innermost, middle),
                 std::invalid_argument);
    EXPECT_THROW(data_set_t::make_contiguous_intersection_of(middle, plain_middle),
                 std::invalid_argument);

    // a shared embedding is kept in the result
    const auto middle_overlap = data_set_t::make_contiguous_intersection_of(middle, middle);
    EXPECT_TRUE(middle_overlap.is_embedded());
    EXPECT_EQ(middle_overlap.corresponding_axes_in_embedding(), (std::vector<size_t>{1}));
}

// Complex in-place transforms use the same data set on both sides: the counterpart
// must be an exact copy, whatever the data set.
TEST_F(data_set_test, inplace_complex_is_unchanged)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // Batch axes may be partial ranges (unlike length axes), so cut a random
        // sub-range along them. Then keep a random subset of the length axes, which
        // gives an embedded data set unless every length axis is kept: the embedding
        // map must survive the copy too. The data set gets a random array type of the
        // complex side (unset included) and a random precision.
        const auto full = random_full_range();
        auto       lo   = full.lower();
        auto       hi   = full.upper();
        for(size_t d = full.get_len_rank(); d < full.get_full_rank(); d++)
        {
            lo[d] = uniform(0, hi[d] - 1);
            hi[d] = uniform(lo[d] + 1, hi[d]);
        }
        auto array_type = random_array_type();
        while(array_type == rocfft_array_type_real)
            array_type = random_array_type();
        const auto batch_range
            = with_type(data_set_t(lo, hi, full.strides_and_distances(), full.get_batch_rank()),
                        array_type,
                        random_precision());
        auto             len_axes = random_permutation(full.get_len_rank());
        len_axes.resize(uniform(1, len_axes.size()));
        const auto data_set
            = batch_range.get_data_set_for_len_axes({len_axes.begin(), len_axes.end()});

        // the same copy is expected for either direction and either side: draw one
        // combination per case rather than checking all four every time
        const auto fft_type = uniform(0, 1) ? rocfft_transform_type_complex_forward
                                            : rocfft_transform_type_complex_inverse;
        const auto io       = uniform(0, 1) ? io_data_label::INPUT : io_data_label::OUTPUT;
        SCOPED_TRACE("data set " + data_set.str() + " / fft_type " + std::to_string(fft_type)
                     + " / io " + (io == io_data_label::INPUT ? "INPUT" : "OUTPUT"));

        const auto other = data_set.get_other_inplace_data_set_for(io, fft_type);
        ASSERT_TRUE(other.has_value());
        EXPECT_TRUE(*other == data_set) << other->str();
        // operator== ignores the embedding map, so compare it separately
        EXPECT_TRUE(other->get_embedding_data_set() == data_set.get_embedding_data_set());
        EXPECT_EQ(other->corresponding_axes_in_embedding(),
                  data_set.corresponding_axes_in_embedding());
    }
}

// real -> Hermitian -> real gives back the original data set, for both real
// transform directions and for even and odd innermost lengths; the wrong parity
// flag must not.
TEST_F(data_set_test, inplace_real_round_trip)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L    = random_vec(uniform(1, 4), 1, 33);
        const auto B    = random_vec(uniform(1, 3), 1, 5);
        const bool odd  = L[0] % 2 == 1;
        // real (or unset) array type and random precision
        const bool typed     = uniform(0, 1);
        const auto precision = random_precision();
        const auto herm_type
            = typed ? rocfft_array_type_hermitian_interleaved : rocfft_array_type_unset;
        const auto full = with_type(data_set_t::default_full_range(L, B, true),
                                    typed ? rocfft_array_type_real : rocfft_array_type_unset,
                                    precision);
        // Multi-device plans also derive in-place data sets for embedded (lower-dimensional)
        // views, which must keep the embedding's innermost axis for real transforms (see
        // plan.cpp). Keep axis 0 and a random subset of the other length axes; keeping all
        // of them gives the plain, non-embedded data set.
        std::set<size_t> kept{0};
        for(size_t d = 1; d < L.size(); d++)
            if(uniform(0, 1))
                kept.insert(d);
        const auto real = full.get_data_set_for_len_axes(kept);
        SCOPED_TRACE("full " + full.str() + " / kept " + to_str({kept.begin(), kept.end()}));

        for(auto [fft_type, herm_io] :
            {std::pair{rocfft_transform_type_real_forward, io_data_label::OUTPUT},
             std::pair{rocfft_transform_type_real_inverse, io_data_label::INPUT}})
        {
            const auto real_io = other(herm_io);
            const auto herm    = real.get_other_inplace_data_set_for(herm_io, fft_type);
            ASSERT_TRUE(herm.has_value());
            EXPECT_EQ(herm->lengths()[0], L[0] / 2 + 1);
            EXPECT_EQ(type_of(*herm), herm_type);
            EXPECT_EQ(precision_of(*herm), precision);

            // the Hermitian side of the view is the view of the Hermitian side of the whole,
            // except for the array type: a view of fewer length axes is not Hermitian itself
            const auto herm_full = full.get_other_inplace_data_set_for(herm_io, fft_type);
            ASSERT_TRUE(herm_full.has_value());
            const auto herm_full_view = herm_full->get_data_set_for_len_axes(kept);
            EXPECT_TRUE(herm->has_same_axes_as(herm_full_view)) << herm->str();
            EXPECT_EQ(type_of(herm_full_view),
                      kept.size() == L.size() ? herm_type : complex_counterpart(herm_type));
            EXPECT_EQ(precision_of(herm_full_view), precision);
            // and so is the embedding of a view
            const auto herm_embedding = herm->get_embedding_data_set();
            EXPECT_TRUE(herm_embedding.has_same_axes_as(*herm_full)) << herm_embedding.str();
            EXPECT_EQ(type_of(herm_embedding),
                      herm->is_embedded() ? complex_counterpart(herm_type) : herm_type);
            EXPECT_EQ(herm->corresponding_axes_in_embedding(),
                      real.corresponding_axes_in_embedding());

            const auto back = herm->get_other_inplace_data_set_for(real_io, fft_type, odd);
            ASSERT_TRUE(back.has_value());
            EXPECT_TRUE(*back == real) << back->str();
            // operator== ignores the embedding map, so compare it separately
            EXPECT_TRUE(back->get_embedding_data_set() == full) << back->str();

            // a Hermitian innermost length of 1 only comes from a real one of 1
            if(L[0] == 1)
            {
                EXPECT_THROW(herm->get_other_inplace_data_set_for(real_io, fft_type, !odd),
                             std::invalid_argument);
                continue;
            }
            const auto wrong = herm->get_other_inplace_data_set_for(real_io, fft_type, !odd);
            ASSERT_TRUE(wrong.has_value());
            EXPECT_FALSE(*wrong == real) << wrong->str();
        }
    }
}

// A Hermitian data set has an in-place real counterpart iff it is packed along its
// innermost axis (and is not self-aliased). A real data set has an in-place Hermitian
// counterpart iff:
// - it is packed along its innermost axis;
// - every other non-unit-span axis has an even stride (a whole number of complex
//   interleaved values);
// - neither it nor its Hermitian interleaved counterpart is self-aliased.
// The data sets below are built so that they are never self-aliased, and neither are
// their counterparts if the first two conditions hold and every other non-unit-span
// axis is strided by no less than an innermost row padded to 2 * (n / 2 + 1) reals.
// That per-axis rule is what the library currently checks instead of robust
// self-aliasing verification. The expected acceptance below restates it which
// weakens the test: That weakness is a temporary artifact which will be resolved
// once self-aliasing detection is implemented in the library.
TEST_F(data_set_test, inplace_real_counterpart_rules)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto len_rank   = uniform(1, 3);
        const auto batch_rank = uniform(1, 2);
        const auto rank       = len_rank + batch_rank;
        const auto spans      = random_vec(rank, 1, 6);
        const auto n          = spans[0];
        const auto P          = 2 * (n / 2 + 1);

        // Axes are nested so that the data set is not self-aliased: axis 0 first (mostly
        // packed), then the other non-unit-span axes in random order, each strided by the
        // extent of the axes before it plus up to 3 elements. Rows count as P elements in
        // that extent, so that the counterpart of a real data set is not self-aliased
        // either if the stride rules hold. Only the first of those axes can thus be
        // strided below P, while any of them can have an odd stride. Unit-span axes span
        // nothing and get any stride.
        std::vector<size_t> strides(rank);
        strides[0]    = uniform(0, 3) ? 1 : uniform(2, 4);
        size_t extent = (n - 1) * strides[0] + 1;
        auto   order  = random_permutation(rank);
        order.erase(std::find(order.begin(), order.end(), 0));
        for(auto d : order)
        {
            if(spans[d] == 1)
            {
                strides[d] = uniform(1, 3 * P);
                continue;
            }
            strides[d] = extent + uniform(0, 3);
            extent     = std::max(extent, P) + (spans[d] - 1) * strides[d];
        }
        const data_set_t untyped(std::vector<size_t>(rank, 0), spans, strides, batch_rank);

        // query the counterpart of either side, for either real transform
        const auto fft_type = uniform(0, 1) ? rocfft_transform_type_real_forward
                                            : rocfft_transform_type_real_inverse;
        const auto herm_io  = fft_type == rocfft_transform_type_real_forward ? io_data_label::OUTPUT
                                                                             : io_data_label::INPUT;
        const bool from_real = uniform(0, 1);
        const auto other_io  = from_real ? herm_io : other(herm_io);
        const bool odd       = uniform(0, 1);
        // random array type of the data set's side (unset included) and random precision
        const bool typed     = uniform(0, 1);
        const auto precision = random_precision();
        const auto data_set  = with_type(untyped,
                                        !typed      ? rocfft_array_type_unset
                                         : from_real ? rocfft_array_type_real
                                                     : rocfft_array_type_hermitian_interleaved,
                                        precision);
        SCOPED_TRACE("data set " + data_set.str() + " / fft_type " + std::to_string(fft_type)
                     + (from_real ? " / real side" : " / Hermitian side"));

        bool follows_rules = spans[0] == 1 || strides[0] == 1;
        if(from_real)
            for(size_t d = 1; d < rank; d++)
                if(spans[d] > 1)
                    follows_rules &= strides[d] % 2 == 0 && strides[d] >= P;

        // expected counterpart: innermost length and strides in units of its own elements
        auto other_spans   = spans;
        auto other_strides = strides;
        other_spans[0]     = from_real ? n / 2 + 1 : 2 * (n - 1) + (odd ? 1 : 0);
        for(size_t d = 1; d < rank; d++)
            other_strides[d] = from_real ? strides[d] / 2 : 2 * strides[d];
        const auto expected = with_type(
            data_set_t(std::vector<size_t>(rank, 0), other_spans, other_strides, batch_rank),
            !typed      ? rocfft_array_type_unset
            : from_real ? rocfft_array_type_hermitian_interleaved
                        : rocfft_array_type_real,
            precision);

        // a Hermitian innermost length of 1 only comes from a real one of 1
        if(!from_real && n == 1 && !odd)
        {
            EXPECT_THROW(data_set.get_other_inplace_data_set_for(other_io, fft_type, odd),
                         std::invalid_argument);
            continue;
        }
        const auto counterpart = data_set.get_other_inplace_data_set_for(other_io, fft_type, odd);
        ASSERT_EQ(counterpart.has_value(), follows_rules);
        if(counterpart)
        {
            EXPECT_TRUE(*counterpart == expected)
                << counterpart->str() << " instead of " << expected.str();
        }
    }
}

TEST_F(data_set_test, inplace_query_rejects_bad_arguments)
{
    auto data_set = data_set_t::default_full_range({8}, 1);
    EXPECT_THROW(data_set.get_other_inplace_data_set_for(io_data_label::OUTPUT,
                                                         static_cast<rocfft_transform_type>(99)),
                 std::invalid_argument);

    // a data set covering nothing has no counterpart
    data_set_t empty({0, 0}, {0, 1}, {1, 1});
    EXPECT_THROW(empty.get_other_inplace_data_set_for(io_data_label::OUTPUT,
                                                      rocfft_transform_type_real_forward),
                 std::logic_error);

    // a Hermitian innermost length of 1 only comes from a real innermost length of 1, so
    // the real side cannot be requested as even (it would have a length of 0)
    const auto herm_of_one = data_set_t::default_full_range({1}, 3);
    EXPECT_THROW(herm_of_one.get_other_inplace_data_set_for(
                     io_data_label::INPUT, rocfft_transform_type_real_forward, false),
                 std::invalid_argument);
    const auto real_of_one = herm_of_one.get_other_inplace_data_set_for(
        io_data_label::INPUT, rocfft_transform_type_real_forward, true);
    ASSERT_TRUE(real_of_one.has_value());
    EXPECT_EQ(real_of_one->lengths(), (std::vector<size_t>{1}));

    // length axes must be full, batch axes need not be
    for(auto fft_type : {rocfft_transform_type_complex_forward, rocfft_transform_type_real_forward})
    {
        data_set_t partial_len({0, 1, 0}, {8, 4, 3}, {1, 10, 40});
        EXPECT_THROW(partial_len.get_other_inplace_data_set_for(io_data_label::OUTPUT, fft_type),
                     std::logic_error);
        data_set_t partial_batch({0, 0, 1}, {8, 4, 3}, {1, 10, 40});
        EXPECT_NO_THROW(
            partial_batch.get_other_inplace_data_set_for(io_data_label::OUTPUT, fft_type));
    }

    // A set array type must match the data set's side of the transform. Accepted array
    // types per side, in all_array_types order:
    const bool side_accepts[3][6] = {
        //unset CI    CP     R      HI     HP
        {true, true, true, false, true, true}, // complex
        {true, false, false, true, false, false}, // real
        {true, false, false, false, true, true}, // Hermitian
    };
    for(auto fft_type : {rocfft_transform_type_complex_forward,
                         rocfft_transform_type_complex_inverse,
                         rocfft_transform_type_real_forward,
                         rocfft_transform_type_real_inverse})
    {
        for(auto other_io : {io_data_label::INPUT, io_data_label::OUTPUT})
        {
            const auto   io   = other(other_io);
            const size_t side = is_real_domain(fft_type, io)        ? 1
                                : is_hermitian_domain(fft_type, io) ? 2
                                                                    : 0;
            for(size_t t = 0; t < std::size(all_array_types); t++)
            {
                const auto data_set = with_type(data_set_t::default_full_range({8}, 3, side == 1),
                                                all_array_types[t],
                                                rocfft_precision_single);
                SCOPED_TRACE("data set " + data_set.str() + " / fft_type "
                             + std::to_string(fft_type) + " / other_io " + to_str(other_io));
                const auto query = outcome_of(
                    [&] { data_set.get_other_inplace_data_set_for(other_io, fft_type); });
                EXPECT_EQ(query, side_accepts[side][t] ? Y : L);
            }
        }
    }
}

TEST_F(data_set_test, axes_by_increasing_stride)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // distinct strides in random axis order
        const auto       rank = uniform(1, 6);
        std::set<size_t> distinct;
        while(distinct.size() < rank)
            distinct.insert(uniform(1, 1000));
        std::vector<size_t> strides(distinct.begin(), distinct.end());
        std::shuffle(strides.begin(), strides.end(), rng);
        const auto data_set
            = data_set_t::full_range(random_vec(rank, 1, 8), strides, 1, 1 + *distinct.rbegin());
        SCOPED_TRACE("strides " + to_str(strides));

        std::map<size_t, size_t> axis_by_stride, rest_by_stride;
        for(size_t d = 0; d < rank; d++)
        {
            axis_by_stride[strides[d]] = d;
            if(d > 0)
                rest_by_stride[strides[d]] = d;
        }
        std::vector<size_t> unpinned, pinned{0};
        for(auto [stride, d] : axis_by_stride)
            unpinned.push_back(d);
        for(auto [stride, d] : rest_by_stride)
            pinned.push_back(d);

        EXPECT_EQ(data_set.length_axes_by_increasing_strides(false), unpinned);
        EXPECT_EQ(data_set.length_axes_by_increasing_strides(true), pinned);
    }
}

TEST_F(data_set_test, embedding_round_trip)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // random array type and precision, both possibly unset: views of fewer length
        // axes (and their embeddings) of a Hermitian data set are not Hermitian themselves
        const auto array_type = random_array_type();
        const auto precision  = random_precision();
        const auto L          = with_type(random_full_range(), array_type, precision);
        const auto L_complex  = with_type(L, complex_counterpart(array_type), precision);
        const auto len_rank   = L.get_len_rank();
        SCOPED_TRACE("L " + L.str());

        // random non-empty subset S of L's length axes
        auto S_vec = random_permutation(len_rank);
        S_vec.resize(uniform(1, len_rank));
        const std::set<size_t>    S(S_vec.begin(), S_vec.end());
        const std::vector<size_t> sorted_S(S.begin(), S.end());
        SCOPED_TRACE("S " + to_str(sorted_S));

        auto       sub       = L.get_data_set_for_len_axes(S);
        const bool S_is_full = S.size() == len_rank;
        EXPECT_EQ(type_of(sub), S_is_full ? array_type : complex_counterpart(array_type));
        EXPECT_EQ(precision_of(sub), precision);
        EXPECT_TRUE(sub.get_embedding_data_set() == (S_is_full ? L : L_complex));
        EXPECT_EQ(sub.corresponding_axes_in_embedding(), sorted_S);

        // subset of a subset still maps back to L
        auto S2_vec = random_permutation(sub.get_len_rank());
        S2_vec.resize(uniform(1, sub.get_len_rank()));
        const std::set<size_t> S2(S2_vec.begin(), S2_vec.end());
        const auto             subsub = sub.get_data_set_for_len_axes(S2);
        std::vector<size_t>    subsub_axes;
        for(auto j : S2)
            subsub_axes.push_back(sorted_S[j]);
        const bool subsub_is_full = S_is_full && S2.size() == len_rank;
        EXPECT_EQ(type_of(subsub), subsub_is_full ? array_type : complex_counterpart(array_type));
        EXPECT_EQ(precision_of(subsub), precision);
        EXPECT_TRUE(subsub.get_embedding_data_set() == (subsub_is_full ? L : L_complex));
        EXPECT_EQ(subsub.corresponding_axes_in_embedding(), subsub_axes);

        // Reordering keeps the embedding consistent only for embedded data sets;
        // selecting every axis returns a plain copy without an embedding map.
        if(S_is_full)
            continue;
        ASSERT_TRUE(sub.is_embedded());
        const auto order = random_permutation(sub.get_len_rank());
        SCOPED_TRACE("order " + to_str(order));
        data_set_test_access::reorder_length_axes(sub, order);
        std::vector<size_t> permuted_axes;
        for(auto j : order)
            permuted_axes.push_back(sorted_S[j]);
        EXPECT_TRUE(sub.get_embedding_data_set() == L_complex);
        EXPECT_EQ(sub.corresponding_axes_in_embedding(), permuted_axes);

        // and a subset of the reordered subset still maps back to L
        const auto          sub3 = sub.get_data_set_for_len_axes(S2);
        std::vector<size_t> sub3_axes;
        for(auto j : S2)
            sub3_axes.push_back(permuted_axes[j]);
        EXPECT_TRUE(sub3.get_embedding_data_set() == L_complex);
        EXPECT_EQ(sub3.corresponding_axes_in_embedding(), sub3_axes);
    }
}

TEST_F(data_set_test, subset_of_axes_rejects_bad_input)
{
    auto data_set = data_set_t::default_full_range({4, 8, 2}, 3);
    EXPECT_THROW(data_set.get_data_set_for_len_axes({}), std::invalid_argument);
    EXPECT_THROW(data_set.get_data_set_for_len_axes({0, 7}), std::invalid_argument);
}

TEST_F(data_set_test, type_and_precision_unset_by_default)
{
    for(const auto& data_set : {data_set_t({0, 0}, {4, 3}, {1, 4}),
                                data_set_t::full_range({4}, {1}, 3, 4),
                                data_set_t::default_full_range({4, 8}, 3),
                                data_set_t::default_full_range({4, 8}, {3, 2}, true)})
    {
        SCOPED_TRACE(data_set.str());
        EXPECT_FALSE(data_set.has_array_type());
        EXPECT_THROW(data_set.get_array_type(), std::logic_error);
        EXPECT_FALSE(data_set.has_precision());
        EXPECT_THROW(data_set.get_precision(), std::logic_error);
        EXPECT_THROW(data_set.element_byte_size(), std::logic_error);
        EXPECT_THROW(data_set.buffer_size_bytes(), std::logic_error);
    }
}

// Expected outcomes for all pairs of array types (unset included), rows and columns in
// all_array_types order: enabling a pair is a one-entry change.
// reinterpret_as(to) on a data set of array type from (rows):
// relabeling only, real <-> complex not implemented yet
const outcome reinterpret_outcomes[6][6] = {
    //unset CI CP R  HI HP
    {I, I, I, I, I, I}, // unset
    {I, Y, I, N, Y, I}, // CI
    {I, I, Y, N, I, Y}, // CP
    {I, N, N, Y, N, N}, // R
    {I, Y, I, N, Y, I}, // HI
    {I, I, Y, N, I, Y}, // HP
};
// make_contiguous_intersection_of, offset_in, is_continuous_in and operator== relate two
// data sets describing the same data, so their array types must be identical (and so
// must their precisions)
const bool same_data_array_types[6][6] = {
    //unset  CI     CP     R      HI     HP
    {true, false, false, false, false, false}, // unset
    {false, true, false, false, false, false}, // CI
    {false, false, true, false, false, false}, // CP
    {false, false, false, true, false, false}, // R
    {false, false, false, false, true, false}, // HI
    {false, false, false, false, false, true}, // HP
};

TEST_F(data_set_test, array_type_pair_rules)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // random data set, embedded about half the time; random precisions, often equal
        const auto full     = random_full_range();
        auto       len_axes = random_permutation(full.get_len_rank());
        len_axes.resize(uniform(1, len_axes.size()));
        const auto data_set
            = uniform(0, 1) ? full.get_data_set_for_len_axes({len_axes.begin(), len_axes.end()})
                            : full;
        const auto prec_a = random_precision();
        const auto prec_b = uniform(0, 1) ? prec_a : random_precision();

        for(size_t ta = 0; ta < std::size(all_array_types); ta++)
        {
            for(size_t tb = 0; tb < std::size(all_array_types); tb++)
            {
                const auto type_a = all_array_types[ta];
                const auto type_b = all_array_types[tb];
                const auto a      = with_type(data_set, type_a, prec_a);
                const auto b      = with_type(data_set, type_b, prec_b);
                SCOPED_TRACE("a " + a.str() + " / b " + b.str());

                // only the array type changes; operator== ignores the embedding map, so
                // compare it separately
                const auto reinterpret = outcome_of([&] {
                    const auto relabeled = a.reinterpret_as(type_b);
                    const auto expected  = with_type(a, type_b, prec_a);
                    EXPECT_TRUE(relabeled == expected) << relabeled.str();
                    EXPECT_TRUE(relabeled.get_embedding_data_set()
                                == expected.get_embedding_data_set());
                    EXPECT_EQ(relabeled.corresponding_axes_in_embedding(),
                              a.corresponding_axes_in_embedding());
                });
                EXPECT_EQ(reinterpret, reinterpret_outcomes[ta][tb]);

                const bool same = same_data_array_types[ta][tb] && prec_a == prec_b;
                EXPECT_TRUE(a.has_same_axes_as(b));
                EXPECT_EQ(a == b, same);
                const auto intersection = outcome_of([&] {
                    const auto xsect = data_set_t::make_contiguous_intersection_of(a, b);
                    EXPECT_EQ(type_of(xsect), type_a);
                    EXPECT_EQ(precision_of(xsect), prec_a);
                });
                const auto offset       = outcome_of([&] { EXPECT_EQ(a.offset_in(b), 0u); });
                const auto continuous   = outcome_of([&] { EXPECT_TRUE(a.is_continuous_in(b)); });
                EXPECT_EQ(intersection, same ? Y : I);
                EXPECT_EQ(offset, same ? Y : I);
                EXPECT_EQ(continuous, same ? Y : I);
            }
        }
    }
}

TEST_F(data_set_test, type_and_precision_in_str)
{
    const auto base = data_set_t::default_full_range({8}, 2);
    EXPECT_NE(base.str().find("precision unset"), std::string::npos) << base.str();
    EXPECT_NE(base.str().find("array type NA"), std::string::npos) << base.str();

    const auto typed = with_type(base, rocfft_array_type_hermitian_planar, rocfft_precision_double);
    EXPECT_NE(typed.str().find("precision double"), std::string::npos) << typed.str();
    EXPECT_NE(typed.str().find("array type HP"), std::string::npos) << typed.str();
}

TEST_F(data_set_test, type_and_precision_survive_clear_and_reset)
{
    auto data_set = with_type(
        data_set_t::default_full_range({8, 4}, 2), rocfft_array_type_real, rocfft_precision_half);

    data_set_test_access::clear(data_set);
    EXPECT_EQ(data_set.get_full_rank(), 0u);
    EXPECT_EQ(data_set.get_array_type(), rocfft_array_type_real);
    EXPECT_EQ(data_set.get_precision(), rocfft_precision_half);

    data_set_test_access::full_range_reset(data_set, {16}, 3);
    EXPECT_EQ(data_set.lengths(), std::vector<size_t>{16});
    EXPECT_EQ(data_set.batches(), std::vector<size_t>{3});
    EXPECT_EQ(data_set.get_array_type(), rocfft_array_type_real);
    EXPECT_EQ(data_set.get_precision(), rocfft_precision_half);
}

TEST_F(data_set_test, element_and_buffer_sizes)
{
    const auto base = data_set_t::default_full_range({8, 4}, 2);
    ASSERT_EQ(base.buffer_element_count(), 64u);
    EXPECT_THROW(base.buffer_element_count(data_part::REAL), std::invalid_argument);

    for(auto precision : all_precisions)
    {
        for(auto array_type : all_array_types)
        {
            const auto data_set = with_type(base, array_type, precision);
            SCOPED_TRACE(data_set.str());
            if(array_type == rocfft_array_type_unset)
            {
                EXPECT_THROW(data_set.element_byte_size(), std::logic_error);
                EXPECT_THROW(data_set.buffer_size_bytes(), std::logic_error);
                continue;
            }
            const auto bytes = data_set.element_byte_size();
            EXPECT_EQ(bytes, ::element_size(precision, array_type));
            // planar data sets are counted per plane, others as a whole
            const bool planar = array_type_is_planar(array_type);
            for(auto part : {data_part::ALL, data_part::REAL, data_part::IMAGINARY})
            {
                if(planar == (part == data_part::ALL))
                {
                    EXPECT_THROW(data_set.buffer_element_count(part), std::invalid_argument);
                    EXPECT_THROW(data_set.buffer_size_bytes(part), std::invalid_argument);
                    continue;
                }
                EXPECT_EQ(data_set.buffer_element_count(part), 64u);
                EXPECT_EQ(data_set.buffer_size_bytes(part), 64 * bytes);
            }
        }
    }
    // 2/4/8 bytes per real component, doubled for interleaved complex types
    EXPECT_EQ(with_type(base, rocfft_array_type_complex_interleaved, rocfft_precision_half)
                  .element_byte_size(),
              4u);
    EXPECT_EQ(with_type(base, rocfft_array_type_real, rocfft_precision_double).element_byte_size(),
              8u);
    EXPECT_EQ(with_type(base, rocfft_array_type_hermitian_planar, rocfft_precision_single)
                  .element_byte_size(),
              4u);

    // padding inside the buffer is counted, trailing padding is not
    const auto padded = with_type(data_set_t::default_full_range({8}, 2, true),
                                  rocfft_array_type_real,
                                  rocfft_precision_single);
    EXPECT_EQ(padded.buffer_size_bytes(), (10 + 8) * 4u);

    // an array type without a precision
    const auto no_precision = with_type(base, rocfft_array_type_real, std::nullopt);
    EXPECT_THROW(no_precision.element_byte_size(), std::logic_error);
    EXPECT_THROW(no_precision.buffer_size_bytes(), std::logic_error);
}

TEST_F(data_set_test, packed_data_set)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // random sub-range along every axis of a random data set, embedded about half the
        // time, with random array type and precision
        const auto full = random_full_range();
        auto       lo   = full.lower();
        auto       hi   = full.upper();
        for(size_t d = 0; d < full.get_full_rank(); d++)
        {
            lo[d] = uniform(0, hi[d] - 1);
            hi[d] = uniform(lo[d] + 1, hi[d]);
        }
        auto data_set
            = with_type(data_set_t(lo, hi, full.strides_and_distances(), full.get_batch_rank()),
                        random_array_type(),
                        random_precision());
        if(uniform(0, 1))
        {
            auto len_axes = random_permutation(full.get_len_rank());
            len_axes.resize(uniform(1, len_axes.size()));
            data_set = data_set.get_data_set_for_len_axes({len_axes.begin(), len_axes.end()});
        }
        const bool padding = uniform(0, 1);
        SCOPED_TRACE("data set " + data_set.str() + " / padding " + std::to_string(padding));

        const auto packed = data_set.get_packed_data_set(padding);
        EXPECT_EQ(packed.get_len_rank(), data_set.get_len_rank());
        EXPECT_EQ(packed.lower(), data_set.lower());
        EXPECT_EQ(packed.upper(), data_set.upper());
        EXPECT_EQ(packed.strides_and_distances(),
                  data_set_t::default_full_range(data_set.lengths(), data_set.batches(), padding)
                      .strides_and_distances());
        EXPECT_EQ(type_of(packed), type_of(data_set));
        EXPECT_EQ(precision_of(packed), precision_of(data_set));
        EXPECT_FALSE(packed.is_embedded());
    }
}
