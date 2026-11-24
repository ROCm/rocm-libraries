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
#include "data_set_test_access.h"
#include "enum_printer.h"
#include <algorithm>
#include <array>
#include <cstdint>
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

    // (a function pointer would drop the default offsets)
    data_set_t with_type(const data_set_t&               data_set,
                         rocfft_array_type               array_type,
                         std::optional<rocfft_precision> precision,
                         const std::array<size_t, 2>&    offsets = {0, 0})
    {
        return data_set_test_access::with_type(data_set, array_type, precision, offsets);
    }

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
    // something not implemented yet, or any other std::logic_error; S marks expected
    // outcomes that depend on the data set (see the dedicated tests)
    enum outcome
    {
        Y,
        I,
        N,
        L,
        S
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

    // In-buffer offset (sum of stride * coordinate) of every logical index, axis 0 fastest.
    std::vector<size_t> all_offsets(const data_set_t& d)
    {
        std::vector<size_t> ret;
        if(d.is_empty())
            return ret;
        const auto lo     = d.lower();
        const auto hi     = d.upper();
        const auto stride = d.strides_and_distances();
        auto       idx    = lo;
        while(true)
        {
            size_t offset = 0;
            for(size_t a = 0; a < idx.size(); a++)
                offset += stride[a] * idx[a];
            ret.push_back(offset);
            size_t a = 0;
            for(; a < idx.size() && ++idx[a] == hi[a]; a++)
                idx[a] = lo[a];
            if(a == idx.size())
                return ret;
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
    data_set_t random_full_range(size_t min_span = 1)
    {
        const auto          len_rank   = uniform(1, 4);
        const auto          batch_rank = uniform(1, 2);
        const auto          spans      = random_vec(len_rank + batch_rank, min_span, 7);
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
        while(array_type == rocfft_array_type_real || array_type_is_hermitian(array_type))
            array_type = random_array_type();
        const auto batch_range
            = with_type(data_set_t(lo, hi, full.strides_and_distances(), full.get_batch_rank()),
                        array_type,
                        random_precision());
        auto len_axes = random_permutation(full.get_len_rank());
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
        const auto L   = random_vec(uniform(1, 4), 1, 33);
        const auto B   = random_vec(uniform(1, 3), 1, 5);
        const bool odd = L[0] % 2 == 1;
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
            EXPECT_TRUE(*herm == with_type(herm_full_view, herm_type, precision)) << herm->str();
            EXPECT_EQ(type_of(herm_full_view),
                      kept.size() == L.size() ? herm_type : complex_counterpart(herm_type));
            EXPECT_EQ(precision_of(herm_full_view), precision);
            // and so is the embedding of a view
            const auto herm_embedding = herm->get_embedding_data_set();
            EXPECT_TRUE(herm_embedding == with_type(*herm_full, type_of(herm_embedding), precision))
                << herm_embedding.str();
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
        {true, true, true, false, false, false}, // complex
        {true, false, false, true, false, false}, // real
        {true, false, false, false, true, false}, // Hermitian (interleaved only)
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
// relabeling; real <-> complex/Hermitian depends on the data set (see the
// dedicated reinterpret_real_* tests)
const outcome reinterpret_outcomes[6][6] = {
    //unset CI CP R  HI HP
    {I, I, I, I, I, I}, // unset
    {I, Y, I, S, Y, I}, // CI
    {I, I, Y, S, I, Y}, // CP
    {I, S, S, Y, S, S}, // R
    {I, Y, I, S, Y, I}, // HI
    {I, I, Y, S, I, Y}, // HP
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
                if(reinterpret_outcomes[ta][tb] != S)
                {
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
                }

                const bool same = same_data_array_types[ta][tb] && prec_a == prec_b;
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

namespace
{
    const rocfft_array_type interleaved_types[]
        = {rocfft_array_type_complex_interleaved, rocfft_array_type_hermitian_interleaved};
    const rocfft_array_type planar_types[]
        = {rocfft_array_type_complex_planar, rocfft_array_type_hermitian_planar};

    // in-buffer position (in elements, offset included) of logical coordinate `coord`
    size_t position_of(const data_set_t&          layout,
                       const std::vector<size_t>& coord,
                       data_part                  part = data_part::ALL)
    {
        const auto strides = layout.strides_and_distances();
        size_t     ret     = layout.get_offset(part);
        for(size_t d = 0; d < coord.size(); d++)
            ret += coord[d] * strides[d];
        return ret;
    }

    // calls f(coord) for every logical coordinate of a non-empty layout
    template <typename F>
    void for_each_coordinate(const data_set_t& layout, F f)
    {
        const auto lower = layout.lower();
        const auto upper = layout.upper();
        auto       coord = lower;
        while(true)
        {
            f(coord);
            size_t d = 0;
            for(; d < coord.size() && ++coord[d] == upper[d]; d++)
                coord[d] = lower[d];
            if(d == coord.size())
                return;
        }
    }
}

TEST_F(data_set_test, reinterpret_real_as_interleaved)
{
    struct layout_t
    {
        std::vector<size_t> lower, upper, strides;
        size_t              offset;
    };
    struct case_t
    {
        layout_t real, cplx;
        size_t   batch_rank;
    };
    // {real layout, expected complex layout, batch rank}: 1D, 2D with lower bounds
    // and an offset, 3D with two batch axes
    const case_t cases[]
        = {{{{0, 0}, {8, 3}, {1, 10}, 0}, {{0, 0}, {4, 3}, {1, 5}, 0}, 1},
           {{{2, 1, 0}, {8, 5, 3}, {1, 10, 60}, 4}, {{1, 1, 0}, {4, 5, 3}, {1, 5, 30}, 2}, 1},
           {{{0, 0, 0, 0, 0}, {6, 4, 2, 3, 2}, {1, 8, 32, 64, 192}, 6},
            {{0, 0, 0, 0, 0}, {3, 4, 2, 3, 2}, {1, 4, 16, 32, 96}, 3},
            2}};
    for(const auto& [r, c, batch_rank] : cases)
    {
        for(auto p : {rocfft_precision_single, rocfft_precision_double})
        {
            for(auto t : interleaved_types)
            {
                const auto real     = with_type(data_set_t(r.lower, r.upper, r.strides, batch_rank),
                                            rocfft_array_type_real,
                                            p,
                                            {r.offset, 0});
                const auto expected = with_type(
                    data_set_t(c.lower, c.upper, c.strides, batch_rank), t, p, {c.offset, 0});
                SCOPED_TRACE(real.str());
                const auto dst = real.reinterpret_as(t);
                EXPECT_TRUE(dst == expected) << dst.str();
                EXPECT_EQ(dst.get_precision(), p);
                EXPECT_TRUE(expected.reinterpret_as(rocfft_array_type_real) == real);
            }
        }
    }
}

TEST_F(data_set_test, reinterpret_real_interleaved_round_trip_and_positions)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // random interleaved layout (any lower bounds, strides and offset) with a
        // unit stride along length axis 0
        const auto len_rank   = uniform(1, 3);
        const auto batch_rank = uniform(1, 2);
        const auto lower      = random_vec(len_rank + batch_rank, 0, 3);
        auto       upper      = random_vec(len_rank + batch_rank, 1, 4);
        for(size_t d = 0; d < upper.size(); d++)
            upper[d] += lower[d];
        auto strides = random_vec(len_rank + batch_rank, 1, 40);
        strides[0]   = 1;
        const auto p = uniform(0, 1) ? rocfft_precision_single : rocfft_precision_double;
        const auto t = interleaved_types[uniform(0, 1)];
        const auto cplx
            = with_type(data_set_t(lower, upper, strides, batch_rank), t, p, {uniform(0, 9), 0});
        SCOPED_TRACE(cplx.str());

        const auto real = cplx.reinterpret_as(rocfft_array_type_real);
        SCOPED_TRACE(real.str());
        EXPECT_EQ(real.get_precision(), p);
        EXPECT_EQ(real.logical_count(), 2 * cplx.logical_count());
        EXPECT_TRUE(real.reinterpret_as(t) == cplx);
        for(auto t2 : interleaved_types)
            EXPECT_TRUE(real.reinterpret_as(t2).reinterpret_as(rocfft_array_type_real) == real);

        // real element 2k (resp. 2k + 1) is the real (resp. imaginary) part of
        // complex element k, at the same byte position
        for_each_coordinate(cplx, [&](const std::vector<size_t>& coord) {
            auto real_coord = coord;
            real_coord[0] *= 2;
            EXPECT_EQ(position_of(real, real_coord), 2 * position_of(cplx, coord));
            EXPECT_EQ(position_of(real, real_coord) * real.element_byte_size(),
                      position_of(cplx, coord) * cplx.element_byte_size());
            real_coord[0] += 1;
            EXPECT_EQ(position_of(real, real_coord), 2 * position_of(cplx, coord) + 1);
        });
    }
}

TEST_F(data_set_test, reinterpret_real_interleaved_rejects_bad_layouts)
{
    // 2D real layout with two batch axes, valid unless modified
    const std::vector<size_t> lower{0, 0, 0, 0}, upper{8, 4, 3, 2}, strides{1, 10, 40, 120};
    const auto                make = [&](rocfft_array_type          t,
                          const std::vector<size_t>& lo,
                          const std::vector<size_t>& up,
                          const std::vector<size_t>& st,
                          size_t                     offset) {
        return with_type(data_set_t(lo, up, st, 2), t, rocfft_precision_single, {offset, 0});
    };
    const auto with = [](std::vector<size_t> v, size_t idx, size_t value) {
        v[idx] = value;
        return v;
    };
    for(auto t : interleaved_types)
    {
        EXPECT_NO_THROW(make(rocfft_array_type_real, lower, upper, strides, 2).reinterpret_as(t));
        const data_set_t rejected[]
            = {// non-unit stride along length axis 0
               make(rocfft_array_type_real, lower, upper, with(strides, 0, 2), 2),
               // odd bounds along length axis 0
               make(rocfft_array_type_real, with(lower, 0, 1), upper, strides, 2),
               make(rocfft_array_type_real, lower, with(upper, 0, 7), strides, 2),
               // odd stride, odd distances
               make(rocfft_array_type_real, lower, upper, with(strides, 1, 11), 2),
               make(rocfft_array_type_real, lower, upper, with(strides, 2, 41), 2),
               make(rocfft_array_type_real, lower, upper, with(strides, 3, 121), 2),
               // odd offset
               make(rocfft_array_type_real, lower, upper, strides, 3)};
        for(const auto& real : rejected)
        {
            SCOPED_TRACE(real.str());
            EXPECT_THROW(real.reinterpret_as(t), std::invalid_argument);
        }

        // complex to real only requires a unit stride along length axis 0
        EXPECT_NO_THROW(make(t, {1, 0, 0, 0}, {7, 3, 3, 2}, {1, 11, 41, 121}, 3)
                            .reinterpret_as(rocfft_array_type_real));
        EXPECT_THROW(
            make(t, lower, upper, with(strides, 0, 2), 2).reinterpret_as(rocfft_array_type_real),
            std::invalid_argument);
    }
}

TEST_F(data_set_test, reinterpret_real_interleaved_embedded)
{
    const auto L = with_type(data_set_t({0, 0, 0, 0}, {8, 4, 6, 3}, {1, 10, 40, 240}),
                             rocfft_array_type_real,
                             rocfft_precision_double,
                             {4, 0});
    for(auto t : interleaved_types)
    {
        // length axis 0 of the sub-layout is L's: the embedding map is kept and
        // reinterpreting commutes with get_embedding_data_set() (whose Hermitian
        // results get the complex counterpart)
        const auto sub = L.get_data_set_for_len_axes({0, 2});
        ASSERT_TRUE(sub.is_embedded());
        const auto dst = sub.reinterpret_as(t);
        EXPECT_TRUE(dst.is_embedded());
        EXPECT_TRUE(dst.get_embedding_data_set()
                    == L.reinterpret_as(rocfft_array_type_complex_interleaved))
            << dst.str();
        EXPECT_EQ(dst.corresponding_axes_in_embedding(), sub.corresponding_axes_in_embedding());
        const auto back = dst.reinterpret_as(rocfft_array_type_real);
        EXPECT_TRUE(back == sub);
        EXPECT_TRUE(back.get_embedding_data_set() == L);

        // length axis 0 of the sub-layout is not L_t's, although it is otherwise
        // reinterpretable (unit stride, even bounds, even other strides and offset)
        const auto L_t = with_type(data_set_t({0, 0, 0, 0}, {4, 8, 6, 3}, {8, 1, 32, 192}),
                                   rocfft_array_type_real,
                                   rocfft_precision_double,
                                   {4, 0});
        EXPECT_THROW(L_t.get_data_set_for_len_axes({1}).reinterpret_as(t), std::invalid_argument);
        EXPECT_THROW(L_t.get_data_set_for_len_axes({1, 2}).reinterpret_as(t),
                     std::invalid_argument);
        const auto L_t_cplx = with_type(data_set_t({0, 0, 0, 0}, {4, 8, 6, 3}, {8, 1, 32, 192}),
                                        t,
                                        rocfft_precision_double,
                                        {3, 0});
        EXPECT_THROW(L_t_cplx.get_data_set_for_len_axes({1}).reinterpret_as(rocfft_array_type_real),
                     std::invalid_argument);
    }
}

TEST_F(data_set_test, reinterpret_real_as_planar)
{
    struct layout_t
    {
        std::vector<size_t>   lower, upper, strides;
        std::array<size_t, 2> offsets;
    };
    struct case_t
    {
        layout_t real, planar;
        size_t   batch_rank;
    };
    // {real layout, expected planar layout, batch rank}: 1D, 2D with lower bounds,
    // a non-unit stride along length axis 0 and an offset, 3D with two batch axes
    // and odd other strides and offset (unchanged)
    const case_t cases[]
        = {{{{0, 0}, {8, 3}, {1, 10}, {0, 0}}, {{0, 0}, {4, 3}, {2, 10}, {0, 1}}, 1},
           {{{2, 1, 0}, {8, 5, 3}, {3, 30, 200}, {5, 0}},
            {{1, 1, 0}, {4, 5, 3}, {6, 30, 200}, {5, 8}},
            1},
           {{{0, 0, 0, 0, 0}, {6, 4, 2, 3, 2}, {1, 7, 29, 61, 190}, {3, 0}},
            {{0, 0, 0, 0, 0}, {3, 4, 2, 3, 2}, {2, 7, 29, 61, 190}, {3, 4}},
            2}};
    for(const auto& [r, c, batch_rank] : cases)
    {
        for(auto p : {rocfft_precision_single, rocfft_precision_double})
        {
            for(auto t : planar_types)
            {
                const auto real     = with_type(data_set_t(r.lower, r.upper, r.strides, batch_rank),
                                            rocfft_array_type_real,
                                            p,
                                            r.offsets);
                const auto expected = with_type(
                    data_set_t(c.lower, c.upper, c.strides, batch_rank), t, p, c.offsets);
                SCOPED_TRACE(real.str());
                const auto dst = real.reinterpret_as(t);
                EXPECT_TRUE(dst == expected) << dst.str();
                EXPECT_EQ(dst.get_precision(), p);
                EXPECT_TRUE(expected.reinterpret_as(rocfft_array_type_real) == real);
            }
        }
    }
}

TEST_F(data_set_test, reinterpret_real_planar_round_trip_and_positions)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // random planar layout (any lower bounds, other strides and first offset)
        // with an even stride along length axis 0 and planes offset by half of it
        const auto len_rank   = uniform(1, 3);
        const auto batch_rank = uniform(1, 2);
        const auto lower      = random_vec(len_rank + batch_rank, 0, 3);
        auto       upper      = random_vec(len_rank + batch_rank, 1, 4);
        for(size_t d = 0; d < upper.size(); d++)
            upper[d] += lower[d];
        auto strides      = random_vec(len_rank + batch_rank, 1, 40);
        strides[0]        = 2 * uniform(1, 5);
        const auto p      = uniform(0, 1) ? rocfft_precision_single : rocfft_precision_double;
        const auto t      = planar_types[uniform(0, 1)];
        const auto first  = uniform(0, 9);
        const auto planar = with_type(
            data_set_t(lower, upper, strides, batch_rank), t, p, {first, first + strides[0] / 2});
        SCOPED_TRACE(planar.str());

        const auto real = planar.reinterpret_as(rocfft_array_type_real);
        SCOPED_TRACE(real.str());
        EXPECT_EQ(real.get_precision(), p);
        EXPECT_EQ(real.logical_count(), 2 * planar.logical_count());
        EXPECT_TRUE(real.reinterpret_as(t) == planar);
        for(auto t2 : planar_types)
            EXPECT_TRUE(real.reinterpret_as(t2).reinterpret_as(rocfft_array_type_real) == real);

        // real element 2k (resp. 2k + 1) is element k of the real (resp. imaginary)
        // part, at the same byte position
        for_each_coordinate(planar, [&](const std::vector<size_t>& coord) {
            auto real_coord = coord;
            real_coord[0] *= 2;
            EXPECT_EQ(position_of(real, real_coord), position_of(planar, coord, data_part::REAL));
            EXPECT_EQ(position_of(real, real_coord) * real.element_byte_size(),
                      position_of(planar, coord, data_part::REAL) * planar.element_byte_size());
            real_coord[0] += 1;
            EXPECT_EQ(position_of(real, real_coord),
                      position_of(planar, coord, data_part::IMAGINARY));
        });
    }
}

TEST_F(data_set_test, reinterpret_real_planar_rejects_bad_layouts)
{
    for(auto t : planar_types)
    {
        // real to planar requires even bounds along length axis 0 only
        const auto real = [](const std::vector<size_t>& lower, const std::vector<size_t>& upper) {
            return with_type(data_set_t(lower, upper, {3, 25, 101, 303}, 2),
                             rocfft_array_type_real,
                             rocfft_precision_single,
                             {7, 0});
        };
        EXPECT_NO_THROW(real({2, 0, 0, 0}, {8, 4, 3, 2}).reinterpret_as(t));
        EXPECT_THROW(real({1, 0, 0, 0}, {8, 4, 3, 2}).reinterpret_as(t), std::invalid_argument);
        EXPECT_THROW(real({2, 0, 0, 0}, {7, 4, 3, 2}).reinterpret_as(t), std::invalid_argument);

        // planar to real requires an even stride along length axis 0 and planes
        // offset by half of it
        const auto planar = [t](size_t stride0, const std::array<size_t, 2>& offsets) {
            return with_type(data_set_t({1, 0, 0, 0}, {7, 4, 3, 2}, {stride0, 25, 101, 303}, 2),
                             t,
                             rocfft_precision_single,
                             offsets);
        };
        EXPECT_NO_THROW(planar(4, {5, 7}).reinterpret_as(rocfft_array_type_real));
        EXPECT_THROW(planar(3, {5, 6}).reinterpret_as(rocfft_array_type_real),
                     std::invalid_argument);
        EXPECT_THROW(planar(3, {5, 7}).reinterpret_as(rocfft_array_type_real),
                     std::invalid_argument);
        for(const auto& offsets :
            std::vector<std::array<size_t, 2>>{{5, 5}, {5, 6}, {5, 8}, {7, 5}})
        {
            EXPECT_THROW(planar(4, offsets).reinterpret_as(rocfft_array_type_real),
                         std::invalid_argument);
        }
    }
}

TEST_F(data_set_test, reinterpret_real_planar_embedded)
{
    const auto L = with_type(data_set_t({0, 0, 0, 0}, {8, 4, 6, 3}, {1, 10, 40, 240}),
                             rocfft_array_type_real,
                             rocfft_precision_double,
                             {4, 0});
    for(auto t : planar_types)
    {
        // length axis 0 of the sub-layout is L's: the embedding map is kept and
        // reinterpreting commutes with get_embedding_data_set() (whose Hermitian
        // results get the complex counterpart)
        const auto sub = L.get_data_set_for_len_axes({0, 2});
        ASSERT_TRUE(sub.is_embedded());
        const auto dst = sub.reinterpret_as(t);
        EXPECT_TRUE(dst.is_embedded());
        EXPECT_TRUE(dst.get_embedding_data_set()
                    == L.reinterpret_as(rocfft_array_type_complex_planar))
            << dst.str();
        EXPECT_EQ(dst.corresponding_axes_in_embedding(), sub.corresponding_axes_in_embedding());
        const auto back = dst.reinterpret_as(rocfft_array_type_real);
        EXPECT_TRUE(back == sub);
        EXPECT_TRUE(back.get_embedding_data_set() == L);

        // length axis 0 of the sub-layout is not the embedding's, although it is
        // otherwise reinterpretable
        const auto L_t = with_type(data_set_t({0, 0, 0, 0}, {4, 8, 6, 3}, {8, 1, 32, 192}),
                                   rocfft_array_type_real,
                                   rocfft_precision_double,
                                   {4, 0});
        EXPECT_THROW(L_t.get_data_set_for_len_axes({1}).reinterpret_as(t), std::invalid_argument);
        EXPECT_THROW(L_t.get_data_set_for_len_axes({1, 2}).reinterpret_as(t),
                     std::invalid_argument);
        const auto L_t_planar = with_type(data_set_t({0, 0, 0, 0}, {4, 8, 6, 3}, {16, 2, 64, 384}),
                                          t,
                                          rocfft_precision_double,
                                          {3, 4});
        EXPECT_THROW(
            L_t_planar.get_data_set_for_len_axes({1}).reinterpret_as(rocfft_array_type_real),
            std::invalid_argument);
    }
}

namespace
{
    // valid offsets for array type t: second offset only set if planar
    std::array<size_t, 2> offsets_for(rocfft_array_type t, size_t first, size_t second)
    {
        return {first, array_type_is_planar(t) ? second : 0};
    }

    // parts of the stored data of a data set of array type t (see data_part)
    std::vector<data_part> parts_of(rocfft_array_type t)
    {
        if(array_type_is_planar(t))
            return {data_part::REAL, data_part::IMAGINARY};
        return {data_part::ALL};
    }
}

TEST_F(data_set_test, offsets_default_to_zero_with_one_per_plane)
{
    EXPECT_EQ(data_set_t::default_full_range({8}, 2).num_planes(), 1u);
    EXPECT_EQ(data_set_t::default_full_range({8}, 2).get_offset(), 0u);
    const auto base = data_set_t::default_full_range({8, 4}, 2);
    for(auto type : all_array_types)
    {
        const auto data_set = with_type(base, type, rocfft_precision_single);
        SCOPED_TRACE(data_set.str());
        EXPECT_EQ(data_set.num_planes(), array_type_is_planar(type) ? 2u : 1u);
        for(auto part : {data_part::ALL, data_part::REAL, data_part::IMAGINARY})
        {
            if(array_type_is_planar(type) == (part == data_part::ALL))
                EXPECT_THROW(data_set.get_offset(part), std::invalid_argument);
            else
                EXPECT_EQ(data_set.get_offset(part), 0u);
        }
    }
}

TEST_F(data_set_test, offsets_enforce_invariant)
{
    for(auto type : all_array_types)
    {
        const auto data_set = with_type(data_set_t({0, 0}, {4, 3}, {1, 4}),
                                        type,
                                        rocfft_precision_double,
                                        offsets_for(type, 3, 5));
        SCOPED_TRACE(data_set.str());
        if(array_type_is_planar(type))
        {
            EXPECT_EQ(data_set.get_offset(data_part::REAL), 3u);
            EXPECT_EQ(data_set.get_offset(data_part::IMAGINARY), 5u);
        }
        else
        {
            EXPECT_EQ(data_set.get_offset(), 3u);
            EXPECT_THROW(
                with_type(
                    data_set_t({0, 0}, {4, 3}, {1, 4}), type, rocfft_precision_double, {3, 5}),
                std::invalid_argument);
        }
    }
}

TEST_F(data_set_test, offsets_setter_enforces_invariant)
{
    auto data_set = data_set_t::default_full_range({8, 4}, 2);
    data_set_test_access::set_array_type_and_offsets(
        data_set, rocfft_array_type_complex_planar, {1, 2});
    EXPECT_EQ(data_set.get_offset(data_part::IMAGINARY), 2u);

    // a violation leaves the object untouched
    for(auto type : all_array_types)
    {
        if(array_type_is_planar(type))
            continue;
        EXPECT_THROW(data_set_test_access::set_array_type_and_offsets(data_set, type, {1, 2}),
                     std::invalid_argument);
        EXPECT_EQ(data_set.get_array_type(), rocfft_array_type_complex_planar);
        EXPECT_EQ(data_set.get_offset(data_part::REAL), 1u);
        EXPECT_EQ(data_set.get_offset(data_part::IMAGINARY), 2u);
    }

    data_set_test_access::set_array_type_and_offsets(
        data_set, rocfft_array_type_complex_interleaved, {1, 0});
    EXPECT_EQ(data_set.get_array_type(), rocfft_array_type_complex_interleaved);
    EXPECT_EQ(data_set.get_offset(), 1u);
    EXPECT_THROW(data_set.get_offset(data_part::IMAGINARY), std::invalid_argument);
}

TEST_F(data_set_test, offsets_survive_reinterpret_as)
{
    const auto base = data_set_t::default_full_range({8, 4}, 2);
    const std::pair<rocfft_array_type, rocfft_array_type> allowed[]
        = {{rocfft_array_type_complex_interleaved, rocfft_array_type_complex_interleaved},
           {rocfft_array_type_complex_interleaved, rocfft_array_type_hermitian_interleaved},
           {rocfft_array_type_hermitian_interleaved, rocfft_array_type_complex_interleaved},
           {rocfft_array_type_hermitian_interleaved, rocfft_array_type_hermitian_interleaved},
           {rocfft_array_type_complex_planar, rocfft_array_type_complex_planar},
           {rocfft_array_type_complex_planar, rocfft_array_type_hermitian_planar},
           {rocfft_array_type_hermitian_planar, rocfft_array_type_complex_planar},
           {rocfft_array_type_hermitian_planar, rocfft_array_type_hermitian_planar},
           {rocfft_array_type_real, rocfft_array_type_real}};
    for(auto [from, to] : allowed)
    {
        const auto src = with_type(base, from, rocfft_precision_single, offsets_for(from, 7, 9));
        const auto dst = src.reinterpret_as(to);
        SCOPED_TRACE(dst.str());
        EXPECT_EQ(dst.num_planes(), src.num_planes());
        for(auto part : parts_of(from))
            EXPECT_EQ(dst.get_offset(part), src.get_offset(part));
        EXPECT_TRUE(dst.reinterpret_as(from) == src);
    }
}

TEST_F(data_set_test, offsets_inplace_mapping)
{
    // complex transforms: identical offsets in all planes
    for(auto type : {rocfft_array_type_complex_interleaved, rocfft_array_type_complex_planar})
    {
        const auto in  = with_type(data_set_t::default_full_range({16}, 4),
                                  type,
                                  rocfft_precision_double,
                                  offsets_for(type, 6, 11));
        const auto out = in.get_other_inplace_data_set_for(io_data_label::OUTPUT,
                                                           rocfft_transform_type_complex_forward);
        ASSERT_TRUE(out.has_value());
        EXPECT_TRUE(*out == in) << out->str();
    }

    // real transforms: real offset == 2 * Hermitian offset
    const auto padded = data_set_t::default_full_range({16}, 4, true);
    const auto real   = with_type(padded, rocfft_array_type_real, rocfft_precision_single, {6, 0});
    const auto odd_real
        = with_type(padded, rocfft_array_type_real, rocfft_precision_single, {7, 0});
    for(auto [fft_type, herm_io] :
        {std::pair{rocfft_transform_type_real_forward, io_data_label::OUTPUT},
         std::pair{rocfft_transform_type_real_inverse, io_data_label::INPUT}})
    {
        const auto herm = real.get_other_inplace_data_set_for(herm_io, fft_type);
        ASSERT_TRUE(herm.has_value());
        EXPECT_EQ(herm->get_offset(), 3u);
        const auto back = herm->get_other_inplace_data_set_for(other(herm_io), fft_type);
        ASSERT_TRUE(back.has_value());
        EXPECT_TRUE(*back == real) << back->str();

        // an odd real offset cannot be halved
        EXPECT_FALSE(odd_real.get_other_inplace_data_set_for(herm_io, fft_type).has_value());
    }

    // planar Hermitian data sets are rejected: in-place real transforms are interleaved only
    const auto herm_planar = with_type(data_set_t::default_full_range({9}, 4),
                                       rocfft_array_type_hermitian_planar,
                                       rocfft_precision_single,
                                       {4, 2});
    EXPECT_THROW(herm_planar.get_other_inplace_data_set_for(io_data_label::OUTPUT,
                                                            rocfft_transform_type_real_inverse),
                 std::logic_error);

    // any Hermitian offset doubles, odd ones included
    const auto herm      = with_type(data_set_t::default_full_range({9}, 4),
                                rocfft_array_type_hermitian_interleaved,
                                rocfft_precision_single,
                                {5, 0});
    const auto from_herm = herm.get_other_inplace_data_set_for(io_data_label::OUTPUT,
                                                               rocfft_transform_type_real_inverse);
    ASSERT_TRUE(from_herm.has_value());
    EXPECT_EQ(from_herm->get_array_type(), rocfft_array_type_real);
    EXPECT_EQ(from_herm->get_offset(), 10u);

    // unset array type: offsets derived alike
    auto unset = padded;
    data_set_test_access::set_array_type_and_offsets(unset, rocfft_array_type_unset, {4, 0});
    const auto unset_herm = unset.get_other_inplace_data_set_for(
        io_data_label::OUTPUT, rocfft_transform_type_real_forward);
    ASSERT_TRUE(unset_herm.has_value());
    EXPECT_FALSE(unset_herm->has_array_type());
    EXPECT_EQ(unset_herm->get_offset(), 2u);
}

TEST_F(data_set_test, offsets_equality_is_strict)
{
    const auto base = data_set_t::default_full_range({8, 4}, 2);
    const auto make = [&base](const std::array<size_t, 2>& offsets) {
        return with_type(base, rocfft_array_type_complex_planar, rocfft_precision_single, offsets);
    };
    EXPECT_TRUE(make({1, 2}) == make({1, 2}));
    EXPECT_FALSE(make({1, 2}) == make({1, 3}));
    EXPECT_FALSE(make({1, 2}) == make({0, 2}));
    EXPECT_FALSE(make({1, 2}) == make({0, 0}));
}

TEST_F(data_set_test, offsets_pass_through_subsets_and_embeddings)
{
    const auto data_set = with_type(data_set_t::default_full_range({4, 8, 2}, 3),
                                    rocfft_array_type_complex_planar,
                                    rocfft_precision_double,
                                    {3, 5});
    for(const auto& len_indices : {std::set<size_t>{0, 1, 2}, std::set<size_t>{0, 2}})
    {
        const auto sub = data_set.get_data_set_for_len_axes(len_indices);
        EXPECT_EQ(sub.get_offset(data_part::REAL), 3u);
        EXPECT_EQ(sub.get_offset(data_part::IMAGINARY), 5u);
        const auto embedding = sub.get_embedding_data_set();
        EXPECT_EQ(embedding.get_offset(data_part::REAL), 3u);
        EXPECT_EQ(embedding.get_offset(data_part::IMAGINARY), 5u);
    }
}

TEST_F(data_set_test, offsets_zero_in_intersection)
{
    const auto base  = data_set_t::default_full_range({8, 4}, 2);
    const auto xsect = data_set_t::make_contiguous_intersection_of(
        with_type(base, rocfft_array_type_complex_planar, rocfft_precision_single, {3, 5}),
        with_type(base, rocfft_array_type_complex_planar, rocfft_precision_single, {1, 2}));
    EXPECT_EQ(xsect.get_offset(data_part::REAL), 0u);
    EXPECT_EQ(xsect.get_offset(data_part::IMAGINARY), 0u);
}

TEST_F(data_set_test, offsets_survive_clear_and_reset)
{
    auto data_set = with_type(data_set_t::default_full_range({8, 4}, 2),
                              rocfft_array_type_hermitian_planar,
                              rocfft_precision_half,
                              {3, 5});
    data_set_test_access::clear(data_set);
    EXPECT_EQ(data_set.get_offset(data_part::REAL), 3u);
    EXPECT_EQ(data_set.get_offset(data_part::IMAGINARY), 5u);
    data_set_test_access::full_range_reset(data_set, {16}, 3);
    EXPECT_EQ(data_set.get_offset(data_part::REAL), 3u);
    EXPECT_EQ(data_set.get_offset(data_part::IMAGINARY), 5u);
}

TEST_F(data_set_test, offsets_in_str)
{
    const auto base = data_set_t::default_full_range({8}, 2);
    const auto planar
        = with_type(base, rocfft_array_type_complex_planar, rocfft_precision_single, {3, 5});
    EXPECT_NE(planar.str().find("offsets 3 5"), std::string::npos) << planar.str();
    const auto interleaved
        = with_type(base, rocfft_array_type_complex_interleaved, rocfft_precision_single, {3, 0});
    EXPECT_NE(interleaved.str().find("offsets 3"), std::string::npos) << interleaved.str();
    EXPECT_EQ(interleaved.str().find("offsets 3 0"), std::string::npos) << interleaved.str();
}

// offset_in, is_continuous_in and is_contiguous are relative to the parts' offsets, which
// they ignore; buffer_element_count includes them
TEST_F(data_set_test, offsets_ignored_by_relative_queries)
{
    const auto whole = with_type(data_set_t::default_full_range({8, 4}, 2),
                                 rocfft_array_type_complex_planar,
                                 rocfft_precision_single);
    const auto chunk = with_type(data_set_t({0, 1, 0}, {8, 3, 2}, whole.strides_and_distances()),
                                 rocfft_array_type_complex_planar,
                                 rocfft_precision_single);
    const auto shifted_whole
        = with_type(whole, rocfft_array_type_complex_planar, rocfft_precision_single, {3, 5});
    const auto shifted_chunk
        = with_type(chunk, rocfft_array_type_complex_planar, rocfft_precision_single, {7, 11});
    EXPECT_EQ(shifted_chunk.offset_in(shifted_whole), chunk.offset_in(whole));
    EXPECT_EQ(shifted_chunk.offset_in(whole), chunk.offset_in(whole));
    EXPECT_EQ(shifted_chunk.is_continuous_in(shifted_whole), chunk.is_continuous_in(whole));
    EXPECT_EQ(shifted_whole.is_contiguous(), whole.is_contiguous());
    for(auto part : parts_of(rocfft_array_type_complex_planar))
    {
        EXPECT_EQ(shifted_whole.buffer_element_count(part),
                  shifted_whole.get_offset(part) + whole.buffer_element_count(part));
        EXPECT_EQ(shifted_chunk.buffer_element_count(part),
                  shifted_chunk.get_offset(part) + chunk.buffer_element_count(part));
    }
    // an empty data set needs no buffer, wherever it would start
    const auto empty = with_type(data_set_t({0, 0}, {0, 2}, {1, 1}),
                                 rocfft_array_type_complex_interleaved,
                                 rocfft_precision_single,
                                 {4, 0});
    EXPECT_EQ(empty.buffer_element_count(), 0u);
}

TEST_F(data_set_test, max_offset_and_stride_contiguous)
{
    // 1D, 2D, 3D: last element of a packed layout
    const auto l1 = data_set_t::default_full_range({16}, 1);
    const auto l2 = data_set_t::default_full_range({4, 8}, 3);
    const auto l3 = data_set_t::default_full_range({2, 3, 4}, 5);
    EXPECT_EQ(l1.max_offset(), 15u);
    EXPECT_EQ(l1.max_stride(), 16u); // distance of the unit batch axis
    EXPECT_EQ(l2.max_offset(), 95u);
    EXPECT_EQ(l2.max_stride(), 32u);
    EXPECT_EQ(l3.max_offset(), 119u);
    EXPECT_EQ(l3.max_stride(), 24u);

    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto packed = data_set_t::default_full_range(random_vec(uniform(1, 4), 1, 16),
                                                           random_vec(uniform(1, 3), 1, 5));
        SCOPED_TRACE(packed.str());
        EXPECT_EQ(packed.max_offset(), packed.logical_count() - 1);
    }
}

TEST_F(data_set_test, max_offset_and_stride_padded_and_unordered)
{
    // real in-place padding: rows of 4 reals occupy 6 slots
    const auto padded = data_set_t::default_full_range({4, 8}, 3, true);
    EXPECT_EQ(padded.max_offset(), 3u + 7u * 6u + 2u * 48u);
    EXPECT_EQ(padded.max_stride(), 48u);

    // largest stride on the first length axis, batch axis interleaved in between
    const auto unordered = data_set_t::full_range({4, 8}, {24, 1}, 3, 8);
    EXPECT_EQ(unordered.max_stride(), 24u);
    EXPECT_EQ(unordered.max_offset(), 3u * 24u + 7u + 2u * 8u);
    EXPECT_EQ(unordered.max_offset(), unordered.logical_count() - 1);

    // two batch axes, the length axis being the slowest
    const data_set_t multi_batch({0, 0, 0}, {4, 8, 3}, {24, 1, 8}, 2);
    EXPECT_EQ(multi_batch.max_stride(), 24u);
    EXPECT_EQ(multi_batch.max_offset(), 95u);
}

// lower bounds are logical coordinates: the first element sits at the offset
TEST_F(data_set_test, max_offset_ignores_lower_bounds)
{
    const data_set_t sub({1, 2, 1}, {4, 8, 3}, {1, 4, 32});
    const data_set_t zero_based({0, 0, 0}, {3, 6, 2}, {1, 4, 32});
    EXPECT_EQ(sub.max_offset(), 2u * 1u + 5u * 4u + 1u * 32u);
    EXPECT_EQ(sub.max_offset(), zero_based.max_offset());
    EXPECT_EQ(sub.max_stride(), 32u);

    // multiple batch axes with non-zero lower bounds
    const data_set_t multi_batch({2, 1, 3}, {4, 8, 5}, {1, 4, 32}, 2);
    EXPECT_EQ(multi_batch.max_offset(), 1u + 6u * 4u + 1u * 32u);
}

// a sub-dimensional embedded layout keeps all axes (as batch axes), so it
// reaches the same elements and carries the same strides
TEST_F(data_set_test, max_offset_and_stride_embedded)
{
    const auto   layout   = with_type(data_set_t({1, 0, 2, 0}, {4, 8, 6, 3}, {200, 1, 16, 1000}),
                                  rocfft_array_type_complex_interleaved,
                                  rocfft_precision_double,
                                  {7, 0});
    const size_t expected = 7u + 2u * 200u + 7u * 1u + 3u * 16u + 2u * 1000u;
    EXPECT_EQ(layout.max_offset(), expected);
    EXPECT_EQ(layout.max_stride(), 1000u);
    for(const auto& len_indices :
        {std::set<size_t>{0}, std::set<size_t>{1, 2}, std::set<size_t>{0, 1, 2}})
    {
        const auto sub = layout.get_data_set_for_len_axes(len_indices);
        SCOPED_TRACE(sub.str());
        EXPECT_EQ(sub.max_offset(), expected);
        EXPECT_EQ(sub.max_stride(), 1000u);
        const auto embedding = sub.get_embedding_data_set();
        EXPECT_EQ(embedding.max_offset(), expected);
        EXPECT_EQ(embedding.max_stride(), 1000u);
    }
}

TEST_F(data_set_test, max_offset_over_parts)
{
    const auto base  = data_set_t::default_full_range({8, 4}, 2);
    const auto count = base.buffer_element_count();
    for(auto type : all_array_types)
    {
        const auto layout = with_type(base, type, rocfft_precision_single, offsets_for(type, 3, 5));
        SCOPED_TRACE(layout.str());
        EXPECT_EQ(layout.max_offset(), (array_type_is_planar(type) ? 5u : 3u) + count - 1);
        EXPECT_EQ(layout.max_stride(), base.max_stride());
    }
    // the larger offset wins whichever part holds it
    const auto first_larger
        = with_type(base, rocfft_array_type_complex_planar, rocfft_precision_single, {9, 2});
    EXPECT_EQ(first_larger.max_offset(), 9u + count - 1);
    for(auto part : parts_of(rocfft_array_type_complex_planar))
        EXPECT_LE(first_larger.buffer_element_count(part) - 1, first_larger.max_offset());
}

TEST_F(data_set_test, max_offset_rejects_empty_range)
{
    const data_set_t empty_len({0, 0}, {0, 4}, {1, 4});
    EXPECT_TRUE(empty_len.is_empty());
    EXPECT_THROW(empty_len.max_offset(), std::logic_error);
    EXPECT_EQ(empty_len.max_stride(), 4u);

    const data_set_t empty_batch({0, 0, 0}, {4, 8, 0}, {1, 4, 32}, 2);
    EXPECT_THROW(empty_batch.max_offset(), std::logic_error);
    EXPECT_EQ(empty_batch.max_stride(), 32u);
}

// values beyond 32 bits are exact, so they can feed 32-bit index limit checks
TEST_F(data_set_test, max_offset_and_stride_beyond_32_bits)
{
    const size_t u32_max = std::numeric_limits<uint32_t>::max();
    const size_t big     = u32_max + 1;

    const auto two_batches = data_set_t::full_range({2}, {1}, 2, big);
    EXPECT_EQ(two_batches.max_stride(), big);
    EXPECT_EQ(two_batches.max_offset(), big + 1);
    EXPECT_GT(two_batches.max_offset(), u32_max);

    // an unused distance (unit batch) exceeds the limit without moving max_offset
    const auto one_batch = data_set_t::full_range({2}, {1}, 1, big);
    EXPECT_GT(one_batch.max_stride(), u32_max);
    EXPECT_EQ(one_batch.max_offset(), 1u);

    // an offset alone may push the last element beyond the limit
    const auto shifted = with_type(data_set_t::default_full_range({8}, 1),
                                   rocfft_array_type_complex_planar,
                                   rocfft_precision_single,
                                   {0, u32_max});
    EXPECT_EQ(shifted.max_offset(), u32_max + 7);
    EXPECT_LE(shifted.max_stride(), u32_max);

    // a layout reaching just below the limit stays below it
    const auto at_limit = data_set_t::full_range({2}, {u32_max - 1}, 1, 1);
    EXPECT_EQ(at_limit.max_offset(), u32_max - 1);
    EXPECT_EQ(at_limit.max_stride(), u32_max - 1);
}

namespace
{
    std::vector<data_set_t> compress(const std::vector<data_set_t>& ds)
    {
        return data_set_t::compress_batch_axes_consistently_in(ds);
    }

    std::vector<data_set_t> compress_lengths(const std::vector<data_set_t>& ds)
    {
        return data_set_t::compress_length_axes_consistently_in(ds);
    }

    void expect_same(const std::vector<data_set_t>& actual, const std::vector<data_set_t>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for(size_t n = 0; n < actual.size(); n++)
            EXPECT_TRUE(actual[n] == expected[n])
                << "data set " << n << ": " << actual[n].str() << " vs " << expected[n].str();
    }

    // For every batch coordinate (relative to the lower bounds, shared by all
    // data sets), the in-buffer location of its first element in each data set;
    // sorted. Equal before and after compression iff every logical element keeps
    // its location consistently across data sets (length axes being unchanged).
    std::vector<std::vector<size_t>> batch_locations(const std::vector<data_set_t>& ds)
    {
        std::vector<std::vector<size_t>> ret;
        const auto                       spans = ds.front().batches();
        std::vector<size_t>              r(spans.size(), 0);
        for(bool done = false; !done;)
        {
            std::vector<size_t> loc;
            for(const auto& d : ds)
            {
                const auto lower = d.lower();
                const auto dist  = d.distances();
                size_t     off   = 0;
                for(size_t k = 0; k < r.size(); k++)
                    off += (lower[d.get_len_rank() + k] + r[k]) * dist[k];
                loc.push_back(off);
            }
            ret.push_back(loc);
            size_t k = 0;
            for(; k < r.size() && ++r[k] == spans[k]; k++)
                r[k] = 0;
            done = k == r.size();
        }
        std::sort(ret.begin(), ret.end());
        return ret;
    }
}

// Legacy leaf axis collapse cases, with the node's high lengths and batch
// as batch axes, inStride/iDist (resp. outStride/oDist) as input (resp. output) strides.
TEST_F(data_set_test, compress_batch_axes_mirrors_collapse_contiguous_dims)
{
    // contiguous in both input and output (output padded): collapses fully
    const data_set_t in({0, 0, 0, 0}, {64, 8, 4, 3}, {1, 64, 512, 2048}, 3);
    const data_set_t out({0, 0, 0, 0}, {64, 8, 4, 3}, {1, 66, 528, 2112}, 3);
    expect_same(compress({in, out}),
                {data_set_t({0, 0}, {64, 96}, {1, 64}), data_set_t({0, 0}, {64, 96}, {1, 66})});

    // contiguous in input only: refused
    const data_set_t in2({0, 0, 0}, {64, 8, 3}, {1, 64, 512}, 2);
    const data_set_t out_gap({0, 0, 0}, {64, 8, 3}, {1, 64, 600}, 2);
    expect_same(compress({in2, out_gap}), {in2, out_gap});

    // contiguous in both but with opposite outer/inner roles (transposed batch): refused
    const data_set_t out_transposed({0, 0, 0}, {64, 8, 3}, {1, 192, 64}, 2);
    expect_same(compress({in2, out_transposed}), {in2, out_transposed});

    // differing multipliers (input and output spans differ along a batch axis):
    // not consistent data sets, rejected
    const data_set_t out_other_multiplier({0, 0, 0}, {64, 4, 3}, {1, 64, 256}, 2);
    EXPECT_THROW(compress({in2, out_other_multiplier}), std::invalid_argument);

    // unit spans merge regardless of their stride, unless their lower bound is not 0
    const data_set_t unit({0, 0, 0}, {64, 1, 5}, {1, 999, 64}, 2);
    expect_same(compress({unit}), {data_set_t({0, 0}, {64, 5}, {1, 64})});
    const data_set_t shifted_unit({0, 3, 0}, {64, 4, 5}, {1, 999, 64}, 2);
    expect_same(compress({shifted_unit}), {shifted_unit});
}

TEST_F(data_set_test, compress_batch_axes_preserves_kernel_order)
{
    // Several directions are valid. Keep the stride in the lowest slot,
    // even when it is not the smallest and the batch span is only one.
    const data_set_t units({0, 0, 0, 0}, {64, 1, 1, 1}, {1, 123, 64, 1024}, 3);
    const auto       merged = data_set_test_access::compress_kernel_batches({units, units});
    ASSERT_EQ(merged.front().get_batch_rank(), 1u);
    EXPECT_EQ(merged.front().distance(), 123u);

    // Packed output spans can differ for a fused leaf. Both higher axes
    // merge, but the final batch cannot use a different multiplier per side.
    const data_set_t in({0, 0, 0, 0}, {64, 8, 4, 2}, {1, 64, 512, 2048}, 3);
    const data_set_t out({0, 0, 0, 0}, {64, 3, 4, 2}, {1, 66, 198, 792}, 3);
    expect_same(data_set_test_access::compress_kernel_batches({in, out}),
                {data_set_t({0, 0, 0}, {64, 32, 2}, {1, 64, 2048}, 2),
                 data_set_t({0, 0, 0}, {64, 12, 2}, {1, 66, 792}, 2)});
}

TEST_F(data_set_test, compress_batch_axes_nominal)
{
    // single data set, partial collapse: the padded outermost axis stays
    const data_set_t padded({0, 0, 0, 0}, {64, 8, 4, 3}, {1, 64, 512, 4096}, 3);
    expect_same(compress({padded}), {data_set_t({0, 0, 0}, {64, 32, 3}, {1, 64, 4096}, 2)});

    // non-adjacent pair: merged at the higher-indexed position, the other one erased
    const data_set_t apart({0, 0, 0, 0}, {64, 8, 5, 3}, {1, 64, 10000, 512}, 3);
    expect_same(compress({apart}), {data_set_t({0, 0, 0}, {64, 5, 24}, {1, 10000, 64}, 2)});

    // nothing merges: no-op
    const data_set_t loose({0, 0, 0}, {64, 8, 3}, {1, 100, 1000}, 2);
    expect_same(compress({loose}), {loose});
    const auto one_batch = data_set_t::default_full_range({64}, 3);
    expect_same(compress({one_batch}), {one_batch});

    // length axes are never touched, even if contiguous with batch axes
    const auto lens = data_set_t::default_full_range({4, 8}, 3);
    expect_same(compress({lens}), {lens});

    // three data sets: a third one that does not allow the merge blocks it...
    const data_set_t a({0, 0, 0}, {16, 8, 3}, {1, 16, 128}, 2);
    const data_set_t b({0, 0, 0}, {9, 8, 3}, {1, 9, 72}, 2);
    const data_set_t c_gap({0, 0, 0}, {16, 8, 3}, {1, 16, 129}, 2);
    expect_same(compress({a, b, c_gap}), {a, b, c_gap});
    // ... and is compressed along with the others when it does
    const data_set_t c({0, 0, 0}, {16, 8, 3}, {2, 32, 256}, 2);
    expect_same(compress({a, b, c}),
                {data_set_t({0, 0}, {16, 24}, {1, 16}),
                 data_set_t({0, 0}, {9, 24}, {1, 9}),
                 data_set_t({0, 0}, {16, 24}, {2, 32})});

    EXPECT_TRUE(compress({}).empty());
}

TEST_F(data_set_test, compress_batch_axes_partial_ranges)
{
    // batch axes [2, 6) x [1, 3) of strides 10 and 40 (= 10 * 4): merged into
    // [2 + 4 * 1, 6 + 8) of stride 10, first element still at 60
    const data_set_t partial({0, 2, 1}, {5, 6, 3}, {1, 10, 40}, 2);
    expect_same(compress({partial}), {data_set_t({0, 6}, {5, 14}, {1, 10})});
    // sub-range of a wider axis (40 != 10 * span): not contiguous, refused
    const data_set_t strided({0, 2, 1}, {5, 6, 3}, {1, 10, 50}, 2);
    expect_same(compress({strided}), {strided});
    // different lower bounds across data sets (same spans) are fine
    const data_set_t other({0, 0, 0}, {5, 4, 2}, {1, 5, 20}, 2);
    expect_same(compress({partial, other}),
                {data_set_t({0, 6}, {5, 14}, {1, 10}), data_set_t({0, 0}, {5, 8}, {1, 5})});
}

TEST_F(data_set_test, compress_batch_axes_embedded)
{
    // sub has batch axes [8 (length axis 1 of L), 3, 5]: the first one is a length
    // axis of the embedding data set, so it is kept even if contiguous with the 3
    const auto L   = data_set_t::default_full_range({4, 8}, {3, 5});
    const auto sub = L.get_data_set_for_len_axes({0});
    const auto out = compress({sub});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(out[0] == data_set_t({0, 0, 0}, {4, 8, 15}, {1, 4, 32}, 2)) << out[0].str();
    EXPECT_TRUE(out[0].get_embedding_data_set() == data_set_t({0, 0, 0}, {4, 8, 15}, {1, 4, 32}))
        << out[0].get_embedding_data_set().str();
    EXPECT_EQ(out[0].corresponding_axes_in_embedding(), std::vector<size_t>({0}));
    // the same axes without embedding collapse fully
    expect_same(compress({data_set_t({0, 0, 0, 0}, {4, 8, 3, 5}, {1, 4, 32, 96}, 3)}),
                {data_set_t({0, 0}, {4, 120}, {1, 4})});

    // embedding axis after the erased one: its index in the map is shifted down
    data_set_t shifted({0, 0, 0, 0}, {4, 3, 5, 7}, {1, 4, 12, 1000}, 3);
    data_set_test_access::set_embedding(shifted, {0, 3});
    const auto out2 = compress({shifted});
    ASSERT_EQ(out2.size(), 1u);
    EXPECT_TRUE(out2[0] == data_set_t({0, 0, 0}, {4, 15, 7}, {1, 4, 1000}, 2)) << out2[0].str();
    EXPECT_TRUE(out2[0].get_embedding_data_set() == data_set_t({0, 0, 0}, {4, 7, 15}, {1, 1000, 4}))
        << out2[0].get_embedding_data_set().str();
    EXPECT_EQ(out2[0].corresponding_axes_in_embedding(), std::vector<size_t>({0}));
}

TEST_F(data_set_test, compress_batch_axes_rejects_bad_input)
{
    const data_set_t two_batch({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 2);
    const data_set_t one_batch({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 1);
    EXPECT_THROW(compress({two_batch, one_batch}), std::invalid_argument);
    const data_set_t other_span({0, 0, 0}, {4, 8, 2}, {1, 4, 32}, 2);
    EXPECT_THROW(compress({two_batch, other_span}), std::invalid_argument);
    EXPECT_THROW(compress({two_batch, two_batch, other_span}), std::invalid_argument);
}

TEST_F(data_set_test, compress_length_axes)
{
    // contiguous lengths in input and output: merged; contiguous batch axes are left alone
    const data_set_t in({0, 0, 0, 0}, {4, 8, 3, 5}, {1, 4, 32, 96}, 2);
    const data_set_t out({0, 0, 0, 0}, {4, 8, 3, 5}, {1, 4, 40, 120}, 2);
    expect_same(compress_lengths({in, out}),
                {data_set_t({0, 0, 0}, {32, 3, 5}, {1, 32, 96}, 2),
                 data_set_t({0, 0, 0}, {32, 3, 5}, {1, 40, 120}, 2)});

    // contiguous in input only: refused
    const data_set_t out_gap({0, 0, 0, 0}, {4, 8, 3, 5}, {1, 5, 40, 120}, 2);
    expect_same(compress_lengths({in, out_gap}), {in, out_gap});

    // only the contiguous pair merges, the padded outermost length stays
    const data_set_t padded({0, 0, 0, 0}, {4, 8, 2, 3}, {1, 4, 64, 128}, 1);
    expect_same(compress_lengths({padded}), {data_set_t({0, 0, 0}, {32, 2, 3}, {1, 64, 128}, 1)});

    // partial range: [0, 4) x [2, 6) of strides 1 and 4 becomes [8, 24) of stride 1
    const data_set_t partial({0, 2, 0}, {4, 6, 3}, {1, 4, 16}, 1);
    expect_same(compress_lengths({partial}), {data_set_t({8, 0}, {24, 3}, {1, 16}, 1)});

    // a single length axis: nothing to merge
    const auto one_len = data_set_t::default_full_range({64}, 3);
    expect_same(compress_lengths({one_len}), {one_len});

    // length axes of an embedded data set are length axes of its embedding: kept
    const auto sub
        = data_set_t::default_full_range({4, 8, 2}, {3}).get_data_set_for_len_axes({0, 1});
    expect_same(compress_lengths({sub}), {sub});

    // real input vs. complex output (8 reals, 5 complex numbers along the fastest
    // axis): that axis is not merged, the higher ones still are
    const data_set_t real_in({0, 0, 0, 0}, {8, 4, 2, 3}, {1, 8, 32, 64}, 1);
    const data_set_t cmplx_out({0, 0, 0, 0}, {5, 4, 2, 3}, {1, 5, 20, 40}, 1);
    expect_same(compress_lengths({real_in, cmplx_out}),
                {data_set_t({0, 0, 0}, {8, 8, 3}, {1, 8, 64}, 1),
                 data_set_t({0, 0, 0}, {5, 8, 3}, {1, 5, 40}, 1)});

    // different ranks: rejected
    EXPECT_THROW(compress_lengths({real_in, data_set_t({0, 0, 0}, {8, 4, 3}, {1, 8, 32}, 1)}),
                 std::invalid_argument);
}

TEST_F(data_set_test, compress_batch_axes_preserves_locations)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // 1 to 3 data sets sharing batch spans, batch axes laid out in a shared
        // (or, sometimes, their own) random order, sometimes padded, random lower bounds
        const auto              len_rank     = uniform(1, 2);
        const auto              batch_rank   = uniform(1, 4);
        const auto              batch_spans  = random_vec(batch_rank, 1, 4);
        const auto              shared_order = random_permutation(batch_rank);
        std::vector<data_set_t> ds;
        std::string             trace;
        for(auto n = uniform(1, 3); n > 0; n--)
        {
            const auto order = uniform(0, 3) == 0 ? random_permutation(batch_rank) : shared_order;
            auto       spans = random_vec(len_rank, 1, 4);
            spans.insert(spans.end(), batch_spans.begin(), batch_spans.end());
            std::vector<size_t> strides(spans.size());
            size_t              next = 1;
            for(size_t d = 0; d < len_rank; d++)
            {
                strides[d] = next;
                next       = next * spans[d] + uniform(0, 1);
            }
            for(auto b : order)
            {
                strides[len_rank + b] = next;
                next                  = next * spans[len_rank + b] + (uniform(0, 2) == 0 ? 1 : 0);
            }
            const auto          lower = random_vec(spans.size(), 0, 2);
            std::vector<size_t> upper(spans.size());
            for(size_t d = 0; d < spans.size(); d++)
                upper[d] = lower[d] + spans[d];
            ds.emplace_back(lower, upper, strides, batch_rank);
            trace += " " + ds.back().str();
        }
        SCOPED_TRACE(trace);

        const auto out = compress(ds);
        ASSERT_EQ(out.size(), ds.size());
        EXPECT_EQ(batch_locations(out), batch_locations(ds));
        for(size_t n = 0; n < ds.size(); n++)
        {
            EXPECT_LE(out[n].get_batch_rank(), batch_rank);
            EXPECT_EQ(out[n].get_batch_rank(), out[0].get_batch_rank());
            EXPECT_EQ(out[n].lengths(), ds[n].lengths());
            EXPECT_EQ(out[n].strides(), ds[n].strides());
            const auto out_lower = out[n].lower();
            const auto ds_lower  = ds[n].lower();
            EXPECT_TRUE(
                std::equal(ds_lower.begin(), ds_lower.begin() + len_rank, out_lower.begin()));
        }
        // nothing is left to merge
        expect_same(compress(out), out);
    }
}

TEST_F(data_set_test, split_length_axis_nominal)
{
    // 1D: 12 -> 3 x 4
    const auto d1 = data_set_t::full_range({12}, {1}, 3, 12);
    const auto s1 = d1.split_length_axis(0, 3);
    EXPECT_EQ(s1.lengths(), (std::vector<size_t>{3, 4}));
    EXPECT_EQ(s1.strides(), (std::vector<size_t>{1, 3}));
    EXPECT_EQ(s1.batches(), d1.batches());
    EXPECT_EQ(s1.distances(), d1.distances());
    EXPECT_EQ(s1.logical_count(), d1.logical_count());
    EXPECT_EQ(all_offsets(s1), all_offsets(d1));

    // 2D, splitting the outer (non-unit stride) axis: 8 x 6 -> 8 x 2 x 3
    const auto d2 = data_set_t::full_range({8, 6}, {1, 8}, 2, 64);
    const auto s2 = d2.split_length_axis(1, 2);
    EXPECT_EQ(s2.lengths(), (std::vector<size_t>{8, 2, 3}));
    EXPECT_EQ(s2.strides(), (std::vector<size_t>{1, 8, 16}));
    EXPECT_EQ(s2.distances(), d2.distances());
    EXPECT_EQ(s2.logical_count(), d2.logical_count());
    EXPECT_EQ(all_offsets(s2), all_offsets(d2));

    // 3D, splitting the middle axis: later length axes shift by one
    const auto d3 = data_set_t::default_full_range({4, 6, 10}, 2);
    const auto s3 = d3.split_length_axis(1, 3);
    EXPECT_EQ(s3.lengths(), (std::vector<size_t>{4, 3, 2, 10}));
    EXPECT_EQ(s3.strides(), (std::vector<size_t>{1, 4, 12, 24}));
    EXPECT_EQ(s3.batches(), d3.batches());
    EXPECT_EQ(s3.distances(), d3.distances());
    EXPECT_EQ(s3.logical_count(), d3.logical_count());
    EXPECT_EQ(all_offsets(s3), all_offsets(d3));
    EXPECT_FALSE(s3.is_embedded());
}

TEST_F(data_set_test, split_length_axis_default_factor)
{
    // divisor closest to sqrt(L), ties (if any) to the smaller one
    const std::map<size_t, std::vector<size_t>> expected = {{64, {8, 8}},
                                                            {48, {6, 8}},
                                                            {1024, {32, 32}},
                                                            {12, {3, 4}},
                                                            {14, {2, 7}},
                                                            {4, {2, 2}},
                                                            {2 * 3 * 5 * 7, {14, 15}}};
    for(const auto& [L, lengths] : expected)
    {
        SCOPED_TRACE("L = " + std::to_string(L));
        const auto d = data_set_t::default_full_range({L}, 2);
        const auto s = d.split_length_axis(0);
        EXPECT_EQ(s.lengths(), lengths);
        EXPECT_EQ(s.strides(), (std::vector<size_t>{1, lengths[0]}));
        EXPECT_EQ(all_offsets(s), all_offsets(d));
    }
}

TEST_F(data_set_test, split_length_axis_rejects_bad_input)
{
    const auto d = data_set_t::default_full_range({12, 7, 1}, 2);
    // not a length axis index (3 is the batch axis)
    EXPECT_THROW(d.split_length_axis(3, 2), std::invalid_argument);
    EXPECT_THROW(d.split_length_axis(100), std::invalid_argument);
    // non-divisor, trivial factors, factor larger than L
    EXPECT_THROW(d.split_length_axis(0, 5), std::invalid_argument);
    EXPECT_THROW(d.split_length_axis(0, 1), std::invalid_argument);
    EXPECT_THROW(d.split_length_axis(0, 12), std::invalid_argument);
    EXPECT_THROW(d.split_length_axis(0, 24), std::invalid_argument);
    EXPECT_THROW(d.split_length_axis(0, 0), std::invalid_argument);
    // no non-trivial factor for prime L, L = 1, or an empty range
    EXPECT_THROW(d.split_length_axis(1), std::invalid_argument);
    EXPECT_THROW(d.split_length_axis(2), std::invalid_argument);
    EXPECT_THROW(d.split_length_axis(2, 1), std::invalid_argument);
    const data_set_t empty({3, 0}, {3, 2}, {1, 8});
    EXPECT_THROW(empty.split_length_axis(0), std::invalid_argument);
    // partial range [2, 14) is not aligned on multiples of 4 (nor of default 3)
    const data_set_t misaligned({2, 0}, {14, 3}, {1, 16});
    EXPECT_THROW(misaligned.split_length_axis(0, 4), std::invalid_argument);
    EXPECT_THROW(misaligned.split_length_axis(0), std::invalid_argument);
    EXPECT_NO_THROW(misaligned.split_length_axis(0, 2));
}

TEST_F(data_set_test, split_length_axis_partial_range)
{
    // [4, 16) along a stride-1 axis, batch [1, 3): inner [0, 4), outer [1, 4)
    const data_set_t d({4, 1}, {16, 3}, {1, 32});
    const auto       s = d.split_length_axis(0, 4);
    EXPECT_EQ(s.lower(), (std::vector<size_t>{0, 1, 1}));
    EXPECT_EQ(s.upper(), (std::vector<size_t>{4, 4, 3}));
    EXPECT_EQ(s.strides_and_distances(), (std::vector<size_t>{1, 4, 32}));
    EXPECT_EQ(s.logical_count(), d.logical_count());
    EXPECT_EQ(all_offsets(s), all_offsets(d));

    // offset of the first element within the full layout is unchanged too
    const auto full = data_set_t::full_range({16}, {1}, 3, 32);
    EXPECT_EQ(s.offset_in(full.split_length_axis(0, 4)), d.offset_in(full));
}

TEST_F(data_set_test, split_length_axis_multi_batch)
{
    const auto d = data_set_t::default_full_range({6, 10}, {3, 2});
    const auto s = d.split_length_axis(0);
    EXPECT_EQ(s.lengths(), (std::vector<size_t>{2, 3, 10}));
    EXPECT_EQ(s.strides(), (std::vector<size_t>{1, 2, 6}));
    EXPECT_EQ(s.get_batch_rank(), 2u);
    EXPECT_EQ(s.batches(), d.batches());
    EXPECT_EQ(s.distances(), d.distances());
    EXPECT_EQ(all_offsets(s), all_offsets(d));
}

TEST_F(data_set_test, split_length_axis_embedded)
{
    const auto L   = data_set_t::default_full_range({4, 8, 6}, 3);
    const auto sub = L.get_data_set_for_len_axes({0, 2});
    ASSERT_TRUE(sub.is_embedded());

    // split the sub-layout's axis 1 (L's axis 2): L's axis 2 is split alike
    const auto s1 = sub.split_length_axis(1, 2);
    EXPECT_EQ(s1.lengths(), (std::vector<size_t>{4, 2, 3}));
    EXPECT_EQ(s1.batches(), sub.batches());
    EXPECT_TRUE(s1.is_embedded());
    EXPECT_TRUE(s1.get_embedding_data_set() == L.split_length_axis(2, 2));
    EXPECT_EQ(s1.corresponding_axes_in_embedding(), (std::vector<size_t>{0, 2, 3}));
    EXPECT_EQ(all_offsets(s1), all_offsets(sub));

    // split the sub-layout's axis 0 (L's axis 0): every later axis shifts by one
    const auto s0 = sub.split_length_axis(0);
    EXPECT_EQ(s0.lengths(), (std::vector<size_t>{2, 2, 6}));
    EXPECT_TRUE(s0.get_embedding_data_set() == L.split_length_axis(0));
    EXPECT_EQ(s0.corresponding_axes_in_embedding(), (std::vector<size_t>{0, 1, 3}));
    EXPECT_EQ(all_offsets(s0), all_offsets(sub));
}

TEST_F(data_set_test, split_length_axis_round_trip)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // random layout whose length axis dim spans an aligned partial range of
        // factor * m elements
        const auto full   = random_full_range(2);
        const auto dim    = uniform(0, full.get_len_rank() - 1);
        const auto factor = uniform(2, 4);
        auto       lo     = full.lower();
        auto       hi     = full.upper();
        lo[dim]           = factor * uniform(0, 2);
        hi[dim]           = lo[dim] + factor * uniform(2, 3);
        const data_set_t d(lo, hi, full.strides_and_distances(), full.get_batch_rank());
        SCOPED_TRACE("d " + d.str() + ", dim " + std::to_string(dim) + ", factor "
                     + std::to_string(factor));

        const auto s = d.split_length_axis(dim, factor);
        EXPECT_EQ(s.get_len_rank(), d.get_len_rank() + 1);
        EXPECT_EQ(s.logical_count(), d.logical_count());
        EXPECT_EQ(all_offsets(s), all_offsets(d));
    }
}

TEST_F(data_set_test, inbuffer_reset_requires_identical_bounds)
{
    auto       layout      = with_type(data_set_t({1, 2, 0}, {5, 5, 2}, {1, 8, 64}, 2),
                            rocfft_array_type_complex_interleaved,
                            rocfft_precision_single);
    const auto replacement = with_type(data_set_t({1, 2, 0}, {5, 5, 2}, {1, 9, 70}, 2),
                                       rocfft_array_type_complex_planar,
                                       rocfft_precision_double,
                                       {3, 11});
    layout.reset_inbuffer(replacement);
    EXPECT_TRUE(layout.same_logical_range(replacement));
    EXPECT_EQ(layout.strides_and_distances(), replacement.strides_and_distances());
    EXPECT_EQ(layout.get_precision(), rocfft_precision_single);
    EXPECT_EQ(layout.get_offset(data_part::REAL), 3);
    EXPECT_EQ(layout.get_offset(data_part::IMAGINARY), 11);

    const data_set_t shifted({1, 3, 0}, {5, 6, 2}, {1, 12, 81}, 2);
    EXPECT_THROW(layout.reset_inbuffer(shifted), std::invalid_argument);
    EXPECT_EQ(layout.strides_and_distances(), replacement.strides_and_distances());
    EXPECT_THROW(layout.reset_inbuffer(replacement.with_axis_span(0, 3)), std::invalid_argument);
    EXPECT_THROW(layout.set_strides_and_distances(std::array<size_t, 2>{1, 2}),
                 std::invalid_argument);
}

TEST_F(data_set_test, reorder_batch_axes_preserves_embedding)
{
    const data_set_t full({0, 0, 0, 0}, {4, 3, 5, 2}, {2, 64, 1, 16});
    auto             child = full.get_data_set_for_len_axes({1});
    const auto       order = child.axes_by_increasing_inbuffer_strides(axis_scope::batch_only);
    ASSERT_EQ(order, (std::vector<size_t>{1, 0, 2}));
    child.reorder_axes(order, axis_scope::batch_only);
    EXPECT_EQ(child.batches(), (std::vector<size_t>{5, 4, 2}));
    EXPECT_EQ(child.distances(), (std::vector<size_t>{1, 2, 16}));
    EXPECT_TRUE(child.get_embedding_data_set() == full);
    EXPECT_THROW(child.reorder_axes(std::array<size_t, 3>{0, 0, 1}, axis_scope::batch_only),
                 std::invalid_argument);
}
