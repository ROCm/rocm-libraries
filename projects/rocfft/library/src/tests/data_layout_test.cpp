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

#include "client_data_layout_helpers.h"
#include "data_layout.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// internal_test_main.cpp
extern size_t random_seed;
extern size_t num_random_cases;

struct data_layout_test_access
{
    static void reorder_length_axes(data_layout_t& layout, const std::vector<size_t>& order)
    {
        layout.reorder_length_axes(order);
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
}

// Every test restarts its generator from random_seed, so a failing case replays
// with --seed <value> --gtest_filter=<test>, and at least as many --num_random_cases.
class data_layout_test : public ::testing::Test
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
    // Full layout of random ranks/spans whose axes are laid out in random order, with
    // random padding between consecutive axes (non-aliasing, not necessarily packed).
    data_layout_t random_full_layout()
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
        return data_layout_t(std::vector<size_t>(spans.size(), 0), spans, strides, batch_rank);
    }
};

TEST_F(data_layout_test, constructor_rejects_bad_input)
{
    // no batch axis at all
    EXPECT_THROW(data_layout_t({0, 0}, {4, 8}, {1, 4}, 0), std::invalid_argument);
    // no length axis left once batch axes are taken
    EXPECT_THROW(data_layout_t({0, 0}, {4, 8}, {1, 4}, 2), std::invalid_argument);
    // lower, upper and strides must be the same size
    EXPECT_THROW(data_layout_t({0, 0, 0}, {4, 8}, {1, 4, 32}), std::invalid_argument);
    // an axis whose lower bound is past its upper bound
    EXPECT_THROW(data_layout_t({5, 0, 0}, {4, 8, 1}, {1, 4, 32}), std::invalid_argument);
}

TEST_F(data_layout_test, single_batch_axis_accessors)
{
    auto one = data_layout_t::default_full_layout({4, 8}, 3);
    EXPECT_EQ(one.batch(), 3u);
    EXPECT_EQ(one.distance(), 32u);

    // two batch axes: length 4, batches 8 and 3
    data_layout_t two({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 2);
    EXPECT_EQ(two.get_len_rank(), 1u);
    EXPECT_EQ(two.get_batch_rank(), 2u);
    EXPECT_THROW(two.batch(), std::logic_error);
    EXPECT_THROW(two.distance(), std::logic_error);
}

// The client helpers are row-major (last index fastest) while data_layout_t lists
// the fastest axis first, so lengths, batches and results are reversed.
TEST_F(data_layout_test, default_layouts_match_client_helpers)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L = random_vec(uniform(1, 4), 1, 16);
        const auto B = random_vec(uniform(1, 3), 1, 5);
        SCOPED_TRACE("lengths " + to_str(L) + " batches " + to_str(B));
        const auto rL = reversed(L);
        const auto rB = reversed(B);

        const auto cplx = data_layout_t::default_full_layout(L, B);
        EXPECT_EQ(cplx.strides(),
                  reversed(default_strides(
                      fft_transform_type_complex_forward, fft_placement_inplace, fft_io_in, rL)));
        EXPECT_EQ(
            cplx.distances(),
            reversed(default_distances(
                fft_transform_type_complex_forward, fft_placement_inplace, fft_io_in, rL, rB)));

        const auto real = data_layout_t::default_full_layout(L, B, true);
        EXPECT_EQ(real.strides(),
                  reversed(default_strides(
                      fft_transform_type_real_forward, fft_placement_inplace, fft_io_in, rL)));
        EXPECT_EQ(real.distances(),
                  reversed(default_distances(
                      fft_transform_type_real_forward, fft_placement_inplace, fft_io_in, rL, rB)));

        const auto herm = real.get_other_inplace_layout_for(io_data_label::OUTPUT,
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
TEST_F(data_layout_test, padded_buffer_element_count)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L = random_vec(uniform(1, 4), 1, 16);
        const auto B = random_vec(uniform(1, 3), 1, 5);
        SCOPED_TRACE("lengths " + to_str(L) + " batches " + to_str(B));

        const auto packed = data_layout_t::default_full_layout(L, B);
        const auto padded = data_layout_t::default_full_layout(L, B, true);
        const auto n      = L[0];
        const auto P      = 2 * (n / 2 + 1);
        const auto N      = packed.buffer_element_count();
        EXPECT_EQ(N, packed.logical_count());
        EXPECT_EQ(padded.logical_count(), packed.logical_count());
        EXPECT_EQ(padded.buffer_element_count(), P * (N / n) - (P - n));
    }
}

// A layout is contiguous iff its elements fill a buffer without gaps: taken by increasing
// stride, each non-unit-span axis must be strided by the number of elements spanned by the
// axes before it (1 for the first one), whatever the axis order.
// contiguous_strides_and_distances() returns the layout's own strides if it is contiguous,
// and packed strides in natural axis order otherwise.
TEST_F(data_layout_test, contiguous_strides)
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
        const data_layout_t layout(std::vector<size_t>(rank, 0), spans, strides, batch_rank);
        SCOPED_TRACE("layout " + layout.str());

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

        EXPECT_EQ(layout.is_contiguous(), contiguous);
        EXPECT_EQ(layout.contiguous_strides_and_distances(), contiguous ? strides : packed_strides);
    }
}

TEST_F(data_layout_test, offset_in)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // outer: random sub-range, along every axis (batch axes included), of a random
        // full layout, so its lower bounds are generally not 0; inner: random sub-range
        // of outer, with arbitrary strides since offset_in only uses outer's strides
        const auto          full         = random_full_layout();
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
        const data_layout_t outer(outer_lo, outer_hi, outer_stride, full.get_batch_rank());
        const data_layout_t inner(
            inner_lo, inner_hi, random_vec(rank, 1, 100), full.get_batch_rank());
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
        const data_layout_t outside(bad_lo, bad_hi, outer_stride, full.get_batch_rank());
        EXPECT_THROW(outside.offset_in(outer), std::invalid_argument);
    }

    // layouts must have the same numbers of length and batch axes
    const auto layout_2d = data_layout_t::full_layout({8, 8}, {1, 8}, 1, 64);
    const auto layout_1d = data_layout_t::full_layout({8}, {1}, 1, 8);
    EXPECT_THROW(layout_1d.offset_in(layout_2d), std::invalid_argument);

    // even with the same total number of axes
    const data_layout_t layout_1d_two_batches({0, 0, 0}, {8, 8, 1}, {1, 8, 64}, 2);
    EXPECT_THROW(layout_1d_two_batches.offset_in(layout_2d), std::invalid_argument);
}

// Any sub-range along the slowest-varying axis of a layout is one unbroken run of it.
// Unit-length axes do not vary, so they are never the slowest-varying axis, whatever
// their stride (e.g. a trailing batch axis of length 1).
TEST_F(data_layout_test, continuous_chunk)
{
    // 4x8 transform, batch 1: rows 2-4 of the length-8 axis are one unbroken run
    const auto          batch_of_one = data_layout_t::default_full_layout({4, 8}, 1);
    const data_layout_t rows({0, 2, 0}, {4, 5, 1}, {1, 4, 32});
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
        const data_layout_t whole(std::vector<size_t>(rank, 0), spans, strides, batch_rank);
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
        const data_layout_t chunk(lo, hi, strides, batch_rank);
        EXPECT_TRUE(chunk.is_continuous_in(whole)) << chunk.str();

        // a chunk running past the end of the slowest-varying axis is not continuous
        auto past_hi  = hi;
        past_hi[slow] = spans[slow] + uniform(1, 3);
        const data_layout_t overflow(lo, past_hi, strides, batch_rank);
        EXPECT_FALSE(overflow.is_continuous_in(whole)) << overflow.str();

        // a different stride along the slowest axis only matters if the chunk spans it
        auto slow_restrided = strides;
        slow_restrided[slow] += 1;
        const data_layout_t slow_chunk(lo, hi, slow_restrided, batch_rank);
        EXPECT_EQ(slow_chunk.is_continuous_in(whole), hi[slow] - lo[slow] == 1) << slow_chunk.str();

        // strides along unit-length axes are irrelevant
        auto unit_restrided = strides;
        for(size_t d = 0; d < rank; d++)
            if(spans[d] == 1)
                unit_restrided[d] += uniform(1, 5);
        const data_layout_t unit_chunk(lo, hi, unit_restrided, batch_rank);
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
        const data_layout_t gap(lo, gap_hi, strides, batch_rank);
        EXPECT_FALSE(gap.is_continuous_in(whole)) << gap.str();

        auto other_restrided = strides;
        other_restrided[d] += 1;
        const data_layout_t other_chunk(lo, hi, other_restrided, batch_rank);
        EXPECT_FALSE(other_chunk.is_continuous_in(whole)) << other_chunk.str();
    }
}

TEST_F(data_layout_test, same_logical_range)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto          full       = random_full_layout();
        const auto          rank       = full.get_full_rank();
        const auto          batch_rank = full.get_batch_rank();
        const auto          upper      = full.upper();
        std::vector<size_t> lo(rank), hi(rank);
        for(size_t d = 0; d < rank; d++)
        {
            lo[d] = uniform(0, upper[d] - 1);
            hi[d] = uniform(lo[d] + 1, upper[d]);
        }
        const data_layout_t a(lo, hi, full.strides_and_distances(), batch_rank);
        SCOPED_TRACE("a " + a.str());

        // strides are irrelevant
        const data_layout_t b(lo, hi, random_vec(rank, 1, 100), batch_rank);
        EXPECT_TRUE(a.has_same_logical_range_as(b)) << b.str();
        EXPECT_TRUE(b.has_same_logical_range_as(a)) << b.str();

        // moving either bound of any one axis, length or batch, makes the ranges differ
        const auto d = uniform(0, rank - 1);
        if(uniform(0, 1))
            lo[d] += 1;
        else
            hi[d] += uniform(1, 3);
        const data_layout_t c(lo, hi, full.strides_and_distances(), batch_rank);
        EXPECT_FALSE(a.has_same_logical_range_as(c)) << c.str();
        EXPECT_FALSE(c.has_same_logical_range_as(a)) << c.str();
    }

    // same full rank, but a different split between length and batch axes
    const data_layout_t one_batch({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 1);
    const data_layout_t two_batches({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 2);
    EXPECT_THROW(one_batch.has_same_logical_range_as(two_batches), std::invalid_argument);
}

TEST_F(data_layout_test, intersection)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto          full  = random_full_layout();
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
        const data_layout_t first(
            lo[0], hi[0], full.strides_and_distances(), full.get_batch_rank());
        const data_layout_t second(lo[1], hi[1], random_vec(rank, 1, 100), full.get_batch_rank());
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
        const auto overlap = data_layout_t::make_contiguous_intersection_of(first, second);
        EXPECT_EQ(overlap.lower(), exp_lo);
        EXPECT_EQ(overlap.upper(), exp_hi);
        EXPECT_EQ(overlap.logical_count(), count);
        EXPECT_EQ(overlap.is_empty(), count == 0);
        EXPECT_EQ(overlap.strides_and_distances(), exp_stride);
    }

    // arguments must have the same length/batch split...
    const auto          full_3d = data_layout_t::default_full_layout({4, 8, 2}, 3);
    const data_layout_t two_batches({0, 0, 0, 0}, {4, 8, 2, 3}, {1, 4, 32, 64}, 2);
    EXPECT_THROW(data_layout_t::make_contiguous_intersection_of(full_3d, two_batches),
                 std::invalid_argument);

    // ...and the same embedding (if any): views of different length axes, or an embedded
    // view and a plain layout with the same axes, are rejected
    const auto          innermost = full_3d.get_layout_for_len_axes({0});
    const auto          middle    = full_3d.get_layout_for_len_axes({1});
    const data_layout_t plain_middle(
        middle.lower(), middle.upper(), middle.strides_and_distances(), middle.get_batch_rank());
    EXPECT_THROW(data_layout_t::make_contiguous_intersection_of(innermost, middle),
                 std::invalid_argument);
    EXPECT_THROW(data_layout_t::make_contiguous_intersection_of(middle, plain_middle),
                 std::invalid_argument);

    // a shared embedding is kept in the result
    const auto middle_overlap = data_layout_t::make_contiguous_intersection_of(middle, middle);
    EXPECT_TRUE(middle_overlap.is_embedded());
    EXPECT_EQ(middle_overlap.corresponding_axes_in_embedding(), (std::vector<size_t>{1}));
}

// Complex in-place transforms use the same layout on both sides: the counterpart must
// be an exact copy, whatever the layout.
TEST_F(data_layout_test, inplace_complex_is_unchanged)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // Batch axes may be partial ranges (unlike length axes), so cut a random
        // sub-range along them. Then keep a random subset of the length axes, which
        // gives an embedded layout unless every length axis is kept: the embedding map
        // must survive the copy too.
        const auto full = random_full_layout();
        auto       lo   = full.lower();
        auto       hi   = full.upper();
        for(size_t d = full.get_len_rank(); d < full.get_full_rank(); d++)
        {
            lo[d] = uniform(0, hi[d] - 1);
            hi[d] = uniform(lo[d] + 1, hi[d]);
        }
        const data_layout_t batch_range(
            lo, hi, full.strides_and_distances(), full.get_batch_rank());
        auto len_axes = random_permutation(full.get_len_rank());
        len_axes.resize(uniform(1, len_axes.size()));
        const auto layout = batch_range.get_layout_for_len_axes({len_axes.begin(), len_axes.end()});

        // the same copy is expected for either direction and either side: draw one
        // combination per case rather than checking all four every time
        const auto fft_type = uniform(0, 1) ? rocfft_transform_type_complex_forward
                                            : rocfft_transform_type_complex_inverse;
        const auto io       = uniform(0, 1) ? io_data_label::INPUT : io_data_label::OUTPUT;
        SCOPED_TRACE("layout " + layout.str() + " / fft_type " + std::to_string(fft_type) + " / io "
                     + (io == io_data_label::INPUT ? "INPUT" : "OUTPUT"));

        const auto other = layout.get_other_inplace_layout_for(io, fft_type);
        ASSERT_TRUE(other.has_value());
        EXPECT_TRUE(*other == layout) << other->str();
        // operator== ignores the embedding map, so compare it separately
        EXPECT_TRUE(other->get_embedding_layout() == layout.get_embedding_layout());
        EXPECT_EQ(other->corresponding_axes_in_embedding(),
                  layout.corresponding_axes_in_embedding());
    }
}

// real -> Hermitian -> real gives back the original layout, for both real
// transform directions and for even and odd innermost lengths; the wrong parity
// flag must not.
TEST_F(data_layout_test, inplace_real_round_trip)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L    = random_vec(uniform(1, 4), 1, 33);
        const auto B    = random_vec(uniform(1, 3), 1, 5);
        const bool odd  = L[0] % 2 == 1;
        const auto full = data_layout_t::default_full_layout(L, B, true);
        // Multi-device plans also derive in-place layouts for embedded (lower-dimensional)
        // views, which must keep the embedding's innermost axis for real transforms (see
        // plan.cpp). Keep axis 0 and a random subset of the other length axes; keeping all
        // of them gives the plain, non-embedded layout.
        std::set<size_t> kept{0};
        for(size_t d = 1; d < L.size(); d++)
            if(uniform(0, 1))
                kept.insert(d);
        const auto real = full.get_layout_for_len_axes(kept);
        SCOPED_TRACE("full " + full.str() + " / kept " + to_str({kept.begin(), kept.end()}));

        for(auto [fft_type, herm_io] :
            {std::pair{rocfft_transform_type_real_forward, io_data_label::OUTPUT},
             std::pair{rocfft_transform_type_real_inverse, io_data_label::INPUT}})
        {
            const auto real_io = other(herm_io);
            const auto herm    = real.get_other_inplace_layout_for(herm_io, fft_type);
            ASSERT_TRUE(herm.has_value());
            EXPECT_EQ(herm->lengths()[0], L[0] / 2 + 1);

            // the Hermitian side of the view is the view of the Hermitian side of the whole
            const auto herm_full = full.get_other_inplace_layout_for(herm_io, fft_type);
            ASSERT_TRUE(herm_full.has_value());
            EXPECT_TRUE(*herm == herm_full->get_layout_for_len_axes(kept)) << herm->str();
            EXPECT_EQ(herm->corresponding_axes_in_embedding(),
                      real.corresponding_axes_in_embedding());

            const auto back = herm->get_other_inplace_layout_for(real_io, fft_type, odd);
            ASSERT_TRUE(back.has_value());
            EXPECT_TRUE(*back == real) << back->str();
            // operator== ignores the embedding map, so compare it separately
            EXPECT_TRUE(back->get_embedding_layout() == full) << back->str();

            // a Hermitian innermost length of 1 only comes from a real one of 1
            if(L[0] == 1)
            {
                EXPECT_THROW(herm->get_other_inplace_layout_for(real_io, fft_type, !odd),
                             std::invalid_argument);
                continue;
            }
            const auto wrong = herm->get_other_inplace_layout_for(real_io, fft_type, !odd);
            ASSERT_TRUE(wrong.has_value());
            EXPECT_FALSE(*wrong == real) << wrong->str();
        }
    }
}

// A Hermitian layout has an in-place real counterpart iff it is packed along its
// innermost axis (and is not self-aliased). A real layout has an in-place Hermitian
// counterpart iff:
// - it is packed along its innermost axis;
// - every other non-unit-span axis has an even stride (a whole number of complex
//   interleaved values);
// - neither it nor its Hermitian interleaved counterpart is self-aliased.
// The layouts below are built so that they are never self-aliased, and neither are
// their counterparts if the first two conditions hold and every other non-unit-span
// axis is strided by no less than an innermost row padded to 2 * (n / 2 + 1) reals.
// That per-axis rule is what the library currently checks instead of robust
// self-aliasing verification. The expected acceptance below restates it which
// weakens the test: That weakness is a temporary artifact which will be resolved
// once self-aliasing detection is implemented in the library.
TEST_F(data_layout_test, inplace_real_counterpart_rules)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto len_rank   = uniform(1, 3);
        const auto batch_rank = uniform(1, 2);
        const auto rank       = len_rank + batch_rank;
        const auto spans      = random_vec(rank, 1, 6);
        const auto n          = spans[0];
        const auto P          = 2 * (n / 2 + 1);

        // Axes are nested so that the layout is not self-aliased: axis 0 first (mostly
        // packed), then the other non-unit-span axes in random order, each strided by the
        // extent of the axes before it plus up to 3 elements. Rows count as P elements in
        // that extent, so that the counterpart of a real layout is not self-aliased
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
        const data_layout_t layout(std::vector<size_t>(rank, 0), spans, strides, batch_rank);

        // query the counterpart of either side, for either real transform
        const auto fft_type = uniform(0, 1) ? rocfft_transform_type_real_forward
                                            : rocfft_transform_type_real_inverse;
        const auto herm_io  = fft_type == rocfft_transform_type_real_forward ? io_data_label::OUTPUT
                                                                             : io_data_label::INPUT;
        const bool from_real = uniform(0, 1);
        const auto other_io  = from_real ? herm_io : other(herm_io);
        const bool odd       = uniform(0, 1);
        SCOPED_TRACE("layout " + layout.str() + " / fft_type " + std::to_string(fft_type)
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
        const data_layout_t expected(
            std::vector<size_t>(rank, 0), other_spans, other_strides, batch_rank);

        // a Hermitian innermost length of 1 only comes from a real one of 1
        if(!from_real && n == 1 && !odd)
        {
            EXPECT_THROW(layout.get_other_inplace_layout_for(other_io, fft_type, odd),
                         std::invalid_argument);
            continue;
        }
        const auto counterpart = layout.get_other_inplace_layout_for(other_io, fft_type, odd);
        ASSERT_EQ(counterpart.has_value(), follows_rules);
        if(counterpart)
        {
            EXPECT_TRUE(*counterpart == expected)
                << counterpart->str() << " instead of " << expected.str();
        }
    }
}

TEST_F(data_layout_test, inplace_query_rejects_bad_arguments)
{
    auto layout = data_layout_t::default_full_layout({8}, 1);
    EXPECT_THROW(layout.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                     static_cast<rocfft_transform_type>(99)),
                 std::invalid_argument);

    // a layout covering nothing has no counterpart
    data_layout_t empty({0, 0}, {0, 1}, {1, 1});
    EXPECT_THROW(empty.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                    rocfft_transform_type_real_forward),
                 std::logic_error);

    // a Hermitian innermost length of 1 only comes from a real innermost length of 1, so
    // the real side cannot be requested as even (it would have a length of 0)
    const auto herm_of_one = data_layout_t::default_full_layout({1}, 3);
    EXPECT_THROW(herm_of_one.get_other_inplace_layout_for(
                     io_data_label::INPUT, rocfft_transform_type_real_forward, false),
                 std::invalid_argument);
    const auto real_of_one = herm_of_one.get_other_inplace_layout_for(
        io_data_label::INPUT, rocfft_transform_type_real_forward, true);
    ASSERT_TRUE(real_of_one.has_value());
    EXPECT_EQ(real_of_one->lengths(), (std::vector<size_t>{1}));

    // length axes must be full, batch axes need not be
    for(auto fft_type : {rocfft_transform_type_complex_forward, rocfft_transform_type_real_forward})
    {
        data_layout_t partial_len({0, 1, 0}, {8, 4, 3}, {1, 10, 40});
        EXPECT_THROW(partial_len.get_other_inplace_layout_for(io_data_label::OUTPUT, fft_type),
                     std::logic_error);
        data_layout_t partial_batch({0, 0, 1}, {8, 4, 3}, {1, 10, 40});
        EXPECT_NO_THROW(
            partial_batch.get_other_inplace_layout_for(io_data_label::OUTPUT, fft_type));
    }
}

TEST_F(data_layout_test, axes_by_increasing_stride)
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
        const auto layout = data_layout_t::full_layout(
            random_vec(rank, 1, 8), strides, 1, 1 + *distinct.rbegin());
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

        EXPECT_EQ(layout.length_axes_by_increasing_strides(false), unpinned);
        EXPECT_EQ(layout.length_axes_by_increasing_strides(true), pinned);
    }
}

TEST_F(data_layout_test, embedding_round_trip)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L        = random_full_layout();
        const auto len_rank = L.get_len_rank();
        SCOPED_TRACE("L " + L.str());

        // random non-empty subset S of L's length axes
        auto S_vec = random_permutation(len_rank);
        S_vec.resize(uniform(1, len_rank));
        const std::set<size_t>    S(S_vec.begin(), S_vec.end());
        const std::vector<size_t> sorted_S(S.begin(), S.end());
        SCOPED_TRACE("S " + to_str(sorted_S));

        auto sub = L.get_layout_for_len_axes(S);
        EXPECT_TRUE(sub.get_embedding_layout() == L);
        EXPECT_EQ(sub.corresponding_axes_in_embedding(), sorted_S);

        // subset of a subset still maps back to L
        auto S2_vec = random_permutation(sub.get_len_rank());
        S2_vec.resize(uniform(1, sub.get_len_rank()));
        const std::set<size_t> S2(S2_vec.begin(), S2_vec.end());
        const auto             subsub = sub.get_layout_for_len_axes(S2);
        std::vector<size_t>    subsub_axes;
        for(auto j : S2)
            subsub_axes.push_back(sorted_S[j]);
        EXPECT_TRUE(subsub.get_embedding_layout() == L);
        EXPECT_EQ(subsub.corresponding_axes_in_embedding(), subsub_axes);

        // Reordering keeps the embedding consistent only for embedded layouts;
        // selecting every axis returns a plain copy without an embedding map.
        if(S.size() == len_rank)
            continue;
        ASSERT_TRUE(sub.is_embedded());
        const auto order = random_permutation(sub.get_len_rank());
        SCOPED_TRACE("order " + to_str(order));
        data_layout_test_access::reorder_length_axes(sub, order);
        std::vector<size_t> permuted_axes;
        for(auto j : order)
            permuted_axes.push_back(sorted_S[j]);
        EXPECT_TRUE(sub.get_embedding_layout() == L);
        EXPECT_EQ(sub.corresponding_axes_in_embedding(), permuted_axes);

        // and a subset of the reordered subset still maps back to L
        const auto          sub3 = sub.get_layout_for_len_axes(S2);
        std::vector<size_t> sub3_axes;
        for(auto j : S2)
            sub3_axes.push_back(permuted_axes[j]);
        EXPECT_TRUE(sub3.get_embedding_layout() == L);
        EXPECT_EQ(sub3.corresponding_axes_in_embedding(), sub3_axes);
    }
}

TEST_F(data_layout_test, subset_of_axes_rejects_bad_input)
{
    auto layout = data_layout_t::default_full_layout({4, 8, 2}, 3);
    EXPECT_THROW(layout.get_layout_for_len_axes({}), std::invalid_argument);
    EXPECT_THROW(layout.get_layout_for_len_axes({0, 7}), std::invalid_argument);
}
