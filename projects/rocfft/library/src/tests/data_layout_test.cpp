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

extern size_t random_seed; // internal_test_main.cpp

struct data_layout_test_access
{
    static void reorder_length_axes(data_layout_t& layout, const std::vector<size_t>& order)
    {
        layout.reorder_length_axes(order);
    }
};

namespace
{
    constexpr size_t num_random_cases = 200;

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
// with --seed <value> --gtest_filter=<test>.
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
    // Full layout of random ranks/spans whose axes are laid out in random order,
    // with random padding between consecutive axes (non-aliasing, not packed).
    data_layout_t random_full_layout(size_t min_span = 1)
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

TEST_F(data_layout_test, contiguous_strides)
{
    auto padded = data_layout_t::default_full_layout({4, 8}, 3, true);
    EXPECT_EQ(padded.contiguous_strides_and_distances(), (std::vector<size_t>{1, 4, 32}));

    // 4x8 with strides 8 and 1 still covers every slot exactly once
    auto swapped = data_layout_t::full_layout({4, 8}, {8, 1}, 1, 32);
    EXPECT_TRUE(swapped.is_contiguous());
    EXPECT_EQ(swapped.contiguous_strides_and_distances(), (std::vector<size_t>{8, 1, 32}));
}

TEST_F(data_layout_test, offset_in)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // outer: random sub-range (nonzero lower bounds, batch axes included) of a
        // random full layout; inner: random sub-range of outer, arbitrary strides
        const auto          full         = random_full_layout(2);
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

    auto outer      = data_layout_t::full_layout({8, 8}, {1, 8}, 1, 64);
    auto other_rank = data_layout_t::full_layout({8}, {1}, 1, 8);
    EXPECT_THROW(other_rank.offset_in(outer), std::invalid_argument);
}

TEST_F(data_layout_test, continuous_chunk)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // packed with spans >= 2: the last batch axis is the slowest-varying one
        const auto whole = data_layout_t::default_full_layout(random_vec(uniform(1, 4), 2, 7),
                                                              random_vec(uniform(1, 2), 2, 7));
        const auto rank  = whole.get_full_rank();
        const auto slow  = rank - 1;
        SCOPED_TRACE("whole " + whole.str());

        auto lo  = whole.lower();
        auto hi  = whole.upper();
        lo[slow] = uniform(0, hi[slow] - 1);
        hi[slow] = uniform(lo[slow] + 1, hi[slow]);
        const data_layout_t chunk(lo, hi, whole.strides_and_distances(), whole.get_batch_rank());
        EXPECT_TRUE(chunk.is_continuous_in(whole)) << chunk.str();

        // a different stride along the slowest axis only matters if the chunk spans it
        auto other_strides = whole.strides_and_distances();
        other_strides[slow] += 1;
        const data_layout_t restrided(lo, hi, other_strides, whole.get_batch_rank());
        EXPECT_EQ(restrided.is_continuous_in(whole), hi[slow] - lo[slow] == 1) << restrided.str();

        // narrowing any faster axis breaks continuity
        const auto d = uniform(0, slow - 1);
        hi[d] -= 1;
        const data_layout_t gap(lo, hi, whole.strides_and_distances(), whole.get_batch_rank());
        EXPECT_FALSE(gap.is_continuous_in(whole)) << gap.str();
    }
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
}

TEST_F(data_layout_test, inplace_complex_is_unchanged)
{
    auto in  = data_layout_t::default_full_layout({16}, 4);
    auto out = in.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                               rocfft_transform_type_complex_forward);
    ASSERT_TRUE(out.has_value());
    EXPECT_TRUE(*out == in);
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
        const auto real = data_layout_t::default_full_layout(L, B, true);
        SCOPED_TRACE("real " + real.str());

        for(auto [fft_type, herm_io] :
            {std::pair{rocfft_transform_type_real_forward, io_data_label::OUTPUT},
             std::pair{rocfft_transform_type_real_inverse, io_data_label::INPUT}})
        {
            const auto real_io = other(herm_io);
            const auto herm    = real.get_other_inplace_layout_for(herm_io, fft_type);
            ASSERT_TRUE(herm.has_value());
            EXPECT_EQ(herm->lengths()[0], L[0] / 2 + 1);

            const auto back = herm->get_other_inplace_layout_for(real_io, fft_type, odd);
            ASSERT_TRUE(back.has_value());
            EXPECT_TRUE(*back == real) << back->str();

            const auto wrong = herm->get_other_inplace_layout_for(real_io, fft_type, !odd);
            ASSERT_TRUE(wrong.has_value());
            EXPECT_FALSE(*wrong == real) << wrong->str();
        }
    }
}

TEST_F(data_layout_test, inplace_real_refuses_impossible_layouts)
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
