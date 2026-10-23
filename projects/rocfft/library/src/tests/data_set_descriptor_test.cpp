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
#include "data_descriptor_test_access.h"
#include "data_set_descriptor.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

extern size_t random_seed; // internal_test_main.cpp

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

    // copy of a full layout (no embedding) with a precision, an array type and offsets
    data_descriptor_t with_type(const data_descriptor_t&     layout,
                                rocfft_precision             p,
                                rocfft_array_type            t,
                                const std::array<size_t, 2>& offsets = {0, 0})
    {
        return data_descriptor_t(p,
                                 t,
                                 layout.lower(),
                                 layout.upper(),
                                 layout.strides_and_distances(),
                                 layout.get_batch_rank(),
                                 offsets);
    }

    // In-buffer offset (sum of stride * coordinate) of every logical index, axis 0 fastest.
    std::vector<size_t> all_offsets(const data_descriptor_t& d)
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
// with --seed <value> --gtest_filter=<test>.
class data_descriptor_test : public ::testing::Test
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
    data_descriptor_t random_full_layout(size_t min_span = 1)
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
        return data_descriptor_t(std::vector<size_t>(spans.size(), 0), spans, strides, batch_rank);
    }
};

TEST_F(data_descriptor_test, constructor_rejects_bad_input)
{
    // no batch axis at all
    EXPECT_THROW(data_descriptor_t({0, 0}, {4, 8}, {1, 4}, 0), std::invalid_argument);
    // no length axis left once batch axes are taken
    EXPECT_THROW(data_descriptor_t({0, 0}, {4, 8}, {1, 4}, 2), std::invalid_argument);
    // lower, upper and strides must be the same size
    EXPECT_THROW(data_descriptor_t({0, 0, 0}, {4, 8}, {1, 4, 32}), std::invalid_argument);
    // an axis whose lower bound is past its upper bound
    EXPECT_THROW(data_descriptor_t({5, 0, 0}, {4, 8, 1}, {1, 4, 32}), std::invalid_argument);
}

TEST_F(data_descriptor_test, single_batch_axis_accessors)
{
    auto one = data_descriptor_t::default_full_layout({4, 8}, 3);
    EXPECT_EQ(one.batch(), 3u);
    EXPECT_EQ(one.distance(), 32u);

    // two batch axes: length 4, batches 8 and 3
    data_descriptor_t two({0, 0, 0}, {4, 8, 3}, {1, 4, 32}, 2);
    EXPECT_EQ(two.get_len_rank(), 1u);
    EXPECT_EQ(two.get_batch_rank(), 2u);
    EXPECT_THROW(two.batch(), std::logic_error);
    EXPECT_THROW(two.distance(), std::logic_error);
}

// The client helpers are row-major (last index fastest) while data_descriptor_t lists
// the fastest axis first, so lengths, batches and results are reversed.
TEST_F(data_descriptor_test, default_layouts_match_client_helpers)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L = random_vec(uniform(1, 4), 1, 16);
        const auto B = random_vec(uniform(1, 3), 1, 5);
        SCOPED_TRACE("lengths " + to_str(L) + " batches " + to_str(B));
        const auto rL = reversed(L);
        const auto rB = reversed(B);

        const auto cplx = data_descriptor_t::default_full_layout(L, B);
        EXPECT_EQ(cplx.strides(),
                  reversed(default_strides(
                      fft_transform_type_complex_forward, fft_placement_inplace, fft_io_in, rL)));
        EXPECT_EQ(
            cplx.distances(),
            reversed(default_distances(
                fft_transform_type_complex_forward, fft_placement_inplace, fft_io_in, rL, rB)));

        const auto real = data_descriptor_t::default_full_layout(L, B, true);
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
TEST_F(data_descriptor_test, padded_buffer_element_count)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L = random_vec(uniform(1, 4), 1, 16);
        const auto B = random_vec(uniform(1, 3), 1, 5);
        SCOPED_TRACE("lengths " + to_str(L) + " batches " + to_str(B));

        const auto packed = data_descriptor_t::default_full_layout(L, B);
        const auto padded = data_descriptor_t::default_full_layout(L, B, true);
        const auto n      = L[0];
        const auto P      = 2 * (n / 2 + 1);
        const auto N      = packed.buffer_element_count();
        EXPECT_EQ(N, packed.logical_count());
        EXPECT_EQ(padded.logical_count(), packed.logical_count());
        EXPECT_EQ(padded.buffer_element_count(), P * (N / n) - (P - n));
    }
}

TEST_F(data_descriptor_test, contiguous_strides)
{
    auto padded = data_descriptor_t::default_full_layout({4, 8}, 3, true);
    EXPECT_EQ(padded.contiguous_strides_and_distances(), (std::vector<size_t>{1, 4, 32}));

    // 4x8 with strides 8 and 1 still covers every slot exactly once
    auto swapped = data_descriptor_t::full_layout({4, 8}, {8, 1}, 1, 32);
    EXPECT_TRUE(swapped.is_contiguous());
    EXPECT_EQ(swapped.contiguous_strides_and_distances(), (std::vector<size_t>{8, 1, 32}));
}

TEST_F(data_descriptor_test, offset_in)
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
        const data_descriptor_t outer(outer_lo, outer_hi, outer_stride, full.get_batch_rank());
        const data_descriptor_t inner(
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
        const data_descriptor_t outside(bad_lo, bad_hi, outer_stride, full.get_batch_rank());
        EXPECT_THROW(outside.offset_in(outer), std::invalid_argument);
    }

    auto outer      = data_descriptor_t::full_layout({8, 8}, {1, 8}, 1, 64);
    auto other_rank = data_descriptor_t::full_layout({8}, {1}, 1, 8);
    EXPECT_THROW(other_rank.offset_in(outer), std::invalid_argument);
}

TEST_F(data_descriptor_test, continuous_chunk)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // packed with spans >= 2: the last batch axis is the slowest-varying one
        const auto whole = data_descriptor_t::default_full_layout(random_vec(uniform(1, 4), 2, 7),
                                                                  random_vec(uniform(1, 2), 2, 7));
        const auto rank  = whole.get_full_rank();
        const auto slow  = rank - 1;
        SCOPED_TRACE("whole " + whole.str());

        auto lo  = whole.lower();
        auto hi  = whole.upper();
        lo[slow] = uniform(0, hi[slow] - 1);
        hi[slow] = uniform(lo[slow] + 1, hi[slow]);
        const data_descriptor_t chunk(
            lo, hi, whole.strides_and_distances(), whole.get_batch_rank());
        EXPECT_TRUE(chunk.is_continuous_in(whole)) << chunk.str();

        // a different stride along the slowest axis only matters if the chunk spans it
        auto other_strides = whole.strides_and_distances();
        other_strides[slow] += 1;
        const data_descriptor_t restrided(lo, hi, other_strides, whole.get_batch_rank());
        EXPECT_EQ(restrided.is_continuous_in(whole), hi[slow] - lo[slow] == 1) << restrided.str();

        // narrowing any faster axis breaks continuity
        const auto d = uniform(0, slow - 1);
        hi[d] -= 1;
        const data_descriptor_t gap(lo, hi, whole.strides_and_distances(), whole.get_batch_rank());
        EXPECT_FALSE(gap.is_continuous_in(whole)) << gap.str();
    }
}

TEST_F(data_descriptor_test, intersection)
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
        const data_descriptor_t first(
            lo[0], hi[0], full.strides_and_distances(), full.get_batch_rank());
        const data_descriptor_t second(
            lo[1], hi[1], random_vec(rank, 1, 100), full.get_batch_rank());
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
        const auto overlap = data_descriptor_t::make_contiguous_intersection_of(first, second);
        EXPECT_EQ(overlap.lower(), exp_lo);
        EXPECT_EQ(overlap.upper(), exp_hi);
        EXPECT_EQ(overlap.logical_count(), count);
        EXPECT_EQ(overlap.is_empty(), count == 0);
        EXPECT_EQ(overlap.strides_and_distances(), exp_stride);
    }
}

TEST_F(data_descriptor_test, inplace_complex_is_unchanged)
{
    auto in  = data_descriptor_t::default_full_layout({16}, 4);
    auto out = in.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                               rocfft_transform_type_complex_forward);
    ASSERT_TRUE(out.has_value());
    EXPECT_TRUE(*out == in);
}

// real -> Hermitian -> real gives back the original layout, for both real
// transform directions and for even and odd innermost lengths; the wrong parity
// flag must not.
TEST_F(data_descriptor_test, inplace_real_round_trip)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L    = random_vec(uniform(1, 4), 1, 33);
        const auto B    = random_vec(uniform(1, 3), 1, 5);
        const bool odd  = L[0] % 2 == 1;
        const auto real = data_descriptor_t::default_full_layout(L, B, true);
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

TEST_F(data_descriptor_test, inplace_real_refuses_impossible_layouts)
{
    // batch distance is odd, so it cannot also be a whole number of complex
    // values.  9 is too small to hold the half spectrum as well, so use 11 to
    // check the odd-ness rule on its own.
    for(size_t odd_distance : {size_t{9}, size_t{11}})
    {
        auto odd = data_descriptor_t::full_layout({8}, {1}, 3, odd_distance);
        EXPECT_FALSE(odd.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                      rocfft_transform_type_real_forward)
                         .has_value())
            << "distance " << odd_distance;
    }

    // batch distance is even but too small to hold the half spectrum
    auto too_short = data_descriptor_t::full_layout({8}, {1}, 3, 8);
    EXPECT_FALSE(
        too_short
            .get_other_inplace_layout_for(io_data_label::OUTPUT, rocfft_transform_type_real_forward)
            .has_value());

    // the real side must be packed along its innermost axis
    auto strided = data_descriptor_t::full_layout({8}, {2}, 3, 20);
    EXPECT_FALSE(
        strided
            .get_other_inplace_layout_for(io_data_label::OUTPUT, rocfft_transform_type_real_forward)
            .has_value());
}

TEST_F(data_descriptor_test, inplace_query_rejects_bad_arguments)
{
    auto layout = data_descriptor_t::default_full_layout({8}, 1);
    EXPECT_THROW(layout.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                     static_cast<rocfft_transform_type>(99)),
                 std::invalid_argument);

    // a layout covering nothing has no counterpart
    data_descriptor_t empty({0, 0}, {0, 1}, {1, 1});
    EXPECT_THROW(empty.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                    rocfft_transform_type_real_forward),
                 std::logic_error);

    // length axes must be full, batch axes need not be
    for(auto fft_type : {rocfft_transform_type_complex_forward, rocfft_transform_type_real_forward})
    {
        data_descriptor_t partial_len({0, 1, 0}, {8, 4, 3}, {1, 10, 40});
        EXPECT_THROW(partial_len.get_other_inplace_layout_for(io_data_label::OUTPUT, fft_type),
                     std::logic_error);
        data_descriptor_t partial_batch({0, 0, 1}, {8, 4, 3}, {1, 10, 40});
        EXPECT_NO_THROW(
            partial_batch.get_other_inplace_layout_for(io_data_label::OUTPUT, fft_type));
    }
}

TEST_F(data_descriptor_test, axes_by_increasing_stride)
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
        const auto layout = data_descriptor_t::full_layout(
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

TEST_F(data_descriptor_test, embedding_round_trip)
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
        data_descriptor_test_access::reorder_length_axes(sub, order);
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

TEST_F(data_descriptor_test, subset_of_axes_rejects_bad_input)
{
    auto layout = data_descriptor_t::default_full_layout({4, 8, 2}, 3);
    EXPECT_THROW(layout.get_layout_for_len_axes({}), std::invalid_argument);
    EXPECT_THROW(layout.get_layout_for_len_axes({0, 7}), std::invalid_argument);
}

TEST_F(data_descriptor_test, type_and_precision_unset_by_default)
{
    for(const auto& layout : {data_descriptor_t({0, 0}, {4, 3}, {1, 4}),
                              data_descriptor_t::full_layout({4}, {1}, 3, 4),
                              data_descriptor_t::default_full_layout({4, 8}, 3),
                              data_descriptor_t::default_full_layout({4, 8}, {3, 2}, true)})
    {
        SCOPED_TRACE(layout.str());
        EXPECT_EQ(layout.get_array_type(), rocfft_array_type_unset);
        EXPECT_FALSE(layout.has_precision());
        EXPECT_THROW(layout.get_precision(), std::logic_error);
        EXPECT_THROW(layout.element_size(), std::logic_error);
        EXPECT_THROW(layout.buffer_size_bytes(), std::logic_error);
    }
}

TEST_F(data_descriptor_test, type_and_precision_constructor)
{
    const data_descriptor_t layout(
        rocfft_precision_double, rocfft_array_type_complex_planar, {0, 1}, {4, 3}, {1, 4});
    EXPECT_EQ(layout.get_array_type(), rocfft_array_type_complex_planar);
    ASSERT_TRUE(layout.has_precision());
    EXPECT_EQ(layout.get_precision(), rocfft_precision_double);
    EXPECT_EQ(layout.lower(), (std::vector<size_t>{0, 1}));
    EXPECT_EQ(layout.upper(), (std::vector<size_t>{4, 3}));
    EXPECT_EQ(layout.strides_and_distances(), (std::vector<size_t>{1, 4}));
    EXPECT_EQ(layout.get_batch_rank(), 1u);

    // same input validation as the overload without precision and array type
    EXPECT_THROW(data_descriptor_t(
                     rocfft_precision_single, rocfft_array_type_real, {0, 0}, {4, 8}, {1, 4}, 0),
                 std::invalid_argument);
    EXPECT_THROW(
        data_descriptor_t(rocfft_precision_single, rocfft_array_type_real, {5, 0}, {4, 8}, {1, 4}),
        std::invalid_argument);
}

TEST_F(data_descriptor_test, type_and_precision_pass_through_subsets_and_embeddings)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto L = with_type(
            random_full_layout(), rocfft_precision_half, rocfft_array_type_hermitian_planar);
        SCOPED_TRACE("L " + L.str());

        auto S_vec = random_permutation(L.get_len_rank());
        S_vec.resize(uniform(1, L.get_len_rank()));
        const auto sub = L.get_layout_for_len_axes(std::set<size_t>(S_vec.begin(), S_vec.end()));
        EXPECT_EQ(sub.get_array_type(), rocfft_array_type_hermitian_planar);
        EXPECT_EQ(sub.get_precision(), rocfft_precision_half);
        // strict equality: the embedding layout carries both values too
        EXPECT_TRUE(sub.get_embedding_layout() == L);
    }

    // unset stays unset
    const auto sub = data_descriptor_t::default_full_layout({4, 8}, 3).get_layout_for_len_axes({1});
    EXPECT_EQ(sub.get_array_type(), rocfft_array_type_unset);
    EXPECT_FALSE(sub.has_precision());
    EXPECT_EQ(sub.get_embedding_layout().get_array_type(), rocfft_array_type_unset);
    EXPECT_FALSE(sub.get_embedding_layout().has_precision());
}

TEST_F(data_descriptor_test, type_and_precision_inplace_mapping)
{
    // real <-> hermitian_interleaved for real transforms, precision kept
    const auto real = with_type(data_descriptor_t::default_full_layout({16}, 4, true),
                                rocfft_precision_single,
                                rocfft_array_type_real);
    for(auto [fft_type, herm_io] :
        {std::pair{rocfft_transform_type_real_forward, io_data_label::OUTPUT},
         std::pair{rocfft_transform_type_real_inverse, io_data_label::INPUT}})
    {
        const auto herm = real.get_other_inplace_layout_for(herm_io, fft_type);
        ASSERT_TRUE(herm.has_value());
        EXPECT_EQ(herm->get_array_type(), rocfft_array_type_hermitian_interleaved);
        EXPECT_EQ(herm->get_precision(), rocfft_precision_single);

        const auto back = herm->get_other_inplace_layout_for(other(herm_io), fft_type);
        ASSERT_TRUE(back.has_value());
        EXPECT_TRUE(*back == real) << back->str();
    }

    // complex transforms keep the array type and precision
    for(auto type : {rocfft_array_type_complex_interleaved, rocfft_array_type_complex_planar})
    {
        const auto in = with_type(
            data_descriptor_t::default_full_layout({16}, 4), rocfft_precision_double, type);
        const auto out = in.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                         rocfft_transform_type_complex_inverse);
        ASSERT_TRUE(out.has_value());
        EXPECT_TRUE(*out == in) << out->str();
    }

    // unset stays unset
    const auto unset = data_descriptor_t::default_full_layout({16}, 4, true)
                           .get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                         rocfft_transform_type_real_forward);
    ASSERT_TRUE(unset.has_value());
    EXPECT_EQ(unset->get_array_type(), rocfft_array_type_unset);
    EXPECT_FALSE(unset->has_precision());
}

TEST_F(data_descriptor_test, type_and_precision_intersection)
{
    const auto base = data_descriptor_t::default_full_layout({8, 4}, 2);
    const std::pair<rocfft_array_type, rocfft_array_type> accepted[]
        = {{rocfft_array_type_complex_interleaved, rocfft_array_type_complex_interleaved},
           {rocfft_array_type_complex_interleaved, rocfft_array_type_hermitian_interleaved},
           {rocfft_array_type_hermitian_interleaved, rocfft_array_type_complex_interleaved},
           {rocfft_array_type_complex_planar, rocfft_array_type_hermitian_planar},
           {rocfft_array_type_hermitian_planar, rocfft_array_type_complex_planar},
           {rocfft_array_type_real, rocfft_array_type_real}};
    for(auto [a, b] : accepted)
    {
        const auto xsect = data_descriptor_t::make_contiguous_intersection_of(
            with_type(base, rocfft_precision_single, a),
            with_type(base, rocfft_precision_single, b));
        EXPECT_EQ(xsect.get_array_type(), a);
        EXPECT_EQ(xsect.get_precision(), rocfft_precision_single);
    }

    // unset matches unset only
    const auto unset = data_descriptor_t::make_contiguous_intersection_of(base, base);
    EXPECT_EQ(unset.get_array_type(), rocfft_array_type_unset);
    EXPECT_FALSE(unset.has_precision());
    const auto ci_single
        = with_type(base, rocfft_precision_single, rocfft_array_type_complex_interleaved);
    EXPECT_THROW(data_descriptor_t::make_contiguous_intersection_of(base, ci_single),
                 std::invalid_argument);

    // precision mismatch
    EXPECT_THROW(
        data_descriptor_t::make_contiguous_intersection_of(
            ci_single,
            with_type(base, rocfft_precision_double, rocfft_array_type_complex_interleaved)),
        std::invalid_argument);

    // interleaved vs planar, real vs complex
    const std::pair<rocfft_array_type, rocfft_array_type> rejected[]
        = {{rocfft_array_type_complex_interleaved, rocfft_array_type_complex_planar},
           {rocfft_array_type_hermitian_interleaved, rocfft_array_type_hermitian_planar},
           {rocfft_array_type_complex_planar, rocfft_array_type_hermitian_interleaved},
           {rocfft_array_type_real, rocfft_array_type_complex_interleaved},
           {rocfft_array_type_hermitian_interleaved, rocfft_array_type_real}};
    for(auto [a, b] : rejected)
    {
        EXPECT_THROW(data_descriptor_t::make_contiguous_intersection_of(
                         with_type(base, rocfft_precision_single, a),
                         with_type(base, rocfft_precision_single, b)),
                     std::invalid_argument);
    }
}

TEST_F(data_descriptor_test, type_and_precision_equality_is_strict)
{
    const auto base = data_descriptor_t::default_full_layout({8, 4}, 2);
    const auto ci_single
        = with_type(base, rocfft_precision_single, rocfft_array_type_complex_interleaved);
    EXPECT_TRUE(ci_single
                == with_type(base, rocfft_precision_single, rocfft_array_type_complex_interleaved));
    EXPECT_FALSE(ci_single == base);
    EXPECT_FALSE(base == ci_single);
    EXPECT_FALSE(
        ci_single
        == with_type(base, rocfft_precision_double, rocfft_array_type_complex_interleaved));
    EXPECT_FALSE(
        ci_single
        == with_type(base, rocfft_precision_single, rocfft_array_type_hermitian_interleaved));

    auto ci_only = base;
    data_descriptor_test_access::set_array_type(ci_only, rocfft_array_type_complex_interleaved);
    EXPECT_FALSE(ci_only == base);
    EXPECT_FALSE(ci_only == ci_single);
}

TEST_F(data_descriptor_test, type_and_precision_in_str)
{
    const auto base = data_descriptor_t::default_full_layout({8}, 2);
    EXPECT_NE(base.str().find("precision unset"), std::string::npos) << base.str();
    EXPECT_NE(base.str().find("array type unset"), std::string::npos) << base.str();

    const auto typed = with_type(base, rocfft_precision_double, rocfft_array_type_hermitian_planar);
    EXPECT_NE(typed.str().find("precision double"), std::string::npos) << typed.str();
    EXPECT_NE(typed.str().find("array type hermitian_planar"), std::string::npos) << typed.str();
}

TEST_F(data_descriptor_test, type_and_precision_survive_clear_and_reset)
{
    auto layout = with_type(data_descriptor_t::default_full_layout({8, 4}, 2),
                            rocfft_precision_half,
                            rocfft_array_type_real);

    data_descriptor_test_access::clear(layout);
    EXPECT_EQ(layout.get_full_rank(), 0u);
    EXPECT_EQ(layout.get_array_type(), rocfft_array_type_real);
    EXPECT_EQ(layout.get_precision(), rocfft_precision_half);

    data_descriptor_test_access::full_range_reset(layout, {16}, 3);
    EXPECT_EQ(layout.lengths(), std::vector<size_t>{16});
    EXPECT_EQ(layout.batches(), std::vector<size_t>{3});
    EXPECT_EQ(layout.get_array_type(), rocfft_array_type_real);
    EXPECT_EQ(layout.get_precision(), rocfft_precision_half);
}

TEST_F(data_descriptor_test, element_and_buffer_sizes)
{
    const auto base = data_descriptor_t::default_full_layout({8, 4}, 2);
    ASSERT_EQ(base.buffer_element_count(), 64u);
    const struct
    {
        rocfft_precision  precision;
        rocfft_array_type array_type;
        size_t            bytes;
    } cases[] = {{rocfft_precision_single, rocfft_array_type_complex_interleaved, 8},
                 {rocfft_precision_double, rocfft_array_type_hermitian_interleaved, 16},
                 {rocfft_precision_half, rocfft_array_type_complex_interleaved, 4},
                 {rocfft_precision_single, rocfft_array_type_real, 4},
                 {rocfft_precision_double, rocfft_array_type_real, 8},
                 // planar: bytes per plane
                 {rocfft_precision_single, rocfft_array_type_complex_planar, 4},
                 {rocfft_precision_double, rocfft_array_type_hermitian_planar, 8}};
    for(const auto& c : cases)
    {
        const auto layout = with_type(base, c.precision, c.array_type);
        SCOPED_TRACE(layout.str());
        EXPECT_EQ(layout.element_size(), c.bytes);
        EXPECT_EQ(layout.buffer_size_bytes(), 64 * c.bytes);
    }

    // padding inside the buffer is counted, trailing padding is not
    const auto padded = with_type(data_descriptor_t::default_full_layout({8}, 2, true),
                                  rocfft_precision_single,
                                  rocfft_array_type_real);
    EXPECT_EQ(padded.buffer_size_bytes(), (10 + 8) * 4u);

    // either value unset throws
    const auto no_type = with_type(base, rocfft_precision_single, rocfft_array_type_unset);
    EXPECT_THROW(no_type.element_size(), std::logic_error);
    EXPECT_THROW(no_type.buffer_size_bytes(), std::logic_error);
    auto no_precision = base;
    data_descriptor_test_access::set_array_type(no_precision, rocfft_array_type_real);
    EXPECT_THROW(no_precision.element_size(), std::logic_error);
    EXPECT_THROW(no_precision.buffer_size_bytes(), std::logic_error);
}

TEST_F(data_descriptor_test, reinterpret_as)
{
    const auto base = data_descriptor_t::default_full_layout({8, 4}, 2);

    // relabeling only, precision unchanged
    const std::pair<rocfft_array_type, rocfft_array_type> allowed[]
        = {{rocfft_array_type_complex_interleaved, rocfft_array_type_complex_interleaved},
           {rocfft_array_type_complex_interleaved, rocfft_array_type_hermitian_interleaved},
           {rocfft_array_type_hermitian_interleaved, rocfft_array_type_complex_interleaved},
           {rocfft_array_type_complex_planar, rocfft_array_type_hermitian_planar},
           {rocfft_array_type_hermitian_planar, rocfft_array_type_complex_planar},
           {rocfft_array_type_real, rocfft_array_type_real}};
    for(auto [from, to] : allowed)
    {
        const auto src = with_type(base, rocfft_precision_double, from);
        const auto dst = src.reinterpret_as(to);
        EXPECT_EQ(dst.get_array_type(), to);
        EXPECT_EQ(dst.get_precision(), rocfft_precision_double);
        EXPECT_TRUE(dst.reinterpret_as(from) == src) << dst.str();
    }

    // an unset precision stays unset
    auto no_precision = base;
    data_descriptor_test_access::set_array_type(no_precision,
                                                rocfft_array_type_complex_interleaved);
    EXPECT_FALSE(
        no_precision.reinterpret_as(rocfft_array_type_hermitian_interleaved).has_precision());

    // no request is left unimplemented: failures are std::invalid_argument only
    for(auto from : {rocfft_array_type_real,
                     rocfft_array_type_complex_interleaved,
                     rocfft_array_type_hermitian_interleaved,
                     rocfft_array_type_complex_planar,
                     rocfft_array_type_hermitian_planar})
    {
        for(auto to : {rocfft_array_type_real,
                       rocfft_array_type_complex_interleaved,
                       rocfft_array_type_hermitian_interleaved,
                       rocfft_array_type_complex_planar,
                       rocfft_array_type_hermitian_planar})
        {
            try
            {
                with_type(base, rocfft_precision_single, from).reinterpret_as(to);
            }
            catch(const std::invalid_argument&)
            {
            }
            catch(const std::logic_error& e)
            {
                ADD_FAILURE() << from << " -> " << to << ": " << e.what();
            }
        }
    }

    // interleaved <-> planar and unset source/target are rejected
    const std::pair<rocfft_array_type, rocfft_array_type> rejected[]
        = {{rocfft_array_type_complex_interleaved, rocfft_array_type_complex_planar},
           {rocfft_array_type_hermitian_planar, rocfft_array_type_hermitian_interleaved},
           {rocfft_array_type_complex_planar, rocfft_array_type_hermitian_interleaved},
           {rocfft_array_type_complex_interleaved, rocfft_array_type_unset}};
    for(auto [from, to] : rejected)
    {
        EXPECT_THROW(with_type(base, rocfft_precision_single, from).reinterpret_as(to),
                     std::invalid_argument);
    }
    EXPECT_THROW(base.reinterpret_as(rocfft_array_type_complex_interleaved), std::invalid_argument);
}

namespace
{
    const rocfft_array_type interleaved_types[]
        = {rocfft_array_type_complex_interleaved, rocfft_array_type_hermitian_interleaved};

    const rocfft_array_type planar_types[]
        = {rocfft_array_type_complex_planar, rocfft_array_type_hermitian_planar};

    // in-buffer position (in elements, offset included) of logical coordinate `coord`
    // in plane `plane`
    size_t position_of(const data_descriptor_t&   layout,
                       const std::vector<size_t>& coord,
                       size_t                     plane = 0)
    {
        const auto strides = layout.strides_and_distances();
        size_t     ret     = layout.get_offset(plane);
        for(size_t d = 0; d < coord.size(); d++)
            ret += coord[d] * strides[d];
        return ret;
    }

    // calls f(coord) for every logical coordinate of a non-empty layout
    template <typename F>
    void for_each_coordinate(const data_descriptor_t& layout, F f)
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

TEST_F(data_descriptor_test, reinterpret_real_as_interleaved)
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
                const data_descriptor_t real(p,
                                             rocfft_array_type_real,
                                             r.lower,
                                             r.upper,
                                             r.strides,
                                             batch_rank,
                                             {r.offset, 0});
                const data_descriptor_t expected(
                    p, t, c.lower, c.upper, c.strides, batch_rank, {c.offset, 0});
                SCOPED_TRACE(real.str());
                const auto dst = real.reinterpret_as(t);
                EXPECT_TRUE(dst == expected) << dst.str();
                EXPECT_EQ(dst.get_precision(), p);
                EXPECT_TRUE(expected.reinterpret_as(rocfft_array_type_real) == real);
            }
        }
    }
}

TEST_F(data_descriptor_test, reinterpret_real_interleaved_round_trip_and_positions)
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
        const data_descriptor_t cplx(p, t, lower, upper, strides, batch_rank, {uniform(0, 9), 0});
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
            EXPECT_EQ(position_of(real, real_coord) * real.element_size(),
                      position_of(cplx, coord) * cplx.element_size());
            real_coord[0] += 1;
            EXPECT_EQ(position_of(real, real_coord), 2 * position_of(cplx, coord) + 1);
        });
    }
}

TEST_F(data_descriptor_test, reinterpret_real_interleaved_rejects_bad_layouts)
{
    // 2D real layout with two batch axes, valid unless modified
    const std::vector<size_t> lower{0, 0, 0, 0}, upper{8, 4, 3, 2}, strides{1, 10, 40, 120};
    const auto                make = [&](rocfft_array_type          t,
                          const std::vector<size_t>& lo,
                          const std::vector<size_t>& up,
                          const std::vector<size_t>& st,
                          size_t                     offset) {
        return data_descriptor_t(rocfft_precision_single, t, lo, up, st, 2, {offset, 0});
    };
    const auto with = [](std::vector<size_t> v, size_t idx, size_t value) {
        v[idx] = value;
        return v;
    };
    for(auto t : interleaved_types)
    {
        EXPECT_NO_THROW(make(rocfft_array_type_real, lower, upper, strides, 2).reinterpret_as(t));
        const data_descriptor_t rejected[]
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

TEST_F(data_descriptor_test, reinterpret_real_interleaved_embedded)
{
    const data_descriptor_t L(rocfft_precision_double,
                              rocfft_array_type_real,
                              {0, 0, 0, 0},
                              {8, 4, 6, 3},
                              {1, 10, 40, 240},
                              1,
                              {4, 0});
    for(auto t : interleaved_types)
    {
        // length axis 0 of the sub-layout is L's: the embedding map is kept and
        // reinterpreting commutes with get_embedding_layout()
        const auto sub = L.get_layout_for_len_axes({0, 2});
        ASSERT_TRUE(sub.is_embedded());
        const auto dst = sub.reinterpret_as(t);
        EXPECT_TRUE(dst.is_embedded());
        EXPECT_TRUE(dst.get_embedding_layout() == L.reinterpret_as(t)) << dst.str();
        EXPECT_EQ(dst.corresponding_axes_in_embedding(), sub.corresponding_axes_in_embedding());
        const auto back = dst.reinterpret_as(rocfft_array_type_real);
        EXPECT_TRUE(back == sub);
        EXPECT_TRUE(back.get_embedding_layout() == L);

        // length axis 0 of the sub-layout is not L_t's, although it is otherwise
        // reinterpretable (unit stride, even bounds, even other strides and offset)
        const data_descriptor_t L_t(rocfft_precision_double,
                                    rocfft_array_type_real,
                                    {0, 0, 0, 0},
                                    {4, 8, 6, 3},
                                    {8, 1, 32, 192},
                                    1,
                                    {4, 0});
        EXPECT_THROW(L_t.get_layout_for_len_axes({1}).reinterpret_as(t), std::invalid_argument);
        EXPECT_THROW(L_t.get_layout_for_len_axes({1, 2}).reinterpret_as(t), std::invalid_argument);
        const data_descriptor_t L_t_cplx(
            rocfft_precision_double, t, {0, 0, 0, 0}, {4, 8, 6, 3}, {8, 1, 32, 192}, 1, {3, 0});
        EXPECT_THROW(L_t_cplx.get_layout_for_len_axes({1}).reinterpret_as(rocfft_array_type_real),
                     std::invalid_argument);
    }
}

TEST_F(data_descriptor_test, reinterpret_real_as_planar)
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
                const data_descriptor_t real(
                    p, rocfft_array_type_real, r.lower, r.upper, r.strides, batch_rank, r.offsets);
                const data_descriptor_t expected(
                    p, t, c.lower, c.upper, c.strides, batch_rank, c.offsets);
                SCOPED_TRACE(real.str());
                const auto dst = real.reinterpret_as(t);
                EXPECT_TRUE(dst == expected) << dst.str();
                EXPECT_EQ(dst.get_precision(), p);
                EXPECT_TRUE(expected.reinterpret_as(rocfft_array_type_real) == real);
            }
        }
    }
}

TEST_F(data_descriptor_test, reinterpret_real_planar_round_trip_and_positions)
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
        auto strides     = random_vec(len_rank + batch_rank, 1, 40);
        strides[0]       = 2 * uniform(1, 5);
        const auto p     = uniform(0, 1) ? rocfft_precision_single : rocfft_precision_double;
        const auto t     = planar_types[uniform(0, 1)];
        const auto first = uniform(0, 9);
        const data_descriptor_t planar(
            p, t, lower, upper, strides, batch_rank, {first, first + strides[0] / 2});
        SCOPED_TRACE(planar.str());

        const auto real = planar.reinterpret_as(rocfft_array_type_real);
        SCOPED_TRACE(real.str());
        EXPECT_EQ(real.get_precision(), p);
        EXPECT_EQ(real.logical_count(), 2 * planar.logical_count());
        EXPECT_TRUE(real.reinterpret_as(t) == planar);
        for(auto t2 : planar_types)
            EXPECT_TRUE(real.reinterpret_as(t2).reinterpret_as(rocfft_array_type_real) == real);

        // real element 2k (resp. 2k + 1) is element k of plane 0 (resp. 1), at the
        // same byte position
        for_each_coordinate(planar, [&](const std::vector<size_t>& coord) {
            auto real_coord = coord;
            real_coord[0] *= 2;
            EXPECT_EQ(position_of(real, real_coord), position_of(planar, coord, 0));
            EXPECT_EQ(position_of(real, real_coord) * real.element_size(),
                      position_of(planar, coord, 0) * planar.element_size());
            real_coord[0] += 1;
            EXPECT_EQ(position_of(real, real_coord), position_of(planar, coord, 1));
        });
    }
}

TEST_F(data_descriptor_test, reinterpret_real_planar_rejects_bad_layouts)
{
    for(auto t : planar_types)
    {
        // real to planar requires even bounds along length axis 0 only
        const auto real = [](const std::vector<size_t>& lower, const std::vector<size_t>& upper) {
            return data_descriptor_t(rocfft_precision_single,
                                     rocfft_array_type_real,
                                     lower,
                                     upper,
                                     {3, 25, 101, 303},
                                     2,
                                     {7, 0});
        };
        EXPECT_NO_THROW(real({2, 0, 0, 0}, {8, 4, 3, 2}).reinterpret_as(t));
        EXPECT_THROW(real({1, 0, 0, 0}, {8, 4, 3, 2}).reinterpret_as(t), std::invalid_argument);
        EXPECT_THROW(real({2, 0, 0, 0}, {7, 4, 3, 2}).reinterpret_as(t), std::invalid_argument);

        // planar to real requires an even stride along length axis 0 and planes
        // offset by half of it
        const auto planar = [t](size_t stride0, const std::array<size_t, 2>& offsets) {
            return data_descriptor_t(rocfft_precision_single,
                                     t,
                                     {1, 0, 0, 0},
                                     {7, 4, 3, 2},
                                     {stride0, 25, 101, 303},
                                     2,
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

TEST_F(data_descriptor_test, reinterpret_real_planar_embedded)
{
    const data_descriptor_t L(rocfft_precision_double,
                              rocfft_array_type_real,
                              {0, 0, 0, 0},
                              {8, 4, 6, 3},
                              {1, 10, 40, 240},
                              1,
                              {4, 0});
    for(auto t : planar_types)
    {
        // length axis 0 of the sub-layout is L's: the embedding map is kept and
        // reinterpreting commutes with get_embedding_layout()
        const auto sub = L.get_layout_for_len_axes({0, 2});
        ASSERT_TRUE(sub.is_embedded());
        const auto dst = sub.reinterpret_as(t);
        EXPECT_TRUE(dst.is_embedded());
        EXPECT_TRUE(dst.get_embedding_layout() == L.reinterpret_as(t)) << dst.str();
        EXPECT_EQ(dst.corresponding_axes_in_embedding(), sub.corresponding_axes_in_embedding());
        const auto back = dst.reinterpret_as(rocfft_array_type_real);
        EXPECT_TRUE(back == sub);
        EXPECT_TRUE(back.get_embedding_layout() == L);

        // length axis 0 of the sub-layout is not the embedding's, although it is
        // otherwise reinterpretable
        const data_descriptor_t L_t(rocfft_precision_double,
                                    rocfft_array_type_real,
                                    {0, 0, 0, 0},
                                    {4, 8, 6, 3},
                                    {8, 1, 32, 192},
                                    1,
                                    {4, 0});
        EXPECT_THROW(L_t.get_layout_for_len_axes({1}).reinterpret_as(t), std::invalid_argument);
        EXPECT_THROW(L_t.get_layout_for_len_axes({1, 2}).reinterpret_as(t), std::invalid_argument);
        const data_descriptor_t L_t_planar(
            rocfft_precision_double, t, {0, 0, 0, 0}, {4, 8, 6, 3}, {16, 2, 64, 384}, 1, {3, 4});
        EXPECT_THROW(L_t_planar.get_layout_for_len_axes({1}).reinterpret_as(rocfft_array_type_real),
                     std::invalid_argument);
    }
}

namespace
{
    const rocfft_array_type all_array_types[] = {rocfft_array_type_unset,
                                                 rocfft_array_type_real,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_array_type_hermitian_interleaved,
                                                 rocfft_array_type_complex_planar,
                                                 rocfft_array_type_hermitian_planar};

    bool is_planar(rocfft_array_type t)
    {
        return t == rocfft_array_type_complex_planar || t == rocfft_array_type_hermitian_planar;
    }

    // valid offsets for array type t: second offset only set if planar
    std::array<size_t, 2> offsets_for(rocfft_array_type t, size_t first, size_t second)
    {
        return {first, is_planar(t) ? second : 0};
    }
}

TEST_F(data_descriptor_test, offsets_default_to_zero_with_one_per_plane)
{
    EXPECT_EQ(data_descriptor_t::default_full_layout({8}, 2).num_planes(), 1u);
    EXPECT_EQ(data_descriptor_t::default_full_layout({8}, 2).get_offset(), 0u);
    const auto base = data_descriptor_t::default_full_layout({8, 4}, 2);
    for(auto type : all_array_types)
    {
        const auto layout = with_type(base, rocfft_precision_single, type);
        SCOPED_TRACE(layout.str());
        EXPECT_EQ(layout.num_planes(), is_planar(type) ? 2u : 1u);
        EXPECT_EQ(layout.get_offset(), 0u);
        EXPECT_EQ(layout.get_offset(0), 0u);
        if(is_planar(type))
        {
            EXPECT_EQ(layout.get_offset(1), 0u);
        }
        else
        {
            EXPECT_THROW(layout.get_offset(1), std::out_of_range);
        }
        EXPECT_THROW(layout.get_offset(2), std::out_of_range);
    }
}

TEST_F(data_descriptor_test, offsets_constructor_enforces_invariant)
{
    for(auto type : all_array_types)
    {
        const data_descriptor_t layout(
            rocfft_precision_double, type, {0, 0}, {4, 3}, {1, 4}, 1, offsets_for(type, 3, 5));
        SCOPED_TRACE(layout.str());
        EXPECT_EQ(layout.get_offset(0), 3u);
        if(is_planar(type))
        {
            EXPECT_EQ(layout.get_offset(1), 5u);
        }
        else
        {
            EXPECT_THROW(layout.get_offset(1), std::out_of_range);
            EXPECT_THROW(
                data_descriptor_t(rocfft_precision_double, type, {0, 0}, {4, 3}, {1, 4}, 1, {3, 5}),
                std::invalid_argument);
        }
    }
}

TEST_F(data_descriptor_test, offsets_setter_enforces_invariant)
{
    auto layout = data_descriptor_t::default_full_layout({8, 4}, 2);
    data_descriptor_test_access::set_array_type_and_offsets(
        layout, rocfft_array_type_complex_planar, {1, 2});
    EXPECT_EQ(layout.get_offset(1), 2u);

    // a violation leaves the object untouched
    for(auto type : all_array_types)
    {
        if(is_planar(type))
            continue;
        EXPECT_THROW(data_descriptor_test_access::set_array_type_and_offsets(layout, type, {1, 2}),
                     std::invalid_argument);
        EXPECT_THROW(data_descriptor_test_access::set_array_type(layout, type),
                     std::invalid_argument);
        EXPECT_EQ(layout.get_array_type(), rocfft_array_type_complex_planar);
        EXPECT_EQ(layout.get_offset(0), 1u);
        EXPECT_EQ(layout.get_offset(1), 2u);
    }

    data_descriptor_test_access::set_array_type_and_offsets(
        layout, rocfft_array_type_complex_interleaved, {1, 0});
    EXPECT_EQ(layout.get_array_type(), rocfft_array_type_complex_interleaved);
    EXPECT_EQ(layout.get_offset(), 1u);
    EXPECT_THROW(layout.get_offset(1), std::out_of_range);
}

TEST_F(data_descriptor_test, offsets_survive_reinterpret_as)
{
    const auto base = data_descriptor_t::default_full_layout({8, 4}, 2);
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
        const auto src = with_type(base, rocfft_precision_single, from, offsets_for(from, 7, 9));
        const auto dst = src.reinterpret_as(to);
        SCOPED_TRACE(dst.str());
        EXPECT_EQ(dst.num_planes(), src.num_planes());
        for(size_t plane = 0; plane < src.num_planes(); plane++)
            EXPECT_EQ(dst.get_offset(plane), src.get_offset(plane));
        EXPECT_TRUE(dst.reinterpret_as(from) == src);
    }
}

TEST_F(data_descriptor_test, offsets_inplace_mapping)
{
    // complex transforms: identical offsets in all planes
    for(auto type : {rocfft_array_type_complex_interleaved, rocfft_array_type_complex_planar})
    {
        const auto in  = with_type(data_descriptor_t::default_full_layout({16}, 4),
                                  rocfft_precision_double,
                                  type,
                                  offsets_for(type, 6, 11));
        const auto out = in.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                         rocfft_transform_type_complex_forward);
        ASSERT_TRUE(out.has_value());
        EXPECT_TRUE(*out == in) << out->str();
    }

    // real transforms: real offset == 2 * hermitian offset
    const auto padded = data_descriptor_t::default_full_layout({16}, 4, true);
    const auto real   = with_type(padded, rocfft_precision_single, rocfft_array_type_real, {6, 0});
    const auto odd_real
        = with_type(padded, rocfft_precision_single, rocfft_array_type_real, {7, 0});
    for(auto [fft_type, herm_io] :
        {std::pair{rocfft_transform_type_real_forward, io_data_label::OUTPUT},
         std::pair{rocfft_transform_type_real_inverse, io_data_label::INPUT}})
    {
        const auto herm = real.get_other_inplace_layout_for(herm_io, fft_type);
        ASSERT_TRUE(herm.has_value());
        EXPECT_EQ(herm->get_offset(), 3u);
        const auto back = herm->get_other_inplace_layout_for(other(herm_io), fft_type);
        ASSERT_TRUE(back.has_value());
        EXPECT_TRUE(*back == real) << back->str();

        // an odd real offset cannot be halved
        EXPECT_FALSE(odd_real.get_other_inplace_layout_for(herm_io, fft_type).has_value());
    }

    // a second (planar) offset cannot be expressed in a single-plane layout
    const auto herm_planar = with_type(data_descriptor_t::default_full_layout({9}, 4),
                                       rocfft_precision_single,
                                       rocfft_array_type_hermitian_planar,
                                       {4, 2});
    EXPECT_FALSE(
        herm_planar
            .get_other_inplace_layout_for(io_data_label::OUTPUT, rocfft_transform_type_real_inverse)
            .has_value());

    // any hermitian offset doubles, odd ones included
    const auto herm      = with_type(data_descriptor_t::default_full_layout({9}, 4),
                                rocfft_precision_single,
                                rocfft_array_type_hermitian_interleaved,
                                {5, 0});
    const auto from_herm = herm.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                             rocfft_transform_type_real_inverse);
    ASSERT_TRUE(from_herm.has_value());
    EXPECT_EQ(from_herm->get_array_type(), rocfft_array_type_real);
    EXPECT_EQ(from_herm->get_offset(), 10u);

    // unset array type: offsets derived alike
    auto unset = padded;
    data_descriptor_test_access::set_array_type_and_offsets(unset, rocfft_array_type_unset, {4, 0});
    const auto unset_herm = unset.get_other_inplace_layout_for(io_data_label::OUTPUT,
                                                               rocfft_transform_type_real_forward);
    ASSERT_TRUE(unset_herm.has_value());
    EXPECT_EQ(unset_herm->get_array_type(), rocfft_array_type_unset);
    EXPECT_EQ(unset_herm->get_offset(), 2u);
}

TEST_F(data_descriptor_test, offsets_equality_is_strict)
{
    const auto base = data_descriptor_t::default_full_layout({8, 4}, 2);
    const auto make = [&base](const std::array<size_t, 2>& offsets) {
        return with_type(base, rocfft_precision_single, rocfft_array_type_complex_planar, offsets);
    };
    EXPECT_TRUE(make({1, 2}) == make({1, 2}));
    EXPECT_FALSE(make({1, 2}) == make({1, 3}));
    EXPECT_FALSE(make({1, 2}) == make({0, 2}));
    EXPECT_FALSE(make({1, 2}) == make({0, 0}));
}

TEST_F(data_descriptor_test, offsets_pass_through_subsets_and_embeddings)
{
    const auto layout = with_type(data_descriptor_t::default_full_layout({4, 8, 2}, 3),
                                  rocfft_precision_double,
                                  rocfft_array_type_complex_planar,
                                  {3, 5});
    for(const auto& len_indices : {std::set<size_t>{0, 1, 2}, std::set<size_t>{0, 2}})
    {
        const auto sub = layout.get_layout_for_len_axes(len_indices);
        EXPECT_EQ(sub.get_offset(0), 3u);
        EXPECT_EQ(sub.get_offset(1), 5u);
        const auto embedding = sub.get_embedding_layout();
        EXPECT_EQ(embedding.get_offset(0), 3u);
        EXPECT_EQ(embedding.get_offset(1), 5u);
    }
}

TEST_F(data_descriptor_test, offsets_zero_in_intersection)
{
    const auto base  = data_descriptor_t::default_full_layout({8, 4}, 2);
    const auto xsect = data_descriptor_t::make_contiguous_intersection_of(
        with_type(base, rocfft_precision_single, rocfft_array_type_complex_planar, {3, 5}),
        with_type(base, rocfft_precision_single, rocfft_array_type_hermitian_planar, {1, 2}));
    EXPECT_EQ(xsect.get_offset(0), 0u);
    EXPECT_EQ(xsect.get_offset(1), 0u);
}

TEST_F(data_descriptor_test, offsets_survive_clear_and_reset)
{
    auto layout = with_type(data_descriptor_t::default_full_layout({8, 4}, 2),
                            rocfft_precision_half,
                            rocfft_array_type_hermitian_planar,
                            {3, 5});
    data_descriptor_test_access::clear(layout);
    EXPECT_EQ(layout.get_offset(0), 3u);
    EXPECT_EQ(layout.get_offset(1), 5u);
    data_descriptor_test_access::full_range_reset(layout, {16}, 3);
    EXPECT_EQ(layout.get_offset(0), 3u);
    EXPECT_EQ(layout.get_offset(1), 5u);
}

TEST_F(data_descriptor_test, offsets_in_str)
{
    const auto base = data_descriptor_t::default_full_layout({8}, 2);
    const auto planar
        = with_type(base, rocfft_precision_single, rocfft_array_type_complex_planar, {3, 5});
    EXPECT_NE(planar.str().find("offsets 3 5"), std::string::npos) << planar.str();
    const auto interleaved
        = with_type(base, rocfft_precision_single, rocfft_array_type_complex_interleaved, {3, 0});
    EXPECT_NE(interleaved.str().find("offsets 3"), std::string::npos) << interleaved.str();
    EXPECT_EQ(interleaved.str().find("offsets 3 0"), std::string::npos) << interleaved.str();
}

// offset_in, is_continuous_in, is_contiguous and buffer_element_count are
// relative to the planes' offsets, which they ignore
TEST_F(data_descriptor_test, offsets_ignored_by_relative_queries)
{
    const auto whole = with_type(data_descriptor_t::default_full_layout({8, 4}, 2),
                                 rocfft_precision_single,
                                 rocfft_array_type_complex_planar);
    const auto chunk = data_descriptor_t(rocfft_precision_single,
                                         rocfft_array_type_complex_planar,
                                         {0, 1, 0},
                                         {8, 3, 2},
                                         whole.strides_and_distances());
    const auto shifted_whole
        = with_type(whole, rocfft_precision_single, rocfft_array_type_complex_planar, {3, 5});
    const auto shifted_chunk = data_descriptor_t(rocfft_precision_single,
                                                 rocfft_array_type_complex_planar,
                                                 chunk.lower(),
                                                 chunk.upper(),
                                                 chunk.strides_and_distances(),
                                                 1,
                                                 {7, 11});
    EXPECT_EQ(shifted_chunk.offset_in(shifted_whole), chunk.offset_in(whole));
    EXPECT_EQ(shifted_chunk.offset_in(whole), chunk.offset_in(whole));
    EXPECT_EQ(shifted_chunk.is_continuous_in(shifted_whole), chunk.is_continuous_in(whole));
    EXPECT_EQ(shifted_whole.is_contiguous(), whole.is_contiguous());
    EXPECT_EQ(shifted_whole.buffer_element_count(), whole.buffer_element_count());
    EXPECT_EQ(shifted_chunk.buffer_element_count(), chunk.buffer_element_count());
}

TEST_F(data_descriptor_test, max_offset_and_stride_contiguous)
{
    // 1D, 2D, 3D: last element of a packed layout
    const auto l1 = data_descriptor_t::default_full_layout({16}, 1);
    const auto l2 = data_descriptor_t::default_full_layout({4, 8}, 3);
    const auto l3 = data_descriptor_t::default_full_layout({2, 3, 4}, 5);
    EXPECT_EQ(l1.max_offset(), 15u);
    EXPECT_EQ(l1.max_stride(), 16u); // distance of the unit batch axis
    EXPECT_EQ(l2.max_offset(), 95u);
    EXPECT_EQ(l2.max_stride(), 32u);
    EXPECT_EQ(l3.max_offset(), 119u);
    EXPECT_EQ(l3.max_stride(), 24u);

    for(size_t i = 0; i < num_random_cases; i++)
    {
        const auto packed = data_descriptor_t::default_full_layout(random_vec(uniform(1, 4), 1, 16),
                                                                   random_vec(uniform(1, 3), 1, 5));
        SCOPED_TRACE(packed.str());
        EXPECT_EQ(packed.max_offset(), packed.logical_count() - 1);
    }
}

TEST_F(data_descriptor_test, max_offset_and_stride_padded_and_unordered)
{
    // real in-place padding: rows of 4 reals occupy 6 slots
    const auto padded = data_descriptor_t::default_full_layout({4, 8}, 3, true);
    EXPECT_EQ(padded.max_offset(), 3u + 7u * 6u + 2u * 48u);
    EXPECT_EQ(padded.max_stride(), 48u);

    // largest stride on the first length axis, batch axis interleaved in between
    const auto unordered = data_descriptor_t::full_layout({4, 8}, {24, 1}, 3, 8);
    EXPECT_EQ(unordered.max_stride(), 24u);
    EXPECT_EQ(unordered.max_offset(), 3u * 24u + 7u + 2u * 8u);
    EXPECT_EQ(unordered.max_offset(), unordered.logical_count() - 1);

    // two batch axes, the length axis being the slowest
    const data_descriptor_t multi_batch({0, 0, 0}, {4, 8, 3}, {24, 1, 8}, 2);
    EXPECT_EQ(multi_batch.max_stride(), 24u);
    EXPECT_EQ(multi_batch.max_offset(), 95u);
}

// lower bounds are logical coordinates: the first element sits at the offset
TEST_F(data_descriptor_test, max_offset_ignores_lower_bounds)
{
    const data_descriptor_t sub({1, 2, 1}, {4, 8, 3}, {1, 4, 32});
    const data_descriptor_t zero_based({0, 0, 0}, {3, 6, 2}, {1, 4, 32});
    EXPECT_EQ(sub.max_offset(), 2u * 1u + 5u * 4u + 1u * 32u);
    EXPECT_EQ(sub.max_offset(), zero_based.max_offset());
    EXPECT_EQ(sub.max_stride(), 32u);

    // multiple batch axes with non-zero lower bounds
    const data_descriptor_t multi_batch({2, 1, 3}, {4, 8, 5}, {1, 4, 32}, 2);
    EXPECT_EQ(multi_batch.max_offset(), 1u + 6u * 4u + 1u * 32u);
}

// a sub-dimensional embedded layout keeps all axes (as batch axes), so it
// reaches the same elements and carries the same strides
TEST_F(data_descriptor_test, max_offset_and_stride_embedded)
{
    const data_descriptor_t layout(rocfft_precision_double,
                                   rocfft_array_type_complex_interleaved,
                                   {1, 0, 2, 0},
                                   {4, 8, 6, 3},
                                   {200, 1, 16, 1000},
                                   1,
                                   {7, 0});
    const size_t            expected = 7u + 2u * 200u + 7u * 1u + 3u * 16u + 2u * 1000u;
    EXPECT_EQ(layout.max_offset(), expected);
    EXPECT_EQ(layout.max_stride(), 1000u);
    for(const auto& len_indices :
        {std::set<size_t>{0}, std::set<size_t>{1, 2}, std::set<size_t>{0, 1, 2}})
    {
        const auto sub = layout.get_layout_for_len_axes(len_indices);
        SCOPED_TRACE(sub.str());
        EXPECT_EQ(sub.max_offset(), expected);
        EXPECT_EQ(sub.max_stride(), 1000u);
        const auto embedding = sub.get_embedding_layout();
        EXPECT_EQ(embedding.max_offset(), expected);
        EXPECT_EQ(embedding.max_stride(), 1000u);
    }
}

TEST_F(data_descriptor_test, max_offset_over_planes)
{
    const auto base  = data_descriptor_t::default_full_layout({8, 4}, 2);
    const auto count = base.buffer_element_count();
    for(auto type : all_array_types)
    {
        const auto layout = with_type(base, rocfft_precision_single, type, offsets_for(type, 3, 5));
        SCOPED_TRACE(layout.str());
        EXPECT_EQ(layout.max_offset(), (is_planar(type) ? 5u : 3u) + count - 1);
        EXPECT_EQ(layout.max_stride(), base.max_stride());
    }
    // the larger offset wins whichever plane holds it
    const auto first_larger
        = with_type(base, rocfft_precision_single, rocfft_array_type_complex_planar, {9, 2});
    EXPECT_EQ(first_larger.max_offset(), 9u + count - 1);
    for(size_t plane = 0; plane < first_larger.num_planes(); plane++)
        EXPECT_LE(first_larger.get_offset(plane) + count - 1, first_larger.max_offset());
}

TEST_F(data_descriptor_test, max_offset_rejects_empty_range)
{
    const data_descriptor_t empty_len({0, 0}, {0, 4}, {1, 4});
    EXPECT_TRUE(empty_len.is_empty());
    EXPECT_THROW(empty_len.max_offset(), std::logic_error);
    EXPECT_EQ(empty_len.max_stride(), 4u);

    const data_descriptor_t empty_batch({0, 0, 0}, {4, 8, 0}, {1, 4, 32}, 2);
    EXPECT_THROW(empty_batch.max_offset(), std::logic_error);
    EXPECT_EQ(empty_batch.max_stride(), 32u);
}

// values beyond 32 bits are exact, so they can feed 32-bit index limit checks
TEST_F(data_descriptor_test, max_offset_and_stride_beyond_32_bits)
{
    const size_t u32_max = std::numeric_limits<uint32_t>::max();
    const size_t big     = u32_max + 1;

    const auto two_batches = data_descriptor_t::full_layout({2}, {1}, 2, big);
    EXPECT_EQ(two_batches.max_stride(), big);
    EXPECT_EQ(two_batches.max_offset(), big + 1);
    EXPECT_GT(two_batches.max_offset(), u32_max);

    // an unused distance (unit batch) exceeds the limit without moving max_offset
    const auto one_batch = data_descriptor_t::full_layout({2}, {1}, 1, big);
    EXPECT_GT(one_batch.max_stride(), u32_max);
    EXPECT_EQ(one_batch.max_offset(), 1u);

    // an offset alone may push the last element beyond the limit
    const auto shifted = with_type(data_descriptor_t::default_full_layout({8}, 1),
                                   rocfft_precision_single,
                                   rocfft_array_type_complex_planar,
                                   {0, u32_max});
    EXPECT_EQ(shifted.max_offset(), u32_max + 7);
    EXPECT_LE(shifted.max_stride(), u32_max);

    // a layout reaching just below the limit stays below it
    const auto at_limit = data_descriptor_t::full_layout({2}, {u32_max - 1}, 1, 1);
    EXPECT_EQ(at_limit.max_offset(), u32_max - 1);
    EXPECT_EQ(at_limit.max_stride(), u32_max - 1);
}

TEST_F(data_descriptor_test, split_length_axis_nominal)
{
    // 1D: 12 -> 3 x 4
    const auto d1 = data_descriptor_t::full_layout({12}, {1}, 3, 12);
    const auto s1 = d1.split_length_axis(0, 3);
    EXPECT_EQ(s1.lengths(), (std::vector<size_t>{3, 4}));
    EXPECT_EQ(s1.strides(), (std::vector<size_t>{1, 3}));
    EXPECT_EQ(s1.batches(), d1.batches());
    EXPECT_EQ(s1.distances(), d1.distances());
    EXPECT_EQ(s1.logical_count(), d1.logical_count());
    EXPECT_EQ(all_offsets(s1), all_offsets(d1));

    // 2D, splitting the outer (non-unit stride) axis: 8 x 6 -> 8 x 2 x 3
    const auto d2 = data_descriptor_t::full_layout({8, 6}, {1, 8}, 2, 64);
    const auto s2 = d2.split_length_axis(1, 2);
    EXPECT_EQ(s2.lengths(), (std::vector<size_t>{8, 2, 3}));
    EXPECT_EQ(s2.strides(), (std::vector<size_t>{1, 8, 16}));
    EXPECT_EQ(s2.distances(), d2.distances());
    EXPECT_EQ(s2.logical_count(), d2.logical_count());
    EXPECT_EQ(all_offsets(s2), all_offsets(d2));

    // 3D, splitting the middle axis: later length axes shift by one
    const auto d3 = data_descriptor_t::default_full_layout({4, 6, 10}, 2);
    const auto s3 = d3.split_length_axis(1, 3);
    EXPECT_EQ(s3.lengths(), (std::vector<size_t>{4, 3, 2, 10}));
    EXPECT_EQ(s3.strides(), (std::vector<size_t>{1, 4, 12, 24}));
    EXPECT_EQ(s3.batches(), d3.batches());
    EXPECT_EQ(s3.distances(), d3.distances());
    EXPECT_EQ(s3.logical_count(), d3.logical_count());
    EXPECT_EQ(all_offsets(s3), all_offsets(d3));
    EXPECT_FALSE(s3.is_embedded());
}

TEST_F(data_descriptor_test, split_length_axis_default_factor)
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
        const auto d = data_descriptor_t::default_full_layout({L}, 2);
        const auto s = d.split_length_axis(0);
        EXPECT_EQ(s.lengths(), lengths);
        EXPECT_EQ(s.strides(), (std::vector<size_t>{1, lengths[0]}));
        EXPECT_EQ(all_offsets(s), all_offsets(d));
    }
}

TEST_F(data_descriptor_test, split_length_axis_rejects_bad_input)
{
    const auto d = data_descriptor_t::default_full_layout({12, 7, 1}, 2);
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
    const data_descriptor_t empty({3, 0}, {3, 2}, {1, 8});
    EXPECT_THROW(empty.split_length_axis(0), std::invalid_argument);
    // partial range [2, 14) is not aligned on multiples of 4 (nor of default 3)
    const data_descriptor_t misaligned({2, 0}, {14, 3}, {1, 16});
    EXPECT_THROW(misaligned.split_length_axis(0, 4), std::invalid_argument);
    EXPECT_THROW(misaligned.split_length_axis(0), std::invalid_argument);
    EXPECT_NO_THROW(misaligned.split_length_axis(0, 2));
}

TEST_F(data_descriptor_test, split_length_axis_partial_range)
{
    // [4, 16) along a stride-1 axis, batch [1, 3): inner [0, 4), outer [1, 4)
    const data_descriptor_t d({4, 1}, {16, 3}, {1, 32});
    const auto              s = d.split_length_axis(0, 4);
    EXPECT_EQ(s.lower(), (std::vector<size_t>{0, 1, 1}));
    EXPECT_EQ(s.upper(), (std::vector<size_t>{4, 4, 3}));
    EXPECT_EQ(s.strides_and_distances(), (std::vector<size_t>{1, 4, 32}));
    EXPECT_EQ(s.logical_count(), d.logical_count());
    EXPECT_EQ(all_offsets(s), all_offsets(d));

    // offset of the first element within the full layout is unchanged too
    const auto full = data_descriptor_t::full_layout({16}, {1}, 3, 32);
    EXPECT_EQ(s.offset_in(full.split_length_axis(0, 4)), d.offset_in(full));
}

TEST_F(data_descriptor_test, split_length_axis_multi_batch)
{
    const auto d = data_descriptor_t::default_full_layout({6, 10}, {3, 2});
    const auto s = d.split_length_axis(0);
    EXPECT_EQ(s.lengths(), (std::vector<size_t>{2, 3, 10}));
    EXPECT_EQ(s.strides(), (std::vector<size_t>{1, 2, 6}));
    EXPECT_EQ(s.get_batch_rank(), 2u);
    EXPECT_EQ(s.batches(), d.batches());
    EXPECT_EQ(s.distances(), d.distances());
    EXPECT_EQ(all_offsets(s), all_offsets(d));
}

TEST_F(data_descriptor_test, split_length_axis_keeps_type_precision_and_offsets)
{
    const auto base = data_descriptor_t::default_full_layout({6, 10}, 2);

    // planar data with two plane offsets, split along length axis 0
    const auto planar
        = with_type(base, rocfft_precision_single, rocfft_array_type_complex_planar, {5, 1000});
    const auto sp = planar.split_length_axis(0, 2);
    EXPECT_EQ(sp.get_array_type(), rocfft_array_type_complex_planar);
    EXPECT_EQ(sp.get_precision(), rocfft_precision_single);
    EXPECT_EQ(sp.num_planes(), 2u);
    EXPECT_EQ(sp.get_offset(0), 5u);
    EXPECT_EQ(sp.get_offset(1), 1000u);
    EXPECT_EQ(sp.max_offset(), planar.max_offset());
    EXPECT_EQ(all_offsets(sp), all_offsets(planar));

    // interleaved data with an offset, split along length axis 1
    const auto interleaved
        = with_type(base, rocfft_precision_double, rocfft_array_type_complex_interleaved, {7, 0});
    const auto si = interleaved.split_length_axis(1);
    EXPECT_EQ(si.get_array_type(), rocfft_array_type_complex_interleaved);
    EXPECT_EQ(si.get_precision(), rocfft_precision_double);
    EXPECT_EQ(si.get_offset(0), 7u);

    // no precision nor array type: still none after the split
    const auto su = base.split_length_axis(0);
    EXPECT_EQ(su.get_array_type(), rocfft_array_type_unset);
    EXPECT_FALSE(su.has_precision());
}

TEST_F(data_descriptor_test, split_length_axis_embedded)
{
    const auto L   = data_descriptor_t::default_full_layout({4, 8, 6}, 3);
    const auto sub = L.get_layout_for_len_axes({0, 2});
    ASSERT_TRUE(sub.is_embedded());

    // split the sub-layout's axis 1 (L's axis 2): L's axis 2 is split alike
    const auto s1 = sub.split_length_axis(1, 2);
    EXPECT_EQ(s1.lengths(), (std::vector<size_t>{4, 2, 3}));
    EXPECT_EQ(s1.batches(), sub.batches());
    EXPECT_TRUE(s1.is_embedded());
    EXPECT_TRUE(s1.get_embedding_layout() == L.split_length_axis(2, 2));
    EXPECT_EQ(s1.corresponding_axes_in_embedding(), (std::vector<size_t>{0, 2, 3}));
    EXPECT_EQ(all_offsets(s1), all_offsets(sub));

    // split the sub-layout's axis 0 (L's axis 0): every later axis shifts by one
    const auto s0 = sub.split_length_axis(0);
    EXPECT_EQ(s0.lengths(), (std::vector<size_t>{2, 2, 6}));
    EXPECT_TRUE(s0.get_embedding_layout() == L.split_length_axis(0));
    EXPECT_EQ(s0.corresponding_axes_in_embedding(), (std::vector<size_t>{0, 1, 3}));
    EXPECT_EQ(all_offsets(s0), all_offsets(sub));
}

TEST_F(data_descriptor_test, split_length_axis_round_trip)
{
    for(size_t i = 0; i < num_random_cases; i++)
    {
        // random layout whose length axis dim spans an aligned partial range of
        // factor * m elements
        const auto full   = random_full_layout(2);
        const auto dim    = uniform(0, full.get_len_rank() - 1);
        const auto factor = uniform(2, 4);
        auto       lo     = full.lower();
        auto       hi     = full.upper();
        lo[dim]           = factor * uniform(0, 2);
        hi[dim]           = lo[dim] + factor * uniform(2, 3);
        const data_descriptor_t d(lo, hi, full.strides_and_distances(), full.get_batch_rank());
        SCOPED_TRACE("d " + d.str() + ", dim " + std::to_string(dim) + ", factor "
                     + std::to_string(factor));

        const auto s = d.split_length_axis(dim, factor);
        EXPECT_EQ(s.get_len_rank(), d.get_len_rank() + 1);
        EXPECT_EQ(s.logical_count(), d.logical_count());
        EXPECT_EQ(all_offsets(s), all_offsets(d));
    }
}
