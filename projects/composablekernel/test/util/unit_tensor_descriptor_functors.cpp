// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_data_to_gemm_v1.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_offset_utils.hpp"

using namespace ck;

constexpr index_t kIndexMax = std::numeric_limits<index_t>::max();
constexpr index_t kIndexMin = std::numeric_limits<index_t>::lowest();

// =============================================================================
// Tests for convert_visible_to_hidden_id functor
// =============================================================================

TEST(ConvertVisibleToHiddenId, SimplePackedDescriptor3D)
{
    // For a 3D packed descriptor created via make_naive_tensor_descriptor_packed,
    // the visible dimension IDs are Sequence<1, 2, 3> (hidden dim 0 is the element space)
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    // Verify the visible dimension IDs for a packed descriptor
    constexpr auto visible_ids = DescType::GetVisibleDimensionIds();
    using ExpectedVisibleIds   = Sequence<1, 2, 3>;
    EXPECT_TRUE((is_same<decltype(visible_ids), const ExpectedVisibleIds>::value));

    // Test the functor
    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // Visible ID 0 -> Hidden ID 1
    constexpr auto hidden_0 = functor(Number<0>{});
    EXPECT_EQ(hidden_0, 1);

    // Visible ID 1 -> Hidden ID 2
    constexpr auto hidden_1 = functor(Number<1>{});
    EXPECT_EQ(hidden_1, 2);

    // Visible ID 2 -> Hidden ID 3
    constexpr auto hidden_2 = functor(Number<2>{});
    EXPECT_EQ(hidden_2, 3);
}

TEST(ConvertVisibleToHiddenId, SimplePackedDescriptor2D)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<32>{}, Number<64>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // For 2D packed: visible IDs are Sequence<1, 2>
    constexpr auto hidden_0 = functor(Number<0>{});
    constexpr auto hidden_1 = functor(Number<1>{});

    EXPECT_EQ(hidden_0, 1);
    EXPECT_EQ(hidden_1, 2);
}

TEST(ConvertVisibleToHiddenId, SimplePackedDescriptor1D)
{
    constexpr auto desc = make_naive_tensor_descriptor_packed(make_tuple(Number<128>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // For 1D packed: visible IDs are Sequence<1>
    constexpr auto hidden_0 = functor(Number<0>{});
    EXPECT_EQ(hidden_0, 1);
}

TEST(ConvertVisibleToHiddenId, SimplePackedDescriptor4D)
{
    constexpr auto desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<2>{}, Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // For 4D packed: visible IDs are Sequence<1, 2, 3, 4>
    EXPECT_EQ(functor(Number<0>{}), 1);
    EXPECT_EQ(functor(Number<1>{}), 2);
    EXPECT_EQ(functor(Number<2>{}), 3);
    EXPECT_EQ(functor(Number<3>{}), 4);
}

// =============================================================================
// Tests for convert_visible_ids_to_hidden_ids functor
// =============================================================================

TEST(ConvertVisibleIdsToHiddenIds, SingleElement)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert single visible ID
    constexpr auto result = functor(Sequence<0>{});
    using ExpectedResult  = Sequence<1>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, MultipleElements)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert multiple visible IDs
    constexpr auto result = functor(Sequence<0, 2>{});
    using ExpectedResult  = Sequence<1, 3>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, AllElements)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert all visible IDs
    constexpr auto result = functor(Sequence<0, 1, 2>{});
    using ExpectedResult  = Sequence<1, 2, 3>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, ReversedOrder)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert visible IDs in reverse order
    constexpr auto result = functor(Sequence<2, 1, 0>{});
    using ExpectedResult  = Sequence<3, 2, 1>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, EmptySequence)
{
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert empty sequence
    constexpr auto result = functor(Sequence<>{});
    using ExpectedResult  = Sequence<>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(ConvertVisibleIdsToHiddenIds, HighDimensional)
{
    constexpr auto desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<2>{}, Number<4>{}, Number<8>{}, Number<16>{}, Number<32>{}));

    using DescType = decltype(desc);

    constexpr auto functor = convert_visible_ids_to_hidden_ids<DescType>{};

    // Convert subset of visible IDs
    constexpr auto result = functor(Sequence<1, 3>{});
    using ExpectedResult  = Sequence<2, 4>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

// =============================================================================
// Tests for generate_arithmetic_sequence_from_scan functor
// =============================================================================

TEST(GenerateArithmeticSequenceFromScan, SingleRange)
{
    // Scan sequence: <0, 3> means:
    //   - Index 0: range from (base + 0) to (base + 3) = Sequence<base, base+1, base+2>
    using ScanSeq          = Sequence<0, 3>;
    constexpr index_t base = 5;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    constexpr auto result = functor(Number<0>{});
    using ExpectedResult  = Sequence<5, 6, 7>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

TEST(GenerateArithmeticSequenceFromScan, MultipleRanges)
{
    // Scan sequence: <0, 2, 5> means:
    //   - Index 0: range from (base + 0) to (base + 2) = 2 elements
    //   - Index 1: range from (base + 2) to (base + 5) = 3 elements
    using ScanSeq          = Sequence<0, 2, 5>;
    constexpr index_t base = 3;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    // First range: base + [0, 2) = Sequence<3, 4>
    constexpr auto result_0 = functor(Number<0>{});
    using ExpectedResult0   = Sequence<3, 4>;
    EXPECT_TRUE((is_same<decltype(result_0), const ExpectedResult0>::value));

    // Second range: base + [2, 5) = Sequence<5, 6, 7>
    constexpr auto result_1 = functor(Number<1>{});
    using ExpectedResult1   = Sequence<5, 6, 7>;
    EXPECT_TRUE((is_same<decltype(result_1), const ExpectedResult1>::value));
}

TEST(GenerateArithmeticSequenceFromScan, SingleElementRanges)
{
    // Scan sequence with single-element ranges: <0, 1, 2, 3>
    using ScanSeq          = Sequence<0, 1, 2, 3>;
    constexpr index_t base = 10;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    // Each range contains exactly one element
    constexpr auto result_0 = functor(Number<0>{});
    using ExpectedResult0   = Sequence<10>;
    EXPECT_TRUE((is_same<decltype(result_0), const ExpectedResult0>::value));

    constexpr auto result_1 = functor(Number<1>{});
    using ExpectedResult1   = Sequence<11>;
    EXPECT_TRUE((is_same<decltype(result_1), const ExpectedResult1>::value));

    constexpr auto result_2 = functor(Number<2>{});
    using ExpectedResult2   = Sequence<12>;
    EXPECT_TRUE((is_same<decltype(result_2), const ExpectedResult2>::value));
}

TEST(GenerateArithmeticSequenceFromScan, ZeroBase)
{
    // Test with base = 0
    using ScanSeq          = Sequence<0, 2, 4>;
    constexpr index_t base = 0;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    constexpr auto result_0 = functor(Number<0>{});
    using ExpectedResult0   = Sequence<0, 1>;
    EXPECT_TRUE((is_same<decltype(result_0), const ExpectedResult0>::value));

    constexpr auto result_1 = functor(Number<1>{});
    using ExpectedResult1   = Sequence<2, 3>;
    EXPECT_TRUE((is_same<decltype(result_1), const ExpectedResult1>::value));
}

TEST(GenerateArithmeticSequenceFromScan, VariableSizeRanges)
{
    // Scan sequence with variable-size ranges: <0, 1, 4, 6, 10>
    using ScanSeq          = Sequence<0, 1, 4, 6, 10>;
    constexpr index_t base = 2;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    // Range 0: [2+0, 2+1) = Sequence<2>
    constexpr auto result_0 = functor(Number<0>{});
    using ExpectedResult0   = Sequence<2>;
    EXPECT_TRUE((is_same<decltype(result_0), const ExpectedResult0>::value));

    // Range 1: [2+1, 2+4) = Sequence<3, 4, 5>
    constexpr auto result_1 = functor(Number<1>{});
    using ExpectedResult1   = Sequence<3, 4, 5>;
    EXPECT_TRUE((is_same<decltype(result_1), const ExpectedResult1>::value));

    // Range 2: [2+4, 2+6) = Sequence<6, 7>
    constexpr auto result_2 = functor(Number<2>{});
    using ExpectedResult2   = Sequence<6, 7>;
    EXPECT_TRUE((is_same<decltype(result_2), const ExpectedResult2>::value));

    // Range 3: [2+6, 2+10) = Sequence<8, 9, 10, 11>
    constexpr auto result_3 = functor(Number<3>{});
    using ExpectedResult3   = Sequence<8, 9, 10, 11>;
    EXPECT_TRUE((is_same<decltype(result_3), const ExpectedResult3>::value));
}

TEST(GenerateArithmeticSequenceFromScan, LargeBase)
{
    // Test with a larger base value
    using ScanSeq          = Sequence<0, 3>;
    constexpr index_t base = 100;

    constexpr auto functor = generate_arithmetic_sequence_from_scan<base, ScanSeq>{};

    constexpr auto result = functor(Number<0>{});
    using ExpectedResult  = Sequence<100, 101, 102>;
    EXPECT_TRUE((is_same<decltype(result), const ExpectedResult>::value));
}

// =============================================================================
// Integration tests - verify functors work together in transform_tensor_descriptor context
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, TransformPreservesMapping)
{
    // Create a simple packed 2D descriptor
    constexpr auto desc = make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}));

    using DescType = decltype(desc);

    // Verify the descriptor structure
    EXPECT_EQ(DescType::GetNumOfVisibleDimension(), 2);
    EXPECT_EQ(DescType::GetNumOfHiddenDimension(), 3); // 1 (element space) + 2 (visible dims)

    // Test that convert_visible_to_hidden_id preserves the expected mapping
    constexpr auto functor = convert_visible_to_hidden_id<DescType>{};

    // The hidden IDs should be consecutive starting from 1
    constexpr auto hidden_0 = functor(Number<0>{});
    constexpr auto hidden_1 = functor(Number<1>{});

    EXPECT_EQ(hidden_0, 1);
    EXPECT_EQ(hidden_1, 2);
    EXPECT_EQ(hidden_1 - hidden_0, 1); // Should be consecutive
}

TEST(TensorDescriptorFunctorsIntegration, ConvertIdsMatchDirectAccess)
{
    // Verify that the functor produces the same result as direct access
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<2>{}, Number<4>{}, Number<8>{}));

    using DescType = decltype(desc);

    constexpr auto visible_ids = DescType::GetVisibleDimensionIds();
    constexpr auto functor     = convert_visible_to_hidden_id<DescType>{};

    // Functor result should match direct sequence access
    EXPECT_EQ(functor(Number<0>{}), visible_ids.At(Number<0>{}));
    EXPECT_EQ(functor(Number<1>{}), visible_ids.At(Number<1>{}));
    EXPECT_EQ(functor(Number<2>{}), visible_ids.At(Number<2>{}));
}

TEST(TensorDescriptorFunctorsIntegration, BatchConvertMatchesSingle)
{
    // Verify that batch conversion produces the same result as individual conversions
    constexpr auto desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<4>{}, Number<8>{}, Number<16>{}));

    using DescType = decltype(desc);

    constexpr auto single_functor = convert_visible_to_hidden_id<DescType>{};
    constexpr auto batch_functor  = convert_visible_ids_to_hidden_ids<DescType>{};

    constexpr auto batch_result = batch_functor(Sequence<0, 1, 2>{});

    // Each element should match the single conversion
    EXPECT_EQ(batch_result.At(Number<0>{}), single_functor(Number<0>{}));
    EXPECT_EQ(batch_result.At(Number<1>{}), single_functor(Number<1>{}));
    EXPECT_EQ(batch_result.At(Number<2>{}), single_functor(Number<2>{}));
}

// =============================================================================
// Custom implicit-GEMM transforms of the convolutions
//
// Two custom transforms replace long chains of Pad / Embed / Merge transforms:
//   * ConvFwdDataImplicitGemmInTransform maps the two GEMM indices (m, k) of an implicit im2col to
//     the coordinates of the input tensor (n, i_0.., c). It is the A matrix of the forward
//     convolution and, with the two indices in the opposite roles, the B matrix of the backward
//     weight convolution.
//   * ConvBwdDataImplicitGemmOutTransform maps the GEMM indices of the backward data A matrix to
//     the (Ho, Wo) padded output tensor. Its "merged padding" mode also reports Ho / Wo out of
//     range.
//
// They must be exactly equivalent to the generic chains (the validity of every position and its
// offset), also for the rows / columns of the GEMM padding, which the custom transforms report as
// invalid by "poisoning" a coordinate. The poison only works while the coordinates stay in a range
// that the host checks (IsConvInputTransformRangeSupported, IsMergedOutPaddingRangeSupported and
// the wrappers of the transforms) verify.
// The reference is always the generic chain: with 32-bit indices on small problems, and with 64-bit
// indices (long_index_t, as the device ops use it for tensors above 2GB: 64-bit offsets and no
// magic division) where a 32-bit offset would overflow. Equality checks are exhaustive on small
// problems; on huge problems only boundary values and random samples are compared (NOT exhaustive).
// =============================================================================

// =============================================================================
// Shared helpers
// =============================================================================

namespace {

constexpr int64_t kMaxIndex = kIndexMax;

// The convolution problems are ConvParam: G, N, K, C and the spatial lengths of the input, filter,
// strides, dilations and pads (the output lengths are computed by ConvParam). A ConvParam has a
// run-time number of dimensions, so the functions below take it with the number of spatial
// dimensions as a template argument. BatchK (the split-K factor of the backward weight
// convolution) is not part of a ConvParam and is passed separately.
using ck::utils::conv::ConvParam;

// The first N elements of a container (a std::array or the vectors of a ConvParam) converted to To
template <typename To, index_t N, typename Container>
std::array<To, N> ToArray(const Container& a)
{
    std::array<To, N> r{};
    for(index_t i = 0; i < N; ++i)
    {
        r[i] = static_cast<To>(a[i]);
    }
    return r;
}

template <index_t N, typename Container>
Array<long_index_t, N> ToLongArray(const Container& a)
{
    Array<long_index_t, N> r{};
    for(index_t i = 0; i < N; ++i)
    {
        r(i) = a[i];
    }
    return r;
}

template <index_t NDim, typename Container>
auto ToMultiIndex(const Container& a)
{
    MultiIndex<NDim> r;
    static_for<0, NDim, 1>{}([&](auto i) { r(i) = static_cast<index_t>(a[i]); });
    return r;
}

// ---------------------------------------------------------------------------------------------
// Convolution problems
// ---------------------------------------------------------------------------------------------

// The implicit im2col matrix has the rows (n, o_0..) and the columns (f_0.., c): M x K with
// M = N * prod(O) and K = C * prod(F). They are the GemmM and GemmK of the forward A matrix and the
// GemmK and GemmN of the backward weight B matrix.
int64_t ImplicitGemmM(const ConvParam& c)
{
    return std::accumulate(c.output_spatial_lengths_.begin(),
                           c.output_spatial_lengths_.end(),
                           c.N_,
                           std::multiplies<int64_t>());
}

int64_t ImplicitGemmK(const ConvParam& c)
{
    return std::accumulate(c.filter_spatial_lengths_.begin(),
                           c.filter_spatial_lengths_.end(),
                           c.C_,
                           std::multiplies<int64_t>());
}

// ---------------------------------------------------------------------------------------------
// Checks shared by all the transforms
// ---------------------------------------------------------------------------------------------

// Moving a coordinate (the UpdateLowerIndex path the kernels use to walk a descriptor) must agree
// with building the coordinate from scratch: same offset, same validity. The walk is random with a
// fixed seed. Its steps go forward and backward, change several dimensions at once and include
// jumps across a whole dimension (from the largest, poisoned coordinate back to the smallest one).
// `extra_steps[d]` adds problem specific step sizes for dimension d (for instance the channel
// count, which makes a merged index carry). NOT exhaustive. Returns the number of mismatches.
template <typename Desc>
int64_t CountCoordinateMoveMismatches(const Desc& desc,
                                      int steps,
                                      uint32_t seed,
                                      const std::vector<std::vector<index_t>>& extra_steps = {})
{
    constexpr index_t N = Desc::GetNumOfDimension();
    std::array<index_t, N> lengths{};
    static_for<0, N, 1>{}([&](auto i) { lengths[i] = desc.GetLength(i); });

    const auto to_index = [&](const std::array<index_t, N>& a) {
        MultiIndex<N> idx;
        static_for<0, N, 1>{}([&](auto i) { idx(i) = a[i]; });
        return idx;
    };

    std::mt19937 rng(seed);
    const auto pick = [&](index_t dim) {
        const index_t length = lengths[dim];
        std::vector<index_t> options{0,
                                     1,
                                     -1,
                                     2,
                                     -2,
                                     3,
                                     -3,
                                     8,
                                     -8,
                                     16,
                                     -16,
                                     length / 2,
                                     -(length / 2),
                                     length - 1,
                                     -(length - 1)};
        if(static_cast<std::size_t>(dim) < extra_steps.size())
        {
            for(const index_t step : extra_steps[dim])
            {
                options.push_back(step);
                options.push_back(-step);
            }
        }
        return options[rng() % options.size()];
    };

    std::array<index_t, N> current{};
    auto coordinate    = make_tensor_coordinate(desc, to_index(current));
    int64_t mismatches = 0;
    for(int step = 0; step < steps; ++step)
    {
        std::array<index_t, N> delta{}, next{};
        bool inside = true;
        for(index_t i = 0; i < N; ++i)
        {
            delta[i] = pick(i);
            next[i]  = current[i] + delta[i];
            inside &= next[i] >= 0 && next[i] < lengths[i];
        }
        if(!inside)
        {
            continue;
        }
        current = next;
        move_tensor_coordinate(
            desc, coordinate, make_tensor_coordinate_step(desc, to_index(delta)));
        const auto fresh = make_tensor_coordinate(desc, to_index(current));
        const bool valid_moved =
            coordinate_has_valid_offset_assuming_visible_index_is_valid(desc, coordinate);
        const bool valid_fresh =
            coordinate_has_valid_offset_assuming_visible_index_is_valid(desc, fresh);
        if(valid_moved != valid_fresh ||
           (valid_fresh && long_index_t{coordinate.GetOffset()} != long_index_t{fresh.GetOffset()}))
        {
            if(mismatches++ < 3)
            {
                ADD_FAILURE() << "moving the coordinate disagrees with a new one at step " << step
                              << ": moved valid=" << valid_moved
                              << " offset=" << coordinate.GetOffset()
                              << ", new valid=" << valid_fresh << " offset=" << fresh.GetOffset();
            }
        }
    }
    return mismatches;
}

// Compares two 3D descriptors of the same lengths (the custom transform and the generic chain) at
// the given positions of each dimension: the validity of the offset and, if valid, the offset.
// Fails on any mismatch and returns the number of positions that the generic chain reports as
// valid.
template <typename GenericIndex = index_t, typename Custom, typename Generic>
int64_t CompareDescriptors3D(const Custom& custom,
                             const Generic& generic,
                             const std::vector<index_t>& positions0,
                             const std::vector<index_t>& positions1,
                             const std::vector<index_t>& positions2)
{
    int64_t valid      = 0;
    int64_t mismatches = 0;
    for(const index_t i0 : positions0)
    {
        for(const index_t i1 : positions1)
        {
            for(const index_t i2 : positions2)
            {
                const auto index = make_multi_index(i0, i1, i2);
                const auto cc    = make_tensor_coordinate(custom, index);
                const auto gc    = make_tensor_coordinate<GenericIndex>(generic, index);
                const bool cv    = coordinate_has_valid_offset(custom, cc);
                const bool gv    = coordinate_has_valid_offset(generic, gc);
                valid += gv;
                if((cv != gv || (cv && cc.GetOffset() != gc.GetOffset())) && mismatches++ < 3)
                {
                    ADD_FAILURE() << "(" << i0 << "," << i1 << "," << i2 << ") custom valid=" << cv
                                  << " offset=" << cc.GetOffset() << ", generic valid=" << gv
                                  << " offset=" << gc.GetOffset();
                }
            }
        }
    }
    return valid;
}

// ---------------------------------------------------------------------------------------------
// Test problems
// ---------------------------------------------------------------------------------------------

// {{ndim, G, N, K, C, filter, input, stride, dilation, left pad, right pad}}; the checks run each
// problem with G = 1 (a packed tensor) and with G = 3 (a tensor with interleaved groups)
const std::vector<ConvParam>& ForwardCases()
{
    static const std::vector<ConvParam> cases = {
        // ---- 1D
        {1, 1, 2, 8, 3, {3}, {17}, {1}, {1}, {1}, {1}},
        {1, 1, 3, 8, 5, {4}, {13}, {2}, {2}, {3}, {0}},
        {1, 1, 2, 8, 1, {1}, {9}, {1}, {1}, {0}, {0}},
        {1, 1, 2, 8, 7, {1}, {10}, {3}, {1}, {0}, {0}},
        {1, 1, 1, 8, 4, {5}, {5}, {1}, {1}, {0}, {0}},
        {1, 1, 2, 8, 4, {3}, {2}, {1}, {1}, {4}, {5}},
        {1, 1, 2, 8, 3, {3}, {31}, {4}, {6}, {6}, {6}},
        {1, 1, 3, 8, 16, {3}, {12}, {1}, {1}, {1}, {1}},
        // ---- 2D
        // original configuration
        {2, 1, 1, 8, 2, {3, 3}, {7, 7}, {2, 2}, {2, 2}, {1, 2}, {5, 4}},
        // N > 1 with every H/W parameter different (Ho != Wo, Y != X, ...)
        {2, 1, 3, 8, 5, {3, 2}, {9, 13}, {2, 3}, {1, 2}, {1, 3}, {4, 0}},
        {2, 1, 2, 8, 4, {2, 5}, {8, 10}, {2, 1}, {3, 2}, {0, 2}, {1, 0}},
        {2, 1, 5, 8, 6, {3, 3}, {14, 11}, {1, 2}, {1, 1}, {1, 1}, {1, 1}},
        // different channel counts: 1, odd, non power of two, power of two
        {2, 1, 2, 8, 1, {3, 3}, {6, 5}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 1, 2, 8, 3, {3, 3}, {6, 5}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 1, 2, 8, 7, {3, 3}, {6, 5}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 1, 2, 8, 16, {3, 3}, {6, 5}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        // 1x1 filters, with and without stride
        {2, 1, 2, 8, 4, {1, 1}, {5, 6}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        {2, 1, 2, 8, 4, {1, 1}, {5, 6}, {2, 3}, {1, 1}, {0, 0}, {0, 0}},
        // Ho == Wo == 1
        {2, 1, 3, 8, 4, {3, 3}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        // padding larger than the input
        {2, 1, 2, 8, 3, {3, 3}, {2, 3}, {1, 1}, {1, 1}, {4, 5}, {4, 5}},
        // right padding not fully consumed by the stride
        {2, 1, 2, 8, 3, {3, 3}, {7, 7}, {3, 3}, {1, 1}, {1, 1}, {3, 3}},
        // large stride and dilation
        {2, 1, 2, 8, 3, {3, 3}, {31, 29}, {4, 5}, {6, 7}, {6, 7}, {6, 7}},
        // ---- 3D
        {3, 1, 2, 8, 3, {3, 2, 3}, {5, 6, 7}, {1, 2, 1}, {1, 1, 2}, {1, 0, 1}, {1, 1, 0}},
        {3, 1, 3, 8, 4, {2, 3, 2}, {4, 9, 5}, {2, 1, 3}, {2, 2, 1}, {0, 2, 1}, {1, 0, 2}},
        {3, 1, 2, 8, 1, {1, 1, 1}, {4, 4, 4}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        {3, 1, 2, 8, 5, {1, 1, 1}, {4, 5, 6}, {2, 3, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        {3, 1, 2, 8, 4, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        {3, 1, 2, 8, 3, {3, 3, 3}, {2, 2, 3}, {1, 1, 1}, {1, 1, 1}, {3, 4, 5}, {3, 4, 5}},
        {3, 1, 2, 8, 3, {3, 3, 3}, {9, 9, 9}, {3, 3, 3}, {2, 1, 2}, {1, 3, 1}, {2, 3, 0}},
        {3, 1, 1, 8, 16, {3, 3, 3}, {5, 4, 6}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
    };
    return cases;
}

// Extreme problems, checked against the generic chain with 64-bit indices (a 32-bit offset would
// overflow here)
const std::vector<ConvParam>& ExtremeForwardCases()
{
    static const std::vector<ConvParam> cases = {
        // clang-format off
        // ---- 1D
        // Wo = 2^29: largest first-dimension extent supported by the padding poison
        {1, 1, 1, 8, 1, {1}, {1 << 29}, {1}, {1}, {0}, {0}},
        // Wo = 65537, C = 65537
        {1, 1, 1, 8, 1, {1}, {65537}, {1}, {1}, {0}, {0}},
        {1, 1, 2, 8, 65537, {1}, {3}, {1}, {1}, {0}, {0}},
        // K = Y * C just below 2^31
        {1, 1, 1, 8, 715827882, {3}, {3}, {1}, {1}, {0}, {0}},
        // very negative and very large lower indices; only wi == 0 is valid
        {1, 1, 1, 8, 1, {1}, {4}, {1 << 28}, {1}, {1 << 29}, {0}},
        {1, 1, 1, 8, 1, {1}, {4}, {1 << 28}, {1}, {0}, {1 << 30}},
        {1, 1, 1, 8, 1, {1}, {4}, {1 << 28}, {1}, {(1 << 29) - 5}, {0}},
        {1, 1, 1, 8, 1, {2}, {1 << 29}, {(1 << 29) - 1}, {1}, {0}, {0}},
        // ---- 2D
        // M just below 2^31 and a large channel count
        {2, 1, 32767, 8, 4096, {3, 3}, {256, 256}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        // power-of-two divisors everywhere
        {2, 1, 4, 8, 64, {2, 2}, {16, 16}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        // divisors of the form 2^k + 1 and 2^k
        {2, 1, 2, 8, 65, {3, 3}, {33, 17}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 1, 2, 8, 65536, {3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        // large divisors: Wo = 65537, C = 65537
        {2, 1, 1, 8, 1, {1, 1}, {1, 65537}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        {2, 1, 2, 8, 65537, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        // very negative lower indices; only hi == 0 / wi == 0 is valid
        {2, 1, 1, 8, 1, {1, 1}, {4, 4}, {1 << 28, 1 << 28}, {1, 1}, {1 << 29, 1 << 30}, {0, 0}},
        // very large positive lower indices; only hi == 0 / wi == 0 is valid
        {2, 1, 1, 8, 1, {1, 1}, {4, 4}, {1 << 28, 1 << 28}, {1, 1}, {0, 0}, {1 << 30, 1 << 30}},
        {2, 1, 1, 8, 1, {1, 1}, {4, 4}, {1 << 28, 1 << 28}, {1, 1}, {0, 0}, {7 << 28, 7 << 28}},
        // large extents: Hi = 2^29, Wi = kIndexMax, filter taps reaching Hi - 1 and Hi
        {2, 1, 1, 8, 1, {2, 2}, {1 << 29, kIndexMax}, {(1 << 29) - 1, kIndexMax - 1}, {1, 1}, {0, 0}, {0, 0}},
        {2, 1, 1, 8, 1, {2, 2}, {1 << 29, kIndexMax}, {1 << 29, kIndexMax}, {1, 1}, {0, 0}, {1, 1}},
        // Wo = kIndexMax (largest magic-division divisor) with M = kIndexMax
        {2, 1, 1, 8, 1, {1, 1}, {1, kIndexMax}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        // K = Y * X * C just below 2^31, with X = 1 (divisor 1) and a huge C divisor
        {2, 1, 1, 8, 715827882, {3, 1}, {3, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        // ---- 3D
        // M just below 2^31, large C
        {3, 1, 2047, 8, 1024, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        // power-of-two divisors everywhere
        {3, 1, 4, 8, 64, {2, 2, 2}, {8, 8, 8}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        // divisors of the form 2^k + 1
        {3, 1, 2, 8, 65, {3, 3, 3}, {5, 17, 33}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        // large divisors on both sides
        {3, 1, 1, 8, 1, {1, 1, 1}, {1, 1, 65537}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        {3, 1, 2, 8, 65537, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        // very negative / large lower indices on every dimension. The pads are 2^20, and not the
        // 2^29..2^30 of the 1D / 2D cases: the generic chain pads all the dimensions, and a 3D
        // padded input of 2^29 x 2^30 x 2^30 elements would overflow the 64-bit element count of
        // its descriptor.
        {3, 1, 1, 8, 1, {1, 1, 1}, {4, 4, 4}, {1 << 19, 1 << 19, 1 << 19}, {1, 1, 1}, {1 << 20, 1 << 20, 1 << 20}, {0, 0, 0}},
        {3, 1, 1, 8, 1, {1, 1, 1}, {4, 4, 4}, {1 << 19, 1 << 19, 1 << 19}, {1, 1, 1}, {0, 0, 0}, {1 << 20, 1 << 20, 1 << 20}},
        // Wo = kIndexMax with the other output dimensions 1
        {3, 1, 1, 8, 1, {1, 1, 1}, {1, 1, kIndexMax}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        // K just below 2^31 with filter 3 x 1 x 1 and a huge C divisor
        {3, 1, 1, 8, 715827882, {3, 1, 1}, {3, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        // clang-format on
    };
    return cases;
}

// Problems with filters of 1 to 16 taps (odd and even; in 1D, 2D and 3D, the long filter is in a
// different dimension each time) with strides, dilations, pads, batches, channels and groups, and a
// small input so that every position of the GEMM can be compared (G is set by the checks). Each
// comes with a split-K factor for the backward weight tests.
// {{ndim, G, N, K, C, filter, input, stride, dilation, pad left, pad right}, split-K}
const std::vector<std::pair<ConvParam, index_t>>& FilterSweepCases()
{
    static const std::vector<std::pair<ConvParam, index_t>> cases = {
        {{1, 1, 2, 8, 4, {1}, {11}, {2}, {2}, {1}, {1}}, 2},
        {{1, 1, 1, 8, 5, {2}, {12}, {3}, {1}, {3}, {2}}, 3},
        {{1, 1, 2, 8, 6, {3}, {10}, {1}, {2}, {0}, {0}}, 1},
        {{1, 1, 1, 8, 7, {4}, {11}, {2}, {1}, {1}, {1}}, 2},
        {{1, 1, 2, 8, 3, {5}, {12}, {3}, {2}, {3}, {2}}, 3},
        {{1, 1, 1, 8, 4, {6}, {10}, {1}, {1}, {0}, {0}}, 1},
        {{1, 1, 2, 8, 5, {7}, {11}, {2}, {2}, {1}, {2}}, 2},
        {{1, 1, 1, 8, 6, {8}, {12}, {3}, {1}, {3}, {2}}, 3},
        {{1, 1, 2, 8, 7, {9}, {10}, {1}, {2}, {0}, {7}}, 1},
        {{1, 1, 1, 8, 3, {10}, {11}, {2}, {1}, {1}, {1}}, 2},
        {{1, 1, 2, 8, 4, {11}, {12}, {3}, {2}, {3}, {8}}, 3},
        {{1, 1, 1, 8, 5, {12}, {10}, {1}, {1}, {0}, {2}}, 1},
        {{1, 1, 2, 8, 6, {13}, {11}, {2}, {2}, {1}, {14}}, 2},
        {{1, 1, 1, 8, 7, {14}, {12}, {3}, {1}, {3}, {2}}, 3},
        {{1, 1, 2, 8, 3, {15}, {10}, {1}, {2}, {0}, {19}}, 1},
        {{1, 1, 1, 8, 4, {16}, {11}, {2}, {1}, {1}, {5}}, 2},

        {{2, 1, 2, 8, 4, {2, 1}, {11, 6}, {2, 3}, {2, 1}, {1, 3}, {1, 2}}, 2},
        {{2, 1, 1, 8, 5, {2, 4}, {12, 5}, {3, 1}, {1, 2}, {3, 0}, {2, 2}}, 3},
        {{2, 1, 2, 8, 6, {4, 3}, {10, 6}, {1, 2}, {2, 1}, {0, 1}, {0, 1}}, 1},
        {{2, 1, 1, 8, 7, {4, 2}, {11, 5}, {2, 3}, {1, 2}, {1, 3}, {1, 2}}, 2},
        {{2, 1, 2, 8, 3, {2, 5}, {12, 6}, {3, 1}, {2, 1}, {3, 0}, {2, 0}}, 3},
        {{2, 1, 1, 8, 4, {6, 4}, {10, 5}, {1, 2}, {1, 2}, {0, 1}, {0, 2}}, 1},
        {{2, 1, 2, 8, 5, {4, 7}, {11, 6}, {2, 3}, {2, 1}, {1, 3}, {1, 2}}, 2},
        {{2, 1, 1, 8, 6, {8, 2}, {12, 5}, {3, 1}, {1, 2}, {3, 0}, {2, 0}}, 3},
        {{2, 1, 2, 8, 7, {2, 9}, {10, 6}, {1, 2}, {2, 1}, {0, 1}, {0, 3}}, 1},
        {{2, 1, 1, 8, 3, {10, 4}, {11, 5}, {2, 3}, {1, 2}, {1, 3}, {1, 2}}, 2},
        {{2, 1, 2, 8, 4, {4, 11}, {12, 6}, {3, 1}, {2, 1}, {3, 0}, {2, 5}}, 3},
        {{2, 1, 1, 8, 5, {12, 2}, {10, 5}, {1, 2}, {1, 2}, {0, 1}, {2, 1}}, 1},
        {{2, 1, 2, 8, 6, {2, 13}, {11, 6}, {2, 3}, {2, 1}, {1, 3}, {1, 6}}, 2},
        {{2, 1, 1, 8, 7, {14, 4}, {12, 5}, {3, 1}, {1, 2}, {3, 0}, {2, 2}}, 3},
        {{2, 1, 2, 8, 3, {4, 15}, {10, 6}, {1, 2}, {2, 1}, {0, 1}, {0, 9}}, 1},
        {{2, 1, 1, 8, 4, {16, 2}, {11, 5}, {2, 3}, {1, 2}, {1, 3}, {5, 2}}, 2},

        {{3, 1, 2, 8, 4, {2, 1, 4}, {11, 6, 4}, {2, 1, 1}, {2, 1, 2}, {1, 1, 0}, {1, 2, 3}}, 2},
        {{3, 1, 1, 8, 5, {3, 4, 2}, {12, 5, 4}, {3, 1, 2}, {1, 2, 1}, {3, 0, 1}, {2, 2, 1}}, 3},
        {{3, 1, 2, 8, 6, {3, 1, 2}, {10, 6, 4}, {1, 2, 3}, {2, 1, 2}, {0, 1, 3}, {0, 1, 2}}, 1},
        {{3, 1, 1, 8, 7, {1, 4, 3}, {11, 5, 4}, {2, 1, 1}, {1, 2, 1}, {1, 1, 0}, {1, 2, 0}}, 2},
        {{3, 1, 2, 8, 3, {2, 1, 5}, {12, 6, 4}, {3, 1, 2}, {2, 1, 2}, {3, 0, 1}, {2, 0, 5}}, 3},
        {{3, 1, 1, 8, 4, {6, 4, 1}, {10, 5, 4}, {1, 2, 3}, {1, 2, 1}, {0, 1, 3}, {0, 2, 2}}, 1},
        {{3, 1, 2, 8, 5, {4, 7, 2}, {11, 6, 4}, {2, 1, 1}, {2, 1, 2}, {1, 1, 0}, {1, 2, 0}}, 2},
        {{3, 1, 1, 8, 6, {1, 2, 8}, {12, 5, 4}, {3, 1, 2}, {1, 2, 1}, {3, 0, 1}, {2, 0, 4}}, 3},
        {{3, 1, 2, 8, 7, {9, 1, 4}, {10, 6, 4}, {1, 2, 3}, {2, 1, 2}, {0, 1, 3}, {7, 1, 2}}, 1},
        {{3, 1, 1, 8, 3, {3, 10, 1}, {11, 5, 4}, {2, 1, 1}, {1, 2, 1}, {1, 1, 0}, {1, 13, 0}}, 2},
        {{3, 1, 2, 8, 4, {4, 1, 11}, {12, 6, 4}, {3, 1, 2}, {2, 1, 2}, {3, 0, 1}, {2, 0, 17}}, 3},
        {{3, 1, 1, 8, 5, {12, 2, 3}, {10, 5, 4}, {1, 2, 3}, {1, 2, 1}, {0, 1, 3}, {2, 1, 2}}, 1},
        {{3, 1, 2, 8, 6, {2, 13, 4}, {11, 6, 4}, {2, 1, 1}, {2, 1, 2}, {1, 1, 0}, {1, 6, 3}}, 2},
        {{3, 1, 1, 8, 7, {3, 4, 14}, {12, 5, 4}, {3, 1, 2}, {1, 2, 1}, {3, 0, 1}, {2, 2, 10}}, 3},
        {{3, 1, 2, 8, 3, {15, 1, 2}, {10, 6, 4}, {1, 2, 3}, {2, 1, 2}, {0, 1, 3}, {19, 1, 2}}, 1},
        {{3, 1, 1, 8, 4, {1, 16, 3}, {11, 5, 4}, {2, 1, 1}, {1, 2, 1}, {1, 1, 0}, {1, 25, 0}}, 2},
    };
    return cases;
}

// {{2, G, N, K, C, filter, input, stride, dilation, left pad, right pad}}, as above
const std::vector<ConvParam>& BwdDataCases()
{
    static const std::vector<ConvParam> cases = {
        {2, 1, 2, 8, 8, {3, 3}, {9, 9}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 1, 2, 24, 8, {3, 3}, {11, 7}, {2, 2}, {1, 1}, {1, 1}, {1, 1}},
        {2, 1, 3, 8, 8, {5, 3}, {10, 13}, {1, 2}, {2, 1}, {2, 1}, {2, 1}},
        {2, 1, 1, 40, 8, {3, 3}, {12, 12}, {2, 3}, {1, 1}, {0, 2}, {0, 2}},
        {2, 1, 2, 8, 8, {1, 1}, {6, 6}, {2, 2}, {1, 1}, {0, 0}, {0, 0}},
        {2, 1, 2, 8, 8, {4, 4}, {8, 9}, {3, 2}, {2, 3}, {3, 4}, {2, 1}},
        // K * YDotSlice * XDotSlice and N * HTildeSlice * WTildeSlice are not block multiples
        {2, 1, 3, 8, 8, {3, 3}, {7, 5}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 1, 1, 72, 8, {3, 3}, {5, 9}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
    };
    return cases;
}

// Problems with filters of 1 to 16 taps (odd and even, the filter of the other dimension has the
// opposite parity most of the time), strides and dilations of both dimensions, and pads that keep
// the output non empty. {2, G, N, K, C, {Y, X}, {Hi, Wi}, {strides}, {dilations}, {left pads},
// {right pads}}
const std::vector<ConvParam>& BwdDataSweepCases()
{
    static const std::vector<ConvParam> cases = {
        {2, 1, 2, 8, 8, {1, 8}, {10, 12}, {2, 1}, {2, 2}, {1, 3}, {1, 1}},
        {2, 1, 2, 8, 8, {2, 15}, {11, 9}, {1, 3}, {1, 3}, {3, 0}, {2, 34}},
        {2, 1, 2, 8, 8, {3, 6}, {12, 10}, {3, 2}, {2, 1}, {0, 1}, {0, 1}},
        {2, 1, 2, 8, 8, {4, 13}, {9, 11}, {2, 2}, {1, 1}, {1, 3}, {1, 0}},
        {2, 1, 2, 8, 8, {5, 4}, {10, 12}, {1, 1}, {2, 2}, {3, 0}, {2, 1}},
        {2, 1, 2, 8, 8, {6, 11}, {11, 9}, {2, 1}, {1, 3}, {0, 1}, {0, 21}},
        {2, 1, 2, 8, 8, {7, 2}, {12, 10}, {1, 3}, {2, 1}, {1, 3}, {1, 1}},
        {2, 1, 2, 8, 8, {8, 9}, {9, 11}, {3, 2}, {1, 1}, {3, 0}, {2, 0}},
        {2, 1, 2, 8, 8, {9, 16}, {10, 12}, {2, 2}, {2, 2}, {0, 1}, {7, 19}},
        {2, 1, 2, 8, 8, {10, 7}, {11, 9}, {1, 1}, {1, 3}, {1, 3}, {1, 7}},
        {2, 1, 2, 8, 8, {11, 14}, {12, 10}, {2, 1}, {2, 1}, {3, 0}, {8, 5}},
        {2, 1, 2, 8, 8, {12, 5}, {9, 11}, {1, 3}, {1, 1}, {0, 1}, {3, 0}},
        {2, 1, 2, 8, 8, {13, 12}, {10, 12}, {3, 2}, {2, 2}, {1, 3}, {15, 9}},
        {2, 1, 2, 8, 8, {14, 3}, {11, 9}, {2, 2}, {1, 3}, {3, 0}, {2, 0}},
        {2, 1, 2, 8, 8, {15, 10}, {12, 10}, {1, 1}, {2, 1}, {0, 1}, {17, 1}},
        {2, 1, 2, 8, 8, {16, 1}, {9, 11}, {2, 1}, {1, 1}, {1, 3}, {7, 0}},
    };
    return cases;
}

// Problems with a filter whose taps are spread by a big dilation along one dimension (H or W) and
// one position in the other: N = G = 1, C = K = 8, Hi or Wi = 8, no left pad. The right pad is
// D * (F - 1) so that the output has as many positions as the input. The taps and the padded
// GemmK columns reach far outside the tensor, which is what the host check of the merged padding
// limits. For each filter and stride, the dilations just below and above the limit of the host
// check, and whether it accepts every group of the problem.
// {{2, G, N, K, C, filter, input, stride, dilation, left pad, right pad}, accepted}
const std::vector<std::pair<ConvParam, bool>>& BwdDataDilationLimitCases()
{
    static const std::vector<std::pair<ConvParam, bool>> cases = {
        // clang-format off
        // H F=2 stride=1 limit=2606164
        {{2, 1, 1, 8, 8, {2, 1}, {8, 1}, {1, 1}, {2606164, 1}, {0, 0}, {2606164, 0}}, true},
        {{2, 1, 1, 8, 8, {2, 1}, {8, 1}, {1, 1}, {2606165, 1}, {0, 0}, {2606165, 0}}, false},
        // H F=2 stride=2 limit=5212332
        {{2, 1, 1, 8, 8, {2, 1}, {8, 1}, {2, 1}, {5212332, 1}, {0, 0}, {5212332, 0}}, true},
        {{2, 1, 1, 8, 8, {2, 1}, {8, 1}, {2, 1}, {5212333, 1}, {0, 0}, {5212333, 0}}, false},
        // H F=3 stride=1 limit=2508737
        {{2, 1, 1, 8, 8, {3, 1}, {8, 1}, {1, 1}, {2508737, 1}, {0, 0}, {5017474, 0}}, true},
        {{2, 1, 1, 8, 8, {3, 1}, {8, 1}, {1, 1}, {2508738, 1}, {0, 0}, {5017476, 0}}, false},
        // H F=3 stride=2 limit=5017478
        {{2, 1, 1, 8, 8, {3, 1}, {8, 1}, {2, 1}, {5017478, 1}, {0, 0}, {10034956, 0}}, true},
        {{2, 1, 1, 8, 8, {3, 1}, {8, 1}, {2, 1}, {5017479, 1}, {0, 0}, {10034958, 0}}, false},
        // H F=16 stride=1 limit=1688269
        {{2, 1, 1, 8, 8, {16, 1}, {8, 1}, {1, 1}, {1688269, 1}, {0, 0}, {25324035, 0}}, true},
        {{2, 1, 1, 8, 8, {16, 1}, {8, 1}, {1, 1}, {1688270, 1}, {0, 0}, {25324050, 0}}, false},
        // H F=16 stride=2 limit=2162392
        {{2, 1, 1, 8, 8, {16, 1}, {8, 1}, {2, 1}, {2162392, 1}, {0, 0}, {32435880, 0}}, true},
        {{2, 1, 1, 8, 8, {16, 1}, {8, 1}, {2, 1}, {2162393, 1}, {0, 0}, {32435895, 0}}, false},
        // W F=2 stride=1 limit=7669546
        {{2, 1, 1, 8, 8, {1, 2}, {1, 8}, {1, 1}, {1, 7669546}, {0, 0}, {0, 7669546}}, true},
        {{2, 1, 1, 8, 8, {1, 2}, {1, 8}, {1, 1}, {1, 7669547}, {0, 0}, {0, 7669547}}, false},
        // W F=3 stride=1 limit=7254976
        {{2, 1, 1, 8, 8, {1, 3}, {1, 8}, {1, 1}, {1, 7254976}, {0, 0}, {0, 14509952}}, true},
        {{2, 1, 1, 8, 8, {1, 3}, {1, 8}, {1, 1}, {1, 7254977}, {0, 0}, {0, 14509954}}, false},
        // W F=16 stride=1 limit=4260859
        {{2, 1, 1, 8, 8, {1, 16}, {1, 8}, {1, 1}, {1, 4260859}, {0, 0}, {0, 63912885}}, true},
        {{2, 1, 1, 8, 8, {1, 16}, {1, 8}, {1, 1}, {1, 4260860}, {0, 0}, {0, 63912900}}, false},
        // clang-format on
    };
    return cases;
}

// {{ndim, G, N, K, C, filter, input, stride, dilation, left pad, right pad}, split-K}, as above
const std::vector<std::pair<ConvParam, index_t>>& BwdWeightCases()
{
    static const std::vector<std::pair<ConvParam, index_t>> cases = {
        // ---- 1D
        {ConvParam{1, 1, 2, 8, 8, {3}, {17}, {1}, {1}, {1}, {1}}, 1},
        {ConvParam{1, 1, 3, 8, 5, {4}, {13}, {2}, {2}, {3}, {0}}, 2},
        {ConvParam{1, 1, 2, 8, 7, {1}, {10}, {3}, {1}, {0}, {0}}, 3},
        // ---- 2D
        {ConvParam{2, 1, 2, 8, 8, {3, 3}, {9, 9}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, 1},
        {ConvParam{2, 1, 2, 8, 8, {3, 3}, {11, 7}, {2, 2}, {1, 1}, {1, 1}, {1, 1}}, 2},
        {ConvParam{2, 1, 3, 8, 5, {5, 3}, {10, 13}, {1, 2}, {2, 1}, {2, 1}, {2, 1}}, 1},
        {ConvParam{2, 1, 1, 8, 6, {3, 3}, {12, 12}, {2, 3}, {1, 1}, {0, 2}, {0, 2}}, 3},
        {ConvParam{2, 1, 2, 8, 8, {1, 1}, {6, 6}, {2, 2}, {1, 1}, {0, 0}, {0, 0}}, 2},
        {ConvParam{2, 1, 2, 8, 3, {4, 4}, {8, 9}, {3, 2}, {2, 3}, {3, 4}, {2, 1}}, 1},
        {ConvParam{2, 1, 3, 8, 8, {3, 3}, {7, 5}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, 4},
        // ---- 3D
        {ConvParam{3, 1, 1, 8, 4, {3, 3, 3}, {5, 6, 7}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
         1},
        {ConvParam{3, 1, 2, 8, 3, {2, 3, 3}, {6, 5, 7}, {2, 1, 2}, {1, 2, 1}, {1, 2, 0}, {0, 1, 1}},
         2},
        {ConvParam{3, 1, 2, 8, 8, {1, 1, 1}, {4, 4, 4}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
         3},
    };
    return cases;
}

// A problem at the limit of the host checks and whether the host check of the forward A matrix and
// the one of the backward weight B matrix accept it.
struct LimitCase
{
    ConvParam problem;
    bool fwd_supported;
    bool wrw_supported;
};

// Problems with a filter whose taps are spread by a big dilation in one spatial dimension, so that
// the lower coordinates (o * stride + f * dilation - left pad, and the padded GEMM columns that
// continue the filter index) reach far outside the input while the tensor stays small. This is
// what the host checks limit: a poisoned coordinate (the poison is length + left pad, twice for a
// padding row and column) must not wrap around into the input, and the offset terms
// (coordinate * stride) must not overflow int32. The dimension is the first one (which gets the
// poison) or the last one (the one with the smallest stride in NHWGC), the right pad is
// D * (F - 1) so that Wo = Wi + left pad, and the other dimensions have an input of 3 (2 if
// N = 3), a filter of 2, a stride of 2 and a pad of 1. Each group of rows is the same problem with
// the dilations just below and above the limit of each host check (the limits are in the comment of
// the group). The accepted problems are compared exactly with the generic chain with 64-bit
// indices, in particular the positions where both the row and the column are padding; the groups
// with N = 3 and an input of 2^20 also have a batch stride and a big stride, to catch a wrongly
// read value of the problem.
// {{ndim, G, N, K, C, filter, input, stride, dilation, left pad, right pad}, forward, backward}
const std::vector<LimitCase>& LimitCases()
{
    static const std::vector<LimitCase> cases = {
        // clang-format off
        // ---- 1D
        // dim 0, F=2, C=8, N=1, in=8, left pad 0: limits fwd 38347883, wrw 89478474
        {{1, 1, 1, 8, 8, {2}, {8}, {1}, {38347883}, {0}, {38347883}}, true, true},
        {{1, 1, 1, 8, 8, {2}, {8}, {1}, {38347884}, {0}, {38347884}}, false, true},
        {{1, 1, 1, 8, 8, {2}, {8}, {1}, {89478474}, {0}, {89478474}}, false, true},
        {{1, 1, 1, 8, 8, {2}, {8}, {1}, {89478475}, {0}, {89478475}}, false, false},
        // dim 0, F=3, C=8, N=1, in=8, left pad 5: limits fwd 38347895, wrw 89478473
        {{1, 1, 1, 8, 8, {3}, {8}, {1}, {38347895}, {5}, {76695790}}, true, true},
        {{1, 1, 1, 8, 8, {3}, {8}, {1}, {38347896}, {5}, {76695792}}, false, true},
        {{1, 1, 1, 8, 8, {3}, {8}, {1}, {89478473}, {5}, {178956946}}, false, true},
        {{1, 1, 1, 8, 8, {3}, {8}, {1}, {89478474}, {5}, {178956948}}, false, false},
        // dim 0, F=16, C=8, N=1, in=8, left pad 0: limits fwd 17895678, wrw 17895694
        {{1, 1, 1, 8, 8, {16}, {8}, {1}, {17895678}, {0}, {268435170}}, true, true},
        {{1, 1, 1, 8, 8, {16}, {8}, {1}, {17895679}, {0}, {268435185}}, false, true},
        {{1, 1, 1, 8, 8, {16}, {8}, {1}, {17895694}, {0}, {268435410}}, false, true},
        {{1, 1, 1, 8, 8, {16}, {8}, {1}, {17895695}, {0}, {268435425}}, false, false},
        // dim 0, F=3, C=40, N=1, in=8, left pad 0: limits fwd 17895606, wrw 17895686
        {{1, 1, 1, 8, 40, {3}, {8}, {1}, {17895606}, {0}, {35791212}}, true, true},
        {{1, 1, 1, 8, 40, {3}, {8}, {1}, {17895607}, {0}, {35791214}}, false, true},
        {{1, 1, 1, 8, 40, {3}, {8}, {1}, {17895686}, {0}, {35791372}}, false, true},
        {{1, 1, 1, 8, 40, {3}, {8}, {1}, {17895687}, {0}, {35791374}}, false, false},
        // dim 0, F=3, C=8, N=1, in=8, left pad 1048576: limits fwd 38048290, wrw 88779424
        {{1, 1, 1, 8, 8, {3}, {8}, {1}, {38048290}, {1048576}, {76096580}}, true, true},
        {{1, 1, 1, 8, 8, {3}, {8}, {1}, {38048291}, {1048576}, {76096582}}, false, true},
        {{1, 1, 1, 8, 8, {3}, {8}, {1}, {88779424}, {1048576}, {177558848}}, false, true},
        {{1, 1, 1, 8, 8, {3}, {8}, {1}, {88779425}, {1048576}, {177558850}}, false, false},
        // dim 0, F=3, C=8, N=3, in=1048576, left pad 5: limits fwd 37598903, wrw 87730855
        {{1, 1, 3, 8, 8, {3}, {1048576}, {1}, {37598903}, {5}, {75197806}}, true, true},
        {{1, 1, 3, 8, 8, {3}, {1048576}, {1}, {37598904}, {5}, {75197808}}, false, true},
        {{1, 1, 3, 8, 8, {3}, {1048576}, {1}, {87730855}, {5}, {175461710}}, false, true},
        {{1, 1, 3, 8, 8, {3}, {1048576}, {1}, {87730856}, {5}, {175461712}}, false, false},

        // ---- 2D
        // dim 0, F=2, C=8, N=1, in=8, left pad 0: limits fwd 29826113, wrw 89478469
        {{2, 1, 1, 8, 8, {2, 2}, {8, 3}, {1, 2}, {29826113, 1}, {0, 1}, {29826113, 1}}, true, true},
        {{2, 1, 1, 8, 8, {2, 2}, {8, 3}, {1, 2}, {29826114, 1}, {0, 1}, {29826114, 1}}, false, true},
        {{2, 1, 1, 8, 8, {2, 2}, {8, 3}, {1, 2}, {89478469, 1}, {0, 1}, {89478469, 1}}, false, true},
        {{2, 1, 1, 8, 8, {2, 2}, {8, 3}, {1, 2}, {89478470, 1}, {0, 1}, {89478470, 1}}, false, false},
        // dim 0, F=3, C=8, N=1, in=8, left pad 5: limits fwd 29826124, wrw 29826149
        {{2, 1, 1, 8, 8, {3, 2}, {8, 3}, {1, 2}, {29826124, 1}, {5, 1}, {59652248, 1}}, true, true},
        {{2, 1, 1, 8, 8, {3, 2}, {8, 3}, {1, 2}, {29826125, 1}, {5, 1}, {59652250, 1}}, false, true},
        {{2, 1, 1, 8, 8, {3, 2}, {8, 3}, {1, 2}, {29826149, 1}, {5, 1}, {59652298, 1}}, false, true},
        {{2, 1, 1, 8, 8, {3, 2}, {8, 3}, {1, 2}, {29826150, 1}, {5, 1}, {59652300, 1}}, false, false},
        // dim 0, F=16, C=8, N=1, in=8, left pad 0: limits fwd 5965222, wrw 5965231
        {{2, 1, 1, 8, 8, {16, 2}, {8, 3}, {1, 2}, {5965222, 1}, {0, 1}, {89478330, 1}}, true, true},
        {{2, 1, 1, 8, 8, {16, 2}, {8, 3}, {1, 2}, {5965223, 1}, {0, 1}, {89478345, 1}}, false, true},
        {{2, 1, 1, 8, 8, {16, 2}, {8, 3}, {1, 2}, {5965231, 1}, {0, 1}, {89478465, 1}}, false, true},
        {{2, 1, 1, 8, 8, {16, 2}, {8, 3}, {1, 2}, {5965232, 1}, {0, 1}, {89478480, 1}}, false, false},
        // dim 0, F=3, C=40, N=1, in=8, left pad 0: limits fwd 5965184, wrw 5965226
        {{2, 1, 1, 8, 40, {3, 2}, {8, 3}, {1, 2}, {5965184, 1}, {0, 1}, {11930368, 1}}, true, true},
        {{2, 1, 1, 8, 40, {3, 2}, {8, 3}, {1, 2}, {5965185, 1}, {0, 1}, {11930370, 1}}, false, true},
        {{2, 1, 1, 8, 40, {3, 2}, {8, 3}, {1, 2}, {5965226, 1}, {0, 1}, {11930452, 1}}, false, true},
        {{2, 1, 1, 8, 40, {3, 2}, {8, 3}, {1, 2}, {5965227, 1}, {0, 1}, {11930454, 1}}, false, false},
        // dim 0, F=3, C=8, N=1, in=8, left pad 1048576: limits fwd 29127063, wrw 29127105
        {{2, 1, 1, 8, 8, {3, 2}, {8, 3}, {1, 2}, {29127063, 1}, {1048576, 1}, {58254126, 1}}, true, true},
        {{2, 1, 1, 8, 8, {3, 2}, {8, 3}, {1, 2}, {29127064, 1}, {1048576, 1}, {58254128, 1}}, false, true},
        {{2, 1, 1, 8, 8, {3, 2}, {8, 3}, {1, 2}, {29127105, 1}, {1048576, 1}, {58254210, 1}}, false, true},
        {{2, 1, 1, 8, 8, {3, 2}, {8, 3}, {1, 2}, {29127106, 1}, {1048576, 1}, {58254212, 1}}, false, false},
        // dim 0, F=3, C=8, N=3, in=1048576, left pad 5: limits fwd 42991574, wrw 42991612
        {{2, 1, 3, 8, 8, {3, 2}, {1048576, 2}, {1, 2}, {42991574, 1}, {5, 1}, {85983148, 1}}, true, true},
        {{2, 1, 3, 8, 8, {3, 2}, {1048576, 2}, {1, 2}, {42991575, 1}, {5, 1}, {85983150, 1}}, false, true},
        {{2, 1, 3, 8, 8, {3, 2}, {1048576, 2}, {1, 2}, {42991612, 1}, {5, 1}, {85983224, 1}}, false, true},
        {{2, 1, 3, 8, 8, {3, 2}, {1048576, 2}, {1, 2}, {42991613, 1}, {5, 1}, {85983226, 1}}, false, false},
        // dim 1, F=2, C=8, N=1, in=8, left pad 0: limits fwd 268434984, wrw 268435392
        {{2, 1, 1, 8, 8, {2, 2}, {3, 8}, {2, 1}, {1, 268434984}, {1, 0}, {1, 268434984}}, true, true},
        {{2, 1, 1, 8, 8, {2, 2}, {3, 8}, {2, 1}, {1, 268434985}, {1, 0}, {1, 268434985}}, false, true},
        {{2, 1, 1, 8, 8, {2, 2}, {3, 8}, {2, 1}, {1, 268435392}, {1, 0}, {1, 268435392}}, false, true},
        {{2, 1, 1, 8, 8, {2, 2}, {3, 8}, {2, 1}, {1, 268435393}, {1, 0}, {1, 268435393}}, false, false},
        // dim 1, F=3, C=64, N=3, in=1048576, left pad 0: limits fwd 11010048, wrw 11010048
        {{2, 1, 3, 8, 64, {2, 3}, {2, 1048576}, {2, 1}, {1, 11010048}, {1, 0}, {1, 22020096}}, true, true},
        {{2, 1, 3, 8, 64, {2, 3}, {2, 1048576}, {2, 1}, {1, 11010049}, {1, 0}, {1, 22020098}}, false, false},
        // dim 1, F=16, C=8, N=1, in=8, left pad 0: limits fwd 17895666, wrw 17895692
        {{2, 1, 1, 8, 8, {2, 16}, {3, 8}, {2, 1}, {1, 17895666}, {1, 0}, {1, 268434990}}, true, true},
        {{2, 1, 1, 8, 8, {2, 16}, {3, 8}, {2, 1}, {1, 17895667}, {1, 0}, {1, 268435005}}, false, true},
        {{2, 1, 1, 8, 8, {2, 16}, {3, 8}, {2, 1}, {1, 17895692}, {1, 0}, {1, 268435380}}, false, true},
        {{2, 1, 1, 8, 8, {2, 16}, {3, 8}, {2, 1}, {1, 17895693}, {1, 0}, {1, 268435395}}, false, false},

        // ---- 3D
        // dim 0, F=2, C=8, N=1, in=8, left pad 0: limits fwd 29826081, wrw 29826145
        {{3, 1, 1, 8, 8, {2, 2, 2}, {8, 3, 3}, {1, 2, 2}, {29826081, 1, 1}, {0, 1, 1}, {29826081, 1, 1}}, true, true},
        {{3, 1, 1, 8, 8, {2, 2, 2}, {8, 3, 3}, {1, 2, 2}, {29826082, 1, 1}, {0, 1, 1}, {29826082, 1, 1}}, false, true},
        {{3, 1, 1, 8, 8, {2, 2, 2}, {8, 3, 3}, {1, 2, 2}, {29826145, 1, 1}, {0, 1, 1}, {29826145, 1, 1}}, false, true},
        {{3, 1, 1, 8, 8, {2, 2, 2}, {8, 3, 3}, {1, 2, 2}, {29826146, 1, 1}, {0, 1, 1}, {29826146, 1, 1}}, false, false},
        // dim 0, F=3, C=8, N=1, in=8, left pad 5: limits fwd 9942030, wrw 14913062
        {{3, 1, 1, 8, 8, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {9942030, 1, 1}, {5, 1, 1}, {19884060, 1, 1}}, true, true},
        {{3, 1, 1, 8, 8, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {9942031, 1, 1}, {5, 1, 1}, {19884062, 1, 1}}, false, true},
        {{3, 1, 1, 8, 8, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {14913062, 1, 1}, {5, 1, 1}, {29826124, 1, 1}}, false, true},
        {{3, 1, 1, 8, 8, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {14913063, 1, 1}, {5, 1, 1}, {29826126, 1, 1}}, false, false},
        // dim 0, F=16, C=8, N=1, in=8, left pad 0: limits fwd 1988405, wrw 1988409
        {{3, 1, 1, 8, 8, {16, 2, 2}, {8, 3, 3}, {1, 2, 2}, {1988405, 1, 1}, {0, 1, 1}, {29826075, 1, 1}}, true, true},
        {{3, 1, 1, 8, 8, {16, 2, 2}, {8, 3, 3}, {1, 2, 2}, {1988406, 1, 1}, {0, 1, 1}, {29826090, 1, 1}}, false, true},
        {{3, 1, 1, 8, 8, {16, 2, 2}, {8, 3, 3}, {1, 2, 2}, {1988409, 1, 1}, {0, 1, 1}, {29826135, 1, 1}}, false, true},
        {{3, 1, 1, 8, 8, {16, 2, 2}, {8, 3, 3}, {1, 2, 2}, {1988410, 1, 1}, {0, 1, 1}, {29826150, 1, 1}}, false, false},
        // dim 0, F=3, C=40, N=1, in=8, left pad 0: limits fwd 1988383, wrw 2982607
        {{3, 1, 1, 8, 40, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {1988383, 1, 1}, {0, 1, 1}, {3976766, 1, 1}}, true, true},
        {{3, 1, 1, 8, 40, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {1988384, 1, 1}, {0, 1, 1}, {3976768, 1, 1}}, false, true},
        {{3, 1, 1, 8, 40, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {2982607, 1, 1}, {0, 1, 1}, {5965214, 1, 1}}, false, true},
        {{3, 1, 1, 8, 40, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {2982608, 1, 1}, {0, 1, 1}, {5965216, 1, 1}}, false, false},
        // dim 0, F=3, C=8, N=1, in=8, left pad 1048576: limits fwd 9242976, wrw 13864496
        {{3, 1, 1, 8, 8, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {9242976, 1, 1}, {1048576, 1, 1}, {18485952, 1, 1}}, true, true},
        {{3, 1, 1, 8, 8, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {9242977, 1, 1}, {1048576, 1, 1}, {18485954, 1, 1}}, false, true},
        {{3, 1, 1, 8, 8, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {13864496, 1, 1}, {1048576, 1, 1}, {27728992, 1, 1}}, false, true},
        {{3, 1, 1, 8, 8, {3, 2, 2}, {8, 3, 3}, {1, 2, 2}, {13864497, 1, 1}, {1048576, 1, 1}, {27728994, 1, 1}}, false, false},
        // dim 0, F=3, C=8, N=3, in=1048576, left pad 5: limits fwd 20621974, wrw 30932985
        {{3, 1, 3, 8, 8, {3, 2, 2}, {1048576, 2, 2}, {1, 2, 2}, {20621974, 1, 1}, {5, 1, 1}, {41243948, 1, 1}}, true, true},
        {{3, 1, 3, 8, 8, {3, 2, 2}, {1048576, 2, 2}, {1, 2, 2}, {20621975, 1, 1}, {5, 1, 1}, {41243950, 1, 1}}, false, true},
        {{3, 1, 3, 8, 8, {3, 2, 2}, {1048576, 2, 2}, {1, 2, 2}, {30932985, 1, 1}, {5, 1, 1}, {61865970, 1, 1}}, false, true},
        {{3, 1, 3, 8, 8, {3, 2, 2}, {1048576, 2, 2}, {1, 2, 2}, {30932986, 1, 1}, {5, 1, 1}, {61865972, 1, 1}}, false, false},
        // dim 2, F=2, C=8, N=1, in=8, left pad 0: limits fwd 268434656, wrw 268435256
        {{3, 1, 1, 8, 8, {2, 2, 2}, {3, 3, 8}, {2, 2, 1}, {1, 1, 268434656}, {1, 1, 0}, {1, 1, 268434656}}, true, true},
        {{3, 1, 1, 8, 8, {2, 2, 2}, {3, 3, 8}, {2, 2, 1}, {1, 1, 268434657}, {1, 1, 0}, {1, 1, 268434657}}, false, true},
        {{3, 1, 1, 8, 8, {2, 2, 2}, {3, 3, 8}, {2, 2, 1}, {1, 1, 268435256}, {1, 1, 0}, {1, 1, 268435256}}, false, true},
        {{3, 1, 1, 8, 8, {2, 2, 2}, {3, 3, 8}, {2, 2, 1}, {1, 1, 268435257}, {1, 1, 0}, {1, 1, 268435257}}, false, false},
        // dim 2, F=3, C=64, N=3, in=1048576, left pad 0: limits fwd 4194304, wrw 4194304
        {{3, 1, 3, 8, 64, {2, 2, 3}, {2, 2, 1048576}, {2, 2, 1}, {1, 1, 4194304}, {1, 1, 0}, {1, 1, 8388608}}, true, true},
        {{3, 1, 3, 8, 64, {2, 2, 3}, {2, 2, 1048576}, {2, 2, 1}, {1, 1, 4194305}, {1, 1, 0}, {1, 1, 8388610}}, false, false},
        // dim 2, F=16, C=8, N=1, in=8, left pad 0: limits fwd 17895643, wrw 17895683
        {{3, 1, 1, 8, 8, {2, 2, 16}, {3, 3, 8}, {2, 2, 1}, {1, 1, 17895643}, {1, 1, 0}, {1, 1, 268434645}}, true, true},
        {{3, 1, 1, 8, 8, {2, 2, 16}, {3, 3, 8}, {2, 2, 1}, {1, 1, 17895644}, {1, 1, 0}, {1, 1, 268434660}}, false, true},
        {{3, 1, 1, 8, 8, {2, 2, 16}, {3, 3, 8}, {2, 2, 1}, {1, 1, 17895683}, {1, 1, 0}, {1, 1, 268435245}}, false, true},
        {{3, 1, 1, 8, 8, {2, 2, 16}, {3, 3, 8}, {2, 2, 1}, {1, 1, 17895684}, {1, 1, 0}, {1, 1, 268435260}}, false, false},
        // clang-format on
    };
    return cases;
}

} // namespace

// =============================================================================
// Forward A matrix: ConvFwdDataImplicitGemmInTransform against the generic chain
// =============================================================================

namespace {

template <index_t NDim>
using InputLayout = tuple_element_t<NDim - 1,
                                    Tuple<tensor_layout::convolution::NWGC,
                                          tensor_layout::convolution::NHWGC,
                                          tensor_layout::convolution::NDHWGC>>;

template <index_t NDim>
using OutputLayout = tuple_element_t<NDim - 1,
                                     Tuple<tensor_layout::convolution::NWGK,
                                           tensor_layout::convolution::NHWGK,
                                           tensor_layout::convolution::NDHWGK>>;

// Strides (in elements) of the packed input (N spatial... G C) and output (N spatial... G K)
// tensors of a problem, in the order of the G N C spatial... tensors of the device ops. The
// descriptors describe one group, so G only changes the strides: the channels of the other groups
// are interleaved.
template <index_t NDim, typename IndexType = index_t>
std::array<IndexType, NDim + 3> InputStrides(const ConvParam& c)
{
    return ToArray<IndexType, NDim + 3>(
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InputLayout<NDim>>(c)
            .GetStrides());
}

template <index_t NDim, typename IndexType = index_t>
std::array<IndexType, NDim + 3> OutputStrides(const ConvParam& c)
{
    return ToArray<IndexType, NDim + 3>(
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutputLayout<NDim>>(c)
            .GetStrides());
}

// The forward transformer of the device ops (the production code that builds the A descriptor)
// for a problem. `Custom` enables the custom transform; IndexType = long_index_t selects the 64-bit
// chain the device ops use for tensors above 2GB. The input and output tensors are packed NHWGC /
// NHWGK (G interleaved groups); the weight tensor only provides lengths.
template <index_t NDim, typename IndexType = index_t, bool Custom = false>
auto MakeFwdTransformer(const ConvParam& c)
{
    using Transformer = ck::tensor_operation::TransformConvFwdToGemm<
        NDim,
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        false, // SplitN
        float,
        float,
        1, // NumGroupsToMerge
        IndexType,
        false, // CTranspose
        Custom>;

    std::array<IndexType, NDim + 3> a_len{}, b_len{}, c_len{};
    a_len[0] = b_len[0] = c_len[0] = c.G_;
    a_len[1] = c_len[1] = c.N_;
    a_len[2] = b_len[2] = c.C_;
    b_len[1] = c_len[2] = c.K_;
    for(index_t d = 0; d < NDim; ++d)
    {
        a_len[3 + d] = c.input_spatial_lengths_[d];
        b_len[3 + d] = c.filter_spatial_lengths_[d];
        c_len[3 + d] = c.output_spatial_lengths_[d];
    }
    std::array<IndexType, NDim + 3> b_str{};
    b_str.fill(1);
    return Transformer{a_len,
                       InputStrides<NDim, IndexType>(c),
                       b_len,
                       b_str,
                       c_len,
                       OutputStrides<NDim, IndexType>(c),
                       ToArray<IndexType, NDim>(c.conv_filter_strides_),
                       ToArray<IndexType, NDim>(c.conv_filter_dilations_),
                       ToArray<IndexType, NDim>(c.input_left_pads_),
                       ToArray<IndexType, NDim>(c.input_right_pads_)};
}

// The generic A descriptor [M, K] (pad -> embed -> merge chain, no GEMM padding): M = (N, O_0..)
// and K = (F_0.., C)
template <index_t NDim, typename IndexType = index_t>
auto MakeGenericA(const ConvParam& c)
{
    return MakeFwdTransformer<NDim, IndexType, false>(c)
        .template MakeADescriptor_M_K<InputLayout<NDim>>();
}

// The A descriptor [M, K] of the custom transform, padded to multiples of m_per_block x k_per_block
// as the device op pads it (the padding rows / columns are invalid)
template <index_t NDim>
auto MakeCustomA(const ConvParam& c, index_t m_per_block, index_t k_per_block)
{
    return MakeFwdTransformer<NDim, index_t, true>(c).MakeADescriptorWithCustomTransform_M_K(
        m_per_block, k_per_block);
}

// The custom transform alone, with the padded GEMM lengths m_padded x k_padded
template <index_t NDim>
auto MakeCustomTransform(const ConvParam& c, index_t m_padded, index_t k_padded)
{
    return make_conv_fwd_data_in_transform<NDim>(c.N_,
                                                 c.C_,
                                                 ToMultiIndex<NDim>(c.input_spatial_lengths_),
                                                 ToMultiIndex<NDim>(c.filter_spatial_lengths_),
                                                 ToMultiIndex<NDim>(c.output_spatial_lengths_),
                                                 ToMultiIndex<NDim>(c.conv_filter_strides_),
                                                 ToMultiIndex<NDim>(c.conv_filter_dilations_),
                                                 ToMultiIndex<NDim>(c.input_left_pads_),
                                                 m_padded,
                                                 k_padded);
}

// Number of times the 32-bit computation of the offset of the lower index (batch, idx..., channel)
// overflows: a product or the running sum leaving the int32 range. The 32-bit arithmetic is
// emulated with 64 bit.
template <index_t NDim>
int CountOffsetOverflows(const std::array<index_t, NDim + 3>& strides,
                         int64_t batch,
                         const std::array<int64_t, NDim>& idx,
                         int64_t channel)
{
    int overflows  = 0;
    int64_t sum    = 0;
    const auto add = [&](int64_t index, int64_t stride) {
        const int64_t term = index * stride;
        overflows += std::llabs(term) > kMaxIndex;
        sum += term;
        overflows += std::llabs(sum) > kMaxIndex;
    };
    add(batch, strides[1]);
    for(index_t d = 0; d < NDim; ++d)
    {
        add(idx[d], strides[3 + d]);
    }
    add(channel, strides[2]);
    return overflows;
}

// Compares the custom A descriptor, padded to multiples of m_per_block x k_per_block as the device
// op pads it, with the generic chain (which has no GEMM padding) at every position, also in the
// padding, which the custom transform must report as invalid itself: validity and offset. The
// reference is the generic chain with 64-bit indices (the chain the device ops use for tensors
// above 2GB, independent of the 32-bit limits that the big problems exercise), and the 32-bit
// chain (the chain used for small tensors; only for problems where it is valid). The generic chains
// compared are the ones of the index types `GenericIndex`. Also checks that no offset term or sum
// of the 32-bit lower index leaves the range of int32, and that moving a coordinate agrees with
// building a new one.
template <index_t NDim, typename... GenericIndex>
void CompareCustomAWithGeneric(const ConvParam& cfg, index_t m_per_block, index_t k_per_block)
{
    const auto custom      = MakeCustomA<NDim>(cfg, m_per_block, k_per_block);
    const index_t m_padded = custom.GetLength(Number<0>{});
    const index_t k_padded = custom.GetLength(Number<1>{});
    EXPECT_EQ(m_padded, math::integer_least_multiple(ImplicitGemmM(cfg), m_per_block));
    EXPECT_EQ(k_padded, math::integer_least_multiple(ImplicitGemmK(cfg), k_per_block));

    const auto compare = [&](auto generic_index) {
        using Index        = decltype(generic_index);
        const auto generic = MakeGenericA<NDim, Index>(cfg);
        int64_t mismatches = 0;
        for(index_t m = 0; m < m_padded; ++m)
        {
            for(index_t k = 0; k < k_padded; ++k)
            {
                const auto index = make_tuple(m, k);
                const auto cc    = make_tensor_coordinate(custom, index);
                const auto gc    = make_tensor_coordinate<Index>(generic, index);
                const bool cv    = coordinate_has_valid_offset(custom, cc);
                const bool gv    = coordinate_has_valid_offset(generic, gc);
                if((cv != gv || (gv && int64_t{cc.GetOffset()} != int64_t{gc.GetOffset()})) &&
                   mismatches++ < 3)
                {
                    ADD_FAILURE() << "(" << m << "," << k << ") custom valid=" << cv
                                  << " offset=" << cc.GetOffset() << ", generic valid=" << gv
                                  << " offset=" << gc.GetOffset();
                }
            }
        }
    };
    (compare(GenericIndex{}), ...);

    // the transform alone gives the lower index, to emulate the 32-bit offset computation
    const auto transform = MakeCustomTransform<NDim>(cfg, m_padded, k_padded);
    const auto strides   = InputStrides<NDim>(cfg);
    int64_t overflows    = 0;
    for(index_t m = 0; m < m_padded; ++m)
    {
        for(index_t k = 0; k < k_padded; ++k)
        {
            MultiIndex<NDim + 2> low;
            transform.CalculateLowerIndex(low, make_tuple(m, k));
            std::array<int64_t, NDim> low_spatial{};
            static_for<0, NDim, 1>{}([&](auto d) { low_spatial[d] = low[Number<d + 1>{}]; });
            overflows += CountOffsetOverflows<NDim>(
                strides, low[Number<0>{}], low_spatial, low[Number<NDim + 1>{}]);
        }
    }
    EXPECT_EQ(overflows, 0) << "the offset computation overflows";
    EXPECT_EQ(CountCoordinateMoveMismatches(custom, 3000, 25), 0);
}

// Compares the custom (M, K) descriptor with the generic chains on every position, with the GEMM
// padded to M_ext x K_ext and without padding, and checks the moves that carry the merged indices.
void CheckForwardConvInputTransformMatchesGeneric(ConvParam cfg)
{
    for(const index_t groups : {1, 3})
    {
        cfg.G_ = groups;
        static_for<1, 4, 1>{}([&](auto ndim) {
            constexpr index_t NDim = decltype(ndim)::value;
            if(cfg.num_dim_spatial_ != NDim)
            {
                return;
            }
            const index_t M        = static_cast<index_t>(ImplicitGemmM(cfg));
            const index_t K        = static_cast<index_t>(ImplicitGemmK(cfg));
            const index_t out_size = static_cast<index_t>(M / cfg.N_); // Ho * Wo * ...

            // The transform is built with the padded GEMM lengths (as the GEMM padder would produce
            // them) and must report the padding rows / columns (m >= M or k >= K) as invalid
            // itself, with the same offsets and validity as the generic chain (which has no
            // padding) inside the real M x K.
            const index_t M_ext = M + out_size + 1;
            const index_t K_ext = K + cfg.C_ + 1;
            CompareCustomAWithGeneric<NDim, long_index_t, index_t>(cfg, M_ext, K_ext);
            CompareCustomAWithGeneric<NDim, long_index_t, index_t>(cfg, 1, 1);

            // Steps that carry the merged indices: one output row, the whole output, the channels
            // and a filter row times the channels
            const index_t out_last  = cfg.output_spatial_lengths_[NDim - 1];
            const index_t filt_last = cfg.filter_spatial_lengths_[NDim - 1];
            const index_t channels  = cfg.C_;
            EXPECT_EQ(CountCoordinateMoveMismatches(
                          MakeCustomA<NDim>(cfg, M_ext, K_ext),
                          5000,
                          2024,
                          {{out_last, out_size}, {4, 32, channels, filt_last * channels}}),
                      0);
        });
    }
}

// Checks the transform alone (lower index and validity) against the generic chain with 64-bit
// indices (MakeGenericA<NDim, long_index_t>), the chain the device ops use for tensors above 2GB.
// Used for index ranges where a 32-bit tensor offset would overflow, so the generic chain with
// 32-bit indices cannot be the reference; the 64-bit one uses 64-bit offsets and no magic division,
// so it is independent of the limits this check exercises. NOTE: this check is NOT exhaustive. The
// (m, k) space is up to 2^31 x 2^31, so only boundary values (around every divisor, the real M and
// K, and the largest supported index) and a fixed-seed random sample are tested.
void CheckForwardConvInputTransformLowerIndices(const ConvParam& cfg)
{
    static_for<1, 4, 1>{}([&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        if(cfg.num_dim_spatial_ != NDim)
        {
            return;
        }
        const int64_t M = ImplicitGemmM(cfg), K = ImplicitGemmK(cfg);
        ASSERT_LT(M, int64_t(1) << 31);
        const auto generic = MakeGenericA<NDim, long_index_t>(cfg);
        const auto transform =
            MakeCustomTransform<NDim>(cfg, static_cast<index_t>(M), static_cast<index_t>(K));

        // Upper-index boundaries around every divisor of the two merges, plus values up to the
        // largest dividend (2^31 - 1) magic division supports; random samples cover the rest.
        std::vector<int64_t> ms{0, 1, M - 1, M, kMaxIndex - 1, kMaxIndex};
        std::vector<int64_t> ks{0, 1, cfg.C_ - 1, cfg.C_, cfg.C_ + 1, K - 1, K, K + cfg.C_};
        int64_t prod = 1;
        for(index_t d = NDim - 1; d >= 0; --d)
        {
            prod *= cfg.output_spatial_lengths_[d];
            for(const int64_t delta : {-1, 0, 1})
            {
                ms.push_back(prod + delta);
            }
        }
        prod = cfg.C_;
        for(index_t d = NDim - 1; d >= 0; --d)
        {
            prod *= cfg.filter_spatial_lengths_[d];
            for(const int64_t delta : {-1, 0, 1})
            {
                ks.push_back(prod + delta);
            }
        }
        std::mt19937_64 rng(7);
        for(int i = 0; i < 2000; ++i)
        {
            ms.push_back(rng() % (kMaxIndex + 1));
            ks.push_back(rng() % (K + cfg.C_));
        }

        int64_t mismatches = 0;
        for(const int64_t m : ms)
        {
            for(const int64_t k : ks)
            {
                if(m < 0 || m > kMaxIndex || k < 0 || k > kMaxIndex)
                {
                    continue;
                }
                const auto up = make_tuple(static_cast<index_t>(m), static_cast<index_t>(k));

                // The generic chain has no GEMM padding: the rows / columns beyond the real M x K
                // are GEMM padding, always invalid, and their lower index is meaningless
                const bool real = m < M && k < K;
                bool valid      = false;
                if(real)
                {
                    // the lower index (n, i_0.., c) is the hidden index of the input tensor
                    // dimensions (hidden dimension 0 is the offset), before the pad: it is not
                    // clamped
                    const auto coordinate = make_tensor_coordinate<long_index_t>(generic, up);
                    const auto& hidden    = coordinate.GetHiddenIndex();
                    valid                 = coordinate_has_valid_offset(generic, coordinate);

                    // The transform computes lower indices in 32-bit arithmetic; a lower index
                    // outside the int32 range is not representable and therefore outside the
                    // supported range.
                    bool fits_int32 = true;
                    static_for<1, NDim + 3, 1>{}([&](auto i) {
                        fits_int32 &= hidden[i] >= kIndexMin && hidden[i] <= kMaxIndex;
                    });
                    if(!fits_int32)
                    {
                        continue;
                    }
                    MultiIndex<NDim + 2> low;
                    transform.CalculateLowerIndex(low, up);
                    static_for<0, NDim + 2, 1>{}(
                        [&](auto i) { mismatches += low[i] != hidden[i + Number<1>{}]; });
                }
                mismatches += transform.IsValidUpperIndexMappedToValidLowerIndex(up) != valid;
            }
        }
        EXPECT_EQ(mismatches, 0);
    });
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, ForwardConvInputTransformMatchesGeneric)
{
    for(const auto& c : ForwardCases())
    {
        CheckForwardConvInputTransformMatchesGeneric(c);
    }
}

TEST(TensorDescriptorFunctorsIntegration, ForwardConvInputTransformExtremeProblems)
{
    for(const auto& c : ExtremeForwardCases())
    {
        CheckForwardConvInputTransformLowerIndices(c);
    }
}

TEST(TensorDescriptorFunctorsIntegration, ForwardConvInputTransformWithFilters)
{
    for(const auto& [problem, split_k] : FilterSweepCases())
    {
        CheckForwardConvInputTransformMatchesGeneric(problem);
    }
}

// =============================================================================
// Backward data A matrix: ConvBwdDataImplicitGemmOutTransform against the generic chain
//
// The custom transform (index_t) is compared with the generic chain of Pad / Embed / Slice / Merge
// / Pad / UnMerge transforms (long_index_t). It has two modes: "separate padding" (outer Pad
// transforms report Ho / Wo out of range) and the opt-in "merged padding" (the transform reports
// that itself, with a poison like the forward transform); only the merged padding has a host check
// (IsMergedOutPaddingRangeSupported). The device op runs one GEMM per (ytilde, xtilde) group.
// =============================================================================

namespace {

template <typename IndexType, bool MergeOutPadding = false, bool SplitConvN = false>
auto MakeBwdDataTransformer(const ConvParam& c, index_t ytilde, index_t xtilde)
{
    namespace ctc   = ck::tensor_layout::convolution;
    using Transform = ck::tensor_operation::TransformConvBwdDataToGemm_v1<
        2,
        ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default,
        8,  // AK1
        8,  // BK1
        64, // MPerBlock
        64, // NPerBlock
        32, // KPerBlock
        true,
        true,
        ctc::NHWGK,
        ctc::GKYXC,
        ctc::NHWGC,
        SplitConvN,
        float,
        float,
        1,
        IndexType,
        false,
        MergeOutPadding>;

    const auto& in    = c.input_spatial_lengths_;
    const auto& f     = c.filter_spatial_lengths_;
    const auto& o     = c.output_spatial_lengths_;
    const IndexType G = c.G_, N = c.N_, K = c.K_, C = c.C_;
    const IndexType Hi = in[0], Wi = in[1], Y = f[0], X = f[1], Ho = o[0], Wo = o[1];

    // NHWGK output, GKYXC weight, NHWGC input
    const std::array<IndexType, 5> a_lengths{G, N, K, Ho, Wo};
    const auto a_strides = OutputStrides<2, IndexType>(c);
    const std::array<IndexType, 5> b_lengths{G, K, C, Y, X};
    const std::array<IndexType, 5> b_strides{K * Y * X * C, Y * X * C, 1, X * C, C};
    const std::array<IndexType, 5> c_lengths{G, N, C, Hi, Wi};
    const auto c_strides      = InputStrides<2, IndexType>(c);
    const auto conv_strides   = ToArray<IndexType, 2>(c.conv_filter_strides_);
    const auto conv_dilations = ToArray<IndexType, 2>(c.conv_filter_dilations_);
    const auto left_pads      = ToArray<IndexType, 2>(c.input_left_pads_);
    const auto right_pads     = ToArray<IndexType, 2>(c.input_right_pads_);
    const std::array<IndexType, 2> tildes{ytilde, xtilde};

    return Transform{a_lengths,
                     a_strides,
                     b_lengths,
                     b_strides,
                     c_lengths,
                     c_strides,
                     conv_strides,
                     conv_dilations,
                     left_pads,
                     right_pads,
                     tildes};
}

// The A descriptor [K0, M, K1] of one group
template <typename IndexType, bool MergeOutPadding = false, bool SplitConvN = false>
auto MakeBwdDataADescriptor(const ConvParam& c, index_t ytilde, index_t xtilde)
{
    return MakeBwdDataTransformer<IndexType, MergeOutPadding, SplitConvN>(c, ytilde, xtilde)
        .MakeADescriptor_AK0_M_AK1();
}

// Compares the A descriptor of the custom transform with the generic chain for every group of the
// problem, if the host check accepts the group (the merged padding transform is not meant to be
// used otherwise); fails on any mismatch. All the positions are compared if `exhaustive`, else the
// first rows / columns, the last rows / columns (where the GEMM padding is) and random ones, with
// every K1 index. Returns the number of groups the host check rejected.
template <bool MergePadding, bool SplitConvN = false>
int CheckBwdDataAllGroups(const ConvParam& c, bool exhaustive)
{
    int rejected  = 0;
    int compared  = 0;
    int64_t valid = 0;

    // one GEMM per group (ytilde, xtilde) that has filter taps, as in the device op
    const index_t sh = c.conv_filter_strides_[0], sw = c.conv_filter_strides_[1];
    const index_t dh = c.conv_filter_dilations_[0], dw = c.conv_filter_dilations_[1];
    const index_t y_tilde = sh / math::gcd(sh, dh);
    const index_t x_tilde = sw / math::gcd(sw, dw);
    for(index_t ytilde = 0; ytilde < y_tilde; ++ytilde)
    {
        for(index_t xtilde = 0; xtilde < x_tilde; ++xtilde)
        {
            if(ytilde >= c.filter_spatial_lengths_[0] || xtilde >= c.filter_spatial_lengths_[1])
            {
                continue;
            }
            const auto transformer =
                MakeBwdDataTransformer<index_t, MergePadding, SplitConvN>(c, ytilde, xtilde);
            const auto custom = transformer.MakeADescriptor_AK0_M_AK1();
            const index_t k0  = custom.GetLength(Number<0>{});
            const index_t m   = custom.GetLength(Number<1>{});
            const index_t k1  = custom.GetLength(Number<2>{});
            if(!transformer.IsMergedOutPaddingRangeSupported(m))
            {
                ++rejected;
                continue;
            }

            const auto generic =
                MakeBwdDataADescriptor<long_index_t, false, SplitConvN>(c, ytilde, xtilde);
            EXPECT_EQ(k0, static_cast<index_t>(generic.GetLength(Number<0>{})));
            EXPECT_EQ(m, static_cast<index_t>(generic.GetLength(Number<1>{})));
            EXPECT_EQ(k1, static_cast<index_t>(generic.GetLength(Number<2>{})));

            // The positions of a dimension: all of them, or the first `head`, the last `tail` (the
            // GEMM padding is at the end) and `random` random ones
            std::mt19937 rng(13);
            const auto positions = [&](index_t length, index_t head, index_t tail, int random) {
                std::vector<index_t> p;
                if(exhaustive)
                {
                    head = length;
                    tail = random = 0;
                }
                for(index_t i = 0; i < head && i < length; ++i)
                {
                    p.push_back(i);
                }
                for(index_t i = 0; i < tail && i < length; ++i)
                {
                    p.push_back(length - 1 - i);
                }
                for(int i = 0; i < random; ++i)
                {
                    p.push_back(static_cast<index_t>(rng() % length));
                }
                return p;
            };
            ++compared;
            valid += CompareDescriptors3D(custom,
                                          generic,
                                          positions(k0, 4, 4, 40),
                                          positions(m, 4, 160, 40),
                                          positions(k1, k1, 0, 0)); // every K1 index
            EXPECT_EQ(CountCoordinateMoveMismatches(custom, 3000, 21), 0);
        }
    }
    if(exhaustive && compared > 0)
    {
        EXPECT_GT(valid, 0) << "no valid position was compared";
    }
    return rejected;
}

// Checks the problem with separate padding, and with merged padding (the host check must accept
// it), for G = 1 (a packed tensor) and G = 3 (a tensor with interleaved groups: NHWGK / NHWGC)
void CheckBwdDataMatchesGeneric(ConvParam c)
{
    for(const index_t groups : {1, 3})
    {
        c.G_ = groups;
        CheckBwdDataAllGroups<false>(c, true);
        EXPECT_EQ(CheckBwdDataAllGroups<true>(c, true), 0);
    }
}

// True if the host check rejects the problem for a GemmM padded to `m_padded` rows. The descriptor
// is not built: for the biggest problems it would have more than 2^31 elements.
bool BwdDataRowsRejected(const ConvParam& c, int64_t m_padded)
{
    return !MakeBwdDataTransformer<index_t, true>(c, 0, 0).IsMergedOutPaddingRangeSupported(
        m_padded);
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, BackwardDataOutputTransformMatchesGeneric)
{
    for(const auto& c : BwdDataCases())
    {
        CheckBwdDataMatchesGeneric(c);
    }
}

TEST(TensorDescriptorFunctorsIntegration, BackwardDataMergedPaddingWithFilters)
{
    for(const auto& c : BwdDataSweepCases())
    {
        CheckBwdDataMatchesGeneric(c);
    }
}

TEST(TensorDescriptorFunctorsIntegration, BackwardData3dHasNoMergedPaddingLimit)
{
    // There is no 3D custom transform with merged padding: the 3D descriptors use the outer Pad
    // transforms, so the range check of the merged padding must not apply to them, even if the
    // opt-in flag is set (as the device op does for the wavelet direct load kernels) and the H
    // dimension is big enough for the 2D check to reject it.
    namespace ctc   = ck::tensor_layout::convolution;
    using Transform = ck::tensor_operation::TransformConvBwdDataToGemm_v1<
        3,
        ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default,
        8,
        8,
        64,
        64,
        32,
        true,
        true,
        ctc::NDHWGK,
        ctc::GKZYXC,
        ctc::NDHWGC,
        false,
        float,
        float,
        1,
        index_t,
        false,
        true>;

    // G N K Do Ho Wo output, G K C Z Y X weight and G N C Di Hi Wi input with Ho = Hi = 2^27
    const index_t hi = 1 << 27;
    const std::array<index_t, 6> a_lengths{1, 1, 8, 1, hi, 1};
    const std::array<index_t, 6> a_strides{8, 8 * hi, 1, 8 * hi, 8, 8};
    const std::array<index_t, 6> b_lengths{1, 8, 8, 1, 1, 1};
    const std::array<index_t, 6> b_strides{64, 64, 1, 8, 8, 8};
    const std::array<index_t, 6> c_lengths{1, 1, 8, 1, hi, 1};
    const std::array<index_t, 6> c_strides{8, 8 * hi, 1, 8 * hi, 8, 8};
    const std::array<index_t, 3> ones{1, 1, 1};
    const std::array<index_t, 3> zeros{0, 0, 0};
    const Transform transformer{a_lengths,
                                a_strides,
                                b_lengths,
                                b_strides,
                                c_lengths,
                                c_strides,
                                ones,
                                ones,
                                zeros,
                                zeros,
                                zeros};
    EXPECT_TRUE(transformer.IsMergedOutPaddingRangeSupported(index_t{1} << 27));
}

// =============================================================================
// Backward data merged padding with split N. The transform splits N when the output or the input
// tensor is bigger than 2GB: N_ is then the batch of one workgroup (a divisor of N, or 1) and the
// descriptors describe N_ batches. (The xdl v3 op that uses the merged padding instantiates the
// transform without split N today; this checks that the merged padding follows N_ if it is
// enabled.)
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, BackwardDataMergedPaddingSplitN)
{
    struct SplitCase
    {
        index_t n, hi, expected_n;
    };
    // 1x1 filter, stride 1, 8 channels, float tensors (4 bytes):
    //  N = 4, 4200 x 4200 : 2.3 GB, split in 2 workgroups of 2 batches
    //  N = 6, 3500 x 3500 : 2.3 GB, split in 2 workgroups of 3 batches
    //  N = 5, 4096 x 4096 : 2.7 GB, no divisor of 5 found, 1 batch per workgroup
    //  N = 4, 2048 x 2048 : 0.5 GB, no split
    const SplitCase cases[] = {{4, 4200, 2}, {6, 3500, 3}, {5, 4096, 1}, {4, 2048, 4}};
    for(const auto& sc : cases)
    {
        ConvParam c(2, 1, sc.n, 8, 8, {1, 1}, {sc.hi, sc.hi}, {1, 1}, {1, 1}, {0, 0}, {0, 0});

        for(const index_t groups : {1, 3})
        {
            c.G_ = groups;
            // with G groups the tensor has G times more elements, so more batches are split
            const auto transformer = MakeBwdDataTransformer<index_t, true, true>(c, 0, 0);
            if(c.G_ == 1)
            {
                EXPECT_EQ(transformer.N_, sc.expected_n);
            }
            else
            {
                EXPECT_LE(transformer.N_, sc.expected_n);
                EXPECT_EQ(sc.n % transformer.N_, 0);
            }

            EXPECT_EQ((CheckBwdDataAllGroups<true, true>(c, false)), 0);
        }
    }
}

// =============================================================================
// Backward weight B matrix: the forward transform on the input tensor, against the generic chain
//
// The B descriptor [K0, GemmN, K1] of the backward weight convolution is the same implicit im2col
// as the forward A matrix: GemmK = (n, o_0..) plays the role of M and GemmN = (f_0.., c) the role
// of K (GemmK is split into K0 x K1 and padded to K0PerBlock * K1 * BatchK, GemmN to NPerBlock).
// =============================================================================

namespace {

// The transform with small tiles (MPerBlock, NPerBlock, K1, K0PerBlock), so that the problems of
// the tests have GEMM padding; `CustomB` selects the custom B descriptor or the generic chain.
constexpr index_t kWrwNPerBlock  = 32;
constexpr index_t kWrwK1         = 8;
constexpr index_t kWrwK0PerBlock = 2;

template <index_t NDim, bool CustomB, typename IndexType = index_t>
using WrwTransform = ck::tensor_operation::TransformConvBwdWeightToGemmV2<
    NDim,
    32, // MPerBlock
    kWrwNPerBlock,
    kWrwK1,
    kWrwK0PerBlock,
    1, // NumGroupsToMerge
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,
    IndexType,
    false, // UseTdm
    CustomB>;

// The A, B and C descriptors of the transform with the split-K factor `batch_k`.
// `use_full_batch_kindex` is the V1 mode of the device op, whose K0 dimension covers all the K
// batches. With
// IndexType = long_index_t (and the generic chain, CustomB = false) this is the chain the device
// ops use for tensors above 2GB.
template <index_t NDim, bool CustomB, typename IndexType = index_t>
auto MakeBwdWeightDescriptors(const ConvParam& c,
                              index_t batch_k,
                              bool use_full_batch_kindex = false)
{
    const IndexType C = c.C_, K = c.K_;

    // N spatial G C input and N spatial G K output (the groups are interleaved: G = 1 is packed),
    // G K filter C weight
    const auto in_strides  = InputStrides<NDim, IndexType>(c);
    const auto out_strides = OutputStrides<NDim, IndexType>(c);
    std::array<IndexType, NDim + 3> wei_strides{K * C, C, 1};
    IndexType wei_s = C;
    for(index_t d = NDim - 1; d >= 0; --d)
    {
        wei_strides[3 + d] = wei_s;
        wei_s *= c.filter_spatial_lengths_[d];
    }

    return WrwTransform<NDim, CustomB, IndexType>::
        template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDim>(
            c.N_,
            c.K_,
            c.C_,
            ToArray<IndexType, NDim>(c.input_spatial_lengths_),
            ToArray<IndexType, NDim>(c.filter_spatial_lengths_),
            ToArray<IndexType, NDim>(c.output_spatial_lengths_),
            in_strides,
            wei_strides,
            out_strides,
            ToArray<IndexType, NDim>(c.conv_filter_strides_),
            ToArray<IndexType, NDim>(c.conv_filter_dilations_),
            ToArray<IndexType, NDim>(c.input_left_pads_),
            ToArray<IndexType, NDim>(c.input_right_pads_),
            batch_k,
            false,
            use_full_batch_kindex);
}

template <index_t NDim, bool CustomB, typename IndexType = index_t>
auto MakeBwdWeightBDescriptor(const ConvParam& c, index_t batch_k)
{
    return MakeBwdWeightDescriptors<NDim, CustomB, IndexType>(c, batch_k)[Number<1>{}];
}

// Compares the custom B descriptor with the generic chains of the index types `GenericIndex` on
// every position (long_index_t is the chain the device ops use for tensors above 2GB, index_t the
// one for small tensors, only valid for problems where 32-bit offsets fit), and checks the moves.
// `require_valid` asserts that the problem has valid elements (small problems may not have any).
template <index_t NDim, typename... GenericIndex>
void CompareBwdWeightB(const ConvParam& c, index_t batch_k, bool require_valid)
{
    const auto custom = MakeBwdWeightBDescriptor<NDim, true>(c, batch_k);

    const index_t k0 = custom.GetLength(Number<0>{});
    const index_t n  = custom.GetLength(Number<1>{});
    const index_t k1 = custom.GetLength(Number<2>{});
    std::vector<index_t> positions0(k0), positions1(n), positions2(k1);
    std::iota(positions0.begin(), positions0.end(), 0);
    std::iota(positions1.begin(), positions1.end(), 0);
    std::iota(positions2.begin(), positions2.end(), 0);

    const auto compare = [&](auto generic_index) {
        using Index        = decltype(generic_index);
        const auto generic = MakeBwdWeightBDescriptor<NDim, false, Index>(c, batch_k);
        EXPECT_EQ(k0, generic.GetLength(Number<0>{}));
        EXPECT_EQ(n, generic.GetLength(Number<1>{}));
        EXPECT_EQ(k1, generic.GetLength(Number<2>{}));
        const int64_t valid =
            CompareDescriptors3D<Index>(custom, generic, positions0, positions1, positions2);
        if(require_valid)
        {
            EXPECT_GT(valid, 0);
        }
    };
    (compare(GenericIndex{}), ...);
    EXPECT_EQ(CountCoordinateMoveMismatches(custom, 3000, 22), 0);
}

// Compares the custom B descriptor with the generic chains with 64-bit and with 32-bit indices on
// every position, for the problem with G = 1 (a packed tensor) and with G = 3 (a tensor with
// interleaved groups: the descriptors describe one group, so G only changes the strides)
void CheckBwdWeightBTransformMatchesGeneric(ConvParam c, index_t batch_k, bool require_valid = true)
{
    for(const index_t groups : {1, 3})
    {
        c.G_ = groups;
        static_for<1, 4, 1>{}([&](auto ndim) {
            if(c.num_dim_spatial_ == ndim)
            {
                CompareBwdWeightB<decltype(ndim)::value, long_index_t, index_t>(
                    c, batch_k, require_valid);
            }
        });
    }
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightInputTransformMatchesGeneric)
{
    for(const auto& [c, batch_k] : BwdWeightCases())
    {
        CheckBwdWeightBTransformMatchesGeneric(c, batch_k);
    }
}

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightWithFilters)
{
    for(const auto& [problem, split_k] : FilterSweepCases())
    {
        CheckBwdWeightBTransformMatchesGeneric(problem, split_k, false);
    }
}

// =============================================================================
// Backward weight: split-K offset hack with the custom B descriptor. When the hack is eligible
// (SplitKHackEligibility) the device op shifts the B pointer by ElementSpaceSize / k_batch for
// each K batch and the kernel indexes the first batch of the descriptor only. The custom B must
// take the same eligibility decision as the generic chain (so the custom transform does not
// enable or disable the hack) and, when the hack is on, the shifted view must give the same
// elements as the descriptor of the whole GemmK.
// =============================================================================

namespace {

// Returns whether the split-K hack is eligible
template <index_t NDim>
bool CheckSplitKHackWithCustomB(const ConvParam& c, index_t k_batch)
{
    // the descriptors of the device op (full K batch index)
    const auto custom  = MakeBwdWeightDescriptors<NDim, true>(c, k_batch, true);
    const auto generic = MakeBwdWeightDescriptors<NDim, false>(c, k_batch, true);

    using Eligibility = ck::tensor_operation::device::SplitKHackEligibility<
        NDim,
        tuple_element_t<NDim - 1,
                        Tuple<tensor_layout::convolution::NWGC,
                              tensor_layout::convolution::NHWGC,
                              tensor_layout::convolution::NDHWGC>>,
        tuple_element_t<NDim - 1,
                        Tuple<tensor_layout::convolution::GKXC,
                              tensor_layout::convolution::GKYXC,
                              tensor_layout::convolution::GKZYXC>>,
        tuple_element_t<NDim - 1,
                        Tuple<tensor_layout::convolution::NWGK,
                              tensor_layout::convolution::NHWGK,
                              tensor_layout::convolution::NDHWGK>>>;
    const auto out                 = ToArray<index_t, NDim>(c.output_spatial_lengths_);
    constexpr index_t k_block_size = kWrwK0PerBlock * kWrwK1;
    const bool eligible_custom     = Eligibility::Check(custom[Number<0>{}],
                                                    custom[Number<1>{}],
                                                    k_batch,
                                                    static_cast<index_t>(c.N_),
                                                    out,
                                                    k_block_size);
    const bool eligible_generic    = Eligibility::Check(generic[Number<0>{}],
                                                     generic[Number<1>{}],
                                                     k_batch,
                                                     static_cast<index_t>(c.N_),
                                                     out,
                                                     k_block_size);
    EXPECT_EQ(eligible_custom, eligible_generic)
        << "the custom B changes the eligibility of the split-K hack";
    if(!eligible_custom)
    {
        return false;
    }

    // The device op runs batch kb on the pointer shifted by kb * stride_b with the descriptor of
    // the first batch: it must see what the descriptor of all the batches has at K0 offset kb *
    // k0_batch.
    const auto& b          = custom[Number<1>{}];
    const index_t k0_total = b.GetLength(Number<0>{});
    const index_t n        = b.GetLength(Number<1>{});
    const index_t k1       = b.GetLength(Number<2>{});
    EXPECT_EQ(k0_total % k_batch, 0);
    const index_t k0_batch      = k0_total / k_batch;
    const long_index_t stride_b = b.GetElementSpaceSize() / k_batch;

    int64_t mismatches = 0;
    for(index_t kb = 0; kb < k_batch; ++kb)
    {
        for(index_t k0 = 0; k0 < k0_batch; ++k0)
        {
            for(index_t i1 = 0; i1 < n; ++i1)
            {
                for(index_t i2 = 0; i2 < k1; ++i2)
                {
                    const auto whole =
                        make_tensor_coordinate(b, make_multi_index(kb * k0_batch + k0, i1, i2));
                    const auto first = make_tensor_coordinate(b, make_multi_index(k0, i1, i2));
                    const bool whole_valid = coordinate_has_valid_offset(b, whole);
                    const bool first_valid = coordinate_has_valid_offset(b, first);
                    if(whole_valid != first_valid ||
                       (whole_valid && long_index_t{whole.GetOffset()} !=
                                           kb * stride_b + long_index_t{first.GetOffset()}))
                    {
                        if(mismatches++ < 3)
                        {
                            ADD_FAILURE() << "batch " << kb << " (" << k0 << "," << i1 << "," << i2
                                          << "): whole valid=" << whole_valid
                                          << " offset=" << whole.GetOffset()
                                          << ", shifted valid=" << first_valid
                                          << " offset=" << first.GetOffset() << " + " << kb << " * "
                                          << stride_b;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0);
    return true;
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightSplitKHackWithCustomB)
{
    struct Case2d
    {
        ConvParam c;
        index_t k_batch;
        bool eligible;
    };
    // {2, G, N, K = 32, C, filter, input, stride, dilation, left pad, right pad}, split-K, eligible
    const Case2d cases[] = {
        // 1x1, stride 1, pad 0: GemmK = N * Hi * Wi, the input is read as it is stored
        {{2, 1, 4, 32, 32, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 2, true},
        {{2, 1, 4, 32, 32, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 4, true},
        {{2, 1, 6, 32, 32, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 3, true},
        {{2, 1, 2, 32, 64, {1, 1}, {8, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 2, true},
        // 2x2 stride 2 pad 0: the patches do not overlap, the im2col is a permutation of the input
        {{2, 1, 4, 32, 8, {2, 2}, {8, 8}, {2, 2}, {1, 1}, {0, 0}, {0, 0}}, 2, true},
        // not eligible: overlapping filter (3x3 with a pad of 1), strided 1x1, C not a multiple of
        // the GemmN block, groups, N not divisible by the K batch, GemmK padded, one K batch
        {{2, 1, 4, 32, 32, {3, 3}, {4, 4}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, 2, false},
        {{2, 1, 4, 32, 32, {1, 1}, {8, 8}, {2, 2}, {1, 1}, {0, 0}, {0, 0}}, 2, false},
        {{2, 1, 4, 32, 8, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 2, false},
        {{2, 3, 4, 32, 32, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 2, false},
        {{2, 1, 3, 32, 32, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 2, false},
        {{2, 1, 4, 32, 32, {1, 1}, {5, 5}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 2, false},
        {{2, 1, 4, 32, 32, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 1, false},
    };
    for(const auto& tc : cases)
    {
        EXPECT_EQ(CheckSplitKHackWithCustomB<2>(tc.c, tc.k_batch), tc.eligible);
    }

    // 1D and 3D, 1x1 stride 1 pad 0
    const ConvParam c1{1, 1, 4, 32, 32, {1}, {16}, {1}, {1}, {0}, {0}};
    EXPECT_TRUE(CheckSplitKHackWithCustomB<1>(c1, 2));
    EXPECT_TRUE(CheckSplitKHackWithCustomB<1>(c1, 4));
    EXPECT_FALSE(CheckSplitKHackWithCustomB<1>(
        ConvParam{1, 1, 4, 32, 32, {3}, {16}, {1}, {1}, {1}, {1}}, 2));

    const ConvParam c3{
        3, 1, 4, 32, 32, {1, 1, 1}, {2, 2, 4}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    EXPECT_TRUE(CheckSplitKHackWithCustomB<3>(c3, 2));
    EXPECT_TRUE(CheckSplitKHackWithCustomB<3>(c3, 4));
}

// =============================================================================
// Host checks: limits of the poison and of the 32-bit offsets
//
// The custom transforms report the GEMM padding (rows m >= M, columns k >= K) as invalid by adding
// a "poison" to the first spatial lower coordinate, so that its unsigned range check fails (twice
// the poison for a padding row and column). This only works while a poisoned coordinate does not
// wrap around into the input, and the 32-bit offset (coordinate * stride) of the padding positions
// must not overflow either. The ops check this on the host before using the custom transforms:
//   * IsConvInputTransformRangeSupported (the forward A and backward weight B matrix), with a
//     wrapper in TransformConvFwdToGemm / TransformConvBwdWeightToGemmV2 that reads the problem
//     from the members of the transform,
//   * IsMergedOutPaddingRangeSupported (the backward data A matrix, merged padding).
// An accepted problem must be exact: the tests find the limits and compare every padding position
// of the accepted problems closest to them with the generic chain with 64-bit indices.
// =============================================================================

namespace {

// A 1D problem (N = C = 1, 1x1 filter, stride 1) with the given input, dilation and pads
ConvParam Conv1d(int64_t in, int64_t dilation, int64_t pad_left, int64_t pad_right)
{
    return ConvParam(1,
                     1,
                     1,
                     8,
                     1,
                     std::vector<long_index_t>{1},
                     std::vector<long_index_t>{in},
                     std::vector<long_index_t>{1},
                     std::vector<long_index_t>{dilation},
                     std::vector<long_index_t>{pad_left},
                     std::vector<long_index_t>{pad_right});
}

constexpr long_index_t kMaxWrap = (long_index_t{1} << 31) - 1;

} // namespace

// The formulas of the host checks, pinned one term at a time with hand picked numbers. The
// LimitCases rows below test whole problems in NHWGC, where the offset bound always binds before
// the poison bound, so the poison bound alone is only tested here.

TEST(TensorDescriptorFunctorsIntegration, ConvPaddingPoisonValue)
{
    // The poison of the forward transform is the smallest value that moves the smallest first
    // spatial coordinate (-left_pad) to the length of the dimension: length + max(0, left_pad).
    const auto poison = [](index_t length, index_t left_pad) {
        const ConvParam cfg = Conv1d(length, 1, left_pad, 0);
        return MakeCustomTransform<1>(cfg, static_cast<index_t>(cfg.output_spatial_lengths_[0]), 1)
            .Poison();
    };
    EXPECT_EQ(poison(8, 0), 8u);
    EXPECT_EQ(poison(8, 5), 13u);
}

TEST(TensorDescriptorFunctorsIntegration, ConvPaddingPoisonRange)
{
    // IsConvPaddingPoisonRangeSupported: the largest coordinate + 2 poisons + the distance to the
    // smallest coordinate (a kernel moves a coordinate by such a difference) must fit in 31 bits
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, 0, kMaxWrap - 16));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(8, 0, kMaxWrap - 15));
    // a left pad increases the poison (13) and the range of the coordinate (5 below 0)
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, -5, kMaxWrap - 31));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(8, -5, kMaxWrap - 30));
    // a negative left pad (positive lower index) does not
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, 5, kMaxWrap - 16));
    // a big input and a big pad
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(1 << 28, -(1 << 27), 0));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(1 << 30, -(1 << 29), 0));
}

TEST(TensorDescriptorFunctorsIntegration, ConvInputOffsetRange)
{
    // The offset terms of IsConvInputTransformRangeSupported<1> for a tall input (the poison is
    // the length): N, C, input, filter, output, stride, dilation, left pad, m_padded, k_padded,
    // N stride, spatial stride, C stride
    const auto offset_supported = [](index_t length, index_t spatial_stride) {
        return IsConvInputTransformRangeSupported<1>(
            1, 1, {length}, {1}, {length}, {1}, {1}, {0}, length + 1, 2, 1, {spatial_stride}, 1);
    };
    EXPECT_TRUE(offset_supported(1 << 26, 8));
    EXPECT_FALSE(offset_supported(1 << 27, 8));
    EXPECT_TRUE(offset_supported(1 << 28, 1));
    EXPECT_FALSE(offset_supported(1 << 30, 1));

    // The same check through the backward weight transform, which reads the problem from its
    // arguments: input, filter, output, dilation, left pad and the strides of the spatial
    // dimensions of the input (N = C = 1, stride 1, GemmK padded by 1 and GemmN by 64 columns)
    const auto wrw_supported = [](std::array<index_t, 2> in,
                                  std::array<index_t, 2> filter,
                                  std::array<index_t, 2> out,
                                  std::array<index_t, 2> dilation,
                                  std::array<index_t, 2> pad,
                                  std::array<index_t, 2> spatial_stride) {
        const std::array<index_t, 5> input_strides{1, 1, 1, spatial_stride[0], spatial_stride[1]};
        return WrwTransform<2, true>::IsCustomTransformPaddingRangeSupported(1,
                                                                             1,
                                                                             in,
                                                                             filter,
                                                                             out,
                                                                             input_strides,
                                                                             {1, 1},
                                                                             dilation,
                                                                             pad,
                                                                             out[0] * out[1] + 1,
                                                                             filter[0] * filter[1] +
                                                                                 64);
    };
    // big inputs and pads are fine (unit strides)
    EXPECT_TRUE(wrw_supported({1 << 29, 1}, {1, 1}, {1 << 29, 1}, {1, 1}, {0, 0}, {1, 1}));
    EXPECT_TRUE(wrw_supported({1 << 29, 1}, {1, 1}, {(1 << 29) + 1, 1}, {1, 1}, {1, 0}, {1, 1}));
    // offset of the poisoned coordinate: about 3 * length * stride must stay below 2^31
    EXPECT_TRUE(wrw_supported({1 << 26, 1}, {1, 1}, {1 << 26, 1}, {1, 1}, {0, 0}, {8, 1}));
    EXPECT_FALSE(wrw_supported({1 << 27, 1}, {1, 1}, {1 << 27, 1}, {1, 1}, {0, 0}, {8, 1}));
    // ... also with a left pad, which increases the poison (the range of the coordinate is from
    // -left_pad to 2 poisons + the largest one: about 2^28)
    EXPECT_TRUE(wrw_supported(
        {1 << 26, 1}, {1, 1}, {(1 << 26) + (1 << 26), 1}, {1, 1}, {1 << 26, 0}, {4, 1}));
    EXPECT_FALSE(wrw_supported(
        {1 << 26, 1}, {1, 1}, {(1 << 26) + (1 << 26), 1}, {1, 1}, {1 << 26, 0}, {8, 1}));
}

TEST(TensorDescriptorFunctorsIntegration, ConvPaddingStaysInvalidAtHugeSizes)
{
    // What the host check protects: the padding position (m = M, k = k_pad) of an input with a 1x1
    // filter has m >= M and k >= K, so both poisons are added: 2 * poison, with
    // poison = length + left_pad. It must stay invalid. The transform is only evaluated if the
    // host check accepts the problem (and not the whole descriptor, whose element count would
    // overflow an int). `supported` returns the result of the host check.
    const auto padding_valid = [](const ConvParam& cfg, index_t k_pad, bool& supported) {
        const index_t M = cfg.output_spatial_lengths_[0];
        supported =
            MakeFwdTransformer<1, index_t, true>(cfg).IsCustomTransformPaddingRangeSupported(M + 1,
                                                                                             192);
        if(!supported)
        {
            return false;
        }
        return MakeCustomTransform<1>(cfg, M + 1, 192)
            .IsValidUpperIndexMappedToValidLowerIndex(make_tuple(M, k_pad));
    };

    // a big first spatial length, with and without a left pad, and huge left pads on a small
    // input: all of them keep the padding invalid
    for(const index_t pad : {0, 1})
    {
        bool supported = false;
        EXPECT_FALSE(padding_valid(Conv1d(1 << 29, 1, pad, 0), 1, supported));
        EXPECT_TRUE(supported);
    }
    for(const index_t pad : {(1 << 29) - 8, (1 << 29) + 5})
    {
        for(const index_t right_pad : {0, 5})
        {
            bool supported = false;
            EXPECT_FALSE(padding_valid(Conv1d(8, 1, pad, right_pad), 1, supported));
            EXPECT_TRUE(supported);
        }
    }

    // upper side: a 1x1 filter with a huge dilation. The padded K columns (k >= K) get f0 > 0 and
    // therefore a lower coordinate f0 * dilation. With the dilation 2^28 - 1 and f0 = 16 the
    // coordinate is 2^32 - 16, and the two poisons (2 * 8) would wrap it to 0, inside the input:
    // the padding would be read as real data. The host check rejects this problem.
    {
        bool supported = false;
        EXPECT_FALSE(padding_valid(Conv1d(8, 1 << 20, 0, 0), 16, supported));
        EXPECT_TRUE(supported);

        supported = true;
        padding_valid(Conv1d(8, (1 << 28) - 1, 0, 0), 16, supported);
        EXPECT_FALSE(supported);
    }
    {
        bool supported = false;
        EXPECT_FALSE(padding_valid(Conv1d(8, 1 << 20, 0, 0), 128, supported));
        EXPECT_TRUE(supported);
    }
}

namespace {

// Block sizes of the GEMM padding in the tests of the host checks. The padding changes the limits
// (the padded columns continue the filter index), so these are arbitrary but fixed; the device ops
// take the padded sizes from their descriptors.
constexpr int64_t kFwdMPerBlock = 256;
constexpr int64_t kFwdKPerBlock = 64;

// Sizes {m, k} of the padded im2col matrix (the GEMM padding of the custom transforms). The
// forward A matrix has the rows (n, o_0..) padded to MPerBlock and the columns (f_0.., c) padded to
// KPerBlock.
template <index_t NDim>
std::pair<int64_t, int64_t> FwdPaddedGemm(const ConvParam& c)
{
    return {math::integer_least_multiple(ImplicitGemmM(c), kFwdMPerBlock),
            math::integer_least_multiple(ImplicitGemmK(c), kFwdKPerBlock)};
}

// The rows of the backward weight B matrix are GemmK, padded to K0PerBlock * K1 * BatchK, and the
// columns are GemmN, padded to NPerBlock
template <index_t NDim>
std::pair<int64_t, int64_t> WrwPaddedGemm(const ConvParam& c, index_t batch_k)
{
    return {math::integer_least_multiple(ImplicitGemmM(c), kWrwK0PerBlock * kWrwK1 * batch_k),
            math::integer_least_multiple(ImplicitGemmK(c), kWrwNPerBlock)};
}

// The host checks assume a non empty output and an input tensor addressable with 32-bit offsets
template <index_t NDim>
bool IsUsableForHostCheck(const ConvParam& c)
{
    if(*std::min_element(c.output_spatial_lengths_.begin(), c.output_spatial_lengths_.end()) < 1)
    {
        ADD_FAILURE() << "the test problem has an empty output";
        return false;
    }
    if(InputStrides<NDim, long_index_t>(c)[1] * c.N_ > kMaxIndex)
    {
        ADD_FAILURE() << "the test input is too big";
        return false;
    }
    return true;
}

// The host check tests compare an accepted problem with the generic chain at every position of its
// padded GEMM only up to this size (m_padded * k_padded); bigger problems only check the answer of
// the host check
constexpr int64_t kMaxComparedPositions = int64_t{1} << 22;

// Checks that the host check of the forward A matrix (the check of the transform, which reads the
// problem from its members) gives `expected_supported`. If the problem is accepted (and small
// enough), every position of the padded GEMM is compared with the generic chain with 64-bit
// indices.
void CheckConvInputRangeLimit(const ConvParam& cfg, bool expected_supported)
{
    static_for<1, 4, 1>{}([&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        if(cfg.num_dim_spatial_ != NDim)
        {
            return;
        }
        if(!IsUsableForHostCheck<NDim>(cfg))
        {
            return;
        }
        const auto [m_padded, k_padded] = FwdPaddedGemm<NDim>(cfg);
        const auto transformer          = MakeFwdTransformer<NDim, index_t, true>(cfg);
        const bool supported            = transformer.IsCustomTransformPaddingRangeSupported(
            static_cast<index_t>(m_padded), static_cast<index_t>(k_padded));
        EXPECT_EQ(supported, expected_supported);
        if(!supported || m_padded * k_padded > kMaxComparedPositions)
        {
            return;
        }

        CompareCustomAWithGeneric<NDim, long_index_t>(cfg, kFwdMPerBlock, kFwdKPerBlock);
    });
}

// The same for the backward weight B matrix, with the host check of its transform.
void CheckBwdWeightBRangeLimit(const ConvParam& c, bool expected_supported)
{
    static_for<1, 4, 1>{}([&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        if(c.num_dim_spatial_ != NDim)
        {
            return;
        }
        constexpr index_t batch_k = 1; // no split-K in the limit tests
        if(!IsUsableForHostCheck<NDim>(c))
        {
            return;
        }
        const auto [m_padded, k_padded] = WrwPaddedGemm<NDim>(c, batch_k);
        const bool supported = WrwTransform<NDim, true>::IsCustomTransformPaddingRangeSupported(
            c.N_,
            c.C_,
            ToArray<index_t, NDim>(c.input_spatial_lengths_),
            ToArray<index_t, NDim>(c.filter_spatial_lengths_),
            ToArray<index_t, NDim>(c.output_spatial_lengths_),
            InputStrides<NDim>(c),
            ToArray<index_t, NDim>(c.conv_filter_strides_),
            ToArray<index_t, NDim>(c.conv_filter_dilations_),
            ToArray<index_t, NDim>(c.input_left_pads_),
            static_cast<index_t>(m_padded),
            static_cast<index_t>(k_padded));
        EXPECT_EQ(supported, expected_supported);
        if(!supported || m_padded * k_padded > kMaxComparedPositions)
        {
            return;
        }

        const auto custom = MakeBwdWeightBDescriptor<NDim, true>(c, batch_k);
        EXPECT_EQ(int64_t{custom.GetLength(Number<0>{})} * custom.GetLength(Number<2>{}), m_padded);
        EXPECT_EQ(int64_t{custom.GetLength(Number<1>{})}, k_padded);
        CompareBwdWeightB<NDim, long_index_t>(c, batch_k, false);
    });
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, ForwardConvInputRangeLimits)
{
    for(const auto& [problem, fwd_supported, wrw_supported] : LimitCases())
    {
        CheckConvInputRangeLimit(problem, fwd_supported);
    }
}

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightRangeLimits)
{
    for(const auto& [problem, fwd_supported, wrw_supported] : LimitCases())
    {
        CheckBwdWeightBRangeLimit(problem, wrw_supported);
    }
}

// =============================================================================
// Backward data: limits of the merged padding (IsMergedOutPaddingRangeSupported)
//
// The poison is added to the Ho coordinate: Ho + (the largest ydot, including the ydot of the
// padded GemmK columns, times the ydot step). The host check must accept ordinary problems and
// reject the ones where a poisoned coordinate or its offset (coordinate * stride) is out of range.
// Every accepted problem is compared with the generic chain on the first and last rows / columns,
// where the GEMM padding is, and on random positions: a wrap around or an overflow of the 32-bit
// offset shows up as a mismatch. The bound of the host check is conservative: it also rejects some
// problems that would work (for instance the biggest left pads and wide outputs below).
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, BackwardDataMergedPaddingRangeLimit)
{
    // 1. a tall output: Ho = Hi, 1x1 filter, stride 1
    for(const int log_hi : {10, 20, 24})
    {
        const int rejected = CheckBwdDataAllGroups<true>(
            ConvParam{2, 1, 1, 8, 8, {1, 1}, {1 << log_hi, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
            false);
        EXPECT_EQ(rejected, 0) << "Hi = 2^" << log_hi;
    }
    {
        // around the limit of the offset of a poisoned coordinate, (Ho + 2 poisons) * HoStride_.
        // Ho is not a multiple of the block size, so that there are padding rows as well as
        // padding columns (K = 8 is padded to the block size too): two poisons.
        int accepted = 0;
        for(int step = 0; step <= 15; ++step)
        {
            const auto c = ConvParam{2,
                                     1,
                                     1,
                                     8,
                                     8,
                                     {1, 1},
                                     {(1 << 25) + step * (1 << 21) + 5, 1},
                                     {1, 1},
                                     {1, 1},
                                     {0, 0},
                                     {0, 0}};
            accepted += CheckBwdDataAllGroups<true>(c, false) == 0;
        }
        EXPECT_GT(accepted, 0);
    }
    // above the limit: a poisoned Ho coordinate times HoStride_ would pass 2^31
    EXPECT_TRUE(BwdDataRowsRejected(
        ConvParam{2, 1, 1, 8, 8, {1, 1}, {1 << 27, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 1 << 27));

    // 2. big dilations: the filter taps and the padded columns reach far outside the tensor. The
    // problems just below the limit of the host check must be accepted (and exact), the ones just
    // above it rejected.
    for(const auto& [c, accepted] : BwdDataDilationLimitCases())
    {
        EXPECT_EQ(CheckBwdDataAllGroups<true>(c, false) == 0, accepted);
    }
}

TEST(TensorDescriptorFunctorsIntegration, BackwardDataMergedPaddingRangeLimitMoreCases)
{
    // 1. groups: the strides of the output tensor (and so the offsets) depend on G, a tall output
    // up to the limit
    for(const index_t groups : {1, 3})
    {
        int accepted = 0;
        for(const int log_hi : {10, 20, 22, 23, 24})
        {
            const auto c = ConvParam{
                2, groups, 1, 8, 8, {1, 1}, {(1 << log_hi) + 5, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
            accepted += CheckBwdDataAllGroups<true>(c, false) == 0;
        }
        EXPECT_GE(accepted, 3) << "G=" << groups;
    }

    // 2. strides 1, 2 and 3 with filters 3 and 4 (several (ytilde, xtilde) groups)
    for(const index_t stride : {1, 2, 3})
    {
        for(const index_t y : {3, 4})
        {
            for(const int log_hi : {10, 20})
            {
                const auto c       = ConvParam{2,
                                         1,
                                         1,
                                         8,
                                         8,
                                               {y, 1},
                                               {(1 << log_hi) + 5, 1},
                                               {stride, 1},
                                               {1, 1},
                                               {1, 0},
                                               {1, 0}};
                const int rejected = CheckBwdDataAllGroups<true>(c, false);
                EXPECT_EQ(rejected, 0) << "stride=" << stride << " Y=" << y;
            }
        }
    }

    // 3. the W dimension: Ho = 1 and a wide output. The poison is on Ho only, Wo is a plain
    // coordinate, but its stride and the batch stride (= Wo * K) grow with it.
    for(const int log_wi : {10, 20, 24})
    {
        const int rejected = CheckBwdDataAllGroups<true>(
            ConvParam{2, 1, 1, 8, 8, {1, 1}, {1, 1 << log_wi}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
            false);
        // the host check is conservative for the biggest ones (the padding of the GemmK adds up to
        // GemmKPerBlock ydot steps to the poison, which is multiplied by the huge H stride)
        EXPECT_TRUE(rejected == 0 || log_wi > 20) << "Wi = 2^" << log_wi;
    }
    // the output has 2^30 elements: the padding rows (n = N) would be at an offset of 2^30 and the
    // other terms add up to more than 2^31
    EXPECT_TRUE(BwdDataRowsRejected(
        ConvParam{2, 1, 1, 8, 8, {1, 1}, {1, 1 << 27}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}, 1 << 27));
    // 4. big left pads: Ho is about the left pad. Only the first 2^3 rows of such a problem are
    // real (the rest of the output is padding), and the host check bounds the batch of the padding
    // rows by the smallest possible number of real rows, so it rejects the biggest pads.
    for(const int log_pad : {10, 16, 20, 23})
    {
        const int rejected = CheckBwdDataAllGroups<true>(
            ConvParam{2, 1, 1, 8, 8, {3, 1}, {8, 1}, {1, 1}, {1, 1}, {1 << log_pad, 0}, {0, 0}},
            false);
        EXPECT_TRUE(rejected == 0 || log_pad > 20) << "left pad = 2^" << log_pad;
    }
}
