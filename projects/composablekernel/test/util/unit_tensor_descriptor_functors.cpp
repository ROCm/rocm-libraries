// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_data_to_gemm_v1.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
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

namespace {

template <index_t NDim>
struct ForwardConvCase
{
    index_t N, C;
    std::array<index_t, NDim> In, Filter, Stride, Dilation, PadLeft, PadRight;
    // Number of groups of the NHWGC input tensor. The descriptor describes one group, so the
    // group only changes the strides: the channels of the other groups are interleaved.
    index_t G = 1;
};

// Input descriptor (N, spatial..., C) of one group of an N spatial G C tensor
// Moving a coordinate (the UpdateLowerIndex path the kernels use to walk the descriptor) must agree
// with building the coordinate from scratch: same offset, same validity. The walk is random with a
// fixed seed and the steps go forward and backward, change several dimensions at once and include
// jumps across the whole length of a dimension (between the smallest and the largest lower
// coordinates). It is not exhaustive. Returns the number of mismatches.
template <typename Desc>
int64_t CountCoordinateMoveMismatches(const Desc& desc, int steps, uint32_t seed)
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
    const auto pick = [&](index_t length) {
        const index_t options[] = {0,
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
        return options[rng() % (sizeof(options) / sizeof(options[0]))];
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
            delta[i] = pick(lengths[i]);
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

template <index_t NDim>
auto MakeInputDescriptor(const ForwardConvCase<NDim>& cfg,
                         const std::array<index_t, NDim + 2>* strides_override = nullptr)
{
    std::array<index_t, NDim + 2> lengths{};
    std::array<index_t, NDim + 2> strides{};
    lengths[0]        = cfg.N;
    lengths[NDim + 1] = cfg.C;
    strides[NDim + 1] = 1;
    index_t stride    = cfg.G * cfg.C;
    for(index_t d = NDim - 1; d >= 0; --d)
    {
        lengths[1 + d] = cfg.In[d];
        strides[1 + d] = stride;
        stride *= cfg.In[d];
    }
    strides[0] = stride;
    if(strides_override != nullptr)
    {
        strides = *strides_override;
    }
    if constexpr(NDim == 1)
        return make_naive_tensor_descriptor(make_tuple(lengths[0], lengths[1], lengths[2]),
                                            make_tuple(strides[0], strides[1], strides[2]));
    else if constexpr(NDim == 2)
        return make_naive_tensor_descriptor(
            make_tuple(lengths[0], lengths[1], lengths[2], lengths[3]),
            make_tuple(strides[0], strides[1], strides[2], strides[3]));
    else
        return make_naive_tensor_descriptor(
            make_tuple(lengths[0], lengths[1], lengths[2], lengths[3], lengths[4]),
            make_tuple(strides[0], strides[1], strides[2], strides[3], strides[4]));
}

template <index_t NDim>
auto ToMultiIndex(const std::array<index_t, NDim>& a)
{
    MultiIndex<NDim> r;
    static_for<0, NDim, 1>{}([&](auto i) { r(i) = a[i]; });
    return r;
}

template <index_t NDim>
std::array<int64_t, NDim> OutputLengths(const ForwardConvCase<NDim>& cfg)
{
    std::array<int64_t, NDim> out{};
    for(index_t d = 0; d < NDim; ++d)
    {
        out[d] = (int64_t(cfg.In[d]) + cfg.PadLeft[d] + cfg.PadRight[d] -
                  int64_t(cfg.Dilation[d]) * (cfg.Filter[d] - 1) - 1) /
                     cfg.Stride[d] +
                 1;
    }
    return out;
}

template <index_t NDim>
std::string Describe(const ForwardConvCase<NDim>& cfg)
{
    const auto out = OutputLengths(cfg);
    std::ostringstream os;
    os << NDim << "D N=" << cfg.N << " C=" << cfg.C << " G=" << cfg.G;
    for(index_t d = 0; d < NDim; ++d)
    {
        os << " | in=" << cfg.In[d] << " filter=" << cfg.Filter[d] << " stride=" << cfg.Stride[d]
           << " dilation=" << cfg.Dilation[d] << " pad=" << cfg.PadLeft[d] << "/" << cfg.PadRight[d]
           << " out=" << out[d];
    }
    return os.str();
}

// Pad -> embed -> merge chain the custom transform replaces
template <index_t NDim>
auto MakeGenericDescriptor(const ForwardConvCase<NDim>& cfg, const std::array<index_t, NDim>& out)
{
    const index_t N = cfg.N;
    const index_t C = cfg.C;
    if constexpr(NDim == 1)
    {
        const auto input_desc = MakeInputDescriptor(cfg);
        const auto padded     = transform_tensor_descriptor(
            input_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(cfg.In[0], cfg.PadLeft[0], cfg.PadRight[0]),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        const auto embedded = transform_tensor_descriptor(
            padded,
            make_tuple(make_pass_through_transform(N),
                       make_embed_transform(make_tuple(cfg.Filter[0], out[0]),
                                            make_tuple(cfg.Dilation[0], cfg.Stride[0])),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));
        return transform_tensor_descriptor(
            embedded,
            make_tuple(make_merge_transform(make_tuple(N, out[0])),
                       make_merge_transform(make_tuple(cfg.Filter[0], C))),
            make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }
    else if constexpr(NDim == 2)
    {
        const auto input_desc = MakeInputDescriptor(cfg);
        const auto padded     = transform_tensor_descriptor(
            input_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(cfg.In[0], cfg.PadLeft[0], cfg.PadRight[0]),
                       make_pad_transform(cfg.In[1], cfg.PadLeft[1], cfg.PadRight[1]),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
        const auto embedded = transform_tensor_descriptor(
            padded,
            make_tuple(make_pass_through_transform(N),
                       make_embed_transform(make_tuple(cfg.Filter[0], out[0]),
                                            make_tuple(cfg.Dilation[0], cfg.Stride[0])),
                       make_embed_transform(make_tuple(cfg.Filter[1], out[1]),
                                            make_tuple(cfg.Dilation[1], cfg.Stride[1])),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));
        return transform_tensor_descriptor(
            embedded,
            make_tuple(make_merge_transform(make_tuple(N, out[0], out[1])),
                       make_merge_transform(make_tuple(cfg.Filter[0], cfg.Filter[1], C))),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }
    else
    {
        static_assert(NDim == 3);
        const auto input_desc = MakeInputDescriptor(cfg);
        const auto padded     = transform_tensor_descriptor(
            input_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(cfg.In[0], cfg.PadLeft[0], cfg.PadRight[0]),
                       make_pad_transform(cfg.In[1], cfg.PadLeft[1], cfg.PadRight[1]),
                       make_pad_transform(cfg.In[2], cfg.PadLeft[2], cfg.PadRight[2]),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));
        const auto embedded = transform_tensor_descriptor(
            padded,
            make_tuple(make_pass_through_transform(N),
                       make_embed_transform(make_tuple(cfg.Filter[0], out[0]),
                                            make_tuple(cfg.Dilation[0], cfg.Stride[0])),
                       make_embed_transform(make_tuple(cfg.Filter[1], out[1]),
                                            make_tuple(cfg.Dilation[1], cfg.Stride[1])),
                       make_embed_transform(make_tuple(cfg.Filter[2], out[2]),
                                            make_tuple(cfg.Dilation[2], cfg.Stride[2])),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1, 2>{},
                       Sequence<3, 4>{},
                       Sequence<5, 6>{},
                       Sequence<7>{}));
        return transform_tensor_descriptor(
            embedded,
            make_tuple(
                make_merge_transform(make_tuple(N, out[0], out[1], out[2])),
                make_merge_transform(make_tuple(cfg.Filter[0], cfg.Filter[1], cfg.Filter[2], C))),
            make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }
}

template <index_t NDim>
auto MakeCustomTransform(const ForwardConvCase<NDim>& cfg,
                         const std::array<index_t, NDim>& out,
                         index_t m_padded,
                         index_t k_padded)
{
    return make_conv_fwd_data_in_transform<NDim>(cfg.N,
                                                 cfg.C,
                                                 ToMultiIndex<NDim>(cfg.In),
                                                 ToMultiIndex<NDim>(cfg.Filter),
                                                 ToMultiIndex<NDim>(out),
                                                 ToMultiIndex<NDim>(cfg.Stride),
                                                 ToMultiIndex<NDim>(cfg.Dilation),
                                                 ToMultiIndex<NDim>(cfg.PadLeft),
                                                 m_padded,
                                                 k_padded);
}

template <index_t NDim>
void CheckForwardConvInputTransformMatchesGeneric(const ForwardConvCase<NDim>& cfg)
{
    SCOPED_TRACE(Describe(cfg));

    std::array<index_t, NDim> out{};
    const auto out64    = OutputLengths(cfg);
    index_t out_spatial = 1;
    index_t filter_size = 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        out[d] = static_cast<index_t>(out64[d]);
        out_spatial *= out[d];
        filter_size *= cfg.Filter[d];
    }

    const auto generic_desc = MakeGenericDescriptor<NDim>(cfg, out);
    const auto input_desc   = MakeInputDescriptor(cfg);
    // The transform is built with the padded GEMM lengths (as the GEMM padder would produce them)
    // and must report the padding rows / columns (m >= M or k >= K) as invalid itself, with the
    // same offsets and validity as the generic pad / embed / merge chain inside the real M x K.
    const index_t M     = cfg.N * out_spatial;
    const index_t K     = filter_size * cfg.C;
    const index_t M_ext = M + out_spatial + 1;
    const index_t K_ext = K + cfg.C + 1;

    const auto make_custom_desc = [&](index_t m_padded, index_t k_padded) {
        return transform_tensor_descriptor(
            input_desc,
            make_tuple(MakeCustomTransform<NDim>(cfg, out, m_padded, k_padded)),
            make_tuple(typename arithmetic_sequence_gen<0, NDim + 2, 1>::type{}),
            make_tuple(Sequence<0, 1>{}));
    };
    const auto custom_desc = make_custom_desc(M_ext, K_ext);
    EXPECT_EQ(custom_desc.GetLength(Number<0>{}), M_ext);
    EXPECT_EQ(custom_desc.GetLength(Number<1>{}), K_ext);

    const auto check_against_generic = [&](const auto& desc, const char* what) {
        int64_t mismatches  = 0;
        index_t first_bad_m = -1;
        index_t first_bad_k = -1;
        for(index_t m = 0; m < desc.GetLength(Number<0>{}); ++m)
        {
            for(index_t k = 0; k < desc.GetLength(Number<1>{}); ++k)
            {
                const auto index              = make_tuple(m, k);
                const auto custom_coordinate  = make_tensor_coordinate(desc, index);
                const auto generic_coordinate = make_tensor_coordinate(generic_desc, index);
                const bool custom_valid =
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(desc,
                                                                                custom_coordinate);
                const bool generic_valid =
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(generic_desc,
                                                                                generic_coordinate);
                const bool expect = m < M && k < K && generic_valid;
                const bool offset_ok =
                    !expect || custom_coordinate.GetOffset() == generic_coordinate.GetOffset();
                if(custom_valid != expect || !offset_ok)
                {
                    if(mismatches++ == 0)
                    {
                        first_bad_m = m;
                        first_bad_k = k;
                    }
                }
            }
        }
        EXPECT_EQ(mismatches, 0) << what << ": first mismatch at m=" << first_bad_m
                                 << " k=" << first_bad_k;
    };
    check_against_generic(custom_desc, "padded GEMM");
    check_against_generic(make_custom_desc(M, K), "no GEMM padding");

    // Moving a coordinate (UpdateLowerIndex path used by the kernels) must agree with building
    // the coordinate from scratch. Steps go forward and backward, change one or both dimensions.
    // NOTE: this check is NOT exhaustive: it follows a fixed-seed random walk of 5000 steps.
    const index_t out_last  = out[NDim - 1];
    const index_t filt_last = cfg.Filter[NDim - 1];
    const index_t dms[]     = {
        0, 1, -1, 3, -3, 16, -16, out_last, -out_last, out_spatial, -out_spatial};
    const index_t dks[] = {
        0, 1, -1, cfg.C, -cfg.C, 4, -4, 8, -8, 32, -32, filt_last * cfg.C, -filt_last * cfg.C};
    std::mt19937 rng(2024);
    index_t cur_m           = 0;
    index_t cur_k           = 0;
    auto coordinate         = make_tensor_coordinate(custom_desc, make_tuple(cur_m, cur_k));
    int64_t move_mismatches = 0;
    int first_bad_step      = -1;
    index_t first_bad_m     = -1;
    index_t first_bad_k     = -1;
    index_t first_bad_dm    = 0;
    index_t first_bad_dk    = 0;
    for(int step = 0; step < 5000; ++step)
    {
        const index_t dm = dms[rng() % (sizeof(dms) / sizeof(dms[0]))];
        const index_t dk = dks[rng() % (sizeof(dks) / sizeof(dks[0]))];
        if(cur_m + dm < 0 || cur_m + dm >= M_ext || cur_k + dk < 0 || cur_k + dk >= K_ext)
        {
            continue;
        }
        cur_m += dm;
        cur_k += dk;
        const auto coordinate_step = make_tensor_coordinate_step(custom_desc, make_tuple(dm, dk));
        move_tensor_coordinate(custom_desc, coordinate, coordinate_step);

        const auto fresh = make_tensor_coordinate(custom_desc, make_tuple(cur_m, cur_k));
        const bool mismatch =
            coordinate.GetOffset() != fresh.GetOffset() ||
            coordinate_has_valid_offset_assuming_visible_index_is_valid(custom_desc, coordinate) !=
                coordinate_has_valid_offset_assuming_visible_index_is_valid(custom_desc, fresh);
        if(mismatch && move_mismatches++ == 0)
        {
            first_bad_step = step;
            first_bad_m    = cur_m;
            first_bad_k    = cur_k;
            first_bad_dm   = dm;
            first_bad_dk   = dk;
        }
    }
    EXPECT_EQ(move_mismatches, 0) << "first mismatch at step " << first_bad_step
                                  << " (moved by dm=" << first_bad_dm << " dk=" << first_bad_dk
                                  << " to m=" << first_bad_m << " k=" << first_bad_k << ")";
}

// Checks the transform alone (lower index and validity) against an int64 reference derived from
// the convolution definition. Used for index ranges where a tensor offset would overflow.
// The generic pad / embed / merge descriptor is not used as the reference here: it computes tensor
// offsets with 32-bit indices, which wrap for these extreme lengths, and its merge transforms rely
// on the same magic division (dividends below 2^31) whose limits this check exercises, so it would
// not be an independent reference. The generic descriptor is the reference for realistic sizes in
// CheckForwardConvInputTransformMatchesGeneric.
// NOTE: this check is NOT exhaustive. The (m, k) space is up to 2^31 x 2^31, so only boundary
// values (around every divisor, the real M and K, and the largest supported index) and a fixed-seed
// random sample are tested.
template <index_t NDim>
void CheckForwardConvInputTransformLowerIndices(const ForwardConvCase<NDim>& cfg)
{
    SCOPED_TRACE(Describe(cfg));

    const auto out64 = OutputLengths(cfg);
    std::array<index_t, NDim> out{};
    int64_t out_spatial = 1;
    int64_t filter_size = 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        out[d] = static_cast<index_t>(out64[d]);
        out_spatial *= out64[d];
        filter_size *= cfg.Filter[d];
    }
    const int64_t M = int64_t(cfg.N) * out_spatial;
    const int64_t K = filter_size * cfg.C;
    ASSERT_LT(M, int64_t(1) << 31);

    const auto transform =
        MakeCustomTransform<NDim>(cfg, out, static_cast<index_t>(M), static_cast<index_t>(K));

    // Upper-index boundaries around every divisor, plus values up to the largest dividend
    // (2^31 - 1) magic division supports; random samples cover the rest.
    std::vector<int64_t> ms{0, 1, M - 1, M, kIndexMax - 1, kIndexMax};
    std::vector<int64_t> ks{0, 1, cfg.C - 1, cfg.C, cfg.C + 1, K - 1, K, K + cfg.C};
    int64_t prod = 1;
    for(index_t d = NDim - 1; d >= 0; --d)
    {
        prod *= out64[d];
        for(int64_t delta : {-1, 0, 1})
            ms.push_back(prod + delta);
    }
    prod = cfg.C;
    for(index_t d = NDim - 1; d >= 0; --d)
    {
        prod *= cfg.Filter[d];
        for(int64_t delta : {-1, 0, 1})
            ks.push_back(prod + delta);
    }
    std::mt19937_64 rng(7);
    for(int i = 0; i < 2000; ++i)
    {
        ms.push_back(rng() % (int64_t(kIndexMax) + 1));
        ks.push_back(rng() % (K + cfg.C));
    }

    int64_t mismatches = 0;
    for(const int64_t m : ms)
    {
        for(const int64_t k : ks)
        {
            if(m < 0 || m > kIndexMax || k < 0 || k > kIndexMax)
            {
                continue;
            }
            const auto up = make_tuple(static_cast<index_t>(m), static_cast<index_t>(k));

            std::array<int64_t, NDim> o{}, f{}, in_idx{};
            int64_t q = m;
            for(index_t d = NDim - 1; d >= 0; --d)
            {
                o[d] = q % out64[d];
                q /= out64[d];
            }
            const int64_t n = q;
            q               = k;
            const int64_t c = q % cfg.C;
            q /= cfg.C;
            for(index_t d = NDim - 1; d >= 1; --d)
            {
                f[d] = q % cfg.Filter[d];
                q /= cfg.Filter[d];
            }
            f[0]          = q;
            bool valid    = true;
            bool in_range = true;
            for(index_t d = 0; d < NDim; ++d)
            {
                in_idx[d] = o[d] * cfg.Stride[d] + f[d] * cfg.Dilation[d] - cfg.PadLeft[d];
                valid &= in_idx[d] >= 0 && in_idx[d] < cfg.In[d];
                in_range &= in_idx[d] >= kIndexMin && in_idx[d] <= kIndexMax;
            }

            // The transform computes lower indices in 32-bit arithmetic; a lower index outside
            // the int32 range is not representable and therefore outside the supported range.
            if(!in_range)
            {
                continue;
            }

            // rows / columns beyond the real M x K are GEMM padding: always invalid, and their
            // lower index is meaningless
            const bool inside = m < M && k < K;
            if(inside)
            {
                MultiIndex<NDim + 2> low;
                transform.CalculateLowerIndex(low, up);

                mismatches += low[Number<0>{}] != n;
                mismatches += low[Number<NDim + 1>{}] != c;
                static_for<0, NDim, 1>{}([&](auto d) {
                    mismatches += static_cast<int64_t>(low[Number<d + 1>{}]) != in_idx[d];
                });
            }
            mismatches +=
                transform.IsValidUpperIndexMappedToValidLowerIndex(up) != (inside && valid);
        }
    }
    EXPECT_EQ(mismatches, 0);
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, ForwardConvInputTransformMatchesGeneric)
{
    // {N, C, In, Filter, Stride, Dilation, PadLeft, PadRight}
    const ForwardConvCase<1> cases1d[] = {
        {2, 3, {17}, {3}, {1}, {1}, {1}, {1}},
        {3, 5, {13}, {4}, {2}, {2}, {3}, {0}},
        {2, 1, {9}, {1}, {1}, {1}, {0}, {0}},
        {2, 7, {10}, {1}, {3}, {1}, {0}, {0}},
        {1, 4, {5}, {5}, {1}, {1}, {0}, {0}},
        {2, 4, {2}, {3}, {1}, {1}, {4}, {5}},
        {2, 3, {31}, {3}, {4}, {6}, {6}, {6}},
        {3, 16, {12}, {3}, {1}, {1}, {1}, {1}},
    };
    const ForwardConvCase<2> cases2d[] = {
        // original configuration
        {1, 2, {7, 7}, {3, 3}, {2, 2}, {2, 2}, {1, 2}, {5, 4}},
        // N > 1 with every H/W parameter different (Ho != Wo, Y != X, ...)
        {3, 5, {9, 13}, {3, 2}, {2, 3}, {1, 2}, {1, 3}, {4, 0}},
        {2, 4, {8, 10}, {2, 5}, {2, 1}, {3, 2}, {0, 2}, {1, 0}},
        {5, 6, {14, 11}, {3, 3}, {1, 2}, {1, 1}, {1, 1}, {1, 1}},
        // different channel counts: 1, odd, non power of two, power of two
        {2, 1, {6, 5}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 3, {6, 5}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 7, {6, 5}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 16, {6, 5}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        // 1x1 filters, with and without stride
        {2, 4, {5, 6}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        {2, 4, {5, 6}, {1, 1}, {2, 3}, {1, 1}, {0, 0}, {0, 0}},
        // Ho == Wo == 1
        {3, 4, {3, 3}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        // padding larger than the input
        {2, 3, {2, 3}, {3, 3}, {1, 1}, {1, 1}, {4, 5}, {4, 5}},
        // right padding not fully consumed by the stride
        {2, 3, {7, 7}, {3, 3}, {3, 3}, {1, 1}, {1, 1}, {3, 3}},
        // large stride and dilation
        {2, 3, {31, 29}, {3, 3}, {4, 5}, {6, 7}, {6, 7}, {6, 7}},
    };
    const ForwardConvCase<3> cases3d[] = {
        {2, 3, {5, 6, 7}, {3, 2, 3}, {1, 2, 1}, {1, 1, 2}, {1, 0, 1}, {1, 1, 0}},
        {3, 4, {4, 9, 5}, {2, 3, 2}, {2, 1, 3}, {2, 2, 1}, {0, 2, 1}, {1, 0, 2}},
        {2, 1, {4, 4, 4}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        {2, 5, {4, 5, 6}, {1, 1, 1}, {2, 3, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        {2, 4, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        {2, 3, {2, 2, 3}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {3, 4, 5}, {3, 4, 5}},
        {2, 3, {9, 9, 9}, {3, 3, 3}, {3, 3, 3}, {2, 1, 2}, {1, 3, 1}, {2, 3, 0}},
        {1, 16, {5, 4, 6}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
    };
    for(auto cfg : cases1d)
    {
        // packed tensor (G = 1) and a tensor with interleaved groups (NHWGC strides, G = 3)
        for(const index_t groups : {1, 3})
        {
            cfg.G = groups;
            CheckForwardConvInputTransformMatchesGeneric<1>(cfg);
        }
    }
    for(auto cfg : cases2d)
    {
        // packed tensor (G = 1) and a tensor with interleaved groups (NHWGC strides, G = 3)
        for(const index_t groups : {1, 3})
        {
            cfg.G = groups;
            CheckForwardConvInputTransformMatchesGeneric<2>(cfg);
        }
    }
    for(auto cfg : cases3d)
    {
        // packed tensor (G = 1) and a tensor with interleaved groups (NHWGC strides, G = 3)
        for(const index_t groups : {1, 3})
        {
            cfg.G = groups;
            CheckForwardConvInputTransformMatchesGeneric<3>(cfg);
        }
    }

    // Extreme configurations, checked against an int64 reference (offsets would overflow here).
    const ForwardConvCase<1> extreme1d[] = {
        // Wo = 2^29: largest first-dimension extent supported by the padding poison
        {1, 1, {1 << 29}, {1}, {1}, {1}, {0}, {0}},
        // Wo = 65537, C = 65537
        {1, 1, {65537}, {1}, {1}, {1}, {0}, {0}},
        {2, 65537, {3}, {1}, {1}, {1}, {0}, {0}},
        // K = Y * C just below 2^31
        {1, 715827882, {3}, {3}, {1}, {1}, {0}, {0}},
        // very negative and very large lower indices; only wi == 0 is valid
        {1, 1, {4}, {1}, {1 << 28}, {1}, {1 << 29}, {0}},
        {1, 1, {4}, {1}, {1 << 28}, {1}, {0}, {1 << 30}},
        {1, 1, {4}, {1}, {1 << 28}, {1}, {(1 << 29) - 5}, {0}},
        {1, 1, {1 << 29}, {2}, {(1 << 29) - 1}, {1}, {0}, {0}},
    };
    const ForwardConvCase<2> extreme2d[] = {
        // M just below 2^31 and a large channel count
        {32767, 4096, {256, 256}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        // power-of-two divisors everywhere
        {4, 64, {16, 16}, {2, 2}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        // divisors of the form 2^k + 1 and 2^k
        {2, 65, {33, 17}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {2, 65536, {2, 2}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        // large divisors: Wo = 65537, C = 65537
        {1, 1, {1, 65537}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        {2, 65537, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        // very negative lower indices; only hi == 0 / wi == 0 is valid
        {1, 1, {4, 4}, {1, 1}, {1 << 28, 1 << 28}, {1, 1}, {1 << 29, 1 << 30}, {0, 0}},
        // very large positive lower indices; only hi == 0 / wi == 0 is valid
        {1, 1, {4, 4}, {1, 1}, {1 << 28, 1 << 28}, {1, 1}, {0, 0}, {1 << 30, 1 << 30}},
        {1, 1, {4, 4}, {1, 1}, {1 << 28, 1 << 28}, {1, 1}, {0, 0}, {7 << 28, 7 << 28}},
        // large extents: Hi = 2^29, Wi = kIndexMax, filter taps reaching Hi - 1 and Hi
        {1,
         1,
         {1 << 29, kIndexMax},
         {2, 2},
         {(1 << 29) - 1, kIndexMax - 1},
         {1, 1},
         {0, 0},
         {0, 0}},
        {1, 1, {1 << 29, kIndexMax}, {2, 2}, {1 << 29, kIndexMax}, {1, 1}, {0, 0}, {1, 1}},
        // Wo = kIndexMax (largest magic-division divisor) with M = kIndexMax
        {1, 1, {1, kIndexMax}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
        // K = Y * X * C just below 2^31, with X = 1 (divisor 1) and a huge C divisor
        {1, 715827882, {3, 1}, {3, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
    };
    const ForwardConvCase<3> extreme3d[] = {
        // M just below 2^31, large C
        {2047, 1024, {32, 32, 32}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        // power-of-two divisors everywhere
        {4, 64, {8, 8, 8}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        // divisors of the form 2^k + 1
        {2, 65, {5, 17, 33}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        // large divisors on both sides
        {1, 1, {1, 1, 65537}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        {2, 65537, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        // very negative / large lower indices on every dimension
        {1,
         1,
         {4, 4, 4},
         {1, 1, 1},
         {1 << 28, 1 << 28, 1 << 28},
         {1, 1, 1},
         {1 << 29, 1 << 30, 1 << 30},
         {0, 0, 0}},
        {1,
         1,
         {4, 4, 4},
         {1, 1, 1},
         {1 << 28, 1 << 28, 1 << 28},
         {1, 1, 1},
         {0, 0, 0},
         {1 << 30, 1 << 30, 1 << 30}},
        // Wo = kIndexMax with the other output dimensions 1
        {1, 1, {1, 1, kIndexMax}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
        // K just below 2^31 with filter 3 x 1 x 1 and a huge C divisor
        {1, 715827882, {3, 1, 1}, {3, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
    };
    for(const auto& cfg : extreme1d)
    {
        CheckForwardConvInputTransformLowerIndices<1>(cfg);
    }
    for(const auto& cfg : extreme2d)
    {
        CheckForwardConvInputTransformLowerIndices<2>(cfg);
    }
    for(const auto& cfg : extreme3d)
    {
        CheckForwardConvInputTransformLowerIndices<3>(cfg);
    }
}

// =============================================================================
// Backward data output (A) descriptor: fused custom transform (index_t) against the generic chain
// of Pad / Embed / Slice / Merge / Pad / UnMerge transforms (long_index_t)
// =============================================================================

namespace {

struct BwdDataCase
{
    index_t N, C, K;
    index_t Hi, Wi;
    index_t Y, X;
    index_t StrideH, StrideW, DilationH, DilationW;
    index_t LeftPadH, LeftPadW, RightPadH, RightPadW;
    // Number of groups of the NHWGK / NHWGC tensors (only the strides depend on it)
    index_t G = 1;
};

template <typename IndexType, bool MergeOutPadding = false, bool SplitConvN = false>
auto MakeBwdDataTransformer(const BwdDataCase& c, index_t ytilde, index_t xtilde)
{
    namespace ctc   = ck::tensor_layout::convolution;
    using Transform = ck::tensor_operation::TransformConvBwdDataToGemm_v1<
        2,
        ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default,
        8,
        8,
        64,
        64,
        32,
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

    const IndexType Ho =
        (c.Hi + c.LeftPadH + c.RightPadH - c.DilationH * (c.Y - 1) - 1) / c.StrideH + 1;
    const IndexType Wo =
        (c.Wi + c.LeftPadW + c.RightPadW - c.DilationW * (c.X - 1) - 1) / c.StrideW + 1;
    const IndexType G = c.G, N = c.N, K = c.K, C = c.C;

    // NHWGK output, GKYXC weight, NHWGC input
    const std::array<IndexType, 5> a_lengths{G, N, K, Ho, Wo};
    const std::array<IndexType, 5> a_strides{K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
    const std::array<IndexType, 5> b_lengths{G, K, C, c.Y, c.X};
    const std::array<IndexType, 5> b_strides{K * c.Y * c.X * C, c.Y * c.X * C, 1, c.X * C, C};
    const std::array<IndexType, 5> c_lengths{G, N, C, c.Hi, c.Wi};
    const std::array<IndexType, 5> c_strides{C, c.Hi * c.Wi * G * C, 1, c.Wi * G * C, G * C};
    const std::array<IndexType, 2> conv_strides{c.StrideH, c.StrideW};
    const std::array<IndexType, 2> conv_dilations{c.DilationH, c.DilationW};
    const std::array<IndexType, 2> left_pads{c.LeftPadH, c.LeftPadW};
    const std::array<IndexType, 2> right_pads{c.RightPadH, c.RightPadW};
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

template <typename IndexType, bool MergeOutPadding = false, bool SplitConvN = false>
auto MakeBwdDataADescriptor(const BwdDataCase& c, index_t ytilde, index_t xtilde)
{
    return MakeBwdDataTransformer<IndexType, MergeOutPadding, SplitConvN>(c, ytilde, xtilde)
        .MakeADescriptor_AK0_M_AK1();
}

template <bool MergePadding>
void CheckBwdDataOutTransformMatchesGeneric(const BwdDataCase& c)
{
    std::ostringstream os;
    os << "N=" << c.N << " G=" << c.G << " C=" << c.C << " K=" << c.K << " In=" << c.Hi << "x"
       << c.Wi << " Filter=" << c.Y << "x" << c.X << " Stride=" << c.StrideH << "x" << c.StrideW
       << " Dilation=" << c.DilationH << "x" << c.DilationW << " Pad=" << c.LeftPadH << ","
       << c.LeftPadW << "," << c.RightPadH << "," << c.RightPadW;
    SCOPED_TRACE(os.str() + (MergePadding ? " merged padding" : " separate padding"));

    const index_t y_tilde = c.StrideH / math::gcd(c.StrideH, c.DilationH);
    const index_t x_tilde = c.StrideW / math::gcd(c.StrideW, c.DilationW);

    for(index_t ytilde = 0; ytilde < y_tilde; ++ytilde)
    {
        for(index_t xtilde = 0; xtilde < x_tilde; ++xtilde)
        {
            SCOPED_TRACE("ytilde=" + std::to_string(ytilde) + " xtilde=" + std::to_string(xtilde));

            // Skip (ytilde, xtilde) pairs without filter taps (same condition as the device op)
            if(ytilde >= c.Y || xtilde >= c.X)
            {
                continue;
            }

            const auto custom  = MakeBwdDataADescriptor<index_t, MergePadding>(c, ytilde, xtilde);
            const auto generic = MakeBwdDataADescriptor<long_index_t>(c, ytilde, xtilde);

            const index_t k0 = custom.GetLength(Number<0>{});
            const index_t m  = custom.GetLength(Number<1>{});
            const index_t k1 = custom.GetLength(Number<2>{});
            ASSERT_EQ(k0, static_cast<index_t>(generic.GetLength(Number<0>{})));
            ASSERT_EQ(m, static_cast<index_t>(generic.GetLength(Number<1>{})));
            ASSERT_EQ(k1, static_cast<index_t>(generic.GetLength(Number<2>{})));

            index_t mismatches  = 0;
            index_t valid_count = 0;
            for(index_t i0 = 0; i0 < k0; ++i0)
            {
                for(index_t i1 = 0; i1 < m; ++i1)
                {
                    for(index_t i2 = 0; i2 < k1; ++i2)
                    {
                        const auto cc =
                            make_tensor_coordinate(custom, make_multi_index(i0, i1, i2));
                        const auto gc =
                            make_tensor_coordinate(generic, make_multi_index(i0, i1, i2));
                        const bool cv = coordinate_has_valid_offset(custom, cc);
                        const bool gv = coordinate_has_valid_offset(generic, gc);
                        if(cv != gv || (cv && cc.GetOffset() != gc.GetOffset()))
                        {
                            if(mismatches++ < 5)
                            {
                                ADD_FAILURE()
                                    << "(" << i0 << "," << i1 << "," << i2
                                    << ") custom valid=" << cv << " offset=" << cc.GetOffset()
                                    << ", generic valid=" << gv << " offset=" << gc.GetOffset();
                            }
                        }
                        valid_count += gv;
                    }
                }
            }
            EXPECT_EQ(mismatches, 0);
            EXPECT_GT(valid_count, 0);
            EXPECT_EQ(CountCoordinateMoveMismatches(custom, 3000, 21), 0);
        }
    }
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, BackwardDataOutputTransformMatchesGeneric)
{
    // {N, C, K, Hi, Wi, Y, X, StrideH, StrideW, DilationH, DilationW, PadL_H, PadL_W, PadR_H,
    // PadR_W}
    const BwdDataCase cases[] = {
        {2, 8, 8, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 8, 24, 11, 7, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1},
        {3, 8, 8, 10, 13, 5, 3, 1, 2, 2, 1, 2, 1, 2, 1},
        {1, 8, 40, 12, 12, 3, 3, 2, 3, 1, 1, 0, 2, 0, 2},
        {2, 8, 8, 6, 6, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0},
        {2, 8, 8, 8, 9, 4, 4, 3, 2, 2, 3, 3, 4, 2, 1},
        // K * YDotSlice * XDotSlice and N * HTildeSlice * WTildeSlice are not block multiples
        {3, 8, 8, 7, 5, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 8, 72, 5, 9, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0},
    };
    for(auto c : cases)
    {
        // packed tensors (G = 1) and tensors with interleaved groups (NHWGK / NHWGC, G = 3)
        for(const index_t groups : {1, 3})
        {
            c.G = groups;
            CheckBwdDataOutTransformMatchesGeneric<false>(c);
            CheckBwdDataOutTransformMatchesGeneric<true>(c);
        }
    }
}

// =============================================================================
// Backward weight input (B) descriptor: fused custom transform against the generic chain of Pad /
// Embed / Merge / right-pad / UnMerge transforms
// =============================================================================

namespace {

template <index_t NDim>
struct BwdWeightCase
{
    index_t N, C, K;
    std::array<index_t, NDim> In, Filter, Stride, Dilation, PadLeft, PadRight;
    index_t BatchK;
    // Number of groups of the N spatial G C input tensor (only the strides depend on it)
    index_t G = 1;
};

template <index_t NDim, bool Custom>
auto MakeBwdWeightDescriptors(const BwdWeightCase<NDim>& c,
                              const std::array<index_t, NDim + 3>* input_strides = nullptr,
                              bool use_full_batch_kindex                         = false)
{
    using Transform = ck::tensor_operation::TransformConvBwdWeightToGemmV2<
        NDim,
        32,
        32,
        8,
        2,
        1,
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,
        index_t,
        false,
        Custom>;

    std::array<index_t, NDim> out{};
    for(index_t d = 0; d < NDim; ++d)
    {
        out[d] = (c.In[d] + c.PadLeft[d] + c.PadRight[d] - c.Dilation[d] * (c.Filter[d] - 1) - 1) /
                     c.Stride[d] +
                 1;
    }

    // N spatial G C input, N spatial G K output (the groups are interleaved: G = 1 is packed),
    // G K filter C weight
    std::array<index_t, NDim + 3> in_strides{}, wei_strides{}, out_strides{};
    in_strides[0]  = c.C;
    in_strides[1]  = c.G * c.C;
    in_strides[2]  = 1;
    out_strides[0] = c.K;
    out_strides[1] = c.G * c.K;
    out_strides[2] = 1;
    wei_strides[0] = c.K * c.C;
    wei_strides[1] = c.C;
    wei_strides[2] = 1;
    index_t in_s = c.G * c.C, out_s = c.G * c.K, wei_s = c.C;
    for(index_t d = NDim - 1; d >= 0; --d)
    {
        in_strides[3 + d]  = in_s;
        out_strides[3 + d] = out_s;
        wei_strides[3 + d] = wei_s;
        in_s *= c.In[d];
        out_s *= out[d];
        wei_s *= c.Filter[d];
    }
    for(index_t d = 0; d < NDim; ++d)
    {
        in_strides[1] *= c.In[d];
        out_strides[1] *= out[d];
    }
    if(input_strides != nullptr)
    {
        in_strides = *input_strides;
    }

    return Transform::template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDim>(
        c.N,
        c.K,
        c.C,
        c.In,
        c.Filter,
        out,
        in_strides,
        wei_strides,
        out_strides,
        c.Stride,
        c.Dilation,
        c.PadLeft,
        c.PadRight,
        c.BatchK,
        false,
        use_full_batch_kindex);
}

template <index_t NDim, bool Custom>
auto MakeBwdWeightBDescriptor(const BwdWeightCase<NDim>& c,
                              const std::array<index_t, NDim + 3>* input_strides = nullptr)
{
    return MakeBwdWeightDescriptors<NDim, Custom>(c, input_strides)[Number<1>{}];
}

template <index_t NDim>
void CheckBwdWeightBTransformMatchesGeneric(const BwdWeightCase<NDim>& c, bool require_valid = true)
{
    SCOPED_TRACE("N=" + std::to_string(c.N) + " G=" + std::to_string(c.G) +
                 " C=" + std::to_string(c.C) + " In0=" + std::to_string(c.In[0]) +
                 " Filter0=" + std::to_string(c.Filter[0]) + " BatchK=" + std::to_string(c.BatchK));

    const auto custom  = MakeBwdWeightBDescriptor<NDim, true>(c);
    const auto generic = MakeBwdWeightBDescriptor<NDim, false>(c);

    const index_t k0 = custom.GetLength(Number<0>{});
    const index_t n  = custom.GetLength(Number<1>{});
    const index_t k1 = custom.GetLength(Number<2>{});
    ASSERT_EQ(k0, generic.GetLength(Number<0>{}));
    ASSERT_EQ(n, generic.GetLength(Number<1>{}));
    ASSERT_EQ(k1, generic.GetLength(Number<2>{}));

    index_t mismatches  = 0;
    index_t valid_count = 0;
    for(index_t i0 = 0; i0 < k0; ++i0)
    {
        for(index_t i1 = 0; i1 < n; ++i1)
        {
            for(index_t i2 = 0; i2 < k1; ++i2)
            {
                const auto cc = make_tensor_coordinate(custom, make_multi_index(i0, i1, i2));
                const auto gc = make_tensor_coordinate(generic, make_multi_index(i0, i1, i2));
                const bool cv = coordinate_has_valid_offset(custom, cc);
                const bool gv = coordinate_has_valid_offset(generic, gc);
                if(cv != gv || (cv && cc.GetOffset() != gc.GetOffset()))
                {
                    if(mismatches++ < 5)
                    {
                        ADD_FAILURE() << "(" << i0 << "," << i1 << "," << i2
                                      << ") custom valid=" << cv << " offset=" << cc.GetOffset()
                                      << ", generic valid=" << gv << " offset=" << gc.GetOffset();
                    }
                }
                valid_count += gv;
            }
        }
    }
    EXPECT_EQ(mismatches, 0);
    if(require_valid)
    {
        EXPECT_GT(valid_count, 0);
    }
    EXPECT_EQ(CountCoordinateMoveMismatches(custom, 3000, 22), 0);
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightInputTransformMatchesGeneric)
{
    // {N, C, K, In, Filter, Stride, Dilation, PadLeft, PadRight, BatchK}
    const BwdWeightCase<1> cases1d[] = {
        {2, 8, 8, {17}, {3}, {1}, {1}, {1}, {1}, 1},
        {3, 5, 8, {13}, {4}, {2}, {2}, {3}, {0}, 2},
        {2, 7, 8, {10}, {1}, {3}, {1}, {0}, {0}, 3},
    };
    const BwdWeightCase<2> cases2d[] = {
        {2, 8, 8, {9, 9}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, 1},
        {2, 8, 8, {11, 7}, {3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, 2},
        {3, 5, 8, {10, 13}, {5, 3}, {1, 2}, {2, 1}, {2, 1}, {2, 1}, 1},
        {1, 6, 8, {12, 12}, {3, 3}, {2, 3}, {1, 1}, {0, 2}, {0, 2}, 3},
        {2, 8, 8, {6, 6}, {1, 1}, {2, 2}, {1, 1}, {0, 0}, {0, 0}, 2},
        {2, 3, 8, {8, 9}, {4, 4}, {3, 2}, {2, 3}, {3, 4}, {2, 1}, 1},
        {3, 8, 8, {7, 5}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, 4},
    };
    const BwdWeightCase<3> cases3d[] = {
        {1, 4, 8, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1},
        {2, 3, 8, {6, 5, 7}, {2, 3, 3}, {2, 1, 2}, {1, 2, 1}, {1, 2, 0}, {0, 1, 1}, 2},
        {2, 8, 8, {4, 4, 4}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, 3},
    };
    for(auto c : cases1d)
    {
        // packed tensors (G = 1) and tensors with interleaved groups (G = 3)
        for(const index_t groups : {1, 3})
        {
            c.G = groups;
            CheckBwdWeightBTransformMatchesGeneric<1>(c);
        }
    }
    for(auto c : cases2d)
    {
        // packed tensors (G = 1) and tensors with interleaved groups (G = 3)
        for(const index_t groups : {1, 3})
        {
            c.G = groups;
            CheckBwdWeightBTransformMatchesGeneric<2>(c);
        }
    }
    for(auto c : cases3d)
    {
        // packed tensors (G = 1) and tensors with interleaved groups (G = 3)
        for(const index_t groups : {1, 3})
        {
            c.G = groups;
            CheckBwdWeightBTransformMatchesGeneric<3>(c);
        }
    }
}

// =============================================================================
// Range of the poison used by the custom transforms for the GEMM padding
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, ConvPaddingPoisonRangeLimit)
{
    constexpr long_index_t max_wrap = (long_index_t{1} << 31) - 1;

    // the poison is length + max(0, left_pad), and max_lower_index + 2 * poison must be below 2^31
    const auto fwd_poison = [](index_t length, index_t left_pad) {
        const ForwardConvCase<1> cfg{1, 1, {length}, {1}, {1}, {1}, {left_pad}, {0}};
        const std::array<index_t, 1> out{static_cast<index_t>(OutputLengths(cfg)[0])};
        return MakeCustomTransform<1>(cfg, out, out[0], 1).Poison();
    };
    EXPECT_EQ(fwd_poison(8, 0), 8u);
    EXPECT_EQ(fwd_poison(8, 5), 13u);

    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, 0, max_wrap - 16));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(8, 0, max_wrap - 15));
    // a left pad increases the poison, and the range of the coordinate: poison = 13, and the
    // smallest coordinate is 5 below 0
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, -5, max_wrap - 31));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(8, -5, max_wrap - 30));
    // a negative left pad (positive lower index) does not
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, 5, max_wrap - 16));
    // a big input and a big pad
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(1 << 28, -(1 << 27), 0));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(1 << 30, -(1 << 29), 0));

    // IsConvInputTransformRangeSupported<1>: N, C, input, filter, output, stride, dilation, left
    // pad, m_padded, k_padded, N stride, spatial stride, C stride
    const auto offset_supported = [](index_t length, index_t spatial_stride) {
        return IsConvInputTransformRangeSupported<1>(
            1, 1, {length}, {1}, {length}, {1}, {1}, {0}, length + 1, 2, 1, {spatial_stride}, 1);
    };
    EXPECT_TRUE(offset_supported(1 << 26, 8));
    EXPECT_FALSE(offset_supported(1 << 27, 8));
    EXPECT_TRUE(offset_supported(1 << 28, 1));
    EXPECT_FALSE(offset_supported(1 << 30, 1));

    // wrw: input, filter, output, dilation, left pad, stride of the spatial dimensions of the input
    // (N = C = 1, stride 1, GemmK padded by 1 and GemmN by 64 columns)
    using WrwTransform = ck::tensor_operation::TransformConvBwdWeightToGemmV2<
        2,
        32,
        32,
        8,
        2,
        1,
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,
        index_t,
        false,
        true>;
    const auto wrw_supported = [](std::array<index_t, 2> in,
                                  std::array<index_t, 2> filter,
                                  std::array<index_t, 2> out,
                                  std::array<index_t, 2> dilation,
                                  std::array<index_t, 2> pad,
                                  std::array<index_t, 2> spatial_stride) {
        const std::array<index_t, 5> input_strides{1, 1, 1, spatial_stride[0], spatial_stride[1]};
        return WrwTransform::IsCustomTransformPaddingRangeSupported(1,
                                                                    1,
                                                                    in,
                                                                    filter,
                                                                    out,
                                                                    input_strides,
                                                                    {1, 1},
                                                                    dilation,
                                                                    pad,
                                                                    out[0] * out[1] + 1,
                                                                    filter[0] * filter[1] + 64);
    };
    // big inputs and pads are fine (unit strides)
    EXPECT_TRUE(wrw_supported({1 << 29, 1}, {1, 1}, {1 << 29, 1}, {1, 1}, {0, 0}, {1, 1}));
    EXPECT_TRUE(wrw_supported({1 << 29, 1}, {1, 1}, {(1 << 29) + 1, 1}, {1, 1}, {1, 0}, {1, 1}));
    // upper side: the filter index of the padded columns times the dilation
    EXPECT_TRUE(wrw_supported({8, 8}, {1, 1}, {8, 8}, {1 << 20, 1}, {0, 0}, {8, 1}));
    EXPECT_FALSE(wrw_supported({8, 8}, {1, 1}, {8, 8}, {1 << 26, 1}, {0, 0}, {8, 1}));
    // offset of the poisoned coordinate: about 3 * length * stride must stay below 2^31
    EXPECT_TRUE(wrw_supported({1 << 26, 1}, {1, 1}, {1 << 26, 1}, {1, 1}, {0, 0}, {8, 1}));
    EXPECT_FALSE(wrw_supported({1 << 27, 1}, {1, 1}, {1 << 27, 1}, {1, 1}, {0, 0}, {8, 1}));
    // ... also with a left pad, which increases the poison
    // (the range of the coordinate is from -left_pad to 2 poisons + the largest one: about 2^28)
    EXPECT_TRUE(wrw_supported(
        {1 << 26, 1}, {1, 1}, {(1 << 26) + (1 << 26), 1}, {1, 1}, {1 << 26, 0}, {4, 1}));
    EXPECT_FALSE(wrw_supported(
        {1 << 26, 1}, {1, 1}, {(1 << 26) + (1 << 26), 1}, {1, 1}, {1 << 26, 0}, {8, 1}));

    // fwd: padding position (m = M, k = k_pad) of an input with a 1x1 filter. m >= M and
    // k >= K, so both poisons are added: 2 * poison, with poison = length + left_pad.
    const auto fwd_padding_valid =
        [](const ForwardConvCase<1>& cfg, index_t k_pad, bool& supported) {
            const auto out64 = OutputLengths(cfg);
            const std::array<index_t, 1> out{static_cast<index_t>(out64[0])};
            const index_t M      = out[0];
            const auto transform = MakeCustomTransform<1>(cfg, out, M + 1, 192);
            // channel-first packed tensor (N = C = 1): spatial stride 1
            supported = IsConvInputTransformRangeSupported<1>(cfg.N,
                                                              cfg.C,
                                                              {cfg.In[0]},
                                                              {cfg.Filter[0]},
                                                              {out[0]},
                                                              {cfg.Stride[0]},
                                                              {cfg.Dilation[0]},
                                                              {cfg.PadLeft[0]},
                                                              M + 1,
                                                              192,
                                                              cfg.In[0],
                                                              {1},
                                                              cfg.In[0]);
            if(!supported)
            {
                return false; // not evaluated: the transform is not meant to be used then
            }
            return transform.IsValidUpperIndexMappedToValidLowerIndex(make_tuple(M, k_pad));
        };

    // a big first spatial length, with and without a left pad, and huge left pads on a small
    // input: all of them keep the padding invalid
    for(const index_t pad : {0, 1})
    {
        bool supported = false;
        EXPECT_FALSE(fwd_padding_valid({1, 1, {1 << 29}, {1}, {1}, {1}, {pad}, {0}}, 1, supported));
        EXPECT_TRUE(supported);
    }
    for(const index_t pad : {(1 << 29) - 8, (1 << 29) + 5})
    {
        for(const index_t right_pad : {0, 5})
        {
            bool supported = false;
            EXPECT_FALSE(
                fwd_padding_valid({1, 1, {8}, {1}, {1}, {1}, {pad}, {right_pad}}, 1, supported));
            EXPECT_TRUE(supported);
        }
    }

    // upper side: a 1x1 filter with a huge dilation. The padded K columns (k >= K) get f0 > 0 and
    // therefore a lower coordinate f0 * dilation. With the dilation 2^28 - 1 and f0 = 16 the
    // coordinate is 2^32 - 16, and the two poisons (2 * 8) would wrap it to 0, inside the input:
    // the padding would be read as real data. The host check rejects this problem.
    {
        bool supported = false;
        EXPECT_FALSE(fwd_padding_valid({1, 1, {8}, {1}, {1}, {1 << 20}, {0}, {0}}, 16, supported));
        EXPECT_TRUE(supported);

        // rejected by the host check (the transform is not evaluated for it)
        supported = true;
        fwd_padding_valid({1, 1, {8}, {1}, {1}, {(1 << 28) - 1}, {0}, {0}}, 16, supported);
        EXPECT_FALSE(supported);
    }
    {
        bool supported = false;
        EXPECT_FALSE(fwd_padding_valid({1, 1, {8}, {1}, {1}, {1 << 20}, {0}, {0}}, 128, supported));
        EXPECT_TRUE(supported);
    }
}

// =============================================================================
// Range limits of the custom input transform with filters larger than 1
// =============================================================================

namespace {

// Runs the host check IsConvInputTransformRangeSupported on a packed channel-first tensor (the
// first spatial dimension is the innermost one if first_dim_innermost, the last one otherwise: the
// coordinates of the dimension with stride 1 are the ones that can reach the farthest) and, if the
// problem is accepted, checks every position of the padded GEMM (all m < m_padded and
// k < k_padded) against an int64 reference: the padding must be invalid, the real positions must
// map to the right lower index, and no offset term or sum may leave the range of int32. Returns
// the result of the host check.
template <index_t NDim>
bool CheckConvInputRangeLimit(const ForwardConvCase<NDim>& cfg,
                              bool first_dim_innermost,
                              bool compare_positions = true)
{
    SCOPED_TRACE(Describe(cfg));

    constexpr int64_t limit = std::numeric_limits<index_t>::max();

    const auto out64 = OutputLengths(cfg);
    std::array<index_t, NDim> out{};
    int64_t prod_out    = 1;
    int64_t filter_size = 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        if(out64[d] < 1)
        {
            ADD_FAILURE() << "the test problem has an empty output";
            return false;
        }
        out[d] = static_cast<index_t>(out64[d]);
        prod_out *= out64[d];
        filter_size *= cfg.Filter[d];
    }
    const int64_t M        = cfg.N * prod_out;
    const int64_t K        = filter_size * cfg.C;
    const int64_t m_padded = (M + 63) / 64 * 64;
    const int64_t k_padded = (K + 63) / 64 * 64;

    std::array<long_index_t, NDim> spatial_stride{};
    long_index_t acc = 1;
    for(index_t i = 0; i < NDim; ++i)
    {
        const index_t d   = first_dim_innermost ? i : NDim - 1 - i;
        spatial_stride[d] = acc;
        acc *= cfg.In[d];
    }
    const long_index_t c_stride = acc;
    const long_index_t n_stride = acc * cfg.C;

    const auto to_long = [](const std::array<index_t, NDim>& a) {
        std::array<long_index_t, NDim> r{};
        for(index_t d = 0; d < NDim; ++d)
        {
            r[d] = a[d];
        }
        return r;
    };
    const bool supported = IsConvInputTransformRangeSupported<NDim>(cfg.N,
                                                                    cfg.C,
                                                                    to_long(cfg.In),
                                                                    to_long(cfg.Filter),
                                                                    to_long(out),
                                                                    to_long(cfg.Stride),
                                                                    to_long(cfg.Dilation),
                                                                    to_long(cfg.PadLeft),
                                                                    m_padded,
                                                                    k_padded,
                                                                    n_stride,
                                                                    spatial_stride,
                                                                    c_stride);
    if(!supported || !compare_positions)
    {
        return supported;
    }

    const auto transform = MakeCustomTransform<NDim>(
        cfg, out, static_cast<index_t>(m_padded), static_cast<index_t>(k_padded));

    int64_t mismatches = 0;
    int64_t overflows  = 0;
    for(int64_t m = 0; m < m_padded; ++m)
    {
        std::array<int64_t, NDim> o{};
        int64_t q = m;
        for(index_t d = NDim - 1; d >= 0; --d)
        {
            o[d] = q % out64[d];
            q /= out64[d];
        }
        const int64_t n = q;
        for(int64_t k = 0; k < k_padded; ++k)
        {
            std::array<int64_t, NDim> f{};
            q               = k;
            const int64_t c = q % cfg.C;
            q /= cfg.C;
            for(index_t d = NDim - 1; d >= 1; --d)
            {
                f[d] = q % cfg.Filter[d];
                q /= cfg.Filter[d];
            }
            f[0] = q;

            bool valid = m < M && k < K;
            std::array<int64_t, NDim> in_idx{};
            for(index_t d = 0; d < NDim; ++d)
            {
                in_idx[d] = o[d] * cfg.Stride[d] + f[d] * cfg.Dilation[d] - cfg.PadLeft[d];
                valid &= in_idx[d] >= 0 && in_idx[d] < cfg.In[d];
            }

            const auto up = make_tuple(static_cast<index_t>(m), static_cast<index_t>(k));
            if(transform.IsValidUpperIndexMappedToValidLowerIndex(up) != valid)
            {
                ++mismatches;
            }
            MultiIndex<NDim + 2> low;
            transform.CalculateLowerIndex(low, up);
            if(valid)
            {
                bool equal = low[Number<0>{}] == n && low[Number<NDim + 1>{}] == c;
                static_for<0, NDim, 1>{}(
                    [&](auto d) { equal &= low[Number<d + 1>{}] == in_idx[d]; });
                mismatches += !equal;
            }

            // offset in 32 bit arithmetic (emulated with 64 bit)
            int64_t sum    = 0;
            const auto add = [&](int64_t idx, int64_t stride) {
                const int64_t term = idx * stride;
                overflows += std::llabs(term) > limit;
                sum += term;
                overflows += std::llabs(sum) > limit;
            };
            add(low[Number<0>{}], n_stride);
            static_for<0, NDim, 1>{}([&](auto d) { add(low[Number<d + 1>{}], spatial_stride[d]); });
            add(low[Number<NDim + 1>{}], c_stride);
        }
    }
    EXPECT_EQ(mismatches, 0) << "an accepted problem has a wrong padding or lower index";
    EXPECT_EQ(overflows, 0) << "an accepted problem overflows the offset computation";

    // the (M, K) descriptor of the transform on an input tensor with the strides of the host check
    std::array<index_t, NDim + 2> strides{};
    strides[0] = static_cast<index_t>(n_stride);
    for(index_t d = 0; d < NDim; ++d)
    {
        strides[1 + d] = static_cast<index_t>(spatial_stride[d]);
    }
    strides[NDim + 1] = static_cast<index_t>(c_stride);
    const auto desc   = transform_tensor_descriptor(
        MakeInputDescriptor<NDim>(cfg, &strides),
        make_tuple(transform),
        make_tuple(typename arithmetic_sequence_gen<0, NDim + 2, 1>::type{}),
        make_tuple(Sequence<0, 1>{}));
    EXPECT_EQ(CountCoordinateMoveMismatches(desc, 3000, 25), 0);
    return true;
}

// 1D / 2D / 3D problem with a first spatial dimension of length 8 whose filter (F taps, dilation
// D) reaches far out of the input: the right pad is chosen so that Wo = Wi + left_pad.
template <index_t NDim>
ForwardConvCase<NDim>
MakeLargeReachCase(index_t channels, index_t filter0, int64_t dilation0, index_t left_pad0)
{
    ForwardConvCase<NDim> cfg{};
    cfg.N = 1;
    cfg.C = channels;
    for(index_t d = 0; d < NDim; ++d)
    {
        cfg.In[d]       = d == 0 ? 8 : 3;
        cfg.Filter[d]   = d == 0 ? filter0 : 2;
        cfg.Stride[d]   = d == 0 ? 1 : 2;
        cfg.Dilation[d] = 1;
        cfg.PadLeft[d]  = d == 0 ? left_pad0 : 1;
        cfg.PadRight[d] = d == 0 ? 0 : 1;
    }
    cfg.Dilation[0] = static_cast<index_t>(dilation0);
    // Wo = Wi + left_pad: right_pad = D * (F - 1), clamped to the index range
    cfg.PadRight[0] = static_cast<index_t>(std::min<int64_t>(dilation0 * (filter0 - 1), kIndexMax));
    return cfg;
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, ConvInputTransformRangeLimitWithFilters)
{
    constexpr int64_t limit = std::numeric_limits<index_t>::max();

    // First spatial dimension: the largest lower coordinate is (Wo - 1) + f0_max * D - left_pad,
    // with f0_max the filter index of the last (padded) GEMM column, and the poison is
    // Wi + left_pad. The documented limit is max_lower + 2 * poison <= 2^31 - 1.
    const auto first_dimension = [&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        for(const index_t filter0 : {2, 3, 4, 7, 8, 15, 16})
        {
            for(const index_t channels : {8, 40})
            {
                for(const index_t left_pad : {0, 5})
                {
                    int64_t filter_rest = channels;
                    for(index_t d = 1; d < NDim; ++d)
                    {
                        filter_rest *= 2; // filter of the other dimensions
                    }
                    int64_t filter_size    = filter_rest * filter0;
                    const int64_t k_padded = (filter_size + 63) / 64 * 64;
                    const int64_t f0_max   = (k_padded - 1) / filter_rest;
                    EXPECT_GE(f0_max, filter0 - 1);

                    const int64_t wi     = 8;
                    const int64_t wo     = wi + left_pad;
                    const int64_t poison = wi + left_pad;
                    // largest f0_max * D for which the poison range holds
                    const int64_t max_reach = limit - (wo - 1) - 2 * poison;
                    const int64_t d_max     = max_reach / f0_max;

                    // far from the limit, and a margin below it for the terms of the offset that
                    // are not part of the poison range (batch, channel, other dimensions)
                    for(const int64_t dilation :
                        {int64_t{1}, int64_t{1} << 20, (max_reach - 65536) / f0_max})
                    {
                        EXPECT_TRUE(CheckConvInputRangeLimit<NDim>(
                            MakeLargeReachCase<NDim>(channels, filter0, dilation, left_pad), true))
                            << "F=" << filter0 << " C=" << channels << " D=" << dilation
                            << " left_pad=" << left_pad;
                    }
                    // Closest accepted problem below the limit of the poison range. The host check
                    // also covers the other terms of the offset, so it stops accepting a bit below
                    // the limit. CheckConvInputRangeLimit then compares every row and column of the
                    // GEMM padding, including the positions where both are padding (two poisons on
                    // the largest filter index), so this is where a wrap around would show.
                    bool accepted_below_limit = false;
                    for(int64_t dilation = d_max;
                        dilation > d_max - 100000 && !accepted_below_limit;
                        --dilation)
                    {
                        accepted_below_limit = CheckConvInputRangeLimit<NDim>(
                            MakeLargeReachCase<NDim>(channels, filter0, dilation, left_pad), true);
                    }
                    EXPECT_TRUE(accepted_below_limit);
                    // just above the limit of the poison range: always rejected
                    for(const int64_t dilation : {d_max + 1})
                    {
                        EXPECT_FALSE(CheckConvInputRangeLimit<NDim>(
                            MakeLargeReachCase<NDim>(channels, filter0, dilation, left_pad), true))
                            << "F=" << filter0 << " C=" << channels << " D=" << dilation
                            << " left_pad=" << left_pad;
                    }
                }
            }
        }
    };
    // A big left pad (the poison is Wi + left_pad and Wo is about left_pad): the host check only,
    // the output would be too big to compare every position. The limit moves with the pad.
    const auto big_pad = [&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        for(const index_t filter0 : {3, 8, 16})
        {
            const index_t left_pad    = 1 << 20;
            const int64_t channels    = 8;
            const int64_t filter_rest = channels << (NDim - 1);
            const int64_t k_padded    = (filter_rest * filter0 + 63) / 64 * 64;
            const int64_t f0_max      = (k_padded - 1) / filter_rest;
            const int64_t poison      = 8 + left_pad;
            const int64_t max_reach   = limit - (8 + left_pad - 1) - 2 * poison;
            const int64_t d_max       = max_reach / f0_max;
            EXPECT_TRUE(CheckConvInputRangeLimit<NDim>(
                MakeLargeReachCase<NDim>(channels, filter0, (max_reach - 65536) / f0_max, left_pad),
                true,
                false))
                << "F=" << filter0;
            EXPECT_FALSE(CheckConvInputRangeLimit<NDim>(
                MakeLargeReachCase<NDim>(channels, filter0, d_max + 1, left_pad), true, false))
                << "F=" << filter0;
        }
    };
    big_pad(std::integral_constant<index_t, 1>{});
    big_pad(std::integral_constant<index_t, 2>{});
    big_pad(std::integral_constant<index_t, 3>{});

    first_dimension(std::integral_constant<index_t, 1>{});
    first_dimension(std::integral_constant<index_t, 2>{});
    first_dimension(std::integral_constant<index_t, 3>{});

    // Last spatial dimension: its coordinates are not poisoned, but they are multiplied by their
    // stride in 32 bit, so (F - 1) * D must stay below 2^31 whatever the other terms are.
    const auto last_dimension = [&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        for(const index_t filter_last : {2, 3, 4, 7, 8, 15, 16})
        {
            const auto make = [&](int64_t dilation) {
                ForwardConvCase<NDim> cfg{};
                cfg.N = 1;
                cfg.C = 64; // K is a multiple of the GEMM block: no padded columns
                for(index_t d = 0; d < NDim; ++d)
                {
                    // the clamped right pad of the rejected probes loses up to F - 2 elements
                    cfg.In[d]       = d == NDim - 1 ? 32 : 8;
                    cfg.Filter[d]   = d == NDim - 1 ? filter_last : 1;
                    cfg.Stride[d]   = 1;
                    cfg.Dilation[d] = d == NDim - 1 ? static_cast<index_t>(dilation) : 1;
                    cfg.PadLeft[d]  = 0;
                    cfg.PadRight[d] = d == NDim - 1 ? static_cast<index_t>(std::min<int64_t>(
                                                          dilation * (filter_last - 1), kIndexMax))
                                                    : 0;
                }
                return cfg;
            };
            for(const int64_t dilation :
                {int64_t{1}, int64_t{1} << 20, (int64_t{1} << 30) / (filter_last - 1)})
            {
                EXPECT_TRUE(CheckConvInputRangeLimit<NDim>(make(dilation), false))
                    << "F=" << filter_last << " D=" << dilation;
            }
            // the dilation that gives (F - 1) * D > 2^31 - 1 is not an index for F = 2
            if(filter_last > 2)
            {
                const int64_t dilation = limit / (filter_last - 1) + 1;
                EXPECT_FALSE(CheckConvInputRangeLimit<NDim>(make(dilation), false))
                    << "F=" << filter_last << " D=" << dilation;
            }
        }
    };
    last_dimension(std::integral_constant<index_t, 1>{});
    last_dimension(std::integral_constant<index_t, 2>{});
    last_dimension(std::integral_constant<index_t, 3>{});
}

// =============================================================================
// Backward data merged padding: filters, strides and dilations, and the range limits
// =============================================================================

namespace {

// Output (A matrix) of a bwd-data problem with N = G = 1, C = K = 8 and a 1 wide W dimension. The
// H dimension is the one that is varied: filter y, stride, dilation, pads.
BwdDataCase MakeBwdDataTallCase(index_t hi,
                                index_t y,
                                index_t stride,
                                index_t dilation,
                                index_t left_pad,
                                index_t right_pad,
                                index_t groups = 1)
{
    BwdDataCase c{};
    c.N = 1;
    c.G = groups;
    c.C = c.K = 8;
    c.Hi      = hi;
    c.Wi      = 1;
    c.Y       = y;
    c.X       = 1;
    c.StrideH = stride;
    c.StrideW = c.DilationW = 1;
    c.DilationH             = dilation;
    c.LeftPadH              = left_pad;
    c.RightPadH             = right_pad;
    c.LeftPadW = c.RightPadW = 0;
    return c;
}

// Result of CheckBwdDataMergedPadding
struct BwdDataMergedResult
{
    bool supported;
    int64_t mismatches;
    int64_t positions;
};

// Compares the A descriptor of the merged padding custom transform against the generic chain for
// one (ytilde, xtilde) group, if the host check IsMergedOutPaddingRangeSupported accepts the
// problem (the transform is not meant to be used otherwise). All positions are compared if
// exhaustive, else the first rows / columns, the last rows / columns (where the GEMM padding is)
// and random ones, with every K1 index.
template <bool SplitConvN = false>
BwdDataMergedResult
CheckBwdDataMergedPadding(const BwdDataCase& c, index_t ytilde, index_t xtilde, bool exhaustive)
{
    const auto transformer = MakeBwdDataTransformer<index_t, true, SplitConvN>(c, ytilde, xtilde);
    const auto custom      = transformer.MakeADescriptor_AK0_M_AK1();
    const index_t k0       = custom.GetLength(Number<0>{});
    const index_t m        = custom.GetLength(Number<1>{});
    const index_t k1       = custom.GetLength(Number<2>{});

    BwdDataMergedResult result{transformer.IsMergedOutPaddingRangeSupported(m), 0, 0};
    if(!result.supported)
    {
        return result;
    }

    const auto generic = MakeBwdDataADescriptor<long_index_t, false, SplitConvN>(c, ytilde, xtilde);
    EXPECT_EQ(k0, static_cast<index_t>(generic.GetLength(Number<0>{})));
    EXPECT_EQ(m, static_cast<index_t>(generic.GetLength(Number<1>{})));
    EXPECT_EQ(k1, static_cast<index_t>(generic.GetLength(Number<2>{})));

    const auto check = [&](index_t i0, index_t i1, index_t i2) {
        const auto idx = make_multi_index(i0, i1, i2);
        const auto cc  = make_tensor_coordinate(custom, idx);
        const auto gc  = make_tensor_coordinate(generic, idx);
        const bool cv  = coordinate_has_valid_offset(custom, cc);
        const bool gv  = coordinate_has_valid_offset(generic, gc);
        ++result.positions;
        if(cv != gv || (cv && cc.GetOffset() != gc.GetOffset()))
        {
            if(result.mismatches++ < 3)
            {
                ADD_FAILURE() << "(" << i0 << "," << i1 << "," << i2 << ") custom valid=" << cv
                              << " offset=" << cc.GetOffset() << ", generic valid=" << gv
                              << " offset=" << gc.GetOffset();
            }
        }
    };

    std::vector<index_t> k0_list, m_list;
    if(exhaustive)
    {
        for(index_t i = 0; i < k0; ++i)
        {
            k0_list.push_back(i);
        }
        for(index_t i = 0; i < m; ++i)
        {
            m_list.push_back(i);
        }
    }
    else
    {
        std::mt19937 rng(13);
        for(index_t i = 0; i < 4 && i < k0; ++i)
        {
            k0_list.push_back(i);
            k0_list.push_back(k0 - 1 - i);
        }
        for(index_t i = 0; i < 4 && i < m; ++i)
        {
            m_list.push_back(i);
        }
        for(index_t i = 0; i < 160 && i < m; ++i)
        {
            m_list.push_back(m - 1 - i);
        }
        for(int i = 0; i < 40; ++i)
        {
            k0_list.push_back(static_cast<index_t>(rng() % k0));
            m_list.push_back(static_cast<index_t>(rng() % m));
        }
    }
    for(const index_t i0 : k0_list)
    {
        for(const index_t i1 : m_list)
        {
            for(index_t i2 = 0; i2 < k1; ++i2)
            {
                check(i0, i1, i2);
            }
        }
    }
    EXPECT_EQ(CountCoordinateMoveMismatches(custom, 3000, 23), 0);
    return result;
}

// All (ytilde, xtilde) groups of a problem (the ones with filter taps, as in the device op)
template <typename F>
void ForEachBwdDataGroup(const BwdDataCase& c, F&& f)
{
    const index_t y_tilde = c.StrideH / math::gcd(c.StrideH, c.DilationH);
    const index_t x_tilde = c.StrideW / math::gcd(c.StrideW, c.DilationW);
    for(index_t ytilde = 0; ytilde < y_tilde; ++ytilde)
    {
        for(index_t xtilde = 0; xtilde < x_tilde; ++xtilde)
        {
            if(ytilde < c.Y && xtilde < c.X)
            {
                f(ytilde, xtilde);
            }
        }
    }
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, BackwardDataMergedPaddingWithFilters)
{
    // filters 1..16 (odd and even, with the other dimension's filter of the opposite parity most
    // of the time), strides and dilations of both dimensions, groups 1 and 3, and pads that keep
    // the output non empty. These problems are small: the host check must accept all of them and
    // every position of every group is compared with the generic chain.
    const index_t strides[][2]   = {{1, 1}, {2, 1}, {1, 3}, {3, 2}, {2, 2}};
    const index_t dilations[][2] = {{1, 1}, {2, 1}, {1, 2}, {2, 3}};
    const index_t left_pads[]    = {0, 1, 3};
    int64_t positions            = 0;
    for(index_t y = 1; y <= 16; ++y)
    {
        BwdDataCase c{};
        c.N = 2;
        c.C = c.K   = 8;
        c.G         = y % 3 == 0 ? 3 : 1;
        c.Y         = y;
        c.X         = (y * 7) % 16 + 1;
        c.StrideH   = strides[y % 5][0];
        c.StrideW   = strides[y % 5][1];
        c.DilationH = dilations[y % 4][0];
        c.DilationW = dilations[(y + 1) % 4][1];
        c.LeftPadH  = left_pads[y % 3];
        c.LeftPadW  = left_pads[(y + 1) % 3];
        c.Hi        = 9 + y % 4;
        c.Wi        = 9 + (y + 2) % 4;
        // right pads so that Ho and Wo are at least 1 (and a bit more for some)
        c.RightPadH = std::max<index_t>(0, c.DilationH * (c.Y - 1) + 1 - c.Hi - c.LeftPadH) + y % 3;
        c.RightPadW = std::max<index_t>(0, c.DilationW * (c.X - 1) + 1 - c.Wi - c.LeftPadW) + y % 2;

        SCOPED_TRACE("N=2 G=" + std::to_string(c.G) + " In=" + std::to_string(c.Hi) + "x" +
                     std::to_string(c.Wi) + " Filter=" + std::to_string(c.Y) + "x" +
                     std::to_string(c.X) + " Stride=" + std::to_string(c.StrideH) + "x" +
                     std::to_string(c.StrideW) + " Dilation=" + std::to_string(c.DilationH) + "x" +
                     std::to_string(c.DilationW) + " Pad=" + std::to_string(c.LeftPadH) + "," +
                     std::to_string(c.LeftPadW) + "," + std::to_string(c.RightPadH) + "," +
                     std::to_string(c.RightPadW));
        ForEachBwdDataGroup(c, [&](index_t ytilde, index_t xtilde) {
            SCOPED_TRACE("ytilde=" + std::to_string(ytilde) + " xtilde=" + std::to_string(xtilde));
            const auto result = CheckBwdDataMergedPadding(c, ytilde, xtilde, true);
            EXPECT_TRUE(result.supported);
            EXPECT_EQ(result.mismatches, 0);
            positions += result.positions;
        });
    }
    EXPECT_GT(positions, 0);
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

TEST(TensorDescriptorFunctorsIntegration, BackwardDataMergedPaddingRangeLimit)
{
    // The poison is added to the Ho coordinate: Ho + (the largest ydot, with the ydot of the padded
    // GemmK columns, times the ydot step). The host check must accept ordinary problems and reject
    // the ones where a poisoned coordinate or its offset (coordinate * stride) is out of range.
    // Every accepted problem is compared with the generic chain on the first and last rows /
    // columns, where the GEMM padding is, and on random positions. A wrap around or an overflow of
    // the 32-bit offset shows up as a mismatch of the validity or of the offset.

    // 1. a tall output: Ho = Hi, 1x1 filter, stride 1
    for(const int log_hi : {10, 20, 24})
    {
        const auto c      = MakeBwdDataTallCase(1 << log_hi, 1, 1, 1, 0, 0);
        const auto result = CheckBwdDataMergedPadding(c, 0, 0, false);
        EXPECT_TRUE(result.supported) << "Hi = 2^" << log_hi;
        EXPECT_EQ(result.mismatches, 0) << "Hi = 2^" << log_hi;
    }
    {
        // around the limit of the offset of a poisoned coordinate (Ho + 2 poisons) * HoStride_,
        // an accepted problem must be exact
        int accepted = 0;
        for(int step = 0; step <= 15; ++step)
        {
            // Ho is not a multiple of the block size, so that there are padding rows as well as
            // padding columns (K = 8 is padded to the block size too): two poisons
            const auto c = MakeBwdDataTallCase((1 << 25) + step * (1 << 21) + 5, 1, 1, 1, 0, 0);
            const auto result = CheckBwdDataMergedPadding(c, 0, 0, false);
            EXPECT_EQ(result.mismatches, 0) << "Hi = " << c.Hi;
            accepted += result.supported;
        }
        EXPECT_GT(accepted, 0);
    }
    {
        // above the limit: a poisoned Ho coordinate times HoStride_ would pass 2^31
        const auto c           = MakeBwdDataTallCase(1 << 27, 1, 1, 1, 0, 0);
        const auto transformer = MakeBwdDataTransformer<index_t, true>(c, 0, 0);
        EXPECT_FALSE(transformer.IsMergedOutPaddingRangeSupported(int64_t{1} << 27));
    }

    // 2. big dilations (the filter taps, and the padded columns, reach far outside the tensor) for
    // several filter sizes, odd and even, with stride 1 and 2. The right pad is the one that keeps
    // Ho = Hi / stride.
    for(const index_t y : {2, 3, 4, 7, 8, 15, 16})
    {
        for(const index_t stride : {1, 2})
        {
            SCOPED_TRACE("Y=" + std::to_string(y) + " stride=" + std::to_string(stride));
            const auto make_case = [&](int64_t dilation) {
                const int64_t right_pad = std::min<int64_t>(dilation * (y - 1), kIndexMax);
                return MakeBwdDataTallCase(8,
                                           y,
                                           stride,
                                           static_cast<index_t>(dilation),
                                           0,
                                           static_cast<index_t>(right_pad));
            };
            const auto output_length = [&](const BwdDataCase& c) {
                return (int64_t(c.Hi) + c.LeftPadH + c.RightPadH -
                        int64_t(c.DilationH) * (c.Y - 1) - 1) /
                           c.StrideH +
                       1;
            };

            // moderate dilations are accepted
            for(const int64_t dilation : {int64_t{1}, int64_t{1} << 10, int64_t{1} << 16})
            {
                const auto c = make_case(dilation);
                ASSERT_GE(output_length(c), 1);
                ForEachBwdDataGroup(c, [&](index_t ytilde, index_t xtilde) {
                    const auto result = CheckBwdDataMergedPadding(c, ytilde, xtilde, false);
                    EXPECT_TRUE(result.supported) << "dilation=" << dilation;
                    EXPECT_EQ(result.mismatches, 0) << "dilation=" << dilation;
                });
            }

            // around the limit the host check decides, and an accepted problem must be exact
            int accepted = 0;
            for(int step = 0; step < 24; ++step)
            {
                const int64_t dilation = (int64_t{1} << 20) + step * (int64_t{1} << 17);
                const auto c           = make_case(dilation);
                if(output_length(c) < 1)
                {
                    continue;
                }
                ForEachBwdDataGroup(c, [&](index_t ytilde, index_t xtilde) {
                    const auto result = CheckBwdDataMergedPadding(c, ytilde, xtilde, false);
                    EXPECT_EQ(result.mismatches, 0) << "dilation=" << dilation;
                    accepted += result.supported;
                });
            }
            EXPECT_GT(accepted, 0);

            // a dilation of 2^26 is rejected (the right pad of the larger filters is clamped)
            const auto c = make_case(int64_t{1} << 26);
            if(output_length(c) >= 1)
            {
                bool any_rejected = false;
                ForEachBwdDataGroup(c, [&](index_t ytilde, index_t xtilde) {
                    const auto transformer =
                        MakeBwdDataTransformer<index_t, true>(c, ytilde, xtilde);
                    const auto desc = transformer.MakeADescriptor_AK0_M_AK1();
                    any_rejected |=
                        !transformer.IsMergedOutPaddingRangeSupported(desc.GetLength(Number<1>{}));
                });
                EXPECT_TRUE(any_rejected);
            }
        }
    }
}

// =============================================================================
// Backward data merged padding: groups, strides, the W dimension and big left pads
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, BackwardDataMergedPaddingRangeLimitMoreCases)
{
    // 1. groups: the strides of the output tensor (and so the offsets) depend on G, a tall output
    // up to the limit
    for(const index_t groups : {1, 3})
    {
        int accepted = 0;
        for(const int log_hi : {10, 20, 22, 23, 24})
        {
            const auto c      = MakeBwdDataTallCase((1 << log_hi) + 5, 1, 1, 1, 0, 0, groups);
            const auto result = CheckBwdDataMergedPadding(c, 0, 0, false);
            EXPECT_EQ(result.mismatches, 0) << "G=" << groups << " Hi = 2^" << log_hi;
            accepted += result.supported;
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
                const auto c = MakeBwdDataTallCase((1 << log_hi) + 5, y, stride, 1, 1, 1);
                ForEachBwdDataGroup(c, [&](index_t ytilde, index_t xtilde) {
                    const auto result = CheckBwdDataMergedPadding(c, ytilde, xtilde, false);
                    EXPECT_TRUE(result.supported);
                    EXPECT_EQ(result.mismatches, 0) << "stride=" << stride << " Y=" << y;
                });
            }
        }
    }

    // 3. the W dimension: Ho = 1 and a wide output. The poison is on Ho only, Wo is a plain
    // coordinate, but its stride and the batch stride (= Wo * K) grow with it.
    const auto wide_case = [](index_t wi, index_t x, index_t dilation, index_t right_pad) {
        BwdDataCase c{};
        c.N = c.G = 1;
        c.C = c.K = 8;
        c.Hi = c.Y = 1;
        c.Wi       = wi;
        c.X        = x;
        c.StrideH = c.StrideW = c.DilationH = 1;
        c.DilationW                         = dilation;
        c.LeftPadH = c.LeftPadW = c.RightPadH = 0;
        c.RightPadW                           = right_pad;
        return c;
    };
    for(const int log_wi : {10, 20, 24})
    {
        const auto c      = wide_case(1 << log_wi, 1, 1, 0);
        const auto result = CheckBwdDataMergedPadding(c, 0, 0, false);
        // the host check is conservative for the biggest ones (the padding of the GemmK adds up to
        // GemmKPerBlock ydot steps to the poison, which is multiplied by the huge H stride)
        EXPECT_TRUE(result.supported || log_wi > 20) << "Wi = 2^" << log_wi;
        EXPECT_EQ(result.mismatches, 0) << "Wi = 2^" << log_wi;
    }
    {
        // the output has 2^30 elements: the padding rows (n = N) would be at an offset of 2^30 and
        // the other terms add up to more than 2^31
        const auto c           = wide_case(1 << 27, 1, 1, 0);
        const auto transformer = MakeBwdDataTransformer<index_t, true>(c, 0, 0);
        EXPECT_FALSE(transformer.IsMergedOutPaddingRangeSupported(int64_t{1} << 27));
    }
    // filters 2 .. 16 in the W dimension with a big dilation
    for(const index_t x : {2, 3, 4, 7, 8, 15, 16})
    {
        for(const index_t dilation : {1, 1 << 10, 1 << 16, 1 << 20})
        {
            const auto c = wide_case(
                8,
                x,
                dilation,
                static_cast<index_t>(std::min<int64_t>(int64_t(dilation) * (x - 1), kIndexMax)));
            ForEachBwdDataGroup(c, [&](index_t ytilde, index_t xtilde) {
                const auto result = CheckBwdDataMergedPadding(c, ytilde, xtilde, false);
                EXPECT_EQ(result.mismatches, 0) << "X=" << x << " dilation=" << dilation;
                if(dilation <= (1 << 16))
                {
                    EXPECT_TRUE(result.supported) << "X=" << x << " dilation=" << dilation;
                }
            });
        }
    }

    // 4. big left pads: Ho is about the left pad
    for(const int log_pad : {10, 16, 20, 23})
    {
        const auto c      = MakeBwdDataTallCase(8, 3, 1, 1, 1 << log_pad, 0);
        const auto result = CheckBwdDataMergedPadding(c, 0, 0, false);
        // Only the first 2^3 rows of such a problem are real (the rest of the output is padding),
        // and the host check bounds the batch of the padding rows by the smallest possible
        // number of real rows, so it rejects the biggest pads
        EXPECT_TRUE(result.supported || log_pad > 20) << "left pad = 2^" << log_pad;
        EXPECT_EQ(result.mismatches, 0) << "left pad = 2^" << log_pad;
    }
}

// =============================================================================
// Backward weight B matrix: filters 1..16 and the range limits of the host check
// (TransformConvBwdWeightToGemmV2::IsCustomTransformPaddingRangeSupported)
// =============================================================================

namespace {

template <index_t NDim>
using WrwCustomTransform = ck::tensor_operation::TransformConvBwdWeightToGemmV2<
    NDim,
    32,
    32,
    8,
    2,
    1,
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,
    index_t,
    false,
    true>;

// Checks the host check of a packed channel-first input (N C spatial...: the innermost spatial
// dimension is the first or the last one, the one whose coordinates reach farthest) and, if the
// problem is accepted, every position of the B descriptor [K0, GemmN, K1] against an int64
// reference: GemmK = k0 * K1 + k1 is (n, o...), GemmN is (f..., c). Returns the host check.
template <index_t NDim>
bool CheckBwdWeightBRangeLimit(const BwdWeightCase<NDim>& c,
                               bool first_dim_innermost,
                               bool compare_positions = true)
{
    SCOPED_TRACE("N=" + std::to_string(c.N) + " C=" + std::to_string(c.C) +
                 " In0=" + std::to_string(c.In[0]) + " Filter0=" + std::to_string(c.Filter[0]) +
                 " Dilation0=" + std::to_string(c.Dilation[0]) +
                 " PadLeft0=" + std::to_string(c.PadLeft[0]) +
                 " innermost=" + (first_dim_innermost ? "first" : "last"));

    constexpr int64_t limit = std::numeric_limits<index_t>::max();

    std::array<int64_t, NDim> out64{};
    std::array<index_t, NDim> out{};
    int64_t prod_out = 1, prod_filter = 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        out64[d] = (int64_t(c.In[d]) + c.PadLeft[d] + c.PadRight[d] -
                    int64_t(c.Dilation[d]) * (c.Filter[d] - 1) - 1) /
                       c.Stride[d] +
                   1;
        if(out64[d] < 1)
        {
            ADD_FAILURE() << "the test problem has an empty output";
            return false;
        }
        out[d] = static_cast<index_t>(out64[d]);
        prod_out *= out64[d];
        prod_filter *= c.Filter[d];
    }

    std::array<index_t, NDim + 3> in_strides{};
    int64_t acc = 1;
    for(index_t i = 0; i < NDim; ++i)
    {
        const index_t d   = first_dim_innermost ? i : NDim - 1 - i;
        in_strides[3 + d] = static_cast<index_t>(acc);
        acc *= c.In[d];
    }
    if(acc * c.C * c.N > limit)
    {
        ADD_FAILURE() << "the test input is too big";
        return false;
    }
    in_strides[2] = static_cast<index_t>(acc);
    in_strides[1] = static_cast<index_t>(acc * c.C);
    in_strides[0] = 0;

    // padded GemmK (a multiple of K0PerBlock * K1 * BatchK = 16 * BatchK) and GemmN (32)
    const auto round_up         = [](int64_t x, int64_t m) { return (x + m - 1) / m * m; };
    const int64_t gemm_k_pad    = round_up(c.N * prod_out, 16 * c.BatchK);
    const int64_t gemm_n_padded = round_up(c.C * prod_filter, 32);

    const bool supported = WrwCustomTransform<NDim>::IsCustomTransformPaddingRangeSupported(
        c.N,
        c.C,
        c.In,
        c.Filter,
        out,
        in_strides,
        c.Stride,
        c.Dilation,
        c.PadLeft,
        static_cast<index_t>(gemm_k_pad),
        static_cast<index_t>(gemm_n_padded));
    if(!supported || !compare_positions)
    {
        return supported;
    }

    const auto custom = MakeBwdWeightBDescriptor<NDim, true>(c, &in_strides);
    const index_t k0  = custom.GetLength(Number<0>{});
    const index_t n   = custom.GetLength(Number<1>{});
    const index_t k1  = custom.GetLength(Number<2>{});
    EXPECT_EQ(int64_t(k0) * k1, gemm_k_pad);
    EXPECT_EQ(n, gemm_n_padded);

    int64_t mismatches = 0;
    for(index_t i0 = 0; i0 < k0; ++i0)
    {
        for(index_t i1 = 0; i1 < n; ++i1)
        {
            for(index_t i2 = 0; i2 < k1; ++i2)
            {
                const int64_t m = int64_t(i0) * k1 + i2;
                const int64_t k = i1;

                std::array<int64_t, NDim> o{}, f{}, in_idx{};
                int64_t q = m;
                for(index_t d = NDim - 1; d >= 0; --d)
                {
                    o[d] = q % out64[d];
                    q /= out64[d];
                }
                const int64_t batch = q;
                q                   = k;
                const int64_t ch    = q % c.C;
                q /= c.C;
                for(index_t d = NDim - 1; d >= 1; --d)
                {
                    f[d] = q % c.Filter[d];
                    q /= c.Filter[d];
                }
                f[0] = q;

                bool valid     = m < c.N * prod_out && k < c.C * prod_filter;
                int64_t offset = batch * in_strides[1] + ch * in_strides[2];
                for(index_t d = 0; d < NDim; ++d)
                {
                    in_idx[d] = o[d] * c.Stride[d] + f[d] * c.Dilation[d] - c.PadLeft[d];
                    valid &= in_idx[d] >= 0 && in_idx[d] < c.In[d];
                    offset += in_idx[d] * in_strides[3 + d];
                }

                const auto cc = make_tensor_coordinate(custom, make_multi_index(i0, i1, i2));
                const bool cv = coordinate_has_valid_offset(custom, cc);
                if(cv != valid || (valid && cc.GetOffset() != offset))
                {
                    if(mismatches++ < 3)
                    {
                        ADD_FAILURE() << "(" << i0 << "," << i1 << "," << i2
                                      << ") custom valid=" << cv << " offset=" << cc.GetOffset()
                                      << ", reference valid=" << valid << " offset=" << offset;
                    }
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "an accepted problem has a wrong padding or offset";
    EXPECT_EQ(CountCoordinateMoveMismatches(custom, 3000, 24), 0);
    return true;
}

// First spatial dimension of length 8 whose filter (F taps, dilation D) reaches far out of the
// input; the right pad is such that Wo = Wi + left_pad. The other dimensions are small.
template <index_t NDim>
BwdWeightCase<NDim>
MakeWrwLargeReachCase(index_t channels, index_t filter0, int64_t dilation0, index_t left_pad0)
{
    BwdWeightCase<NDim> c{};
    c.N      = 1;
    c.C      = channels;
    c.K      = 8;
    c.BatchK = 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        c.In[d]       = d == 0 ? 8 : 3;
        c.Filter[d]   = d == 0 ? filter0 : 2;
        c.Stride[d]   = d == 0 ? 1 : 2;
        c.Dilation[d] = d == 0 ? static_cast<index_t>(dilation0) : 1;
        c.PadLeft[d]  = d == 0 ? left_pad0 : 1;
        c.PadRight[d] =
            d == 0 ? static_cast<index_t>(std::min<int64_t>(dilation0 * (filter0 - 1), kIndexMax))
                   : 1;
    }
    return c;
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightWithFilters)
{
    // filters 1..16 (odd and even) in 1D, 2D and 3D with strides, dilations, pads, batch splits and
    // groups; every position of the B descriptor is compared with the generic chain
    const index_t strides[]   = {1, 2, 3};
    const index_t dilations[] = {1, 2};
    const index_t left_pads[] = {0, 1, 3};
    const auto right_pad =
        [](index_t in, index_t filter, index_t dilation, index_t left, index_t extra) {
            return std::max<index_t>(0, dilation * (filter - 1) + 1 - in - left) + extra;
        };
    for(index_t y = 1; y <= 16; ++y)
    {
        {
            BwdWeightCase<1> c{};
            c.N        = 1 + y % 2;
            c.C        = 3 + y % 5;
            c.K        = 8;
            c.In       = {20 + y % 3};
            c.Filter   = {y};
            c.Stride   = {strides[y % 3]};
            c.Dilation = {dilations[y % 2]};
            c.PadLeft  = {left_pads[y % 3]};
            c.PadRight = {right_pad(c.In[0], y, c.Dilation[0], c.PadLeft[0], y % 3)};
            c.BatchK   = 1 + y % 3;
            c.G        = y % 2 ? 3 : 1;
            CheckBwdWeightBTransformMatchesGeneric<1>(c, false);
        }
        {
            BwdWeightCase<2> c{};
            const index_t other = 1 + (y * 5) % 4;
            c.N                 = 1 + y % 2;
            c.C                 = 3 + y % 5;
            c.K                 = 8;
            c.In                = {18 + y % 3, 6 + y % 4};
            c.Filter = y % 2 ? std::array<index_t, 2>{y, other} : std::array<index_t, 2>{other, y};
            c.Stride = {strides[y % 3], strides[(y + 1) % 3]};
            c.Dilation = {dilations[y % 2], dilations[(y + 1) % 2]};
            c.PadLeft  = {left_pads[y % 3], left_pads[(y + 1) % 3]};
            c.PadRight = {right_pad(c.In[0], c.Filter[0], c.Dilation[0], c.PadLeft[0], y % 3),
                          right_pad(c.In[1], c.Filter[1], c.Dilation[1], c.PadLeft[1], y % 2)};
            c.BatchK   = 1 + y % 3;
            c.G        = y % 2 ? 1 : 3;
            CheckBwdWeightBTransformMatchesGeneric<2>(c, false);
        }
        {
            BwdWeightCase<3> c{};
            c.N        = 1 + y % 2;
            c.C        = 3 + y % 5;
            c.K        = 8;
            c.In       = {18 + y % 3, 5 + y % 2, 5};
            c.Filter   = y % 3 == 0   ? std::array<index_t, 3>{y, 2, 1}
                         : y % 3 == 1 ? std::array<index_t, 3>{1, y, 2}
                                      : std::array<index_t, 3>{2, 1, y};
            c.Stride   = {strides[y % 3], strides[(y + 1) % 3], strides[(y + 2) % 3]};
            c.Dilation = {dilations[y % 2], dilations[(y + 1) % 2], 1};
            c.PadLeft  = {left_pads[y % 3], left_pads[(y + 1) % 3], left_pads[(y + 2) % 3]};
            c.PadRight = {right_pad(c.In[0], c.Filter[0], c.Dilation[0], c.PadLeft[0], y % 3),
                          right_pad(c.In[1], c.Filter[1], c.Dilation[1], c.PadLeft[1], y % 2),
                          right_pad(c.In[2], c.Filter[2], c.Dilation[2], c.PadLeft[2], 0)};
            c.BatchK   = 1 + y % 3;
            c.G        = y % 2 ? 3 : 1;
            CheckBwdWeightBTransformMatchesGeneric<3>(c, false);
        }
    }
}

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightRangeLimitWithFilters)
{
    constexpr int64_t limit = std::numeric_limits<index_t>::max();

    // First spatial dimension: the same limit as the forward A matrix (max_lower + 2 * poison <=
    // 2^31 - 1, where max_lower comes from the filter index of the last padded GemmN column)
    const auto first_dimension = [&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        for(const index_t filter0 : {2, 3, 4, 7, 8, 15, 16})
        {
            for(const index_t channels : {8, 40})
            {
                for(const index_t left_pad : {0, 5})
                {
                    const int64_t filter_rest = int64_t(channels) << (NDim - 1);
                    const int64_t n_padded =
                        (filter_rest * filter0 + 31) / 32 * 32; // GemmN padded to NPerBlock
                    const int64_t f0_max    = (n_padded - 1) / filter_rest;
                    const int64_t wi        = 8;
                    const int64_t wo        = wi + left_pad;
                    const int64_t poison    = wi + left_pad;
                    const int64_t max_reach = limit - (wo - 1) - 2 * poison;
                    const int64_t d_max     = max_reach / f0_max;

                    for(const int64_t dilation :
                        {int64_t{1}, int64_t{1} << 20, (max_reach - 65536) / f0_max})
                    {
                        EXPECT_TRUE(CheckBwdWeightBRangeLimit<NDim>(
                            MakeWrwLargeReachCase<NDim>(channels, filter0, dilation, left_pad),
                            true))
                            << "F=" << filter0 << " C=" << channels << " D=" << dilation
                            << " left_pad=" << left_pad;
                    }
                    // the closest accepted problem below the limit: every position, including
                    // the ones where the GemmK row and the GemmN column are both padding
                    bool accepted_below_limit = false;
                    for(int64_t dilation = d_max;
                        dilation > d_max - 100000 && !accepted_below_limit;
                        --dilation)
                    {
                        accepted_below_limit = CheckBwdWeightBRangeLimit<NDim>(
                            MakeWrwLargeReachCase<NDim>(channels, filter0, dilation, left_pad),
                            true);
                    }
                    EXPECT_TRUE(accepted_below_limit);
                    EXPECT_FALSE(CheckBwdWeightBRangeLimit<NDim>(
                        MakeWrwLargeReachCase<NDim>(channels, filter0, d_max + 1, left_pad), true))
                        << "F=" << filter0 << " C=" << channels << " left_pad=" << left_pad;
                }
            }
        }
    };
    // A big left pad (the poison is Wi + left_pad and Wo is about left_pad): the host check only,
    // the GemmK would be too big to compare every position. The limit moves with the pad.
    const auto big_pad = [&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        for(const index_t filter0 : {3, 8, 16})
        {
            const index_t left_pad    = 1 << 20;
            const int64_t channels    = 8;
            const int64_t filter_rest = channels << (NDim - 1);
            const int64_t n_padded    = (filter_rest * filter0 + 31) / 32 * 32;
            const int64_t f0_max      = (n_padded - 1) / filter_rest;
            const int64_t poison      = 8 + left_pad;
            const int64_t max_reach   = limit - (8 + left_pad - 1) - 2 * poison;
            const int64_t d_max       = max_reach / f0_max;
            EXPECT_TRUE(CheckBwdWeightBRangeLimit<NDim>(
                MakeWrwLargeReachCase<NDim>(
                    channels, filter0, (max_reach - 65536) / f0_max, left_pad),
                true,
                false))
                << "F=" << filter0;
            EXPECT_FALSE(CheckBwdWeightBRangeLimit<NDim>(
                MakeWrwLargeReachCase<NDim>(channels, filter0, d_max + 1, left_pad), true, false))
                << "F=" << filter0;
        }
    };
    big_pad(std::integral_constant<index_t, 1>{});
    big_pad(std::integral_constant<index_t, 2>{});
    big_pad(std::integral_constant<index_t, 3>{});

    first_dimension(std::integral_constant<index_t, 1>{});
    first_dimension(std::integral_constant<index_t, 2>{});
    first_dimension(std::integral_constant<index_t, 3>{});

    // Last spatial dimension: only its coordinate times its stride (1) has to fit
    const auto last_dimension = [&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        for(const index_t filter_last : {2, 3, 4, 7, 8, 15, 16})
        {
            const auto make = [&](int64_t dilation) {
                BwdWeightCase<NDim> c{};
                c.N      = 1;
                c.C      = 64; // GemmN is a multiple of the block: no padded columns
                c.K      = 8;
                c.BatchK = 1;
                for(index_t d = 0; d < NDim; ++d)
                {
                    c.In[d]       = d == NDim - 1 ? 32 : 8;
                    c.Filter[d]   = d == NDim - 1 ? filter_last : 1;
                    c.Stride[d]   = 1;
                    c.Dilation[d] = d == NDim - 1 ? static_cast<index_t>(dilation) : 1;
                    c.PadLeft[d]  = 0;
                    c.PadRight[d] = d == NDim - 1 ? static_cast<index_t>(std::min<int64_t>(
                                                        dilation * (filter_last - 1), kIndexMax))
                                                  : 0;
                }
                return c;
            };
            for(const int64_t dilation :
                {int64_t{1}, int64_t{1} << 20, (int64_t{1} << 30) / (filter_last - 1)})
            {
                EXPECT_TRUE(CheckBwdWeightBRangeLimit<NDim>(make(dilation), false))
                    << "F=" << filter_last << " D=" << dilation;
            }
            // the dilation that gives (F - 1) * D > 2^31 - 1 is not an index for F = 2
            if(filter_last > 2)
            {
                EXPECT_FALSE(
                    CheckBwdWeightBRangeLimit<NDim>(make(limit / (filter_last - 1) + 1), false))
                    << "F=" << filter_last;
            }
        }
    };
    last_dimension(std::integral_constant<index_t, 1>{});
    last_dimension(std::integral_constant<index_t, 2>{});
    last_dimension(std::integral_constant<index_t, 3>{});
}

// =============================================================================
// Forward: the host check of the transform (TransformConvFwdToGemm) reads the problem from its
// members; it must give the same result as IsConvInputTransformRangeSupported called with the
// problem description
// =============================================================================

namespace {

template <index_t NDim>
std::pair<bool, bool> FwdTransformHostCheck(const ForwardConvCase<NDim>& cfg,
                                            bool first_dim_innermost)
{
    using Transform = ck::tensor_operation::TransformConvFwdToGemm<
        NDim,
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        false,
        float,
        float,
        1,
        index_t,
        false,
        true>;

    const auto out64 = OutputLengths(cfg);
    std::array<index_t, NDim> out{};
    int64_t prod_out = 1, prod_filter = 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        if(out64[d] < 1)
        {
            ADD_FAILURE() << "the test problem has an empty output";
            return {false, false};
        }
        out[d] = static_cast<index_t>(out64[d]);
        prod_out *= out64[d];
        prod_filter *= cfg.Filter[d];
    }
    const int64_t m_padded = (cfg.N * prod_out + 63) / 64 * 64;
    const int64_t k_padded = (cfg.C * prod_filter + 63) / 64 * 64;

    // G N C spatial... input (channel-first packed), G K C filter... weight, G N K spatial...
    // output (any packed layout: the host check does not read them)
    std::array<index_t, NDim + 3> a_len{}, a_str{}, b_len{}, b_str{}, c_len{}, c_str{};
    std::array<long_index_t, NDim> spatial_stride{};
    int64_t acc = 1;
    for(index_t i = 0; i < NDim; ++i)
    {
        const index_t d   = first_dim_innermost ? i : NDim - 1 - i;
        spatial_stride[d] = acc;
        a_str[3 + d]      = static_cast<index_t>(acc);
        acc *= cfg.In[d];
    }
    if(acc * cfg.C * cfg.N > int64_t{std::numeric_limits<index_t>::max()})
    {
        ADD_FAILURE() << "the test input is too big";
        return {false, false};
    }
    a_len[0] = b_len[0] = c_len[0] = 1;
    a_len[1] = c_len[1] = cfg.N;
    a_len[2] = b_len[2] = cfg.C;
    b_len[1] = c_len[2] = 8;
    a_str[0]            = static_cast<index_t>(acc * cfg.C * cfg.N);
    a_str[1]            = static_cast<index_t>(acc * cfg.C);
    a_str[2]            = static_cast<index_t>(acc);
    for(index_t d = 0; d < NDim; ++d)
    {
        a_len[3 + d] = cfg.In[d];
        b_len[3 + d] = cfg.Filter[d];
        c_len[3 + d] = out[d];
        b_str[3 + d] = 1;
        c_str[3 + d] = 1;
    }
    b_str[0] = b_str[1] = b_str[2] = 1;
    c_str[0] = c_str[1] = c_str[2] = 1;

    const Transform transform{a_len,
                              a_str,
                              b_len,
                              b_str,
                              c_len,
                              c_str,
                              cfg.Stride,
                              cfg.Dilation,
                              cfg.PadLeft,
                              cfg.PadRight};

    const auto to_long = [](const std::array<index_t, NDim>& a) {
        std::array<long_index_t, NDim> r{};
        for(index_t d = 0; d < NDim; ++d)
        {
            r[d] = a[d];
        }
        return r;
    };
    const bool expected = IsConvInputTransformRangeSupported<NDim>(cfg.N,
                                                                   cfg.C,
                                                                   to_long(cfg.In),
                                                                   to_long(cfg.Filter),
                                                                   to_long(out),
                                                                   to_long(cfg.Stride),
                                                                   to_long(cfg.Dilation),
                                                                   to_long(cfg.PadLeft),
                                                                   m_padded,
                                                                   k_padded,
                                                                   a_str[1],
                                                                   spatial_stride,
                                                                   a_str[2]);
    return {transform.IsCustomTransformPaddingRangeSupported(static_cast<index_t>(m_padded),
                                                             static_cast<index_t>(k_padded)),
            expected};
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, ForwardTransformHostCheckMatchesDescription)
{
    constexpr int64_t limit = std::numeric_limits<index_t>::max();

    // The dilation of one dimension is searched (by bisection, on the description based check)
    // where the host check flips from accepting to rejecting; the check of the transform must agree
    // with it right before, at and after the flip, so that a wrongly read member (a swapped
    // dimension, a missing pad, the wrong stride, the batch) changes the result.
    const auto run = [&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        const auto make = [&](bool first, index_t filter, int64_t dilation, index_t left_pad) {
            ForwardConvCase<NDim> cfg{};
            cfg.N             = 3;
            cfg.C             = first ? 8 : 64;
            const index_t big = first ? 0 : NDim - 1;
            for(index_t d = 0; d < NDim; ++d)
            {
                cfg.In[d]       = d == big ? (1 << 20) : 2;
                cfg.Filter[d]   = d == big ? filter : 2;
                cfg.Stride[d]   = d == big ? 1 : 2;
                cfg.Dilation[d] = d == big ? static_cast<index_t>(dilation) : 1;
                cfg.PadLeft[d]  = d == big ? left_pad : 1;
                cfg.PadRight[d] =
                    d == big
                        ? static_cast<index_t>(std::min<int64_t>(dilation * (filter - 1), limit))
                        : 1;
            }
            return cfg;
        };
        for(const bool first : {true, false})
        {
            for(const index_t filter : {3, 8, 16})
            {
                for(const index_t left_pad : {0, 5, 1 << 20})
                {
                    if(!first && left_pad != 0)
                    {
                        continue;
                    }
                    // largest dilation accepted by the description based check
                    int64_t lo = 1, hi = limit / (filter - 1);
                    ASSERT_TRUE(
                        FwdTransformHostCheck<NDim>(make(first, filter, lo, left_pad), first)
                            .second);
                    while(lo < hi)
                    {
                        const int64_t mid = lo + (hi - lo + 1) / 2;
                        if(FwdTransformHostCheck<NDim>(make(first, filter, mid, left_pad), first)
                               .second)
                        {
                            lo = mid;
                        }
                        else
                        {
                            hi = mid - 1;
                        }
                    }
                    for(int64_t dilation = lo - 2; dilation <= lo + 2; ++dilation)
                    {
                        const auto cfg    = make(first, filter, dilation, left_pad);
                        const auto result = FwdTransformHostCheck<NDim>(cfg, first);
                        EXPECT_EQ(result.first, result.second)
                            << "F=" << filter << " left_pad=" << left_pad << " D=" << dilation
                            << " first=" << first;
                        EXPECT_EQ(result.second, dilation <= lo);
                    }
                }
            }
        }
    };
    run(std::integral_constant<index_t, 1>{});
    run(std::integral_constant<index_t, 2>{});
    run(std::integral_constant<index_t, 3>{});
}

// =============================================================================
// Backward weight: the static host check of the transform must give the same result as
// IsConvInputTransformRangeSupported called with the problem description (GemmK is N * prod(O),
// GemmN is C * prod(F), see the B descriptor)
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightHostCheckMatchesDescription)
{
    constexpr int64_t limit = std::numeric_limits<index_t>::max();

    const auto run = [&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        const auto make = [&](bool first, index_t filter, int64_t dilation, index_t left_pad) {
            BwdWeightCase<NDim> c{};
            c.N               = 3;
            c.C               = first ? 8 : 64;
            c.K               = 8;
            c.BatchK          = 2;
            const index_t big = first ? 0 : NDim - 1;
            for(index_t d = 0; d < NDim; ++d)
            {
                c.In[d]       = d == big ? (1 << 20) : 2;
                c.Filter[d]   = d == big ? filter : 2;
                c.Stride[d]   = d == big ? 2 : 3;
                c.Dilation[d] = d == big ? static_cast<index_t>(dilation) : 1;
                c.PadLeft[d]  = d == big ? left_pad : 1;
                c.PadRight[d] =
                    d == big
                        ? static_cast<index_t>(std::min<int64_t>(dilation * (filter - 1), limit))
                        : 1;
            }
            return c;
        };
        // {wrapper, description based}
        const auto checks = [&](const BwdWeightCase<NDim>& c, bool first) {
            std::array<int64_t, NDim> out64{};
            std::array<index_t, NDim> out{};
            int64_t prod_out = 1, prod_filter = 1;
            for(index_t d = 0; d < NDim; ++d)
            {
                out64[d] = (int64_t(c.In[d]) + c.PadLeft[d] + c.PadRight[d] -
                            int64_t(c.Dilation[d]) * (c.Filter[d] - 1) - 1) /
                               c.Stride[d] +
                           1;
                EXPECT_GE(out64[d], 1);
                out[d] = static_cast<index_t>(out64[d]);
                prod_out *= out64[d];
                prod_filter *= c.Filter[d];
            }
            std::array<index_t, NDim + 3> in_strides{};
            std::array<long_index_t, NDim> spatial_stride{};
            int64_t acc = 1;
            for(index_t i = 0; i < NDim; ++i)
            {
                const index_t d   = first ? i : NDim - 1 - i;
                in_strides[3 + d] = static_cast<index_t>(acc);
                spatial_stride[d] = acc;
                acc *= c.In[d];
            }
            EXPECT_LE(acc * c.C * c.N, limit);
            in_strides[2] = static_cast<index_t>(acc);
            in_strides[1] = static_cast<index_t>(acc * c.C);

            const auto round_up      = [](int64_t x, int64_t m) { return (x + m - 1) / m * m; };
            const int64_t gemm_k_pad = round_up(c.N * prod_out, 16 * c.BatchK);
            const int64_t gemm_n_pad = round_up(c.C * prod_filter, 32);
            const auto to_long       = [](const std::array<index_t, NDim>& a) {
                std::array<long_index_t, NDim> r{};
                for(index_t d = 0; d < NDim; ++d)
                {
                    r[d] = a[d];
                }
                return r;
            };
            const bool wrapper = WrwCustomTransform<NDim>::IsCustomTransformPaddingRangeSupported(
                c.N,
                c.C,
                c.In,
                c.Filter,
                out,
                in_strides,
                c.Stride,
                c.Dilation,
                c.PadLeft,
                static_cast<index_t>(gemm_k_pad),
                static_cast<index_t>(gemm_n_pad));
            const bool expected = IsConvInputTransformRangeSupported<NDim>(c.N,
                                                                           c.C,
                                                                           to_long(c.In),
                                                                           to_long(c.Filter),
                                                                           to_long(out),
                                                                           to_long(c.Stride),
                                                                           to_long(c.Dilation),
                                                                           to_long(c.PadLeft),
                                                                           gemm_k_pad,
                                                                           gemm_n_pad,
                                                                           in_strides[1],
                                                                           spatial_stride,
                                                                           in_strides[2]);
            return std::make_pair(wrapper, expected);
        };
        for(const bool first : {true, false})
        {
            for(const index_t filter : {3, 8, 16})
            {
                for(const index_t left_pad : {0, 5, 1 << 20})
                {
                    if(!first && left_pad != 0)
                    {
                        continue;
                    }
                    int64_t lo = 1, hi = limit / (filter - 1);
                    ASSERT_TRUE(checks(make(first, filter, lo, left_pad), first).second);
                    while(lo < hi)
                    {
                        const int64_t mid = lo + (hi - lo + 1) / 2;
                        if(checks(make(first, filter, mid, left_pad), first).second)
                        {
                            lo = mid;
                        }
                        else
                        {
                            hi = mid - 1;
                        }
                    }
                    for(int64_t dilation = lo - 2; dilation <= lo + 2; ++dilation)
                    {
                        const auto result = checks(make(first, filter, dilation, left_pad), first);
                        EXPECT_EQ(result.first, result.second)
                            << "F=" << filter << " left_pad=" << left_pad << " D=" << dilation
                            << " first=" << first;
                        EXPECT_EQ(result.second, dilation <= lo);
                    }
                }
            }
        }
    };
    run(std::integral_constant<index_t, 1>{});
    run(std::integral_constant<index_t, 2>{});
    run(std::integral_constant<index_t, 3>{});
}

// =============================================================================
// Randomized problems (fixed seed): all the custom transforms against the generic chain
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, RandomizedCustomTransformsMatchGeneric)
{
    std::mt19937 rng(20241125);
    const auto rnd = [&](index_t lo, index_t hi) {
        return lo + static_cast<index_t>(rng() % static_cast<uint32_t>(hi - lo + 1));
    };
    // right pad that keeps the output non empty, plus a random extra
    const auto right_pad = [&](index_t in, index_t filter, index_t dilation, index_t left) {
        return std::max<index_t>(0, dilation * (filter - 1) + 1 - in - left) + rnd(0, 3);
    };

    // forward A matrix
    const auto forward = [&](auto ndim, int iterations) {
        constexpr index_t NDim = decltype(ndim)::value;
        for(int it = 0; it < iterations; ++it)
        {
            ForwardConvCase<NDim> cfg{};
            cfg.N = rnd(1, 3);
            cfg.C = rnd(1, 9);
            cfg.G = rnd(0, 1) ? 3 : 1;
            for(index_t d = 0; d < NDim; ++d)
            {
                cfg.In[d]       = rnd(1, NDim == 3 ? 5 : 9);
                cfg.Filter[d]   = rnd(1, NDim == 3 ? 4 : 8);
                cfg.Stride[d]   = rnd(1, 3);
                cfg.Dilation[d] = rnd(1, 3);
                cfg.PadLeft[d]  = rnd(0, 4);
                cfg.PadRight[d] =
                    right_pad(cfg.In[d], cfg.Filter[d], cfg.Dilation[d], cfg.PadLeft[d]);
            }
            CheckForwardConvInputTransformMatchesGeneric<NDim>(cfg);
        }
    };
    forward(std::integral_constant<index_t, 1>{}, 25);
    forward(std::integral_constant<index_t, 2>{}, 25);
    forward(std::integral_constant<index_t, 3>{}, 8);

    // backward weight B matrix
    const auto backward_weight = [&](auto ndim, int iterations) {
        constexpr index_t NDim = decltype(ndim)::value;
        for(int it = 0; it < iterations; ++it)
        {
            BwdWeightCase<NDim> c{};
            c.N      = rnd(1, 3);
            c.C      = rnd(1, 9);
            c.K      = 8;
            c.G      = rnd(0, 1) ? 3 : 1;
            c.BatchK = rnd(1, 3);
            for(index_t d = 0; d < NDim; ++d)
            {
                c.In[d]       = rnd(1, NDim == 3 ? 5 : 9);
                c.Filter[d]   = rnd(1, NDim == 3 ? 4 : 8);
                c.Stride[d]   = rnd(1, 3);
                c.Dilation[d] = rnd(1, 3);
                c.PadLeft[d]  = rnd(0, 4);
                c.PadRight[d] = right_pad(c.In[d], c.Filter[d], c.Dilation[d], c.PadLeft[d]);
            }
            CheckBwdWeightBTransformMatchesGeneric<NDim>(c, false);
        }
    };
    backward_weight(std::integral_constant<index_t, 1>{}, 25);
    backward_weight(std::integral_constant<index_t, 2>{}, 25);
    backward_weight(std::integral_constant<index_t, 3>{}, 8);

    // backward data A matrix with the merged padding (2D)
    for(int it = 0; it < 30; ++it)
    {
        BwdDataCase c{};
        c.N         = rnd(1, 3);
        c.C         = 8;
        c.K         = 8 * rnd(1, 2);
        c.G         = rnd(0, 1) ? 3 : 1;
        c.Y         = rnd(1, 8);
        c.X         = rnd(1, 8);
        c.Hi        = rnd(1, 12);
        c.Wi        = rnd(1, 12);
        c.StrideH   = rnd(1, 3);
        c.StrideW   = rnd(1, 3);
        c.DilationH = rnd(1, 3);
        c.DilationW = rnd(1, 3);
        c.LeftPadH  = rnd(0, 4);
        c.LeftPadW  = rnd(0, 4);
        c.RightPadH = right_pad(c.Hi, c.Y, c.DilationH, c.LeftPadH);
        c.RightPadW = right_pad(c.Wi, c.X, c.DilationW, c.LeftPadW);
        ForEachBwdDataGroup(c, [&](index_t ytilde, index_t xtilde) {
            const auto result = CheckBwdDataMergedPadding(c, ytilde, xtilde, true);
            EXPECT_TRUE(result.supported);
            EXPECT_EQ(result.mismatches, 0);
        });
    }
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
        for(const index_t groups : {1, 3})
        {
            BwdDataCase c{};
            c.N = sc.n;
            c.G = groups;
            c.C = c.K = 8;
            c.Hi = c.Wi = sc.hi;
            c.Y = c.X = 1;
            c.StrideH = c.StrideW = c.DilationH = c.DilationW = 1;
            c.LeftPadH = c.LeftPadW = c.RightPadH = c.RightPadW = 0;

            // with G groups the tensor has G times more elements, so more batches are split
            const auto transformer = MakeBwdDataTransformer<index_t, true, true>(c, 0, 0);
            if(groups == 1)
            {
                EXPECT_EQ(transformer.N_, sc.expected_n) << "N=" << sc.n << " Hi=" << sc.hi;
            }
            else
            {
                EXPECT_LE(transformer.N_, sc.expected_n) << "N=" << sc.n << " Hi=" << sc.hi;
                EXPECT_EQ(sc.n % transformer.N_, 0);
            }

            const auto result = CheckBwdDataMergedPadding<true>(c, 0, 0, false);
            EXPECT_TRUE(result.supported) << "N=" << sc.n << " Hi=" << sc.hi << " G=" << groups;
            EXPECT_EQ(result.mismatches, 0) << "N=" << sc.n << " Hi=" << sc.hi << " G=" << groups;
        }
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

template <index_t NDim>
struct SplitKLayouts;

template <>
struct SplitKLayouts<1>
{
    using In  = ck::tensor_layout::convolution::NWGC;
    using Wei = ck::tensor_layout::convolution::GKXC;
    using Out = ck::tensor_layout::convolution::NWGK;
};

template <>
struct SplitKLayouts<2>
{
    using In  = ck::tensor_layout::convolution::NHWGC;
    using Wei = ck::tensor_layout::convolution::GKYXC;
    using Out = ck::tensor_layout::convolution::NHWGK;
};

template <>
struct SplitKLayouts<3>
{
    using In  = ck::tensor_layout::convolution::NDHWGC;
    using Wei = ck::tensor_layout::convolution::GKZYXC;
    using Out = ck::tensor_layout::convolution::NDHWGK;
};

// Returns whether the split-K hack is eligible
template <index_t NDim>
bool CheckSplitKHackWithCustomB(BwdWeightCase<NDim> c, index_t k_batch)
{
    SCOPED_TRACE("N=" + std::to_string(c.N) + " G=" + std::to_string(c.G) +
                 " C=" + std::to_string(c.C) + " In0=" + std::to_string(c.In[0]) +
                 " Filter0=" + std::to_string(c.Filter[0]) +
                 " Stride0=" + std::to_string(c.Stride[0]) + " KBatch=" + std::to_string(k_batch));
    c.BatchK = k_batch;

    // the descriptors of the device op (full K batch index)
    const auto custom  = MakeBwdWeightDescriptors<NDim, true>(c, nullptr, true);
    const auto generic = MakeBwdWeightDescriptors<NDim, false>(c, nullptr, true);

    std::array<index_t, NDim> out{};
    index_t out_prod = 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        out[d] = (c.In[d] + c.PadLeft[d] + c.PadRight[d] - c.Dilation[d] * (c.Filter[d] - 1) - 1) /
                     c.Stride[d] +
                 1;
        out_prod *= out[d];
    }

    using Layouts                  = SplitKLayouts<NDim>;
    using Eligibility              = ck::tensor_operation::device::SplitKHackEligibility<NDim,
                                                                                         typename Layouts::In,
                                                                                         typename Layouts::Wei,
                                                                                         typename Layouts::Out>;
    constexpr index_t k_block_size = 16; // K0PerBlock * K1 of the transform
    const bool eligible_custom     = Eligibility::Check(
        custom[Number<0>{}], custom[Number<1>{}], k_batch, c.N, out, k_block_size);
    const bool eligible_generic = Eligibility::Check(
        generic[Number<0>{}], generic[Number<1>{}], k_batch, c.N, out, k_block_size);
    EXPECT_EQ(eligible_custom, eligible_generic)
        << "the custom B changes the eligibility of the split-K hack";
    if(!eligible_custom)
    {
        return false;
    }

    const auto& b          = custom[Number<1>{}];
    const index_t k0_total = b.GetLength(Number<0>{});
    const index_t n        = b.GetLength(Number<1>{});
    const index_t k1       = b.GetLength(Number<2>{});
    EXPECT_EQ(k0_total % k_batch, 0);
    const index_t k0_batch = k0_total / k_batch;
    // the stride of the pointer shift of the device op
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
    // {N, C, K, In, Filter, Stride, Dilation, PadLeft, PadRight, BatchK (unused), G}
    struct Case2d
    {
        BwdWeightCase<2> c;
        index_t k_batch;
        bool eligible;
    };
    const auto make2d = [](index_t n,
                           index_t ch,
                           std::array<index_t, 2> in,
                           std::array<index_t, 2> filter,
                           std::array<index_t, 2> stride,
                           std::array<index_t, 2> pad,
                           index_t groups) {
        BwdWeightCase<2> c{};
        c.N        = n;
        c.C        = ch;
        c.K        = 32; // GemmM is a multiple of MPerBlock (the hack needs a compact A)
        c.In       = in;
        c.Filter   = filter;
        c.Stride   = stride;
        c.Dilation = {1, 1};
        c.PadLeft  = pad;
        c.PadRight = pad;
        c.BatchK   = 1;
        c.G        = groups;
        return c;
    };
    const Case2d cases[] = {
        // 1x1, stride 1, pad 0: GemmK = N * Hi * Wi, the input is read as it is stored
        {make2d(4, 32, {4, 4}, {1, 1}, {1, 1}, {0, 0}, 1), 2, true},
        {make2d(4, 32, {4, 4}, {1, 1}, {1, 1}, {0, 0}, 1), 4, true},
        {make2d(6, 32, {4, 4}, {1, 1}, {1, 1}, {0, 0}, 1), 3, true},
        {make2d(2, 64, {8, 4}, {1, 1}, {1, 1}, {0, 0}, 1), 2, true},
        // 2x2 stride 2 pad 0: the patches do not overlap, the im2col is a permutation of the input
        {make2d(4, 8, {8, 8}, {2, 2}, {2, 2}, {0, 0}, 1), 2, true},
        // not eligible: overlapping filter, strided 1x1, C not a multiple of GemmN block, groups,
        // N not divisible by the K batch, GemmK padded, one K batch
        {make2d(4, 32, {4, 4}, {3, 3}, {1, 1}, {1, 1}, 1), 2, false},
        {make2d(4, 32, {8, 8}, {1, 1}, {2, 2}, {0, 0}, 1), 2, false},
        {make2d(4, 8, {4, 4}, {1, 1}, {1, 1}, {0, 0}, 1), 2, false},
        {make2d(4, 32, {4, 4}, {1, 1}, {1, 1}, {0, 0}, 3), 2, false},
        {make2d(3, 32, {4, 4}, {1, 1}, {1, 1}, {0, 0}, 1), 2, false},
        {make2d(4, 32, {5, 5}, {1, 1}, {1, 1}, {0, 0}, 1), 2, false},
        {make2d(4, 32, {4, 4}, {1, 1}, {1, 1}, {0, 0}, 1), 1, false},
    };
    for(const auto& tc : cases)
    {
        EXPECT_EQ(CheckSplitKHackWithCustomB<2>(tc.c, tc.k_batch), tc.eligible);
    }

    // 1D and 3D, 1x1 stride 1 pad 0
    {
        BwdWeightCase<1> c{};
        c.N        = 4;
        c.C        = 32;
        c.K        = 32;
        c.In       = {16};
        c.Filter   = {1};
        c.Stride   = {1};
        c.Dilation = {1};
        c.PadLeft = c.PadRight = {0};
        c.BatchK               = 1;
        EXPECT_TRUE((CheckSplitKHackWithCustomB<1>(c, 2)));
        EXPECT_TRUE((CheckSplitKHackWithCustomB<1>(c, 4)));
        c.Filter  = {3};
        c.PadLeft = c.PadRight = {1};
        EXPECT_FALSE((CheckSplitKHackWithCustomB<1>(c, 2)));
    }
    {
        BwdWeightCase<3> c{};
        c.N        = 4;
        c.C        = 32;
        c.K        = 32;
        c.In       = {2, 2, 4};
        c.Filter   = {1, 1, 1};
        c.Stride   = {1, 1, 1};
        c.Dilation = {1, 1, 1};
        c.PadLeft = c.PadRight = {0, 0, 0};
        c.BatchK               = 1;
        EXPECT_TRUE((CheckSplitKHackWithCustomB<3>(c, 2)));
        EXPECT_TRUE((CheckSplitKHackWithCustomB<3>(c, 4)));
    }
}
