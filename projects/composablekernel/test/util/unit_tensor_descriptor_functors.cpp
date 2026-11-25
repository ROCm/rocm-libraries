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
template <index_t NDim>
auto MakeInputDescriptor(const ForwardConvCase<NDim>& cfg)
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
        // second dimension down to -(2^31 - 1)
        {1, 1, {4, 4}, {1, 1}, {1 << 28, 1 << 28}, {1, 1}, {1 << 29, kIndexMax}, {0, 0}},
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

template <typename IndexType, bool MergeOutPadding = false>
auto MakeBwdDataADescriptor(const BwdDataCase& c, index_t ytilde, index_t xtilde)
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
        false,
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
                     tildes}
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
auto MakeBwdWeightBDescriptor(const BwdWeightCase<NDim>& c)
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
        c.BatchK)[Number<1>{}];
}

template <index_t NDim>
void CheckBwdWeightBTransformMatchesGeneric(const BwdWeightCase<NDim>& c)
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
    EXPECT_GT(valid_count, 0);
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
    EXPECT_EQ(GetConvPaddingPoison(8, 0), 8u);
    EXPECT_EQ(GetConvPaddingPoison(8, 5), 13u);
    EXPECT_EQ(GetConvPaddingPoison(8, -5), 8u);

    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, 0, max_wrap - 16));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(8, 0, max_wrap - 15));
    // a left pad increases the poison
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, -5, max_wrap - 26));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(8, -5, max_wrap - 25));
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
    EXPECT_TRUE(wrw_supported(
        {1 << 26, 1}, {1, 1}, {(1 << 26) + (1 << 26), 1}, {1, 1}, {1 << 26, 0}, {8, 1}));
    EXPECT_FALSE(wrw_supported(
        {1 << 26, 1}, {1, 1}, {(1 << 26) + (1 << 26), 1}, {1, 1}, {1 << 26, 0}, {16, 1}));

    // fwd: padding position (m = M, k = k_pad) of an input with a 1x1 filter. m >= M and
    // k >= K, so both poisons are added: 2 * poison, with poison = length + left_pad.
    const auto fwd_padding_valid =
        [](const ForwardConvCase<1>& cfg, index_t k_pad, bool& supported) {
            const auto out64 = OutputLengths(cfg);
            const std::array<index_t, 1> out{static_cast<index_t>(out64[0])};
            const index_t M      = out[0];
            const auto transform = MakeCustomTransform<1>(cfg, out, M + 1, 192);
            supported            = transform.IsPaddingRangeSupported();
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
    // coordinate is 2^32 - 16, and the two poisons (2 * 8) wrap it to 0, inside the input. The
    // generic chain would report this padding as invalid.
    {
        bool supported = false;
        EXPECT_FALSE(fwd_padding_valid({1, 1, {8}, {1}, {1}, {1 << 20}, {0}, {0}}, 16, supported));
        EXPECT_TRUE(supported);

        supported = true;
        const bool valid =
            fwd_padding_valid({1, 1, {8}, {1}, {1}, {(1 << 28) - 1}, {0}, {0}}, 16, supported);
        EXPECT_FALSE(supported);
        EXPECT_TRUE(valid) << "the padding of this problem is reported as valid";
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
bool CheckConvInputRangeLimit(const ForwardConvCase<NDim>& cfg, bool first_dim_innermost)
{
    SCOPED_TRACE(Describe(cfg));

    constexpr int64_t limit = std::numeric_limits<index_t>::max();

    const auto out64 = OutputLengths(cfg);
    std::array<index_t, NDim> out{};
    int64_t prod_out    = 1;
    int64_t filter_size = 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        EXPECT_GE(out64[d], 1);
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
    if(!supported)
    {
        return false;
    }

    const auto transform = MakeCustomTransform<NDim>(
        cfg, out, static_cast<index_t>(m_padded), static_cast<index_t>(k_padded));
    EXPECT_TRUE(transform.IsPaddingRangeSupported());

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
        cfg.Stride[d]   = 1;
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
        for(const index_t filter0 : {3, 7})
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
                    const int64_t max_reach = limit - (wo - 1) + left_pad - 2 * poison;
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
                    // just above the limit of the poison range: always rejected
                    for(const int64_t dilation : {d_max + 1, d_max + 1000})
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
    first_dimension(std::integral_constant<index_t, 1>{});
    first_dimension(std::integral_constant<index_t, 2>{});
    first_dimension(std::integral_constant<index_t, 3>{});

    // Last spatial dimension: its coordinates are not poisoned, but they are multiplied by their
    // stride in 32 bit, so (F - 1) * D must stay below 2^31 whatever the other terms are.
    const auto last_dimension = [&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        for(const index_t filter_last : {3, 7})
        {
            const auto make = [&](int64_t dilation) {
                ForwardConvCase<NDim> cfg{};
                cfg.N = 1;
                cfg.C = 64; // K is a multiple of the GEMM block: no padded columns
                for(index_t d = 0; d < NDim; ++d)
                {
                    cfg.In[d]       = 8;
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
            for(const int64_t dilation : {limit / (filter_last - 1) + 1})
            {
                EXPECT_FALSE(CheckConvInputRangeLimit<NDim>(make(dilation), false))
                    << "F=" << filter_last << " D=" << dilation;
            }
        }
    };
    last_dimension(std::integral_constant<index_t, 1>{});
    last_dimension(std::integral_constant<index_t, 2>{});
    last_dimension(std::integral_constant<index_t, 3>{});
}
