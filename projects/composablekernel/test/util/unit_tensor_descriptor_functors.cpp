// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"

using namespace ck;

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
};

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
    os << NDim << "D N=" << cfg.N << " C=" << cfg.C;
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
        const auto input_desc = make_naive_tensor_descriptor_packed(make_tuple(N, cfg.In[0], C));
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
        const auto input_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, cfg.In[0], cfg.In[1], C));
        const auto padded = transform_tensor_descriptor(
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
        const auto input_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, cfg.In[0], cfg.In[1], cfg.In[2], C));
        const auto padded = transform_tensor_descriptor(
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
auto MakeCustomTransform(const ForwardConvCase<NDim>& cfg, const std::array<index_t, NDim>& out)
{
    return make_conv_fwd_data_in_transform<NDim>(cfg.N,
                                                 cfg.C,
                                                 ToMultiIndex<NDim>(cfg.In),
                                                 ToMultiIndex<NDim>(cfg.Filter),
                                                 ToMultiIndex<NDim>(out),
                                                 ToMultiIndex<NDim>(cfg.Stride),
                                                 ToMultiIndex<NDim>(cfg.Dilation),
                                                 ToMultiIndex<NDim>(cfg.PadLeft));
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
    const auto input_desc   = [&] {
        if constexpr(NDim == 1)
            return make_naive_tensor_descriptor_packed(make_tuple(cfg.N, cfg.In[0], cfg.C));
        else if constexpr(NDim == 2)
            return make_naive_tensor_descriptor_packed(
                make_tuple(cfg.N, cfg.In[0], cfg.In[1], cfg.C));
        else
            return make_naive_tensor_descriptor_packed(
                make_tuple(cfg.N, cfg.In[0], cfg.In[1], cfg.In[2], cfg.C));
    }();
    const auto custom_desc = transform_tensor_descriptor(
        input_desc,
        make_tuple(MakeCustomTransform<NDim>(cfg, out)),
        make_tuple(typename arithmetic_sequence_gen<0, NDim + 2, 1>::type{}),
        make_tuple(Sequence<0, 1>{}));

    // Upper indices beyond M and K are also checked: GEMM padding produces them and the custom
    // transform must behave exactly like the generic merge/embed/pad chain there.
    const index_t M     = cfg.N * out_spatial;
    const index_t K     = filter_size * cfg.C;
    const index_t M_ext = M + out_spatial + 1;
    const index_t K_ext = K + cfg.C + 1;

    int64_t mismatches  = 0;
    index_t first_bad_m = -1;
    index_t first_bad_k = -1;
    for(index_t m = 0; m < M_ext; ++m)
    {
        for(index_t k = 0; k < K_ext; ++k)
        {
            const auto index              = make_tuple(m, k);
            const auto custom_coordinate  = make_tensor_coordinate(custom_desc, index);
            const auto generic_coordinate = make_tensor_coordinate(generic_desc, index);

            if(custom_coordinate.GetOffset() != generic_coordinate.GetOffset() ||
               coordinate_has_valid_offset_assuming_visible_index_is_valid(custom_desc,
                                                                           custom_coordinate) !=
                   coordinate_has_valid_offset_assuming_visible_index_is_valid(generic_desc,
                                                                               generic_coordinate))
            {
                if(mismatches++ == 0)
                {
                    first_bad_m = m;
                    first_bad_k = k;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "first mismatch at m=" << first_bad_m << " k=" << first_bad_k;

    // Moving a coordinate (UpdateLowerIndex path used by the kernels) must agree with building
    // the coordinate from scratch. Steps go forward and backward, change one or both dimensions.
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
        move_mismatches +=
            coordinate.GetOffset() != fresh.GetOffset() ||
            coordinate_has_valid_offset_assuming_visible_index_is_valid(custom_desc, coordinate) !=
                coordinate_has_valid_offset_assuming_visible_index_is_valid(custom_desc, fresh);
    }
    EXPECT_EQ(move_mismatches, 0);
}

// Checks the transform alone (lower index and validity) against an int64 reference derived from
// the convolution definition. Used for index ranges where a tensor offset would overflow.
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

    const auto transform = MakeCustomTransform<NDim>(cfg, out);

    // Upper-index boundaries around every divisor, plus values up to the largest dividend
    // (2^31 - 1) magic division supports; random samples cover the rest.
    std::vector<int64_t> ms{0, 1, M - 1, M, INT32_MAX - 1, INT32_MAX};
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
        ms.push_back(rng() % (int64_t(INT32_MAX) + 1));
        ks.push_back(rng() % (K + cfg.C));
    }

    int64_t mismatches = 0;
    for(const int64_t m : ms)
    {
        for(const int64_t k : ks)
        {
            if(m < 0 || m > INT32_MAX || k < 0 || k > INT32_MAX)
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
                in_range &= in_idx[d] >= INT32_MIN && in_idx[d] <= INT32_MAX;
            }

            // The transform computes lower indices in 32-bit arithmetic; a lower index outside
            // the int32 range (only reachable with absurd dilation/pad past the filter) is not
            // representable and therefore outside the supported range.
            if(!in_range)
            {
                continue;
            }

            MultiIndex<NDim + 2> low;
            transform.CalculateLowerIndex(low, up);

            mismatches += low[Number<0>{}] != n;
            mismatches += low[Number<NDim + 1>{}] != c;
            static_for<0, NDim, 1>{}([&](auto d) {
                mismatches += static_cast<int64_t>(low[Number<d + 1>{}]) != in_idx[d];
            });
            mismatches += transform.IsValidUpperIndexMappedToValidLowerIndex(up) != valid;
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
    for(const auto& cfg : cases1d)
    {
        CheckForwardConvInputTransformMatchesGeneric<1>(cfg);
    }
    for(const auto& cfg : cases2d)
    {
        CheckForwardConvInputTransformMatchesGeneric<2>(cfg);
    }
    for(const auto& cfg : cases3d)
    {
        CheckForwardConvInputTransformMatchesGeneric<3>(cfg);
    }

    // Extreme configurations, checked against an int64 reference (offsets would overflow here).
    const ForwardConvCase<1> extreme1d[] = {
        // Wo = INT32_MAX: largest magic-division divisor on the M side
        {1, 1, {INT32_MAX}, {1}, {1}, {1}, {0}, {0}},
        // Wo = 65537, C = 65537
        {1, 1, {65537}, {1}, {1}, {1}, {0}, {0}},
        {2, 65537, {3}, {1}, {1}, {1}, {0}, {0}},
        // K = Y * C just below 2^31
        {1, 715827882, {3}, {3}, {1}, {1}, {0}, {0}},
        // very negative and very large lower indices; only wi == 0 is valid
        {1, 1, {4}, {1}, {1 << 28}, {1}, {1 << 30}, {0}},
        {1, 1, {4}, {1}, {1 << 28}, {1}, {0}, {1 << 30}},
        {1, 1, {4}, {1}, {1 << 28}, {1}, {INT32_MAX}, {0}},
        {1, 1, {INT32_MAX}, {2}, {INT32_MAX - 1}, {1}, {0}, {0}},
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
        {1, 1, {4, 4}, {1, 1}, {1 << 28, 1 << 28}, {1, 1}, {1 << 30, 1 << 30}, {0, 0}},
        // very large positive lower indices; only hi == 0 / wi == 0 is valid
        {1, 1, {4, 4}, {1, 1}, {1 << 28, 1 << 28}, {1, 1}, {0, 0}, {1 << 30, 1 << 30}},
        {1, 1, {4, 4}, {1, 1}, {1 << 28, 1 << 28}, {1, 1}, {0, 0}, {7 << 28, 7 << 28}},
        // lower indices down to -(2^31 - 1)
        {1, 1, {4, 4}, {1, 1}, {1 << 28, 1 << 28}, {1, 1}, {INT32_MAX, INT32_MAX}, {0, 0}},
        // extents at the 32-bit limit: Hi = Wi = INT32_MAX, filter taps reaching Hi - 1 and Hi
        {1,
         1,
         {INT32_MAX, INT32_MAX},
         {2, 2},
         {INT32_MAX - 1, INT32_MAX - 1},
         {1, 1},
         {0, 0},
         {0, 0}},
        {1, 1, {INT32_MAX, INT32_MAX}, {2, 2}, {INT32_MAX, INT32_MAX}, {1, 1}, {0, 0}, {1, 1}},
        // Wo = INT32_MAX (largest magic-division divisor) with M = INT32_MAX
        {1, 1, {1, INT32_MAX}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
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
         {1 << 30, 1 << 30, 1 << 30},
         {0, 0, 0}},
        {1,
         1,
         {4, 4, 4},
         {1, 1, 1},
         {1 << 28, 1 << 28, 1 << 28},
         {1, 1, 1},
         {0, 0, 0},
         {1 << 30, 1 << 30, 1 << 30}},
        // Wo = INT32_MAX with the other output dimensions 1
        {1, 1, {1, 1, INT32_MAX}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
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
