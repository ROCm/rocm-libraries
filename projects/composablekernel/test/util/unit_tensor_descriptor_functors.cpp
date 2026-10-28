// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"

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

TEST(TensorDescriptorFunctorsIntegration, ForwardConvInputTransformMatchesGeneric)
{
    constexpr index_t N         = 1;
    constexpr index_t Hi        = 7;
    constexpr index_t Wi        = 7;
    constexpr index_t C         = 2;
    constexpr index_t Y         = 3;
    constexpr index_t X         = 3;
    constexpr index_t Ho        = 5;
    constexpr index_t Wo        = 5;
    constexpr index_t StrideH   = 2;
    constexpr index_t StrideW   = 2;
    constexpr index_t DilationH = 2;
    constexpr index_t DilationW = 2;
    constexpr index_t PadHLeft  = 1;
    constexpr index_t PadHRight = 5;
    constexpr index_t PadWLeft  = 2;
    constexpr index_t PadWRight = 4;

    const auto input_desc = make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));
    const auto custom_desc =
        transform_tensor_descriptor(input_desc,
                                    make_tuple(make_conv_fwd_data_in_transform(N,
                                                                               Hi,
                                                                               Wi,
                                                                               C,
                                                                               Y,
                                                                               X,
                                                                               Ho,
                                                                               Wo,
                                                                               StrideH,
                                                                               StrideW,
                                                                               DilationH,
                                                                               DilationW,
                                                                               PadHLeft,
                                                                               PadWLeft)),
                                    make_tuple(Sequence<0, 1, 2, 3>{}),
                                    make_tuple(Sequence<0, 1>{}));

    const auto padded_input_desc = transform_tensor_descriptor(
        input_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pad_transform(Hi, PadHLeft, PadHRight),
                   make_pad_transform(Wi, PadWLeft, PadWRight),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    const auto embedded_input_desc = transform_tensor_descriptor(
        padded_input_desc,
        make_tuple(make_pass_through_transform(N),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(DilationH, StrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(DilationW, StrideW)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto generic_desc =
        transform_tensor_descriptor(embedded_input_desc,
                                    make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                               make_merge_transform(make_tuple(Y, X, C))),
                                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}));

    using FwdTransform = ck::tensor_operation::TransformConvFwdToGemm<
        2,
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        false,
        float,
        float,
        1,
        index_t,
        false>;
    using CustomFwdTransform = ck::tensor_operation::TransformConvFwdToGemm<
        2,
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        false,
        float,
        float,
        1,
        index_t,
        false,
        true>;

    const std::array<index_t, 5> input_lengths{1, N, C, Hi, Wi};
    const std::array<index_t, 5> input_strides{C, N * Hi * Wi * C, 1, Wi * C, C};
    const std::array<index_t, 5> weight_lengths{1, 4, C, Y, X};
    const std::array<index_t, 5> weight_strides{4 * Y * X * C, Y * X * C, 1, X * C, C};
    const std::array<index_t, 5> output_lengths{1, N, 4, Ho, Wo};
    const std::array<index_t, 5> output_strides{4, Ho * Wo * 4, 1, Wo * 4, 4};
    const std::array<index_t, 2> conv_strides{StrideH, StrideW};
    const std::array<index_t, 2> conv_dilations{DilationH, DilationW};
    const std::array<index_t, 2> input_left_pads{PadHLeft, PadWLeft};
    const std::array<index_t, 2> input_right_pads{PadHRight, PadWRight};

    const FwdTransform default_transform{input_lengths,
                                         input_strides,
                                         weight_lengths,
                                         weight_strides,
                                         output_lengths,
                                         output_strides,
                                         conv_strides,
                                         conv_dilations,
                                         input_left_pads,
                                         input_right_pads};
    const CustomFwdTransform custom_transform{input_lengths,
                                              input_strides,
                                              weight_lengths,
                                              weight_strides,
                                              output_lengths,
                                              output_strides,
                                              conv_strides,
                                              conv_dilations,
                                              input_left_pads,
                                              input_right_pads};
    const auto default_fwd_desc =
        default_transform.template MakeADescriptor_M_K<tensor_layout::convolution::NHWGC>();
    const auto custom_fwd_desc =
        custom_transform.template MakeADescriptor_M_K<tensor_layout::convolution::NHWGC>();

    for(index_t m = 0; m < N * Ho * Wo; ++m)
    {
        for(index_t k = 0; k < Y * X * C; ++k)
        {
            const auto index                  = make_tuple(m, k);
            const auto custom_coordinate      = make_tensor_coordinate(custom_desc, index);
            const auto generic_coordinate     = make_tensor_coordinate(generic_desc, index);
            const auto default_fwd_coordinate = make_tensor_coordinate(default_fwd_desc, index);
            const auto custom_fwd_coordinate  = make_tensor_coordinate(custom_fwd_desc, index);

            EXPECT_EQ(custom_coordinate.GetOffset(), generic_coordinate.GetOffset());
            EXPECT_EQ(coordinate_has_valid_offset_assuming_visible_index_is_valid(
                          custom_desc, custom_coordinate),
                      coordinate_has_valid_offset_assuming_visible_index_is_valid(
                          generic_desc, generic_coordinate));
            EXPECT_EQ(default_fwd_coordinate.GetOffset(), custom_fwd_coordinate.GetOffset());
            EXPECT_EQ(coordinate_has_valid_offset_assuming_visible_index_is_valid(
                          default_fwd_desc, default_fwd_coordinate),
                      coordinate_has_valid_offset_assuming_visible_index_is_valid(
                          custom_fwd_desc, custom_fwd_coordinate));
        }
    }
}

TEST(TensorDescriptorFunctorsIntegration, BackwardConvOutputTransformMatchesGeneric)
{
    constexpr index_t N             = 1;
    constexpr index_t Ho            = 2;
    constexpr index_t Wo            = 2;
    constexpr index_t K             = 2;
    constexpr index_t YDot          = 1;
    constexpr index_t XDot          = 2;
    constexpr index_t HTilde        = 2;
    constexpr index_t WTilde        = 2;
    constexpr index_t ConvDilationH = 1;
    constexpr index_t ConvDilationW = 1;
    constexpr index_t K0            = 1;
    constexpr index_t K1            = 5;
    constexpr index_t MPerBlock     = 3;
    constexpr index_t KPerBlock     = 5;

    constexpr index_t YDotSlice         = 1;
    constexpr index_t XDotSlice         = 2;
    constexpr index_t HTildeSlice       = 2;
    constexpr index_t WTildeSlice       = 2;
    constexpr index_t IHTildeSliceBegin = 0;
    constexpr index_t IWTildeSliceBegin = 0;
    constexpr index_t HRatio            = -1;
    constexpr index_t WRatio            = -1;
    constexpr index_t KRaw              = YDotSlice * XDotSlice * K;
    constexpr index_t MRaw              = N * HTildeSlice * WTildeSlice;
    constexpr index_t KPadded           = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;
    constexpr index_t MPadded           = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
    constexpr index_t KPad              = KPadded - KRaw;
    constexpr index_t MPad              = MPadded - MRaw;

    const auto output_desc         = make_naive_tensor_descriptor_packed(make_tuple(N, Ho, Wo, K));
    const auto padded_spatial_desc = transform_tensor_descriptor(
        output_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pad_transform(Ho, Number<0>{}, Number<0>{}),
                   make_pad_transform(Wo, Number<0>{}, Number<0>{}),
                   make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
    const auto custom_desc =
        transform_tensor_descriptor(padded_spatial_desc,
                                    make_tuple(make_conv_bwd_data_out_transform(N,
                                                                                Ho,
                                                                                Wo,
                                                                                K,
                                                                                YDot,
                                                                                XDot,
                                                                                HTilde,
                                                                                WTilde,
                                                                                ConvDilationH,
                                                                                ConvDilationW,
                                                                                HTildeSlice,
                                                                                WTildeSlice,
                                                                                YDotSlice,
                                                                                XDotSlice,
                                                                                IHTildeSliceBegin,
                                                                                IWTildeSliceBegin,
                                                                                1,
                                                                                1,
                                                                                K0,
                                                                                K1,
                                                                                MPerBlock,
                                                                                KPerBlock)),
                                    make_tuple(Sequence<0, 1, 2, 3>{}),
                                    make_tuple(Sequence<0, 1, 2>{}));

    const auto embedded_desc = transform_tensor_descriptor(
        padded_spatial_desc,
        make_tuple(make_pass_through_transform(N),
                   make_embed_transform(make_tuple(YDot, HTilde), make_tuple(HRatio, 1)),
                   make_embed_transform(make_tuple(XDot, WTilde), make_tuple(WRatio, 1)),
                   make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto sliced_desc = transform_tensor_descriptor(
        embedded_desc,
        make_tuple(make_pass_through_transform(N),
                   make_slice_transform(YDot, Number<0>{}, YDotSlice),
                   make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                   make_slice_transform(XDot, Number<0>{}, XDotSlice),
                   make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                   make_pass_through_transform(K)),
        make_tuple(Sequence<0>{},
                   Sequence<1>{},
                   Sequence<2>{},
                   Sequence<3>{},
                   Sequence<4>{},
                   Sequence<5>{}),
        make_tuple(Sequence<0>{},
                   Sequence<1>{},
                   Sequence<2>{},
                   Sequence<3>{},
                   Sequence<4>{},
                   Sequence<5>{}));

    const auto raw_gemm_desc = transform_tensor_descriptor(
        sliced_desc,
        make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K)),
                   make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice))),
        make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    const auto padded_gemm_desc =
        transform_tensor_descriptor(raw_gemm_desc,
                                    make_tuple(make_pad_transform(KRaw, Number<0>{}, KPad),
                                               make_pad_transform(MRaw, Number<0>{}, MPad)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}));

    const auto generic_desc =
        transform_tensor_descriptor(padded_gemm_desc,
                                    make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                               make_pass_through_transform(MPadded)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    for(index_t k0 = 0; k0 < K0; ++k0)
    {
        for(index_t m = 0; m < MPadded; ++m)
        {
            for(index_t k1 = 0; k1 < K1; ++k1)
            {
                const auto index              = make_tuple(k0, m, k1);
                const auto custom_coordinate  = make_tensor_coordinate(custom_desc, index);
                const auto generic_coordinate = make_tensor_coordinate(generic_desc, index);

                SCOPED_TRACE(::testing::Message() << "k0=" << k0 << ", m=" << m << ", k1=" << k1);
                EXPECT_EQ(custom_coordinate.GetOffset(), generic_coordinate.GetOffset());
                EXPECT_EQ(coordinate_has_valid_offset_assuming_visible_index_is_valid(
                              custom_desc, custom_coordinate),
                          coordinate_has_valid_offset_assuming_visible_index_is_valid(
                              generic_desc, generic_coordinate));
            }
        }
    }
}

TEST(TensorDescriptorFunctorsIntegration, ZeroPadTransformValidityBounds)
{
    constexpr index_t Length = 5;
    const auto index_pad     = make_pad_transform(Length, Number<0>{}, Number<0>{});
    const auto long_pad      = make_pad_transform(long_index_t{Length}, Number<0>{}, Number<0>{});

    EXPECT_FALSE(index_pad.IsValidUpperIndexMappedToValidLowerIndex(make_multi_index(-1)));
    EXPECT_TRUE(index_pad.IsValidUpperIndexMappedToValidLowerIndex(make_multi_index(0)));
    EXPECT_TRUE(index_pad.IsValidUpperIndexMappedToValidLowerIndex(make_multi_index(Length - 1)));
    EXPECT_FALSE(index_pad.IsValidUpperIndexMappedToValidLowerIndex(make_multi_index(Length)));

    EXPECT_FALSE(
        long_pad.IsValidUpperIndexMappedToValidLowerIndex(make_multi_index<long_index_t>(-1)));
    EXPECT_TRUE(
        long_pad.IsValidUpperIndexMappedToValidLowerIndex(make_multi_index<long_index_t>(0)));
    EXPECT_TRUE(long_pad.IsValidUpperIndexMappedToValidLowerIndex(
        make_multi_index<long_index_t>(Length - 1)));
    EXPECT_FALSE(
        long_pad.IsValidUpperIndexMappedToValidLowerIndex(make_multi_index<long_index_t>(Length)));
}
