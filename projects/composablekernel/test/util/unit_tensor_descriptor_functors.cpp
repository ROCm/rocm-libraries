// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <cstdint>
#include <random>
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

struct ForwardConvCase
{
    index_t N, Hi, Wi, C, Y, X;
    index_t StrideH, StrideW, DilationH, DilationW;
    index_t PadHLeft, PadHRight, PadWLeft, PadWRight;
};

void CheckForwardConvInputTransformMatchesGeneric(const ForwardConvCase& cfg)
{
    const index_t N         = cfg.N;
    const index_t Hi        = cfg.Hi;
    const index_t Wi        = cfg.Wi;
    const index_t C         = cfg.C;
    const index_t Y         = cfg.Y;
    const index_t X         = cfg.X;
    const index_t StrideH   = cfg.StrideH;
    const index_t StrideW   = cfg.StrideW;
    const index_t DilationH = cfg.DilationH;
    const index_t DilationW = cfg.DilationW;
    const index_t PadHLeft  = cfg.PadHLeft;
    const index_t PadHRight = cfg.PadHRight;
    const index_t PadWLeft  = cfg.PadWLeft;
    const index_t PadWRight = cfg.PadWRight;
    const index_t Ho        = (Hi + PadHLeft + PadHRight - DilationH * (Y - 1) - 1) / StrideH + 1;
    const index_t Wo        = (Wi + PadWLeft + PadWRight - DilationW * (X - 1) - 1) / StrideW + 1;

    SCOPED_TRACE(::testing::Message()
                 << "N=" << N << " Hi=" << Hi << " Wi=" << Wi << " C=" << C << " Y=" << Y << " X="
                 << X << " stride=" << StrideH << "x" << StrideW << " dilation=" << DilationH << "x"
                 << DilationW << " padH=" << PadHLeft << "/" << PadHRight << " padW=" << PadWLeft
                 << "/" << PadWRight << " Ho=" << Ho << " Wo=" << Wo);

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

    // Upper indices beyond M and K are also checked: GEMM padding produces them and the custom
    // transform must behave exactly like the generic merge/embed/pad chain there.
    const index_t M      = N * Ho * Wo;
    const index_t K      = Y * X * C;
    const index_t M_ext  = M + Ho * Wo + 1;
    const index_t K_ext  = K + X * C + 1;
    int64_t mismatches   = 0;
    index_t first_bad_m  = -1;
    index_t first_bad_k  = -1;
    auto record_mismatch = [&](index_t m, index_t k) {
        if(mismatches++ == 0)
        {
            first_bad_m = m;
            first_bad_k = k;
        }
    };

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
                record_mismatch(m, k);
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "first mismatch at m=" << first_bad_m << " k=" << first_bad_k;

    // Moving a coordinate (UpdateLowerIndex path used by the kernels) must agree with building
    // the coordinate from scratch. Steps go forward and backward, change one or both dimensions.
    const index_t dms[] = {0, 1, -1, 3, -3, Wo, -Wo, Ho * Wo, -Ho * Wo, 16, -16};
    const index_t dks[] = {0, 1, -1, C, -C, X * C, -X * C, 4, -4, 8, -8, 32, -32};
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
void CheckForwardConvInputTransformLowerIndices(const ForwardConvCase& cfg)
{
    const int64_t Ho = (int64_t(cfg.Hi) + cfg.PadHLeft + cfg.PadHRight -
                        int64_t(cfg.DilationH) * (cfg.Y - 1) - 1) /
                           cfg.StrideH +
                       1;
    const int64_t Wo = (int64_t(cfg.Wi) + cfg.PadWLeft + cfg.PadWRight -
                        int64_t(cfg.DilationW) * (cfg.X - 1) - 1) /
                           cfg.StrideW +
                       1;
    const int64_t M = int64_t(cfg.N) * Ho * Wo;
    const int64_t K = int64_t(cfg.Y) * cfg.X * cfg.C;
    ASSERT_LT(M, int64_t(1) << 31);

    SCOPED_TRACE(::testing::Message()
                 << "N=" << cfg.N << " Hi=" << cfg.Hi << " Wi=" << cfg.Wi << " C=" << cfg.C
                 << " Y=" << cfg.Y << " X=" << cfg.X << " stride=" << cfg.StrideH << "x"
                 << cfg.StrideW << " dilation=" << cfg.DilationH << "x" << cfg.DilationW
                 << " padH=" << cfg.PadHLeft << "/" << cfg.PadHRight << " padW=" << cfg.PadWLeft
                 << "/" << cfg.PadWRight << " Ho=" << Ho << " Wo=" << Wo);

    const auto transform = make_conv_fwd_data_in_transform(cfg.N,
                                                           cfg.Hi,
                                                           cfg.Wi,
                                                           cfg.C,
                                                           cfg.Y,
                                                           cfg.X,
                                                           Ho,
                                                           Wo,
                                                           cfg.StrideH,
                                                           cfg.StrideW,
                                                           cfg.DilationH,
                                                           cfg.DilationW,
                                                           cfg.PadHLeft,
                                                           cfg.PadWLeft);

    // Upper-index boundaries around every divisor, plus values up to the largest dividend
    // (2^31 - 1) magic division supports; random samples cover the rest.
    std::vector<int64_t> ms{0,
                            1,
                            Wo - 1,
                            Wo,
                            Wo + 1,
                            Ho * Wo - 1,
                            Ho * Wo,
                            Ho * Wo + 1,
                            M - 1,
                            M,
                            INT32_MAX - 1,
                            INT32_MAX};
    std::vector<int64_t> ks{0,
                            1,
                            cfg.C - 1,
                            cfg.C,
                            cfg.C + 1,
                            int64_t(cfg.X) * cfg.C - 1,
                            int64_t(cfg.X) * cfg.C,
                            K - 1,
                            K,
                            K + int64_t(cfg.X) * cfg.C};
    std::mt19937_64 rng(7);
    for(int i = 0; i < 2000; ++i)
    {
        ms.push_back(rng() % (int64_t(INT32_MAX) + 1));
        ks.push_back(rng() % (K + int64_t(cfg.X) * cfg.C));
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

            const int64_t n  = m / (Wo * Ho);
            const int64_t ho = (m / Wo) % Ho;
            const int64_t wo = m % Wo;
            const int64_t y  = k / (int64_t(cfg.X) * cfg.C);
            const int64_t x  = (k / cfg.C) % cfg.X;
            const int64_t c  = k % cfg.C;
            const int64_t hi = ho * cfg.StrideH + y * cfg.DilationH - cfg.PadHLeft;
            const int64_t wi = wo * cfg.StrideW + x * cfg.DilationW - cfg.PadWLeft;
            const bool valid = hi >= 0 && hi < cfg.Hi && wi >= 0 && wi < cfg.Wi;

            // The transform computes lower indices in 32-bit arithmetic; a lower index outside
            // the int32 range (only reachable with absurd dilation/pad past the filter) is not
            // representable and therefore outside the supported range.
            if(hi < INT32_MIN || hi > INT32_MAX || wi < INT32_MIN || wi > INT32_MAX)
            {
                continue;
            }

            MultiIndex<4> low;
            transform.CalculateLowerIndex(low, up);

            mismatches += low[Number<0>{}] != n;
            mismatches += low[Number<1>{}] != hi;
            mismatches += low[Number<2>{}] != wi;
            mismatches += low[Number<3>{}] != c;
            mismatches += transform.IsValidUpperIndexMappedToValidLowerIndex(up) != valid;
        }
    }
    EXPECT_EQ(mismatches, 0);
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, ForwardConvInputTransformMatchesGeneric)
{
    // {N, Hi, Wi, C, Y, X, StrideH, StrideW, DilationH, DilationW,
    //  PadHLeft, PadHRight, PadWLeft, PadWRight}
    const ForwardConvCase cases[] = {
        // original configuration
        {1, 7, 7, 2, 3, 3, 2, 2, 2, 2, 1, 5, 2, 4},
        // N > 1 with every H/W parameter different (Ho != Wo, Y != X, ...)
        {3, 9, 13, 5, 3, 2, 2, 3, 1, 2, 1, 4, 3, 0},
        {2, 8, 10, 4, 2, 5, 2, 1, 3, 2, 0, 1, 2, 0},
        {5, 14, 11, 6, 3, 3, 1, 2, 1, 1, 1, 1, 1, 1},
        // different channel counts: 1, odd, non power of two, power of two
        {2, 6, 5, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 6, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 6, 5, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 6, 5, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        // 1x1 filters, with and without stride
        {2, 5, 6, 4, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {2, 5, 6, 4, 1, 1, 2, 3, 1, 1, 0, 0, 0, 0},
        // Ho == Wo == 1
        {3, 3, 3, 4, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0},
        // padding larger than the input
        {2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 4, 4, 5, 5},
        // right padding not fully consumed by the stride
        {2, 7, 7, 3, 3, 3, 3, 3, 1, 1, 1, 3, 1, 3},
        // large stride and dilation
        {2, 31, 29, 3, 3, 3, 4, 5, 6, 7, 6, 6, 7, 7},
    };

    for(const auto& cfg : cases)
    {
        CheckForwardConvInputTransformMatchesGeneric(cfg);
    }

    // Extreme configurations, checked against an int64 reference (offsets would overflow here).
    const ForwardConvCase extreme_cases[] = {
        // M just below 2^31 and a large channel count
        {32767, 256, 256, 4096, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        // power-of-two divisors everywhere
        {4, 16, 16, 64, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0},
        // divisors of the form 2^k + 1 and 2^k
        {2, 33, 17, 65, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 2, 2, 65536, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
        // large divisors: Wo = 65537, C = 65537
        {1, 1, 65537, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {2, 3, 3, 65537, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        // very negative lower indices; only hi == 0 / wi == 0 is valid
        {1, 4, 4, 1, 1, 1, 1 << 28, 1 << 28, 1, 1, 1 << 30, 0, 1 << 30, 0},
        // very large positive lower indices; only hi == 0 / wi == 0 is valid
        {1, 4, 4, 1, 1, 1, 1 << 28, 1 << 28, 1, 1, 0, 1 << 30, 0, 1 << 30},
        {1, 4, 4, 1, 1, 1, 1 << 28, 1 << 28, 1, 1, 0, 7 << 28, 0, 7 << 28},
        // lower indices down to -(2^31 - 1)
        {1, 4, 4, 1, 1, 1, 1 << 28, 1 << 28, 1, 1, INT32_MAX, 0, INT32_MAX, 0},
        // extents at the 32-bit limit: Hi = Wi = INT32_MAX, filter taps reaching Hi - 1 and Hi
        {1, INT32_MAX, INT32_MAX, 1, 2, 2, 1, 1, INT32_MAX - 1, INT32_MAX - 1, 0, 0, 0, 0},
        {1, INT32_MAX, INT32_MAX, 1, 2, 2, 1, 1, INT32_MAX, INT32_MAX, 0, 1, 0, 1},
        // Wo = INT32_MAX (largest magic-division divisor) with M = INT32_MAX
        {1, 1, INT32_MAX, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        // K = Y * X * C just below 2^31, with X = 1 (divisor 1) and a huge C divisor
        {1, 3, 1, 715827882, 3, 1, 1, 1, 1, 1, 0, 0, 0, 0},
    };

    for(const auto& cfg : extreme_cases)
    {
        CheckForwardConvInputTransformLowerIndices(cfg);
    }
}
