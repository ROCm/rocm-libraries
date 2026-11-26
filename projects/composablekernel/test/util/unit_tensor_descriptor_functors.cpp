// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <cstdint>
#include <random>
#include <sstream>
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
// Equality checks are exhaustive on small problems. On huge problems (where the generic chain
// itself overflows) the transforms are compared with an int64 reference derived from the definition
// of the convolution on boundary values and random samples: this is NOT exhaustive.
// =============================================================================

// =============================================================================
// Shared helpers
// =============================================================================

namespace {

constexpr int64_t kMaxIndex = kIndexMax;

// Calls f(std::integral_constant<index_t, D>{}) for D = 1, 2, 3 spatial dimensions
template <typename F>
void ForEachSpatialDim(F&& f)
{
    f(std::integral_constant<index_t, 1>{});
    f(std::integral_constant<index_t, 2>{});
    f(std::integral_constant<index_t, 3>{});
}

// Calls f(case) twice: with G = 1 (a packed tensor) and with G = 3 (a tensor with interleaved
// groups: NHWGC). The descriptors describe one group, so G only changes the strides.
template <typename Case, typename F>
void ForEachGroupCount(Case c, F&& f)
{
    for(const index_t groups : {1, 3})
    {
        c.G = groups;
        f(c);
    }
}

constexpr int64_t RoundUp(int64_t x, int64_t multiple)
{
    return (x + multiple - 1) / multiple * multiple;
}

// Output length of a convolution along one dimension (64 bit: a huge dilation overflows 32 bit)
constexpr int64_t ConvOutputLength(int64_t in,
                                   int64_t left_pad,
                                   int64_t right_pad,
                                   int64_t dilation,
                                   int64_t filter,
                                   int64_t stride)
{
    return (in + left_pad + right_pad - dilation * (filter - 1) - 1) / stride + 1;
}

// Smallest right pad that gives at least one output along a dimension, plus `extra`
constexpr index_t
RightPadForOutput(index_t in, index_t filter, index_t dilation, index_t left_pad, index_t extra)
{
    return std::max<index_t>(0, dilation * (filter - 1) + 1 - in - left_pad) + extra;
}

template <index_t NDim>
auto ToMultiIndex(const std::array<index_t, NDim>& a)
{
    MultiIndex<NDim> r;
    static_for<0, NDim, 1>{}([&](auto i) { r(i) = a[i]; });
    return r;
}

template <typename T, std::size_t N>
Array<long_index_t, static_cast<index_t>(N)> ToLongArray(const std::array<T, N>& a)
{
    Array<long_index_t, static_cast<index_t>(N)> r{};
    for(std::size_t d = 0; d < N; ++d)
    {
        r(static_cast<index_t>(d)) = a[d];
    }
    return r;
}

// Fixed-seed random numbers in [lo, hi]
struct SeededRandom
{
    explicit SeededRandom(uint32_t seed) : rng(seed) {}
    index_t operator()(index_t lo, index_t hi)
    {
        return lo + static_cast<index_t>(rng() % static_cast<uint32_t>(hi - lo + 1));
    }
    std::mt19937 rng;
};

// ---------------------------------------------------------------------------------------------
// Convolution problems
// ---------------------------------------------------------------------------------------------

// Forward convolution problem. The input tensor is N spatial... G C (NHWGC for 2D) and the
// descriptors describe one group, so G only changes the strides: the channels of the other groups
// are interleaved.
template <index_t NDim>
struct ForwardConvCase
{
    static constexpr index_t kNDim = NDim;
    index_t N, C;
    std::array<index_t, NDim> In, Filter, Stride, Dilation, PadLeft, PadRight;
    index_t G = 1;
};

// Backward weight convolution problem. K is the number of output channels and BatchK the split-K
// factor; G as above.
template <index_t NDim>
struct BwdWeightCase
{
    static constexpr index_t kNDim = NDim;
    index_t N, C, K;
    std::array<index_t, NDim> In, Filter, Stride, Dilation, PadLeft, PadRight;
    index_t BatchK;
    index_t G = 1;
};

// Backward data convolution problem (2D, the only dimension with a custom transform)
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

template <typename Case>
std::array<int64_t, Case::kNDim> OutputLengths(const Case& c)
{
    std::array<int64_t, Case::kNDim> out{};
    for(index_t d = 0; d < Case::kNDim; ++d)
    {
        out[d] = ConvOutputLength(
            c.In[d], c.PadLeft[d], c.PadRight[d], c.Dilation[d], c.Filter[d], c.Stride[d]);
    }
    return out;
}

template <index_t NDim>
std::string ExtraDescription(const ForwardConvCase<NDim>&)
{
    return "";
}

template <index_t NDim>
std::string ExtraDescription(const BwdWeightCase<NDim>& c)
{
    return " K=" + std::to_string(c.K) + " BatchK=" + std::to_string(c.BatchK);
}

// Text for SCOPED_TRACE
template <typename Case>
std::string Describe(const Case& c)
{
    const auto out = OutputLengths(c);
    std::ostringstream os;
    os << Case::kNDim << "D N=" << c.N << " C=" << c.C << " G=" << c.G << ExtraDescription(c);
    for(index_t d = 0; d < Case::kNDim; ++d)
    {
        os << " | in=" << c.In[d] << " filter=" << c.Filter[d] << " stride=" << c.Stride[d]
           << " dilation=" << c.Dilation[d] << " pad=" << c.PadLeft[d] << "/" << c.PadRight[d]
           << " out=" << out[d];
    }
    return os.str();
}

std::string Describe(const BwdDataCase& c)
{
    std::ostringstream os;
    os << "N=" << c.N << " G=" << c.G << " C=" << c.C << " K=" << c.K << " In=" << c.Hi << "x"
       << c.Wi << " Filter=" << c.Y << "x" << c.X << " Stride=" << c.StrideH << "x" << c.StrideW
       << " Dilation=" << c.DilationH << "x" << c.DilationW << " Pad=" << c.LeftPadH << ","
       << c.LeftPadW << "," << c.RightPadH << "," << c.RightPadW;
    return os.str();
}

// Sizes derived from the problem. The implicit im2col matrix has the rows (n, o_0..) and the
// columns (f_0.., c): M x K with M = N * prod(O) and K = C * prod(F). They are the GemmM and GemmK
// of the forward A matrix and the GemmK and GemmN of the backward weight B matrix.
template <index_t NDim>
struct ConvGeometry
{
    std::array<int64_t, NDim> out64{};
    std::array<index_t, NDim> out{};
    int64_t out_spatial = 1; // Ho * Wo * ...
    int64_t filter_size = 1; // Y * X * ...
    int64_t M           = 0;
    int64_t K           = 0;
    bool empty          = false; // some output dimension is empty: not a valid problem
};

template <typename Case>
ConvGeometry<Case::kNDim> MakeGeometry(const Case& c)
{
    ConvGeometry<Case::kNDim> g;
    g.out64 = OutputLengths(c);
    for(index_t d = 0; d < Case::kNDim; ++d)
    {
        g.empty |= g.out64[d] < 1;
        g.out[d] = static_cast<index_t>(g.out64[d]);
        g.out_spatial *= g.out64[d];
        g.filter_size *= c.Filter[d];
    }
    g.M = c.N * g.out_spatial;
    g.K = c.C * g.filter_size;
    return g;
}

// Strides (in elements) of an input tensor packed as N C spatial...: one spatial dimension has
// stride 1 (the innermost) and the others are packed outside it. The coordinates that can reach the
// farthest are the ones multiplied by the biggest strides, so the limit tests use both the first
// and the last spatial dimension as the innermost one.
template <index_t NDim>
struct ChannelFirstStrides
{
    long_index_t n = 0, c = 0;
    Array<long_index_t, NDim> spatial{};
    long_index_t elements = 0; // size of the whole tensor

    ChannelFirstStrides() = default;
    ChannelFirstStrides(const std::array<index_t, NDim>& in,
                        int64_t channels,
                        int64_t batch,
                        bool first_dim_innermost)
    {
        long_index_t acc = 1;
        for(index_t i = 0; i < NDim; ++i)
        {
            const index_t d = first_dim_innermost ? i : NDim - 1 - i;
            spatial(d)      = acc;
            acc *= in[d];
        }
        c        = acc;
        n        = acc * channels;
        elements = n * batch;
    }

    // [n, spatial..., c]: the strides of the (N, spatial..., C) descriptor of MakeInputDescriptor
    std::array<index_t, NDim + 2> NSpatialC() const
    {
        std::array<index_t, NDim + 2> s{};
        s[0] = static_cast<index_t>(n);
        for(index_t d = 0; d < NDim; ++d)
        {
            s[1 + d] = static_cast<index_t>(spatial[d]);
        }
        s[NDim + 1] = static_cast<index_t>(c);
        return s;
    }

    // [group, n, c, spatial...]: the stride order of the G N C spatial... tensors of the ops
    std::array<index_t, NDim + 3> GroupNCSpatial(long_index_t group_stride = 0) const
    {
        std::array<index_t, NDim + 3> s{};
        s[0] = static_cast<index_t>(group_stride);
        s[1] = static_cast<index_t>(n);
        s[2] = static_cast<index_t>(c);
        for(index_t d = 0; d < NDim; ++d)
        {
            s[3 + d] = static_cast<index_t>(spatial[d]);
        }
        return s;
    }

    // Offset of an input element (int64)
    int64_t Offset(int64_t batch, int64_t channel, const std::array<int64_t, NDim>& idx) const
    {
        int64_t offset = batch * n + channel * c;
        for(index_t d = 0; d < NDim; ++d)
        {
            offset += idx[d] * spatial[d];
        }
        return offset;
    }

    // Number of times the 32-bit offset computation overflows for the lower index
    // (batch, idx..., channel): a product or the running sum leaving the int32 range. The 32-bit
    // arithmetic is emulated with 64 bit.
    int OverflowCount(int64_t batch, const std::array<int64_t, NDim>& idx, int64_t channel) const
    {
        int overflows  = 0;
        int64_t sum    = 0;
        const auto add = [&](int64_t index, int64_t stride) {
            const int64_t term = index * stride;
            overflows += std::llabs(term) > kMaxIndex;
            sum += term;
            overflows += std::llabs(sum) > kMaxIndex;
        };
        add(batch, n);
        for(index_t d = 0; d < NDim; ++d)
        {
            add(idx[d], spatial[d]);
        }
        add(channel, c);
        return overflows;
    }
};

// int64 reference of the implicit im2col: the GEMM position (m, k) with m = (n, o_0..) and
// k = (f_0.., c) reads the input element (n, i_0.., c) with
// i_d = o_d * stride_d + f_d * dilation_d - left_pad_d. It follows the definition of the
// convolution and does not use any transform. Rows beyond M and columns beyond K are GEMM padding.
template <index_t NDim>
class Im2colReference
{
    public:
    struct Position
    {
        int64_t n = 0, c = 0;
        std::array<int64_t, NDim> in_idx{};
        bool in_bounds  = true; // every i_d is inside the input (otherwise: convolution padding)
        bool fits_int32 = true; // every i_d can be represented in 32 bit
        bool real       = true; // m < M and k < K (otherwise: GEMM padding)
        bool valid      = true; // real and in_bounds: the element is read from the input
    };

    template <typename Case>
    explicit Im2colReference(const Case& c)
        : channels_(c.C),
          in_(ToInt64(c.In)),
          filter_(ToInt64(c.Filter)),
          stride_(ToInt64(c.Stride)),
          dilation_(ToInt64(c.Dilation)),
          left_pad_(ToInt64(c.PadLeft))
    {
        const auto g = MakeGeometry(c);
        out_         = g.out64;
        M_           = g.M;
        K_           = g.K;
    }

    Position At(int64_t m, int64_t k) const
    {
        Position p;
        std::array<int64_t, NDim> o{}, f{};
        int64_t q = m;
        for(index_t d = NDim - 1; d >= 0; --d)
        {
            o[d] = q % out_[d];
            q /= out_[d];
        }
        p.n = q;
        q   = k;
        p.c = q % channels_;
        q /= channels_;
        for(index_t d = NDim - 1; d >= 1; --d)
        {
            f[d] = q % filter_[d];
            q /= filter_[d];
        }
        f[0] = q;
        for(index_t d = 0; d < NDim; ++d)
        {
            p.in_idx[d] = o[d] * stride_[d] + f[d] * dilation_[d] - left_pad_[d];
            p.in_bounds &= p.in_idx[d] >= 0 && p.in_idx[d] < in_[d];
            p.fits_int32 &= p.in_idx[d] >= kIndexMin && p.in_idx[d] <= kMaxIndex;
        }
        p.real  = m < M_ && k < K_;
        p.valid = p.real && p.in_bounds;
        return p;
    }

    private:
    static std::array<int64_t, NDim> ToInt64(const std::array<index_t, NDim>& a)
    {
        std::array<int64_t, NDim> r{};
        for(index_t d = 0; d < NDim; ++d)
        {
            r[d] = a[d];
        }
        return r;
    }

    int64_t channels_, M_, K_;
    std::array<int64_t, NDim> in_, filter_, stride_, dilation_, left_pad_, out_;
};

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

struct DescriptorComparison
{
    int64_t positions  = 0;
    int64_t valid      = 0; // positions that the generic chain reports as valid
    int64_t mismatches = 0;
};

// Compares two 3D descriptors of the same lengths (the custom transform and the generic chain) at
// the given positions of each dimension: the validity of the offset and, if valid, the offset.
template <typename Custom, typename Generic>
DescriptorComparison CompareDescriptors3D(const Custom& custom,
                                          const Generic& generic,
                                          const std::vector<index_t>& positions0,
                                          const std::vector<index_t>& positions1,
                                          const std::vector<index_t>& positions2)
{
    DescriptorComparison result;
    for(const index_t i0 : positions0)
    {
        for(const index_t i1 : positions1)
        {
            for(const index_t i2 : positions2)
            {
                const auto index = make_multi_index(i0, i1, i2);
                const auto cc    = make_tensor_coordinate(custom, index);
                const auto gc    = make_tensor_coordinate(generic, index);
                const bool cv    = coordinate_has_valid_offset(custom, cc);
                const bool gv    = coordinate_has_valid_offset(generic, gc);
                ++result.positions;
                result.valid += gv;
                if(cv != gv || (cv && cc.GetOffset() != gc.GetOffset()))
                {
                    if(result.mismatches++ < 3)
                    {
                        ADD_FAILURE() << "(" << i0 << "," << i1 << "," << i2
                                      << ") custom valid=" << cv << " offset=" << cc.GetOffset()
                                      << ", generic valid=" << gv << " offset=" << gc.GetOffset();
                    }
                }
            }
        }
    }
    return result;
}

// All the positions of a dimension
std::vector<index_t> AllPositions(index_t length)
{
    std::vector<index_t> positions(length);
    for(index_t i = 0; i < length; ++i)
    {
        positions[i] = i;
    }
    return positions;
}

// The first `head` and the last `tail` positions of a dimension (the GEMM padding is at the end)
// and `random` random ones
std::vector<index_t>
SamplePositions(index_t length, index_t head, index_t tail, int random, std::mt19937& rng)
{
    std::vector<index_t> positions;
    for(index_t i = 0; i < head && i < length; ++i)
    {
        positions.push_back(i);
    }
    for(index_t i = 0; i < tail && i < length; ++i)
    {
        positions.push_back(length - 1 - i);
    }
    for(int i = 0; i < random; ++i)
    {
        positions.push_back(static_cast<index_t>(rng() % length));
    }
    return positions;
}

// A problem with the filter, stride, dilation and pad of every dimension depending on `y`, a
// filter size from 1 to 16 (odd and even, in a different dimension each time) and small inputs, so
// that every position of the GEMM can be compared.
template <typename Case>
Case MakeFilterSweepCase(index_t y)
{
    constexpr index_t NDim  = Case::kNDim;
    const index_t strides[] = {1, 2, 3}, dilations[] = {1, 2}, left_pads[] = {0, 1, 3};
    Case c{};
    c.N = 1 + y % 2;
    c.C = 3 + y % 5;
    c.G = y % 2 ? 3 : 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        c.In[d]       = d == 0 ? 10 + y % 3 : d == 1 ? 5 + y % 2 : 4;
        c.Filter[d]   = d == y % NDim ? y : 1 + (y * 5 + d) % 4;
        c.Stride[d]   = strides[(y + d) % 3];
        c.Dilation[d] = dilations[(y + d) % 2];
        c.PadLeft[d]  = left_pads[(y + d) % 3];
        c.PadRight[d] =
            RightPadForOutput(c.In[d], c.Filter[d], c.Dilation[d], c.PadLeft[d], (y + d) % 3);
    }
    return c;
}

// A random small problem
template <typename Case>
Case MakeRandomCase(SeededRandom& rnd)
{
    constexpr index_t NDim = Case::kNDim;
    Case c{};
    c.N = rnd(1, 3);
    c.C = rnd(1, 9);
    c.G = rnd(0, 1) ? 3 : 1;
    for(index_t d = 0; d < NDim; ++d)
    {
        c.In[d]       = rnd(1, NDim == 3 ? 5 : 9);
        c.Filter[d]   = rnd(1, NDim == 3 ? 4 : 8);
        c.Stride[d]   = rnd(1, 3);
        c.Dilation[d] = rnd(1, 3);
        c.PadLeft[d]  = rnd(0, 4);
        c.PadRight[d] =
            RightPadForOutput(c.In[d], c.Filter[d], c.Dilation[d], c.PadLeft[d], rnd(0, 3));
    }
    return c;
}

// A problem in which one spatial dimension (the "reach" dimension, the first or the last one) has
// a filter whose taps are spread by a big dilation, so that the lower coordinates reach far out of
// the input, while the other dimensions are small. The right pad is D * (F - 1), clamped to the
// index range, so that Wo = Wi + left_pad. These are the problems at the limits of the host checks.
struct FarReachSpec
{
    index_t n         = 1;
    index_t channels  = 8;
    index_t reach_dim = 0;
    // reach dimension
    index_t in       = 8;
    index_t filter   = 3;
    int64_t dilation = 1;
    index_t left_pad = 0;
    index_t stride   = 1;
    // the other dimensions (the padding is the same on both sides)
    index_t other_in     = 3;
    index_t other_filter = 2;
    index_t other_stride = 2;
    index_t other_pad    = 1;
};

template <typename Case>
Case MakeFarReachCase(const FarReachSpec& s)
{
    Case c{};
    c.N = s.n;
    c.C = s.channels;
    for(index_t d = 0; d < Case::kNDim; ++d)
    {
        const bool reach = d == s.reach_dim;
        c.In[d]          = reach ? s.in : s.other_in;
        c.Filter[d]      = reach ? s.filter : s.other_filter;
        c.Stride[d]      = reach ? s.stride : s.other_stride;
        c.Dilation[d]    = reach ? static_cast<index_t>(s.dilation) : 1;
        c.PadLeft[d]     = reach ? s.left_pad : s.other_pad;
        c.PadRight[d] =
            reach ? static_cast<index_t>(std::min<int64_t>(s.dilation * (s.filter - 1), kMaxIndex))
                  : s.other_pad;
    }
    return c;
}

} // namespace

// =============================================================================
// Forward A matrix: ConvFwdDataImplicitGemmInTransform against the generic chain
// =============================================================================

namespace {

// Input descriptor (N, spatial..., C) of one group of an N spatial G C tensor, optionally with
// other strides
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
    return make_naive_tensor_descriptor(
        generate_tuple([&](auto i) { return lengths[decltype(i)::value]; }, Number<NDim + 2>{}),
        generate_tuple([&](auto i) { return strides[decltype(i)::value]; }, Number<NDim + 2>{}));
}

// The pad -> embed -> merge chain the custom transform replaces. Its lower dimensions are the
// (N, spatial..., C) input; its upper dimensions are M = (N, O_0..) and K = (F_0.., C).
template <index_t NDim>
auto MakeGenericDescriptor(const ForwardConvCase<NDim>& cfg, const std::array<index_t, NDim>& out)
{
    const index_t N = cfg.N;
    const index_t C = cfg.C;
    // lower dimension i -> upper dimension i
    const auto identity =
        generate_tuple([](auto i) { return Sequence<decltype(i)::value>{}; }, Number<NDim + 2>{});

    // 1. pad the spatial dimensions: (N, I_0.., C) -> (N, Ip_0.., C)
    const auto padded =
        transform_tensor_descriptor(MakeInputDescriptor(cfg),
                                    generate_tuple(
                                        [&](auto i) {
                                            constexpr index_t I = decltype(i)::value;
                                            if constexpr(I == 0)
                                                return make_pass_through_transform(N);
                                            else if constexpr(I == NDim + 1)
                                                return make_pass_through_transform(C);
                                            else
                                                return make_pad_transform(cfg.In[I - 1],
                                                                          cfg.PadLeft[I - 1],
                                                                          cfg.PadRight[I - 1]);
                                        },
                                        Number<NDim + 2>{}),
                                    identity,
                                    identity);

    // 2. embed: every padded spatial dimension d becomes the pair (filter tap f_d, output index
    // o_d), at the dimensions (2d + 1, 2d + 2) of the result; N and C pass through
    const auto embedded = transform_tensor_descriptor(
        padded,
        generate_tuple(
            [&](auto i) {
                constexpr index_t I = decltype(i)::value;
                if constexpr(I == 0)
                    return make_pass_through_transform(N);
                else if constexpr(I == NDim + 1)
                    return make_pass_through_transform(C);
                else
                    return make_embed_transform(make_tuple(cfg.Filter[I - 1], out[I - 1]),
                                                make_tuple(cfg.Dilation[I - 1], cfg.Stride[I - 1]));
            },
            Number<NDim + 2>{}),
        identity,
        generate_tuple(
            [](auto i) {
                constexpr index_t I = decltype(i)::value;
                if constexpr(I == 0)
                    return Sequence<0>{};
                else if constexpr(I == NDim + 1)
                    return Sequence<2 * NDim + 1>{};
                else
                    return Sequence<2 * I - 1, 2 * I>{};
            },
            Number<NDim + 2>{}));

    // 3. merge: M = (N, O_0..) from the dimensions (0, 2, 4, ..), K = (F_0.., C) from (1, 3, 5, ..)
    return transform_tensor_descriptor(
        embedded,
        make_tuple(make_merge_transform(generate_tuple(
                       [&](auto i) {
                           constexpr index_t I = decltype(i)::value;
                           if constexpr(I == 0)
                               return N;
                           else
                               return out[I - 1];
                       },
                       Number<NDim + 1>{})),
                   make_merge_transform(generate_tuple(
                       [&](auto i) {
                           constexpr index_t I = decltype(i)::value;
                           if constexpr(I == NDim)
                               return C;
                           else
                               return cfg.Filter[I];
                       },
                       Number<NDim + 1>{}))),
        make_tuple(typename arithmetic_sequence_gen<0, 2 * NDim + 2, 2>::type{},
                   typename arithmetic_sequence_gen<1, 2 * NDim + 3, 2>::type{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));
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

// The (M, K) descriptor of the custom transform on the input tensor (optionally with other
// strides). m_padded / k_padded are the GEMM sizes padded to the block sizes.
template <index_t NDim>
auto MakeCustomDescriptor(const ForwardConvCase<NDim>& cfg,
                          const std::array<index_t, NDim>& out,
                          index_t m_padded,
                          index_t k_padded,
                          const std::array<index_t, NDim + 2>* strides = nullptr)
{
    return transform_tensor_descriptor(
        MakeInputDescriptor<NDim>(cfg, strides),
        make_tuple(MakeCustomTransform<NDim>(cfg, out, m_padded, k_padded)),
        make_tuple(typename arithmetic_sequence_gen<0, NDim + 2, 1>::type{}),
        make_tuple(Sequence<0, 1>{}));
}

// Compares the custom (M, K) descriptor with the generic chain on every position, also in the
// GEMM padding, and checks that moving a coordinate agrees with building a new one.
template <index_t NDim>
void CheckForwardConvInputTransformMatchesGeneric(const ForwardConvCase<NDim>& cfg)
{
    SCOPED_TRACE(Describe(cfg));

    const auto g            = MakeGeometry(cfg);
    const index_t M         = static_cast<index_t>(g.M);
    const index_t K         = static_cast<index_t>(g.K);
    const auto generic_desc = MakeGenericDescriptor<NDim>(cfg, g.out);

    // The transform is built with the padded GEMM lengths (as the GEMM padder would produce them)
    // and must report the padding rows / columns (m >= M or k >= K) as invalid itself, with the
    // same offsets and validity as the generic chain (which has no padding) inside the real M x K.
    const index_t M_ext    = M + static_cast<index_t>(g.out_spatial) + 1;
    const index_t K_ext    = K + cfg.C + 1;
    const auto custom_desc = MakeCustomDescriptor<NDim>(cfg, g.out, M_ext, K_ext);
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
                if((custom_valid != expect || !offset_ok) && mismatches++ == 0)
                {
                    first_bad_m = m;
                    first_bad_k = k;
                }
            }
        }
        EXPECT_EQ(mismatches, 0) << what << ": first mismatch at m=" << first_bad_m
                                 << " k=" << first_bad_k;
    };
    check_against_generic(custom_desc, "padded GEMM");
    check_against_generic(MakeCustomDescriptor<NDim>(cfg, g.out, M, K), "no GEMM padding");

    // Steps that carry the merged indices: one output row, the whole output, the channels and a
    // filter row times the channels
    const index_t out_last  = g.out[NDim - 1];
    const index_t filt_last = cfg.Filter[NDim - 1];
    EXPECT_EQ(CountCoordinateMoveMismatches(custom_desc,
                                            5000,
                                            2024,
                                            {{out_last, static_cast<index_t>(g.out_spatial)},
                                             {4, 32, cfg.C, filt_last * cfg.C}}),
              0);
}

// Checks the transform alone (lower index and validity) against the int64 reference. Used for
// index ranges where a tensor offset would overflow.
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

    const auto g = MakeGeometry(cfg);
    ASSERT_LT(g.M, int64_t(1) << 31);
    const Im2colReference<NDim> reference(cfg);
    const auto transform =
        MakeCustomTransform<NDim>(cfg, g.out, static_cast<index_t>(g.M), static_cast<index_t>(g.K));

    // Upper-index boundaries around every divisor of the two merges, plus values up to the largest
    // dividend (2^31 - 1) magic division supports; random samples cover the rest.
    std::vector<int64_t> ms{0, 1, g.M - 1, g.M, kMaxIndex - 1, kMaxIndex};
    std::vector<int64_t> ks{0, 1, cfg.C - 1, cfg.C, cfg.C + 1, g.K - 1, g.K, g.K + cfg.C};
    int64_t prod = 1;
    for(index_t d = NDim - 1; d >= 0; --d)
    {
        prod *= g.out64[d];
        for(const int64_t delta : {-1, 0, 1})
        {
            ms.push_back(prod + delta);
        }
    }
    prod = cfg.C;
    for(index_t d = NDim - 1; d >= 0; --d)
    {
        prod *= cfg.Filter[d];
        for(const int64_t delta : {-1, 0, 1})
        {
            ks.push_back(prod + delta);
        }
    }
    std::mt19937_64 rng(7);
    for(int i = 0; i < 2000; ++i)
    {
        ms.push_back(rng() % (kMaxIndex + 1));
        ks.push_back(rng() % (g.K + cfg.C));
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
            const auto p = reference.At(m, k);
            // The transform computes lower indices in 32-bit arithmetic; a lower index outside the
            // int32 range is not representable and therefore outside the supported range.
            if(!p.fits_int32)
            {
                continue;
            }
            const auto up = make_tuple(static_cast<index_t>(m), static_cast<index_t>(k));

            // rows / columns beyond the real M x K are GEMM padding: always invalid, and their
            // lower index is meaningless
            if(p.real)
            {
                MultiIndex<NDim + 2> low;
                transform.CalculateLowerIndex(low, up);
                mismatches += low[Number<0>{}] != p.n;
                mismatches += low[Number<NDim + 1>{}] != p.c;
                static_for<0, NDim, 1>{}([&](auto d) {
                    mismatches += static_cast<int64_t>(low[Number<d + 1>{}]) != p.in_idx[d];
                });
            }
            mismatches += transform.IsValidUpperIndexMappedToValidLowerIndex(up) != p.valid;
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
    // every case with a packed tensor (G = 1) and with interleaved groups (G = 3)
    const auto check = [](const auto& cfg) {
        ForEachGroupCount(cfg, [](const auto& c) {
            CheckForwardConvInputTransformMatchesGeneric<std::decay_t<decltype(c)>::kNDim>(c);
        });
    };
    for(const auto& cfg : cases1d)
    {
        check(cfg);
    }
    for(const auto& cfg : cases2d)
    {
        check(cfg);
    }
    for(const auto& cfg : cases3d)
    {
        check(cfg);
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

TEST(TensorDescriptorFunctorsIntegration, ForwardConvInputTransformWithFilters)
{
    // filters of 1 to 16 taps, odd and even, in every dimension, with strides, dilations, pads and
    // groups; every position of the GEMM is compared with the generic chain
    for(index_t y = 1; y <= 16; ++y)
    {
        ForEachSpatialDim([&](auto ndim) {
            constexpr index_t NDim = decltype(ndim)::value;
            CheckForwardConvInputTransformMatchesGeneric<NDim>(
                MakeFilterSweepCase<ForwardConvCase<NDim>>(y));
        });
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
auto MakeBwdDataTransformer(const BwdDataCase& c, index_t ytilde, index_t xtilde)
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

    const IndexType Ho = static_cast<IndexType>(
        ConvOutputLength(c.Hi, c.LeftPadH, c.RightPadH, c.DilationH, c.Y, c.StrideH));
    const IndexType Wo = static_cast<IndexType>(
        ConvOutputLength(c.Wi, c.LeftPadW, c.RightPadW, c.DilationW, c.X, c.StrideW));
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

// The A descriptor [K0, M, K1] of one group
template <typename IndexType, bool MergeOutPadding = false, bool SplitConvN = false>
auto MakeBwdDataADescriptor(const BwdDataCase& c, index_t ytilde, index_t xtilde)
{
    return MakeBwdDataTransformer<IndexType, MergeOutPadding, SplitConvN>(c, ytilde, xtilde)
        .MakeADescriptor_AK0_M_AK1();
}

// Calls f(ytilde, xtilde) for every group of a problem that has filter taps (the same condition as
// the device op)
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

struct BwdDataGroupResult
{
    bool supported = false; // accepted by the host check (always true without merged padding)
    DescriptorComparison comparison;
};

// Compares the A descriptor of the custom transform with the generic chain for one group, if the
// host check accepts the problem (the merged padding transform is not meant to be used otherwise).
// All the positions are compared if `exhaustive`, else the first rows / columns, the last rows /
// columns (where the GEMM padding is) and random ones, with every K1 index.
template <bool MergePadding, bool SplitConvN = false>
BwdDataGroupResult
CheckBwdDataGroup(const BwdDataCase& c, index_t ytilde, index_t xtilde, bool exhaustive)
{
    BwdDataGroupResult result;
    const auto transformer =
        MakeBwdDataTransformer<index_t, MergePadding, SplitConvN>(c, ytilde, xtilde);
    const auto custom = transformer.MakeADescriptor_AK0_M_AK1();
    const index_t k0  = custom.GetLength(Number<0>{});
    const index_t m   = custom.GetLength(Number<1>{});
    const index_t k1  = custom.GetLength(Number<2>{});

    result.supported = transformer.IsMergedOutPaddingRangeSupported(m);
    if(!result.supported)
    {
        return result;
    }

    const auto generic = MakeBwdDataADescriptor<long_index_t, false, SplitConvN>(c, ytilde, xtilde);
    EXPECT_EQ(k0, static_cast<index_t>(generic.GetLength(Number<0>{})));
    EXPECT_EQ(m, static_cast<index_t>(generic.GetLength(Number<1>{})));
    EXPECT_EQ(k1, static_cast<index_t>(generic.GetLength(Number<2>{})));

    std::mt19937 rng(13);
    result.comparison =
        exhaustive ? CompareDescriptors3D(
                         custom, generic, AllPositions(k0), AllPositions(m), AllPositions(k1))
                   : CompareDescriptors3D(custom,
                                          generic,
                                          SamplePositions(k0, 4, 4, 40, rng),
                                          SamplePositions(m, 4, 160, 40, rng),
                                          AllPositions(k1));
    EXPECT_EQ(CountCoordinateMoveMismatches(custom, 3000, 21), 0);
    return result;
}

struct BwdDataSummary
{
    int groups   = 0;
    int accepted = 0; // groups accepted by the host check
    DescriptorComparison comparison;
};

// CheckBwdDataGroup for every group of the problem; fails on any mismatch of an accepted group
template <bool MergePadding, bool SplitConvN = false>
BwdDataSummary CheckBwdDataAllGroups(const BwdDataCase& c, bool exhaustive)
{
    SCOPED_TRACE(Describe(c) + (MergePadding ? " merged padding" : " separate padding"));
    BwdDataSummary summary;
    ForEachBwdDataGroup(c, [&](index_t ytilde, index_t xtilde) {
        SCOPED_TRACE("ytilde=" + std::to_string(ytilde) + " xtilde=" + std::to_string(xtilde));
        const auto r = CheckBwdDataGroup<MergePadding, SplitConvN>(c, ytilde, xtilde, exhaustive);
        ++summary.groups;
        summary.accepted += r.supported;
        summary.comparison.positions += r.comparison.positions;
        summary.comparison.valid += r.comparison.valid;
        summary.comparison.mismatches += r.comparison.mismatches;
    });
    EXPECT_EQ(summary.comparison.mismatches, 0);
    return summary;
}

// True if the host check rejects at least one group of the problem
bool AnyBwdDataGroupRejected(const BwdDataCase& c)
{
    bool rejected = false;
    ForEachBwdDataGroup(c, [&](index_t ytilde, index_t xtilde) {
        const auto transformer = MakeBwdDataTransformer<index_t, true>(c, ytilde, xtilde);
        rejected |= !transformer.IsMergedOutPaddingRangeSupported(
            transformer.MakeADescriptor_AK0_M_AK1().GetLength(Number<1>{}));
    });
    return rejected;
}

// True if the host check rejects the problem for a GemmM padded to `m_padded` rows. The descriptor
// is not built: for the biggest problems it would have more than 2^31 elements.
bool BwdDataRowsRejected(const BwdDataCase& c, int64_t m_padded)
{
    return !MakeBwdDataTransformer<index_t, true>(c, 0, 0).IsMergedOutPaddingRangeSupported(
        m_padded);
}

enum class Axis
{
    H,
    W
};

// A problem with a single varying dimension (`axis`): N = G = 1 (or `groups`), C = K = 8, and the
// other dimension has one position (a 1x1 filter and no pad)
BwdDataCase MakeBwdDataLineCase(Axis axis,
                                index_t in,
                                index_t filter,
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
    c.Hi = c.Wi = 1;
    c.Y = c.X = 1;
    c.StrideH = c.StrideW = c.DilationH = c.DilationW = 1;
    c.LeftPadH = c.LeftPadW = c.RightPadH = c.RightPadW = 0;
    if(axis == Axis::H)
    {
        c.Hi        = in;
        c.Y         = filter;
        c.StrideH   = stride;
        c.DilationH = dilation;
        c.LeftPadH  = left_pad;
        c.RightPadH = right_pad;
    }
    else
    {
        c.Wi        = in;
        c.X         = filter;
        c.StrideW   = stride;
        c.DilationW = dilation;
        c.LeftPadW  = left_pad;
        c.RightPadW = right_pad;
    }
    return c;
}

// The line problem with a far reaching filter: the right pad is D * (F - 1) (clamped to the index
// range) so that the output has as many positions as the input
BwdDataCase MakeBwdDataFarReachLineCase(
    Axis axis, index_t in, index_t filter, index_t stride, int64_t dilation, index_t left_pad = 0)
{
    return MakeBwdDataLineCase(
        axis,
        in,
        filter,
        stride,
        static_cast<index_t>(dilation),
        left_pad,
        static_cast<index_t>(std::min<int64_t>(dilation * (filter - 1), kMaxIndex)));
}

// A 2D problem with the filter, stride, dilation and pad of both dimensions depending on y, a
// filter size from 1 to 16 (see MakeFilterSweepCase)
BwdDataCase MakeBwdDataSweepCase(index_t y)
{
    const index_t strides[][2]   = {{1, 1}, {2, 1}, {1, 3}, {3, 2}, {2, 2}};
    const index_t dilations[][2] = {{1, 1}, {2, 1}, {1, 2}, {2, 3}};
    const index_t left_pads[]    = {0, 1, 3};
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
    c.RightPadH = RightPadForOutput(c.Hi, c.Y, c.DilationH, c.LeftPadH, y % 3);
    c.RightPadW = RightPadForOutput(c.Wi, c.X, c.DilationW, c.LeftPadW, y % 2);
    return c;
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
    for(const auto& problem : cases)
    {
        // packed tensors (G = 1) and tensors with interleaved groups (NHWGK / NHWGC, G = 3)
        ForEachGroupCount(problem, [](const BwdDataCase& c) {
            const auto separate = CheckBwdDataAllGroups<false>(c, true);
            EXPECT_GT(separate.comparison.valid, 0);

            const auto merged = CheckBwdDataAllGroups<true>(c, true);
            EXPECT_EQ(merged.accepted, merged.groups);
            EXPECT_GT(merged.comparison.valid, 0);
        });
    }
}

TEST(TensorDescriptorFunctorsIntegration, BackwardDataMergedPaddingWithFilters)
{
    // filters of 1 to 16 taps (odd and even, the other dimension's filter has the opposite
    // parity most of the time), strides and dilations of both dimensions, groups 1 and 3, and pads
    // that keep the output non empty. These problems are small: the host check must accept all of
    // them and every position of every group is compared with the generic chain.
    int64_t positions = 0;
    for(index_t y = 1; y <= 16; ++y)
    {
        const auto c = MakeBwdDataSweepCase(y);
        CheckBwdDataAllGroups<false>(c, true);
        const auto merged = CheckBwdDataAllGroups<true>(c, true);
        EXPECT_EQ(merged.accepted, merged.groups);
        positions += merged.comparison.positions;
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
        BwdDataCase problem{};
        problem.N = sc.n;
        problem.C = problem.K = 8;
        problem.Hi = problem.Wi = sc.hi;
        problem.Y = problem.X = 1;
        problem.StrideH = problem.StrideW = problem.DilationH = problem.DilationW = 1;
        problem.LeftPadH = problem.LeftPadW = problem.RightPadH = problem.RightPadW = 0;

        ForEachGroupCount(problem, [&](const BwdDataCase& c) {
            SCOPED_TRACE("N=" + std::to_string(sc.n) + " Hi=" + std::to_string(sc.hi) +
                         " G=" + std::to_string(c.G));
            // with G groups the tensor has G times more elements, so more batches are split
            const auto transformer = MakeBwdDataTransformer<index_t, true, true>(c, 0, 0);
            if(c.G == 1)
            {
                EXPECT_EQ(transformer.N_, sc.expected_n);
            }
            else
            {
                EXPECT_LE(transformer.N_, sc.expected_n);
                EXPECT_EQ(sc.n % transformer.N_, 0);
            }

            const auto s = CheckBwdDataAllGroups<true, true>(c, false);
            EXPECT_EQ(s.accepted, s.groups);
        });
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

template <index_t NDim, bool CustomB>
using WrwTransform = ck::tensor_operation::TransformConvBwdWeightToGemmV2<
    NDim,
    32, // MPerBlock
    kWrwNPerBlock,
    kWrwK1,
    kWrwK0PerBlock,
    1, // NumGroupsToMerge
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default,
    index_t,
    false, // UseTdm
    CustomB>;

// The A, B and C descriptors of the transform. `input_strides` replaces the strides of the packed
// input tensor (G N C spatial... order); `use_full_batch_kindex` is the V1 mode of the device op,
// whose K0 dimension covers all the K batches.
template <index_t NDim, bool CustomB>
auto MakeBwdWeightDescriptors(const BwdWeightCase<NDim>& c,
                              const std::array<index_t, NDim + 3>* input_strides = nullptr,
                              bool use_full_batch_kindex                         = false)
{
    const auto g = MakeGeometry(c);

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
        out_s *= g.out[d];
        wei_s *= c.Filter[d];
    }
    for(index_t d = 0; d < NDim; ++d)
    {
        in_strides[1] *= c.In[d];
        out_strides[1] *= g.out[d];
    }
    if(input_strides != nullptr)
    {
        in_strides = *input_strides;
    }

    return WrwTransform<NDim, CustomB>::template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
        NDim>(c.N,
              c.K,
              c.C,
              c.In,
              c.Filter,
              g.out,
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

template <index_t NDim, bool CustomB>
auto MakeBwdWeightBDescriptor(const BwdWeightCase<NDim>& c,
                              const std::array<index_t, NDim + 3>* input_strides = nullptr)
{
    return MakeBwdWeightDescriptors<NDim, CustomB>(c, input_strides)[Number<1>{}];
}

// Backward weight problem of the filter sweep (see MakeFilterSweepCase)
template <index_t NDim>
BwdWeightCase<NDim> MakeWrwSweepCase(index_t y)
{
    auto c   = MakeFilterSweepCase<BwdWeightCase<NDim>>(y);
    c.K      = 8;
    c.BatchK = 1 + y % 3;
    return c;
}

// Compares the custom B descriptor with the generic chain on every position and checks the moves.
// `require_valid` asserts that the problem has valid elements (small problems may not have any).
template <index_t NDim>
void CheckBwdWeightBTransformMatchesGeneric(const BwdWeightCase<NDim>& c, bool require_valid = true)
{
    SCOPED_TRACE(Describe(c));

    const auto custom  = MakeBwdWeightBDescriptor<NDim, true>(c);
    const auto generic = MakeBwdWeightBDescriptor<NDim, false>(c);

    const index_t k0 = custom.GetLength(Number<0>{});
    const index_t n  = custom.GetLength(Number<1>{});
    const index_t k1 = custom.GetLength(Number<2>{});
    ASSERT_EQ(k0, generic.GetLength(Number<0>{}));
    ASSERT_EQ(n, generic.GetLength(Number<1>{}));
    ASSERT_EQ(k1, generic.GetLength(Number<2>{}));

    const auto r =
        CompareDescriptors3D(custom, generic, AllPositions(k0), AllPositions(n), AllPositions(k1));
    EXPECT_EQ(r.mismatches, 0);
    if(require_valid)
    {
        EXPECT_GT(r.valid, 0);
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
    // every case with a packed tensor (G = 1) and with interleaved groups (G = 3)
    const auto check = [](const auto& cfg) {
        ForEachGroupCount(cfg, [](const auto& c) {
            CheckBwdWeightBTransformMatchesGeneric<std::decay_t<decltype(c)>::kNDim>(c);
        });
    };
    for(const auto& c : cases1d)
    {
        check(c);
    }
    for(const auto& c : cases2d)
    {
        check(c);
    }
    for(const auto& c : cases3d)
    {
        check(c);
    }
}

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightWithFilters)
{
    // filters of 1 to 16 taps (odd and even) in 1D, 2D and 3D with strides, dilations, pads, batch
    // splits and groups; every position of the B descriptor is compared with the generic chain
    for(index_t y = 1; y <= 16; ++y)
    {
        ForEachSpatialDim([&](auto ndim) {
            constexpr index_t NDim = decltype(ndim)::value;
            CheckBwdWeightBTransformMatchesGeneric<NDim>(MakeWrwSweepCase<NDim>(y), false);
        });
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

// A problem without dilation and with the same pad on both sides. K = 32 so that GemmM is a
// multiple of MPerBlock: the hack needs a compact A descriptor.
template <index_t NDim>
BwdWeightCase<NDim> MakeSplitKCase(index_t n,
                                   index_t channels,
                                   const std::array<index_t, NDim>& in,
                                   const std::array<index_t, NDim>& filter,
                                   const std::array<index_t, NDim>& stride,
                                   const std::array<index_t, NDim>& pad,
                                   index_t groups = 1)
{
    BwdWeightCase<NDim> c{};
    c.N        = n;
    c.C        = channels;
    c.K        = 32;
    c.In       = in;
    c.Filter   = filter;
    c.Stride   = stride;
    c.PadLeft  = pad;
    c.PadRight = pad;
    c.BatchK   = 1;
    c.G        = groups;
    c.Dilation.fill(1);
    return c;
}

// Returns whether the split-K hack is eligible
template <index_t NDim>
bool CheckSplitKHackWithCustomB(BwdWeightCase<NDim> c, index_t k_batch)
{
    SCOPED_TRACE(Describe(c) + " KBatch=" + std::to_string(k_batch));
    c.BatchK = k_batch;

    // the descriptors of the device op (full K batch index)
    const auto custom  = MakeBwdWeightDescriptors<NDim, true>(c, nullptr, true);
    const auto generic = MakeBwdWeightDescriptors<NDim, false>(c, nullptr, true);

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
    const auto out                 = MakeGeometry(c).out;
    constexpr index_t k_block_size = kWrwK0PerBlock * kWrwK1;
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
        BwdWeightCase<2> c;
        index_t k_batch;
        bool eligible;
    };
    const auto make = [](index_t n,
                         index_t channels,
                         std::array<index_t, 2> in,
                         std::array<index_t, 2> filter,
                         std::array<index_t, 2> stride,
                         index_t groups) {
        return MakeSplitKCase<2>(n, channels, in, filter, stride, {0, 0}, groups);
    };
    const Case2d cases[] = {
        // 1x1, stride 1, pad 0: GemmK = N * Hi * Wi, the input is read as it is stored
        {make(4, 32, {4, 4}, {1, 1}, {1, 1}, 1), 2, true},
        {make(4, 32, {4, 4}, {1, 1}, {1, 1}, 1), 4, true},
        {make(6, 32, {4, 4}, {1, 1}, {1, 1}, 1), 3, true},
        {make(2, 64, {8, 4}, {1, 1}, {1, 1}, 1), 2, true},
        // 2x2 stride 2 pad 0: the patches do not overlap, the im2col is a permutation of the input
        {make(4, 8, {8, 8}, {2, 2}, {2, 2}, 1), 2, true},
        // not eligible: overlapping filter (3x3 with a pad of 1), strided 1x1, C not a multiple of
        // the GemmN block, groups, N not divisible by the K batch, GemmK padded, one K batch
        {MakeSplitKCase<2>(4, 32, {4, 4}, {3, 3}, {1, 1}, {1, 1}), 2, false},
        {make(4, 32, {8, 8}, {1, 1}, {2, 2}, 1), 2, false},
        {make(4, 8, {4, 4}, {1, 1}, {1, 1}, 1), 2, false},
        {make(4, 32, {4, 4}, {1, 1}, {1, 1}, 3), 2, false},
        {make(3, 32, {4, 4}, {1, 1}, {1, 1}, 1), 2, false},
        {make(4, 32, {5, 5}, {1, 1}, {1, 1}, 1), 2, false},
        {make(4, 32, {4, 4}, {1, 1}, {1, 1}, 1), 1, false},
    };
    for(const auto& tc : cases)
    {
        EXPECT_EQ(CheckSplitKHackWithCustomB<2>(tc.c, tc.k_batch), tc.eligible);
    }

    // 1D and 3D, 1x1 stride 1 pad 0
    const auto c1 = MakeSplitKCase<1>(4, 32, {16}, {1}, {1}, {0});
    EXPECT_TRUE(CheckSplitKHackWithCustomB<1>(c1, 2));
    EXPECT_TRUE(CheckSplitKHackWithCustomB<1>(c1, 4));
    EXPECT_FALSE(CheckSplitKHackWithCustomB<1>(MakeSplitKCase<1>(4, 32, {16}, {3}, {1}, {1}), 2));

    const auto c3 = MakeSplitKCase<3>(4, 32, {2, 2, 4}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0});
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
// of the accepted problems closest to them with the int64 reference.
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, ConvPaddingPoisonRangeLimit)
{
    constexpr long_index_t max_wrap = (long_index_t{1} << 31) - 1;

    // 1. The poison of the forward transform is the smallest value that moves the smallest first
    // spatial coordinate (-left_pad) to the length of the dimension: length + max(0, left_pad).
    const auto fwd_poison = [](index_t length, index_t left_pad) {
        const ForwardConvCase<1> cfg{1, 1, {length}, {1}, {1}, {1}, {left_pad}, {0}};
        const std::array<index_t, 1> out{static_cast<index_t>(OutputLengths(cfg)[0])};
        return MakeCustomTransform<1>(cfg, out, out[0], 1).Poison();
    };
    EXPECT_EQ(fwd_poison(8, 0), 8u);
    EXPECT_EQ(fwd_poison(8, 5), 13u);

    // 2. IsConvPaddingPoisonRangeSupported: the largest coordinate + 2 poisons + the distance to
    // the smallest coordinate (a kernel moves a coordinate by such a difference) must fit in 31
    // bits
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, 0, max_wrap - 16));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(8, 0, max_wrap - 15));
    // a left pad increases the poison (13) and the range of the coordinate (5 below 0)
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, -5, max_wrap - 31));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(8, -5, max_wrap - 30));
    // a negative left pad (positive lower index) does not
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(8, 5, max_wrap - 16));
    // a big input and a big pad
    EXPECT_TRUE(IsConvPaddingPoisonRangeSupported(1 << 28, -(1 << 27), 0));
    EXPECT_FALSE(IsConvPaddingPoisonRangeSupported(1 << 30, -(1 << 29), 0));

    // 3. The offset terms of IsConvInputTransformRangeSupported<1>: N, C, input, filter, output,
    // stride, dilation, left pad, m_padded, k_padded, N stride, spatial stride, C stride
    const auto offset_supported = [](index_t length, index_t spatial_stride) {
        return IsConvInputTransformRangeSupported<1>(
            1, 1, {length}, {1}, {length}, {1}, {1}, {0}, length + 1, 2, 1, {spatial_stride}, 1);
    };
    EXPECT_TRUE(offset_supported(1 << 26, 8));
    EXPECT_FALSE(offset_supported(1 << 27, 8));
    EXPECT_TRUE(offset_supported(1 << 28, 1));
    EXPECT_FALSE(offset_supported(1 << 30, 1));

    // 4. The same host check through the backward weight transform: input, filter, output,
    // dilation, left pad and the strides of the spatial dimensions of the input (N = C = 1, stride
    // 1, GemmK padded by 1 and GemmN by 64 columns)
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
    // upper side: the filter index of the padded columns times the dilation
    EXPECT_TRUE(wrw_supported({8, 8}, {1, 1}, {8, 8}, {1 << 20, 1}, {0, 0}, {8, 1}));
    EXPECT_FALSE(wrw_supported({8, 8}, {1, 1}, {8, 8}, {1 << 26, 1}, {0, 0}, {8, 1}));
    // offset of the poisoned coordinate: about 3 * length * stride must stay below 2^31
    EXPECT_TRUE(wrw_supported({1 << 26, 1}, {1, 1}, {1 << 26, 1}, {1, 1}, {0, 0}, {8, 1}));
    EXPECT_FALSE(wrw_supported({1 << 27, 1}, {1, 1}, {1 << 27, 1}, {1, 1}, {0, 0}, {8, 1}));
    // ... also with a left pad, which increases the poison (the range of the coordinate is from
    // -left_pad to 2 poisons + the largest one: about 2^28)
    EXPECT_TRUE(wrw_supported(
        {1 << 26, 1}, {1, 1}, {(1 << 26) + (1 << 26), 1}, {1, 1}, {1 << 26, 0}, {4, 1}));
    EXPECT_FALSE(wrw_supported(
        {1 << 26, 1}, {1, 1}, {(1 << 26) + (1 << 26), 1}, {1, 1}, {1 << 26, 0}, {8, 1}));

    // 5. What the host check protects: the padding position (m = M, k = k_pad) of an input with a
    // 1x1 filter has m >= M and k >= K, so both poisons are added: 2 * poison, with
    // poison = length + left_pad. It must stay invalid. The transform is only evaluated if the
    // host check accepts the problem. `supported` returns the result of the host check.
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
                return false;
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

namespace {

// Block sizes of the GEMM padding in the tests of the host checks. The padding changes the limits
// (the padded columns continue the filter index), so these are arbitrary but fixed; the device ops
// take the padded sizes from their descriptors.
constexpr int64_t kFwdMPerBlock = 256;
constexpr int64_t kFwdKPerBlock = 64;

// What the host checks need to know about a problem on an input tensor packed as N C spatial...
template <index_t NDim>
struct HostCheckInput
{
    bool ok = false; // false (with a test failure) if the problem cannot be used
    ConvGeometry<NDim> geometry;
    ChannelFirstStrides<NDim> strides;
    int64_t m_padded = 0; // padded rows (n, o_0..) of the im2col matrix
    int64_t k_padded = 0; // padded columns (f_0.., c)
};

template <typename Case>
HostCheckInput<Case::kNDim>
MakeHostCheckInput(const Case& c, bool first_dim_innermost, int64_t m_block, int64_t k_block)
{
    HostCheckInput<Case::kNDim> h;
    h.geometry = MakeGeometry(c);
    if(h.geometry.empty)
    {
        ADD_FAILURE() << "the test problem has an empty output";
        return h;
    }
    h.strides = ChannelFirstStrides<Case::kNDim>(c.In, c.C, c.N, first_dim_innermost);
    if(h.strides.elements > kMaxIndex)
    {
        ADD_FAILURE() << "the test input is too big";
        return h;
    }
    h.m_padded = RoundUp(h.geometry.M, m_block);
    h.k_padded = RoundUp(h.geometry.K, k_block);
    h.ok       = true;
    return h;
}

// IsConvInputTransformRangeSupported called with the description of the problem
template <typename Case>
bool DescriptionSupported(const Case& c, const HostCheckInput<Case::kNDim>& h)
{
    return IsConvInputTransformRangeSupported<Case::kNDim>(c.N,
                                                           c.C,
                                                           ToLongArray(c.In),
                                                           ToLongArray(c.Filter),
                                                           ToLongArray(h.geometry.out),
                                                           ToLongArray(c.Stride),
                                                           ToLongArray(c.Dilation),
                                                           ToLongArray(c.PadLeft),
                                                           h.m_padded,
                                                           h.k_padded,
                                                           h.strides.n,
                                                           h.strides.spatial,
                                                           h.strides.c);
}

// Rows of the backward weight B matrix are GemmK, padded to K0PerBlock * K1 * BatchK, and the
// columns are GemmN, padded to NPerBlock
template <index_t NDim>
HostCheckInput<NDim> MakeWrwHostCheckInput(const BwdWeightCase<NDim>& c, bool first_dim_innermost)
{
    return MakeHostCheckInput(
        c, first_dim_innermost, kWrwK0PerBlock * kWrwK1 * c.BatchK, kWrwNPerBlock);
}

// The host check of the backward weight transform (it reads the problem from its arguments)
template <index_t NDim>
bool WrwWrapperSupported(const BwdWeightCase<NDim>& c, const HostCheckInput<NDim>& h)
{
    return WrwTransform<NDim, true>::IsCustomTransformPaddingRangeSupported(
        c.N,
        c.C,
        c.In,
        c.Filter,
        h.geometry.out,
        h.strides.GroupNCSpatial(),
        c.Stride,
        c.Dilation,
        c.PadLeft,
        static_cast<index_t>(h.m_padded),
        static_cast<index_t>(h.k_padded));
}

// Runs the host check IsConvInputTransformRangeSupported on a forward problem and, if the problem
// is accepted (and `compare_positions`), checks every position of the padded GEMM (all m < m_padded
// and k < k_padded) against the int64 reference: the padding must be invalid, the real positions
// must map to the right lower index, and no offset term or sum may leave the range of int32.
// Returns the result of the host check.
template <index_t NDim>
bool CheckConvInputRangeLimit(const ForwardConvCase<NDim>& cfg,
                              bool first_dim_innermost,
                              bool compare_positions = true)
{
    SCOPED_TRACE(Describe(cfg));

    const auto h = MakeHostCheckInput(cfg, first_dim_innermost, kFwdMPerBlock, kFwdKPerBlock);
    if(!h.ok)
    {
        return false;
    }
    const bool supported = DescriptionSupported(cfg, h);
    if(!supported || !compare_positions)
    {
        return supported;
    }

    const Im2colReference<NDim> reference(cfg);
    const auto transform = MakeCustomTransform<NDim>(
        cfg, h.geometry.out, static_cast<index_t>(h.m_padded), static_cast<index_t>(h.k_padded));

    int64_t mismatches = 0;
    int64_t overflows  = 0;
    for(int64_t m = 0; m < h.m_padded; ++m)
    {
        for(int64_t k = 0; k < h.k_padded; ++k)
        {
            const auto p  = reference.At(m, k);
            const auto up = make_tuple(static_cast<index_t>(m), static_cast<index_t>(k));
            mismatches += transform.IsValidUpperIndexMappedToValidLowerIndex(up) != p.valid;

            MultiIndex<NDim + 2> low;
            transform.CalculateLowerIndex(low, up);
            std::array<int64_t, NDim> low_spatial{};
            static_for<0, NDim, 1>{}([&](auto d) { low_spatial[d] = low[Number<d + 1>{}]; });
            if(p.valid)
            {
                bool equal = low[Number<0>{}] == p.n && low[Number<NDim + 1>{}] == p.c;
                for(index_t d = 0; d < NDim; ++d)
                {
                    equal &= low_spatial[d] == p.in_idx[d];
                }
                mismatches += !equal;
            }
            overflows +=
                h.strides.OverflowCount(low[Number<0>{}], low_spatial, low[Number<NDim + 1>{}]);
        }
    }
    EXPECT_EQ(mismatches, 0) << "an accepted problem has a wrong padding or lower index";
    EXPECT_EQ(overflows, 0) << "an accepted problem overflows the offset computation";

    // the (M, K) descriptor of the transform on an input tensor with the strides of the host check
    const auto strides = h.strides.NSpatialC();
    const auto desc    = MakeCustomDescriptor<NDim>(cfg,
                                                 h.geometry.out,
                                                 static_cast<index_t>(h.m_padded),
                                                 static_cast<index_t>(h.k_padded),
                                                 &strides);
    EXPECT_EQ(CountCoordinateMoveMismatches(desc, 3000, 25), 0);
    return true;
}

// The same for the backward weight B matrix, with the host check of its transform. The reference
// of a position (i0, i1, i2) of the descriptor [K0, GemmN, K1] is the im2col position
// (m = i0 * K1 + i2, k = i1).
template <index_t NDim>
bool CheckBwdWeightBRangeLimit(const BwdWeightCase<NDim>& c,
                               bool first_dim_innermost,
                               bool compare_positions = true)
{
    SCOPED_TRACE(Describe(c) + (first_dim_innermost ? " innermost=first" : " innermost=last"));

    const auto h = MakeWrwHostCheckInput(c, first_dim_innermost);
    if(!h.ok)
    {
        return false;
    }
    const bool supported = WrwWrapperSupported(c, h);
    if(!supported || !compare_positions)
    {
        return supported;
    }

    const auto in_strides = h.strides.GroupNCSpatial();
    const auto custom     = MakeBwdWeightBDescriptor<NDim, true>(c, &in_strides);
    const index_t k0      = custom.GetLength(Number<0>{});
    const index_t n       = custom.GetLength(Number<1>{});
    const index_t k1      = custom.GetLength(Number<2>{});
    EXPECT_EQ(int64_t(k0) * k1, h.m_padded);
    EXPECT_EQ(n, h.k_padded);

    const Im2colReference<NDim> reference(c);
    int64_t mismatches = 0;
    for(index_t i0 = 0; i0 < k0; ++i0)
    {
        for(index_t i1 = 0; i1 < n; ++i1)
        {
            for(index_t i2 = 0; i2 < k1; ++i2)
            {
                const auto p      = reference.At(int64_t(i0) * k1 + i2, i1);
                const int64_t off = h.strides.Offset(p.n, p.c, p.in_idx);

                const auto cc = make_tensor_coordinate(custom, make_multi_index(i0, i1, i2));
                const bool cv = coordinate_has_valid_offset(custom, cc);
                if((cv != p.valid || (p.valid && cc.GetOffset() != off)) && mismatches++ < 3)
                {
                    ADD_FAILURE() << "(" << i0 << "," << i1 << "," << i2 << ") custom valid=" << cv
                                  << " offset=" << cc.GetOffset() << ", reference valid=" << p.valid
                                  << " offset=" << off;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "an accepted problem has a wrong padding or offset";
    EXPECT_EQ(CountCoordinateMoveMismatches(custom, 3000, 24), 0);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Limit suites shared by the forward A matrix and the backward weight B matrix. They only differ in
// the problem type, the host check, and the padding of the dimension that holds (f_0.., c).
// ---------------------------------------------------------------------------------------------

template <index_t NDim>
struct FwdRangeLimitOps
{
    static constexpr int64_t kKBlock = kFwdKPerBlock;
    static auto Make(const FarReachSpec& s) { return MakeFarReachCase<ForwardConvCase<NDim>>(s); }
    static bool Check(const ForwardConvCase<NDim>& c, bool first_dim_innermost, bool compare)
    {
        return CheckConvInputRangeLimit<NDim>(c, first_dim_innermost, compare);
    }
};

template <index_t NDim>
struct WrwRangeLimitOps
{
    static constexpr int64_t kKBlock = kWrwNPerBlock;
    static auto Make(const FarReachSpec& s)
    {
        auto c   = MakeFarReachCase<BwdWeightCase<NDim>>(s);
        c.K      = 8;
        c.BatchK = 1;
        return c;
    }
    static bool Check(const BwdWeightCase<NDim>& c, bool first_dim_innermost, bool compare)
    {
        return CheckBwdWeightBRangeLimit<NDim>(c, first_dim_innermost, compare);
    }
};

// Limit of the first dimension of a far reach problem. The largest lower coordinate is
// (Wo - 1) + f0_max * D - left_pad, with f0_max the filter index of the last (padded) GEMM column,
// and the poison is Wi + left_pad. The limit is max_lower + 2 * poison <= 2^31 - 1, so the largest
// dilation that fits is d_max. The filter of the other dimensions is FarReachSpec::other_filter.
struct ReachLimit
{
    int64_t f0_max;
    int64_t max_reach; // largest value of f0_max * D
    int64_t d_max;
};

template <index_t NDim>
ReachLimit
ComputeReachLimit(int64_t channels, int64_t filter0, int64_t k_block, int64_t wi, int64_t left_pad)
{
    int64_t filter_rest = channels;
    for(index_t d = 1; d < NDim; ++d)
    {
        filter_rest *= FarReachSpec{}.other_filter;
    }
    const int64_t k_padded = RoundUp(filter_rest * filter0, k_block);
    const int64_t wo       = wi + left_pad;
    const int64_t poison   = wi + left_pad;
    ReachLimit limit;
    limit.f0_max    = (k_padded - 1) / filter_rest;
    limit.max_reach = kMaxIndex - (wo - 1) - 2 * poison;
    limit.d_max     = limit.max_reach / limit.f0_max;
    return limit;
}

// First spatial dimension: the poison range
template <index_t NDim, typename Ops>
void CheckFirstDimensionLimit()
{
    for(const index_t filter0 : {2, 3, 4, 7, 8, 15, 16})
    {
        for(const index_t channels : {8, 40})
        {
            for(const index_t left_pad : {0, 5})
            {
                FarReachSpec spec;
                spec.channels = channels;
                spec.filter   = filter0;
                spec.left_pad = left_pad;
                const auto limit =
                    ComputeReachLimit<NDim>(channels, filter0, Ops::kKBlock, spec.in, left_pad);
                EXPECT_GE(limit.f0_max, filter0 - 1);

                const auto check = [&](int64_t dilation) {
                    spec.dilation = dilation;
                    return Ops::Check(Ops::Make(spec), true, true);
                };
                const std::string what = "F=" + std::to_string(filter0) +
                                         " C=" + std::to_string(channels) +
                                         " left_pad=" + std::to_string(left_pad);

                // far from the limit, and a margin below it for the terms of the offset that are
                // not part of the poison range (batch, channel, other dimensions)
                for(const int64_t dilation :
                    {int64_t{1}, int64_t{1} << 20, (limit.max_reach - 65536) / limit.f0_max})
                {
                    EXPECT_TRUE(check(dilation)) << what << " D=" << dilation;
                }
                // Closest accepted problem below the limit of the poison range. The host check
                // also covers the other terms of the offset, so it stops accepting a bit below the
                // limit. The check then compares every row and column of the GEMM padding,
                // including the positions where both are padding (two poisons on the largest
                // filter index), so this is where a wrap around would show.
                bool accepted_below_limit = false;
                for(int64_t dilation = limit.d_max;
                    dilation > limit.d_max - 100000 && !accepted_below_limit;
                    --dilation)
                {
                    accepted_below_limit = check(dilation);
                }
                EXPECT_TRUE(accepted_below_limit) << what;
                // just above the limit of the poison range: always rejected
                EXPECT_FALSE(check(limit.d_max + 1)) << what;
            }
        }
    }
}

// A big left pad (the poison is Wi + left_pad and Wo is about left_pad): the host check only, the
// output would be too big to compare every position. The limit moves with the pad.
template <index_t NDim, typename Ops>
void CheckBigLeftPadLimit()
{
    for(const index_t filter0 : {3, 8, 16})
    {
        FarReachSpec spec;
        spec.filter   = filter0;
        spec.left_pad = 1 << 20;
        const auto limit =
            ComputeReachLimit<NDim>(spec.channels, filter0, Ops::kKBlock, spec.in, spec.left_pad);

        spec.dilation = (limit.max_reach - 65536) / limit.f0_max;
        EXPECT_TRUE(Ops::Check(Ops::Make(spec), true, false)) << "F=" << filter0;
        spec.dilation = limit.d_max + 1;
        EXPECT_FALSE(Ops::Check(Ops::Make(spec), true, false)) << "F=" << filter0;
    }
}

// Last spatial dimension: its coordinates are not poisoned, but they are multiplied by their
// stride in 32 bit, so (F - 1) * D must stay below 2^31 whatever the other terms are.
template <index_t NDim, typename Ops>
void CheckLastDimensionLimit()
{
    for(const index_t filter_last : {2, 3, 4, 7, 8, 15, 16})
    {
        FarReachSpec spec;
        spec.channels     = 64; // a multiple of the block: no padded columns
        spec.reach_dim    = NDim - 1;
        spec.in           = 32; // the clamped right pad of the rejected probes loses up to F - 2
        spec.filter       = filter_last;
        spec.other_in     = 8;
        spec.other_filter = 1;
        spec.other_stride = 1;
        spec.other_pad    = 0;

        for(const int64_t dilation :
            {int64_t{1}, int64_t{1} << 20, (int64_t{1} << 30) / (filter_last - 1)})
        {
            spec.dilation = dilation;
            EXPECT_TRUE(Ops::Check(Ops::Make(spec), false, true))
                << "F=" << filter_last << " D=" << dilation;
        }
        // the dilation that gives (F - 1) * D > 2^31 - 1 is not an index for F = 2
        if(filter_last > 2)
        {
            spec.dilation = kMaxIndex / (filter_last - 1) + 1;
            EXPECT_FALSE(Ops::Check(Ops::Make(spec), false, true))
                << "F=" << filter_last << " D=" << spec.dilation;
        }
    }
}

template <index_t NDim, typename Ops>
void RunRangeLimitSuite()
{
    CheckBigLeftPadLimit<NDim, Ops>();
    CheckFirstDimensionLimit<NDim, Ops>();
    CheckLastDimensionLimit<NDim, Ops>();
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, ConvInputTransformRangeLimitWithFilters)
{
    // filters of 2 to 16 taps (odd and even) in 1D, 2D and 3D
    ForEachSpatialDim([](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        RunRangeLimitSuite<NDim, FwdRangeLimitOps<NDim>>();
    });
}

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightRangeLimitWithFilters)
{
    // the same limits through the host check of the backward weight transform
    ForEachSpatialDim([](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        RunRangeLimitSuite<NDim, WrwRangeLimitOps<NDim>>();
    });
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
        const auto s = CheckBwdDataAllGroups<true>(
            MakeBwdDataLineCase(Axis::H, 1 << log_hi, 1, 1, 1, 0, 0), false);
        EXPECT_EQ(s.accepted, s.groups) << "Hi = 2^" << log_hi;
    }
    {
        // around the limit of the offset of a poisoned coordinate, (Ho + 2 poisons) * HoStride_.
        // Ho is not a multiple of the block size, so that there are padding rows as well as
        // padding columns (K = 8 is padded to the block size too): two poisons.
        int accepted = 0;
        for(int step = 0; step <= 15; ++step)
        {
            const auto c =
                MakeBwdDataLineCase(Axis::H, (1 << 25) + step * (1 << 21) + 5, 1, 1, 1, 0, 0);
            accepted += CheckBwdDataAllGroups<true>(c, false).accepted;
        }
        EXPECT_GT(accepted, 0);
    }
    // above the limit: a poisoned Ho coordinate times HoStride_ would pass 2^31
    EXPECT_TRUE(BwdDataRowsRejected(MakeBwdDataLineCase(Axis::H, 1 << 27, 1, 1, 1, 0, 0), 1 << 27));

    // 2. big dilations (the filter taps, and the padded columns, reach far outside the tensor) for
    // several filter sizes, odd and even, with stride 1 and 2
    for(const index_t y : {2, 3, 4, 7, 8, 15, 16})
    {
        for(const index_t stride : {1, 2})
        {
            SCOPED_TRACE("Y=" + std::to_string(y) + " stride=" + std::to_string(stride));
            const auto make_case = [&](int64_t dilation) {
                return MakeBwdDataFarReachLineCase(Axis::H, 8, y, stride, dilation);
            };
            const auto output_length = [](const BwdDataCase& c) {
                return ConvOutputLength(c.Hi, c.LeftPadH, c.RightPadH, c.DilationH, c.Y, c.StrideH);
            };

            // moderate dilations are accepted
            for(const int64_t dilation : {int64_t{1}, int64_t{1} << 10, int64_t{1} << 16})
            {
                const auto c = make_case(dilation);
                ASSERT_GE(output_length(c), 1);
                const auto s = CheckBwdDataAllGroups<true>(c, false);
                EXPECT_EQ(s.accepted, s.groups) << "dilation=" << dilation;
            }

            // around the limit the host check decides, and an accepted problem must be exact
            int accepted = 0;
            for(int step = 0; step < 24; ++step)
            {
                const auto c = make_case((int64_t{1} << 20) + step * (int64_t{1} << 17));
                if(output_length(c) >= 1)
                {
                    accepted += CheckBwdDataAllGroups<true>(c, false).accepted;
                }
            }
            EXPECT_GT(accepted, 0);

            // a dilation of 2^26 is rejected (the right pad of the larger filters is clamped)
            const auto c = make_case(int64_t{1} << 26);
            if(output_length(c) >= 1)
            {
                EXPECT_TRUE(AnyBwdDataGroupRejected(c));
            }
        }
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
            const auto c = MakeBwdDataLineCase(Axis::H, (1 << log_hi) + 5, 1, 1, 1, 0, 0, groups);
            accepted += CheckBwdDataAllGroups<true>(c, false).accepted;
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
                const auto c = MakeBwdDataLineCase(Axis::H, (1 << log_hi) + 5, y, stride, 1, 1, 1);
                const auto s = CheckBwdDataAllGroups<true>(c, false);
                EXPECT_EQ(s.accepted, s.groups) << "stride=" << stride << " Y=" << y;
            }
        }
    }

    // 3. the W dimension: Ho = 1 and a wide output. The poison is on Ho only, Wo is a plain
    // coordinate, but its stride and the batch stride (= Wo * K) grow with it.
    for(const int log_wi : {10, 20, 24})
    {
        const auto s = CheckBwdDataAllGroups<true>(
            MakeBwdDataLineCase(Axis::W, 1 << log_wi, 1, 1, 1, 0, 0), false);
        // the host check is conservative for the biggest ones (the padding of the GemmK adds up to
        // GemmKPerBlock ydot steps to the poison, which is multiplied by the huge H stride)
        EXPECT_TRUE(s.accepted == s.groups || log_wi > 20) << "Wi = 2^" << log_wi;
    }
    // the output has 2^30 elements: the padding rows (n = N) would be at an offset of 2^30 and the
    // other terms add up to more than 2^31
    EXPECT_TRUE(BwdDataRowsRejected(MakeBwdDataLineCase(Axis::W, 1 << 27, 1, 1, 1, 0, 0), 1 << 27));
    // filters 2 .. 16 in the W dimension with a big dilation
    for(const index_t x : {2, 3, 4, 7, 8, 15, 16})
    {
        for(const index_t dilation : {1, 1 << 10, 1 << 16, 1 << 20})
        {
            const auto s = CheckBwdDataAllGroups<true>(
                MakeBwdDataFarReachLineCase(Axis::W, 8, x, 1, dilation), false);
            if(dilation <= (1 << 16))
            {
                EXPECT_EQ(s.accepted, s.groups) << "X=" << x << " dilation=" << dilation;
            }
        }
    }

    // 4. big left pads: Ho is about the left pad. Only the first 2^3 rows of such a problem are
    // real (the rest of the output is padding), and the host check bounds the batch of the padding
    // rows by the smallest possible number of real rows, so it rejects the biggest pads.
    for(const int log_pad : {10, 16, 20, 23})
    {
        const auto s = CheckBwdDataAllGroups<true>(
            MakeBwdDataLineCase(Axis::H, 8, 3, 1, 1, 1 << log_pad, 0), false);
        EXPECT_TRUE(s.accepted == s.groups || log_pad > 20) << "left pad = 2^" << log_pad;
    }
}

// =============================================================================
// The host checks of the transforms (TransformConvFwdToGemm::IsCustomTransformPaddingRangeSupported
// and TransformConvBwdWeightToGemmV2::IsCustomTransformPaddingRangeSupported) read the problem from
// the members / arguments of the transform; they must give the same result as
// IsConvInputTransformRangeSupported called with the description of the problem. A wrongly read
// value (a swapped dimension, a missing pad, the wrong stride, the batch) would change the limit,
// so the two are compared right before, at and after the limit.
// =============================================================================

namespace {

// Problem in which the dimension `first ? first : last` has a far reaching filter, with the
// strides of the reach dimension and of the other dimensions given. The batch is 3 so that the
// batch stride matters.
template <typename Case>
Case MakeWrapperTestCase(bool first,
                         index_t filter,
                         int64_t dilation,
                         index_t left_pad,
                         index_t stride,
                         index_t other_stride)
{
    FarReachSpec s;
    s.n            = 3;
    s.channels     = first ? 8 : 64;
    s.reach_dim    = first ? 0 : Case::kNDim - 1;
    s.in           = 1 << 20;
    s.filter       = filter;
    s.dilation     = dilation;
    s.left_pad     = left_pad;
    s.stride       = stride;
    s.other_in     = 2;
    s.other_filter = 2;
    s.other_stride = other_stride;
    s.other_pad    = 1;
    return MakeFarReachCase<Case>(s);
}

// Finds by bisection the largest dilation that the description based host check accepts, then
// checks that the host check of the transform agrees with it right before, at and after the limit.
// make(first, filter, dilation, left_pad) builds the problem and checks(problem, first) returns
// {host check of the transform, host check of the description}.
template <typename MakeFn, typename ChecksFn>
void CheckWrapperAgreesAroundLimit(MakeFn&& make, ChecksFn&& checks)
{
    for(const bool first : {true, false})
    {
        for(const index_t filter : {3, 8, 16})
        {
            for(const index_t left_pad : {0, 5, 1 << 20})
            {
                // the last dimension has no poison, so a left pad does not move its limit
                if(!first && left_pad != 0)
                {
                    continue;
                }
                const auto accepted = [&](int64_t dilation) {
                    return checks(make(first, filter, dilation, left_pad), first).second;
                };
                int64_t lo = 1, hi = kMaxIndex / (filter - 1);
                ASSERT_TRUE(accepted(lo));
                while(lo < hi)
                {
                    const int64_t mid = lo + (hi - lo + 1) / 2;
                    if(accepted(mid))
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
}

// {host check of the forward transform, host check of the description} of a forward problem. The
// transform of the device op reads G N C spatial... lengths and strides of the three tensors.
template <index_t NDim>
std::pair<bool, bool> FwdWrapperAndDescriptionChecks(const ForwardConvCase<NDim>& cfg, bool first)
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

    const auto h = MakeHostCheckInput(cfg, first, kFwdMPerBlock, kFwdKPerBlock);
    if(!h.ok)
    {
        return {false, false};
    }

    // the weight and output strides are not read by the host check
    std::array<index_t, NDim + 3> a_len{}, b_len{}, c_len{}, b_str{}, c_str{};
    a_len[0] = b_len[0] = c_len[0] = 1;
    a_len[1] = c_len[1] = cfg.N;
    a_len[2] = b_len[2] = cfg.C;
    b_len[1] = c_len[2] = 8;
    for(index_t d = 0; d < NDim; ++d)
    {
        a_len[3 + d] = cfg.In[d];
        b_len[3 + d] = cfg.Filter[d];
        c_len[3 + d] = h.geometry.out[d];
    }
    b_str.fill(1);
    c_str.fill(1);
    const auto a_str = h.strides.GroupNCSpatial(h.strides.elements);
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
    return {transform.IsCustomTransformPaddingRangeSupported(static_cast<index_t>(h.m_padded),
                                                             static_cast<index_t>(h.k_padded)),
            DescriptionSupported(cfg, h)};
}

// {host check of the backward weight transform, host check of the description}
template <index_t NDim>
std::pair<bool, bool> WrwWrapperAndDescriptionChecks(const BwdWeightCase<NDim>& c, bool first)
{
    const auto h = MakeWrwHostCheckInput(c, first);
    if(!h.ok)
    {
        return {false, false};
    }
    return {WrwWrapperSupported(c, h), DescriptionSupported(c, h)};
}

} // namespace

TEST(TensorDescriptorFunctorsIntegration, ForwardTransformHostCheckMatchesDescription)
{
    ForEachSpatialDim([](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        CheckWrapperAgreesAroundLimit(
            [](bool first, index_t filter, int64_t dilation, index_t left_pad) {
                return MakeWrapperTestCase<ForwardConvCase<NDim>>(
                    first, filter, dilation, left_pad, 1, 2);
            },
            [](const ForwardConvCase<NDim>& c, bool first) {
                return FwdWrapperAndDescriptionChecks<NDim>(c, first);
            });
    });
}

TEST(TensorDescriptorFunctorsIntegration, BackwardWeightHostCheckMatchesDescription)
{
    ForEachSpatialDim([](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        CheckWrapperAgreesAroundLimit(
            [](bool first, index_t filter, int64_t dilation, index_t left_pad) {
                auto c = MakeWrapperTestCase<BwdWeightCase<NDim>>(
                    first, filter, dilation, left_pad, 2, 3);
                c.K      = 8;
                c.BatchK = 2;
                return c;
            },
            [](const BwdWeightCase<NDim>& c, bool first) {
                return WrwWrapperAndDescriptionChecks<NDim>(c, first);
            });
    });
}

// =============================================================================
// Randomized problems (fixed seeds): all the custom transforms against the generic chain
// =============================================================================

TEST(TensorDescriptorFunctorsIntegration, RandomizedCustomTransformsMatchGeneric)
{
    SeededRandom rnd(20241125);

    // forward A matrix and backward weight B matrix, in 1D, 2D and 3D
    ForEachSpatialDim([&](auto ndim) {
        constexpr index_t NDim = decltype(ndim)::value;
        const int iterations   = NDim == 3 ? 8 : 25;
        for(int it = 0; it < iterations; ++it)
        {
            CheckForwardConvInputTransformMatchesGeneric<NDim>(
                MakeRandomCase<ForwardConvCase<NDim>>(rnd));
        }
        for(int it = 0; it < iterations; ++it)
        {
            auto c   = MakeRandomCase<BwdWeightCase<NDim>>(rnd);
            c.K      = 8;
            c.BatchK = rnd(1, 3);
            CheckBwdWeightBTransformMatchesGeneric<NDim>(c, false);
        }
    });

    // backward data A matrix with the merged padding (2D)
    for(int it = 0; it < 30; ++it)
    {
        BwdDataCase c{};
        c.N          = rnd(1, 3);
        c.C          = 8;
        c.K          = 8 * rnd(1, 2);
        c.G          = rnd(0, 1) ? 3 : 1;
        c.Y          = rnd(1, 8);
        c.X          = rnd(1, 8);
        c.Hi         = rnd(1, 12);
        c.Wi         = rnd(1, 12);
        c.StrideH    = rnd(1, 3);
        c.StrideW    = rnd(1, 3);
        c.DilationH  = rnd(1, 3);
        c.DilationW  = rnd(1, 3);
        c.LeftPadH   = rnd(0, 4);
        c.LeftPadW   = rnd(0, 4);
        c.RightPadH  = RightPadForOutput(c.Hi, c.Y, c.DilationH, c.LeftPadH, rnd(0, 3));
        c.RightPadW  = RightPadForOutput(c.Wi, c.X, c.DilationW, c.LeftPadW, rnd(0, 3));
        const auto s = CheckBwdDataAllGroups<true>(c, true);
        EXPECT_EQ(s.accepted, s.groups);
    }
}
