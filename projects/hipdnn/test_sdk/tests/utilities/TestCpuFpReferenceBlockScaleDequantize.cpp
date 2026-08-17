// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

template <typename T1, typename T2, typename T3>
struct TypeTriple
{
    using InputType = T1;
    using ScaleType = T2;
    using OutputType = T3;
};

// ============================================================================
// Typed tests over input type: float/half/bfloat16 with float scale
// ============================================================================

using TypesBlockScaleDequantize = ::testing::Types<TypeTriple<float, float, float>,
                                                   TypeTriple<half, float, float>,
                                                   TypeTriple<bfloat16, float, float>>;

template <class T>
class CpuFpReferenceBlockScaleDequantizeTyped : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceBlockScaleDequantizeTyped, TypesBlockScaleDequantize, );

TYPED_TEST(CpuFpReferenceBlockScaleDequantizeTyped, IdentityScale)
{
    using XType = typename TypeParam::InputType;
    using ScaleType = typename TypeParam::ScaleType;
    using YType = typename TypeParam::OutputType;

    // X: 2x4, Scale: 2x2 (block_size=2 along dim 1)
    Tensor<XType> xTensor({2, 4});
    Tensor<ScaleType> scaleTensor({2, 2});
    Tensor<YType> yTensor({2, 4});

    xTensor.fillWithValue(safeTestTypeCast<XType>(3.0f));
    scaleTensor.fillWithValue(safeTestTypeCast<ScaleType>(1.0f));

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    for(int b = 0; b < 2; ++b)
    {
        for(int c = 0; c < 4; ++c)
        {
            EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(b, c)), 3.0f, tolerance);
        }
    }
}

TEST(TestCpuFpReferenceBlockScaleDequantizeTyped, BFloat16InBFloat16Out)
{
    // bfloat16 input, float scale, bfloat16 output — registered signature gap
    Tensor<bfloat16> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<bfloat16> yTensor({1, 4});

    xTensor.setHostValue(bfloat16(2.0f), 0, 0);
    xTensor.setHostValue(bfloat16(4.0f), 0, 1);
    xTensor.setHostValue(bfloat16(1.0f), 0, 2);
    xTensor.setHostValue(bfloat16(3.0f), 0, 3);

    scaleTensor.setHostValue(0.5f, 0, 0); // scales elements [0,1]
    scaleTensor.setHostValue(2.0f, 0, 1); // scales elements [2,3]

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-2f; // bfloat16 has ~2 decimal digits of precision
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 0)), 2.0f * 0.5f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 1)), 4.0f * 0.5f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 2)), 1.0f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 3)), 3.0f * 2.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, NonTrivialScale)
{
    // X: 1x4, Scale: 1x2 => block_size=2 along dim 1
    Tensor<float> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(1.0f, 0, 0);
    xTensor.setHostValue(2.0f, 0, 1);
    xTensor.setHostValue(3.0f, 0, 2);
    xTensor.setHostValue(4.0f, 0, 3);

    scaleTensor.setHostValue(2.0f, 0, 0); // scales elements [0,1]
    scaleTensor.setHostValue(0.5f, 0, 1); // scales elements [2,3]

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 1.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 2.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 3.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 4.0f * 0.5f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, MultiDimBlocking)
{
    // X: 2x4x8, Scale: 2x4x2 => block along trailing dim, block_size=4
    // Use distinct per-block scale values so spot-checking can detect index mapping bugs.
    Tensor<float> xTensor({2, 4, 8});
    Tensor<float> scaleTensor({2, 4, 2});
    Tensor<float> yTensor({2, 4, 8});

    xTensor.fillWithValue(1.0f);

    // Assign distinct scale values: scale[b][r][s] covers x[b][r][s*4..(s+1)*4-1]
    float scaleVal = 1.0f;
    for(int b = 0; b < 2; ++b)
    {
        for(int r = 0; r < 4; ++r)
        {
            for(int s = 0; s < 2; ++s)
            {
                scaleTensor.setHostValue(scaleVal, b, r, s);
                scaleVal += 1.0f;
            }
        }
    }

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {4}, false);

    auto tolerance = 1e-5f;
    // Block (0,0,0): x[0][0][0..3] => scale[0][0][0] = 1.0
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 0), 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 3), 1.0f, tolerance);
    // Block (0,0,1): x[0][0][4..7] => scale[0][0][1] = 2.0
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 4), 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 7), 2.0f, tolerance);
    // Block (0,1,0): x[0][1][0..3] => scale[0][1][0] = 3.0
    EXPECT_NEAR(yTensor.getHostValue(0, 1, 0), 3.0f, tolerance);
    // Block (1,3,1): x[1][3][4..7] => scale[1][3][1] = 16.0 (last block)
    EXPECT_NEAR(yTensor.getHostValue(1, 3, 4), 16.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(1, 3, 7), 16.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, MultiTrailingDimBlockSize)
{
    // X: 1x4x6, Scale: 1x2x3 => blockSize={2,2} blocks trailing two dims simultaneously
    // dim 1: block_size=2 (4/2=2 scale entries), dim 2: block_size=2 (6/2=3 scale entries)
    Tensor<float> xTensor({1, 4, 6});
    Tensor<float> scaleTensor({1, 2, 3});
    Tensor<float> yTensor({1, 4, 6});

    xTensor.fillWithValue(1.0f);

    // Assign distinct scale values per block so we can verify correct index mapping
    // scale[0][s1][s2] covers x[0][s1*2..s1*2+1][s2*2..s2*2+1]
    scaleTensor.setHostValue(1.0f, 0, 0, 0);
    scaleTensor.setHostValue(2.0f, 0, 0, 1);
    scaleTensor.setHostValue(3.0f, 0, 0, 2);
    scaleTensor.setHostValue(4.0f, 0, 1, 0);
    scaleTensor.setHostValue(5.0f, 0, 1, 1);
    scaleTensor.setHostValue(6.0f, 0, 1, 2);

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2, 2}, false);

    auto tolerance = 1e-5f;
    // Block (0,0): x[0][0..1][0..1] => scale[0][0][0] = 1.0
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 0), 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1, 1), 1.0f, tolerance);
    // Block (0,1): x[0][0..1][2..3] => scale[0][0][1] = 2.0
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 2), 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1, 3), 2.0f, tolerance);
    // Block (0,2): x[0][0..1][4..5] => scale[0][0][2] = 3.0
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 4), 3.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1, 5), 3.0f, tolerance);
    // Block (1,0): x[0][2..3][0..1] => scale[0][1][0] = 4.0
    EXPECT_NEAR(yTensor.getHostValue(0, 2, 0), 4.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3, 1), 4.0f, tolerance);
    // Block (1,2): x[0][2..3][4..5] => scale[0][1][2] = 6.0
    EXPECT_NEAR(yTensor.getHostValue(0, 2, 4), 6.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3, 5), 6.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, NegativeXValues)
{
    // Dequantize is linear: negative x values should scale correctly
    Tensor<float> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(-2.0f, 0, 0);
    xTensor.setHostValue(-1.0f, 0, 1);
    xTensor.setHostValue(0.0f, 0, 2);
    xTensor.setHostValue(-4.0f, 0, 3);

    scaleTensor.setHostValue(3.0f, 0, 0); // scales elements [0,1]
    scaleTensor.setHostValue(0.5f, 0, 1); // scales elements [2,3]

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), -2.0f * 3.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), -1.0f * 3.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 0.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), -4.0f * 0.5f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, IsNegativeScaleFloat)
{
    // With is_negative_scale=true and float scale, Y = X * 2^(-scale)
    Tensor<float> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(8.0f, 0, 0);
    xTensor.setHostValue(8.0f, 0, 1);
    xTensor.setHostValue(16.0f, 0, 2);
    xTensor.setHostValue(16.0f, 0, 3);

    scaleTensor.setHostValue(3.0f, 0, 0); // 2^(-3) = 0.125
    scaleTensor.setHostValue(1.0f, 0, 1); // 2^(-1) = 0.5

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, true);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 8.0f * 0.125f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 8.0f * 0.125f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 16.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 16.0f * 0.5f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, NormalScaleMultiplication)
{
    // Normal (non-negative) scale: Y = X * scale
    Tensor<float> xTensor({1, 2});
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> yTensor({1, 2});

    xTensor.setHostValue(5.0f, 0, 0);
    xTensor.setHostValue(10.0f, 0, 1);
    scaleTensor.setHostValue(0.1f, 0, 0);

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 1.0f, tolerance);
}

// ============================================================================
// FP8 E8M0 scale: standalone scale semantics test
// ============================================================================

TEST(TestCpuFpReferenceBlockScaleDequantizeFp8, E8M0ScaleDequantize)
{
    // Test with fp8_e8m0 scale: scale value is 2^(biased_exp - 127)
    // fp8_e8m0 with bits=127 => 2^0 = 1.0
    // fp8_e8m0 with bits=128 => 2^1 = 2.0
    Tensor<float> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(3.0f, 0, 0);
    xTensor.setHostValue(3.0f, 0, 1);
    xTensor.setHostValue(5.0f, 0, 2);
    xTensor.setHostValue(5.0f, 0, 3);

    // scale[0] = 1.0 (bits=127), scale[1] = 2.0 (bits=128)
    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 0);
    scaleTensor.setHostValue(fp8_e8m0::from_bits(128), 0, 1);

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 3.0f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 3.0f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 5.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 5.0f * 2.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp8, IsNegativeScaleE8M0)
{
    // isNegativeScale=true with fp8_e8m0: Y = X * 2^(-float(scale))
    // float(fp8_e8m0::from_bits(127)) = 2^0 = 1.0, so 2^(-1.0) = 0.5
    // float(fp8_e8m0::from_bits(128)) = 2^1 = 2.0, so 2^(-2.0) = 0.25
    Tensor<float> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(8.0f, 0, 0);
    xTensor.setHostValue(8.0f, 0, 1);
    xTensor.setHostValue(16.0f, 0, 2);
    xTensor.setHostValue(16.0f, 0, 3);

    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 0); // float value = 1.0 => 2^(-1.0) = 0.5
    scaleTensor.setHostValue(
        fp8_e8m0::from_bits(128), 0, 1); // float value = 2.0 => 2^(-2.0) = 0.25

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, true);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 8.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 8.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 16.0f * 0.25f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 16.0f * 0.25f, tolerance);
}

// ============================================================================
// Validation error path tests
// ============================================================================

TEST(TestCpuFpReferenceBlockScaleDequantizeValidation, ScaleRankExceedsXRank)
{
    // Scale tensor has more dimensions than x — should throw
    const Tensor<float> xTensor({4});
    const Tensor<float> scaleTensor({2, 2});
    Tensor<float> yTensor({4});

    EXPECT_THROW(
        CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false),
        std::invalid_argument);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeValidation, ScaleRankLessThanXRank)
{
    // Scale tensor has fewer dimensions than x — should throw (no broadcast support)
    const Tensor<float> xTensor({2, 4});
    const Tensor<float> scaleTensor({2});
    Tensor<float> yTensor({2, 4});

    EXPECT_THROW(
        CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false),
        std::invalid_argument);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeValidation, ScaleDimMismatch)
{
    // X: 1x4 with block_size=2 expects scale dims {1, 2}, but scale is {1, 3}
    const Tensor<float> xTensor({1, 4});
    const Tensor<float> scaleTensor({1, 3});
    Tensor<float> yTensor({1, 4});

    EXPECT_THROW(
        CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false),
        std::invalid_argument);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeValidation, EmptyXDimsThrows)
{
    const Tensor<float> xTensor({});
    const Tensor<float> scaleTensor({});
    Tensor<float> yTensor({});

    EXPECT_THROW(
        CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false),
        std::runtime_error);
}

// ============================================================================
// MX dequantize typed tests: all narrow types with fp8_e8m0 scale
// ============================================================================

using MxDequantizeTypes = ::testing::Types<TypeTriple<fp8_e4m3, fp8_e8m0, float>,
                                           TypeTriple<fp8_e4m3, fp8_e8m0, half>,
                                           TypeTriple<fp8_e5m2, fp8_e8m0, float>,
                                           TypeTriple<fp8_e5m2, fp8_e8m0, half>,
                                           TypeTriple<fp4_e2m1, fp8_e8m0, float>,
                                           TypeTriple<fp4_e2m1, fp8_e8m0, half>,
                                           TypeTriple<fp6_e2m3, fp8_e8m0, float>,
                                           TypeTriple<fp6_e2m3, fp8_e8m0, half>,
                                           TypeTriple<fp6_e3m2, fp8_e8m0, float>,
                                           TypeTriple<fp6_e3m2, fp8_e8m0, half>>;

template <class T>
class CpuFpReferenceMxDequantize : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceMxDequantize, MxDequantizeTypes, );

TYPED_TEST(CpuFpReferenceMxDequantize, WithE8m0Scale)
{
    using XType = typename TypeParam::InputType;
    using ScaleType = typename TypeParam::ScaleType;
    using YType = typename TypeParam::OutputType;

    // MX dequantize: narrow input, fp8_e8m0 scale, float/half output
    // scale[0] = 1.0 (bits=127), scale[1] = 2.0 (bits=128)
    // x: {1, 1, 2, 2}, expected y: {1, 1, 4, 4}
    Tensor<XType> xTensor({1, 4});
    Tensor<ScaleType> scaleTensor({1, 2});
    Tensor<YType> yTensor({1, 4});

    xTensor.setHostValue(XType(1.0f), 0, 0);
    xTensor.setHostValue(XType(1.0f), 0, 1);
    xTensor.setHostValue(XType(2.0f), 0, 2);
    xTensor.setHostValue(XType(2.0f), 0, 3);

    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 0); // 1.0
    scaleTensor.setHostValue(fp8_e8m0::from_bits(128), 0, 1); // 2.0

    CpuFpReferenceBlockScaleDequantize::dequantize(xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-2f;
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 0)), 1.0f * 1.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 1)), 1.0f * 1.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 2)), 2.0f * 2.0f, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 3)), 2.0f * 2.0f, tolerance);
}
