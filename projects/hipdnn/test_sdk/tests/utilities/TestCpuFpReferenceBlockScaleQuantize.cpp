// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleQuantize.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

template <typename T1, typename T2, typename T3>
struct TypeTriple
{
    using InputType = T1;
    using OutputType = T2;
    using ScaleType = T3;
};

// ============================================================================
// Typed tests over output type: float/half/bfloat16 with float scale
// ============================================================================

using TypesBlockScaleQuantize = ::testing::Types<TypeTriple<float, float, float>,
                                                 TypeTriple<float, half, float>,
                                                 TypeTriple<float, bfloat16, float>>;

template <class T>
class CpuFpReferenceBlockScaleQuantizeTyped : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceBlockScaleQuantizeTyped, TypesBlockScaleQuantize, );

TYPED_TEST(CpuFpReferenceBlockScaleQuantizeTyped, UniformInput)
{
    using InputType = typename TypeParam::InputType;
    using OutputType = typename TypeParam::OutputType;
    using ScaleType = typename TypeParam::ScaleType;

    Tensor<InputType> xTensor({2, 4});
    Tensor<OutputType> yTensor({2, 4});
    Tensor<ScaleType> scaleTensor({2, 2});

    const float uniformInputVal = 100.0f;
    xTensor.fillWithValue(safeTestTypeCast<InputType>(uniformInputVal));

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2, 1);

    const auto maxOutVal = static_cast<float>(std::numeric_limits<OutputType>::max());
    const float expectedScale = uniformInputVal / maxOutVal;

    const auto scaleTolerance = 1e-5f;
    for(int b = 0; b < 2; ++b)
    {
        for(int s = 0; s < 2; ++s)
        {
            EXPECT_NEAR(static_cast<float>(scaleTensor.getHostValue(b, s)) / expectedScale,
                        1.0,
                        scaleTolerance);
        }
    }

    // The output values should be equal to maxOutVal, since uniformInputVal / expectedScale = maxOutVal
    const auto outputTolerance = std::is_same_v<OutputType, float> ? 1.0e-5f : 1.0e-2f;
    for(int b = 0; b < 2; ++b)
    {
        for(int c = 0; c < 4; ++c)
        {
            EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(b, c)), maxOutVal, outputTolerance);
        }
    }

    // Check that the dequantized values are within one quantization step of the original value
    for(int b = 0; b < 2; ++b)
    {
        for(int c = 0; c < 4; ++c)
        {
            const auto quantized = static_cast<float>(yTensor.getHostValue(b, c));
            const auto scale = static_cast<float>(scaleTensor.getHostValue(b, c / 2));
            const auto reconstructed = quantized * scale;
            EXPECT_NEAR(reconstructed, uniformInputVal, scale);
        }
    }
}

TYPED_TEST(CpuFpReferenceBlockScaleQuantizeTyped, RoundTrip)
{
    using InputType = typename TypeParam::InputType;
    using OutputType = typename TypeParam::OutputType;
    using ScaleType = typename TypeParam::ScaleType;

    Tensor<InputType> xTensor({2, 4});
    Tensor<OutputType> yTensor({2, 4});
    Tensor<ScaleType> scaleTensor({2, 2});
    const int32_t blockSize = 2;
    const int64_t axis = 1;

    xTensor.fillWithRandomValues(safeTestTypeCast<InputType>(-1.0f),
                                 safeTestTypeCast<InputType>(1.0f));
    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, blockSize, axis);

    Tensor<InputType> reconstructedXTensor({2, 4});
    CpuFpReferenceBlockScaleDequantize::dequantize(
        yTensor, scaleTensor, reconstructedXTensor, {blockSize}, false);

    const auto tolerance = std::is_same_v<OutputType, float> ? 1.0e-5f : 1.0e-2f;
    for(int b = 0; b < 2; ++b)
    {
        for(int c = 0; c < 4; ++c)
        {
            const auto original = static_cast<float>(xTensor.getHostValue(b, c));
            const auto reconstructed = static_cast<float>(reconstructedXTensor.getHostValue(b, c));
            EXPECT_NEAR(reconstructed, original, tolerance);
        }
    }
}

TEST(TestCpuFpReferenceBlockScaleQuantizeFp32, NonTrivialScale)
{
    Tensor<float> xTensor({1, 4});
    Tensor<int8_t> yTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});

    xTensor.setHostValue(1.0f, 0, 0);
    xTensor.setHostValue(2.0f, 0, 1);
    xTensor.setHostValue(30.0f, 0, 2);
    xTensor.setHostValue(100.0f, 0, 3);

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2, 1);

    const auto maxOutVal = static_cast<float>(std::numeric_limits<int8_t>::max());
    const float expectedScale0 = 2.0f / maxOutVal;
    const float expectedScale1 = 100.0f / maxOutVal;
    EXPECT_NEAR(scaleTensor.getHostValue(0, 0), expectedScale0, 1e-5f);
    EXPECT_NEAR(scaleTensor.getHostValue(0, 1), expectedScale1, 1e-5f);

    // Block 0:
    //
    // 1 / (2/127) = 63.5 => 63 after cast to int8_t
    // 2 / (2/127) = 127
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 0)), 63);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 1)), 127);

    // Block 1:
    //
    // 30 / (100/127) = 38.1 => 38 after cast to int8_t
    // 100 / (100/127) = 127
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 2)), 38);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 3)), 127);

    // Check the dequantized values
    // Reconstructed values should be within one quantization step of the original value
    const float reconstructed0
        = static_cast<float>(yTensor.getHostValue(0, 0)) * scaleTensor.getHostValue(0, 0);
    const float reconstructed1
        = static_cast<float>(yTensor.getHostValue(0, 1)) * scaleTensor.getHostValue(0, 0);
    const float reconstructed2
        = static_cast<float>(yTensor.getHostValue(0, 2)) * scaleTensor.getHostValue(0, 1);
    const float reconstructed3
        = static_cast<float>(yTensor.getHostValue(0, 3)) * scaleTensor.getHostValue(0, 1);

    EXPECT_NEAR(reconstructed0, 1.0f, expectedScale0);
    EXPECT_NEAR(reconstructed1, 2.0f, expectedScale0);
    EXPECT_NEAR(reconstructed2, 30.0f, expectedScale1);
    EXPECT_NEAR(reconstructed3, 100.0f, expectedScale1);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeFp32, MultiDimBlockingDefaultAxis)
{
    Tensor<float> xTensor({2, 4, 8});
    Tensor<float> scaleTensor({2, 4, 2});
    Tensor<int8_t> yTensor({2, 4, 8});

    xTensor.fillWithValue(0.0f);
    // max(x[0, 0, 0:4]) = 4.0
    // max(x[0, 0, 4:8]) = 8.0
    xTensor.setHostValue(4.0f, 0, 0, 2);
    xTensor.setHostValue(8.0f, 0, 0, 6);
    // max(x[1, 3, 4:8]) = 16.0
    xTensor.setHostValue(16.0f, 1, 3, 4);

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 4);

    const auto maxOutVal = static_cast<float>(std::numeric_limits<int8_t>::max());
    EXPECT_NEAR(scaleTensor.getHostValue(0, 0, 0), 4.0f / maxOutVal, 1e-5f);
    EXPECT_NEAR(scaleTensor.getHostValue(0, 0, 1), 8.0f / maxOutVal, 1e-5f);
    EXPECT_NEAR(scaleTensor.getHostValue(1, 3, 1), 16.0f / maxOutVal, 1e-5f);
    // All other scale values should be 0.0
    for(int b = 0; b < 2; ++b)
    {
        for(int c = 0; c < 4; ++c)
        {
            for(int s = 0; s < 2; ++s)
            {
                if((b == 0 && c == 0 && s == 0) || (b == 0 && c == 0 && s == 1)
                   || (b == 1 && c == 3 && s == 1))
                {
                    continue;
                }
                EXPECT_FLOAT_EQ(scaleTensor.getHostValue(b, c, s), 0.0f);
            }
        }
    }

    // Max values in each block should be quantized to maxOutVal, all other values should be 0
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 0, 2)), 127);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 0, 6)), 127);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(1, 3, 4)), 127);
    for(int b = 0; b < 2; ++b)
    {
        for(int c = 0; c < 4; ++c)
        {
            for(int s = 0; s < 8; ++s)
            {
                if((b == 0 && c == 0 && s == 2) || (b == 0 && c == 0 && s == 6)
                   || (b == 1 && c == 3 && s == 4))
                {
                    continue;
                }
                EXPECT_EQ(static_cast<int>(yTensor.getHostValue(b, c, s)), 0);
            }
        }
    }
}

TEST(TestCpuFpReferenceBlockScaleQuantizeFp32, NonDefaultAxisBlocking)
{
    Tensor<float> xTensor({4, 8});
    Tensor<float> scaleTensor({2, 8});
    Tensor<float> yTensor({4, 8});

    xTensor.fillWithValue(1.0f);
    // max(x[0:2, 3]) = 50.0
    xTensor.setHostValue(50.0f, 1, 3);
    // max(x[2:4, 5]) = 100.0
    xTensor.setHostValue(100.0f, 3, 5);

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2, 0);

    const auto maxOutVal = std::numeric_limits<float>::max();
    const float expectedScale0 = 50.0f / maxOutVal;
    const float expectedScale1 = 100.0f / maxOutVal;
    const float expectedScale2 = 1.0f / maxOutVal;
    const auto scaleTolerance = 1e-5f;
    EXPECT_NEAR(scaleTensor.getHostValue(0, 3) / expectedScale0, 1.0f, scaleTolerance);
    EXPECT_NEAR(scaleTensor.getHostValue(1, 5) / expectedScale1, 1.0f, scaleTolerance);
    for(int b = 0; b < 2; ++b)
    {
        for(int c = 0; c < 8; ++c)
        {
            if((b == 0 && c == 3) || (b == 1 && c == 5))
            {
                continue;
            }
            EXPECT_NEAR(scaleTensor.getHostValue(b, c) / expectedScale2, 1.0f, scaleTolerance);
        }
    }

    // Max values in each block should be quantized to maxOutVal
    EXPECT_NEAR(yTensor.getHostValue(1, 3), maxOutVal, 1e-5f);
    EXPECT_NEAR(yTensor.getHostValue(3, 5), maxOutVal, 1e-5f);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeFp32, NegativeXValuesPreserveSign)
{
    // The max-abs search must ignore sign, but the output must preserve it
    Tensor<float> xTensor({1, 3});
    Tensor<float> scaleTensor({1, 1});
    Tensor<int8_t> yTensor({1, 3});

    xTensor.setHostValue(-10.0f, 0, 0);
    xTensor.setHostValue(3.0f, 0, 1);
    xTensor.setHostValue(-2.0f, 0, 2);

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 3);

    const auto maxOutVal = static_cast<float>(std::numeric_limits<int8_t>::max());
    EXPECT_NEAR(scaleTensor.getHostValue(0, 0), 10.0f / maxOutVal, 1e-5f);

    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 0)), -127);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 1)), 38);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 2)), -25);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeFp32, AllZerosInput)
{
    Tensor<float> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> yTensor({1, 4});

    xTensor.fillWithValue(0.0f);

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 4);

    EXPECT_FLOAT_EQ(scaleTensor.getHostValue(0, 0), 0.0f);
    for(int c = 0; c < 4; ++c)
    {
        EXPECT_FLOAT_EQ(yTensor.getHostValue(0, c), 0.0f);
    }
}

TEST(TestCpuFpReferenceBlockScaleQuantizeFp32, RaggedLastBlock)
{
    Tensor<float> xTensor({1, 5});
    Tensor<float> scaleTensor({1, 3});
    Tensor<int8_t> yTensor({1, 5});

    xTensor.setHostValue(2.0f, 0, 0);
    xTensor.setHostValue(4.0f, 0, 1);
    xTensor.setHostValue(6.0f, 0, 2);
    xTensor.setHostValue(8.0f, 0, 3);
    xTensor.setHostValue(127.0f, 0, 4); // only element in the last block

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2);

    const auto maxOutVal = static_cast<float>(std::numeric_limits<int8_t>::max());
    EXPECT_NEAR(scaleTensor.getHostValue(0, 0), 4.0f / maxOutVal, 1e-5f);
    EXPECT_NEAR(scaleTensor.getHostValue(0, 1), 8.0f / maxOutVal, 1e-5f);
    EXPECT_FLOAT_EQ(scaleTensor.getHostValue(0, 2), 1.0f);

    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 0)), 63);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 1)), 127);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 2)), 95);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 3)), 127);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 4)), 127);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeFp32, OutputStaysWithinLimits)
{
    Tensor<float> xTensor({1, 4});
    Tensor<int8_t> yTensor({1, 4});
    Tensor<float> scaleTensor({1, 1});

    xTensor.setHostValue(127.0f, 0, 0);
    xTensor.setHostValue(-128.0f, 0, 1); // exceeds int8_t positive max
    xTensor.setHostValue(50.0f, 0, 2);
    xTensor.setHostValue(-127.0f, 0, 3);

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 4);

    const auto minOutVal = std::numeric_limits<int8_t>::lowest();
    const auto maxOutVal = std::numeric_limits<int8_t>::max();
    for(int c = 0; c < 4; ++c)
    {
        const auto yVal = yTensor.getHostValue(0, c);
        EXPECT_GE(yVal, minOutVal);
        EXPECT_LE(yVal, maxOutVal);
    }
}

TEST(TestCpuFpReferenceBlockScaleQuantizeFp32, Tensors1D)
{
    Tensor<float> xTensor({4});
    Tensor<int8_t> yTensor({4});
    Tensor<float> scaleTensor({2});

    xTensor.setHostValue(1.0f, 0);
    xTensor.setHostValue(2.0f, 1);
    xTensor.setHostValue(30.0f, 2);
    xTensor.setHostValue(100.0f, 3);

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2);

    const auto maxOutVal = static_cast<float>(std::numeric_limits<int8_t>::max());
    const float expectedScale0 = 2.0f / maxOutVal;
    const float expectedScale1 = 100.0f / maxOutVal;
    EXPECT_NEAR(scaleTensor.getHostValue(0), expectedScale0, 1e-5f);
    EXPECT_NEAR(scaleTensor.getHostValue(1), expectedScale1, 1e-5f);

    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0)), 63);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(1)), 127);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(2)), 38);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(3)), 127);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeFp32, Fp8E8M0Scale)
{
    Tensor<float> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<int8_t> yTensor({1, 4});

    // Block 0: maxAbsVal == maxOutVal => true ratio == 1.0 == 2^0, bits = 127
    xTensor.setHostValue(127.0f, 0, 0);
    xTensor.setHostValue(-100.0f, 0, 1);
    // Block 1: maxAbsVal == 2 * maxOutVal => true ratio == 2.0 == 2^1, bits = 128
    xTensor.setHostValue(254.0f, 0, 2);
    xTensor.setHostValue(-40.0f, 0, 3);

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2, 1);

    EXPECT_EQ(scaleTensor.getHostValue(0, 0).data, fp8_e8m0::from_bits(127).data);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 0)), 127);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 1)), -100);

    EXPECT_EQ(scaleTensor.getHostValue(0, 1).data, fp8_e8m0::from_bits(128).data);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 2)), 127);
    EXPECT_EQ(static_cast<int>(yTensor.getHostValue(0, 3)), -20);
}

// ============================================================================
// Validation error path tests
// ============================================================================

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, IORankMismatch)
{
    const Tensor<float> xTensor({2, 4});
    Tensor<float> yTensor({2, 4, 1});
    Tensor<float> scaleTensor({2, 2});

    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, IODimsMismatch)
{
    const Tensor<float> xTensor({2, 4});
    Tensor<float> yTensor({2, 5});
    Tensor<float> scaleTensor({2, 2});

    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, EmptyDimsThrows)
{
    const Tensor<float> xTensor({});
    Tensor<float> yTensor({});
    Tensor<float> scaleTensor({});

    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, ZeroBlockSizeThrows)
{
    const Tensor<float> xTensor({1, 4});
    Tensor<float> yTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});

    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 0),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, NegativeBlockSizeThrows)
{
    const Tensor<float> xTensor({1, 4});
    Tensor<float> yTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});

    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, -2),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, AxisBelowZeroThrows)
{
    const Tensor<float> xTensor({2, 4});
    Tensor<float> yTensor({2, 4});
    Tensor<float> scaleTensor({2, 2});

    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2, -1),
                 std::out_of_range);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, AxisAtOrAboveRankThrows)
{
    const Tensor<float> xTensor({2, 4});
    Tensor<float> yTensor({2, 4});
    Tensor<float> scaleTensor({2, 2});

    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2, 2),
                 std::out_of_range);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, ScaleDimMismatchDefaultAxis)
{
    const Tensor<float> xTensor({1, 4});
    Tensor<float> yTensor({1, 4});
    Tensor<float> scaleTensor({1, 3});

    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, ScaleDimMismatchNonDefaultAxis)
{
    const Tensor<float> xTensor({4, 8});
    Tensor<float> yTensor({4, 8});
    Tensor<float> scaleTensor({3, 8});

    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2, 0),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceBlockScaleQuantizeValidation, ThrowsOnNaNAndInfInput)
{
    Tensor<float> xTensorNaN({1, 4});
    Tensor<float> xTensorInf({1, 4});
    Tensor<float> yTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});

    xTensorNaN.setHostValue(std::numeric_limits<float>::quiet_NaN(), 0, 0);
    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensorNaN, yTensor, scaleTensor, 2),
                 std::runtime_error);

    xTensorInf.setHostValue(std::numeric_limits<float>::infinity(), 0, 0);
    EXPECT_THROW(CpuFpReferenceBlockScaleQuantize::quantize(xTensorInf, yTensor, scaleTensor, 2),
                 std::runtime_error);
}

// ===========================================================================
// MX quantize typed tests: all narrow types with fp8_e8m0 scale
// ===========================================================================

using MxQuantizeTypes = ::testing::Types<TypeTriple<float, fp8_e4m3, fp8_e8m0>,
                                         TypeTriple<half, fp8_e4m3, fp8_e8m0>,
                                         TypeTriple<float, fp8_e5m2, fp8_e8m0>,
                                         TypeTriple<half, fp8_e5m2, fp8_e8m0>,
                                         TypeTriple<float, fp4_e2m1, fp8_e8m0>,
                                         TypeTriple<half, fp4_e2m1, fp8_e8m0>,
                                         TypeTriple<float, fp6_e2m3, fp8_e8m0>,
                                         TypeTriple<half, fp6_e2m3, fp8_e8m0>,
                                         TypeTriple<float, fp6_e3m2, fp8_e8m0>,
                                         TypeTriple<half, fp6_e3m2, fp8_e8m0>>;

template <class T>
class CpuFpReferenceBlockScaleQuantizeMxTyped : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceBlockScaleQuantizeMxTyped, MxQuantizeTypes, );

TYPED_TEST(CpuFpReferenceBlockScaleQuantizeMxTyped, WithE8m0Scale)
{
    using InputType = typename TypeParam::InputType;
    using OutputType = typename TypeParam::OutputType;
    using ScaleType = typename TypeParam::ScaleType;

    Tensor<InputType> xTensor({1, 4});
    Tensor<OutputType> yTensor({1, 4});
    Tensor<ScaleType> scaleTensor({1, 2});

    const auto maxOutVal = static_cast<float>(std::numeric_limits<OutputType>::max());
    xTensor.setHostValue(safeTestTypeCast<InputType>(maxOutVal / 2.0f), 0, 0);
    xTensor.setHostValue(safeTestTypeCast<InputType>(maxOutVal / 2.0f), 0, 1);
    xTensor.setHostValue(safeTestTypeCast<InputType>(maxOutVal), 0, 2);
    xTensor.setHostValue(safeTestTypeCast<InputType>(maxOutVal), 0, 3);

    CpuFpReferenceBlockScaleQuantize::quantize(xTensor, yTensor, scaleTensor, 2);

    const auto tolerance = 1e-2f;
    EXPECT_NEAR(static_cast<float>(scaleTensor.getHostValue(0, 0)), 0.5f, tolerance);
    EXPECT_NEAR(static_cast<float>(scaleTensor.getHostValue(0, 1)), 1.0f, tolerance);

    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 0)), maxOutVal, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 1)), maxOutVal, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 2)), maxOutVal, tolerance);
    EXPECT_NEAR(static_cast<float>(yTensor.getHostValue(0, 3)), maxOutVal, tolerance);
}
