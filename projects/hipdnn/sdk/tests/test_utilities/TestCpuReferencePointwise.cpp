// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestConstants.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::test_utilities::constants;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

template <typename Input1Type, typename Input2Type = Input1Type, typename OutputType = Input1Type>
class CpuReferencePointwiseBinaryTemplate : public ::testing::Test
{
protected:
    OutputType getTolerance() const
    {
        if constexpr(std::is_same_v<OutputType, half>)
        {
            return static_cast<OutputType>(TOLERANCE_HALF);
        }
        else if constexpr(std::is_same_v<OutputType, hip_bfloat16>)
        {
            return static_cast<OutputType>(TOLERANCE_BFLOAT16);
        }
        else
        {
            return static_cast<OutputType>(TOLERANCE_FLOAT);
        }
    }

    void testBinaryAddOperation()
    {
        Tensor<Input1Type> input1({1, 3, 2, 2});
        Tensor<Input2Type> input2({1, 3, 2, 2});
        Tensor<OutputType> output({1, 3, 2, 2});

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_1));
        input2.fillWithValue(static_cast<Input2Type>(TEST_VALUE_2));

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        Tensor<OutputType> expected({1, 3, 2, 2});
        expected.fillWithValue(static_cast<OutputType>(TEST_VALUE_3));

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBinarySubtractOperation()
    {
        Tensor<Input1Type> input1({1, 3, 2, 2});
        Tensor<Input2Type> input2({1, 3, 2, 2});
        Tensor<OutputType> output({1, 3, 2, 2});

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_5));
        input2.fillWithValue(static_cast<Input2Type>(TEST_VALUE_2));

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::SUB, output, input1, input2);

        Tensor<OutputType> expected({1, 3, 2, 2});
        expected.fillWithValue(static_cast<OutputType>(TEST_VALUE_3));

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBinaryAddOperationSanityValidation()
    {
        Tensor<Input1Type> input1({1, 1, 2, 2});
        Tensor<Input2Type> input2({1, 1, 2, 2});
        Tensor<OutputType> output({1, 1, 2, 2});

        input1.setHostValue(static_cast<Input1Type>(PI), 0, 0, 0, 0); // π
        input1.setHostValue(static_cast<Input1Type>(E), 0, 0, 0, 1); // e
        input1.setHostValue(static_cast<Input1Type>(SQRT_2), 0, 0, 1, 0); // √2
        input1.setHostValue(static_cast<Input1Type>(GOLDEN_RATIO), 0, 0, 1, 1); // φ (golden ratio)

        input2.setHostValue(static_cast<Input2Type>(LN_2), 0, 0, 0, 0); // ln(2)
        input2.setHostValue(static_cast<Input2Type>(std::sin(1.0f)), 0, 0, 0, 1);
        input2.setHostValue(static_cast<Input2Type>(std::cos(1.0f)), 0, 0, 1, 0);
        input2.setHostValue(static_cast<Input2Type>(std::tan(1.0f)), 0, 0, 1, 1);

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        // Create expected tensor with computed results
        Tensor<OutputType> expected({1, 1, 2, 2});
        expected.setHostValue(static_cast<OutputType>(PI + LN_2), 0, 0, 0, 0); // π + ln(2)
        expected.setHostValue(
            static_cast<OutputType>(E + std::sin(1.0f)), 0, 0, 0, 1); // e + sin(1)
        expected.setHostValue(
            static_cast<OutputType>(SQRT_2 + std::cos(1.0f)), 0, 0, 1, 0); // √2 + cos(1)
        expected.setHostValue(
            static_cast<OutputType>(GOLDEN_RATIO + std::tan(1.0f)), 0, 0, 1, 1); // φ + tan(1)

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBinarySubtractOperationSanityValidation()
    {
        Tensor<Input1Type> input1({1, 1, 2, 2});
        Tensor<Input2Type> input2({1, 1, 2, 2});
        Tensor<OutputType> output({1, 1, 2, 2});

        input1.setHostValue(static_cast<Input1Type>(TEST_VALUE_2 * PI), 0, 0, 0, 0); // 2π
        input1.setHostValue(static_cast<Input1Type>(E * E), 0, 0, 0, 1); // e²
        input1.setHostValue(static_cast<Input1Type>(SQRT_5), 0, 0, 1, 0); // √5
        input1.setHostValue(static_cast<Input1Type>(TEST_VALUE_2), 0, 0, 1, 1); // Simple test value

        input2.setHostValue(static_cast<Input2Type>(PI / TEST_VALUE_2), 0, 0, 0, 0); // π/2
        input2.setHostValue(static_cast<Input2Type>(E), 0, 0, 0, 1); // e
        input2.setHostValue(static_cast<Input2Type>(SQRT_3), 0, 0, 1, 0); // √3
        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_1), 0, 0, 1, 1); // Simple test value

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::SUB, output, input1, input2);

        // Create expected tensor with computed results
        Tensor<OutputType> expected({1, 1, 2, 2});
        expected.setHostValue(static_cast<OutputType>((TEST_VALUE_2 * PI) - (PI / TEST_VALUE_2)),
                              0,
                              0,
                              0,
                              0); // 2π - π/2 = 3π/2
        expected.setHostValue(static_cast<OutputType>((E * E) - E), 0, 0, 0,
                              1); // e² - e
        expected.setHostValue(static_cast<OutputType>(SQRT_5 - SQRT_3), 0, 0, 1, 0); // √5 - √3
        expected.setHostValue(
            static_cast<OutputType>(TEST_VALUE_2 - TEST_VALUE_1), 0, 0, 1, 1); // 2 - 1

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBinaryAddOperation2D()
    {
        Tensor<Input1Type> input1({4, 3});
        Tensor<Input2Type> input2({4, 3});
        Tensor<OutputType> output({4, 3});

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_1));
        input2.fillWithValue(static_cast<Input2Type>(TEST_VALUE_2));

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        Tensor<OutputType> expected({4, 3});
        expected.fillWithValue(static_cast<OutputType>(TEST_VALUE_3));

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBinaryAddOperation3D()
    {
        Tensor<Input1Type> input1({2, 3, 10});
        Tensor<Input2Type> input2({2, 3, 10});
        Tensor<OutputType> output({2, 3, 10});

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_2_5));
        input2.fillWithValue(static_cast<Input2Type>(TEST_VALUE_1_5));

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        Tensor<OutputType> expected({2, 3, 10});
        expected.fillWithValue(static_cast<OutputType>(TEST_VALUE_4));

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBinarySingleElementTensors()
    {
        Tensor<Input1Type> input1({1, 1, 1, 1});
        Tensor<Input2Type> input2({1, 1, 1, 1});
        Tensor<OutputType> output({1, 1, 1, 1});

        input1.setHostValue(static_cast<Input1Type>(E * E), 0, 0, 0, 0); // e²
        input2.setHostValue(static_cast<Input2Type>(E), 0, 0, 0, 0); // e

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::SUB, output, input1, input2);

        Tensor<OutputType> expected({1, 1, 1, 1});
        expected.setHostValue(static_cast<OutputType>((E * E) - E), 0, 0, 0,
                              0); // e² - e

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBinaryNumericalPrecision()
    {
        Tensor<Input1Type> input1({1, 1, 1, 1});
        Tensor<Input2Type> input2({1, 1, 1, 1});
        Tensor<OutputType> output({1, 1, 1, 1});

        input1.setHostValue(static_cast<Input1Type>(PRECISION_TEST_A), 0, 0, 0, 0);
        input2.setHostValue(static_cast<Input2Type>(PRECISION_TEST_B), 0, 0, 0, 0);

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        Tensor<OutputType> expected({1, 1, 1, 1});
        expected.setHostValue(
            static_cast<OutputType>(PRECISION_TEST_A + PRECISION_TEST_B), 0, 0, 0, 0);

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testElementwise1D()
    {
        Tensor<Input1Type> input1({5});
        Tensor<Input2Type> input2({5});
        Tensor<OutputType> output({5});

        for(int i = 0; i < 5; ++i)
        {
            input1.setHostValue(static_cast<Input1Type>(static_cast<float>(i + 1)), i);
            input2.setHostValue(static_cast<Input2Type>(static_cast<float>(i * 2)), i);
        }

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        // Create expected tensor: [1,2,3,4,5] + [0,2,4,6,8] = [1,4,7,10,13]
        Tensor<OutputType> expected({5});
        expected.setHostValue(static_cast<OutputType>(static_cast<float>(1)), 0);
        expected.setHostValue(static_cast<OutputType>(static_cast<float>(4)), 1);
        expected.setHostValue(static_cast<OutputType>(static_cast<float>(7)), 2);
        expected.setHostValue(static_cast<OutputType>(static_cast<float>(10)), 3);
        expected.setHostValue(static_cast<OutputType>(static_cast<float>(13)), 4);

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBroadcast2Dx1D()
    {
        Tensor<Input1Type> input1({3, 4}); // [M,N] = [3,4]
        Tensor<Input2Type> input2({4}); // [N] = [4]
        Tensor<OutputType> output({3, 4}); // Output: [3,4]

        // Fill input1 with pattern: row*10 + col
        for(int m = 0; m < 3; ++m)
        {
            for(int n = 0; n < 4; ++n)
            {
                input1.setHostValue(
                    static_cast<Input1Type>(static_cast<float>((m * 10) + n)), m, n);
            }
        }

        // Fill input2 with pattern: [100, 200, 300, 400]
        for(int n = 0; n < 4; ++n)
        {
            input2.setHostValue(static_cast<Input2Type>(static_cast<float>((n + 1) * 100)), n);
        }

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        // Create expected tensor: broadcasting input2[n] to all input1[m,n]
        Tensor<OutputType> expected({3, 4});
        for(int m = 0; m < 3; ++m)
        {
            for(int n = 0; n < 4; ++n)
            {
                auto input1Val = static_cast<float>((m * 10) + n);
                auto input2Val = static_cast<float>((n + 1) * 100);
                expected.setHostValue(static_cast<OutputType>(input1Val + input2Val), m, n);
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBroadcast3D()
    {
        Tensor<Input1Type> input1({2, 3, 4}); // [2,3,4]
        Tensor<Input2Type> input2({1, 3, 1}); // [1,3,1] - broadcasts to [2,3,4]
        Tensor<OutputType> output({2, 3, 4}); // Output: [2,3,4]

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_5));

        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_1), 0, 0, 0); // Channel 0
        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_2), 0, 1, 0); // Channel 1
        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_3), 0, 2, 0); // Channel 2

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::SUB, output, input1, input2);

        // Create expected tensor: broadcasting subtraction
        Tensor<OutputType> expected({2, 3, 4});
        for(int n = 0; n < 2; ++n)
        {
            for(int c = 0; c < 3; ++c)
            {
                for(int h = 0; h < 4; ++h)
                {
                    float input1Val = TEST_VALUE_5;
                    auto input2Val = static_cast<float>(c + 1); // Channel values: 1.0, 2.0, 3.0
                    expected.setHostValue(static_cast<OutputType>(input1Val - input2Val), n, c, h);
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBroadcast3DImplicitLeading()
    {
        Tensor<Input1Type> input1({2, 3, 4}); // [2,3,4]
        Tensor<Input2Type> input2(
            {3, 1}); // [3,1] - broadcasts to [2,3,4] (implicit leading dimension)
        Tensor<OutputType> output({2, 3, 4}); // Output: [2,3,4]

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_5));

        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_1), 0, 0); // Channel 0
        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_2), 1, 0); // Channel 1
        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_3), 2, 0); // Channel 2

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::SUB, output, input1, input2);

        // Create expected tensor: broadcasting subtraction with implicit leading dimension
        Tensor<OutputType> expected({2, 3, 4});
        for(int n = 0; n < 2; ++n)
        {
            for(int c = 0; c < 3; ++c)
            {
                for(int h = 0; h < 4; ++h)
                {
                    float input1Val = TEST_VALUE_5;
                    auto input2Val = static_cast<float>(c + 1); // Channel values: 1.0, 2.0, 3.0
                    expected.setHostValue(static_cast<OutputType>(input1Val - input2Val), n, c, h);
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    // Test case: 4D × 4D: [N,C,H,W] + [1,C,1,1] → broadcast to [N,C,H,W]
    void testBroadcast4Dx4D()
    {
        Tensor<Input1Type> input1({2, 3, 2, 2}); // [N,C,H,W] = [2,3,2,2]
        Tensor<Input2Type> input2({1, 3, 1, 1}); // [1,C,1,1] = [1,3,1,1]
        Tensor<OutputType> output({2, 3, 2, 2}); // Output: [2,3,2,2]

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_1));

        input2.setHostValue(
            static_cast<Input2Type>(BROADCAST_MULTIPLIER_10), 0, 0, 0, 0); // Channel 0
        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_2 * BROADCAST_MULTIPLIER_10),
                            0,
                            1,
                            0,
                            0); // Channel 1
        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_3 * BROADCAST_MULTIPLIER_10),
                            0,
                            2,
                            0,
                            0); // Channel 2

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        // Create expected tensor: broadcasting addition
        Tensor<OutputType> expected({2, 3, 2, 2});
        for(int n = 0; n < 2; ++n)
        {
            for(int c = 0; c < 3; ++c)
            {
                for(int h = 0; h < 2; ++h)
                {
                    for(int w = 0; w < 2; ++w)
                    {
                        float input1Val = TEST_VALUE_1;
                        auto input2Val = static_cast<float>(
                            (static_cast<float>(c) + 1.0f)
                            * BROADCAST_MULTIPLIER_10); // Channel values: 10.0, 20.0, 30.0
                        expected.setHostValue(
                            static_cast<OutputType>(input1Val + input2Val), n, c, h, w);
                    }
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    // Test case: Complex N-D broadcasting: [2,1,3,1] + [1,2,1,4] → [2,2,3,4]
    void testBroadcastComplexND()
    {
        Tensor<Input1Type> input1({2, 1, 3, 1});
        Tensor<Input2Type> input2({1, 2, 1, 4});
        Tensor<OutputType> output({2, 2, 3, 4});

        for(int n = 0; n < 2; ++n)
        {
            for(int h = 0; h < 3; ++h)
            {
                input1.setHostValue(
                    static_cast<Input1Type>(static_cast<float>((n * 10) + h)), n, 0, h, 0);
            }
        }

        for(int c = 0; c < 2; ++c)
        {
            for(int w = 0; w < 4; ++w)
            {
                input2.setHostValue(
                    static_cast<Input2Type>(static_cast<float>((c * 100) + w)), 0, c, 0, w);
            }
        }

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        // Create expected tensor: broadcasting addition
        Tensor<OutputType> expected({2, 2, 3, 4});
        for(int n = 0; n < 2; ++n)
        {
            for(int c = 0; c < 2; ++c)
            {
                for(int h = 0; h < 3; ++h)
                {
                    for(int w = 0; w < 4; ++w)
                    {
                        // input1[n,0,h,0] broadcasts to input1[n,c,h,w]
                        auto input1Val = static_cast<float>((n * 10) + h);
                        // input2[0,c,0,w] broadcasts to input2[n,c,h,w]
                        auto input2Val = static_cast<float>((c * 100) + w);
                        expected.setHostValue(
                            static_cast<OutputType>(input1Val + input2Val), n, c, h, w);
                    }
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBroadcast5D()
    {
        std::vector<int64_t> dims1 = {2, 3, 2, 2, 2}; // [N,C,D,H,W] = [2,3,2,2,2]
        std::vector<int64_t> strides1 = {24, 8, 4, 2, 1}; // Row-major strides for [2,3,2,2,2]
        Tensor<Input1Type> input1(dims1, strides1);

        std::vector<int64_t> dims2 = {1, 3, 1, 1, 1}; // [1,C,1,1,1] = [1,3,1,1,1]
        std::vector<int64_t> strides2 = {3, 1, 1, 1, 1}; // Row-major strides for [1,3,1,1,1]
        Tensor<Input2Type> input2(dims2, strides2);

        std::vector<int64_t> outputDims = {2, 3, 2, 2, 2}; // Output: [2,3,2,2,2]
        std::vector<int64_t> outputStrides = {24, 8, 4, 2, 1}; // Row-major strides for [2,3,2,2,2]
        Tensor<OutputType> output(outputDims, outputStrides);

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_2));

        // Set channel-specific values in input2
        input2.setHostValue(
            static_cast<Input2Type>(BROADCAST_MULTIPLIER_10), 0, 0, 0, 0, 0); // Channel 0
        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_2 * BROADCAST_MULTIPLIER_10),
                            0,
                            1,
                            0,
                            0,
                            0); // Channel 1
        input2.setHostValue(static_cast<Input2Type>(TEST_VALUE_3 * BROADCAST_MULTIPLIER_10),
                            0,
                            2,
                            0,
                            0,
                            0); // Channel 2

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseForward(
            PointwiseMode::ADD, output, input1, input2);

        // Create expected tensor: broadcasting addition
        Tensor<OutputType> expected(outputDims, outputStrides);
        for(int n = 0; n < 2; ++n)
        {
            for(int c = 0; c < 3; ++c)
            {
                for(int d = 0; d < 2; ++d)
                {
                    for(int h = 0; h < 2; ++h)
                    {
                        for(int w = 0; w < 2; ++w)
                        {
                            auto input1Val = TEST_VALUE_2;
                            auto input2Val = static_cast<float>(
                                (static_cast<float>(c) + 1.0f)
                                * BROADCAST_MULTIPLIER_10); // Channel values: 10.0, 20.0, 30.0
                            expected.setHostValue(
                                static_cast<OutputType>(input1Val + input2Val), n, c, d, h, w);
                        }
                    }
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }
};

using TestTypes = ::testing::Types<float, double, half, hip_bfloat16>;
// Empty third argument required for C++17 compatibility with TYPED_TEST_SUITE macro
TYPED_TEST_SUITE(CpuReferencePointwiseBinaryTemplate, TestTypes, );

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryAddOperation)
{
    this->testBinaryAddOperation();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinarySubtractOperation)
{
    this->testBinarySubtractOperation();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryAddOperationSanityValidation)
{
    this->testBinaryAddOperationSanityValidation();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinarySubtractOperationSanityValidation)
{
    this->testBinarySubtractOperationSanityValidation();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryAddOperation2D)
{
    this->testBinaryAddOperation2D();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryAddOperation3D)
{
    this->testBinaryAddOperation3D();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinarySingleElementTensors)
{
    this->testBinarySingleElementTensors();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryNumericalPrecision)
{
    this->testBinaryNumericalPrecision();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryElementwise1D)
{
    this->testElementwise1D();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryBroadcast2Dx1D)
{
    this->testBroadcast2Dx1D();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryBroadcast3D)
{
    this->testBroadcast3D();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryBroadcast3DImplicitLeading)
{
    this->testBroadcast3DImplicitLeading();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryBroadcast4Dx4D)
{
    this->testBroadcast4Dx4D();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryBroadcastComplexND)
{
    this->testBroadcastComplexND();
}

TYPED_TEST(CpuReferencePointwiseBinaryTemplate, BinaryBroadcast5D)
{
    this->testBroadcast5D();
}

// Mixed-type binary test instantiations
using TestPointwiseMixed1 = CpuReferencePointwiseBinaryTemplate<float, half, hip_bfloat16>;
using TestPointwiseMixed2 = CpuReferencePointwiseBinaryTemplate<half, float, hip_bfloat16>;
using TestPointwiseMixed3 = CpuReferencePointwiseBinaryTemplate<float, float, half>;
using TestPointwiseMixed4 = CpuReferencePointwiseBinaryTemplate<hip_bfloat16, float, half>;
using TestPointwiseMixed5 = CpuReferencePointwiseBinaryTemplate<hip_bfloat16, half, float>;
using TestPointwiseMixed6 = CpuReferencePointwiseBinaryTemplate<float, hip_bfloat16, half>;
using TestPointwiseMixed7 = CpuReferencePointwiseBinaryTemplate<half, hip_bfloat16, float>;

// Test a sample of mixed-type binary operations
TEST_F(TestPointwiseMixed1, BinaryMixedTypeAddOperation)
{
    this->testBinaryAddOperation();
}

TEST_F(TestPointwiseMixed1, BinaryMixedTypeSubtractOperation)
{
    this->testBinarySubtractOperation();
}

TEST_F(TestPointwiseMixed2, BinaryMixedTypeAddOperation)
{
    this->testBinaryAddOperation();
}

TEST_F(TestPointwiseMixed2, BinaryMixedTypeSubtractOperation)
{
    this->testBinarySubtractOperation();
}

TEST_F(TestPointwiseMixed3, BinaryMixedTypeAddOperation)
{
    this->testBinaryAddOperation();
}

TEST_F(TestPointwiseMixed3, BinaryMixedTypeSubtractOperation)
{
    this->testBinarySubtractOperation();
}

TEST_F(TestPointwiseMixed4, BinaryMixedTypeAddOperation)
{
    this->testBinaryAddOperation();
}

TEST_F(TestPointwiseMixed4, BinaryMixedTypeSubtractOperation)
{
    this->testBinarySubtractOperation();
}

TEST_F(TestPointwiseMixed5, BinaryMixedTypeAddOperation)
{
    this->testBinaryAddOperation();
}

TEST_F(TestPointwiseMixed5, BinaryMixedTypeSubtractOperation)
{
    this->testBinarySubtractOperation();
}

TEST_F(TestPointwiseMixed6, BinaryMixedTypeAddOperation)
{
    this->testBinaryAddOperation();
}

TEST_F(TestPointwiseMixed6, BinaryMixedTypeSubtractOperation)
{
    this->testBinarySubtractOperation();
}

TEST_F(TestPointwiseMixed7, BinaryMixedTypeAddOperation)
{
    this->testBinaryAddOperation();
}

TEST_F(TestPointwiseMixed7, BinaryMixedTypeSubtractOperation)
{
    this->testBinarySubtractOperation();
}
