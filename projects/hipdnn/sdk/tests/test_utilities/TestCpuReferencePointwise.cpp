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
class CpuReferencePointwiseTemplate : public ::testing::Test
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

    // ======================= BINARY OPERATIONS =======================

    void testBinaryAddOperation()
    {
        Tensor<Input1Type> input1({1, 3, 2, 2});
        Tensor<Input2Type> input2({1, 3, 2, 2});
        Tensor<OutputType> output({1, 3, 2, 2});

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_1));
        input2.fillWithValue(static_cast<Input2Type>(TEST_VALUE_2));

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
            PointwiseMode::ADD, output, input1, input2);

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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

    void testBinarySubtractOperation()
    {
        Tensor<Input1Type> input1({1, 3, 2, 2});
        Tensor<Input2Type> input2({1, 3, 2, 2});
        Tensor<OutputType> output({1, 3, 2, 2});

        input1.fillWithValue(static_cast<Input1Type>(TEST_VALUE_5));
        input2.fillWithValue(static_cast<Input2Type>(TEST_VALUE_2));

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
            PointwiseMode::SUB, output, input1, input2);

        Tensor<OutputType> expected({1, 3, 2, 2});
        expected.fillWithValue(static_cast<OutputType>(TEST_VALUE_3));

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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

        CpuReferencePointwiseImpl<OutputType, Input1Type, Input2Type>::pointwiseCompute(
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

    // ======================= UNARY OPERATIONS =======================

    void testReluForwardOperation()
    {
        Tensor<Input1Type> input({1, 3, 2, 2});
        Tensor<OutputType> output({1, 3, 2, 2});

        // Fill with mix of positive and negative values
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1), 0, 0, 0, 0); // 1.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_2), 0, 0, 0, 1); // -2.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_3), 0, 0, 1, 0); // 3.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_1), 0, 0, 1, 1); // -1.0
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 1, 0, 0); // 0.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_5), 0, 1, 0, 1); // 5.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_5), 0, 1, 1, 0); // -5.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_2_5), 0, 1, 1, 1); // 2.5
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_1_5), 0, 2, 0, 0); // -1.5
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_4), 0, 2, 0, 1); // 4.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_3), 0, 2, 1, 0); // -3.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1_5), 0, 2, 1, 1); // 1.5

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::RELU_FWD, output, input);

        // Create expected tensor: ReLU(x) = max(0, x)
        Tensor<OutputType> expected({1, 3, 2, 2});
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_1), 0, 0, 0, 0); // max(0, 1) = 1
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 0, 0, 1); // max(0, -2) = 0
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_3), 0, 0, 1, 0); // max(0, 3) = 3
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 0, 1, 1); // max(0, -1) = 0
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 1, 0, 0); // max(0, 0) = 0
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_5), 0, 1, 0, 1); // max(0, 5) = 5
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 1, 1, 0); // max(0, -5) = 0
        expected.setHostValue(
            static_cast<OutputType>(TEST_VALUE_2_5), 0, 1, 1, 1); // max(0, 2.5) = 2.5
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 2, 0, 0); // max(0, -1.5) = 0
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_4), 0, 2, 0, 1); // max(0, 4) = 4
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 2, 1, 0); // max(0, -3) = 0
        expected.setHostValue(
            static_cast<OutputType>(TEST_VALUE_1_5), 0, 2, 1, 1); // max(0, 1.5) = 1.5

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testReluBackwardOperation()
    {
        Tensor<Input1Type> input({1, 2, 2, 2});
        Tensor<OutputType> output({1, 2, 2, 2});

        // Fill with mix of positive and negative values
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1), 0, 0, 0, 0); // 1.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_2), 0, 0, 0, 1); // -2.0
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 0, 1, 0); // 0.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_3), 0, 0, 1, 1); // 3.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_1), 0, 1, 0, 0); // -1.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_2_5), 0, 1, 0, 1); // 2.5
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_5), 0, 1, 1, 0); // -5.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1_5), 0, 1, 1, 1); // 1.5

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::RELU_BWD, output, input);

        // Create expected tensor: ReLU_backward(x) = x > 0 ? 1 : 0
        Tensor<OutputType> expected({1, 2, 2, 2});
        expected.setHostValue(static_cast<OutputType>(1.0f), 0, 0, 0, 0); // 1 > 0 ? 1 : 0 = 1
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 0, 0, 1); // -2 > 0 ? 1 : 0 = 0
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 0, 1, 0); // 0 > 0 ? 1 : 0 = 0
        expected.setHostValue(static_cast<OutputType>(1.0f), 0, 0, 1, 1); // 3 > 0 ? 1 : 0 = 1
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 1, 0, 0); // -1 > 0 ? 1 : 0 = 0
        expected.setHostValue(static_cast<OutputType>(1.0f), 0, 1, 0, 1); // 2.5 > 0 ? 1 : 0 = 1
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 1, 1, 0); // -5 > 0 ? 1 : 0 = 0
        expected.setHostValue(static_cast<OutputType>(1.0f), 0, 1, 1, 1); // 1.5 > 0 ? 1 : 0 = 1

        // Add debug output for the zero case
        std::cout << "DEBUG: For input x=0.0, output="
                  << static_cast<float>(output.getHostValue(0, 0, 1, 0))
                  << ", expected=" << static_cast<float>(expected.getHostValue(0, 0, 1, 0)) << "\n";

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testParameterizedReluForwardOperation()
    {
        Tensor<Input1Type> input({1, 2, 2, 2});
        Tensor<OutputType> output({1, 2, 2, 2});

        // Fill with values that will test all three regions: below lower_clip, between clips, above upper_clip
        input.setHostValue(
            static_cast<Input1Type>(-TEST_VALUE_5), 0, 0, 0, 0); // -5.0 (below lower_clip)
        input.setHostValue(
            static_cast<Input1Type>(-TEST_VALUE_1), 0, 0, 0, 1); // -1.0 (between clips)
        input.setHostValue(
            static_cast<Input1Type>(TEST_VALUE_2), 0, 0, 1, 0); // 2.0 (between clips)
        input.setHostValue(
            static_cast<Input1Type>(TEST_VALUE_5), 0, 0, 1, 1); // 5.0 (above upper_clip)
        input.setHostValue(
            static_cast<Input1Type>(-TEST_VALUE_2), 0, 1, 0, 0); // -2.0 (at lower_clip)
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 1, 0, 1); // 0.0 (between clips)
        input.setHostValue(
            static_cast<Input1Type>(TEST_VALUE_4), 0, 1, 1, 0); // 4.0 (at upper_clip)
        input.setHostValue(
            static_cast<Input1Type>(TEST_VALUE_1), 0, 1, 1, 1); // 1.0 (between clips)

        // Parameters: lower_clip = -2.0, upper_clip = 4.0, lower_slope = 0.1
        float lowerClip = -TEST_VALUE_2; // -2.0
        float upperClip = TEST_VALUE_4; // 4.0
        auto lowerSlope = static_cast<float>(0.1);

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::RELU_FWD, output, input, lowerClip, upperClip, lowerSlope);

        // Create expected tensor: Generalized ReLU with parameters
        Tensor<OutputType> expected({1, 2, 2, 2});
        // For x < lower_clip: output = lower_clip + lower_slope * (x - lower_clip)
        // For lower_clip <= x <= upper_clip: output = x
        // For x > upper_clip: output = upper_clip
        expected.setHostValue(
            static_cast<OutputType>(lowerClip + (lowerSlope * (-TEST_VALUE_5 - lowerClip))),
            0,
            0,
            0,
            0); // -2 + 0.1*(-5-(-2)) = -2.3
        expected.setHostValue(
            static_cast<OutputType>(-TEST_VALUE_1), 0, 0, 0, 1); // -1.0 (in range)
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_2), 0, 0, 1, 0); // 2.0 (in range)
        expected.setHostValue(static_cast<OutputType>(upperClip), 0, 0, 1, 1); // 4.0 (clamped)
        expected.setHostValue(static_cast<OutputType>(lowerClip), 0, 1, 0, 0); // -2.0 (at boundary)
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 1, 0, 1); // 0.0 (in range)
        expected.setHostValue(static_cast<OutputType>(upperClip), 0, 1, 1, 0); // 4.0 (at boundary)
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_1), 0, 1, 1, 1); // 1.0 (in range)

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testParameterizedReluBackwardOperation()
    {
        Tensor<Input1Type> input({1, 2, 2, 2});
        Tensor<OutputType> output({1, 2, 2, 2});

        // Fill with values that will test all three regions: below lower_clip, between clips, above upper_clip
        input.setHostValue(
            static_cast<Input1Type>(-TEST_VALUE_5), 0, 0, 0, 0); // -5.0 (below lower_clip)
        input.setHostValue(
            static_cast<Input1Type>(-TEST_VALUE_1), 0, 0, 0, 1); // -1.0 (between clips)
        input.setHostValue(
            static_cast<Input1Type>(TEST_VALUE_2), 0, 0, 1, 0); // 2.0 (between clips)
        input.setHostValue(
            static_cast<Input1Type>(TEST_VALUE_5), 0, 0, 1, 1); // 5.0 (above upper_clip)
        input.setHostValue(
            static_cast<Input1Type>(-TEST_VALUE_2), 0, 1, 0, 0); // -2.0 (at lower_clip)
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 1, 0, 1); // 0.0 (between clips)
        input.setHostValue(
            static_cast<Input1Type>(TEST_VALUE_4), 0, 1, 1, 0); // 4.0 (at upper_clip)
        input.setHostValue(
            static_cast<Input1Type>(TEST_VALUE_1), 0, 1, 1, 1); // 1.0 (between clips)

        // Parameters: lower_clip = -2.0, upper_clip = 4.0, lower_slope = 0.1
        float lowerClip = -TEST_VALUE_2; // -2.0
        float upperClip = TEST_VALUE_4; // 4.0
        auto lowerSlope = static_cast<float>(0.1);

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::RELU_BWD, output, input, lowerClip, upperClip, lowerSlope);

        // Create expected tensor: Generalized ReLU backward with parameters
        Tensor<OutputType> expected({1, 2, 2, 2});
        // For x < lower_clip: derivative = lower_slope
        // For lower_clip <= x <= upper_clip: derivative = 1.0
        // For x > upper_clip: derivative = 0.0
        expected.setHostValue(static_cast<OutputType>(lowerSlope),
                              0,
                              0,
                              0,
                              0); // below lower_clip -> lower_slope = 0.1
        expected.setHostValue(static_cast<OutputType>(1.0f), 0, 0, 0, 1); // in range -> 1.0
        expected.setHostValue(static_cast<OutputType>(1.0f), 0, 0, 1, 0); // in range -> 1.0
        expected.setHostValue(static_cast<OutputType>(0.0f), 0, 0, 1, 1); // above upper_clip -> 0.0
        expected.setHostValue(
            static_cast<OutputType>(1.0f), 0, 1, 0, 0); // at lower_clip boundary -> 1.0
        expected.setHostValue(static_cast<OutputType>(1.0f), 0, 1, 0, 1); // in range -> 1.0
        expected.setHostValue(
            static_cast<OutputType>(1.0f), 0, 1, 1, 0); // at upper_clip boundary -> 1.0
        expected.setHostValue(static_cast<OutputType>(1.0f), 0, 1, 1, 1); // in range -> 1.0

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testSigmoidForwardOperation()
    {
        Tensor<Input1Type> input({1, 2, 2, 2});
        Tensor<OutputType> output({1, 2, 2, 2});

        // Fill with values that will test sigmoid function
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 0, 0, 0); // 0.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1), 0, 0, 0, 1); // 1.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_1), 0, 0, 1, 0); // -1.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_2), 0, 0, 1, 1); // 2.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_2), 0, 1, 0, 0); // -2.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_5), 0, 1, 0, 1); // 5.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_5), 0, 1, 1, 0); // -5.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1_5), 0, 1, 1, 1); // 1.5

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::SIGMOID_FWD, output, input);

        // Create expected tensor: sigmoid(x) = 1 / (1 + exp(-x))
        Tensor<OutputType> expected({1, 2, 2, 2});
        expected.setHostValue(static_cast<OutputType>(1.0f / (1.0f + std::exp(-0.0f))),
                              0,
                              0,
                              0,
                              0); // sigmoid(0) = 0.5
        expected.setHostValue(static_cast<OutputType>(1.0f / (1.0f + std::exp(-TEST_VALUE_1))),
                              0,
                              0,
                              0,
                              1); // sigmoid(1)
        expected.setHostValue(static_cast<OutputType>(1.0f / (1.0f + std::exp(TEST_VALUE_1))),
                              0,
                              0,
                              1,
                              0); // sigmoid(-1)
        expected.setHostValue(static_cast<OutputType>(1.0f / (1.0f + std::exp(-TEST_VALUE_2))),
                              0,
                              0,
                              1,
                              1); // sigmoid(2)
        expected.setHostValue(static_cast<OutputType>(1.0f / (1.0f + std::exp(TEST_VALUE_2))),
                              0,
                              1,
                              0,
                              0); // sigmoid(-2)
        expected.setHostValue(static_cast<OutputType>(1.0f / (1.0f + std::exp(-TEST_VALUE_5))),
                              0,
                              1,
                              0,
                              1); // sigmoid(5)
        expected.setHostValue(static_cast<OutputType>(1.0f / (1.0f + std::exp(TEST_VALUE_5))),
                              0,
                              1,
                              1,
                              0); // sigmoid(-5)
        expected.setHostValue(static_cast<OutputType>(1.0f / (1.0f + std::exp(-TEST_VALUE_1_5))),
                              0,
                              1,
                              1,
                              1); // sigmoid(1.5)

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testSigmoidBackwardOperation()
    {
        Tensor<Input1Type> input({1, 2, 2, 2});
        Tensor<OutputType> output({1, 2, 2, 2});

        // Fill with values that will test sigmoid backward function
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 0, 0, 0); // 0.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1), 0, 0, 0, 1); // 1.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_1), 0, 0, 1, 0); // -1.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_2), 0, 0, 1, 1); // 2.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_2), 0, 1, 0, 0); // -2.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_3), 0, 1, 0, 1); // 3.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_3), 0, 1, 1, 0); // -3.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1_5), 0, 1, 1, 1); // 1.5

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::SIGMOID_BWD, output, input);

        // Create expected tensor: sigmoid_backward(x) = sigmoid(x) * (1 - sigmoid(x))
        Tensor<OutputType> expected({1, 2, 2, 2});
        for(int n = 0; n < 1; ++n)
        {
            for(int c = 0; c < 2; ++c)
            {
                for(int h = 0; h < 2; ++h)
                {
                    for(int w = 0; w < 2; ++w)
                    {
                        auto inputVal = input.getHostValue(n, c, h, w);
                        auto sigmoid = 1.0f / (1.0f + std::exp(-static_cast<float>(inputVal)));
                        auto derivative = sigmoid * (1.0f - sigmoid);
                        expected.setHostValue(static_cast<OutputType>(derivative), n, c, h, w);
                    }
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testTanhForwardOperation()
    {
        Tensor<Input1Type> input({1, 2, 2, 2});
        Tensor<OutputType> output({1, 2, 2, 2});

        // Fill with values that will test tanh function
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 0, 0, 0); // 0.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1), 0, 0, 0, 1); // 1.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_1), 0, 0, 1, 0); // -1.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_2), 0, 0, 1, 1); // 2.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_2), 0, 1, 0, 0); // -2.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_3), 0, 1, 0, 1); // 3.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_3), 0, 1, 1, 0); // -3.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1_5), 0, 1, 1, 1); // 1.5

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::TANH_FWD, output, input);

        // Create expected tensor: tanh(x)
        Tensor<OutputType> expected({1, 2, 2, 2});
        expected.setHostValue(static_cast<OutputType>(std::tanh(0.0f)), 0, 0, 0, 0); // tanh(0) = 0
        expected.setHostValue(
            static_cast<OutputType>(std::tanh(TEST_VALUE_1)), 0, 0, 0, 1); // tanh(1)
        expected.setHostValue(
            static_cast<OutputType>(std::tanh(-TEST_VALUE_1)), 0, 0, 1, 0); // tanh(-1)
        expected.setHostValue(
            static_cast<OutputType>(std::tanh(TEST_VALUE_2)), 0, 0, 1, 1); // tanh(2)
        expected.setHostValue(
            static_cast<OutputType>(std::tanh(-TEST_VALUE_2)), 0, 1, 0, 0); // tanh(-2)
        expected.setHostValue(
            static_cast<OutputType>(std::tanh(TEST_VALUE_3)), 0, 1, 0, 1); // tanh(3)
        expected.setHostValue(
            static_cast<OutputType>(std::tanh(-TEST_VALUE_3)), 0, 1, 1, 0); // tanh(-3)
        expected.setHostValue(
            static_cast<OutputType>(std::tanh(TEST_VALUE_1_5)), 0, 1, 1, 1); // tanh(1.5)

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testTanhBackwardOperation()
    {
        Tensor<Input1Type> input({1, 2, 2, 2});
        Tensor<OutputType> output({1, 2, 2, 2});

        // Fill with values that will test tanh backward function
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 0, 0, 0); // 0.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1), 0, 0, 0, 1); // 1.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_1), 0, 0, 1, 0); // -1.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_2), 0, 0, 1, 1); // 2.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_2), 0, 1, 0, 0); // -2.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_3), 0, 1, 0, 1); // 3.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_3), 0, 1, 1, 0); // -3.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1_5), 0, 1, 1, 1); // 1.5

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::TANH_BWD, output, input);

        // Create expected tensor: tanh_backward(x) = 1 - tanh²(x)
        Tensor<OutputType> expected({1, 2, 2, 2});
        for(int n = 0; n < 1; ++n)
        {
            for(int c = 0; c < 2; ++c)
            {
                for(int h = 0; h < 2; ++h)
                {
                    for(int w = 0; w < 2; ++w)
                    {
                        auto inputVal = input.getHostValue(n, c, h, w);
                        auto tanhVal = std::tanh(static_cast<float>(inputVal));
                        auto derivative = 1.0f - (tanhVal * tanhVal);
                        expected.setHostValue(static_cast<OutputType>(derivative), n, c, h, w);
                    }
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testAbsoluteValueOperation()
    {
        Tensor<Input1Type> input({1, 2, 2, 2});
        Tensor<OutputType> output({1, 2, 2, 2});

        // Fill with mix of positive, negative, and zero values
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1), 0, 0, 0, 0); // 1.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_2), 0, 0, 0, 1); // -2.0
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 0, 1, 0); // 0.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_3), 0, 0, 1, 1); // -3.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_5), 0, 1, 0, 0); // 5.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_1_5), 0, 1, 0, 1); // -1.5
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_2_5), 0, 1, 1, 0); // 2.5
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_4), 0, 1, 1, 1); // -4.0

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::ABS, output, input);

        // Create expected tensor: abs(x)
        Tensor<OutputType> expected({1, 2, 2, 2});
        expected.setHostValue(
            static_cast<OutputType>(std::abs(TEST_VALUE_1)), 0, 0, 0, 0); // |1| = 1
        expected.setHostValue(
            static_cast<OutputType>(std::abs(-TEST_VALUE_2)), 0, 0, 0, 1); // |-2| = 2
        expected.setHostValue(static_cast<OutputType>(std::abs(0.0f)), 0, 0, 1, 0); // |0| = 0
        expected.setHostValue(
            static_cast<OutputType>(std::abs(-TEST_VALUE_3)), 0, 0, 1, 1); // |-3| = 3
        expected.setHostValue(
            static_cast<OutputType>(std::abs(TEST_VALUE_5)), 0, 1, 0, 0); // |5| = 5
        expected.setHostValue(
            static_cast<OutputType>(std::abs(-TEST_VALUE_1_5)), 0, 1, 0, 1); // |-1.5| = 1.5
        expected.setHostValue(
            static_cast<OutputType>(std::abs(TEST_VALUE_2_5)), 0, 1, 1, 0); // |2.5| = 2.5
        expected.setHostValue(
            static_cast<OutputType>(std::abs(-TEST_VALUE_4)), 0, 1, 1, 1); // |-4| = 4

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testNegationOperation()
    {
        Tensor<Input1Type> input({1, 2, 2, 2});
        Tensor<OutputType> output({1, 2, 2, 2});

        // Fill with mix of positive, negative, and zero values
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_1), 0, 0, 0, 0); // 1.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_2), 0, 0, 0, 1); // -2.0
        input.setHostValue(static_cast<Input1Type>(0.0f), 0, 0, 1, 0); // 0.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_3), 0, 0, 1, 1); // -3.0
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_5), 0, 1, 0, 0); // 5.0
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_1_5), 0, 1, 0, 1); // -1.5
        input.setHostValue(static_cast<Input1Type>(TEST_VALUE_2_5), 0, 1, 1, 0); // 2.5
        input.setHostValue(static_cast<Input1Type>(-TEST_VALUE_4), 0, 1, 1, 1); // -4.0

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::NEG, output, input);

        // Create expected tensor: -x
        Tensor<OutputType> expected({1, 2, 2, 2});
        expected.setHostValue(static_cast<OutputType>(-TEST_VALUE_1), 0, 0, 0, 0); // -1
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_2), 0, 0, 0, 1); // -(-2) = 2
        expected.setHostValue(static_cast<OutputType>(-0.0f), 0, 0, 1, 0); // -0 = 0
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_3), 0, 0, 1, 1); // -(-3) = 3
        expected.setHostValue(static_cast<OutputType>(-TEST_VALUE_5), 0, 1, 0, 0); // -5
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_1_5), 0, 1, 0, 1); // -(-1.5) = 1.5
        expected.setHostValue(static_cast<OutputType>(-TEST_VALUE_2_5), 0, 1, 1, 0); // -2.5
        expected.setHostValue(static_cast<OutputType>(TEST_VALUE_4), 0, 1, 1, 1); // -(-4) = 4

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testUnary1DOperation()
    {
        Tensor<Input1Type> input({5});
        Tensor<OutputType> output({5});

        // Fill with simple values for 1D testing
        for(int i = 0; i < 5; ++i)
        {
            input.setHostValue(static_cast<Input1Type>(static_cast<float>(i - 2)),
                               i); // [-2, -1, 0, 1, 2]
        }

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::RELU_FWD, output, input);

        // Create expected tensor: ReLU applied to [-2, -1, 0, 1, 2] = [0, 0, 0, 1, 2]
        Tensor<OutputType> expected({5});
        expected.setHostValue(static_cast<OutputType>(0.0f), 0); // max(0, -2) = 0
        expected.setHostValue(static_cast<OutputType>(0.0f), 1); // max(0, -1) = 0
        expected.setHostValue(static_cast<OutputType>(0.0f), 2); // max(0, 0) = 0
        expected.setHostValue(static_cast<OutputType>(1.0f), 3); // max(0, 1) = 1
        expected.setHostValue(static_cast<OutputType>(2.0f), 4); // max(0, 2) = 2

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testUnary2DOperation()
    {
        Tensor<Input1Type> input({3, 4});
        Tensor<OutputType> output({3, 4});

        // Fill with pattern for 2D testing
        for(int m = 0; m < 3; ++m)
        {
            for(int n = 0; n < 4; ++n)
            {
                input.setHostValue(
                    static_cast<Input1Type>(static_cast<float>((m - 1) + (n - 2))), m, n);
            }
        }

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::ABS, output, input);

        // Create expected tensor: abs applied to the pattern
        Tensor<OutputType> expected({3, 4});
        for(int m = 0; m < 3; ++m)
        {
            for(int n = 0; n < 4; ++n)
            {
                auto val = static_cast<float>((m - 1) + (n - 2));
                expected.setHostValue(static_cast<OutputType>(std::abs(val)), m, n);
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testUnary3DOperation()
    {
        Tensor<Input1Type> input({2, 3, 4});
        Tensor<OutputType> output({2, 3, 4});

        input.fillWithValue(static_cast<Input1Type>(-TEST_VALUE_2_5));

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::NEG, output, input);

        // Create expected tensor: negation of -2.5 = 2.5
        Tensor<OutputType> expected({2, 3, 4});
        expected.fillWithValue(static_cast<OutputType>(TEST_VALUE_2_5));

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testUnarySingleElementTensor()
    {
        Tensor<Input1Type> input({1, 1, 1, 1});
        Tensor<OutputType> output({1, 1, 1, 1});

        input.setHostValue(static_cast<Input1Type>(E), 0, 0, 0, 0); // e

        CpuReferencePointwiseImpl<OutputType, Input1Type>::pointwiseCompute(
            PointwiseMode::TANH_FWD, output, input);

        // Create expected tensor: tanh(e)
        Tensor<OutputType> expected({1, 1, 1, 1});
        expected.setHostValue(static_cast<OutputType>(std::tanh(E)), 0, 0, 0, 0);

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<OutputType> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }
};

using TestTypes = ::testing::Types<float, double, half, hip_bfloat16>;
// Empty third argument required for C++17 compatibility with TYPED_TEST_SUITE macro
TYPED_TEST_SUITE(CpuReferencePointwiseTemplate, TestTypes, );

// Binary operation tests
TYPED_TEST(CpuReferencePointwiseTemplate, BinaryAddOperation)
{
    this->testBinaryAddOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinarySubtractOperation)
{
    this->testBinarySubtractOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinaryAddOperationSanityValidation)
{
    this->testBinaryAddOperationSanityValidation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinarySubtractOperationSanityValidation)
{
    this->testBinarySubtractOperationSanityValidation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinarySingleElementTensors)
{
    this->testBinarySingleElementTensors();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinaryNumericalPrecision)
{
    this->testBinaryNumericalPrecision();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinaryElementwise1D)
{
    this->testElementwise1D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinaryBroadcast2Dx1D)
{
    this->testBroadcast2Dx1D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinaryBroadcast3D)
{
    this->testBroadcast3D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinaryBroadcast3DImplicitLeading)
{
    this->testBroadcast3DImplicitLeading();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinaryBroadcast4Dx4D)
{
    this->testBroadcast4Dx4D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinaryBroadcastComplexND)
{
    this->testBroadcastComplexND();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BinaryBroadcast5D)
{
    this->testBroadcast5D();
}

// Unary operation tests
TYPED_TEST(CpuReferencePointwiseTemplate, UnaryReluForward)
{
    this->testReluForwardOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnaryReluBackward)
{
    this->testReluBackwardOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnaryParameterizedReluForward)
{
    this->testParameterizedReluForwardOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnaryParameterizedReluBackward)
{
    this->testParameterizedReluBackwardOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnarySigmoidForward)
{
    this->testSigmoidForwardOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnarySigmoidBackward)
{
    this->testSigmoidBackwardOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnaryTanhForward)
{
    this->testTanhForwardOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnaryTanhBackward)
{
    this->testTanhBackwardOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnaryAbsoluteValue)
{
    this->testAbsoluteValueOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnaryNegation)
{
    this->testNegationOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, Unary1DOperation)
{
    this->testUnary1DOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, Unary2DOperation)
{
    this->testUnary2DOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, Unary3DOperation)
{
    this->testUnary3DOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, UnarySingleElementTensor)
{
    this->testUnarySingleElementTensor();
}

// Mixed-type binary test instantiations
using TestPointwiseMixed1 = CpuReferencePointwiseTemplate<float, half, hip_bfloat16>;
using TestPointwiseMixed2 = CpuReferencePointwiseTemplate<half, float, hip_bfloat16>;
using TestPointwiseMixed3 = CpuReferencePointwiseTemplate<float, float, half>;
using TestPointwiseMixed4 = CpuReferencePointwiseTemplate<hip_bfloat16, float, half>;
using TestPointwiseMixed5 = CpuReferencePointwiseTemplate<hip_bfloat16, half, float>;
using TestPointwiseMixed6 = CpuReferencePointwiseTemplate<float, hip_bfloat16, half>;
using TestPointwiseMixed7 = CpuReferencePointwiseTemplate<half, hip_bfloat16, float>;

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

// Mixed-type unary test instantiations
using TestPointwiseUnaryMixed1 = CpuReferencePointwiseTemplate<float, half>;
using TestPointwiseUnaryMixed2 = CpuReferencePointwiseTemplate<half, float>;
using TestPointwiseUnaryMixed3 = CpuReferencePointwiseTemplate<float, hip_bfloat16>;
using TestPointwiseUnaryMixed4 = CpuReferencePointwiseTemplate<hip_bfloat16, float>;
using TestPointwiseUnaryMixed5 = CpuReferencePointwiseTemplate<half, hip_bfloat16>;
using TestPointwiseUnaryMixed6 = CpuReferencePointwiseTemplate<hip_bfloat16, half>;

// Test a sample of mixed-type unary operations
TEST_F(TestPointwiseUnaryMixed1, UnaryMixedTypeReluForward)
{
    this->testReluForwardOperation();
}

TEST_F(TestPointwiseUnaryMixed1, UnaryMixedTypeSigmoidForward)
{
    this->testSigmoidForwardOperation();
}

TEST_F(TestPointwiseUnaryMixed2, UnaryMixedTypeAbsoluteValue)
{
    this->testAbsoluteValueOperation();
}

TEST_F(TestPointwiseUnaryMixed2, UnaryMixedTypeNegation)
{
    this->testNegationOperation();
}

TEST_F(TestPointwiseUnaryMixed3, UnaryMixedTypeTanhForward)
{
    this->testTanhForwardOperation();
}

TEST_F(TestPointwiseUnaryMixed3, UnaryMixedTypeReluBackward)
{
    this->testReluBackwardOperation();
}

TEST_F(TestPointwiseUnaryMixed4, UnaryMixedTypeParameterizedRelu)
{
    this->testParameterizedReluForwardOperation();
}

TEST_F(TestPointwiseUnaryMixed4, UnaryMixedTypeSigmoidBackward)
{
    this->testSigmoidBackwardOperation();
}

TEST_F(TestPointwiseUnaryMixed5, UnaryMixedTypeTanhBackward)
{
    this->testTanhBackwardOperation();
}

TEST_F(TestPointwiseUnaryMixed5, UnaryMixedTypeAbsoluteValue)
{
    this->testAbsoluteValueOperation();
}

TEST_F(TestPointwiseUnaryMixed6, UnaryMixedTypeReluForward)
{
    this->testReluForwardOperation();
}

TEST_F(TestPointwiseUnaryMixed6, UnaryMixedTypeParameterizedReluBackward)
{
    this->testParameterizedReluBackwardOperation();
}
