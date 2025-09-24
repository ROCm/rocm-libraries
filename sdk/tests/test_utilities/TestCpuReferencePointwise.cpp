// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <numbers>
using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

template <typename T>
class CpuReferencePointwiseTemplate : public ::testing::Test
{
protected:
    T getTolerance() const
    {
        if constexpr(std::is_same_v<T, half>)
        {
            return static_cast<T>(1e-3f);
        }
        else if constexpr(std::is_same_v<T, hip_bfloat16>)
        {
            return static_cast<T>(1e-2f);
        }
        else
        {
            return static_cast<T>(1e-5f);
        }
    }

    void testAddOperation()
    {
        Tensor<T> input1({1, 3, 2, 2});
        Tensor<T> input2({1, 3, 2, 2});
        Tensor<T> output({1, 3, 2, 2});

        input1.fillWithValue(static_cast<T>(1.0));
        input2.fillWithValue(static_cast<T>(2.0));

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        Tensor<T> expected({1, 3, 2, 2});
        expected.fillWithValue(static_cast<T>(3.0));

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testSubtractOperation()
    {
        Tensor<T> input1({1, 3, 2, 2});
        Tensor<T> input2({1, 3, 2, 2});
        Tensor<T> output({1, 3, 2, 2});

        input1.fillWithValue(static_cast<T>(5.0));
        input2.fillWithValue(static_cast<T>(2.0));

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::SUB, input1, input2, output);

        Tensor<T> expected({1, 3, 2, 2});
        expected.fillWithValue(static_cast<T>(3.0));

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testAddOperationSanityValidation()
    {
        Tensor<T> input1({1, 1, 2, 2});
        Tensor<T> input2({1, 1, 2, 2});
        Tensor<T> output({1, 1, 2, 2});

        input1.setHostValue(static_cast<T>(std::numbers::pi_v<float>), 0, 0, 0, 0);
        input1.setHostValue(static_cast<T>(std::numbers::e_v<float>), 0, 0, 0, 1);
        input1.setHostValue(static_cast<T>(std::numbers::sqrt2_v<float>), 0, 0, 1, 0);
        input1.setHostValue(static_cast<T>(std::numbers::phi_v<float>), 0, 0, 1, 1);

        input2.setHostValue(static_cast<T>(std::numbers::ln2_v<float>), 0, 0, 0, 0);
        input2.setHostValue(static_cast<T>(std::sin(1.0f)), 0, 0, 0, 1);
        input2.setHostValue(static_cast<T>(std::cos(1.0f)), 0, 0, 1, 0);
        input2.setHostValue(static_cast<T>(std::tan(1.0f)), 0, 0, 1, 1);

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        // Create expected tensor with computed results
        Tensor<T> expected({1, 1, 2, 2});
        expected.setHostValue(
            static_cast<T>(std::numbers::pi_v<float> + std::numbers::ln2_v<float>), 0, 0, 0, 0);
        expected.setHostValue(
            static_cast<T>(std::numbers::e_v<float> + std::sin(1.0f)), 0, 0, 0, 1);
        expected.setHostValue(
            static_cast<T>(std::numbers::sqrt2_v<float> + std::cos(1.0f)), 0, 0, 1, 0);
        expected.setHostValue(
            static_cast<T>(std::numbers::phi_v<float> + std::tan(1.0f)), 0, 0, 1, 1);

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testSubtractOperationSanityValidation()
    {
        Tensor<T> input1({1, 1, 2, 2});
        Tensor<T> input2({1, 1, 2, 2});
        Tensor<T> output({1, 1, 2, 2});

        input1.setHostValue(static_cast<T>(2.0f * std::numbers::pi_v<float>), 0, 0, 0, 0);
        input1.setHostValue(
            static_cast<T>(std::numbers::e_v<float> * std::numbers::e_v<float>), 0, 0, 0, 1); // e²
        input1.setHostValue(static_cast<T>(std::sqrt(5.0f)), 0, 0, 1, 0); // √5
        input1.setHostValue(static_cast<T>(2.0f), 0, 0, 1, 1); // log₁₀(100)

        input2.setHostValue(static_cast<T>(std::numbers::pi_v<float> / 2.0f), 0, 0, 0, 0); // π/2
        input2.setHostValue(static_cast<T>(std::numbers::e_v<float>), 0, 0, 0, 1); // e
        input2.setHostValue(static_cast<T>(std::numbers::sqrt3_v<float>), 0, 0, 1, 0); // √3
        input2.setHostValue(static_cast<T>(1.0f), 0, 0, 1, 1); // log₁₀(10)

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::SUB, input1, input2, output);

        // Create expected tensor with computed results
        Tensor<T> expected({1, 1, 2, 2});
        expected.setHostValue(
            static_cast<T>((2.0f * std::numbers::pi_v<float>)-(std::numbers::pi_v<float> / 2.0f)),
            0,
            0,
            0,
            0); // 2π - π/2 = 3π/2
        expected.setHostValue(static_cast<T>((std::numbers::e_v<float>
                                              * std::numbers::e_v<float>)-std::numbers::e_v<float>),
                              0,
                              0,
                              0,
                              1); // e² - e
        expected.setHostValue(
            static_cast<T>(std::sqrt(5.0f) - std::numbers::sqrt3_v<float>), 0, 0, 1, 0); // √5 - √3
        expected.setHostValue(static_cast<T>(2.0f - 1.0f), 0, 0, 1, 1); // log₁₀(100) - log₁₀(10)

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testAddOperation2D()
    {
        Tensor<T> input1({4, 3});
        Tensor<T> input2({4, 3});
        Tensor<T> output({4, 3});

        input1.fillWithValue(static_cast<T>(1.0));
        input2.fillWithValue(static_cast<T>(2.0));

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        Tensor<T> expected({4, 3});
        expected.fillWithValue(static_cast<T>(3.0));

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testAddOperation3D()
    {
        Tensor<T> input1({2, 3, 10});
        Tensor<T> input2({2, 3, 10});
        Tensor<T> output({2, 3, 10});

        input1.fillWithValue(static_cast<T>(2.5));
        input2.fillWithValue(static_cast<T>(1.5));

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        Tensor<T> expected({2, 3, 10});
        expected.fillWithValue(static_cast<T>(4.0));

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testSingleElementTensors()
    {
        Tensor<T> input1({1, 1, 1, 1});
        Tensor<T> input2({1, 1, 1, 1});
        Tensor<T> output({1, 1, 1, 1});

        input1.setHostValue(
            static_cast<T>(std::numbers::e_v<float> * std::numbers::e_v<float>), 0, 0, 0, 0); // e²
        input2.setHostValue(static_cast<T>(std::numbers::e_v<float>), 0, 0, 0, 0); // e

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::SUB, input1, input2, output);

        Tensor<T> expected({1, 1, 1, 1});
        expected.setHostValue(static_cast<T>((std::numbers::e_v<float>
                                              * std::numbers::e_v<float>)-std::numbers::e_v<float>),
                              0,
                              0,
                              0,
                              0); // e² - e

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testNumericalPrecision()
    {
        Tensor<T> input1({1, 1, 1, 1});
        Tensor<T> input2({1, 1, 1, 1});
        Tensor<T> output({1, 1, 1, 1});

        input1.setHostValue(static_cast<T>(0.123456789f), 0, 0, 0, 0);
        input2.setHostValue(static_cast<T>(0.987654321f), 0, 0, 0, 0);

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        Tensor<T> expected({1, 1, 1, 1});
        expected.setHostValue(static_cast<T>(0.123456789f + 0.987654321f), 0, 0, 0, 0);

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBroadcast1Dx1D()
    {
        Tensor<T> input1({5});
        Tensor<T> input2({5});
        Tensor<T> output({5});

        for(int i = 0; i < 5; ++i)
        {
            input1.setHostValue(static_cast<T>(static_cast<float>(i + 1)), i);
            input2.setHostValue(static_cast<T>(static_cast<float>(i * 2)), i);
        }

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        // Create expected tensor: [1,2,3,4,5] + [0,2,4,6,8] = [1,4,7,10,13]
        Tensor<T> expected({5});
        expected.setHostValue(static_cast<T>(static_cast<float>(1)), 0);
        expected.setHostValue(static_cast<T>(static_cast<float>(4)), 1);
        expected.setHostValue(static_cast<T>(static_cast<float>(7)), 2);
        expected.setHostValue(static_cast<T>(static_cast<float>(10)), 3);
        expected.setHostValue(static_cast<T>(static_cast<float>(13)), 4);

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBroadcast2Dx1D()
    {
        Tensor<T> input1({3, 4}); // [M,N] = [3,4]
        Tensor<T> input2({4}); // [N] = [4]
        Tensor<T> output({3, 4}); // Output: [3,4]

        // Fill input1 with pattern: row*10 + col
        for(int m = 0; m < 3; ++m)
        {
            for(int n = 0; n < 4; ++n)
            {
                input1.setHostValue(static_cast<T>(static_cast<float>((m * 10) + n)), m, n);
            }
        }

        // Fill input2 with pattern: [100, 200, 300, 400]
        for(int n = 0; n < 4; ++n)
        {
            input2.setHostValue(static_cast<T>(static_cast<float>((n + 1) * 100)), n);
        }

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        // Create expected tensor: broadcasting input2[n] to all input1[m,n]
        Tensor<T> expected({3, 4});
        for(int m = 0; m < 3; ++m)
        {
            for(int n = 0; n < 4; ++n)
            {
                auto input1Val = static_cast<float>((m * 10) + n);
                auto input2Val = static_cast<float>((n + 1) * 100);
                expected.setHostValue(static_cast<T>(input1Val + input2Val), m, n);
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBroadcast3D()
    {
        Tensor<T> input1({2, 3, 4}); // [2,3,4]
        Tensor<T> input2({1, 3, 1}); // [1,3,1] - broadcasts to [2,3,4]
        Tensor<T> output({2, 3, 4}); // Output: [2,3,4]

        input1.fillWithValue(static_cast<T>(5.0));

        input2.setHostValue(static_cast<T>(1.0), 0, 0, 0); // Channel 0
        input2.setHostValue(static_cast<T>(2.0), 0, 1, 0); // Channel 1
        input2.setHostValue(static_cast<T>(3.0), 0, 2, 0); // Channel 2

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::SUB, input1, input2, output);

        // Create expected tensor: broadcasting subtraction
        Tensor<T> expected({2, 3, 4});
        for(int n = 0; n < 2; ++n)
        {
            for(int c = 0; c < 3; ++c)
            {
                for(int h = 0; h < 4; ++h)
                {
                    float input1Val = 5.0f;
                    auto input2Val = static_cast<float>(c + 1); // Channel values: 1.0, 2.0, 3.0
                    expected.setHostValue(static_cast<T>(input1Val - input2Val), n, c, h);
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    // Test case: 4D × 4D: [N,C,H,W] + [1,C,1,1] → broadcast to [N,C,H,W]
    void testBroadcast4Dx4D()
    {
        Tensor<T> input1({2, 3, 2, 2}); // [N,C,H,W] = [2,3,2,2]
        Tensor<T> input2({1, 3, 1, 1}); // [1,C,1,1] = [1,3,1,1]
        Tensor<T> output({2, 3, 2, 2}); // Output: [2,3,2,2]

        input1.fillWithValue(static_cast<T>(1.0));

        input2.setHostValue(static_cast<T>(10.0), 0, 0, 0, 0); // Channel 0
        input2.setHostValue(static_cast<T>(20.0), 0, 1, 0, 0); // Channel 1
        input2.setHostValue(static_cast<T>(30.0), 0, 2, 0, 0); // Channel 2

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        // Create expected tensor: broadcasting addition
        Tensor<T> expected({2, 3, 2, 2});
        for(int n = 0; n < 2; ++n)
        {
            for(int c = 0; c < 3; ++c)
            {
                for(int h = 0; h < 2; ++h)
                {
                    for(int w = 0; w < 2; ++w)
                    {
                        float input1Val = 1.0f;
                        auto input2Val
                            = static_cast<float>((c + 1) * 10); // Channel values: 10.0, 20.0, 30.0
                        expected.setHostValue(static_cast<T>(input1Val + input2Val), n, c, h, w);
                    }
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    // Test case: Complex N-D broadcasting: [2,1,3,1] + [1,2,1,4] → [2,2,3,4]
    void testBroadcastComplexND()
    {
        Tensor<T> input1({2, 1, 3, 1});
        Tensor<T> input2({1, 2, 1, 4});
        Tensor<T> output({2, 2, 3, 4});

        for(int n = 0; n < 2; ++n)
        {
            for(int h = 0; h < 3; ++h)
            {
                input1.setHostValue(static_cast<T>(static_cast<float>((n * 10) + h)), n, 0, h, 0);
            }
        }

        for(int c = 0; c < 2; ++c)
        {
            for(int w = 0; w < 4; ++w)
            {
                input2.setHostValue(static_cast<T>(static_cast<float>((c * 100) + w)), 0, c, 0, w);
            }
        }

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        // Create expected tensor: broadcasting addition
        Tensor<T> expected({2, 2, 3, 4});
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
                        expected.setHostValue(static_cast<T>(input1Val + input2Val), n, c, h, w);
                    }
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

    void testBroadcast5D()
    {
        std::vector<int64_t> dims1 = {2, 3, 2, 2, 2}; // [N,C,D,H,W] = [2,3,2,2,2]
        std::vector<int64_t> strides1 = {24, 8, 4, 2, 1}; // Row-major strides for [2,3,2,2,2]
        Tensor<T> input1(dims1, strides1);

        std::vector<int64_t> dims2 = {1, 3, 1, 1, 1}; // [1,C,1,1,1] = [1,3,1,1,1]
        std::vector<int64_t> strides2 = {3, 1, 1, 1, 1}; // Row-major strides for [1,3,1,1,1]
        Tensor<T> input2(dims2, strides2);

        std::vector<int64_t> outputDims = {2, 3, 2, 2, 2}; // Output: [2,3,2,2,2]
        std::vector<int64_t> outputStrides = {24, 8, 4, 2, 1}; // Row-major strides for [2,3,2,2,2]
        Tensor<T> output(outputDims, outputStrides);

        input1.fillWithValue(static_cast<T>(2.0));

        // Set channel-specific values in input2
        input2.setHostValue(static_cast<T>(10.0), 0, 0, 0, 0, 0); // Channel 0
        input2.setHostValue(static_cast<T>(20.0), 0, 1, 0, 0, 0); // Channel 1
        input2.setHostValue(static_cast<T>(30.0), 0, 2, 0, 0, 0); // Channel 2

        CpuReferencePointwiseImpl<T, T, T>::pointwiseForward(PointwiseMode::ADD, input1, input2, output);

        // Create expected tensor: broadcasting addition
        Tensor<T> expected(outputDims, outputStrides);
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
                            auto input1Val = 2.0f;
                            auto input2Val = static_cast<float>(
                                (c + 1) * 10); // Channel values: 10.0, 20.0, 30.0
                            expected.setHostValue(
                                static_cast<T>(input1Val + input2Val), n, c, d, h, w);
                        }
                    }
                }
            }
        }

        auto tolerance = getTolerance();
        CpuFpReferenceValidation<T> validator(tolerance, tolerance);
        EXPECT_TRUE(validator.allClose(expected.memory(), output.memory()));
    }

};

using TestTypes = ::testing::Types<float, double, half, hip_bfloat16>;
TYPED_TEST_SUITE(CpuReferencePointwiseTemplate, TestTypes);

TYPED_TEST(CpuReferencePointwiseTemplate, AddOperation)
{
    this->testAddOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, SubtractOperation)
{
    this->testSubtractOperation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, AddOperationSanityValidation)
{
    this->testAddOperationSanityValidation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, SubtractOperationSanityValidation)
{
    this->testSubtractOperationSanityValidation();
}

TYPED_TEST(CpuReferencePointwiseTemplate, AddOperation2D)
{
    this->testAddOperation2D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, AddOperation3D)
{
    this->testAddOperation3D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, SingleElementTensors)
{
    this->testSingleElementTensors();
}

TYPED_TEST(CpuReferencePointwiseTemplate, NumericalPrecision)
{
    this->testNumericalPrecision();
}

TYPED_TEST(CpuReferencePointwiseTemplate, Broadcast1Dx1D)
{
    this->testBroadcast1Dx1D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, Broadcast2Dx1D)
{
    this->testBroadcast2Dx1D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, Broadcast3D)
{
    this->testBroadcast3D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, Broadcast4Dx4D)
{
    this->testBroadcast4Dx4D();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BroadcastComplexND)
{
    this->testBroadcastComplexND();
}

TYPED_TEST(CpuReferencePointwiseTemplate, Broadcast5D)
{
    this->testBroadcast5D();
}
