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
            return static_cast<T>(1e-3f); // Half has better precision than bfloat16
        }
        else if constexpr(std::is_same_v<T, hip_bfloat16>)
        {
            return static_cast<T>(1e-2f); // Bfloat16 has lower precision due to fewer mantissa bits
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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::SUB);

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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};
        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};
        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::SUB);

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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};
        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::SUB);

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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};
        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

        // Verify results: [1,2,3,4,5] + [0,2,4,6,8] = [1,4,7,10,13]
        EXPECT_EQ(output.getHostValue(0), static_cast<T>(static_cast<float>(1)));
        EXPECT_EQ(output.getHostValue(1), static_cast<T>(static_cast<float>(4)));
        EXPECT_EQ(output.getHostValue(2), static_cast<T>(static_cast<float>(7)));
        EXPECT_EQ(output.getHostValue(3), static_cast<T>(static_cast<float>(10)));
        EXPECT_EQ(output.getHostValue(4), static_cast<T>(static_cast<float>(13)));
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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

        // Verify broadcasting: input2[n] is added to all input1[m,n]
        EXPECT_EQ(output.getHostValue(0, 0),
                  static_cast<T>(static_cast<float>(0 + 100))); // 0 + 100 = 100
        EXPECT_EQ(output.getHostValue(0, 1),
                  static_cast<T>(static_cast<float>(1 + 200))); // 1 + 200 = 201
        EXPECT_EQ(output.getHostValue(1, 2),
                  static_cast<T>(static_cast<float>(12 + 300))); // 12 + 300 = 312
        EXPECT_EQ(output.getHostValue(2, 3),
                  static_cast<T>(static_cast<float>(23 + 400))); // 23 + 400 = 423
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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::SUB);

        EXPECT_EQ(output.getHostValue(0, 0, 0), static_cast<T>(4.0)); // 5.0 - 1.0
        EXPECT_EQ(output.getHostValue(1, 1, 3), static_cast<T>(3.0)); // 5.0 - 2.0
        EXPECT_EQ(output.getHostValue(0, 2, 2), static_cast<T>(2.0)); // 5.0 - 3.0
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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

        // Channel 0: all should be 1.0 + 10.0 = 11.0
        EXPECT_EQ(output.getHostValue(0, 0, 0, 0), static_cast<T>(11.0));
        EXPECT_EQ(output.getHostValue(0, 0, 1, 1), static_cast<T>(11.0));
        EXPECT_EQ(output.getHostValue(1, 0, 0, 1), static_cast<T>(11.0));

        // Channel 1: all should be 1.0 + 20.0 = 21.0
        EXPECT_EQ(output.getHostValue(0, 1, 0, 0), static_cast<T>(21.0));
        EXPECT_EQ(output.getHostValue(1, 1, 1, 1), static_cast<T>(21.0));

        // Channel 2: all should be 1.0 + 30.0 = 31.0
        EXPECT_EQ(output.getHostValue(0, 2, 0, 0), static_cast<T>(31.0));
        EXPECT_EQ(output.getHostValue(1, 2, 1, 0), static_cast<T>(31.0));
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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

        // output[0,0,0,0] = input1[0,0,0,0] + input2[0,0,0,0] = 0 + 0 = 0
        EXPECT_EQ(output.getHostValue(0, 0, 0, 0), static_cast<T>(static_cast<float>(0)));

        // output[1,1,2,3] = input1[1,0,2,0] + input2[0,1,0,3] = 12 + 103 = 115
        EXPECT_EQ(output.getHostValue(1, 1, 2, 3), static_cast<T>(static_cast<float>(115)));

        // output[0,1,1,2] = input1[0,0,1,0] + input2[0,1,0,2] = 1 + 102 = 103
        EXPECT_EQ(output.getHostValue(0, 1, 1, 2), static_cast<T>(static_cast<float>(103)));
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

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

        // Channel 0: all should be 2.0 + 10.0 = 12.0
        EXPECT_EQ(output.getHostValue(0, 0, 0, 0, 0), static_cast<T>(12.0));
        EXPECT_EQ(output.getHostValue(0, 0, 1, 1, 1), static_cast<T>(12.0));
        EXPECT_EQ(output.getHostValue(1, 0, 0, 1, 0), static_cast<T>(12.0));

        // Channel 1: all should be 2.0 + 20.0 = 22.0
        EXPECT_EQ(output.getHostValue(0, 1, 0, 0, 0), static_cast<T>(22.0));
        EXPECT_EQ(output.getHostValue(1, 1, 1, 1, 1), static_cast<T>(22.0));
        EXPECT_EQ(output.getHostValue(0, 1, 1, 0, 1), static_cast<T>(22.0));

        // Channel 2: all should be 2.0 + 30.0 = 32.0
        EXPECT_EQ(output.getHostValue(0, 2, 0, 0, 0), static_cast<T>(32.0));
        EXPECT_EQ(output.getHostValue(1, 2, 1, 0, 1), static_cast<T>(32.0));
        EXPECT_EQ(output.getHostValue(1, 2, 0, 1, 1), static_cast<T>(32.0));
    }

    void testErrorHandlingWrongInputCount()
    {
        Tensor<T> input1({1, 3, 2, 2});
        Tensor<T> output({1, 3, 2, 2});

        std::vector<const TensorBase<T>*> inputs = {&input1};
        EXPECT_THROW(
            CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD),
            std::runtime_error);
    }

    void testErrorHandlingNullInput()
    {
        Tensor<T> input2({1, 3, 2, 2});
        Tensor<T> output({1, 3, 2, 2});

        std::vector<const TensorBase<T>*> inputs = {nullptr, &input2};

        EXPECT_THROW(
            CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD),
            std::runtime_error);
    }

    void testErrorHandlingDimensionMismatch()
    {
        Tensor<T> input1({1, 3, 2, 2});
        Tensor<T> input2({1, 2, 3, 2});
        Tensor<T> output({1, 3, 2, 2});

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        EXPECT_THROW(
            CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD),
            std::runtime_error);
    }

    void testErrorHandlingEmptyInputs()
    {
        Tensor<T> output({1, 3, 2, 2});
        std::vector<const TensorBase<T>*> inputs = {};

        EXPECT_THROW(
            CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD),
            std::runtime_error);
    }

    void testErrorHandlingTooManyInputs()
    {
        Tensor<T> input1({1, 3, 2, 2});
        Tensor<T> input2({1, 3, 2, 2});
        Tensor<T> input3({1, 3, 2, 2});
        Tensor<T> output({1, 3, 2, 2});

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2, &input3};

        EXPECT_THROW(
            CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD),
            std::runtime_error);
    }

    void testBroadcastErrorIncompatibleShapes()
    {
        Tensor<T> input1({2, 3, 2, 2});
        Tensor<T> input2({2, 2, 2, 2});
        Tensor<T> output({2, 3, 2, 2});

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        EXPECT_THROW(
            CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD),
            std::runtime_error);
    }

    void testBroadcastErrorWrongOutputShape()
    {
        Tensor<T> input1({2, 2, 2, 2});
        Tensor<T> input2({1, 2, 1, 1});
        Tensor<T> output({2, 3, 2, 2});
        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        EXPECT_THROW(
            CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD),
            std::runtime_error);
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

TYPED_TEST(CpuReferencePointwiseTemplate, ErrorHandlingWrongInputCount)
{
    this->testErrorHandlingWrongInputCount();
}

TYPED_TEST(CpuReferencePointwiseTemplate, ErrorHandlingNullInput)
{
    this->testErrorHandlingNullInput();
}

TYPED_TEST(CpuReferencePointwiseTemplate, ErrorHandlingDimensionMismatch)
{
    this->testErrorHandlingDimensionMismatch();
}

TYPED_TEST(CpuReferencePointwiseTemplate, ErrorHandlingEmptyInputs)
{
    this->testErrorHandlingEmptyInputs();
}

TYPED_TEST(CpuReferencePointwiseTemplate, ErrorHandlingTooManyInputs)
{
    this->testErrorHandlingTooManyInputs();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BroadcastErrorIncompatibleShapes)
{
    this->testBroadcastErrorIncompatibleShapes();
}

TYPED_TEST(CpuReferencePointwiseTemplate, BroadcastErrorWrongOutputShape)
{
    this->testBroadcastErrorWrongOutputShape();
}
