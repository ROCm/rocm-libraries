// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

template <typename T>
class CpuReferencePointwiseTemplate : public ::testing::Test
{
protected:
    T getTolerance() const
    {
        if constexpr(std::is_same_v<T, half> || std::is_same_v<T, hip_bfloat16>)
        {
            return static_cast<T>(1e-2f);
        }
        else
        {
            return static_cast<T>(1e-4f);
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

        T expected = static_cast<T>(3.0);
        EXPECT_EQ(output.getHostValue(0, 0, 0, 0), expected);
        EXPECT_EQ(output.getHostValue(0, 1, 1, 1), expected);
        EXPECT_EQ(output.getHostValue(0, 2, 0, 1), expected);
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

        T expected = static_cast<T>(3.0);
        EXPECT_EQ(output.getHostValue(0, 0, 0, 0), expected);
        EXPECT_EQ(output.getHostValue(0, 1, 1, 1), expected);
        EXPECT_EQ(output.getHostValue(0, 2, 0, 1), expected);
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

    void testAddOperationSanityValidation()
    {
        Tensor<T> input1({1, 1, 2, 2});
        Tensor<T> input2({1, 1, 2, 2});
        Tensor<T> output({1, 1, 2, 2});

        // Set known values: [1, 2, 3, 4] + [0.1, 0.2, 0.3, 0.4]
        input1.setHostValue(static_cast<T>(1.0f), 0, 0, 0, 0);
        input1.setHostValue(static_cast<T>(2.0f), 0, 0, 0, 1);
        input1.setHostValue(static_cast<T>(3.0f), 0, 0, 1, 0);
        input1.setHostValue(static_cast<T>(4.0f), 0, 0, 1, 1);

        input2.setHostValue(static_cast<T>(0.1f), 0, 0, 0, 0);
        input2.setHostValue(static_cast<T>(0.2f), 0, 0, 0, 1);
        input2.setHostValue(static_cast<T>(0.3f), 0, 0, 1, 0);
        input2.setHostValue(static_cast<T>(0.4f), 0, 0, 1, 1);

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};
        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

        // Expected: [1.1, 2.2, 3.3, 4.4]
        auto tolerance = getTolerance();
        EXPECT_NEAR(static_cast<float>(output.getHostValue(0, 0, 0, 0)),
                    1.1f,
                    static_cast<float>(tolerance));
        EXPECT_NEAR(static_cast<float>(output.getHostValue(0, 0, 0, 1)),
                    2.2f,
                    static_cast<float>(tolerance));
        EXPECT_NEAR(static_cast<float>(output.getHostValue(0, 0, 1, 0)),
                    3.3f,
                    static_cast<float>(tolerance));
        EXPECT_NEAR(static_cast<float>(output.getHostValue(0, 0, 1, 1)),
                    4.4f,
                    static_cast<float>(tolerance));
    }

    void testSubtractOperationSanityValidation()
    {
        Tensor<T> input1({1, 1, 2, 2});
        Tensor<T> input2({1, 1, 2, 2});
        Tensor<T> output({1, 1, 2, 2});

        // Set known values: [5, 4, 3, 2] - [1, 1, 1, 1]
        input1.setHostValue(static_cast<T>(5.0), 0, 0, 0, 0);
        input1.setHostValue(static_cast<T>(4.0), 0, 0, 0, 1);
        input1.setHostValue(static_cast<T>(3.0), 0, 0, 1, 0);
        input1.setHostValue(static_cast<T>(2.0), 0, 0, 1, 1);

        input2.setHostValue(static_cast<T>(1.0), 0, 0, 0, 0);
        input2.setHostValue(static_cast<T>(1.0), 0, 0, 0, 1);
        input2.setHostValue(static_cast<T>(1.0), 0, 0, 1, 0);
        input2.setHostValue(static_cast<T>(1.0), 0, 0, 1, 1);

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};
        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::SUB);

        // Expected: [4, 3, 2, 1]
        EXPECT_EQ(output.getHostValue(0, 0, 0, 0), static_cast<T>(4.0));
        EXPECT_EQ(output.getHostValue(0, 0, 0, 1), static_cast<T>(3.0));
        EXPECT_EQ(output.getHostValue(0, 0, 1, 0), static_cast<T>(2.0));
        EXPECT_EQ(output.getHostValue(0, 0, 1, 1), static_cast<T>(1.0));
    }

    void testAddOperation2D()
    {
        Tensor<T> input1({4, 3});
        Tensor<T> input2({4, 3});
        Tensor<T> output({4, 3});

        input1.fillWithValue(static_cast<T>(1.0));
        input2.fillWithValue(static_cast<T>(2.0));

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        EXPECT_THROW(
            CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD),
            std::runtime_error);
    }

    void testAddOperation3D()
    {
        Tensor<T> input1({2, 3, 10});
        Tensor<T> input2({2, 3, 10});
        Tensor<T> output({2, 3, 10});

        input1.fillWithValue(static_cast<T>(2.5));
        input2.fillWithValue(static_cast<T>(1.5));

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};

        EXPECT_THROW(
            CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD),
            std::runtime_error);
    }

    void testSingleElementTensors()
    {
        Tensor<T> input1({1, 1, 1, 1});
        Tensor<T> input2({1, 1, 1, 1});
        Tensor<T> output({1, 1, 1, 1});

        input1.setHostValue(static_cast<T>(5.0), 0, 0, 0, 0);
        input2.setHostValue(static_cast<T>(3.0), 0, 0, 0, 0);

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};
        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::SUB);

        EXPECT_EQ(output.getHostValue(0, 0, 0, 0), static_cast<T>(2.0));
    }

    void testNumericalPrecision()
    {
        Tensor<T> input1({1, 1, 1, 1});
        Tensor<T> input2({1, 1, 1, 1});
        Tensor<T> output({1, 1, 1, 1});

        // Test with values that should be exactly representable in all types
        input1.setHostValue(static_cast<T>(0.125), 0, 0, 0, 0); // 1/8 - exactly representable
        input2.setHostValue(static_cast<T>(0.25), 0, 0, 0, 0); // 1/4 - exactly representable

        std::vector<const TensorBase<T>*> inputs = {&input1, &input2};
        CpuReferencePointwiseImpl<T>::pointwiseForward(inputs, output, PointwiseMode::ADD);

        T expected = static_cast<T>(0.375); // 3/8 - exactly representable
        EXPECT_EQ(output.getHostValue(0, 0, 0, 0), expected);
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
