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
