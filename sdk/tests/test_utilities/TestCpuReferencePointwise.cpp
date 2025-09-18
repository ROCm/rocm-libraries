// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

TEST(TestCpuReferencePointwiseFp32, AddOperation)
{
    Tensor<float> input1({1, 3, 2, 2});
    Tensor<float> input2({1, 3, 2, 2});
    Tensor<float> output({1, 3, 2, 2});

    // Fill input1 with 1.0, input2 with 2.0, expect output = 3.0
    input1.fillWithValue(1.0f);
    input2.fillWithValue(2.0f);

    std::vector<const TensorBase<float>*> inputs = {&input1, &input2};

    CpuReferencePointwiseImpl<float>::pointwiseForward(inputs, output, PointwiseMode::ADD);

    // Verify all elements are 3.0 (1.0 + 2.0)
    EXPECT_FLOAT_EQ(output.getHostValue(0, 0, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 1, 1, 1), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 2, 0, 1), 3.0f);
}

TEST(TestCpuReferencePointwiseFp32, SubtractOperation)
{
    Tensor<float> input1({1, 3, 2, 2});
    Tensor<float> input2({1, 3, 2, 2});
    Tensor<float> output({1, 3, 2, 2});

    // Fill input1 with 5.0, input2 with 2.0, expect output = 3.0
    input1.fillWithValue(5.0f);
    input2.fillWithValue(2.0f);

    std::vector<const TensorBase<float>*> inputs = {&input1, &input2};

    CpuReferencePointwiseImpl<float>::pointwiseForward(inputs, output, PointwiseMode::SUB);

    // Verify all elements are 3.0 (5.0 - 2.0)
    EXPECT_FLOAT_EQ(output.getHostValue(0, 0, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 1, 1, 1), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 2, 0, 1), 3.0f);
}

TEST(TestCpuReferencePointwiseFp32, MultiplyOperation)
{
    Tensor<float> input1({1, 3, 2, 2});
    Tensor<float> input2({1, 3, 2, 2});
    Tensor<float> output({1, 3, 2, 2});

    // Fill input1 with 1.5, input2 with 2.0, expect output = 3.0
    input1.fillWithValue(1.5f);
    input2.fillWithValue(2.0f);

    std::vector<const TensorBase<float>*> inputs = {&input1, &input2};

    CpuReferencePointwiseImpl<float>::pointwiseForward(inputs, output, PointwiseMode::MUL);

    // Verify all elements are 3.0 (1.5 * 2.0)
    EXPECT_FLOAT_EQ(output.getHostValue(0, 0, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 1, 1, 1), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 2, 0, 1), 3.0f);
}

TEST(TestCpuReferencePointwiseFp32, IdentityOperation)
{
    Tensor<float> input1({1, 3, 2, 2});
    Tensor<float> output({1, 3, 2, 2});

    // Fill input1 with 3.0, expect output = 3.0
    input1.fillWithValue(3.0f);

    std::vector<const TensorBase<float>*> inputs = {&input1};

    CpuReferencePointwiseImpl<float>::pointwiseForward(inputs, output, PointwiseMode::IDENTITY);

    // Verify all elements are 3.0 (identity of 3.0)
    EXPECT_FLOAT_EQ(output.getHostValue(0, 0, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 1, 1, 1), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 2, 0, 1), 3.0f);
}

TEST(TestCpuReferencePointwiseFp32, NegateOperation)
{
    Tensor<float> input1({1, 3, 2, 2});
    Tensor<float> output({1, 3, 2, 2});

    // Fill input1 with -3.0, expect output = 3.0
    input1.fillWithValue(-3.0f);

    std::vector<const TensorBase<float>*> inputs = {&input1};

    CpuReferencePointwiseImpl<float>::pointwiseForward(inputs, output, PointwiseMode::NEG);

    // Verify all elements are 3.0 (negate of -3.0)
    EXPECT_FLOAT_EQ(output.getHostValue(0, 0, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 1, 1, 1), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 2, 0, 1), 3.0f);
}

TEST(TestCpuReferencePointwiseFp32, AbsOperation)
{
    Tensor<float> input1({1, 3, 2, 2});
    Tensor<float> output({1, 3, 2, 2});

    // Fill input1 with -3.0, expect output = 3.0
    input1.fillWithValue(-3.0f);

    std::vector<const TensorBase<float>*> inputs = {&input1};

    CpuReferencePointwiseImpl<float>::pointwiseForward(inputs, output, PointwiseMode::ABS);

    // Verify all elements are 3.0 (abs of -3.0)
    EXPECT_FLOAT_EQ(output.getHostValue(0, 0, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 1, 1, 1), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 2, 0, 1), 3.0f);
}

TEST(TestCpuReferencePointwiseFp32, BinarySelectOperation)
{
    Tensor<float> condition({1, 3, 2, 2});
    Tensor<float> input1({1, 3, 2, 2});
    Tensor<float> input2({1, 3, 2, 2});
    Tensor<float> output({1, 3, 2, 2});

    // Fill condition with 1.0 (true), input1 with 3.0, input2 with 5.0
    // Expect output = 3.0 (select input1 when condition is true)
    condition.fillWithValue(1.0f);
    input1.fillWithValue(3.0f);
    input2.fillWithValue(5.0f);

    std::vector<const TensorBase<float>*> inputs = {&condition, &input1, &input2};

    CpuReferencePointwiseImpl<float>::pointwiseForward(
        inputs, output, PointwiseMode::BINARY_SELECT);

    // Verify all elements are 3.0 (selected input1)
    EXPECT_FLOAT_EQ(output.getHostValue(0, 0, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 1, 1, 1), 3.0f);
    EXPECT_FLOAT_EQ(output.getHostValue(0, 2, 0, 1), 3.0f);
}

TEST(TestCpuReferencePointwiseFp32, ErrorHandling_WrongInputCount)
{
    Tensor<float> input1({1, 3, 2, 2});
    Tensor<float> output({1, 3, 2, 2});

    // Test binary operation with wrong number of inputs
    std::vector<const TensorBase<float>*> inputs = {&input1}; // Only 1 input for binary op

    EXPECT_THROW(
        CpuReferencePointwiseImpl<float>::pointwiseForward(inputs, output, PointwiseMode::ADD),
        std::runtime_error);
}

TEST(TestCpuReferencePointwiseFp32, ErrorHandling_NullInput)
{
    Tensor<float> input2({1, 3, 2, 2});
    Tensor<float> output({1, 3, 2, 2});

    // Test with null input
    std::vector<const TensorBase<float>*> inputs = {nullptr, &input2};

    EXPECT_THROW(
        CpuReferencePointwiseImpl<float>::pointwiseForward(inputs, output, PointwiseMode::ADD),
        std::runtime_error);
}

TEST(TestCpuReferencePointwiseFp32, ErrorHandling_DimensionMismatch)
{
    Tensor<float> input1({1, 3, 2, 2});
    Tensor<float> input2({1, 2, 3, 2}); // Different dimensions
    Tensor<float> output({1, 3, 2, 2});

    std::vector<const TensorBase<float>*> inputs = {&input1, &input2};

    EXPECT_THROW(
        CpuReferencePointwiseImpl<float>::pointwiseForward(inputs, output, PointwiseMode::ADD),
        std::runtime_error);
}

TEST(TestCpuReferencePointwiseFp64, AddOperation)
{
    Tensor<double> input1({1, 3, 2, 2});
    Tensor<double> input2({1, 3, 2, 2});
    Tensor<double> output({1, 3, 2, 2});

    // Fill input1 with 1.0, input2 with 2.0, expect output = 3.0
    input1.fillWithValue(1.0);
    input2.fillWithValue(2.0);

    std::vector<const TensorBase<double>*> inputs = {&input1, &input2};

    CpuReferencePointwiseImpl<double>::pointwiseForward(inputs, output, PointwiseMode::ADD);

    // Verify all elements are 3.0 (1.0 + 2.0)
    EXPECT_DOUBLE_EQ(output.getHostValue(0, 0, 0, 0), 3.0);
    EXPECT_DOUBLE_EQ(output.getHostValue(0, 1, 1, 1), 3.0);
    EXPECT_DOUBLE_EQ(output.getHostValue(0, 2, 0, 1), 3.0);
}
