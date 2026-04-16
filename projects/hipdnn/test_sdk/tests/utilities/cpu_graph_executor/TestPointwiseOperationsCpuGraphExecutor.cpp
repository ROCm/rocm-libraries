// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "PointwiseGraphUtils.hpp"
#include "PointwiseTensorBundles.hpp"

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

struct ReluPointwiseTestParams
{
    hipdnn_frontend::PointwiseMode mode;
    DataType inputDataType = DataType::FLOAT;
    DataType accumulatorDataType = DataType::FLOAT;
    float in0TensorValue = 0.0f;
    float in1TensorValue = 0.0f; // For binary operations
    std::vector<int64_t> inputDims = {1, 3, 4, 4};
    std::vector<int64_t> outputDims = {1, 3, 4, 4};

    // Optional ReLU parameters
    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;
    std::optional<float> reluLowerClipSlope;

    // Test metadata
    std::string testDescription;
};

class PointwiseReluTestHelper
{
public:
    template <typename T>
    static void runReluFwdTest(const ReluPointwiseTestParams& params)
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseUnaryGraph(params.inputDims,
                                       params.outputDims,
                                       params.inputDataType,
                                       params.accumulatorDataType,
                                       params.inputDataType,
                                       params.mode,
                                       1,
                                       TensorLayout::NCHW,
                                       params.reluLowerClip,
                                       params.reluUpperClip,
                                       params.reluLowerClipSlope);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        auto graphWrap = GraphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& nodeWrap = graphWrap.getNodeWrapper(0);
        const auto& attributes = nodeWrap.attributesAs<PointwiseAttributes>();

        tensorBundle.tensors[attributes.in_0_tensor_uid()]->fillTensorWithValue(
            params.in0TensorValue);

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        const auto& outTensor = tensorBundle.tensors.at(attributes.out_0_tensor_uid());

        EXPECT_EQ(params.mode, hipdnn_frontend::PointwiseMode::RELU_FWD);
        EXPECT_TRUE(PointwiseReluTestHelper::verifyReluForwardOutput<T>(*outTensor, params));
    }

    template <typename T>
    static void runReluBwdTest(const ReluPointwiseTestParams& params)
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseBinaryGraph(params.inputDims,
                                        params.inputDims,
                                        params.outputDims,
                                        params.inputDataType,
                                        params.inputDataType,
                                        params.accumulatorDataType,
                                        params.inputDataType,
                                        params.mode,
                                        1,
                                        TensorLayout::NCHW,
                                        params.reluLowerClip,
                                        params.reluUpperClip,
                                        params.reluLowerClipSlope);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        auto graphWrap = GraphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& nodeWrap = graphWrap.getNodeWrapper(0);
        const auto& attributes = nodeWrap.attributesAs<PointwiseAttributes>();

        tensorBundle.tensors[attributes.in_0_tensor_uid()]->fillTensorWithValue(
            params.in0TensorValue);
        tensorBundle.tensors[attributes.in_1_tensor_uid().value()]->fillTensorWithValue(
            params.in1TensorValue);

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        const auto& outTensor = tensorBundle.tensors.at(attributes.out_0_tensor_uid());

        EXPECT_EQ(params.mode, hipdnn_frontend::PointwiseMode::RELU_BWD);
        EXPECT_TRUE(PointwiseReluTestHelper::verifyReluBackwardOutput<T>(*outTensor, params));
    }

    template <typename InputType>
    static bool verifyReluForwardOutput(ITensor& outTensor, const ReluPointwiseTestParams& params)
    {
        Tensor<InputType> input0(params.inputDims);
        input0.fillTensorWithValue(params.in0TensorValue);
        Tensor<InputType> output(params.outputDims);

        if(params.reluLowerClip.has_value() || params.reluUpperClip.has_value()
           || params.reluLowerClipSlope.has_value())
        {
            CpuReferencePointwiseImpl<InputType, InputType, InputType>::pointwiseCompute(
                PointwiseMode::RELU_FWD,
                output,
                input0,
                params.reluLowerClip.has_value() ? params.reluLowerClip.value() : 0.0f,
                params.reluUpperClip.has_value() ? params.reluUpperClip.value()
                                                 : std::numeric_limits<float>::max(),
                params.reluLowerClipSlope.has_value() ? params.reluLowerClipSlope.value() : 0.0f);
        }
        else
        {
            CpuReferencePointwiseImpl<InputType, InputType, InputType>::pointwiseCompute(
                PointwiseMode::RELU_FWD, output, input0);
        }

        const CpuFpReferenceValidation<InputType> validator;
        return validator.allClose(output, outTensor);
    }

    template <typename InputType>
    static bool verifyReluBackwardOutput(ITensor& outTensor, const ReluPointwiseTestParams& params)
    {
        Tensor<InputType> input0(params.inputDims);
        Tensor<InputType> input1(params.inputDims);
        input0.fillTensorWithValue(params.in0TensorValue);
        input1.fillTensorWithValue(params.in1TensorValue);

        Tensor<InputType> output(params.outputDims);

        if(params.reluLowerClip.has_value() || params.reluUpperClip.has_value()
           || params.reluLowerClipSlope.has_value())
        {
            CpuReferencePointwiseImpl<InputType, InputType, InputType>::pointwiseCompute(
                PointwiseMode::RELU_BWD,
                output,
                input0,
                input1,
                params.reluLowerClip.has_value() ? params.reluLowerClip.value() : 0.0f,
                params.reluUpperClip.has_value() ? params.reluUpperClip.value()
                                                 : std::numeric_limits<float>::max(),
                params.reluLowerClipSlope.has_value() ? params.reluLowerClipSlope.value() : 0.0f);
        }
        else
        {
            CpuReferencePointwiseImpl<InputType, InputType, InputType>::pointwiseCompute(
                PointwiseMode::RELU_BWD, output, input0, input1);
        }

        const CpuFpReferenceValidation<InputType> validator;
        return validator.allClose(output, outTensor);
    }
};

template <typename T>
class ReluPointwiseOperationsCpuGraphExecutor : public ::testing::Test
{
protected:
    using DataTypeT = T;
};

using TestTypes = ::testing::Types<float, half, bfloat16>;
TYPED_TEST_SUITE(ReluPointwiseOperationsCpuGraphExecutor, TestTypes, );

// =============================================================================
// ReLU Forward Tests
// =============================================================================

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluForwardStandardNegativeInput)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -10.0f;
    params.testDescription = "Standard ReLU with negative input should clamp to zero";

    PointwiseReluTestHelper::runReluFwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluForwardStandardPositiveInput)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 10.0f;
    params.testDescription = "Standard ReLU with positive input should pass through";

    PointwiseReluTestHelper::runReluFwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluForwardClampedBelowLowerBound)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 0.0f;
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU with input below lower bound";

    PointwiseReluTestHelper::runReluFwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluForwardClampedAboveUpperBound)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 2.0f;
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU with input above upper bound";

    PointwiseReluTestHelper::runReluFwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluForwardClampedWithinBounds)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 0.2f;
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU with input within bounds";

    PointwiseReluTestHelper::runReluFwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluForwardLeakyNegativeInput)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -0.5f;
    params.reluLowerClipSlope = 0.1f;
    params.testDescription = "Leaky ReLU with negative input";

    PointwiseReluTestHelper::runReluFwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluForwardLeakyPositiveInput)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 0.5f;
    params.reluLowerClipSlope = 0.1f;
    params.testDescription = "Leaky ReLU with positive input";

    PointwiseReluTestHelper::runReluFwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluForwardUpperBoundOnlyBelowBound)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Upper-bounded ReLU with input below bound";

    PointwiseReluTestHelper::runReluFwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluForwardUpperBoundOnlyAboveBound)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 0.9f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Upper-bounded ReLU with input above bound";

    PointwiseReluTestHelper::runReluFwdTest<TypeParam>(params);
}

// =============================================================================
// ReLU Backward Tests
// =============================================================================
TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardStandardNegativeX)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -10.0f; // X
    params.in1TensorValue = 5.0f; // Dy
    params.testDescription = "Standard ReLU backward with negative X";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardStandardPositiveX)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 10.0f; // X
    params.in1TensorValue = 5.0f; // Dy
    params.testDescription = "Standard ReLU backward with positive X";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardClampedBelowBounds)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -1.0f; // X
    params.in1TensorValue = 5.0f; // Dy
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU backward with X below bounds";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardClampedAboveBounds)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f; // X
    params.in1TensorValue = 2.0f; // Dy
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU backward with X above bounds";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardClampedWithinBounds)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 0.2f; // X
    params.in1TensorValue = 1.0f; // Dy
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU backward with X within bounds";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardLeakyNegativeX)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -1.0f; // X
    params.in1TensorValue = 5.0f; // Dy
    params.reluLowerClipSlope = 0.1f;
    params.testDescription = "Leaky ReLU backward with negative X";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardLeakyPositiveX)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f; // X
    params.in1TensorValue = 2.0f; // Dy
    params.reluLowerClipSlope = 0.1f;
    params.testDescription = "Leaky ReLU backward with positive X";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardUpperBoundOnlyBelowBound)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f; // X
    params.in1TensorValue = 5.0f; // Dy
    params.reluUpperClip = 10.0f;
    params.testDescription = "Upper-bounded ReLU backward with X below bound";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardUpperBoundOnlyAboveBound)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 20.0f; // X
    params.in1TensorValue = 2.0f; // Dy
    params.reluUpperClip = 10.0f;
    params.testDescription = "Upper-bounded ReLU backward with X above bound";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}
