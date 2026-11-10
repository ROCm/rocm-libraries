// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "PointwiseGraphUtils.hpp"
#include "PointwiseTensorBundles.hpp"

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
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
    float in0TensorValue = 0.0f; // RELU_FWD: x; RELU_BWD: dy.
    float in1TensorValue = 0.0f; // RELU_BWD: x.
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
        Tensor<InputType> upstreamGrad(params.inputDims);
        Tensor<InputType> input(params.inputDims);
        upstreamGrad.fillTensorWithValue(params.in0TensorValue);
        input.fillTensorWithValue(params.in1TensorValue);

        Tensor<InputType> output(params.outputDims);

        if(params.reluLowerClip.has_value() || params.reluUpperClip.has_value()
           || params.reluLowerClipSlope.has_value())
        {
            CpuReferencePointwiseImpl<InputType, InputType, InputType>::pointwiseCompute(
                PointwiseMode::RELU_BWD,
                output,
                upstreamGrad,
                input,
                params.reluLowerClip.has_value() ? params.reluLowerClip.value() : 0.0f,
                params.reluUpperClip.has_value() ? params.reluUpperClip.value()
                                                 : std::numeric_limits<float>::max(),
                params.reluLowerClipSlope.has_value() ? params.reluLowerClipSlope.value() : 0.0f);
        }
        else
        {
            CpuReferencePointwiseImpl<InputType, InputType, InputType>::pointwiseCompute(
                PointwiseMode::RELU_BWD, output, upstreamGrad, input);
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
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = -10.0f; // X
    params.testDescription = "Standard ReLU backward with negative X";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardStandardPositiveX)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = 10.0f; // X
    params.testDescription = "Standard ReLU backward with positive X";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TYPED_TEST(ReluPointwiseOperationsCpuGraphExecutor, ReluBackwardClampedBelowBounds)
{
    ReluPointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = -1.0f; // X
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
    params.in0TensorValue = 2.0f; // Dy
    params.in1TensorValue = 1.0f; // X
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
    params.in0TensorValue = 1.0f; // Dy
    params.in1TensorValue = 0.2f; // X
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
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = -1.0f; // X
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
    params.in0TensorValue = 2.0f; // Dy
    params.in1TensorValue = 1.0f; // X
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
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = 1.0f; // X
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
    params.in0TensorValue = 2.0f; // Dy
    params.in1TensorValue = 20.0f; // X
    params.reluUpperClip = 10.0f;
    params.testDescription = "Upper-bounded ReLU backward with X above bound";

    PointwiseReluTestHelper::runReluBwdTest<TypeParam>(params);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, PreluForwardBackwardComposedGraph)
{
    const std::vector<int64_t> tensorDims = {1, 2, 2, 2};
    const std::vector<int64_t> alphaDims = {1, 2, 1, 1};
    const std::vector<int64_t> scalarDims = {1, 1, 1, 1};

    Tensor<float> dy(tensorDims);
    Tensor<float> x(tensorDims);
    Tensor<float> alpha(alphaDims);
    Tensor<float> zero(scalarDims);
    Tensor<float> y(tensorDims);
    Tensor<float> dx(tensorDims);
    Tensor<float> dalpha(alphaDims);

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("PreluBackwardComposedGraph");
    graph->set_io_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

    int64_t uid = 1;
    const auto makeInput = [&](const std::string& name, const std::vector<int64_t>& dims) {
        auto attributes = hipdnn_frontend::graph::makeTensorAttributes(
            name,
            hipdnn_frontend::DataType::FLOAT,
            dims,
            generateStrides(dims, TensorLayout::NCHW.strideOrder));
        attributes.set_uid(uid++);
        return std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(attributes));
    };
    const auto configureOutput
        = [&](const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& tensor,
              hipdnn_frontend::DataType dataType,
              const std::vector<int64_t>& dims) {
              if(!tensor->has_uid())
              {
                  tensor->set_uid(uid++);
              }
              tensor->set_data_type(dataType).set_dim(dims).set_stride(
                  generateStrides(dims, TensorLayout::NCHW.strideOrder));
          };

    auto dyTensor = makeInput("dy", tensorDims);
    auto xTensor = makeInput("x", tensorDims);
    auto alphaTensor = makeInput("alpha", alphaDims);
    auto zeroTensor = makeInput("zero", scalarDims);

    hipdnn_frontend::graph::PointwiseAttributes compareAttrs;
    compareAttrs.set_name("positive_mask").set_mode(hipdnn_frontend::PointwiseMode::CMP_GT);
    auto positiveMask = graph->pointwise(xTensor, zeroTensor, compareAttrs);
    configureOutput(positiveMask, hipdnn_frontend::DataType::BOOLEAN, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes multiplyXAlphaAttrs;
    multiplyXAlphaAttrs.set_name("x_times_alpha").set_mode(hipdnn_frontend::PointwiseMode::MUL);
    auto xTimesAlpha = graph->pointwise(xTensor, alphaTensor, multiplyXAlphaAttrs);
    configureOutput(xTimesAlpha, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes ySelectAttrs;
    ySelectAttrs.set_name("y_select").set_mode(hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto yTensor = graph->pointwise(xTensor, xTimesAlpha, positiveMask, ySelectAttrs);
    configureOutput(yTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);
    yTensor->set_output(true);

    hipdnn_frontend::graph::PointwiseAttributes multiplyAlphaAttrs;
    multiplyAlphaAttrs.set_name("dy_times_alpha").set_mode(hipdnn_frontend::PointwiseMode::MUL);
    auto dyTimesAlpha = graph->pointwise(dyTensor, alphaTensor, multiplyAlphaAttrs);
    configureOutput(dyTimesAlpha, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes dxSelectAttrs;
    dxSelectAttrs.set_name("dx_select").set_mode(hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto dxTensor = graph->pointwise(dyTensor, dyTimesAlpha, positiveMask, dxSelectAttrs);
    configureOutput(dxTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);
    dxTensor->set_output(true);

    hipdnn_frontend::graph::PointwiseAttributes multiplyXAttrs;
    multiplyXAttrs.set_name("dy_times_x").set_mode(hipdnn_frontend::PointwiseMode::MUL);
    auto dyTimesX = graph->pointwise(dyTensor, xTensor, multiplyXAttrs);
    configureOutput(dyTimesX, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes dalphaSelectAttrs;
    dalphaSelectAttrs.set_name("dalpha_select")
        .set_mode(hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto dalphaTerms = graph->pointwise(zeroTensor, dyTimesX, positiveMask, dalphaSelectAttrs);
    configureOutput(dalphaTerms, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::ReductionAttributes reduceAttrs;
    reduceAttrs.set_name("dalpha_reduce").set_mode(hipdnn_frontend::ReductionMode::ADD);
    auto dalphaTensor = graph->reduction(dalphaTerms, reduceAttrs);
    configureOutput(dalphaTensor, hipdnn_frontend::DataType::FLOAT, alphaDims);
    dalphaTensor->set_output(true);

    constexpr std::array<std::array<float, 4>, 2> X_VALUES = {
        std::array{-2.0f, 0.0f, 1.0f, 2.0f},
        std::array{-3.0f, -1.0f, 0.0f, 4.0f},
    };
    for(int64_t channel = 0; channel < 2; ++channel)
    {
        const auto channelIndex = static_cast<size_t>(channel);
        alpha.setHostValue(channel == 0 ? 0.1f : 0.25f, 0, channel, 0, 0);
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                const auto dyValue = static_cast<float>(1 + channel * 4 + height * 2 + width);
                const auto xValue = X_VALUES[channelIndex][static_cast<size_t>(height * 2 + width)];
                dy.setHostValue(dyValue, 0, channel, height, width);
                x.setHostValue(xValue, 0, channel, height, width);
            }
        }
    }
    zero.setHostValue(0.0f, 0, 0, 0, 0);
    dy.markHostModified();
    x.markHostModified();
    alpha.markHostModified();
    zero.markHostModified();

    const auto validationResult = graph->validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const std::unordered_map<int64_t, void*> variantPack = {
        {dyTensor->get_uid(), dy.memory().hostData()},
        {xTensor->get_uid(), x.memory().hostData()},
        {alphaTensor->get_uid(), alpha.memory().hostData()},
        {zeroTensor->get_uid(), zero.memory().hostData()},
        {yTensor->get_uid(), y.memory().hostData()},
        {dxTensor->get_uid(), dx.memory().hostData()},
        {dalphaTensor->get_uid(), dalpha.memory().hostData()},
    };
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    std::array<float, 2> expectedDalpha = {0.0f, 0.0f};
    for(int64_t channel = 0; channel < 2; ++channel)
    {
        const auto channelIndex = static_cast<size_t>(channel);
        const float alphaValue = alpha.getHostValue(0, channel, 0, 0);
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                const float dyValue = dy.getHostValue(0, channel, height, width);
                const float xValue = x.getHostValue(0, channel, height, width);
                const bool positive = xValue > 0.0f;
                EXPECT_FLOAT_EQ(y.getHostValue(0, channel, height, width),
                                positive ? xValue : xValue * alphaValue);
                EXPECT_FLOAT_EQ(dx.getHostValue(0, channel, height, width),
                                positive ? dyValue : dyValue * alphaValue);
                if(!positive)
                {
                    expectedDalpha[channelIndex] += dyValue * xValue;
                }
            }
        }
        EXPECT_FLOAT_EQ(dalpha.getHostValue(0, channel, 0, 0), expectedDalpha[channelIndex]);
    }
}

namespace
{

// Calls `visit` with every logical index of a tensor of shape `dims` and its row-major position.
template <typename Visit>
void forEachIndex(const std::vector<int64_t>& dims, Visit visit)
{
    size_t total = 1;
    for(const int64_t dim : dims)
    {
        total *= static_cast<size_t>(dim);
    }

    std::vector<int64_t> index(dims.size());
    for(size_t position = 0; position < total; ++position)
    {
        size_t remaining = position;
        for(size_t axis = dims.size(); axis-- > 0;)
        {
            index[axis] = static_cast<int64_t>(remaining % static_cast<size_t>(dims[axis]));
            remaining /= static_cast<size_t>(dims[axis]);
        }
        visit(index, position);
    }
}

// Runs ADD or MUL over the tensors, whatever their strides, and checks every output element by
// logical index. The inputs differ at every position, so a stride applied to the wrong tensor, or
// ignored, gives a wrong value. The values are small integers and halves, so the results are exact
// in FP32 whatever the executor computes in. `in1` may broadcast (dimensions of 1).
void expectBinaryMatchesLogicalIndexing(PointwiseMode mode,
                                        Tensor<float>& in0,
                                        Tensor<float>& in1,
                                        Tensor<float>& out)
{
    // The uids createPointwiseGraph gives the first input, the output and the second input.
    constexpr int64_t IN0_UID = 1;
    constexpr int64_t OUT_UID = 2;
    constexpr int64_t IN1_UID = 3;

    const auto graph = createPointwiseGraph(PointwiseGraphSpec::binary(
        mode, in0.dims(), in0.strides(), out.dims(), out.strides(), in1.dims(), in1.strides()));

    forEachIndex(in0.dims(), [&in0](const std::vector<int64_t>& index, size_t position) {
        in0.setHostValue(static_cast<float>(position), index);
    });
    forEachIndex(in1.dims(), [&in1](const std::vector<int64_t>& index, size_t position) {
        in1.setHostValue(static_cast<float>(position) * 0.5f - 1.0f, index);
    });

    const std::unordered_map<int64_t, void*> variantPack = {{IN0_UID, in0.memory().hostData()},
                                                            {OUT_UID, out.memory().hostData()},
                                                            {IN1_UID, in1.memory().hostData()}};
    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
    executor.execute(graph.GetBufferPointer(), graph.GetSize(), variantPack);

    forEachIndex(out.dims(), [&](const std::vector<int64_t>& index, size_t position) {
        std::vector<int64_t> operandIndex(index.size());
        for(size_t axis = 0; axis < index.size(); ++axis)
        {
            operandIndex[axis] = in1.dims()[axis] == 1 ? 0 : index[axis];
        }
        const float a = in0.getHostValue(index);
        const float b = in1.getHostValue(operandIndex);
        EXPECT_FLOAT_EQ(out.getHostValue(index), mode == PointwiseMode::ADD ? a + b : a * b)
            << "at position " << position;
    });
}

} // namespace

// The layout conv and batchnorm activations use: channels last, with a per-channel operand.
TEST(TestPointwiseOperationsCpuGraphExecutor, BinaryAddNhwcWithPerChannelOperand)
{
    Tensor<float> in0({2, 3, 4, 5}, TensorLayout::NHWC);
    Tensor<float> in1({1, 3, 1, 1}, TensorLayout::NHWC);
    Tensor<float> out({2, 3, 4, 5}, TensorLayout::NHWC);

    expectBinaryMatchesLogicalIndexing(PointwiseMode::ADD, in0, in1, out);
}

// Each tensor keeps its own strides: the first input channels first, the other two channels last.
TEST(TestPointwiseOperationsCpuGraphExecutor, BinaryAddMixedLayouts)
{
    Tensor<float> in0({2, 3, 4, 5}, TensorLayout::NCHW);
    Tensor<float> in1({2, 3, 4, 5}, TensorLayout::NHWC);
    Tensor<float> out({2, 3, 4, 5}, TensorLayout::NHWC);

    expectBinaryMatchesLogicalIndexing(PointwiseMode::ADD, in0, in1, out);
}

// Rows of 5 padded to a pitch of 8 (first input) and 9 (output), as in a view into a larger buffer.
TEST(TestPointwiseOperationsCpuGraphExecutor, BinaryMulPaddedStrides)
{
    const std::vector<int64_t> dims = {2, 3, 4, 5};
    const std::vector<int64_t> paddedIn0 = {96, 32, 8, 1};
    const std::vector<int64_t> paddedOut = {108, 36, 9, 1};
    Tensor<float> in0(dims, paddedIn0);
    Tensor<float> in1(dims);
    Tensor<float> out(dims, paddedOut);

    expectBinaryMatchesLogicalIndexing(PointwiseMode::MUL, in0, in1, out);
}

// A unary op that also changes layout: channels last in, channels first out.
TEST(TestPointwiseOperationsCpuGraphExecutor, UnaryReluNhwcInputToNchwOutput)
{
    constexpr int64_t IN_UID = 1;
    constexpr int64_t OUT_UID = 2;
    Tensor<float> in({2, 3, 4, 5}, TensorLayout::NHWC);
    Tensor<float> out({2, 3, 4, 5}, TensorLayout::NCHW);
    const auto graph = createPointwiseGraph(PointwiseGraphSpec::unary(
        PointwiseMode::RELU_FWD, in.dims(), in.strides(), out.dims(), out.strides()));

    // 0 to 119 less 60, so ReLU sees both signs.
    forEachIndex(in.dims(), [&in](const std::vector<int64_t>& index, size_t position) {
        in.setHostValue(static_cast<float>(position) - 60.0f, index);
    });

    const std::unordered_map<int64_t, void*> variantPack
        = {{IN_UID, in.memory().hostData()}, {OUT_UID, out.memory().hostData()}};
    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
    executor.execute(graph.GetBufferPointer(), graph.GetSize(), variantPack);

    forEachIndex(out.dims(), [&](const std::vector<int64_t>& index, size_t position) {
        const float a = in.getHostValue(index);
        EXPECT_FLOAT_EQ(out.getHostValue(index), a > 0.0f ? a : 0.0f) << "at position " << position;
    });
}

// Above the GPU reference's 5-D cap (GpuReferencePointwise::MAX_DIMS), so only the CPU runs it.
TEST(TestPointwiseOperationsCpuGraphExecutor, BinaryAddBroadcastAboveFiveDimensions)
{
    const std::vector<int64_t> dims = {2, 3, 2, 2, 2, 2};
    Tensor<float> in0(dims);
    Tensor<float> in1({1, 3, 1, 1, 1, 1});
    Tensor<float> out(dims);

    expectBinaryMatchesLogicalIndexing(PointwiseMode::ADD, in0, in1, out);
}
