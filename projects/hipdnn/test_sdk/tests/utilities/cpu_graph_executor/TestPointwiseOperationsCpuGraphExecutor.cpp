// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>

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

// =============================================================================
// Additional unary, binary, comparison and logical pointwise operations
// =============================================================================

struct PointwiseOperationTestParams
{
    hipdnn_frontend::PointwiseMode mode;
    DataType inputDataType = DataType::FLOAT;
    DataType input1DataType = DataType::FLOAT;
    DataType accumulatorDataType = DataType::FLOAT;
    float in0TensorValue = 0.0f;
    float in1TensorValue = 0.0f;
    std::vector<int64_t> in0Dims = {1, 3, 4, 4};
    std::vector<int64_t> in1Dims = {1, 3, 4, 4};
    std::vector<int64_t> outputDims = {1, 3, 4, 4};

    // Optional parameters
    std::optional<float> eluAlpha;
    std::optional<float> softplusBeta;

    // Test metadata
    std::string testDescription;
};

class PointwiseOperationsTestHelper
{
public:
    template <typename T>
    static void runUnaryTest(const PointwiseOperationTestParams& params)
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseUnaryGraph(params.in0Dims,
                                       params.outputDims,
                                       params.inputDataType,
                                       params.accumulatorDataType,
                                       params.inputDataType,
                                       params.mode,
                                       1,
                                       TensorLayout::NCHW,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       params.eluAlpha,
                                       params.softplusBeta);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        auto graphWrap = GraphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrap.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

        tensorBundle.tensors[attributes.in_0_tensor_uid()]->fillTensorWithValue(
            params.in0TensorValue);

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        const auto& outTensor = tensorBundle.tensors.at(attributes.out_0_tensor_uid());
        EXPECT_TRUE(verifyUnaryOutput<T>(*outTensor, params));
    }

    template <typename T>
    static void runBinaryTest(const PointwiseOperationTestParams& params)
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseBinaryGraph(params.in0Dims,
                                        params.in1Dims,
                                        params.outputDims,
                                        params.inputDataType,
                                        params.inputDataType,
                                        params.accumulatorDataType,
                                        params.inputDataType,
                                        params.mode);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        auto graphWrap = GraphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrap.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

        tensorBundle.tensors[attributes.in_0_tensor_uid()]->fillTensorWithValue(
            params.in0TensorValue);
        tensorBundle.tensors[attributes.in_1_tensor_uid().value()]->fillTensorWithValue(
            params.in1TensorValue);

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        const auto& outTensor = tensorBundle.tensors.at(attributes.out_0_tensor_uid());
        EXPECT_TRUE((verifyBinaryOutput<T, T, T>(*outTensor, params)));
    }

    // In0 and in1 may have different data types; the output uses the in0 data type.
    template <typename T0, typename T1>
    static void runMixedTypeBinaryTest(const PointwiseOperationTestParams& params)
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseBinaryGraph(params.in0Dims,
                                        params.in1Dims,
                                        params.outputDims,
                                        params.inputDataType,
                                        params.input1DataType,
                                        params.accumulatorDataType,
                                        params.inputDataType,
                                        params.mode);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        auto graphWrap = GraphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrap.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

        tensorBundle.tensors[attributes.in_0_tensor_uid()]->fillTensorWithValue(
            params.in0TensorValue);
        tensorBundle.tensors[attributes.in_1_tensor_uid().value()]->fillTensorWithValue(
            params.in1TensorValue);

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        const auto& outTensor = tensorBundle.tensors.at(attributes.out_0_tensor_uid());
        EXPECT_TRUE((verifyBinaryOutput<T0, T1, T0>(*outTensor, params)));
    }

    // Comparison operations produce a BOOLEAN output tensor.
    template <typename T0, typename T1>
    static void runCompareTest(const PointwiseOperationTestParams& params, bool expectedResult)
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseBinaryGraph(params.in0Dims,
                                        params.in1Dims,
                                        params.outputDims,
                                        params.inputDataType,
                                        params.input1DataType,
                                        params.accumulatorDataType,
                                        DataType::BOOLEAN,
                                        params.mode);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        auto graphWrap = GraphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrap.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

        tensorBundle.tensors[attributes.in_0_tensor_uid()]->fillTensorWithValue(
            params.in0TensorValue);
        tensorBundle.tensors[attributes.in_1_tensor_uid().value()]->fillTensorWithValue(
            params.in1TensorValue);

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        const auto* outTensor = static_cast<Tensor<bool>*>(
            tensorBundle.tensors.at(attributes.out_0_tensor_uid()).get());
        verifyBooleanOutput(*outTensor, params.outputDims, expectedResult);
    }

    static void runLogicalBinaryTest(hipdnn_frontend::PointwiseMode mode,
                                     bool in0Value,
                                     bool in1Value,
                                     bool expectedResult)
    {
        const std::vector<int64_t> dims = {1, 2, 2, 2};
        auto [graph, tensorBundle, variantPack] = buildPointwiseBinaryGraph(dims,
                                                                            dims,
                                                                            dims,
                                                                            DataType::BOOLEAN,
                                                                            DataType::BOOLEAN,
                                                                            DataType::FLOAT,
                                                                            DataType::BOOLEAN,
                                                                            mode);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        auto graphWrap = GraphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrap.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

        auto* input0 = static_cast<Tensor<bool>*>(
            tensorBundle.tensors.at(attributes.in_0_tensor_uid()).get());
        auto* input1 = static_cast<Tensor<bool>*>(
            tensorBundle.tensors.at(attributes.in_1_tensor_uid().value()).get());
        input0->fillWithValue(in0Value);
        input1->fillWithValue(in1Value);

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        const auto* outTensor = static_cast<Tensor<bool>*>(
            tensorBundle.tensors.at(attributes.out_0_tensor_uid()).get());
        verifyBooleanOutput(*outTensor, dims, expectedResult);
    }

    static void runLogicalNotTest(bool inValue, bool expectedResult)
    {
        const std::vector<int64_t> dims = {1, 2, 2, 2};
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseUnaryGraph(dims,
                                       dims,
                                       DataType::BOOLEAN,
                                       DataType::FLOAT,
                                       DataType::BOOLEAN,
                                       hipdnn_frontend::PointwiseMode::LOGICAL_NOT);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        auto graphWrap = GraphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrap.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

        auto* input = static_cast<Tensor<bool>*>(
            tensorBundle.tensors.at(attributes.in_0_tensor_uid()).get());
        input->fillWithValue(inValue);

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        const auto* outTensor = static_cast<Tensor<bool>*>(
            tensorBundle.tensors.at(attributes.out_0_tensor_uid()).get());
        verifyBooleanOutput(*outTensor, dims, expectedResult);
    }

    template <typename InputType>
    static bool verifyUnaryOutput(ITensor& outTensor, const PointwiseOperationTestParams& params)
    {
        Tensor<InputType> input0(params.in0Dims);
        input0.fillTensorWithValue(params.in0TensorValue);
        Tensor<InputType> output(params.outputDims);

        if(params.eluAlpha.has_value() || params.softplusBeta.has_value())
        {
            CpuReferencePointwiseImpl<InputType, InputType>::pointwiseCompute(
                sdkPointwiseMode(params.mode),
                output,
                input0,
                0.0f,
                std::numeric_limits<float>::max(),
                0.0f,
                1.0f,
                params.eluAlpha.has_value() ? params.eluAlpha.value() : 1.0f,
                params.softplusBeta.has_value() ? params.softplusBeta.value() : 1.0f);
        }
        else
        {
            CpuReferencePointwiseImpl<InputType, InputType>::pointwiseCompute(
                sdkPointwiseMode(params.mode), output, input0);
        }

        const CpuFpReferenceValidation<InputType> validator;
        return validator.allClose(output, outTensor);
    }

    template <typename Input0Type, typename Input1Type, typename OutputType>
    static bool verifyBinaryOutput(ITensor& outTensor, const PointwiseOperationTestParams& params)
    {
        Tensor<Input0Type> input0(params.in0Dims);
        Tensor<Input1Type> input1(params.in1Dims);
        input0.fillTensorWithValue(params.in0TensorValue);
        input1.fillTensorWithValue(params.in1TensorValue);
        Tensor<OutputType> output(params.outputDims);

        CpuReferencePointwiseImpl<OutputType, Input0Type, Input1Type>::pointwiseCompute(
            sdkPointwiseMode(params.mode), output, input0, input1);

        const CpuFpReferenceValidation<OutputType> validator;
        return validator.allClose(output, outTensor);
    }

private:
    static PointwiseMode sdkPointwiseMode(hipdnn_frontend::PointwiseMode mode)
    {
        return hipdnn_test_sdk::utilities::frontendToSdkPointwiseMode(mode);
    }

    static void verifyBooleanOutput(const Tensor<bool>& output,
                                    const std::vector<int64_t>& dims,
                                    bool expectedResult)
    {
        for(int64_t n = 0; n < dims[0]; ++n)
        {
            for(int64_t c = 0; c < dims[1]; ++c)
            {
                for(int64_t h = 0; h < dims[2]; ++h)
                {
                    for(int64_t w = 0; w < dims[3]; ++w)
                    {
                        EXPECT_EQ(output.getHostValue(n, c, h, w), expectedResult);
                    }
                }
            }
        }
    }
};

template <typename T>
class PointwiseOperationsCpuGraphExecutor : public ::testing::Test
{
protected:
    using DataTypeT = T;
};

TYPED_TEST_SUITE(PointwiseOperationsCpuGraphExecutor, TestTypes, );

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryExp)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::EXP;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.5f;
    params.testDescription = "Exponential of a positive input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryLog)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::LOG;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 2.0f;
    params.testDescription = "Natural logarithm of a positive input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnarySqrt)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SQRT;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 4.0f;
    params.testDescription = "Square root of a positive input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryRsqrt)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RSQRT;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 4.0f;
    params.testDescription = "Reciprocal square root of a positive input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryReciprocal)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RECIPROCAL;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 2.0f;
    params.testDescription = "Reciprocal of a positive input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnarySin)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SIN;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.testDescription = "Sine of a positive input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryTan)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::TAN;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 0.5f;
    params.testDescription = "Tangent of a positive input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryErf)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ERF;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 0.5f;
    params.testDescription = "Error function of a positive input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryCeilPositive)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CEIL;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.5f;
    params.testDescription = "Ceil of a positive fractional input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryCeilNegative)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CEIL;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -1.5f;
    params.testDescription = "Ceil of a negative fractional input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryFloorPositive)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::FLOOR;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.5f;
    params.testDescription = "Floor of a positive fractional input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryFloorNegative)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::FLOOR;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -1.5f;
    params.testDescription = "Floor of a negative fractional input";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryIdentity)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::IDENTITY;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 3.0f;
    params.testDescription = "Identity passes the input through";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryEluPositiveInput)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 2.0f;
    params.testDescription = "ELU with positive input passes through";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryEluNegativeInput)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -1.0f;
    params.testDescription = "ELU with negative input, default alpha";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnarySoftplusPositiveInput)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.testDescription = "Softplus with positive input, default beta";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnarySoftplusNegativeInput)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -1.0f;
    params.testDescription = "Softplus with negative input, default beta";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryEluWithAlpha)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ELU_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = -1.0f;
    params.eluAlpha = 0.5f;
    params.testDescription = "ELU with negative input and custom alpha";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnarySoftplusWithBeta)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.softplusBeta = 2.0f;
    params.testDescription = "Softplus with custom beta";

    PointwiseOperationsTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryDivide)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::DIV;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 6.0f;
    params.in1TensorValue = 3.0f;
    params.testDescription = "Divide with same-shape inputs";

    PointwiseOperationsTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryDivideBroadcastChannel)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::DIV;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 6.0f;
    params.in1TensorValue = 3.0f;
    params.in1Dims = {1, 3, 1, 1};
    params.testDescription = "Divide with in1 broadcast over H and W";

    PointwiseOperationsTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryDivideBroadcastScalarIn0)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::DIV;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 6.0f;
    params.in1TensorValue = 3.0f;
    params.in0Dims = {1, 1, 1, 1};
    params.testDescription = "Divide with in0 broadcast to the output shape";

    PointwiseOperationsTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryAddSquare)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ADD_SQUARE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "AddSquare with same-shape inputs";

    PointwiseOperationsTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryAddSquareBroadcastChannel)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ADD_SQUARE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.in1Dims = {1, 3, 1, 1};
    params.testDescription = "AddSquare with in1 broadcast over H and W";

    PointwiseOperationsTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryAddSquareBroadcastScalarIn0)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ADD_SQUARE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.in0Dims = {1, 1, 1, 1};
    params.testDescription = "AddSquare with in0 broadcast to the output shape";

    PointwiseOperationsTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareEqualEqualInputs)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_EQ;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareEqual with EqualInputs";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareEqualFirstSmaller)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_EQ;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareEqual with FirstSmaller";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, false);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareEqualFirstLarger)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_EQ;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 3.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareEqual with FirstLarger";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, false);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareEqualBroadcastChannel)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_EQ;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.in1Dims = {1, 3, 1, 1};
    params.testDescription = "CompareEqual with in1 broadcast";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, false);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareNotEqualEqualInputs)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_NEQ;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareNotEqual with EqualInputs";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, false);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareNotEqualFirstSmaller)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_NEQ;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareNotEqual with FirstSmaller";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareNotEqualFirstLarger)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_NEQ;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 3.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareNotEqual with FirstLarger";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareNotEqualBroadcastChannel)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_NEQ;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.in1Dims = {1, 3, 1, 1};
    params.testDescription = "CompareNotEqual with in1 broadcast";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareGreaterEqualEqualInputs)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_GE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareGreaterEqual with EqualInputs";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareGreaterEqualFirstSmaller)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_GE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareGreaterEqual with FirstSmaller";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, false);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareGreaterEqualFirstLarger)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_GE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 3.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareGreaterEqual with FirstLarger";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareGreaterEqualBroadcastChannel)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_GE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.in1Dims = {1, 3, 1, 1};
    params.testDescription = "CompareGreaterEqual with in1 broadcast";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, false);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessEqualInputs)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LT;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareLess with EqualInputs";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, false);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessFirstSmaller)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LT;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareLess with FirstSmaller";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessFirstLarger)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LT;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 3.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareLess with FirstLarger";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, false);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessBroadcastChannel)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LT;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.in1Dims = {1, 3, 1, 1};
    params.testDescription = "CompareLess with in1 broadcast";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessEqualEqualInputs)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareLessEqual with EqualInputs";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessEqualFirstSmaller)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareLessEqual with FirstSmaller";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessEqualFirstLarger)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 3.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "CompareLessEqual with FirstLarger";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, false);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessEqualBroadcastChannel)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LE;
    params.inputDataType = nativeTypeToDataType<TypeParam>();
    params.input1DataType = nativeTypeToDataType<TypeParam>();
    params.accumulatorDataType = nativeTypeToDataType<float>();
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.in1Dims = {1, 3, 1, 1};
    params.testDescription = "CompareLessEqual with in1 broadcast";

    PointwiseOperationsTestHelper::runCompareTest<TypeParam, TypeParam>(params, true);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, DivideMixedTypeFloatInputHalfDivisor)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::DIV;
    params.inputDataType = DataType::FLOAT;
    params.input1DataType = DataType::HALF;
    params.accumulatorDataType = DataType::FLOAT;
    params.in0TensorValue = 6.0f;
    params.in1TensorValue = 4.0f;
    params.testDescription = "DIV with float dividend and half divisor";

    PointwiseOperationsTestHelper::runMixedTypeBinaryTest<float, half>(params);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, AddSquareMixedTypeFloatInputBfloat16Squared)
{
    PointwiseOperationTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ADD_SQUARE;
    params.inputDataType = DataType::FLOAT;
    params.input1DataType = DataType::BFLOAT16;
    params.accumulatorDataType = DataType::FLOAT;
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.testDescription = "ADD_SQUARE with float first input and bfloat16 second input";

    PointwiseOperationsTestHelper::runMixedTypeBinaryTest<float, bfloat16>(params);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, LogicalAndTrueAndTrue)
{
    PointwiseOperationsTestHelper::runLogicalBinaryTest(
        hipdnn_frontend::PointwiseMode::LOGICAL_AND, true, true, true);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, LogicalAndTrueAndFalse)
{
    PointwiseOperationsTestHelper::runLogicalBinaryTest(
        hipdnn_frontend::PointwiseMode::LOGICAL_AND, true, false, false);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, LogicalOrFalseOrFalse)
{
    PointwiseOperationsTestHelper::runLogicalBinaryTest(
        hipdnn_frontend::PointwiseMode::LOGICAL_OR, false, false, false);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, LogicalOrFalseOrTrue)
{
    PointwiseOperationsTestHelper::runLogicalBinaryTest(
        hipdnn_frontend::PointwiseMode::LOGICAL_OR, false, true, true);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, LogicalNotTrue)
{
    PointwiseOperationsTestHelper::runLogicalNotTest(true, false);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, LogicalNotFalse)
{
    PointwiseOperationsTestHelper::runLogicalNotTest(false, true);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, ExpReciprocalDivideComposedGraph)
{
    const std::vector<int64_t> tensorDims = {1, 2, 2, 2};
    const std::vector<int64_t> scalarDims = {1, 1, 1, 1};

    Tensor<float> x(tensorDims);
    Tensor<float> zero(scalarDims);
    Tensor<float> selected(tensorDims);

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("ExpReciprocalDivideComposedGraph");
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

    auto xTensor = makeInput("x", tensorDims);
    auto zeroTensor = makeInput("zero", scalarDims);

    hipdnn_frontend::graph::PointwiseAttributes expAttrs;
    expAttrs.set_name("exp").set_mode(hipdnn_frontend::PointwiseMode::EXP);
    auto expTensor = graph->pointwise(xTensor, expAttrs);
    configureOutput(expTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes reciprocalAttrs;
    reciprocalAttrs.set_name("reciprocal").set_mode(hipdnn_frontend::PointwiseMode::RECIPROCAL);
    auto reciprocalTensor = graph->pointwise(expTensor, reciprocalAttrs);
    configureOutput(reciprocalTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);

    // exp(-x) / exp(x) = exp(-2x)
    hipdnn_frontend::graph::PointwiseAttributes divideAttrs;
    divideAttrs.set_name("divide").set_mode(hipdnn_frontend::PointwiseMode::DIV);
    auto quotientTensor = graph->pointwise(reciprocalTensor, expTensor, divideAttrs);
    configureOutput(quotientTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes compareAttrs;
    compareAttrs.set_name("positive").set_mode(hipdnn_frontend::PointwiseMode::CMP_GT);
    auto positiveMask = graph->pointwise(xTensor, zeroTensor, compareAttrs);
    configureOutput(positiveMask, hipdnn_frontend::DataType::BOOLEAN, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes notAttrs;
    notAttrs.set_name("non_positive").set_mode(hipdnn_frontend::PointwiseMode::LOGICAL_NOT);
    auto nonPositiveTensor = graph->pointwise(positiveMask, notAttrs);
    configureOutput(nonPositiveTensor, hipdnn_frontend::DataType::BOOLEAN, tensorDims);

    // exp(-2x) where x <= 0, exp(x) otherwise
    hipdnn_frontend::graph::PointwiseAttributes selectAttrs;
    selectAttrs.set_name("select").set_mode(hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto selectedTensor
        = graph->pointwise(quotientTensor, expTensor, nonPositiveTensor, selectAttrs);
    configureOutput(selectedTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);
    selectedTensor->set_output(true);

    constexpr std::array<float, 8> X_VALUES = {-1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f, 1.5f, -2.0f};
    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                x.setHostValue(X_VALUES[static_cast<size_t>(channel * 4 + height * 2 + width)],
                               0,
                               channel,
                               height,
                               width);
            }
        }
    }
    zero.setHostValue(0.0f, 0, 0, 0, 0);
    x.markHostModified();
    zero.markHostModified();

    const auto validationResult = graph->validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const std::unordered_map<int64_t, void*> variantPack = {
        {xTensor->get_uid(), x.memory().hostData()},
        {zeroTensor->get_uid(), zero.memory().hostData()},
        {selectedTensor->get_uid(), selected.memory().hostData()},
    };
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                const float xValue = x.getHostValue(0, channel, height, width);
                const float expected = xValue <= 0.0f ? std::exp(-2.0f * xValue) : std::exp(xValue);
                EXPECT_NEAR(selected.getHostValue(0, channel, height, width), expected, 1e-4f);
            }
        }
    }
}
