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

struct PointwiseTestParams
{
    hipdnn_frontend::PointwiseMode mode;
    float in0TensorValue = 0.0f; // Unary: x; backward modes: dy.
    float in1TensorValue = 0.0f; // Binary: second operand; backward modes: x.
    std::vector<int64_t> in0Dims = {1, 3, 4, 4};
    std::vector<int64_t> in1Dims = {1, 3, 4, 4};
    std::vector<int64_t> outputDims = {1, 3, 4, 4};

    // Optional mode parameters
    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;
    std::optional<float> reluLowerClipSlope;
    std::optional<float> swishBeta;
    std::optional<float> eluAlpha;
    std::optional<float> softplusBeta;
    std::optional<int64_t> axis; // GEN_INDEX

    // Expected value of every element of a BOOLEAN output (comparison and logical modes)
    bool expectedBoolean = false;

    // Test metadata
    std::string testDescription;
};

class PointwiseTestHelper
{
public:
    template <typename InputType, typename OutputType = InputType>
    static void runUnaryTest(const PointwiseTestParams& params)
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseUnaryGraph(params.in0Dims,
                                       params.outputDims,
                                       nativeTypeToDataType<InputType>(),
                                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                       nativeTypeToDataType<OutputType>(),
                                       params.mode,
                                       1,
                                       TensorLayout::NCHW,
                                       params.reluLowerClip,
                                       params.reluUpperClip,
                                       params.reluLowerClipSlope,
                                       params.swishBeta,
                                       params.eluAlpha,
                                       params.softplusBeta,
                                       params.axis);

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
        EXPECT_TRUE((verifyOutput<InputType, InputType, OutputType>(*outTensor, params)));
    }

    template <typename Input0Type,
              typename Input1Type = Input0Type,
              typename OutputType = Input0Type>
    static void runBinaryTest(const PointwiseTestParams& params)
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseBinaryGraph(params.in0Dims,
                                        params.in1Dims,
                                        params.outputDims,
                                        nativeTypeToDataType<Input0Type>(),
                                        nativeTypeToDataType<Input1Type>(),
                                        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                        nativeTypeToDataType<OutputType>(),
                                        params.mode,
                                        1,
                                        TensorLayout::NCHW,
                                        params.reluLowerClip,
                                        params.reluUpperClip,
                                        params.reluLowerClipSlope,
                                        params.swishBeta,
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
        tensorBundle.tensors[attributes.in_1_tensor_uid().value()]->fillTensorWithValue(
            params.in1TensorValue);

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        const auto& outTensor = tensorBundle.tensors.at(attributes.out_0_tensor_uid());
        EXPECT_TRUE((verifyOutput<Input0Type, Input1Type, OutputType>(*outTensor, params)));
    }

private:
    static bool hasModeParameters(const PointwiseTestParams& params)
    {
        return params.reluLowerClip.has_value() || params.reluUpperClip.has_value()
               || params.reluLowerClipSlope.has_value() || params.swishBeta.has_value()
               || params.eluAlpha.has_value() || params.softplusBeta.has_value();
    }

    template <typename Input0Type, typename Input1Type, typename OutputType>
    static bool verifyOutput(ITensor& outTensor, const PointwiseTestParams& params)
    {
        if constexpr(std::is_same_v<OutputType, bool>)
        {
            const auto& output = static_cast<const Tensor<bool>&>(outTensor);
            return allElementsEqual(
                output, params.outputDims, [&](int64_t, int64_t, int64_t, int64_t) {
                    return params.expectedBoolean;
                });
        }
        else if constexpr(std::is_same_v<OutputType, int32_t>) // GEN_INDEX
        {
            const auto& output = static_cast<const Tensor<OutputType>&>(outTensor);
            const auto axis = static_cast<size_t>(params.axis.value());
            return allElementsEqual(
                output, params.outputDims, [&](int64_t n, int64_t c, int64_t h, int64_t w) {
                    return static_cast<OutputType>(std::array<int64_t, 4>{n, c, h, w}[axis]);
                });
        }
        else
        {
            const auto mode = frontendToSdkPointwiseMode(params.mode);
            Tensor<Input0Type> input0(params.in0Dims);
            input0.fillTensorWithValue(params.in0TensorValue);
            Tensor<OutputType> output(params.outputDims);

            if(hipdnn_flatbuffers_sdk::utilities::isBinaryPointwiseMode(mode))
            {
                Tensor<Input1Type> input1(params.in1Dims);
                input1.fillTensorWithValue(params.in1TensorValue);
                if(hasModeParameters(params))
                {
                    CpuReferencePointwiseImpl<OutputType, Input0Type, Input1Type>::pointwiseCompute(
                        mode,
                        output,
                        input0,
                        input1,
                        params.reluLowerClip.value_or(0.0f),
                        params.reluUpperClip.value_or(std::numeric_limits<float>::max()),
                        params.reluLowerClipSlope.value_or(0.0f),
                        params.swishBeta.value_or(1.0f),
                        params.eluAlpha.value_or(1.0f),
                        params.softplusBeta.value_or(1.0f));
                }
                else
                {
                    CpuReferencePointwiseImpl<OutputType, Input0Type, Input1Type>::pointwiseCompute(
                        mode, output, input0, input1);
                }
            }
            else if(hasModeParameters(params))
            {
                CpuReferencePointwiseImpl<OutputType, Input0Type>::pointwiseCompute(
                    mode,
                    output,
                    input0,
                    params.reluLowerClip.value_or(0.0f),
                    params.reluUpperClip.value_or(std::numeric_limits<float>::max()),
                    params.reluLowerClipSlope.value_or(0.0f),
                    params.swishBeta.value_or(1.0f),
                    params.eluAlpha.value_or(1.0f),
                    params.softplusBeta.value_or(1.0f));
            }
            else
            {
                CpuReferencePointwiseImpl<OutputType, Input0Type>::pointwiseCompute(
                    mode, output, input0);
            }

            const CpuFpReferenceValidation<OutputType> validator;
            return validator.allClose(output, outTensor);
        }
    }

    template <typename T, typename ExpectedFn>
    static bool allElementsEqual(const Tensor<T>& output,
                                 const std::vector<int64_t>& dims,
                                 ExpectedFn expected)
    {
        for(int64_t n = 0; n < dims[0]; ++n)
        {
            for(int64_t c = 0; c < dims[1]; ++c)
            {
                for(int64_t h = 0; h < dims[2]; ++h)
                {
                    for(int64_t w = 0; w < dims[3]; ++w)
                    {
                        if(output.getHostValue(n, c, h, w) != expected(n, c, h, w))
                        {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
};

template <typename T>
class PointwiseOperationsCpuGraphExecutor : public ::testing::Test
{
protected:
    using DataTypeT = T;
};

using TestTypes = ::testing::Types<float, half, bfloat16>;
TYPED_TEST_SUITE(PointwiseOperationsCpuGraphExecutor, TestTypes, );

// =============================================================================
// ReLU Forward Tests
// =============================================================================

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluForwardStandardNegativeInput)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.in0TensorValue = -10.0f;
    params.testDescription = "Standard ReLU with negative input should clamp to zero";

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluForwardStandardPositiveInput)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.in0TensorValue = 10.0f;
    params.testDescription = "Standard ReLU with positive input should pass through";

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluForwardClampedBelowLowerBound)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.in0TensorValue = 0.0f;
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU with input below lower bound";

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluForwardClampedAboveUpperBound)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.in0TensorValue = 2.0f;
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU with input above upper bound";

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluForwardClampedWithinBounds)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.in0TensorValue = 0.2f;
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU with input within bounds";

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluForwardLeakyNegativeInput)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.in0TensorValue = -0.5f;
    params.reluLowerClipSlope = 0.1f;
    params.testDescription = "Leaky ReLU with negative input";

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluForwardLeakyPositiveInput)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.in0TensorValue = 0.5f;
    params.reluLowerClipSlope = 0.1f;
    params.testDescription = "Leaky ReLU with positive input";

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluForwardUpperBoundOnlyBelowBound)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.in0TensorValue = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Upper-bounded ReLU with input below bound";

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluForwardUpperBoundOnlyAboveBound)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_FWD;
    params.in0TensorValue = 0.9f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Upper-bounded ReLU with input above bound";

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

// =============================================================================
// ReLU Backward Tests
// =============================================================================
TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluBackwardStandardNegativeX)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = -10.0f; // X
    params.testDescription = "Standard ReLU backward with negative X";

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluBackwardStandardPositiveX)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = 10.0f; // X
    params.testDescription = "Standard ReLU backward with positive X";

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluBackwardClampedBelowBounds)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = -1.0f; // X
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU backward with X below bounds";

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluBackwardClampedAboveBounds)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.in0TensorValue = 2.0f; // Dy
    params.in1TensorValue = 1.0f; // X
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU backward with X above bounds";

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluBackwardClampedWithinBounds)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.in0TensorValue = 1.0f; // Dy
    params.in1TensorValue = 0.2f; // X
    params.reluLowerClip = 0.1f;
    params.reluUpperClip = 0.3f;
    params.testDescription = "Clamped ReLU backward with X within bounds";

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluBackwardLeakyNegativeX)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = -1.0f; // X
    params.reluLowerClipSlope = 0.1f;
    params.testDescription = "Leaky ReLU backward with negative X";

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluBackwardLeakyPositiveX)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.in0TensorValue = 2.0f; // Dy
    params.in1TensorValue = 1.0f; // X
    params.reluLowerClipSlope = 0.1f;
    params.testDescription = "Leaky ReLU backward with positive X";

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluBackwardUpperBoundOnlyBelowBound)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.in0TensorValue = 5.0f; // Dy
    params.in1TensorValue = 1.0f; // X
    params.reluUpperClip = 10.0f;
    params.testDescription = "Upper-bounded ReLU backward with X below bound";

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, ReluBackwardUpperBoundOnlyAboveBound)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::RELU_BWD;
    params.in0TensorValue = 2.0f; // Dy
    params.in1TensorValue = 20.0f; // X
    params.reluUpperClip = 10.0f;
    params.testDescription = "Upper-bounded ReLU backward with X above bound";

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

namespace
{
// Hands out sequential uids for the tensors of a hand-built composed graph.
struct ComposedGraphTensorFactory
{
    int64_t uid = 1;

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
        makeInput(const std::string& name, const std::vector<int64_t>& dims)
    {
        auto attributes = hipdnn_frontend::graph::makeTensorAttributes(
            name,
            hipdnn_frontend::DataType::FLOAT,
            dims,
            generateStrides(dims, TensorLayout::NCHW.strideOrder));
        attributes.set_uid(uid++);
        return std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(attributes));
    }

    void configureOutput(const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& tensor,
                         hipdnn_frontend::DataType dataType,
                         const std::vector<int64_t>& dims)
    {
        if(!tensor->has_uid())
        {
            tensor->set_uid(uid++);
        }
        tensor->set_data_type(dataType).set_dim(dims).set_stride(
            generateStrides(dims, TensorLayout::NCHW.strideOrder));
    }
};
} // namespace

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

    ComposedGraphTensorFactory tensors;

    auto dyTensor = tensors.makeInput("dy", tensorDims);
    auto xTensor = tensors.makeInput("x", tensorDims);
    auto alphaTensor = tensors.makeInput("alpha", alphaDims);
    auto zeroTensor = tensors.makeInput("zero", scalarDims);

    hipdnn_frontend::graph::PointwiseAttributes compareAttrs;
    compareAttrs.set_name("positive_mask").set_mode(hipdnn_frontend::PointwiseMode::CMP_GT);
    auto positiveMask = graph->pointwise(xTensor, zeroTensor, compareAttrs);
    tensors.configureOutput(positiveMask, hipdnn_frontend::DataType::BOOLEAN, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes multiplyXAlphaAttrs;
    multiplyXAlphaAttrs.set_name("x_times_alpha").set_mode(hipdnn_frontend::PointwiseMode::MUL);
    auto xTimesAlpha = graph->pointwise(xTensor, alphaTensor, multiplyXAlphaAttrs);
    tensors.configureOutput(xTimesAlpha, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes ySelectAttrs;
    ySelectAttrs.set_name("y_select").set_mode(hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto yTensor = graph->pointwise(xTensor, xTimesAlpha, positiveMask, ySelectAttrs);
    tensors.configureOutput(yTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);
    yTensor->set_output(true);

    hipdnn_frontend::graph::PointwiseAttributes multiplyAlphaAttrs;
    multiplyAlphaAttrs.set_name("dy_times_alpha").set_mode(hipdnn_frontend::PointwiseMode::MUL);
    auto dyTimesAlpha = graph->pointwise(dyTensor, alphaTensor, multiplyAlphaAttrs);
    tensors.configureOutput(dyTimesAlpha, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes dxSelectAttrs;
    dxSelectAttrs.set_name("dx_select").set_mode(hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto dxTensor = graph->pointwise(dyTensor, dyTimesAlpha, positiveMask, dxSelectAttrs);
    tensors.configureOutput(dxTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);
    dxTensor->set_output(true);

    hipdnn_frontend::graph::PointwiseAttributes multiplyXAttrs;
    multiplyXAttrs.set_name("dy_times_x").set_mode(hipdnn_frontend::PointwiseMode::MUL);
    auto dyTimesX = graph->pointwise(dyTensor, xTensor, multiplyXAttrs);
    tensors.configureOutput(dyTimesX, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes dalphaSelectAttrs;
    dalphaSelectAttrs.set_name("dalpha_select")
        .set_mode(hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto dalphaTerms = graph->pointwise(zeroTensor, dyTimesX, positiveMask, dalphaSelectAttrs);
    tensors.configureOutput(dalphaTerms, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::ReductionAttributes reduceAttrs;
    reduceAttrs.set_name("dalpha_reduce").set_mode(hipdnn_frontend::ReductionMode::ADD);
    auto dalphaTensor = graph->reduction(dalphaTerms, reduceAttrs);
    tensors.configureOutput(dalphaTensor, hipdnn_frontend::DataType::FLOAT, alphaDims);
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
// Unary math operations
// =============================================================================

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryExp)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::EXP;
    params.in0TensorValue = 1.5f;

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnarySqrt)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SQRT;
    params.in0TensorValue = 4.0f;

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryFloorNegativeInput)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::FLOOR;
    params.in0TensorValue = -1.5f;

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryIdentity)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::IDENTITY;
    params.in0TensorValue = 3.0f;

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnaryEluWithAlpha)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ELU_FWD;
    params.in0TensorValue = -1.0f;
    params.eluAlpha = 0.5f;

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, UnarySoftplusWithBeta)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD;
    params.in0TensorValue = 1.0f;
    params.softplusBeta = 2.0f;

    PointwiseTestHelper::runUnaryTest<TypeParam>(params);
}

// =============================================================================
// Binary arithmetic and backward operations
// =============================================================================

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryDivide)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::DIV;
    params.in0TensorValue = 6.0f;
    params.in1TensorValue = 3.0f;

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryDivideBroadcastChannel)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::DIV;
    params.in0TensorValue = 6.0f;
    params.in1TensorValue = 3.0f;
    params.in1Dims = {1, 3, 1, 1};

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryAddSquare)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ADD_SQUARE;
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryAddSquareBroadcastScalarIn0)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ADD_SQUARE;
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.in0Dims = {1, 1, 1, 1};

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryGeluBackwardPositiveX)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::GELU_BWD;
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = 1.0f;

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryGeluBackwardNegativeXBroadcastChannel)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::GELU_BWD;
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = -1.0f;
    params.in1Dims = {1, 3, 1, 1};

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryGeluApproxTanhBackwardNegativeX)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_BWD;
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = -1.0f;

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinarySwishBackwardWithBeta)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SWISH_BWD;
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = 1.0f;
    params.swishBeta = 0.5f;

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryEluBackwardWithAlpha)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::ELU_BWD;
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = -1.0f;
    params.eluAlpha = 0.5f;

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinarySoftplusBackwardNegativeX)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD;
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = -1.0f;

    PointwiseTestHelper::runBinaryTest<TypeParam>(params);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, DivideMixedTypeFloatInputHalfDivisor)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::DIV;
    params.in0TensorValue = 6.0f;
    params.in1TensorValue = 4.0f;

    PointwiseTestHelper::runBinaryTest<float, half>(params);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, SoftplusBackwardMixedTypeFloatDyHalfX)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD;
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = 1.0f;

    PointwiseTestHelper::runBinaryTest<float, half>(params);
}

// =============================================================================
// Comparison operations
// =============================================================================

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareEqualEqualInputs)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_EQ;
    params.in0TensorValue = 2.0f;
    params.in1TensorValue = 2.0f;
    params.expectedBoolean = true;

    PointwiseTestHelper::runBinaryTest<TypeParam, TypeParam, bool>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareNotEqualBroadcastChannel)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_NEQ;
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.in1Dims = {1, 3, 1, 1};
    params.expectedBoolean = true;

    PointwiseTestHelper::runBinaryTest<TypeParam, TypeParam, bool>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareGreaterEqualFirstLarger)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_GE;
    params.in0TensorValue = 3.0f;
    params.in1TensorValue = 2.0f;
    params.expectedBoolean = true;

    PointwiseTestHelper::runBinaryTest<TypeParam, TypeParam, bool>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessFirstSmaller)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LT;
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 2.0f;
    params.expectedBoolean = true;

    PointwiseTestHelper::runBinaryTest<TypeParam, TypeParam, bool>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, BinaryCompareLessFirstLarger)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::CMP_LT;
    params.in0TensorValue = 3.0f;
    params.in1TensorValue = 2.0f;
    params.expectedBoolean = false;

    PointwiseTestHelper::runBinaryTest<TypeParam, TypeParam, bool>(params);
}

// =============================================================================
// Logical operations
// =============================================================================

TEST(TestPointwiseOperationsCpuGraphExecutor, LogicalAndTrueAndFalse)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::LOGICAL_AND;
    params.in0TensorValue = 1.0f;
    params.in1TensorValue = 0.0f;
    params.expectedBoolean = false;

    PointwiseTestHelper::runBinaryTest<bool, bool, bool>(params);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, LogicalOrFalseOrTrue)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::LOGICAL_OR;
    params.in0TensorValue = 0.0f;
    params.in1TensorValue = 1.0f;
    params.expectedBoolean = true;

    PointwiseTestHelper::runBinaryTest<bool, bool, bool>(params);
}

TEST(TestPointwiseOperationsCpuGraphExecutor, LogicalNotTrue)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::LOGICAL_NOT;
    params.in0TensorValue = 1.0f;
    params.expectedBoolean = false;

    PointwiseTestHelper::runUnaryTest<bool>(params);
}

// =============================================================================
// GEN_INDEX (INT32 output)
// =============================================================================

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, GenIndexChannelAxis)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::GEN_INDEX;
    params.in0TensorValue = 0.0f;
    params.axis = 1;
    params.in0Dims = {2, 3, 2, 2};
    params.outputDims = {2, 3, 2, 2};

    PointwiseTestHelper::runUnaryTest<TypeParam, int32_t>(params);
}

TYPED_TEST(PointwiseOperationsCpuGraphExecutor, GenIndexLastAxis)
{
    PointwiseTestParams params;
    params.mode = hipdnn_frontend::PointwiseMode::GEN_INDEX;
    params.in0TensorValue = 0.0f;
    params.axis = 3;
    params.in0Dims = {2, 3, 2, 4};
    params.outputDims = {2, 3, 2, 4};

    PointwiseTestHelper::runUnaryTest<TypeParam, int32_t>(params);
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

    ComposedGraphTensorFactory tensors;

    auto xTensor = tensors.makeInput("x", tensorDims);
    auto zeroTensor = tensors.makeInput("zero", scalarDims);

    hipdnn_frontend::graph::PointwiseAttributes expAttrs;
    expAttrs.set_name("exp").set_mode(hipdnn_frontend::PointwiseMode::EXP);
    auto expTensor = graph->pointwise(xTensor, expAttrs);
    tensors.configureOutput(expTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes reciprocalAttrs;
    reciprocalAttrs.set_name("reciprocal").set_mode(hipdnn_frontend::PointwiseMode::RECIPROCAL);
    auto reciprocalTensor = graph->pointwise(expTensor, reciprocalAttrs);
    tensors.configureOutput(reciprocalTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);

    // exp(-x) / exp(x) = exp(-2x)
    hipdnn_frontend::graph::PointwiseAttributes divideAttrs;
    divideAttrs.set_name("divide").set_mode(hipdnn_frontend::PointwiseMode::DIV);
    auto quotientTensor = graph->pointwise(reciprocalTensor, expTensor, divideAttrs);
    tensors.configureOutput(quotientTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes compareAttrs;
    compareAttrs.set_name("positive").set_mode(hipdnn_frontend::PointwiseMode::CMP_GT);
    auto positiveMask = graph->pointwise(xTensor, zeroTensor, compareAttrs);
    tensors.configureOutput(positiveMask, hipdnn_frontend::DataType::BOOLEAN, tensorDims);

    hipdnn_frontend::graph::PointwiseAttributes notAttrs;
    notAttrs.set_name("non_positive").set_mode(hipdnn_frontend::PointwiseMode::LOGICAL_NOT);
    auto nonPositiveTensor = graph->pointwise(positiveMask, notAttrs);
    tensors.configureOutput(nonPositiveTensor, hipdnn_frontend::DataType::BOOLEAN, tensorDims);

    // exp(-2x) where x <= 0, exp(x) otherwise
    hipdnn_frontend::graph::PointwiseAttributes selectAttrs;
    selectAttrs.set_name("select").set_mode(hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto selectedTensor
        = graph->pointwise(quotientTensor, expTensor, nonPositiveTensor, selectAttrs);
    tensors.configureOutput(selectedTensor, hipdnn_frontend::DataType::FLOAT, tensorDims);
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
