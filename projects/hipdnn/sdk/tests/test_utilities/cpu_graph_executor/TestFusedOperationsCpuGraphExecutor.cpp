// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/BinaryOperationFunctors.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace ::testing;

class TestFusedOperationsCpuGraphExecutor : public ::testing::Test
{
};

TEST_F(TestFusedOperationsCpuGraphExecutor, ConvAddMulFusedGraph)
{
    auto tolerance = pointwise::getTolerance<float>();
    std::vector<int64_t> xDims = {1, 1, 2, 2};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 2, 2};

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    float addConstant = 5.0f;
    float multiplyConstant = 2.0f;
    unsigned int seed = getGlobalTestSeed();

    // DIRECT TENSOR MANAGEMENT - Expert Architecture
    // Graph execution tensors
    Tensor<float> xTensor(xDims, TensorLayout::NHWC);
    Tensor<float> wTensor(wDims, TensorLayout::NHWC);
    Tensor<float> addConstantTensor({1}, TensorLayout::NHWC);
    Tensor<float> multiplyConstantTensor({1}, TensorLayout::NHWC);
    Tensor<float> yTensor(yDims, TensorLayout::NHWC);

    // Reference computation tensors (separate copy for validation)
    Tensor<float> refXTensor(xDims, TensorLayout::NHWC);
    Tensor<float> refWTensor(wDims, TensorLayout::NHWC);
    Tensor<float> refYTensor(yDims, TensorLayout::NHWC);

    // Initialize tensors with consistent values
    xTensor.fillWithRandomValues(0.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(0.0f, 1.0f, seed);
    refXTensor.fillWithRandomValues(0.0f, 1.0f, seed);
    refWTensor.fillWithRandomValues(0.0f, 1.0f, seed);

    // Set constant values
    static_cast<float*>(addConstantTensor.memory().hostData())[0] = addConstant;
    static_cast<float*>(multiplyConstantTensor.memory().hostData())[0] = multiplyConstant;

    // BUILD 3-STEP FUSED GRAPH MANUALLY
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("ConvolutionAddMultiplyFusedTest");
    graph->set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    int64_t uid = 1;

    // Create tensor attributes for all tensors
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "X", hipdnn_frontend::DataType::FLOAT, xTensor);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto wAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "W", hipdnn_frontend::DataType::FLOAT, wTensor);
    wAttr.set_uid(uid++);
    auto wTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(wAttr));

    auto addConstantAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "AddConstant", hipdnn_frontend::DataType::FLOAT, addConstantTensor);
    addConstantAttr.set_uid(uid++);
    auto addConstantTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(addConstantAttr));

    auto multiplyConstantAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "MultiplyConstant", hipdnn_frontend::DataType::FLOAT, multiplyConstantTensor);
    multiplyConstantAttr.set_uid(uid++);
    auto multiplyConstantTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        std::move(multiplyConstantAttr));

    // Step 1: CONVOLUTION OPERATION
    hipdnn_frontend::graph::ConvFpropAttributes convAttrs;
    convAttrs.set_name("Convolution_fwd_inference");
    convAttrs.set_stride(strides);
    convAttrs.set_dilation(dilation);
    convAttrs.set_pre_padding(padding);
    convAttrs.set_post_padding(padding);
    convAttrs.set_convolution_mode(hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    auto convOutputTensorAttr = graph->conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

    if(!convOutputTensorAttr->has_uid())
    {
        convOutputTensorAttr->set_uid(uid++);
    }
    convOutputTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);
    convOutputTensorAttr->set_dim(yDims);
    convOutputTensorAttr->set_stride(generateStrides(yDims, TensorLayout::NHWC.strideOrder));

    // Step 2: POINTWISE ADD OPERATION (conv + 5.0)
    hipdnn_frontend::graph::PointwiseAttributes addAttrs;
    addAttrs.set_name("PointwiseAdd");
    addAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);

    auto addOutputTensorAttr
        = graph->pointwise(convOutputTensorAttr, addConstantTensorAttr, addAttrs);

    if(!addOutputTensorAttr->has_uid())
    {
        addOutputTensorAttr->set_uid(uid++);
    }
    addOutputTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);
    addOutputTensorAttr->set_dim(yDims);
    addOutputTensorAttr->set_stride(generateStrides(yDims, TensorLayout::NHWC.strideOrder));

    // Step 3: POINTWISE MULTIPLY OPERATION ((conv + 5.0) * 2.0)
    hipdnn_frontend::graph::PointwiseAttributes multiplyAttrs;
    multiplyAttrs.set_name("PointwiseMultiply");
    multiplyAttrs.set_mode(hipdnn_frontend::PointwiseMode::MUL);

    auto finalOutputTensorAttr
        = graph->pointwise(addOutputTensorAttr, multiplyConstantTensorAttr, multiplyAttrs);

    if(!finalOutputTensorAttr->has_uid())
    {
        finalOutputTensorAttr->set_uid(uid++);
    }
    finalOutputTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);
    finalOutputTensorAttr->set_dim(yDims);
    finalOutputTensorAttr->set_stride(generateStrides(yDims, TensorLayout::NHWC.strideOrder));

    // Create variant pack for input/output tensors (virtual tensors auto-allocated by executor)
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[xTensorAttr->get_uid()] = xTensor.memory().hostData();
    variantPack[wTensorAttr->get_uid()] = wTensor.memory().hostData();
    variantPack[addConstantTensorAttr->get_uid()] = addConstantTensor.memory().hostData();
    variantPack[multiplyConstantTensorAttr->get_uid()] = multiplyConstantTensor.memory().hostData();
    variantPack[finalOutputTensorAttr->get_uid()] = yTensor.memory().hostData();

    // Execute the graph using CPU graph executor
    CpuReferenceGraphExecutor graphExecutor;
    // Serialize the frontend graph to flatbuffer format
    auto serializedGraph = graph->buildFlatbufferOperationGraph();
    // Execute with correct 3-parameter signature
    graphExecutor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    // Compute reference result manually: (conv(X, W) + 5.0) * 2.0
    // Step 1: Perform convolution
    Tensor<float> tempConvOutput(yDims, TensorLayout::NHWC);
    CpuFpReferenceConvolutionImpl<float, float>::convFwdInference(
        refXTensor, refWTensor, tempConvOutput, strides, dilation, padding);

    // Step 2: Add constant (5.0) to convolution result
    Tensor<float> tempAddOutput(yDims, TensorLayout::NHWC);
    auto* convOutputData = static_cast<float*>(tempConvOutput.memory().hostData());
    auto* addOutputData = static_cast<float*>(tempAddOutput.memory().hostData());
    size_t elementCount = static_cast<size_t>(tempConvOutput.dims()[0])
                          * static_cast<size_t>(tempConvOutput.dims()[1])
                          * static_cast<size_t>(tempConvOutput.dims()[2])
                          * static_cast<size_t>(tempConvOutput.dims()[3]);

    for(size_t i = 0; i < elementCount; ++i)
    {
        addOutputData[i] = convOutputData[i] + addConstant;
    }

    // Step 3: Multiply by constant (2.0)
    auto* finalOutputData = static_cast<float*>(refYTensor.memory().hostData());
    for(size_t i = 0; i < elementCount; ++i)
    {
        finalOutputData[i] = addOutputData[i] * multiplyConstant;
    }

    // Validate results
    CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);

    EXPECT_TRUE(cpuRefOutputValidation.allClose(refYTensor, yTensor));
}
