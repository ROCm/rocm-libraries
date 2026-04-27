// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/CpuReferencePointwise.hpp>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running fused convolution fprop + bias + activ graph " << inputType << " ["
              << layout << "]" << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    constexpr int64_t n = 16; // Batch size

    // Input
    constexpr int64_t c = 16; // Number of input (x) channels
    constexpr int64_t h = 16; // Height
    constexpr int64_t w = 16; // Width

    // Filter
    constexpr int64_t k = 16; // Number of output (y) channels
    constexpr int64_t r = 3; // Height
    constexpr int64_t s = 3; // Width
    constexpr int64_t u = 1; // Height stride
    constexpr int64_t v = 1; // Width stride
    constexpr int64_t padH = 1; // Height padding
    constexpr int64_t padW = 1; // Width padding
    constexpr int64_t dilH = 1; // Height dilation
    constexpr int64_t dilW = 1; // Width dilation

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(
            hipdnn_frontend::DataType::FLOAT); // MIOpen requires FLOAT compute type

    auto xAttr = createTensor({n, c, h, w}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvFpropAttributes convAttributes;
    convAttributes.set_name("conv_fprop_node");
    convAttributes.set_padding({padH, padW});
    convAttributes.set_stride({u, v});
    convAttributes.set_dilation({dilH, dilW});

    auto convOutAttr = graph->conv_fprop(xAttr, wAttr, convAttributes);
    // Explicitly set output dimensions and strides so we can derive the bias shape.
    // The output dimensions aren't automatically populated until after graph->build_operation_graph(),
    // but we need them now to create the bias tensor with the correct per-channel shape.
    convOutAttr->set_dim({n, k, h, w});
    convOutAttr->set_stride(utilities::generateStrides({n, k, h, w}, layout.strideOrder));

    // Create bias tensor with per-channel shape (1, k, 1, 1) derived from output dims
    const auto biasDims = utilities::getDerivedShape(convOutAttr->get_dim());
    auto biasAttr = createTensor(biasDims, inputType, layout);

    // Add bias using pointwise ADD operation
    graph::PointwiseAttributes biasAddAttributes;
    biasAddAttributes.set_name("bias_add_node");
    biasAddAttributes.set_mode(hipdnn_frontend::PointwiseMode::ADD);
    biasAddAttributes.set_compute_data_type(inputType); // MIOpen requires FLOAT compute type

    auto biasOutAttr = graph->pointwise(convOutAttr, biasAttr, biasAddAttributes);

    // Apply ReLU activation
    graph::PointwiseAttributes activationAttributes;
    activationAttributes.set_name("activation_node");
    activationAttributes.set_mode(hipdnn_frontend::PointwiseMode::RELU_FWD);

    auto yAttr = graph->pointwise(biasOutAttr, activationAttributes);
    yAttr->set_output(true);

    HIPDNN_FE_CHECK(graph->build(handle));
    std::cout << "Graph build successful.\n";

    utilities::Tensor<InputType> xTensor(xAttr->get_dim(), layout);
    utilities::Tensor<InputType> wTensor(wAttr->get_dim(), layout);
    utilities::Tensor<InputType> biasTensor(biasDims, layout);
    utilities::Tensor<InputType> yTensor(yAttr->get_dim(), layout);

    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    wTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    yTensor.fillWithValue(static_cast<InputType>(0.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
    variantPack[biasAttr->get_uid()] = biasTensor.memory().deviceData();
    variantPack[yAttr->get_uid()] = yTensor.memory().deviceData();

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    yTensor.memory().markDeviceModified();

    auto yHostPtr = yTensor.memory().hostData();

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(yHostPtr[i]) << " ";
    }
    std::cout << '\n';

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        // Step 1: Compute convolution output
        utilities::Tensor<InputType> convRefTensor(convOutAttr->get_dim(), layout);
        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
            xTensor, wTensor, convRefTensor, {u, v}, {dilH, dilW}, {padH, padW});

        // Step 2: Add bias using pointwise ADD with broadcasting
        utilities::Tensor<InputType> biasRefTensor(convOutAttr->get_dim(), layout);
        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<InputType>::pointwiseCompute(
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
            biasRefTensor,
            convRefTensor,
            biasTensor);

        // Step 3: Apply ReLU activation
        utilities::Tensor<InputType> yRefTensor(yAttr->get_dim(), layout);
        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<InputType>::pointwiseCompute(
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            yRefTensor,
            biasRefTensor);

        auto tolerance = hipdnn_test_sdk::utilities::conv::getToleranceFwd<InputType>();

        auto outValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        std::cout << "CPU reference validation:\n";
        bool outValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "output", outValidator, yRefTensor, yTensor, tolerance, tolerance);

        validationPassed = outValid;
    }

    std::cout << "Fused Convolution fprop + Bias + Activ graph execution complete for " << inputType
              << ".\n\n";
    return validationPassed;
}

int main(int argc, char* argv[])
{
    auto config = parseCommandLineArgs(argc, argv);

    auto [handle, handleError] = createHipdnnHandle();
    HIPDNN_FE_CHECK(handleError);

    bool allPassed = run(SampleRunner{*handle, config});

    if(allPassed)
    {
        std::cout << "All fused Conv fwd + Bias + Activation runs completed successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "One or more fused Conv fwd + Bias + Activation runs failed validation.\n";
        return 1;
    }
}
