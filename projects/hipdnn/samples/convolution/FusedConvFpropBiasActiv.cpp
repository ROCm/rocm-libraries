// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
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

    // Input
    const int64_t n = config.dims.size() > 0 ? config.dims[0] : 16; // Batch size
    const int64_t c = config.dims.size() > 1 ? config.dims[1] : 16; // Channels
    const int64_t h = config.dims.size() > 2 ? config.dims[2] : 16; // Height
    const int64_t w = config.dims.size() > 3 ? config.dims[3] : 16; // Width

    // Filter
    const int64_t k = config.filter.size() > 0 ? config.filter[0] : 16; // Output channels
    const int64_t r = config.filter.size() > 1 ? config.filter[1] : 3; // Filter height
    const int64_t s = config.filter.size() > 2 ? config.filter[2] : 3; // Filter width

    // Stride
    const int64_t u = config.stride.size() > 0 ? config.stride[0] : 1;
    const int64_t v = config.stride.size() > 1 ? config.stride[1] : 1;

    // Padding
    const int64_t padH = config.padding.size() > 0 ? config.padding[0] : 1;
    const int64_t padW = config.padding.size() > 1 ? config.padding[1] : 1;

    // Dilation
    const int64_t dilH = config.dilation.size() > 0 ? config.dilation[0] : 1;
    const int64_t dilW = config.dilation.size() > 1 ? config.dilation[1] : 1;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(
            hipdnn_frontend::DataType::FLOAT); // MIOpen requires FLOAT compute type

    setPreferredEngine(graph, config);

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
    const int64_t outH = (h + 2 * padH - dilH * (r - 1) - 1) / u + 1;
    const int64_t outW = (w + 2 * padW - dilW * (s - 1) - 1) / v + 1;

    convOutAttr->set_dim({n, k, outH, outW});
    convOutAttr->set_stride(utilities::generateStrides({n, k, outH, outW}, layout.strideOrder));

    // Create bias tensor with per-channel shape (1, k, 1, 1) derived from output dims
    const auto biasDims = utilities::getDerivedShape(convOutAttr->get_dim());
    auto biasAttr = createTensor(biasDims, inputType, layout);

    // Add bias using pointwise ADD operation
    graph::PointwiseAttributes biasAddAttributes;
    biasAddAttributes.set_name("bias_add_node");
    biasAddAttributes.set_mode(hipdnn_frontend::PointwiseMode::ADD);
    biasAddAttributes.set_compute_data_type(inputType);

    auto biasOutAttr = graph->pointwise(convOutAttr, biasAttr, biasAddAttributes);

    // Apply ReLU activation
    graph::PointwiseAttributes activationAttributes;
    activationAttributes.set_name("activation_node");
    activationAttributes.set_mode(hipdnn_frontend::PointwiseMode::RELU_FWD);

    auto yAttr = graph->pointwise(biasOutAttr, activationAttributes);
    yAttr->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));

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

    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

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
        const bool outValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "output", outValidator, yRefTensor, yTensor, tolerance, tolerance);

        validationPassed = outValid;
    }

    std::cout << "Fused Convolution fprop + Bias + Activ graph execution complete for " << inputType
              << ".\n\n";

    return validationPassed;
}

int main(int argc, char* argv[])
{
    try
    {
        RETURN_SUCCESS_IF_NO_DEVICE();

        auto config = parseCommandLineArgs(argc, argv);

        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

        const bool allPassed = run(SampleRunner{*handle, config});

        if(allPassed)
        {
            std::cout << "All fused Conv fwd + Bias + Activation runs completed successfully.\n";
            return 0;
        }
        std::cout << "One or more fused Conv fwd + Bias + Activation runs failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
