// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_frontend.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/Workspace.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk;

template <typename InputType, typename IntermediateType>
void SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running convolution forward graph " << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    constexpr int64_t n = 16; // Batch size

    // Input
    constexpr int64_t c = 16; // Number of channels
    constexpr int64_t h = 16; // Height
    constexpr int64_t w = 16; // Width

    // Filter
    constexpr int64_t k = 16; // Number of filters
    constexpr int64_t r = 3; // Height
    constexpr int64_t s = 3; // Width
    constexpr int64_t u = 1; // Height stride
    constexpr int64_t v = 1; // Width stride
    constexpr int64_t padH = 1; // Height padding
    constexpr int64_t padW = 1; // Width padding
    constexpr int64_t dilH = 1; // Height dilation
    constexpr int64_t dilW = 1; // Width dilation

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(inputType);

    auto xAttr = createTensor({n, c, h, w}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvFpropAttributes convAttributes;
    convAttributes.set_name("conv_forward_node");
    convAttributes.set_padding({padH, padW});
    convAttributes.set_stride({u, v});
    convAttributes.set_dilation({dilH, dilW});

    auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttributes);
    yAttr->set_output(true);

    HIPDNN_FE_CHECK(graph->validate());
    std::cout << "Graph validation successful.\n";

    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));
    std::cout << "Operation graph build successful.\n";

    HIPDNN_FE_CHECK(graph->create_execution_plans());
    std::cout << "Execution plans created successfully.\n";

    HIPDNN_FE_CHECK(graph->check_support());
    std::cout << "Graph support check successful.\n";

    HIPDNN_FE_CHECK(graph->build_plans());
    std::cout << "Plans build successful.\n";

    utilities::Tensor<InputType> xTensor(xAttr->get_dim(), layout);
    utilities::Tensor<InputType> wTensor(wAttr->get_dim(), layout);
    utilities::Tensor<InputType> yTensor(yAttr->get_dim(), layout);

    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    wTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    yTensor.fillWithValue(static_cast<InputType>(0.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
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

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> yRefTensor(yAttr->get_dim(), layout);

        test_utilities::CpuFpReferenceConvolutionImpl<InputType, float>::convFwdInference(
            xTensor, wTensor, yRefTensor, {u, v}, {dilH, dilW}, {padH, padW});

        auto tolerance = test_utilities::conv::getToleranceFwd<InputType>();

        auto yValidator = test_utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        bool yValid = yValidator.allClose(yRefTensor, yTensor);

        std::cout << "CPU reference validation:\n";
        std::cout << "  y: " << (yValid ? "successful" : "failed") << "\n";
    }

    std::cout << "Convolution forward graph execution complete for " << inputType << ".\n\n";
}

int main(int argc, char* argv[])
{
    auto config = parseCommandLineArgs(argc, argv);

    initializeFrontendLogging();

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    run(SampleRunner{handle, config});

    HIPDNN_CHECK(hipdnnDestroy(handle));
    std::cout << "All convolution forward runs completed.\n";
    return 0;
}
