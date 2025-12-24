// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

template <typename InputType, typename IntermediateType>
void SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running convolution backward weights graph " << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    constexpr int64_t n = 16; // Batch size

    // Input (x dimensions)
    constexpr int64_t c = 16; // Number of input channels
    constexpr int64_t h = 16; // Height
    constexpr int64_t w = 16; // Width

    // Filter (dw dimensions)
    constexpr int64_t k = 16; // Number of output channels
    constexpr int64_t r = 3; // Filter height
    constexpr int64_t s = 3; // Filter width
    constexpr int64_t u = 1; // Height stride
    constexpr int64_t v = 1; // Width stride
    constexpr int64_t padH = 1; // Height padding
    constexpr int64_t padW = 1; // Width padding
    constexpr int64_t dilH = 1; // Height dilation
    constexpr int64_t dilW = 1; // Width dilation

    // Output gradient (dy dimensions) - computed based on input and conv parameters
    const int64_t outH = (h + 2 * padH - dilH * (r - 1) - 1) / u + 1;
    const int64_t outW = (w + 2 * padW - dilW * (s - 1) - 1) / v + 1;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto dyAttr = createTensor({n, k, outH, outW}, inputType, layout);
    auto xAttr = createTensor({n, c, h, w}, inputType, layout);

    graph::ConvWgradAttributes convAttributes;
    convAttributes.set_name("conv_backward_weights_node");
    convAttributes.set_pre_padding({padH, padW});
    convAttributes.set_post_padding({padH, padW});
    convAttributes.set_stride({u, v});
    convAttributes.set_dilation({dilH, dilW});

    auto dwAttr = graph->conv_wgrad(dyAttr, xAttr, convAttributes);
    dwAttr->set_output(true);

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

    utilities::Tensor<InputType> dyTensor(dyAttr->get_dim(), layout);
    utilities::Tensor<InputType> xTensor(xAttr->get_dim(), layout);
    utilities::Tensor<InputType> dwTensor(dwAttr->get_dim(), layout);

    dyTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    dwTensor.fillWithValue(static_cast<InputType>(0.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[dyAttr->get_uid()] = dyTensor.memory().deviceData();
    variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
    variantPack[dwAttr->get_uid()] = dwTensor.memory().deviceData();

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    dwTensor.memory().markDeviceModified();

    auto dwHostPtr = dwTensor.memory().hostData();

    std::cout << "First 10 dw values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(dwHostPtr[i]) << " ";
    }
    std::cout << '\n';

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> dwRefTensor(dwAttr->get_dim(), layout);

        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::wgrad(
            xTensor, dwRefTensor, dyTensor, {u, v}, {dilH, dilW}, {padH, padW});

        auto tolerance = hipdnn_test_sdk::utilities::conv::getToleranceWrw<InputType>();

        auto dwValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        bool dwValid = dwValidator.allClose(dwRefTensor, dwTensor);

        std::cout << "CPU reference validation:\n";
        std::cout << "  dw: " << (dwValid ? "successful" : "failed") << "\n";
    }

    std::cout << "Convolution backward weights graph execution complete for " << inputType
              << ".\n\n";
}

int main(int argc, char* argv[])
{
    auto config = parseCommandLineArgs(argc, argv);

    initializeFrontendLogging();

    auto backend = hipdnnBackend();
    hipdnnHandle_t handle;
    HIPDNN_CHECK(backend->create(&handle));

    run(SampleRunner{handle, config});

    HIPDNN_CHECK(backend->destroy(handle));
    std::cout << "All convolution backward weights runs completed.\n";
    return 0;
}
