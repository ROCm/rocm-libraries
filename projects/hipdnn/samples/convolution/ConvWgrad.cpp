// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running convolution backward weights graph " << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    constexpr int64_t N = 16; // Batch size

    // Input (x dimensions)
    constexpr int64_t C = 16; // Number of input channels
    constexpr int64_t H = 16; // Height
    constexpr int64_t W = 16; // Width

    // Filter (dw dimensions)
    constexpr int64_t K = 16; // Number of output channels
    constexpr int64_t R = 3; // Filter height
    constexpr int64_t S = 3; // Filter width
    constexpr int64_t U = 1; // Height stride
    constexpr int64_t V = 1; // Width stride
    constexpr int64_t PAD_H = 1; // Height padding
    constexpr int64_t PAD_W = 1; // Width padding
    constexpr int64_t DIL_H = 1; // Height dilation
    constexpr int64_t DIL_W = 1; // Width dilation

    // Output gradient (dy dimensions) - computed based on input and conv parameters
    const int64_t outH = (H + 2 * PAD_H - DIL_H * (R - 1) - 1) / U + 1;
    const int64_t outW = (W + 2 * PAD_W - DIL_W * (S - 1) - 1) / V + 1;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto dyAttr = createTensor({N, K, outH, outW}, inputType, layout);
    auto xAttr = createTensor({N, C, H, W}, inputType, layout);

    graph::ConvWgradAttributes convAttributes;
    convAttributes.set_name("conv_backward_weights_node");
    convAttributes.set_pre_padding({PAD_H, PAD_W});
    convAttributes.set_post_padding({PAD_H, PAD_W});
    convAttributes.set_stride({U, V});
    convAttributes.set_dilation({DIL_H, DIL_W});

    auto dwAttr = graph->conv_wgrad(dyAttr, xAttr, convAttributes);
    dwAttr->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful.\n";

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

    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    dwTensor.memory().markDeviceModified();

    auto dwHostPtr = dwTensor.memory().hostData();

    std::cout << "First 10 dw values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(dwHostPtr[i]) << " ";
    }
    std::cout << '\n';

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> dwRefTensor(dwAttr->get_dim(), layout);

        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::wgrad(
            xTensor, dwRefTensor, dyTensor, {U, V}, {DIL_H, DIL_W}, {PAD_H, PAD_W});

        auto absoluteTolerance = hipdnn_test_sdk::utilities::conv::
            calculateConvWrwTolerance<InputType, InputType, float>(
                0.0, 1.0, 0.0, 1.0, dyAttr->get_dim());
        constexpr float RELATIVE_TOLERANCE = 0.01f;

        auto dwValidator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(
            absoluteTolerance, RELATIVE_TOLERANCE);

        std::cout << "CPU reference validation:\n";
        const bool dwValid
            = hipdnn_test_sdk::utilities::validateAndReport<InputType>(std::cout,
                                                                       "dw",
                                                                       dwValidator,
                                                                       dwRefTensor,
                                                                       dwTensor,
                                                                       absoluteTolerance,
                                                                       RELATIVE_TOLERANCE);

        validationPassed = dwValid;
    }

    std::cout << "Convolution backward weights graph execution complete for " << inputType
              << ".\n\n";
    return validationPassed;
}

int main(int argc, char* argv[])
{
    try
    {
        auto config = parseCommandLineArgs(argc, argv);

        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

        const bool allPassed = run(SampleRunner{*handle, config});

        if(allPassed)
        {
            std::cout << "All convolution backward weights runs completed successfully.\n";
            return 0;
        }
        std::cout << "One or more convolution backward weights runs failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
