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
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running convolution backward data graph " << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    constexpr int64_t n = 16; // Batch size

    // Input (dx dimensions)
    constexpr int64_t c = 16; // Number of dx channels
    constexpr int64_t h = 16; // Height
    constexpr int64_t w = 16; // Width

    // Filter
    constexpr int64_t k = 16; // Number of dy channels
    constexpr int64_t r = 3; // Height
    constexpr int64_t s = 3; // Width
    constexpr int64_t u = 1; // Height stride
    constexpr int64_t v = 1; // Width stride
    constexpr int64_t padH = 1; // Height padding
    constexpr int64_t padW = 1; // Width padding
    constexpr int64_t dilH = 1; // Height dilation
    constexpr int64_t dilW = 1; // Width dilation

    // Output (dy dimensions) - computed based on input and conv parameters
    const int64_t outH = (h + 2 * padH - dilH * (r - 1) - 1) / u + 1;
    const int64_t outW = (w + 2 * padW - dilW * (s - 1) - 1) / v + 1;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto dyAttr = createTensor({n, k, outH, outW}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvDgradAttributes convAttributes;
    convAttributes.set_name("conv_backward_data_node");
    convAttributes.set_pre_padding({padH, padW});
    convAttributes.set_post_padding({padH, padW});
    convAttributes.set_stride({u, v});
    convAttributes.set_dilation({dilH, dilW});

    auto dxAttr = graph->conv_dgrad(dyAttr, wAttr, convAttributes);
    dxAttr->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful.\n";

    utilities::Tensor<InputType> dyTensor(dyAttr->get_dim(), layout);
    utilities::Tensor<InputType> wTensor(wAttr->get_dim(), layout);
    utilities::Tensor<InputType> dxTensor(dxAttr->get_dim(), layout);

    dyTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    wTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    dxTensor.fillWithValue(static_cast<InputType>(0.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[dyAttr->get_uid()] = dyTensor.memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
    variantPack[dxAttr->get_uid()] = dxTensor.memory().deviceData();

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    dxTensor.memory().markDeviceModified();

    auto dxHostPtr = dxTensor.memory().hostData();

    std::cout << "First 10 dx values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(dxHostPtr[i]) << " ";
    }
    std::cout << '\n';

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> dxRefTensor(dxAttr->get_dim(), layout);

        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::dgrad(
            dxRefTensor, wTensor, dyTensor, {u, v}, {dilH, dilW}, {padH, padW});

        auto absoluteTolerance = hipdnn_test_sdk::utilities::conv::
            calculateConvDgradTolerance<InputType, InputType, float>(
                0.0, 1.0, 0.0, 1.0, wAttr->get_dim());
        constexpr float relativeTolerance = 0.01f;

        auto dxValidator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(
            absoluteTolerance, relativeTolerance);

        std::cout << "CPU reference validation:\n";
        bool dxValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(std::cout,
                                                                                "dx",
                                                                                dxValidator,
                                                                                dxRefTensor,
                                                                                dxTensor,
                                                                                absoluteTolerance,
                                                                                relativeTolerance);

        validationPassed = dxValid;
    }

    std::cout << "Convolution backward data graph execution complete for " << inputType << ".\n\n";
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
        std::cout << "All convolution backward data runs completed successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "One or more convolution backward data runs failed validation.\n";
        return 1;
    }
}
