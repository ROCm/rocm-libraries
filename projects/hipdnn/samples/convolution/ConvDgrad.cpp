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

    std::cout << "Running convolution backward data graph " << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    // Input (dx)
    auto n = config.dims.size() > 0 ? config.dims[0] : 16;
    auto c = config.dims.size() > 1 ? config.dims[1] : 16;
    auto h = config.dims.size() > 2 ? config.dims[2] : 16;
    auto w = config.dims.size() > 3 ? config.dims[3] : 16;

    // Filter + channels
    auto k = config.filter.size() > 0 ? config.filter[0] : 16;
    auto r = config.filter.size() > 0 ? config.filter[0] : 3;
    auto s = config.filter.size() > 1 ? config.filter[1] : 3;

    // Stride
    auto u = config.stride.size() > 0 ? config.stride[0] : 1;
    auto v = config.stride.size() > 1 ? config.stride[1] : 1;

    // Padding
    auto padH = config.padding.size() > 0 ? config.padding[0] : 1;
    auto padW = config.padding.size() > 1 ? config.padding[1] : 1;

    // Dilation
    auto dilH = config.dilation.size() > 0 ? config.dilation[0] : 1;
    auto dilW = config.dilation.size() > 1 ? config.dilation[1] : 1;

    // Output (dy shape)
    const int64_t outH = (h + 2 * padH - dilH * (r - 1) - 1) / u + 1;
    const int64_t outW = (w + 2 * padW - dilW * (s - 1) - 1) / v + 1;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    if(config.engine_id != -1)
    {
        graph->set_preferred_engine_id_ext(config.engine_id);
    }

    auto dyAttr = createTensor({n, k, outH, outW}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvDgradAttributes convAttributes;
    convAttributes.set_name("conv_backward_data_node");
    convAttributes.set_pre_padding({PAD_H, PAD_W});
    convAttributes.set_post_padding({PAD_H, PAD_W});
    convAttributes.set_stride({U, V});
    convAttributes.set_dilation({DIL_H, DIL_W});

    auto dxAttr = graph->conv_dgrad(dyAttr, wAttr, convAttributes);
    dxAttr->set_output(true);

    HIPDNN_FE_CHECK(graph->build(handle));
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

    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

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
            dxRefTensor, wTensor, dyTensor, {U, V}, {DIL_H, DIL_W}, {PAD_H, PAD_W});

        auto absoluteTolerance = hipdnn_test_sdk::utilities::conv::
            calculateConvDgradTolerance<InputType, InputType, float>(
                0.0, 1.0, 0.0, 1.0, wAttr->get_dim());

        constexpr float relativeTolerance = 0.01f;

        auto dxValidator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(
            absoluteTolerance, RELATIVE_TOLERANCE);

        std::cout << "CPU reference validation:\n";
        const bool dxValid
            = hipdnn_test_sdk::utilities::validateAndReport<InputType>(std::cout,
                                                                       "dx",
                                                                       dxValidator,
                                                                       dxRefTensor,
                                                                       dxTensor,
                                                                       absoluteTolerance,
                                                                       RELATIVE_TOLERANCE);

        validationPassed = dxValid;
    }

    std::cout << "Convolution backward data graph execution complete for " << inputType << ".\n\n";

    return validationPassed;
}

int main(int argc, char* argv[])
{
    try
    {
        auto config = parseCommandLineArgs(argc, argv);

        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

    bool allPassed = run(SampleRunner{*handle, config}, config);

        if(allPassed)
        {
            std::cout << "All convolution backward data runs completed successfully.\n";
            return 0;
        }
        std::cout << "One or more convolution backward data runs failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
