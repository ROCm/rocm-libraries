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

    std::cout << "Running fused convolution fprop + activ graph " << inputType << " [" << layout
              << "]" << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    constexpr int64_t N = 16; // Batch size

    // Input
    constexpr int64_t C = 16; // Number of input (x) channels
    constexpr int64_t H = 16; // Height
    constexpr int64_t W = 16; // Width

    // Filter
    constexpr int64_t K = 16; // Number of output (y) channels
    constexpr int64_t R = 3; // Height
    constexpr int64_t S = 3; // Width
    constexpr int64_t U = 1; // Height stride
    constexpr int64_t V = 1; // Width stride
    constexpr int64_t PAD_H = 1; // Height padding
    constexpr int64_t PAD_W = 1; // Width padding
    constexpr int64_t DIL_H = 1; // Height dilation
    constexpr int64_t DIL_W = 1; // Width dilation

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

    auto xAttr = createTensor({N, C, H, W}, inputType, layout);
    auto wAttr = createTensor({K, C, R, S}, inputType, layout);

    graph::ConvFpropAttributes convAttributes;
    convAttributes.set_name("conv_fprop_node");
    convAttributes.set_padding({PAD_H, PAD_W});
    convAttributes.set_stride({U, V});
    convAttributes.set_dilation({DIL_H, DIL_W});

    auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttributes);
    yAttr->set_output(false);

    graph::PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_mode(hipdnn_frontend::PointwiseMode::RELU_FWD);
    // Set values to clamp between 0.2 - 0.7
    pointwiseAttributes.set_relu_lower_clip(0.2f);
    pointwiseAttributes.set_relu_upper_clip(0.7f);
    auto pointwiseOutAttr = graph->pointwise(yAttr, pointwiseAttributes);
    pointwiseOutAttr->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful.\n";

    utilities::Tensor<InputType> xTensor(xAttr->get_dim(), layout);
    utilities::Tensor<InputType> wTensor(wAttr->get_dim(), layout);
    utilities::Tensor<InputType> pointwiseOutTensor(pointwiseOutAttr->get_dim(), layout);

    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    wTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    pointwiseOutTensor.fillWithValue(static_cast<InputType>(0.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
    variantPack[pointwiseOutAttr->get_uid()] = pointwiseOutTensor.memory().deviceData();

    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    pointwiseOutTensor.memory().markDeviceModified();

    auto pointwiseOutHostPtr = pointwiseOutTensor.memory().hostData();

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(pointwiseOutHostPtr[i]) << " ";
    }
    std::cout << '\n';

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> yRefTensor(yAttr->get_dim(), layout);
        utilities::Tensor<InputType> pointwiseOutRefTensor(pointwiseOutAttr->get_dim(), layout);

        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
            xTensor, wTensor, yRefTensor, {U, V}, {DIL_H, DIL_W}, {PAD_H, PAD_W});

        hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl<InputType>::pointwiseCompute(
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            pointwiseOutRefTensor,
            yRefTensor,
            pointwiseAttributes.get_relu_lower_clip().value(),
            pointwiseAttributes.get_relu_upper_clip().value(),
            0.0f);

        auto tolerance = hipdnn_test_sdk::utilities::conv::getToleranceFwd<InputType>();

        auto outValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        std::cout << "CPU reference validation:\n";
        const bool outValid
            = hipdnn_test_sdk::utilities::validateAndReport<InputType>(std::cout,
                                                                       "pointwise out",
                                                                       outValidator,
                                                                       pointwiseOutRefTensor,
                                                                       pointwiseOutTensor,
                                                                       tolerance,
                                                                       tolerance);

        validationPassed = outValid;
    }

    std::cout << "Fused Convolution fprop + Activ graph execution complete for " << inputType
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
            std::cout << "All fused Conv fwd + Activation runs completed successfully.\n";
            return 0;
        }
        std::cout << "One or more fused Conv fwd + Activation runs failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
