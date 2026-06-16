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
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

    if(config.engine_id != -1)
    {
        graph->set_preferred_engine_id_ext(config.engine_id);
    }
    else if(!config.engine_name.empty())
    {
        if(!hipdnn_data_sdk::utilities::isEngineNameRegistered(config.engine_name))
        {
            std::cerr << "Warning: Unknown engine name: " << config.engine_name << "\n";
        }

        graph->set_preferred_engine_id_ext(
            hipdnn_data_sdk::utilities::engineNameToId(config.engine_name));
    }

    auto xAttr = createTensor({n, c, h, w}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvFpropAttributes convAttributes;
    convAttributes.set_name("conv_fprop_node");
    convAttributes.set_padding({PAD_H, PAD_W});
    convAttributes.set_stride({U, V});
    convAttributes.set_dilation({DIL_H, DIL_W});

    auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttributes);
    yAttr->set_output(false);

    graph::PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);
    // Set values to clamp between 0.2 - 0.7
    pointwiseAttributes.set_relu_lower_clip(0.2f);
    pointwiseAttributes.set_relu_upper_clip(0.7f);

    auto pointwiseOutAttr = graph->pointwise(yAttr, pointwiseAttributes);
    pointwiseOutAttr->set_output(true);

    HIPDNN_FE_CHECK(graph->build(handle));
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

    std::cout << "First 10 pointwise out values: ";
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
