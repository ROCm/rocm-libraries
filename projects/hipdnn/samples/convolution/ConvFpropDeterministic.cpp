// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;
using namespace hipdnn_data_sdk::utilities;

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running deterministic convolution fprop graph " << inputType << " [" << layout
              << "]" << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    auto n = config.dims.size() > 0 ? config.dims[0] : 16;
    auto c = config.dims.size() > 1 ? config.dims[1] : 16;
    auto h = config.dims.size() > 2 ? config.dims[2] : 16;
    auto w = config.dims.size() > 3 ? config.dims[3] : 16;

    auto k = config.filter.size() > 0 ? config.filter[0] : 16;
    auto r = config.filter.size() > 0 ? config.filter[0] : 3;
    auto s = config.filter.size() > 1 ? config.filter[1] : 3;

    auto u = config.stride.size() > 0 ? config.stride[0] : 1;
    auto v = config.stride.size() > 1 ? config.stride[1] : 1;

    auto padH = config.padding.size() > 0 ? config.padding[0] : 1;
    auto padW = config.padding.size() > 1 ? config.padding[1] : 1;

    auto dilH = config.dilation.size() > 0 ? config.dilation[0] : 1;
    auto dilW = config.dilation.size() > 1 ? config.dilation[1] : 1;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    graph->set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

    auto xAttr = createTensor({N, C, H, W}, inputType, layout);
    auto wAttr = createTensor({K, C, R, S}, inputType, layout);

    graph::ConvFpropAttributes convAttributes;
    convAttributes.set_name("conv_fprop_deterministic_node");
    convAttributes.set_padding({PAD_H, PAD_W});
    convAttributes.set_stride({U, V});
    convAttributes.set_dilation({DIL_H, DIL_W});

    auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttributes);
    yAttr->set_output(true);

    HIPDNN_FE_CHECK(graph->build(handle));

    std::cout << "Graph build successful (using deterministic engine).\n";

    utilities::Tensor<InputType> xTensor(xAttr->get_dim(), layout);
    utilities::Tensor<InputType> wTensor(wAttr->get_dim(), layout);
    utilities::Tensor<InputType> yTensor1(yAttr->get_dim(), layout);
    utilities::Tensor<InputType> yTensor2(yAttr->get_dim(), layout);

    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    wTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    yTensor1.fillWithValue(static_cast<InputType>(0.0f));
    yTensor2.fillWithValue(static_cast<InputType>(0.0f));

    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
        variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
        variantPack[yAttr->get_uid()] = yTensor1.memory().deviceData();

        HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));
        yTensor1.memory().markDeviceModified();
    }

    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
        variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
        variantPack[yAttr->get_uid()] = yTensor2.memory().deviceData();

        HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));
        yTensor2.memory().markDeviceModified();
    }

    auto y1HostPtr = yTensor1.memory().hostData();
    auto y2HostPtr = yTensor2.memory().hostData();

    std::cout << "First 10 y values (run 1): ";
    for(int i = 0; i < 10; ++i)
        std::cout << static_cast<float>(y1HostPtr[i]) << " ";
    std::cout << '\n';

    std::cout << "First 10 y values (run 2): ";
    for(int i = 0; i < 10; ++i)
        std::cout << static_cast<float>(y2HostPtr[i]) << " ";
    std::cout << '\n';

    bool determinismPassed = true;
    auto elementCount = getTensorElementCount(yAttr);

    for(int64_t i = 0; i < elementCount; ++i)
    {
        if(y1HostPtr[i] != y2HostPtr[i])
        {
            std::cerr << "Determinism check failed at index " << i << ": " << y1HostPtr[i]
                      << " != " << y2HostPtr[i] << '\n';
            determinismPassed = false;
            break;
        }
    }

    if(determinismPassed)
        std::cout << "Determinism check: PASSED (results are bit-exact)\n";
    else
        std::cout << "Determinism check: FAILED (results differ)\n";

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> yRefTensor(yAttr->get_dim(), layout);

        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
            xTensor, wTensor, yRefTensor, {U, V}, {DIL_H, DIL_W}, {PAD_H, PAD_W});

        auto tolerance = hipdnn_test_sdk::utilities::conv::getToleranceFwd<InputType>();

        auto yValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        bool yValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "y", yValidator, yRefTensor, yTensor1, tolerance, tolerance);

        validationPassed = yValid;
    }

    std::cout << "Deterministic convolution fprop graph execution complete for " << inputType
              << ".\n\n";

    return determinismPassed && validationPassed;
}

int main(int argc, char* argv[])
{
    try
    {
        auto config = parseCommandLineArgs(argc, argv);

        initializeFrontendLogging();

        hipdnnHandle_t handle = nullptr;
        HIPDNN_CHECK(hipdnnCreate(&handle));

    bool allPassed = run(SampleRunner{handle, config}, config);

        HIPDNN_CHECK(hipdnnDestroy(handle));

        if(allPassed)
        {
            std::cout << "All deterministic convolution fprop runs completed successfully.\n";
            return 0;
        }
        std::cout << "One or more deterministic convolution fprop runs failed.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
