// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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
    graph->set_io_data_type(inputType).set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    // Set the preferred engine to deterministic for bit-reproducible results
    graph->set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

    auto xAttr = createTensor({n, c, h, w}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvFpropAttributes convAttributes;
    convAttributes.set_name("conv_fprop_deterministic_node");
    convAttributes.set_padding({padH, padW});
    convAttributes.set_stride({u, v});
    convAttributes.set_dilation({dilH, dilW});

    auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttributes);
    yAttr->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful (using deterministic engine).\n";

    utilities::Tensor<InputType> xTensor(xAttr->get_dim(), layout);
    utilities::Tensor<InputType> wTensor(wAttr->get_dim(), layout);
    utilities::Tensor<InputType> yTensor1(yAttr->get_dim(), layout);
    utilities::Tensor<InputType> yTensor2(yAttr->get_dim(), layout);

    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    wTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    yTensor1.fillWithValue(static_cast<InputType>(0.0f));
    yTensor2.fillWithValue(static_cast<InputType>(0.0f));

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    // First execution
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
        variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
        variantPack[yAttr->get_uid()] = yTensor1.memory().deviceData();

        HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));
        yTensor1.memory().markDeviceModified();
    }

    // Second execution with same inputs
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
    {
        std::cout << static_cast<float>(y1HostPtr[i]) << " ";
    }
    std::cout << '\n';

    std::cout << "First 10 y values (run 2): ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(y2HostPtr[i]) << " ";
    }
    std::cout << '\n';

    // Verify determinism - results should be bit-exact
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
    {
        std::cout << "Determinism check: PASSED (results are bit-exact)\n";
    }
    else
    {
        std::cout << "Determinism check: FAILED (results differ)\n";
    }

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> yRefTensor(yAttr->get_dim(), layout);

        hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
            xTensor, wTensor, yRefTensor, {u, v}, {dilH, dilW}, {padH, padW});

        auto tolerance = hipdnn_test_sdk::utilities::conv::getToleranceFwd<InputType>();

        auto yValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        std::cout << "CPU reference validation:\n";
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
    auto config = parseCommandLineArgs(argc, argv);

    initializeFrontendLogging();

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    bool allPassed = run(SampleRunner{handle, config});

    HIPDNN_CHECK(hipdnnDestroy(handle));

    if(allPassed)
    {
        std::cout << "All deterministic convolution fprop runs completed successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "One or more deterministic convolution fprop runs failed.\n";
        return 1;
    }
}
