// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

template <typename InputType, typename ComputeType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    using OutputType = InputType;

    auto inputType = getDataTypeEnumFromType<InputType>();
    auto computeType = getDataTypeEnumFromType<ComputeType>();

    std::cout << "Running batch normalization inference + ReLU activation graph " << inputType
              << " [" << layout << "]" << (config.cpuValidation ? " (with CPU validation)" : "")
              << "...\n";

    const int64_t n = 16; // BATCH SIZE
    const int64_t c = 16; // CHANNELS (FEATURES)
    const int64_t h = 16; // HEIGHT (SPATIAL DIMENSION)
    const int64_t w = 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(computeType)
        .set_compute_data_type(computeType);

    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, computeType);
    auto bias = createTensor({1, c, 1, 1}, computeType);
    auto mean = createTensor({1, c, 1, 1}, computeType);
    // invVariance = 1/sqrt(variance + epsilon), epsilon is already baked in
    auto invVariance = createTensor({1, c, 1, 1}, computeType);

    // Step 1: Batchnorm Inference (using pre-computed invVariance, so epsilon not needed)
    auto bnAttributes = graph::BatchnormInferenceAttributes();
    bnAttributes.set_name("bn_inference_node");

    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);

    // Step 2: Pointwise ReLU Activation
    auto pwAttributes = graph::PointwiseAttributes();
    pwAttributes.set_name("activation_node");
    pwAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto activatedY = graph->pointwise(y, pwAttributes);
    activatedY->set_name("activated_y");
    activatedY->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful.\n";

    // Allocate tensors
    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<ComputeType> scaleTensor(scale->get_dim());
    utilities::Tensor<ComputeType> biasTensor(bias->get_dim());
    utilities::Tensor<ComputeType> meanTensor(mean->get_dim());
    utilities::Tensor<ComputeType> invVarianceTensor(invVariance->get_dim());
    utilities::Tensor<OutputType> activatedYTensor(activatedY->get_dim(), layout);

    // Initialize tensors
    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    scaleTensor.fillWithRandomValues(static_cast<ComputeType>(0.0f),
                                     static_cast<ComputeType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<ComputeType>(0.0f), static_cast<ComputeType>(1.0f));
    meanTensor.fillWithRandomValues(static_cast<ComputeType>(0.0f), static_cast<ComputeType>(1.0f));
    invVarianceTensor.fillWithRandomValues(static_cast<ComputeType>(0.1f),
                                           static_cast<ComputeType>(1.0f));

    // Build variant pack
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[mean->get_uid()] = meanTensor.memory().deviceData();
    variantPack[invVariance->get_uid()] = invVarianceTensor.memory().deviceData();
    variantPack[activatedY->get_uid()] = activatedYTensor.memory().deviceData();

    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    activatedYTensor.memory().markDeviceModified();
    auto activatedYHostPtr = activatedYTensor.memory().hostData();

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation using CpuReferenceGraphExecutor...\n";

        // Create reference tensor
        utilities::Tensor<OutputType> activatedYRefTensor(activatedY->get_dim(), layout);

        // Build variant pack for CPU execution (using host pointers)
        const std::unordered_map<int64_t, void*> cpuVariantPack{
            {x->get_uid(), xTensor.memory().hostData()},
            {scale->get_uid(), scaleTensor.memory().hostData()},
            {bias->get_uid(), biasTensor.memory().hostData()},
            {mean->get_uid(), meanTensor.memory().hostData()},
            {invVariance->get_uid(), invVarianceTensor.memory().hostData()},
            {activatedY->get_uid(), activatedYRefTensor.memory().hostData()}};

        // Execute on CPU using graph executor
        auto [serializedGraph, serErr] = graph->to_binary();
        if(serErr.is_bad())
        {
            std::cerr << "Failed to serialize graph: " << serErr.get_message() << '\n';
            return false;
        }
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
        cpuExecutor.execute(serializedGraph.data(), serializedGraph.size(), cpuVariantPack);

        auto tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceInference<OutputType>();
        auto yValidator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<OutputType>(
            tolerance, tolerance);

        std::cout << "CPU reference validation:\n";
        const bool yValid
            = hipdnn_test_sdk::utilities::validateAndReport<OutputType>(std::cout,
                                                                        "activated_y",
                                                                        yValidator,
                                                                        activatedYRefTensor,
                                                                        activatedYTensor,
                                                                        tolerance,
                                                                        tolerance);

        validationPassed = yValid;
    }

    std::cout << "First 10 activated_y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(activatedYHostPtr[i]) << " ";
    }

    std::cout << "\nBatch normalization inference + activation graph execution complete for "
              << inputType << ".\n\n";
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
            std::cout
                << "All batch normalization inference + activation runs completed successfully.\n";
            return 0;
        }
        std::cout
            << "One or more batch normalization inference + activation runs failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
