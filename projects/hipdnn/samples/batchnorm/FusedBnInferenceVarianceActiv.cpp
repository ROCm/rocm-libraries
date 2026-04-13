// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
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

    std::cout << "Running batch normalization inference with variance + ReLU activation graph "
              << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    int64_t n = 16; // BATCH SIZE
    int64_t c = 16; // CHANNELS (FEATURES)
    int64_t h = 16; // HEIGHT (SPATIAL DIMENSION)
    int64_t w = 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(computeType)
        .set_compute_data_type(computeType);

    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, computeType);
    auto bias = createTensor({1, c, 1, 1}, computeType);
    auto mean = createTensor({1, c, 1, 1}, computeType);
    auto variance = createTensor({1, c, 1, 1}, computeType);

    // Epsilon is a tensor for the variance extension
    auto epsilon = std::make_shared<graph::TensorAttributes>();
    epsilon->set_value(utilities::BATCHNORM_DEFAULT_EPSILON);

    // Step 1: Batchnorm Inference with Variance
    auto bnAttributes = graph::BatchnormInferenceAttributesVarianceExt();
    bnAttributes.set_name("bn_inference_variance_ext_node");

    auto y = graph->batchnorm_inference_variance_ext(
        x, mean, variance, scale, bias, epsilon, bnAttributes);

    // Step 2: Pointwise ReLU Activation
    auto pwAttributes = graph::PointwiseAttributes();
    pwAttributes.set_name("activation_node");
    pwAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto activatedY = graph->pointwise(y, pwAttributes);
    activatedY->set_name("activated_y");
    activatedY->set_output(true);

    HIPDNN_FE_CHECK(graph->build(handle));
    std::cout << "Graph build successful.\n";

    // Allocate tensors
    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<ComputeType> scaleTensor(scale->get_dim());
    utilities::Tensor<ComputeType> biasTensor(bias->get_dim());
    utilities::Tensor<ComputeType> meanTensor(mean->get_dim());
    utilities::Tensor<ComputeType> varianceTensor(variance->get_dim());
    utilities::Tensor<OutputType> activatedYTensor(activatedY->get_dim(), layout);

    // Initialize tensors
    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    scaleTensor.fillWithRandomValues(static_cast<ComputeType>(0.0f),
                                     static_cast<ComputeType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<ComputeType>(0.0f), static_cast<ComputeType>(1.0f));
    meanTensor.fillWithRandomValues(static_cast<ComputeType>(0.0f), static_cast<ComputeType>(1.0f));
    varianceTensor.fillWithRandomValues(static_cast<ComputeType>(0.1f),
                                        static_cast<ComputeType>(1.0f));

    // Build variant pack
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[mean->get_uid()] = meanTensor.memory().deviceData();
    variantPack[variance->get_uid()] = varianceTensor.memory().deviceData();
    variantPack[activatedY->get_uid()] = activatedYTensor.memory().deviceData();

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

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
        std::unordered_map<int64_t, void*> cpuVariantPack;
        cpuVariantPack[x->get_uid()] = xTensor.memory().hostData();
        cpuVariantPack[scale->get_uid()] = scaleTensor.memory().hostData();
        cpuVariantPack[bias->get_uid()] = biasTensor.memory().hostData();
        cpuVariantPack[mean->get_uid()] = meanTensor.memory().hostData();
        cpuVariantPack[variance->get_uid()] = varianceTensor.memory().hostData();
        cpuVariantPack[activatedY->get_uid()] = activatedYRefTensor.memory().hostData();

        // Execute on CPU using graph executor
        auto [serializedGraph, serErr] = graph->to_binary();
        if(serErr.is_bad())
        {
            std::cerr << "Failed to serialize graph: " << serErr.get_message() << std::endl;
            return false;
        }
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
        cpuExecutor.execute(serializedGraph.data(), serializedGraph.size(), cpuVariantPack);

        auto tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceInferenceWithVariance<
            OutputType>();
        auto yValidator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<OutputType>(
            tolerance, tolerance);

        std::cout << "CPU reference validation:\n";
        bool yValid = hipdnn_test_sdk::utilities::validateAndReport<OutputType>(std::cout,
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

    std::cout << "\nBatch normalization inference with variance + activation graph execution "
                 "complete for "
              << inputType << ".\n\n";
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
        std::cout << "All batch normalization inference with variance + activation runs completed "
                     "successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "One or more batch normalization inference with variance + activation runs "
                     "failed validation.\n";
        return 1;
    }
}
