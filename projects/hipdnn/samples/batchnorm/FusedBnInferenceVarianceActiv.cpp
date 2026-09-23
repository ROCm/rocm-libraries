// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
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
              << (config.cpuValidation ? " (with CPU validation)" : "")
              << (config.useRuntimePassByValue ? " [runtime-pass-by-value]" : "") << "...\n";

    // Input dimensions
    const int64_t n = config.dims.size() > 0 ? config.dims[0] : 16; // BATCH SIZE
    const int64_t c = config.dims.size() > 1 ? config.dims[1] : 16; // CHANNELS (FEATURES)
    const int64_t h = config.dims.size() > 2 ? config.dims[2] : 16; // HEIGHT (SPATIAL DIMENSION)
    const int64_t w = config.dims.size() > 3 ? config.dims[3] : 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(computeType)
        .set_compute_data_type(computeType);

    setPreferredEngine(graph, config);

    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, computeType);
    auto bias = createTensor({1, c, 1, 1}, computeType);
    auto mean = createTensor({1, c, 1, 1}, computeType);
    auto variance = createTensor({1, c, 1, 1}, computeType);

    // Epsilon is a pass-by-value scalar, not a device buffer.
    // Compile-time constant: value is baked into the op-graph at build() and requires no
    // variant-pack entry at execute(). Works with any plugin version.
    // Runtime pass-by-value (--runtime-pass-by-value): value is supplied as a host pointer in the
    // variant pack at execute(), allowing it to vary across executions without rebuilding the
    // graph. Requires plugin SDK >= 1.2.0.
    constexpr auto EPSILON = utilities::BATCHNORM_DEFAULT_EPSILON;
    auto epsilon = std::make_shared<graph::TensorAttributes>();
    epsilon->set_dim({1}).set_stride({1}).set_data_type(getDataTypeEnumFromType<double>());
    if(config.useRuntimePassByValue)
    {
        // No baked value; the value is supplied as a host pointer in the variant pack at
        // execute() instead.
        epsilon->set_as_runtime_parameter();
    }
    else
    {
        epsilon->set_compile_time_constant(EPSILON);
    }

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

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
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

    // Runtime pass-by-value tensors are supplied as host pointers in the variant pack.
    // Compile-time constants are baked at build() and must NOT appear here.
    // Using the same default here keeps output numerically identical to the compile-time path,
    // so CPU validation passes regardless of which mode is active. In a real application this
    // value could differ per execution without rebuilding the graph.
    double epsilonVal = EPSILON;
    if(config.useRuntimePassByValue)
    {
        variantPack[epsilon->get_uid()] = &epsilonVal;
    }

    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    activatedYTensor.memory().markDeviceModified();

    auto activatedYHostPtr = activatedYTensor.memory().hostData();

    auto runCpuValidation = [&]() -> bool {
        std::cout << "Running CPU reference validation using CpuReferenceGraphExecutor...\n";

        // Create reference tensor
        utilities::Tensor<OutputType> activatedYRefTensor(activatedY->get_dim(), layout);

        // Build variant pack for CPU execution (using host pointers)
        std::unordered_map<int64_t, void*> cpuVariantPack{
            {x->get_uid(), xTensor.memory().hostData()},
            {scale->get_uid(), scaleTensor.memory().hostData()},
            {bias->get_uid(), biasTensor.memory().hostData()},
            {mean->get_uid(), meanTensor.memory().hostData()},
            {variance->get_uid(), varianceTensor.memory().hostData()},
            {activatedY->get_uid(), activatedYRefTensor.memory().hostData()}};

        if(config.useRuntimePassByValue)
        {
            cpuVariantPack[epsilon->get_uid()] = &epsilonVal;
        }

        // Execute on CPU using graph executor
        auto [serializedGraph, serErr] = graph->to_binary();
        if(serErr.is_bad())
        {
            std::cerr << "Failed to serialize graph: " << serErr.get_message() << '\n';
            return false;
        }
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
        cpuExecutor.execute(serializedGraph.data(), serializedGraph.size(), cpuVariantPack);

        auto tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceInferenceWithVariance<
            OutputType>();

        auto yValidator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<OutputType>(
            tolerance, tolerance);

        std::cout << "CPU reference validation:\n";
        return hipdnn_test_sdk::utilities::validateAndReport<OutputType>(std::cout,
                                                                         "activated_y",
                                                                         yValidator,
                                                                         activatedYRefTensor,
                                                                         activatedYTensor,
                                                                         tolerance,
                                                                         tolerance);
    };

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        validationPassed = runCpuValidation();
    }

    std::cout << "First 10 activated_y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(activatedYHostPtr[i]) << " ";
    }

    // Demonstrate that the graph can be re-executed with a different scalar value without
    // rebuilding. The variant pack already holds a pointer to epsilonVal, so updating the
    // value in-place is all that is needed.
    if(config.useRuntimePassByValue)
    {
        std::cout << "\nRe-executing with epsilon = 1.0 (no rebuild required)...\n";
        epsilonVal = 1.0;
        HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));
        activatedYTensor.memory().markDeviceModified();
        activatedYHostPtr = activatedYTensor.memory().hostData();

        std::cout << "First 10 activated_y values (epsilon = 1.0): ";
        for(int i = 0; i < 10; ++i)
        {
            std::cout << static_cast<float>(activatedYHostPtr[i]) << " ";
        }

        if(config.cpuValidation)
        {
            std::cout << "\nRe-validating against CPU reference with the updated scalar value...\n";
            validationPassed = runCpuValidation() && validationPassed;
        }
    }

    std::cout << "\nBatch normalization inference with variance + activation graph execution "
                 "complete for "
              << inputType << ".\n\n";

    return validationPassed;
}

int main(int argc, char* argv[])
{
    try
    {
        RETURN_SUCCESS_IF_NO_DEVICE();

        auto config = parseCommandLineArgs(argc, argv, SampleType::BN_WITH_PASS_BY_VALUE);

        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

        const bool allPassed = run(SampleRunner{*handle, config});

        if(allPassed)
        {
            std::cout
                << "All batch normalization inference with variance + activation runs completed "
                   "successfully.\n";
            return 0;
        }
        std::cout << "One or more batch normalization inference with variance + activation runs "
                     "failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
