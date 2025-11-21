// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_frontend.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/Workspace.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk;

template <typename InputType, typename IntermediateType>
void SampleRunner::operator()(const TensorLayout& layout)
{
    auto inputType = getDataTypeEnumFromType<InputType>();
    auto intermediateType = getDataTypeEnumFromType<IntermediateType>();

    std::cout << "Running batch normalization inference + activation graph " << inputType << " ["
              << layout << "]" << (config.cpuValidation ? " (with CPU validation)" : "")
              << " [activation: " << config.activationType << "]...\n";

    int64_t n = 16; // BATCH SIZE
    int64_t c = 16; // CHANNELS (FEATURES)
    int64_t h = 16; // HEIGHT (SPATIAL DIMENSION)
    int64_t w = 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, intermediateType);
    auto bias = createTensor({1, c, 1, 1}, intermediateType);
    auto mean = createTensor({1, c, 1, 1}, intermediateType);
    auto invVariance = createTensor({1, c, 1, 1}, intermediateType);

    // Step 1: Batchnorm Inference
    auto bnAttributes = graph::BatchnormInferenceAttributes();
    bnAttributes.set_name("bn_inference_node");

    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);

    // Mark BN output as virtual to enable fusion with activation
    y->set_is_virtual(true);

    // Step 2: Pointwise Activation
    auto pwAttributes = graph::PointwiseAttributes();
    pwAttributes.set_name("activation_node");
    pwAttributes.set_mode(PointwiseMode::RELU_FWD);

    // Configure activation based on type
    if(config.activationType == "relu6")
    {
        // Clipped ReLU with upper clip at 6.0
        pwAttributes.set_relu_upper_clip(6.0f);
    }
    else if(config.activationType == "clamp")
    {
        // CLAMP with both lower and upper clips
        pwAttributes.set_relu_lower_clip(0.1f);
        pwAttributes.set_relu_upper_clip(0.5f);
    }
    // For "relu", no additional parameters needed

    auto activatedY = graph->pointwise(y, pwAttributes);
    activatedY->set_name("activated_y");
    activatedY->set_output(true);
    activatedY->set_is_virtual(false);

    HIPDNN_FE_CHECK(graph->validate());
    std::cout << "Graph validation successful.\n";

    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));
    std::cout << "Operation graph build successful.\n";

    HIPDNN_FE_CHECK(graph->create_execution_plans());
    std::cout << "Execution plans created successfully.\n";

    HIPDNN_FE_CHECK(graph->check_support());
    std::cout << "Graph support check successful.\n";

    HIPDNN_FE_CHECK(graph->build_plans());
    std::cout << "Plans build successful.\n";

    // Allocate tensors
    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
    utilities::Tensor<IntermediateType> biasTensor(bias->get_dim());
    utilities::Tensor<IntermediateType> meanTensor(mean->get_dim());
    utilities::Tensor<IntermediateType> invVarianceTensor(invVariance->get_dim());
    utilities::Tensor<InputType> activatedYTensor(activatedY->get_dim(), layout);

    // Initialize tensors
    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    scaleTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                     static_cast<IntermediateType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                    static_cast<IntermediateType>(1.0f));
    meanTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                    static_cast<IntermediateType>(1.0f));
    invVarianceTensor.fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                           static_cast<IntermediateType>(1.0f));

    // Build variant pack
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[mean->get_uid()] = meanTensor.memory().deviceData();
    variantPack[invVariance->get_uid()] = invVarianceTensor.memory().deviceData();
    variantPack[activatedY->get_uid()] = activatedYTensor.memory().deviceData();

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    activatedYTensor.memory().markDeviceModified();
    auto activatedYHostPtr = activatedYTensor.memory().hostData();

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation using CpuReferenceGraphExecutor...\n";

        // Create reference tensor
        utilities::Tensor<InputType> activatedYRefTensor(activatedY->get_dim(), layout);

        // Build variant pack for CPU execution (using host pointers)
        std::unordered_map<int64_t, void*> cpuVariantPack;
        cpuVariantPack[x->get_uid()] = xTensor.memory().hostData();
        cpuVariantPack[scale->get_uid()] = scaleTensor.memory().hostData();
        cpuVariantPack[bias->get_uid()] = biasTensor.memory().hostData();
        cpuVariantPack[mean->get_uid()] = meanTensor.memory().hostData();
        cpuVariantPack[invVariance->get_uid()] = invVarianceTensor.memory().hostData();
        cpuVariantPack[activatedY->get_uid()] = activatedYRefTensor.memory().hostData();

        // Execute on CPU using graph executor
        auto serializedGraph = graph->buildFlatbufferOperationGraph();
        test_utilities::CpuReferenceGraphExecutor cpuExecutor;
        cpuExecutor.execute(serializedGraph.data(), serializedGraph.size(), cpuVariantPack);

        auto tolerance = test_utilities::batchnorm::getToleranceInference<InputType>();
        auto yValidator = test_utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        bool yValid = yValidator.allClose(activatedYRefTensor, activatedYTensor);

        std::cout << "CPU reference validation:\n";
        std::cout << "  activated_y: " << (yValid ? "successful" : "failed") << "\n";
    }

    std::cout << "First 10 activated_y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(activatedYHostPtr[i]) << " ";
    }

    std::cout << "\nBatch normalization inference + activation graph execution complete for "
              << inputType << ".\n\n";
}

int main(int argc, char* argv[])
{
    auto config = parseCommandLineArgs(argc, argv);

    initializeFrontendLogging();

    auto backend = hipdnnBackend();
    hipdnnHandle_t handle;
    HIPDNN_CHECK(backend->create(&handle));

    run(SampleRunner{handle, config});

    HIPDNN_CHECK(backend->destroy(handle));
    std::cout << "All batch normalization inference + activation runs completed.\n";
    return 0;
}
