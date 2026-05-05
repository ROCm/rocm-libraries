// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
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

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    auto inputType = getDataTypeEnumFromType<InputType>();
    auto intermediateType = getDataTypeEnumFromType<IntermediateType>();

    std::cout << "Running fused BN Inference + Activation Backward + BN Backward graph "
              << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    int64_t n = 1; // Batch size
    int64_t c = 3; // Channels
    int64_t h = 14; // Height
    int64_t w = 14; // Width

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    // Input tensors
    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto dy = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto bias = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto savedMean = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto savedInvVariance = createTensor({1, c, 1, 1}, intermediateType, layout);

    // Build fused graph: BN Inference -> Activation Backward -> BN Backward

    // Step 1: Batchnorm Inference
    auto bnInfAttributes = graph::BatchnormInferenceAttributes();
    bnInfAttributes.set_name("bn_inference_node");

    auto bnY
        = graph->batchnorm_inference(x, savedMean, savedInvVariance, scale, bias, bnInfAttributes);
    bnY->set_name("bn_y");

    // Step 2: Activation Backward (ReLU backward)
    auto activBwdAttributes = graph::PointwiseAttributes();
    activBwdAttributes.set_name("activation_backward_node");
    activBwdAttributes.set_mode(hipdnn_frontend::PointwiseMode::RELU_BWD);

    auto dxDrelu = graph->pointwise(bnY, dy, activBwdAttributes);
    dxDrelu->set_name("dx_drelu");

    // Step 3: Batchnorm Backward
    auto bnBwdAttributes = graph::BatchnormBackwardAttributes();
    bnBwdAttributes.set_name("bn_backward_node");
    bnBwdAttributes.set_saved_mean_and_inv_variance(savedMean, savedInvVariance);

    auto [dx, dscale, dbias] = graph->batchnorm_backward(dxDrelu, x, scale, bnBwdAttributes);
    dx->set_name("dx");
    dx->set_output(true);

    dscale->set_name("dscale");
    dscale->set_data_type(intermediateType);
    dscale->set_output(true);

    dbias->set_name("dbias");
    dbias->set_data_type(intermediateType);
    dbias->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful.\n";

    // Create tensors for execution
    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<InputType> dyTensor(dy->get_dim(), layout);
    utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
    utilities::Tensor<IntermediateType> biasTensor(bias->get_dim());
    utilities::Tensor<IntermediateType> savedMeanTensor(savedMean->get_dim());
    utilities::Tensor<IntermediateType> savedInvVarTensor(savedInvVariance->get_dim());

    utilities::Tensor<InputType> dxTensor(dx->get_dim(), layout);
    utilities::Tensor<IntermediateType> dscaleTensor(dscale->get_dim());
    utilities::Tensor<IntermediateType> dbiasTensor(dbias->get_dim());

    // Initialize input tensors with random values
    xTensor.fillWithRandomValues(static_cast<InputType>(-1.8f), static_cast<InputType>(1.8f));
    dyTensor.fillWithRandomValues(static_cast<InputType>(-1.8f), static_cast<InputType>(1.8f));
    scaleTensor.fillWithRandomValues(static_cast<IntermediateType>(0.5f),
                                     static_cast<IntermediateType>(1.5f));
    biasTensor.fillWithRandomValues(static_cast<IntermediateType>(-0.1f),
                                    static_cast<IntermediateType>(0.1f));
    savedMeanTensor.fillWithRandomValues(static_cast<IntermediateType>(-0.1f),
                                         static_cast<IntermediateType>(0.1f));
    savedInvVarTensor.fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                           static_cast<IntermediateType>(2.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[dy->get_uid()] = dyTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[savedMean->get_uid()] = savedMeanTensor.memory().deviceData();
    variantPack[savedInvVariance->get_uid()] = savedInvVarTensor.memory().deviceData();
    variantPack[dx->get_uid()] = dxTensor.memory().deviceData();
    variantPack[dscale->get_uid()] = dscaleTensor.memory().deviceData();
    variantPack[dbias->get_uid()] = dbiasTensor.memory().deviceData();

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    dxTensor.memory().markDeviceModified();
    dscaleTensor.memory().markDeviceModified();
    dbiasTensor.memory().markDeviceModified();

    auto dxHostPtr = dxTensor.memory().hostData();
    auto dscaleHostPtr = dscaleTensor.memory().hostData();
    auto dbiasHostPtr = dbiasTensor.memory().hostData();

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation using CpuReferenceGraphExecutor...\n";

        // Create reference tensors
        utilities::Tensor<InputType> dxRefTensor(dx->get_dim(), layout);
        utilities::Tensor<IntermediateType> dscaleRefTensor(dscale->get_dim());
        utilities::Tensor<IntermediateType> dbiasRefTensor(dbias->get_dim());

        // Build variant pack for CPU execution (using host pointers)
        std::unordered_map<int64_t, void*> cpuVariantPack;
        cpuVariantPack[x->get_uid()] = xTensor.memory().hostData();
        cpuVariantPack[dy->get_uid()] = dyTensor.memory().hostData();
        cpuVariantPack[scale->get_uid()] = scaleTensor.memory().hostData();
        cpuVariantPack[bias->get_uid()] = biasTensor.memory().hostData();
        cpuVariantPack[savedMean->get_uid()] = savedMeanTensor.memory().hostData();
        cpuVariantPack[savedInvVariance->get_uid()] = savedInvVarTensor.memory().hostData();
        cpuVariantPack[dx->get_uid()] = dxRefTensor.memory().hostData();
        cpuVariantPack[dscale->get_uid()] = dscaleRefTensor.memory().hostData();
        cpuVariantPack[dbias->get_uid()] = dbiasRefTensor.memory().hostData();

        // Execute on CPU using graph executor
        auto [serializedGraph, serErr] = graph->to_binary();
        if(serErr.is_bad())
        {
            std::cerr << "Failed to serialize graph: " << serErr.get_message() << std::endl;
            return false;
        }
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
        cpuExecutor.execute(serializedGraph.data(), serializedGraph.size(), cpuVariantPack);

        // Tolerance range is high due to data-type mismatch between plugin and reference impl.
        // This will be fixed in a follow-up.
        // Issue is due to the reference not splitting the input / output datatypes.
        const auto inputTol = 4e-2f;

        auto dxValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(inputTol, inputTol);
        auto dscaleDbiasValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<IntermediateType>(inputTol,
                                                                                     inputTol);

        std::cout << "CPU reference validation:\n";
        bool dxValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "dx", dxValidator, dxRefTensor, dxTensor, inputTol, inputTol);
        bool dscaleValid
            = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(std::cout,
                                                                              "dscale",
                                                                              dscaleDbiasValidator,
                                                                              dscaleRefTensor,
                                                                              dscaleTensor,
                                                                              inputTol,
                                                                              inputTol);
        bool dbiasValid
            = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(std::cout,
                                                                              "dbias",
                                                                              dscaleDbiasValidator,
                                                                              dbiasRefTensor,
                                                                              dbiasTensor,
                                                                              inputTol,
                                                                              inputTol);

        validationPassed = dxValid && dscaleValid && dbiasValid;
    }

    std::cout << "First 10 dx values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<InputType>(dxHostPtr[i]) << " ";
    }
    std::cout << "\nFirst 10 dscale values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<IntermediateType>(dscaleHostPtr[i]) << " ";
    }
    std::cout << "\nFirst 10 dbias values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<IntermediateType>(dbiasHostPtr[i]) << " ";
    }

    std::cout << "\nFused BN Inference + Activation Backward + BN Backward graph execution "
              << "complete for " << inputType << ".\n\n";
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
        std::cout << "All fused BN Inference + Activation Backward + BN Backward runs completed "
                  << "successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "One or more fused BN Inference + Activation Backward + BN Backward runs "
                  << "failed validation.\n";
        return 1;
    }
}
