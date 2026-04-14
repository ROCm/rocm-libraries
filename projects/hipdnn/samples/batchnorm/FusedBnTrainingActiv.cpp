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

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    auto inputType = getDataTypeEnumFromType<InputType>();
    auto intermediateType = getDataTypeEnumFromType<IntermediateType>();

    std::cout << "Running batch normalization training + ReLU activation graph " << inputType
              << " [" << layout << "]" << (config.cpuValidation ? " (with CPU validation)" : "");

    if(config.useRunningStats)
    {
        std::cout << " [FULL_TRAINING mode]...\n";
    }
    else
    {
        std::cout << " [BATCH_STATS_ONLY mode]...\n";
    }

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

    // Epsilon is a pass-by-value scalar, not a buffer
    auto epsilon = std::make_shared<graph::TensorAttributes>();
    epsilon->set_value(utilities::BATCHNORM_DEFAULT_EPSILON);

    auto bnAttributes = graph::BatchnormAttributes();
    bnAttributes.set_name("bn_training_node");
    bnAttributes.set_epsilon(epsilon);

    // Declare running statistics tensors at broader scope
    std::shared_ptr<graph::TensorAttributes> prevRunningMean;
    std::shared_ptr<graph::TensorAttributes> prevRunningVar;

    double momentumVal = 0.1;

    // Conditionally setup running statistics inputs
    if(config.useRunningStats)
    {
        prevRunningMean = createTensor({1, c, 1, 1}, intermediateType);
        prevRunningVar = createTensor({1, c, 1, 1}, intermediateType);

        // Momentum: use pass-by-value with double (matches MIOpen API)
        auto momentum = std::make_shared<graph::TensorAttributes>();
        momentum->set_value(momentumVal);

        bnAttributes.set_previous_running_stats(prevRunningMean, prevRunningVar, momentum);
    }

    // Step 1: Batchnorm Training
    auto [y, savedMean, savedInvVariance, nextRunningMean, nextRunningVariance]
        = graph->batchnorm(x, scale, bias, bnAttributes);

    // Step 2: Pointwise ReLU Activation
    auto pwAttributes = graph::PointwiseAttributes();
    pwAttributes.set_name("activation_node");
    pwAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto activatedY = graph->pointwise(y, pwAttributes);
    activatedY->set_name("activated_y");
    activatedY->set_output(true);

    // Configure output tensors for batch statistics
    savedMean->set_output(true).set_data_type(intermediateType);
    savedInvVariance->set_output(true).set_data_type(intermediateType);

    // Configure running statistics output tensors
    if(config.useRunningStats)
    {
        nextRunningMean->set_output(true).set_data_type(intermediateType);
        nextRunningVariance->set_output(true).set_data_type(intermediateType);
    }

    HIPDNN_FE_CHECK(graph->build(handle));
    std::cout << "Graph build successful.\n";

    // Allocate tensors for BATCH_STATS_ONLY mode
    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
    utilities::Tensor<IntermediateType> biasTensor(bias->get_dim());
    utilities::Tensor<InputType> activatedYTensor(activatedY->get_dim(), layout);
    utilities::Tensor<IntermediateType> savedMeanTensor(savedMean->get_dim());
    utilities::Tensor<IntermediateType> savedInvVarTensor(savedInvVariance->get_dim());

    // Declare running statistics tensors at broader scope (conditionally initialized)
    utilities::Tensor<IntermediateType> prevMeanTensor(
        config.useRunningStats ? prevRunningMean->get_dim() : std::vector<int64_t>{1});
    utilities::Tensor<IntermediateType> prevVarTensor(
        config.useRunningStats ? prevRunningVar->get_dim() : std::vector<int64_t>{1});
    utilities::Tensor<IntermediateType> nextMeanTensor(
        config.useRunningStats ? nextRunningMean->get_dim() : std::vector<int64_t>{1});
    utilities::Tensor<IntermediateType> nextVarTensor(
        config.useRunningStats ? nextRunningVariance->get_dim() : std::vector<int64_t>{1});

    // Initialize tensors
    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    scaleTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                     static_cast<IntermediateType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                    static_cast<IntermediateType>(1.0f));

    if(config.useRunningStats)
    {
        prevMeanTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                            static_cast<IntermediateType>(1.0f));
        prevVarTensor.fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                           static_cast<IntermediateType>(1.0f));
    }

    // Build variant pack
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[activatedY->get_uid()] = activatedYTensor.memory().deviceData();
    variantPack[savedMean->get_uid()] = savedMeanTensor.memory().deviceData();
    variantPack[savedInvVariance->get_uid()] = savedInvVarTensor.memory().deviceData();

    if(config.useRunningStats)
    {
        variantPack[prevRunningMean->get_uid()] = prevMeanTensor.memory().deviceData();
        variantPack[prevRunningVar->get_uid()] = prevVarTensor.memory().deviceData();
        variantPack[nextRunningMean->get_uid()] = nextMeanTensor.memory().deviceData();
        variantPack[nextRunningVariance->get_uid()] = nextVarTensor.memory().deviceData();
    }

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    activatedYTensor.memory().markDeviceModified();
    savedMeanTensor.memory().markDeviceModified();
    savedInvVarTensor.memory().markDeviceModified();

    if(config.useRunningStats)
    {
        nextMeanTensor.memory().markDeviceModified();
        nextVarTensor.memory().markDeviceModified();
    }

    auto activatedYHostPtr = activatedYTensor.memory().hostData();
    auto savedMeanHostPtr = savedMeanTensor.memory().hostData();
    auto savedInvVarHostPtr = savedInvVarTensor.memory().hostData();

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation using CpuReferenceGraphExecutor...\n";

        // Create reference tensors
        utilities::Tensor<InputType> activatedYRefTensor(activatedY->get_dim(), layout);
        utilities::Tensor<IntermediateType> savedMeanRefTensor(savedMean->get_dim());
        utilities::Tensor<IntermediateType> savedInvVarRefTensor(savedInvVariance->get_dim());

        utilities::Tensor<IntermediateType> nextMeanRefTensor(
            config.useRunningStats ? nextRunningMean->get_dim() : std::vector<int64_t>{1});
        utilities::Tensor<IntermediateType> nextVarRefTensor(
            config.useRunningStats ? nextRunningVariance->get_dim() : std::vector<int64_t>{1});

        // Build variant pack for CPU execution (using host pointers)
        std::unordered_map<int64_t, void*> cpuVariantPack;
        cpuVariantPack[x->get_uid()] = xTensor.memory().hostData();
        cpuVariantPack[scale->get_uid()] = scaleTensor.memory().hostData();
        cpuVariantPack[bias->get_uid()] = biasTensor.memory().hostData();
        cpuVariantPack[activatedY->get_uid()] = activatedYRefTensor.memory().hostData();
        cpuVariantPack[savedMean->get_uid()] = savedMeanRefTensor.memory().hostData();
        cpuVariantPack[savedInvVariance->get_uid()] = savedInvVarRefTensor.memory().hostData();

        if(config.useRunningStats)
        {
            cpuVariantPack[prevRunningMean->get_uid()] = prevMeanTensor.memory().hostData();
            cpuVariantPack[prevRunningVar->get_uid()] = prevVarTensor.memory().hostData();
            cpuVariantPack[nextRunningMean->get_uid()] = nextMeanRefTensor.memory().hostData();
            cpuVariantPack[nextRunningVariance->get_uid()] = nextVarRefTensor.memory().hostData();
        }

        // Execute on CPU using graph executor
        auto [serializedGraph, serErr] = graph->to_binary();
        if(serErr.is_bad())
        {
            std::cerr << "Failed to serialize graph: " << serErr.get_message() << std::endl;
            return false;
        }
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
        cpuExecutor.execute(serializedGraph.data(), serializedGraph.size(), cpuVariantPack);

        auto tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceTraining<InputType>();
        auto floatTolerance = static_cast<float>(tolerance);
        auto yValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);
        auto statsValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<IntermediateType>(
                static_cast<IntermediateType>(tolerance), static_cast<IntermediateType>(tolerance));

        std::cout << "CPU reference validation:\n";
        bool yValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(std::cout,
                                                                               "activated_y",
                                                                               yValidator,
                                                                               activatedYRefTensor,
                                                                               activatedYTensor,
                                                                               floatTolerance,
                                                                               floatTolerance);
        bool meanValid
            = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(std::cout,
                                                                              "saved_mean",
                                                                              statsValidator,
                                                                              savedMeanRefTensor,
                                                                              savedMeanTensor,
                                                                              floatTolerance,
                                                                              floatTolerance);
        bool invVarValid
            = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(std::cout,
                                                                              "saved_inv_variance",
                                                                              statsValidator,
                                                                              savedInvVarRefTensor,
                                                                              savedInvVarTensor,
                                                                              floatTolerance,
                                                                              floatTolerance);

        bool nextMeanValid = true;
        bool nextVarValid = true;
        if(config.useRunningStats)
        {
            nextMeanValid = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                std::cout,
                "next_running_mean",
                statsValidator,
                nextMeanRefTensor,
                nextMeanTensor,
                floatTolerance,
                floatTolerance);
            nextVarValid = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                std::cout,
                "next_running_variance",
                statsValidator,
                nextVarRefTensor,
                nextVarTensor,
                floatTolerance,
                floatTolerance);
        }

        validationPassed = yValid && meanValid && invVarValid && nextMeanValid && nextVarValid;
    }

    std::cout << "First 10 activated_y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(activatedYHostPtr[i]) << " ";
    }
    std::cout << "\nFirst 10 saved_mean values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(savedMeanHostPtr[i]) << " ";
    }
    std::cout << "\nFirst 10 saved_inv_variance values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(savedInvVarHostPtr[i]) << " ";
    }

    if(config.useRunningStats)
    {
        auto nextMeanHostPtr = nextMeanTensor.memory().hostData();
        auto nextVarHostPtr = nextVarTensor.memory().hostData();

        std::cout << "\nFirst 10 next_running_mean values: ";
        for(int i = 0; i < 10; ++i)
        {
            std::cout << static_cast<float>(nextMeanHostPtr[i]) << " ";
        }
        std::cout << "\nFirst 10 next_running_variance values: ";
        for(int i = 0; i < 10; ++i)
        {
            std::cout << static_cast<float>(nextVarHostPtr[i]) << " ";
        }
    }

    std::cout << "\nBatch normalization training + activation graph execution complete for "
              << inputType << ".\n\n";
    return validationPassed;
}

int main(int argc, char* argv[])
{
    auto config = parseCommandLineArgs(argc, argv, SampleType::BN_TRAINING);

    auto [handle, handleError] = createHipdnnHandle();
    HIPDNN_FE_CHECK(handleError);

    bool allPassed = run(SampleRunner{*handle, config});

    if(allPassed)
    {
        std::cout << "All batch normalization training + activation runs completed successfully.\n";
        return 0;
    }
    else
    {
        std::cout
            << "One or more batch normalization training + activation runs failed validation.\n";
        return 1;
    }
}
