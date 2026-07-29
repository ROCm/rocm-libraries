// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    auto inputType = getDataTypeEnumFromType<InputType>();
    auto intermediateType = getDataTypeEnumFromType<IntermediateType>();

    std::cout << "Running batch normalization training graph " << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "")
              << (config.useRuntimePassByValue ? " [runtime-pass-by-value]" : "");

    if(config.useRunningStats)
    {
        std::cout << " [FULL_TRAINING mode]...\n";
    }
    else
    {
        std::cout << " [BATCH_STATS_ONLY mode]...\n";
    }

    // Input dimensions
    const int64_t n = !config.dims.empty() ? config.dims[0] : 16; // BATCH SIZE
    const int64_t c = config.dims.size() > 1 ? config.dims[1] : 16; // CHANNELS (FEATURES)
    const int64_t h = config.dims.size() > 2 ? config.dims[2] : 16; // HEIGHT (SPATIAL DIMENSION)
    const int64_t w = config.dims.size() > 3 ? config.dims[3] : 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    setPreferredEngine(graph, config);

    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, intermediateType);
    auto bias = createTensor({1, c, 1, 1}, intermediateType);
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

    auto bnAttributes = graph::BatchnormAttributes();
    bnAttributes.set_name("bn_training_node");
    bnAttributes.set_epsilon(epsilon);

    std::shared_ptr<graph::TensorAttributes> prevRunningMean;
    std::shared_ptr<graph::TensorAttributes> prevRunningVar;
    std::shared_ptr<graph::TensorAttributes> momentum;

    constexpr double MOMENTUM = 0.1;

    // Conditionally setup running statistics inputs
    if(config.useRunningStats)
    {
        prevRunningMean = createTensor({1, c, 1, 1}, intermediateType);
        prevRunningVar = createTensor({1, c, 1, 1}, intermediateType);
        // Momentum follows the same compile-time vs. runtime pattern as epsilon.
        momentum = std::make_shared<graph::TensorAttributes>();
        momentum->set_dim({1}).set_stride({1}).set_data_type(
            getDataTypeEnumFromType<decltype(MOMENTUM)>());
        if(config.useRuntimePassByValue)
        {
            momentum->set_as_runtime_parameter();
        }
        else
        {
            momentum->set_compile_time_constant(MOMENTUM);
        }

        bnAttributes.set_previous_running_stats(prevRunningMean, prevRunningVar, momentum);
    }
    // API always returns 5 values regardless of whether running stats are used
    auto [y, savedMean, savedInvVariance, nextRunningMean, nextRunningVariance]
        = graph->batchnorm(x, scale, bias, bnAttributes);
    // Configure output tensors (always needed for BATCH_STATS_ONLY mode)
    y->set_output(true);
    savedMean->set_output(true).set_data_type(intermediateType);
    savedInvVariance->set_output(true).set_data_type(intermediateType);

    if(config.useRunningStats)
    {
        nextRunningMean->set_output(true).set_data_type(intermediateType);
        nextRunningVariance->set_output(true).set_data_type(intermediateType);
    }

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));

    std::cout << "Graph build successful.\n";

    // Allocate tensors for BATCH_STATS_ONLY mode
    // Note: epsilon is pass-by-value, no buffer allocation needed
    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
    utilities::Tensor<IntermediateType> biasTensor(bias->get_dim());
    utilities::Tensor<InputType> yTensor(y->get_dim(), layout);
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
    xTensor.fillWithRandomValues(static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f));
    scaleTensor.fillWithRandomValues(static_cast<IntermediateType>(-2.0f),
                                     static_cast<IntermediateType>(2.0f));
    biasTensor.fillWithRandomValues(static_cast<IntermediateType>(-2.0f),
                                    static_cast<IntermediateType>(2.0f));

    if(config.useRunningStats)
    {
        prevMeanTensor.fillWithRandomValues(static_cast<IntermediateType>(-2.0f),
                                            static_cast<IntermediateType>(2.0f));
        prevVarTensor.fillWithRandomValues(static_cast<IntermediateType>(-2.0f),
                                           static_cast<IntermediateType>(2.0f));
    }

    // Build variant pack with batch statistics.
    // Runtime pass-by-value tensors are supplied as host pointers in the variant pack.
    // Compile-time constants are baked at build() and must NOT appear here.
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[y->get_uid()] = yTensor.memory().deviceData();
    variantPack[savedMean->get_uid()] = savedMeanTensor.memory().deviceData();
    variantPack[savedInvVariance->get_uid()] = savedInvVarTensor.memory().deviceData();

    // Using the same default here keeps output numerically identical to the compile-time path,
    // so CPU validation passes regardless of which mode is active. In a real application this
    // value could differ per execution without rebuilding the graph.
    double epsilonVal = EPSILON;
    double momentumVal = MOMENTUM;
    if(config.useRuntimePassByValue)
    {
        variantPack[epsilon->get_uid()] = &epsilonVal;
    }

    if(config.useRunningStats)
    {
        variantPack[prevRunningMean->get_uid()] = prevMeanTensor.memory().deviceData();
        variantPack[prevRunningVar->get_uid()] = prevVarTensor.memory().deviceData();
        variantPack[nextRunningMean->get_uid()] = nextMeanTensor.memory().deviceData();
        variantPack[nextRunningVariance->get_uid()] = nextVarTensor.memory().deviceData();
        if(config.useRuntimePassByValue)
        {
            variantPack[momentum->get_uid()] = &momentumVal;
        }
    }

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));

    yTensor.memory().markDeviceModified();
    savedMeanTensor.memory().markDeviceModified();
    savedInvVarTensor.memory().markDeviceModified();

    if(config.useRunningStats)
    {
        nextMeanTensor.memory().markDeviceModified();
        nextVarTensor.memory().markDeviceModified();
    }

    auto yHostPtr = yTensor.memory().hostData();
    auto savedMeanHostPtr = savedMeanTensor.memory().hostData();
    auto savedInvVarHostPtr = savedInvVarTensor.memory().hostData();

    // Runs CPU reference validation against the current GPU output buffers. Called once after
    // the initial execute() and again after the runtime pass-by-value re-execution demo below,
    // so both scalar values get verified against the CPU reference.
    auto runCpuValidation = [&]() -> bool {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> yRefTensor(y->get_dim(), layout);
        utilities::Tensor<IntermediateType> savedMeanRefTensor(savedMean->get_dim());
        utilities::Tensor<IntermediateType> savedInvVarRefTensor(savedInvVariance->get_dim());

        if(config.useRunningStats)
        {
            // FULL_TRAINING mode validation
            utilities::Tensor<IntermediateType> nextMeanRefTensor(nextRunningMean->get_dim());
            utilities::Tensor<IntermediateType> nextVarRefTensor(nextRunningVariance->get_dim());

            hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::fwdTraining<
                InputType, // XDataType
                IntermediateType, // ScaleBiasDataType
                IntermediateType, // MeanVarianceDataType
                InputType // YDataType
                >(xTensor,
                  scaleTensor,
                  biasTensor,
                  yRefTensor,
                  epsilonVal,
                  momentumVal, // momentum value used
                  &savedMeanRefTensor,
                  &savedInvVarRefTensor,
                  &prevMeanTensor, // used
                  &prevVarTensor, // used
                  &nextMeanRefTensor, // used
                  &nextVarRefTensor // used
            );

            auto tolerance
                = hipdnn_test_sdk::utilities::batchnorm::getToleranceTraining<InputType>();
            auto floatTolerance = static_cast<float>(tolerance);

            auto yValidator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(
                tolerance, tolerance);

            auto statsValidator
                = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<IntermediateType>(
                    static_cast<IntermediateType>(tolerance),
                    static_cast<IntermediateType>(tolerance));

            std::cout << "CPU reference validation:\n";
            const bool yValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
                std::cout, "y", yValidator, yRefTensor, yTensor, floatTolerance, floatTolerance);
            const bool meanValid = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                std::cout,
                "saved_mean",
                statsValidator,
                savedMeanRefTensor,
                savedMeanTensor,
                floatTolerance,
                floatTolerance);
            const bool invVarValid
                = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                    std::cout,
                    "saved_inv_variance",
                    statsValidator,
                    savedInvVarRefTensor,
                    savedInvVarTensor,
                    floatTolerance,
                    floatTolerance);
            const bool nextMeanValid
                = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                    std::cout,
                    "next_running_mean",
                    statsValidator,
                    nextMeanRefTensor,
                    nextMeanTensor,
                    floatTolerance,
                    floatTolerance);
            const bool nextVarValid
                = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                    std::cout,
                    "next_running_var",
                    statsValidator,
                    nextVarRefTensor,
                    nextVarTensor,
                    floatTolerance,
                    floatTolerance);

            return yValid && meanValid && invVarValid && nextMeanValid && nextVarValid;
        }
        // BATCH_STATS_ONLY mode validation
        hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::fwdTraining<
            InputType, // XDataType
            IntermediateType, // ScaleBiasDataType
            IntermediateType, // MeanVarianceDataType
            InputType // YDataType
            >(xTensor,
              scaleTensor,
              biasTensor,
              yRefTensor,
              epsilonVal,
              momentumVal, // momentum (not used in BATCH_STATS_ONLY mode but required by API)
              &savedMeanRefTensor,
              &savedInvVarRefTensor,
              nullptr, // prevRunningMean (not used)
              nullptr, // prevRunningVariance (not used)
              nullptr, // nextRunningMean (not used)
              nullptr // nextRunningVariance (not used)
        );

        auto tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceTraining<InputType>();
        auto floatTolerance = static_cast<float>(tolerance);

        auto yValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        auto statsValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<IntermediateType>(
                static_cast<IntermediateType>(tolerance), static_cast<IntermediateType>(tolerance));

        std::cout << "CPU reference validation:\n";
        const bool yValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "y", yValidator, yRefTensor, yTensor, floatTolerance, floatTolerance);
        const bool meanValid
            = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(std::cout,
                                                                              "saved_mean",
                                                                              statsValidator,
                                                                              savedMeanRefTensor,
                                                                              savedMeanTensor,
                                                                              floatTolerance,
                                                                              floatTolerance);
        const bool invVarValid
            = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(std::cout,
                                                                              "saved_inv_variance",
                                                                              statsValidator,
                                                                              savedInvVarRefTensor,
                                                                              savedInvVarTensor,
                                                                              floatTolerance,
                                                                              floatTolerance);

        return yValid && meanValid && invVarValid;
    };

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        validationPassed = runCpuValidation();
    }

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(yHostPtr[i]) << " ";
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

    std::cout << '\n';

    // Demonstrate that the graph can be re-executed with different scalar values without
    // rebuilding. The variant pack already holds pointers to epsilonVal (and momentumVal),
    // so updating their values in-place is all that is needed.
    if(config.useRuntimePassByValue)
    {
        std::cout << "Re-executing with epsilon = 1.0, momentum = 0.9 (no rebuild required)...\n";
        epsilonVal = 1.0;
        momentumVal = 0.9;
        HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));
        yTensor.memory().markDeviceModified();
        savedMeanTensor.memory().markDeviceModified();
        savedInvVarTensor.memory().markDeviceModified();
        if(config.useRunningStats)
        {
            nextMeanTensor.memory().markDeviceModified();
            nextVarTensor.memory().markDeviceModified();
        }
        yHostPtr = yTensor.memory().hostData();

        std::cout << "First 10 y values (epsilon = 1.0, momentum = 0.9): ";
        for(int i = 0; i < 10; ++i)
        {
            std::cout << static_cast<float>(yHostPtr[i]) << " ";
        }
        std::cout << '\n';

        if(config.cpuValidation)
        {
            std::cout << "Re-validating against CPU reference with the updated scalar values...\n";
            validationPassed = runCpuValidation() && validationPassed;
        }
    }

    std::cout << "\nBatch normalization training graph execution complete for " << inputType
              << ".\n\n";

    return validationPassed;
}

int main(int argc, char* argv[])
{
    try
    {
        RETURN_SUCCESS_IF_NO_DEVICE();

        auto config = parseCommandLineArgs(argc, argv, SampleType::BN_TRAINING);

        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

        const bool allPassed = run(SampleRunner{*handle, config});

        if(allPassed)
        {
            std::cout << "All batch normalization training runs completed successfully.\n";
            return 0;
        }
        std::cout << "One or more batch normalization training runs failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
