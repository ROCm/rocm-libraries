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
              << (config.cpuValidation ? " (with CPU validation)" : "");

    if(config.useRunningStats)
        std::cout << " [FULL_TRAINING mode]...\n";
    else
        std::cout << " [BATCH_STATS_ONLY mode]...\n";

    auto n = config.dims.size() > 0 ? config.dims[0] : 16;
    auto c = config.dims.size() > 1 ? config.dims[1] : 16;
    auto h = config.dims.size() > 2 ? config.dims[2] : 16;
    auto w = config.dims.size() > 3 ? config.dims[3] : 16;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, intermediateType);
    auto bias = createTensor({1, c, 1, 1}, intermediateType);

    auto epsilon = std::make_shared<graph::TensorAttributes>();
    epsilon->set_value(utilities::BATCHNORM_DEFAULT_EPSILON);

    auto bnAttributes = graph::BatchnormAttributes();
    bnAttributes.set_name("bn_training_node");
    bnAttributes.set_epsilon(epsilon);

    std::shared_ptr<graph::TensorAttributes> prevRunningMean;
    std::shared_ptr<graph::TensorAttributes> prevRunningVar;

    const double momentumVal = 0.1;

    if(config.useRunningStats)
    {
        prevRunningMean = createTensor({1, c, 1, 1}, intermediateType);
        prevRunningVar = createTensor({1, c, 1, 1}, intermediateType);

        auto momentum = std::make_shared<graph::TensorAttributes>();
        momentum->set_value(momentumVal);

        bnAttributes.set_previous_running_stats(prevRunningMean, prevRunningVar, momentum);
    }

    auto [y, savedMean, savedInvVariance, nextRunningMean, nextRunningVariance]
        = graph->batchnorm(x, scale, bias, bnAttributes);

    y->set_output(true);
    savedMean->set_output(true).set_data_type(intermediateType);
    savedInvVariance->set_output(true).set_data_type(intermediateType);

    if(config.useRunningStats)
    {
        nextRunningMean->set_output(true).set_data_type(intermediateType);
        nextRunningVariance->set_output(true).set_data_type(intermediateType);
    }

    HIPDNN_FE_CHECK(graph->build(handle));

    std::cout << "Graph build successful.\n";

    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
    utilities::Tensor<IntermediateType> biasTensor(bias->get_dim());
    utilities::Tensor<InputType> yTensor(y->get_dim(), layout);
    utilities::Tensor<IntermediateType> savedMeanTensor(savedMean->get_dim());
    utilities::Tensor<IntermediateType> savedInvVarTensor(savedInvVariance->get_dim());

    utilities::Tensor<IntermediateType> prevMeanTensor(
        config.useRunningStats ? prevRunningMean->get_dim() : std::vector<int64_t>{1});
    utilities::Tensor<IntermediateType> prevVarTensor(
        config.useRunningStats ? prevRunningVar->get_dim() : std::vector<int64_t>{1});
    utilities::Tensor<IntermediateType> nextMeanTensor(
        config.useRunningStats ? nextRunningMean->get_dim() : std::vector<int64_t>{1});
    utilities::Tensor<IntermediateType> nextVarTensor(
        config.useRunningStats ? nextRunningVariance->get_dim() : std::vector<int64_t>{1});

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

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[y->get_uid()] = yTensor.memory().deviceData();
    variantPack[savedMean->get_uid()] = savedMeanTensor.memory().deviceData();
    variantPack[savedInvVariance->get_uid()] = savedInvVarTensor.memory().deviceData();

    if(config.useRunningStats)
    {
        variantPack[prevRunningMean->get_uid()] = prevMeanTensor.memory().deviceData();
        variantPack[prevRunningVar->get_uid()] = prevVarTensor.memory().deviceData();
        variantPack[nextRunningMean->get_uid()] = nextMeanTensor.memory().deviceData();
        variantPack[nextRunningVariance->get_uid()] = nextVarTensor.memory().deviceData();
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

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> yRefTensor(y->get_dim(), layout);
        utilities::Tensor<IntermediateType> savedMeanRefTensor(savedMean->get_dim());
        utilities::Tensor<IntermediateType> savedInvVarRefTensor(savedInvVariance->get_dim());

        if(config.useRunningStats)
        {
            utilities::Tensor<IntermediateType> nextMeanRefTensor(nextRunningMean->get_dim());
            utilities::Tensor<IntermediateType> nextVarRefTensor(nextRunningVariance->get_dim());

            hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::
                fwdTraining<InputType, IntermediateType, IntermediateType, InputType>(
                    xTensor,
                    scaleTensor,
                    biasTensor,
                    yRefTensor,
                    utilities::BATCHNORM_DEFAULT_EPSILON,
                    momentumVal,
                    &savedMeanRefTensor,
                    &savedInvVarRefTensor,
                    &prevMeanTensor,
                    &prevVarTensor,
                    &nextMeanRefTensor,
                    &nextVarRefTensor);

            auto tolerance
                = hipdnn_test_sdk::utilities::batchnorm::getToleranceTraining<InputType>();
            auto floatTolerance = static_cast<float>(tolerance);

            auto yValidator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(
                tolerance, tolerance);

            auto statsValidator
                = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<IntermediateType>(
                    static_cast<IntermediateType>(tolerance),
                    static_cast<IntermediateType>(tolerance));

            bool yValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
                std::cout, "y", yValidator, yRefTensor, yTensor, floatTolerance, floatTolerance);

            bool meanValid = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                std::cout,
                "saved_mean",
                statsValidator,
                savedMeanRefTensor,
                savedMeanTensor,
                floatTolerance,
                floatTolerance);

            bool invVarValid = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                std::cout,
                "saved_inv_variance",
                statsValidator,
                savedInvVarRefTensor,
                savedInvVarTensor,
                floatTolerance,
                floatTolerance);

            bool nextMeanValid = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                std::cout,
                "next_running_mean",
                statsValidator,
                nextMeanRefTensor,
                nextMeanTensor,
                floatTolerance,
                floatTolerance);

            bool nextVarValid = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                std::cout,
                "next_running_var",
                statsValidator,
                nextVarRefTensor,
                nextVarTensor,
                floatTolerance,
                floatTolerance);

            validationPassed = yValid && meanValid && invVarValid && nextMeanValid && nextVarValid;
        }
        else
        {
            hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::
                fwdTraining<InputType, IntermediateType, IntermediateType, InputType>(
                    xTensor,
                    scaleTensor,
                    biasTensor,
                    yRefTensor,
                    utilities::BATCHNORM_DEFAULT_EPSILON,
                    momentumVal,
                    &savedMeanRefTensor,
                    &savedInvVarRefTensor,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr);

            auto tolerance
                = hipdnn_test_sdk::utilities::batchnorm::getToleranceTraining<InputType>();
            auto floatTolerance = static_cast<float>(tolerance);

            auto yValidator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(
                tolerance, tolerance);

            auto statsValidator
                = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<IntermediateType>(
                    static_cast<IntermediateType>(tolerance),
                    static_cast<IntermediateType>(tolerance));

            bool yValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
                std::cout, "y", yValidator, yRefTensor, yTensor, floatTolerance, floatTolerance);

            bool meanValid = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                std::cout,
                "saved_mean",
                statsValidator,
                savedMeanRefTensor,
                savedMeanTensor,
                floatTolerance,
                floatTolerance);

            bool invVarValid = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(
                std::cout,
                "saved_inv_variance",
                statsValidator,
                savedInvVarRefTensor,
                savedInvVarTensor,
                floatTolerance,
                floatTolerance);

            validationPassed = yValid && meanValid && invVarValid;
        }
    }

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
        std::cout << static_cast<float>(yHostPtr[i]) << " ";

    std::cout << "\nFirst 10 saved_mean values: ";
    for(int i = 0; i < 10; ++i)
        std::cout << static_cast<float>(savedMeanHostPtr[i]) << " ";

    std::cout << "\nFirst 10 saved_inv_variance values: ";
    for(int i = 0; i < 10; ++i)
        std::cout << static_cast<float>(savedInvVarHostPtr[i]) << " ";

    if(config.useRunningStats)
    {
        auto nextMeanHostPtr = nextMeanTensor.memory().hostData();
        auto nextVarHostPtr = nextVarTensor.memory().hostData();

        std::cout << "\nFirst 10 next_running_mean values: ";
        for(int i = 0; i < 10; ++i)
            std::cout << static_cast<float>(nextMeanHostPtr[i]) << " ";

        std::cout << "\nFirst 10 next_running_variance values: ";
        for(int i = 0; i < 10; ++i)
            std::cout << static_cast<float>(nextVarHostPtr[i]) << " ";
    }

    std::cout << '\n';
    std::cout << "\nBatch normalization training graph execution complete for " << inputType
              << ".\n\n";

    return validationPassed;
}

int main(int argc, char* argv[])
{
    try
    {
        auto config = parseCommandLineArgs(argc, argv, SampleType::BN_TRAINING);

        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

    bool allPassed = run(SampleRunner{*handle, config}, config);

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
