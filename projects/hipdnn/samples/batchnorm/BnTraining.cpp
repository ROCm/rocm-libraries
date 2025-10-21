// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_frontend.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/utilities/Constants.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk;

// TODO: verify this sample when applicable engines are added
template <typename InputType, typename IntermediateType>
void SampleRunner::operator()(const TensorLayout& layout)
{
    auto inputType = getDataTypeEnumFromType<InputType>();
    auto intermediateType = getDataTypeEnumFromType<IntermediateType>();

    std::cout << "Running batch normalization training graph " << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    int64_t n = 16; // BATCH SIZE
    int64_t c = 16; // CHANNELS (FEATURES)
    int64_t h = 16; // HEIGHT (SPATIAL DIMENSION)
    int64_t w = 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(intermediateType);

    auto x = createTensor({n, c, h, w}, inputType);
    auto scale = createTensor({1, c, 1, 1}, intermediateType);
    auto bias = createTensor({1, c, 1, 1}, intermediateType);
    auto prevRunningMean = createTensor({1, c, 1, 1}, intermediateType);
    auto prevRunningVar = createTensor({1, c, 1, 1}, intermediateType);
    auto momentum = createTensor({1, 1, 1, 1}, intermediateType);
    auto epsilon = createTensor({1, 1, 1, 1}, intermediateType);

    auto bnAttributes = graph::BatchnormAttributes();
    bnAttributes.set_name("bn_training_node");
    bnAttributes.set_previous_running_stats(prevRunningMean, prevRunningVar, momentum)
        .set_epsilon(epsilon);

    auto [y, nextRunningMean, nextRunningVar, savedMean, savedInvVariance]
        = graph->batchnorm(x, scale, bias, bnAttributes);

    y->set_output(true);
    nextRunningMean->set_output(true);
    nextRunningVar->set_output(true);
    savedMean->set_output(true);
    savedInvVariance->set_output(true);

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

    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
    utilities::Tensor<IntermediateType> biasTensor(bias->get_dim());
    utilities::Tensor<IntermediateType> prevMeanTensor(prevRunningMean->get_dim());
    utilities::Tensor<IntermediateType> prevVarTensor(prevRunningVar->get_dim());
    utilities::Tensor<IntermediateType> momentumTensor(momentum->get_dim());
    utilities::Tensor<IntermediateType> epsilonTensor(epsilon->get_dim());

    utilities::Tensor<InputType> yTensor(y->get_dim(), layout);
    utilities::Tensor<IntermediateType> nextMeanTensor(nextRunningMean->get_dim());
    utilities::Tensor<IntermediateType> nextVarTensor(nextRunningVar->get_dim());
    utilities::Tensor<IntermediateType> savedMeanTensor(savedMean->get_dim());
    utilities::Tensor<IntermediateType> savedInvVarTensor(savedInvVariance->get_dim());

    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    scaleTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                     static_cast<IntermediateType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                    static_cast<IntermediateType>(1.0f));
    prevMeanTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                        static_cast<IntermediateType>(1.0f));
    prevVarTensor.fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                       static_cast<IntermediateType>(1.0f));

    momentumTensor.memory().hostData()[0] = 0.1f;
    epsilonTensor.memory().hostData()[0] = utilities::BATCHNORM_DEFAULT_EPSILON;

    std::unordered_map<int64_t, void*> variantPack;

    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[prevRunningMean->get_uid()] = prevMeanTensor.memory().deviceData();
    variantPack[prevRunningVar->get_uid()] = prevVarTensor.memory().deviceData();
    variantPack[momentum->get_uid()] = momentumTensor.memory().deviceData();
    variantPack[epsilon->get_uid()] = epsilonTensor.memory().deviceData();
    variantPack[y->get_uid()] = yTensor.memory().deviceData();
    variantPack[nextRunningMean->get_uid()] = nextMeanTensor.memory().deviceData();
    variantPack[nextRunningVar->get_uid()] = nextVarTensor.memory().deviceData();
    variantPack[savedMean->get_uid()] = savedMeanTensor.memory().deviceData();
    variantPack[savedInvVariance->get_uid()] = savedInvVarTensor.memory().deviceData();

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));

    yTensor.memory().markDeviceModified();
    nextMeanTensor.memory().markDeviceModified();
    nextVarTensor.memory().markDeviceModified();
    savedMeanTensor.memory().markDeviceModified();
    savedInvVarTensor.memory().markDeviceModified();

    auto yHostPtr = yTensor.memory().hostData();

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> yRefTensor(y->get_dim(), layout);
        utilities::Tensor<IntermediateType> nextMeanRefTensor(nextRunningMean->get_dim());
        utilities::Tensor<IntermediateType> nextVarRefTensor(nextRunningVar->get_dim());
        utilities::Tensor<IntermediateType> savedMeanRefTensor(savedMean->get_dim());
        utilities::Tensor<IntermediateType> savedInvVarRefTensor(savedInvVariance->get_dim());

        // TODO: Uncomment when CPU reference implemented
        // CpuFpReferenceBatchnormImpl<InputType, IntermediateType>::batchnorm_fwd_training(x_tensor,
        //                                scale_tensor,
        //                                bias_tensor,
        //                                prev_mean_tensor,
        //                                prev_var_tensor,
        //                                momentum_tensor,
        //                                epsilon_tensor,
        //                                y_ref_tensor,
        //                                next_mean_ref_tensor,
        //                                next_var_ref_tensor,
        //                                saved_mean_ref_tensor,
        //                                saved_inv_var_ref_tensor);

        // auto tolerance = test_utilities::batchnorm::getToleranceTraining<InputType>();
        //
        // auto y_validator
        //     = test_utilities::CpuFpReferenceValidation<InputType>(
        //         tolerance, tolerance);
        //
        // auto stats_validator
        //     = test_utilities::CpuFpReferenceValidation<IntermediateType>(
        //         static_cast<IntermediateType>(tolerance), static_cast<IntermediateType>(tolerance));
        // bool y_valid = y_validator.allClose(y_ref_tensor.memory(), y_tensor.memory());
        // bool next_mean_valid = stats_validator.allClose(next_mean_ref_tensor.memory(),
        //                                                        next_mean_tensor.memory());
        // bool next_var_valid = stats_validator.allClose(next_var_ref_tensor.memory(),
        //                                                       next_var_tensor.memory());
        // TODO: consider adding validation for other output buffers, but they are verified indirectly by y
        // std::cout << "CPU reference validation:\n";
        // std::cout << "  y: " << (y_valid ? "successful" : "failed") << "\n";
        // std::cout << "  next_running_mean: " << (next_mean_valid ? "successful" : "failed") << "\n";
        // std::cout << "  next_running_var: " << (next_var_valid ? "successful" : "failed") << "\n";

        std::cout << "CPU reference validation skipped - batchnorm training forward not yet "
                     "implemented.\n";
    }

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(yHostPtr[i]) << " ";
    }

    std::cout << "\nBatch normalization training graph execution complete for " << inputType
              << ".\n\n";
}

int main(int argc, char* argv[])
{
    auto config = parseCommandLineArgs(argc, argv);

    initializeFrontendLogging();

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    run(SampleRunner{handle, config});

    HIPDNN_CHECK(hipdnnDestroy(handle));
    std::cout << "All batch normalization training runs completed.\n";
    return 0;
}
