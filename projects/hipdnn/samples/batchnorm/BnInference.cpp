// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "../utils/Helpers.hpp"

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>

#include <iostream>
#include <string>
#include <unordered_map>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;

template <typename InputType, typename IntermediateType>
void SampleRunner::operator()(const TensorLayout& layout)
{
    auto inputType = getDataTypeEnumFromType<InputType>();
    auto intermediateType = getDataTypeEnumFromType<IntermediateType>();

    std::cout << "Running batch normalization inference graph " << inputType << " [" << layout
              << "]" << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

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
    auto mean = createTensor({1, c, 1, 1}, intermediateType);
    auto invVariance = createTensor({1, c, 1, 1}, intermediateType);

    auto bnAttributes = graph::BatchnormInferenceAttributes();
    bnAttributes.set_name("bn_inference_node");

    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);
    y->set_output(true).set_data_type(inputType);

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

    Tensor<InputType> xTensor(x->get_dim(), layout);
    Tensor<IntermediateType> scaleTensor(scale->get_dim());
    Tensor<IntermediateType> biasTensor(bias->get_dim());
    Tensor<IntermediateType> meanTensor(mean->get_dim());
    Tensor<IntermediateType> invVarianceTensor(invVariance->get_dim());
    Tensor<InputType> yTensor(y->get_dim(), layout);

    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));

    scaleTensor.fillWithValue(static_cast<IntermediateType>(1.0f));

    biasTensor.fillWithValue(static_cast<IntermediateType>(0.0f));

    meanTensor.fillWithValue(static_cast<IntermediateType>(0.5f));

    invVarianceTensor.fillWithValue(static_cast<IntermediateType>(1.0f));

    std::unordered_map<int64_t, void*> variantPack;

    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[mean->get_uid()] = meanTensor.memory().deviceData();
    variantPack[invVariance->get_uid()] = invVarianceTensor.memory().deviceData();
    variantPack[y->get_uid()] = yTensor.memory().deviceData();

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));

    yTensor.memory().markDeviceModified();
    auto yHostPtr = yTensor.memory().hostData();

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        Tensor<InputType> yRefTensor(y->get_dim(), layout);

        // Convert inverse variance to variance for CPU reference
        Tensor<IntermediateType> varianceTensor(invVariance->get_dim());
        auto invVarianceHostPtr = invVarianceTensor.memory().hostData();
        auto varianceHostPtr = varianceTensor.memory().hostData();

        for(size_t i = 0; i < invVarianceTensor.memory().count(); ++i)
        {
            varianceHostPtr[i] = static_cast<IntermediateType>(1.0f)
                                 / (invVarianceHostPtr[i] * invVarianceHostPtr[i]);
        }

        auto epsilon = getEpsilon<InputType>();

        hipdnn_sdk::test_utilities::CpuFpReferenceBatchnormImpl<InputType, IntermediateType>::
            batchnormFwdInference(
                xTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, yRefTensor, epsilon);

        auto validator = hipdnn_sdk::test_utilities::CpuFpReferenceValidation<InputType>(
            static_cast<InputType>(epsilon), static_cast<InputType>(epsilon));

        std::cout << "CPU reference validation "
                  << (validator.allClose(yRefTensor.memory(), yTensor.memory()) ? "successful"
                                                                                : "failed")
                  << ".\n";
    }

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(yHostPtr[i]) << " ";
    }

    std::cout << "\nBatch normalization inference graph execution complete for " << inputType
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
    std::cout << "All batch normalization inference runs completed successfully.\n";
    return 0;
}
