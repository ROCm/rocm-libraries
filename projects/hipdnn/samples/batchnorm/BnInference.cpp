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

    std::cout << "Running batch normalization inference graph " << inputType << " [" << layout
              << "]" << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    const int64_t n = 16; // BATCH SIZE
    const int64_t c = 16; // CHANNELS (FEATURES)
    const int64_t h = 16; // HEIGHT (SPATIAL DIMENSION)
    const int64_t w = 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto bias = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto mean = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto invVariance = createTensor({1, c, 1, 1}, intermediateType, layout);

    auto bnAttributes = graph::BatchnormInferenceAttributes();
    bnAttributes.set_name("bn_inference_node");

    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);
    y->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful.\n";

    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
    utilities::Tensor<IntermediateType> biasTensor(bias->get_dim());
    utilities::Tensor<IntermediateType> meanTensor(mean->get_dim());
    utilities::Tensor<IntermediateType> invVarianceTensor(invVariance->get_dim());
    utilities::Tensor<InputType> yTensor(y->get_dim(), layout);

    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    scaleTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                     static_cast<IntermediateType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                    static_cast<IntermediateType>(1.0f));
    meanTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                    static_cast<IntermediateType>(1.0f));
    invVarianceTensor.fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                           static_cast<IntermediateType>(1.0f));

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

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> yRefTensor(y->get_dim(), layout);

        auto tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceInference<InputType>();

        hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::fwdInference(
            xTensor, scaleTensor, biasTensor, meanTensor, invVarianceTensor, yRefTensor);

        auto validator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        std::cout << "CPU reference validation:\n";
        const bool yValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "y", validator, yRefTensor, yTensor, tolerance, tolerance);

        validationPassed = yValid;
    }

    std::cout << "First 10 y values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(yHostPtr[i]) << " ";
    }

    std::cout << "\nBatch normalization inference graph execution complete for " << inputType
              << ".\n\n";
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
            std::cout << "All batch normalization inference runs completed successfully.\n";
            return 0;
        }
        std::cout << "One or more batch normalization inference runs failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
