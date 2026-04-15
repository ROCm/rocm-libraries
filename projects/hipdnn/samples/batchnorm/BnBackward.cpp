// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <iostream>
#include <string>
#include <unordered_map>

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

    std::cout << "Running batch normalization backwards graph " << inputType << " [" << layout
              << "]" << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    int64_t n = 16; // BATCH SIZE
    int64_t c = 16; // CHANNELS (FEATURES)
    int64_t h = 16; // HEIGHT (SPATIAL DIMENSION)
    int64_t w = 16; // WIDTH (SPATIAL DIMENSION)

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(intermediateType)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto dy = createTensor({n, c, h, w}, inputType, layout);
    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, intermediateType);
    auto savedMean = createTensor({1, c, 1, 1}, intermediateType);
    auto savedInvVariance = createTensor({1, c, 1, 1}, intermediateType);
    auto bnBwdAttributes = graph::BatchnormBackwardAttributes();
    bnBwdAttributes.set_name("bn_backward_node");
    bnBwdAttributes.set_saved_mean_and_inv_variance(savedMean, savedInvVariance);

    auto [dx, dscale, dbias] = graph->batchnorm_backward(dy, x, scale, bnBwdAttributes);
    dx->set_output(true);
    dscale->set_output(true).set_data_type(intermediateType);
    dbias->set_output(true).set_data_type(intermediateType);

    HIPDNN_FE_CHECK(graph->build(handle));
    std::cout << "Graph build successful.\n";

    utilities::Tensor<InputType> dyTensor(dy->get_dim(), layout);
    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
    utilities::Tensor<IntermediateType> savedMeanTensor(savedMean->get_dim());
    utilities::Tensor<IntermediateType> savedInvVarTensor(savedInvVariance->get_dim());
    utilities::Tensor<InputType> dxTensor(dx->get_dim(), layout);
    utilities::Tensor<IntermediateType> dscaleTensor(dscale->get_dim());
    utilities::Tensor<IntermediateType> dbiasTensor(dbias->get_dim());

    dyTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    scaleTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                     static_cast<IntermediateType>(1.0f));
    savedMeanTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                         static_cast<IntermediateType>(1.0f));
    savedInvVarTensor.fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                           static_cast<IntermediateType>(1.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[dy->get_uid()] = dyTensor.memory().deviceData();
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[savedMean->get_uid()] = savedMeanTensor.memory().deviceData();
    variantPack[savedInvVariance->get_uid()] = savedInvVarTensor.memory().deviceData();
    variantPack[dx->get_uid()] = dxTensor.memory().deviceData();
    variantPack[dscale->get_uid()] = dscaleTensor.memory().deviceData();
    variantPack[dbias->get_uid()] = dbiasTensor.memory().deviceData();

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));

    dxTensor.memory().markDeviceModified();
    dscaleTensor.memory().markDeviceModified();
    dbiasTensor.memory().markDeviceModified();

    auto dxHostPtr = dxTensor.memory().hostData();
    auto dscaleHostPtr = dscaleTensor.memory().hostData();
    auto dbiasHostPtr = dbiasTensor.memory().hostData();

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> dxRefTensor(dx->get_dim(), layout);
        utilities::Tensor<IntermediateType> dscaleRefTensor(dscale->get_dim());
        utilities::Tensor<IntermediateType> dbiasRefTensor(dbias->get_dim());

        hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::backward(dyTensor,
                                                                      xTensor,
                                                                      scaleTensor,
                                                                      dxRefTensor,
                                                                      dscaleRefTensor,
                                                                      dbiasRefTensor,
                                                                      &savedMeanTensor,
                                                                      &savedInvVarTensor);

        auto tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceBackward<InputType>();
        auto floatTolerance = static_cast<float>(tolerance);

        auto dxValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);
        auto dscaleDbiasValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<IntermediateType>(
                static_cast<IntermediateType>(tolerance), static_cast<IntermediateType>(tolerance));

        std::cout << "CPU reference validation:\n";
        bool dxValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "dx", dxValidator, dxRefTensor, dxTensor, floatTolerance, floatTolerance);
        bool dscaleValid
            = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(std::cout,
                                                                              "dscale",
                                                                              dscaleDbiasValidator,
                                                                              dscaleRefTensor,
                                                                              dscaleTensor,
                                                                              floatTolerance,
                                                                              floatTolerance);
        bool dbiasValid
            = hipdnn_test_sdk::utilities::validateAndReport<IntermediateType>(std::cout,
                                                                              "dbias",
                                                                              dscaleDbiasValidator,
                                                                              dbiasRefTensor,
                                                                              dbiasTensor,
                                                                              floatTolerance,
                                                                              floatTolerance);

        validationPassed = dxValid && dscaleValid && dbiasValid;
    }

    std::cout << "First 10 dx values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(dxHostPtr[i]) << " ";
    }
    std::cout << "\nFirst 10 dscale values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(dscaleHostPtr[i]) << " ";
    }
    std::cout << "\nFirst 10 dbias values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(dbiasHostPtr[i]) << " ";
    }

    std::cout << "\nBatch normalization backward graph execution complete for " << inputType
              << ".\n\n";
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
        std::cout << "All batch normalization backward runs completed successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "One or more batch normalization backward runs failed validation.\n";
        return 1;
    }
}
