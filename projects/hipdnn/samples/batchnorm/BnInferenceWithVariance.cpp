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

    std::cout << "Running batch normalization inference with variance graph " << inputType << " ["
              << layout << "]" << (config.cpuValidation ? " (with CPU validation)" : "")
              << (config.useRuntimePassByValue ? " [runtime-pass-by-value]" : "") << "...\n";

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
    auto mean = createTensor({1, c, 1, 1}, intermediateType);
    auto variance = createTensor({1, c, 1, 1}, intermediateType);
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

    auto bnAttributes = graph::BatchnormInferenceAttributesVarianceExt();
    bnAttributes.set_name("bn_inference_variance_ext_node");

    auto y = graph->batchnorm_inference_variance_ext(
        x, mean, variance, scale, bias, epsilon, bnAttributes);
    y->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));

    std::cout << "Graph build successful.\n";

    utilities::Tensor<InputType> xTensor(x->get_dim(), layout);
    utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
    utilities::Tensor<IntermediateType> biasTensor(bias->get_dim());
    utilities::Tensor<IntermediateType> meanTensor(mean->get_dim());
    utilities::Tensor<IntermediateType> varianceTensor(variance->get_dim());
    utilities::Tensor<InputType> yTensor(y->get_dim(), layout);

    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    scaleTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                     static_cast<IntermediateType>(1.0f));
    biasTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                    static_cast<IntermediateType>(1.0f));
    meanTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                    static_cast<IntermediateType>(1.0f));
    varianceTensor.fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                        static_cast<IntermediateType>(1.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[mean->get_uid()] = meanTensor.memory().deviceData();
    variantPack[variance->get_uid()] = varianceTensor.memory().deviceData();
    variantPack[y->get_uid()] = yTensor.memory().deviceData();

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

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));

    yTensor.memory().markDeviceModified();

    auto yHostPtr = yTensor.memory().hostData();

    auto runCpuValidation = [&]() -> bool {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> yRefTensor(y->get_dim(), layout);

        auto tolerance
            = hipdnn_test_sdk::utilities::batchnorm::getToleranceInferenceWithVariance<InputType>();

        hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
            xTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, yRefTensor, epsilonVal);

        auto validator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        std::cout << "CPU reference validation:\n";

        return hipdnn_test_sdk::utilities::validateAndReport<InputType>(
            std::cout, "y", validator, yRefTensor, yTensor, tolerance, tolerance);
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

    // Demonstrate that the graph can be re-executed with a different scalar value without
    // rebuilding. The variant pack already holds a pointer to epsilonVal, so updating the
    // value in-place is all that is needed.
    if(config.useRuntimePassByValue)
    {
        std::cout << "\nRe-executing with epsilon = 1.0 (no rebuild required)...\n";
        epsilonVal = 1.0;
        HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));
        yTensor.memory().markDeviceModified();
        yHostPtr = yTensor.memory().hostData();

        std::cout << "First 10 y values (epsilon = 1.0): ";
        for(int i = 0; i < 10; ++i)
        {
            std::cout << static_cast<float>(yHostPtr[i]) << " ";
        }

        if(config.cpuValidation)
        {
            std::cout << "\nRe-validating against CPU reference with the updated scalar value...\n";
            validationPassed = runCpuValidation() && validationPassed;
        }
    }

    std::cout << "\nBatch normalization inference with variance graph execution complete for "
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
                << "All batch normalization inference with variance runs completed successfully.\n";
            return 0;
        }

        std::cout
            << "One or more batch normalization inference with variance runs failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
