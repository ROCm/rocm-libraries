// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

// SDPA-specific runner: iterates over data types with BHSD and BSHD layouts.
// Both layouts are controlled via strides on TensorAttributes.
template <typename F>
bool runSdpa(F&& f)
{
    bool allPassed = true;
    allPassed &= f.template operator()<float, float>(TensorLayout::BHSD);
    allPassed &= f.template operator()<half, float>(TensorLayout::BHSD);
    allPassed &= f.template operator()<bfloat16, float>(TensorLayout::BHSD);
    allPassed &= f.template operator()<float, float>(TensorLayout::BSHD);
    allPassed &= f.template operator()<half, float>(TensorLayout::BSHD);
    allPassed &= f.template operator()<bfloat16, float>(TensorLayout::BSHD);
    return allPassed;
}

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running SDPA forward graph " << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    // SDPA dimensions: [batch, num_heads, seq_len, head_dim]
    constexpr int64_t batch = 2;
    constexpr int64_t numHeads = 4;
    constexpr int64_t seqLen = 128;
    constexpr int64_t headDim = 128;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto qAttr = createTensor({batch, numHeads, seqLen, headDim}, inputType, layout);
    auto kAttr = createTensor({batch, numHeads, seqLen, headDim}, inputType, layout);
    auto vAttr = createTensor({batch, numHeads, seqLen, headDim}, inputType, layout);

    graph::SdpaAttributes sdpaAttributes;
    sdpaAttributes.set_name("sdpa_fprop_node");
    sdpaAttributes.set_attn_scale_value(1.0f / std::sqrt(static_cast<float>(headDim)));

    auto [oAttr, statsAttr] = graph->sdpa(qAttr, kAttr, vAttr, std::move(sdpaAttributes));
    oAttr->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
    std::cout << "Graph build successful.\n";

    utilities::Tensor<InputType> qTensor(qAttr->get_dim(), layout);
    utilities::Tensor<InputType> kTensor(kAttr->get_dim(), layout);
    utilities::Tensor<InputType> vTensor(vAttr->get_dim(), layout);
    utilities::Tensor<InputType> oTensor(oAttr->get_dim(), layout);

    qTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    kTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    vTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    oTensor.fillWithValue(static_cast<InputType>(0.0f));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[qAttr->get_uid()] = qTensor.memory().deviceData();
    variantPack[kAttr->get_uid()] = kTensor.memory().deviceData();
    variantPack[vAttr->get_uid()] = vTensor.memory().deviceData();
    variantPack[oAttr->get_uid()] = oTensor.memory().deviceData();

    int64_t workspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    oTensor.memory().markDeviceModified();

    auto oHostPtr = oTensor.memory().hostData();

    std::cout << "First 10 output values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(oHostPtr[i]) << " ";
    }
    std::cout << '\n';

    bool validationPassed = true;

    if(config.cpuValidation)
    {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> oRefTensor(oAttr->get_dim(), layout);

        auto attnScale = 1.0f / std::sqrt(static_cast<float>(headDim));

        hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(
            qTensor, kTensor, vTensor, oRefTensor, attnScale);

        // SDPA involves two matrix multiplies and softmax, requiring more generous
        // tolerances than single-operation validation.
        float tolerance;
        if constexpr(std::is_same_v<InputType, float>)
        {
            tolerance = 2e-4f;
        }
        else if constexpr(std::is_same_v<InputType, half>)
        {
            tolerance = 5e-3f;
        }
        else
        {
            tolerance = 1e-2f;
        }

        auto oValidator
            = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);

        bool oValid = oValidator.allClose(oRefTensor, oTensor);

        std::cout << "CPU reference validation:\n";
        std::cout << "  output: " << (oValid ? "successful" : "failed") << "\n";

        validationPassed = oValid;
    }

    std::cout << "SDPA forward graph execution complete for " << inputType << ".\n\n";
    return validationPassed;
}

int main(int argc, char* argv[])
{
    auto config = parseCommandLineArgs(argc, argv);

    auto [handle, handleError] = createHipdnnHandle();
    HIPDNN_FE_CHECK(handleError);

    bool allPassed = runSdpa(SampleRunner{*handle, config});

    if(allPassed)
    {
        std::cout << "All SDPA forward runs completed successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "One or more SDPA forward runs failed validation.\n";
        return 1;
    }
}
