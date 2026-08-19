// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <cstdio>
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

namespace
{

// SDPA-specific runner: iterates over data types with BHSD and BSHD layouts.
// Both layouts are controlled via strides on TensorAttributes. Filtering mirrors
// the shared run() in Helpers.hpp: an empty config.dtype/config.layout means
// "run all", otherwise only the requested combination is run.
template <typename F>
bool runSdpa(F&& f)
{
    bool allPassed = true;

    const std::vector<std::string> dtypes = {"fp32", "fp16", "bf16"};
    const std::vector<std::pair<std::string, TensorLayout>> layouts
        = {{"bhsd", TensorLayout::BHSD}, {"bshd", TensorLayout::BSHD}};

    for(const auto& dt : dtypes)
    {
        if(!f.config.dtype.empty() && f.config.dtype != dt)
        {
            continue;
        }

        for(const auto& [layoutName, layout] : layouts)
        {
            if(!f.config.layout.empty() && f.config.layout != layoutName)
            {
                continue;
            }

            if(dt == "fp32")
            {
                allPassed &= f.template operator()<float, float>(layout);
            }
            else if(dt == "fp16")
            {
                allPassed &= f.template operator()<half, float>(layout);
            }
            else if(dt == "bf16")
            {
                allPassed &= f.template operator()<bfloat16, float>(layout);
            }
        }
    }

    return allPassed;
}

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "Running SDPA forward graph " << inputType << " [" << layout << "]"
              << (config.cpuValidation ? " (with CPU validation)" : "")
              << (config.useRuntimePassByValue ? " [runtime-pass-by-value]" : "") << "...\n";

    // SDPA dimensions: [batch, num_heads, seq_len, head_dim]
    constexpr int64_t BATCH = 2;
    constexpr int64_t NUM_HEADS = 4;
    constexpr int64_t SEQ_LEN = 128;
    constexpr int64_t HEAD_DIM = 128;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    setPreferredEngine(graph, config);

    auto qAttr = createTensor({BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM}, inputType, layout);
    auto kAttr = createTensor({BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM}, inputType, layout);
    auto vAttr = createTensor({BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM}, inputType, layout);

    // Attn_scale is a pass-by-value scalar, not a device buffer.
    // Compile-time constant: value is baked into the op-graph at build() and requires no
    // variant-pack entry at execute(). Works with any plugin version.
    // Runtime pass-by-value (--runtime-pass-by-value): value is supplied as a host pointer in the
    // variant pack at execute(), allowing it to vary across executions without rebuilding the
    // graph. Requires plugin SDK >= 1.2.0.
    const float attnScaleDefault = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    auto attnScale = std::make_shared<graph::TensorAttributes>();
    attnScale->set_dim({1}).set_stride({1}).set_data_type(getDataTypeEnumFromType<float>());
    if(config.useRuntimePassByValue)
    {
        // No baked value; the value is supplied as a host pointer in the variant pack at
        // execute() instead.
        attnScale->set_as_runtime_parameter();
    }
    else
    {
        attnScale->set_compile_time_constant(attnScaleDefault);
    }

    graph::SdpaAttributes sdpaAttributes;
    sdpaAttributes.set_name("sdpa_fprop_node");
    sdpaAttributes.set_attn_scale(attnScale);

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

    // Using the same default here keeps output numerically identical to the compile-time path,
    // so CPU validation passes regardless of which mode is active. In a real application this
    // value could differ per execution without rebuilding the graph.
    float attnScaleVal = attnScaleDefault;
    if(config.useRuntimePassByValue)
    {
        variantPack[attnScale->get_uid()] = &attnScaleVal;
    }

    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));

    oTensor.memory().markDeviceModified();

    auto oHostPtr = oTensor.memory().hostData();

    std::cout << "First 10 output values: ";
    for(int i = 0; i < 10; ++i)
    {
        std::cout << static_cast<float>(oHostPtr[i]) << " ";
    }
    std::cout << '\n';

    // Structured bindings (oAttr, statsAttr above) cannot be captured directly by a C++17 lambda,
    // so bind a plain reference for use inside runCpuValidation below.
    auto& oAttrRef = oAttr;

    // Runs CPU reference validation against the current GPU output buffer. Called once after
    // the initial execute() and again after the runtime pass-by-value re-execution demo below,
    // so both scalar values get verified against the CPU reference.
    auto runCpuValidation = [&]() -> bool {
        std::cout << "Running CPU reference validation...\n";

        utilities::Tensor<InputType> oRefTensor(oAttrRef->get_dim(), layout);

        hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(
            qTensor, kTensor, vTensor, oRefTensor, attnScaleVal);

        // SDPA involves two matrix multiplies and softmax, requiring more generous
        // tolerances than single-operation validation.
        float tolerance = 0.0f;
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

        const bool oValid = oValidator.allClose(oRefTensor, oTensor);

        std::cout << "CPU reference validation:\n";
        std::cout << "  output: " << (oValid ? "successful" : "failed") << "\n";

        return oValid;
    };

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        validationPassed = runCpuValidation();
    }

    // Demonstrate that the graph can be re-executed with a different attn_scale value without
    // rebuilding. The variant pack already holds a pointer to attnScaleVal, so updating its
    // value in-place is all that is needed.
    if(config.useRuntimePassByValue)
    {
        std::cout << "Re-executing with attn_scale = 1.0 (no rebuild required)...\n";
        attnScaleVal = 1.0f;
        HIPDNN_FE_CHECK(graph->execute(handle, variantPack, workspace.get()));
        oTensor.memory().markDeviceModified();
        oHostPtr = oTensor.memory().hostData();

        std::cout << "First 10 output values (attn_scale = 1.0): ";
        for(int i = 0; i < 10; ++i)
        {
            std::cout << static_cast<float>(oHostPtr[i]) << " ";
        }
        std::cout << '\n';

        if(config.cpuValidation)
        {
            std::cout << "Re-validating against CPU reference with the updated attn_scale...\n";
            validationPassed = runCpuValidation() && validationPassed;
        }
    }

    std::cout << "SDPA forward graph execution complete for " << inputType << ".\n\n";
    return validationPassed;
}

int main(int argc, char* argv[])
{
    try
    {
        RETURN_SUCCESS_IF_NO_DEVICE();

        auto config = parseCommandLineArgs(argc, argv, SampleType::SDPA);

        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

        const bool allPassed = runSdpa(SampleRunner{*handle, config});

        if(allPassed)
        {
            std::cout << "All SDPA forward runs completed successfully.\n";
            return 0;
        }

        std::cout << "One or more SDPA forward runs failed validation.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
