// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ReluPlan.hpp"

#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/ExamplePluginUtils.hpp"
#include "hip/IKernelCompiler.hpp"

namespace example_plugin
{

ReluPlan::ReluPlan(ReluParams&& params)
    : _params(std::move(params))
{
}

void ReluPlan::compile(const IKernelCompiler& kernelCompiler,
                       const hipDeviceProp_t& deviceProperties)
{
    // Extract base GPU architecture from gcnArchName
    // e.g., "gfx90a:sramecc+:xnack-" -> "gfx90a"
    std::string archName(deviceProperties.gcnArchName);
    auto colonPos = archName.find(':');
    if(colonPos != std::string::npos)
    {
        archName = archName.substr(0, colonPos);
    }

    HIPDNN_PLUGIN_LOG_INFO("Compiling ReluPlan for architecture: " << archName);

    _compiledProgram = kernelCompiler.compile("ReluForward.cpp", {"--offload-arch=" + archName});
    _kernel = _compiledProgram->getRunnableKernel("relu_forward_kernel");
}

size_t ReluPlan::getWorkspaceSize(const ExamplePluginHandle& /*handle*/) const
{
    return 0;
}

void ReluPlan::execute(const ExamplePluginHandle& /*handle*/,
                       const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                       uint32_t numDeviceBuffers,
                       void* /*workspace*/) const
{
    auto inputBuffer = findDeviceBuffer(_params.inputUid, deviceBuffers, numDeviceBuffers);
    auto outputBuffer = findDeviceBuffer(_params.outputUid, deviceBuffers, numDeviceBuffers);

    auto* input = static_cast<const float*>(inputBuffer.ptr);
    auto* output = static_cast<float*>(outputBuffer.ptr);

    static constexpr unsigned int kBlockSize = 256;
    auto numElementsU = static_cast<unsigned int>(_params.numElements);
    unsigned int gridSize = (numElementsU + kBlockSize - 1) / kBlockSize;

    _kernel->setBlockSize(kBlockSize, 1, 1);
    _kernel->setGridSize(gridSize, 1, 1);

    auto negSlope = static_cast<float>(_params.negativeSlope);
    _kernel->launch(nullptr, input, output, numElementsU, negSlope);
}

} // namespace example_plugin
