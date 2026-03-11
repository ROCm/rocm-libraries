// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ReluPlan.hpp"

#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/ExamplePluginUtils.hpp"

namespace example_plugin
{

ReluPlan::ReluPlan(int64_t inputUid,
                   int64_t outputUid,
                   int64_t numElements,
                   double negativeSlope,
                   const IKernelCompiler& compiler)
    : _inputUid(inputUid)
    , _outputUid(outputUid)
    , _numElements(numElements)
    , _negativeSlope(negativeSlope)
    , _compiler(compiler)
{
}

void ReluPlan::compile(const hipDeviceProp_t& deviceProperties)
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

    _compiledProgram = _compiler.compile("ReluForward.cpp", {"--offload-arch=" + archName});
    _kernel = _compiledProgram->getKernel("relu_forward_kernel");
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
    auto inputBuffer = findDeviceBuffer(_inputUid, deviceBuffers, numDeviceBuffers);
    auto outputBuffer = findDeviceBuffer(_outputUid, deviceBuffers, numDeviceBuffers);

    auto* input = static_cast<const float*>(inputBuffer.ptr);
    auto* output = static_cast<float*>(outputBuffer.ptr);

    static constexpr unsigned int kBlockSize = 256;
    auto numElementsU = static_cast<unsigned int>(_numElements);
    unsigned int gridSize = (numElementsU + kBlockSize - 1) / kBlockSize;

    _kernel->setBlockSize(kBlockSize, 1, 1);
    _kernel->setGridSize(gridSize, 1, 1);

    auto negSlope = static_cast<float>(_negativeSlope);
    _kernel->launch(nullptr, input, output, numElementsU, negSlope);
}

} // namespace example_plugin
