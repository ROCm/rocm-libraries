// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ReluPlan.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/ExampleProviderUtils.hpp"
#include "hip/IKernelCompiler.hpp"

#include <limits>

namespace example_provider
{

ReluPlan::ReluPlan(const ReluParams& params)
    : _params(params)
{
}

void ReluPlan::compile(const IKernelCompiler& kernelCompiler)
{
    HIPDNN_PLUGIN_LOG_INFO("Compiling ReluPlan");

    _compiledProgram = kernelCompiler.compile("ReluForward.cpp", {});
    _kernel = _compiledProgram->getRunnableKernel("relu_forward_kernel");
}

size_t ReluPlan::getWorkspaceSize(const ExampleProviderHandle& /*handle*/) const
{
    return 0;
}

void ReluPlan::execute(const ExampleProviderHandle& handle,
                       const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                       uint32_t numDeviceBuffers,
                       void* /*workspace*/) const
{
    auto inputBuffer = findDeviceBuffer(_params.inputUid, deviceBuffers, numDeviceBuffers);
    auto outputBuffer = findDeviceBuffer(_params.outputUid, deviceBuffers, numDeviceBuffers);

    auto* input = static_cast<const float*>(inputBuffer.ptr);
    auto* output = static_cast<float*>(outputBuffer.ptr);

    if(_params.numElements > std::numeric_limits<unsigned int>::max())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "Number of elements exceeds unsigned int maximum");
    }

    static constexpr unsigned int BLOCK_SIZE = 256;
    auto numElementsU = static_cast<unsigned int>(_params.numElements);
    const unsigned int gridSize = (numElementsU + BLOCK_SIZE - 1) / BLOCK_SIZE;

    _kernel->setBlockSize(BLOCK_SIZE, 1, 1);
    _kernel->setGridSize(gridSize, 1, 1);

    auto negSlope = static_cast<float>(_params.negativeSlope);
    _kernel->launch(handle.getStream(), input, output, numElementsU, negSlope);
}

} // namespace example_provider
