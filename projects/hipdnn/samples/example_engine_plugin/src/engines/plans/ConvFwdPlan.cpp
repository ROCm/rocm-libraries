// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvFwdPlan.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/ExampleProviderUtils.hpp"
#include "hip/IKernelCompiler.hpp"

#include <limits>

namespace example_provider
{

ConvFwdPlan::ConvFwdPlan(const ConvFwdParams& params)
    : _params(params)
{
}

void ConvFwdPlan::compile(const IKernelCompiler& kernelCompiler)
{
    HIPDNN_PLUGIN_LOG_INFO("Compiling ConvFwdPlan");

    _compiledProgram = kernelCompiler.compile("ConvForwardNaive.cpp", {});
    _kernel = _compiledProgram->getRunnableKernel("conv_forward_naive_kernel");
}

size_t ConvFwdPlan::getWorkspaceSize(const ExampleProviderHandle& /*handle*/) const
{
    return 0;
}

void ConvFwdPlan::execute(const ExampleProviderHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* /*workspace*/) const
{
    auto inputBuffer = findDeviceBuffer(_params.inputUid, deviceBuffers, numDeviceBuffers);
    auto weightBuffer = findDeviceBuffer(_params.weightUid, deviceBuffers, numDeviceBuffers);
    auto outputBuffer = findDeviceBuffer(_params.outputUid, deviceBuffers, numDeviceBuffers);

    auto* input = static_cast<const float*>(inputBuffer.ptr);
    auto* weight = static_cast<const float*>(weightBuffer.ptr);
    auto* output = static_cast<float*>(outputBuffer.ptr);

    const int64_t totalOutputElementsI64 = _params.n * _params.k * _params.outH * _params.outW;
    if(totalOutputElementsI64 > std::numeric_limits<unsigned int>::max())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "Total output elements exceeds unsigned int maximum");
    }

    auto totalOutputElements = static_cast<unsigned int>(totalOutputElementsI64);
    auto blockSizeU = static_cast<unsigned int>(_params.blockSize);
    const unsigned int gridSize = (totalOutputElements + blockSizeU - 1) / blockSizeU;

    _kernel->setBlockSize(blockSizeU, 1, 1);
    _kernel->setGridSize(gridSize, 1, 1);

    auto n = static_cast<int>(_params.n);
    auto c = static_cast<int>(_params.c);
    auto h = static_cast<int>(_params.h);
    auto w = static_cast<int>(_params.w);
    auto k = static_cast<int>(_params.k);
    auto r = static_cast<int>(_params.r);
    auto s = static_cast<int>(_params.s);
    auto outH = static_cast<int>(_params.outH);
    auto outW = static_cast<int>(_params.outW);
    auto padH = static_cast<int>(_params.padH);
    auto padW = static_cast<int>(_params.padW);
    auto strideH = static_cast<int>(_params.strideH);
    auto strideW = static_cast<int>(_params.strideW);

    _kernel->launch(handle.getStream(),
                    input,
                    weight,
                    output,
                    n,
                    c,
                    h,
                    w,
                    k,
                    r,
                    s,
                    outH,
                    outW,
                    padH,
                    padW,
                    strideH,
                    strideW);
}

} // namespace example_provider
