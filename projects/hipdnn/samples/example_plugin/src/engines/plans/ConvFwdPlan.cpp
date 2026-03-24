// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvFwdPlan.hpp"

#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/ExamplePluginUtils.hpp"
#include "hip/IKernelCompiler.hpp"

namespace example_plugin
{

ConvFwdPlan::ConvFwdPlan(ConvFwdParams&& params)
    : _params(std::move(params))
{
}

void ConvFwdPlan::compile(const IKernelCompiler& kernelCompiler,
                          const hipDeviceProp_t& deviceProperties)
{
    // Extract base GPU architecture from gcnArchName
    std::string archName(deviceProperties.gcnArchName);
    auto colonPos = archName.find(':');
    if(colonPos != std::string::npos)
    {
        archName = archName.substr(0, colonPos);
    }

    HIPDNN_PLUGIN_LOG_INFO("Compiling ConvFwdPlan for architecture: " << archName);

    _compiledProgram
        = kernelCompiler.compile("ConvForwardNaive.cpp", {"--offload-arch=" + archName});
    _kernel = _compiledProgram->getKernel("conv_forward_naive_kernel");
}

size_t ConvFwdPlan::getWorkspaceSize(const ExamplePluginHandle& /*handle*/) const
{
    return 0;
}

void ConvFwdPlan::execute(const ExamplePluginHandle& /*handle*/,
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

    // Total output elements: N * K * outH * outW
    auto totalOutputElements
        = static_cast<unsigned int>(_params.n * _params.k * _params.outH * _params.outW);
    auto blockSizeU = static_cast<unsigned int>(_params.blockSize);
    unsigned int gridSize = (totalOutputElements + blockSizeU - 1) / blockSizeU;

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

    _kernel->launch(nullptr,
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

} // namespace example_plugin
