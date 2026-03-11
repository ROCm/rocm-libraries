// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvFwdPlan.hpp"

#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/ExamplePluginUtils.hpp"

namespace example_plugin
{

ConvFwdPlan::ConvFwdPlan(int64_t inputUid,
                         int64_t weightUid,
                         int64_t outputUid,
                         int64_t n,
                         int64_t c,
                         int64_t h,
                         int64_t w,
                         int64_t k,
                         int64_t r,
                         int64_t s,
                         int64_t outH,
                         int64_t outW,
                         int64_t padH,
                         int64_t padW,
                         int64_t strideH,
                         int64_t strideW,
                         int64_t blockSize,
                         const IKernelCompiler& compiler)
    : _inputUid(inputUid)
    , _weightUid(weightUid)
    , _outputUid(outputUid)
    , _n(n)
    , _c(c)
    , _h(h)
    , _w(w)
    , _k(k)
    , _r(r)
    , _s(s)
    , _outH(outH)
    , _outW(outW)
    , _padH(padH)
    , _padW(padW)
    , _strideH(strideH)
    , _strideW(strideW)
    , _blockSize(blockSize)
    , _compiler(compiler)
{
}

void ConvFwdPlan::compile(const hipDeviceProp_t& deviceProperties)
{
    // Extract base GPU architecture from gcnArchName
    std::string archName(deviceProperties.gcnArchName);
    auto colonPos = archName.find(':');
    if(colonPos != std::string::npos)
    {
        archName = archName.substr(0, colonPos);
    }

    HIPDNN_PLUGIN_LOG_INFO("Compiling ConvFwdPlan for architecture: " << archName);

    _compiledProgram = _compiler.compile("ConvForwardNaive.cpp", {"--offload-arch=" + archName});
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
    auto inputBuffer = findDeviceBuffer(_inputUid, deviceBuffers, numDeviceBuffers);
    auto weightBuffer = findDeviceBuffer(_weightUid, deviceBuffers, numDeviceBuffers);
    auto outputBuffer = findDeviceBuffer(_outputUid, deviceBuffers, numDeviceBuffers);

    auto* input = static_cast<const float*>(inputBuffer.ptr);
    auto* weight = static_cast<const float*>(weightBuffer.ptr);
    auto* output = static_cast<float*>(outputBuffer.ptr);

    // Total output elements: N * K * outH * outW
    auto totalOutputElements = static_cast<unsigned int>(_n * _k * _outH * _outW);
    auto blockSizeU = static_cast<unsigned int>(_blockSize);
    unsigned int gridSize = (totalOutputElements + blockSizeU - 1) / blockSizeU;

    _kernel->setBlockSize(blockSizeU, 1, 1);
    _kernel->setGridSize(gridSize, 1, 1);

    auto n = static_cast<int>(_n);
    auto c = static_cast<int>(_c);
    auto h = static_cast<int>(_h);
    auto w = static_cast<int>(_w);
    auto k = static_cast<int>(_k);
    auto r = static_cast<int>(_r);
    auto s = static_cast<int>(_s);
    auto outH = static_cast<int>(_outH);
    auto outW = static_cast<int>(_outW);
    auto padH = static_cast<int>(_padH);
    auto padW = static_cast<int>(_padW);
    auto strideH = static_cast<int>(_strideH);
    auto strideW = static_cast<int>(_strideW);

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
