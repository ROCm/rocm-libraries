// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/ConvFwdPlan.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <unordered_map>

namespace rocke_conv_engine
{

// ---------------------------------------------------------------------------
// ConvModuleGuard
// ---------------------------------------------------------------------------

ConvModuleGuard::~ConvModuleGuard()
{
    if(_module != nullptr)
    {
        auto err = hipModuleUnload(_module);
        if(err != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "ConvModuleGuard: hipModuleUnload failed: " << hipGetErrorString(err));
        }
    }
}

ConvModuleGuard& ConvModuleGuard::operator=(ConvModuleGuard&& other) noexcept
{
    if(this != &other)
    {
        if(_module != nullptr)
        {
            auto err = hipModuleUnload(_module);
            if(err != hipSuccess)
            {
                HIPDNN_PLUGIN_LOG_ERROR("ConvModuleGuard: hipModuleUnload failed during move: "
                                        << hipGetErrorString(err));
            }
        }
        _module = other._module;
        _function = other._function;
        other._module = nullptr;
        other._function = nullptr;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// ConvFwdPlan
// ---------------------------------------------------------------------------

ConvFwdPlan::ConvFwdPlan(ConvModuleGuard kernel, ConvFwdParams params)
    : _kernel(std::move(kernel))
    , _params(std::move(params))
{
}

size_t ConvFwdPlan::getWorkspaceSize(const Handle& /*handle*/) const
{
    return 0;
}

void ConvFwdPlan::execute(const Handle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* /*workspace*/) const
{
    // Build UID -> device pointer map
    std::unordered_map<int64_t, void*> uidToPtr;
    uidToPtr.reserve(numDeviceBuffers);
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtr[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    auto lookupPtr = [&](int64_t uid, const char* name) -> void* {
        auto it = uidToPtr.find(uid);
        if(it == uidToPtr.end())
        {
            HIPDNN_PLUGIN_LOG_ERROR("ConvFwdPlan::execute: tensor uid "
                                    << uid << " (" << name << ") not in device buffer map");
            return nullptr;
        }
        return it->second;
    };
    void* pA = lookupPtr(_params.xUid, "x");
    void* pB = lookupPtr(_params.wUid, "w");
    void* pD = lookupPtr(_params.yUid, "y");
    if(!pA || !pB || !pD)
        return;

    // Kernel argument layout (matches conv_implicit_gemm.py build_implicit_gemm_conv):
    //   ptr A (8B), ptr B (8B), ptr D (8B), i32 A_bytes (4B), i32 B_bytes (4B), i32 D_bytes (4B)
    struct ConvKernelArgs
    {
        const void* pA;
        const void* pB;
        void* pD;
        int aBytes;
        int bBytes;
        int dBytes;
    } args{pA, pB, pD, _params.aBytes, _params.bBytes, _params.dBytes};

    // Grid: X = ceil(K / tileN), Y = ceil(M / tileM), Z = 1
    // where M = N * Ho * Wo, following the block_n_axis="x", block_m_axis="y" convention
    const unsigned int gridX = _params.gridN; // N_gemm tiles
    const unsigned int gridY = _params.gridM; // M tiles
    const unsigned int gridZ = 1;

    size_t argSize = sizeof(args);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays) - HIP API requires C-style array
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize,
                      HIP_LAUNCH_PARAM_END};

    const hipError_t err = hipModuleLaunchKernel(_kernel.function(),
                                                 gridX,
                                                 gridY,
                                                 gridZ,
                                                 _params.blockSize,
                                                 1,
                                                 1,
                                                 0,
                                                 handle.getStream(),
                                                 nullptr,
                                                 config);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "ConvFwdPlan::execute: hipModuleLaunchKernel failed: " << hipGetErrorString(err));
    }
    else
    {
        HIPDNN_PLUGIN_LOG_INFO("ConvFwdPlan::execute: launched "
                               << _params.kernelName << " grid=[" << gridX << "," << gridY << ",1]"
                               << " block=[" << _params.blockSize << ",1,1]");
    }
}

} // namespace rocke_conv_engine
