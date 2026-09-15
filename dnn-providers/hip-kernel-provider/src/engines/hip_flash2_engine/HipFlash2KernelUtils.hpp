// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Kernel loading and launch utilities for the Flash-Attention 2 V7 engine.
// Mirrors the pattern in asm_sdpa_engine/plans/SdpaKernelUtils.hpp.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <optional>
#include <string>
#include <utility>

namespace hip_flash2_engine
{

// =============================================================================
// HipModuleGuard -- RAII wrapper for hipModule_t
// =============================================================================
class HipModuleGuard
{
public:
    HipModuleGuard() = default;

    explicit HipModuleGuard(hipModule_t mod, hipFunction_t func = nullptr)
        : _module(mod)
        , _function(func)
    {
    }

    ~HipModuleGuard()
    {
        if(_module != nullptr)
        {
            const hipError_t err = hipModuleUnload(_module);
            if(err != hipSuccess)
            {
                HIPDNN_PLUGIN_LOG_ERROR(
                    "HipFlash2: failed to unload kernel module: " << hipGetErrorString(err));
            }
        }
    }

    HipModuleGuard(const HipModuleGuard&) = delete;
    HipModuleGuard& operator=(const HipModuleGuard&) = delete;

    HipModuleGuard(HipModuleGuard&& o) noexcept
        : _module(std::exchange(o._module, nullptr))
        , _function(std::exchange(o._function, nullptr))
    {
    }

    HipModuleGuard& operator=(HipModuleGuard&& o) noexcept
    {
        if(this != &o)
        {
            if(_module != nullptr)
            {
                // Log unload errors on move-assignment (mirrors SdpaKernelUtils pattern)
                const hipError_t err = hipModuleUnload(_module);
                if(err != hipSuccess)
                {
                    HIPDNN_PLUGIN_LOG_ERROR(
                        "HipFlash2: failed to unload kernel module on move-assign: "
                        << hipGetErrorString(err));
                }
            }
            _module = std::exchange(o._module, nullptr);
            _function = std::exchange(o._function, nullptr);
        }
        return *this;
    }

    hipModule_t module() const
    {
        return _module;
    }
    hipFunction_t function() const
    {
        return _function;
    }
    void setFunction(hipFunction_t f)
    {
        _function = f;
    }

private:
    hipModule_t _module = nullptr;
    hipFunction_t _function = nullptr;
};

// =============================================================================
// loadKernelModule -- load .co and get named function
// =============================================================================
inline std::optional<HipModuleGuard> loadKernelModule(const std::string& coPath,
                                                      const char* funcName)
{
    hipModule_t rawModule = nullptr;
    hipError_t err = hipModuleLoad(&rawModule, coPath.c_str());
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2: failed to load .co from '"
                                << coPath << "': " << hipGetErrorString(err));
        return std::nullopt;
    }

    HipModuleGuard guard(rawModule);

    hipFunction_t func = nullptr;
    err = hipModuleGetFunction(&func, guard.module(), funcName);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2: hipModuleGetFunction('"
                                << funcName << "'): " << hipGetErrorString(err));
        return std::nullopt; // guard destructs -> hipModuleUnload
    }
    guard.setFunction(func);
    return guard;
}

// =============================================================================
// Flash2KernelArgs -- argument struct passed to the kernel via
// HIP_LAUNCH_PARAM_BUFFER_POINTER/SIZE (matches the kernel's parameter order)
// =============================================================================
struct Flash2KernelArgs
{
    // Input tensors (device pointers, FP16)
    const void* ptrQ = nullptr;
    const void* ptrK = nullptr;
    const void* ptrV = nullptr;
    // Output tensor (device pointer, FP16)
    void* ptrO = nullptr;

    // Attention geometry
    int batch = 1;
    int numHeadsQ = 32;
    int numHeadsK = 32;
    int seqLenQ = 2048;
    int seqLenKv = 2048;
    int headDim = 128; // compile-time template in kernel, but kept for reference
    float scale = 0.0f;
    int causal = 0; // bool as int

    // Strides (in elements, not bytes) -- BHSD layout [B, H, S, D]
    int qStrideBatch = 0;
    int qStrideHead = 0;
    int qStrideSeq = 0;
    int kStrideBatch = 0;
    int kStrideHead = 0;
    int kStrideSeq = 0;
    int vStrideBatch = 0;
    int vStrideHead = 0;
    int vStrideSeq = 0;
    int oStrideBatch = 0;
    int oStrideHead = 0;
    int oStrideSeq = 0;
};

// =============================================================================
// launchFlash2Kernel -- wrapper around hipModuleLaunchKernel
// =============================================================================
inline bool launchFlash2Kernel(hipFunction_t func,
                               Flash2KernelArgs& args,
                               unsigned int gridX,
                               unsigned int gridY,
                               unsigned int gridZ,
                               unsigned int blockDim,
                               hipStream_t stream)
{
    // All Flash2 V7 tiles use 1-D thread blocks (256 or 512 threads per CTA)
    constexpr unsigned int K_BLOCK_DIM_Y = 1;
    constexpr unsigned int K_BLOCK_DIM_Z = 1;

    size_t argSize = sizeof(Flash2KernelArgs);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize,
                      HIP_LAUNCH_PARAM_END};

    const hipError_t err = hipModuleLaunchKernel(func,
                                                 gridX,
                                                 gridY,
                                                 gridZ,
                                                 blockDim,
                                                 K_BLOCK_DIM_Y,
                                                 K_BLOCK_DIM_Z,
                                                 0, // LDS allocated by kernel
                                                 stream,
                                                 nullptr, // params via config
                                                 config);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "HipFlash2: hipModuleLaunchKernel failed: " << hipGetErrorString(err));
        return false;
    }

    HIPDNN_PLUGIN_LOG_INFO("HipFlash2: kernel launched grid=[" << gridX << "," << gridY << ","
                                                               << gridZ << "] block=[" << blockDim
                                                               << ",1,1]");
    return true;
}

// =============================================================================
// Kernel symbol names (extern "C" wrappers in HipFlash2FwdPlan.hip)
// =============================================================================
inline const char* flash2KernelName(int headDim)
{
    switch(headDim)
    {
    case 64:
        return "flash2_v7_hipdnn_d64";
    case 128:
        return "flash2_v7_hipdnn_d128";
    default:
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2: unsupported head_dim=" << headDim);
        return nullptr;
    }
}

// =============================================================================
// .co path helper -- builds the .co path for this device.
// Only gfx942 is supported; isApplicable() gates on arch before buildPlan is called.
// =============================================================================
// flash2CoPath: resolve the directory containing the precompiled .co files.
//
// Resolution order (mirrors asm_sdpa_engine pattern):
//   1. Runtime env var HIP_FLASH2_KERNEL_DIR (allows deployment overrides)
//   2. Compile-time HIP_FLASH2_KERNEL_DIR (set to absolute install path by CMake)
//   3. Built-in fallback (standard ROCm install location)
//
// The CMakeLists sets HIP_FLASH2_KERNEL_DIR via target_compile_definitions to
// "${CMAKE_INSTALL_PREFIX}/lib/hipdnn/engines/hip_flash2_kernels" so that the
// path is always absolute in a normal build.
#ifndef HIP_FLASH2_KERNEL_DIR
#define HIP_FLASH2_KERNEL_DIR "/opt/rocm/lib/hipdnn/engines/hip_flash2_kernels"
#endif

inline std::string flash2CoPath(const std::string& archId, const std::string& variantTag = "")
{
    // Prefer runtime env override so tests and non-standard installs work.
    const char* envDir = std::getenv("HIP_FLASH2_KERNEL_DIR");
    std::string dir = (envDir != nullptr && envDir[0] != '\0') ? envDir : HIP_FLASH2_KERNEL_DIR;
    if(!dir.empty() && dir.back() != '/')
    {
        dir += '/';
    }
    if(variantTag.empty())
    {
        return dir + "hip_flash2_fwd_" + archId + ".co";
    }
    return dir + "hip_flash2_fwd_" + archId + "_" + variantTag + ".co";
}

} // namespace hip_flash2_engine
