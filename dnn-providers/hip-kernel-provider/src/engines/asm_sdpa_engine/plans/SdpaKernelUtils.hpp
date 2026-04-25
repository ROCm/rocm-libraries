// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <utility>

namespace asm_sdpa_engine
{

// =============================================================================
// Workspace alignment utilities
// =============================================================================
//
// In AITER (upstream), each workspace buffer (D buffer, dq_acc) is a separate PyTorch tensor
// allocation. Each torch::empty() call invokes hipMalloc(), which guarantees 256-byte alignment
// per allocation. So AITER never explicitly aligns — every buffer pointer is automatically aligned.
//
// In hip-kernel-provider, hipDNN provides a single contiguous workspace buffer (one hipMalloc).
// The execute() method must carve this into sub-buffers using pointer arithmetic:
//   D buffer    starts at: workspace + 0                     (aligned by hipMalloc)
//   dq_acc      starts at: workspace + sizeof(D buffer)      (NOT automatically aligned)
//
// We round each sub-buffer size up to a 64-byte boundary (MI300X L2 cache line size) so the
// next sub-buffer starts cache-line-aligned. This prevents false sharing between buffers and
// ensures vector memory instructions (e.g. global_load_b128) don't span cache line boundaries.
//
// TODO(Task I8.9): POC hardcodes 64 bytes; production should query hipGetDeviceProperties()
// NOLINTNEXTLINE(readability-redundant-inline-specifier)
inline constexpr size_t K_WORKSPACE_ALIGNMENT_BYTES = 64;

constexpr size_t alignUp(size_t size, size_t alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

// =============================================================================
// Kernel launch helper
// =============================================================================
//
// Wraps hipModuleLaunchKernel with HIP_LAUNCH_PARAM config for ASM kernels.
// Logs error on failure, logs grid/block info on success.
// Returns true on success, false on failure.

inline bool launchKernel(const char* kernelName,
                         hipFunction_t func,
                         void* args,
                         size_t argSize,
                         unsigned int gridX,
                         unsigned int gridY,
                         unsigned int gridZ,
                         unsigned int blockDim)
{
    // NOLINTNEXTLINE(modernize-avoid-c-arrays) - HIP API requires C-style array
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize,
                      HIP_LAUNCH_PARAM_END};

    hipError_t err = hipModuleLaunchKernel(func,
                                           gridX,
                                           gridY,
                                           gridZ,
                                           blockDim,
                                           1,
                                           1,
                                           0, // shared memory (kernel uses LDS internally)
                                           nullptr, // stream (use default)
                                           nullptr, // kernel args (not used with config)
                                           config);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to launch "
                                << kernelName << " kernel, error: " << hipGetErrorString(err));
        return false;
    }

    HIPDNN_PLUGIN_LOG_INFO(kernelName << " kernel launched: grid=[" << gridX << "," << gridY << ","
                                      << gridZ << "] block=[" << blockDim << ",1,1]");
    return true;
}

// =============================================================================
// HipModuleGuard — RAII wrapper for hipModule_t
// =============================================================================
//
// Owns a hipModule_t and calls hipModuleUnload in its destructor.
// Optionally stores the associated hipFunction_t (non-owning — lifetime tied to module).
// Move-only; compiler-generated move/destructor on classes holding this member
// will do the right thing, eliminating manual resource management boilerplate.

class HipModuleGuard
{
public:
    HipModuleGuard() = default;

    explicit HipModuleGuard(hipModule_t module, hipFunction_t function = nullptr)
        : _module(module)
        , _function(function)
    {
    }

    ~HipModuleGuard()
    {
        if(_module != nullptr)
        {
            hipError_t err = hipModuleUnload(_module);
            if(err != hipSuccess)
            {
                HIPDNN_PLUGIN_LOG_ERROR(
                    "Failed to unload kernel module, error: " << hipGetErrorString(err));
            }
        }
    }

    HipModuleGuard(const HipModuleGuard&) = delete;
    HipModuleGuard& operator=(const HipModuleGuard&) = delete;

    HipModuleGuard(HipModuleGuard&& other) noexcept
        : _module(std::exchange(other._module, nullptr))
        , _function(std::exchange(other._function, nullptr))
    {
    }

    HipModuleGuard& operator=(HipModuleGuard&& other) noexcept
    {
        if(this != &other)
        {
            if(_module != nullptr)
            {
                hipError_t err = hipModuleUnload(_module);
                if(err != hipSuccess)
                {
                    HIPDNN_PLUGIN_LOG_ERROR("Failed to unload kernel module during move, error: "
                                            << hipGetErrorString(err));
                }
            }
            _module = std::exchange(other._module, nullptr);
            _function = std::exchange(other._function, nullptr);
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
    void setFunction(hipFunction_t func)
    {
        _function = func;
    }

private:
    hipModule_t _module = nullptr;
    hipFunction_t _function = nullptr;
};

} // namespace asm_sdpa_engine
