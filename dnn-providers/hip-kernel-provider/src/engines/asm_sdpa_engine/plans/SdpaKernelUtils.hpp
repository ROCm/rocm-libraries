// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <optional>
#include <string>
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
// SDPA backward workspace buffer sizes
// =============================================================================
//
// Shared by SdpaBwdPlanBuilder::getMaxWorkspaceSize() (pre-plan, extracts dims
// from the graph) and SdpaBwdPlan::getWorkspaceSize() (post-plan, reads from
// SdpaBwdParams).  Having a single source of truth prevents the two from
// silently diverging.

/// D buffer: row-wise dot(O, dO) output, shape [B, H_q, S_q] in FP32.
constexpr size_t sdpaBwdDBufferSize(size_t batch, size_t headsQ, size_t seqLenQ)
{
    return alignUp(batch * headsQ * seqLenQ * sizeof(float), K_WORKSPACE_ALIGNMENT_BYTES);
}

/// dq_acc buffer: FP32 accumulator for dQ, shape [B, H_q, S_q, D_qk] in FP32.
/// Only needed for a32-accumulator kernels; a16 kernels write dQ in BF16 directly.
// TODO(Task I8.2): POC assumes a32 accumulator — always allocates FP32 dq_acc buffer.
// For a16 accumulator kernels, dQ is written directly in BF16 (no dq_acc buffer needed,
// no dq_convert kernel launched). Provider should check accumulator type and skip
// dq_acc allocation for a16.
constexpr size_t sdpaBwdDqAccBufferSize(size_t batch, size_t headsQ, size_t seqLenQ, size_t headDim)
{
    return alignUp(batch * headsQ * seqLenQ * headDim * sizeof(float), K_WORKSPACE_ALIGNMENT_BYTES);
}

// =============================================================================
// Kernel launch helper
// =============================================================================
//
// Wraps hipModuleLaunchKernel with HIP_LAUNCH_PARAM config for ASM kernels.
// Uses HIP_LAUNCH_PARAM_BUFFER_POINTER/SIZE mechanism which is required for
// passing large argument structures (e.g. 656 bytes for fwd, 784 bytes for bwd)
// to ASM kernels.
// Logs error on failure, logs grid/block info on success.
// Returns true on success, false on failure.

inline bool launchKernel(const char* kernelName,
                         hipFunction_t func,
                         void* args,
                         size_t argSize,
                         unsigned int gridX,
                         unsigned int gridY,
                         unsigned int gridZ,
                         unsigned int blockDim,
                         hipStream_t stream = nullptr)
{
    // All AITER fmha kernels (fwd, bwd ODO/DQDKDV/DQ_CONVERT, all 319 variants
    // across gfx942/gfx950) use 1D thread blocks.
    constexpr unsigned int K_BLOCK_DIM_Y = 1;
    constexpr unsigned int K_BLOCK_DIM_Z = 1;

    // NOLINTNEXTLINE(modernize-avoid-c-arrays) - HIP API requires C-style array
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize,
                      HIP_LAUNCH_PARAM_END};

    // The stream determines execution ordering.  All kernels on the same
    // stream run sequentially; using the handle's stream allows the caller
    // to overlap SDPA work with independent operations on other streams.
    const hipError_t err = hipModuleLaunchKernel(func,
                                                 gridX,
                                                 gridY,
                                                 gridZ,
                                                 blockDim,
                                                 K_BLOCK_DIM_Y,
                                                 K_BLOCK_DIM_Z,
                                                 0, // shared memory (kernel uses LDS internally)
                                                 stream,
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

    explicit HipModuleGuard(hipModule_t moduleIn, hipFunction_t functionIn = nullptr)
        : _module(moduleIn)
        , _function(functionIn)
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
                const hipError_t err = hipModuleUnload(_module);
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

// =============================================================================
// Kernel module loading helper
// =============================================================================
//
// Combines hipModuleLoad + hipModuleGetFunction into a single call.
// Returns std::nullopt on failure (with error logging); on success returns
// a HipModuleGuard that owns the module and holds the function pointer.

inline std::optional<HipModuleGuard> loadKernelModule(const std::string& coPath,
                                                      const char* funcName)
{
    hipModule_t rawModule = nullptr;
    hipError_t err = hipModuleLoad(&rawModule, coPath.c_str());
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "Failed to load kernel module: " << coPath << " error: " << hipGetErrorString(err));
        return std::nullopt;
    }
    HipModuleGuard guard(rawModule);

    hipFunction_t func = nullptr;
    err = hipModuleGetFunction(&func, guard.module(), funcName);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to get kernel function '"
                                << funcName << "' error: " << hipGetErrorString(err));
        return std::nullopt; // guard destructor unloads module
    }
    guard.setFunction(func);

    return guard; // moved out
}

} // namespace asm_sdpa_engine
