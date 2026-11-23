// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaBwdParams.hpp"
#include "SdpaPlanUtils.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

/// The dq_acc buffer the DQDKDV kernel accumulates dQ into when a post stage
/// follows it. Mirrors the batch-mode layout of AITER v0.1.29 mha_bwd.cu
/// fmha_v3_bwd() (:577-596):
/// - A32: FP32 [B, H_q, S_q, D_qk].
/// - A16: Q dtype [B, H_q, ceil16(S_q), 192 if D_qk == 192 else 128].
/// The DQDKDV kernel takes no dq_acc strides, so the layout must match AITER's
/// exactly. Strides are in elements; the batch stride is 0 when B == 1.
struct SdpaBwdDqAccLayout
{
    size_t elementBytes;
    size_t stride;
    size_t nheadStride;
    size_t batchStride;
    size_t bytes; // before workspace alignment
};

constexpr SdpaBwdDqAccLayout sdpaBwdDqAccLayout(
    size_t batch, size_t headsQ, size_t seqLenQ, size_t headDimQk, AccumulatorType accType)
{
    const bool a32 = accType == AccumulatorType::A32;
    const size_t elementBytes = a32 ? 4 : 2;
    const size_t rows = a32 ? seqLenQ : (seqLenQ + 15) / 16 * 16;
    size_t stride = headDimQk;
    if(!a32)
    {
        stride = headDimQk == 192 ? 192 : 128;
    }
    const size_t perBatch = headsQ * rows * stride;
    return {elementBytes,
            stride,
            rows * stride,
            batch == 1 ? 0 : perBatch,
            batch * perBatch * elementBytes};
}

/// True when every dq_acc offset the DQDKDV kernels form in 32 bits within one
/// batch fits: the row and head strides and the per-batch span of `headsQ`
/// heads, in bytes. The span is checked for B == 1 as well, where the layout's
/// batch stride is 0. Offsets across batches are bounded per row by the dqdkdv
/// lookup (bwd_dispatch::psskOffOffsetsFitU32 for the pssk = 0 rows, through
/// sdpaBwdDqAccBytesFitU32). Logs each offending field.
inline bool sdpaBwdDqAccFitsU32(const SdpaBwdDqAccLayout& layout, size_t headsQ)
{
    const auto elementBytes = static_cast<int64_t>(layout.elementBytes);
    const auto nheadStride = static_cast<int64_t>(layout.nheadStride);
    bool ok = plan_utils::byteStrideFitsU32(
        "stride_dq_acc", static_cast<int64_t>(layout.stride), elementBytes);
    ok &= plan_utils::byteStrideFitsU32("nhead_stride_dq_acc", nheadStride, elementBytes);
    // The per-batch element count saturates instead of overflowing; any
    // saturated count is far past the bound.
    int64_t perBatch = 0;
    if(__builtin_mul_overflow(static_cast<int64_t>(headsQ), nheadStride, &perBatch))
    {
        perBatch = INT64_MAX;
    }
    ok &= plan_utils::byteStrideFitsU32("dq_acc per-batch span", perBatch, elementBytes);
    return ok;
}

/// True when the whole dq_acc of `batch` batches of `headsQ` heads spans at
/// most UINT32_MAX bytes, so offsets across batches fit 32 bits as well. The
/// byte count is formed without overflow (layout.bytes wraps for huge shapes).
inline bool sdpaBwdDqAccBytesFitU32(const SdpaBwdDqAccLayout& layout, size_t batch, size_t headsQ)
{
    int64_t bytes = 0;
    return !__builtin_mul_overflow(
               static_cast<int64_t>(batch), static_cast<int64_t>(headsQ), &bytes)
           && !__builtin_mul_overflow(bytes, static_cast<int64_t>(layout.nheadStride), &bytes)
           && !__builtin_mul_overflow(bytes, static_cast<int64_t>(layout.elementBytes), &bytes)
           && bytes <= static_cast<int64_t>(UINT32_MAX);
}

/// Total backward workspace size. Layout: D buffer | dq_acc, where dq_acc is
/// present only when a post stage reads it (`hasDqAcc`, see
/// bwd_dispatch::postStage()).
constexpr size_t sdpaBwdWorkspaceSize(size_t batch,
                                      size_t headsQ,
                                      size_t seqLenQ,
                                      size_t headDimQk,
                                      AccumulatorType accType,
                                      bool hasDqAcc)
{
    size_t size = sdpaBwdDBufferSize(batch, headsQ, seqLenQ);
    if(hasDqAcc)
    {
        size += alignUp(sdpaBwdDqAccLayout(batch, headsQ, seqLenQ, headDimQk, accType).bytes,
                        K_WORKSPACE_ALIGNMENT_BYTES);
    }
    return size;
}

/// How to zero a [B, H, S, D] tensor whose D axis is contiguous with
/// hipMemset2DAsync: one call per element of `loops`, each zeroing `height`
/// rows of `widthBytes` that start `pitchBytes` apart. Axes whose stride
/// continues the region below them merge into it, so packed, [B, S, H, D] and
/// row-padded layouts take a single call. Unused loops are {1, 0}.
struct StridedZeroPlan
{
    struct Axis
    {
        size_t extent;
        size_t strideBytes;
    };
    size_t widthBytes;
    size_t pitchBytes;
    size_t height;
    std::array<Axis, 3> loops;
};

/// `extents` and `strides` (in elements) are the B, H and S axes.
inline StridedZeroPlan planBhsdZero(const std::array<size_t, 3>& extents,
                                    const std::array<size_t, 3>& strides,
                                    size_t headDim,
                                    size_t elementBytes)
{
    std::array<StridedZeroPlan::Axis, 3> axes{};
    size_t numAxes = 0;
    for(size_t i = 0; i < extents.size(); ++i)
    {
        if(extents[i] > 1)
        {
            axes[numAxes++] = {extents[i], strides[i] * elementBytes};
        }
    }
    std::sort(axes.begin(),
              axes.begin() + static_cast<std::ptrdiff_t>(numAxes),
              [](const StridedZeroPlan::Axis& a, const StridedZeroPlan::Axis& b) {
                  return a.strideBytes < b.strideBytes;
              });

    size_t next = 0;
    size_t width = headDim * elementBytes;
    while(next < numAxes && axes[next].strideBytes == width)
    {
        width *= axes[next++].extent;
    }
    StridedZeroPlan plan{width, width, 1, {}};
    if(next < numAxes && axes[next].strideBytes >= width)
    {
        plan.pitchBytes = axes[next].strideBytes;
        plan.height = axes[next++].extent;
        while(next < numAxes && axes[next].strideBytes == plan.height * plan.pitchBytes)
        {
            plan.height *= axes[next++].extent;
        }
    }
    plan.loops.fill({1, 0});
    for(size_t i = 0; next < numAxes; ++i)
    {
        plan.loops[i] = axes[next++];
    }
    return plan;
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

// =============================================================================
// Kernel module loading from in-memory image
// =============================================================================
//
// Same as loadKernelModule but takes a pre-loaded HSACO image (e.g. extracted
// from a .kpack archive) instead of a filesystem path.

inline std::optional<HipModuleGuard>
    loadKernelModuleFromMemory(const void* data, size_t size, const char* funcName)
{
    if(data == nullptr || size == 0)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Cannot load kernel module from null/empty image");
        return std::nullopt;
    }

    hipModule_t rawModule = nullptr;
    hipError_t err = hipModuleLoadData(&rawModule, data);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Failed to load kernel module from memory ("
                                << size << " bytes), error: " << hipGetErrorString(err));
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
