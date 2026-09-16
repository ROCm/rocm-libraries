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
        , _mergeFunction(std::exchange(o._mergeFunction, nullptr))
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
            _mergeFunction = std::exchange(o._mergeFunction, nullptr);
        }
        return *this;
    }

    hipModule_t module() const
    {
        return _module;
    }
    /// Optional second entry point from the SAME module. Split-K needs two
    /// kernels (the per-chunk split pass and the merge pass) and both live in
    /// one .co, so this is one hipModuleLoad and two hipModuleGetFunction
    /// calls -- the guard's move/unload semantics are unchanged.
    void setMergeFunction(hipFunction_t f)
    {
        _mergeFunction = f;
    }

    hipFunction_t mergeFunction() const
    {
        return _mergeFunction;
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
    hipFunction_t _mergeFunction = nullptr;
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

// Overload for split-K: one module, two entry points. Returns nullopt if
// either function is missing, so a truncated or mismatched object is caught at
// build-plan time rather than at launch.
inline std::optional<HipModuleGuard>
    loadKernelModule(const std::string& coPath, const char* funcName, const char* mergeFuncName)
{
    auto guard = loadKernelModule(coPath, funcName);
    if(!guard)
        return std::nullopt;

    hipFunction_t mergeFunc = nullptr;
    const hipError_t err = hipModuleGetFunction(&mergeFunc, guard->module(), mergeFuncName);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2: hipModuleGetFunction('"
                                << mergeFuncName << "'): " << hipGetErrorString(err));
        return std::nullopt; // guard destructs -> hipModuleUnload
    }
    guard->setMergeFunction(mergeFunc);
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
// Split-K argument structs -- field order must match the kernel signatures in
// HipFlash2FwdPlanVariantSplitK.hip byte for byte. These go across as a packed
// buffer, so a reordered or resized field is silent corruption rather than a
// compile error.
// =============================================================================

/// Args for flash2_split_d128 / flash2_split_d64.
/// Writes fp32 partials: po[B*H][nsplit][Sq][D], pm/pl[B*H][nsplit][Sq].
/// Takes no output strides -- the partial buffers are always packed.
struct Flash2SplitKernelArgs
{
    const void* ptr_q = nullptr;
    const void* ptr_k = nullptr;
    const void* ptr_v = nullptr;
    float* ptr_po = nullptr;
    float* ptr_pm = nullptr;
    float* ptr_pl = nullptr;

    int batch = 1;
    int num_heads_q = 32;
    int num_heads_k = 32;
    int seq_len_q = 2048;
    int seq_len_kv = 2048;
    int head_dim = 128;
    float scale = 0.0f;
    int causal = 0;
    int nsplit = 1;

    int q_stride_batch = 0;
    int q_stride_head = 0;
    int q_stride_seq = 0;
    int k_stride_batch = 0;
    int k_stride_head = 0;
    int k_stride_seq = 0;
    int v_stride_batch = 0;
    int v_stride_head = 0;
    int v_stride_seq = 0;
};

/// Args for flash2_merge: combines the per-chunk partials with the exact
/// online-softmax rescale (m = max m_j, l = sum l_j*exp(m_j - m)).
struct Flash2MergeKernelArgs
{
    const float* ptr_po = nullptr;
    const float* ptr_pm = nullptr;
    const float* ptr_pl = nullptr;
    void* ptr_o = nullptr;

    int batch = 1;
    int num_heads_q = 32;
    int seq_len_q = 2048;
    int head_dim = 128;
    int nsplit = 1;

    int o_stride_batch = 0;
    int o_stride_head = 0;
    int o_stride_seq = 0;
};

/// Launch the per-chunk split pass. Grid is (qTiles, B*H, nsplit): the kernel
/// decodes blockIdx.y as the batch-head pair and blockIdx.z as the chunk.
inline bool launchFlash2SplitKernel(hipFunction_t func,
                                    Flash2SplitKernelArgs& args,
                                    unsigned int gridX,
                                    unsigned int gridY,
                                    unsigned int gridZ,
                                    unsigned int blockDim,
                                    hipStream_t stream)
{
    size_t argSize = sizeof(Flash2SplitKernelArgs);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize,
                      HIP_LAUNCH_PARAM_END};

    const hipError_t err = hipModuleLaunchKernel(
        func, gridX, gridY, gridZ, blockDim, 1, 1, 0, stream, nullptr, config);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "HipFlash2: split kernel launch failed: " << hipGetErrorString(err));
        return false;
    }
    return true;
}

/// Launch the merge pass. Four query rows per CTA of 256 threads, so the grid
/// is (ceil(Sq/4), B*H, 1).
inline bool launchFlash2MergeKernel(hipFunction_t func,
                                    Flash2MergeKernelArgs& args,
                                    unsigned int gridX,
                                    unsigned int gridY,
                                    hipStream_t stream)
{
    constexpr unsigned int K_MERGE_BLOCK_DIM = 256;

    size_t argSize = sizeof(Flash2MergeKernelArgs);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize,
                      HIP_LAUNCH_PARAM_END};

    const hipError_t err = hipModuleLaunchKernel(
        func, gridX, gridY, 1, K_MERGE_BLOCK_DIM, 1, 1, 0, stream, nullptr, config);
    if(err != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "HipFlash2: merge kernel launch failed: " << hipGetErrorString(err));
        return false;
    }
    return true;
}

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
/// Variant tag for the split-K object. It is not in the dispatch variant table
/// because it is not a tiling choice: it carries a different pair of entry
/// points (a per-chunk split pass plus a merge pass) and is selected by the
/// split factor rather than by CTA geometry.
constexpr const char* K_FLASH2_SPLITK_TAG = "splitk";

/// Merge entry point, shared by both head dims.
constexpr const char* K_FLASH2_MERGE_FUNC = "flash2_merge";

/// Split-pass entry point for a given head dim. Mirrors flash2KernelName().
inline const char* flash2SplitKernelName(int headDim)
{
    switch(headDim)
    {
    case 64:
        return "flash2_split_d64";
    case 128:
        return "flash2_split_d128";
    default:
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2: unsupported head_dim for split-K=" << headDim);
        return nullptr;
    }
}

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
