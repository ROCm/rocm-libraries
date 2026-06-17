// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Kernel loading and launch utilities for the Flash-Attention 2 V7 engine.
// Mirrors the pattern in asm_sdpa_engine/plans/SdpaKernelUtils.hpp.

#pragma once

#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <optional>
#include <string>
#include <utility>

namespace hip_flash2_engine {

// =============================================================================
// HipModuleGuard — RAII wrapper for hipModule_t
// =============================================================================
class HipModuleGuard {
   public:
    HipModuleGuard() = default;

    explicit HipModuleGuard(hipModule_t mod, hipFunction_t func = nullptr)
        : _module(mod), _function(func) {}

    ~HipModuleGuard() {
        if (_module != nullptr) {
            const hipError_t err = hipModuleUnload(_module);
            if (err != hipSuccess) {
                HIPDNN_PLUGIN_LOG_ERROR(
                    "HipFlash2: failed to unload kernel module: " << hipGetErrorString(err));
            }
        }
    }

    HipModuleGuard(const HipModuleGuard&) = delete;
    HipModuleGuard& operator=(const HipModuleGuard&) = delete;

    HipModuleGuard(HipModuleGuard&& o) noexcept
        : _module(std::exchange(o._module, nullptr)),
          _function(std::exchange(o._function, nullptr)) {}

    HipModuleGuard& operator=(HipModuleGuard&& o) noexcept {
        if (this != &o) {
            if (_module != nullptr) hipModuleUnload(_module);
            _module = std::exchange(o._module, nullptr);
            _function = std::exchange(o._function, nullptr);
        }
        return *this;
    }

    hipModule_t module() const {
        return _module;
    }
    hipFunction_t function() const {
        return _function;
    }
    void setFunction(hipFunction_t f) {
        _function = f;
    }

   private:
    hipModule_t _module = nullptr;
    hipFunction_t _function = nullptr;
};

// =============================================================================
// loadKernelModule — load .co and get named function
// =============================================================================
inline std::optional<HipModuleGuard> loadKernelModule(const std::string& coPath,
                                                      const char* funcName) {
    hipModule_t rawModule = nullptr;
    hipError_t err = hipModuleLoad(&rawModule, coPath.c_str());
    if (err != hipSuccess) {
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2: failed to load .co from '"
                                << coPath << "': " << hipGetErrorString(err));
        return std::nullopt;
    }

    HipModuleGuard guard(rawModule);

    hipFunction_t func = nullptr;
    err = hipModuleGetFunction(&func, guard.module(), funcName);
    if (err != hipSuccess) {
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2: hipModuleGetFunction('"
                                << funcName << "'): " << hipGetErrorString(err));
        return std::nullopt;  // guard destructs → hipModuleUnload
    }
    guard.setFunction(func);
    return guard;
}

// =============================================================================
// Flash2KernelArgs — argument struct passed to the kernel via
// HIP_LAUNCH_PARAM_BUFFER_POINTER/SIZE (matches the kernel's parameter order)
// =============================================================================
struct Flash2KernelArgs {
    // Input tensors (device pointers, FP16)
    const void* ptr_q = nullptr;
    const void* ptr_k = nullptr;
    const void* ptr_v = nullptr;
    // Output tensor (device pointer, FP16)
    void* ptr_o = nullptr;

    // Attention geometry
    int batch = 1;
    int num_heads_q = 32;
    int num_heads_k = 32;
    int seq_len_q = 2048;
    int seq_len_kv = 2048;
    int head_dim = 128;  // compile-time template in kernel, but kept for reference
    float scale = 0.0f;
    int causal = 0;  // bool as int

    // Strides (in elements, not bytes) — BHSD layout [B, H, S, D]
    int q_stride_batch = 0;
    int q_stride_head = 0;
    int q_stride_seq = 0;
    int k_stride_batch = 0;
    int k_stride_head = 0;
    int k_stride_seq = 0;
    int v_stride_batch = 0;
    int v_stride_head = 0;
    int v_stride_seq = 0;
    int o_stride_batch = 0;
    int o_stride_head = 0;
    int o_stride_seq = 0;
};

// =============================================================================
// launchFlash2Kernel — wrapper around hipModuleLaunchKernel
// =============================================================================
inline bool launchFlash2Kernel(hipFunction_t func, Flash2KernelArgs& args, unsigned int gridX,
                               unsigned int gridY, unsigned int gridZ, unsigned int blockDim,
                               hipStream_t stream) {
    // All Flash2 V7 tiles use 1-D thread blocks (256 or 512 threads per CTA)
    constexpr unsigned int K_BLOCK_DIM_Y = 1;
    constexpr unsigned int K_BLOCK_DIM_Z = 1;

    size_t argSize = sizeof(Flash2KernelArgs);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argSize, HIP_LAUNCH_PARAM_END};

    const hipError_t err =
        hipModuleLaunchKernel(func, gridX, gridY, gridZ, blockDim, K_BLOCK_DIM_Y, K_BLOCK_DIM_Z,
                              0,  // LDS allocated by kernel
                              stream,
                              nullptr,  // params via config
                              config);
    if (err != hipSuccess) {
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
inline const char* flash2KernelName(int headDim) {
    switch (headDim) {
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
// .co path helper — selects gfx942 or gfx950 based on device string
// =============================================================================
inline std::string flash2CoPath(const std::string& archId) {
    // HIP_FLASH2_KERNEL_DIR is set at compile time in CMakeLists.txt
#ifndef HIP_FLASH2_KERNEL_DIR
#define HIP_FLASH2_KERNEL_DIR "/opt/rocm/lib/hipdnn/engines/hip_flash2_kernels"
#endif
    std::string dir = HIP_FLASH2_KERNEL_DIR;
    if (dir.back() != '/') dir += '/';
    return dir + "hip_flash2_fwd_" + archId + ".co";
}

}  // namespace hip_flash2_engine
