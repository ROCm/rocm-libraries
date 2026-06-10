// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/validation/reference_kernels.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include <hip/hip_runtime.h>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>

namespace ck_tile {
namespace dispatcher {
namespace backends {

/**
 * Kernel instance wrapper for unified_gemm_codegen.py generated kernels
 *
 * These kernels have structure:
 * - Types defined outside: using ADataType = ...; using BDataType = ...;
 * - struct SelectedKernel with static constexpr config and launch() method
 * - constexpr const char* KERNEL_NAME = "...";
 *
 * This is different from tile_engine style where everything is in SelectedKernel.
 */
template <typename SelectedKernelType,
          typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_>
class GeneratedTileKernelInstance : public KernelInstance
{
    public:
    using ADataType      = ADataType_;
    using BDataType      = BDataType_;
    using CDataType      = CDataType_;
    using AccDataType    = AccDataType_;
    using SelectedKernel = SelectedKernelType;

    GeneratedTileKernelInstance(const KernelKey& key, const std::string& name)
        : key_(key), name_(name)
    {
    }

    const KernelKey& get_key() const override { return key_; }

    bool supports(const Problem& problem) const override
    {
        // Check dimension divisibility if padding not enabled
        constexpr bool pad_m = SelectedKernel::kPadM;
        constexpr bool pad_n = SelectedKernel::kPadN;
        constexpr bool pad_k = SelectedKernel::kPadK;

        if(pad_m && pad_n && pad_k)
        {
            return true; // Padding enabled - supports any size
        }

        // Check divisibility
        constexpr int tile_m = SelectedKernel::TileM;
        constexpr int tile_n = SelectedKernel::TileN;
        constexpr int tile_k = SelectedKernel::TileK;

        if(!pad_m && problem.M % tile_m != 0)
            return false;
        if(!pad_n && problem.N % tile_n != 0)
            return false;
        if(!pad_k && problem.K % tile_k != 0)
            return false;

        return true;
    }

    std::string get_name() const override { return name_; }

    float run(const void* a_ptr,
              const void* b_ptr,
              void* c_ptr,
              const void** d_ptrs,
              const Problem& problem,
              void* stream) const override
    {
        (void)d_ptrs; // Not used in basic GEMM

        // Create arguments using constructor (correct order!)
        // Order from GemmHostArgs constructor: a_ptr, b_ptr, e_ptr, k_batch, M, N, K, stride_A,
        // stride_B, stride_E
        ck_tile::GemmHostArgs args(a_ptr,           // a_ptr
                                   b_ptr,           // b_ptr
                                   c_ptr,           // e_ptr/c_ptr
                                   problem.k_batch, // k_batch (4th argument!)
                                   problem.M,       // M
                                   problem.N,       // N
                                   problem.K,       // K
                                   problem.K,       // stride_A (row-major A: stride = K)
                                   problem.K,       // stride_B (column-major B: stride = K)
                                   problem.N        // stride_E/C (row-major C: stride = N)
        );

        // Benchmark parameters. Defaults match old Tile Engine
        // (gemm_common.hpp: warmup=50, repeat=100, flush_cache=true,
        // rotating_count=1000) so bridge-vs-old-TE comparisons are
        // apples-to-apples and a too-small warmup can't leave the GPU clock
        // un-ramped (the transient that produced PR #8123's spurious 2048^3 dip).
        // Each is overridable at run time via env var so a caller can match a
        // different harness without recompiling.
        const bool bench = this->benchmarking_;
        ck_tile::stream_config stream_cfg;
        stream_cfg.stream_id_      = reinterpret_cast<hipStream_t>(stream);
        stream_cfg.time_kernel_    = bench;
        stream_cfg.log_level_      = 0;
        stream_cfg.cold_niters_    = bench ? env_int("CK_TILE_BENCH_WARMUP", 50) : 0;
        stream_cfg.nrepeat_        = bench ? env_int("CK_TILE_BENCH_REPEAT", 100) : 1;
        stream_cfg.is_gpu_timer_   = bench;
        stream_cfg.flush_cache_    = bench && env_bool("CK_TILE_BENCH_FLUSH", true);
        stream_cfg.rotating_count_ = bench ? env_int("CK_TILE_BENCH_ROTATING", 1000) : 1;

        // Call the generated kernel's launch method
        return SelectedKernel::launch(args, stream_cfg);
    }

    bool validate(const void* a_ptr,
                  const void* b_ptr,
                  const void* c_ptr,
                  const void** d_ptrs,
                  const Problem& problem,
                  float tolerance) const override
    {
        (void)a_ptr;
        (void)b_ptr;
        (void)c_ptr;
        (void)d_ptrs;
        (void)problem;
        (void)tolerance;
        // Validation would require reference implementation
        return true;
    }

    private:
    // Read an integer benchmark knob from the environment, falling back to
    // `fallback` when unset or unparseable.
    static int env_int(const char* name, int fallback)
    {
        const char* v = std::getenv(name);
        if(v == nullptr || *v == '\0')
            return fallback;
        char* end      = nullptr;
        const long out = std::strtol(v, &end, 10);
        if(end == v)
            return fallback;
        return static_cast<int>(out);
    }

    // Read a boolean benchmark knob ("0"/"false"/"off" => false, else true).
    static bool env_bool(const char* name, bool fallback)
    {
        const char* v = std::getenv(name);
        if(v == nullptr || *v == '\0')
            return fallback;
        std::string s(v);
        return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
    }

    KernelKey key_;
    std::string name_;
};

/// Helper function to create a generated tile kernel instance wrapper
template <typename SelectedKernel,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType>
std::shared_ptr<KernelInstance> create_generated_tile_kernel(const KernelKey& key,
                                                             const std::string& name)
{
    return std::make_shared<
        GeneratedTileKernelInstance<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>>(
        key, name);
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
