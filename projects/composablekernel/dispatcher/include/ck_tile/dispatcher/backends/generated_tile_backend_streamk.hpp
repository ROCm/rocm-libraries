// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_kernel.hpp"
#include <hip/hip_runtime.h>
#include <string>

namespace ck_tile {
namespace dispatcher {
namespace backends {

/**
 * Kernel-instance wrapper for unified_gemm_codegen.py Stream-K kernels.
 *
 * Counterpart of GeneratedTileKernelInstance (regular GEMM) for the Stream-K
 * variant. The difference is the host-args type: Stream-K needs
 * ck_tile::StreamKHostArgs (workspace pointer + reduction strategy), which is
 * ABI-incompatible with the GemmHostArgs path -- this is exactly why Stream-K
 * could not previously ride the registry. With this backend it can: the
 * Dispatcher selects the instance by KernelKey (which now carries streamk +
 * reduction_strategy) and calls run().
 *
 * supports() gates on the requested reduction strategy so that the registry can
 * hold atomic/linear/tree side by side and the Dispatcher's first-fit selection
 * picks the one the caller asked for via Problem::reduction_strategy.
 *
 * NOTE (PR-C): the generated SelectedKernel::launch(StreamKHostArgs, stream)
 * still owns the reduction workspace internally (DeviceMem) and does the
 * per-iter reset. PR-D relocates workspace ownership + reset to Dispatcher::run()
 * via get_workspace_size()/the workspace-aware run() overload.
 */
template <typename SelectedKernelType,
          typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_>
class GeneratedStreamKKernelInstance : public KernelInstance
{
    public:
    using ADataType      = ADataType_;
    using BDataType      = BDataType_;
    using CDataType      = CDataType_;
    using AccDataType    = AccDataType_;
    using SelectedKernel = SelectedKernelType;

    GeneratedStreamKKernelInstance(const KernelKey& key, const std::string& name)
        : key_(key), name_(name)
    {
    }

    const KernelKey& get_key() const override { return key_; }

    std::string get_name() const override { return name_; }

    /// Accept ONLY when the caller requested a Stream-K kernel with THIS
    /// instance's reduction strategy. Lets atomic/linear/tree coexist in the
    /// registry and be selected by Problem::reduction_strategy.
    bool supports(const Problem& problem) const override
    {
        if(!problem.streamk)
            return false;
        if(problem.reduction_strategy != key_.algorithm.reduction_strategy)
            return false;

        // Stream-K distributes K-iterations across workgroups; padding flags
        // mirror the regular backend's divisibility guard. Final feasibility
        // (enough tiles to partition) is enforced by the kernel at launch.
        constexpr bool pad_m = SelectedKernel::kPadM;
        constexpr bool pad_n = SelectedKernel::kPadN;
        constexpr bool pad_k = SelectedKernel::kPadK;
        if(!pad_m && problem.M % SelectedKernel::TileM != 0)
            return false;
        if(!pad_n && problem.N % SelectedKernel::TileN != 0)
            return false;
        if(!pad_k && problem.K % SelectedKernel::TileK != 0)
            return false;
        return true;
    }

    float run(const void* a_ptr,
              const void* b_ptr,
              void* c_ptr,
              const void** d_ptrs,
              const Problem& problem,
              void* stream) const override
    {
        (void)d_ptrs; // Not used for Stream-K GEMM

        // rcr strides: row-major A (K), column-major B (K), row-major C (N).
        // k_batch is owned by the Stream-K tile partitioner, not passed here.
        // StreamKHostArgs uses ck_tile::index_t (int32); cast from Problem's int64.
        using idx = ck_tile::index_t;
        ck_tile::StreamKHostArgs args{a_ptr,
                                      b_ptr,
                                      c_ptr,
                                      static_cast<idx>(problem.M),
                                      static_cast<idx>(problem.N),
                                      static_cast<idx>(problem.K),
                                      static_cast<idx>(problem.K),
                                      static_cast<idx>(problem.K),
                                      static_cast<idx>(problem.N)};

        const bool bench = this->benchmarking_;
        ck_tile::stream_config stream_cfg;
        stream_cfg.stream_id_      = reinterpret_cast<hipStream_t>(stream);
        stream_cfg.time_kernel_    = bench;
        stream_cfg.log_level_      = 0;
        stream_cfg.cold_niters_    = bench ? 5 : 0;
        stream_cfg.nrepeat_        = bench ? 10 : 1;
        stream_cfg.is_gpu_timer_   = bench;
        stream_cfg.flush_cache_    = false;
        stream_cfg.rotating_count_ = 1; // atomic accumulates into C; never rotate

        // PR-C: workspace + reset are owned inside the generated launch().
        // PR-D will switch to the workspace-aware run() overload so the
        // Dispatcher owns the buffer.
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
        return true; // reference validation handled by the TE/driver harness
    }

    private:
    KernelKey key_;
    std::string name_;
};

/// Helper to create a Stream-K kernel-instance wrapper.
template <typename SelectedKernel,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType>
std::shared_ptr<KernelInstance> create_generated_streamk_kernel(const KernelKey& key,
                                                                const std::string& name)
{
    return std::make_shared<GeneratedStreamKKernelInstance<SelectedKernel,
                                                           ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           AccDataType>>(key, name);
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
