// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
#include "ck_tile/ops/swiglu/swiglu_pipeline.hpp"

namespace ck_tile {

struct SwiGLUHostArgs
{
    const void* a_ptr;
    const void* b0_ptr;
    const void* b1_ptr;
    void* c_ptr;

    index_t m;
    index_t n;
    index_t k;

    index_t a_stride;
    index_t b0_stride;
    index_t b1_stride;
    index_t c_stride;
    index_t k_batch = 1;
};

template <typename TilePartitioner_,
          typename BaseGemmPipeline_,
          typename EpiloguePipeline_,
          typename ActMulOp_>
struct SwiGLUKernel
{
    using ActMulOp         = ActMulOp_;
    using TilePartitioner  = TilePartitioner_;
    using GemmPipeline     = SwiGLUPipeline<BaseGemmPipeline_, ActMulOp>;
    using EpiloguePipeline = EpiloguePipeline_;

    // static_assert(GemmEpilogue::DsDataType == {});

    using Kernel = ck_tile::UniversalGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;

    static constexpr Kernel gemm{};

    using KernelArgs = typename Kernel::KernelArgs;
    using HostArgs   = SwiGLUHostArgs;

    CK_TILE_HOST static KernelArgs MakeKernelArgs(const HostArgs& args)
    {
        UniversalGemmHostArgs<1, 2, 0> host_args{
            {args.a_ptr},
            {args.b0_ptr, args.b1_ptr},
            {},
            args.c_ptr,
            1,
            args.m,
            args.n,
            args.k,
            {args.a_stride},
            {args.b0_stride, args.b1_stride},
            {},
            args.c_stride,
        };

        return Kernel::MakeKernelArgs(std::move(host_args));
    }

    CK_TILE_HOST static auto GetName() -> std::string
    {
        return std::string("SwiGLU_") + Kernel::GetName();
    }

    CK_TILE_HOST static auto IsSupportedArgument(const KernelArgs& kargs) -> bool
    {
        return Kernel::IsSupportedArgument(kargs);
    }

    constexpr static int kBlockSize = Kernel::kBlockSize;

    constexpr static auto BlockSize() -> dim3 { return Kernel::BlockSize(); }

    constexpr static auto GridSize(index_t m, index_t n, index_t k_batch) -> dim3
    {
        return Kernel::GridSize(m, n, k_batch);
    }

    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        return Kernel::MaxOccupancyGridSize(s);
    }

    CK_TILE_DEVICE auto operator()(KernelArgs args) const -> void { gemm(std::move(args)); }
};
} // namespace ck_tile
