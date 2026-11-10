// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/ops/epilogue/mx_alpha_beta_epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/mx_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v4.hpp"

namespace ck_tile {

namespace mx_tdm_v4 {

/// The MxTdmV4GemmTraits values rocKE derives from `spec`'s shape, as host values.
struct ShapeTraits
{
    bool edge_m;
    bool edge_n;
    bool wide_a;
    bool wide_b;
    bool wide_d;
    index_t odd_tiles;
    index_t tail_steps;
    index_t store_width;
};

CK_TILE_HOST_DEVICE constexpr ShapeTraits shape_traits(const MxTdmV4Spec& s)
{
    return {s.edge_m(),
            s.edge_n(),
            s.wide_a(),
            s.wide_b(),
            s.wide_d(),
            static_cast<index_t>(s.K / s.tile_k % 2),
            s.tail_steps(),
            s.store_width()};
}

} // namespace mx_tdm_v4

/**
 * @brief The host arguments of MxGemmTdmV4Kernel: an MX GEMM whose one D tensor is C, with alpha
 *        and beta.
 *
 * A is M x K and B is N x K, K-contiguous (stride K, two FP4 values per byte); C and D are
 * M x N row-major (stride N); the scales are K-major, [ceil(K / 128)][M or N][4] E8M0. C is not
 * read when beta == 0 and may be null. k_batch must be 1.
 */
struct MxGemmTdmV4HostArgs : MxGemmHostArgs<1, 1, 1>
{
    CK_TILE_HOST MxGemmTdmV4HostArgs(const void* a_ptr_,
                                     const void* a_scale_ptr_,
                                     const void* b_ptr_,
                                     const void* b_scale_ptr_,
                                     const void* c_ptr_,
                                     void* d_ptr_,
                                     index_t k_batch_,
                                     index_t M_,
                                     index_t N_,
                                     index_t K_,
                                     index_t stride_A_,
                                     index_t stride_B_,
                                     index_t stride_C_,
                                     index_t stride_D_,
                                     float alpha_ = 1.0f,
                                     float beta_  = 0.0f)
        : MxGemmHostArgs<1, 1, 1>({a_ptr_},
                                  {a_scale_ptr_},
                                  {b_ptr_},
                                  {b_scale_ptr_},
                                  {c_ptr_},
                                  d_ptr_,
                                  k_batch_,
                                  M_,
                                  N_,
                                  K_,
                                  {stride_A_},
                                  {stride_B_},
                                  {stride_C_},
                                  stride_D_),
          alpha(alpha_),
          beta(beta_)
    {
    }

    float alpha;
    float beta;
};

/// The device arguments of MxGemmTdmV4Kernel, made by MakeKernelArgs.
struct MxGemmTdmV4KernelArgs
{
    const void* a;       ///< M x K, K-contiguous, packed (two FP4 values per byte)
    const void* b;       ///< N x K, K-contiguous, packed
    const void* a_scale; ///< [ceil(K / 128)][M][4] E8M0
    const void* b_scale; ///< [ceil(K / 128)][N][4] E8M0
    const void* c;       ///< M x N row-major; not read when beta == 0
    void* d;             ///< M x N row-major
    index_t M;
    index_t N;
    index_t K;
    float alpha;
    float beta;
};

/**
 * @brief MX GEMM kernel of GemmPipelineAgBgCrCompTDMV4: D = alpha * A @ B^T + beta * C (rocKE
 *        mx_gemm at commit b32989439dc).
 *
 * One workgroup of 128 threads per tile_m x tile_n tile, mapped as TilePartitioner (a
 * GemmTile2DNMajorPartitioner) maps it: grid x walks N tiles, y walks M tiles. The pipeline reads
 * its tile from the workgroup ids itself, and the epilogue stores the accumulators straight to D
 * (EpiloguePipeline is an MxAlphaBetaEpilogue of this pipeline). With a cluster tuning the launch
 * must use ClusterSize() (ClusterLaunch; make_kernel's cluster overload). Compile with rocKE's
 * flags for the same schedule: -mllvm -amdgpu-expert-scheduling-mode -mllvm -enable-misched=false
 * -mllvm -enable-post-misched=false -mllvm -greedy-regclass-priority-trumps-globalness.
 */
template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct MxGemmTdmV4Kernel
    : UniversalGemmKernel<TilePartitioner_,
                          GemmPipeline_,
                          EpiloguePipeline_,
                          MxGemmTdmV4Kernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>>
{
    using Base =
        UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_, MxGemmTdmV4Kernel>;
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using Traits           = typename GemmPipeline::Traits;
    using Problem          = typename GemmPipeline::Problem;
    using Policy           = typename GemmPipeline::Policy;
    using ODataType        = typename GemmPipeline::ODataType;
    using HostArgs         = MxGemmTdmV4HostArgs;
    using KernelArgs       = MxGemmTdmV4KernelArgs;

    static constexpr index_t kBlockSize = GemmPipeline::kThreads;
    static constexpr MxTdmV4Spec kSpec  = GemmPipeline::kSpec;
    static constexpr bool kHasCluster   = GemmPipeline::kHasCluster;
    /// The cluster is the tuning's (cluster_x along N, cluster_y along M), not the tile shape's.
    static constexpr bool ClusterLaunch = kHasCluster;

    static_assert(TilePartitioner::MPerBlock == kSpec.tile_m &&
                      TilePartitioner::NPerBlock == kSpec.tile_n &&
                      TilePartitioner::KPerBlock == kSpec.tile_k,
                  "comp_tdm_v4: the tile partitioner's tile must be the pipeline's");
    static_assert(
        std::is_same_v<TilePartitioner,
                       GemmTile2DNMajorPartitioner<typename TilePartitioner::BlockGemmShape>>,
        "comp_tdm_v4: the pipeline maps grid x to N tiles and y to M tiles, as "
        "GemmTile2DNMajorPartitioner");
    static_assert(std::is_same_v<typename EpiloguePipeline::Pipeline, GemmPipeline> &&
                      EpiloguePipeline::kStoreWidth == Traits::kStoreWidth &&
                      EpiloguePipeline::kEdgeM == Traits::kEdgeM &&
                      EpiloguePipeline::kEdgeN == Traits::kEdgeN &&
                      EpiloguePipeline::kWideD == Traits::kWideD,
                  "comp_tdm_v4: the epilogue must store this pipeline's accumulators with its "
                  "traits' store width, EdgeM, EdgeN and WideD");
    static_assert(Traits::kStoreWidth >= 1 && Traits::kStoreWidth <= kSpec.max_store_width() &&
                      kSpec.wave_repeat_n() % Traits::kStoreWidth == 0 &&
                      (Traits::kStoreWidth & (Traits::kStoreWidth - 1)) == 0,
                  "comp_tdm_v4: the D store width must be a power of 2 dividing the N repeat, "
                  "at most 16 B");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        std::string name = std::string("mx_gemm_tdm_v4_mx") + mx_tdm_v4_dtype_name(kSpec.a_dtype);
        if(kSpec.b_dtype != kSpec.a_dtype)
            name += mx_tdm_v4_dtype_name(kSpec.b_dtype);
        name += "_t" + std::to_string(kSpec.tile_m) + "x" + std::to_string(kSpec.tile_n) + "x" +
                std::to_string(kSpec.tile_k) + mx_tdm_v4_tuning_name(kSpec);
        if(Traits::kEdgeM || Traits::kEdgeN)
            name += "_edge";
        if(Traits::kWideA || Traits::kWideB || Traits::kWideD)
            name += "_wide";
        if(Traits::kOddTiles >= 0)
            name += Traits::kOddTiles ? "_odd" : "_even";
        if(Traits::kHasTail)
            name += "_tail" + std::to_string(Traits::kTailSteps);
        return name + "_w" + std::to_string(Traits::kStoreWidth);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return GemmPipeline::GetSmemSize();
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static dim3 GridSize(index_t M, index_t N)
    {
        return TilePartitioner::GridSize(M, N);
    }

    CK_TILE_HOST static constexpr dim3 ClusterSize()
    {
        return dim3(kSpec.cluster_x, kSpec.cluster_y, kSpec.cluster_z);
    }

    CK_TILE_HOST static constexpr KernelArgs MakeKernelArgs(const HostArgs& h)
    {
        return KernelArgs{h.as_ptr[0],
                          h.bs_ptr[0],
                          h.as_scale_ptr[0],
                          h.bs_scale_ptr[0],
                          h.ds_ptr[0],
                          h.e_ptr,
                          h.M,
                          h.N,
                          h.K,
                          h.alpha,
                          h.beta};
    }

    /// nullptr if this instance runs `kargs`, else why not: rocKE's validate() of the full spec,
    /// or a shape path the traits lack.
    CK_TILE_HOST static const char* GetUnsupportedReason(const KernelArgs& kargs)
    {
        const MxTdmV4Spec spec     = Policy::template GetSpec<Problem>(kargs.M, kargs.N, kargs.K);
        const MxTdmV4Reason reason = mx_tdm_v4_validate(spec);
        if(reason != MxTdmV4Reason::Ok)
            return mx_tdm_v4_reason_message(reason);
        const mx_tdm_v4::ShapeTraits need = mx_tdm_v4::shape_traits(spec);
        if(need.edge_m && !Traits::kEdgeM)
            return "M is not a multiple of tile_m: needs an EdgeM instance";
        if(need.edge_n && !Traits::kEdgeN)
            return "N is not a multiple of tile_n: needs an EdgeN instance";
        if(need.wide_a && !Traits::kWideA)
            return "A's offsets pass 2^31: needs a WideA instance";
        if(need.wide_b && !Traits::kWideB)
            return "B's offsets pass 2^31: needs a WideB instance";
        if(need.wide_d && !Traits::kWideD)
            return "D's offsets pass 2^31: needs a WideD instance";
        if(Traits::kOddTiles >= 0 && need.odd_tiles != Traits::kOddTiles)
            return "the tile count K / tile_k does not have the instance's OddTiles parity";
        if(need.tail_steps != Traits::kTailSteps)
            return "the K tail (K mod tile_k, in 128-wide K steps) is not the instance's TailSteps";
        if(kargs.N % Traits::kStoreWidth != 0)
            return "the store width does not divide N";
        if(kargs.a == nullptr || kargs.b == nullptr || kargs.a_scale == nullptr ||
           kargs.b_scale == nullptr || kargs.d == nullptr ||
           (kargs.beta != 0.0f && kargs.c == nullptr))
            return "null operand";
        return nullptr;
    }

    /// GetUnsupportedReason of MakeKernelArgs(h), after what KernelArgs cannot express: packed
    /// strides and k_batch == 1.
    CK_TILE_HOST static const char* GetUnsupportedReason(const HostArgs& h)
    {
        if(h.k_batch != 1)
            return "k_batch must be 1";
        if(h.stride_As[0] != h.K || h.stride_Bs[0] != h.K)
            return "A and B must be packed: their strides must be K";
        if(h.stride_E != h.N || (h.beta != 0.0f && h.stride_Ds[0] != h.N))
            return "C and D must be packed: their strides must be N";
        return GetUnsupportedReason(MakeKernelArgs(h));
    }

    template <typename Args>
    CK_TILE_HOST static bool IsSupportedArgument(const Args& args)
    {
        const char* why = GetUnsupportedReason(args);
        if(why != nullptr && EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            CK_TILE_ERROR(GetName(), ": ", why);
        return why == nullptr;
    }

    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
#if defined(__gfx125__)
        __shared__ alignas(1024) uint8_t smem[GetSmemSize()];
        const mx_tdm_v4::Coords coords = GemmPipeline::MakeCoords();
        const mx_tdm_v4::Operands operands{static_cast<const uint8_t*>(kargs.a),
                                           static_cast<const uint8_t*>(kargs.b),
                                           static_cast<const uint8_t*>(kargs.a_scale),
                                           static_cast<const uint8_t*>(kargs.b_scale),
                                           kargs.M,
                                           kargs.N,
                                           kargs.K};
        typename GemmPipeline::CVec acc[GemmPipeline::kNumAcc];
        GemmPipeline{}(
            operands, coords, c_style_pointer_cast<CK_TILE_LDS_ADDR uint8_t*>(smem), acc);
        EpiloguePipeline{}(
            kargs.c, kargs.d, kargs.M, kargs.N, kargs.alpha, kargs.beta, coords, acc);
#else
        ignore = kargs;
#endif
    }
};

} // namespace ck_tile
