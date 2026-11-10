// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/ops/epilogue/mx_alpha_beta_epilogue.hpp"
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
 * @brief MX GEMM kernel of GemmPipelineAgBgCrCompTDMV4: D = alpha * A @ B^T + beta * C (rocKE
 *        mx_gemm at commit b32989439dc).
 *
 * One workgroup of 128 threads per tile_m x tile_n tile: grid x walks N tiles, y walks M tiles.
 * With a cluster the launch must use ClusterSize() (make_kernel's cluster overload). Compile with
 * rocKE's flags for the same schedule: -mllvm -amdgpu-expert-scheduling-mode -mllvm
 * -enable-misched=false -mllvm -enable-post-misched=false -mllvm
 * -greedy-regclass-priority-trumps-globalness.
 */
template <typename Pipeline_>
struct MxGemmTdmV4Kernel
{
    using Pipeline  = remove_cvref_t<Pipeline_>;
    using Traits    = typename Pipeline::Traits;
    using Problem   = typename Pipeline::Problem;
    using Policy    = typename Pipeline::Policy;
    using ODataType = typename Pipeline::ODataType;
    using Epilogue  = MxAlphaBetaEpilogue<MxAlphaBetaEpilogueProblem<Pipeline,
                                                                     Traits::kStoreWidth,
                                                                     Traits::kEdgeM,
                                                                     Traits::kEdgeN,
                                                                     Traits::kWideD>>;

    static constexpr index_t kBlockSize = Pipeline::kThreads;
    static constexpr MxTdmV4Spec kSpec  = Pipeline::kSpec;
    static constexpr bool kHasCluster   = Pipeline::kHasCluster;

    static_assert(Traits::kStoreWidth >= 1 && Traits::kStoreWidth <= kSpec.max_store_width() &&
                      kSpec.wave_repeat_n() % Traits::kStoreWidth == 0 &&
                      (Traits::kStoreWidth & (Traits::kStoreWidth - 1)) == 0,
                  "comp_tdm_v4: the D store width must be a power of 2 dividing the N repeat, "
                  "at most 16 B");

    struct Args
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

    CK_TILE_HOST static std::string GetName()
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

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static dim3 GridSize(index_t M, index_t N)
    {
        return dim3(static_cast<uint32_t>(integer_divide_ceil(N, kSpec.tile_n)),
                    static_cast<uint32_t>(integer_divide_ceil(M, kSpec.tile_m)),
                    1);
    }

    CK_TILE_HOST static constexpr dim3 ClusterSize()
    {
        return dim3(kSpec.cluster_x, kSpec.cluster_y, kSpec.cluster_z);
    }

    CK_TILE_HOST static constexpr Args MakeKernelArgs(const Args& args) { return args; }

    /// nullptr if this instance runs `args`, else why not: rocKE's validate() of the full spec,
    /// or a shape path the traits lack.
    CK_TILE_HOST static const char* GetUnsupportedReason(const Args& args)
    {
        const MxTdmV4Spec spec     = Policy::template GetSpec<Problem>(args.M, args.N, args.K);
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
        if(args.N % Traits::kStoreWidth != 0)
            return "the store width does not divide N";
        if(args.a == nullptr || args.b == nullptr || args.a_scale == nullptr ||
           args.b_scale == nullptr || args.d == nullptr || (args.beta != 0.0f && args.c == nullptr))
            return "null operand";
        return nullptr;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& args)
    {
        return GetUnsupportedReason(args) == nullptr;
    }

    CK_TILE_DEVICE void operator()(Args args) const
    {
#if defined(__gfx125__)
        __shared__ alignas(1024) uint8_t smem[GetSmemSize()];
        const mx_tdm_v4::Coords coords = Pipeline::MakeCoords();
        const mx_tdm_v4::Operands operands{static_cast<const uint8_t*>(args.a),
                                           static_cast<const uint8_t*>(args.b),
                                           static_cast<const uint8_t*>(args.a_scale),
                                           static_cast<const uint8_t*>(args.b_scale),
                                           args.M,
                                           args.N,
                                           args.K};
        typename Pipeline::CVec acc[Pipeline::kNumAcc];
        Pipeline{}(operands, coords, c_style_pointer_cast<CK_TILE_LDS_ADDR uint8_t*>(smem), acc);
        Epilogue{}(args.c, args.d, args.M, args.N, args.alpha, args.beta, coords, acc);
#else
        ignore = args;
#endif
    }
};

} // namespace ck_tile
