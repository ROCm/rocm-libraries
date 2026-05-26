// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"

// Intentional duplication of the MoE FlatMM host-side helpers from
// `example/ck_tile/18_flatmm/moe_flatmm_impl.hpp`.
//
// Keeping these test-local avoids a `test -> example` include dependency while
// still exercising the same host-side shuffle and dispatch logic shape.

template <typename Layout>
static constexpr inline auto is_row_major(Layout /*layout*/)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename FlatmmConfig, typename T>
auto flatmm_shuffle_b(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    constexpr int MaxVecSize     = 16 / sizeof(T);
    constexpr int KLane          = ck_tile::get_warp_size() / FlatmmConfig::N_Warp_Tile;
    constexpr int ItemsPerAccess = std::min(MaxVecSize, FlatmmConfig::K_Warp_Tile / KLane);

    ck_tile::HostTensor<T> t_view({n_ / FlatmmConfig::N_Warp_Tile,
                                   FlatmmConfig::N_Warp_Tile,
                                   k_ / ItemsPerAccess,
                                   ItemsPerAccess});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 2, 1, 3});
}

template <class FlatmmConfig, ck_tile::MoeFlatmmKind moe_kind, class IterSrc, class IterDst>
void shuffle_mxfp4_weight(const IterSrc src, IterDst dst, int experts_cnt, int N, int K)
{
    int KPack = 16;
    int NLane = FlatmmConfig::N_Warp_Tile;
    int KLane = 64 / NLane;
    int K_pk  = K / 2;
    int K0    = K_pk / (KLane * KPack);
    int tempk;

    if constexpr(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        const int up_stride = N / 2 / NLane;

        for(long eid = 0; eid < experts_cnt; ++eid)
        {
            for(int n = 0; n < N; ++n)
            {
                for(int k = 0; k < K_pk; ++k)
                {
                    int n0 = n / NLane;
                    int n1 = n % NLane;

                    int n0_interleave = n >= N / 2 ? (n0 - up_stride) * 2 + 1 : n0 * 2;

                    int k0 = k / (KLane * KPack);
                    tempk  = k % (KLane * KPack);
                    int k1 = tempk / KPack;
                    int k2 = tempk % KPack;

                    long outputIndex = eid * N * K_pk + n0_interleave * KPack * NLane * KLane * K0 +
                                       k0 * KPack * NLane * KLane + k1 * KPack * NLane +
                                       n1 * KPack + k2;

                    dst[outputIndex] = src[eid * N * K_pk + n * K_pk + k];
                }
            }
        }
    }
    else
    {
        for(long eid = 0; eid < experts_cnt; ++eid)
        {
            for(int n = 0; n < N; ++n)
            {
                for(int k = 0; k < K_pk; ++k)
                {
                    int n0 = n / NLane;
                    int n1 = n % NLane;

                    int k0 = k / (KLane * KPack);
                    tempk  = k % (KLane * KPack);
                    int k1 = tempk / KPack;
                    int k2 = tempk % KPack;

                    long outputIndex = eid * N * K_pk + n0 * KPack * NLane * KLane * K0 +
                                       k0 * KPack * NLane * KLane + k1 * KPack * NLane +
                                       n1 * KPack + k2;

                    dst[outputIndex] = src[eid * N * K_pk + n * K_pk + k];
                }
            }
        }
    }
}

template <typename FlatmmConfig, ck_tile::MoeFlatmmKind moe_kind, typename T>
auto shuffle_mxfp4_scale(const ck_tile::HostTensor<T>& scale, int experts_cnt)
{
    assert(scale.get_lengths().size() == 2);
    int n_ = scale.get_lengths()[1];
    int k_ = scale.get_lengths()[0];

    int k_per_expert = k_ / experts_cnt;

    constexpr int K_Pack       = 2;
    constexpr int N_Pack       = 2;
    constexpr int GranularityK = 32;

    constexpr int K_Lane = 64 / FlatmmConfig::N_Warp_Tile;

    static_assert(FlatmmConfig::N_Warp_Tile == 16, "only support XDL_N == 16");
    static_assert(FlatmmConfig::N_Repeat % N_Pack == 0);
    static_assert(FlatmmConfig::K_Tile % (K_Pack * K_Lane * GranularityK) == 0);

    if constexpr(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        ck_tile::HostTensor<T> shfl_scale({
            experts_cnt,
            k_per_expert / K_Pack / K_Lane,
            K_Pack,
            K_Lane,
            N_Pack,
            n_ / FlatmmConfig::N_Warp_Tile / N_Pack,
            FlatmmConfig::N_Warp_Tile,
        });
        std::copy(scale.begin(), scale.end(), shfl_scale.begin());
        return ck_tile::reference_permute(shfl_scale, {0, 5, 1, 3, 6, 2, 4});
    }
    else
    {
        ck_tile::HostTensor<T> shfl_scale({
            experts_cnt,
            k_per_expert / K_Pack / K_Lane,
            K_Pack,
            K_Lane,
            n_ / FlatmmConfig::N_Warp_Tile / N_Pack,
            N_Pack,
            FlatmmConfig::N_Warp_Tile,
        });
        std::copy(scale.begin(), scale.end(), shfl_scale.begin());
        return ck_tile::reference_permute(shfl_scale, {0, 4, 1, 3, 6, 2, 5});
    }
}

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ck_tile::MoeFlatmmKind moe_kind = ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only,
          typename CDEElementWise         = ck_tile::element_wise::PassThrough,
          typename ScaleM,
          typename ScaleN>
float moe_gemm(const ck_tile::MoeFlatmmHostArgs<ScaleM, ScaleN>& args,
               const ck_tile::stream_config& s)
{
    using CodegenFlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                           FlatmmConfig::kPadN,
                                           FlatmmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           FlatmmConfig::NumWaveGroups>;

    using CodegenGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                               FlatmmConfig::kPadN,
                                                               FlatmmConfig::kPadK,
                                                               FlatmmConfig::DoubleSmemBuffer,
                                                               ALayout,
                                                               BLayout,
                                                               ELayout,
                                                               FlatmmConfig::TransposeC,
                                                               FlatmmConfig::UseStructuredSparsity,
                                                               false,
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>;

    if constexpr(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        static_assert(
            FlatmmConfig::N_Tile % (FlatmmConfig::N_Warp * FlatmmConfig::N_Warp_Tile * 2) == 0,
            "requires NRepeat is multiple of 2 for FFN_gemm1_gate_up");
    }

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenFlatmmShape, Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr auto scheduler      = FlatmmConfig::Scheduler;

        using CodegenPipelineProblem = ck_tile::FlatmmPipelineProblem<ADataType,
                                                                      BDataType,
                                                                      AccDataType,
                                                                      CodegenFlatmmShape,
                                                                      CodegenGemmTraits,
                                                                      scheduler,
                                                                      has_hot_loop_v,
                                                                      tail_number_v>;

        constexpr int BlockedXDLN_PerWarp =
            moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up ? 2 : 1;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDatatype,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             FlatmmConfig::NumWaveGroups,
                                             false,
                                             1,
                                             BlockedXDLN_PerWarp>>;

        using CodegenFlatmmPipeline =
            ck_tile::MoeFlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using Kernel = ck_tile::
            MoeFlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue, moe_kind>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:" << CodegenFlatmmShape::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;
            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm2 ? args.NumTokens * args.TopK
                                                               : args.NumTokens,
                args.K,
                args.stride_A,
                is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N * args.NumExperts, args.stride_B, is_row_major(BLayout{})));

            const int outputN =
                moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up ? args.N / 2 : args.N;

            auto size_a_buffer = a_m.get_element_space_size_in_bytes();
            auto size_b_buffer = b_n.get_element_space_size_in_bytes();

            ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                ck_tile::flush_icache();
                rotating_mem.Next();
                if(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm2)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.NumTokens * args.N * sizeof(CDataType), s.stream_id_));
                else if(args.k_batch > 1)
                    hipGetErrorString(
                        hipMemsetAsync(args.e_ptr,
                                       0,
                                       args.NumTokens * args.TopK * outputN * sizeof(CDataType),
                                       s.stream_id_));
            };
            return ck_tile::launch_kernel_time_mask(
                s,
                run_flush_cache,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            return ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
    };

    return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
}

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ck_tile::MoeFlatmmKind moe_kind = ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only,
          typename CDEElementWise         = ck_tile::element_wise::PassThrough,
          typename MoeFlatmmHostArgs>
float mx_moe_gemm(const MoeFlatmmHostArgs& args, const ck_tile::stream_config& s)
{
    using CodegenFlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using CodegenGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                               FlatmmConfig::kPadN,
                                                               FlatmmConfig::kPadK,
                                                               FlatmmConfig::DoubleSmemBuffer,
                                                               ALayout,
                                                               BLayout,
                                                               ELayout,
                                                               FlatmmConfig::TransposeC,
                                                               FlatmmConfig::UseStructuredSparsity,
                                                               false,
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>;

    using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                           FlatmmConfig::kPadN,
                                           FlatmmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           FlatmmConfig::NumWaveGroups>;

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "MX MoE requires sizeof(ADataType) >= sizeof(BDataType)");

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ComputeDataType,
                                                             ComputeDataType,
                                                             AccDataType,
                                                             CodegenFlatmmShape,
                                                             Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr auto scheduler      = FlatmmConfig::Scheduler;

        using CodegenPipelineProblem = ck_tile::F8xMXF4FlatmmPipelineProblem<ADataType,
                                                                             BDataType,
                                                                             AccDataType,
                                                                             CodegenFlatmmShape,
                                                                             CodegenGemmTraits,
                                                                             scheduler,
                                                                             has_hot_loop_v,
                                                                             tail_number_v>;

        constexpr int BlockedXDLN_PerWarp = 2;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                             ComputeDataType,
                                             DsDatatype,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             FlatmmConfig::NumWaveGroups,
                                             false,
                                             1,
                                             BlockedXDLN_PerWarp>>;

        using CodegenFlatmmPipeline =
            ck_tile::F8xMXF4FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using Kernel = ck_tile::
            MoeFlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue, moe_kind>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel " << Kernel::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << "\n"
                      << "k_batch: " << kargs.k_batch << std::endl;
        }

        ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        return ave_time;
    };

    BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    return ave_time;
}

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ck_tile::MoeFlatmmKind moe_kind = ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up,
          typename CDEElementWise         = ck_tile::element_wise::PassThrough,
          typename MoeFlatmmHostArgs>
float a16w4_moe_gemm(const MoeFlatmmHostArgs& args, const ck_tile::stream_config& s)
{
    using CodegenFlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                           FlatmmConfig::kPadN,
                                           FlatmmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           FlatmmConfig::NumWaveGroups>;

    using CodegenGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                               FlatmmConfig::kPadN,
                                                               FlatmmConfig::kPadK,
                                                               FlatmmConfig::DoubleSmemBuffer,
                                                               ALayout,
                                                               BLayout,
                                                               ELayout,
                                                               FlatmmConfig::TransposeC,
                                                               FlatmmConfig::UseStructuredSparsity,
                                                               false,
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>;

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "A16W4 MoE requires ADataType is a wider type than BDataType");
    static_assert(std::is_same_v<BDataType, ck_tile::pk_fp4_t>,
                  "a16w4_moe_gemm only supports BDataType = pk_fp4_t");

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ComputeDataType,
                                                             ComputeDataType,
                                                             AccDataType,
                                                             CodegenFlatmmShape,
                                                             Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr auto scheduler      = FlatmmConfig::Scheduler;

        using CodegenPipelineProblem = ck_tile::F16xMXF4FlatmmPipelineProblem<ADataType,
                                                                              BDataType,
                                                                              AccDataType,
                                                                              CodegenFlatmmShape,
                                                                              CodegenGemmTraits,
                                                                              scheduler,
                                                                              has_hot_loop_v,
                                                                              tail_number_v>;

        constexpr int BlockedXDLN_PerWarp = 2;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                             ComputeDataType,
                                             DsDatatype,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             FlatmmConfig::NumWaveGroups,
                                             false,
                                             1,
                                             BlockedXDLN_PerWarp>>;

        using CodegenFlatmmPipeline =
            ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using FusedAct = ck_tile::moe::Swiglu;

        using Kernel = ck_tile::MoeFlatmmKernel<TilePartitioner,
                                                CodegenFlatmmPipeline,
                                                GemmEpilogue,
                                                moe_kind,
                                                FusedAct>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel " << Kernel::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << "\nk_batch: " << kargs.k_batch << std::endl;
        }

        ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        return ave_time;
    };

    BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    return ave_time;
}
