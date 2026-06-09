// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

template <typename Tuple>
class TestCkTileGemmFusedAQuantStandalone : public ::testing::Test
{
    protected:
    using ALayout    = std::tuple_element_t<0, Tuple>;
    using BLayout    = std::tuple_element_t<1, Tuple>;
    using CLayout    = std::tuple_element_t<2, Tuple>;
    using AQLayout   = std::tuple_element_t<3, Tuple>;
    using ADataType  = std::tuple_element_t<4, Tuple>;
    using BDataType  = std::tuple_element_t<5, Tuple>;
    using QDataType  = std::tuple_element_t<6, Tuple>;
    using CDataType  = std::tuple_element_t<7, Tuple>;
    using GemmConfig = std::tuple_element_t<9, Tuple>;

    using QuantGroupSize  = std::tuple_element_t<10, Tuple>;
    using AQuantGroupSize = QuantGroupSize;
    using BQuantGroupSize = std::tuple_element_t<11, Tuple>;
    using BQLayout        = std::tuple_element_t<12, Tuple>;

    using QuantTypeTag              = std::tuple_element_t<8, Tuple>;
    static constexpr auto QuantType = QuantTypeTag::value;

    using AccDataType = float;

    static_assert(QuantType == ck_tile::QuantType::ABQuantGrouped,
                  "Standalone fusedAquant tests currently support ABQuantGrouped only.");
    using ComputeDataType = void;

    static constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
    static constexpr ck_tile::index_t N_Tile = GemmConfig::N_Tile;
    static constexpr ck_tile::index_t K_Tile = GemmConfig::K_Tile;

    static constexpr ck_tile::index_t M_Warp = GemmConfig::M_Warp;
    static constexpr ck_tile::index_t N_Warp = GemmConfig::N_Warp;
    static constexpr ck_tile::index_t K_Warp = GemmConfig::K_Warp;

    static constexpr ck_tile::index_t M_Warp_Tile = GemmConfig::M_Warp_Tile;
    static constexpr ck_tile::index_t N_Warp_Tile = GemmConfig::N_Warp_Tile;
    static constexpr ck_tile::index_t K_Warp_Tile = GemmConfig::K_Warp_Tile;

    static constexpr bool APreshuffleQuant = GemmConfig::APreshuffleQuant;
    static constexpr bool BPreshuffleQuant = GemmConfig::BPreshuffleQuant;
    static constexpr bool PreshuffleB      = GemmConfig::PreshuffleB;
    static constexpr bool TiledMMAPermuteN = GemmConfig::TiledMMAPermuteN;
    static constexpr bool DoubleSmemBuffer = GemmConfig::DoubleSmemBuffer;
    static constexpr bool FuseAQuant       = GemmConfig::FuseAQuant;

    static constexpr bool kPadM = GemmConfig::kPadM;
    static constexpr bool kPadN = GemmConfig::kPadN;
    static constexpr bool kPadK = GemmConfig::kPadK;

    static_assert(FuseAQuant, "This fixture is only for FuseAQuant configurations.");

    template <typename Layout>
    static constexpr auto is_row_major(Layout)
    {
        return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(Layout{})>,
                                                     ck_tile::tensor_layout::gemm::RowMajor>>{};
    }

    template <typename ADataType_, typename BDataType_, typename AccDataType_, typename CDataType_>
    auto calculate_rtol_atol(const ck_tile::index_t K,
                             const ck_tile::index_t kbatch,
                             const float max_accumulated_value)
    {
        using ComputeType = std::conditional_t<
            std::is_same_v<BDataType_, ck_tile::pk_fp4_t>,
            ADataType_,
            std::conditional_t<sizeof(ADataType_) < sizeof(BDataType_), ADataType_, BDataType_>>;

        const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType_, AccDataType_>(
            ck_tile::integer_divide_ceil(K, kbatch));
        const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType_, AccDataType_>(
            max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

        const auto rtol_split_k =
            ck_tile::get_relative_threshold<CDataType_, CDataType_, CDataType_>(kbatch);
        const auto atol_split_k =
            ck_tile::get_absolute_threshold<CDataType_, CDataType_, CDataType_>(
                max_accumulated_value, kbatch);

        return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
    }

    void invoke_quant_gemm(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t WaveSize     = 32;
        constexpr ck_tile::index_t MIterPerWarp = M_Tile / (M_Warp * M_Warp_Tile);
        constexpr bool SupportVectorSize16 =
            (M_Warp_Tile * K_Warp_Tile * sizeof(ADataType) * MIterPerWarp / WaveSize) % 16 == 0;
        constexpr int VectorSize = PreshuffleB ? (SupportVectorSize16 ? 16 : 8) : 16;

        using CodegenGemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

        using CodegenGemmTraits = ck_tile::TileGemmQuantTraits<kPadM,
                                                               kPadN,
                                                               kPadK,
                                                               APreshuffleQuant,
                                                               BPreshuffleQuant,
                                                               PreshuffleB,
                                                               ALayout,
                                                               BLayout,
                                                               CLayout,
                                                               QuantType,
                                                               AQLayout,
                                                               BQLayout,
                                                               GemmConfig::TransposeC,
                                                               DoubleSmemBuffer,
                                                               false,
                                                               VectorSize,
                                                               FuseAQuant>;

        this->template run_quant_gemm_impl<CodegenGemmShape, TilePartitioner, CodegenGemmTraits>(
            args, s);
    }

    void run_test_with_validation(ck_tile::index_t M,
                                  ck_tile::index_t N,
                                  ck_tile::index_t K,
                                  ck_tile::index_t k_batch      = 1,
                                  ck_tile::index_t stride_B_pad = 0)
    {
        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{})) + stride_B_pad;
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));

        const ck_tile::index_t AQK = ck_tile::integer_divide_ceil(K, AQuantGroupSize::kK);
        const ck_tile::index_t BQN = ck_tile::integer_divide_ceil(N, BQuantGroupSize::kN);
        const ck_tile::index_t BQK = ck_tile::integer_divide_ceil(K, BQuantGroupSize::kK);
        const ck_tile::index_t stride_BQ =
            ck_tile::get_default_stride(BQK, BQN, 0, is_row_major(BQLayout{}));

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> bq_bqk_bqn(
            ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, is_row_major(BQLayout{})));

        ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(bq_bqk_bqn);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem bq_bqk_bqn_dev_buf(bq_bqk_bqn.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(M * N * sizeof(CDataType));

        a_m_k_dev_buf.ToDevice(a_m_k.data());

        ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
        if constexpr(PreshuffleB)
        {
            if constexpr(TiledMMAPermuteN && BQuantGroupSize::kN == 1)
            {
                b_k_n_dev = ck_tile::shuffle_b_permuteN<GemmConfig>(b_k_n);
            }
            else
            {
                b_k_n_dev = ck_tile::shuffle_b<GemmConfig>(b_k_n);
            }
        }
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
        }
        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());

        if constexpr(PreshuffleB && TiledMMAPermuteN && BQuantGroupSize::kN == 1)
        {
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::bq_permuteN<GemmConfig>(bq_bqk_bqn, BQuantGroupSize::kN);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else if constexpr(GemmConfig::BPreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::shuffle_bq(&bq_bqk_bqn, GemmConfig::K_Tile / BQuantGroupSize::kK);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else
        {
            bq_bqk_bqn_dev_buf.ToDevice(bq_bqk_bqn.data());
        }

        if(k_batch > 1)
        {
            c_m_n_dev_buf.SetZero();
        }

        ck_tile::QuantGemmHostArgs args{a_m_k_dev_buf.GetDeviceBuffer(),
                                        b_k_n_dev_buf.GetDeviceBuffer(),
                                        c_m_n_dev_buf.GetDeviceBuffer(),
                                        nullptr,
                                        bq_bqk_bqn_dev_buf.GetDeviceBuffer(),
                                        k_batch,
                                        M,
                                        N,
                                        K,
                                        AQK,
                                        BQK,
                                        stride_A,
                                        stride_B,
                                        stride_C,
                                        0,
                                        stride_BQ};

        ck_tile::stream_config stream_config{};
        invoke_quant_gemm(args, stream_config);

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        run_cpu_reference_fused_aquant(a_m_k, b_k_n, bq_bqk_bqn, c_m_n_host_ref);

        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<ck_tile::fp8_t, BDataType, AccDataType, CDataType>(
                K, k_batch, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "FusedAQuantGrouped validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;

        if(!pass)
        {
            std::cout << "FusedAQuantGrouped - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    private:
    void run_cpu_reference_fused_aquant(const ck_tile::HostTensor<ADataType>& a_m_k,
                                        const ck_tile::HostTensor<BDataType>& b_k_n,
                                        const ck_tile::HostTensor<QDataType>& bq_bqk_bqn,
                                        ck_tile::HostTensor<CDataType>& c_m_n_host_ref)
    {
        ck_tile::reference_gemm_fused_aquant<ADataType,
                                             QDataType,
                                             BDataType,
                                             QDataType,
                                             AccDataType,
                                             CDataType,
                                             AQuantGroupSize,
                                             BQuantGroupSize>(
            a_m_k, b_k_n, bq_bqk_bqn, c_m_n_host_ref);
    }

    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        static_assert(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        constexpr bool transpose_c = CodegenGemmTraits::TransposeC;
        constexpr bool eight_waves =
#ifdef CK_GFX950_SUPPORT
            IS_FP8BLOCKSCALE &&
            (GemmConfig::M_Warp * GemmConfig::N_Warp * GemmConfig::K_Warp == 8) &&
            GemmConfig::K_Warp_Tile == 128;
#else
            false;
#endif

        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType>;

        constexpr auto base_gemm_pipeline = []() {
            if constexpr(eight_waves)
                return ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>{};
            else if constexpr(PreshuffleB)
                return ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>{};
            else
                return ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>{};
        }();
        using BaseGemmPipeline = std::decay_t<decltype(base_gemm_pipeline)>;

        const ck_tile::index_t K_split =
            ck_tile::integer_least_multiple(args.K, GemmConfig::K_Tile);
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using PipelineProblem =
                ck_tile::GemmABQuantPipelineProblem<ADataType,
                                                    QDataType,
                                                    BDataType,
                                                    QDataType,
                                                    AccDataType,
                                                    CodegenGemmShape,
                                                    CodegenGemmTraits,
                                                    AQuantGroupSize,
                                                    BQuantGroupSize,
                                                    transpose_c,
                                                    ComputeDataType,
                                                    ck_tile::GemmPipelineScheduler::Intrawave,
                                                    has_hot_loop_v,
                                                    tail_number_v>;

            using GemmPipeline = std::conditional_t<
                eight_waves,
                ck_tile::ABQuantGemmPipelineAgBgCrEightWaves<PipelineProblem>,
                std::conditional_t<
                    PreshuffleB,
                    ck_tile::WPABQuantBPipelineAgBgCrV2<PipelineProblem>,
                    ck_tile::FusedAQuantBQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>>;

            using GemmEpilogue = std::conditional_t<
                TiledMMAPermuteN,
                ck_tile::PermuteNEpilogue<
                    ck_tile::PermuteNEpilogueProblem<typename PipelineProblem::AComputeDataType,
                                                     typename PipelineProblem::BComputeDataType,
                                                     ck_tile::tuple<>,
                                                     AccDataType,
                                                     CDataType,
                                                     ck_tile::tuple<>,
                                                     CLayout,
                                                     ck_tile::element_wise::PassThrough,
                                                     TilePartitioner::MPerBlock,
                                                     TilePartitioner::NPerBlock,
                                                     M_Warp,
                                                     N_Warp,
                                                     M_Warp_Tile,
                                                     N_Warp_Tile,
                                                     K_Warp_Tile,
                                                     transpose_c,
                                                     false,
                                                     1>>,
                ck_tile::CShuffleEpilogue<
                    ck_tile::CShuffleEpilogueProblem<typename PipelineProblem::AComputeDataType,
                                                     typename PipelineProblem::BComputeDataType,
                                                     ck_tile::tuple<>,
                                                     AccDataType,
                                                     CDataType,
                                                     ck_tile::tuple<>,
                                                     CLayout,
                                                     ck_tile::element_wise::PassThrough,
                                                     TilePartitioner::MPerBlock,
                                                     TilePartitioner::NPerBlock,
                                                     M_Warp,
                                                     N_Warp,
                                                     M_Warp_Tile,
                                                     N_Warp_Tile,
                                                     K_Warp_Tile,
                                                     transpose_c>>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::ABQuantGrouped>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for standalone fusedAquant");
            }

            ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};
