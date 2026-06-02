// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#include "ck_tile/core/tensor/load_tile.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tile_elementwise.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/utility/reduce_operator.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_quant_pipeline_problem.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d_problem.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_problem.hpp"

namespace ck_tile {

// Compute optimized pipeline
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1

template <typename Problem, typename Policy = GemmABQuantPipelineAgBgCrDefaultPolicy>
struct FusedAQuantBQuantGemmPipelineAgBgCrCompV3 : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmABQuantPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType       = remove_cvref_t<typename Problem::ADataType>;
    using AQDataType      = remove_cvref_t<typename Problem::AQDataType>;
    using BDataType       = remove_cvref_t<typename Problem::BDataType>;
    using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
    using CDataType       = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;
    using AQuantGroupSize = remove_cvref_t<typename Problem::AQuantGroupSize>;
    using BQuantGroupSize = remove_cvref_t<typename Problem::BQuantGroupSize>;

    static_assert(BQuantGroupSize::kM == 1, "only N/K blocks for BQuant kernel!");
    static_assert(AQuantGroupSize::kN == 1, "only M/K blocks for AQuant kernel!");
    static_assert(AQuantGroupSize::kM == 1, "no block M for AQuant kernel supported yet!");
    static_assert(AQuantGroupSize::kK == BQuantGroupSize::kK,
                  "AQuantGroupSize::kK should be equal to BQuantGroupSize::kK");

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t AQPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<AQDataType>>::PackedSize;

    static constexpr index_t BQPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BQDataType>>::PackedSize;

    using ALayout  = remove_cvref_t<typename Problem::ALayout>;
    using AQLayout = remove_cvref_t<typename Problem::AQLayout>;
    using BLayout  = remove_cvref_t<typename Problem::BLayout>;
    using BQLayout = remove_cvref_t<typename Problem::BQLayout>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    // Keeping rest the same, change ADataType to FP8 for the BlockGemm call
    using QuantizedABQuantProblem = GemmABQuantPipelineProblem<fp8_t,
                                                               typename Problem::AQDataType,
                                                               typename Problem::BDataType,
                                                               typename Problem::BQDataType,
                                                               typename Problem::CDataType,
                                                               typename Problem::BlockGemmShape,
                                                               typename Problem::Traits,
                                                               typename Problem::AQuantGroupSize,
                                                               typename Problem::BQuantGroupSize,
                                                               Problem::TransposeC,
                                                               fp8_t,
                                                               Problem::Scheduler,
                                                               Problem::HasHotLoop,
                                                               Problem::TailNum>;
    using BlockGemm =
        remove_cvref_t<decltype(Policy::template GetBlockGemm<QuantizedABQuantProblem>())>;

    // A/B DataType gets converted from PkInt4/PkFp4 during loading
    // using OverrideADataType = typename BlockGemm::OverrideADataType;
    // using OverrideBDataType = typename BlockGemm::OverrideBDataType;
    using OverrideADataType = fp8_t;
    using OverrideBDataType = fp8_t;

    static constexpr index_t BlockSize   = Problem::kBlockSize;
    static constexpr index_t MPerBlock   = BlockGemmShape::kM;
    static constexpr index_t NPerBlock   = BlockGemmShape::kN;
    static constexpr index_t KPerBlock   = BlockGemmShape::kK;
    static constexpr index_t KPerBlockAQ = BlockGemmShape::kK / AQuantGroupSize::kK;
    static constexpr index_t NPerBlockBQ =
        (BQuantGroupSize::kN <= BlockGemmShape::kN)
            ? integer_divide_ceil(BlockGemmShape::kN, BQuantGroupSize::kN)
            : 1;
    static constexpr index_t KPerBlockBQ = BlockGemmShape::kK / BQuantGroupSize::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }
    static constexpr index_t GetVectorSizeAQ()
    {
        return Policy::template GetVectorSizeAQ<Problem>();
    }
    static constexpr index_t GetVectorSizeBQ()
    {
        return Policy::template GetVectorSizeBQ<Problem>();
    }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr bool APreshuffleQuant = Problem::Traits::APreshuffleQuant;
    static constexpr bool BPreshuffleQuant = Problem::Traits::BPreshuffleQuant;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    using Base::PrefetchStages;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_', "fusedaquant_bquant_pipeline_AgBgCrCompV3",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', BlockGemm::WarpGemm::kM, BlockGemm::WarpGemm::kN, BlockGemm::WarpGemm::kK),
                      concat('x', kPadM, kPadN, kPadK), AQuantGroupSize::GetName(), BQuantGroupSize::GetName());
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // We are not storing the original packed type in LDS, so we need to multiply the smem size
        // by the packed size.
        constexpr index_t smem_size_a = Policy::template GetSmemSizeA<Problem>() * APackedSize;
        constexpr index_t smem_size_b = Policy::template GetSmemSizeB<Problem>() * BPackedSize;

        return smem_size_a + smem_size_b;
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        CK_TILE_HOST_DEVICE static constexpr auto MakeAReduceTileDistribution()
        {
            constexpr index_t VecLoadSize = gcd(problem_fixed_vector_size_v<Problem>
                                                    ? Problem::VectorSizeA
                                                    : Policy::template GetVectorSizeA<Problem>(),
                                                AQuantGroupSize::kK);
            constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      MPerBlock * KPerBlockAQ,
                                                      AQuantGroupSize::kK,
                                                      VecLoadSize,
                                                      Policy::getATileAccessPattern(),
                                                      NumWaveGroups>;

            return TileEncodingPattern::make_2d_static_tile_distribution();
        }

        template <typename ADramWindow>
        CK_TILE_DEVICE static auto MakeAReduceDramWindow(const ADramWindow& a_dram_window)
        {
            const auto& a_tensor_view = a_dram_window.get_bottom_tensor_view();
            const auto& a_desc        = a_tensor_view.get_tensor_descriptor();

            const auto total_m        = a_desc.get_lengths()[I0{}];
            const auto total_k        = a_desc.get_lengths()[I1{}];
            const auto total_k_groups = total_k / AQuantGroupSize::kK;

            const auto a_unmerged = transform_tensor_view(
                a_tensor_view,
                make_tuple(make_pass_through_transform(total_m),
                           make_unmerge_transform(
                               make_tuple(total_k_groups, number<AQuantGroupSize::kK>{}))),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}));

            const auto a_grouped = transform_tensor_view(
                a_unmerged,
                make_tuple(make_merge_transform(make_tuple(total_m, total_k_groups)),
                           make_pass_through_transform(number<AQuantGroupSize::kK>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            const auto& a_origin = a_dram_window.get_window_origin();
            const index_t grouped_row_origin =
                a_origin[I0{}] * total_k_groups + a_origin[I1{}] / AQuantGroupSize::kK;

            return make_tile_window(
                a_grouped,
                make_tuple(number<MPerBlock * KPerBlockAQ>{}, number<AQuantGroupSize::kK>{}),
                make_array(grouped_row_origin, index_t{0}),
                MakeAReduceTileDistribution());
        }

        template <index_t... I>
        CK_TILE_DEVICE static constexpr auto getIdx(tile_distributed_index<I...>)
        {
            constexpr auto idxs = make_tuple(I...);
            return idxs[number<0>{}];
        }

        template <typename ADstStaticTileDist,
                  typename AQDstStaticTileDistribution,
                  typename ADramWindow>
        CK_TILE_DEVICE static auto LoadAndQuantizeATile(
            static_distributed_tensor<fp8_t, ADstStaticTileDist>& a_block_tile,
            static_distributed_tensor<AQDataType, AQDstStaticTileDistribution>& aq_block_tile,
            const ADramWindow& a_dram_window)
        {

            static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>,
                          "Grouped fused A quantization currently supports RowMajor A only.");
            static_assert(std::is_same_v<ADataType, ck_tile::bf16_t>);
            // ADstStaticTileDist{} -> AReduceTileDist
            // Modify the window and match the temp tensor
            auto a_reduce =
                make_static_distributed_tensor<ADataType>(MakeAReduceTileDistribution());
            auto a_reduce_dram_window = MakeAReduceDramWindow(a_dram_window);
            load_tile(a_reduce, a_reduce_dram_window);

            // ADRAM
            // MPerBlock x KPerBlock
            // = MPerBlock x (VecSize * ThreadPerK)
            // = MPerBlock x (VecSize * warpsize / MPerWarp)

            // a_reduce_tile:
            // (MPerBlock * BlockAQPerK) x AQGroupSize::kK
            //
            // => aq_reduced
            // (MPerBlock * BlockAQPerK) x 1
            //
            // => aq_dst
            // MPerBlock x BlockAQPerK
            //
            // => a_dst
            //

            // Define the reduce problem for quantization
            constexpr index_t MWarp = BlockGemmShape::BlockWarps::at(number<0>{});
            constexpr index_t KWarp = BlockGemmShape::BlockWarps::at(number<2>{});

            static_assert(KWarp == 1, "Only single KWarp supported!");

            using BlockWarps = ck_tile::sequence<MWarp, KWarp>;
            using BlockTile  = ck_tile::sequence<MPerBlock * KPerBlockAQ, AQuantGroupSize::kK>;
            using WarpTile =
                ck_tile::sequence<MPerBlock * KPerBlockAQ / MWarp, AQuantGroupSize::kK / KWarp>;
            using ThreadTile =
                ck_tile::sequence<BlockGemm::MIterPerWarp, BlockGemm::WarpGemm::kKPerThread>;
            using ReduceShape = Reduce2dShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;

            // TODO: Computedatatype float?
            using ReduceProblem = BlockReduce2dProblem<ADataType, AQDataType, ReduceShape>;

            using ReducePolicy = Reduce2dDefaultPolicy;

            auto blockreduce      = ReducePolicy::template GetBlockReduce2d<ReduceProblem>();
            auto blockreduce_sync = ReducePolicy::template GetBlockReduce2dSync<ReduceProblem>();
            //  Crosswarp sync should not be needed, as
            // quant scales are computed for per-warp values

            // Only absmax computed during reduction; range scaling applied later
            auto reduce_func = ReduceOp::AbsMax{};

            const float fp8_inv_range = 1.f / (type_convert<float>(numeric<fp8_t>::max()) -
                                               type_convert<float>(numeric<fp8_t>::min()));

            auto aq_reduce = blockreduce.template MakeYBlockTile<decltype(a_reduce)>();

            set_tile(aq_reduce, ReduceOp::AbsMax::GetIdentityValue<float>());

            blockreduce(a_reduce, aq_reduce, reduce_func);
            blockreduce_sync(aq_reduce, reduce_func);

            // TODO: Copy/Sync values across threads to match blockgemm expectation
            // aq_tmp after reduction is one value per warp (aka per m) at lane 0, whereas for
            // the MFMA the full scale data needs to be available for each thread
            constexpr auto thread_buf_size = aq_reduce.get_thread_buffer_size();
            constexpr auto aq_thread_buf_size =
                static_distributed_tensor<AQDataType,
                                          AQDstStaticTileDistribution>::get_thread_buffer_size();
            static_assert(thread_buf_size == aq_thread_buf_size);

            set_tile(aq_block_tile, 1.f * fp8_inv_range);
            static_for<0, thread_buf_size, 1>{}([&](auto i) {
                // Copy the first lanes values to all threads
                float abs_max = amd_wave_read_first_lane(aq_reduce.get_thread_buffer()[i]);
                // if(abs_max == 0.f)
                // {
                abs_max = 1.f;
                // };
                aq_block_tile.get_thread_buffer()[i] =
                    type_convert<AQDataType>(abs_max * fp8_inv_range);
            });

            auto a_raw_tile = make_static_distributed_tensor<ADataType>(ADstStaticTileDist{});
            load_tile(a_raw_tile, a_dram_window);
            sweep_tile(aq_block_tile, [&](auto aq_idx) {
                constexpr auto m_idx       = aq_idx[number<0>{}];
                constexpr auto k_group_idx = aq_idx[number<1>{}];

                const float scale_value = type_convert<float>(aq_block_tile(aq_idx));

                static_for<0, AQuantGroupSize::kK, 1>{}([&](auto kk) {
                    constexpr auto k_idx =
                        tile_distributed_index<getIdx(k_group_idx) * AQuantGroupSize::kK +
                                               kk.value>{};

                    constexpr auto a_idx = make_tuple(m_idx, k_idx);

                    float raw_a_value   = type_convert<float>(a_raw_tile(a_idx));
                    a_block_tile(a_idx) = type_convert<fp8_t>(raw_a_value / scale_value);
                });
            });
            // Apply scales and convert data to the original block tile distribution
            // sweep_tile(a_block_tile, [&](auto idx) {
            //     // constexpr auto m_idx = idx[number<0>{}];
            //     // constexpr auto k_idx = idx[number<1>{}];
            //     // constexpr auto k     = getIdx(k_idx);

            //     constexpr auto x_idx = [&]() {
            //         return get_x_indices_from_distributed_indices(ADstStaticTileDist{}, idx);
            //     }();
            //     constexpr auto m = x_idx[number<0>{}];
            //     constexpr auto k = x_idx[number<1>{}];

            //     constexpr auto k_group_idx =
            //         tile_distributed_index<k / number<AQuantGroupSize::kK>{}>{};

            //     // constexpr auto m     = getIdx(m_idx);
            //     // constexpr auto grouped_k_idx =
            //     //     tile_distributed_index<k % number<AQuantGroupSize::kK>{}>{};
            //     // constexpr auto m_reduce =
            //     //     tile_distributed_index<m * KPerBlockAQ + k /
            //     //     number<AQuantGroupSize::kK>{}>{};

            //     // auto raw_a_value  = a_reduce(make_tuple(m_reduce, grouped_k_idx));
            //     float raw_a_value = type_convert<float>(a_raw_tile(idx));
            //     auto scale_value  = aq_block_tile(make_tuple(m, k_group_idx));
            //     a_block_tile(idx) = type_convert<fp8_t>(raw_a_value / scale_value);
            // });
        }

        template <typename BDramWindow, typename BBlockTile_>
        CK_TILE_DEVICE static void LoadAndConvertBTile(BBlockTile_& b_block_tile,
                                                       const BDramWindow& b_dram_window)
        {
            constexpr index_t UnaryOpSize = 8;
            load_and_convert_tile<UnaryOpSize>(b_block_tile, b_dram_window);
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename AQDramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename BQDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto
        operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                   const AElementFunction& a_element_func,
                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                   const BElementFunction& b_element_func,
                   [[maybe_unused]] const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                   [[maybe_unused]] index_t m,
                   index_t n,
                   index_t num_loop,
                   void* p_smem) const
        {

            static_assert(std::is_same_v<ADataType, bf16_t>, "Only BF16 input is supported!");
            static_assert(is_null_tile_window<AQDramBlockWindowTmp>,
                          "AQ Dram Block window is not used with FusedAQuant!");
            static_assert(KPerBlock >= AQuantGroupSize::kK,
                          "Quantization across blocks is not supported!");
            static_assert(KPerBlock % AQuantGroupSize::kK ==
                          0); // KPerBlock = AQuantGroupSize * KPerBlockAQ

            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BQDataType,
                                   remove_cvref_t<typename BQDramBlockWindowTmp::DataType>>,
                "A/B/BQ Dram block window should have the same data type as appropriate "
                "([A|B|BQ]DataType) defined in Problem definition!");

            // TODO: We only support RowMajor A, clean this
            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            // constexpr bool is_aq_col_major =
            //     std::is_same_v<AQLayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
            constexpr bool is_bq_row_major =
                std::is_same_v<BQLayout, tensor_layout::gemm::RowMajor>;

            // static_assert(is_a_col_major
            //                   ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
            //                      MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
            //                   : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
            //                      KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
            //               "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");
            static_assert(
                BPreshuffleQuant ||
                    (is_bq_row_major
                         ? (KPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                            NPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                         : (NPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                            KPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I1{}])),
                "Bq block window has incorrect lengths for defined BqLayout!");

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;
            // using AQDramTileWindowStep = typename AQDramBlockWindowTmp::BottomTensorIndex;
            using BQDramTileWindowStep = typename BQDramBlockWindowTmp::BottomTensorIndex;

            // Note: A/B DataType PkInt4/PkFp4 gets converted during loading, before going to
            // LDS
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);
            auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);
            // auto aq_copy_dram_window = Base::GetAQDramLoadWindow(aq_dram_block_window_tmp);
            auto bq_copy_dram_window = Base::GetBQDramLoadWindow(bq_dram_block_window_tmp);

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());
            using AQBlockTileDistr =
                decltype(Policy::template MakeAQDramTileDistribution<Problem>());
            using BQBlockTileDistr = decltype(bq_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<OverrideADataType>(ABlockTileDistr{}));
            using AQBlockTile =
                decltype(make_static_distributed_tensor<AQDataType>(AQBlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<OverrideBDataType>(BBlockTileDistr{}));
            using BQBlockTile =
                decltype(make_static_distributed_tensor<BQDataType>(BQBlockTileDistr{}));

            auto block_gemm = BlockGemm();

            ABlockTile a_block_tile;
            BBlockTile b_block_tile;
            AQBlockTile aq_block_tile[3];
            BQBlockTile bq_block_tile[2];
            int AQIdx = 0;
            int BQIdx = 0;

            auto c_block_tile = block_gemm.MakeCBlockTile();

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                false ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            const BQDramTileWindowStep bq_dram_tile_window_step =
                (BPreshuffleQuant)
                    ? make_array(((NPerBlockBQ <= BlockGemmShape::BlockWarps::at(number<1>{}))
                                      ? ck_tile::integer_divide_ceil(n, BQuantGroupSize::kN)
                                      : ck_tile::integer_least_multiple(n, NPerBlock) /
                                            BlockGemmShape::WarpTile::at(number<1>{})),
                                 0)
                : is_bq_row_major ? make_array(KPerBlockBQ, 0)
                                  : make_array(0, KPerBlockBQ);

            LoadAndQuantizeATile(a_block_tile, aq_block_tile[AQIdx], a_copy_dram_window);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            // B tile gets converted to A datatype during loading
            LoadAndConvertBTile(b_block_tile, b_copy_dram_window);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            Base::GlobalPrefetch(
                bq_block_tile[BQIdx], bq_copy_dram_window, bq_dram_tile_window_step);

            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<OverrideADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tile);
                Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
            }

            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<OverrideBDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_block_tile);
                Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
            }

            LoadAndQuantizeATile(a_block_tile, aq_block_tile[AQIdx + 1], a_copy_dram_window);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

            LoadAndConvertBTile(b_block_tile, b_copy_dram_window);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
            block_sync_lds();

            block_gemm.LocalPrefetch(
                a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);

            __builtin_amdgcn_sched_barrier(0);

            if constexpr(HasHotLoop)
            {
                constexpr index_t tail_count =
                    ((TailNum == TailNumber::Full) || (TailNum == TailNumber::Odd)) ? 1 : 2;
                index_t i = 0;
                do
                {
                    block_sync_lds();

                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        // Note: ABDataType PkInt4/PkFp4 gets converted during loading earlier
                        auto a_shuffle_tmp = make_static_distributed_tensor<OverrideADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tile);
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        // Note: BDataType PkInt4/PkFp4 gets converted during loading earlier
                        auto b_shuffle_tmp = make_static_distributed_tensor<OverrideBDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_block_tile);
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
                    }

                    // Base::GlobalPrefetch(a_block_tile, a_copy_dram_window,
                    // a_dram_tile_window_step);
                    // Base::GlobalPrefetch(b_block_tile, b_copy_dram_window,
                    // b_dram_tile_window_step);
                    LoadAndQuantizeATile(
                        a_block_tile, aq_block_tile[(AQIdx + 2) % 3], a_copy_dram_window);
                    move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

                    LoadAndConvertBTile(b_block_tile, b_copy_dram_window);
                    move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

                    // Base::GlobalPrefetch(aq_block_tile[(currIdx + 1) % 2],
                    //                      aq_copy_dram_window,
                    //                      aq_dram_tile_window_step);
                    Base::GlobalPrefetch(bq_block_tile[(BQIdx + 1) % 2],
                                         bq_copy_dram_window,
                                         bq_dram_tile_window_step);

                    block_gemm(c_block_tile,
                               aq_block_tile[AQIdx],
                               bq_block_tile[BQIdx],
                               a_lds_gemm_window,
                               b_lds_gemm_window);

                    AQIdx = (AQIdx + 1) % 3;
                    BQIdx = (BQIdx + 1) % 2;

                    block_sync_lds();

                    block_gemm.LocalPrefetch(
                        a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                    __builtin_amdgcn_sched_barrier(0);

                    i += 1;
                } while(i < (num_loop - tail_count));
            }
            // tail
            if constexpr((TailNum == TailNumber::Full) || (TailNum == TailNumber::Odd))
            {
                block_gemm(c_block_tile,
                           aq_block_tile[AQIdx],
                           bq_block_tile[BQIdx],
                           a_lds_gemm_window,
                           b_lds_gemm_window);
            }
            else
            {
                Base::GlobalPrefetch(
                    bq_block_tile[(BQIdx + 1) % 2], bq_copy_dram_window, bq_dram_tile_window_step);
                block_gemm(c_block_tile,
                           aq_block_tile[AQIdx],
                           bq_block_tile[BQIdx],
                           a_lds_gemm_window,
                           b_lds_gemm_window);
                block_sync_lds();

                AQIdx = (AQIdx + 1) % 3;
                BQIdx = (BQIdx + 1) % 2;

                if constexpr(is_a_col_major && !is_a_load_tr_v())
                {
                    // Note: ADataType gets converted during loading from PkInt4/PkFp4
                    auto a_shuffle_tmp = make_static_distributed_tensor<OverrideADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, a_block_tile);
                    Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                }
                else
                {
                    Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                }
                if constexpr(is_b_row_major && !is_b_load_tr_v())
                {
                    // Note: BDataType gets converted during loading from PkInt4
                    auto b_shuffle_tmp = make_static_distributed_tensor<OverrideBDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, b_block_tile);
                    Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                }
                else
                {
                    Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
                }
                block_sync_lds();
                block_gemm.LocalPrefetch(
                    a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                block_gemm(c_block_tile,
                           aq_block_tile[AQIdx],
                           bq_block_tile[BQIdx],
                           a_lds_gemm_window,
                           b_lds_gemm_window);
            }
            return c_block_tile;
        }
    };
    // Overload for PreshuffleQuant = true
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AQDramBlockWindowTmp,
              typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem,
                                   index_t m = 0,
                                   index_t n = 0) const
    {

        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const OverrideADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const OverrideBDataType& b) { return b; },
            aq_dram_block_window_tmp,
            bq_dram_block_window_tmp,
            m,
            n,
            num_loop,
            p_smem);
    }

    /// @brief Runtime pipeline dispatch operator for grouped GEMM kernels.
    ///
    /// This operator is used by grouped GEMM kernels where pipeline parameters
    /// (has_hot_loop, num_loop, tail_number) are calculated on the device side
    /// at runtime, not on the host side during compilation. This is necessary
    /// because different GEMM problems in the group may have different K dimensions,
    /// requiring different pipeline configurations that cannot be determined at
    /// compile time.
    ///
    /// @param a_dram_block_window_tmp Block window for A tensor in DRAM
    /// @param b_dram_block_window_tmp Block window for B tensor in DRAM
    /// @param aq_dram_block_window_tmp Block window for AQ (quantization scale) tensor in DRAM
    /// @param bq_dram_block_window_tmp Block window for BQ (quantization scale) tensor in DRAM
    /// @param num_loop Number of main loop iterations (calculated on device)
    /// @param has_hot_loop Whether the pipeline has a hot loop (calculated on device)
    /// @param tail_number Type of tail handling required (calculated on device)
    /// @param p_smem Pointer to shared memory
    /// @return Accumulated result tile in registers
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AQDramBlockWindowTmp,
              typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem,
                                   index_t m = 0,
                                   index_t n = 0) const
    {
        const auto RunPipeline = [&](auto has_hot_loop_, auto tail_number_) {
            constexpr bool hot_loop = has_hot_loop_.value;
            constexpr auto tail_num = tail_number_.value;

            return PipelineImpl<Scheduler>{}.template operator()<hot_loop, tail_num>(
                a_dram_block_window_tmp,
                // Note: ADataType PkInt4/PkFp4 gets converted during loading
                [](const OverrideADataType& a) { return a; },
                b_dram_block_window_tmp,
                // Note: BDataType PkInt4/PkFp4 gets converted during loading
                [](const OverrideBDataType& b) { return b; },
                aq_dram_block_window_tmp,
                bq_dram_block_window_tmp,
                m,
                n, // dummy value, won't be used
                num_loop,
                p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }
};

} // namespace ck_tile
