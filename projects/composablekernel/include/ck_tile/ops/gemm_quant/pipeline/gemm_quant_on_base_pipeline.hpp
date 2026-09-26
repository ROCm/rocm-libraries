// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"

// Runs a group-quant GEMM (AQuant / BQuant / ABQuant) on a plain GEMM pipeline that accepts
// per-iteration scale windows (e.g. GemmPipelineAgBgCrCompAsync). The plain pipeline keeps its own
// data movement and LDS layout; only its block gemm is replaced by the quant block gemm, and the
// quant scale windows are streamed through the pipeline's scale-window path.

namespace ck_tile {

// Replaces the policy of a pipeline template Pipeline<Problem, Policy>.
template <typename Pipeline, typename NewPolicy>
struct rebind_policy;

template <template <typename, typename> class Pipeline,
          typename Problem,
          typename Policy,
          typename NewPolicy>
struct rebind_policy<Pipeline<Problem, Policy>, NewPolicy>
{
    using type    = Pipeline<Problem, NewPolicy>;
    using problem = Problem;
    using policy  = Policy;
};

// Adapts a quant block gemm to the register-tile call of the plain pipelines:
// (c, a_reg, b_reg, scale_a, scale_b), where an absent scale is a null tensor.
template <typename QuantBlockGemm>
struct BlockGemmQuantRegAdaptor : QuantBlockGemm
{
    template <index_t SubTileIdx = 0,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename ScaleATensor,
              typename ScaleBTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor,
                                   const ScaleATensor& scale_a,
                                   const ScaleBTensor& scale_b)
    {
        // The quant block gemm consumes its own register tiles, which share the distribution
        // of the tiles the pipeline loaded from LDS.
        this->block_gemm_impl_.a_warp_tile_.get_thread_buffer() =
            a_block_tensor.get_thread_buffer();
        this->block_gemm_impl_.b_warp_tile_.get_thread_buffer() =
            b_block_tensor.get_thread_buffer();
        auto sa = scale_a;
        auto sb = scale_b;
        if constexpr(is_null_tensor_v<ScaleATensor>)
            QuantBlockGemm::operator()(c_block_tensor, sb, a_block_tensor, b_block_tensor);
        else if constexpr(is_null_tensor_v<ScaleBTensor>)
            QuantBlockGemm::operator()(c_block_tensor, sa, a_block_tensor, b_block_tensor);
        else
            QuantBlockGemm::operator()(c_block_tensor, sa, sb, a_block_tensor, b_block_tensor);
    }
};

// Policy of the plain pipeline with the quant block gemm and quant scale-window steps.
template <typename BasePolicy, typename QuantPolicy, QuantType QT>
struct GemmQuantBasePolicy : BasePolicy
{
    template <typename Problem, bool = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return BlockGemmQuantRegAdaptor<
            remove_cvref_t<decltype(QuantPolicy::template GetBlockGemm<Problem>())>>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetScaleDramTileWindowSteps()
    {
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr auto aq_step      = [] {
            if constexpr(QT == QuantType::BQuantGrouped)
                return make_array(0, 0);
            else
            {
                constexpr index_t KPerBlockAQ = KPerBlock / Problem::AQuantGroupSize::kK;
                if constexpr(std::is_same_v<typename Problem::AQLayout,
                                                 tensor_layout::gemm::ColumnMajor>)
                    return make_array(KPerBlockAQ, 0);
                else
                    return make_array(0, KPerBlockAQ);
            }
        }();
        constexpr auto bq_step = [] {
            if constexpr(QT == QuantType::AQuantGrouped)
                return make_array(0, 0);
            else
            {
                constexpr index_t KPerBlockBQ = KPerBlock / Problem::BQuantGroupSize::kK;
                if constexpr(std::is_same_v<typename Problem::BQLayout,
                                            tensor_layout::gemm::RowMajor>)
                    return make_array(KPerBlockBQ, 0);
                else
                    return make_array(0, KPerBlockBQ);
            }
        }();
        return make_tuple(aq_step, bq_step);
    }
};

// Quant pipeline (e.g. BQuantGemmPipelineAgBgCrCompV3) whose main loop runs on BasePipeline.
// Keeps the quant pipeline's interface (quant block sizes, scale vector sizes) and takes the
// data-movement interface (vector sizes, LDS size, loop structure) from BasePipeline.
template <QuantType QT, typename QuantPipeline, typename BasePipeline>
struct GemmQuantOnBasePipeline : QuantPipeline
{
    using Problem         = typename rebind_policy<QuantPipeline, void>::problem;
    using QuantPolicy     = typename rebind_policy<QuantPipeline, void>::policy;
    using NullWindow      = decltype(make_null_tile_window(make_tuple(number<0>{}, number<0>{})));
    using RebasedPipeline = typename rebind_policy<
        BasePipeline,
        GemmQuantBasePolicy<typename rebind_policy<BasePipeline, void>::policy, QuantPolicy, QT>>::
        type;

    static constexpr index_t PrefetchStages = BasePipeline::PrefetchStages;
    static constexpr bool DoubleSmemBuffer  = BasePipeline::DoubleSmemBuffer;
    static constexpr index_t GetVectorSizeA() { return BasePipeline::GetVectorSizeA(); }
    static constexpr index_t GetVectorSizeB() { return BasePipeline::GetVectorSizeB(); }
    static constexpr index_t GetSmemPackB() { return BasePipeline::GetSmemPackB(); }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return RebasedPipeline::GetSmemSize();
    }
    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return BasePipeline::BlockHasHotloop(num_loop);
    }
    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        return BasePipeline::GetBlockLoopTailNum(num_loop);
    }
    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_', QuantPipeline::GetName(), BasePipeline::GetName());
    }

    template <typename ADramWindow,
              typename BDramWindow,
              typename AQDramWindow,
              typename BQDramWindow>
    CK_TILE_DEVICE auto Run(const ADramWindow& a,
                            const BDramWindow& b,
                            const AQDramWindow& aq,
                            const BQDramWindow& bq,
                            index_t num_loop,
                            void* p_smem) const
    {
        auto aq_window = [&] {
            if constexpr(QT == QuantType::BQuantGrouped)
                return NullWindow{};
            else
                return GemmAQuantPipelineAgBgCrImplBase<Problem, QuantPolicy>{}.GetAQDramLoadWindow(
                    aq);
        }();
        auto bq_window = [&] {
            if constexpr(QT == QuantType::AQuantGrouped)
                return NullWindow{};
            else
                return GemmBQuantPipelineAgBgCrImplBase<Problem, QuantPolicy>{}.GetBQDramLoadWindow(
                    bq);
        }();
        return RebasedPipeline{}(make_tuple(a),
                                 element_wise::PassThrough{},
                                 make_tuple(b),
                                 element_wise::PassThrough{},
                                 make_tuple(aq_window),
                                 make_tuple(bq_window),
                                 num_loop,
                                 p_smem);
    }

    // AQuant kernel call
    template <typename ADramWindow, typename BDramWindow, typename AQDramWindow>
    CK_TILE_DEVICE auto operator()(const ADramWindow& a,
                                   const BDramWindow& b,
                                   const AQDramWindow& aq,
                                   index_t num_loop,
                                   void* p_smem,
                                   index_t = 0) const
    {
        if constexpr(QT == QuantType::AQuantGrouped)
            return Run(a, b, aq, NullWindow{}, num_loop, p_smem);
        else
            return Run(a, b, NullWindow{}, aq, num_loop, p_smem);
    }

    // BQuant kernel call with runtime loop structure (already derived from num_loop by Run)
    template <typename ADramWindow, typename BDramWindow, typename BQDramWindow>
    CK_TILE_DEVICE auto operator()(const ADramWindow& a,
                                   const BDramWindow& b,
                                   const BQDramWindow& bq,
                                   index_t num_loop,
                                   bool,
                                   TailNumber,
                                   void* p_smem,
                                   index_t = 0) const
    {
        return Run(a, b, NullWindow{}, bq, num_loop, p_smem);
    }

    // ABQuant kernel calls
    template <typename ADramWindow,
              typename BDramWindow,
              typename AQDramWindow,
              typename BQDramWindow>
    CK_TILE_DEVICE auto operator()(const ADramWindow& a,
                                   const BDramWindow& b,
                                   const AQDramWindow& aq,
                                   const BQDramWindow& bq,
                                   index_t num_loop,
                                   void* p_smem,
                                   index_t = 0,
                                   index_t = 0) const
    {
        return Run(a, b, aq, bq, num_loop, p_smem);
    }

    template <typename ADramWindow,
              typename BDramWindow,
              typename AQDramWindow,
              typename BQDramWindow>
    CK_TILE_DEVICE auto operator()(const ADramWindow& a,
                                   const BDramWindow& b,
                                   const AQDramWindow& aq,
                                   const BQDramWindow& bq,
                                   index_t num_loop,
                                   bool,
                                   TailNumber,
                                   void* p_smem,
                                   index_t = 0,
                                   index_t = 0) const
    {
        return Run(a, b, aq, bq, num_loop, p_smem);
    }
};

} // namespace ck_tile
