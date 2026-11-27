// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled.hpp"

namespace ck_tile {

template <typename Pipeline, typename Shape>
using MxTdmStoreEpilogueBase =
    TdmEpilogueStaged<CShuffleEpilogueProblem<typename Pipeline::ADataType,
                                              typename Pipeline::BDataType,
                                              tuple<>,
                                              float,
                                              typename Pipeline::ODataType,
                                              tuple<>,
                                              tensor_layout::gemm::RowMajor,
                                              element_wise::PassThrough,
                                              Shape::kM,
                                              Shape::kN,
                                              Shape::BlockWarps::at(number<0>{}),
                                              Shape::BlockWarps::at(number<1>{}),
                                              Shape::WarpTile::at(number<0>{}),
                                              Shape::WarpTile::at(number<1>{}),
                                              Shape::WarpTile::at(number<2>{}),
                                              /*isCTransposed=*/false>,
                      /*MIterPerStage=*/0,
                      /*NumStageBuffers=*/2,
                      std::conditional_t<std::is_same_v<typename Pipeline::ODataType, fp8_t>,
                                         element_wise::SaturateFp8E4M3,
                                         element_wise::PassThrough>>;

/**
 * @brief TdmEpilogue for MxGemmPipelineAgBgCrCompTDMScheduled: E = acc staged in LDS
 *        (kTileM * kTileN * sizeof(E) bytes) and stored by TDM. E is row-major and there are no
 *        Ds tensors. An FP8 E saturates: SaturateFp8E4M3 maps each accumulator before the cast.
 *        With WideD, E's view is rebased at the tile's first row, as the pipeline's A and B
 *        component windows are: the store's element offset is 32-bit, and E's pass 2^31.
 */
template <typename Pipeline_,
          typename Pipeline = remove_cvref_t<Pipeline_>,
          typename Shape    = typename Pipeline::BlockGemmShape>
struct MxTdmStoreEpilogue : MxTdmStoreEpilogueBase<Pipeline, Shape>
{
    using Base = MxTdmStoreEpilogueBase<Pipeline, Shape>;

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem)
    {
        if constexpr(Pipeline::Traits::kWideD)
        {
            using View = remove_cvref_t<decltype(out_dram_window.get_bottom_tensor_view())>;
            static_assert(numeric_traits<typename View::DataType>::PackedSize == 1,
                          "E's rows are addressed in elements");
            const auto& view   = out_dram_window.get_bottom_tensor_view();
            const auto origin  = out_dram_window.get_window_origin();
            const auto lengths = detail::tdm_real_lengths(view.get_tensor_descriptor());
            const index_t row0 = origin[number<0>{}];
            const index_t stride =
                view.get_tensor_descriptor().calculate_offset(make_multi_index(1, 0));
            // The tile's rows only: the rows past the end of E stay clipped.
            const index_t rows = min(lengths[number<0>{}] - row0, index_t{Shape::kM});
            auto rebased = make_naive_tensor_view<address_space_enum::global, View::DstInMemOp>(
                view.get_buffer_view().p_data_ + static_cast<long_index_t>(row0) * stride,
                make_tuple(rows, lengths[number<1>{}]),
                make_tuple(stride, 1));
            auto window = make_tile_window(rebased,
                                           make_tuple(number<Shape::kM>{}, number<Shape::kN>{}),
                                           {0, origin[number<1>{}]});
            return Base::operator()(window, o_acc_tile, ds_dram_windows, p_smem);
        }
        else
        {
            return Base::operator()(out_dram_window, o_acc_tile, ds_dram_windows, p_smem);
        }
    }
};

} // namespace ck_tile
