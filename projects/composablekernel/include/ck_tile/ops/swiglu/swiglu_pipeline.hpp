// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#include <type_traits>

namespace ck_tile {
template <typename BasePipeline_, typename ActMulOp_>
struct SwiGLUPipeline : BasePipeline_
{
    private:
    template <typename T>
    static constexpr bool is_tuple_v = is_detected<is_tuple, T>::value;

    public:
    // ActMulOp: y = x0 * act(x1)
    using ActMulOp = remove_cvref_t<ActMulOp_>;

    // note we also inherit from BasePipeline_
    using BasePipeline = remove_cvref_t<BasePipeline_>;

    using BaseBsDataType = typename BasePipeline ::BsDataType;
    using BaseBsLayout   = typename BasePipeline ::BsLayout;
    static_assert(is_tuple_v<BaseBsDataType>);
    static_assert(is_tuple_v<BaseBsLayout>);
    static_assert(BaseBsDataType::size() == 1);
    static_assert(BaseBsLayout::size() == 1);

    using BDataType = remove_cvref_t<std::tuple_element_t<0, BaseBsDataType>>;
    using BLayout   = remove_cvref_t<std::tuple_element_t<0, BaseBsLayout>>;

    using AsDataType = typename BasePipeline ::AsDataType;
    using AsLayout   = typename BasePipeline ::AsLayout;

    // Repeat datatype/layout of base pipeline, because this class does 2 gemms instead of 1
    using BsDataType = tuple<BDataType, BDataType>;
    using BsLayout   = tuple<BLayout, BLayout>;

    using CDataType = typename BasePipeline ::CDataType;
    using CLayout   = typename BasePipeline ::CLayout;

    template <typename AsWindow, typename BsWindow>
    CK_TILE_DEVICE auto operator()(const AsWindow& a_dram_block_window_tmp,
                                   const BsWindow& b_dram_block_window_tmp,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem) const
    {
        static_assert(is_tuple_v<AsWindow>);
        static_assert(is_tuple_v<BsWindow>);
        static_assert(AsWindow::size() == 1);
        static_assert(BsWindow::size() == 2);

        const auto a  = ck_tile::make_tuple(a_dram_block_window_tmp[number<0>{}]);
        const auto b0 = ck_tile::make_tuple(b_dram_block_window_tmp[number<0>{}]);
        const auto b1 = ck_tile::make_tuple(b_dram_block_window_tmp[number<1>{}]);

        auto c1 = BasePipeline{}(a, b1, num_loop, p_smem);
        auto c0 = BasePipeline{}(a, b0, num_loop, p_smem);
        sweep_tile(c0, [&c0, &c1](auto idx) { ActMulOp{}(c0(idx), c0[idx], c1[idx]); });
        return c0;
    }

    template <typename AsWindow, typename BsWindow, typename AElemOp, typename BElemOp>
    CK_TILE_DEVICE auto operator()(const AsWindow& a_dram_block_window_tmp,
                                   const AElemOp& /*unused*/,
                                   const BsWindow& b_dram_block_window_tmp,
                                   const BElemOp& /*unused*/,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        static_assert(std::is_same_v<AElemOp, element_wise::PassThrough>);
        static_assert(std::is_same_v<BElemOp, element_wise::PassThrough>);

        return this->operator()(a_dram_block_window_tmp, b_dram_block_window_tmp, num_loop, p_smem);
    }
};
} // namespace ck_tile
