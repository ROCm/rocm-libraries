// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

struct BatchedTransposeHostArgs
{
    const void* p_input;
    void* p_output;
    index_t batch;
    index_t height;
    index_t width;
    index_t dim_stride;
    index_t dim_block_h;
    index_t dim_block_w;
};

template <typename Pipeline_>
struct BatchedTransposeKernel
{

    CK_TILE_DEVICE static index_t counter = 0;
    using Pipeline                        = remove_cvref_t<Pipeline_>;
    using Problem                         = remove_cvref_t<typename Pipeline::Problem>;

    using Type = typename Problem::InputType;

    struct BatchedTransposeKargs
    {
        const void* p_input;
        void* p_output;
        index_t batch;
        index_t height;
        index_t width;
        index_t dim_stride;
    };

    using Kargs = BatchedTransposeKargs;
    using Hargs = BatchedTransposeHostArgs;

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& host_args)
    {
        size_t grid_size_x = (host_args.height + host_args.dim_block_h - 1) / host_args.dim_block_h;
        size_t grid_size_y = (host_args.width + host_args.dim_block_w - 1) / host_args.dim_block_w;
        size_t grid_size_z = host_args.batch;
        return dim3(grid_size_x, grid_size_y, grid_size_z);
    }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_input    = h.p_input;
        k.p_output   = h.p_output;
        k.batch      = h.batch;
        k.height     = h.height;
        k.width      = h.width;
        k.dim_stride = h.dim_stride;
        return k;
    }

    CK_TILE_HOST_DEVICE static constexpr auto BlockSize() { return Problem::kBlockSize; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        static constexpr ck_tile::index_t kMPerBlock       = Problem::kMPerBlock;
        static constexpr ck_tile::index_t kNPerBlock       = Problem::kNPerBlock;
        static constexpr bool kPadM                        = Problem::kPadM;
        static constexpr bool kPadN                        = Problem::kPadN;
        static constexpr ck_tile::index_t VectorSizeInput  = Problem::VectorSizeInput;
        static constexpr ck_tile::index_t VectorSizeOutput = Problem::VectorSizeOutput;

        const auto iM   = __builtin_amdgcn_readfirstlane(blockIdx.x * kMPerBlock);
        const auto iN   = __builtin_amdgcn_readfirstlane(blockIdx.y * kNPerBlock);
        const auto iDim = blockIdx.z;

        const auto x_m_n = [&]() {
            const auto x_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const Type*>(kargs.p_input) + iDim * kargs.dim_stride,
                make_tuple(kargs.height, kargs.width),
                make_tuple(kargs.width, 1),
                number<VectorSizeInput>{},
                number<1>{});

            return pad_tensor_view(x_dram_naive,
                                   make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                   sequence<kPadN, kPadM>{});
        }();

        const auto y_n_m = [&]() {
            const auto y_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<Type*>(kargs.p_output) + iDim * kargs.dim_stride,
                make_tuple(kargs.width, kargs.height),
                make_tuple(kargs.height, 1),
                number<VectorSizeOutput>{},
                number<1>{});

            return pad_tensor_view(y_dram_naive,
                                   make_tuple(number<kNPerBlock>{}, number<kMPerBlock>{}),
                                   sequence<kPadM, kPadN>{});
        }();

        auto x_block_window = make_tile_window(
            x_m_n,
            make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
            {static_cast<ck_tile::index_t>(iM), static_cast<ck_tile::index_t>(iN)});

        auto y_block_window = make_tile_window(
            y_n_m,
            make_tuple(number<kNPerBlock>{}, number<kMPerBlock>{}),
            {static_cast<ck_tile::index_t>(iN), static_cast<ck_tile::index_t>(iM)});

        Pipeline{}(x_block_window, y_block_window);
    }
};
} // namespace ck_tile
