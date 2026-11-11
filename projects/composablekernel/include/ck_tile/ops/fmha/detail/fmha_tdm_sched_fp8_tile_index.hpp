// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Lane and tile index helpers of the FP8 sched pipeline's distributed tiles.
struct FmhaTdmSchedFp8TileIndex
{
    // Row of lane 0 of wavefront `wave` in M iteration `band` of a block GEMM C tile (Y0 = M).
    template <typename CDstr>
    CK_TILE_HOST_DEVICE static constexpr index_t GetBandRowOrigin(index_t wave, index_t band)
    {
        constexpr index_t kNDimP = CDstr::get_num_of_dimension_p();
        array<index_t, kNDimP + CDstr::NDimY> ps_ys{};
        ps_ys(0)      = wave;
        ps_ys(kNDimP) = band;
        return make_tensor_adaptor_coordinate(CDstr{}.get_ps_ys_to_xs_adaptor(), ps_ys)
            .get_bottom_index()[number<0>{}];
    }

    // True if wavefront w owns rows [(b * Waves + w) * BandRows, +BandRows) of M iteration b.
    template <typename CDstr, index_t NumBands, index_t BandRows, index_t Waves>
    CK_TILE_HOST_DEVICE static constexpr bool IsWaveInterleavedBands()
    {
        for(index_t b = 0; b < NumBands; ++b)
            for(index_t w = 0; w < Waves; ++w)
                if(GetBandRowOrigin<CDstr>(w, b) != (b * Waves + w) * BandRows)
                    return false;
        return true;
    }

    // Tile row that this lane holds at distributed index idx of a row tensor.
    template <typename Tensor, typename Idx>
    CK_TILE_DEVICE static index_t LaneRowOf(Idx idx)
    {
        constexpr auto dstr  = typename Tensor::StaticTileDistribution{};
        using PartitionIndex = remove_cvref_t<decltype(get_partition_index(dstr))>;
        const auto lane_x    = make_tensor_adaptor_coordinate(
                                dstr.get_ps_ys_to_xs_adaptor(),
                                container_concat(get_partition_index(dstr),
                                                 array<index_t, dstr.get_num_of_dimension_y()>{}))
                                .get_bottom_index();
        return (lane_x + get_x_indices_from_distributed_indices(dstr, idx, PartitionIndex{}))
            .at(number<0>{});
    }

    // Sets tensor elements where pred(row position, col0 + column) holds; positions per lane row.
    template <typename Tensor, typename RowPositions, typename DataType, typename Predicate>
    CK_TILE_DEVICE static void SetTileIfRowPos(
        Tensor& tensor, DataType value, const RowPositions& pos, index_t col0, Predicate pred)
    {
        constexpr auto dstr  = typename Tensor::StaticTileDistribution{};
        constexpr auto spans = Tensor::get_distributed_spans();
        using PartitionIndex = remove_cvref_t<decltype(get_partition_index(dstr))>;
        const auto lane_x    = make_tensor_adaptor_coordinate(
                                dstr.get_ps_ys_to_xs_adaptor(),
                                container_concat(get_partition_index(dstr),
                                                 array<index_t, dstr.get_num_of_dimension_y()>{}))
                                .get_bottom_index();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            const index_t row = pos[make_tuple(idx0)];
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                const index_t col =
                    col0 + lane_x.at(number<1>{}) +
                    get_x_indices_from_distributed_indices(dstr, idx, PartitionIndex{})
                        .at(number<1>{});
                if(pred(row, col))
                    tensor(idx) = value;
            });
        });
    }

    // set_tile_if with X split into one lane base plus a constant per element; pred may take both.
    template <typename Tensor, typename DataType, typename Predicate>
    CK_TILE_DEVICE static void SetTileIfLaneBase(Tensor& tensor, DataType value, Predicate pred)
    {
        constexpr auto dstr  = typename Tensor::StaticTileDistribution{};
        constexpr auto spans = Tensor::get_distributed_spans();
        using PartitionIndex = remove_cvref_t<decltype(get_partition_index(dstr))>;
        const auto lane_x    = make_tensor_adaptor_coordinate(
                                dstr.get_ps_ys_to_xs_adaptor(),
                                container_concat(get_partition_index(dstr),
                                                 array<index_t, dstr.get_num_of_dimension_y()>{}))
                                .get_bottom_index();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                if constexpr(std::is_invocable_v<Predicate, decltype(lane_x), decltype(lane_x)>)
                {
                    const auto cx =
                        get_x_indices_from_distributed_indices(dstr, idx, PartitionIndex{});
                    if(pred(lane_x + cx, cx))
                        tensor(idx) = value;
                }
                else
                {
                    const auto x = lane_x + get_x_indices_from_distributed_indices(
                                                dstr, idx, PartitionIndex{});
                    if(pred(x))
                        tensor(idx) = value;
                }
            });
        });
    }

    // (row, col) of the lane's element whose distributed index is zero.
    template <typename Tensor>
    CK_TILE_DEVICE static auto TileLaneOrigin()
    {
        constexpr auto dstr = typename Tensor::StaticTileDistribution{};
        return make_tensor_adaptor_coordinate(
                   dstr.get_ps_ys_to_xs_adaptor(),
                   container_concat(get_partition_index(dstr),
                                    array<index_t, dstr.get_num_of_dimension_y()>{}))
            .get_bottom_index();
    }

    // Sets the elements whose lane-independent (row, col) part passes pred.
    template <typename Tensor, typename DataType, typename Predicate>
    CK_TILE_DEVICE static void SetTileIfConstPart(Tensor& tensor, DataType value, Predicate pred)
    {
        constexpr auto dstr  = typename Tensor::StaticTileDistribution{};
        constexpr auto spans = Tensor::get_distributed_spans();
        using PartitionIndex = remove_cvref_t<decltype(get_partition_index(dstr))>;
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                const auto x = get_x_indices_from_distributed_indices(dstr, idx, PartitionIndex{});
                if(pred(x.at(number<0>{}), x.at(number<1>{})))
                    tensor(idx) = value;
            });
        });
    }
};

} // namespace ck_tile
