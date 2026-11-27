// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_tile_index.hpp"

namespace ck_tile {

// O store of the FP8 sched pipeline: LDS staging tile, then one TDM store per row band.
template <typename Geometry, typename ODataType>
struct FmhaTdmSchedFp8OutputStore
{
    static constexpr index_t kM0 = Geometry::kM;
    static constexpr index_t kN1 = Geometry::kDv;

    // O is written by the pipeline through a dedicated LDS staging tile and per-wavefront TDM.
    static constexpr bool kStoresOutputViaTdm = true;
    // 16 B row pad: the 16 rows of one ds_store_b128 then start on distinct bank quads.
    static constexpr index_t kOLdsRowPitch = kN1 + 16 / sizeof(ODataType);
    // Own staging; an epilogue class reusing the K/V LDS would need a K/V drain barrier.
    static constexpr index_t kOLdsBytes = kM0 * kOLdsRowPitch * sizeof(ODataType);
    static constexpr index_t kOBandRows = Geometry::PvWmma::kM;

    // o_ptr/rows_left/cols_left are relative to the tile origin (i_m0, i_n1).
    template <typename OAccTile>
    CK_TILE_DEVICE static void StoreOutput(const OAccTile& o_acc_tile,
                                           ODataType* o_ptr,
                                           index_t rows_left,
                                           index_t cols_left,
                                           index_t stride_o,
                                           void* smem_o)
    {
        static_assert(16 % sizeof(ODataType) == 0, "O LDS rows are padded by one 16 B vector");
        static_assert(kM0 == Geometry::kPvMIter * Geometry::kWaves * kOBandRows,
                      "each wavefront must own kPvMIter bands of kOBandRows rows");
        static_assert(FmhaTdmSchedFp8TileIndex::IsWaveInterleavedBands<
                          typename OAccTile::StaticTileDistribution,
                          Geometry::kPvMIter,
                          kOBandRows,
                          Geometry::kWaves>(),
                      "the barrier-free O store needs each wavefront to read back only its rows");

        auto* smem = static_cast<ODataType*>(smem_o);
        auto o_lds = make_tensor_view<address_space_enum::lds>(
            smem,
            make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN1>{}),
                                         make_tuple(number<kOLdsRowPitch>{}, number<1>{}),
                                         number<16 / sizeof(ODataType)>{},
                                         number<1>{}));
        auto o_lds_window = make_tile_window(o_lds,
                                             make_tuple(number<kM0>{}, number<kN1>{}),
                                             {0, 0},
                                             o_acc_tile.get_tile_distribution());
        store_tile(o_lds_window, cast_tile<ODataType>(o_acc_tile));
        // Each TDM store below only reads rows this wavefront wrote, so no workgroup barrier.
        s_wait_dscnt<0>();

        // TDM moves sizeof(ODataType) elements, so o_ptr and stride_o need only element alignment.
        const index_t wave = get_warp_id();
        static_for<0, Geometry::kPvMIter, 1>{}([&](auto band) {
            const index_t row0 =
                (static_cast<index_t>(band) * Geometry::kWaves + wave) * kOBandRows;
            const index_t rows = min(rows_left - row0, kOBandRows);
            if(rows > 0)
            {
                // Box spans the pad; relies on the TDM store dropping columns past tensor_dim0.
                const uint32_t dims[2]    = {static_cast<uint32_t>(cols_left),
                                             static_cast<uint32_t>(rows)};
                const uint64_t strides[2] = {static_cast<uint64_t>(stride_o), 0};
                constexpr uint16_t box[2] = {kOLdsRowPitch, kOBandRows};
                const auto desc           = createTDMDescriptor<ODataType, 2>(
                    o_ptr + static_cast<long_index_t>(row0) * stride_o,
                    smem + row0 * kOLdsRowPitch,
                    dims,
                    strides,
                    box,
                    TDMConfig{});
                amd_tdm_store(desc);
            }
        });
        s_wait_tensorcnt<0>();
    }
};

} // namespace ck_tile
