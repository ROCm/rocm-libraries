// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/amd_tdm_descriptor.hpp"

namespace ck_tile {

// LDS layout and the KV cache copies of BlockFmhaFwdDecodePagedKVPipeline.
//
// K and V pages are copied to LDS with TDM gathers of rows of 16-byte vectors of one kv head. In
// the vectorized cache layout those bytes are WMMA operands as they are (S^T = K Q^T and
// O^T += V^T P^T), so the pipeline reads them back without transposing.
struct BlockFmhaFwdDecodePagedKVPipelineDefaultPolicy
{
    // 16-token pages per cache page
    template <typename Problem>
    static constexpr index_t kSubPages = Problem::kPageBlockSize / Problem::kN0;
    // pages per wave per step: one (cooperative) or four (wave split)
    template <typename Problem>
    static constexpr index_t kWavePages = Problem::kWaveSplit ? 4 : 1;
    // steps whose copies are in flight or in use
    template <typename Problem>
    static constexpr index_t kStages = Problem::kWaveSplit ? 2 : 3;
    template <typename Problem>
    static constexpr index_t kPageBytes = Problem::kN0 * Problem::kHeadDim;
    // LDS K of a wave-split step: four page-16 images, or with larger cache pages the 64 tokens as
    // they lie in the cache ([hdim / 16][64 tokens][16 B]); offsets of a page and a 16-byte chunk
    template <typename Problem>
    static constexpr index_t kKPageStride =
        Problem::kWaveSplit && kSubPages<Problem> > 1 ? 256 : kPageBytes<Problem>;
    template <typename Problem>
    static constexpr index_t kKChunkStride =
        Problem::kWaveSplit && kSubPages<Problem> > 1 ? 1024 : 256;

    // cooperative layout: the four waves share each step's four pages
    template <typename Problem>
    struct alignas(16) SmemCoop
    {
        alignas(16) uint8_t k[kStages<Problem>][Problem::kNumWarps][kPageBytes<Problem>];
        alignas(16) uint8_t v[kStages<Problem>][Problem::kNumWarps][kPageBytes<Problem>];
        alignas(16) uint8_t p[Problem::kM0][Problem::kNumWarps * Problem::kN0];
        alignas(16) float red[2][Problem::kNumWarps][Problem::kM0]; // per-wave row max / row sum
    };

    // wave-split layout
    template <typename Problem>
    struct alignas(16) SmemWaveSplit
    {
        // per wave, so that a wave's O for the intra-workgroup merge reuses only its own stages
        union alignas(16) PerWave
        {
            struct
            {
                alignas(16) uint8_t k[kStages<Problem>][kWavePages<Problem>][kPageBytes<Problem>];
                alignas(16) uint8_t v[kStages<Problem>][kWavePages<Problem>][kPageBytes<Problem>];
            } kv;
            // padded rows keep the column-wise stores of 16 rows off one bank
            alignas(16) float o[Problem::kM0][Problem::kHeadDim + 4];
        } wave[Problem::kNumWarps];
        alignas(16) float ml[2][Problem::kNumWarps][Problem::kM0];
    };

    template <typename Problem>
    using Smem = std::conditional_t<Problem::kWaveSplit, SmemWaveSplit<Problem>, SmemCoop<Problem>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return sizeof(Smem<Problem>);
    }

#if defined(__gfx125__)
    // LDS buffers of page q of wave i_wave's pages of stage st
    template <typename Problem>
    CK_TILE_DEVICE static uint8_t*
    GetKPage(Smem<Problem>& sm, index_t i_wave, index_t st, index_t q)
    {
        if constexpr(Problem::kWaveSplit)
            return &sm.wave[i_wave].kv.k[st][0][0] + q * kKPageStride<Problem>;
        else
        {
            ignore = q;
            return &sm.k[st][i_wave][0];
        }
    }
    template <typename Problem>
    CK_TILE_DEVICE static uint8_t*
    GetVPage(Smem<Problem>& sm, index_t i_wave, index_t st, index_t q)
    {
        if constexpr(Problem::kWaveSplit)
            return &sm.wave[i_wave].kv.v[st][q][0];
        else
        {
            ignore = q;
            return &sm.v[st][i_wave][0];
        }
    }

    // TDM gather of kCount rows of kRowBytes (row indices up to 2^31) to consecutive LDS rows
    template <index_t kRowBytes, index_t kCount>
    CK_TILE_DEVICE static void
    TdmGatherRows(const uint8_t* base, uint8_t* lds, const int32_t (&rows)[kCount])
    {
        static_assert(kCount <= 8 && kRowBytes % 8 == 0);
        TDM_GROUP0 g0{static_cast<uint32_t>(reinterpret_cast<uintptr_t>(to_lds(lds))),
                      reinterpret_cast<uintptr_t>(base),
                      static_cast<uint32_t>(TDMGatherIndexSize::Row32bit_Index),
                      1};
        TDM_GROUP1 g1;
        g1.data_size = 3; // 8-byte units
        g1.tensorDim0(kRowBytes / 8);
        g1.tensorDim1(0x7fffffff);
        g1.tensorDim0Stride(kRowBytes / 8);
        g1.tile_dim0 = kRowBytes / 8;
        g1.tile_dim1 = kCount; // row indices
        TDM_GROUP2 g2;
        TDM_GROUP3 g3;
        for(index_t i = 0; i < kCount; ++i)
        {
            if(i < 4)
                g2.mode1.setRowIndex32(i, static_cast<uint32_t>(rows[i]));
            else
                g3.mode1.setRowIndex32(i - 4, static_cast<uint32_t>(rows[i]));
        }
        __builtin_amdgcn_tensor_load_to_lds(
            bit_cast<uint32x4_t>(amd_wave_read_first_lane(g0.bitfield)),
            amd_wave_read_first_lane(g1.bitfield),
            amd_wave_read_first_lane(g2.bitfield),
            amd_wave_read_first_lane(g3.bitfield),
            int32x8_t{},
            0);
    }
#endif
};

} // namespace ck_tile
