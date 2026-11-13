// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// MX scale relayout kernel. Embedded in the plugin and compiled at run time with
// HIPRTC for the device in use (compilation/Program.hpp); the host side is
// engines/plans/HipblasltMxScaleLayout.cpp. It rewrites packed UE8M0 block scales into
// the swizzled layouts hipBLASLt's Tensile MX kernels read, which hipBLASLt has no
// function to produce.

#include "MxScaleRelayoutParams.hpp"

namespace hipblaslt_plugin::mx_scale_relayout
{
namespace
{

enum class Swizzle
{
    GFX950,
    GFX1250,
};

constexpr uint32_t TILE_BYTES = TILE * TILE;

// Maps the i-th byte of a tile, in target-contiguous order, to its tile-local (m, k)
// position and its offset in the target buffer. Consecutive i give consecutive
// offsets within each group of 4 bytes.
template <Swizzle S>
__device__ void targetPosition(const Params& p,
                               uint32_t m0,
                               uint32_t k0,
                               uint32_t i,
                               uint32_t& lm,
                               uint32_t& lk,
                               size_t& offset)
{
    if constexpr(S == Swizzle::GFX950)
    {
        // A 64 x 64 tile holds 2 x 8 granules of 32 rows x 8 K blocks; each granule
        // is 256 contiguous target bytes: d1 + 2*d4 + 4*d2 + 64*d5.
        const uint32_t granule = i / 256;
        const uint32_t w = i % 256;
        const uint32_t gm = granule / 8;
        const uint32_t gk = granule % 8;
        const uint32_t d5 = w / 64;
        const uint32_t d2 = (w % 64) / 4;
        const uint32_t d4 = (w / 2) % 2;
        const uint32_t d1 = w % 2;
        lm = 32 * gm + 16 * d1 + d2;
        lk = 8 * gk + 4 * d4 + d5;
        const uint32_t d0 = (m0 + lm) / 32;
        const uint32_t d3 = (k0 + lk) / 8;
        offset = static_cast<size_t>(32) * p.outKb * d0 + 256 * static_cast<size_t>(d3) + w;
    }
    else
    {
        // 16 groups of 4 K blocks; each group is 64 rows x 4 contiguous bytes.
        const uint32_t group = i / (TILE * 4);
        const uint32_t w = i % (TILE * 4);
        lm = w / 4;
        lk = 4 * group + w % 4;
        offset = static_cast<size_t>((k0 + lk) / 4) * p.outMn * 4 + static_cast<size_t>(m0 + lm) * 4
                 + (k0 + lk) % 4;
    }
}

// Rewrites one TILE x TILE block through LDS: coalesced reads in the source order,
// coalesced writes in the target order.
template <Swizzle S>
__device__ void relayout(const Batch& batch)
{
    __shared__ uint8_t tile[TILE][TILE + 4];

    const bool second = blockIdx.x >= batch.firstBlockOfSecond;
    const Params& p = second ? batch.second : batch.first;
    const uint32_t block = second ? blockIdx.x - batch.firstBlockOfSecond : blockIdx.x;
    const uint32_t tilesK = second ? batch.tilesKSecond : batch.tilesKFirst;
    const uint32_t m0 = (block / tilesK) * TILE;
    const uint32_t k0 = (block % tilesK) * TILE;

    // Load the tile walking the source in its own order, so neighbouring lanes read
    // neighbouring bytes. Positions outside the logical scale are zero (the padding).
    if(p.wordLoads)
    {
#pragma unroll
        for(uint32_t w = threadIdx.x; w < TILE_BYTES / 4; w += THREADS)
        {
            const uint32_t row = w / (TILE / 4);
            const uint32_t col = 4 * (w % (TILE / 4));
            const uint32_t m = p.srcKContiguous ? m0 + row : m0 + col;
            const uint32_t k = p.srcKContiguous ? k0 + col : k0 + row;
            uint32_t value = 0;
            if(m < p.mn && k < p.kBlocks)
            {
                const size_t offset = p.srcKContiguous ? static_cast<size_t>(m) * p.kBlocks + k
                                                       : static_cast<size_t>(k) * p.mn + m;
                value = *reinterpret_cast<const uint32_t*>(p.src + offset);
            }
#pragma unroll
            for(uint32_t j = 0; j < 4; ++j)
            {
                const auto byte = static_cast<uint8_t>(value >> (8 * j));
                if(p.srcKContiguous)
                {
                    tile[row][col + j] = byte;
                }
                else
                {
                    tile[col + j][row] = byte;
                }
            }
        }
    }
    else
    {
#pragma unroll
        for(uint32_t i = threadIdx.x; i < TILE_BYTES; i += THREADS)
        {
            const uint32_t lm = p.srcKContiguous ? i / TILE : i % TILE;
            const uint32_t lk = p.srcKContiguous ? i % TILE : i / TILE;
            const uint32_t m = m0 + lm;
            const uint32_t k = k0 + lk;
            uint8_t value = 0;
            if(m < p.mn && k < p.kBlocks)
            {
                value = p.srcKContiguous ? p.src[static_cast<size_t>(m) * p.kBlocks + k]
                                         : p.src[static_cast<size_t>(k) * p.mn + m];
            }
            tile[lm][lk] = value;
        }
    }
    __syncthreads();

    if(p.wordStores)
    {
        // The 4 bytes of a word share their row (or granule), so the bounds of the
        // first byte hold for all four.
#pragma unroll
        for(uint32_t w = threadIdx.x; w < TILE_BYTES / 4; w += THREADS)
        {
            uint32_t packed = 0;
            size_t offset = 0;
            bool inBounds = false;
#pragma unroll
            for(uint32_t j = 0; j < 4; ++j)
            {
                uint32_t lm = 0;
                uint32_t lk = 0;
                size_t byteOffset = 0;
                targetPosition<S>(p, m0, k0, 4 * w + j, lm, lk, byteOffset);
                if(j == 0)
                {
                    offset = byteOffset;
                    inBounds = m0 + lm < p.outMn && k0 + lk < p.outKb;
                }
                packed |= static_cast<uint32_t>(tile[lm][lk]) << (8 * j);
            }
            if(inBounds)
            {
                *reinterpret_cast<uint32_t*>(p.dst + offset) = packed;
            }
        }
    }
    else
    {
#pragma unroll
        for(uint32_t i = threadIdx.x; i < TILE_BYTES; i += THREADS)
        {
            uint32_t lm = 0;
            uint32_t lk = 0;
            size_t offset = 0;
            targetPosition<S>(p, m0, k0, i, lm, lk, offset);
            if(m0 + lm < p.outMn && k0 + lk < p.outKb)
            {
                p.dst[offset] = tile[lm][lk];
            }
        }
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(THREADS) void MxScaleRelayoutGfx950(Batch batch)
{
    relayout<Swizzle::GFX950>(batch);
}

extern "C" __global__ __launch_bounds__(THREADS) void MxScaleRelayoutGfx1250(Batch batch)
{
    relayout<Swizzle::GFX1250>(batch);
}

} // namespace hipblaslt_plugin::mx_scale_relayout
