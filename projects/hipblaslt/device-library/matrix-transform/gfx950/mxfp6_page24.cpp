// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The near-die placement of MXFP6 operands and their A scales for the rocKE GEMM on gfx950:
// HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT (A), HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT
// (B) and HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT (A scales). The host checks
// every shape before a launch; these kernels bounds-check nothing.

#include "mxfp6_page24.h"

#include <cstdint>
#include <hip/hip_runtime.h>

namespace
{
    constexpr uint32_t kThreads      = mxfp6Page24Threads;
    constexpr uint64_t kPageBytes    = uint64_t(2) << 20;
    constexpr uint64_t kBlockBytes   = 65536; // one block row at one K step
    constexpr uint32_t kTileBytes    = 24576; // a 48-byte-group tile: 256 rows by 128 K
    constexpr uint32_t kWaveBytes    = 6144;
    constexpr uint32_t kPieceBytes   = 3072;
    constexpr uint32_t kStepsPerPage = 24;
    constexpr uint32_t kPagesPerRow  = 32;
    constexpr uint32_t kScaleStep    = 16384; // 128x4 scales of 4096 rows at one K step
    constexpr uint32_t kHalfBytes    = 8192;

    // Offset in its block of the 16 bytes at tile byte x, in half 0.
    __device__ uint32_t blockOffset(uint32_t x)
    {
        return 16384 * (x / kWaveBytes) + 4096 * ((x % kWaveBytes) / kPieceBytes) + x % kPieceBytes;
    }
}

// One work-group per tile and K step: blockIdx.x = tile * kTiles + s. A split operand has 16
// tiles in 8 block rows, tile t in half (t / 8) % 2 of row t % 8; an unsplit one has one row per
// tile, in both halves. Row r is shifted by (r % 8) blocks within its page.
extern "C" __global__
    __launch_bounds__(kThreads) void hipblaslt_mxfp6_page24_data(uint8_t* __restrict__ dst,
                                                                 const uint8_t* __restrict__ src,
                                                                 uint32_t kTiles,
                                                                 uint32_t split)
{
    const uint32_t tile     = blockIdx.x / kTiles;
    const uint32_t s        = blockIdx.x % kTiles;
    const uint32_t r        = split ? tile % 8 : tile;
    const uint64_t q        = uint64_t(kPagesPerRow) * r + s / kStepsPerPage;
    const uint64_t page     = 2 * q + ((q >> 6) & 1);
    uint8_t* const block    = dst + page * kPageBytes + (s % kStepsPerPage + r % 8) * kBlockBytes;
    const uint8_t* const in = src + uint64_t(blockIdx.x) * kTileBytes;

    for(uint32_t x = threadIdx.x * 16; x < kTileBytes; x += kThreads * 16)
    {
        const uint4    v   = *reinterpret_cast<const uint4*>(in + x);
        const uint32_t off = blockOffset(x);
        if(split)
        {
            *reinterpret_cast<uint4*>(block + off + kHalfBytes * ((tile / 8) % 2)) = v;
        }
        else
        {
            *reinterpret_cast<uint4*>(block + off)              = v;
            *reinterpret_cast<uint4*>(block + off + kHalfBytes) = v;
        }
    }
}

// One work-group per K step s: its 16 KiB of 128x4 scales go to slot s % 24 (32 KiB) of page
// 2 * (s / 24), each 8 KiB granule g as two identical halves.
extern "C" __global__ __launch_bounds__(kThreads) void hipblaslt_mxfp6_page24_scales(
    uint8_t* __restrict__ dst, const uint8_t* __restrict__ src)
{
    const uint32_t s    = blockIdx.x;
    uint8_t* const slot = dst + 2 * uint64_t(s / kStepsPerPage) * kPageBytes
                          + (s % kStepsPerPage) * uint64_t(2 * kScaleStep);
    const uint8_t* const in = src + uint64_t(s) * kScaleStep;

    for(uint32_t y = threadIdx.x * 16; y < kScaleStep; y += kThreads * 16)
    {
        const uint4    v                      = *reinterpret_cast<const uint4*>(in + y);
        const uint32_t off                    = 2 * kHalfBytes * (y / kHalfBytes) + y % kHalfBytes;
        *reinterpret_cast<uint4*>(slot + off) = v;
        *reinterpret_cast<uint4*>(slot + off + kHalfBytes) = v;
    }
}
