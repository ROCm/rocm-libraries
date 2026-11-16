// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host-side layouts of HIPBLASLT_ORDER_COL256_R128_EXT (FP6 data) and
// HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT (E8M0 scales), from a
// K-contiguous operand: transposed A or non-transposed B, one row per M (N) index.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx_tile128x4
{
    constexpr size_t kTileRows      = 256; // M (N) rows per data tile
    constexpr size_t kTileK         = 128; // K elements per data tile
    constexpr size_t kGroupK        = 32; // K elements per 24-byte group
    constexpr size_t kGroupBytes    = kGroupK * 6 / 8;
    constexpr size_t kTileBytes     = kTileRows * kTileK * 6 / 8;
    constexpr size_t kLowPlaneBytes = 16; // leading bytes of a group, in the first plane
    constexpr size_t kScaleRows     = 128; // rows per scale tile
    constexpr size_t kScaleBlocks   = 4; // K blocks per scale tile

    inline void checkDataShape(size_t rows, size_t k)
    {
        if(rows % kTileRows != 0 || k % kTileK != 0)
            throw std::invalid_argument("HIPBLASLT_ORDER_COL256_R128_EXT needs rows % 256 == 0 "
                                        "and K % 128 == 0; got rows="
                                        + std::to_string(rows) + " K=" + std::to_string(k));
    }

    inline void checkScaleShape(size_t rows, size_t kBlocks)
    {
        if(rows % kScaleRows != 0 || kBlocks % kScaleBlocks != 0)
            throw std::invalid_argument("HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT needs "
                                        "rows % 128 == 0 and K blocks % 4 == 0; got rows="
                                        + std::to_string(rows)
                                        + " K blocks=" + std::to_string(kBlocks));
    }

    // Byte offset of the 24-byte group (row, k / 32): bytes [0, 16) of the group go to
    // the returned offset, bytes [16, 24) to tiledGroupHighOffset.
    inline size_t tiledGroupLowOffset(size_t row, size_t kGroup, size_t kTotal)
    {
        size_t const kTiles = kTotal / kTileK;
        size_t const tile   = (row / kTileRows) * kTiles + kGroup / (kTileK / kGroupK);
        size_t const o
            = 64 * ((row % kTileRows) / 16) + 16 * (kGroup % (kTileK / kGroupK)) + row % 16;
        return tile * kTileBytes + kLowPlaneBytes * o;
    }

    inline size_t tiledGroupHighOffset(size_t row, size_t kGroup, size_t kTotal)
    {
        size_t const lowOffset = tiledGroupLowOffset(row, kGroup, kTotal);
        size_t const tileBase  = lowOffset / kTileBytes * kTileBytes;
        size_t const o         = (lowOffset - tileBase) / kLowPlaneBytes;
        return tileBase + kTileRows * kTileK / kGroupK * kLowPlaneBytes
               + (kGroupBytes - kLowPlaneBytes) * o;
    }

    // dst and src hold rows * k * 6 / 8 bytes; src rows are the 6-bit LSB-first packed K
    // elements of each M (N) index.
    inline void tileFp6Data(uint8_t* dst, uint8_t const* src, size_t rows, size_t k)
    {
        checkDataShape(rows, k);
        size_t const rowBytes = k * 6 / 8;
        for(size_t r = 0; r < rows; ++r)
        {
            for(size_t g = 0; g < k / kGroupK; ++g)
            {
                uint8_t const* group = src + r * rowBytes + g * kGroupBytes;
                std::memcpy(dst + tiledGroupLowOffset(r, g, k), group, kLowPlaneBytes);
                std::memcpy(dst + tiledGroupHighOffset(r, g, k),
                            group + kLowPlaneBytes,
                            kGroupBytes - kLowPlaneBytes);
            }
        }
    }

    // Byte offset of the scale of row `row` and K block `kBlock`.
    inline size_t shuffledScaleOffset(size_t row, size_t kBlock, size_t rows)
    {
        size_t const tile = (kBlock / kScaleBlocks) * (rows / kScaleRows) + row / kScaleRows;
        return ((tile * 64 + 16 * (kBlock % kScaleBlocks) + row % 16) * 8)
               + (row % kScaleRows) / 16;
    }

    // src is [rows][kBlocks], K block fastest.
    inline std::vector<uint8_t>
        shuffleScales(std::vector<uint8_t> const& src, size_t rows, size_t kBlocks)
    {
        checkScaleShape(rows, kBlocks);
        if(src.size() != rows * kBlocks)
            throw std::invalid_argument("128x4 scale shuffle: expected "
                                        + std::to_string(rows * kBlocks) + " scales, got "
                                        + std::to_string(src.size()));
        std::vector<uint8_t> dst(src.size());
        for(size_t r = 0; r < rows; ++r)
            for(size_t kb = 0; kb < kBlocks; ++kb)
                dst[shuffledScaleOffset(r, kb, rows)] = src[r * kBlocks + kb];
        return dst;
    }
} // namespace mx_tile128x4
