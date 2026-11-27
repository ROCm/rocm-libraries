// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host-side layouts of the HIPBLASLT_ORDER_COL256_R128_* orders (FP6 data) and the
// HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT and _128_4_PAGE24_EXT scale modes
// (E8M0 scales), from a K-contiguous operand: transposed A or non-transposed B, one row
// per M (N) index.

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

    // The order inside a 24576-byte tile: HIPBLASLT_ORDER_COL256_R128_EXT,
    // HIPBLASLT_ORDER_COL256_R128_P48_EXT or HIPBLASLT_ORDER_COL256_R128_P16_EXT. The
    // The near-die orders use the 48-byte tiles.
    enum class TileOrder
    {
        COL256_R128,
        P48,
        P16,
    };

    // Offset in its tile of byte b of the 24-byte group of row `row` and K group `kGroup`.
    inline size_t tileByteOffset(TileOrder order, size_t row, size_t kGroup, size_t b)
    {
        size_t const c  = row % kTileRows;
        size_t const kg = kGroup % (kTileK / kGroupK);
        size_t const j  = c / 32;
        size_t const p  = (c / 16) % 2;
        size_t const l  = 16 * kg + c % 16;
        switch(order)
        {
        case TileOrder::COL256_R128:
        {
            size_t const o = 64 * (c / 16) + 16 * kg + c % 16;
            return b < kLowPlaneBytes
                       ? kLowPlaneBytes * o + b
                       : kTileRows * kTileK / kGroupK * kLowPlaneBytes
                             + (kGroupBytes - kLowPlaneBytes) * o + (b - kLowPlaneBytes);
        }
        case TileOrder::P48:
            return 3072 * j + 48 * l + 24 * p + b;
        case TileOrder::P16:
        {
            size_t const t = 24 * p + b;
            return 3072 * j + 1024 * (t / 16) + 16 * l + t % 16;
        }
        }
        throw std::invalid_argument("unknown TileOrder");
    }

    // dst and src hold rows * k * 6 / 8 bytes, as for tileFp6Data.
    inline void
        tileFp6Data(uint8_t* dst, uint8_t const* src, size_t rows, size_t k, TileOrder order)
    {
        checkDataShape(rows, k);
        size_t const rowBytes = k * 6 / 8;
        size_t const kTiles   = k / kTileK;
        for(size_t r = 0; r < rows; ++r)
            for(size_t g = 0; g < k / kGroupK; ++g)
            {
                size_t const   tile  = (r / kTileRows) * kTiles + g / (kTileK / kGroupK);
                uint8_t const* group = src + r * rowBytes + g * kGroupBytes;
                for(size_t b = 0; b < kGroupBytes; ++b)
                    dst[tile * kTileBytes + tileByteOffset(order, r, g, b)] = group[b];
            }
    }

    // The placed orders: HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT (A) and
    // HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT (B).
    constexpr size_t kPlacedBlockBytes = 65536;
    constexpr size_t kPlacedPieceBytes = 3072; // contiguous run of a tile in a block
    constexpr size_t kPageBytes        = size_t(2) << 20;
    constexpr size_t kPage24MaxKTiles  = 768;
    constexpr size_t kSplitTiles       = 16; // tiles of the split order, 4096 rows

    // Offset in a 64 KiB block of byte x of a tile written to half h.
    inline size_t placedBlockOffset(size_t x, size_t h)
    {
        return 16384 * (x / 6144) + 8192 * h + 4096 * ((x % 6144) / 3072) + x % 3072;
    }

    // Start of block row r at K tile s.
    inline size_t page24BlockOffset(size_t r, size_t s)
    {
        return (64 * r + 2 * (s / 24) + (r / 2) % 2) * kPageBytes
               + (s % 24 + r % 8) * kPlacedBlockBytes;
    }

    inline void checkPlacedShape(size_t rows, size_t k, bool split)
    {
        checkDataShape(rows, k);
        if(k / kTileK > kPage24MaxKTiles || (split && rows != kSplitTiles * kTileRows))
            throw std::invalid_argument("the placed orders need K / 128 <= 768, and 4096 rows "
                                        "when split; got rows="
                                        + std::to_string(rows) + " K=" + std::to_string(k));
    }

    inline size_t placedFp6Bytes(size_t rows, bool split)
    {
        return (64 * (split ? kSplitTiles / 2 : rows / kTileRows) + 2) * kPageBytes;
    }

    // Offset of tile byte x of tile t at K tile s, in half h; a split tile is in half t / 8
    // only, an unsplit one in both halves.
    inline size_t placedFp6Offset(size_t t, size_t s, size_t x, size_t h, bool split)
    {
        size_t const r = split ? t % (kSplitTiles / 2) : t;
        return page24BlockOffset(r, s) + placedBlockOffset(x, h);
    }

    // dst holds placedFp6Bytes(rows, split) bytes, which the caller zeroes; tiled is rows x k
    // in HIPBLASLT_ORDER_COL256_R128_P48_EXT.
    inline void
        placeP48Tiles(uint8_t* dst, uint8_t const* tiled, size_t rows, size_t k, bool split)
    {
        checkPlacedShape(rows, k, split);
        size_t const kTiles = k / kTileK;
        for(size_t t = 0; t < rows / kTileRows; ++t)
            for(size_t s = 0; s < kTiles; ++s)
                for(size_t h = 0; h < 2; ++h)
                {
                    if(split && h != t / (kSplitTiles / 2))
                        continue;
                    uint8_t const* tile = tiled + (t * kTiles + s) * kTileBytes;
                    for(size_t x = 0; x < kTileBytes; x += kPlacedPieceBytes)
                        std::memcpy(dst + placedFp6Offset(t, s, x, h, split),
                                    tile + x,
                                    kPlacedPieceBytes);
                }
    }

    // dst as for placeP48Tiles; src as for tileFp6Data.
    inline void placeFp6Data(uint8_t* dst, uint8_t const* src, size_t rows, size_t k, bool split)
    {
        checkPlacedShape(rows, k, split);
        std::vector<uint8_t> tiled(rows * k * 6 / 8);
        tileFp6Data(tiled.data(), src, rows, k, TileOrder::P48);
        placeP48Tiles(dst, tiled.data(), rows, k, split);
    }

    // HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT: the 128x4 scales of
    // 4096 rows, 16384 bytes per K tile, placed in 2 MiB pages.
    constexpr size_t kPlacedScaleRows = 4096;

    // Offset of byte x of the 128x4 scales in half h.
    inline size_t page24ScaleOffset(size_t x, size_t h)
    {
        size_t const s = x / 16384;
        size_t const y = x % 16384;
        return 2 * (s / 24) * kPageBytes + (s % 24) * 32768 + 16384 * (y / 8192)
               + 8192 * h + y % 8192;
    }

    inline size_t page24ScaleBytes(size_t kBlocks)
    {
        size_t const kTiles = kBlocks / kScaleBlocks;
        return (2 * ((kTiles + 23) / 24) + 2) * kPageBytes;
    }

    // src is [4096][kBlocks], K block fastest; returns page24ScaleBytes(kBlocks) bytes.
    inline std::vector<uint8_t> placeScales(std::vector<uint8_t> const& src, size_t kBlocks)
    {
        if(kBlocks / kScaleBlocks > kPage24MaxKTiles)
            throw std::invalid_argument("placed scales need K / 128 <= 768; got K blocks="
                                        + std::to_string(kBlocks));
        auto const           dense = shuffleScales(src, kPlacedScaleRows, kBlocks);
        std::vector<uint8_t> dst(page24ScaleBytes(kBlocks), 0);
        for(size_t x = 0; x < dense.size(); ++x)
            for(size_t h = 0; h < 2; ++h)
                dst[page24ScaleOffset(x, h)] = dense[x];
        return dst;
    }
} // namespace mx_tile128x4
