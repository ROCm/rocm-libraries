// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>
#include <stdexcept>

#include "detail/checked_arithmetic.hpp"
#include "parallel_chunks.hpp"

namespace roc::hostnumerics::amd_gpu_layout {
namespace {
using detail::parallelForChunks;
using ::roc::hostnumerics::detail::checkedAdd;
using ::roc::hostnumerics::detail::checkedMultiply;

constexpr std::string_view sizeOverflow = "MX scale storage size overflows size_t.";

size_t roundUp(size_t value, size_t multiple) {
    const size_t remainder = value % multiple;
    return remainder == 0 ? value : checkedAdd(value, multiple - remainder, sizeOverflow);
}

// Number of consecutive fast-dimension scales in each gfx1250 tile.
size_t gfx1250TileSize(size_t blockSize) {
    if (blockSize != 16 && blockSize != 32)
        throw std::runtime_error("gfx1250 MX scale layout requires a block size of 16 or 32.");
    return 128 / blockSize;
}

size_t physicalByteCountFor(std::array<size_t, 2> naturalShape, size_t blockSize,
                            MxScaleStorageLayout layout, size_t naturalByteCount) {
    switch (layout) {
        case MxScaleStorageLayout::Natural:
            return naturalByteCount;
        case MxScaleStorageLayout::Gfx950:
            return checkedMultiply(roundUp(naturalShape[0], 32), roundUp(naturalShape[1], 8),
                                   sizeOverflow);
        case MxScaleStorageLayout::Gfx1250:
            return checkedMultiply(naturalShape[0],
                                   roundUp(naturalShape[1], gfx1250TileSize(blockSize)),
                                   sizeOverflow);
    }
    throw std::runtime_error("Invalid MX scale storage layout.");
}

// gfx950 stores each zero-padded 32 x 8 tile of [row, column] scales
// contiguously, column tiles fastest. Within a tile, with
// row = 16 * rowHalf + rowLane and column = 4 * columnHalf + columnLane, the
// scales are ordered [columnLane, rowLane, columnHalf, rowHalf], rowHalf fastest.
void copyGfx950Scales(const std::byte* natural, std::byte* physical, size_t rows, size_t columns) {
    const size_t columnTiles = roundUp(columns, 8) / 8;
    parallelForChunks(rows, rows * columns, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            const size_t rowTile = row / 32;
            const size_t rowHalf = row / 16 % 2;
            const size_t rowLane = row % 16;
            for (size_t column = 0; column < columns; ++column) {
                const size_t tile = rowTile * columnTiles + column / 8;
                const size_t columnHalf = column / 4 % 2;
                const size_t columnLane = column % 4;
                const size_t destination =
                    (((tile * 4 + columnLane) * 16 + rowLane) * 2 + columnHalf) * 2 + rowHalf;
                physical[destination] = natural[row * columns + column];
            }
        }
    });
}

void copyGfx1250Scales(const std::byte* natural, std::byte* physical, size_t slow, size_t fast,
                       size_t tileSize) {
    parallelForChunks(slow, slow * fast, [&](size_t begin, size_t end) {
        for (size_t slowIndex = begin; slowIndex < end; ++slowIndex)
            for (size_t fastIndex = 0; fastIndex < fast; ++fastIndex)
                physical[(fastIndex / tileSize * slow + slowIndex) * tileSize +
                         fastIndex % tileSize] = natural[slowIndex * fast + fastIndex];
    });
}
}  // namespace

MxScaleStorageLayout mxScaleStorageLayoutForArchitectureName(std::string_view architectureName) {
    const auto matchesArchitectureToken = [architectureName](std::string_view token) {
        return architectureName == token ||
               (architectureName.size() > token.size() &&
                architectureName.compare(0, token.size(), token) == 0 &&
                architectureName[token.size()] == ':');
    };

    if (matchesArchitectureToken("gfx950")) return MxScaleStorageLayout::Gfx950;
    if (matchesArchitectureToken("gfx1250") || matchesArchitectureToken("gfx1250-strict"))
        return MxScaleStorageLayout::Gfx1250;
    return MxScaleStorageLayout::Natural;
}

MxScaleStoragePlan::MxScaleStoragePlan(std::array<size_t, 2> naturalShape, size_t blockSize,
                                       MxScaleStorageLayout layout)
    : m_layout(layout), m_naturalShape(naturalShape), m_blockSize(blockSize) {
    if (blockSize == 0) throw std::runtime_error("MX scale block size must be nonzero.");
    m_naturalByteCount = checkedMultiply(naturalShape[0], naturalShape[1], sizeOverflow);
    m_physicalByteCount = physicalByteCountFor(naturalShape, blockSize, layout, m_naturalByteCount);
}

std::vector<std::byte> copyMxScaleStorageToPhysicalLayout(const std::byte* naturalScaleStorage,
                                                          size_t naturalScaleByteCount,
                                                          const MxScaleStoragePlan& plan) {
    if (naturalScaleStorage == nullptr && naturalScaleByteCount != 0)
        throw std::runtime_error("MX scale layout input storage is null.");
    if (naturalScaleByteCount != plan.naturalByteCount())
        throw std::runtime_error("MX scale storage does not match its plan.");

    std::vector<std::byte> physical(plan.physicalByteCount());
    const auto [slow, fast] = plan.naturalShape();
    switch (plan.layout()) {
        case MxScaleStorageLayout::Natural:
            std::copy_n(naturalScaleStorage, naturalScaleByteCount, physical.begin());
            break;
        case MxScaleStorageLayout::Gfx950:
            copyGfx950Scales(naturalScaleStorage, physical.data(), slow, fast);
            break;
        case MxScaleStorageLayout::Gfx1250:
            copyGfx1250Scales(naturalScaleStorage, physical.data(), slow, fast,
                              gfx1250TileSize(plan.blockSize()));
            break;
    }
    return physical;
}

}  // namespace roc::hostnumerics::amd_gpu_layout
