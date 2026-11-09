// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstring>
#include <roc/hostnumerics/amd_gpu_layout/operand.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "copy_permutation.hpp"
#include "detail/checked_arithmetic.hpp"

namespace roc::hostnumerics::amd_gpu_layout::detail {
namespace {
constexpr size_t lanesPerSimd = 16;
constexpr size_t simdsPerWave = 4;

// A swizzle tile split into the factors that the kernel's VGPRs, SIMDs and
// lanes index, in source order along K then MN.
struct SwizzleFactors {
    size_t vgprIndex = 0;
    size_t vgprBlock = 0;
    size_t simdBlock = 0;
    size_t simdIndexIndex = 0;
    size_t simdIndexBlock = 0;
};

SwizzleFactors swizzleFactors(const OperandSwizzle& swizzle) {
    if (swizzle.tileMN != 64 && swizzle.tileMN != 32)
        throw std::runtime_error("Operand swizzle tileMN must be 32 or 64.");
    if (swizzle.tileK == 0 || swizzle.tileK % 4 != 0)
        throw std::runtime_error("Operand swizzle tileK must be a nonzero multiple of 4.");
    if (swizzle.subTileK == 0)
        throw std::runtime_error("Operand swizzle subTileK must be nonzero.");
    if (swizzle.tileMN == 32 && swizzle.subTileK != 2 && swizzle.subTileK != 4)
        throw std::runtime_error("Operand swizzle with tileMN 32 requires subTileK 2 or 4.");

    const size_t simdIndex = swizzle.tileMN / lanesPerSimd;
    SwizzleFactors factors;
    factors.simdBlock = simdsPerWave / simdIndex;
    factors.vgprIndex = std::min(simdIndex, swizzle.subTileK);
    factors.vgprBlock = swizzle.tileK / factors.simdBlock / factors.vgprIndex;
    factors.simdIndexBlock = factors.vgprIndex;
    factors.simdIndexIndex = simdIndex / factors.simdIndexBlock;
    if (factors.vgprIndex * factors.vgprBlock * factors.simdBlock != swizzle.tileK)
        throw std::runtime_error("Operand swizzle tileK does not divide into its VGPR blocks.");
    if (lanesPerSimd * factors.simdIndexIndex * factors.simdIndexBlock != swizzle.tileMN)
        throw std::runtime_error("Operand swizzle tileMN does not divide into its SIMD lanes.");
    return factors;
}

void requireDivisible(size_t extent, size_t tile, const char* message) {
    if (extent % tile != 0) throw std::runtime_error(message);
}

// Factor the contiguous [MN, K] input into tile and lane axes. The shapes and
// permutations both list axes from slowest to fastest.
ReshapePermutation operandPermutation(OperandExtents extents,
                                      const std::optional<OperandSwizzle>& swizzle,
                                      const std::optional<OperandTile>& tile) {
    const Shape natural{extents.mn, extents.k};
    if (tile && (tile->k == 0 || tile->mn == 0))
        throw std::runtime_error("Operand tile extents must be nonzero.");

    if (!swizzle) {
        requireDivisible(extents.k, tile->k, "Operand K extent must be divisible by the tile.");
        requireDivisible(extents.mn, tile->mn, "Operand MN extent must be divisible by the tile.");
        return {natural, natural,
                Shape{extents.mn / tile->mn, tile->mn, extents.k / tile->k, tile->k},
                Permutation{0, 2, 1, 3}};
    }

    const SwizzleFactors factors = swizzleFactors(*swizzle);
    if (!tile) {
        requireDivisible(extents.k, swizzle->tileK,
                         "Operand K extent must be divisible by the swizzle tile.");
        requireDivisible(extents.mn, swizzle->tileMN,
                         "Operand MN extent must be divisible by the swizzle tile.");
        const Shape factorsShape{extents.mn / swizzle->tileMN,
                                 factors.simdIndexBlock,
                                 factors.simdIndexIndex,
                                 lanesPerSimd,
                                 extents.k / swizzle->tileK,
                                 factors.simdBlock,
                                 factors.vgprBlock,
                                 factors.vgprIndex};
        std::vector<size_t> order;
        if (swizzle->tileMN == 64)
            order = {0, 7, 2, 3, 4, 5, 6, 1};
        else if (swizzle->subTileK == 4)
            order = {0, 7, 2, 3, 4, 6, 5, 1};
        else
            order = {0, 1, 2, 3, 4, 7, 5, 6};
        return {natural, natural, factorsShape, Permutation(std::move(order))};
    }

    requireDivisible(tile->k, swizzle->tileK,
                     "Operand tile K extent must be divisible by the swizzle tile.");
    requireDivisible(tile->mn, swizzle->tileMN,
                     "Operand tile MN extent must be divisible by the swizzle tile.");
    requireDivisible(extents.k, tile->k, "Operand K extent must be divisible by the tile.");
    requireDivisible(extents.mn, tile->mn, "Operand MN extent must be divisible by the tile.");
    const Shape factorsShape{extents.mn / tile->mn,
                             tile->mn / swizzle->tileMN,
                             factors.simdIndexBlock,
                             factors.simdIndexIndex,
                             lanesPerSimd,
                             extents.k / tile->k,
                             tile->k / swizzle->tileK,
                             factors.simdBlock,
                             factors.vgprBlock,
                             factors.vgprIndex};
    std::vector<size_t> order;
    if (swizzle->tileMN == 64)
        order = {0, 5, 1, 9, 3, 4, 6, 7, 8, 2};
    else if (swizzle->subTileK == 4)
        order = {0, 5, 1, 9, 3, 4, 6, 8, 7, 2};
    else
        order = {0, 5, 1, 2, 3, 4, 6, 9, 7, 8};
    return {natural, natural, factorsShape, Permutation(std::move(order))};
}
}  // namespace

void copyOperandBytes(const std::byte* operand, std::byte* output, size_t elementCount,
                      size_t elementSize, OperandExtents extents,
                      const std::optional<OperandSwizzle>& swizzle,
                      const std::optional<OperandTile>& tile) {
    const size_t expectedElementCount = ::roc::hostnumerics::detail::checkedMultiply(
        extents.k, extents.mn, "Operand extents: size multiplication overflow");
    if (elementCount != expectedElementCount) {
        std::ostringstream message;
        message << "Operand holds " << elementCount << " elements, but its extents require "
                << expectedElementCount << ".";
        throw std::runtime_error(message.str());
    }

    if (!swizzle && !tile) {
        if (elementCount != 0) std::memcpy(output, operand, elementCount * elementSize);
        return;
    }

    const auto transform = operandPermutation(extents, swizzle, tile);
    copyPermutationBytes(operand, output, elementSize, transform);
}

}  // namespace roc::hostnumerics::amd_gpu_layout::detail
