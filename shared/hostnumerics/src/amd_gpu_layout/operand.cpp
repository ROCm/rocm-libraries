// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstring>
#include <roc/hostnumerics/amd_gpu_layout/operand.hpp>
#include <sstream>
#include <stdexcept>

#include "detail/checked_arithmetic.hpp"
#include "detail/mx_layout_common.hpp"

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

DimensionShufflePlan makeOperandPlan(size_t elementCount, OperandExtents extents,
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

    DimensionShufflePlan plan;
    if (!swizzle && !tile) {
        plan.identity = true;
        return plan;
    }
    if (tile && (tile->k == 0 || tile->mn == 0))
        throw std::runtime_error("Operand tile extents must be nonzero.");

    std::vector<size_t> dimensionOrder;
    if (!swizzle) {
        requireDivisible(extents.k, tile->k, "Operand K extent must be divisible by the tile.");
        requireDivisible(extents.mn, tile->mn, "Operand MN extent must be divisible by the tile.");
        plan.sizes = {tile->k, extents.k / tile->k, tile->mn, extents.mn / tile->mn};
        dimensionOrder = {0, 2, 1, 3};
    } else {
        const SwizzleFactors factors = swizzleFactors(*swizzle);
        if (!tile) {
            requireDivisible(extents.k, swizzle->tileK,
                             "Operand K extent must be divisible by the swizzle tile.");
            requireDivisible(extents.mn, swizzle->tileMN,
                             "Operand MN extent must be divisible by the swizzle tile.");
            plan.sizes = {factors.vgprIndex,      factors.vgprBlock,
                          factors.simdBlock,      extents.k / swizzle->tileK,
                          lanesPerSimd,           factors.simdIndexIndex,
                          factors.simdIndexBlock, extents.mn / swizzle->tileMN};
            if (swizzle->tileMN == 64)
                dimensionOrder = {6, 1, 2, 3, 4, 5, 0, 7};
            else if (swizzle->subTileK == 4)
                dimensionOrder = {6, 2, 1, 3, 4, 5, 0, 7};
            else
                dimensionOrder = {1, 2, 0, 3, 4, 5, 6, 7};
        } else {
            requireDivisible(tile->k, swizzle->tileK,
                             "Operand tile K extent must be divisible by the swizzle tile.");
            requireDivisible(tile->mn, swizzle->tileMN,
                             "Operand tile MN extent must be divisible by the swizzle tile.");
            requireDivisible(extents.k, tile->k, "Operand K extent must be divisible by the tile.");
            requireDivisible(extents.mn, tile->mn,
                             "Operand MN extent must be divisible by the tile.");
            plan.sizes = {factors.vgprIndex,          factors.vgprBlock,
                          factors.simdBlock,          tile->k / swizzle->tileK,
                          extents.k / tile->k,        lanesPerSimd,
                          factors.simdIndexIndex,     factors.simdIndexBlock,
                          tile->mn / swizzle->tileMN, extents.mn / tile->mn};
            if (swizzle->tileMN == 64)
                dimensionOrder = {7, 1, 2, 3, 5, 6, 0, 8, 4, 9};
            else if (swizzle->subTileK == 4)
                dimensionOrder = {7, 2, 1, 3, 5, 6, 0, 8, 4, 9};
            else
                dimensionOrder = {1, 2, 0, 3, 5, 6, 7, 8, 4, 9};
        }
    }

    plan.sourceStrides = computeStrides(plan.sizes);
    plan.destinationStrides = computeShuffledStrides(plan.sizes, dimensionOrder);
    return plan;
}
}  // namespace

void copyOperandBytes(const std::byte* operand, size_t elementCount, size_t elementSize,
                      OperandExtents extents, const std::optional<OperandSwizzle>& swizzle,
                      const std::optional<OperandTile>& tile, std::byte* output) {
    const DimensionShufflePlan plan = makeOperandPlan(elementCount, extents, swizzle, tile);
    validateByteStorage(elementCount, elementCount, elementSize, "Operand layout");
    if (output == nullptr) return;

    if (plan.identity) {
        if (elementCount != 0) std::memcpy(output, operand, elementCount * elementSize);
        return;
    }

    const size_t totalElements = validateShuffle(elementCount, plan);
    forEachShuffledElement(plan, totalElements, [&](size_t sourceIndex, size_t destinationIndex) {
        copyElement(operand, sourceIndex, output, destinationIndex, elementSize);
    });
}
}  // namespace roc::hostnumerics::amd_gpu_layout::detail
