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
// lanes index, in source order along reduction then non-reduction.
struct SwizzleFactors {
    size_t vgprIndex = 0;
    size_t vgprBlock = 0;
    size_t simdBlock = 0;
    size_t simdIndexIndex = 0;
    size_t simdIndexBlock = 0;
};

SwizzleFactors swizzleFactors(const OperandSwizzle& swizzle) {
    if (swizzle.tileNonReduction != 64 && swizzle.tileNonReduction != 32)
        throw std::runtime_error("Operand swizzle tileNonReduction must be 32 or 64.");
    if (swizzle.tileReduction == 0 || swizzle.tileReduction % 4 != 0)
        throw std::runtime_error("Operand swizzle tileReduction must be a nonzero multiple of 4.");
    if (swizzle.subTileReduction == 0)
        throw std::runtime_error("Operand swizzle subTileReduction must be nonzero.");
    if (swizzle.tileNonReduction == 32 && swizzle.subTileReduction != 2 &&
        swizzle.subTileReduction != 4)
        throw std::runtime_error(
            "Operand swizzle with tileNonReduction 32 requires subTileReduction 2 or 4.");

    const size_t simdIndex = swizzle.tileNonReduction / lanesPerSimd;
    SwizzleFactors factors;
    factors.simdBlock = simdsPerWave / simdIndex;
    factors.vgprIndex = std::min(simdIndex, swizzle.subTileReduction);
    factors.vgprBlock = swizzle.tileReduction / factors.simdBlock / factors.vgprIndex;
    factors.simdIndexBlock = factors.vgprIndex;
    factors.simdIndexIndex = simdIndex / factors.simdIndexBlock;
    if (factors.vgprIndex * factors.vgprBlock * factors.simdBlock != swizzle.tileReduction)
        throw std::runtime_error(
            "Operand swizzle tileReduction does not divide into its VGPR blocks.");
    if (lanesPerSimd * factors.simdIndexIndex * factors.simdIndexBlock != swizzle.tileNonReduction)
        throw std::runtime_error(
            "Operand swizzle tileNonReduction does not divide into its SIMD lanes.");
    return factors;
}

void requireDivisible(size_t extent, size_t tile, const char* message) {
    if (extent % tile != 0) throw std::runtime_error(message);
}

// Factor the contiguous [non-reduction, reduction] input into tile and lane axes. The shapes and
// permutations both list axes from slowest to fastest.
ReshapePermutation operandPermutation(OperandExtents extents,
                                      const std::optional<OperandSwizzle>& swizzle,
                                      const std::optional<OperandTile>& tile) {
    const Shape natural{extents.nonReduction, extents.reduction};
    if (tile && (tile->reduction == 0 || tile->nonReduction == 0))
        throw std::runtime_error("Operand tile extents must be nonzero.");

    if (!swizzle) {
        requireDivisible(extents.reduction, tile->reduction,
                         "Operand reduction extent must be divisible by the tile.");
        requireDivisible(extents.nonReduction, tile->nonReduction,
                         "Operand non-reduction extent must be divisible by the tile.");
        return {natural, natural,
                Shape{extents.nonReduction / tile->nonReduction, tile->nonReduction,
                      extents.reduction / tile->reduction, tile->reduction},
                Permutation{0, 2, 1, 3}};
    }

    const SwizzleFactors factors = swizzleFactors(*swizzle);
    if (!tile) {
        requireDivisible(extents.reduction, swizzle->tileReduction,
                         "Operand reduction extent must be divisible by the swizzle tile.");
        requireDivisible(extents.nonReduction, swizzle->tileNonReduction,
                         "Operand non-reduction extent must be divisible by the swizzle tile.");
        const Shape factorsShape{extents.nonReduction / swizzle->tileNonReduction,
                                 factors.simdIndexBlock,
                                 factors.simdIndexIndex,
                                 lanesPerSimd,
                                 extents.reduction / swizzle->tileReduction,
                                 factors.simdBlock,
                                 factors.vgprBlock,
                                 factors.vgprIndex};
        std::vector<size_t> order;
        if (swizzle->tileNonReduction == 64)
            order = {0, 7, 2, 3, 4, 5, 6, 1};
        else if (swizzle->subTileReduction == 4)
            order = {0, 7, 2, 3, 4, 6, 5, 1};
        else
            order = {0, 1, 2, 3, 4, 7, 5, 6};
        return {natural, natural, factorsShape, Permutation(std::move(order))};
    }

    requireDivisible(tile->reduction, swizzle->tileReduction,
                     "Operand tile reduction extent must be divisible by the swizzle tile.");
    requireDivisible(tile->nonReduction, swizzle->tileNonReduction,
                     "Operand tile non-reduction extent must be divisible by the swizzle tile.");
    requireDivisible(extents.reduction, tile->reduction,
                     "Operand reduction extent must be divisible by the tile.");
    requireDivisible(extents.nonReduction, tile->nonReduction,
                     "Operand non-reduction extent must be divisible by the tile.");
    const Shape factorsShape{extents.nonReduction / tile->nonReduction,
                             tile->nonReduction / swizzle->tileNonReduction,
                             factors.simdIndexBlock,
                             factors.simdIndexIndex,
                             lanesPerSimd,
                             extents.reduction / tile->reduction,
                             tile->reduction / swizzle->tileReduction,
                             factors.simdBlock,
                             factors.vgprBlock,
                             factors.vgprIndex};
    std::vector<size_t> order;
    if (swizzle->tileNonReduction == 64)
        order = {0, 5, 1, 9, 3, 4, 6, 7, 8, 2};
    else if (swizzle->subTileReduction == 4)
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
        extents.reduction, extents.nonReduction, "Operand extents: size multiplication overflow");
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
