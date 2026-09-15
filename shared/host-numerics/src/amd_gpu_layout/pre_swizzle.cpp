// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/pre_swizzle.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "detail/mx_layout_common.hpp"

namespace roc::host_numerics::amd_gpu_layout::detail {
namespace {
DimensionShufflePlan makePreSwizzlePlan(size_t inputElementCount, const std::vector<size_t>& sizes,
                                        const std::vector<size_t>& preSwizzleSize,
                                        const std::vector<size_t>& preTileSize) {
    if (!preSwizzleSize.empty() && preSwizzleSize.size() != 3) {
        std::ostringstream message;
        message << "preSwizzle: preSwizzleSize must have 3 elements, got " << preSwizzleSize.size();
        throw std::runtime_error(message.str());
    }
    if (!preTileSize.empty() && preTileSize.size() != 2) {
        std::ostringstream message;
        message << "preSwizzle: preTileSize must have 2 elements, got " << preTileSize.size();
        throw std::runtime_error(message.str());
    }
    if (sizes.size() != 2) {
        std::ostringstream message;
        message << "preSwizzle: Batch dimension not yet supported. sizes.size()=" << sizes.size();
        throw std::runtime_error(message.str());
    }

    const size_t totalElements = checkedProduct(sizes);
    if (totalElements != inputElementCount) {
        std::ostringstream message;
        message << "preSwizzle: input size " << inputElementCount << " doesn't match sizes product "
                << totalElements;
        throw std::runtime_error(message.str());
    }

    DimensionShufflePlan plan;
    if (preSwizzleSize.empty() && preTileSize.empty()) {
        plan.identity = true;
        return plan;
    }

    if (!preSwizzleSize.empty()) {
        const size_t tileMN = preSwizzleSize[0];
        const size_t tileK = preSwizzleSize[1];
        const size_t subTileK = preSwizzleSize[2];
        if (tileMN != 64 && tileMN != 32)
            throw std::runtime_error("preSwizzle: tileMN must be 32 or 64");
        if (tileK == 0 || tileK % 4 != 0)
            throw std::runtime_error("preSwizzle: tileK must be a non-zero multiple of 4");
        if (subTileK == 0) throw std::runtime_error("preSwizzle: subTileK must be non-zero");
        if (tileMN == 32 && subTileK != 2 && subTileK != 4)
            throw std::runtime_error("preSwizzle: tileMN 32 supports subTileK 2 or 4");
    }
    if (!preTileSize.empty() && (preTileSize[0] == 0 || preTileSize[1] == 0))
        throw std::runtime_error("preSwizzle: pre-tile dimensions must be non-zero");

    std::vector<size_t> sourceSizes;
    std::vector<size_t> dimensionOrder;

    if (!preSwizzleSize.empty() && preTileSize.empty()) {
        const size_t tileMN = preSwizzleSize[0];
        const size_t tileK = preSwizzleSize[1];
        const size_t subTileK = preSwizzleSize[2];

        if (sizes[0] % tileK != 0 || sizes[1] % tileMN != 0)
            throw std::runtime_error(
                "preSwizzle: tensor dimensions must be divisible by the swizzle tile");

        const size_t lanesPerSIMD = 16;
        const size_t SIMDsPerWave = 4;
        const size_t SIMDIndex = tileMN / lanesPerSIMD;
        const size_t SIMDBlock = SIMDsPerWave / SIMDIndex;
        const size_t VGPRIndex = std::min(SIMDIndex, subTileK);
        const size_t VGPRBlock = tileK / SIMDBlock / VGPRIndex;
        const size_t SIMDIndexBlock = VGPRIndex;
        const size_t SIMDIndexIndex = SIMDIndex / SIMDIndexBlock;

        if (VGPRIndex * VGPRBlock * SIMDBlock != tileK)
            throw std::runtime_error("preSwizzle: nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK");
        if (lanesPerSIMD * SIMDIndexIndex * SIMDIndexBlock != tileMN)
            throw std::runtime_error(
                "preSwizzle: nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN");

        sourceSizes = {VGPRIndex,    VGPRBlock,      SIMDBlock,      sizes[0] / tileK,
                       lanesPerSIMD, SIMDIndexIndex, SIMDIndexBlock, sizes[1] / tileMN};

        if (tileMN == 64)
            dimensionOrder = {6, 1, 2, 3, 4, 5, 0, 7};
        else if (subTileK == 4)
            dimensionOrder = {6, 2, 1, 3, 4, 5, 0, 7};
        else
            dimensionOrder = {1, 2, 0, 3, 4, 5, 6, 7};
    } else if (preSwizzleSize.empty()) {
        if (sizes[0] % preTileSize[0] != 0 || sizes[1] % preTileSize[1] != 0)
            throw std::runtime_error(
                "preSwizzle: tensor dimensions must be divisible by the pre-tile");
        sourceSizes = {preTileSize[0], sizes[0] / preTileSize[0], preTileSize[1],
                       sizes[1] / preTileSize[1]};
        dimensionOrder = {0, 2, 1, 3};
    } else {
        const size_t tileMN = preSwizzleSize[0];
        const size_t tileK = preSwizzleSize[1];
        const size_t subTileK = preSwizzleSize[2];
        const size_t preTileK = preTileSize[0];
        const size_t preTileMN = preTileSize[1];
        const size_t lanesPerSIMD = 16;
        const size_t SIMDsPerWave = 4;
        const size_t SIMDIndex = tileMN / lanesPerSIMD;
        const size_t SIMDBlock = SIMDsPerWave / SIMDIndex;
        const size_t VGPRIndex = std::min(SIMDIndex, subTileK);
        const size_t VGPRBlock = tileK / SIMDBlock / VGPRIndex;
        const size_t SIMDIndexBlock = VGPRIndex;
        const size_t SIMDIndexIndex = SIMDIndex / SIMDIndexBlock;

        if (preTileK % tileK != 0 || preTileMN % tileMN != 0)
            throw std::runtime_error(
                "preSwizzle: pre-tile dimensions must be divisible by the swizzle tile");
        if (sizes[0] % preTileK != 0 || sizes[1] % preTileMN != 0)
            throw std::runtime_error(
                "preSwizzle: tensor dimensions must be divisible by the pre-tile");
        if (VGPRIndex * VGPRBlock * SIMDBlock != tileK)
            throw std::runtime_error("preSwizzle: nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK");
        if (lanesPerSIMD * SIMDIndexIndex * SIMDIndexBlock != tileMN)
            throw std::runtime_error(
                "preSwizzle: nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN");

        sourceSizes = {VGPRIndex,           VGPRBlock,           SIMDBlock,      preTileK / tileK,
                       sizes[0] / preTileK, lanesPerSIMD,        SIMDIndexIndex, SIMDIndexBlock,
                       preTileMN / tileMN,  sizes[1] / preTileMN};

        if (tileMN == 64)
            dimensionOrder = {7, 1, 2, 3, 5, 6, 0, 8, 4, 9};
        else if (subTileK == 4)
            dimensionOrder = {7, 2, 1, 3, 5, 6, 0, 8, 4, 9};
        else
            dimensionOrder = {1, 2, 0, 3, 5, 6, 7, 8, 4, 9};
    }

    const size_t sourceElements = checkedProduct(sourceSizes);
    if (sourceElements != totalElements) {
        std::ostringstream message;
        message << "PreSwizzle size mismatch: product(srcSizes)=" << sourceElements
                << " != product(sizes)=" << totalElements;
        throw std::runtime_error(message.str());
    }
    if (sourceSizes.empty()) throw std::runtime_error("PreSwizzle source size not populated.");
    if (dimensionOrder.empty())
        throw std::runtime_error("PreSwizzle permutation order not populated.");

    plan.sizes = std::move(sourceSizes);
    plan.sourceStrides = computeStrides(plan.sizes);
    plan.destinationStrides = computeShuffledStrides(plan.sizes, dimensionOrder);
    return plan;
}
}  // namespace

size_t preSwizzleBytes(const std::byte* input, size_t inputElementCount, size_t elementSize,
                       const std::vector<size_t>& sizes, const std::vector<size_t>& preSwizzleSize,
                       const std::vector<size_t>& preTileSize, std::byte* output) {
    const DimensionShufflePlan plan =
        makePreSwizzlePlan(inputElementCount, sizes, preSwizzleSize, preTileSize);
    validateByteStorage(inputElementCount, inputElementCount, elementSize, "preSwizzle");
    if (output == nullptr) return inputElementCount;

    if (plan.identity) {
        const size_t byteCount = inputElementCount * elementSize;
        if (byteCount != 0) std::memcpy(output, input, byteCount);
        return inputElementCount;
    }

    const size_t totalElements = validateShuffle(inputElementCount, plan);
    forEachShuffledElement(plan, totalElements, [&](size_t sourceIndex, size_t destinationIndex) {
        copyElement(input, sourceIndex, output, destinationIndex, elementSize);
    });
    return totalElements;
}
}  // namespace roc::host_numerics::amd_gpu_layout::detail
