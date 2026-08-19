// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstring>
#include <limits>
#include <roc/host_validation/amd_gpu_layout/mx.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mx_threading.hpp"

namespace roc::host_validation::amd_gpu_layout::detail {
namespace {

struct DimensionShufflePlan {
    bool identity = false;
    std::vector<size_t> sizes;
    std::vector<size_t> destinationStrides;
    std::vector<size_t> sourceStrides;
};

struct GFX950ScalePlan {
    size_t rowCount = 0;
    size_t columnCount = 0;
    size_t paddedRowCount = 0;
    size_t paddedColumnCount = 0;
    size_t outputElementCount = 0;
    DimensionShufflePlan shuffle;
};

struct GFX1250ScalePlan {
    size_t slowDimension = 0;
    size_t fastDimension = 0;
    size_t dimK = 0;
    size_t outputElementCount = 0;
};

size_t checkedMultiply(size_t left, size_t right, const std::string& context) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        throw std::overflow_error(context + ": size multiplication overflow");
    return left * right;
}

size_t checkedAdd(size_t left, size_t right, const std::string& context) {
    if (right > std::numeric_limits<size_t>::max() - left)
        throw std::overflow_error(context + ": size addition overflow");
    return left + right;
}

size_t roundUp(size_t value, size_t multiple) {
    if (multiple == 0) throw std::runtime_error("roundUp: multiple must be non-zero");
    const size_t remainder = value % multiple;
    return remainder == 0 ? value : checkedAdd(value, multiple - remainder, "roundUp");
}

size_t checkedProduct(const std::vector<size_t>& values) {
    size_t result = 1;
    for (const size_t value : values) result = checkedMultiply(result, value, "product");
    return result;
}

std::vector<size_t> computeStrides(const std::vector<size_t>& sizes) {
    std::vector<size_t> strides(sizes.size());
    if (sizes.empty()) return strides;

    strides[0] = 1;
    for (size_t index = 1; index < sizes.size(); ++index)
        strides[index] = checkedMultiply(strides[index - 1], sizes[index - 1], "computeStrides");
    return strides;
}

std::vector<size_t> computeShuffledStrides(const std::vector<size_t>& sizes,
                                           const std::vector<size_t>& dimensionOrder) {
    if (dimensionOrder.size() != sizes.size())
        throw std::runtime_error(
            "computeShuffledStrides: dimension order must contain every dimension");

    std::vector<size_t> strides(sizes.size(), 0);
    std::vector<bool> seen(sizes.size(), false);
    size_t stride = 1;
    for (const size_t index : dimensionOrder) {
        if (index >= sizes.size() || seen[index])
            throw std::runtime_error(
                "computeShuffledStrides: dimension order must be a permutation");
        seen[index] = true;
        strides[index] = stride;
        stride = checkedMultiply(stride, sizes[index], "computeShuffledStrides");
    }
    return strides;
}

size_t maximumOffset(const std::vector<size_t>& sizes, const std::vector<size_t>& strides,
                     const std::string& context) {
    size_t offset = 0;
    for (size_t dimension = 0; dimension < sizes.size(); ++dimension) {
        if (sizes[dimension] == 0) return 0;
        const size_t contribution =
            checkedMultiply(sizes[dimension] - 1, strides[dimension], context);
        offset = checkedAdd(offset, contribution, context);
    }
    return offset;
}

size_t validateShuffle(size_t inputElementCount, const DimensionShufflePlan& plan) {
    if (plan.sizes.size() != plan.destinationStrides.size() ||
        plan.sizes.size() != plan.sourceStrides.size())
        throw std::runtime_error("shuffleDims: size/stride dimension mismatch");
    if (plan.sizes.size() < 2) throw std::runtime_error("shuffleDims: need at least 2 dimensions");

    const size_t totalElements = checkedProduct(plan.sizes);
    if (inputElementCount != totalElements) {
        std::ostringstream message;
        message << "shuffleDims: input size " << inputElementCount << " doesn't match expected "
                << totalElements;
        throw std::runtime_error(message.str());
    }

    if (totalElements != 0 &&
        (maximumOffset(plan.sizes, plan.sourceStrides, "shuffleDims source") >= inputElementCount ||
         maximumOffset(plan.sizes, plan.destinationStrides, "shuffleDims destination") >=
             inputElementCount))
        throw std::runtime_error("shuffleDims: strides address outside the storage");
    return totalElements;
}

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

GFX950ScalePlan makeGFX950ScalePlan(size_t inputElementCount, const std::vector<size_t>& sizes) {
    if (sizes.size() != 2) {
        std::ostringstream message;
        message << "preSwizzleAITER: sizes must have 2 elements, got " << sizes.size();
        throw std::runtime_error(message.str());
    }

    GFX950ScalePlan plan;
    plan.rowCount = sizes[0];
    plan.columnCount = sizes[1];
    const size_t inputElements =
        checkedMultiply(plan.rowCount, plan.columnCount, "preSwizzleScalesGFX950");
    if (inputElements != inputElementCount) {
        std::ostringstream message;
        message << "preSwizzleAITER: input size " << inputElementCount
                << " doesn't match sizes product " << inputElements;
        throw std::runtime_error(message.str());
    }

    plan.paddedRowCount = roundUp(plan.rowCount, 32);
    plan.paddedColumnCount = roundUp(plan.columnCount, 8);
    plan.outputElementCount =
        checkedMultiply(plan.paddedRowCount, plan.paddedColumnCount, "preSwizzleScalesGFX950");
    plan.shuffle.sizes = {plan.paddedRowCount / 32, 2, 16, plan.paddedColumnCount / 8, 2, 4};
    plan.shuffle.sourceStrides = {
        checkedMultiply(32, plan.paddedColumnCount, "preSwizzleScalesGFX950 strides"),
        checkedMultiply(16, plan.paddedColumnCount, "preSwizzleScalesGFX950 strides"),
        plan.paddedColumnCount,
        8,
        4,
        1};
    plan.shuffle.destinationStrides =
        computeShuffledStrides(plan.shuffle.sizes, {1, 4, 2, 5, 3, 0});
    validateShuffle(plan.outputElementCount, plan.shuffle);
    return plan;
}

GFX1250ScalePlan makeGFX1250ScalePlan(size_t inputElementCount, size_t slowDimension,
                                      size_t fastDimension, size_t mxBlock) {
    if (mxBlock != 16 && mxBlock != 32)
        throw std::runtime_error("preSwizzleScalesGFX1250: mxBlock must be 16 or 32");

    const size_t expectedInput =
        checkedMultiply(slowDimension, fastDimension, "preSwizzleScalesGFX1250 input");
    if (expectedInput != inputElementCount) {
        std::ostringstream message;
        message << "preSwizzleScalesGFX1250: input size " << inputElementCount
                << " doesn't match slowDim*fastDim = " << expectedInput;
        throw std::runtime_error(message.str());
    }

    GFX1250ScalePlan plan;
    plan.slowDimension = slowDimension;
    plan.fastDimension = fastDimension;
    plan.dimK = 128 / mxBlock;
    const size_t paddedFastDimension = roundUp(fastDimension, plan.dimK);
    plan.outputElementCount =
        checkedMultiply(slowDimension, paddedFastDimension, "preSwizzleScalesGFX1250 output");
    return plan;
}

void validateByteStorage(size_t inputElementCount, size_t outputElementCount, size_t elementSize,
                         const std::string& context) {
    checkedMultiply(inputElementCount, elementSize, context + " input bytes");
    checkedMultiply(outputElementCount, elementSize, context + " output bytes");
}

void copyElement(const std::byte* input, size_t sourceIndex, std::byte* output,
                 size_t destinationIndex, size_t elementSize) {
    std::memcpy(output + destinationIndex * elementSize, input + sourceIndex * elementSize,
                elementSize);
}

template <typename Function>
void forEachShuffledElement(const DimensionShufflePlan& plan, size_t totalElements,
                            Function function) {
    parallelForChunks(totalElements, totalElements, [&](size_t begin, size_t end) {
        std::vector<size_t> coordinate(plan.sizes.size());
        for (size_t coordinateNumber = begin; coordinateNumber < end; ++coordinateNumber) {
            size_t remaining = coordinateNumber;
            for (size_t dimension = 0; dimension < plan.sizes.size(); ++dimension) {
                coordinate[dimension] = remaining % plan.sizes[dimension];
                remaining /= plan.sizes[dimension];
            }

            size_t sourceIndex = 0;
            size_t destinationIndex = 0;
            for (size_t dimension = 0; dimension < plan.sizes.size(); ++dimension) {
                sourceIndex += coordinate[dimension] * plan.sourceStrides[dimension];
                destinationIndex += coordinate[dimension] * plan.destinationStrides[dimension];
            }
            function(sourceIndex, destinationIndex);
        }
    });
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

size_t preSwizzleScalesGFX950Bytes(const std::byte* input, size_t inputElementCount,
                                   size_t elementSize, const std::vector<size_t>& sizes,
                                   std::byte* output) {
    const GFX950ScalePlan plan = makeGFX950ScalePlan(inputElementCount, sizes);
    validateByteStorage(inputElementCount, plan.outputElementCount, elementSize,
                        "preSwizzleScalesGFX950");
    if (output == nullptr) return plan.outputElementCount;

    forEachShuffledElement(
        plan.shuffle, plan.outputElementCount,
        [&](size_t paddedSourceIndex, size_t destinationIndex) {
            const size_t sourceRow = paddedSourceIndex / plan.paddedColumnCount;
            const size_t sourceColumn = paddedSourceIndex % plan.paddedColumnCount;
            if (sourceRow < plan.rowCount && sourceColumn < plan.columnCount) {
                const size_t sourceIndex = sourceRow * plan.columnCount + sourceColumn;
                copyElement(input, sourceIndex, output, destinationIndex, elementSize);
            }
        });
    return plan.outputElementCount;
}

size_t preSwizzleScalesGFX1250Bytes(const std::byte* input, size_t inputElementCount,
                                    size_t elementSize, size_t slowDimension, size_t fastDimension,
                                    size_t mxBlock, std::byte* output) {
    const GFX1250ScalePlan plan =
        makeGFX1250ScalePlan(inputElementCount, slowDimension, fastDimension, mxBlock);
    validateByteStorage(inputElementCount, plan.outputElementCount, elementSize,
                        "preSwizzleScalesGFX1250");
    if (output == nullptr) return plan.outputElementCount;

    const size_t copyGroupCount = plan.outputElementCount / plan.dimK;
    parallelForChunks(copyGroupCount, plan.outputElementCount, [&](size_t begin, size_t end) {
        for (size_t copyGroup = begin; copyGroup < end; ++copyGroup) {
            const size_t slowIndex = copyGroup % plan.slowDimension;
            const size_t tile = copyGroup / plan.slowDimension;
            const size_t outputBase = copyGroup * plan.dimK;
            const size_t sourceFastBase = tile * plan.dimK;
            for (size_t elementInTile = 0; elementInTile < plan.dimK; ++elementInTile) {
                const size_t sourceFast = sourceFastBase + elementInTile;
                if (sourceFast < plan.fastDimension) {
                    const size_t sourceIndex = slowIndex * plan.fastDimension + sourceFast;
                    copyElement(input, sourceIndex, output, outputBase + elementInTile,
                                elementSize);
                }
            }
        }
    });
    return plan.outputElementCount;
}

}  // namespace roc::host_validation::amd_gpu_layout::detail
