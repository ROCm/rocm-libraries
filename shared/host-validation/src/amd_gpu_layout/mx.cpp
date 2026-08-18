// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <limits>
#include <roc/host_validation/amd_gpu_layout/mx.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace roc::host_validation::amd_gpu_layout {
namespace detail {
namespace {

size_t checkedProduct(std::vector<size_t> const& values) {
    size_t result = 1;
    for (const size_t value : values) result = checkedMultiply(result, value, "product");
    return result;
}

}  // namespace

size_t checkedMultiply(size_t left, size_t right, std::string_view context) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        throw std::overflow_error(std::string(context) + ": size multiplication overflow");
    return left * right;
}

size_t checkedAdd(size_t left, size_t right, std::string_view context) {
    if (right > std::numeric_limits<size_t>::max() - left)
        throw std::overflow_error(std::string(context) + ": size addition overflow");
    return left + right;
}

size_t roundUp(size_t value, size_t multiple) {
    if (multiple == 0) throw std::runtime_error("roundUp: multiple must be non-zero");
    const size_t remainder = value % multiple;
    return remainder == 0 ? value : checkedAdd(value, multiple - remainder, "roundUp");
}

int operationThreadCount(size_t workItemCount, size_t minimumWorkItemsPerThread,
                         int defaultMaximumThreadCount) {
#ifdef _OPENMP
    if (workItemCount == 0 || omp_in_parallel()) return 1;

    const int runtimeMaximum = std::max(1, omp_get_max_threads());
    const char* configuredThreadCount = std::getenv("OMP_NUM_THREADS");
    const int maximum = configuredThreadCount != nullptr && configuredThreadCount[0] != '\0'
                            ? runtimeMaximum
                            : std::min(runtimeMaximum, defaultMaximumThreadCount);
    const size_t usefulThreadCount = std::max(
        size_t{1}, workItemCount / minimumWorkItemsPerThread +
                       static_cast<size_t>(workItemCount % minimumWorkItemsPerThread != 0));
    return static_cast<int>(std::min(usefulThreadCount, static_cast<size_t>(maximum)));
#else
    (void)workItemCount;
    (void)minimumWorkItemsPerThread;
    (void)defaultMaximumThreadCount;
    return 1;
#endif
}

std::vector<size_t> computeStrides(std::vector<size_t> const& sizes) {
    std::vector<size_t> strides(sizes.size());
    if (sizes.empty()) return strides;

    strides[0] = 1;
    for (size_t index = 1; index < sizes.size(); ++index)
        strides[index] = checkedMultiply(strides[index - 1], sizes[index - 1], "computeStrides");

    return strides;
}

std::vector<size_t> computeShuffledStrides(std::vector<size_t> const& sizes,
                                           std::vector<size_t> const& dimOrder) {
    if (dimOrder.size() != sizes.size())
        throw std::runtime_error(
            "computeShuffledStrides: dimension order must contain every dimension");

    std::vector<size_t> strides(sizes.size(), 0);
    std::vector<bool> seen(sizes.size(), false);
    size_t stride = 1;
    for (const size_t index : dimOrder) {
        if (index >= sizes.size() || seen[index])
            throw std::runtime_error(
                "computeShuffledStrides: dimension order must be a permutation");
        seen[index] = true;
        strides[index] = stride;
        stride = checkedMultiply(stride, sizes[index], "computeShuffledStrides");
    }
    return strides;
}

size_t maximumOffset(std::vector<size_t> const& sizes, std::vector<size_t> const& strides,
                     std::string_view context) {
    size_t offset = 0;
    for (size_t dimension = 0; dimension < sizes.size(); ++dimension) {
        if (sizes[dimension] == 0) return 0;
        const size_t contribution =
            checkedMultiply(sizes[dimension] - 1, strides[dimension], context);
        offset = checkedAdd(offset, contribution, context);
    }
    return offset;
}

void parallelForChunks(size_t iterationCount, size_t workItemCount, ParallelChunkFunction function,
                       void* context) {
    if (function == nullptr)
        throw std::invalid_argument("parallelForChunks: function must be non-null");
    if (iterationCount == 0) return;

    const int threadCount = operationThreadCount(workItemCount);
    if (threadCount <= 1) {
        function(0, iterationCount, context);
        return;
    }

#ifdef _OPENMP
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
#pragma omp parallel for schedule(static, 1) num_threads(threadCount)
    for (int chunkIndex = 0; chunkIndex < threadCount; ++chunkIndex) {
        if (failed.load(std::memory_order_relaxed)) continue;

        const size_t chunk = static_cast<size_t>(chunkIndex);
        const size_t chunks = static_cast<size_t>(threadCount);
        const size_t baseSize = iterationCount / chunks;
        const size_t extraItems = iterationCount % chunks;
        const size_t begin = chunk * baseSize + std::min(chunk, extraItems);
        const size_t end = begin + baseSize + static_cast<size_t>(chunk < extraItems);
        try {
            function(begin, end, context);
        } catch (...) {
#pragma omp critical(roc_host_validation_amd_gpu_layout_exception)
            {
                if (!failure) failure = std::current_exception();
            }
            failed.store(true, std::memory_order_relaxed);
        }
    }
    if (failure) std::rethrow_exception(failure);
#else
    function(0, iterationCount, context);
#endif
}

size_t validateShuffle(size_t inputElementCount, std::vector<size_t> const& sizes,
                       std::vector<size_t> const& destinationStrides,
                       std::vector<size_t> const& sourceStrides) {
    if (sizes.size() != destinationStrides.size() || sizes.size() != sourceStrides.size())
        throw std::runtime_error("shuffleDims: size/stride dimension mismatch");

    if (sizes.size() < 2) throw std::runtime_error("shuffleDims: need at least 2 dimensions");

    const size_t totalElements = checkedProduct(sizes);
    if (inputElementCount != totalElements) {
        std::ostringstream message;
        message << "shuffleDims: input size " << inputElementCount << " doesn't match expected "
                << totalElements;
        throw std::runtime_error(message.str());
    }

    if (totalElements != 0 &&
        (maximumOffset(sizes, sourceStrides, "shuffleDims source") >= inputElementCount ||
         maximumOffset(sizes, destinationStrides, "shuffleDims destination") >= inputElementCount))
        throw std::runtime_error("shuffleDims: strides address outside the storage");

    return totalElements;
}

DimensionShufflePlan makePreSwizzlePlan(size_t inputElementCount, std::vector<size_t> const& sizes,
                                        std::vector<size_t> const& preSwizzleSize,
                                        std::vector<size_t> const& preTileSize) {
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
        else if (tileMN == 32 && subTileK == 4)
            dimensionOrder = {6, 2, 1, 3, 4, 5, 0, 7};
        else if (tileMN == 32 && subTileK == 2)
            dimensionOrder = {1, 2, 0, 3, 4, 5, 6, 7};
    } else if (preSwizzleSize.empty() && !preTileSize.empty()) {
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
        else if (tileMN == 32 && subTileK == 4)
            dimensionOrder = {7, 2, 1, 3, 5, 6, 0, 8, 4, 9};
        else if (tileMN == 32 && subTileK == 2)
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

GFX950ScalePlan makeGFX950ScalePlan(size_t inputElementCount, std::vector<size_t> const& sizes) {
    if (sizes.size() != 2) {
        std::ostringstream message;
        message << "preSwizzleAITER: sizes must have 2 elements, got " << sizes.size();
        throw std::runtime_error(message.str());
    }

    GFX950ScalePlan plan;
    plan.numRows = sizes[0];
    plan.numCols = sizes[1];
    const size_t totalElements =
        checkedMultiply(plan.numRows, plan.numCols, "preSwizzleScalesGFX950");
    if (totalElements != inputElementCount) {
        std::ostringstream message;
        message << "preSwizzleAITER: input size " << inputElementCount
                << " doesn't match sizes product " << totalElements;
        throw std::runtime_error(message.str());
    }

    plan.paddedRows = roundUp(plan.numRows, 32);
    plan.paddedCols = roundUp(plan.numCols, 8);
    plan.paddedElements =
        checkedMultiply(plan.paddedRows, plan.paddedCols, "preSwizzleScalesGFX950");
    plan.shuffle.sizes = {plan.paddedRows / 32, 2, 16, plan.paddedCols / 8, 2, 4};
    plan.shuffle.sourceStrides = {
        checkedMultiply(32, plan.paddedCols, "preSwizzleScalesGFX950 strides"),
        checkedMultiply(16, plan.paddedCols, "preSwizzleScalesGFX950 strides"),
        plan.paddedCols,
        8,
        4,
        1};
    plan.shuffle.destinationStrides =
        computeShuffledStrides(plan.shuffle.sizes, {1, 4, 2, 5, 3, 0});
    return plan;
}

GFX1250ScalePlan makeGFX1250ScalePlan(size_t inputElementCount, size_t slowDim, size_t fastDim,
                                      size_t mxBlock) {
    if (mxBlock != 16 && mxBlock != 32)
        throw std::runtime_error("preSwizzleScalesGFX1250: mxBlock must be 16 or 32");

    const size_t expectedInput = checkedMultiply(slowDim, fastDim, "preSwizzleScalesGFX1250 input");
    if (expectedInput != inputElementCount) {
        std::ostringstream message;
        message << "preSwizzleScalesGFX1250: input size " << inputElementCount
                << " doesn't match slowDim*fastDim = " << expectedInput;
        throw std::runtime_error(message.str());
    }

    GFX1250ScalePlan plan;
    plan.slowDim = slowDim;
    plan.fastDim = fastDim;
    plan.dimk = 128 / mxBlock;
    const size_t paddedFast = roundUp(fastDim, plan.dimk);
    plan.outputElements = checkedMultiply(slowDim, paddedFast, "preSwizzleScalesGFX1250 output");
    return plan;
}

}  // namespace detail

size_t preSwizzleScalesGFX950PaddedSize(size_t numRows, size_t numCols) {
    return detail::checkedMultiply(detail::roundUp(numRows, 32), detail::roundUp(numCols, 8),
                                   "preSwizzleScalesGFX950PaddedSize");
}

size_t preSwizzleScalesGFX1250PaddedSize(size_t slowDim, size_t fastDim, size_t mxBlock) {
    if (mxBlock != 16 && mxBlock != 32)
        throw std::runtime_error("preSwizzleScalesGFX1250PaddedSize: mxBlock must be 16 or 32");
    const size_t dimk = 128 / mxBlock;
    return detail::checkedMultiply(slowDim, detail::roundUp(fastDim, dimk),
                                   "preSwizzleScalesGFX1250PaddedSize");
}

}  // namespace roc::host_validation::amd_gpu_layout
