// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <string_view>
#include <vector>

namespace roc::host_validation::amd_gpu_layout {
namespace detail {

// Layout validation, planning, and parallel scheduling are compiled into the
// AMDGPULayout component. The templates below retain only type-dependent
// element construction and copying.
size_t checkedMultiply(size_t left, size_t right, std::string_view context);
size_t checkedAdd(size_t left, size_t right, std::string_view context);
size_t roundUp(size_t value, size_t multiple);
int operationThreadCount(size_t workItemCount, size_t minimumWorkItemsPerThread = 4096,
                         int defaultMaximumThreadCount = 8);

std::vector<size_t> computeStrides(std::vector<size_t> const& sizes);
std::vector<size_t> computeShuffledStrides(std::vector<size_t> const& sizes,
                                           std::vector<size_t> const& dimOrder);
size_t maximumOffset(std::vector<size_t> const& sizes, std::vector<size_t> const& strides,
                     std::string_view context);

using ParallelChunkFunction = void (*)(size_t begin, size_t end, void* context);
void parallelForChunks(size_t iterationCount, size_t workItemCount, ParallelChunkFunction function,
                       void* context);

struct DimensionShufflePlan {
    bool identity = false;
    std::vector<size_t> sizes;
    std::vector<size_t> destinationStrides;
    std::vector<size_t> sourceStrides;
};

struct GFX950ScalePlan {
    size_t numRows = 0;
    size_t numCols = 0;
    size_t paddedRows = 0;
    size_t paddedCols = 0;
    size_t paddedElements = 0;
    DimensionShufflePlan shuffle;
};

struct GFX1250ScalePlan {
    size_t slowDim = 0;
    size_t fastDim = 0;
    size_t dimk = 0;
    size_t outputElements = 0;
};

size_t validateShuffle(size_t inputElementCount, std::vector<size_t> const& sizes,
                       std::vector<size_t> const& destinationStrides,
                       std::vector<size_t> const& sourceStrides);
DimensionShufflePlan makePreSwizzlePlan(size_t inputElementCount, std::vector<size_t> const& sizes,
                                        std::vector<size_t> const& preSwizzleSize,
                                        std::vector<size_t> const& preTileSize);
GFX950ScalePlan makeGFX950ScalePlan(size_t inputElementCount, std::vector<size_t> const& sizes);
GFX1250ScalePlan makeGFX1250ScalePlan(size_t inputElementCount, size_t slowDim, size_t fastDim,
                                      size_t mxBlock);

/**
 * @brief Helper to compute product of elements in a range
 */
template <typename T>
inline size_t product(std::vector<T> const& x) {
    size_t result = 1;
    for (const auto value : x)
        result = checkedMultiply(result, static_cast<size_t>(value), "product");
    return result;
}

/**
 * @brief Shuffle data according to dimension reordering
 *
 * This performs a dimension shuffle where:
 * - input is arranged according to srcStrides
 * - output is arranged according to dstStrides
 * - both have the same dimension sizes
 */
template <typename T>
inline std::vector<T> shuffleDims(std::vector<T> const& input, std::vector<size_t> const& sizes,
                                  std::vector<size_t> const& dstStrides,
                                  std::vector<size_t> const& srcStrides) {
    const size_t totalElements = validateShuffle(input.size(), sizes, dstStrides, srcStrides);
    std::vector<T> output(input.size());

    struct ShuffleContext {
        std::vector<T> const* input;
        std::vector<size_t> const* sizes;
        std::vector<size_t> const* destinationStrides;
        std::vector<size_t> const* sourceStrides;
        std::vector<T>* output;
    };
    ShuffleContext context{&input, &sizes, &dstStrides, &srcStrides, &output};

    parallelForChunks(
        totalElements, totalElements,
        [](size_t begin, size_t end, void* opaqueContext) {
            auto& shuffle = *static_cast<ShuffleContext*>(opaqueContext);
            std::vector<size_t> coordinate(shuffle.sizes->size());
            for (size_t coordinateNumber = begin; coordinateNumber < end; ++coordinateNumber) {
                size_t remaining = coordinateNumber;
                for (size_t dimension = 0; dimension < shuffle.sizes->size(); ++dimension) {
                    coordinate[dimension] = remaining % (*shuffle.sizes)[dimension];
                    remaining /= (*shuffle.sizes)[dimension];
                }

                size_t sourceIndex = 0;
                size_t destinationIndex = 0;
                for (size_t dimension = 0; dimension < shuffle.sizes->size(); ++dimension) {
                    sourceIndex += coordinate[dimension] * (*shuffle.sourceStrides)[dimension];
                    destinationIndex +=
                        coordinate[dimension] * (*shuffle.destinationStrides)[dimension];
                }

                (*shuffle.output)[destinationIndex] = (*shuffle.input)[sourceIndex];
            }
        },
        &context);

    return output;
}

}  // namespace detail

/**
 * @brief Pre-swizzle and optionally pre-tile the input.
 *
 * This function rearranges tensor data according to swizzle and tile configurations.
 * The incoming data should be in row-major order with the 0 dimension being the
 * fastest (smallest stride).
 *
 * @param input The input data vector
 * @param sizes The dimension sizes {size0, size1}
 * @param preSwizzleSize The swizzle configuration {tileMN, tileK, subTileK}, or empty
 * @param preTileSize The pre-tile configuration {tileSize0, tileSize1}, or empty
 * @return The pre-swizzled/pre-tiled data
 */
template <typename T>
inline std::vector<T> preSwizzle(std::vector<T> const& input, std::vector<size_t> const& sizes,
                                 std::vector<size_t> const& preSwizzleSize,
                                 std::vector<size_t> const& preTileSize) {
    const auto plan = detail::makePreSwizzlePlan(input.size(), sizes, preSwizzleSize, preTileSize);
    if (plan.identity) return input;
    return detail::shuffleDims(input, plan.sizes, plan.destinationStrides, plan.sourceStrides);
}

/**
 * @brief Compute the output size of preSwizzleScalesGFX950 after padding
 *
 * @param numRows The number of rows (may not be a multiple of 32)
 * @param numCols The number of columns (may not be a multiple of 8)
 * @return The total number of elements in the padded output
 */
size_t preSwizzleScalesGFX950PaddedSize(size_t numRows, size_t numCols);

/**
 * @brief Pre-swizzle scale data.
 *
 * This implements the e8m0_shuffle algorithm from:
 * https://github.com/ROCm/aiter/blob/main/aiter/utility/fp4_utils.py
 *
 * The algorithm is:
 *   scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
 *   scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
 *   scale = scale.view(sm, sn)
 *
 * @param input The input scale data vector (row-major, M x numScaleCols)
 * @param sizes The dimension sizes {numScaleRows, numScaleCols} where numScaleRows = M
 * @return The swizzled scale data
 */
template <typename T>
inline std::vector<T> preSwizzleScalesGFX950(std::vector<T> const& input,
                                             std::vector<size_t> const& sizes) {
    const auto plan = detail::makeGFX950ScalePlan(input.size(), sizes);

    std::vector<T> const* inputPointer = &input;
    std::vector<T> paddedInput;
    if (plan.paddedRows != plan.numRows || plan.paddedCols != plan.numCols) {
        paddedInput.resize(plan.paddedElements, T{});
        for (size_t row = 0; row < plan.numRows; ++row) {
            std::copy(input.begin() + row * plan.numCols,
                      input.begin() + row * plan.numCols + plan.numCols,
                      paddedInput.begin() + row * plan.paddedCols);
        }
        inputPointer = &paddedInput;
    }

    return detail::shuffleDims(*inputPointer, plan.shuffle.sizes, plan.shuffle.destinationStrides,
                               plan.shuffle.sourceStrides);
}

/**
 * @brief Compute the output size of preSwizzleScalesGFX1250 after padding
 *
 * The gfx1250 dimk swizzle pads the "fast" dimension (the one varying with
 * the K-block index in MX storage) to a multiple of `dimk = 128 / mxBlock`.
 * The slow dimension is left unchanged.
 *
 * @param slowDim Number of elements along the slow dimension of the
 *                natural-layout scale buffer (M for transA=T, K/MX for transA=N).
 * @param fastDim Number of elements along the fast dimension (K/MX for
 *                transA=T, M for transA=N).
 * @param mxBlock The MX block size (16 or 32).
 * @return The total number of elements in the swizzled, padded output.
 */
size_t preSwizzleScalesGFX1250PaddedSize(size_t slowDim, size_t fastDim, size_t mxBlock);

/**
 * @brief Pre-swizzle scale data for the gfx1250-class block-scaled MX layout.
 *
 * The kernel expects the scale tensor to be viewed as
 *   `{slowDim, ceil(fastDim / dimk), dimk}`
 * where `dimk = 128 / mxBlock`, and then permuted by `(1, 0, 2)` to
 *   `{ceil(fastDim / dimk), slowDim, dimk}`.
 * The fast dimension is padded with zero scales to a multiple of `dimk`.
 *
 * "Slow" / "fast" refer to the natural column-major scale layout the MX
 * generator produces. For block-scaled inputs:
 *   - transA = T scaleA: scale is (K/MX rows x M cols) col-major.
 *     Fast (stride 1) = K/MX. Slow = M.
 *   - transA = N scaleA: fast = M, slow = K/MX.
 *   - transB = N scaleB: fast = K/MX, slow = N.
 *   - transB = T scaleB: fast = N, slow = K/MX.
 *
 * @param input    The natural-layout scale data (size = slowDim * fastDim).
 * @param slowDim  Slow (stride > 1) dimension of the natural layout.
 * @param fastDim  Fast (stride 1) dimension of the natural layout.
 * @param mxBlock  MX block size (16 or 32).
 * @return The swizzled scale buffer of size
 *         `slowDim * roundUp(fastDim, dimk)` (i.e.
 *         `preSwizzleScalesGFX1250PaddedSize(slowDim, fastDim, mxBlock)`).
 */
template <typename T>
inline std::vector<T> preSwizzleScalesGFX1250(std::vector<T> const& input, size_t slowDim,
                                              size_t fastDim, size_t mxBlock) {
    const auto plan = detail::makeGFX1250ScalePlan(input.size(), slowDim, fastDim, mxBlock);
    std::vector<T> output(plan.outputElements, T{});

    struct CopyContext {
        std::vector<T> const* input;
        std::vector<T>* output;
        size_t slowDim;
        size_t fastDim;
        size_t dimk;
    };
    CopyContext context{&input, &output, plan.slowDim, plan.fastDim, plan.dimk};
    const size_t copyGroupCount = plan.outputElements / plan.dimk;

    detail::parallelForChunks(
        copyGroupCount, plan.outputElements,
        [](size_t begin, size_t end, void* opaqueContext) {
            auto& copy = *static_cast<CopyContext*>(opaqueContext);
            for (size_t copyGroup = begin; copyGroup < end; ++copyGroup) {
                const size_t slowIndex = copyGroup % copy.slowDim;
                const size_t tile = copyGroup / copy.slowDim;
                const size_t outputBase = copyGroup * copy.dimk;
                const size_t sourceFastBase = tile * copy.dimk;
                for (size_t elementInTile = 0; elementInTile < copy.dimk; ++elementInTile) {
                    const size_t sourceFast = sourceFastBase + elementInTile;
                    if (sourceFast < copy.fastDim)
                        (*copy.output)[outputBase + elementInTile] =
                            (*copy.input)[slowIndex * copy.fastDim + sourceFast];
                }
            }
        },
        &context);

    return output;
}

}  // namespace roc::host_validation::amd_gpu_layout
