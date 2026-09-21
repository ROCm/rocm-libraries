// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace DGen
{

    /**
     * @brief Round up a value to the next multiple of a given number
     */
    inline size_t roundUp(size_t value, size_t multiple)
    {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    /**
     * @brief Compute the output size of preSwizzleScalesGFX950 after padding
     *
     * @param numRows The number of rows (may not be a multiple of 32)
     * @param numCols The number of columns (may not be a multiple of 8)
     * @return The total number of elements in the padded output
     */
    inline size_t preSwizzleScalesGFX950PaddedSize(size_t numRows, size_t numCols)
    {
        return roundUp(numRows, 32) * roundUp(numCols, 8);
    }

    /**
     * @brief Helper to compute product of elements in a range
     */
    template <typename T>
    inline size_t product(std::vector<T> const& x)
    {
        return std::accumulate(x.begin(), x.end(), size_t(1), std::multiplies<size_t>());
    }

    /**
     * @brief Compute strides for column-major layout given sizes
     */
    inline std::vector<size_t> computeStrides(std::vector<size_t> const& sizes)
    {
        std::vector<size_t> strides(sizes.size());
        if(sizes.empty())
            return strides;

        strides[0] = 1;
        for(size_t i = 1; i < sizes.size(); ++i)
            strides[i] = strides[i - 1] * sizes[i - 1];

        return strides;
    }

    /**
     * @brief Compute shuffled strides given dimension order
     */
    inline std::vector<size_t> computeShuffledStrides(std::vector<size_t> const& sizes,
                                                       std::vector<size_t> const& dimOrder)
    {
        std::vector<size_t> strides(sizes.size(), 0);
        size_t              stride = 1;
        for(auto idx : dimOrder)
        {
            strides.at(idx) = stride;
            stride *= sizes.at(idx);
        }
        return strides;
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
    inline std::vector<T> shuffleDims(std::vector<T> const&      input,
                                      std::vector<size_t> const& sizes,
                                      std::vector<size_t> const& dstStrides,
                                      std::vector<size_t> const& srcStrides)
    {
        if(sizes.size() != dstStrides.size() || sizes.size() != srcStrides.size())
            throw std::runtime_error("shuffleDims: size/stride dimension mismatch");

        if(sizes.size() < 2)
            throw std::runtime_error("shuffleDims: need at least 2 dimensions");

        size_t totalElements = product(sizes);
        if(input.size() != totalElements)
        {
            std::ostringstream msg;
            msg << "shuffleDims: input size " << input.size() << " doesn't match expected "
                << totalElements;
            throw std::runtime_error(msg.str());
        }

        std::vector<T> output(input.size());

        // Compute total number of coordinates
        size_t totalCoords = 1;
        for(size_t i = 0; i < sizes.size(); ++i)
            totalCoords *= sizes[i];

#pragma omp parallel for
        for(size_t coordNum = 0; coordNum < totalCoords; ++coordNum)
        {
            // Convert coordNum to N-D coordinates
            std::vector<size_t> coord(sizes.size());
            size_t              remaining = coordNum;
            for(size_t i = 0; i < sizes.size(); ++i)
            {
                coord[i] = remaining % sizes[i];
                remaining /= sizes[i];
            }

            // Compute source and destination indices using strides
            size_t srcIdx = 0;
            size_t dstIdx = 0;
            for(size_t i = 0; i < sizes.size(); ++i)
            {
                srcIdx += coord[i] * srcStrides[i];
                dstIdx += coord[i] * dstStrides[i];
            }

            output[dstIdx] = input[srcIdx];
        }

        return output;
    }

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
    inline std::vector<T> preSwizzle(std::vector<T> const&      input,
                                     std::vector<size_t> const& sizes,
                                     std::vector<size_t> const& preSwizzleSize,
                                     std::vector<size_t> const& preTileSize)
    {
        if(!preSwizzleSize.empty())
        {
            if(preSwizzleSize.size() != 3)
            {
                std::ostringstream msg;
                msg << "preSwizzle: preSwizzleSize must have 3 elements, got "
                    << preSwizzleSize.size();
                throw std::runtime_error(msg.str());
            }
        }

        if(sizes.size() != 2)
        {
            std::ostringstream msg;
            msg << "preSwizzle: Batch dimension not yet supported. sizes.size()=" << sizes.size();
            throw std::runtime_error(msg.str());
        }

        size_t totalElements = product(sizes);
        if(totalElements != input.size())
        {
            std::ostringstream msg;
            msg << "preSwizzle: input size " << input.size() << " doesn't match sizes product "
                << totalElements;
            throw std::runtime_error(msg.str());
        }

        std::vector<size_t> srcSizes, dimOrder;

        if((!preSwizzleSize.empty()) && (preTileSize.empty()))
        {
            auto tileMN   = preSwizzleSize[0];
            auto tileK    = preSwizzleSize[1];
            auto subTileK = preSwizzleSize[2];

            if(tileMN != 64 && tileMN != 32)
            {
                std::ostringstream msg;
                msg << "preSwizzle: tileMN must be 32 or 64, got " << tileMN;
                throw std::runtime_error(msg.str());
            }

            if(tileK % 4 != 0)
            {
                std::ostringstream msg;
                msg << "preSwizzle: tileK must be a multiple of 4, got " << tileK;
                throw std::runtime_error(msg.str());
            }

            size_t nLanesPerSIMD   = 16;
            size_t nSIMDsPerWave   = 4;
            size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
            size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
            size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
            size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
            size_t nSIMDIndexBlock = nVGPRIndex;
            size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

            if(nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK)
            {
                std::ostringstream msg;
                msg << "preSwizzle: nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK";
                throw std::runtime_error(msg.str());
            }

            if(nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN)
            {
                std::ostringstream msg;
                msg << "preSwizzle: nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN";
                throw std::runtime_error(msg.str());
            }

            srcSizes = {nVGPRIndex,
                        nVGPRBlock,
                        nSIMDBlock,
                        sizes[0] / (tileK),
                        nLanesPerSIMD,
                        nSIMDIndexIndex,
                        nSIMDIndexBlock,
                        sizes[1] / (tileMN)};

            if(tileMN == 64)
            {
                // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
                dimOrder = {6, 1, 2, 3, 4, 5, 0, 7};
            }
            else if(tileMN == 32 && subTileK == 4)
            {
                // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
                //              swap nSIMDBlock (2) and nVGPRBlock (1)
                dimOrder = {6, 2, 1, 3, 4, 5, 0, 7};
            }
            else if(tileMN == 32 && subTileK == 2)
            {
                // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
                dimOrder = {1, 2, 0, 3, 4, 5, 6, 7};
            }
        }
        else if((preSwizzleSize.empty()) && (!preTileSize.empty()))
        {
            srcSizes = {preTileSize[0],
                        sizes[0] / preTileSize[0],
                        preTileSize[1],
                        sizes[1] / preTileSize[1]};

            // Pre-tiling: 1 and 3 are pushed to the back (they become the slowest)
            dimOrder = {0, 2, 1, 3};
        }
        else
        {
            auto tileMN   = preSwizzleSize[0];
            auto tileK    = preSwizzleSize[1];
            auto subTileK = preSwizzleSize[2];

            if(tileMN != 64 && tileMN != 32)
            {
                std::ostringstream msg;
                msg << "preSwizzle: tileMN must be 32 or 64, got " << tileMN;
                throw std::runtime_error(msg.str());
            }

            if(tileK % 4 != 0)
            {
                std::ostringstream msg;
                msg << "preSwizzle: tileK must be a multiple of 4, got " << tileK;
                throw std::runtime_error(msg.str());
            }

            size_t ptTileSizeK     = preTileSize[0];
            size_t ptTileSizeMN    = preTileSize[1];
            size_t nLanesPerSIMD   = 16;
            size_t nSIMDsPerWave   = 4;
            size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
            size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
            size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
            size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
            size_t nSIMDIndexBlock = nVGPRIndex;
            size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

            if(ptTileSizeK / tileK == 0)
            {
                std::ostringstream msg;
                msg << "preSwizzle: ptTileSizeK / tileK == 0, ptTileSizeK=" << ptTileSizeK
                    << ", tileK=" << tileK;
                throw std::runtime_error(msg.str());
            }

            if(ptTileSizeMN / tileMN == 0)
            {
                std::ostringstream msg;
                msg << "preSwizzle: ptTileSizeMN / tileMN == 0, ptTileSizeMN=" << ptTileSizeMN
                    << ", tileMN=" << tileMN;
                throw std::runtime_error(msg.str());
            }

            if(nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK)
            {
                std::ostringstream msg;
                msg << "preSwizzle: nVGPRIndex * nVGPRBlock * nSIMDBlock != tileK";
                throw std::runtime_error(msg.str());
            }

            if(nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN)
            {
                std::ostringstream msg;
                msg << "preSwizzle: nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock != tileMN";
                throw std::runtime_error(msg.str());
            }

            srcSizes = {nVGPRIndex,
                        nVGPRBlock,
                        nSIMDBlock,
                        ptTileSizeK / tileK,
                        sizes[0] / ptTileSizeK,
                        nLanesPerSIMD,
                        nSIMDIndexIndex,
                        nSIMDIndexBlock,
                        ptTileSizeMN / tileMN,
                        sizes[1] / ptTileSizeMN};

            if(tileMN == 64)
            {
                // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
                // Pre tile: push workgroup tiles (4 and 9) to the end
                dimOrder = {7, 1, 2, 3, 5, 6, 0, 8, 4, 9};
            }
            else if(tileMN == 32 && subTileK == 4)
            {
                // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
                //              swap nSIMDBlock (2) and nVGPRBlock (1)
                // Pre tile: push workgroup tiles (4 and 9) to the end
                dimOrder = {7, 2, 1, 3, 5, 6, 0, 8, 4, 9};
            }
            else if(tileMN == 32 && subTileK == 2)
            {
                // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
                // Pre tile: push workgroup tiles (4 and 9) to the end
                dimOrder = {1, 2, 0, 3, 5, 6, 7, 8, 4, 9};
            }
        }

        if(product(srcSizes) != product(sizes))
        {
            std::ostringstream msg;
            msg << "PreSwizzle size mismatch: product(srcSizes)=" << product(srcSizes)
                << " != product(sizes)=" << product(sizes);
            throw std::runtime_error(msg.str());
        }

        if(srcSizes.empty())
            throw std::runtime_error("PreSwizzle source size not populated.");

        if(dimOrder.empty())
            throw std::runtime_error("PreSwizzle permutation order not populated.");

        auto srcStrides = computeStrides(srcSizes);
        auto dstStrides = computeShuffledStrides(srcSizes, dimOrder);

        return shuffleDims(input, srcSizes, dstStrides, srcStrides);
    }

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
     * For output position (outRow, outCol), we compute the source position
     * by decomposing into 6D indices and applying the inverse permutation.
     *
     * @param input The input scale data vector (row-major, M x numScaleCols)
     * @param sizes The dimension sizes {numScaleRows, numScaleCols} where numScaleRows = M
     * @return The swizzled scale data
     */
    template <typename T>
    inline std::vector<T> preSwizzleScalesGFX950(std::vector<T> const&      input,
                                                 std::vector<size_t> const& sizes)
    {
        if(sizes.size() != 2)
        {
            std::ostringstream msg;
            msg << "preSwizzleAITER: sizes must have 2 elements, got " << sizes.size();
            throw std::runtime_error(msg.str());
        }

        size_t numRows = sizes[0]; // M dimension (number of scale rows)
        size_t numCols = sizes[1]; // K/32 dimension (number of scale columns)

        size_t totalElements = numRows * numCols;
        if(totalElements != input.size())
        {
            std::ostringstream msg;
            msg << "preSwizzleAITER: input size " << input.size() << " doesn't match sizes product "
                << totalElements;
            throw std::runtime_error(msg.str());
        }

        // Pad rows to multiple of 32 and cols to multiple of 8 if needed
        size_t paddedRows = roundUp(numRows, 32);
        size_t paddedCols = roundUp(numCols, 8);

        // Create padded input if dimensions are not already aligned
        std::vector<T> const* inputPtr = &input;
        std::vector<T>        paddedInput;
        if(paddedRows != numRows || paddedCols != numCols)
        {
            paddedInput.resize(paddedRows * paddedCols, T{});
            // Copy each row of original data into the padded buffer (row-major layout)
            for(size_t r = 0; r < numRows; ++r)
            {
                std::copy(input.begin() + r * numCols,
                          input.begin() + r * numCols + numCols,
                          paddedInput.begin() + r * paddedCols);
            }
            inputPtr = &paddedInput;
        }

        // AITER shuffle algorithm using shuffleDims:
        // view as (paddedRows // 32, 2, 16, paddedCols // 8, 2, 4)
        // permute (0, 3, 5, 2, 4, 1)
        // view as (paddedRows, paddedCols)

        // 6D view of the 2D row-major input
        std::vector<size_t> srcSizes = {paddedRows / 32, 2, 16, paddedCols / 8, 2, 4};

        // Row-major strides for the 6D view:
        // row = d0*32 + d1*16 + d2, col = d3*8 + d4*4 + d5
        // linear = row*paddedCols + col
        std::vector<size_t> srcStrides
            = {32 * paddedCols, 16 * paddedCols, paddedCols, 8, 4, 1};

        // Dimension order derived from inverse of permute(0, 3, 5, 2, 4, 1).
        // The inverse permutation is {0, 5, 3, 1, 4, 2}.
        // For row-major output (last dimension fastest), we process dimensions
        // in the order that makes the output contiguous: {1, 4, 2, 5, 3, 0}
        std::vector<size_t> dimOrder = {1, 4, 2, 5, 3, 0};
        auto                dstStrides = computeShuffledStrides(srcSizes, dimOrder);

        return shuffleDims(*inputPtr, srcSizes, dstStrides, srcStrides);
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
     * @return The total number of bytes in the swizzled, padded output.
     */
    inline size_t
        preSwizzleScalesGFX1250PaddedSize(size_t slowDim, size_t fastDim, size_t mxBlock)
    {
        if(mxBlock == 0)
            throw std::runtime_error("preSwizzleScalesGFX1250PaddedSize: mxBlock must be > 0");
        size_t const dimk = 128 / mxBlock;
        if(dimk == 0)
            throw std::runtime_error(
                "preSwizzleScalesGFX1250PaddedSize: dimk = 128 / mxBlock must be > 0");
        return slowDim * roundUp(fastDim, dimk);
    }

    /**
     * @brief Pre-swizzle scale data for the gfx1250 (and other non-rocroller WMMA)
     *        block-scaled MX layout.
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
     * In hipBLASLt-bench terms `kAlongFast = (isMatrixA == isTranspose)`.
     *
     * The algorithm is byte-equivalent to the reference implementation in
     * mengzcai's `swizzle_mx_scale` (mengzcai/develop-gfx1250-mx-pick-v2),
     * but is self-contained (no dependency on `TensorDataManipulation.hpp`).
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
    inline std::vector<T> preSwizzleScalesGFX1250(std::vector<T> const& input,
                                                  size_t                slowDim,
                                                  size_t                fastDim,
                                                  size_t                mxBlock)
    {
        if(mxBlock == 0)
            throw std::runtime_error("preSwizzleScalesGFX1250: mxBlock must be > 0");
        size_t const dimk = 128 / mxBlock;
        if(dimk == 0)
            throw std::runtime_error(
                "preSwizzleScalesGFX1250: dimk = 128 / mxBlock must be > 0");

        size_t const expectedInput = slowDim * fastDim;
        if(expectedInput != input.size())
        {
            std::ostringstream msg;
            msg << "preSwizzleScalesGFX1250: input size " << input.size()
                << " doesn't match slowDim*fastDim = " << expectedInput;
            throw std::runtime_error(msg.str());
        }

        size_t const paddedFast = roundUp(fastDim, dimk);
        size_t const numTiles   = paddedFast / dimk;

        // Output layout: {numTiles, slowDim, dimk} in row-major (last dim fastest).
        // Output linear index: outIdx = tile*(slowDim*dimk) + s*dimk + j
        // where j in [0, dimk), s in [0, slowDim), tile in [0, numTiles).
        // Source (natural layout) linear index for the same (s, tile, j):
        //   srcFast = tile*dimk + j
        //   srcIdx  = s * fastDim + srcFast    (when srcFast < fastDim; else padded)
        // Padded entries (srcFast >= fastDim) are written as default-constructed T.
        std::vector<T> output(slowDim * paddedFast, T{});

#pragma omp parallel for collapse(2)
        for(size_t tile = 0; tile < numTiles; ++tile)
        {
            for(size_t s = 0; s < slowDim; ++s)
            {
                size_t const outBase = tile * (slowDim * dimk) + s * dimk;
                for(size_t j = 0; j < dimk; ++j)
                {
                    size_t const srcFast = tile * dimk + j;
                    if(srcFast < fastDim)
                        output[outBase + j] = input[s * fastDim + srcFast];
                    // else: leave T{} (zero scale -> kernel ignores padded MX block)
                }
            }
        }

        return output;
    }

} // namespace DGen

