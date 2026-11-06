// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

enum class MXScaleLayout
{
    None    = 0,
    GFX950  = 1,
    GFX1250 = 2,
    // 128-row by 4-K-block tiles (Block_32_UE8M0_128_4_EXT); K-contiguous operands only.
    Tile128x4 = 3,
};

#include "hipblaslt_scaling_format.hpp"
#include <cstdint>
#include <string_view>

MXScaleLayout mxScaleLayoutForArchName(std::string_view archName);

// Maps a block-scaling format and device arch to the client scale swizzle layout.
// Only Block_32_UE8M0_32_8_EXT uses GFX950 swizzle and only Block_32_UE8M0_128_4_EXT
// uses Tile128x4; gfx1250 uses GFX1250 for other block formats; everything else stays
// natural-packed (None).
MXScaleLayout mxScaleLayoutForFormat(hipblaslt_scaling_format scalingFormat,
                                     std::string_view       archName);

// integer_exact MX elements are integers of magnitude at most kIntegerExactMXMaxElement, and their
// E8M0 scales at most kIntegerExactMXMaxScale, so each product of a dequantized A and B element is
// an integer of magnitude at most kIntegerExactMXMaxProduct. K of them bound every partial sum of
// D, in any summation order. fp32 holds every integer below 2^24 exactly, so the device and the
// host reference agree exactly while K * kIntegerExactMXMaxProduct < 2^24, that is K < 262144.
constexpr int     kIntegerExactMXMaxElement = 2;
constexpr int     kIntegerExactMXMaxScale   = 4;
constexpr int64_t kIntegerExactMXMaxProduct = int64_t(kIntegerExactMXMaxElement)
                                              * kIntegerExactMXMaxScale * kIntegerExactMXMaxElement
                                              * kIntegerExactMXMaxScale;
constexpr int64_t kIntegerExactMXExactLimit = int64_t(1) << 24;

constexpr int64_t integerExactMXSumBound(int64_t K)
{
    return K * kIntegerExactMXMaxProduct;
}

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-export.h>
#include <hipblaslt/hipblaslt-types.h>
#include <stdint.h>

#include <vector>

// The seedIndex of A (isMatrixA) or B in batch `batch`: abc + 3 * batch, abc = 0 for A and 1 for
// B. A broadcast matrix (batch stride 0) is one matrix, so every batch takes batch 0's index and
// regenerates the same data, scales and reference.
inline uint32_t mxSeedIndex(bool isMatrixA, int64_t batch, int64_t batchStride)
{
    return static_cast<uint32_t>((isMatrixA ? 0 : 1) + 3 * (batchStride == 0 ? 0 : batch));
}

// seedIndex (mxSeedIndex) salts the maf2_0 seed and, for maf1_0,
// picks sin (A) or cos (B) and the batch's index offset batch * row * col; every other
// initMethod uses one fixed seed for all matrices and batches.
// integer_exact supports fp8 and fp6 data. With swizzled scales it currently requires K along
// stored rows (transposed A or non-transposed B); other orientations throw before writing.
// With K along the stored rows, a non-positive row or column throws std::invalid_argument
// before writing: the row count divides the padded-layout check. Otherwise the reference
// covers all stride * col elements the generator writes, and stride * col must be a multiple
// of row, or it throws before writing.
// With K along the stored columns (non-transposed A or transposed B), K must be a multiple of
// the scale block. A partial block makes getAlignedFloat read past the packed buffer, so
// integer_exact throws before writing.
// With returnReference false the returned dequantized reference vector is not built, and
// an empty vector is returned; the data and scale bytes written are the same either way.
// Decoupled scale initialization can still use internal floats to generate those bytes.
std::vector<float> generateMXInput(hipDataType            dataType,
                                   hipDataType            scaleType,
                                   void*                  data,
                                   void*                  scale,
                                   uint64_t               row,
                                   uint64_t               col,
                                   uint64_t               stride,
                                   bool                   isTranspose,
                                   int const              scaleBlockRowSize,
                                   int const              scaleBlockColSize,
                                   bool                   isMatrixA,
                                   MXScaleLayout          scaleLayout = MXScaleLayout::None,
                                   std::string_view const initMethod  = "Bounded",
                                   float                  min_val     = -1.0f,
                                   float                  max_val     = 1.0f,
                                   std::string_view const scaleInitMethod = "",
                                   uint32_t               seedIndex       = 0,
                                   bool                   returnReference = true);

// generateMXInput emits scales packed for the unpadded data K, but setMXScaleA/B
// on gfx950 pad ceil(K/mxBlock) up to a multiple of 8. K-fast layouts need this
// in-place restride before scale swizzle / H2D (see tensile DataInitialization).
void restrideMXScaleBufferKFast(uint8_t* buffer,
                                size_t   compactFreeDim,
                                size_t   compactKBlocks,
                                size_t   paddedKBlocks,
                                size_t   elemBytes);

void applyMXScaleLayoutInPlace(uint8_t*      scale,
                               size_t        scaleElemCount,
                               MXScaleLayout scaleLayout,
                               size_t        slowDim,
                               size_t        fastDim,
                               size_t        mxBlock);

// The compact canonical scale generateMXInput(..., MXScaleLayout::None) writes for these
// arguments (`elements` one-byte scales) and the {mnExtent, kBlocks} grid its
// MXScaleLayout::GFX950 output swizzles that scale as: swizzleScaleGFX950(scale, grid) on the
// compact scale gives the GFX950 call's scale. The generator sizes the scale from `stride`, so
// with a padded leading dimension it covers more MN rows than the problem has.
struct MXScaleGrid
{
    size_t elements;
    size_t mnExtent;
    size_t kBlocks;
    // K runs along the stored rows (transposed A, non-transposed B); otherwise the compact
    // scale is [K block][MN].
    bool   kIsRows;
    size_t kExtent;
    size_t elementsPerMXBlock;
};

MXScaleGrid mxScaleGridGFX950(uint64_t  row,
                              uint64_t  col,
                              uint64_t  stride,
                              bool      isTranspose,
                              int const scaleBlockRowSize,
                              int const scaleBlockColSize,
                              bool      isMatrixA);

// The scale generateMXInput(..., MXScaleLayout::GFX950) writes, from the compact canonical
// scale (grid.elements scales) of the same arguments and their mxScaleGridGFX950 grid.
std::vector<uint8_t> swizzleScaleGFX950(std::vector<uint8_t> const& scaleBytes,
                                        MXScaleGrid const&          grid);

#endif
