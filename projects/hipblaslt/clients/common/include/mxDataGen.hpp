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

// integer_exact supports fp8 and fp6 data. With swizzled scales it currently requires K along
// stored rows (transposed A or non-transposed B); other orientations throw before writing.
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
                                   std::string_view const scaleInitMethod = "");

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

#endif
