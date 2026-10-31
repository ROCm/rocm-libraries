// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

enum class MXScaleLayout
{
    None    = 0,
    GFX950  = 1,
    GFX1250 = 2,
};

#include "hipblaslt_scaling_format.hpp"
#include <string_view>

MXScaleLayout mxScaleLayoutForArchName(std::string_view archName);

// Maps a block-scaling format and device arch to the client scale swizzle layout.
// Only Block_32_UE8M0_32_8_EXT uses GFX950 swizzle; gfx1250 uses GFX1250 for other
// block formats; everything else stays natural-packed (None).
MXScaleLayout mxScaleLayoutForFormat(hipblaslt_scaling_format scalingFormat,
                                     std::string_view       archName);

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-export.h>
#include <hipblaslt/hipblaslt-types.h>
#include <stdint.h>

#include <vector>

// integer_exact supports fp8 data. With swizzled scales it currently requires K along
// stored rows (transposed A or non-transposed B); other orientations throw before writing.
// With returnReference false the dequantized reference floats are not computed and an empty
// vector is returned; the data and scale bytes written are the same either way.
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
// MXScaleLayout::GFX950 output swizzles that scale as: applyMXScaleLayoutInPlace(scale,
// elements, MXScaleLayout::GFX950, mnExtent, kBlocks, mxBlock) on the compact scale gives the
// GFX950 call's scale. The generator sizes the scale from `stride`, so with a padded leading
// dimension it covers more MN rows than the problem has.
struct MXScaleGrid
{
    size_t elements;
    size_t mnExtent;
    size_t kBlocks;
};

MXScaleGrid mxScaleGridGFX950(uint64_t  row,
                              uint64_t  col,
                              uint64_t  stride,
                              bool      isTranspose,
                              int const scaleBlockRowSize,
                              int const scaleBlockColSize,
                              bool      isMatrixA);

#endif
