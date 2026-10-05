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

// Optional helper: expand compact K-fast scales to a wider K stride in-place.
// Used by tensilelite client when a descriptor reports padded K-blocks;
// hipblaslt testing_matmul relies on preSwizzleScalesGFX950's internal pad
// instead. NoSwizzle Compact descriptors leave compactK == paddedK (no-op).
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
