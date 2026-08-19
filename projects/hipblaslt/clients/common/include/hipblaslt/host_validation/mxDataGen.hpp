// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

// hipBLASLt MX generation adapter over shared host-validation.

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
                                     std::string_view         archName);

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-export.h>
#include <hipblaslt/hipblaslt-types.h>
#include <roc/host_validation/mx.hpp>
#include <stdint.h>

#include <span>
#include <vector>

inline constexpr uint32_t hipblasltMxDefaultSeed = 1713573849U;

std::vector<float> generateMXInput(hipDataType            dataType,
                                   hipDataType            scaleType,
                                   std::span<uint8_t>     data,
                                   std::span<uint8_t>     scale,
                                   uint64_t               row,
                                   uint64_t               col,
                                   uint64_t               stride,
                                   int const              scaleBlockRowSize,
                                   int const              scaleBlockColSize,
                                   MXScaleLayout          scaleLayout     = MXScaleLayout::None,
                                   std::string_view const initMethod      = "Bounded",
                                   float                  min_val         = -1.0f,
                                   float                  max_val         = 1.0f,
                                   std::string_view const scaleInitMethod = "",
                                   uint32_t               seed = hipblasltMxDefaultSeed);

// generateMXInput emits scales packed for the unpadded data K, but setMXScaleA/B
// on gfx950 pad ceil(K/mxBlock) up to a multiple of 8. K-fast layouts need this
// in-place restride before scale swizzle / H2D (see tensile DataInitialization).
void restrideMXScaleBufferKFast(std::span<uint8_t> buffer,
                                size_t             compactFreeDim,
                                size_t             compactKBlocks,
                                size_t             paddedKBlocks,
                                size_t             elemBytes);

#endif
