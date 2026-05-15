/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

// Pure-value enums declared outside the HIPBLASLT_ENABLE_MXDATAGENERATOR guard
// so widely-included headers can hold them as member variables without
// transitively dragging in HIP / hipblaslt when the feature is disabled.

// Cpu = host PRNG, host pointers (deterministic baseline).
// Gpu = device PRNG into device pointers; reference float is read back from
// the device buffer. Layouts that need PRNG rearrangement
// (getAlignedFloat) silently fall back to the CPU path; only
// (isMatrixA == isTranspose) actually exercises the GPU PRNG.
enum class MXInitDevice
{
    Cpu = 0,
    Gpu = 1,
};

// Scale tensor memory layout left in the `scale` buffer.
//   kNone    : natural mxDataGenerator layout (no swizzle).
//   kGFX950  : preSwizzleScalesGFX950 layout for gfx950 subtile kernels.
//   kGFX1250 : preSwizzleScalesGFX1250 (dimk) layout for gfx1250 +
//              non-rocroller WMMA kernels.
enum class MXScaleLayout
{
    kNone    = 0,
    kGFX950  = 1,
    kGFX1250 = 2,
};

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-export.h>
#include <hipblaslt/hipblaslt-types.h>
#include <stdint.h>

#include <string_view>
#include <vector>

// `scaleLayout` selects the post-generation scale memory layout. `initDevice`
// selects the host vs device PRNG path; with MXInitDevice::Gpu, data/scale
// must be device pointers (see MXInitDevice for full semantics).
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
                                   MXScaleLayout          scaleLayout = MXScaleLayout::kNone,
                                   std::string_view const initMethod  = "Bounded",
                                   float                  min_val     = -1.0f,
                                   float                  max_val     = 1.0f,
                                   MXInitDevice           initDevice  = MXInitDevice::Cpu);

#endif
