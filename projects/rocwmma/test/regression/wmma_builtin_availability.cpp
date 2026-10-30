/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// WMMA support and selection do not depend on builtin availability: with every
// WMMA builtin reported unavailable, metadata and explicit-target selection are
// unchanged in the host pass and in every device pass. Only instantiating an
// operation's exec would fail, and this file never does.
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_bf16_16x16x16_bf16_w32 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_bf16_16x16x16_bf16_w32_gfx12 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_bf16_16x16x32_bf16 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x128_bf8_bf8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x128_bf8_fp8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x128_fp8_bf8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x128_fp8_fp8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x16_f16_w32 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x16_f16_w32_gfx12 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x32_f16 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x64_bf8_bf8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x64_bf8_fp8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x64_fp8_bf8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f16_16x16x64_fp8_fp8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x128_bf8_bf8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x128_bf8_fp8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x128_fp8_bf8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x128_fp8_fp8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x16_bf16_w32 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x16_bf16_w32_gfx12 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x16_bf8_bf8_w32_gfx12 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x16_bf8_fp8_w32_gfx12 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x16_f16_w32 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x16_f16_w32_gfx12 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x16_fp8_bf8_w32_gfx12 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x32_bf16 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x32_f16 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x4_f32 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x64_bf8_bf8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x64_bf8_fp8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x64_fp8_bf8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_f32_16x16x64_fp8_fp8 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_i32_16x16x16_iu8_w32 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_i32_16x16x16_iu8_w32_gfx12 1
#define ROCWMMA_TEST_NO_AMDGCN_BUILTIN_wmma_i32_16x16x64_iu8 1

#include <gtest/gtest.h>
#include <rocwmma/rocwmma.hpp>

#include "wmma_target_backend.hpp"

namespace
{
    using namespace rocwmma;

    template <uint32_t TargetId,
              typename InputT,
              typename ComputeT,
              uint32_t BlockKTest,
              uint32_t ExpectedBlockK>
    constexpr bool selects()
    {
        using Op = SelectedWmmaFor<TargetId, InputT, InputT, ComputeT, 16u, 16u, BlockKTest>;
        return MmaTraits<Op>::is_supported && MmaTraits<Op>::BlockK == ExpectedBlockK;
    }

    static_assert(selects<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float32_t, 64u, 16u>());
    static_assert(selects<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int32_t, 64u, 16u>());
    static_assert(selects<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, int32_t, 8u, 8u>());
    static_assert(selects<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float32_t, 64u, 16u>());
    static_assert(selects<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float32_t, 64u, 32u>());
    static_assert(selects<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float16_t, 256u, 128u>());
    static_assert(!MmaTraits<SelectedWmmaFor<Constants::AMDGCN_ARCH_ID_GFX942,
                                             float16_t,
                                             float16_t,
                                             float32_t,
                                             16u,
                                             16u,
                                             64u>>::is_supported);
}

TEST(WmmaBuiltinAvailability, MetadataWithoutBuiltins)
{
    // All checks are static_asserts, evaluated in every compilation pass.
    SUCCEED();
}
