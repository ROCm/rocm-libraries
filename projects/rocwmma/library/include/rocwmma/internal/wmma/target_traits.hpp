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
#ifndef ROCWMMA_WMMA_TARGET_TRAITS_HPP
#define ROCWMMA_WMMA_TARGET_TRAITS_HPP

#include "../constants.hpp"
#include "descriptor.hpp"
#include "native_gfx11.hpp"
#include "native_gfx12.hpp"
#include "native_gfx125x.hpp"

// Which native WMMA operations each target has. A target gets an explicit list;
// targets share a list only when their capabilities are identical.
namespace rocwmma
{
    namespace detail
    {
        using WmmaGfx11Ops = wmma_op_list<wmma_f16_16x16x16_f16_gfx11,
                                          wmma_f32_16x16x16_f16_gfx11,
                                          wmma_f32_16x16x16_bf16_gfx11,
                                          wmma_bf16_16x16x16_bf16_gfx11,
                                          wmma_i32_16x16x16_i8_gfx11>;

        // gfx117x and gfx12
        using WmmaGfx12Ops = wmma_op_list<wmma_f32_16x16x16_f16_gfx12,
                                          wmma_i32_16x16x16_u8_gfx12,
                                          wmma_f16_16x16x16_f16_gfx12,
                                          wmma_f32_16x16x16_bf16_gfx12,
                                          wmma_bf16_16x16x16_bf16_gfx12,
                                          wmma_i32_16x16x16_i8_gfx12>;

        using WmmaGfx1250Ops = wmma_op_list<wmma_f32_16x16x32_f16_gfx1250,
                                            wmma_bf16_16x16x32_bf16_gfx1250,
                                            wmma_f16_16x16x32_f16_gfx1250,
                                            wmma_f32_16x16x32_bf16_gfx1250,
                                            wmma_i32_16x16x64_i8_gfx1250,
                                            wmma_i32_16x16x64_u8_gfx1250,
                                            wmma_f32_16x16x4_f32_gfx1250>;

        template <uint32_t TargetId>
        struct wmma_target_ops
        {
            using type = wmma_op_list<>;
        };

        // clang-format off
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1100> { using type = WmmaGfx11Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1101> { using type = WmmaGfx11Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1102> { using type = WmmaGfx11Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1103> { using type = WmmaGfx11Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1150> { using type = WmmaGfx11Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1151> { using type = WmmaGfx11Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1152> { using type = WmmaGfx11Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1153> { using type = WmmaGfx11Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1170> { using type = WmmaGfx12Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1171> { using type = WmmaGfx12Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1200> { using type = WmmaGfx12Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1201> { using type = WmmaGfx12Ops; };
        template <> struct wmma_target_ops<Constants::AMDGCN_ARCH_ID_GFX1250> { using type = WmmaGfx1250Ops; };
        // clang-format on

        template <uint32_t TargetId>
        using wmma_target_ops_t = typename wmma_target_ops<TargetId>::type;

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_TARGET_TRAITS_HPP
