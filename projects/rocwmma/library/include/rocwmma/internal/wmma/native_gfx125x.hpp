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
#ifndef ROCWMMA_WMMA_NATIVE_GFX125X_HPP
#define ROCWMMA_WMMA_NATIVE_GFX125X_HPP

#include "descriptor.hpp"
#include "register_abi.hpp"

// Native WMMA operations of the gfx1250 interface: larger K, A/B/C input
// modifiers and operand-reuse hints. 16-bit accumulators are packed.
namespace rocwmma
{
    namespace detail
    {
        struct wmma_f32_16x16x32_f16_gfx1250
        {
            using Key       = wmma_key<float16_t, float16_t, float32_t, 16u, 16u, 32u>;
            using PaddableK = wmma_k_list<4u, 8u, 16u>;

            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            constexpr static WmmaInputNeg NegA   = WmmaInputNeg::None;
            constexpr static WmmaInputNeg NegB   = WmmaInputNeg::None;
            constexpr static WmmaCMod     CMod   = WmmaCMod::None;
            constexpr static WmmaReuse    ReuseA = WmmaReuse::None;
            constexpr static WmmaReuse    ReuseB = WmmaReuse::None;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f32_16x16x32_f16)
                using F16x16 = wmma_native_vec_t<float16_t, 16>;
                using F32x8  = wmma_native_vec_t<float32_t, 8>;

                DRegsT result;
                wmma_abi::as<F32x8>(result)
                    = __builtin_amdgcn_wmma_f32_16x16x32_f16((bool)NegA,
                                                             wmma_abi::as<F16x16>(regsA),
                                                             (bool)NegB,
                                                             wmma_abi::as<F16x16>(regsB),
                                                             (int16_t)CMod,
                                                             wmma_abi::as<F32x8>(regsC),
                                                             (bool)ReuseA,
                                                             (bool)ReuseB);
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_f32_16x16x32_f16 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_bf16_16x16x32_bf16_gfx1250
        {
            using Key       = wmma_key<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 32u>;
            using PaddableK = wmma_k_list<4u, 8u, 16u>;

            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            constexpr static WmmaInputNeg NegA   = WmmaInputNeg::None;
            constexpr static WmmaInputNeg NegB   = WmmaInputNeg::None;
            constexpr static WmmaCMod     CMod   = WmmaCMod::None;
            constexpr static WmmaReuse    ReuseA = WmmaReuse::None;
            constexpr static WmmaReuse    ReuseB = WmmaReuse::None;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_bf16_16x16x32_bf16)
                using BF16x16 = wmma_native_vec_t<__bf16, 16>;
                using BF16x8  = wmma_native_vec_t<__bf16, 8>;

                DRegsT result;
                wmma_abi::as<BF16x8>(result)
                    = __builtin_amdgcn_wmma_bf16_16x16x32_bf16((bool)NegA,
                                                               wmma_abi::as<BF16x16>(regsA),
                                                               (bool)NegB,
                                                               wmma_abi::as<BF16x16>(regsB),
                                                               (int16_t)CMod,
                                                               wmma_abi::as<BF16x8>(regsC),
                                                               (bool)ReuseA,
                                                               (bool)ReuseB);
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_bf16_16x16x32_bf16 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_f16_16x16x32_f16_gfx1250
        {
            using Key       = wmma_key<float16_t, float16_t, float16_t, 16u, 16u, 32u>;
            using PaddableK = wmma_k_list<4u, 8u, 16u>;

            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            constexpr static WmmaInputNeg NegA   = WmmaInputNeg::None;
            constexpr static WmmaInputNeg NegB   = WmmaInputNeg::None;
            constexpr static WmmaCMod     CMod   = WmmaCMod::None;
            constexpr static WmmaReuse    ReuseA = WmmaReuse::None;
            constexpr static WmmaReuse    ReuseB = WmmaReuse::None;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f16_16x16x32_f16)
                using F16x16 = wmma_native_vec_t<float16_t, 16>;
                using F16x8  = wmma_native_vec_t<float16_t, 8>;

                DRegsT result;
                wmma_abi::as<F16x8>(result)
                    = __builtin_amdgcn_wmma_f16_16x16x32_f16((bool)NegA,
                                                             wmma_abi::as<F16x16>(regsA),
                                                             (bool)NegB,
                                                             wmma_abi::as<F16x16>(regsB),
                                                             (int16_t)CMod,
                                                             wmma_abi::as<F16x8>(regsC),
                                                             (bool)ReuseA,
                                                             (bool)ReuseB);
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_f16_16x16x32_f16 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_f32_16x16x32_bf16_gfx1250
        {
            using Key       = wmma_key<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 32u>;
            using PaddableK = wmma_k_list<4u, 8u, 16u>;

            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            constexpr static WmmaInputNeg NegA   = WmmaInputNeg::None;
            constexpr static WmmaInputNeg NegB   = WmmaInputNeg::None;
            constexpr static WmmaCMod     CMod   = WmmaCMod::None;
            constexpr static WmmaReuse    ReuseA = WmmaReuse::None;
            constexpr static WmmaReuse    ReuseB = WmmaReuse::None;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f32_16x16x32_bf16)
                using BF16x16 = wmma_native_vec_t<__bf16, 16>;
                using F32x8   = wmma_native_vec_t<float32_t, 8>;

                DRegsT result;
                wmma_abi::as<F32x8>(result)
                    = __builtin_amdgcn_wmma_f32_16x16x32_bf16((bool)NegA,
                                                              wmma_abi::as<BF16x16>(regsA),
                                                              (bool)NegB,
                                                              wmma_abi::as<BF16x16>(regsB),
                                                              (int16_t)CMod,
                                                              wmma_abi::as<F32x8>(regsC),
                                                              (bool)ReuseA,
                                                              (bool)ReuseB);
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_f32_16x16x32_bf16 for this target");
                return DRegsT{};
#endif
            }
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_NATIVE_GFX125X_HPP
