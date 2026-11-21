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
#ifndef ROCWMMA_WMMA_NATIVE_GFX12_HPP
#define ROCWMMA_WMMA_NATIVE_GFX12_HPP

#include "descriptor.hpp"
#include "register_abi.hpp"

// Native WMMA operations of the gfx12 interface (_w32_gfx12 builtins), used by
// gfx117x and gfx12 targets.
namespace rocwmma
{
    namespace detail
    {
        struct wmma_f32_16x16x16_f16_gfx12
        {
            using Key       = wmma_key<float16_t, float16_t, float32_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<4u, 8u>;

            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f32_16x16x16_f16_w32_gfx12)
                using F16x8 = wmma_native_vec_t<float16_t, 8>;
                using F32x8 = wmma_native_vec_t<float32_t, 8>;

                DRegsT result;
                wmma_abi::as<F32x8>(result)
                    = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(wmma_abi::as<F16x8>(regsA),
                                                                       wmma_abi::as<F16x8>(regsB),
                                                                       wmma_abi::as<F32x8>(regsC));
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

        // Unsigned 8-bit inputs. Shares its builtin with the int8 operation; the
        // packed bytes travel in signed words.
        struct wmma_i32_16x16x16_u8_gfx12
        {
            using Key       = wmma_key<uint8_t, uint8_t, int32_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<8u>;

            using ARegsT = VRegUI32x2;
            using BRegsT = VRegUI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            constexpr static WmmaInputSign InputSign = WmmaInputSign::Unsigned;
            // Wrap on int32 overflow (no saturation), as on every other target
            constexpr static bool Clamp = false;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_i32_16x16x16_iu8_w32_gfx12)
                using I32x2 = wmma_native_vec_t<int32_t, 2>;
                using I32x8 = wmma_native_vec_t<int32_t, 8>;

                DRegsT result;
                wmma_abi::as<I32x8>(result)
                    = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12((bool)InputSign,
                                                                       wmma_abi::as<I32x2>(regsA),
                                                                       (bool)InputSign,
                                                                       wmma_abi::as<I32x2>(regsB),
                                                                       wmma_abi::as<I32x8>(regsC),
                                                                       Clamp);
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_f16_16x16x16_f16_gfx12
        {
            using Key       = wmma_key<float16_t, float16_t, float16_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<4u, 8u>;

            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f16_16x16x16_f16_w32_gfx12)
                using F16x8 = wmma_native_vec_t<float16_t, 8>;

                DRegsT result;
                wmma_abi::as<F16x8>(result)
                    = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(wmma_abi::as<F16x8>(regsA),
                                                                       wmma_abi::as<F16x8>(regsB),
                                                                       wmma_abi::as<F16x8>(regsC));
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_f32_16x16x16_bf16_gfx12
        {
            using Key       = wmma_key<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<4u, 8u>;

            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f32_16x16x16_bf16_w32_gfx12)
                // The gfx12 bf16 builtins take raw 16-bit integers
                using S16x8 = wmma_native_vec_t<int16_t, 8>;
                using F32x8 = wmma_native_vec_t<float32_t, 8>;

                DRegsT result;
                wmma_abi::as<F32x8>(result)
                    = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(wmma_abi::as<S16x8>(regsA),
                                                                        wmma_abi::as<S16x8>(regsB),
                                                                        wmma_abi::as<F32x8>(regsC));
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_bf16_16x16x16_bf16_gfx12
        {
            using Key       = wmma_key<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<4u, 8u>;

            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_bf16_16x16x16_bf16_w32_gfx12)
                // The gfx12 bf16 builtins take raw 16-bit integers
                using S16x8 = wmma_native_vec_t<int16_t, 8>;

                DRegsT result;
                wmma_abi::as<S16x8>(result) = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(
                    wmma_abi::as<S16x8>(regsA),
                    wmma_abi::as<S16x8>(regsB),
                    wmma_abi::as<S16x8>(regsC));
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

        // Signed 8-bit inputs. Shares its builtin with the uint8 operation.
        struct wmma_i32_16x16x16_i8_gfx12
        {
            using Key       = wmma_key<int8_t, int8_t, int32_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<8u>;

            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            constexpr static WmmaInputSign InputSign = WmmaInputSign::Signed;
            // Wrap on int32 overflow (no saturation), as on every other target
            constexpr static bool Clamp = false;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_i32_16x16x16_iu8_w32_gfx12)
                using I32x2 = wmma_native_vec_t<int32_t, 2>;
                using I32x8 = wmma_native_vec_t<int32_t, 8>;

                DRegsT result;
                wmma_abi::as<I32x8>(result)
                    = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12((bool)InputSign,
                                                                       wmma_abi::as<I32x2>(regsA),
                                                                       (bool)InputSign,
                                                                       wmma_abi::as<I32x2>(regsB),
                                                                       wmma_abi::as<I32x8>(regsC),
                                                                       Clamp);
                return result;
#else
                static_assert(!ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                              "rocWMMA: the compiler does not provide "
                              "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_f32_16x16x16_fp8_fp8_gfx12
        {
            using Key       = wmma_key<float8_t, float8_t, float32_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<8u>;

            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f32_16x16x16_fp8_fp8_w32_gfx12)
                // The builtin takes the packed fp8 data as ints
                using I32x2 = wmma_native_vec_t<int, 2>;
                using F32x8 = wmma_native_vec_t<float32_t, 8>;

                DRegsT result;
                wmma_abi::as<F32x8>(result) = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    wmma_abi::as<I32x2>(regsA),
                    wmma_abi::as<I32x2>(regsB),
                    wmma_abi::as<F32x8>(regsC));
                return result;
#else
                static_assert(
                    !ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                    "rocWMMA: the compiler does not provide "
                    "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_f32_16x16x16_bf8_bf8_gfx12
        {
            using Key       = wmma_key<bfloat8_t, bfloat8_t, float32_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<8u>;

            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f32_16x16x16_bf8_bf8_w32_gfx12)
                // The builtin takes the packed fp8 data as ints
                using I32x2 = wmma_native_vec_t<int, 2>;
                using F32x8 = wmma_native_vec_t<float32_t, 8>;

                DRegsT result;
                wmma_abi::as<F32x8>(result) = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                    wmma_abi::as<I32x2>(regsA),
                    wmma_abi::as<I32x2>(regsB),
                    wmma_abi::as<F32x8>(regsC));
                return result;
#else
                static_assert(
                    !ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                    "rocWMMA: the compiler does not provide "
                    "__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_f32_16x16x16_fp8_bf8_gfx12
        {
            using Key       = wmma_key<float8_t, bfloat8_t, float32_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<>;

            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f32_16x16x16_fp8_bf8_w32_gfx12)
                // The builtin takes the packed fp8 data as ints
                using I32x2 = wmma_native_vec_t<int, 2>;
                using F32x8 = wmma_native_vec_t<float32_t, 8>;

                DRegsT result;
                wmma_abi::as<F32x8>(result) = __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
                    wmma_abi::as<I32x2>(regsA),
                    wmma_abi::as<I32x2>(regsB),
                    wmma_abi::as<F32x8>(regsC));
                return result;
#else
                static_assert(
                    !ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                    "rocWMMA: the compiler does not provide "
                    "__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

        struct wmma_f32_16x16x16_bf8_fp8_gfx12
        {
            using Key       = wmma_key<bfloat8_t, float8_t, float32_t, 16u, 16u, 16u>;
            using PaddableK = wmma_k_list<>;

            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            template <typename Dummy = void>
            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
#if ROCWMMA_HAS_AMDGCN_BUILTIN(wmma_f32_16x16x16_bf8_fp8_w32_gfx12)
                // The builtin takes the packed fp8 data as ints
                using I32x2 = wmma_native_vec_t<int, 2>;
                using F32x8 = wmma_native_vec_t<float32_t, 8>;

                DRegsT result;
                wmma_abi::as<F32x8>(result) = __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
                    wmma_abi::as<I32x2>(regsA),
                    wmma_abi::as<I32x2>(regsB),
                    wmma_abi::as<F32x8>(regsC));
                return result;
#else
                static_assert(
                    !ROCWMMA_DEVICE_COMPILE || wmma_dependent_false_v<Dummy>,
                    "rocWMMA: the compiler does not provide "
                    "__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12 for this target");
                return DRegsT{};
#endif
            }
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_NATIVE_GFX12_HPP
