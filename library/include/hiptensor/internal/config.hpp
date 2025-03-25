/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef HIPTENSOR_CONFIG_HPP
#define HIPTENSOR_CONFIG_HPP

namespace hiptensor
{

///
/// Architecture support
/// Guaranteed symbols:
/// HIPTENSOR_ARCH_GFX908
/// HIPTENSOR_ARCH_GFX90a
/// HIPTENSOR_ARCH_GFX942
/// HIPTENSOR_ARCH_GFX950
#if defined(__gfx908__)
#define HIPTENSOR_ARCH_GFX908 __gfx908__
#elif defined(__gfx90a__)
#define HIPTENSOR_ARCH_GFX90A __gfx90a__
#elif defined(__gfx942__)
#define HIPTENSOR_ARCH_GFX942 __gfx942__
#elif defined(__gfx950__)
#define HIPTENSOR_ARCH_GFX950 __gfx950__
#else
#define HIPTENSOR_ARCH_HOST 1
#endif

#if !defined(HIPTENSOR_ARCH_GFX908)
#define HIPTENSOR_ARCH_GFX908 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX90A)
#define HIPTENSOR_ARCH_GFX90A 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX942)
#define HIPTENSOR_ARCH_GFX942 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX950)
#define HIPTENSOR_ARCH_GFX950 0
#endif
#if !defined(HIPTENSOR_ARCH_HOST)
#define HIPTENSOR_ARCH_HOST 0
#endif

#if defined(NDEBUG)
#define HIPTENSOR_UNSUPPORTED_IMPL(MSG)
#else
#define HIPTENSOR_UNSUPPORTED_IMPL(MSG) __attribute__((deprecated(MSG)))
#endif

#if defined(HIP_NO_HALF)
#define HIPTENSOR_NO_HALF 1
#else
#define HIPTENSOR_NO_HALF 0
#endif // HIP_NO_HALF

#if HIPTENSOR_NO_HALF || (!HIPTENSOR_NO_HALF && defined(__HIP_NO_HALF_CONVERSIONS__))
#define HIPTENSOR_TESTS_NO_HALF 1
#else
#define HIPTENSOR_TESTS_NO_HALF 0
#endif // !HIPTENSOR_NO_HALF && defined(__HIP_NO_HALF_CONVERSIONS__)

///
/// Host and Device symbols
///
#define HIPTENSOR_DEVICE __device__

#define HIPTENSOR_HOST __host__

#define HIPTENSOR_HOST_DEVICE HIPTENSOR_HOST HIPTENSOR_DEVICE

#define HIPTENSOR_KERNEL __global__

} // namespace hiptensor

#endif // HIPTENSOR_CONFIG_HPP
