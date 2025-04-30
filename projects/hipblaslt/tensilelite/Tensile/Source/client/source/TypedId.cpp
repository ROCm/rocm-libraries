/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <TypedId.hpp>

namespace TensileLite
{
    template struct TypedGemm<float>;
    template struct TypedGemm<double>;
    template struct TypedGemm<std::complex<float>>;
    template struct TypedGemm<std::complex<double>>;
    template struct TypedGemm<Int8x4, Int8x4, int32_t, int32_t>;
    template struct TypedGemm<int32_t>;
    template struct TypedGemm<int8_t, int8_t, int8_t, int8_t, int32_t, int32_t>;
    template struct TypedGemm<int8_t, int8_t, int32_t, int32_t>;
    template struct TypedGemm<int8_t, int8_t, int32_t, int32_t, float, float>;
    template struct TypedGemm<int8_t, int8_t, int8_t, int8_t, float, float>;
#ifdef TENSILE_USE_HALF
    template struct TypedGemm<Half>;
    template struct TypedGemm<Half, Half, Half, Half, float, float>;
    template struct TypedGemm<Half, Half, float, float>;
#endif
#ifdef TENSILE_USE_BF16
    template struct TypedGemm<BFloat16, BFloat16, BFloat16, BFloat16, float, float>;
    template struct TypedGemm<BFloat16, BFloat16, float, float>;
#endif
#ifdef TENSILE_USE_FP8_BF8
    template struct TypedGemm<Half, Half, Half, Half, float, float, Float8>;
    template struct TypedGemm<Half, Half, Half, Half, float, float, Float8BFloat8>;
    template struct TypedGemm<Half, Half, Half, Half, float, float, BFloat8Float8>;
    template struct TypedGemm<Float8, Float8, Float8, Float8, float, float>;
    template struct TypedGemm<Float8_fnuz, Float8_fnuz, Float8_fnuz, Float8_fnuz, float, float>;
    template struct TypedGemm<Float8, Float8, BFloat8, BFloat8, float, float>;
    template struct TypedGemm<Float8_fnuz, Float8_fnuz, BFloat8_fnuz, BFloat8_fnuz, float, float>;
    template struct TypedGemm<Float8, Float8, float, float>;
    template struct TypedGemm<Float8_fnuz, Float8_fnuz, float, float>;
    template struct TypedGemm<BFloat8, BFloat8, BFloat8, BFloat8, float, float>;
    template struct TypedGemm<BFloat8_fnuz, BFloat8_fnuz, BFloat8_fnuz, BFloat8_fnuz, float, float>;
    template struct TypedGemm<BFloat8, BFloat8, Float8, Float8, float, float>;
    template struct TypedGemm<BFloat8_fnuz, BFloat8_fnuz, Float8_fnuz, Float8_fnuz, float, float>;
    template struct TypedGemm<BFloat8, BFloat8, float, float>;
    template struct TypedGemm<BFloat8_fnuz, BFloat8_fnuz, float, float>;
    // hybrid
    template struct TypedGemm<Float8, BFloat8, Float8, Float8, float, float>;
    template struct TypedGemm<Float8_fnuz, BFloat8_fnuz, Float8_fnuz, Float8_fnuz, float, float>;
    template struct TypedGemm<Float8, BFloat8, float, float>;
    template struct TypedGemm<Float8_fnuz, BFloat8_fnuz, float, float>;
    template struct TypedGemm<BFloat8, Float8, Float8, Float8, float, float>;
    template struct TypedGemm<BFloat8_fnuz, Float8_fnuz, Float8_fnuz, Float8_fnuz, float, float>;
    template struct TypedGemm<BFloat8, Float8, float, float>;
    template struct TypedGemm<BFloat8_fnuz, Float8_fnuz, float, float>;
    template struct TypedGemm<Float8, BFloat8, BFloat8, BFloat8, float, float>;
    template struct TypedGemm<Float8_fnuz, BFloat8_fnuz, BFloat8_fnuz, BFloat8_fnuz, float, float>;
    template struct TypedGemm<BFloat8, Float8, BFloat8, BFloat8, float, float>;
    template struct TypedGemm<BFloat8_fnuz, Float8_fnuz, BFloat8_fnuz, BFloat8_fnuz, float, float>;
#endif
} // namespace TensileLite
