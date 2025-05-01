/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <string>

#if defined(__clang__)
#if __has_feature(cxx_rtti)
#define __RTTI_ENABLED
#endif
#elif defined(__GNUG__)
#if defined(__GXX_RTTI)
#define __RTTI_ENABLED
#endif
#else
#undef __RTTI_ENABLED
#endif

namespace
{
    // TODO: It might be desirable to have a similar method implemented for
    // RR types that do not rely on RTTI. In that case, RTTI could be
    // disabled for RR code.
    template <typename T>
    inline std::string typeName()
    {
#ifdef __RTTI_ENABLED
        return typeid(T).name();
#else
        return "unknown type (RTTI is disabled)";
#endif
    }
}
