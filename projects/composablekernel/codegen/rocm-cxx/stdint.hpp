/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef ROCM_GUARD_ROCM_STDINT_HPP
#define ROCM_GUARD_ROCM_STDINT_HPP

#include <rocm/config.hpp>

#ifndef __HIPCC_RTC__
#include <cstdint>
#endif

namespace rocm {
inline namespace ROCM_INLINE_NS {

#ifdef __HIPCC_RTC__
using int8_t   = __hip_int8_t;
using uint8_t  = __hip_uint8_t;
using int16_t  = __hip_int16_t;
using uint16_t = __hip_uint16_t;
using int32_t  = __hip_int32_t;
using uint32_t = __hip_uint32_t;
using int64_t  = __hip_int64_t;
using uint64_t = __hip_uint64_t;
#else
using int8_t   = std::int8_t;
using uint8_t  = std::uint8_t;
using int16_t  = std::int16_t;
using uint16_t = std::uint16_t;
using int32_t  = std::int32_t;
using uint32_t = std::uint32_t;
using int64_t  = std::int64_t;
using uint64_t = std::uint64_t;
#endif

using intptr_t  = int64_t;
using uintptr_t = uint64_t;
using intmax_t  = int64_t;
using uintmax_t = uint64_t;
using intmax_t  = int64_t;

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_STDINT_HPP
