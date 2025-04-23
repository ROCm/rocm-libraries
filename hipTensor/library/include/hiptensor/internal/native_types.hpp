/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef HIPTENSOR_NATIVE_TYPES_HPP
#define HIPTENSOR_NATIVE_TYPES_HPP

#include <array>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_vector_types.h>
#include <type_traits>
#include <utility>

namespace hiptensor
{

    /**
 * \defgroup DataTypes Data Type Metadata
 *
 * @brief Definition and metadata on supported data types of matrices.
 *
 * @{
 *
 * Native Data Types:
 * float64_t = f64 = double
 * float = f32
 * _Float16 = f16
 * int8
 * uint8
 * int16
 * int32
 * uint32
 *
 *
 * Non-Native Data Types:
 * h16 = __half
 * bf16 = bfloat16
 *
 */

    // Native types
    using float16_t = _Float16;
    using float32_t = float;
    using float64_t = double;

    using int8_t    = ::int8_t;
    using uint8_t   = ::uint8_t;
    using int16_t   = ::int16_t;
    using uint16_t  = ::uint16_t;
    using int32_t   = ::int32_t;
    using uint32_t  = ::uint32_t;
    using int64_t   = ::int64_t;
    using uint64_t  = ::uint64_t;
    using index_t   = ::int32_t;
    using index64_t = ::int64_t;

    // Non-native types
    using bfloat16_t = hip_bfloat16;

#if !HIPTENSOR_NO_HALF
    using hfloat16_t = __half;
#endif // !HIPTENSOR_NO_HALF
    // clang-format off


} // namespace hiptensor

// Add in some extensions to basic type support.
// Some of these are required for vector implementations.
// #include "type_traits.hpp"
// #include "types_ext.hpp"

#include "native_types_impl.hpp"

#endif // HIPTENSOR_NATIVE_TYPES_HPP
