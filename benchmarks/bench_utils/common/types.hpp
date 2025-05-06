// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef ROCTHRUST_BENCHMARKS_BENCH_UTILS_TYPES_HPP_
#define ROCTHRUST_BENCHMARKS_BENCH_UTILS_TYPES_HPP_

#include <cstdint>

// Types used in the benchmarks
#if(defined(__GNUC__) || defined(__clang__)) && (defined(__GLIBCXX__) || defined(_LIBCPP_VERSION))
#define THRUST_BENCHMARKS_HAVE_INT128_SUPPORT 1
#else
#define THRUST_BENCHMARKS_HAVE_INT128_SUPPORT 0
#endif

using int8_t   = std::int8_t;
using int16_t  = std::int16_t;
using int32_t  = std::int32_t;
using int64_t  = std::int64_t;
using uint8_t  = std::uint8_t;
using uint16_t = std::uint16_t;
using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;
#if THRUST_BENCHMARKS_HAVE_INT128_SUPPORT
using int128_t  = __int128_t;
using uint128_t = __uint128_t;
#endif
using float32_t = float;
using float64_t = double;

#endif // ROCTHRUST_BENCHMARKS_BENCH_UTILS_TYPES_HPP_