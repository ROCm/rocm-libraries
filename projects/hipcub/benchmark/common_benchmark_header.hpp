// MIT License
//
// Copyright (c) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// Primbench
#include "primbench.hpp"

// Google Benchmark
#include "benchmark/benchmark.h"

// CmdParser
#include "cmdparser.hpp"

// HIP API
#include <hip/hip_runtime.h>

// benchmark_utils.hpp should only be included by this header.
// The following definition is used as guard in benchmark_utils.hpp
// Including benchmark_utils.hpp by itself will cause a compile error.
#define BENCHMARK_UTILS_INCLUDE_GUARD
#include "benchmark_utils.hpp"

#define HIP_CHECK(condition)                                                           \
    {                                                                                  \
        hipError_t error = condition;                                                  \
        if(error != hipSuccess)                                                        \
        {                                                                              \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error);                                                               \
        }                                                                              \
    }

PRIMBENCH_REGISTER_TYPE(int8_t, "i8")
PRIMBENCH_REGISTER_TYPE(int16_t, "i16")
PRIMBENCH_REGISTER_TYPE(int32_t, "i32")
PRIMBENCH_REGISTER_TYPE(int64_t, "i64")
PRIMBENCH_REGISTER_TYPE(uint8_t, "u8")
PRIMBENCH_REGISTER_TYPE(uint16_t, "u16")
PRIMBENCH_REGISTER_TYPE(uint32_t, "u32")
PRIMBENCH_REGISTER_TYPE(uint64_t, "u64")
PRIMBENCH_REGISTER_TYPE(float, "f32")
PRIMBENCH_REGISTER_TYPE(double, "f64")
PRIMBENCH_REGISTER_TYPE(long long, "i64")
PRIMBENCH_REGISTER_TYPE(unsigned long long, "u64")
PRIMBENCH_REGISTER_TYPE(__half, "f16")

using custom_int_t       = benchmark_utils::custom_type<int>;
using custom_float2      = benchmark_utils::custom_type<float, float>;
using custom_double2     = benchmark_utils::custom_type<double, double>;
using custom_char_double = benchmark_utils::custom_type<char, double>;
using custom_double_char = benchmark_utils::custom_type<double, char>;
using custom_int_double  = benchmark_utils::custom_type<int, double>;

PRIMBENCH_REGISTER_TYPE(custom_int_t, "custom<i32>");
PRIMBENCH_REGISTER_TYPE(custom_float2, "custom<f32,f32>")
PRIMBENCH_REGISTER_TYPE(custom_double2, "custom<f64,f64>")
PRIMBENCH_REGISTER_TYPE(custom_char_double, "custom<i8,f64>")
PRIMBENCH_REGISTER_TYPE(custom_double_char, "custom<f64,i8>")
PRIMBENCH_REGISTER_TYPE(custom_int_double, "custom<i32,f64>")
