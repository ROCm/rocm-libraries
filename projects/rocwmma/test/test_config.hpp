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
#ifndef ROCWMMA_TEST_CONFIG_HPP
#define ROCWMMA_TEST_CONFIG_HPP

#include <rocwmma/internal/config.hpp>

///
/// Testing symbols
///
#if defined(ROCWMMA_EXTENDED_TESTS)
#define ROCWMMA_EXTENDED_TESTS 1
#else
#define ROCWMMA_EXTENDED_TESTS 0
#endif

#if defined(ROCWMMA_VALIDATION_TESTS)
#define ROCWMMA_VALIDATION_TESTS 1
#else
#define ROCWMMA_VALIDATION_TESTS 0
#endif

#if defined(ROCWMMA_BENCHMARK_TESTS)
#define ROCWMMA_BENCHMARK_TESTS 1
#else
#define ROCWMMA_BENCHMARK_TESTS 0
#endif

#if defined(ROCWMMA_BENCHMARK_WITH_ROCBLAS)
#define ROCWMMA_BENCHMARK_WITH_ROCBLAS 1
#else
#define ROCWMMA_BENCHMARK_WITH_ROCBLAS 0
#endif

#if defined(ROCWMMA_VALIDATE_WITH_ROCBLAS)
#define ROCWMMA_VALIDATE_WITH_ROCBLAS 1
#else
#define ROCWMMA_VALIDATE_WITH_ROCBLAS 0
#endif

#if ROCWMMA_BENCHMARK_WITH_ROCBLAS || ROCWMMA_VALIDATE_WITH_ROCBLAS
#define ROCWMMA_ROCBLAS_INTEGRATION 1
#else
#define ROCWMMA_ROCBLAS_INTEGRATION 0
#endif

#if ROCWMMA_VALIDATION_TESTS && !ROCWMMA_VALIDATE_WITH_ROCBLAS
#define ROCWMMA_VALIDATE_WITH_CPU 1
#else
#define ROCWMMA_VALIDATE_WITH_CPU 0
#endif

#if ROCWMMA_NO_HALF || (!ROCWMMA_NO_HALF && defined(__HIP_NO_HALF_CONVERSIONS__))
#define ROCWMMA_TESTS_NO_HALF 1
#else
#define ROCWMMA_TESTS_NO_HALF 0
#endif // !ROCWMMA_NO_HALF && defined(__HIP_NO_HALF_CONVERSIONS__)

#endif // ROCWMMA_TEST_CONFIG_HPP
