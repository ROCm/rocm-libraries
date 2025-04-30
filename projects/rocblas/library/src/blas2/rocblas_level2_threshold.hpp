/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#include "rocblas.h"

/*********************************************************************gemv**********************************************************************/

// Threshold values of (M, N) in gfx908 and gfx906 below which the threads per block should be 512 or less to get better performance
constexpr int gemvn_gfx908_threshold        = 15000;
constexpr int zgemvn_gfx908_threshold       = 18000;
constexpr int gemvn_gfx906_threshold        = 6000;
constexpr int dgemvn_gfx906_lower_threshold = 15000;
constexpr int dgemvn_gfx906_upper_threshold = 24000;
constexpr int gemvt_threshold               = 6000;

// Threshold values of (M, N) in gfx10, gfx11, gfx12
constexpr int sgemvt_gfx_arch_10_11_12_threshold = 4000;

// small mn large batched thresholds
constexpr int gemv_sm_mn_gfx90a_batch_min_threshold = 256;
constexpr int gemv_sm_mn_gfx942_batch_min_threshold = 1024;

// Double buffered load optimized for single and double precision for gemv (transpose)
constexpr int sgemvt_gfx908_lower_threshold = 7000;
constexpr int dgemvt_gfx908_lower_threshold = 3000;

constexpr int sgemvn_gfx942_double_buffered_higher_threshold = 30000;
constexpr int dgemvn_gfx942_double_buffered_higher_threshold = 40000;

/*********************************************************************symv**********************************************************************/

// Double buffered load optimized for single and double precision for symv (upper)
constexpr int ssymv_U_gfx908_gfx90a_higher_threshold = 22000;
constexpr int dsymv_U_gfx908_higher_threshold        = 23000;
constexpr int dsymv_U_gfx90a_higher_threshold        = 16000;

// Double buffered load optimized for single and double precision for symv (lower)
constexpr int ssymv_L_gfx90a_higher_threshold = 29000;
constexpr int dsymv_L_gfx90a_higher_threshold = 20000;

// Double buffered load optimized for double precision for symv (lower) generic cases
constexpr int dsymv_L_gfx90a_generic_higher_threshold = 26000;

// Double buffered load optimized for double precision for symv (upper) generic cases
constexpr int dsymv_U_gfx908_generic_higher_threshold = 14000;
constexpr int dsymv_U_gfx908_generic_lower_threshold  = 19000;
