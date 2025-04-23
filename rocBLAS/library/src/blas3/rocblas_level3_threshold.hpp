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

/*********************************************************************syrk**********************************************************************/
// Threshold values of N and K in gfx942 to get better performance
constexpr int zsyrk_gfx942_n_higher_threshold  = 3000;
constexpr int sdsyrk_gfx942_n_higher_threshold = 4000;
constexpr int csyrk_gfx942_n_higher_threshold  = 1600;
constexpr int syrk_k_lower_threshold           = 500;

// Threshold values of N and K in gfx90a to get better performance
constexpr int sdsyrk_gfx90a_n_higher_threshold = 3000;
constexpr int czsyrk_gfx90a_n_higher_threshold = 2500;

/*********************************************************************dgmm**********************************************************************/
// Threshold values of M in gfx942 to get better performance
constexpr int dcdgmm_gfx942_m_lower_threshold = 2500;
