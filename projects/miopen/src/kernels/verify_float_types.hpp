/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include "bfloat16_dev.hpp"

#ifndef MIOPEN_VERIFY_USE_DOUBLE_ACCUM
#define MIOPEN_VERIFY_USE_DOUBLE_ACCUM 0
#endif

#ifndef MIOPEN_UUT_USE_FP64
#define MIOPEN_UUT_USE_FP64 0
#endif

#ifndef MIOPEN_UUT_USE_FP32
#define MIOPEN_UUT_USE_FP32 0
#endif

#ifndef MIOPEN_UUT_USE_FP16
#define MIOPEN_UUT_USE_FP16 0
#endif

#ifndef MIOPEN_UUT_USE_BFP16
#define MIOPEN_UUT_USE_BFP16 0
#endif

#ifndef MIOPEN_REF_USE_FP64
#define MIOPEN_REF_USE_FP64 0
#endif

#ifndef MIOPEN_REF_USE_FP32
#define MIOPEN_REF_USE_FP32 0
#endif

#ifndef MIOPEN_REF_USE_FP16
#define MIOPEN_REF_USE_FP16 0
#endif

#ifndef MIOPEN_REF_USE_BFP16
#define MIOPEN_REF_USE_BFP16 0
#endif

#if MIOPEN_VERIFY_USE_DOUBLE_ACCUM == 1
#define FLOAT_ACCUM double
#else
#define FLOAT_ACCUM float
#endif

// Have to make sure that if uut/ref are double that verifying result will be also double
#if MIOPEN_UUT_USE_FP64 == 1
#define FLOAT_UUT double
#define CVT_FLOAT_UUT2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#endif

#if MIOPEN_UUT_USE_FP32 == 1
#define FLOAT_UUT float
#define CVT_FLOAT_UUT2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#endif

#if MIOPEN_UUT_USE_FP16 == 1
#define FLOAT_UUT _Float16
#define CVT_FLOAT_UUT2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#endif

#if MIOPEN_UUT_USE_BFP16 == 1
#define FLOAT_UUT ushort
#define CVT_FLOAT_UUT2ACCUM(x) (bfloat16_to_float(x))
#endif

// Have to make sure that if uut/ref are double that verifying result will be also double
#if MIOPEN_REF_USE_FP64 == 1
#define FLOAT_REF double
#define CVT_FLOAT_REF2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#endif

#if MIOPEN_REF_USE_FP32 == 1
#define FLOAT_REF float
#define CVT_FLOAT_REF2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#endif

#if MIOPEN_REF_USE_FP16 == 1
#define FLOAT_REF _Float16
#define CVT_FLOAT_REF2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#endif

#if MIOPEN_REF_USE_BFP16 == 1
#define FLOAT_REF ushort
#define CVT_FLOAT_REF2ACCUM(x) (bfloat16_to_float(x))
#endif
