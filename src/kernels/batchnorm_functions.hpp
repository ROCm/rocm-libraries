/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_USE_FPMIX
#define MIOPEN_USE_FPMIX 0
#endif

#ifndef MIOPEN_USE_BFPMIX
#define MIOPEN_USE_BFPMIX 0
#endif

#include "bfloat16_dev.hpp"

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#ifndef MIOPEN_USE_FPMIX
#define MIOPEN_USE_FPMIX 0
#endif

#ifndef MIOPEN_USE_BFPMIX
#define MIOPEN_USE_BFPMIX 0
#endif

#define _FLOAT_ACCUM float

#if MIOPEN_USE_FP16 == 1
    #define FP_TYPE half
    #define FP_TYPE_PREC float
    #define EPSILON static_cast<FP_TYPE>(0.0001)
    #ifndef HALF_MAX
    #define MAX_VAL 65504 /* max value */
    #else
    #define MAX_VAL HALF_MAX
    #endif
#endif
#if MIOPEN_USE_FP32 == 1
    #define FP_TYPE float
    #define FP_TYPE_PREC float
    #define EPSILON static_cast<FP_TYPE>(0.000001)
    #ifndef FLT_MAX
    #define MAX_VAL 3.402823466e+38F /* max value */
    #else
    #define MAX_VAL FLT_MAX
    #endif
#endif

#if MIOPEN_USE_FPMIX == 1
    #define FP_TYPE half
    #ifdef MIO_BN_NODPP
    #undef MIO_BN_NODPP
    #define MIO_BN_NODPP 0
    #endif

    #ifdef FP_TYPE_PREC
    #undef FP_TYPE_PREC
    #endif
    #define FP_TYPE_PREC float

    #ifdef EPSILON
    #undef EPSILON
    #endif
    #define EPSILON static_cast<FP_TYPE>(0.000001)

#endif

#if MIOPEN_USE_BFPMIX == 1
    // Enables bfloat16, stored in ushort
    #define FP_TYPE uint16_t

    #ifdef MIO_BN_NODPP
    #undef MIO_BN_NODPP
    #define MIO_BN_NODPP 0
    #endif

    #ifdef FP_TYPE_PREC
    #undef FP_TYPE_PREC
    #endif
    #define FP_TYPE_PREC float

    #ifdef EPSILON
    #undef EPSILON
    #endif
    #define EPSILON static_cast<FP_TYPE_PREC>(0.000001)

    // TODO bfloat16_to_float and float_to_bfloat16 is not directly supported in HIP
    #define FLOAT2FLOATPREC(x) (bfloat16_to_float(x))
    #define FLOATPREC2FLOAT(x) (float_to_bfloat16(x))

    #define FLOAT2ACCUM(x) (FLOAT2FLOATPREC(x))
    #define ACCUM2FLOAT(x) (FLOATPREC2FLOAT(x))

#else
    #define FLOAT2FLOATPREC(x) (static_cast<FP_TYPE_PREC>(x))
    #define FLOATPREC2FLOAT(x) (static_cast<FP_TYPE>(x))
    #define FLOAT2ACCUM(x) (static_cast<_FLOAT_ACCUM>(x))
    #define ACCUM2FLOAT(x) (static_cast<FP_TYPE>(x))
#endif

#define _FLOAT2 PPCAT(FP_TYPE, TWO)
#define _FLOAT4 PPCAT(FP_TYPE, FOUR)
#define _FLOAT8 PPCAT(FP_TYPE, EIGHT)
#define _AS_FLOAT PPCAT(as_, FP_TYPE)

#define _FLOAT_PREC4 PPCAT(FP_TYPE_PREC, FOUR)
#define _FLOAT_ACCUM4 PPCAT(_FLOAT_ACCUM, FOUR)


// Env Configs
#ifndef MIO_BN_LDSGCN_SIZE
#define MIO_BN_LDSGCN_SIZE 16
#endif

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 256
#endif

#ifndef MIO_BN_C
#define MIO_BN_C 1
#endif

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

#ifndef MIO_BN_GRP0
#define MIO_BN_GRP0 1
#endif

#ifndef MIO_BN_GRP1
#define MIO_BN_GRP1 1
#endif

#ifndef MIO_BN_GRP2
#define MIO_BN_GRP2 1
#endif

#ifndef MIO_BN_NGRPS
#define MIO_BN_NGRPS 1
#endif

#ifndef MIO_BN_LOOP_UNROLL_MAXN
#define MIO_BN_LOOP_UNROLL_MAXN 768
#endif

#ifndef MIO_BN_LOOP_UNROLL_MAXHW
#define MIO_BN_LOOP_UNROLL_MAXHW 2500
#endif

#ifndef MIO_BN_NCHW
#define MIO_BN_NCHW 1
#endif

#ifndef MIO_BN_VARIANT
#define MIO_BN_VARIANT 255
#endif

#ifndef MIO_BN_MAXN
#define MIO_BN_MAXN 65
#endif

