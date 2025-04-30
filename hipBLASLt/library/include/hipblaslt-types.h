/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

/*! \file
 * \brief hipblaslt-types.h defines data types used by hipblaslt
 */

#pragma once
#ifndef _HIPBLASLT_TYPES_H_
#define _HIPBLASLT_TYPES_H_

#if (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 2 && HIP_VERSION_PATCH > 42130) \
	|| (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 3)//tmp before gfx94 use hip f8 header

#define HIPBLASLT_USE_F8_FNUZ_BC 1 // Always use custom impl for now
#define HIPBLASLT_USE_F8_OCP_BC 0 // Always use ocp impl for hip header
#define ROCM_USE_FLOAT8 // TODO: Remove when bc is not needed

#if defined(__HIPCC__)
#include <hip/hip_fp8.h>
#define HIPBLASLT_FP8_TYPE_FNUZ HIP_FP8_TYPE_FNUZ
#define HIPBLASLT_FP8_TYPE_OCP HIP_FP8_TYPE_OCP
#endif

#else // HIP_VERSION Check

#define HIPBLASLT_USE_F8_FNUZ_BC 1
#define HIPBLASLT_USE_F8_OCP_BC 1

#if !defined(HIP_FP8_TYPE_FNUZ)
#define HIPBLASLT_FP8_TYPE_FNUZ 1
#endif

#if !defined(HIP_FP8_TYPE_OCP)
#define HIPBLASLT_FP8_TYPE_OCP 0
#endif

#endif // HIP_VERSION Check

#include "hipblaslt_float8_bc.h"
#include "hipblaslt_float8.h"
#include <float.h>

// Generic API

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Single precision floating point type */
typedef float hipblasLtFloat;

#ifdef ROCM_USE_FLOAT16
typedef _Float16 hipblasLtHalf;
#else
/*! \brief Structure definition for hipblasLtHalf */
typedef struct _hipblasLtHalf
{
    uint16_t data;
} hipblasLtHalf;
#endif

typedef hip_bfloat16 hipblasLtBfloat16;

typedef int8_t  hipblasLtInt8;
typedef int32_t hipblasLtInt32;

#ifdef __cplusplus
}
#endif

#endif /* _HIPBLASLT_TYPES_H_ */
