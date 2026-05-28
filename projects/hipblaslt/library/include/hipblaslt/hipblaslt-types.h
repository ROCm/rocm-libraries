/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
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


#if defined(__HIPCC__)
#include <hip/hip_fp8.h>
#endif

#include "hipblaslt_float8.h"
#if defined(__HIP__)
#include "hipblaslt_bfloat6.h"
#include "hipblaslt_float6.h"
#include "hipblaslt_float4.h"
#endif
#include "hipblaslt_e8.h"
#include "hipblaslt_e5m3.h"
#include <float.h>
#include <complex>

// Generic API

#ifdef __cplusplus
typedef std::complex<float>  hipblaslt_complex_float;
typedef std::complex<double> hipblaslt_complex_double;
#endif

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

int const HIP_R_6F_E2M3_EXT = 31;
int const HIP_R_6F_E3M2_EXT = 32;
int const HIP_R_4F_E2M1_EXT = 33;
int const HIP_R_8F_E5M3_EXT = 34;

#if defined(ROCM_USE_FLOAT16) && defined(__cplusplus) && __cplusplus >= 201103L \
    && (defined(__HCC__) || defined(__HIPCC__))
#include <cmath>

// Mirrors HIP's __hip_cvt_float_to_fp8(float, saturation, interpretation):
// saturation is a parameter of the conversion primitive, not hardcoded,
// because callers may need to opt in only on architectures whose HW
// fp32->fp16 conversion saturates (e.g. gfx12 with .amdhsa_fp16_overflow=1).
//
// Two independent flags because two different HW paths drive saturation
// differently under MODE.FP16_OVFL=1:
//
//   saturate_finite   -- finite overflow clamps to +/-fp16_max.
//                        Governed by the VALU cvt rule, which "preserves true
//                        INF values" (Inf input -> Inf output). Holds on
//                        every gfx12 arch (gfx1200/1201/1250).
//
//   saturate_inf      -- +/-Inf inputs also clamp to +/-fp16_max.
//                        Governed by the WMMA/SWMMAC rule: with FP16_OVFL=1,
//                        "F16/BF16/...: overflow to +/-MAX instead of
//                        infinity" -- a *large result becomes +/-maxValue*,
//                        including the Inf-input case (the Inf product/sum
//                        is what gets replaced). Empirically verified on
//                        gfx1250: fp16 +Inf into v_wmma_f32_16x16x32_f16
//                        yields fp32_max in the accumulator, not Inf.
//                        Only set this true when the GPU GEMM path lowers
//                        to WMMA (currently gfx1250). gfx1200/1201 fp16
//                        kernels lower to VALU FMA which preserves Inf.
//
// NaN always passes through unchanged ("Any NaN input produces a NaN output,
// and 0*Inf produces NaN" per the same spec).
HIP_HOST_DEVICE inline hipblasLtHalf
    hipblaslt_cvt_float_to_half(float f, bool saturate_finite, bool saturate_inf)
{
    if(!std::isnan(f))
    {
        constexpr float fp16_max = 65504.0f;
        if(std::isinf(f))
        {
            if(saturate_inf)
                f = (f > 0) ? fp16_max : -fp16_max;
        }
        else if(saturate_finite)
        {
            if(f > fp16_max)
                f = fp16_max;
            else if(f < -fp16_max)
                f = -fp16_max;
        }
    }
    return static_cast<hipblasLtHalf>(f);
}

// Companion to hipblaslt_cvt_float_to_half for bf16 destinations. Same two-flag
// semantics; see comment above. WMMA path -> set both true; VALU FMA path ->
// set only saturate_finite.
HIP_HOST_DEVICE inline hipblasLtBfloat16
    hipblaslt_cvt_float_to_bfloat16(float f, bool saturate_finite, bool saturate_inf)
{
    if(!std::isnan(f))
    {
        // bf16 max bit pattern 0x7F7F: sign=0, exp=254, mantissa=0x7F.
        // Value = (1 + 127/128) * 2^127 ≈ 3.38953139e38.
        constexpr float bf16_max = 0x1.FEp+127f;
        if(std::isinf(f))
        {
            if(saturate_inf)
                f = (f > 0) ? bf16_max : -bf16_max;
        }
        else if(saturate_finite)
        {
            if(f > bf16_max)
                f = bf16_max;
            else if(f < -bf16_max)
                f = -bf16_max;
        }
    }
    return static_cast<hipblasLtBfloat16>(f);
}
#endif

#endif /* _HIPBLASLT_TYPES_H_ */
