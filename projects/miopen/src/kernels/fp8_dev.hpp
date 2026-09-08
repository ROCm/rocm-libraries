// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef FP8_DEVICE_HPP
#define FP8_DEVICE_HPP

#if defined(__HIP_PLATFORM_AMD__) && !defined(MIOPEN_USE_CLANG_TIDY) && !defined(CPPCHECK) && \
    !defined(MIOPEN_DISABLE_NATIVE_FP8_CONVERSION) &&                                         \
    ((defined(MIOPEN_USE_FP8) && MIOPEN_USE_FP8) || (defined(MIOPEN_USE_BFP8) && MIOPEN_USE_BFP8))
#include <hip/hip_fp8.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef EXECUTION_SPECIFIER
#ifdef __HIP_PLATFORM_AMD__
#define EXECUTION_SPECIFIER __device__
#else
#define EXECUTION_SPECIFIER
#endif
#endif

#ifndef MIOPEN_USE_FP8
#define MIOPEN_USE_FP8 0
#endif

#ifndef MIOPEN_USE_BFP8
#define MIOPEN_USE_BFP8 0
#endif

#if MIOPEN_USE_FP8 || MIOPEN_USE_BFP8

#if defined(__HIP_PLATFORM_AMD__) && !defined(MIOPEN_USE_CLANG_TIDY) && !defined(CPPCHECK) && \
    !defined(MIOPEN_DISABLE_NATIVE_FP8_CONVERSION) &&                                         \
    ((MIOPEN_FP8_IEEE_EXPONENT_BIAS && HIP_FP8_TYPE_OCP) ||                                   \
     (!MIOPEN_FP8_IEEE_EXPONENT_BIAS && HIP_FP8_TYPE_FNUZ))
#if MIOPEN_FP8_IEEE_EXPONENT_BIAS
#define MIOPEN_FP8_TYPE __hip_fp8_e4m3
#define MIOPEN_BFP8_TYPE __hip_fp8_e5m2
#define MIOPEN_FP8_INTERPRETATION __HIP_E4M3
#define MIOPEN_BFP8_INTERPRETATION __HIP_E5M2
#else
#define MIOPEN_FP8_TYPE __hip_fp8_e4m3_fnuz
#define MIOPEN_BFP8_TYPE __hip_fp8_e5m2_fnuz
#define MIOPEN_FP8_INTERPRETATION __HIP_E4M3_FNUZ
#define MIOPEN_BFP8_INTERPRETATION __HIP_E5M2_FNUZ
#endif

EXECUTION_SPECIFIER inline float fp8_to_float(uchar x)
{
    MIOPEN_FP8_TYPE value;
    value.__x = x;
    return static_cast<float>(value);
}

EXECUTION_SPECIFIER inline float bfp8_to_float(uchar x)
{
    MIOPEN_BFP8_TYPE value;
    value.__x = x;
    return static_cast<float>(value);
}

#if MIOPEN_FP8_CLIPPING == 1
#define MIOPEN_FP8_SATURATION __HIP_SATFINITE
#else
#define MIOPEN_FP8_SATURATION __HIP_NOSAT
#endif

EXECUTION_SPECIFIER inline uchar float_to_fp8(float x)
{
    return __hip_cvt_float_to_fp8(x, MIOPEN_FP8_SATURATION, MIOPEN_FP8_INTERPRETATION);
}

EXECUTION_SPECIFIER inline uchar float_to_bfp8(float x)
{
    return __hip_cvt_float_to_fp8(x, MIOPEN_FP8_SATURATION, MIOPEN_BFP8_INTERPRETATION);
}

#undef MIOPEN_BFP8_INTERPRETATION
#undef MIOPEN_FP8_INTERPRETATION
#undef MIOPEN_BFP8_TYPE
#undef MIOPEN_FP8_TYPE
#undef MIOPEN_FP8_SATURATION
#else
// Software fallback when the configured FP8 interpretation is unavailable in HIP.
typedef union
{
    float fp32;
    uint uint32;
} cvt_fp8_fp32_t;

EXECUTION_SPECIFIER float fp8_to_float_impl(uchar x, const int wm, const int we)
{
    bool negative_zero_nan = MIOPEN_FP8_IEEE_EXPONENT_BIAS ? false : true;

    const int weo = 8;
    const int wmo = 23;

    cvt_fp8_fp32_t fInf, fNegInf, fNaN, fNeg0;
    fInf.uint32    = 0x7F800000;
    fNegInf.uint32 = 0xFF800000;
    fNaN.uint32    = 0x7F800001;
    fNeg0.uint32   = 0x80000000;

    if(x == 0)
        return (float)(0);

    uint sign     = x >> 7;
    uint mantissa = x & ((1 << wm) - 1);
    int exponent  = (x & 0x7F) >> wm;
    if(negative_zero_nan)
    {
        if(x == 0x80)
            return fNaN.fp32;
    }
    else
    {
        if(x == 0x80)
            return fNeg0.fp32;
        if(we == 4 && (x & 0x7F) == 0x7F)
            return fNaN.fp32;
        if(we == 5 && exponent == ((1 << we) - 1))
            return (mantissa == 0) ? (sign ? fNegInf.fp32 : fInf.fp32) : fNaN.fp32;
    }
    cvt_fp8_fp32_t retval;
    const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        // TODO: verify __builtin_clz and clz do the same thing
#ifdef __HIP_PLATFORM_AMD__
        int sh = 1 + __clz(mantissa) - (32 - wm);
#else
        int sh = 1 + clz(mantissa) - (32 - wm);
#endif
        mantissa <<= sh;
        exponent += 1 - sh;
        /*
        exponent++;
        while(mantissa<(1<<wm)) {
          mantissa <<= 1;
          exponent--;
        }
        */
        mantissa &= ((1 << wm) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= wmo - wm;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << wmo;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    retval.uint32 = (sign << 31) | (exponent << 23) | mantissa;
    return retval.fp32;
}

EXECUTION_SPECIFIER float fp8_to_float(uchar x) { return fp8_to_float_impl(x, 3, 4); }

EXECUTION_SPECIFIER float bfp8_to_float(uchar x) { return fp8_to_float_impl(x, 2, 5); }

EXECUTION_SPECIFIER inline uchar
float_to_fp8_impl(float _x, const int wm, const int we) // bool stoch, uint rng)
{
    bool negative_zero_nan = MIOPEN_FP8_IEEE_EXPONENT_BIAS ? false : true;
    bool clip              = MIOPEN_FP8_CLIPPING;

    // Conserve the logic for stochastic rounding:
    bool stoch     = false;
    uint rng       = 0;
    const int mfmt = 23;
    cvt_fp8_fp32_t value;
    value.fp32   = _x;
    const uint x = value.uint32;

    uint head, mantissa;
    int exponent;
    uint sign;

    head     = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head >> 23) & 0xFF;
    sign     = head >> 31;

    uint signed_inf;
    uint nan;
    if(negative_zero_nan)
    {
        signed_inf = clip ? (sign << 7) + 0x7F : 0x80;
        nan        = 0x80;
    }
    else if(we == 4)
    {
        signed_inf = (sign << 7) + (clip ? 0x7E : 0x7F);
        nan        = (sign << 7) + 0x7F;
    }
    else
    {
        signed_inf = (sign << 7) + (clip ? 0x7B : 0x7C);
        nan        = (sign << 7) + 0x7F;
    }

    if(negative_zero_nan)
    {
        if((x & 0x7F800000) == 0x7F800000)
            return nan;
    }
    else
    {
        if((x & 0x7F800000) == 0x7F800000)
        {
            if(we == 4 || mantissa != 0)
                return nan;
            return sign == 0 ? 0x7C : 0xFC;
        }
    }
    if(x == 0)
        return 0;

    uint drop_mask           = (1 << (mfmt - wm)) - 1;
    const int max_exp        = (1 << we) - ((!negative_zero_nan && we == 5) ? 2 : 1);
    const int exp_low_cutoff = (128) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

    exponent -= exp_low_cutoff - 1;
    if(exponent <= 0)
        drop_mask = (1 << (mfmt - wm + 1 - exponent)) - 1;
    mantissa += 1 << mfmt;
    mantissa += (stoch ? rng : mantissa) & drop_mask;
    if(mantissa >= (2 << mfmt))
    {
        mantissa >>= 1;
        exponent++;
    }
    mantissa >>= (mfmt - wm);

    if(exponent <= 0)
    {
        if(x == 0) // cppcheck-suppress identicalConditionAfterEarlyExit
            return 0;
        else
        {
            // subnormal range; represented by a subnormal float8 (exponent 0)
            // and involves loss of accuracy
            mantissa >>= 1 - exponent;
            exponent = 0;
        }
    }
    // above range: quantize to maximum possible float of the same sign
    else if(exponent > max_exp)
    {
        if(clip)
        {
            mantissa = (!negative_zero_nan && we == 4) ? (1 << wm) - 2 : (1 << wm) - 1;
            exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }
    if(exponent == 0 && mantissa == 0)
        return negative_zero_nan ? 0 : (sign << 7);
    if(clip && !negative_zero_nan && we == 4 && exponent == max_exp && mantissa == (1 << wm) - 1)
        mantissa--;
    mantissa &= (1 << wm) - 1;
    return (sign << 7) | (exponent << wm) | mantissa;
}

EXECUTION_SPECIFIER uchar float_to_fp8(float _x) // bool stoch, uint rng)
{
    return float_to_fp8_impl(_x, 3, 4);
}

EXECUTION_SPECIFIER uchar float_to_bfp8(float _x) // bool stoch, uint rng)
{
    return float_to_fp8_impl(_x, 2, 5);
}
#endif // HIP target supports the configured FP8 interpretation
#endif // MIOPEN_USE_FP8 || MIOPEN_USE_BFP8

#ifdef __cplusplus
}
#endif

#endif // FP8_DEVICE_HPP
