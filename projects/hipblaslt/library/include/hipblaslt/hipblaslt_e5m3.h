/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <hip/hip_fp16.h>          // __half_raw
#include <iostream>

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

#if defined(__gfx1250__) && __HIP_DEVICE_COMPILE__
#define HIP_E5M3_CVT_FAST_PATH 1
#else
#define HIP_E5M3_CVT_FAST_PATH 0
#endif

static_assert(sizeof(unsigned char) == 1);
static_assert(sizeof(unsigned short int) == 2);
static_assert(sizeof(unsigned int) == 4);


HIP_HOST_DEVICE static uint8_t hipblaslt_cast_to_uf8(float _x, int wm, int we, bool clip = false,
                                                          bool stoch = false,
                                                          unsigned int rng = 0);

HIP_HOST_DEVICE static float hipblaslt_cast_from_uf8(uint8_t x, int wm, int we,
                                          bool clip = false);

#if HIP_E5M3_CVT_FAST_PATH
template <bool stochastic_rounding = false>
HIP_DEVICE static uint8_t hipblaslt_cast_to_uf8_from_f32(float v, bool saturate,
                                                   unsigned int rng = 0);

HIP_DEVICE static float hipblaslt_cast_to_f32_from_uf8(uint8_t v);

#endif

struct hipblaslt_e5m3 {
    uint8_t data;

    HIP_HOST_DEVICE hipblaslt_e5m3() = default;

    HIP_HOST_DEVICE hipblaslt_e5m3(const hipblaslt_e5m3&) = default;

    explicit HIP_HOST_DEVICE hipblaslt_e5m3(const float f)
    {
#if HIP_E5M3_CVT_FAST_PATH
        data = hipblaslt_cast_to_uf8_from_f32(f, true);
#else
        data = hipblaslt_cast_to_uf8(f, 3, 5, true);
#endif
    }

    /*! convert fp8 e4m3 to float */
    HIP_HOST_DEVICE operator float() const {
// #if HIP_E5M3_CVT_FAST_PATH
//        return hipblaslt_cast_to_f32_from_uf8(data);
//#else
        return hipblaslt_cast_from_uf8(data, 3, 5);
//#endif
    }

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        return data == 0;
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        return data == 0xff;
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        return false;
    }

    inline HIP_HOST_DEVICE hipblaslt_e5m3 operator-() const
    {
        return hipblaslt_e5m3(*this);
    }
};

inline float operator*(hipblaslt_e5m3 a, float b)
{
    return static_cast<float>(static_cast<float>(a) * b);
}

inline float operator*(float a, hipblaslt_e5m3 b)
{
    return static_cast<float>(a * static_cast<float>(b));
}

namespace std
{
    inline std::string to_string(const hipblaslt_e5m3& a)
    {
        return std::to_string(static_cast<float>(a));
    }

    inline std::ostream& operator<<(std::ostream& stream, const hipblaslt_e5m3& a)
    {
        float val = static_cast<float>(a);
        stream << val;
        return stream;
    }
} // namespace std

HIP_HOST_DEVICE static uint8_t hipblaslt_cast_to_uf8(float _x, int wm, int we, bool clip ,
                                                          bool stoch,
                                                          unsigned int rng) {
  const int mfmt = 23;
  unsigned int x;

  x = reinterpret_cast<unsigned int&>(_x);


  unsigned int head, mantissa;
  int exponent, bias;
  unsigned int sign;
  unsigned int fInf, mask;

  head = x & 0xFF800000;
  mantissa = x & 0x7FFFFF;
  exponent = (head >> 23) & 0xFF;
  sign = head >> 31;
  bias = 127;
  fInf = 0x7F800000;
  mask = 0x7FFFFFFF;

  sign = 0;

  unsigned int inf = 0;
  unsigned int nan = 0;
  if (we == 4) {  // e4m3
    inf = clip ? 0x7e : 0x7f;
    nan = 0x7f;
  } else {  // e5m3
    inf = clip ? 0xfe : 0xff;
    nan = 0xff;
  }

  // Max values
  unsigned long long ifmax = 0;
  if (we == 5) { // E5M3
    ifmax = 0x47E00000;
  } else { // E4M3
    ifmax = 0x43E00000;
  }
  // Deal with inf and NaNs
  if ((x & fInf) == fInf) {
    return nan;
  }

  if ((x & mask) > ifmax) {
    return inf;
  }

  if (x == 0) {
    return 0;
  }

  // First need to check if it is normal or denorm as there is a difference of implict 1
  // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
  // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for
  // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
  // exponent and mantissa again

  // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
  const int f8_bias = (1 << (we - 1)) - 1;
  const int f8_denormal_act_exponent = 1 - f8_bias;  // actual exponent of f8 denormal
  // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
  // f8_exponent is the converted f8 exponent with bias encoding
  // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
  // the difference needs to be adjusted and mantissa shifted
  int act_exponent, f8_exponent, exponent_diff;

  if (exponent == 0) {  // fp32/fp16 is in denormal.
    /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16
here. In this case, f8 is usually in denormal. But there could be exceptions. fp16 denormal has
exponent bias 15 while bf8 with NANOO has exponent bias 16. It means that there are some numbers in
fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. fp16 numbers
where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal. In
this case, the fp16 mantissa should be shift left by 1  */
    act_exponent = exponent - bias + 1;
    exponent_diff = f8_denormal_act_exponent -
        act_exponent;  // actual exponent is exponent-bias+1 as it is denormal
  } else {             // fp32/fp16 is normal with implicit 1
    act_exponent = exponent - bias;
    if (act_exponent <= f8_denormal_act_exponent) {
      /* This is the case where fp32/fp16 is normal but it is in f8 denormal range.
For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
actual exponent is -7, it is actually larger due to the implict 1,
Therefore it needs to be adjust to -6 and mantissa shift right by 1.
So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
      exponent_diff = f8_denormal_act_exponent - act_exponent;
    } else {              // both fp32/fp16 and f8 are in normal range
      exponent_diff = 0;  // exponent_diff=0 does not mean there is no difference for this case,
                          // act_exponent could be larger. Just that it does not need shift mantissa
    }
    mantissa += (1ull << mfmt);  // Add the implicit 1 into mantissa
  }

  bool midpoint = (mantissa & ((1ull << (mfmt - wm + exponent_diff)) - 1)) ==
      (1ull << (mfmt - wm + exponent_diff - 1));
  /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we shift
right as shift right could rip off some residual part and make something not midpoint look like
midpoint. For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than midpoint, but
after shift right by 4 bits, it would look like midpoint.
*/

  if (exponent_diff > 0)
    mantissa >>= exponent_diff;
  else if (exponent_diff == -1)
    mantissa <<= -exponent_diff;
  bool implicit_one = mantissa & (1ull << mfmt);
  // if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
  f8_exponent =
      (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias - (implicit_one ? 0 : 1);

  // Now we have the exponent and mantissa adjusted
  unsigned long long drop_mask = (1ull << (mfmt - wm)) - 1;
  bool odd =
      mantissa & (1ull << (mfmt - wm));  // if the least significant bit that is not truncated is 1
  mantissa +=
      (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1ull) : mantissa)) & drop_mask;

  // Now we deal with overflow
  if (f8_exponent == 0) {
    if ((1ull << mfmt) & mantissa) {
      f8_exponent = 1;  // denormal overflow to become normal, promote exponent
    }
  } else {
    if ((1ull << (mfmt + 1)) & mantissa) {
      mantissa >>= 1;
      f8_exponent++;
    }
  }

  mantissa >>= (mfmt - wm);

  // above range: quantize to maximum possible float of the same sign
  const int max_exp = (1 << we) - 1;
  if (f8_exponent > max_exp) {
    return inf;
  }

  if (f8_exponent == 0 && mantissa == 0)
    return 0;

  mantissa &= (1 << wm) - 1;
  return (f8_exponent << wm) | mantissa;
}

// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/test/rocblas_hip_f8_impl.h#L220
// This has been modified to handle double types as well
HIP_HOST_DEVICE static float hipblaslt_cast_from_uf8(uint8_t x, int wm, int we,
                                          bool clip) {
  constexpr int weo = 8;
  constexpr int wmo = 23;

  float fInf, fNegInf, fNaN, fNeg0, fmax, fmin;

  const unsigned int ifInf = 0x7F800000;
  const unsigned int ifNegInf = 0xFF800000;
  const unsigned int ifNaN = 0x7F800001;
  const unsigned int ifNeg0 = 0x80000000;
  /* Max number in e5m2 57344*/
  const unsigned int ifmax = 0x47600000;
  const unsigned int ifmin = 0xC7600000;
  fInf = reinterpret_cast<const float&>(ifInf);
  fNegInf = reinterpret_cast<const float&>(ifNegInf);
  fNaN = reinterpret_cast<const float&>(ifNaN);
  fNeg0 = reinterpret_cast<const float&>(ifNeg0);
  fmax = reinterpret_cast<const float&>(ifmax);
  fmin = reinterpret_cast<const float&>(ifmin);

  if (x == 0) {
    return 0;
  }

  unsigned long long mantissa = x & ((1 << wm) - 1);
  int exponent = ((we == 4) ? (x & 0x7F) : (x & 0xFF)) >> wm;
  if (we == 4) {  // e4m3
    if ((x & 0x7F) == 0x7F) {
      return fNaN;
    }
  } else if ((x & 0xFF) == 0xFF) {  // e5m2 NaN/Inf
    return fNaN;
  }

  unsigned int retval;

  const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1;

  // subnormal input
  if (exponent == 0) {
#if __HIP_DEVICE_COMPILE__
    // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
    int sh = 1 + __clz(mantissa) - (32 - wm);
#else
    int sh = 1 + __builtin_clz(mantissa) - (32 - wm);
#endif
    mantissa <<= sh;
    exponent += 1 - sh;
    mantissa &= ((1ull << wm) - 1);
  }
  exponent += exp_low_cutoff - 1;
  mantissa <<= wmo - wm;

  // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
  if (exponent <= 0) {
    mantissa |= 1ull << wmo;
    mantissa >>= 1 - exponent;
    exponent = 0;
  }

  retval = (exponent << 23) | mantissa;

  return reinterpret_cast<const float&>(retval);
}

#if HIP_E5M3_CVT_FAST_PATH
template <bool stochastic_rounding>
HIP_DEVICE static uint8_t hipblaslt_cast_to_uf8_from_f32(float v, bool saturate,
                                                   unsigned int rng) {
  uint8_t i8data;
  union {
    float fval;
    unsigned int i32val;
    uint8_t i8val[4];  // NOTE: not endian independent
  } val;

  unsigned int ival = 0;
  val.fval = v;

  if (saturate) {
      if ((val.i32val & 0x7F800000) != 0x7F800000) {  /// propagate NAN/INF, no clipping
          val.fval = __builtin_amdgcn_fmed3f(val.fval, 114688.0, -114688.0);
      }
  }

  if (stochastic_rounding) {
    ival =  __builtin_amdgcn_cvt_sr_fp8_f32_e5m3(val.fval, rng, ival, 0);
    val.i32val = ival;
    i8data = val.i8val[0];  // little endian
  } else {                  // RNE CVT
    ival = __builtin_amdgcn_cvt_pk_fp8_f32_e5m3(val.fval, val.fval, ival, false);
    val.i32val = ival;
    i8data = val.i8val[0];
  }
  return i8data;
}

HIP_DEVICE static float hipblaslt_cast_to_f32_from_uf8(uint8_t v) {
  union {
    unsigned int i32val;
    unsigned char i8val[4];
  } val;
  val.i8val[0] = v;

  float fval = __builtin_amdgcn_cvt_f32_fp8_e5m3(val.i32val, 0);
  return fval;
}

#endif
