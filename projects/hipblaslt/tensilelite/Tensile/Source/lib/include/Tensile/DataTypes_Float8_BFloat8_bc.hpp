/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc.
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

#ifdef TENSILE_USE_HIP
#include <hip/hip_runtime.h>
#endif

// comment out following macro to disable FP8/BF8 types
#define TENSILE_USE_FP8_BF8

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

namespace tensile_hip_f8_impl
{

    __host__ inline int clz(uint32_t x)
    {
        return __builtin_clz(x);
    }
    __device__ inline int clz(uint32_t x)
    {
        return __clz(x);
    }

    template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
    HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch, uint32_t rng)
    {
        constexpr bool is_half  = std::is_same<T, _Float16>::value;
        constexpr bool is_float = std::is_same<T, float>::value;
        static_assert(wm + we == 7, "wm+we==7");
        static_assert(is_half || is_float, "Only half and float can be cast to f8");

        //if(sizeof(T)==2 && we==5 && !negative_zero_nan)
        //return cast_to_f8_no_range_reduce<2, 5, _Float16>(_x, stoch, rng);

        const int mfmt = (sizeof(T) == 4) ? 23 : 10;
        uint32_t  x;
        if(sizeof(T) == 4)
            x = reinterpret_cast<uint32_t&>(_x);
        else
            x = reinterpret_cast<uint16_t&>(_x);

        uint32_t y, head, mantissa;
        int      exponent, bias;
        uint32_t sign;

        if(sizeof(T) == 4)
        {
            head     = x & 0xFF800000;
            mantissa = x & 0x7FFFFF;
            exponent = (head >> 23) & 0xFF;
            sign     = head >> 31;
            bias     = 127;
        }
        else
        {
            head     = x & 0xFC00;
            mantissa = x & 0x3FF;
            exponent = (head >> 10) & 0x1F;
            sign     = head >> 15;
            bias     = 15;
        }

        uint32_t signed_inf = (sign << 7) + (((1 << we) - 1) << wm);

        // Deal with inf and NaNs
        if(negative_zero_nan)
        {
            if(sizeof(T) == 4)
            {
                if((x & 0x7F800000) == 0x7F800000)
                    return 0x80;
            }
            else
            {
                //if(__hisinf(x) || __hisnan(x))
                if((x & 0x7C00) == 0x7C00)
                    return 0x80;
            }
        }
        else
        {
            if(sizeof(T) == 4)
            {
                if((x & 0x7F800000) == 0x7F800000)
                    return signed_inf + (mantissa != 0 ? 1 : 0);
            }
            else
            {
                if((x & 0x7C00) == 0x7C00)
                    return signed_inf + (mantissa != 0 ? 1 : 0);
            }
        }
        if(x == 0)
            return 0;

        // First need to check if it is normal or denorm as there is a difference of implict 1
        // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
        // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for
        // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
        // exponent and mantissa again

        // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
        const int f8_bias                  = (1 << (we - 1)) - 1 + (negative_zero_nan ? 1 : 0);
        const int f8_denormal_act_exponent = 1 - f8_bias; //actual exponent of f8 denormal
        // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
        // f8_exponent is the converted f8 exponent with bias encoding
        // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
        // the difference needs to be adjusted and mantissa shifted
        int act_exponent, f8_exponent, exponent_diff;

        if(exponent == 0)
        { // fp32/fp16 is in denormal.
            /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16 here.
   In this case, f8 is usually in denormal. But there could be exceptions.
   fp16 denormal has exponent bias 15 while bf8 with NANOO has exponent bias 16.
   It means that there are some numbers in fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15.
   fp16 numbers where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal.
   In this case, the fp16 mantissa should be shift left by 1  */
            act_exponent  = exponent - bias + 1;
            exponent_diff = f8_denormal_act_exponent
                            - act_exponent; // actual exponent is exponent-bias+1 as it is denormal
        }
        else
        { // fp32/fp16 is normal with implicit 1
            act_exponent = exponent - bias;
            if(act_exponent <= f8_denormal_act_exponent)
            {
                /* This is the case where fp32/fp16 is normal but it is in f8 denormal range.
       For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
       actual exponent is -7, it is actually larger due to the implict 1,
       Therefore it needs to be adjust to -6 and mantissa shift right by 1.
       So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
                exponent_diff = f8_denormal_act_exponent - act_exponent;
            }
            else
            { //both fp32/fp16 and f8 are in normal range
                exponent_diff
                    = 0; // exponent_diff=0 does not mean there is no difference for this case,
                //act_exponent could be larger. Just that it does not need shift mantissa
            }
            mantissa += (1 << mfmt); //Add the implicit 1 into mantissa
        }

        bool midpoint = (mantissa & ((1 << (mfmt - wm + exponent_diff)) - 1))
                        == (1 << (mfmt - wm + exponent_diff - 1));
        /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we shift right
     as shift right could rip off some residual part and make something not midpoint look like midpoint.
     For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than midpoint,
     but after shift right by 4 bits, it would look like midpoint.
  */

        if(exponent_diff > 0)
            mantissa >>= exponent_diff;
        else if(exponent_diff == -1)
            mantissa <<= -exponent_diff;
        bool implicit_one = mantissa & (1 << mfmt);
        //if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
        f8_exponent = (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias
                      - (implicit_one ? 0 : 1);

        //Now we have the exponent and mantissa adjusted
        uint32_t drop_mask = (1 << (mfmt - wm)) - 1;
        //bool midpoint = (mantissa & drop_mask) == ( 1 << (mfmt-wm-1) );
        bool odd = mantissa
                   & (1 << (mfmt - wm)); // if the least significant bit that is not truncated is 1
        mantissa
            += (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa)) & drop_mask;

        //Now we deal with overflow
        if(f8_exponent == 0)
        {
            if((1 << mfmt) & mantissa)
            {
                f8_exponent = 1; //denormal overflow to become normal, promote exponent
                //mantissa &=  (1<<mfmt) -1 ; //No need to make 1 implicit now as it will be addressed later
            }
        }
        else
        {
            if((1 << (mfmt + 1)) & mantissa)
            {
                mantissa >>= 1;
                f8_exponent++;
                //mantissa &=  (1<<mfmt) -1 ; // No need to make 1 implicit now as it will be addressed later
            }
        }

        mantissa >>= (mfmt - wm);

        // above range: quantize to maximum possible float of the same sign
        const int max_exp = (1 << we) - (negative_zero_nan ? 1 : 2);
        if(f8_exponent > max_exp)
        {
            if(clip)
            {
                mantissa    = (1 << wm) - 1;
                f8_exponent = max_exp;
            }
            else
            {
                return signed_inf;
            }
        }

        if(f8_exponent == 0 && mantissa == 0)
            return negative_zero_nan ? 0 : (sign << 7);
        mantissa &= (1 << wm) - 1;
        return (sign << 7) | (f8_exponent << wm) | mantissa;
    }

    template <int wm, int we, typename T, bool negative_zero_nan>
    HIP_HOST_DEVICE T cast_from_f8(uint8_t x)
    {
        constexpr bool is_half  = std::is_same<T, _Float16>::value;
        constexpr bool is_float = std::is_same<T, float>::value;
        //constexpr bool is_bf16 = std::is_same<T,hip_bfloat16>::value;
        static_assert(is_half || is_float, "only half and float are supported");

        constexpr int weo = is_half ? 5 : 8;
        constexpr int wmo = is_half ? 10 : (is_float ? 23 : 7);

        T fInf, fNegInf, fNaN, fNeg0;
        if(is_half)
        {
            const uint16_t ihInf    = 0x7C00;
            const uint16_t ihNegInf = 0xFC00;
            const uint16_t ihNaN    = 0x7C01;
            const uint16_t ihNeg0   = 0x8000;
            fInf                    = reinterpret_cast<const _Float16&>(ihInf);
            fNegInf                 = reinterpret_cast<const _Float16&>(ihNegInf);
            fNaN                    = reinterpret_cast<const _Float16&>(ihNaN);
            fNeg0                   = reinterpret_cast<const _Float16&>(ihNeg0);
        }
        else if(is_float)
        {
            const uint32_t ifInf    = 0x7F800000;
            const uint32_t ifNegInf = 0xFF800000;
            const uint32_t ifNaN    = 0x7F800001;
            const uint32_t ifNeg0   = 0x80000000;
            fInf                    = reinterpret_cast<const float&>(ifInf);
            fNegInf                 = reinterpret_cast<const float&>(ifNegInf);
            fNaN                    = reinterpret_cast<const float&>(ifNaN);
            fNeg0                   = reinterpret_cast<const float&>(ifNeg0);
        }

        if(x == 0)
            return 0;

        uint32_t sign     = x >> 7;
        uint32_t mantissa = x & ((1 << wm) - 1);
        int      exponent = (x & 0x7F) >> wm;
        if(negative_zero_nan)
        {
            if(x == 0x80)
                return fNaN;
        }
        else
        {
            if(x == 0x80)
                return fNeg0;
            if(exponent == ((1 << we) - 1))
                return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
        }
        typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type retval;
        if(we == 5 && is_half && !negative_zero_nan)
        {
            retval = x << 8;
            return reinterpret_cast<const T&>(retval);
        }

        const int exp_low_cutoff
            = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

        //subnormal input
        if(exponent == 0)
        {
            //guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
            int sh = 1 + clz(mantissa) - (32 - wm);
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

        if(sizeof(T) == 2)
            retval = (sign << 15) | (exponent << 10) | mantissa;
        else
            retval = (sign << 31) | (exponent << 23) | mantissa;
        return reinterpret_cast<const T&>(retval);
    }

} // namespace hip_f8_impl

//#include "hip_f8_impl.h"

//  Naming convension of datatype in hip header file
//      float8: fp8
//      bfloat8: bf8
//      f8 is used to consider both float8 and bfloat8

namespace TensileLite
{
    enum class hip_f8_type
    {
        bf8_fnuz    = 0, // 1:5:2
        fp8_fnuz    = 1, // 1:4:3
        //fp8bf8_fnuz = 2, // Only use for computeInputType
        //bf8fp8_fnuz = 3, // Only use for computeInputType
        bf8         = 4, // Placeholder, should not be used
        fp8         = 5, // Placeholder, should not be used
        //fp8bf8      = 6, // Placeholder, should not be used
        //bf8fp8      = 7, // Placeholder, should not be used
    };

    enum class hip_f8_rounding_mode
    {
        standard,
        stochastic
    };

    // bias mode bit implementation
    //
    // "bias mode optimial"
    //    => "bias mode bit" = 1
    //    => bias = 16 for 152, 8 for 143
    //    => NAN/INF are represented as negative_zero
    //
    // "bias mode ieee"
    //    => "bias mode bit" = 0
    //    => bias = 15 for 152, 7 for 143
    //    => NAN/INF are represented as per IEEE conventions

    // NOTE: made optimal bias mode default assuming that's the case on device
    static __device__ bool hip_f8_bias_mode_bit_device = true;
    static bool            hip_f8_bias_mode_bit_host   = true;

    static __global__ void set_hip_f8_bias_mode_bit(bool v)
    {
        hip_f8_bias_mode_bit_device = v;
    }

    static void set_hip_f8_bias_mode_ieee()
    {
        hipLaunchKernelGGL(set_hip_f8_bias_mode_bit, dim3(1), dim3(1), 0, 0, false);
        hip_f8_bias_mode_bit_host = false;
    }

    static void set_hip_f8_bias_mode_optimal()
    {
        hipLaunchKernelGGL(set_hip_f8_bias_mode_bit, dim3(1), dim3(1), 0, 0, true);
        hip_f8_bias_mode_bit_host = true;
    }

    static inline HIP_HOST_DEVICE bool get_hip_f8_bias_mode()
    {
#if defined(__HIP_DEVICE_COMPILE__)
        return hip_f8_bias_mode_bit_device;
#else
        return hip_f8_bias_mode_bit_host;
#endif
    }

    // data type
    template <hip_f8_type T>
    struct Float8_BFloat8
    {
        uint8_t __x;

        // default constructor
        HIP_HOST_DEVICE Float8_BFloat8() = default;

        // constructor from bits, no casting to floating point
        //explicit HIP_HOST_DEVICE Float8_BFloat8<T>(uint8_t v)
        //{
        //    __x = v;
        //}

        // constructor from float
#if defined(__gfx940__)
        // constructor from float using s/w simulation
        // Device implementation using intrinsic code
        explicit HIP_DEVICE Float8_BFloat8(float                v,
                                           hip_f8_rounding_mode rm = hip_f8_rounding_mode::standard,
                                           uint32_t             rng = 0)
        {
            union
            {
                float    fval;
                uint32_t i32val;
                uint8_t  i8val[4]; // not endian independent
            } val;

            uint32_t ival = 0;
            val.fval      = v;

            if(T == hip_f8_type::bf8_fnuz)
            {
                // add clipping code.. by default, always clipping for now
                if((val.i32val & 0x7F800000) != 0x7F800000) // all exp bits  are 1 --> NaN or INF
                    val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);

                // TODO: make it compile-time
                if(rm == hip_f8_rounding_mode::stochastic)
                {
                    ival       = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
                    val.i32val = ival;
                    __x       = val.i8val[0]; // little endian
                }
                else // RNE CVT
                {
                    ival = __builtin_amdgcn_cvt_pk_bf8_f32(
                        val.fval, val.fval, ival, false); // false -> WORD0
                    val.i32val = ival;
                    __x       = val.i8val[0];
                }
            }
            else // fp8
            {
                if((val.i32val & 0x7F800000) != 0x7F800000) // all exp bits  are 1 --> NaN or INF
                    val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);

                // TODO: make this if-statement compile-time
                if(rm == hip_f8_rounding_mode::stochastic)
                {
                    ival       = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
                    val.i32val = ival;
                    __x       = val.i8val[0]; // little endian
                }
                else // RNE CVT
                {
                    ival = __builtin_amdgcn_cvt_pk_fp8_f32(
                        val.fval, val.fval, ival, false); // false -> WORD0
                    val.i32val = ival;
                    __x       = val.i8val[0];
                }
            }
        }

        // only host code is simulated
        explicit HIP_HOST
#else // gfx940
        explicit HIP_HOST_DEVICE
#endif // gfx940
            Float8_BFloat8(float                v,
                           hip_f8_rounding_mode rm  = hip_f8_rounding_mode::standard,
                           uint32_t             rng = 0)
        {
            assert(T == hip_f8_type::fp8_fnuz || T == hip_f8_type::bf8_fnuz);
            // NOTE: made clipping default again
            if(T == hip_f8_type::bf8_fnuz)
            {
                if(get_hip_f8_bias_mode())
                {
                    __x = tensile_hip_f8_impl::
                        cast_to_f8<2, 5, float, true /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
                else
                {
                    __x = tensile_hip_f8_impl::
                        cast_to_f8<2, 5, float, false /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
            }
            else /* fp8*/
            {
                if(get_hip_f8_bias_mode())
                {
                    __x = tensile_hip_f8_impl::
                        cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
                else
                {
                    __x = tensile_hip_f8_impl::
                        cast_to_f8<3, 4, float, false /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
            }
        }

        // constructor from double
#if defined(__gfx940__)
        // convert double to f32 and call constructor
        explicit HIP_DEVICE Float8_BFloat8(double               v,
                                           hip_f8_rounding_mode rm = hip_f8_rounding_mode::standard,
                                           uint32_t             rng = 0)
            : Float8_BFloat8(static_cast<float>(v), rm, rng)
        {
        }

        explicit HIP_HOST
#else
        explicit HIP_HOST_DEVICE
#endif
            Float8_BFloat8(double               v,
                           hip_f8_rounding_mode rm  = hip_f8_rounding_mode::standard,
                           uint32_t             rng = 0)
            : Float8_BFloat8(static_cast<float>(v), rm, rng)
        {
        }

        // constructor from half
#if defined(__gfx940__)
        // no h/w inst for cvt from f16, just convert f16 to f32 and call constructor
        explicit HIP_DEVICE Float8_BFloat8(_Float16             v,
                                           hip_f8_rounding_mode rm = hip_f8_rounding_mode::standard,
                                           uint32_t             rng = 0)
            : Float8_BFloat8((float)v, rm, rng)
        {
        }

        explicit HIP_HOST
#else
        explicit HIP_HOST_DEVICE
#endif
            Float8_BFloat8(_Float16             v,
                           hip_f8_rounding_mode rm  = hip_f8_rounding_mode::standard,
                           uint32_t             rng = 0)
        {
            assert(T == hip_f8_type::fp8_fnuz || T == hip_f8_type::bf8_fnuz);
            // NOTE: made clipping default again
            if(T == hip_f8_type::bf8_fnuz)
            {
                if(get_hip_f8_bias_mode())
                {
                    __x = tensile_hip_f8_impl::
                        cast_to_f8<2, 5, _Float16, true /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
                else
                {
                    __x = tensile_hip_f8_impl::
                        cast_to_f8<2, 5, _Float16, false /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
            }
            else /* fp8*/
            {
                if(get_hip_f8_bias_mode())
                {
                    __x = tensile_hip_f8_impl::
                        cast_to_f8<3, 4, _Float16, true /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
                else
                {
                    __x = tensile_hip_f8_impl::
                        cast_to_f8<3, 4, _Float16, false /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
            }
        }

        // NOTE: need constructor from int for tensile
        // constructor from int, converted into float first
        explicit HIP_HOST_DEVICE Float8_BFloat8(int                  v,
                                                hip_f8_rounding_mode rm
                                                = hip_f8_rounding_mode::standard,
                                                uint32_t rng = 0)
            : Float8_BFloat8((float)v, rm, rng)
        {
        }
        explicit HIP_HOST_DEVICE Float8_BFloat8(size_t               v,
                                                hip_f8_rounding_mode rm
                                                = hip_f8_rounding_mode::standard,
                                                uint32_t rng = 0)
            : Float8_BFloat8((float)v, rm, rng)
        {
        }

        // constructor from hip_bfloat16
        // explicit HIP_HOST_DEVICE Float8_BFloat8(hip_bfloat16 v, hip_f8_rounding_mode r=hip_f8_rounding_mode::standard, uint32_t rng=0);

        // convert to float
#if defined(__gfx940__)
        // builtin conversion
        explicit inline HIP_DEVICE operator float() const
        {
            float    fval;
            uint32_t i32val = static_cast<uint32_t>(__x);
            if(T == hip_f8_type::bf8_fnuz)
                // workaround: use inline asm instead of builtin function
                // fval = __builtin_amdgcn_cvt_f32_bf8(i32val, 0);
                asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
            else
                // workaround: use inline asm instead of builtin function
                // fval = __builtin_amdgcn_cvt_f32_fp8(i32val, 0);
                asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
            return fval;
        }
        explicit inline HIP_HOST operator float() const
#else // non gfx940
        explicit inline HIP_HOST_DEVICE operator float() const
#endif
        {
            assert(T == hip_f8_type::fp8_fnuz || T == hip_f8_type::bf8_fnuz);
            if(T == hip_f8_type::bf8_fnuz)
            {
                if(get_hip_f8_bias_mode())
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(__x);
                }
                else
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<2, 5, float, false /*negative_zero_nan*/>(__x);
                }
            }
            else /* fp8*/
            {
                if(get_hip_f8_bias_mode())
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(__x);
                }
                else
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<3, 4, float, false /*negative_zero_nan*/>(__x);
                }
            }
        }

        // convert to double
        explicit inline HIP_HOST_DEVICE operator double() const
        {
            return double(float(*this)); // convert to float, then convert to f16
        }

        // convert to bf16
        explicit inline HIP_HOST_DEVICE operator BFloat16() const
        {
            return BFloat16(float(*this)); // convert to float, then convert to bf16
        }

        // convert to int
        explicit inline HIP_HOST_DEVICE operator int() const
        {
            return int(float(*this)); // convert to float, then convert to f16
        }

        // convert to half
        // NOTE: no hardware instruction to convert from and to f8, may want to convert it f32 first
#if defined(__gfx940__)
        explicit inline HIP_DEVICE operator _Float16() const
        {
            return _Float16(float(*this)); // convert to float, then convert to f16
        }

        explicit inline HIP_HOST operator _Float16() const
#else
        explicit inline HIP_HOST_DEVICE operator _Float16() const
#endif
        {
            assert(T == hip_f8_type::fp8_fnuz || T == hip_f8_type::bf8_fnuz);
            if(T == hip_f8_type::bf8_fnuz)
            {
                if(get_hip_f8_bias_mode())
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<2, 5, _Float16, true /*negative_zero_nan*/>(__x);
                }
                else
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<2, 5, _Float16, false /*negative_zero_nan*/>(__x);
                }
            }
            else /* fp8*/
            {
                if(get_hip_f8_bias_mode())
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<3, 4, _Float16, true /*negative_zero_nan*/>(__x);
                }
                else
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<3, 4, _Float16, false /*negative_zero_nan*/>(__x);
                }
            }
        }

        // convert to hip_bfloat16
        // NOTE: no hardware instruction to convert from and to f8, may want to convert it f32 first
        // explicit inline HIP_HOST_DEVICE operator hip_bfloat16() const;

        // check for zero
        inline HIP_HOST_DEVICE bool is_zero() const
        {
            if(get_hip_f8_bias_mode())
            {
                return __x == 0x00;
            }
            else
            {
                return (__x == 0x00) || (__x == 0x80);
            }
        }

        // check for nan
        inline HIP_HOST_DEVICE bool is_nan() const
        {
            if(get_hip_f8_bias_mode())
            {
                return __x == 0x80;
            }
            else
            {
                if(T == hip_f8_type::bf8_fnuz)
                {
                    return (__x == 0x7d) || (__x == 0x7e) || (__x == 0x7f) || (__x == 0xfd)
                        || (__x == 0xfe) || (__x == 0xff);
                }
                else
                {
                    return (__x == 0x79) || (__x == 0x7a) || (__x == 0x7b) || (__x == 0x7c)
                        || (__x == 0x7d) || (__x == 0x7e) || (__x == 0x7f)
                        || (__x == 0xf9) || (__x == 0xfa) || (__x == 0xfb)
                        || (__x == 0xfc) || (__x == 0xfd) || (__x == 0xfe)
                        || (__x == 0xff);
                }
            }
        }

        // check for inf
        inline HIP_HOST_DEVICE bool is_inf() const
        {
            if(get_hip_f8_bias_mode())
            {
                return __x == 0x80;
            }
            else
            {
                if(T == hip_f8_type::bf8_fnuz)
                {
                    return (__x == 0x7c) || (__x == 0xfc);
                }
                else
                {
                    return (__x == 0x78) || (__x == 0xf8);
                }
            }
        }
        //
        //  assignment operator overloading
        //
        // TODO: need to verify whether it produces extra copy, need to investigate the assembly?
        // use cast_to_f8 function otherwise
        inline HIP_HOST_DEVICE Float8_BFloat8<T>& operator=(const float& a)
        {
            __x = Float8_BFloat8<T>(a).__x;
            return *this;
        }

        inline HIP_HOST_DEVICE Float8_BFloat8<T>& operator=(const double& a)
        {
            __x = Float8_BFloat8<T>((float)a).__x;
            return *this;
        }

        inline __host__ __device__ Float8_BFloat8<T>& operator=(const Float8_BFloat8<T>& a)
        {
            __x = a.__x;
            return *this;
        }

        //inline __host__ __device__ Float8_BFloat8<T>& operator=(const rocblas_half& a)
        inline __host__ __device__ Float8_BFloat8<T>& operator=(const _Float16& a)
        {
            __x = Float8_BFloat8<T>(a).__x;
            return *this;
        }

        //  += operator
        inline __host__ __device__ Float8_BFloat8<T>& operator+=(const Float8_BFloat8<T>& a)
        {
            __x = Float8_BFloat8<T>(float(this->__x) + float(a.__x)).__x;
            return *this;
        }
    };


    typedef Float8_BFloat8<hip_f8_type::fp8_fnuz>    Float8_fnuz;
    typedef Float8_BFloat8<hip_f8_type::bf8_fnuz>    BFloat8_fnuz;
    typedef Float8_BFloat8<hip_f8_type::fp8>    Float8;
    typedef Float8_BFloat8<hip_f8_type::bf8>    BFloat8;

    // Dummy data type just to differentiate hybrid case, no conversion is used.
    typedef struct Float8BFloat8_fnuz{ uint8_t data;} Float8BFloat8_fnuz;
    typedef struct BFloat8Float8_fnuz{ uint8_t data;} BFloat8Float8_fnuz;
    typedef struct Float8BFloat8{ uint8_t data;} Float8BFloat8;
    typedef struct BFloat8Float8{ uint8_t data;} BFloat8Float8;

    //  Other operator overloading
    inline std::ostream& operator<<(std::ostream& os, const Float8& f8)
    {
        os << static_cast<float>(f8);
        return os;
    }
    inline std::ostream& operator<<(std::ostream& os, const BFloat8& bf8)
    {
        os << static_cast<float>(bf8);
        return os;
    }

    inline Float8 operator+(Float8 a, Float8 b)
    {
        return static_cast<Float8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline Float8 operator+(Float8 a, float b)
    {
        return static_cast<Float8>(static_cast<float>(a) + b);
    }
    inline Float8 operator+(float a, Float8 b)
    {
        return static_cast<Float8>(a + static_cast<float>(b));
    }
    inline BFloat8 operator+(BFloat8 a, BFloat8 b)
    {
        return static_cast<BFloat8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline Float8 operator-(Float8 a, Float8 b)
    {
        return static_cast<Float8>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline BFloat8 operator-(BFloat8 a, BFloat8 b)
    {
        return static_cast<BFloat8>(static_cast<float>(a) - static_cast<float>(b));
    }
    //  NOTE: It is not used in reference solution directly, we want to return float otherwise
    inline Float8 operator*(Float8 a, Float8 b)
    {
        return static_cast<Float8>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline Float8 operator*(float a, Float8 b)
    {
        return static_cast<Float8>(a * static_cast<float>(b));
    }
    inline Float8 operator*(Float8 a, float b)
    {
        return static_cast<Float8>(static_cast<float>(a) * b);
    }
    inline BFloat8 operator*(BFloat8 a, BFloat8 b)
    {
        return static_cast<BFloat8>(static_cast<float>(a) * static_cast<float>(b));
    }

    inline Float8 operator/(Float8 a, Float8 b)
    {
        return static_cast<Float8>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline BFloat8 operator/(BFloat8 a, BFloat8 b)
    {
        return static_cast<BFloat8>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline bool operator<(Float8 a, Float8 b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<(float a, Float8 b)
    {
        return a < static_cast<float>(b);
    }
    inline bool operator<(Float8 a, float b)
    {
        return static_cast<float>(a) < b;
    }
    inline bool operator<(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<=(Float8 a, Float8 b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator<=(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator==(Float8 a, Float8 b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator==(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator!=(Float8 a, Float8 b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator!=(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator>(Float8 a, Float8 b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>(float a, Float8 b)
    {
        return a > static_cast<float>(b);
    }
    inline bool operator>(Float8 a, float b)
    {
        return static_cast<float>(a) > b;
    }
    inline bool operator>(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>=(Float8 a, Float8 b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }
    inline bool operator>=(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }



    inline std::ostream& operator<<(std::ostream& os, const Float8_fnuz& f8)
    {
        os << static_cast<float>(f8);
        return os;
    }
    inline std::ostream& operator<<(std::ostream& os, const BFloat8_fnuz& bf8)
    {
        os << static_cast<float>(bf8);
        return os;
    }

    inline Float8_fnuz operator+(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline Float8_fnuz operator+(Float8_fnuz a, float b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) + b);
    }
    inline Float8_fnuz operator+(float a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(a + static_cast<float>(b));
    }
    inline BFloat8_fnuz operator+(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<BFloat8_fnuz>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline Float8_fnuz operator-(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline BFloat8_fnuz operator-(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<BFloat8_fnuz>(static_cast<float>(a) - static_cast<float>(b));
    }
    //  NOTE: It is not used in reference solution directly, we want to return float otherwise
    inline Float8_fnuz operator*(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline Float8_fnuz operator*(float a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(a * static_cast<float>(b));
    }
    inline Float8_fnuz operator*(Float8_fnuz a, float b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) * b);
    }
    inline BFloat8_fnuz operator*(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<BFloat8_fnuz>(static_cast<float>(a) * static_cast<float>(b));
    }

    inline Float8_fnuz operator/(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline BFloat8_fnuz operator/(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<BFloat8_fnuz>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline bool operator<(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<(float a, Float8_fnuz b)
    {
        return a < static_cast<float>(b);
    }
    inline bool operator<(Float8_fnuz a, float b)
    {
        return static_cast<float>(a) < b;
    }
    inline bool operator<(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<=(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator<=(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator==(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator==(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator!=(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator!=(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator>(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>(float a, Float8_fnuz b)
    {
        return a > static_cast<float>(b);
    }
    inline bool operator>(Float8_fnuz a, float b)
    {
        return static_cast<float>(a) > b;
    }
    inline bool operator>(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>=(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }
    inline bool operator>=(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    // =========== Explicit downcasting to support Stochastic Rounding ===============
    // NOTE: s/w clipping is enabled by default (Fp32toFp8SWClip)

    // same types, no casting needed
    template <typename T, typename Ta = T>
    inline T explicit_downcast(Ta a, hip_f8_rounding_mode rm, uint32_t rng)
    {
        return a;
    }
    // float8 ........
    template <>
    inline Float8_fnuz explicit_downcast(float a, hip_f8_rounding_mode rm, uint32_t rng)
    {
        return Float8_fnuz(a, rm, rng);
    }

    // bfloat8 ........
    template <>
    inline BFloat8_fnuz explicit_downcast(float a, hip_f8_rounding_mode rm, uint32_t rng)
    {
        return BFloat8_fnuz(a, rm, rng);
    }
} // end of namespace TensileLite

namespace std
{



    inline bool isinf(const TensileLite::Float8& a)
    {
        return a.is_inf();
    }
    inline bool isinf(const TensileLite::BFloat8& a)
    {
        return a.is_inf();
    }

    inline bool isnan(const TensileLite::Float8& a)
    {
        return a.is_nan();
    }
    inline bool isnan(const TensileLite::BFloat8& a)
    {
        return a.is_nan();
    }
    inline bool iszero(const TensileLite::Float8& a)
    {
        return a.is_zero();
    }
    inline bool iszero(const TensileLite::BFloat8& a)
    {
        return a.is_zero();
    }

    inline TensileLite::Float8 abs(const TensileLite::Float8& a)
    {
        return TensileLite::Float8(std::abs(float(a)));
    }
    inline TensileLite::BFloat8 abs(const TensileLite::BFloat8& a)
    {
        return TensileLite::BFloat8(std::abs(float(a)));
    }

    inline TensileLite::Float8 sin(const TensileLite::Float8& a)
    {
        return TensileLite::Float8(std::sin(float(a)));
    }
    inline TensileLite::BFloat8 sin(const TensileLite::BFloat8& a)
    {
        return TensileLite::BFloat8(std::sin(float(a)));
    }

    inline TensileLite::Float8 cos(const TensileLite::Float8& a)
    {
        return TensileLite::Float8(std::cos(float(a)));
    }
    inline TensileLite::BFloat8 cos(const TensileLite::BFloat8& a)
    {
        return TensileLite::BFloat8(std::cos(float(a)));
    }

    inline std::string to_string(const TensileLite::Float8& a)
    {
        return std::to_string(static_cast<float>(a));
    }
    inline std::string to_string(const TensileLite::BFloat8& a)
    {
        return std::to_string(static_cast<float>(a));
    }




    inline bool isinf(const TensileLite::Float8_fnuz& a)
    {
        return a.is_inf();
    }
    inline bool isinf(const TensileLite::BFloat8_fnuz& a)
    {
        return a.is_inf();
    }

    inline bool isnan(const TensileLite::Float8_fnuz& a)
    {
        return a.is_nan();
    }
    inline bool isnan(const TensileLite::BFloat8_fnuz& a)
    {
        return a.is_nan();
    }
    inline bool iszero(const TensileLite::Float8_fnuz& a)
    {
        return a.is_zero();
    }
    inline bool iszero(const TensileLite::BFloat8_fnuz& a)
    {
        return a.is_zero();
    }

    inline TensileLite::Float8_fnuz abs(const TensileLite::Float8_fnuz& a)
    {
        return TensileLite::Float8_fnuz(std::abs(float(a)));
    }
    inline TensileLite::BFloat8_fnuz abs(const TensileLite::BFloat8_fnuz& a)
    {
        return TensileLite::BFloat8_fnuz(std::abs(float(a)));
    }

    inline TensileLite::Float8_fnuz sin(const TensileLite::Float8_fnuz& a)
    {
        return TensileLite::Float8_fnuz(std::sin(float(a)));
    }
    inline TensileLite::BFloat8_fnuz sin(const TensileLite::BFloat8_fnuz& a)
    {
        return TensileLite::BFloat8_fnuz(std::sin(float(a)));
    }

    inline TensileLite::Float8_fnuz cos(const TensileLite::Float8_fnuz& a)
    {
        return TensileLite::Float8_fnuz(std::cos(float(a)));
    }
    inline TensileLite::BFloat8_fnuz cos(const TensileLite::BFloat8_fnuz& a)
    {
        return TensileLite::BFloat8_fnuz(std::cos(float(a)));
    }

    inline std::string to_string(const TensileLite::Float8_fnuz& a)
    {
        return std::to_string(static_cast<float>(a));
    }
    inline std::string to_string(const TensileLite::BFloat8_fnuz& a)
    {
        return std::to_string(static_cast<float>(a));
    }
} // namespace std
