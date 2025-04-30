/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022-2023 Advanced Micro Devices, Inc.
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

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

// We are clipping in down--conversion by default
#define DOWNCAST_CLIPPING_ON 1

namespace tensile_hip_f8_impl
{

    template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
    HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

    template <int wm, int we, typename T, bool negative_zero_nan>
    HIP_HOST_DEVICE T cast_from_f8(uint8_t x);

} // namespace hip_f8_impl

#include "hip_f8_impl.h"

// device specific optimized code
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
namespace tensile_gfx940_f8_impl
{
    template <bool isE2M5, bool stochastic_rounding>
    inline HIP_DEVICE uint8_t cast_to_f8_from_f32(float v, uint32_t rng = 0)
    {
        uint8_t i8data;

        union
        {
            float    fval;
            uint32_t i32val;
            uint8_t  i8val[4]; // not endian independent
        } val;

        uint32_t ival = 0;
        val.fval      = v;

        if(isE2M5)
        {
#ifdef DOWNCAST_CLIPPING_ON
            if((val.i32val & 0x7F800000) != 0x7F800000) // all exp bits  are 1 --> NaN or INF
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
#endif

            if(stochastic_rounding)
            {
                ival       = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
                val.i32val = ival;
                i8data     = val.i8val[0]; // little endian
            }
            else // RNE CVT
            {
                ival = __builtin_amdgcn_cvt_pk_bf8_f32(
                    val.fval, val.fval, ival, false); // false -> WORD0
                val.i32val = ival;
                i8data     = val.i8val[0];
            }
        }
        else // fp8
        {
#ifdef DOWNCAST_CLIPPING_ON
            if((val.i32val & 0x7F800000) != 0x7F800000) // all exp bits  are 1 --> NaN or INF
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
#endif

            if(stochastic_rounding)
            {
                ival       = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
                val.i32val = ival;
                i8data     = val.i8val[0]; // little endian
            }
            else // RNE CVT
            {
                ival = __builtin_amdgcn_cvt_pk_fp8_f32(
                    val.fval, val.fval, ival, false); // false -> WORD0
                val.i32val = ival;
                i8data     = val.i8val[0];
            }
        }
        return i8data;
    }

}

#endif

//  Naming convension of datatype in hip header file
//      float8: fp8
//      bfloat8: bf8
//      f8 is used to consider both float8 and bfloat8

//namespace Tensile
//{
enum class hip_f8_type
{
    bf8 = 0, // 1:5:2
    fp8 = 1 // 1:4:3
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
    uint8_t data;

    // default constructor
    HIP_HOST_DEVICE Float8_BFloat8() = default;

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // NOTE: ON-DEVICE... always optimal bias
    explicit HIP_DEVICE Float8_BFloat8(float                v,
                                       hip_f8_rounding_mode rm  = hip_f8_rounding_mode::standard,
                                       uint32_t             rng = 0)
    {
        // runtime branch, use default constructor and explicit_cast() if want to avoid it

        if(rm == hip_f8_rounding_mode::stochastic)
            data = tensile_gfx940_f8_impl::cast_to_f8_from_f32<T == hip_f8_type::bf8, true>(v, rng);
        else
            data
                = tensile_gfx940_f8_impl::cast_to_f8_from_f32<T == hip_f8_type::bf8, false>(v, rng);
    }
    // only host code is simulated
    explicit HIP_HOST
#else // gfx940/gfx941/gfx942
    explicit HIP_HOST_DEVICE
#endif // gfx940/gfx941/gfx942
        Float8_BFloat8(float                v,
                       hip_f8_rounding_mode rm  = hip_f8_rounding_mode::standard,
                       uint32_t             rng = 0)
    {
        // NOTE: made clipping default again
        if(T == hip_f8_type::bf8)
        {
            if(get_hip_f8_bias_mode())
            {
                data = tensile_hip_f8_impl::
#ifdef DOWNCAST_CLIPPING_ON
                    cast_to_f8<2, 5, float, true /*negative_zero_nan*/, true /*clip*/>(
#else
                    cast_to_f8<2, 5, float, true /*negative_zero_nan*/, false /*clip*/>(
#endif
                        v, (rm == hip_f8_rounding_mode::stochastic), rng);
            }
            else
            {
                data = tensile_hip_f8_impl::
#ifdef DOWNCAST_CLIPPING_ON
                    cast_to_f8<2, 5, float, false /*negative_zero_nan*/, true /*clip*/>(
#else
                    cast_to_f8<2, 5, float, false /*negative_zero_nan*/, false /*clip*/>(
#endif
                        v, (rm == hip_f8_rounding_mode::stochastic), rng);
            }
        }
        else /* fp8*/
        {
            if(get_hip_f8_bias_mode())
            {
                data = tensile_hip_f8_impl::
#ifdef DOWNCAST_CLIPPING_ON
                    cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
#else
                    cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
#endif
                        v, (rm == hip_f8_rounding_mode::stochastic), rng);
            }
            else
            {
                data = tensile_hip_f8_impl::
#ifdef DOWNCAST_CLIPPING_ON
                    cast_to_f8<3, 4, float, false /*negative_zero_nan*/, true /*clip*/>(
#else
                    cast_to_f8<3, 4, float, false /*negative_zero_nan*/, false /*clip*/>(
#endif
                        v, (rm == hip_f8_rounding_mode::stochastic), rng);
            }
        }
    }

    // constructor from half
    // no h/w inst for cvt from f16, just convert f16 to f32 and call constructor
    explicit HIP_HOST_DEVICE Float8_BFloat8(_Float16             v,
                                            hip_f8_rounding_mode rm
                                            = hip_f8_rounding_mode::standard,
                                            uint32_t rng = 0)
        : Float8_BFloat8((float)v, rm, rng)
    {
    }

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
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // builtin conversion
    explicit inline HIP_DEVICE operator float() const
    {
        float    fval;
        uint32_t i32val = static_cast<uint32_t>(data);
        if(T == hip_f8_type::bf8)
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

#else // non gfx940/gfx941/gfx942

    explicit inline HIP_HOST_DEVICE operator float() const
#endif // gfx940/gfx941/gfx942
    {
        if(T == hip_f8_type::bf8)
        {
            if(get_hip_f8_bias_mode())
            {
                return tensile_hip_f8_impl::cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(
                    data);
            }
            else
            {
                return tensile_hip_f8_impl::cast_from_f8<2, 5, float, false /*negative_zero_nan*/>(
                    data);
            }
        }
        else /* fp8*/
        {
            if(get_hip_f8_bias_mode())
            {
                return tensile_hip_f8_impl::cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(
                    data);
            }
            else
            {
                return tensile_hip_f8_impl::cast_from_f8<3, 4, float, false /*negative_zero_nan*/>(
                    data);
            }
        }
    }

    // convert to half
    explicit inline HIP_HOST_DEVICE operator _Float16() const
    {
        return _Float16(float(*this)); // convert to float, then convert to f16
    }

    // convert to hip_bfloat16
    // NOTE: no hardware instruction to convert from and to f8, may want to convert it f32 first
    // explicit inline HIP_HOST_DEVICE operator hip_bfloat16() const;

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        if(get_hip_f8_bias_mode())
        {
            return data == 0x00;
        }
        else
        {
            return (data == 0x00) || (data == 0x80);
        }
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        if(get_hip_f8_bias_mode())
        {
            return data == 0x80;
        }
        else
        {
            if(T == hip_f8_type::bf8)
            {
                return (data == 0x7d) || (data == 0x7e) || (data == 0x7f) || (data == 0xfd)
                       || (data == 0xfe) || (data == 0xff);
            }
            else
            {
                return (data == 0x79) || (data == 0x7a) || (data == 0x7b) || (data == 0x7c)
                       || (data == 0x7d) || (data == 0x7e) || (data == 0x7f) || (data == 0xf9)
                       || (data == 0xfa) || (data == 0xfb) || (data == 0xfc) || (data == 0xfd)
                       || (data == 0xfe) || (data == 0xff);
            }
        }
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        if(get_hip_f8_bias_mode())
        {
            return data == 0x80;
        }
        else
        {
            if(T == hip_f8_type::bf8)
            {
                return (data == 0x7c) || (data == 0xfc);
            }
            else
            {
                return (data == 0x78) || (data == 0xf8);
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
        data = Float8_BFloat8<T>(a).data;
        return *this;
    }

    inline HIP_HOST_DEVICE Float8_BFloat8<T>& operator=(const double& a)
    {
        data = Float8_BFloat8<T>((float)a).data;
        return *this;
    }

    inline __host__ __device__ Float8_BFloat8<T>& operator=(const Float8_BFloat8<T>& a)
    {
        data = a.data;
        return *this;
    }

    //inline __host__ __device__ Float8_BFloat8<T>& operator=(const rocblas_half& a)
    inline __host__ __device__ Float8_BFloat8<T>& operator=(const _Float16& a)
    {
        data = Float8_BFloat8<T>(a).data;
        return *this;
    }

    //  += operator
    inline __host__ __device__ Float8_BFloat8<T>& operator+=(const Float8_BFloat8<T>& a)
    {
        data = Float8_BFloat8<T>(float(this->data) + float(a.data)).data;
        return *this;
    }
};

// TODO: place it in appropriate header
typedef Float8_BFloat8<hip_f8_type::fp8> tensile_float8;
typedef Float8_BFloat8<hip_f8_type::bf8> tensile_bfloat8;

//  Other operator overloading
inline std::ostream& operator<<(std::ostream& os, const tensile_float8& f8)
{
    os << static_cast<float>(f8);
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const tensile_bfloat8& bf8)
{
    os << static_cast<float>(bf8);
    return os;
}
inline tensile_float8 operator+(tensile_float8 a, tensile_float8 b)
{
    return static_cast<tensile_float8>(static_cast<float>(a) + static_cast<float>(b));
}
inline tensile_bfloat8 operator+(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<tensile_bfloat8>(static_cast<float>(a) + static_cast<float>(b));
}
inline tensile_float8 operator-(tensile_float8 a, tensile_float8 b)
{
    return static_cast<tensile_float8>(static_cast<float>(a) - static_cast<float>(b));
}
inline tensile_bfloat8 operator-(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<tensile_bfloat8>(static_cast<float>(a) - static_cast<float>(b));
}
//  NOTE: It is not used in reference solution directly, we want to return float otherwise
inline tensile_float8 operator*(tensile_float8 a, tensile_float8 b)
{
    return static_cast<tensile_float8>(static_cast<float>(a) * static_cast<float>(b));
}
inline tensile_bfloat8 operator*(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<tensile_bfloat8>(static_cast<float>(a) * static_cast<float>(b));
}

inline tensile_float8 operator/(tensile_float8 a, tensile_float8 b)
{
    return static_cast<tensile_float8>(static_cast<float>(a) / static_cast<float>(b));
}
inline tensile_bfloat8 operator/(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<tensile_bfloat8>(static_cast<float>(a) / static_cast<float>(b));
}
inline bool operator<(tensile_float8 a, tensile_float8 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}
inline bool operator<(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}
inline bool operator<=(tensile_float8 a, tensile_float8 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}
inline bool operator<=(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}
inline bool operator==(tensile_float8 a, tensile_float8 b)
{
    return static_cast<float>(a) == static_cast<float>(b);
}
inline bool operator==(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<float>(a) == static_cast<float>(b);
}
inline bool operator!=(tensile_float8 a, tensile_float8 b)
{
    return static_cast<float>(a) != static_cast<float>(b);
}
inline bool operator!=(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<float>(a) != static_cast<float>(b);
}
inline bool operator>(tensile_float8 a, tensile_float8 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}
inline bool operator>(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}
inline bool operator>=(tensile_float8 a, tensile_float8 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}
inline bool operator>=(tensile_bfloat8 a, tensile_bfloat8 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}

// ================ Explicit downcasting to support Stochastic Rounding and clipping ===============

#if 0 // enable_if_t supported from C++14 and above, not C++11! enable it when compiler updated 
    template <typename T,
              typename Ta,
              bool stochastic_rounding,
            std::enable_if_t<std::is_same<T, Ta>{}, int> = 0>
    inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng = 0, bool clip = true)
    {
        // same type, no conversion
        return a;
    }

    // Use h/w intrinsic and optimized version when __gfx940__/__gfx941__/__gfx942__
    template <typename T,
              typename Ta,
              bool stochastic_rounding,
              std::enable_if_t<!(std::is_same<T, Ta>{}), int> = 0>
    inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng = 0, bool clip = true)
    {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        T val;
        if(std::is_same<T, rocblas_f8>::value)
            val.data
                = tensile_gfx940_f8_impl::cast_to_f8_from_f32<false, stochastic_rounding>(float(a), rng, clip);
        else
            val.data
                = tensile_gfx940_f8_impl::cast_to_f8_from_f32<true, stochastic_rounding>(float(a), rng, clip);
        return val;
#else //gfx940/gfx941/gfx942
        return T(float(a),
          stochastic_rounding ? hip_f8_rounding_mode::stochastic : hip_f8_rounding_mode::standard,
          rng,
          clip);
#endif //gfx940/gfx941/gfx942
    }
#else // without enable_if_t, we have to use explicit template specialization

template <typename T, typename Ta, bool stochastic_rounding>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng = 0);

template <typename T, typename Ta = T, bool stochastic_rounding>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng)
{
    // same type, no conversion
    return a;
}

// NOTE: using explicit specialization
template <>
inline __host__ __device__ tensile_float8
    explicit_downcast<tensile_float8, float, true>(float a, uint32_t rng)
{
    // Use h/w intrinsic and optimized version when __gfx940__/__gfx941__/__gfx942__
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    tensile_float8 val;
    val.data = tensile_gfx940_f8_impl::cast_to_f8_from_f32<false, true>(a, rng);
    return val;
#else
    return tensile_float8(float(a), hip_f8_rounding_mode::stochastic, rng);
#endif
}

template <>
inline __host__ __device__ tensile_float8
    explicit_downcast<tensile_float8, float, false>(float a, uint32_t rng)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    tensile_float8 val;
    val.data = tensile_gfx940_f8_impl::cast_to_f8_from_f32<false, false>(a, rng);
    return val;
#else
    return tensile_float8(float(a), hip_f8_rounding_mode::standard, rng);
#endif
}

template <>
inline __host__ __device__ tensile_bfloat8
    explicit_downcast<tensile_bfloat8, float, true>(float a, uint32_t rng)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    tensile_bfloat8 val;
    val.data = tensile_gfx940_f8_impl::cast_to_f8_from_f32<true, true>(a, rng);
    return val;
#else
    return tensile_bfloat8(float(a), hip_f8_rounding_mode::stochastic, rng);
#endif
}

template <>
inline __host__ __device__ tensile_bfloat8
    explicit_downcast<tensile_bfloat8, float, false>(float a, uint32_t rng)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    tensile_bfloat8 val;
    val.data = tensile_gfx940_f8_impl::cast_to_f8_from_f32<true, false>(a, rng);
    return val;
#else
    return tensile_bfloat8(float(a), hip_f8_rounding_mode::standard, rng);
#endif
}

#endif // end of explicit specialization

//} // end of namespace Tensile

namespace std
{
    inline bool isinf(const tensile_float8& a)
    {
        return a.is_inf();
    }
    inline bool isinf(const tensile_bfloat8& a)
    {
        return a.is_inf();
    }

    inline bool isnan(const tensile_float8& a)
    {
        return a.is_nan();
    }
    inline bool isnan(const tensile_bfloat8& a)
    {
        return a.is_nan();
    }
    inline bool iszero(const tensile_float8& a)
    {
        return a.is_zero();
    }
    inline bool iszero(const tensile_bfloat8& a)
    {
        return a.is_zero();
    }

    inline tensile_float8 abs(const tensile_float8& a)
    {
        return tensile_float8(std::abs(float(a)));
    }
    inline tensile_bfloat8 abs(const tensile_bfloat8& a)
    {
        return tensile_bfloat8(std::abs(float(a)));
    }

    inline tensile_float8 sin(const tensile_float8& a)
    {
        return tensile_float8(std::sin(float(a)));
    }
    inline tensile_bfloat8 sin(const tensile_bfloat8& a)
    {
        return tensile_bfloat8(std::sin(float(a)));
    }

    inline tensile_float8 cos(const tensile_float8& a)
    {
        return tensile_float8(std::cos(float(a)));
    }
    inline tensile_bfloat8 cos(const tensile_bfloat8& a)
    {
        return tensile_bfloat8(std::cos(float(a)));
    }

} // namespace std
