/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCBLAS_FLOAT8_H
#define ROCBLAS_FLOAT8_H

#if __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))
/*! \brief Struct to represent a 8 bit floating-point number. */
typedef struct
{
    uint8_t data;
} rocblas_f8;

typedef struct
{
    uint8_t data;
} rocblas_bf8;

#else // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

// We are clipping in down conversion by default
#define rocblas_F8_downcast_clipping 1

namespace rocblas_hip_f8_impl
{

    template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
    HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

    template <int wm, int we, typename T, bool negative_zero_nan>
    HIP_HOST_DEVICE T cast_from_f8(uint8_t x);

} // namespace rocblas_hip_f8_impl

#include "rocblas_hip_f8_impl.h"

static __device__ bool rocblas_hip_f8_bias_mode_bit_device = true;
static bool            rocblas_hip_f8_bias_mode_bit_host   = true;

struct ROCBLAS_EXPORT rocblas_f8
{
    uint8_t data;
    enum class rocblas_hip_f8_rounding_mode
    {
        standard,
        stochastic
    };

    // default constructor
    HIP_HOST_DEVICE rocblas_f8() = default;

#if defined(__gfx942__)
    // device specific optimized F8 down-conversion code

    template <bool stochastic_rounding = false>
    static HIP_DEVICE uint8_t cast_to_f8_from_f32(float v, uint32_t rng = 0)
    {
        uint8_t i8data;
        union
        {
            float    fval;
            uint32_t i32val;
            uint8_t  i8val[4]; // NOTE: not endian independent
        } val;

        uint32_t ival = 0;
        val.fval      = v;

#ifdef rocblas_F8_downcast_clipping
        if((val.i32val & 0x7F800000) != 0x7F800000) /// propagate NAN/INF, no clipping
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
        return i8data;
    }

#endif // __gfx942__

    // constructor from float
#if defined(__gfx942__)

    // NOTE: ON-DEVICE... always optimal bias
    explicit HIP_DEVICE rocblas_f8(float                        v,
                                   rocblas_hip_f8_rounding_mode rm
                                   = rocblas_hip_f8_rounding_mode::standard,
                                   uint32_t rng = 0)
    {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
        if(rm == rocblas_hip_f8_rounding_mode::stochastic)
            data = cast_to_f8_from_f32<true>(v, rng);
        else
            data = cast_to_f8_from_f32<false>(v);
    }

    // Host only implementation using s/w simulation
    explicit HIP_HOST
#else
    // both Host and DEVICE for non-gfx942 using s/w simulation
    explicit HIP_HOST_DEVICE
#endif
        rocblas_f8(float                        v,
                   rocblas_hip_f8_rounding_mode rm  = rocblas_hip_f8_rounding_mode::standard,
                   uint32_t                     rng = 0)
    {
#ifdef rocblas_F8_downcast_clipping
        data = rocblas_hip_f8_impl::
            cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
                v, (rm == rocblas_hip_f8_rounding_mode::stochastic), rng);
#else // rocblas_F8_downcast_clipping
        data = rocblas_hip_f8_impl::
            cast_to_f8<3, 4, float, true /*negative_zero_nan*/, false /*clip*/>(
                v, (rm == rocblas_hip_f8_rounding_mode::stochastic), rng);
#endif // rocblas_F8_downcast_clipping
    }

    // Constructor from half
    explicit HIP_HOST_DEVICE rocblas_f8(_Float16                     v,
                                        rocblas_hip_f8_rounding_mode rm
                                        = rocblas_hip_f8_rounding_mode::standard,
                                        uint32_t rng = 0)
        : rocblas_f8((float)v, rm, rng)
    {
    }
    // constructor from bfloat16
    explicit HIP_HOST_DEVICE rocblas_f8(rocblas_bfloat16             v,
                                        rocblas_hip_f8_rounding_mode rm
                                        = rocblas_hip_f8_rounding_mode::standard,
                                        uint32_t rng = 0)
        : rocblas_f8((float)v, rm, rng)
    {
    }
    // constructor from int
    explicit HIP_HOST_DEVICE rocblas_f8(int                          v,
                                        rocblas_hip_f8_rounding_mode rm
                                        = rocblas_hip_f8_rounding_mode::standard,
                                        uint32_t rng = 0)
        : rocblas_f8((float)v, rm, rng)
    {
    }
    // constructor from double
    explicit HIP_HOST_DEVICE rocblas_f8(double                       v,
                                        rocblas_hip_f8_rounding_mode rm
                                        = rocblas_hip_f8_rounding_mode::standard,
                                        uint32_t rng = 0)
        : rocblas_f8((float)v, rm, rng)
    {
    }

    // convert to float
#if defined(__gfx942__)
    // upcast using device specific intrinsic
    explicit inline HIP_DEVICE operator float() const
    {
        float    fval;
        uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
        asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));

        return fval;
    }

    explicit inline HIP_HOST operator float() const
#else // non __gfx942__
    explicit inline HIP_HOST_DEVICE operator float() const
#endif
    {
        return rocblas_hip_f8_impl::cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(data);
    }

    // convert to half
    explicit inline HIP_HOST_DEVICE operator _Float16() const
    {
        return _Float16(float(*this)); // convert to float, then convert to f16
    }

    // convert to bfloat16
    explicit inline HIP_HOST_DEVICE operator rocblas_bfloat16() const
    {
        return rocblas_bfloat16(float(*this)); // convert to float, then convert to f16
    }

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        return data == 0x00;
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        return data == 0x80;
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        return data == 0x80;
    }

    // assignment overloading only from the same F8 types
    inline __host__ __device__ rocblas_f8& operator=(const rocblas_f8& a)
    {
        data = a.data;
        return *this;
    }
};

struct ROCBLAS_EXPORT rocblas_bf8
{
    uint8_t data;
    enum class rocblas_hip_f8_rounding_mode
    {
        standard,
        stochastic
    };

    // default constructor
    HIP_HOST_DEVICE rocblas_bf8() = default;

#if defined(__gfx942__)
    // device specific optimized F8 down-conversion code

    template <bool stochastic_rounding = false>
    static HIP_DEVICE uint8_t cast_to_bf8_from_f32(float v, uint32_t rng = 0)
    {
        uint8_t i8data;
        union
        {
            float    fval;
            uint32_t i32val;
            uint8_t  i8val[4]; // NOTE: not endian independent
        } val;

        uint32_t ival = 0;
        val.fval      = v;

#ifdef rocblas_F8_downcast_clipping
        if((val.i32val & 0x7F800000) != 0x7F800000) // propagate NAN/INF, no clipping
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
        return i8data;
    }

#endif // __gfx942__

    // constructor from float
#if defined(__gfx942__)

    // NOTE: ON-DEVICE... always optimal bias
    explicit HIP_DEVICE rocblas_bf8(float                        v,
                                    rocblas_hip_f8_rounding_mode rm
                                    = rocblas_hip_f8_rounding_mode::standard,
                                    uint32_t rng = 0)
    {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
        if(rm == rocblas_hip_f8_rounding_mode::stochastic)
            data = cast_to_bf8_from_f32<true>(v, rng);
        else
            data = cast_to_bf8_from_f32<false>(v);
    }

    // Host only implementation using s/w simulation
    explicit HIP_HOST
#else
    // both Host and DEVICE for non-gfx942 using s/w simulation
    explicit HIP_HOST_DEVICE
#endif
        rocblas_bf8(float                        v,
                    rocblas_hip_f8_rounding_mode rm  = rocblas_hip_f8_rounding_mode::standard,
                    uint32_t                     rng = 0)
    {
#ifdef rocblas_F8_downcast_clipping
        data = rocblas_hip_f8_impl::
            cast_to_f8<2, 5, float, true /*negative_zero_nan*/, true /*clip*/>(
                v, (rm == rocblas_hip_f8_rounding_mode::stochastic), rng);
#else
        data = rocblas_hip_f8_impl::
            cast_to_f8<2, 5, float, true /*negative_zero_nan*/, false /*clip*/>(
                v, (rm == rocblas_hip_f8_rounding_mode::stochastic), rng);
#endif // rocblas_F8_downcast_clipping
    }

    // Constructor from half
    explicit HIP_HOST_DEVICE rocblas_bf8(_Float16                     v,
                                         rocblas_hip_f8_rounding_mode rm
                                         = rocblas_hip_f8_rounding_mode::standard,
                                         uint32_t rng = 0)
        : rocblas_bf8((float)v, rm, rng)
    {
    }
    // constructor from bfloat16
    explicit HIP_HOST_DEVICE rocblas_bf8(rocblas_bfloat16             v,
                                         rocblas_hip_f8_rounding_mode rm
                                         = rocblas_hip_f8_rounding_mode::standard,
                                         uint32_t rng = 0)
        : rocblas_bf8((float)v, rm, rng)
    {
    }
    // constructor from int
    explicit HIP_HOST_DEVICE rocblas_bf8(int                          v,
                                         rocblas_hip_f8_rounding_mode rm
                                         = rocblas_hip_f8_rounding_mode::standard,
                                         uint32_t rng = 0)
        : rocblas_bf8((float)v, rm, rng)
    {
    }
    // constructor from double
    explicit HIP_HOST_DEVICE rocblas_bf8(double                       v,
                                         rocblas_hip_f8_rounding_mode rm
                                         = rocblas_hip_f8_rounding_mode::standard,
                                         uint32_t rng = 0)
        : rocblas_bf8((float)v, rm, rng)
    {
    }

    // convert to float
#if defined(__gfx942__)
    // upcast using device specific intrinsic
    explicit inline HIP_DEVICE operator float() const
    {
        float    fval;
        uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
        asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));

        return fval;
    }

    explicit inline HIP_HOST operator float() const
#else // non __gfx942__
    explicit inline HIP_HOST_DEVICE operator float() const
#endif
    {
        return rocblas_hip_f8_impl::cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(data);
    }

    // convert to half
    explicit inline HIP_HOST_DEVICE operator _Float16() const
    {
        return _Float16(float(*this)); // convert to float, then convert to f16
    }

    // convert to bfloat16
    explicit inline HIP_HOST_DEVICE operator rocblas_bfloat16() const
    {
        return rocblas_bfloat16(float(*this)); // convert to float, then convert to f16
    }

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        return data == 0x00;
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        return data == 0x80;
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        return data == 0x80;
    }

    // assignment overloading only from the same F8 types
    inline __host__ __device__ rocblas_bf8& operator=(const rocblas_bf8& a)
    {
        data = a.data;
        return *this;
    }
};

namespace std
{
    inline rocblas_f8 sin(rocblas_f8 a)
    {
        return rocblas_f8(sinf(float(a)));
    }
    inline rocblas_f8 cos(rocblas_f8 a)
    {
        return rocblas_f8(cosf(float(a)));
    }
    inline rocblas_bf8 sin(rocblas_bf8 a)
    {
        return rocblas_bf8(sinf(float(a)));
    }
    inline rocblas_bf8 cos(rocblas_bf8 a)
    {
        return rocblas_bf8(cosf(float(a)));
    }
    __device__ __host__ constexpr rocblas_f8 real(const rocblas_f8& a)
    {
        return a;
    }
    __device__ __host__ constexpr rocblas_bf8 real(const rocblas_bf8& a)
    {
        return a;
    }
}

// Special operator overloading
inline std::ostream& operator<<(std::ostream& os, const rocblas_f8& f8)
{
    return os << float(f8);
}

inline std::ostream& operator<<(std::ostream& os, const rocblas_bf8& bf8)
{
    return os << float(bf8);
}

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline __host__ __device__ float operator+(const float fa, rocblas_f8 b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(const float fa, rocblas_bf8 b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(rocblas_f8 a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(rocblas_bf8 a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(rocblas_f8 a, rocblas_bf8 b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ float operator+(rocblas_bf8 a, rocblas_f8 b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ rocblas_f8 operator+(rocblas_f8 a, rocblas_f8 b)
{
    return rocblas_f8(float(a) + float(b));
}

inline __host__ __device__ rocblas_bf8 operator+(rocblas_bf8 a, rocblas_bf8 b)
{
    return rocblas_bf8(float(a) + float(b));
}

inline __host__ __device__ rocblas_f8& operator+=(rocblas_f8& a, rocblas_f8 b)
{
    return a = rocblas_f8(float(a) + float(b));
}

inline __host__ __device__ rocblas_bf8& operator+=(rocblas_bf8& a, rocblas_bf8 b)
{
    return a = rocblas_bf8(float(a) + float(b));
}

// overloading multiplication, always returns float,
inline __host__ __device__ float operator*(rocblas_f8 a, rocblas_f8 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, rocblas_f8 b)
{
    return (a * float(b));
}

inline __host__ __device__ float operator*(rocblas_f8 a, float b)
{
    return (float(a) * b);
}

inline __host__ __device__ float operator*(int32_t a, rocblas_f8 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, rocblas_f8 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(rocblas_bf8 a, rocblas_bf8 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, rocblas_bf8 b)
{
    return (a * float(b));
}

inline __host__ __device__ float operator*(rocblas_bf8 a, float b)
{
    return (float(a) * b);
}

inline __host__ __device__ float operator*(int32_t a, rocblas_bf8 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, rocblas_bf8 b)
{
    return ((float)a * float(b));
}

// overloading for mixed f8 and bf8 types
inline __host__ __device__ float operator*(rocblas_f8 a, rocblas_bf8 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(rocblas_bf8 a, rocblas_f8 b)
{
    return float(a) * float(b);
}

// all - operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline __host__ __device__ float operator-(const float fa, rocblas_f8 b)
{
    return (fa - float(b));
}

inline __host__ __device__ float operator-(const float fa, rocblas_bf8 b)
{
    return (fa - float(b));
}

inline __host__ __device__ float operator-(rocblas_f8 a, const float fb)
{
    return (float(a) - fb);
}

inline __host__ __device__ float operator-(rocblas_bf8 a, const float fb)
{
    return (float(a) - fb);
}

inline __host__ __device__ float operator-(rocblas_f8 a, rocblas_bf8 b)
{
    return (float(a) - float(b));
}

inline __host__ __device__ float operator-(rocblas_bf8 a, rocblas_f8 b)
{
    return (float(a) - float(b));
}

inline __host__ __device__ rocblas_f8 operator-(rocblas_f8 a, rocblas_f8 b)
{
    return rocblas_f8(float(a) - float(b));
}

inline __host__ __device__ rocblas_bf8 operator-(rocblas_bf8 a, rocblas_bf8 b)
{
    return rocblas_bf8(float(a) - float(b));
}

inline __host__ __device__ rocblas_f8& operator-=(rocblas_f8& a, rocblas_f8 b)
{
    return a = rocblas_f8(float(a) - float(b));
}

inline __host__ __device__ rocblas_bf8& operator-=(rocblas_bf8& a, rocblas_bf8 b)
{
    return a = rocblas_bf8(float(a) - float(b));
}

// overloading division, always returns float,
inline __host__ __device__ float operator/(rocblas_f8 a, rocblas_f8 b)
{
    return float(a) / float(b);
}

inline __host__ __device__ float operator/(float a, rocblas_f8 b)
{
    return (a / float(b));
}

inline __host__ __device__ float operator/(rocblas_f8 a, float b)
{
    return (float(a) / b);
}

inline __host__ __device__ float operator/(int32_t a, rocblas_f8 b)
{
    return ((float)a / float(b));
}

inline __host__ __device__ float operator/(double a, rocblas_f8 b)
{
    return ((float)a / float(b));
}

inline __host__ __device__ float operator/(rocblas_bf8 a, rocblas_bf8 b)
{
    return float(a) / float(b);
}

inline __host__ __device__ float operator/(float a, rocblas_bf8 b)
{
    return (a / float(b));
}

inline __host__ __device__ float operator/(rocblas_bf8 a, float b)
{
    return (float(a) / b);
}

inline __host__ __device__ float operator/(int32_t a, rocblas_bf8 b)
{
    return ((float)a / float(b));
}

inline __host__ __device__ float operator/(double a, rocblas_bf8 b)
{
    return ((float)a / float(b));
}

// overloading for mixed f8 and bf8 types
inline __host__ __device__ float operator/(rocblas_f8 a, rocblas_bf8 b)
{
    return float(a) / float(b);
}

inline __host__ __device__ float operator/(rocblas_bf8 a, rocblas_f8 b)
{
    return float(a) / float(b);
}

// overloading for compare
inline __host__ __device__ bool operator==(rocblas_f8 a, rocblas_f8 b)
{
    return (a.data == b.data);
}

inline __host__ __device__ bool operator==(rocblas_bf8 a, rocblas_bf8 b)
{
    return (a.data == b.data);
}

inline __host__ __device__ bool operator!=(rocblas_f8 a, rocblas_f8 b)
{
    return (a.data != b.data);
}

inline __host__ __device__ bool operator!=(rocblas_bf8 a, rocblas_bf8 b)
{
    return (a.data != b.data);
}

// ================ Explicit downcasting to support different rounding (RNE, SR) ===============
// NOTE: we going to remove all assignment operator overloading from other types and enforce
// this explicit_downcast function to make any roudning behavior default
// We have to explicitly call this function with SR flag

template <typename T,
          typename Ta,
          bool stochastic_rounding,
          typename std::enable_if<std::is_same<T, Ta>{}, int>::type = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng = 0)
{
    // same type, no conversion
    return a;
}

// Use h/w intrinsic and optimized version when __gfx942__
template <
    typename T,
    typename Ta,
    bool stochastic_rounding,
    typename std::enable_if<(!(std::is_same<T, Ta>{})
                             && (std::is_same<T, rocblas_f8>{} || std::is_same<T, rocblas_bf8>{})),
                            int>::type
    = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng)
{
#if defined(__gfx942__)
    // NOTE: we are directly calling cast_to_f8_from_f32 instead of constructor to optimize away one runtime branch
    T val;
    if(std::is_same<T, rocblas_f8>::value)
        val.data = rocblas_f8::cast_to_f8_from_f32<stochastic_rounding>(float(a), rng);
    else
        val.data = rocblas_bf8::cast_to_bf8_from_f32<stochastic_rounding>(float(a), rng);
    return val;
#else // non __gfx942__
    return T(float(a),
             stochastic_rounding ? T::rocblas_hip_f8_rounding_mode::stochastic
                                 : T::rocblas_hip_f8_rounding_mode::standard,
             rng);
#endif // __gfx942__
}

// NOTE NOTE: The above code is good if we don't consider HIP-GEMM code and only consider the quantization
// However, if we need HIP-GEMM for fall-back, we would need explicit_cast handles Tacc=f32 to To=f16/bf16 conversion
template <
    typename T,
    typename Ta,
    bool stochastic_rounding,
    typename std::enable_if<(!(std::is_same<T, Ta>{})
                             && !(std::is_same<T, rocblas_f8>{} || std::is_same<T, rocblas_bf8>{})),
                            int>::type
    = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng)
{
    // the return type is not a F8 types, no SR for those types
    // not sure if we have direct conversion, so converting to float first
    // no effect if the input type is float
    return T(float(a));
}

// =================================================================================================

#endif // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#endif // ROCBLAS_FLOAT8_H
