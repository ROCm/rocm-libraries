/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
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

    template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
    HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

    template <int wm, int we, typename T, bool negative_zero_nan>
    HIP_HOST_DEVICE T cast_from_f8(uint8_t x);

} // namespace hip_f8_impl

#include "hip_f8_impl.h"

//  Naming convension of datatype in hip header file
//      float8: fp8
//      bfloat8: bf8
//      f8 is used to consider both float8 and bfloat8

namespace Tensile
{
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

        // constructor from bits, no casting to floating point
        //explicit HIP_HOST_DEVICE Float8_BFloat8<T>(uint8_t v)
        //{
        //    data = v;
        //}

        // constructor from float
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
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

            if(T == hip_f8_type::bf8)
            {
                // add clipping code.. by default, always clipping for now
                if((val.i32val & 0x7F800000) != 0x7F800000) // all exp bits  are 1 --> NaN or INF
                    val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);

                // TODO: make it compile-time
                if(rm == hip_f8_rounding_mode::stochastic)
                {
                    ival       = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
                    val.i32val = ival;
                    data       = val.i8val[0]; // little endian
                }
                else // RNE CVT
                {
                    ival = __builtin_amdgcn_cvt_pk_bf8_f32(
                        val.fval, val.fval, ival, false); // false -> WORD0
                    val.i32val = ival;
                    data       = val.i8val[0];
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
                    data       = val.i8val[0]; // little endian
                }
                else // RNE CVT
                {
                    ival = __builtin_amdgcn_cvt_pk_fp8_f32(
                        val.fval, val.fval, ival, false); // false -> WORD0
                    val.i32val = ival;
                    data       = val.i8val[0];
                }
            }
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
                        cast_to_f8<2, 5, float, true /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
                else
                {
                    data = tensile_hip_f8_impl::
                        cast_to_f8<2, 5, float, false /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
            }
            else /* fp8*/
            {
                if(get_hip_f8_bias_mode())
                {
                    data = tensile_hip_f8_impl::
                        cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
                else
                {
                    data = tensile_hip_f8_impl::
                        cast_to_f8<3, 4, float, false /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
            }
        }

        // constructor from half
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        // no h/w inst for cvt from f16, just convert f16 to f32 and call constructor
        explicit HIP_DEVICE Float8_BFloat8(_Float16             v,
                                           hip_f8_rounding_mode rm = hip_f8_rounding_mode::standard,
                                           uint32_t             rng = 0)
            : Float8_BFloat8((float)v, rm, rng)
        {
        }

        explicit HIP_HOST
#else // gfx940/gfx941/gfx942
        explicit HIP_HOST_DEVICE
#endif // gfx940/gfx941/gfx942
            Float8_BFloat8(_Float16             v,
                           hip_f8_rounding_mode rm  = hip_f8_rounding_mode::standard,
                           uint32_t             rng = 0)
        {
            // NOTE: made clipping default again
            if(T == hip_f8_type::bf8)
            {
                if(get_hip_f8_bias_mode())
                {
                    data = tensile_hip_f8_impl::
                        cast_to_f8<2, 5, _Float16, true /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
                else
                {
                    data = tensile_hip_f8_impl::
                        cast_to_f8<2, 5, _Float16, false /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
            }
            else /* fp8*/
            {
                if(get_hip_f8_bias_mode())
                {
                    data = tensile_hip_f8_impl::
                        cast_to_f8<3, 4, _Float16, true /*negative_zero_nan*/, true /*clip*/>(
                            v, (rm == hip_f8_rounding_mode::stochastic), rng);
                }
                else
                {
                    data = tensile_hip_f8_impl::
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
                    return tensile_hip_f8_impl::
                        cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(data);
                }
                else
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<2, 5, float, false /*negative_zero_nan*/>(data);
                }
            }
            else /* fp8*/
            {
                if(get_hip_f8_bias_mode())
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(data);
                }
                else
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<3, 4, float, false /*negative_zero_nan*/>(data);
                }
            }
        }

        // convert to half
        // NOTE: no hardware instruction to convert from and to f8, may want to convert it f32 first
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        explicit inline HIP_DEVICE operator _Float16() const
        {
            return _Float16(float(*this)); // convert to float, then convert to f16
        }

        explicit inline HIP_HOST operator _Float16() const
#else // gfx940/gfx941/gfx942
        explicit inline HIP_HOST_DEVICE operator _Float16() const
#endif // gfx940/gfx941/gfx942
        {
            if(T == hip_f8_type::bf8)
            {
                if(get_hip_f8_bias_mode())
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<2, 5, _Float16, true /*negative_zero_nan*/>(data);
                }
                else
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<2, 5, _Float16, false /*negative_zero_nan*/>(data);
                }
            }
            else /* fp8*/
            {
                if(get_hip_f8_bias_mode())
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<3, 4, _Float16, true /*negative_zero_nan*/>(data);
                }
                else
                {
                    return tensile_hip_f8_impl::
                        cast_from_f8<3, 4, _Float16, false /*negative_zero_nan*/>(data);
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
    typedef Float8_BFloat8<hip_f8_type::fp8> Float8;
    typedef Float8_BFloat8<hip_f8_type::bf8> BFloat8;

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
    inline Float8 explicit_downcast(float a, hip_f8_rounding_mode rm, uint32_t rng)
    {
        return Float8(a, rm, rng);
    }

    // bfloat8 ........
    template <>
    inline BFloat8 explicit_downcast(float a, hip_f8_rounding_mode rm, uint32_t rng)
    {
        return BFloat8(a, rm, rng);
    }
} // end of namespace Tensile

namespace std
{
    inline bool isinf(const Tensile::Float8& a)
    {
        return a.is_inf();
    }
    inline bool isinf(const Tensile::BFloat8& a)
    {
        return a.is_inf();
    }

    inline bool isnan(const Tensile::Float8& a)
    {
        return a.is_nan();
    }
    inline bool isnan(const Tensile::BFloat8& a)
    {
        return a.is_nan();
    }
    inline bool iszero(const Tensile::Float8& a)
    {
        return a.is_zero();
    }
    inline bool iszero(const Tensile::BFloat8& a)
    {
        return a.is_zero();
    }

    inline Tensile::Float8 abs(const Tensile::Float8& a)
    {
        return Tensile::Float8(std::abs(float(a)));
    }
    inline Tensile::BFloat8 abs(const Tensile::BFloat8& a)
    {
        return Tensile::BFloat8(std::abs(float(a)));
    }

    inline Tensile::Float8 sin(const Tensile::Float8& a)
    {
        return Tensile::Float8(std::sin(float(a)));
    }
    inline Tensile::BFloat8 sin(const Tensile::BFloat8& a)
    {
        return Tensile::BFloat8(std::sin(float(a)));
    }

    inline Tensile::Float8 cos(const Tensile::Float8& a)
    {
        return Tensile::Float8(std::cos(float(a)));
    }
    inline Tensile::BFloat8 cos(const Tensile::BFloat8& a)
    {
        return Tensile::BFloat8(std::cos(float(a)));
    }

} // namespace std
