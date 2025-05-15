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

#ifndef BATCHNORM_FUNCTIONS_HPP
#define BATCHNORM_FUNCTIONS_HPP

#ifndef MIOPEN_USE_FPMIX
#define MIOPEN_USE_FPMIX 0
#endif

#ifndef MIOPEN_USE_BFPMIX
#define MIOPEN_USE_BFPMIX 0
#endif

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "bfloat16_dev.hpp"
#include "miopen_type_traits.hpp"

#ifndef MIOPEN_USE_FPMIX
#define MIOPEN_USE_FPMIX 0
#endif

#ifndef MIOPEN_USE_BFPMIX
#define MIOPEN_USE_BFPMIX 0
#endif

#if MIOPEN_USE_FP16 == 1
#define FP_TYPE ::__half
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
#define FP_TYPE ::__half
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
#define FP_TYPE ushort

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
#endif

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

// TODO: Spaghetti code!!!
// MIOPEN_USE_AMDGCN may be defined before this header.
#ifndef MIOPEN_USE_AMDGCN
#if defined(__AMDGCN__) &&                           \
    !((defined(MIO_BN_GFX103X) && MIO_BN_GFX103X) || \
      (defined(MIO_BN_GFX110X) && MIO_BN_GFX110X) || (defined(MIO_BN_GFX120X) && MIO_BN_GFX120X))
#define MIOPEN_USE_AMDGCN 1
#else
#define MIOPEN_USE_AMDGCN 0
#endif
#endif

// MIOPEN_USE_AMDGCN is guaranteed to be defined at this point.

#ifndef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#elif (MIO_BN_NODPP == 1 && MIO_BN_VARIANT != 0)
#undef MIOPEN_USE_AMDGCN
#define MIOPEN_USE_AMDGCN 0
#endif

#ifndef MIO_SAVE_MEAN_VARIANCE
#define MIO_SAVE_MEAN_VARIANCE 0
#endif

#ifndef MIO_RUNNING_RESULT
#define MIO_RUNNING_RESULT 0
#endif

#ifndef MIO_BN_GFX103X
#define MIO_BN_GFX103X 0
#endif

#ifndef MIO_BN_GFX110X
#define MIO_BN_GFX110X 0
#endif

#ifndef MIO_BN_GFX120X
#define MIO_BN_GFX120X 0
#endif

#ifndef MIO_BN_VECTORIZE
#define MIO_BN_VECTORIZE 0
#endif

#ifndef MIO_BN_STASH_METHOD
#define MIO_BN_STASH_METHOD 0
#endif

static constexpr unsigned int VEC_SIZE = static_cast<bool>(MIO_BN_VECTORIZE) ? 4 : 1;
static constexpr unsigned int VEC_SIZE_X =
    static_cast<bool>(MIO_BN_VECTORIZE) && static_cast<bool>(MIO_LAYOUT_NHWC) ? VEC_SIZE : 1;
static constexpr unsigned int VEC_SIZE_Y =
    static_cast<bool>(MIO_BN_VECTORIZE) && (!static_cast<bool>(MIO_LAYOUT_NHWC)) ? VEC_SIZE : 1;

struct __half4
{
    static_assert(MIOPEN_USE_FP16 == 1 || MIOPEN_USE_FPMIX == 1,
                  "__half4 is not provided is this situation.");

    ::__half x, y, z, w;

    __host__ __device__ __half4() = default;

    __host__ __device__ __half4(::__half a, ::__half b, ::__half c, ::__half d)
        : x(a), y(b), z(c), w(d)
    {
    }

    __host__ __device__ __half4& operator=(const __half4&) = default;
};

// TODO: implement __half8

template <typename T, int N>
struct mapped_vector_type
{
    static_assert(false, "there is no specialization for this T & N combination.");
};

// Note: we should make sure there is no exception
template <typename T>
struct mapped_vector_type<T, 1>
{
    using type = T;
};

// float
template <>
struct mapped_vector_type<float, 2>
{
    using type = ::float2;
};

template <>
struct mapped_vector_type<float, 4>
{
    using type = ::float4;
};

// half
template <>
struct mapped_vector_type<::__half, 4>
{
    using type = ::__half4;
};

// TODO: implement half2

// ushort
template <>
struct mapped_vector_type<ushort, 4>
{
    using type = ushort4;
};

template <>
struct mapped_vector_type<ushort, 2>
{
    using type = ushort2;
};

// int
template <>
struct mapped_vector_type<int, 4>
{
    using type = int4;
};

template <>
struct mapped_vector_type<int, 2>
{
    using type = int2;
};

using _FpAccum = float;

// _C suffix means used for computation
using _FpPrec_C =
    typename std::conditional<static_cast<bool>(MIO_BN_VECTORIZE) &&
                                  static_cast<bool>(MIO_LAYOUT_NHWC),
                              typename mapped_vector_type<FP_TYPE_PREC, VEC_SIZE>::type,
                              FP_TYPE_PREC>::type;

// _LS suffix means used for loading / storing
using _FpPrec_LS =
    typename std::conditional<static_cast<bool>(MIO_BN_VECTORIZE),
                              typename mapped_vector_type<FP_TYPE_PREC, VEC_SIZE>::type,
                              FP_TYPE_PREC>::type;

// _C suffix means used for computation
using _Fp_C = typename std::conditional<static_cast<bool>(MIO_BN_VECTORIZE) &&
                                            static_cast<bool>(MIO_LAYOUT_NHWC),
                                        typename mapped_vector_type<FP_TYPE, VEC_SIZE>::type,
                                        FP_TYPE>::type;

// _LS suffix means used for loading / storing
using _Fp_LS = typename std::conditional<static_cast<bool>(MIO_BN_VECTORIZE),
                                         typename mapped_vector_type<FP_TYPE, VEC_SIZE>::type,
                                         FP_TYPE>::type;

// _C suffix means used for computation
using _FpAccum_C = typename std::conditional<static_cast<bool>(MIO_BN_VECTORIZE) &&
                                                 static_cast<bool>(MIO_LAYOUT_NHWC),
                                             typename mapped_vector_type<_FpAccum, VEC_SIZE>::type,
                                             _FpAccum>::type;

// _LS suffix means used for loading / storing
using _FpAccum_LS = typename std::conditional<static_cast<bool>(MIO_BN_VECTORIZE),
                                              typename mapped_vector_type<_FpAccum, VEC_SIZE>::type,
                                              _FpAccum>::type;

// Hip does have fma which does the same thing as OpenCL mad
template <typename T>
struct hip_mad
{
    static_assert(sizeof(T) == sizeof(::__half) || sizeof(T) == sizeof(float) ||
                      sizeof(T) == sizeof(double),
                  "Input floating point type size is wrong!");
    __forceinline__ __device__ auto operator()(T _1, T _2, T _3)
    {
        if constexpr(sizeof(T) == sizeof(::__half))
        {
            // TODO: I don't know if this is right, it uses _Float16
            return fma(
                static_cast<_Float16>(_1), static_cast<_Float16>(_2), static_cast<_Float16>(_3));
        }
        else if constexpr(sizeof(T) == sizeof(float))
        {
            return fma(static_cast<float>(_1), static_cast<float>(_2), static_cast<float>(_3));
        }
        else
        {
            return fma(static_cast<double>(_1), static_cast<double>(_2), static_cast<double>(_3));
        }
    }
};

// Conversion functions

template <typename FpPrecType, typename FpType>
__forceinline__ __device__ __host__ FpPrecType fp_to_fpprec(FpType x)
{
    if constexpr(MIOPEN_USE_BFPMIX == 1)
    {
        static_assert(std::is_same<FpType, ushort>::value && std::is_same<FpPrecType, float>::value,
                      "when MIOPEN_USE_BFPMIX == 1, FpType must be ushort(bfloat16), and "
                      "FpPrecType must be float");
        return bfloat16_to_float(x);
    }
    else
    {
        return static_cast<FpPrecType>(x);
    }
}

template <typename FpType, typename FpPrecType>
__forceinline__ __device__ __host__ FpType fpprec_to_fp(FpPrecType x)
{
    if constexpr(MIOPEN_USE_BFPMIX == 1)
    {
        static_assert(std::is_same<FpType, ushort>::value && std::is_same<FpPrecType, float>::value,
                      "when MIOPEN_USE_BFPMIX == 1, FpType must be ushort(bfloat16), and "
                      "FpPrecType must be float");
        return float_2_bfloat16(x);
    }
    else
    {
        return static_cast<FpType>(x);
    }
}

template <typename FpAccumType, typename FpType>
__forceinline__ __device__ __host__ FpAccumType fp_to_fpaccum(FpType x)
{
    if constexpr(MIOPEN_USE_BFPMIX == 1)
    {
        static_assert(std::is_same<decltype(fp_to_fpaccum(x)), decltype(fp_to_fpprec(x))>::value,
                      "In this case FpAccumType must be equal to "
                      "FpPrecType.");
        return fp_to_fpprec(x);
    }
    else
    {
        return static_cast<FpAccumType>(x);
    }
}

template <typename FpType, typename FpAccumType>
__forceinline__ __device__ __host__ FpType fpaccum_to_fp(FpAccumType x)
{
    if constexpr(MIOPEN_USE_BFPMIX == 1)
    {
        static_assert(std::is_same<decltype(fpaccum_to_fp(x)), decltype(fpprec_to_fp(x))>::value,
                      "In this case FpAccumType must be equal to "
                      "FpPrecType.");
        return fpprec_to_fp(x);
    }
    else
    {
        return static_cast<FpType>(x);
    }
}

template <typename Fp4Type, typename FpPrec4Type>
__forceinline__ __device__ __host__ Fp4Type fpprec4_to_fp4(FpPrec4Type const& val)
{
    return Fp4Type(fpprec_to_fp<FP_TYPE>(val.x),
                   fpprec_to_fp<FP_TYPE>(val.y),
                   fpprec_to_fp<FP_TYPE>(val.z),
                   fpprec_to_fp<FP_TYPE>(val.w));
}

template <typename FpPrec4Type, typename Fp4Type>
__forceinline__ __device__ __host__ FpPrec4Type fp4_to_fpprec4(Fp4Type const& val)
{
    return FpPrec4Type(fp_to_fpprec<FP_TYPE_PREC>(val.x),
                       fp_to_fpprec<FP_TYPE_PREC>(val.y),
                       fp_to_fpprec<FP_TYPE_PREC>(val.z),
                       fp_to_fpprec<FP_TYPE_PREC>(val.w));
}

template <typename FpPrecVecType, typename FpVecType>
__forceinline__ __device__ __host__ FpPrecVecType fp_to_fpprec_vec(FpVecType const& val)
{
    if constexpr(MIO_BN_VECTORIZE)
    {
        return fp4_to_fpprec4<FpPrecVecType>(val);
    }
    else
    {
        return fp_to_fpprec<FpPrecVecType>(val);
    }
}

template <typename FpVecType, typename FpPrecVecType>
__forceinline__ __device__ __host__ FpVecType fpprec_to_fp_vec(FpPrecVecType const& val)
{
    if constexpr(MIO_BN_VECTORIZE)
    {
        return fpprec4_to_fp4<FpVecType>(val);
    }
    else
    {
        return fpprec_to_fp<FpVecType>(val);
    }
}

// Accumulate functions

template <typename T>
__forceinline__ __device__ __host__ void _accumulate1(T& a, T const& b)
{
    a += b;
}

template <typename T>
__forceinline__ __device__ __host__ void _accumulate_mad1(T& a, T const& b, T const& c, T const& d)
{
    a = fma(b, c, d);
}

template <typename T>
__forceinline__ __device__ __host__ void
_accumulate4(T& a, typename mapped_vector_type<T, 4>::type const& b)
{
    a += b.x;
    a += b.y;
    a += b.z;
    a += b.w;
}

template <typename T>
__forceinline__ __device__ __host__ void
_accumulate_mad4(T& a,
                 typename mapped_vector_type<T, 4>::type const& b,
                 typename mapped_vector_type<T, 4>::type const& c,
                 T const& d)
{
    a = fma(b.x, c.x, d);
    a = fma(b.y, c.y, d);
    a = fma(b.z, c.z, d);
    a = fma(b.w, c.w, d);
}

template <typename T>
__forceinline__ __device__ __host__ void _accumulate(T& a, T const& b)
{
    if constexpr(static_cast<bool>(MIO_BN_VECTORIZE) && (!static_cast<bool>(MIO_LAYOUT_NHWC)))
    {
        _accumulate4(a, b);
    }
    else
    {
        _accumulate1(a, b);
    }
}

template <typename T>
__forceinline__ __device__ __host__ void
_accumulate_mad(T& a,
                typename mapped_vector_type<T, 4>::type const& b,
                typename mapped_vector_type<T, 4>::type const& c,
                T const& d)
{
    static_assert(static_cast<bool>(MIO_BN_VECTORIZE) && (!static_cast<bool>(MIO_LAYOUT_NHWC)),
                  "_accumulate_mad for this particular arg list is disabled.");
    _accumulate_mad4(a, b, c, d);
}

template <typename T>
__forceinline__ __device__ __host__ void _accumulate_mad(T& a, T const& b, T const& c, T const& d)
{
    static_assert(!(static_cast<bool>(MIO_BN_VECTORIZE) && (!static_cast<bool>(MIO_LAYOUT_NHWC))),
                  "_accumulate_mad for this particular arg list is disabled.");
    _accumulate_mad1(a, b, c, d);
}

// _C suffix means used for computation
// _LS suffix means used for loading / storing
template <typename FpAccum, typename FpAccum_C, typename FpPrec_C>
__forceinline__ __device__ void running_stash(FpPrec_C* __restrict resultRunningMean,
                                              FpPrec_C* __restrict resultRunningVariance,
                                              double expAvgFactor,
                                              FpAccum_C mean,
                                              FpAccum_C variance,
                                              uint channel)
{
    static_assert(MIO_BN_VARIANT != 4, "running_stash is only compiled when MIO_BN_VARIANT != 4.");

    const auto pvt_runMean = static_cast<FpAccum_C>(resultRunningMean[channel]);

    const auto pvt_newRunMean = fma(static_cast<FpAccum_C>(-expAvgFactor),
                                    static_cast<FpAccum_C>(pvt_runMean),
                                    static_cast<FpAccum_C>(pvt_runMean)); // tmp = oldRunMean

    resultRunningMean[channel] =
        static_cast<FpPrec_C>(fma(static_cast<FpAccum_C>(mean),
                                  static_cast<FpAccum_C>(expAvgFactor),
                                  static_cast<FpAccum_C>(pvt_newRunMean))); // newMean*factor + tmp

    const FpAccum_C adjust = static_cast<FpAccum_C>(
        (MIO_BN_NHW == 1) ? variance
                          : variance * (static_cast<FpAccum>(MIO_BN_NHW) /
                                        (static_cast<FpAccum>(MIO_BN_NHW) - FpAccum{1.0})));

    resultRunningVariance[channel] =
        static_cast<FpPrec_C>((FpAccum{1.0} - static_cast<FpAccum>(expAvgFactor)) *
                                  static_cast<FpAccum_C>(resultRunningVariance[channel]) +
                              static_cast<FpAccum>(expAvgFactor) * adjust);
}

// _C suffix means used for computation
// _LS suffix means used for loading / storing
template <typename FpAccum_C, typename FpPrec_C>
__forceinline__ __device__ void saved_stash(FpPrec_C* __restrict resultSaveMean,
                                            FpPrec_C* __restrict resultSaveInvVariance,
                                            FpAccum_C mean,
                                            FpAccum_C invVariance,
                                            unsigned int channel)
{
    resultSaveMean[channel]        = static_cast<FpPrec_C>(mean);
    resultSaveInvVariance[channel] = static_cast<FpPrec_C>(invVariance);
}

#endif // BATCHNORM_FUNCTIONS_HPP
