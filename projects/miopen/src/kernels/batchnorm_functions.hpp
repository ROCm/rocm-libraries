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

#include "configuration.hpp"
#include "vector_types.hpp"
#include "miopen_math.hpp"

// TODO: refactor after the initial phase of porting OpenCL kernels to HIP
#if(MIO_BN_STASH_METHOD == 0)
// store values in HW dimension
#define NSTRIDE ystride
#else
// store values in N dimension
#define NSTRIDE                                                             \
    (miopen::batchnorm::config::c / miopen::batchnorm::config::vec_size_x * \
     miopen::batchnorm::config::hw)
#endif

namespace miopen {

namespace batchnorm {

template <typename OutType, typename InType>
__forceinline__ __device__ __host__ OutType cast(InType in)
{
    if constexpr(std::is_same<OutType, InType>::value)
    {
        return in;
    }
    else if constexpr(std::is_same<OutType, ushort>::value && std::is_same<InType, float>::value)
    {
        return float_to_bfloat16(in);
    }
    else if constexpr(std::is_same<InType, ushort>::value && std::is_same<OutType, float>::value)
    {
        return bfloat16_to_float(in);
    }
    else if constexpr(std::is_same<InType, typename mapped_vector_type<ushort, 4>::type>::value &&
                      std::is_same<OutType, typename mapped_vector_type<float, 4>::type>::value)
    {
        return typename mapped_vector_type<float, 4>::type{
            bfloat16_to_float(in.x),
            bfloat16_to_float(in.y),
            bfloat16_to_float(in.z),
            bfloat16_to_float(in.w),
        };
    }
    else if constexpr(std::is_same<InType, typename mapped_vector_type<float, 4>::type>::value &&
                      std::is_same<OutType, typename mapped_vector_type<ushort, 4>::type>::value)
    {
        return typename mapped_vector_type<ushort, 4>::type{
            float_to_bfloat16(in.x),
            float_to_bfloat16(in.y),
            float_to_bfloat16(in.z),
            float_to_bfloat16(in.w),
        };
    }
    else if constexpr(std::is_same<InType, typename mapped_vector_type<float, 4>::type>::value &&
                      std::is_same<OutType, typename mapped_vector_type<_Float16, 4>::type>::value)
    {
        return typename mapped_vector_type<_Float16, 4>::type{
            cast<_Float16>(in.x), cast<_Float16>(in.y), cast<_Float16>(in.z), cast<_Float16>(in.w)};
    }
    else if constexpr(std::is_same<InType, typename mapped_vector_type<_Float16, 4>::type>::value &&
                      std::is_same<OutType, typename mapped_vector_type<float, 4>::type>::value)
    {
        return typename mapped_vector_type<float, 4>::type{
            cast<float>(in.x), cast<float>(in.y), cast<float>(in.z), cast<float>(in.w)};
    }
    else
    {
        return static_cast<OutType>(in);
    }
}

// Just temporary, only to get the tests to pass; TODO: think of how to make it more robust
template <>
__forceinline__ __device__ __host__ HIP_vector_type<float, 4>
cast(typename miopen::mapped_vector_type<_Float16, 4>::type in)
{
    return {cast<float>(in.x), cast<float>(in.y), cast<float>(in.z), cast<float>(in.w)};
}

template <>
__forceinline__ __device__ __host__ typename miopen::mapped_vector_type<_Float16, 4>::type
cast(HIP_vector_type<float, 4> in)
{
    return {cast<_Float16>(in.x), cast<_Float16>(in.y), cast<_Float16>(in.z), cast<_Float16>(in.w)};
}

template <typename FpType, typename FpPrecType>
__forceinline__ __device__ __host__ auto
fpprec4_to_fp4(typename mapped_vector_type<FpPrecType, 4>::type const& val)
{
    return typename mapped_vector_type<FpType, 4>::type(
        cast<FpType>(val.x), cast<FpType>(val.y), cast<FpType>(val.z), cast<FpType>(val.w));
}

template <typename FpPrecType, typename FpType>
__forceinline__ __device__ __host__ auto
fp4_to_fpprec4(typename mapped_vector_type<FpType, 4>::type const& val)
{
    return typename mapped_vector_type<FpPrecType, 4>::type(cast<FpPrecType>(val.x),
                                                            cast<FpPrecType>(val.y),
                                                            cast<FpPrecType>(val.z),
                                                            cast<FpPrecType>(val.w));
}

template <typename FpPrecType, typename FpType, size_t VecSize>
__forceinline__ __device__ __host__ auto
fp_to_fpprec_vec(typename mapped_vector_type<FpType, VecSize>::type const& val)
{
    if constexpr(miopen::batchnorm::config::vectorize)
    {
        return fp4_to_fpprec4<FpPrecType, FpType>(val);
    }
    else
    {
        return cast<FpPrecType, FpType>(val);
    }
}

template <typename FpType, typename FpPrecType, size_t VecSize>
__forceinline__ __device__ __host__ auto
fpprec_to_fp_vec(typename mapped_vector_type<FpPrecType, VecSize>::type const& val)
{
    if constexpr(miopen::batchnorm::config::vectorize)
    {
        return fpprec4_to_fp4<FpType, FpPrecType>(val);
    }
    else
    {
        return cast<FpType, FpPrecType>(val);
    }
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void _accumulate1(T1& a, T2 const& b)
{
    a += cast<T1>(b);
}

template <typename T>
__forceinline__ __device__ __host__ void _accumulate_mad1(T& a, T const& b, T const& c, T const& d)
{
    a = miopen::fma(b, c, d);
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void _accumulate2(T1& a, T2 const& b)
{
    a += cast<T1>(b.x);
    a += cast<T1>(b.y);
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void _accumulate4(T1& a, T2 const& b)
{
    a += cast<T1>(b.x);
    a += cast<T1>(b.y);
    a += cast<T1>(b.z);
    a += cast<T1>(b.w);
}

template <typename T1, typename T2, typename T3, typename T4>
__forceinline__ __device__ __host__ void
_accumulate_mad4(T1& a, T2 const& b, T3 const& c, T4 const& d)
{
    a = miopen::fma(cast<T1>(b.x), cast<T1>(c.x), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.y), cast<T1>(c.y), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.z), cast<T1>(c.z), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.w), cast<T1>(c.w), cast<T1>(d));
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void _accumulate8(T1& a, T2 const& b)
{
    a += cast<T1>(b.s0);
    a += cast<T1>(b.s1);
    a += cast<T1>(b.s2);
    a += cast<T1>(b.s3);
    a += cast<T1>(b.s4);
    a += cast<T1>(b.s5);
    a += cast<T1>(b.s6);
    a += cast<T1>(b.s7);
}
template <typename T1, typename T2, typename T3, typename T4>
__forceinline__ __device__ __host__ void
_accumulate_mad8(T1& a, T2 const& b, T3 const& c, T4 const& d)
{
    a = miopen::fma(cast<T1>(b.s0), cast<T1>(c.s0), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.s1), cast<T1>(c.s1), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.s2), cast<T1>(c.s2), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.s3), cast<T1>(c.s3), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.s4), cast<T1>(c.s4), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.s5), cast<T1>(c.s5), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.s6), cast<T1>(c.s6), cast<T1>(d));
    a = miopen::fma(cast<T1>(b.s7), cast<T1>(c.s7), cast<T1>(d));
}

template <typename T>
__forceinline__ __device__ __host__ void
_accumulate_mad(T& a,
                typename mapped_vector_type<T, 8>::type const& b,
                typename mapped_vector_type<T, 8>::type const& c,
                T const& d)
{
    // static_assert(miopen::batchnorm::config::vectorize && (!miopen::config::layout_nhwc),
    //               "_accumulate_mad for this particular arg list is disabled.");
    _accumulate_mad8(a, b, c, d);
}

template <typename T>
__forceinline__ __device__ __host__ void
_accumulate_mad(T& a,
                typename mapped_vector_type<T, 4>::type const& b,
                typename mapped_vector_type<T, 4>::type const& c,
                T const& d)
{
    // static_assert(miopen::batchnorm::config::vectorize && (!miopen::config::layout_nhwc),
    //               "_accumulate_mad for this particular arg list is disabled.");
    _accumulate_mad4(a, b, c, d);
}

template <typename T>
__forceinline__ __device__ __host__ void _accumulate_mad(T& a, T const& b, T const& c, T const& d)
{
    // static_assert(!miopen::batchnorm::config::vectorize && (!miopen::config::layout_nhwc),
    //               "_accumulate_mad for this particular arg list is disabled.");
    _accumulate_mad1(a, b, c, d);
}

template <typename T>
__forceinline__ __device__ __host__ void _accumulate(T& a, T const& b)
{
    _accumulate1(a, b);
}

template <typename T>
__forceinline__ __device__ __host__ void
_accumulate(T& a, typename mapped_vector_type<T, 2>::type const& b)
{
    _accumulate2(a, b);
}

template <typename T>
__forceinline__ __device__ __host__ void
_accumulate(T& a, typename mapped_vector_type<T, 4>::type const& b)
{
    _accumulate4(a, b);
}

template <typename T>
__forceinline__ __device__ __host__ void
_accumulate(T& a, typename mapped_vector_type<T, 8>::type const& b)
{
    _accumulate8(a, b);
}

__forceinline__ __device__ unsigned int getStashIndex(unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrp_sz,
                                                      unsigned int xgrp_id,
                                                      unsigned int xlid,
                                                      unsigned int xstride)
{

    if constexpr(MIOPEN_USE_FPMIX || MIOPEN_USE_BFPMIX)
    {
        // 2 _FLOAT values are used to store 1 _FLOAT_PREC value.
        if constexpr(miopen::config::layout_nhwc)
        {
            if constexpr(miopen::batchnorm::config::c % 2 == 0)
            {
                // xgrp_sz values are split in two parts: even threads use 2 values at even rows,
                // odd threads - at odd rows. The only restriction for C and xgrp_sz is that they
                // must be even.
                return zgroupoffset *
                           (miopen::batchnorm::config::c / miopen::batchnorm::config::vec_size_x *
                            miopen::batchnorm::config::hw) +
                       (vindex * 2 + xlid % 2) * NSTRIDE + ygroupoffset * ystride +
                       (xgrp_sz * xgrp_id + xlid / 2 * 2) * xstride;
            }
            else
            {
                // Values are stored consecutively in y dim.
                return zgroupoffset *
                           (miopen::batchnorm::config::c / miopen::batchnorm::config::vec_size_x *
                            miopen::batchnorm::config::hw) +
                       (vindex * 2) * NSTRIDE + ygroupoffset * ystride +
                       (xgrp_sz * xgrp_id + xlid) * xstride;
            }
        }
        else
        {
            // Values are stored consecutively in y dim, indices are aligned up by 2 (_FLOAT_PREC).
            return zgroupoffset *
                       (miopen::batchnorm::config::c / miopen::batchnorm::config::vec_size_x *
                        miopen::batchnorm::config::hw) +
                   ((vindex * 2) * NSTRIDE + ygroupoffset * ystride +
                    (xgrp_sz * xgrp_id + xlid) * xstride + 1) /
                       2 * 2;
        }
    }
    else
    {
        return zgroupoffset *
                   (miopen::batchnorm::config::c / miopen::batchnorm::config::vec_size_x *
                    miopen::batchnorm::config::hw) +
               vindex * NSTRIDE + ygroupoffset * ystride + (xgrp_sz * xgrp_id + xlid) * xstride;
    }
}

template <typename FpPrecType_C, typename FpType_C>
__forceinline__ __device__ FpPrecType_C loadFromStash(const FpType_C* stash,
                                                      unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrp_sz,
                                                      unsigned int xgrp_id,
                                                      unsigned int xlid,
                                                      unsigned int xstride)
{
    unsigned int index =
        getStashIndex(vindex, zgroupoffset, ygroupoffset, ystride, xgrp_sz, xgrp_id, xlid, xstride);

    if constexpr(miopen::batchnorm::config::stash_method == 0 ||
                 miopen::batchnorm::config::stash_method == 1)
    {
        return *((const FpPrecType_C*)(stash + index));
    }
    else
    {
        FpPrecType_C value;
        *((FpType_C*)(&value)) = *(stash + index);
        index += NSTRIDE;
        *((FpType_C*)(&value) + 1) = *(stash + index);

        return value;
    }
}

template <typename FpPrecType_C, typename FpType_C>
__forceinline__ __device__ void storeToStash(FpPrecType_C value,
                                             FpType_C* stash,
                                             unsigned int vindex,
                                             unsigned int zgroupoffset,
                                             unsigned int ygroupoffset,
                                             unsigned int ystride,
                                             unsigned int xgrp_sz,
                                             unsigned int xgrp_id,
                                             unsigned int xlid,
                                             unsigned int xstride)
{
    unsigned int index =
        getStashIndex(vindex, zgroupoffset, ygroupoffset, ystride, xgrp_sz, xgrp_id, xlid, xstride);

    if constexpr(miopen::batchnorm::config::stash_method == 0 ||
                 miopen::batchnorm::config::stash_method == 1)
    {
        *(reinterpret_cast<FpPrecType_C*>(stash + index)) = value;
    }
    else
    {
        *(stash + index) = *((FpType_C*)(&value));
        index += NSTRIDE;
        *(stash + index) = *((FpType_C*)(&value) + 1);
    }
}

template <typename FpAccumType, typename FpAccumType_C, typename FpPrecType_C>
__forceinline__ __device__ void
running_stash(const FpPrecType_C* __restrict prevResultRunningMean,
              const FpPrecType_C* __restrict prevResultRunningVariance,
              FpPrecType_C* __restrict nextResultRunningMean,
              FpPrecType_C* __restrict nextResultRunningVariance,
              double expAvgFactor,
              FpAccumType_C mean,
              FpAccumType_C variance,
              uint channel)
{
    static_assert(miopen::batchnorm::config::variant != 4,
                  "running_stash is only compiled when MIO_BN_VARIANT != 4.");

    const auto pvt_runMean = static_cast<FpAccumType_C>(prevResultRunningMean[channel]);

    const auto pvt_newRunMean =
        miopen::fma(static_cast<FpAccumType_C>(-expAvgFactor),
                    static_cast<FpAccumType_C>(pvt_runMean),
                    static_cast<FpAccumType_C>(pvt_runMean)); // tmp = oldRunMean

    nextResultRunningMean[channel] = static_cast<FpPrecType_C>(
        miopen::fma(static_cast<FpAccumType_C>(mean),
                    static_cast<FpAccumType_C>(expAvgFactor),
                    static_cast<FpAccumType_C>(pvt_newRunMean))); // newMean*factor + tmp

    const FpAccumType_C adjust = static_cast<FpAccumType_C>(
        (miopen::batchnorm::config::nhw == 1)
            ? variance
            : variance *
                  (static_cast<FpAccumType>(miopen::batchnorm::config::nhw) /
                   (static_cast<FpAccumType>(miopen::batchnorm::config::nhw) - FpAccumType{1.0})));

    nextResultRunningVariance[channel] = static_cast<FpPrecType_C>(
        (FpAccumType{1.0} - static_cast<FpAccumType>(expAvgFactor)) *
            static_cast<FpAccumType_C>(prevResultRunningVariance[channel]) +
        static_cast<FpAccumType>(expAvgFactor) * adjust);
}

template <typename FpAccumType_C, typename FpPrecType_C>
__forceinline__ __device__ void saved_stash(FpPrecType_C* __restrict resultSaveMean,
                                            FpPrecType_C* __restrict resultSaveInvVariance,
                                            FpAccumType_C mean,
                                            FpAccumType_C invVariance,
                                            unsigned int channel)
{
    resultSaveMean[channel]        = static_cast<FpPrecType_C>(mean);
    resultSaveInvVariance[channel] = static_cast<FpPrecType_C>(invVariance);
}

} // namespace batchnorm
} // namespace miopen

#endif // BATCHNORM_FUNCTIONS_HPP
