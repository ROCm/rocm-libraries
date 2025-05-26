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

namespace miopen {
namespace batchnorm {

template <typename FpPrecType, typename FpType>
__forceinline__ __device__ __host__ FpPrecType fp_to_fpprec(FpType x)
{
    if constexpr(std::is_same<FpPrecType, FpType>::value)
    {
        // type is the same, no need conversion
        return x;
    }
    else
    {
        // TODO also change this
        if constexpr(miopen::batchnorm::config::input_type_strategy ==
                     miopen::type_strategy::bfpmix)
        {
            static_assert(std::is_same<FpType, ushort>::value &&
                              std::is_same<FpPrecType, float>::value,
                          "when MIOPEN_USE_BFPMIX == 1, FpType must be ushort(bfloat16), and "
                          "FpPrecType must be float");
            return bfloat16_to_float(x);
        }
        else
        {
            return static_cast<FpPrecType>(x);
        }
    }
}

template <typename FpType, typename FpPrecType>
__forceinline__ __device__ __host__ FpType fpprec_to_fp(FpPrecType x)
{
    if constexpr(std::is_same<FpType, FpPrecType>::value)
    {
        // type is the same, no need conversion
        return x;
    }
    else
    {

        if constexpr(miopen::batchnorm::config::input_type_strategy ==
                     miopen::type_strategy::bfpmix)
        {
            static_assert(std::is_same<FpType, ushort>::value &&
                              std::is_same<FpPrecType, float>::value,
                          "when MIOPEN_USE_BFPMIX == 1, FpType must be ushort(bfloat16), and "
                          "FpPrecType must be float");
            return float_to_bfloat16(x);
        }
        else
        {
            return static_cast<FpType>(x);
        }
    }
}

template <typename FpAccumType, typename FpType>
__forceinline__ __device__ __host__ FpAccumType fp_to_fpaccum(FpType x)
{
    if constexpr(miopen::batchnorm::config::input_type_strategy == miopen::type_strategy::bfpmix)
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
    if constexpr(miopen::batchnorm::config::input_type_strategy == miopen::type_strategy::bfpmix)
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

template <typename FpType, typename FpPrecType>
__forceinline__ __device__ __host__ auto
fpprec4_to_fp4(typename mapped_vector_type<FpPrecType, 4>::type const& val)
{
    return typename mapped_vector_type<FpType, 4>::type(fpprec_to_fp<FpType>(val.x),
                                                        fpprec_to_fp<FpType>(val.y),
                                                        fpprec_to_fp<FpType>(val.z),
                                                        fpprec_to_fp<FpType>(val.w));
}

template <typename FpPrecType, typename FpType>
__forceinline__ __device__ __host__ auto
fp4_to_fpprec4(typename mapped_vector_type<FpType, 4>::type const& val)
{
    return typename mapped_vector_type<FpPrecType, 4>::type(fp_to_fpprec<FpPrecType>(val.x),
                                                            fp_to_fpprec<FpPrecType>(val.y),
                                                            fp_to_fpprec<FpPrecType>(val.z),
                                                            fp_to_fpprec<FpPrecType>(val.w));
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
        return fp_to_fpprec<FpPrecType, FpType>(val);
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
        return fpprec_to_fp<FpType, FpPrecType>(val);
    }
}

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
    if constexpr(miopen::batchnorm::config::vectorize && !miopen::config::layout_nhwc)
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
    static_assert(miopen::batchnorm::config::vectorize && (!miopen::config::layout_nhwc),
                  "_accumulate_mad for this particular arg list is disabled.");
    _accumulate_mad4(a, b, c, d);
}

template <typename T>
__forceinline__ __device__ __host__ void _accumulate_mad(T& a, T const& b, T const& c, T const& d)
{
    static_assert(!miopen::batchnorm::config::vectorize && (!miopen::config::layout_nhwc),
                  "_accumulate_mad for this particular arg list is disabled.");
    _accumulate_mad1(a, b, c, d);
}

template <typename FpAccumType, typename FpAccumType_C, typename FpPrecType_C>
__forceinline__ __device__ void running_stash(FpPrecType_C* __restrict resultRunningMean,
                                              FpPrecType_C* __restrict resultRunningVariance,
                                              double expAvgFactor,
                                              FpAccumType_C mean,
                                              FpAccumType_C variance,
                                              uint channel)
{
    static_assert(miopen::batchnorm::config::variant != 4,
                  "running_stash is only compiled when MIO_BN_VARIANT != 4.");

    const auto pvt_runMean = static_cast<FpAccumType_C>(resultRunningMean[channel]);

    const auto pvt_newRunMean = fma(static_cast<FpAccumType_C>(-expAvgFactor),
                                    static_cast<FpAccumType_C>(pvt_runMean),
                                    static_cast<FpAccumType_C>(pvt_runMean)); // tmp = oldRunMean

    resultRunningMean[channel] = static_cast<FpPrecType_C>(
        fma(static_cast<FpAccumType_C>(mean),
            static_cast<FpAccumType_C>(expAvgFactor),
            static_cast<FpAccumType_C>(pvt_newRunMean))); // newMean*factor + tmp

    const FpAccumType_C adjust = static_cast<FpAccumType_C>(
        (miopen::batchnorm::config::nhw == 1)
            ? variance
            : variance *
                  (static_cast<FpAccumType>(miopen::batchnorm::config::nhw) /
                   (static_cast<FpAccumType>(miopen::batchnorm::config::nhw) - FpAccumType{1.0})));

    resultRunningVariance[channel] =
        static_cast<FpPrecType_C>((FpAccumType{1.0} - static_cast<FpAccumType>(expAvgFactor)) *
                                      static_cast<FpAccumType_C>(resultRunningVariance[channel]) +
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
