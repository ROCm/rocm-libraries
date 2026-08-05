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
// Product-private adapter from hipBLASLt descriptors to roc::host-validation.
#include <omp.h>

#include <bitset>
#include <iostream>
#include <roc/host_validation/adapters/hipblaslt/HipblasltReferenceGemm.hpp>
#include <roc/host_validation/backends/blas.hpp>
#include <roc/host_validation/validation.hpp>
#include <span>
#include <stdexcept>
#include <utility>

#include "datatype_interface.hpp"
#include "hipblaslt_vector.hpp"
#include "utility.hpp"

template <typename T>
class customVector
{
public:
    void initialize(size_t size)
    {
        m_data.resize(size);
        m_pointer = m_data.data();
    }
    void initialize(const void* buffer)
    {
        m_pointer = const_cast<void*>(buffer);
    }

    operator T*()
    {
        return (T*)m_pointer;
    }

    operator const T*() const
    {
        return (const T*)m_pointer;
    }

    T& operator[](std::size_t i)
    {
        return ((T*)m_pointer)[i];
    }

private:
    std::vector<T> m_data;
    void*          m_pointer = nullptr;
};

template <typename T>
struct is_hip_custom_type
    : std::integral_constant<bool,
                             std::is_same_v<T, hipblasLtHalf> || // HIP_R_16F
                                 std::is_same_v<T, hip_bfloat16> || // HIP_R_16BF
                                 std::is_same_v<T, hipblaslt_f8> || // HIP_R_8F_E4M3
                                 std::is_same_v<T, hipblaslt_bf8> || // HIP_R_8F_E5M2
                                 std::is_same_v<T, hipblaslt_f8_fnuz> || // HIP_R_8F_E4M3_FNUZ
                                 std::is_same_v<T, hipblaslt_bf8_fnuz> // HIP_R_8F_E5M2_FNUZ
                             >
{
};

template <typename T1, typename T2>
constexpr bool is_any_custom_type_v
    = is_hip_custom_type<T1>::value || is_hip_custom_type<T2>::value;


template <typename T>
constexpr auto get_real_if_complex(const T& val)
{
    if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
    {
        return val.real();
    }
    else
    {
        return val;
    }
}

template <typename T1, typename T2, std::enable_if_t<!is_any_custom_type_v<T1, T2>, int> = 0>
constexpr auto safe_multiply(const T1& a, const T2& b)
{
    using is_T1_complex = std::integral_constant<bool, is_std_complex_v<T1>>;
    using is_T2_complex = std::integral_constant<bool, is_std_complex_v<T2>>;

    if constexpr(std::is_integral_v<T1> && is_T2_complex::value)
    {
        using ComplexRealT = typename T2::value_type;
        return static_cast<ComplexRealT>(a) * b; 
    }
    else if constexpr(std::is_floating_point_v<T1> && is_T2_complex::value)
    {
        using ComplexRealT = typename T2::value_type;
        return static_cast<ComplexRealT>(a) * b;
    }
    else if constexpr(is_T1_complex::value && std::is_integral_v<T2>)
    {
        using ComplexRealT = typename T1::value_type;
        return a * static_cast<ComplexRealT>(b); 
    }
    else if constexpr(is_T1_complex::value && std::is_floating_point_v<T2>)
    {
        using ComplexRealT = typename T1::value_type;
        return a * static_cast<ComplexRealT>(b);
    }
    else if constexpr(is_T1_complex::value && is_T2_complex::value)
    {
        using T1_Val = typename T1::value_type;
        using T2_Val = typename T2::value_type;
        
        if constexpr(sizeof(T1_Val) >= sizeof(T2_Val))
        {
             return a * static_cast<T1>(b);
        }
        else
        {
             return static_cast<T2>(a) * b;
        }
    }
    else
    {
        return a * b;
    }
}

template <
    typename CustomReal,
    typename StandardT,
    std::enable_if_t<is_hip_custom_type<CustomReal>::value && !is_hip_custom_type<StandardT>::value,
                     int>
    = 0>
constexpr auto safe_multiply(const CustomReal& a, const StandardT& b)
{
    if constexpr(is_std_complex_v<StandardT>)
    {
        using RealT = typename StandardT::value_type;
        return static_cast<RealT>(a) * b; // Results in a complex type
    }
    else
    {
        return static_cast<float>(a) * b; // Results in a standard float type
    }
}

template <
    typename StandardT,
    typename CustomReal,
    std::enable_if_t<!is_hip_custom_type<StandardT>::value && is_hip_custom_type<CustomReal>::value,
                     int>
    = 0>
constexpr auto safe_multiply(const StandardT& a, const CustomReal& b)
{
    if constexpr(is_std_complex_v<StandardT>)
    {
        using RealT = typename StandardT::value_type;
        return a * static_cast<RealT>(b); // Results in a complex type
    }
    else
    {
        return a * static_cast<float>(b); // Results in a standard float type
    }
}

template <typename CustomReal1,
          typename CustomReal2,
          std::enable_if_t<is_hip_custom_type<CustomReal1>::value
                               && is_hip_custom_type<CustomReal2>::value,
                           int>
          = 0>
constexpr auto safe_multiply(const CustomReal1& a, const CustomReal2& b)
{
    return static_cast<float>(a) * static_cast<float>(b);
}

template <typename TD, typename TcCast, typename Tc>
void sat_cast_mul(TD* dst, customVector<TcCast>& src, Tc scale, size_t size)
{
    constexpr bool is_TD_complex
        = std::is_same_v<TD, std::complex<float>> || std::is_same_v<TD, std::complex<double>>;
    constexpr bool is_src_complex = is_std_complex_v<TcCast>; // TcCast is the type of src/result

    constexpr bool is_TD_real_or_int = !is_TD_complex;

    if constexpr(std::is_same<TcCast, float>::value
                 || (!std::is_same<TD, hipblaslt_bf8_fnuz>::value
                     && !std::is_same<TD, hipblaslt_f8_fnuz>::value))
    {
        if(scale != static_cast<Tc>(1))
        {
            for(size_t i = 0; i < size; i++)
            {
                auto result
                    = src[i] * scale;

                if constexpr(is_TD_real_or_int && is_src_complex)
                {
                    dst[i] = saturate_cast<TD>(get_real_if_complex(result));
                }
                else
                {
                    dst[i] = saturate_cast<TD>(result);
                }
            }
        }
        else
        {
            for(size_t i = 0; i < size; i++)
            {
                if constexpr(is_TD_real_or_int && is_src_complex)
                {
                    dst[i] = saturate_cast<TD>(get_real_if_complex(src[i]));
                }
                else
                {
                    dst[i] = saturate_cast<TD>(src[i]);
                }
            }
        }
    }
}

template <typename TcCast, typename Tc>
void sat_cast_mul(void* dst, hipDataType typeD, customVector<TcCast>& src, Tc scale, size_t size)
{
    switch(typeD)
    {
    case HIP_R_32F:
        sat_cast_mul<float, TcCast, Tc>(static_cast<float*>(dst), src, scale, size);
        break;
    case HIP_R_64F:
        sat_cast_mul<double, TcCast, Tc>(static_cast<double*>(dst), src, scale, size);
        break;
    case HIP_C_32F:
        sat_cast_mul<std::complex<float>, TcCast, Tc>(
            static_cast<std::complex<float>*>(dst), src, scale, size);
        break;
    case HIP_C_64F:
        sat_cast_mul<std::complex<double>, TcCast, Tc>(
            static_cast<std::complex<double>*>(dst), src, scale, size);
        break;
    case HIP_R_16F:
        sat_cast_mul<hipblasLtHalf, TcCast, Tc>(static_cast<hipblasLtHalf*>(dst), src, scale, size);
        break;
    case HIP_R_16BF:
        sat_cast_mul<hip_bfloat16, TcCast, Tc>(static_cast<hip_bfloat16*>(dst), src, scale, size);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        sat_cast_mul<hipblaslt_f8_fnuz, TcCast, Tc>(
            static_cast<hipblaslt_f8_fnuz*>(dst), src, scale, size);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        sat_cast_mul<hipblaslt_bf8_fnuz, TcCast, Tc>(
            static_cast<hipblaslt_bf8_fnuz*>(dst), src, scale, size);
        break;
    case HIP_R_8F_E4M3:
        sat_cast_mul<hipblaslt_f8, TcCast, Tc>(static_cast<hipblaslt_f8*>(dst), src, scale, size);
        break;
    case HIP_R_8F_E5M2:
        sat_cast_mul<hipblaslt_bf8, TcCast, Tc>(static_cast<hipblaslt_bf8*>(dst), src, scale, size);
        break;
    case HIP_R_32I:
        sat_cast_mul<int32_t, TcCast, Tc>(static_cast<int32_t*>(dst), src, scale, size);
        break;
    case HIP_R_8I:
        sat_cast_mul<hipblasLtInt8, TcCast, Tc>(static_cast<hipblasLtInt8*>(dst), src, scale, size);
        break;
    default:
        hipblaslt_cerr << "Error type in sat_cast_mul" << std::endl;
        break;
    }
}

template <typename TcCast, typename TiA>
void cast_mul(customVector<TcCast>& dst, const TiA* src, size_t size)
{
    // Logic: Only extract real part if Destination is Real AND Source is Complex
    constexpr bool requires_real_extraction = !is_std_complex_v<TcCast> && is_std_complex_v<TiA>;

    if constexpr(std::is_same<TcCast, float>::value
                 || (!std::is_same<TiA, hipblaslt_bf8_fnuz>::value
                     && !std::is_same<TiA, hipblaslt_f8_fnuz>::value))
    {
        if constexpr(std::is_same<TcCast, float>::value
                     || !(std::is_same<TiA, hipblaslt_bf8>::value
                          || std::is_same<TiA, hipblaslt_f8>::value))
            for(size_t i = 0; i < size; i++)
            {
                if constexpr(requires_real_extraction)
                {
                    dst[i] = static_cast<TcCast>(get_real_if_complex(src[i]));
                }
                else
                {
                    dst[i] = static_cast<TcCast>(src[i]);
                }
            }
    }
}

template <typename TcCast>
void cast_mul(customVector<TcCast>& dst, const void* src, hipDataType TiA, size_t size)
{
    switch(TiA)
    {
    case HIP_R_32F:
        cast_mul<TcCast, float>(dst, static_cast<const float*>(src), size);
        break;
    case HIP_R_64F:
        cast_mul<TcCast, double>(dst, static_cast<const double*>(src), size);
        break;
    case HIP_C_32F:
        cast_mul<TcCast, std::complex<float>>(
            dst, static_cast<const std::complex<float>*>(src), size);
        break;
    case HIP_C_64F:
        cast_mul<TcCast, std::complex<double>>(
            dst, static_cast<const std::complex<double>*>(src), size);
        break;
    case HIP_R_16F:
        cast_mul<TcCast, hipblasLtHalf>(dst, static_cast<const hipblasLtHalf*>(src), size);
        break;
    case HIP_R_16BF:
        cast_mul<TcCast, hip_bfloat16>(dst, static_cast<const hip_bfloat16*>(src), size);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        cast_mul<TcCast, hipblaslt_f8_fnuz>(dst, static_cast<const hipblaslt_f8_fnuz*>(src), size);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        cast_mul<TcCast, hipblaslt_bf8_fnuz>(
            dst, static_cast<const hipblaslt_bf8_fnuz*>(src), size);
        break;
    case HIP_R_8F_E4M3:
        cast_mul<TcCast, hipblaslt_f8>(dst, static_cast<const hipblaslt_f8*>(src), size);
        break;
    case HIP_R_8F_E5M2:
        cast_mul<TcCast, hipblaslt_bf8>(dst, static_cast<const hipblaslt_bf8*>(src), size);
        break;
    case HIP_R_32I:
        cast_mul<TcCast, int32_t>(dst, static_cast<const int32_t*>(src), size);
        break;
    case HIP_R_8I:
        cast_mul<TcCast, hipblasLtInt8>(dst, static_cast<const hipblasLtInt8*>(src), size);
        break;
    default:
        hipblaslt_cerr << "Error type in cast_mul" << std::endl;
        break;
    }
}

template <typename TcCast, typename Tc, typename TiA>
void cast_mul(customVector<TcCast>& dst,
              const TiA* A,
              bool                  isScaleAVec,
              const TcCast* scaleAVec,
              const TcCast* AlphaVec,
              bool                  transA,
              int64_t               m,
              int64_t               k,
              size_t                size,
              bool                  isMXFormat = false)
{
    constexpr bool requires_real_extraction = !is_std_complex_v<TcCast> && is_std_complex_v<TiA>;
    constexpr bool destination_is_real      = !is_std_complex_v<TcCast>;

    // If we are casting Complex -> Real, extract real part.
    // Otherwise return val as-is.
    auto get_cast_val = [&](auto val) {
        if constexpr(requires_real_extraction)                              // complex→real
            return get_real_if_complex(val);
        else if constexpr(is_std_complex_v<TcCast>
                      && !is_std_complex_v<std::decay_t<decltype(val)>>)    // real→complex
           return static_cast<scalar_of_t<TcCast>>(val);                    // always valid: complex<float>→float, complex<double>→double
        else                                                                // same type
            return val;
};

    if constexpr((std::is_same<TcCast, float>::value)
                 || (!std::is_same<TiA, hipblaslt_bf8_fnuz>::value
                     && !std::is_same<TiA, hipblaslt_f8_fnuz>::value))
    {
        if constexpr(std::is_same<TcCast, float>::value
                     || !(std::is_same<TiA, hipblaslt_bf8>::value
                          || std::is_same<TiA, hipblaslt_f8>::value))
        {
            if constexpr(false
#if defined(HIPBLASLT_USE_FP4)
                         || std::is_same<TiA, hipblaslt_f4x2>::value
#endif
#if defined(HIPBLASLT_USE_FP6) && defined(HIPBLASLT_USE_BF6)
                         || std::is_same<TiA, hipblaslt_f6x16>::value
                         || std::is_same<TiA, hipblaslt_bf6x16>::value
#endif
                         )
                size = size / TiA::packed_size;
            if(AlphaVec != nullptr)
            {
                if(transA)
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto A_val = A[i];

                        if(isMXFormat)
                        {
                            auto result = safe_multiply(static_cast<TcCast>(get_cast_val(A_val)), AlphaVec[i % m]);
                            if constexpr(destination_is_real)
                            {
                                dst[i] = static_cast<TcCast>(get_real_if_complex(result));
                            }
                            else
                            {
                                dst[i] = static_cast<TcCast>(result);
                            }
                        }
                        else
                        {
                            auto scaleA = isScaleAVec ? scaleAVec[i % m] : scaleAVec[0];
                            if constexpr(false
#if defined(HIPBLASLT_USE_FP4)
                                         || std::is_same<TiA, hipblaslt_f4x2>::value
#endif
#if defined(HIPBLASLT_USE_FP6) && defined(HIPBLASLT_USE_BF6)
                                         || std::is_same<TiA, hipblaslt_f6x16>::value
                                         || std::is_same<TiA, hipblaslt_bf6x16>::value
#endif
                                         )
                            {
                                using type = TiA;
                                for(int j = 0; j < type::packed_size; j++)
                                {
                                    dst[type::packed_size * i + j]
                                        = static_cast<TcCast>(A[i].castElement(j)) * scaleA
                                          * AlphaVec[i % m];
                                }
                            }
                            else
                            {
                                auto scaled_A = safe_multiply(static_cast<TcCast>(get_cast_val(A_val)), scaleA);
                                auto result   = safe_multiply(scaled_A, AlphaVec[i % m]);

                                if constexpr(destination_is_real)
                                {
                                    dst[i] = static_cast<TcCast>(get_real_if_complex(result));
                                }
                                else
                                {
                                    dst[i] = static_cast<TcCast>(result);
                                }
                            }
                        }
                    }
                } // transA
                else
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto A_val = A[i];

                        if(isMXFormat)
                        {
                            auto result = safe_multiply(static_cast<TcCast>(get_cast_val(A_val)), AlphaVec[i / k]);
                            if constexpr(destination_is_real)
                            {
                                dst[i] = static_cast<TcCast>(get_real_if_complex(result));
                            }
                            else
                            {
                                dst[i] = static_cast<TcCast>(result);
                            }
                        }
                        else
                        {
                            auto scaleA = isScaleAVec ? scaleAVec[i / k] : scaleAVec[0];
                            if constexpr(false
#if defined(HIPBLASLT_USE_FP4)
                                         || std::is_same<TiA, hipblaslt_f4x2>::value
#endif
#if defined(HIPBLASLT_USE_FP6) && defined(HIPBLASLT_USE_BF6)
                                         || std::is_same<TiA, hipblaslt_f6x16>::value
                                         || std::is_same<TiA, hipblaslt_bf6x16>::value
#endif
                                         )
                            {
                                using type = TiA;
                                for(int j = 0; j < type::packed_size; j++)
                                {
                                    dst[type::packed_size * i + j]
                                        = static_cast<TcCast>(A[i].castElement(j)) * scaleA
                                          * AlphaVec[i / k];
                                }
                            }
                            else
                            {
                                auto scaled_A = safe_multiply(static_cast<TcCast>(get_cast_val(A_val)), scaleA);
                                auto result   = safe_multiply(scaled_A, AlphaVec[i / k]);

                                if constexpr(destination_is_real)
                                {
                                    dst[i] = static_cast<TcCast>(get_real_if_complex(result));
                                }
                                else
                                {
                                    dst[i] = static_cast<TcCast>(result);
                                }
                            }
                        }
                    }
                }
            } // AlphaVec != nullptr
            else
            {
                if(transA)
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto A_val = A[i];

                        if(isMXFormat)
                        {
                            dst[i] = static_cast<TcCast>(get_cast_val(A_val));
                        }
                        else
                        {
                            auto scaleA = isScaleAVec ? scaleAVec[i % m] : scaleAVec[0];
                            if constexpr(false
#if defined(HIPBLASLT_USE_FP4)
                                         || std::is_same<TiA, hipblaslt_f4x2>::value
#endif
#if defined(HIPBLASLT_USE_FP6) && defined(HIPBLASLT_USE_BF6)
                                         || std::is_same<TiA, hipblaslt_f6x16>::value
                                         || std::is_same<TiA, hipblaslt_bf6x16>::value
#endif
                                         )
                            {
                                using type = TiA;
                                for(int j = 0; j < type::packed_size; j++)
                                {
                                    dst[type::packed_size * i + j]
                                        = static_cast<TcCast>(static_cast<TcCast>(A[i].castElement(j)) * scaleA);
                                }
                            }
                            else
                            {
                                auto result = safe_multiply(static_cast<TcCast>(get_cast_val(A_val)), scaleA);

                                if constexpr(destination_is_real)
                                {
                                    dst[i] = static_cast<TcCast>(get_real_if_complex(result));
                                }
                                else
                                {
                                    dst[i] = static_cast<TcCast>(result);
                                }
                            }
                        }
                    }
                }
                else // not transA
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto A_val = A[i];

                        if(isMXFormat)
                        {
                            dst[i] = static_cast<TcCast>(get_cast_val(A_val));
                        }
                        else
                        {
                            auto scaleA = isScaleAVec ? scaleAVec[i / k] : scaleAVec[0];
                            if constexpr(false
#if defined(HIPBLASLT_USE_FP4)
                                         || std::is_same<TiA, hipblaslt_f4x2>::value
#endif
#if defined(HIPBLASLT_USE_FP6) && defined(HIPBLASLT_USE_BF6)
                                         || std::is_same<TiA, hipblaslt_f6x16>::value
                                         || std::is_same<TiA, hipblaslt_bf6x16>::value
#endif
                                         )
                            {
                                using type = TiA;
                                for(int j = 0; j < type::packed_size; j++)
                                {
                                    dst[type::packed_size * i + j]
                                        = static_cast<TcCast>(static_cast<TcCast>(A[i].castElement(j)) * scaleA);
                                }
                            }
                            else
                            {
                                auto result = safe_multiply(static_cast<TcCast>(get_cast_val(A_val)), scaleA);

                                if constexpr(destination_is_real)
                                {
                                    dst[i] = static_cast<TcCast>(get_real_if_complex(result));
                                }
                                else
                                {
                                    dst[i] = static_cast<TcCast>(result);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename TcCast, typename Tc>
void cast_mul(customVector<TcCast>& dst,
              const void*           src,
              hipDataType           TiA,
              bool                  isScaleAVec,
              const TcCast*         scaleAVec,
              const TcCast*         AlphaVec,
              bool                  transA,
              int64_t               m,
              int64_t               k,
              size_t                size,
              bool                  isMXFormat = false)
{
    switch(TiA)
    {
    case HIP_R_32F:
        cast_mul<TcCast, Tc, float>(dst,
                                    static_cast<const float*>(src),
                                    isScaleAVec,
                                    scaleAVec,
                                    AlphaVec,
                                    transA,
                                    m,
                                    k,
                                    size,
                                    isMXFormat);
        break;
    case HIP_R_64F:
        cast_mul<TcCast, Tc, double>(dst,
                                     static_cast<const double*>(src),
                                     isScaleAVec,
                                     scaleAVec,
                                     AlphaVec,
                                     transA,
                                     m,
                                     k,
                                     size,
                                     isMXFormat);
        break;
    case HIP_C_32F:
        cast_mul<TcCast, Tc, std::complex<float>>(dst,
                                                  static_cast<const std::complex<float>*>(src),
                                                  isScaleAVec,
                                                  scaleAVec,
                                                  AlphaVec,
                                                  transA,
                                                  m,
                                                  k,
                                                  size,
                                                  isMXFormat);
        break;
    case HIP_C_64F:
        cast_mul<TcCast, Tc, std::complex<double>>(dst,
                                                   static_cast<const std::complex<double>*>(src),
                                                   isScaleAVec,
                                                   scaleAVec,
                                                   AlphaVec,
                                                   transA,
                                                   m,
                                                   k,
                                                   size,
                                                   isMXFormat);
        break;
    case HIP_R_16F:
        cast_mul<TcCast, Tc, hipblasLtHalf>(dst,
                                            static_cast<const hipblasLtHalf*>(src),
                                            isScaleAVec,
                                            scaleAVec,
                                            AlphaVec,
                                            transA,
                                            m,
                                            k,
                                            size,
                                            isMXFormat);
        break;
    case HIP_R_16BF:
        cast_mul<TcCast, Tc, hip_bfloat16>(dst,
                                           static_cast<const hip_bfloat16*>(src),
                                           isScaleAVec,
                                           scaleAVec,
                                           AlphaVec,
                                           transA,
                                           m,
                                           k,
                                           size,
                                           isMXFormat);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        cast_mul<TcCast, Tc, hipblaslt_f8_fnuz>(dst,
                                                static_cast<const hipblaslt_f8_fnuz*>(src),
                                                isScaleAVec,
                                                scaleAVec,
                                                AlphaVec,
                                                transA,
                                                m,
                                                k,
                                                size,
                                                isMXFormat);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        cast_mul<TcCast, Tc, hipblaslt_bf8_fnuz>(dst,
                                                 static_cast<const hipblaslt_bf8_fnuz*>(src),
                                                 isScaleAVec,
                                                 scaleAVec,
                                                 AlphaVec,
                                                 transA,
                                                 m,
                                                 k,
                                                 size,
                                                 isMXFormat);
        break;
    case HIP_R_8F_E4M3:
        cast_mul<TcCast, Tc, hipblaslt_f8>(dst,
                                           static_cast<const hipblaslt_f8*>(src),
                                           isScaleAVec,
                                           scaleAVec,
                                           AlphaVec,
                                           transA,
                                           m,
                                           k,
                                           size,
                                           isMXFormat);
        break;
    case HIP_R_8F_E5M2:
        cast_mul<TcCast, Tc, hipblaslt_bf8>(dst,
                                            static_cast<const hipblaslt_bf8*>(src),
                                            isScaleAVec,
                                            scaleAVec,
                                            AlphaVec,
                                            transA,
                                            m,
                                            k,
                                            size,
                                            isMXFormat);
        break;
    case HIP_R_32I:
        cast_mul<TcCast, Tc, int32_t>(dst,
                                      static_cast<const int32_t*>(src),
                                      isScaleAVec,
                                      scaleAVec,
                                      AlphaVec,
                                      transA,
                                      m,
                                      k,
                                      size);
        break;
    case HIP_R_8I:
        cast_mul<TcCast, Tc, hipblasLtInt8>(dst,
                                            static_cast<const hipblasLtInt8*>(src),
                                            isScaleAVec,
                                            scaleAVec,
                                            AlphaVec,
                                            transA,
                                            m,
                                            k,
                                            size);
        break;
#if defined(HIPBLASLT_USE_FP4)
    case static_cast<hipDataType>(HIP_R_4F_E2M1):
        cast_mul<TcCast, Tc, hipblaslt_f4x2>(dst,
                                             static_cast<const hipblaslt_f4x2*>(src),
                                             isScaleAVec,
                                             scaleAVec,
                                             AlphaVec,
                                             transA,
                                             m,
                                             k,
                                             size);
        break;
#endif
#if defined(HIPBLASLT_USE_FP6)
    case static_cast<hipDataType>(HIP_R_6F_E2M3):
        cast_mul<TcCast, Tc, hipblaslt_f6x16>(dst,
                                              static_cast<const hipblaslt_f6x16*>(src),
                                              isScaleAVec,
                                              scaleAVec,
                                              AlphaVec,
                                              transA,
                                              m,
                                              k,
                                              size);
        break;
#endif
#if defined(HIPBLASLT_USE_BF6)
    case static_cast<hipDataType>(HIP_R_6F_E3M2):
        cast_mul<TcCast, Tc, hipblaslt_bf6x16>(dst,
                                               static_cast<const hipblaslt_bf6x16*>(src),
                                               isScaleAVec,
                                               scaleAVec,
                                               AlphaVec,
                                               transA,
                                               m,
                                               k,
                                               size);
        break;
#endif
    default:
        hipblaslt_cerr << "Error type in cast_mul" << std::endl;
        break;
    }
}

template <typename TcCast, typename Tc, typename TciACast, typename TiA>
void cast_mul_with_Tci(customVector<TcCast>& dst,
                       const TiA* A,
                       bool                  isScaleAVec,
                       const TcCast* scaleAVec,
                       const TcCast* AlphaVec,
                       bool                  transA,
                       int64_t               m,
                       int64_t               k,
                       size_t                size)
{
    // NOTE: TciACast is the intermediate type. If the result of the math (A*Scale*Alpha) is Complex,
    // but TciACast is Real, we must extract the real part before casting to TciACast.
    constexpr bool intermediate_is_real = !is_std_complex_v<TciACast>;

    // Helper to extract real only if we are squashing complex to real
    auto get_intermediate_val = [&](auto val) {
        if constexpr(intermediate_is_real)
        {
            return get_real_if_complex(val);
        }
        else
        {
            return val;
        }
    };

    if constexpr(std::is_same<TcCast, float>::value
                 || (!std::is_same<TciACast, hipblaslt_bf8_fnuz>::value
                     && !std::is_same<TciACast, hipblaslt_f8_fnuz>::value)
                        && (!std::is_same<TiA, hipblaslt_bf8_fnuz>::value
                            && !std::is_same<TiA, hipblaslt_f8_fnuz>::value))
    {
        if constexpr(std::is_same<TcCast, float>::value
                     || (!std::is_same<TciACast, hipblaslt_bf8>::value
                         && !std::is_same<TciACast, hipblaslt_f8>::value)
                            && (!std::is_same<TiA, hipblaslt_bf8>::value
                                && !std::is_same<TiA, hipblaslt_f8>::value))
        {
            if(AlphaVec != nullptr)
            {
                if(transA)
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto scaleA_val = isScaleAVec ? scaleAVec[i % m] : scaleAVec[0];
                        auto scaled_val = safe_multiply(A[i], scaleA_val);
                        auto result     = safe_multiply(scaled_val, AlphaVec[i % m]);

                        auto intermediate_val = static_cast<TciACast>(get_intermediate_val(result));
                        
                        constexpr bool final_cast_requires_real = !is_std_complex_v<TcCast> && is_std_complex_v<TciACast>;
                        if constexpr(final_cast_requires_real)
                        {
                            dst[i] = static_cast<TcCast>(get_real_if_complex(intermediate_val));
                        }
                        else
                        {
                            dst[i] = static_cast<TcCast>(intermediate_val);
                        }
                    }
                }
                else // not transA
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto scaleA_val = isScaleAVec ? scaleAVec[i / k] : scaleAVec[0];
                        auto scaled_val = safe_multiply(A[i], scaleA_val);
                        auto result     = safe_multiply(scaled_val, AlphaVec[i / k]);

                        auto intermediate_val = static_cast<TciACast>(get_intermediate_val(result));
                        
                        constexpr bool final_cast_requires_real = !is_std_complex_v<TcCast> && is_std_complex_v<TciACast>;
                        if constexpr(final_cast_requires_real)
                        {
                            dst[i] = static_cast<TcCast>(get_real_if_complex(intermediate_val));
                        }
                        else
                        {
                            dst[i] = static_cast<TcCast>(intermediate_val);
                        }
                    }
                }
            }
            else // AlphaVec == nullptr
            {
                if(transA)
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto scaleA_val = isScaleAVec ? scaleAVec[i % m] : scaleAVec[0];
                        auto scaled_val = safe_multiply(A[i], scaleA_val);

                        auto intermediate_val = static_cast<TciACast>(get_intermediate_val(scaled_val));

                        constexpr bool final_cast_requires_real = !is_std_complex_v<TcCast> && is_std_complex_v<TciACast>;
                        if constexpr(final_cast_requires_real)
                        {
                            dst[i] = static_cast<TcCast>(get_real_if_complex(intermediate_val));
                        }
                        else
                        {
                            dst[i] = static_cast<TcCast>(intermediate_val);
                        }
                    }
                }
                else // not transA
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto scaleA_val = isScaleAVec ? scaleAVec[i / k] : scaleAVec[0];
                        auto scaled_val = safe_multiply(A[i], scaleA_val);

                        auto intermediate_val = static_cast<TciACast>(get_intermediate_val(scaled_val));
                        
                        constexpr bool final_cast_requires_real = !is_std_complex_v<TcCast> && is_std_complex_v<TciACast>;
                        if constexpr(final_cast_requires_real)
                        {
                            dst[i] = static_cast<TcCast>(get_real_if_complex(intermediate_val));
                        }
                        else
                        {
                            dst[i] = static_cast<TcCast>(intermediate_val);
                        }
                    }
                }
            }
        }
    }
}

template <typename TcCast, typename Tc, typename TciACast>
void cast_mul_with_Tci(customVector<TcCast>& dst,
                       const void*           src,
                       hipDataType           TiA,
                       bool                  isScaleAVec,
                       const TcCast*         scaleAVec,
                       const TcCast*         AlphaVec,
                       bool                  transA,
                       int64_t               m,
                       int64_t               k,
                       size_t                size)
{
    switch(TiA)
    {
    case HIP_R_32F:
        cast_mul_with_Tci<TcCast, Tc, TciACast, float>(dst,
                                                       static_cast<const float*>(src),
                                                       isScaleAVec,
                                                       scaleAVec,
                                                       AlphaVec,
                                                       transA,
                                                       m,
                                                       k,
                                                       size);
        break;
    case HIP_R_64F:
        cast_mul_with_Tci<TcCast, Tc, TciACast, double>(dst,
                                                        static_cast<const double*>(src),
                                                        isScaleAVec,
                                                        scaleAVec,
                                                        AlphaVec,
                                                        transA,
                                                        m,
                                                        k,
                                                        size);
        break;
    case HIP_C_32F:
        cast_mul_with_Tci<TcCast, Tc, TciACast, std::complex<float>>(dst,
                                                       static_cast<const std::complex<float>*>(src),
                                                       isScaleAVec,
                                                       scaleAVec,
                                                       AlphaVec,
                                                       transA,
                                                       m,
                                                       k,
                                                       size);
        break;
    case HIP_C_64F:
        cast_mul_with_Tci<TcCast, Tc, TciACast, std::complex<double>>(dst,
                                                        static_cast<const std::complex<double>*>(src),
                                                        isScaleAVec,
                                                        scaleAVec,
                                                        AlphaVec,
                                                        transA,
                                                        m,
                                                        k,
                                                        size);
        break;
    case HIP_R_16F:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblasLtHalf>(
            dst,
            static_cast<const hipblasLtHalf*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    case HIP_R_16BF:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hip_bfloat16>(dst,
                                                              static_cast<const hip_bfloat16*>(src),
                                                              isScaleAVec,
                                                              scaleAVec,
                                                              AlphaVec,
                                                              transA,
                                                              m,
                                                              k,
                                                              size);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblaslt_f8_fnuz>(
            dst,
            static_cast<const hipblaslt_f8_fnuz*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblaslt_bf8_fnuz>(
            dst,
            static_cast<const hipblaslt_bf8_fnuz*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    case HIP_R_8F_E4M3:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblaslt_f8>(dst,
                                                              static_cast<const hipblaslt_f8*>(src),
                                                              isScaleAVec,
                                                              scaleAVec,
                                                              AlphaVec,
                                                              transA,
                                                              m,
                                                              k,
                                                              size);
        break;
    case HIP_R_8F_E5M2:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblaslt_bf8>(
            dst,
            static_cast<const hipblaslt_bf8*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    case HIP_R_32I:
        cast_mul_with_Tci<TcCast, Tc, TciACast, int32_t>(dst,
                                                         static_cast<const int32_t*>(src),
                                                         isScaleAVec,
                                                         scaleAVec,
                                                         AlphaVec,
                                                         transA,
                                                         m,
                                                         k,
                                                         size);
        break;
    case HIP_R_8I:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblasLtInt8>(
            dst,
            static_cast<const hipblasLtInt8*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    default:
        hipblaslt_cerr << "Error type in cast_mul_with_Tci" << std::endl;
        break;
    }
}

template <typename TcCast, typename Tc>
void cast_mul_with_Tci(customVector<TcCast>& dst,
                       const void*           src,
                       hipDataType           TiA,
                       bool                  isScaleAVec,
                       const TcCast*         scaleAVec,
                       const TcCast*         AlphaVec,
                       bool                  transA,
                       int64_t               m,
                       int64_t               k,
                       hipDataType           TciACast,
                       size_t                size)
{
    switch(TciACast)
    {
    case HIP_R_32F:
        cast_mul_with_Tci<TcCast, Tc, float>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_64F:
        cast_mul_with_Tci<TcCast, Tc, double>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_C_32F:
        cast_mul_with_Tci<TcCast, Tc, std::complex<float>>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_C_64F:
        cast_mul_with_Tci<TcCast, Tc, std::complex<double>>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_16F:
        cast_mul_with_Tci<TcCast, Tc, hipblasLtHalf>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_16BF:
        cast_mul_with_Tci<TcCast, Tc, hip_bfloat16>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        cast_mul_with_Tci<TcCast, Tc, hipblaslt_f8_fnuz>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        cast_mul_with_Tci<TcCast, Tc, hipblaslt_bf8_fnuz>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8F_E4M3:
        cast_mul_with_Tci<TcCast, Tc, hipblaslt_f8>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8F_E5M2:
        cast_mul_with_Tci<TcCast, Tc, hipblaslt_bf8>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_32I:
        cast_mul_with_Tci<TcCast, Tc, int32_t>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8I:
        cast_mul_with_Tci<TcCast, Tc, hipblasLtInt8>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    default:
        hipblaslt_cerr << "Error type in cast_mul_with_Tci" << std::endl;
        break;
    }
}

template <typename Tc>
void hipblaslt_reference_gemm(hipblasOperation_t       transA,
                hipblasOperation_t       transB,
                int64_t                  m,
                int64_t                  n,
                int64_t                  k,
                Tc                       alpha,
                const void*              A,
                int64_t                  lda,
                const void*              B,
                int64_t                  ldb,
                Tc                       beta,
                std::add_pointer_t<void> C,
                int64_t                  ldc,
                const void*              AlphaVec,
                const void*              scaleAVec,
                const void*              scaleBVec,
                Tc                       scaleD,
                bool                     isScaleAVec,
                bool                     isScaleBVec,
                hipDataType              TiA,
                hipDataType              TiB,
                hipDataType              To,
                hipDataType              Tc_enum,
                hipDataType              TciA,
                hipDataType              TciB,
                bool                     alt,
                bool                     isScaleAMXFormat,
                bool                     isScaleBMXFormat)
{
    using IntTcCast = std::conditional_t<std::is_same<Tc, int32_t>::value, double, Tc>;
    // Promote half accumulation to float for the host reference.
    using HalfTcCast = std::conditional_t<std::is_same<Tc, hipblasLtHalf>::value, float, Tc>;
    using TcCast     = std::conditional_t<std::is_same<Tc, int32_t>::value, IntTcCast, HalfTcCast>;

    if(Tc_enum == HIP_R_32I)
    {
        Tc_enum = HIP_R_64F;
    }
    else if(Tc_enum == HIP_R_16F)
    {
        Tc_enum = HIP_R_32F;
    }

    hipDataType TciACast = (TciA == HIP_R_32I) ? HIP_R_64F : TciA;
    hipDataType TciBCast = (TciB == HIP_R_32I) ? HIP_R_64F : TciB;

    size_t sizeA          = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB          = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC          = n * size_t(ldc);
    size_t scaleAVec_size = isScaleAVec ? m : 1;
    size_t scaleBVec_size = isScaleBVec ? n : 1;

    customVector<TcCast> A_Tc, B_Tc, C_Tc, AlphaVec_Tc, scaleA_Tc, scaleB_Tc;

    if(AlphaVec)
    {
        AlphaVec_Tc.initialize(m);
        cast_mul(AlphaVec_Tc, (Tc*)AlphaVec, m);
    }

    if(scaleAVec)
    {
        scaleA_Tc.initialize(scaleAVec_size);
        cast_mul(scaleA_Tc, (Tc*)scaleAVec, scaleAVec_size);
    }
    else
    {
        scaleA_Tc.initialize(1);
        scaleA_Tc[0] = TcCast(1);
    }

    if(scaleBVec)
    {
        scaleB_Tc.initialize(scaleBVec_size);
        cast_mul(scaleB_Tc, (Tc*)scaleBVec, scaleBVec_size);
    }
    else
    {
        scaleB_Tc.initialize(1);
        scaleB_Tc[0] = TcCast(1);
    }

    A_Tc.initialize(sizeA);
    if(realDataTypeSize(TiA) > realDataTypeSize(TciACast))
    {
        cast_mul_with_Tci<TcCast, Tc>(A_Tc,
                                      A,
                                      TiA,
                                      isScaleAVec,
                                      scaleA_Tc,
                                      AlphaVec_Tc,
                                      transA == HIPBLAS_OP_N,
                                      m,
                                      k,
                                      TciACast,
                                      sizeA);
    }
    else
    {
        cast_mul<TcCast, Tc>(A_Tc,
                             A,
                             TiA,
                             isScaleAVec,
                             scaleA_Tc,
                             AlphaVec_Tc,
                             transA == HIPBLAS_OP_N,
                             m,
                             k,
                             sizeA,
                             isScaleAMXFormat);
    }

    B_Tc.initialize(sizeB);
    if(realDataTypeSize(TiB) > realDataTypeSize(TciBCast))
    {
        cast_mul_with_Tci<TcCast, Tc>(B_Tc,
                                      B,
                                      TiB,
                                      isScaleBVec,
                                      scaleB_Tc,
                                      nullptr,
                                      transB != HIPBLAS_OP_N,
                                      n,
                                      k,
                                      TciBCast,
                                      sizeB);
    }
    else
    {
        cast_mul<TcCast, Tc>(B_Tc,
                             B,
                             TiB,
                             isScaleBVec,
                             scaleB_Tc,
                             nullptr,
                             transB != HIPBLAS_OP_N,
                             n,
                             k,
                             sizeB,
                             isScaleBMXFormat);
    }

    if(To == Tc_enum)
    {
        C_Tc.initialize(C);
    }
    else
    {
        C_Tc.initialize(sizeC);
        cast_mul<TcCast>(C_Tc, C, To, sizeC);
    }

    TcCast alphaCast = (TcCast)alpha;
    TcCast betaCast  = (TcCast)beta;

    const ptrdiff_t aRowStride = transA == HIPBLAS_OP_N ? 1 : lda;
    const ptrdiff_t aColumnStride = transA == HIPBLAS_OP_N ? lda : 1;
    const ptrdiff_t bRowStride = transB == HIPBLAS_OP_N ? 1 : ldb;
    const ptrdiff_t bColumnStride = transB == HIPBLAS_OP_N ? ldb : 1;
    const TcCast* aValues = A_Tc;
    const TcCast* bValues = B_Tc;
    TcCast* cValues = C_Tc;

    static constexpr int64_t blasThreshold = 600;
    const bool useBlas = m > blasThreshold || n > blasThreshold || k > blasThreshold
                         || lda > blasThreshold || ldb > blasThreshold || ldc > blasThreshold;

    using namespace roc::host_validation;
    GemmOperand operandA(TensorView::fromNative<TcCast>(
        Layout(Shape{size_t(m), size_t(k)}, {aRowStride, aColumnStride}),
        std::span<const TcCast>(aValues, sizeA)));
    GemmOperand operandB(TensorView::fromNative<TcCast>(
        Layout(Shape{size_t(k), size_t(n)}, {bRowStride, bColumnStride}),
        std::span<const TcCast>(bValues, sizeB)));
    operandA.conjugate = transA == HIPBLAS_OP_C;
    operandB.conjugate = transB == HIPBLAS_OP_C;

    GemmProblem problem(
        std::move(operandA), std::move(operandB),
        TensorView::fromNative<TcCast>(Layout(Shape{size_t(m), size_t(n)}, {1, ldc}),
                                       std::span<const TcCast>(cValues, sizeC)),
        MutableTensorView::fromNative<TcCast>(Layout(Shape{size_t(m), size_t(n)}, {1, ldc}),
                                              std::span<TcCast>(cValues, sizeC)),
        nativeScalarType<TcCast>);
    if constexpr (is_std_complex_v<TcCast>) {
        problem.epilogue.alpha = {static_cast<double>(alphaCast.real()),
                                  static_cast<double>(alphaCast.imag())};
        problem.epilogue.beta = {static_cast<double>(betaCast.real()),
                                 static_cast<double>(betaCast.imag())};
    } else {
        problem.epilogue.alpha = {static_cast<double>(alphaCast), 0.0};
        problem.epilogue.beta = {static_cast<double>(betaCast), 0.0};
    }

    if (useBlas) {
        static const BlasGemmBackend backend;
        referenceGemm(problem, {
                                   .backend = GemmBackend::Blas,
                                   .requireRequestedBackend = true,
                                   .backendImplementation = &backend,
                               });
    } else {
        referenceGemm(problem);
    }

    if(scaleD != static_cast<Tc>(1))
    {
        sat_cast_mul<TcCast, Tc>(C, To, C_Tc, scaleD, sizeC);
    }
    else
    {
        if(To != Tc_enum)
        {
            sat_cast_mul<TcCast, Tc>(C, To, C_Tc, scaleD, sizeC);
        }
    }
}

#define CREATEFUNCTION(Tc)                                                  \
    template void hipblaslt_reference_gemm<Tc>(hipblasOperation_t       transA,           \
                                 hipblasOperation_t       transB,           \
                                 int64_t                  m,                \
                                 int64_t                  n,                \
                                 int64_t                  k,                \
                                 Tc                       alpha,            \
                                 const void*              A,                \
                                 int64_t                  lda,              \
                                 const void*              B,                \
                                 int64_t                  ldb,              \
                                 Tc                       beta,             \
                                 std::add_pointer_t<void> C,                \
                                 int64_t                  ldc,              \
                                 const void*              AlphaVec,         \
                                 const void*              scaleAVec,        \
                                 const void*              scaleBVec,        \
                                 Tc                       scaleD,           \
                                 bool                     isScaleAVec,      \
                                 bool                     isScaleBVec,      \
                                 hipDataType              TiA,              \
                                 hipDataType              TiB,              \
                                 hipDataType              To,               \
                                 hipDataType              Tc_enum,          \
                                 hipDataType              TciA,             \
                                 hipDataType              TciB,             \
                                 bool                     alt,              \
                                 bool                     isScaleAMXFormat, \
                                 bool                     isScaleBMXFormat);

CREATEFUNCTION(hipblasLtHalf)
CREATEFUNCTION(float)
CREATEFUNCTION(double)
CREATEFUNCTION(int32_t)
CREATEFUNCTION(std::complex<float>)
CREATEFUNCTION(std::complex<double>)
