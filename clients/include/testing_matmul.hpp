/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "TensorDataManipulation.hpp"
#include "allclose.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "frequency_monitor.hpp"
#include "hipBuffer.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_random.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_vector.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <cstddef>
#include <functional>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <omp.h>
#include <set>

extern "C" __global__ void flush_icache()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}

bool isSwizzleSupported(hipDataType datatype)
{
    switch(datatype)
    {
    case HIP_R_16BF:
    case HIP_R_16F:
    case HIP_R_8F_E4M3_FNUZ:
        return true;
    default:
        return false;
    }
}

hipblasLtOrder_t orderForDatatype(hipDataType datatype)
{
    switch(datatype)
    {
    case HIP_R_16F:
    case HIP_R_16BF:
        return HIPBLASLT_ORDER_COL16_4R8;
    case HIP_R_8F_E4M3_FNUZ:
        return HIPBLASLT_ORDER_COL16_4R16;
    default:
        throw std::runtime_error("unsupported datatype in orderForDatatype");
    }
}

void calculateKforSwizzling(
    hipDataType datatype, const Arguments& arg, size_t& MiK, size_t& MiKv, size_t& PackK)
{
    switch(datatype)
    {
    case HIP_R_32F:
        if(arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_TF32)
        {
            MiK  = 8;
            MiKv = 2;
        }
        else
        {
            MiK  = 4;
            MiKv = 1;
        }
        break;
    case HIP_R_64F:
        MiK  = 4;
        MiKv = 1;
        break;
    case HIP_R_16F:
    case HIP_R_16BF:
        MiK  = 16;
        MiKv = 4;
        break;
    case HIP_R_8I:
    case HIP_R_8F_E5M2_FNUZ:
    case HIP_R_8F_E4M3_FNUZ:
#ifdef ROCM_USE_FLOAT8
    case HIP_R_8F_E4M3:
    case HIP_R_8F_E5M2:
#endif
        MiK  = 32;
        MiKv = 8;
        break;
    default:
        throw std::runtime_error("unsupported datatype in calculateKforSwizzling");
    }

    PackK = 16 / MiKv / realDataTypeSize(datatype);
}

template <typename T>
void swizzle_tensor(T*               dst,
                    const T*         src,
                    hipDataType      datatype,
                    const Arguments& arg,
                    size_t           b,
                    size_t           m,
                    size_t           k,
                    size_t           ld,
                    bool             colMaj)
{
    if(ld < k)
        throw std::runtime_error("invalid value of ld in swizzle_tensor: ld must be >= k.");

    using Tensor = Tensor::Manipulation::Tensor;
    size_t MiM   = 16;
    size_t MiK = 0, MiKv = 0, PackK = 0;
    calculateKforSwizzling(datatype, arg, MiK, MiKv, PackK);
    const size_t numElements = b * m * k;
    auto         tmpTensor   = Tensor::create<T>({b, m, k});

    if(colMaj)
    {
        auto orgTensor = Tensor::create<T>({b, k, m});
        for(size_t i = 0; i < b * k; i++)
        {
            memcpy(orgTensor.template as<T>() + (i * m), src + (i * ld), m * sizeof(T));
        }
        tmpTensor = permute(orgTensor, {0, 2, 1});
    }
    else
    {
        for(size_t i = 0; i < b * m; i++)
        {
            memcpy(tmpTensor.template as<T>() + (i * k), src + (i * ld), k * sizeof(T));
        }
    }

    auto                          MultipleM = MiM;
    auto                          MultipleK = MiK * PackK;
    const auto                    paddedM   = (m / MultipleM + !!(m % MultipleM)) * MultipleM;
    const auto                    paddedK   = (k / MultipleK + !!(k % MultipleK)) * MultipleK;
    ::Tensor::Manipulation::Shape paddedShape{b, paddedM, paddedK};
    auto paddedTensor = ::Tensor::Manipulation::pad(tmpTensor, paddedShape, T(0));
    paddedTensor.reshape(
        {b, paddedM / MiM, MiM, paddedK / (MiK * PackK), MiK / MiKv, MiKv * PackK});
    Tensor permuted = permute(paddedTensor, {0, 1, 3, 4, 2, 5});
    memcpy(dst, permuted.template as<void>(), b * paddedM * paddedK * sizeof(T));
}

void swizzle_tensor_type(HipHostBuffer&       dst,
                         const HipHostBuffer& src,
                         hipDataType          datatype,
                         const Arguments&     arg,
                         size_t               b,
                         size_t               m,
                         size_t               k,
                         size_t               ld,
                         bool                 colMaj)
{
    switch(datatype)
    {
    case HIP_R_32F:
        swizzle_tensor<float>(dst.as<float>(), src.as<float>(), datatype, arg, b, m, k, ld, colMaj);
        return;
    case HIP_R_16F:
        swizzle_tensor<hipblasLtHalf>(
            dst.as<hipblasLtHalf>(), src.as<hipblasLtHalf>(), datatype, arg, b, m, k, ld, colMaj);
        return;
    case HIP_R_16BF:
        swizzle_tensor<hip_bfloat16>(
            dst.as<hip_bfloat16>(), src.as<hip_bfloat16>(), datatype, arg, b, m, k, ld, colMaj);
        return;
    case HIP_R_8F_E4M3_FNUZ:
        swizzle_tensor<hipblaslt_f8_fnuz>(dst.as<hipblaslt_f8_fnuz>(),
                                          src.as<hipblaslt_f8_fnuz>(),
                                          datatype,
                                          arg,
                                          b,
                                          m,
                                          k,
                                          ld,
                                          colMaj);
        return;
    case HIP_R_8F_E5M2_FNUZ:
        swizzle_tensor<hipblaslt_bf8_fnuz>(dst.as<hipblaslt_bf8_fnuz>(),
                                           src.as<hipblaslt_bf8_fnuz>(),
                                           datatype,
                                           arg,
                                           b,
                                           m,
                                           k,
                                           ld,
                                           colMaj);
        return;
#ifdef ROCM_USE_FLOAT8
    case HIP_R_8F_E4M3:
        swizzle_tensor<hipblaslt_f8>(
            dst.as<hipblaslt_f8>(), src.as<hipblaslt_f8>(), datatype, arg, b, m, k, ld, colMaj);
        return;
    case HIP_R_8F_E5M2:
        swizzle_tensor<hipblaslt_bf8>(
            dst.as<hipblaslt_bf8>(), src.as<hipblaslt_bf8>(), datatype, arg, b, m, k, ld, colMaj);
        return;
#endif
    default:
        hipblaslt_cerr << "Error type in swizzle_tensor_type()" << std::endl;
    }
}

inline void pre_gpu_time(bool         use_gpu_timer,
                         hipEvent_t&  event_gpu_time_start,
                         double&      gpu_time_used,
                         hipStream_t& stream)
{
    if(use_gpu_timer)
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
    else
        gpu_time_used = get_time_us_sync(stream);
}
inline void post_gpu_time(bool         use_gpu_timer,
                          hipEvent_t&  event_gpu_time_start,
                          hipEvent_t&  event_gpu_time_end,
                          double&      gpu_time_used,
                          hipStream_t& stream)
{
    if(use_gpu_timer)
    {
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
        CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
        float gpu_time_ms;
        CHECK_HIP_ERROR(
            hipEventElapsedTime(&gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
        gpu_time_used = gpu_time_ms * 1000; // ms to us
    }
    else
    {
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
    }
}

template <typename Tout>
Tout cast_from_type(void* in, hipDataType type, size_t index)
{
    switch(type)
    {
    case HIP_R_32F:
        return static_cast<Tout>((static_cast<float*>(in))[index]);
    case HIP_R_64F:
        return static_cast<Tout>((static_cast<double*>(in))[index]);
    case HIP_R_16F:
        return static_cast<Tout>((static_cast<hipblasLtHalf*>(in))[index]);
    case HIP_R_16BF:
        return static_cast<Tout>((static_cast<hip_bfloat16*>(in))[index]);
    case HIP_R_8F_E4M3_FNUZ:
        if constexpr(std::is_same<Tout, float>::value)
            return static_cast<Tout>((static_cast<hipblaslt_f8_fnuz*>(in))[index]);
        return 0;
    case HIP_R_8F_E5M2_FNUZ:
        if constexpr(std::is_same<Tout, float>::value)
            return static_cast<Tout>((static_cast<hipblaslt_bf8_fnuz*>(in))[index]);
        return 0;
#ifdef ROCM_USE_FLOAT8
    case HIP_R_8F_E4M3:
        if constexpr(std::is_same<Tout, float>::value)
            return static_cast<Tout>((static_cast<hipblaslt_f8*>(in))[index]);
        return 0;
    case HIP_R_8F_E5M2:
        if constexpr(std::is_same<Tout, float>::value)
            return static_cast<Tout>((static_cast<hipblaslt_bf8*>(in))[index]);
        return 0;
#endif
    case HIP_R_32I:
        return static_cast<Tout>((static_cast<int32_t*>(in))[index]);
    case HIP_R_8I:
        return static_cast<Tout>((static_cast<hipblasLtInt8*>(in))[index]);
    default:
        hipblaslt_cerr << "Error type in cast_from_type()" << std::endl;
        return 0;
    }
}

template <typename Tin>
void saturate_cast_to_type(void* dst, Tin src, hipDataType typeD, size_t indexD)
{
    switch(typeD)
    {
    case HIP_R_32F:
        static_cast<float*>(dst)[indexD] = saturate_cast<float>(src);
        return;
    case HIP_R_64F:
        static_cast<double*>(dst)[indexD] = saturate_cast<double>(src);
        return;
    case HIP_R_16F:
        static_cast<hipblasLtHalf*>(dst)[indexD] = saturate_cast<hipblasLtHalf>(src);
        return;
    case HIP_R_16BF:
        static_cast<hip_bfloat16*>(dst)[indexD] = saturate_cast<hip_bfloat16>(src);
        return;
    case HIP_R_8F_E4M3_FNUZ:
        static_cast<hipblaslt_f8_fnuz*>(dst)[indexD] = saturate_cast<hipblaslt_f8_fnuz>(src);
        return;
    case HIP_R_8F_E5M2_FNUZ:
        static_cast<hipblaslt_bf8_fnuz*>(dst)[indexD] = saturate_cast<hipblaslt_bf8_fnuz>(src);
        return;
#ifdef ROCM_USE_FLOAT8
    case HIP_R_8F_E4M3:
        static_cast<hipblaslt_f8*>(dst)[indexD] = saturate_cast<hipblaslt_f8>(src);
        return;
    case HIP_R_8F_E5M2:
        static_cast<hipblaslt_bf8*>(dst)[indexD] = saturate_cast<hipblaslt_bf8>(src);
        return;
#endif
    case HIP_R_32I:
        static_cast<int32_t*>(dst)[indexD] = saturate_cast<int32_t>(src);
        return;
    case HIP_R_8I:
        static_cast<hipblasLtInt8*>(dst)[indexD] = saturate_cast<hipblasLtInt8>(src);
        return;
    default:
        hipblaslt_cerr << "Error type in cast_from_type()" << std::endl;
    }
}

template <typename Ti, typename Tc, typename Tact, typename F>
void epilogue_func(int64_t     m,
                   int64_t     n,
                   int64_t     ld,
                   Ti*         in,
                   void*       out,
                   Tc*         out_raw,
                   Tc*         amaxD,
                   void*       e,
                   Tc          scaleD,
                   Tc          scaleE,
                   bool        enable_bias,
                   void*       bias,
                   hipDataType bias_type,
                   Tact        arg1,
                   Tact        arg2,
                   F&          act_func,
                   bool        gradient,
                   hipDataType To)
{
    for(int i = 0; i < m; i++)
    {
        Ti bias_data = enable_bias ? cast_from_type<Ti>(bias, bias_type, i) : 0;

#define CALCULATE_EPILOGUE_ACT                                                          \
    auto pos     = j * ld + i;                                                          \
    auto in_Tact = static_cast<Tact>(in[pos]) + bias_data;                              \
    if(e && !gradient)                                                                  \
    {                                                                                   \
        saturate_cast_to_type(e, in_Tact* scaleE, To, pos);                             \
    }                                                                                   \
    Tact in_Tact_act = 0;                                                               \
    if(gradient)                                                                        \
    {                                                                                   \
        in_Tact_act = act_func(cast_from_type<Tact>(e, To, pos), arg1, arg2) * in_Tact; \
    }                                                                                   \
    else                                                                                \
        in_Tact_act = act_func(in_Tact, arg1, arg2);

        if(amaxD == nullptr)
        {
#pragma omp parallel for
            for(int j = 0; j < n; j++)
            {
                CALCULATE_EPILOGUE_ACT;
                saturate_cast_to_type(out, in_Tact_act * scaleD, To, pos);
                *(out_raw + pos) = static_cast<Tc>(in_Tact_act * scaleD);
            }
        }
        else
        {
            for(int j = 0; j < n; j++)
            {
                CALCULATE_EPILOGUE_ACT;
                *amaxD = *amaxD > fabs(static_cast<Tc>(in_Tact_act))
                             ? *amaxD
                             : fabs(static_cast<Tc>(in_Tact_act));
                saturate_cast_to_type(out, in_Tact_act * scaleD, To, pos);
                *(out_raw + pos) = static_cast<Tc>(in_Tact_act * scaleD);
            }
        }
    }
}

template <typename Tact, typename F>
void epilogue_func(int64_t     m,
                   int64_t     n,
                   int64_t     ld,
                   void*       in,
                   void*       out,
                   void*       out_raw,
                   void*       amaxD,
                   void*       e,
                   void*       scaleD,
                   void*       scaleE,
                   bool        enable_bias,
                   void*       bias,
                   hipDataType bias_type,
                   Tact        arg1,
                   Tact        arg2,
                   F&          act_func,
                   bool        gradient,
                   hipDataType To,
                   hipDataType Tc)
{
    switch(Tc)
    {
    case HIP_R_32F:
        epilogue_func(m,
                      n,
                      ld,
                      (float*)in,
                      out,
                      (float*)out_raw,
                      (float*)amaxD,
                      e,
                      *(float*)scaleD,
                      *(float*)scaleE,
                      enable_bias,
                      bias,
                      bias_type,
                      arg1,
                      arg2,
                      act_func,
                      gradient,
                      To);
        return;
    case HIP_R_64F:
        epilogue_func(m,
                      n,
                      ld,
                      (double*)in,
                      out,
                      (double*)out_raw,
                      (double*)amaxD,
                      e,
                      *(double*)scaleD,
                      *(double*)scaleE,
                      enable_bias,
                      bias,
                      bias_type,
                      arg1,
                      arg2,
                      act_func,
                      gradient,
                      To);
        return;
    case HIP_R_32I:
        epilogue_func(m,
                      n,
                      ld,
                      (int32_t*)in,
                      out,
                      (int32_t*)out_raw,
                      (int32_t*)amaxD,
                      e,
                      *(int32_t*)scaleD,
                      *(int32_t*)scaleE,
                      enable_bias,
                      bias,
                      bias_type,
                      arg1,
                      arg2,
                      act_func,
                      gradient,
                      To);
        return;
    default:
        hipblaslt_cerr << "Error type in epilogue_func()" << std::endl;
        return;
    }
}

template <typename Ti, typename Tc>
void epilogue_func(int64_t     m,
                   int64_t     n,
                   int64_t     ld,
                   Ti*         in,
                   void*       out,
                   Tc*         out_raw,
                   Tc*         amaxD,
                   void*       e,
                   Tc          scaleD,
                   Tc          scaleE,
                   bool        enable_bias,
                   void*       bias,
                   hipDataType bias_type,
                   bool        gradient,
                   hipDataType To)
{
#define CALCULATE_EPILOGUE_BASIC                          \
    auto pos  = j * ld + i;                               \
    Tc   temp = static_cast<Ti>(*(in + pos)) + bias_data; \
    if(e)                                                 \
    {                                                     \
        saturate_cast_to_type(e, temp* scaleE, To, pos);  \
    }

    for(int i = 0; i < m; i++)
    {
        Ti bias_data = enable_bias ? cast_from_type<Ti>(bias, bias_type, i) : 0;

        if(amaxD == nullptr)
        {
#pragma omp parallel for
            for(int j = 0; j < n; j++)
            {
                CALCULATE_EPILOGUE_BASIC;
                temp *= scaleD;
                saturate_cast_to_type(out, temp, To, pos);
                *(out_raw + pos) = static_cast<Tc>(temp);
            }
        }
        else
        {
            for(int j = 0; j < n; j++)
            {
                CALCULATE_EPILOGUE_BASIC;
                *amaxD
                    = *amaxD > fabs(static_cast<Tc>(temp)) ? *amaxD : fabs(static_cast<Tc>(temp));
                temp *= scaleD;
                saturate_cast_to_type(out, temp, To, pos);
                *(out_raw + pos) = static_cast<Tc>(temp);
            }
        }
    }
}

void epilogue_func(int64_t     m,
                   int64_t     n,
                   int64_t     ld,
                   void*       in,
                   void*       out,
                   void*       out_raw,
                   void*       amaxD,
                   void*       e,
                   void*       scaleD,
                   void*       scaleE,
                   bool        enable_bias,
                   void*       bias,
                   hipDataType bias_type,
                   bool        gradient,
                   hipDataType To,
                   hipDataType Tc)
{
    switch(Tc)
    {
    case HIP_R_32F:
        epilogue_func(m,
                      n,
                      ld,
                      (float*)in,
                      out,
                      (float*)out_raw,
                      (float*)amaxD,
                      e,
                      *(float*)scaleD,
                      *(float*)scaleE,
                      enable_bias,
                      bias,
                      bias_type,
                      gradient,
                      To);
        return;
    case HIP_R_64F:
        epilogue_func(m,
                      n,
                      ld,
                      (double*)in,
                      out,
                      (double*)out_raw,
                      (double*)amaxD,
                      e,
                      *(double*)scaleD,
                      *(double*)scaleE,
                      enable_bias,
                      bias,
                      bias_type,
                      gradient,
                      To);
        return;
    case HIP_R_32I:
        epilogue_func(m,
                      n,
                      ld,
                      (int32_t*)in,
                      out,
                      (int32_t*)out_raw,
                      (int32_t*)amaxD,
                      e,
                      *(int32_t*)scaleD,
                      *(int32_t*)scaleE,
                      enable_bias,
                      bias,
                      bias_type,
                      gradient,
                      To);
        return;
    default:
        hipblaslt_cerr << "Error type in epilogue_func()" << std::endl;
        return;
    }
}

template <bool SumLd, typename Tc>
void reduction_func(void*       workspace,
                    hipDataType ti,
                    void*       bias,
                    hipDataType bias_type,
                    int         length,
                    int         k,
                    int         s1,
                    int         s2,
                    int         s3,
                    int         batch_count)
{
    assert(batch_count == 1);
    for(int batch = 0; batch < batch_count; batch++)
    {
        for(int i1 = 0; i1 < length; i1++)
        {
            Tc sum = 0;
            for(int i2 = 0; i2 < k; i2++)
            {
                if constexpr(SumLd)
                {
                    sum += cast_from_type<Tc>(workspace, ti, i1 * s2 + i2 * s1 + batch * s3);
                }
                else
                {
                    sum += cast_from_type<Tc>(workspace, ti, i1 * s1 + i2 * s2 + batch * s3);
                }
            }
            saturate_cast_to_type(bias, sum, bias_type, i1);
        }
    }
}

auto _relu = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    return static_cast<decltype(in)>(std::max(static_cast<decltype(in)>(0), in));
};

auto _gelu = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    using Tc = float;

    constexpr auto k0    = static_cast<Tc>(0.7978845608028654);
    constexpr auto k1    = static_cast<Tc>(0.044715);
    Tc             in_Tc = static_cast<Tc>(in);

    return static_cast<decltype(in)>(
        0.5f * (in_Tc * (1.f + std::tanh(k0 * (in_Tc * (1.f + k1 * (in_Tc * in_Tc)))))));
};

auto _dgelu = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    using Tc = float;

    constexpr auto k0    = static_cast<Tc>(0.0535161);
    constexpr auto k1    = static_cast<Tc>(0.398942);
    constexpr auto k2    = static_cast<Tc>(0.0356774);
    constexpr auto k3    = static_cast<Tc>(0.797885);
    Tc             in_Tc = static_cast<Tc>(in);

    Tc pow3 = in_Tc * in_Tc * in_Tc;
    Tc x1   = k0 * pow3 + k1 * in_Tc;
    Tc xx   = k2 * pow3 + k3 * in_Tc;
    Tc x2   = 4 / pow(exp(-xx) + exp(xx), 2);
    Tc tmp  = 0.5 * tanh(xx) + x1 * x2 + 0.5;
    return static_cast<decltype(in)>(0.5f * tanh(xx) + x1 * x2 + 0.5f);
};

// swish with beta=1
auto _silu = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    using Tc = float;
    Tc in_Tc = static_cast<Tc>(in);
    return static_cast<decltype(in)>(in_Tc / (1.f + exp(-in_Tc)));
};

void testing_matmul_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const size_t safe_size = N * lda;

    const hipblasOperation_t transA = HIPBLAS_OP_T;
    const hipblasOperation_t transB = HIPBLAS_OP_N;

    // allocate memory on device
    HipDeviceBuffer dA(arg.a_type, safe_size / 2, arg.HMM);
    HipDeviceBuffer dB(arg.b_type, safe_size, arg.HMM);
    HipDeviceBuffer dC(arg.c_type, safe_size, arg.HMM);
    HipDeviceBuffer dD(arg.d_type, safe_size, arg.HMM);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());

    hipblaslt_local_handle        handle{arg};
    hipblaslt_local_matrix_layout matA(M, K, lda, arg.a_type);
    hipblaslt_local_matrix_layout matB(K, N, ldb, arg.b_type);
    hipblaslt_local_matrix_layout matC(M, N, ldc, arg.c_type);
    hipblaslt_local_matrix_layout matD(M, N, ldc, arg.d_type);
    hipblaslt_local_matmul_descr  matmul(transA,
                                        transB,
                                        arg.compute_type,
                                        arg.scale_type,
                                        arg.compute_input_typeA,
                                        arg.compute_input_typeB);

    size_t                     workspace_size = 0;
    hipblaslt_local_preference pref;

    void* workspace = nullptr;
    float alpha = 1.0, beta = 0.0;

    hipStream_t stream = nullptr;
}

void copy_gemm_to_host(hipStream_t                   stream,
                       const uint32_t&               gemm_count,
                       std::vector<HipHostBuffer>&   hDst,
                       std::vector<HipDeviceBuffer>& dSrc)
{

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
    {
        CHECK_HIP_ERROR(synchronize(hDst[gemmIdx], dSrc[gemmIdx]));
    }
}

void check(hipStream_t                   stream,
           const Arguments&              arg,
           const uint32_t&               gemm_count,
           const std::vector<int64_t>&   M,
           const std::vector<int64_t>&   N,
           const std::vector<int64_t>&   ldd,
           const std::vector<int64_t>&   lde,
           const std::vector<int64_t>&   stride_d,
           const std::vector<int64_t>&   stride_e,
           const std::vector<int>&       num_batches,
           const std::vector<size_t>&    size_bias,
           std::vector<HipHostBuffer>&   hD_gold,
           std::vector<HipHostBuffer>&   hD_1,
           std::vector<HipDeviceBuffer>& dD,
           std::vector<HipHostBuffer>&   hAmaxD_gold,
           std::vector<HipHostBuffer>&   hAmaxD,
           std::vector<HipDeviceBuffer>& dAmaxD,
           std::vector<HipHostBuffer>&   hE_gold,
           std::vector<HipHostBuffer>&   hE,
           std::vector<HipDeviceBuffer>& dE,
           std::vector<HipHostBuffer>&   hBias_gold,
           std::vector<HipHostBuffer>&   hBias,
           std::vector<HipDeviceBuffer>& dBias,
           std::vector<double>&          tol,
           double&                       hipblaslt_error,
           double&                       hipblaslt_atol,
           double&                       hipblaslt_rtol,
           hipDataType                   To,
           hipDataType                   Tbias,
           hipDataType                   Tc)
{
    // fetch GPU
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
    {
        if(!arg.gradient && arg.use_e)
        {
            CHECK_HIP_ERROR(synchronize(hE[gemmIdx], dE[gemmIdx]));
        }

        if(arg.amaxD)
        {
            CHECK_HIP_ERROR(synchronize(hAmaxD[gemmIdx], dAmaxD[gemmIdx]));
        }
        if(arg.gradient && arg.bias_vector)
        {
            CHECK_HIP_ERROR(synchronize(hBias[gemmIdx], dBias[gemmIdx]));
        }
        if(arg.unit_check)
        {
            if(tol[gemmIdx] != 0)
            {
                near_check_general(M[gemmIdx],
                                   N[gemmIdx],
                                   ldd[gemmIdx],
                                   stride_d[gemmIdx],
                                   hD_gold[gemmIdx].buf(),
                                   hD_1[gemmIdx].buf(),
                                   num_batches[gemmIdx],
                                   tol[gemmIdx],
                                   To);
            }
            else
            {
                unit_check_general(M[gemmIdx],
                                   N[gemmIdx],
                                   ldd[gemmIdx],
                                   stride_d[gemmIdx],
                                   hD_gold[gemmIdx].buf(),
                                   hD_1[gemmIdx].buf(),
                                   num_batches[gemmIdx],
                                   To);
            }
            if(arg.amaxD)
            {
                if(tol[gemmIdx] != 0)
                {
                    near_check_general(1,
                                       1,
                                       1,
                                       1,
                                       hAmaxD_gold[gemmIdx].buf(),
                                       hAmaxD[gemmIdx].buf(),
                                       num_batches[gemmIdx],
                                       tol[gemmIdx],
                                       Tc);
                }
                else
                {
                    unit_check_general(1,
                                       1,
                                       1,
                                       1,
                                       hAmaxD_gold[gemmIdx].buf(),
                                       hAmaxD[gemmIdx].buf(),
                                       num_batches[gemmIdx],
                                       Tc);
                }
            }
            if(!arg.gradient && arg.use_e)
            {
                if(tol[gemmIdx] != 0)
                {
                    near_check_general(M[gemmIdx],
                                       N[gemmIdx],
                                       lde[gemmIdx],
                                       stride_e[gemmIdx],
                                       hE_gold[gemmIdx].buf(),
                                       hE[gemmIdx].buf(),
                                       num_batches[gemmIdx],
                                       tol[gemmIdx],
                                       To);
                }
                else
                {
                    unit_check_general(M[gemmIdx],
                                       N[gemmIdx],
                                       lde[gemmIdx],
                                       stride_e[gemmIdx],
                                       hE_gold[gemmIdx].buf(),
                                       hE[gemmIdx].buf(),
                                       num_batches[gemmIdx],
                                       To);
                }
            }
            if(arg.gradient && arg.bias_vector)
            {
                if(tol[gemmIdx] != 0)
                {
                    near_check_general(size_bias[gemmIdx],
                                       1,
                                       size_bias[gemmIdx],
                                       size_bias[gemmIdx],
                                       hBias_gold[gemmIdx].buf(),
                                       hBias[gemmIdx].buf(),
                                       num_batches[gemmIdx],
                                       tol[gemmIdx],
                                       Tbias);
                }
                else
                {
                    unit_check_general(size_bias[gemmIdx],
                                       1,
                                       size_bias[gemmIdx],
                                       size_bias[gemmIdx],
                                       hBias_gold[gemmIdx].buf(),
                                       hBias[gemmIdx].buf(),
                                       num_batches[gemmIdx],
                                       Tbias);
                }
            }
        }

        if(arg.norm_check)
        {
            double norm_error = 0.0;
            norm_error        = std::abs(norm_check_general('F',
                                                     M[gemmIdx],
                                                     N[gemmIdx],
                                                     ldd[gemmIdx],
                                                     stride_d[gemmIdx],
                                                     hD_gold[gemmIdx].buf(),
                                                     hD_1[gemmIdx].buf(),
                                                     num_batches[gemmIdx],
                                                     To));
            hipblaslt_error += norm_error;
            if(arg.norm_check_assert)
            {
                CHECK_SUCCESS(norm_check(norm_error, To));
            }

            if(arg.amaxD)
            {
                double norm_error = std::abs(norm_check_general('F',
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                hAmaxD_gold[gemmIdx].buf(),
                                                                hAmaxD[gemmIdx].buf(),
                                                                num_batches[gemmIdx],
                                                                Tc));
                hipblaslt_error += norm_error;
                if(arg.norm_check_assert)
                    CHECK_SUCCESS(norm_check(norm_error, Tc));
            }
            if(!arg.gradient && arg.use_e)
            {
                double norm_error = 0.0;
                norm_error        = std::abs(norm_check_general('F',
                                                         M[gemmIdx],
                                                         N[gemmIdx],
                                                         lde[gemmIdx],
                                                         stride_e[gemmIdx],
                                                         hE_gold[gemmIdx].buf(),
                                                         hE[gemmIdx].buf(),
                                                         num_batches[gemmIdx],
                                                         To));
                hipblaslt_error += norm_error;
                if(arg.norm_check_assert)
                {
                    CHECK_SUCCESS(norm_check(norm_error, To));
                }
            }
            if(arg.gradient && arg.bias_vector)
            {
                double norm_error = 0.0;
                norm_error        = std::abs(norm_check_general('F',
                                                         M[gemmIdx],
                                                         1,
                                                         M[gemmIdx],
                                                         M[gemmIdx],
                                                         hBias_gold[gemmIdx].buf(),
                                                         hBias[gemmIdx].buf(),
                                                         num_batches[gemmIdx],
                                                         Tbias));
                hipblaslt_error += norm_error;
                if(arg.norm_check_assert)
                {
                    CHECK_SUCCESS(norm_check(norm_error, Tbias));
                }
            }
        }

        if(arg.allclose_check)
        {
            bool is_allclose = allclose_check_general('F',
                                                      M[gemmIdx],
                                                      N[gemmIdx],
                                                      ldd[gemmIdx],
                                                      stride_d[gemmIdx],
                                                      hD_gold[gemmIdx].buf(),
                                                      hD_1[gemmIdx].buf(),
                                                      num_batches[gemmIdx],
                                                      hipblaslt_atol,
                                                      hipblaslt_rtol,
                                                      To);
            //TODO: confirm if allclose_check_assert is neccessary
        }
    }
}

// A function to determine the default bias_type
hipDataType derive_unset_bias_type(const Arguments& arg)
{
    // TODO: confirm if HIP_R_64F, HIP_R_32I are neccessary for biastype
    static const std::set<hipDataType> supported_bias_types
        = {HIP_R_32F, HIP_R_16F, HIP_R_16BF, HIP_R_64F, HIP_R_32I};

    hipDataType real_bias_type = arg.bias_type;

    // when bias type is unset
    if(arg.bias_type == HIPBLASLT_DATATYPE_INVALID)
    {
        if(arg.compute_type == HIPBLAS_COMPUTE_32I)
        {
            real_bias_type = HIP_R_32I;
        }
        else if(arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_TF32)
        {
            real_bias_type = HIP_R_32F;
        }
        else if((arg.a_type == HIP_R_8F_E4M3_FNUZ || arg.a_type == HIP_R_8F_E5M2_FNUZ)
                && (arg.b_type == HIP_R_8F_E4M3_FNUZ || arg.b_type == HIP_R_8F_E5M2_FNUZ))
        {
            if(arg.d_type == HIP_R_32F || arg.d_type == HIP_R_16BF)
                real_bias_type = HIP_R_16BF;
            else if(arg.d_type == HIP_R_16F)
                real_bias_type = HIP_R_16F;
            else //more default cases once support C != D
                real_bias_type = HIP_R_16F;
        }
#ifdef ROCM_USE_FLOAT8
        else if((arg.a_type == HIP_R_8F_E4M3 || arg.a_type == HIP_R_8F_E5M2)
                && (arg.b_type == HIP_R_8F_E4M3 || arg.b_type == HIP_R_8F_E5M2))
        {
            if(arg.d_type == HIP_R_32F || arg.d_type == HIP_R_16BF)
                real_bias_type = HIP_R_16BF;
            else if(arg.d_type == HIP_R_16F)
                real_bias_type = HIP_R_16F;
            else //more default cases once support C != D
                real_bias_type = HIP_R_16F;
        }
#endif
        else
        {
            real_bias_type = arg.d_type;
        }
    }

    if(supported_bias_types.count(real_bias_type) == 0)
        throw std::invalid_argument("Invalid bias type "
                                    + std::string(hip_datatype_to_string(real_bias_type)));

    return real_bias_type;
}

// A function to determine the default compute_input_type
std::tuple<hipDataType, hipDataType> derive_unset_compute_input_type(const Arguments& arg)
{
    static const std::set<hipDataType> supported_compute_input_types = {
        HIP_R_32F,
        HIP_R_16BF,
        HIP_R_16F,
#ifdef ROCM_USE_FLOAT8
        HIP_R_8F_E4M3,
        HIP_R_8F_E5M2,
#endif
        HIP_R_8F_E4M3_FNUZ,
        HIP_R_8F_E5M2_FNUZ,
    };

    hipDataType real_compute_input_typeA = arg.compute_input_typeA;
    hipDataType real_compute_input_typeB = arg.compute_input_typeB;

    if(real_compute_input_typeA != HIPBLASLT_DATATYPE_INVALID
       && !supported_compute_input_types.count(real_compute_input_typeA))
        throw std::invalid_argument(
            "Invalid compute_input_typeA "
            + std::string(hip_datatype_to_string(real_compute_input_typeA)));

    if(real_compute_input_typeA != HIPBLASLT_DATATYPE_INVALID
       && !supported_compute_input_types.count(real_compute_input_typeB))
        throw std::invalid_argument(
            "Invalid compute_input_typeB "
            + std::string(hip_datatype_to_string(real_compute_input_typeB)));

    // when compute_input_type type is unset
    if(real_compute_input_typeA == HIPBLASLT_DATATYPE_INVALID)
    {
        real_compute_input_typeA = computeTypeToRealDataType(arg.compute_type);
    }

    if(real_compute_input_typeB == HIPBLASLT_DATATYPE_INVALID)
    {
        real_compute_input_typeB = computeTypeToRealDataType(arg.compute_type);
    }

    return {real_compute_input_typeA, real_compute_input_typeB};
}

void testing_matmul_with_bias(const Arguments& arg,
                              hipDataType      TiA,
                              hipDataType      TiB,
                              hipDataType      To,
                              hipDataType      Tc,
                              hipDataType      TciA,
                              hipDataType      TciB,
                              hipDataType      Tbias);

void testing_matmul(const Arguments& arg)
{
    hipDataType tiA = arg.a_type;
    hipDataType tiB = arg.b_type;
    hipDataType to  = arg.c_type;
    hipDataType tc  = computeTypeToRealDataType(arg.compute_type);
    hipDataType tciA, tciB;

    // after this, tciA and tciB should not be invalid
    std::tie(tciA, tciB) = derive_unset_compute_input_type(arg);

    // after this, real bias type should not be invalid
    hipDataType real_bias_type = derive_unset_bias_type(arg);
    Arguments   arg_revised    = arg;
    arg_revised.bias_type      = real_bias_type;

    // Set the values of flush, rotating size, cold_iters and hot_iters only for internal use
    hipblasltSetFlushValue(arg.flush);
    hipblasltSetRotatingBufferSizeValue(arg.rotating);
    hipblasltSetColdIterationsValue(arg.cold_iters);
    hipblasltSetHotIterationsValue(arg.iters);

    // for all f8/bf8 cases including mix mode
    if((realDataTypeSize(tiA) == 1 || realDataTypeSize(tiB) == 1) && tc != HIP_R_32I)
    {
        if(to == HIP_R_16BF || to == HIP_R_32F)
        {
            if(real_bias_type == HIP_R_16BF)
            {
                return testing_matmul_with_bias(
                    arg_revised, tiA, tiB, to, tc, tciA, tciB, HIP_R_16BF);
            }
            else
            {
                return testing_matmul_with_bias(
                    arg_revised, tiA, tiB, to, tc, tciA, tciB, HIP_R_32F);
            }
        }
        else
        {
            if(real_bias_type == HIP_R_16F)
            {
                return testing_matmul_with_bias(
                    arg_revised, tiA, tiB, to, tc, tciA, tciB, HIP_R_16F);
            }
            else
            {
                return testing_matmul_with_bias(
                    arg_revised, tiA, tiB, to, tc, tciA, tciB, HIP_R_32F);
            }
        }
    }
    else if(to == HIP_R_16F)
    {
        if(real_bias_type == HIP_R_16F)
        {
            return testing_matmul_with_bias(arg_revised, tiA, tiB, to, tc, tciA, tciB, HIP_R_16F);
        }
        else
        {
            return testing_matmul_with_bias(arg_revised, tiA, tiB, to, tc, tciA, tciB, HIP_R_32F);
        }
    }
    else if(to == HIP_R_16BF)
    {
        if(real_bias_type == HIP_R_16BF)
        {
            return testing_matmul_with_bias(arg_revised, tiA, tiB, to, tc, tciA, tciB, HIP_R_16BF);
        }
        else
        {
            return testing_matmul_with_bias(arg_revised, tiA, tiB, to, tc, tciA, tciB, HIP_R_32F);
        }
    }
    else if(to == HIP_R_32F || to == HIP_R_32I || to == HIP_R_8I || to == HIP_R_64F)
    {
        //set Tbias to To
        return testing_matmul_with_bias(arg_revised, tiA, tiB, to, tc, tciA, tciB, to);
    }
    // shouldn't arrive here
    hipblaslt_test_invalid{}(arg);
    return;
}

void testing_matmul_with_bias(const Arguments& arg,
                              hipDataType      TiA,
                              hipDataType      TiB,
                              hipDataType      To,
                              hipDataType      Tc,
                              hipDataType      TciA,
                              hipDataType      TciB,
                              hipDataType      Tbias)
{
    double gpu_time_used, cpu_time_used, gpu_mem_gbytes;
    gpu_time_used = cpu_time_used = gpu_mem_gbytes = 0.0;
    bool                   HMM                     = arg.HMM;
    hipblaslt_local_handle handle{arg};
    hipStream_t            stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    hipEvent_t event_gpu_time_start, event_gpu_time_end;
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_end));

    hipblasOperation_t transA(char_to_hipblas_operation(arg.transA));
    hipblasOperation_t transB(char_to_hipblas_operation(arg.transB));

    hipDataType Talpha = Tc;

    bool    do_grouped_gemm = arg.grouped_gemm > 0;
    int32_t gemm_count      = std::max(1, arg.grouped_gemm);
    int64_t rotating        = arg.rotating * 1024 * 1024;

    std::vector<int64_t> M(gemm_count), N(gemm_count), K(gemm_count), lda(gemm_count),
        ldb(gemm_count), ldc(gemm_count), ldd(gemm_count), lde(gemm_count);
    std::vector<computeTypeInterface> h_alpha(gemm_count), h_beta(gemm_count);
    std::vector<int64_t> A_row(gemm_count), A_col(gemm_count), B_row(gemm_count), B_col(gemm_count);
    std::vector<int64_t> stride_a(gemm_count), stride_da(gemm_count), stride_b(gemm_count),
        stride_c(gemm_count), stride_d(gemm_count), stride_e(gemm_count);
    std::vector<bool>   do_batched(gemm_count), epilogue_on(gemm_count, false);
    std::vector<int>    num_batches(gemm_count);
    std::vector<size_t> size_A(gemm_count), size_dA(gemm_count), size_B(gemm_count),
        size_C(gemm_count), size_D(gemm_count), size_D_copy(gemm_count), size_E(gemm_count),
        size_bias(gemm_count), size_scaleAlphaVec(gemm_count), size_scaleAVec(gemm_count),
        size_scaleBVec(gemm_count);

    std::vector<hipblasLtMatrixLayout_t> matA(gemm_count), matB(gemm_count), matC(gemm_count),
        matD(gemm_count);
    std::vector<std::vector<hipblasLtMatmulDesc_t>> matmul;
    std::vector<hipblasLtEpilogue_t> epilogue(gemm_count, HIPBLASLT_EPILOGUE_DEFAULT);

    std::vector<HipDeviceBuffer>  dA, dB, dC, dD, dE, dBias;
    std::vector<HipDeviceBuffer>* dDp;
    std::vector<HipDeviceBuffer>  dScaleAlphaVec, dScaleA, dScaleB, dScaleC, dScaleD, dScaleE,
        dAmaxD;

    std::vector<HipHostBuffer> hE, hE_gold, hBias, hBias_gold;
    std::vector<HipHostBuffer> hA, hB, hC, hD_gold, hD_1;
    std::vector<HipHostBuffer> hScaleAlphaVec, hScaleA, hScaleB, hScaleC, hScaleD, hScaleE,
        hAmaxD_gold, hAmaxD, hD_gold_epl, hD_gold_ScaleAlpha, hBias_gold_epl;

    std::vector<void*> alpha_in(gemm_count);

    // Need to split into two for loop to calculate the rotating buffer
    int64_t totalRotatingSizeNeeded = 0;
    for(int i = 0; i < gemm_count; i++)
    {
        M[i] = arg.M[i];
        N[i] = arg.N[i];
        K[i] = arg.K[i];
        set_alpha_type(h_alpha[i], arg, Tc);
        set_beta_type(h_beta[i], arg, Tc);
        lda[i] = arg.lda[i];
        ldb[i] = arg.ldb[i];
        ldc[i] = arg.ldc[i];
        ldd[i] = arg.ldd[i];
        lde[i] = arg.lde[i];

        A_row[i] = transA == HIPBLAS_OP_N ? M[i] : K[i];
        A_col[i] = transA == HIPBLAS_OP_N ? K[i] : M[i];
        B_row[i] = transB == HIPBLAS_OP_N ? K[i] : N[i];
        B_col[i] = transB == HIPBLAS_OP_N ? N[i] : K[i];

        do_batched[i]  = (arg.batch_count > 1);
        num_batches[i] = (do_batched[i] ? arg.batch_count : 1);

        stride_a[i] = (do_batched[i] && arg.stride_a[i] >= lda[i] * A_col[i]) ? arg.stride_a[i]
                                                                              : lda[i] * A_col[i];
        stride_b[i] = do_batched[i] ? arg.stride_b[i] : ldb[i] * B_col[i];
        stride_c[i] = do_batched[i] ? arg.stride_c[i] : ldc[i] * N[i];
        stride_d[i] = do_batched[i] ? arg.stride_c[i] : ldd[i] * N[i];
        stride_e[i] = do_batched[i] ? arg.stride_e[i] : lde[i] * N[i];

        size_A[i]    = stride_a[i] * num_batches[i];
        size_dA[i]   = size_A[i];
        stride_da[i] = stride_a[i];
        if(arg.swizzle_a && isSwizzleSupported(TiA))
        {
            stride_da[i] = 0;
            size_t MiM = 16, MiK = 0, __ = 0, PackK = 0;
            calculateKforSwizzling(TiA, arg, MiK, __, PackK);
            size_t K_block = MiK * PackK;
            size_t stride_swizzle
                = ((M[i] + MiM - 1) / MiM) * MiM * ((K[i] + K_block - 1) / K_block) * K_block;
            if(do_batched[i] && arg.stride_a[i] >= (int64_t)stride_swizzle)
            {
                stride_da[i]   = arg.stride_a[i];
                stride_swizzle = (size_t)arg.stride_a[i];
            }
            size_dA[i] = num_batches[i] * stride_swizzle;
        }

        size_B[i]
            = stride_b[i] == 0 ? ldb[i] * B_col[i] * num_batches[i] : stride_b[i] * num_batches[i];
        size_C[i]
            = stride_c[i] == 0 ? ldc[i] * N[i] * num_batches[i] : stride_c[i] * num_batches[i];
        size_D[i]
            = stride_d[i] == 0 ? ldd[i] * N[i] * num_batches[i] : stride_d[i] * num_batches[i];

        size_E[i] = arg.use_e ? (stride_e[i] == 0 ? lde[i] * N[i] * num_batches[i]
                                                  : stride_e[i] * num_batches[i])
                              : 0;
        if(arg.c_equal_d)
        {
            ldd[i]      = arg.ldc[i];
            stride_d[i] = stride_c[i];
            size_D[i]   = size_C[i];
        }

        size_D_copy[i] = (arg.unit_check || arg.norm_check || arg.allclose_check) ? size_D[i] : 0;
        size_scaleAlphaVec[i] = arg.scaleAlpha_vector ? M[i] : 0;
        if(arg.scaleA == hipblaslt_scaling_format::Scalar)
            size_scaleAVec[i] = 1;
        else if(arg.scaleA == hipblaslt_scaling_format::Vector)
            size_scaleAVec[i] = M[i];
        else
            size_scaleAVec[i] = 0;
        if(arg.scaleB == hipblaslt_scaling_format::Scalar)
            size_scaleBVec[i] = 1;
        else if(arg.scaleB == hipblaslt_scaling_format::Vector)
            size_scaleBVec[i] = N[i];
        else
            size_scaleBVec[i] = 0;
        if(arg.bias_vector)
        {
            if(arg.bias_source == hipblaslt_bias_source::a
               || arg.bias_source == hipblaslt_bias_source::d)
                size_bias[i] = M[i];
            else if(arg.bias_source == hipblaslt_bias_source::b)
                size_bias[i] = N[i];
        }
        else
        {
            size_bias[i] = 0;
        }
        auto    biasSize = size_bias[i] * realDataTypeSize(Tbias);
        int64_t sizeC    = get_computeInterface(h_beta[i], Tc) == 0 ? 0 : size_C[i] * sizeof(To);
        totalRotatingSizeNeeded
            += size_dA[i] * realDataTypeSize(TiA) + size_B[i] * realDataTypeSize(TiB) + sizeC
               + size_D[i] * realDataTypeSize(To) + size_E[i] * realDataTypeSize(To) + biasSize
               + size_scaleAlphaVec[i] * realDataTypeSize(Talpha)
               + size_scaleAVec[i] * realDataTypeSize(Talpha)
               + size_scaleBVec[i] * realDataTypeSize(Talpha);
    }

    gpu_mem_gbytes = static_cast<double>(totalRotatingSizeNeeded) / (1024 * 1024 * 1024);

    // Calculating block count
    int32_t max_iters   = max(arg.cold_iters, arg.iters);
    int32_t block_count = max(1, min(max_iters, ceil((float)rotating / totalRotatingSizeNeeded)));
    if(rotating > 0)
    {
        hipblaslt_cout << "Rotating buffer " << rotating / (1024 * 1024) << " MiB. "
                       << "Needed Size: " << totalRotatingSizeNeeded / (1024 * 1024) << " MiB. "
                       << "Needed block count: " << block_count
                       << " (Capped to max iters: " << max_iters << ")" << std::endl;
    }
    // Calculating block count end
    matmul.resize(block_count, std::vector<hipblasLtMatmulDesc_t>(gemm_count));

    for(int i = 0; i < gemm_count; i++)
    {
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matA[i]), arg.a_type, A_row[i], A_col[i], lda[i]));
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matB[i]), arg.b_type, B_row[i], B_col[i], ldb[i]));
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matC[i]), arg.c_type, M[i], N[i], ldc[i]));
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matD[i]), arg.d_type, M[i], N[i], ldd[i]));

        if(arg.swizzle_a && isSwizzleSupported(TiA))
        {
            hipblasLtOrder_t orderA = orderForDatatype(TiA);
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matA[i], HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
        }

        if(do_batched[i])
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matA[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matB[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matC[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matD[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matA[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_da[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matB[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_b[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matC[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_c[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matD[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_d[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
        }

        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatmulDescCreate(&(matmul[0][i]), arg.compute_type, arg.scale_type));

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &TciA, sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &TciB, sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);

        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul[0][i], HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul[0][i], HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)));

        if(arg.bias_vector)
        {
            epilogue_on[i] = true;
            switch(arg.activation_type)
            {
            case hipblaslt_activation_type::relu:
                epilogue[i] = HIPBLASLT_EPILOGUE_RELU_BIAS;
                break;
            case hipblaslt_activation_type::gelu:
                epilogue[i] = HIPBLASLT_EPILOGUE_GELU_BIAS;
                break;
            case hipblaslt_activation_type::swish:
                epilogue[i] = HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT;
                break;
            default:
                epilogue[i] = HIPBLASLT_EPILOGUE_BIAS;
                break;
            }
        }
        else
        {
            switch(arg.activation_type)
            {
            case hipblaslt_activation_type::relu:
                epilogue[i]    = HIPBLASLT_EPILOGUE_RELU;
                epilogue_on[i] = true;
                break;
            case hipblaslt_activation_type::gelu:
                epilogue[i]    = HIPBLASLT_EPILOGUE_GELU;
                epilogue_on[i] = true;
                break;
            case hipblaslt_activation_type::swish:
                epilogue[i]    = HIPBLASLT_EPILOGUE_SWISH_EXT;
                epilogue_on[i] = true;
                break;
            default:
                break;
            }
        }
        if(arg.gradient)
        {
            switch(epilogue[i])
            {
            case HIPBLASLT_EPILOGUE_BIAS:
            {
                switch(arg.bias_source)
                {
                case hipblaslt_bias_source::a:
                    epilogue[i] = HIPBLASLT_EPILOGUE_BGRADA;
                    break;
                case hipblaslt_bias_source::b:
                    epilogue[i] = HIPBLASLT_EPILOGUE_BGRADB;
                    break;
                default:
                    break;
                }
            }
            break;
            case HIPBLASLT_EPILOGUE_GELU:
                CHECK_SUCCESS(arg.use_e && "Must enable use e if gradient is enabled with gelu.");
                epilogue[i] = HIPBLASLT_EPILOGUE_DGELU;
                break;
            case HIPBLASLT_EPILOGUE_GELU_BIAS:
                CHECK_SUCCESS(arg.use_e && "Must enable use e if gradient is enabled with gelu.");
                epilogue[i] = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
                break;
            default:
                break;
            }
        }
        if(arg.use_e)
        {
            switch(epilogue[i])
            {
            case HIPBLASLT_EPILOGUE_GELU:
                epilogue[i] = HIPBLASLT_EPILOGUE_GELU_AUX;
                break;
            case HIPBLASLT_EPILOGUE_GELU_BIAS:
                epilogue[i] = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
                break;
            default:
                break;
            }
        }

        if(arg.scaleAlpha_vector)
        {
            epilogue_on[i] = true;
        }

        // allocate memory on device
        dA.emplace_back(TiA, size_dA[i] * block_count, HMM);
        dB.emplace_back(TiB, size_B[i] * block_count, HMM);
        dC.emplace_back(To, size_C[i] * block_count, HMM);

        if(!arg.c_equal_d)
        {
            dD.emplace_back(To, size_D[i] * block_count, HMM);
            dDp = &dD;
        }
        else
            dDp = &dC;

        if(size_bias[i] * block_count != 0)
            dBias.emplace_back(Tbias, size_bias[i] * block_count, HMM);

        if(arg.scaleAlpha_vector)
        {
            dScaleAlphaVec.emplace_back(Talpha, size_scaleAlphaVec[i] * block_count, HMM);
        }

        if(arg.use_e)
        {
            dE.emplace_back(To, size_E[i] * block_count, HMM);
        }

        if(arg.scaleA == hipblaslt_scaling_format::Scalar
           || arg.scaleA == hipblaslt_scaling_format::Vector)
        {
            dScaleA.emplace_back(Talpha, size_scaleAVec[i] * block_count, HMM);
        }
        if(arg.scaleB == hipblaslt_scaling_format::Scalar
           || arg.scaleB == hipblaslt_scaling_format::Vector)
        {
            dScaleB.emplace_back(Talpha, size_scaleBVec[i] * block_count, HMM);
        }
        if(arg.scaleC)
        {
            dScaleC.emplace_back(Talpha, 1, HMM);
        }
        if(arg.scaleD)
        {
            dScaleD.emplace_back(Talpha, 1, HMM);
        }
        if(arg.amaxD)
        {
            epilogue_on[i] = true;
            dAmaxD.emplace_back(Talpha, 1, HMM);
        }
        if(arg.scaleE)
        {
            dScaleE.emplace_back(Talpha, 1, HMM);
        }

        // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
        hA.emplace_back(TiA, size_A[i]);
        hB.emplace_back(TiB, size_B[i]);
        hC.emplace_back(To, size_C[i]);
        hD_gold.emplace_back(To, size_D_copy[i]);
        hD_1.emplace_back(To, size_D_copy[i]);
        if(size_bias[i] * block_count != 0)
        {
            hBias.emplace_back(Tbias, size_bias[i]);
            hBias_gold.emplace_back(Tbias, size_bias[i]);
        }

        hD_gold_epl.emplace_back(Talpha, size_D_copy[i]);
        hD_gold_ScaleAlpha.emplace_back(Talpha, size_D_copy[i]);
        hBias_gold_epl.emplace_back(Talpha, size_D_copy[i]); // Reduction for matrix D

        if(arg.scaleAlpha_vector)
            hScaleAlphaVec.emplace_back(Talpha, size_scaleAlphaVec[i]);

        if(arg.scaleA == hipblaslt_scaling_format::Scalar
           || arg.scaleA == hipblaslt_scaling_format::Vector)
            hScaleA.emplace_back(Talpha, size_scaleAVec[i]);
        if(arg.scaleB == hipblaslt_scaling_format::Scalar
           || arg.scaleB == hipblaslt_scaling_format::Vector)
            hScaleB.emplace_back(Talpha, size_scaleBVec[i]);
        if(arg.scaleC)
            hScaleC.emplace_back(Talpha, 1);
        if(arg.scaleD)
            hScaleD.emplace_back(Talpha, 1);
        if(arg.amaxD)
        {
            hAmaxD_gold.emplace_back(Talpha, 1);
            hAmaxD.emplace_back(Talpha, 1);
        }
        if(arg.scaleE)
            hScaleE.emplace_back(Talpha, 1);

        if(arg.use_e)
        {
            hE.emplace_back(To, size_E[i]);
            if(!arg.gradient)
            {
                hE_gold.emplace_back(To, size_E[i]);
            }
        }

        hipblaslt_seedrand();

        hipblaslt_init_device(ABC::A,
                              arg.initialization,
                              alpha_isnan_type(arg, Talpha),
                              dA[i].buf(),
                              A_row[i],
                              A_col[i],
                              (arg.swizzle_a) ? A_row[i] : lda[i],
                              TiA,
                              (arg.swizzle_a) ? A_row[i] * A_col[i] : stride_a[i],
                              num_batches[i]);
        hipblaslt_init_device(ABC::B,
                              arg.initialization,
                              alpha_isnan_type(arg, Talpha),
                              dB[i].buf(),
                              B_row[i],
                              B_col[i],
                              ldb[i],
                              TiB,
                              stride_b[i],
                              num_batches[i]);
        hipblaslt_init_device(ABC::C,
                              arg.initialization,
                              beta_isnan_type(arg, Talpha),
                              dC[i].buf(),
                              M[i],
                              N[i],
                              ldc[i],
                              To,
                              stride_c[i],
                              num_batches[i]);

        // broadcast first block
        CHECK_HIP_ERROR(broadcast(dA[i], block_count));
        CHECK_HIP_ERROR(broadcast(dB[i], block_count));
        CHECK_HIP_ERROR(broadcast(dC[i], block_count));

        if(arg.unit_check || arg.norm_check || arg.allclose_check || arg.swizzle_a)
        {
            CHECK_HIP_ERROR(synchronize(hA[i],
                                        dA[i],
                                        num_batches[i],
                                        A_row[i],
                                        A_col[i],
                                        lda[i],
                                        realDataTypeSize(TiA),
                                        arg.swizzle_a));
            CHECK_HIP_ERROR(synchronize(hB[i], dB[i]));
            CHECK_HIP_ERROR(synchronize(hC[i], dC[i]));
        }

        if(arg.swizzle_a && isSwizzleSupported(TiA))
        {
            HipHostBuffer tmp(TiA, size_dA[i]);
            swizzle_tensor_type(tmp, hA[i], TiA, arg, num_batches[i], M[i], K[i], lda[i], false);
            CHECK_HIP_ERROR(synchronize(dA[i], tmp, block_count));
        }

        if(arg.gradient && arg.use_e)
        {
            hipblaslt_init(hE[i].buf(), M[i], N[i], lde[i], To, stride_e[i], num_batches[i]);
        }

        if(arg.bias_vector)
        {
            hipblaslt_init(hBias[i].buf(), size_bias[i], 1, size_bias[i], Tbias);
        }

        if(arg.scaleA == hipblaslt_scaling_format::Scalar
           || arg.scaleA == hipblaslt_scaling_format::Vector)
            hipblaslt_init(hScaleA[i].buf(), size_scaleAVec[i], 1, size_scaleAVec[i], Talpha);

        if(arg.scaleB == hipblaslt_scaling_format::Scalar
           || arg.scaleB == hipblaslt_scaling_format::Vector)
            hipblaslt_init(hScaleB[i].buf(), size_scaleBVec[i], 1, size_scaleBVec[i], Talpha);

        if(arg.scaleC)
        {
            if(To == HIP_R_8F_E4M3_FNUZ || To == HIP_R_8F_E5M2_FNUZ)
            {
                hipblaslt_init_small(hScaleC[i].buf(), 1, 1, 1, Talpha);
            }
            else
            {
                hipblaslt_init(hScaleC[i].buf(), 1, 1, 1, Talpha);
            }
        }

        if(arg.scaleD)
        {
            if(To == HIP_R_8F_E4M3_FNUZ || To == HIP_R_8F_E5M2_FNUZ)
            {
                hipblaslt_init_small(hScaleD[i].buf(), 1, 1, 1, Talpha);
            }
            else
            {
                hipblaslt_init(hScaleD[i].buf(), 1, 1, 1, Talpha);
            }
        }

        if(arg.amaxD)
            hipblaslt_init_zero(hAmaxD_gold[i].buf(), 1, 1, 1, Talpha);

        if(arg.scaleE)
            hipblaslt_init(hScaleE[i].buf(), 1, 1, 1, Talpha);

        if(arg.scaleAlpha_vector)
            hipblaslt_init(hScaleAlphaVec[i].buf(), M[i], 1, M[i], Talpha);

        if(arg.gradient && arg.use_e)
        {
            CHECK_HIP_ERROR(synchronize(dE[i], hE[i], block_count));
        }
        if(!arg.gradient && arg.bias_vector)
        {
            CHECK_HIP_ERROR(synchronize(dBias[i], hBias[i], block_count));
        }

        if(arg.scaleAlpha_vector)
        {
            CHECK_HIP_ERROR(synchronize(dScaleAlphaVec[i], hScaleAlphaVec[i], block_count));
            alpha_in[i] = dScaleAlphaVec[i].buf();
            set_computeInterface(
                h_alpha[i], 1.0, Tc); // use dScaleAlphaVec instead, original alpha = 1.0 for verify
        }
        else
            alpha_in[i] = &(h_alpha[i]);

        if(arg.scaleA == hipblaslt_scaling_format::Scalar
           || arg.scaleA == hipblaslt_scaling_format::Vector)
        {
            if(arg.amaxScaleA && (arg.a_type == HIP_R_32F || arg.a_type == HIP_R_16F))
            {
                CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.a_type,
                                                       HIP_R_32F,
                                                       dScaleA[i].buf(),
                                                       dA[i].buf(),
                                                       A_row[i],
                                                       A_col[i],
                                                       stream));

                CHECK_HIP_ERROR(synchronize(hScaleA[i], dScaleA[i]));
            }
            else
                CHECK_HIP_ERROR(synchronize(dScaleA[i], hScaleA[i], block_count));
        }

        if(arg.scaleB == hipblaslt_scaling_format::Scalar
           || arg.scaleB == hipblaslt_scaling_format::Vector)
        {
            if(arg.amaxScaleB && (arg.b_type == HIP_R_32F || arg.b_type == HIP_R_16F))
            {
                CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.b_type,
                                                       HIP_R_32F,
                                                       dScaleB[i].buf(),
                                                       dB[i].buf(),
                                                       B_row[i],
                                                       B_col[i],
                                                       stream));
                CHECK_HIP_ERROR(synchronize(hScaleB[i], dScaleB[i]));
            }
            else
                CHECK_HIP_ERROR(synchronize(dScaleB[i], hScaleB[i], block_count));
        }

        if(arg.scaleC)
            CHECK_HIP_ERROR(synchronize(dScaleC[i], hScaleC[i]));

        if(arg.scaleD)
            CHECK_HIP_ERROR(synchronize(dScaleD[i], hScaleD[i]));

        if(arg.scaleE)
            CHECK_HIP_ERROR(synchronize(dScaleE[i], hScaleE[i]));

        //// copy data from CPU to device end
        if(size_D_copy[i])
        {
            if(epilogue_on[i])
            {
                transform_buf(hC[i], hD_gold_epl[i], To, Talpha);
            }
            else
            {
                copy_buf(hC[i], hD_gold[i], To);
            }
        }

        if(epilogue_on[i])
            EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                                  HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                                  &(epilogue[i]),
                                                                  sizeof(epilogue[i])),
                                  HIPBLAS_STATUS_SUCCESS);

        if(arg.use_e)
        {
            void* e_addr = dE[i].buf();
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &e_addr, sizeof(void*)));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &lde[i], sizeof(int64_t)));
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                                                &stride_e[i],
                                                sizeof(int64_t)));
        }

        if(arg.bias_vector)
        {
            const void* bias_addr;
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                &arg.bias_type,
                                                sizeof(hipDataType)),
                HIPBLAS_STATUS_SUCCESS);
            bias_addr = dBias[i].buf();

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(
                    matmul[0][i], HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_addr, sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);
        }

        if(arg.scaleA == hipblaslt_scaling_format::Scalar
           || arg.scaleA == hipblaslt_scaling_format::Vector)
        {
            hipblasLtMatmulDescAttributes_t attr
                = arg.scaleA == hipblaslt_scaling_format::Vector
                      ? HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT
                      : HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
            void* scaleA_addr = (void*)(dScaleA[i].buf());
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescSetAttribute(matmul[0][i], attr, &scaleA_addr, sizeof(void*)));
        }

        if(arg.scaleB == hipblaslt_scaling_format::Scalar
           || arg.scaleB == hipblaslt_scaling_format::Vector)
        {
            hipblasLtMatmulDescAttributes_t attr
                = arg.scaleB == hipblaslt_scaling_format::Vector
                      ? HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT
                      : HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
            void* scaleB_addr = (void*)(dScaleB[i].buf());
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescSetAttribute(matmul[0][i], attr, &scaleB_addr, sizeof(void*)));
        }

        if(arg.scaleC)
        {
            void* scaleC_addr = dScaleC[i].buf();
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, &scaleC_addr, sizeof(void*)));
        }

        if(arg.scaleD)
        {
            void* scaleD_addr = dScaleD[i].buf();
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scaleD_addr, sizeof(void*)));
        }

        if(arg.amaxD)
        {
            void* amaxD_addr = dAmaxD[i].buf();
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amaxD_addr, sizeof(void*)));
        }

        if(arg.scaleE)
        {
            void* scaleE_addr = dScaleE[i].buf();
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
                                                &scaleE_addr,
                                                sizeof(void*)));
        }

        if(arg.scaleAlpha_vector)
        {
            hipblasLtPointerMode_t scale_mode
                = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                                &scale_mode,
                                                sizeof(scale_mode)),
                HIPBLAS_STATUS_SUCCESS);
        }

        for(int32_t b = 1; b < matmul.size(); b++)
        {
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescCreate(&(matmul[b][i]), arg.compute_type, arg.scale_type));
            CHECK_HIPBLASLT_ERROR(hipblaslt_ext::copyMatmul(matmul[0][i], matmul[b][i]));

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                                &TciA,
                                                sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                                &TciB,
                                                sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);

            // Update bias, E
            if(arg.bias_vector)
            {
                const void* bias_addr = (const void*)(dBias[i].as<char>()
                                                      + b * size_bias[i] * realDataTypeSize(Tbias));
                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                    HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias_addr,
                                                    sizeof(void*)),
                    HIPBLAS_STATUS_SUCCESS);
            }
            if(arg.use_e)
            {
                void* e_addr = (void*)(dE[i].as<char>() + b * size_E[i] * realDataTypeSize(To));
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                    &e_addr,
                                                    sizeof(void*)));
            }
            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                hipblasLtMatmulDescAttributes_t attr
                    = arg.scaleA == hipblaslt_scaling_format::Vector
                          ? HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT
                          : HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
                void* scaleA_addr = (void*)(dScaleA[i].as<char>() + b * size_scaleAVec[i]);
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[b][i], attr, &scaleA_addr, sizeof(void*)));
            }

            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                hipblasLtMatmulDescAttributes_t attr
                    = arg.scaleB == hipblaslt_scaling_format::Vector
                          ? HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT
                          : HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
                void* scaleB_addr = (void*)(dScaleB[i].as<char>() + b * size_scaleBVec[i]);
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[b][i], attr, &scaleB_addr, sizeof(void*)));
            }
        }
    }

    // set preference
    size_t                     max_workspace_size = arg.user_allocated_workspace;
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)),
        HIPBLAS_STATUS_SUCCESS);

    // set workspace
    device_vector<unsigned char>* dWorkspace     = nullptr;
    size_t                        workspace_size = 0;

    // set user args
    hipblaslt_ext::UserArguments* userArgs   = nullptr;
    hipblaslt_ext::UserArguments* d_userArgs = nullptr;

    // Get Heuristic results
    int32_t requestAlgoCount = arg.requested_solution_num < 0 ? HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM
                                                              : arg.requested_solution_num;
    int     returnedAlgoCount = 0;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    std::vector<size_t>                           heuristicTuningIndex;

    // Cpp API
    hipblaslt_ext::GemmPreferenceV2 gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    std::vector<hipblaslt_ext::Gemm>                      gemmVec;
    std::vector<hipblaslt_ext::GroupedGemm>               groupedGemmVec;
    std::vector<std::vector<hipblaslt_ext::GemmInputsV2>> extinputs;

    // C to Cpp API for GG
    std::vector<std::vector<void*>> da(block_count, std::vector<void*>(gemm_count));
    std::vector<std::vector<void*>> db(block_count, std::vector<void*>(gemm_count));
    std::vector<std::vector<void*>> dc(block_count, std::vector<void*>(gemm_count));
    std::vector<std::vector<void*>> dd(block_count, std::vector<void*>(gemm_count));

    for(int32_t b = 0; b < block_count; b++)
    {
        if(!do_grouped_gemm)
            gemmVec.push_back(hipblaslt_ext::Gemm(handle,
                                                  transA,
                                                  transB,
                                                  arg.a_type,
                                                  arg.b_type,
                                                  arg.c_type,
                                                  arg.d_type,
                                                  arg.compute_type));
        else
            groupedGemmVec.push_back(hipblaslt_ext::GroupedGemm(handle,
                                                                transA,
                                                                transB,
                                                                arg.a_type,
                                                                arg.b_type,
                                                                arg.c_type,
                                                                arg.d_type,
                                                                arg.compute_type));
    }

    std::vector<hipblaslt_ext::GemmEpilogueV2> extepilogue;
    hipblaslt_ext::GemmProblemTypeV2           extproblemtype;
    if(arg.use_ext_setproblem)
    {
        extinputs.resize(block_count, std::vector<hipblaslt_ext::GemmInputsV2>(gemm_count));
        extepilogue.resize(gemm_count);

        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            auto  bias_type = HIPBLASLT_DATATYPE_INVALID;
            void* bias_addr = nullptr;
            for(int32_t b = 0; b < block_count; b++)
            {
                if(arg.bias_vector)
                {
                    bias_type = arg.bias_type;
                    bias_addr = (void*)(dBias[gemmIdx].as<char>()
                                        + b * size_bias[gemmIdx] * realDataTypeSize(bias_type));
                }
                if(b == 0)
                {
                    extepilogue[gemmIdx].setMode(epilogue[gemmIdx]);
                    extepilogue[gemmIdx].setBiasDataType(bias_type);
                    extepilogue[gemmIdx].setAuxLeadingDimension(lde[gemmIdx]);
                    extepilogue[gemmIdx].setAuxBatchStride(stride_e[gemmIdx]);
                    extepilogue[gemmIdx].setScalingAType(
                        arg.scaleA == hipblaslt_scaling_format::Vector ? 1 : 0);
                    extepilogue[gemmIdx].setScalingBType(
                        arg.scaleB == hipblaslt_scaling_format::Vector ? 1 : 0);
                }
                extinputs[b][gemmIdx].setA((void*)((dA[gemmIdx].as<char>())
                                                   + b * size_dA[gemmIdx] * realDataTypeSize(TiA)));
                extinputs[b][gemmIdx].setB((void*)((dB[gemmIdx].as<char>())
                                                   + b * size_B[gemmIdx] * realDataTypeSize(TiB)));
                extinputs[b][gemmIdx].setC(
                    (void*)((dC[gemmIdx].as<char>()) + b * size_C[gemmIdx] * realDataTypeSize(To)));
                extinputs[b][gemmIdx].setD((void*)(((*dDp)[gemmIdx].as<char>())
                                                   + b * size_D[gemmIdx] * realDataTypeSize(To)));
                extinputs[b][gemmIdx].setAlpha(&h_alpha[gemmIdx]);
                extinputs[b][gemmIdx].setBeta(&h_beta[gemmIdx]);
                extinputs[b][gemmIdx].setBias(bias_addr);
                extinputs[b][gemmIdx].setScaleA(
                    (arg.scaleA == hipblaslt_scaling_format::Scalar
                     || arg.scaleA == hipblaslt_scaling_format::Vector)
                        ? (void*)((dScaleA[gemmIdx].as<char>()) + b * size_scaleAVec[gemmIdx])
                        : nullptr);
                extinputs[b][gemmIdx].setScaleB(
                    (arg.scaleB == hipblaslt_scaling_format::Scalar
                     || arg.scaleB == hipblaslt_scaling_format::Vector)
                        ? (void*)((dScaleB[gemmIdx].as<char>()) + b * size_scaleBVec[gemmIdx])
                        : nullptr);
                extinputs[b][gemmIdx].setScaleC(arg.scaleC ? dScaleC[gemmIdx].as<char>() : nullptr);
                extinputs[b][gemmIdx].setScaleD(arg.scaleD ? dScaleD[gemmIdx].as<char>() : nullptr);
                extinputs[b][gemmIdx].setScaleAux(arg.scaleE ? dScaleE[gemmIdx].as<char>()
                                                             : nullptr);
                extinputs[b][gemmIdx].setAmaxD(arg.amaxD ? dAmaxD[gemmIdx].as<char>() : nullptr);
                if(arg.use_e)
                    extinputs[b][gemmIdx].setAux(
                        (void*)((dE[gemmIdx].as<char>())
                                + b * size_E[gemmIdx] * realDataTypeSize(To)));
                if(arg.scaleAlpha_vector)
                    extinputs[b][gemmIdx].setScaleAlphaVec(
                        (void*)((dScaleAlphaVec[gemmIdx].as<char>())
                                + b * size_scaleAlphaVec[gemmIdx] * realDataTypeSize(Talpha)));
            }
        }
        extproblemtype.setOpA(transA);
        extproblemtype.setOpB(transB);
        extproblemtype.setTypeA(arg.a_type);
        extproblemtype.setTypeB(arg.b_type);
        extproblemtype.setTypeC(arg.c_type);
        extproblemtype.setTypeD(arg.d_type);
        extproblemtype.setTypeCompute(arg.compute_type);
    }
    else if(arg.grouped_gemm)
    {
        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            for(int32_t b = 0; b < block_count; b++)
            {
                da[b][gemmIdx] = (void*)((dA[gemmIdx].as<char>())
                                         + b * size_dA[gemmIdx] * realDataTypeSize(TiA));
                db[b][gemmIdx] = (void*)((dB[gemmIdx].as<char>())
                                         + b * size_B[gemmIdx] * realDataTypeSize(TiB));
                dc[b][gemmIdx] = (void*)((dC[gemmIdx].as<char>())
                                         + b * size_C[gemmIdx] * realDataTypeSize(To));
                dd[b][gemmIdx] = (void*)(((*dDp)[gemmIdx].as<char>())
                                         + b * size_D[gemmIdx] * realDataTypeSize(To));
            }
        }
    }

    hipblaslt_ext::GemmType gemmType = do_grouped_gemm
                                           ? hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM
                                           : hipblaslt_ext::GemmType::HIPBLASLT_GEMM;

    // Remove duplicate
    std::vector<uint32_t> gsu_vector;
    std::vector<uint32_t> wgm_vector;
    for(int32_t i = 0; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        if(arg.gsu_vector[i] == -1)
            break;
        gsu_vector.push_back(arg.gsu_vector[i]);
    }
    for(int32_t i = 0; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        if(arg.wgm_vector[i] == -1)
            break;
        wgm_vector.push_back(arg.wgm_vector[i]);
    }
    std::set<uint32_t> remove_duplicate(gsu_vector.begin(), gsu_vector.end());
    gsu_vector.assign(remove_duplicate.begin(), remove_duplicate.end());
    remove_duplicate = std::set<uint32_t>(wgm_vector.begin(), wgm_vector.end());
    wgm_vector.assign(remove_duplicate.begin(), remove_duplicate.end());
    std::vector<hipblaslt_ext::GemmTuning> tuningVec;
    if(arg.use_ext)
    {
        for(size_t wgm = 0; wgm < wgm_vector.size(); wgm++)
            for(size_t gsu = 0; gsu < gsu_vector.size(); gsu++)
            {
                hipblaslt_ext::GemmTuning tuning;
                tuning.splitK = gsu_vector[gsu];
                tuning.wgm    = wgm_vector[wgm];
                tuningVec.push_back(tuning);
            }
    }
    else
    {
        // C API does not support
        tuningVec.push_back(hipblaslt_ext::GemmTuning());
    }

    if(arg.algo_method == 2)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
        heuristicResult.clear();
        heuristicTuningIndex.clear();

        int algoIndexCount = 0;
        int algoIndexInc   = 100;
        while(1)
        {
            std::vector<int>                              algoIndex;
            std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
            bool                                          foundAlgo = false;
            if(arg.solution_index == -1)
            {
                // Get algos by index
                // In real cases, the user can use the saved algo index to get the algorithm.
                // isAlgoSupported is not necessary if the user is sure that the algo supports the problem.
                algoIndex.resize(algoIndexInc);
                std::iota(std::begin(algoIndex), std::end(algoIndex), algoIndexCount);
                algoIndexCount += algoIndexInc;
            }
            else
            {
                // Specify the index
                algoIndex.resize(1);
                algoIndex[0] = arg.solution_index;
            }
            if(HIPBLAS_STATUS_INVALID_VALUE
               == hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, tmpAlgo))
            {
                break;
            }
            returnedAlgoCount = tmpAlgo.size();

            if(!do_grouped_gemm)
            {
                if(arg.use_ext)
                {
                    if(arg.use_ext_setproblem)
                    {
                        for(int32_t b = 0; b < block_count; b++)
                            CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(M[0],
                                                                        N[0],
                                                                        K[0],
                                                                        num_batches[0],
                                                                        lda[0],
                                                                        ldb[0],
                                                                        ldc[0],
                                                                        ldd[0],
                                                                        stride_a[0],
                                                                        stride_b[0],
                                                                        stride_c[0],
                                                                        stride_d[0],
                                                                        extepilogue[0],
                                                                        extinputs[b][0],
                                                                        extproblemtype));
                    }
                    else
                    {
                        for(int32_t b = 0; b < block_count; b++)
                            CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(
                                matmul[b][0],
                                alpha_in[0],
                                (dA[0].as<char>()) + b * size_dA[0] * realDataTypeSize(TiA),
                                matA[0],
                                (dB[0].as<char>()) + b * size_B[0] * realDataTypeSize(TiB),
                                matB[0],
                                &h_beta[0],
                                (dC[0].as<char>()) + b * size_C[0] * realDataTypeSize(To),
                                matC[0],
                                ((*dDp)[0].as<char>()) + b * size_D[0] * realDataTypeSize(To),
                                matD[0]));
                    }
                    for(int j = 0; j < returnedAlgoCount; j++)
                    {
                        for(size_t t = 0; t < tuningVec.size(); t++)
                        {
                            size_t tmpWorkspaceSize = 0;
                            if(gemmVec[0].isAlgoSupported(
                                   tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                               == HIPBLAS_STATUS_SUCCESS)
                            {
                                heuristicResult.push_back(tmpAlgo[j]);
                                heuristicTuningIndex.push_back(t);
                                workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                                foundAlgo      = true;
                            }
                        }
                        if(foundAlgo)
                            break;
                    }
                }
                else
                {
                    for(int j = 0; j < returnedAlgoCount; j++)
                    {
                        for(size_t t = 0; t < 1; t++) // CAPI not supported yet
                        {
                            size_t tmpWorkspaceSize = 0;
                            if(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                                    matmul[0][0],
                                                                    alpha_in[0],
                                                                    matA[0],
                                                                    matB[0],
                                                                    &h_beta[0],
                                                                    matC[0],
                                                                    matD[0],
                                                                    tmpAlgo[j].algo,
                                                                    tmpWorkspaceSize)
                               == HIPBLAS_STATUS_SUCCESS)
                            {
                                heuristicResult.push_back(tmpAlgo[j]);
                                heuristicTuningIndex.push_back(t);
                                workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                                foundAlgo      = true;
                                break;
                            }
                        }
                    }
                }
            }
            else
            {
                if(arg.use_ext_setproblem)
                {
                    auto num_batches_64
                        = std::vector<int64_t>{num_batches.begin(), num_batches.end()};
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(M,
                                                                           N,
                                                                           K,
                                                                           num_batches_64,
                                                                           lda,
                                                                           ldb,
                                                                           ldc,
                                                                           ldd,
                                                                           stride_a,
                                                                           stride_b,
                                                                           stride_c,
                                                                           stride_d,
                                                                           extepilogue,
                                                                           extinputs[b],
                                                                           extproblemtype));
                }
                else
                {
                    std::vector<void*> h_alpha_void, h_beta_void;
                    for(size_t i = 0; i < h_alpha.size(); i++)
                    {
                        h_alpha_void.push_back(&h_alpha[i]);
                        h_beta_void.push_back(&h_beta[i]);
                    }
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(matmul[b],
                                                                           h_alpha_void,
                                                                           da[b],
                                                                           matA,
                                                                           db[b],
                                                                           matB,
                                                                           h_beta_void,
                                                                           dc[b],
                                                                           matC,
                                                                           dd[b],
                                                                           matD));
                }

                for(int j = 0; j < returnedAlgoCount; j++)
                {
                    for(size_t t = 0; t < tuningVec.size(); t++)
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(groupedGemmVec[0].isAlgoSupported(
                               tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                           == HIPBLAS_STATUS_SUCCESS)
                        {
                            heuristicResult.push_back(tmpAlgo[j]);
                            heuristicTuningIndex.push_back(t);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                            foundAlgo      = true;
                        }
                    }
                    if(foundAlgo)
                        break;
                }
            }

            if(arg.solution_index != -1)
            {
                CHECK_SOLUTION_FOUND(foundAlgo);
                foundAlgo = true;
            }
            if(foundAlgo)
            {
                break;
            }
        }
    }
    else if(arg.algo_method == 1)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
        EXPECT_HIPBLAS_STATUS(hipblaslt_ext::getAllAlgos(handle,
                                                         gemmType,
                                                         transA,
                                                         transB,
                                                         arg.a_type,
                                                         arg.b_type,
                                                         arg.c_type,
                                                         arg.d_type,
                                                         arg.compute_type,
                                                         tmpAlgo),
                              HIPBLAS_STATUS_SUCCESS);
        returnedAlgoCount = tmpAlgo.size();
        heuristicResult.clear();
        heuristicTuningIndex.clear();
        int requestCount = 0;
        if(!do_grouped_gemm)
        {
            if(arg.use_ext)
            {
                if(arg.use_ext_setproblem)
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(M[0],
                                                                    N[0],
                                                                    K[0],
                                                                    num_batches[0],
                                                                    lda[0],
                                                                    ldb[0],
                                                                    ldc[0],
                                                                    ldd[0],
                                                                    stride_a[0],
                                                                    stride_b[0],
                                                                    stride_c[0],
                                                                    stride_d[0],
                                                                    extepilogue[0],
                                                                    extinputs[b][0],
                                                                    extproblemtype));
                }
                else
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(
                            matmul[b][0],
                            alpha_in[0],
                            (dA[0].as<char>()) + b * size_dA[0] * realDataTypeSize(TiA),
                            matA[0],
                            (dB[0].as<char>()) + b * size_B[0] * realDataTypeSize(TiB),
                            matB[0],
                            &h_beta[0],
                            (dC[0].as<char>()) + b * size_C[0] * realDataTypeSize(To),
                            matC[0],
                            ((*dDp)[0].as<char>()) + b * size_D[0] * realDataTypeSize(To),
                            matD[0]));
                }
                for(int j = 0; j < returnedAlgoCount; j++)
                {
                    int addRequest = 0;
                    for(size_t t = 0; t < tuningVec.size(); t++)
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(gemmVec[0].isAlgoSupported(
                               tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                           == HIPBLAS_STATUS_SUCCESS)
                        {
                            addRequest = 1;
                            heuristicResult.push_back(tmpAlgo[j]);
                            heuristicTuningIndex.push_back(t);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                        }
                    }
                    requestCount += addRequest;
                    if(requestCount >= requestAlgoCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                for(int j = 0; j < returnedAlgoCount; j++)
                {
                    int addRequest = 0;
                    for(size_t t = 0; t < 1; t++) // C API not supported yet
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                                matmul[0][0],
                                                                alpha_in[0],
                                                                matA[0],
                                                                matB[0],
                                                                &h_beta[0],
                                                                matC[0],
                                                                matD[0],
                                                                tmpAlgo[j].algo,
                                                                tmpWorkspaceSize)
                           == HIPBLAS_STATUS_SUCCESS)
                        {
                            addRequest = 1;
                            heuristicResult.push_back(tmpAlgo[j]);
                            heuristicTuningIndex.push_back(t);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                        }
                    }
                    requestCount += addRequest;
                    if(requestCount >= requestAlgoCount)
                    {
                        break;
                    }
                }
            }
        }
        else
        {
            if(arg.use_ext_setproblem)
            {
                auto num_batches_64 = std::vector<int64_t>{num_batches.begin(), num_batches.end()};
                for(int32_t b = 0; b < block_count; b++)
                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(M,
                                                                       N,
                                                                       K,
                                                                       num_batches_64,
                                                                       lda,
                                                                       ldb,
                                                                       ldc,
                                                                       ldd,
                                                                       stride_a,
                                                                       stride_b,
                                                                       stride_c,
                                                                       stride_d,
                                                                       extepilogue,
                                                                       extinputs[b],
                                                                       extproblemtype));
            }
            else
            {
                std::vector<void*> h_alpha_void, h_beta_void;
                for(size_t i = 0; i < h_alpha.size(); i++)
                {
                    h_alpha_void.push_back(&h_alpha[i]);
                    h_beta_void.push_back(&h_beta[i]);
                }

                for(int32_t b = 0; b < block_count; b++)
                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(matmul[b],
                                                                       h_alpha_void,
                                                                       da[b],
                                                                       matA,
                                                                       db[b],
                                                                       matB,
                                                                       h_beta_void,
                                                                       dc[b],
                                                                       matC,
                                                                       dd[b],
                                                                       matD));
            }

            for(int j = 0; j < returnedAlgoCount; j++)
            {
                int    addRequest       = 0;
                size_t tmpWorkspaceSize = 0;
                for(size_t t = 0; t < tuningVec.size(); t++)
                {
                    if(groupedGemmVec[0].isAlgoSupported(
                           tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                       == HIPBLAS_STATUS_SUCCESS)
                    {
                        addRequest = 1;
                        heuristicResult.push_back(tmpAlgo[j]);
                        heuristicTuningIndex.push_back(t);
                        workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                    }
                }
                requestCount += addRequest;
                if(requestCount >= requestAlgoCount)
                {
                    break;
                }
            }
        }
    }
    else
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;

        if(!do_grouped_gemm)
        {
            if(arg.use_ext)
            {
                if(arg.use_ext_setproblem)
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(M[0],
                                                                    N[0],
                                                                    K[0],
                                                                    num_batches[0],
                                                                    lda[0],
                                                                    ldb[0],
                                                                    ldc[0],
                                                                    ldd[0],
                                                                    stride_a[0],
                                                                    stride_b[0],
                                                                    stride_c[0],
                                                                    stride_d[0],
                                                                    extepilogue[0],
                                                                    extinputs[b][0],
                                                                    extproblemtype));
                }
                else
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(
                            matmul[b][0],
                            alpha_in[0],
                            (dA[0].as<char>()) + b * size_dA[0] * realDataTypeSize(TiA),
                            matA[0],
                            (dB[0].as<char>()) + b * size_B[0] * realDataTypeSize(TiB),
                            matB[0],
                            &h_beta[0],
                            (dC[0].as<char>()) + b * size_C[0] * realDataTypeSize(To),
                            matC[0],
                            ((*dDp)[0].as<char>()) + b * size_D[0] * realDataTypeSize(To),
                            matD[0]));
                }
                CHECK_HIPBLASLT_ERROR(
                    gemmVec[0].algoGetHeuristic(requestAlgoCount, gemmPref, tmpAlgo));
                heuristicResult.clear();
                heuristicTuningIndex.clear();
                for(int j = 0; j < tmpAlgo.size(); j++)
                {
                    for(size_t t = 0; t < tuningVec.size(); t++)
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(gemmVec[0].isAlgoSupported(
                               tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                           == HIPBLAS_STATUS_SUCCESS)
                        {
                            heuristicResult.push_back(tmpAlgo[j]);
                            heuristicTuningIndex.push_back(t);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                        }
                    }
                }
                returnedAlgoCount = heuristicResult.size();
            }
            else
            {
                std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo(requestAlgoCount);
                EXPECT_HIPBLAS_STATUS((hipblasLtMatmulAlgoGetHeuristic(handle,
                                                                       matmul[0][0],
                                                                       matA[0],
                                                                       matB[0],
                                                                       matC[0],
                                                                       matD[0],
                                                                       pref,
                                                                       requestAlgoCount,
                                                                       tmpAlgo.data(),
                                                                       &returnedAlgoCount)),
                                      HIPBLAS_STATUS_SUCCESS);
                heuristicResult.clear();
                for(int32_t i = 0; i < returnedAlgoCount; i++)
                {
                    heuristicResult.push_back(tmpAlgo[i]);
                }
                heuristicTuningIndex.resize(heuristicResult.size(), 0); // C API not supported yet
            }

            for(int i = 0; i < returnedAlgoCount; i++)
                workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);
        }
        else
        {
            if(arg.use_ext_setproblem)
            {
                auto num_batches_64 = std::vector<int64_t>{num_batches.begin(), num_batches.end()};
                for(int32_t b = 0; b < block_count; b++)
                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(M,
                                                                       N,
                                                                       K,
                                                                       num_batches_64,
                                                                       lda,
                                                                       ldb,
                                                                       ldc,
                                                                       ldd,
                                                                       stride_a,
                                                                       stride_b,
                                                                       stride_c,
                                                                       stride_d,
                                                                       extepilogue,
                                                                       extinputs[b],
                                                                       extproblemtype));
            }
            else
            {
                std::vector<void*> h_alpha_void, h_beta_void;
                for(size_t i = 0; i < h_alpha.size(); i++)
                {
                    h_alpha_void.push_back(&h_alpha[i]);
                    h_beta_void.push_back(&h_beta[i]);
                }
                for(int32_t b = 0; b < block_count; b++)
                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(matmul[b],
                                                                       h_alpha_void,
                                                                       da[b],
                                                                       matA,
                                                                       db[b],
                                                                       matB,
                                                                       h_beta_void,
                                                                       dc[b],
                                                                       matC,
                                                                       dd[b],
                                                                       matD));
            }

            CHECK_HIPBLASLT_ERROR(
                groupedGemmVec[0].algoGetHeuristic(requestAlgoCount, gemmPref, tmpAlgo));
            heuristicResult.clear();
            heuristicTuningIndex.clear();
            for(int j = 0; j < tmpAlgo.size(); j++)
            {
                for(size_t t = 0; t < tuningVec.size(); t++)
                {
                    size_t tmpWorkspaceSize = 0;
                    if(groupedGemmVec[0].isAlgoSupported(
                           tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                       == HIPBLAS_STATUS_SUCCESS)
                    {
                        heuristicResult.push_back(tmpAlgo[j]);
                        heuristicTuningIndex.push_back(t);
                    }
                }
            }
            workspace_size = max_workspace_size;
        }
    }

    returnedAlgoCount = heuristicResult.size();

    if(returnedAlgoCount == 0)
    {
        int             deviceId;
        hipDeviceProp_t deviceProperties;
        static_cast<void>(hipGetDevice(&deviceId));
        static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
        //workaround before known_bug work
        if((gpu_arch_match(deviceProperties.gcnArchName, "11?")
            || gpu_arch_match(deviceProperties.gcnArchName, "12?"))
           && (arg.gradient || arg.grouped_gemm))
        {
            hipblaslt_cerr << "No Solution Found!!" << std::endl;
            return;
        }
    }

    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    dWorkspace = new device_vector<unsigned char>(workspace_size * block_count, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dWorkspace->memcheck());

    if(arg.use_user_args)
    {
        CHECK_HIP_ERROR(
            hipHostMalloc(&userArgs, gemm_count * sizeof(hipblaslt_ext::UserArguments)));
        CHECK_HIP_ERROR(hipMalloc(&d_userArgs,
                                  block_count * gemm_count * sizeof(hipblaslt_ext::UserArguments)));
    }

    auto ptrs = benchmark_allocation();

    if(arg.print_solution_found)
        hipblaslt_cout << "Is supported " << heuristicResult.size()
                       << " / Total solutions: " << returnedAlgoCount * tuningVec.size()
                       << std::endl;

    if(heuristicResult.size() != heuristicTuningIndex.size())
    {
        hipblaslt_cerr << "Internal error, heuristicResult.size() != heuristicTuningIndex.size() "
                       << heuristicResult.size() << " != " << heuristicTuningIndex.size()
                       << std::endl;
        exit(EXIT_FAILURE);
    }

    // get CPU result
    if(arg.unit_check || arg.norm_check || arg.allclose_check)
    {
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        // For the xf32 xdl math op, cast type of A/B from float to xfloat32 .
        if(TiA == HIP_R_32F && TiB == HIP_R_32F && To == HIP_R_32F && Talpha == HIP_R_32F)
            if(arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_TF32)
                for(int i = 0; i < gemm_count; i++)
                {
                    type_to_xdl_math_op_type<hipblasLtXfloat32, float, float>(
                        static_cast<float*>(hA[i].buf()), size_A[i]);
                    type_to_xdl_math_op_type<hipblasLtXfloat32, float, float>(
                        static_cast<float*>(hB[i].buf()), size_B[i]);
                }

#define epilogue_param                                                                             \
    M[gemmIdx], N[gemmIdx], ldd[gemmIdx],                                                          \
        (hD_gold_epl[gemmIdx].as<char>() + pos * realDataTypeSize(Talpha)),                        \
        (hD_gold[gemmIdx].as<char>() + pos * realDataTypeSize(To)),                                \
        (hBias_gold_epl[gemmIdx].as<char>() + pos * realDataTypeSize(Talpha)),                     \
        arg.amaxD ? hAmaxD_gold[gemmIdx].as<char>() + 0 : nullptr, ePos, scaleDValue, scaleEValue, \
        applyBias
        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            auto                 alpha    = h_alpha[gemmIdx];
            auto                 betaTemp = h_beta[gemmIdx];
            computeTypeInterface tempSC;
            if(arg.scaleC)
            {
                // betaTemp *= hScaleC[gemmIdx][0];
                set_computeInterface(tempSC, hScaleC[gemmIdx].buf(), Tc);
                mul_computeInterface(betaTemp, tempSC, Tc);
            }

            computeTypeInterface scale;
            set_computeInterface(scale, 1, Talpha);
            void* scaleAVec   = (arg.scaleA == hipblaslt_scaling_format::Scalar
                               || arg.scaleA == hipblaslt_scaling_format::Vector)
                                    ? hScaleA[gemmIdx].buf()
                                    : (void*)(&scale);
            void* scaleBVec   = (arg.scaleB == hipblaslt_scaling_format::Scalar
                               || arg.scaleB == hipblaslt_scaling_format::Vector)
                                    ? hScaleB[gemmIdx].buf()
                                    : (void*)(&scale);
            void* scaleDValue = arg.scaleD ? hScaleD[gemmIdx].buf() : (void*)(&scale);
            void* scaleEValue = arg.scaleE ? hScaleE[gemmIdx].buf() : (void*)(&scale);

            for(int batchIdx = 0; batchIdx < num_batches[gemmIdx]; batchIdx++)
            {
                if(epilogue_on[gemmIdx])
                {
                    cblas_gemm(transA,
                               transB,
                               M[gemmIdx],
                               N[gemmIdx],
                               K[gemmIdx],
                               alpha,
                               hA[gemmIdx].as<char>()
                                   + stride_a[gemmIdx] * batchIdx * realDataTypeSize(TiA),
                               lda[gemmIdx],
                               hB[gemmIdx].as<char>()
                                   + stride_b[gemmIdx] * batchIdx * realDataTypeSize(TiB),
                               ldb[gemmIdx],
                               betaTemp,
                               hD_gold_epl[gemmIdx].as<char>()
                                   + stride_d[gemmIdx] * batchIdx * realDataTypeSize(Talpha),
                               ldd[gemmIdx],
                               arg.scaleAlpha_vector ? hScaleAlphaVec[gemmIdx].as<char>() + 0
                                                     : nullptr,
                               scaleAVec,
                               scaleBVec,
                               (void*)(&scale),
                               (arg.scaleA == hipblaslt_scaling_format::Vector),
                               (arg.scaleB == hipblaslt_scaling_format::Vector),
                               TiA,
                               TiB,
                               Tc,
                               Tc,
                               TciA,
                               TciB,
                               false);
                    auto                        pos       = stride_d[gemmIdx] * batchIdx;
                    std::vector<HipHostBuffer>* hEInst    = arg.gradient ? &hE : &hE_gold;
                    void*                       ePos      = ((*hEInst).size() <= gemmIdx)
                                                                ? nullptr
                                                                : ((*hEInst)[gemmIdx].as<char>() + pos * realDataTypeSize(To));
                    auto                        applyBias = arg.gradient ? false : arg.bias_vector;
                    void* hBias_buf = ((hBias).size() <= gemmIdx) ? nullptr : hBias[gemmIdx].buf();

                    switch(arg.activation_type)
                    {
                    case hipblaslt_activation_type::gelu:
                        if(arg.gradient)
                            epilogue_func(epilogue_param,
                                          hBias_buf,
                                          Tbias,
                                          arg.activation_arg1,
                                          arg.activation_arg2,
                                          ::_dgelu,
                                          true,
                                          To,
                                          Talpha);
                        else
                        {
                            epilogue_func(epilogue_param,
                                          hBias_buf,
                                          Tbias,
                                          arg.activation_arg1,
                                          arg.activation_arg2,
                                          ::_gelu,
                                          false,
                                          To,
                                          Talpha);
                        }
                        break;
                    case hipblaslt_activation_type::relu:
                        epilogue_func(epilogue_param,
                                      hBias_buf,
                                      Tbias,
                                      arg.activation_arg1,
                                      arg.activation_arg2,
                                      ::_relu,
                                      arg.gradient,
                                      To,
                                      Talpha);
                        break;
                    case hipblaslt_activation_type::swish:
                        epilogue_func(epilogue_param,
                                      hBias_buf,
                                      Tbias,
                                      arg.activation_arg1,
                                      arg.activation_arg2,
                                      ::_silu,
                                      arg.gradient,
                                      To,
                                      Talpha);
                        break;
                    default:
                        epilogue_func(epilogue_param, hBias_buf, Tbias, false, To, Talpha);
                        break;
                    }
                    if(arg.gradient && arg.bias_vector && batchIdx == num_batches[gemmIdx] - 1)
                    {
                        if(arg.bias_source == hipblaslt_bias_source::d)
                        {
                            reduction_func<false, float>(hBias_gold_epl[gemmIdx].as<char>()
                                                             + pos * realDataTypeSize(Talpha),
                                                         Talpha,
                                                         hBias_gold[gemmIdx].buf(),
                                                         Tbias,
                                                         M[gemmIdx],
                                                         N[gemmIdx],
                                                         1,
                                                         ldd[gemmIdx],
                                                         stride_d[gemmIdx],
                                                         num_batches[gemmIdx]);
                        }
                        else
                        {
                            bool sumLd = false;
                            int  s1 = 1, s2 = 1, s3 = 1;
                            auto reduc = [&sumLd,
                                          &s1,
                                          &s2,
                                          &s3,
                                          &hBias_gold,
                                          &Tbias,
                                          &size_bias,
                                          &K,
                                          &num_batches,
                                          &gemmIdx,
                                          &arg](void* ptr, hipDataType Ti) {
                                if(sumLd)
                                {
                                    reduction_func<true, float>(ptr,
                                                                Ti,
                                                                hBias_gold[gemmIdx].buf(),
                                                                Tbias,
                                                                size_bias[gemmIdx],
                                                                K[gemmIdx],
                                                                s1,
                                                                s2,
                                                                s3,
                                                                num_batches[gemmIdx]);
                                }
                                else
                                {
                                    reduction_func<false, float>(ptr,
                                                                 Ti,
                                                                 hBias_gold[gemmIdx].buf(),
                                                                 Tbias,
                                                                 size_bias[gemmIdx],
                                                                 K[gemmIdx],
                                                                 s1,
                                                                 s2,
                                                                 s3,
                                                                 num_batches[gemmIdx]);
                                }
                            };
                            if(arg.bias_source == hipblaslt_bias_source::a)
                            {
                                void* ptr = hA[gemmIdx].buf();
                                s2        = lda[gemmIdx];
                                s3        = stride_a[gemmIdx];
                                sumLd     = transA == HIPBLAS_OP_N ? false : true;
                                reduc(ptr, TiA);
                            }
                            else if(arg.bias_source == hipblaslt_bias_source::b)
                            {
                                void* ptr = hB[gemmIdx].buf();
                                s2        = ldb[gemmIdx];
                                s3        = stride_b[gemmIdx];
                                sumLd     = transB == HIPBLAS_OP_N ? true : false;
                                reduc(ptr, TiB);
                            }
                        }
                    }
                }
                else
                {
                    cblas_gemm(transA,
                               transB,
                               M[gemmIdx],
                               N[gemmIdx],
                               K[gemmIdx],
                               alpha,
                               hA[gemmIdx].as<char>()
                                   + stride_a[gemmIdx] * batchIdx * realDataTypeSize(TiA),
                               lda[gemmIdx],
                               hB[gemmIdx].as<char>()
                                   + stride_b[gemmIdx] * batchIdx * realDataTypeSize(TiB),
                               ldb[gemmIdx],
                               betaTemp,
                               hD_gold[gemmIdx].as<char>()
                                   + stride_d[gemmIdx] * batchIdx * realDataTypeSize(To),
                               ldd[gemmIdx],
                               nullptr,
                               scaleAVec,
                               scaleBVec,
                               scaleDValue,
                               (arg.scaleA == hipblaslt_scaling_format::Vector),
                               (arg.scaleB == hipblaslt_scaling_format::Vector),
                               TiA,
                               TiB,
                               To,
                               Tc,
                               TciA,
                               TciB,
                               false);
                }
            }
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }
    }

    if(!arg.timing)
    {
        for(size_t sol = 0; sol < heuristicResult.size(); sol++)
        {
            if((arg.unit_check || arg.norm_check || arg.allclose_check) && arg.c_equal_d)
            {
                for(int i = 0; i < gemm_count; i++)
                {
                    CHECK_HIP_ERROR(synchronize(dC[i], hC[i], block_count));
                }
            }
            if(!do_grouped_gemm)
            {
                if(arg.use_ext)
                {
                    CHECK_HIPBLASLT_ERROR(
                        gemmVec[0].initialize(heuristicResult[sol].algo,
                                              tuningVec[heuristicTuningIndex[sol]],
                                              *dWorkspace));
                    CHECK_HIPBLASLT_ERROR(gemmVec[0].run(stream));
                }
                else
                {
                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                          matmul[0][0],
                                                          alpha_in[0],
                                                          dA[0].buf(),
                                                          matA[0],
                                                          dB[0].buf(),
                                                          matB[0],
                                                          &(h_beta[0]),
                                                          dC[0].buf(),
                                                          matC[0],
                                                          (*dDp)[0].buf(),
                                                          matD[0],
                                                          &heuristicResult[sol].algo,
                                                          *dWorkspace,
                                                          workspace_size,
                                                          stream),
                                          HIPBLAS_STATUS_SUCCESS);
                }
            }
            else
            {
                //grouped gemm
                if(arg.use_user_args)
                {
                    CHECK_HIPBLASLT_ERROR(
                        groupedGemmVec[0].initialize(heuristicResult[sol].algo,
                                                     tuningVec[heuristicTuningIndex[0]],
                                                     *dWorkspace));
                    groupedGemmVec[0].getDefaultValueForDeviceUserArguments(userArgs);
                    // Copy them to device memory
                    CHECK_HIP_ERROR(hipMemcpy(d_userArgs,
                                              userArgs,
                                              gemm_count * sizeof(hipblaslt_ext::UserArguments),
                                              hipMemcpyHostToDevice));

                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[0].run(d_userArgs, stream));
                }
                else
                {
                    CHECK_HIPBLASLT_ERROR(
                        groupedGemmVec[0].initialize(heuristicResult[sol].algo,
                                                     tuningVec[heuristicTuningIndex[0]],
                                                     *dWorkspace,
                                                     false,
                                                     stream));

                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[0].run(stream));
                }
            }

            double              hipblaslt_error = 0.0;
            double              hipblaslt_atol  = 1;
            double              hipblaslt_rtol  = 1;
            std::vector<double> tol(gemm_count);
            if(arg.unit_check && (hipblaslt_get_arch_major() == 11) && realDataTypeSize(TiA) == 2
               && realDataTypeSize(TiB) == 2)
            {
                for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
                {
                    tol[gemmIdx] = K[gemmIdx] * sum_error_tolerance_for_gfx11_type(Tc, TiA, To);
                }
            }
            if(arg.unit_check || arg.norm_check || arg.allclose_check)
            {
                copy_gemm_to_host(stream, gemm_count, hD_1, (*dDp));
                check(stream,
                      arg,
                      gemm_count,
                      M,
                      N,
                      ldd,
                      lde,
                      stride_d,
                      stride_e,
                      num_batches,
                      size_bias,
                      hD_gold,
                      hD_1,
                      (*dDp),
                      hAmaxD_gold,
                      hAmaxD,
                      dAmaxD,
                      hE_gold,
                      hE,
                      dE,
                      hBias_gold,
                      hBias,
                      dBias,
                      tol,
                      hipblaslt_error,
                      hipblaslt_atol,
                      hipblaslt_rtol,
                      To,
                      Tbias,
                      Talpha);
            }
        }
    }
    else
    {
        // Get device information
        hipDeviceProp_t deviceProps;
        CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProps, 0));
        int32_t gpu_block3 = deviceProps.multiProcessorCount * 60;

        size_t      best_sol       = -1;
        double      best_flops     = 0.0;
        double      best_gpu_time  = std::numeric_limits<double>::max();
        double      best_warm_time = std::numeric_limits<double>::max();
        std::string best_s_name    = "";
        std::string best_k_name    = "";
        double      best_norm      = 0.0;
        double      best_atol      = 0.0;
        double      best_rtol      = 0.0;
        int         number_cold_calls
            = ((arg.unit_check || arg.norm_check || arg.allclose_check) && arg.cold_iters == 0)
                  ? 1
                  : arg.cold_iters;
        int number_hot_calls = arg.iters;

        int    flush_iter      = 100000;
        double flush_time_used = 0;
        if(arg.flush)
        {
            for(int i = 0; i < flush_iter; i++)
                hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);

            if(arg.use_gpu_timer)
                CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
            else
            {
                flush_time_used = get_time_us_sync(stream);
            }
            for(int i = 0; i < flush_iter; i++)
                hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
            if(arg.use_gpu_timer)
            {
                CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
                CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
                float gpu_time_ms;
                CHECK_HIP_ERROR(
                    hipEventElapsedTime(&gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
                flush_time_used = gpu_time_ms * 1000; // ms to us
            }
            else
            {
                flush_time_used = get_time_us_sync(stream) - flush_time_used;
            }
            flush_time_used /= flush_iter;
        }

        for(size_t sol = 0; sol < heuristicResult.size(); sol++)
        {
            if((arg.unit_check || arg.norm_check || arg.allclose_check) && arg.c_equal_d)
            {
                for(int i = 0; i < gemm_count; i++)
                {
                    CHECK_HIP_ERROR(synchronize(dC[i], hC[i], block_count));
                }
            }
            if(!do_grouped_gemm)
            {
                FrequencyMonitor& freq_monitor = getFrequencyMonitor();
                if(arg.use_ext)
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(
                            gemmVec[b].initialize(heuristicResult[sol].algo,
                                                  tuningVec[heuristicTuningIndex[sol]],
                                                  *dWorkspace));
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(gemmVec[i % block_count].run(stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, gemm_count, hD_1, (*dDp));
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    freq_monitor.start();
                    pre_gpu_time(arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);

                    for(int i = 0; i < number_hot_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(gemmVec[i % block_count].run(stream));
                        if(arg.flush)
                            hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                    }
                }
                else
                {
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        auto ptr_matmul = matmul[i % block_count][0];
                        auto ptr_alpha  = arg.scaleAlpha_vector
                                              ? (dScaleAlphaVec[0].as<char>())
                                                   + (i % block_count) * size_scaleAlphaVec[0]
                                              : alpha_in[0];
                        EXPECT_HIPBLAS_STATUS(
                            hipblasLtMatmul(
                                handle,
                                ptr_matmul,
                                ptr_alpha,
                                dA[0].as<char>()
                                    + (i % block_count) * size_dA[0] * realDataTypeSize(TiA),
                                matA[0],
                                dB[0].as<char>()
                                    + (i % block_count) * size_B[0] * realDataTypeSize(TiB),
                                matB[0],
                                &(h_beta[0]),
                                dC[0].as<char>()
                                    + (i % block_count) * size_C[0] * realDataTypeSize(To),
                                matC[0],
                                (*dDp)[0].as<char>()
                                    + (i % block_count) * size_D[0] * realDataTypeSize(To),
                                matD[0],
                                &heuristicResult[sol].algo,
                                *dWorkspace,
                                workspace_size,
                                stream),
                            HIPBLAS_STATUS_SUCCESS);
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, gemm_count, hD_1, (*dDp));
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    freq_monitor.start();
                    pre_gpu_time(arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_hot_calls; i++)
                    {
                        auto ptr_matmul = matmul[i % block_count][0];
                        auto ptr_alpha  = arg.scaleAlpha_vector
                                              ? (dScaleAlphaVec[0].as<char>())
                                                   + (i % block_count) * size_scaleAlphaVec[0]
                                              : alpha_in[0];
                        EXPECT_HIPBLAS_STATUS(
                            hipblasLtMatmul(
                                handle,
                                ptr_matmul,
                                ptr_alpha,
                                dA[0].as<char>()
                                    + (i % block_count) * size_dA[0] * realDataTypeSize(TiA),
                                matA[0],
                                dB[0].as<char>()
                                    + (i % block_count) * size_B[0] * realDataTypeSize(TiB),
                                matB[0],
                                &(h_beta[0]),
                                dC[0].as<char>()
                                    + (i % block_count) * size_C[0] * realDataTypeSize(To),
                                matC[0],
                                (*dDp)[0].as<char>()
                                    + (i % block_count) * size_D[0] * realDataTypeSize(To),
                                matD[0],
                                &heuristicResult[sol].algo,
                                *dWorkspace,
                                workspace_size,
                                stream),
                            HIPBLAS_STATUS_SUCCESS);
                        if(arg.flush)
                            hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                    }
                }
                post_gpu_time(arg.use_gpu_timer,
                              event_gpu_time_start,
                              event_gpu_time_end,
                              gpu_time_used,
                              stream);
                freq_monitor.stop();
            }
            else
            {
                FrequencyMonitor& freq_monitor = getFrequencyMonitor();
                if(arg.use_user_args)
                {
                    std::vector<unsigned char*> d_userArgsVec(block_count);
                    //grouped gemm
                    for(int32_t b = 0; b < block_count; b++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].initialize(
                            heuristicResult[sol].algo,
                            tuningVec[heuristicTuningIndex[sol]],
                            ((unsigned char*)(*dWorkspace) + b * workspace_size)));
                        groupedGemmVec[b].getDefaultValueForDeviceUserArguments(userArgs);
                        d_userArgsVec[b] = (unsigned char*)d_userArgs
                                           + b * gemm_count * sizeof(hipblaslt_ext::UserArguments);
                        // Copy them to device memory
                        CHECK_HIP_ERROR(hipMemcpy(d_userArgsVec[b],
                                                  userArgs,
                                                  gemm_count * sizeof(hipblaslt_ext::UserArguments),
                                                  hipMemcpyHostToDevice));
                    }
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(
                            d_userArgsVec[i % block_count], stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, gemm_count, hD_1, (*dDp));
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    freq_monitor.start();
                    pre_gpu_time(arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);

                    for(int i = 0; i < number_hot_calls; i++)
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(
                            d_userArgsVec[i % block_count], stream));

                    post_gpu_time(arg.use_gpu_timer,
                                  event_gpu_time_start,
                                  event_gpu_time_end,
                                  gpu_time_used,
                                  stream);
                    freq_monitor.stop();
                }
                else
                {
                    //grouped gemm
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].initialize(
                            heuristicResult[sol].algo,
                            tuningVec[heuristicTuningIndex[sol]],
                            ((unsigned char*)(*dWorkspace) + b * workspace_size),
                            false,
                            stream));

                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, gemm_count, hD_1, (*dDp));
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    freq_monitor.start();
                    pre_gpu_time(arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);

                    for(int i = 0; i < number_hot_calls; i++)
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(stream));

                    post_gpu_time(arg.use_gpu_timer,
                                  event_gpu_time_start,
                                  event_gpu_time_end,
                                  gpu_time_used,
                                  stream);
                    freq_monitor.stop();
                }
            }

            double flops = 0;
            for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
            {
                flops += gemm_gflop_count(M[gemmIdx], N[gemmIdx], K[gemmIdx], Talpha);
                switch(arg.activation_type)
                {
                case hipblaslt_activation_type::relu:
                    flops += relu_gflop_count(M[gemmIdx], N[gemmIdx], Talpha);
                    break;
                case hipblaslt_activation_type::gelu:
                    flops += gelu_gflop_count(M[gemmIdx], N[gemmIdx], Talpha);
                    break;
                case hipblaslt_activation_type::swish:
                    flops += silu_gflop_count(M[gemmIdx], N[gemmIdx], Talpha);
                    break;
                default:
                    break;
                }
            }

            double              hipblaslt_error = 0.0;
            double              hipblaslt_atol  = 1;
            double              hipblaslt_rtol  = 1;
            std::vector<double> tol(gemm_count);
            if(arg.unit_check && (hipblaslt_get_arch_major() == 11) && realDataTypeSize(TiA) == 2
               && realDataTypeSize(TiB) == 2)
            {
                for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
                {
                    tol[gemmIdx] = K[gemmIdx] * sum_error_tolerance_for_gfx11_type(Tc, TiA, To);
                }
            }
            if(arg.unit_check || arg.norm_check || arg.allclose_check)
            {
                check(stream,
                      arg,
                      gemm_count,
                      M,
                      N,
                      ldd,
                      lde,
                      stride_d,
                      stride_e,
                      num_batches,
                      size_bias,
                      hD_gold,
                      hD_1,
                      (*dDp),
                      hAmaxD_gold,
                      hAmaxD,
                      dAmaxD,
                      hE_gold,
                      hE,
                      dE,
                      hBias_gold,
                      hBias,
                      dBias,
                      tol,
                      hipblaslt_error,
                      hipblaslt_atol,
                      hipblaslt_rtol,
                      To,
                      Tbias,
                      Talpha);
            }

#define argument_param                                                                            \
    e_transA, e_transB, e_grouped_gemm, e_batch_count, e_M, e_N, e_K, e_alpha, e_lda, e_stride_a, \
        e_beta, e_ldb, e_stride_b, e_ldc, e_stride_c, e_ldd, e_stride_d, e_a_type, e_b_type,      \
        e_c_type, e_d_type, e_compute_type, e_scaleA, e_scaleB, e_scaleC, e_scaleD, e_amaxD,      \
        e_activation_type, e_bias_vector, e_bias_type, e_rotating

            const char* tuningEnv     = getenv("HIPBLASLT_TUNING_FILE");
            int32_t     solutionIndex = ((tuningEnv && heuristicResult.size() == 1)
                                     || (arg.print_solution_found && arg.print_kernel_info))
                                            ? hipblaslt_ext::getIndexFromAlgo(heuristicResult[sol].algo)
                                            : -1;
            std::string solutionName  = "";
            std::string kernelName    = "";
            std::string archName      = "";
            std::string cuNum         = "";

            if(tuningEnv && heuristicResult.size() == 1)
            {
                archName = deviceProps.gcnArchName;
                cuNum    = std::to_string(deviceProps.multiProcessorCount);
            }

            if(arg.print_solution_found)
            {
                if(arg.print_kernel_info)
                {
                    if(arg.use_ext)
                    {
                        if(!do_grouped_gemm)
                        {
                            solutionName = gemmVec[0].getSolutionName();
                            kernelName   = gemmVec[0].getKernelName();
                        }
                        else
                        {
                            solutionName = groupedGemmVec[0].getSolutionName();
                            kernelName   = groupedGemmVec[0].getKernelName();
                        }
                    }
                    else
                    {
                        solutionName = hipblaslt_ext::getSolutionNameFromAlgo(
                            handle, heuristicResult[sol].algo);
                        kernelName = hipblaslt_ext::getKernelNameFromAlgo(
                            handle, heuristicResult[sol].algo);
                    }
                }
                ArgumentModel<argument_param>{}.log_args(
                    Talpha,
                    hipblaslt_cout,
                    sol,
                    solutionIndex,
                    solutionName,
                    kernelName,
                    archName,
                    cuNum,
                    arg,
                    (uint32_t)tuningVec[heuristicTuningIndex[sol]].splitK,
                    (uint32_t)tuningVec[heuristicTuningIndex[sol]].wgm,
                    gpu_time_used,
                    flush_time_used,
                    flops,
                    gpu_mem_gbytes,
                    cpu_time_used,
                    hipblaslt_error,
                    hipblaslt_atol,
                    hipblaslt_rtol);
            }
            if(best_gpu_time > gpu_time_used)
            {
                best_sol      = sol;
                best_flops    = flops;
                best_gpu_time = gpu_time_used;
                best_s_name   = solutionName;
                best_k_name   = kernelName;
                best_norm     = hipblaslt_error;
                best_atol     = hipblaslt_atol;
                best_rtol     = hipblaslt_rtol;
            }
        }

        if(heuristicResult.size() > 1)
        {
            const char* tuningEnv = getenv("HIPBLASLT_TUNING_FILE");
            int32_t     solutionIndex
                = (tuningEnv || arg.print_kernel_info)
                      ? hipblaslt_ext::getIndexFromAlgo(heuristicResult[best_sol].algo)
                      : -1;
            std::string solutionName = "";
            std::string kernelName   = "";
            std::string archName     = "";
            std::string cuNum        = "";
            if(tuningEnv)
            {
                archName = deviceProps.gcnArchName;
                cuNum    = std::to_string(deviceProps.multiProcessorCount);
            }

            if(arg.print_kernel_info)
            {
                solutionName = best_s_name;
                kernelName   = best_k_name;
            }

            hipblaslt_cout << "Winner: " << std::endl;
            ArgumentModel<argument_param>{}.log_args(
                Talpha,
                hipblaslt_cout,
                best_sol,
                solutionIndex,
                solutionName,
                kernelName,
                archName,
                cuNum,
                arg,
                (uint32_t)tuningVec[heuristicTuningIndex[best_sol]].splitK,
                (uint32_t)tuningVec[heuristicTuningIndex[best_sol]].wgm,
                best_gpu_time,
                flush_time_used,
                best_flops,
                gpu_mem_gbytes,
                cpu_time_used,
                best_norm,
                best_atol,
                best_rtol);
        }
    }

    for(auto it : ptrs)
    {
        CHECK_HIP_ERROR(hipFree(it));
    }

    if(dWorkspace != nullptr)
        delete dWorkspace;
    if(userArgs != nullptr)
        CHECK_HIP_ERROR(hipFree(userArgs));
    if(d_userArgs != nullptr)
        CHECK_HIP_ERROR(hipFree(d_userArgs));

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_end));
}
