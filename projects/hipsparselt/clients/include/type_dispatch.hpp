/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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

#include "hipsparselt_arguments.hpp"
#include <hipsparselt/hipsparselt.h>

template <typename T>
constexpr auto hipsparselt_type2datatype()
{
    if(std::is_same<T, __half>{})
        return HIP_R_16F;
    if(std::is_same<T, hip_bfloat16>{})
        return HIP_R_16BF;
    if(std::is_same<T, char>{})
        return HIP_R_8I;
#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
    if(std::is_same<T, hipsparselt_fp8_e4m3>{})
        return HIP_R_8F_E4M3;
    if(std::is_same<T, hipsparselt_fp8_e5m2>{})
        return HIP_R_8F_E5M2;
#endif
    return HIP_R_16F; // testing purposes we default to f32 ex
}

// ----------------------------------------------------------------------------
// Calls TEST template based on the argument types. TEST<> is expected to
// return a functor which takes a const Arguments& argument. If the types do
// not match a recognized type combination, then TEST<void> is called.  This
// function returns the same type as TEST<...>{}(arg), usually bool or void.
// ----------------------------------------------------------------------------

// Simple functions which take only one datatype
//
// Even if the function can take mixed datatypes, this function can handle the
// cases where the types are uniform, in which case one template type argument
// is passed to TEST, and the rest are assumed to match the first.
template <template <typename...> class TEST>
auto hipsparselt_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
    case HIP_R_16F:
        return TEST<__half>{}(arg);
    case HIP_R_16BF:
        return TEST<hip_bfloat16>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

// gemm functions
template <template <typename...> class TEST>
auto hipsparselt_spmm_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type;
    auto       Tc    = arg.compute_type;
    auto       TBias = arg.bias_type;

    if(arg.b_type == Ti && arg.d_type == To)
    {
        if(Ti == To && To == HIP_R_16F && Tc == HIPSPARSELT_COMPUTE_32F)
        {
            switch(TBias)
            {
            case HIP_R_16F:
                return TEST<__half, __half, float, __half>{}(arg);
            case HIP_R_32F:
                return TEST<__half, __half, float, float>{}(arg);
            default:
                break;
            }
        }
        else if(Ti == To && To == HIP_R_16BF && Tc == HIPSPARSELT_COMPUTE_32F)
        {
            switch(TBias)
            {
            case HIP_R_16BF:
                return TEST<hip_bfloat16, hip_bfloat16, float, hip_bfloat16>{}(arg);
            case HIP_R_32F:
                return TEST<hip_bfloat16, hip_bfloat16, float, float>{}(arg);
            default:
                break;
            }
        }
        if(Ti == To && To == HIP_R_16F && Tc == HIPSPARSELT_COMPUTE_16F && TBias == HIP_R_16F)
        {
            return TEST<__half, __half, __half, __half>{}(arg);
        }
        else if(Ti == To && To == HIP_R_16BF && Tc == HIPSPARSELT_COMPUTE_16F
                && TBias == HIP_R_16BF)
        {
            return TEST<hip_bfloat16, hip_bfloat16, hip_bfloat16, hip_bfloat16>{}(arg);
        }
        else if(Ti == To && To == HIP_R_8I && Tc == HIPSPARSELT_COMPUTE_32I && TBias == HIP_R_32F)
        {
            return TEST<int8_t, int8_t, int32_t, float>{}(arg);
        }
        else if(Ti == HIP_R_8I && To == HIP_R_16F && Tc == HIPSPARSELT_COMPUTE_32I
                && TBias == HIP_R_32F)
        {
            return TEST<int8_t, __half, int32_t, float>{}(arg);
        }
        else if(Ti == HIP_R_8I && To == HIP_R_16BF && Tc == HIPSPARSELT_COMPUTE_32I
                && TBias == HIP_R_32F)
        {
            return TEST<int8_t, hip_bfloat16, int32_t, float>{}(arg);
        }
#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
        else if(Ti == HIP_R_8F_E4M3 && To == HIP_R_32F && Tc == HIPSPARSELT_COMPUTE_32F
                && TBias == HIP_R_32F)
        {
            return TEST<hipsparselt_fp8_e4m3, float, float, float>{}(arg);
        }
        else if(Ti == HIP_R_8F_E5M2 && To == HIP_R_32F && Tc == HIPSPARSELT_COMPUTE_32F
                && TBias == HIP_R_32F)
        {
            return TEST<hipsparselt_fp8_e5m2, float, float, float>{}(arg);
        }
#ifdef __HIP_PLATFORM_NVIDIA__
        else if(Ti == HIP_R_8F_E4M3 && To == HIP_R_16F && Tc == HIPSPARSELT_COMPUTE_32F
                && TBias == HIP_R_16F)
        {
            return TEST<hipsparselt_fp8_e4m3, __half, float, __half>{}(arg);
        }
        else if(Ti == HIP_R_8F_E4M3 && To == HIP_R_16BF && Tc == HIPSPARSELT_COMPUTE_32F
                && TBias == HIP_R_16BF)
        {
            return TEST<hipsparselt_fp8_e4m3, hip_bfloat16, float, hip_bfloat16>{}(arg);
        }
        else if(Ti == HIP_R_8F_E5M2 && To == HIP_R_16F && Tc == HIPSPARSELT_COMPUTE_32F
                && TBias == HIP_R_16F)
        {
            return TEST<hipsparselt_fp8_e5m2, __half, float, __half>{}(arg);
        }
        else if(Ti == HIP_R_8F_E5M2 && To == HIP_R_16BF && Tc == HIPSPARSELT_COMPUTE_32F
                && TBias == HIP_R_16BF)
        {
            return TEST<hipsparselt_fp8_e5m2, hip_bfloat16, float, hip_bfloat16>{}(arg);
        }
#endif
#endif
    }
    return TEST<void>{}(arg);
}
