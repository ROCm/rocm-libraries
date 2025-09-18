/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "rocblas.h"
#include "rocblas_arguments.hpp"

#ifdef WIN32
#define setenv(A, B, C) _putenv_s(A, B)
#define unsetenv(A) _putenv_s(A, "")
#endif

template <typename T>
constexpr auto rocblas_type2datatype()
{
    // rocblas_datatype_f16_r  = 150
    if(std::is_same_v<T, rocblas_half>)
        return rocblas_datatype_f16_r;
    // rocblas_datatype_f32_r  = 151
    if(std::is_same_v<T, rocblas_float>)
        return rocblas_datatype_f32_r;
    // rocblas_datatype_f64_r  = 152
    if(std::is_same_v<T, rocblas_double>)
        return rocblas_datatype_f64_r;
    // rocblas_datatype_f16_c  = 153
    // rocblas_datatype_f32_c  = 154
    if(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_datatype_f32_c;
    // rocblas_datatype_f64_c  = 155
    if(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_datatype_f64_c;
    // rocblas_datatype_i8_r   = 160
    if(std::is_same_v<T, int8_t>)
        return rocblas_datatype_i8_r;
    // rocblas_datatype_u8_r   = 161
    if(std::is_same_v<T, uint8_t>)
        return rocblas_datatype_u8_r;
    // rocblas_datatype_i32_r  = 162
    if(std::is_same_v<T, int32_t>)
        return rocblas_datatype_i32_r;
    // rocblas_datatype_u32_r  = 163
    if(std::is_same_v<T, uint32_t>)
        return rocblas_datatype_u32_r;
    // rocblas_datatype_i8_c   = 164
    // rocblas_datatype_u8_c   = 165
    // rocblas_datatype_i32_c  = 166
    // rocblas_datatype_u32_c  = 167
    // rocblas_datatype_bf16_r = 168
    if(std::is_same_v<T, rocblas_bfloat16>)
        return rocblas_datatype_bf16_r;
    // rocblas_datatype_bf16_c = 169

    return rocblas_datatype_f32_r; // testing purposes we default to f32 ex
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
auto rocblas_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
    case rocblas_datatype_f16_r:
        return TEST<rocblas_half>{}(arg);
    case rocblas_datatype_bf16_r:
        return TEST<rocblas_bfloat16>{}(arg);
    case rocblas_datatype_f32_r:
        return TEST<float>{}(arg);
    case rocblas_datatype_f64_r:
        return TEST<double>{}(arg);
    // case rocblas_datatype_f16_c:
    //     return TEST<rocblas_half_complex>{}(arg);
    case rocblas_datatype_f32_c:
        return TEST<rocblas_float_complex>{}(arg);
    case rocblas_datatype_f64_c:
        return TEST<rocblas_double_complex>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

// BLAS1 functions
template <template <typename...> class TEST>
auto rocblas_blas1_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, Tb = arg.b_type, To = arg.d_type;
    const auto Tc = arg.c_type;

    if(strstr(arg.function, "iamax") || strstr(arg.function, "iamin")
       || strstr(arg.function, "asum") || strstr(arg.function, "nrm2")
       || strstr(arg.function, "swap"))
    {
        // s, d, c, z precisions
        if(Ti == To && Ti == Tb && Ti == Tc)
            if(Ti == rocblas_datatype_f32_r || Ti == rocblas_datatype_f64_r
               || Ti == rocblas_datatype_f32_c || Ti == rocblas_datatype_f64_c)
                return rocblas_simple_dispatch<TEST>(arg);
    }
    else if(strstr(arg.function, "rotm") || strstr(arg.function, "rotmg"))
    {
        // s, d precisions
        if(Ti == To && Ti == Tb && Ti == Tc)
            if(Ti == rocblas_datatype_f32_r || Ti == rocblas_datatype_f64_r)
                return rocblas_simple_dispatch<TEST>(arg);
    }
    else if(strstr(arg.function, "axpy") || strstr(arg.function, "copy"))
    {
        // h, s, d, c, z precisions
        if(Ti == To && Ti == Tb && Ti == Tc)
            if(Ti == rocblas_datatype_f32_r || Ti == rocblas_datatype_f64_r
               || Ti == rocblas_datatype_f32_c || Ti == rocblas_datatype_f64_c
               || Ti == rocblas_datatype_f16_r)
                return rocblas_simple_dispatch<TEST>(arg);
    }
    else if(strstr(arg.function, "dot"))
    {
        // h, bf, s, d, c, z precisions
        if(Ti == To && Ti == Tb && Ti == Tc)
        {
            // h, s, d, c, z
            if(Ti == rocblas_datatype_f32_r || Ti == rocblas_datatype_f64_r
               || Ti == rocblas_datatype_f32_c || Ti == rocblas_datatype_f64_c
               || Ti == rocblas_datatype_f16_r || Ti == rocblas_datatype_bf16_r)
                return rocblas_simple_dispatch<TEST>(arg);
        }
        else if(Ti == To && Ti == Tb)
        {
            // bf
            if(Ti == rocblas_datatype_bf16_r && Tc == rocblas_datatype_f32_r)
                return rocblas_simple_dispatch<TEST>(arg);
        }
    }
    else if(strstr(arg.function, "rotg"))
    {
        if(Ti == To && Ti == Tb && Ti == Tc)
        {
            // s, d
            if(Ti == rocblas_datatype_f32_r || Ti == rocblas_datatype_f64_r)
                return rocblas_simple_dispatch<TEST>(arg);
        }
        else if(Ti == rocblas_datatype_f32_c)
        {
            // c
            return TEST<rocblas_float_complex, float>{}(arg);
        }
        else if(Ti == rocblas_datatype_f64_c)
        {
            // z
            return TEST<rocblas_double_complex, double>{}(arg);
        }
        else
        {
            rocblas_cout << "no dispatch for rotg" << std::endl;
        }
    }
    else if(strstr(arg.function, "rot"))
    {
        // s, d, c, z, cs, zd precisions
        if(Ti == To && Ti == Tb && Ti == Tc)
        {
            // s, d
            if(Ti == rocblas_datatype_f32_r || Ti == rocblas_datatype_f64_r)
                return rocblas_simple_dispatch<TEST>(arg);
        }
        else if(Ti == rocblas_datatype_f32_c && Tb == rocblas_datatype_f32_r
                && Tc == rocblas_datatype_f32_c)
        {
            // c
            return TEST<rocblas_float_complex, float, rocblas_float_complex>{}(arg);
        }
        else if(Ti == rocblas_datatype_f64_c && Tb == rocblas_datatype_f64_r
                && Tc == rocblas_datatype_f64_c)
        {
            // z
            return TEST<rocblas_double_complex, double, rocblas_double_complex>{}(arg);
        }
        else if(Ti == rocblas_datatype_f32_c && Tb == rocblas_datatype_f32_r
                && Tc == rocblas_datatype_f32_r)
        {
            // cs
            return TEST<rocblas_float_complex, float>{}(arg);
        }
        else if(Ti == rocblas_datatype_f64_c && Tb == rocblas_datatype_f64_r
                && Tc == rocblas_datatype_f64_r)
        {
            // zd
            return TEST<rocblas_double_complex, double>{}(arg);
        }
    }
    else if(strstr(arg.function, "scal"))
    {
        // s, d, c, cs, z, zd
        if(Ti == To && Ti == Tb)
        {
            // z, d, c, z
            if(Ti == rocblas_datatype_f32_r || Ti == rocblas_datatype_f64_r
               || Ti == rocblas_datatype_f32_c || Ti == rocblas_datatype_f64_c)
                return rocblas_simple_dispatch<TEST>(arg);
        }
        else if(Ti == rocblas_datatype_f32_c && Tb == rocblas_datatype_f32_r)
        {
            // cs
            return TEST<rocblas_float_complex, float>{}(arg);
        }
        else if(Ti == rocblas_datatype_f64_c && Tb == rocblas_datatype_f64_r)
        {
            // zd
            return TEST<rocblas_double_complex, double>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

// BLAS1_ex functions
template <template <typename...> class TEST>
auto rocblas_blas1_ex_dispatch(const Arguments& arg)
{
    const auto        Ta = arg.a_type, Tx = arg.b_type, Ty = arg.c_type, Tex = arg.compute_type;
    const std::string function = arg.function;
    const bool        is_axpy  = function == "axpy_ex" || function == "axpy_batched_ex"
                         || function == "axpy_strided_batched_ex";
    const bool is_dot = function == "dot_ex" || function == "dot_batched_ex"
                        || function == "dot_strided_batched_ex" || function == "dotc_ex"
                        || function == "dotc_batched_ex" || function == "dotc_strided_batched_ex";
    const bool is_nrm2 = function == "nrm2_ex" || function == "nrm2_batched_ex"
                         || function == "nrm2_strided_batched_ex";
    const bool is_rot = function == "rot_ex" || function == "rot_batched_ex"
                        || function == "rot_strided_batched_ex";
    const bool is_scal = function == "scal_ex" || function == "scal_batched_ex"
                         || function == "scal_strided_batched_ex";

    if(Ta == Tx && Tx == Ty && Ty == Tex)
    {
        return rocblas_simple_dispatch<TEST>(arg); // Ta == Tx == Ty == Tex
    }
    else if(is_scal && Ta == Tx && Tx == Tex)
    {
        // hscal with f16_r compute (scal doesn't care about Ty)
        return rocblas_simple_dispatch<TEST>(arg);
    }
    else if((is_rot || is_dot || is_axpy) && Ta == Tx && Tx == Ty && Ta == rocblas_datatype_f16_r
            && Tex == rocblas_datatype_f32_r)
    {
        return TEST<rocblas_half, rocblas_half, rocblas_half, float>{}(arg);
    }
    else if(is_axpy && Ta == Tex && Tx == Ty && Tx == rocblas_datatype_f16_r
            && Tex == rocblas_datatype_f32_r)
    {
        return TEST<float, rocblas_half, rocblas_half, float>{}(arg);
    }
    else if((is_scal || is_nrm2) && Ta == Tx && Ta == rocblas_datatype_f16_r
            && Tex == rocblas_datatype_f32_r)
    {
        // half scal, nrm2
        return TEST<rocblas_half, rocblas_half, float>{}(arg);
    }
    else if((is_rot || is_dot || is_axpy) && Ta == Tx && Tx == Ty && Ta == rocblas_datatype_bf16_r
            && Tex == rocblas_datatype_f32_r)
    {
        return TEST<rocblas_bfloat16, rocblas_bfloat16, rocblas_bfloat16, float>{}(arg);
    }
    else if((is_scal || is_nrm2) && Ta == Tx && Ta == rocblas_datatype_bf16_r
            && Tex == rocblas_datatype_f32_r)
    {
        // bfloat16 scal, nrm2
        return TEST<rocblas_bfloat16, rocblas_bfloat16, float>{}(arg);
    }
    else if(is_axpy && Ta == Tex && Tx == Ty && Tx == rocblas_datatype_bf16_r
            && Tex == rocblas_datatype_f32_r)
    {
        // axpy bfloat16 with float alpha
        return TEST<float, rocblas_bfloat16, rocblas_bfloat16, float>{}(arg);
    }

    // exclusive functions cases
    else if(is_scal)
    {
        // scal_ex ordering: <alphaType, dataType, exType> opposite order of scal test

        if(Ta == rocblas_datatype_f32_r && Tx == rocblas_datatype_f16_r
           && Tex == rocblas_datatype_f32_r)
        {
            // scal half with float alpha
            return TEST<float, rocblas_half, float>{}(arg);
        }
        else if(Ta == Tex && Tx == rocblas_datatype_bf16_r && Tex == rocblas_datatype_f32_r)
        {
            // scal bfloat16 with float alpha
            return TEST<float, rocblas_bfloat16, float>{}(arg);
        }
        else if(Ta == rocblas_datatype_f32_r && Tx == rocblas_datatype_f32_c
                && Tex == rocblas_datatype_f32_c)
        {
            // csscal-like
            return TEST<float, rocblas_float_complex, rocblas_float_complex>{}(arg);
        }
        else if(Ta == rocblas_datatype_f64_r && Tx == rocblas_datatype_f64_c
                && Tex == rocblas_datatype_f64_c)
        {
            // zdscal-like
            return TEST<double, rocblas_double_complex, rocblas_double_complex>{}(arg);
        }
    }
    else if(is_nrm2)
    {
        if(Ta == rocblas_datatype_f32_c && Tx == rocblas_datatype_f32_r
           && Tex == rocblas_datatype_f32_r)
        {
            // scnrm2
            return TEST<rocblas_float_complex, float, float>{}(arg);
        }
        else if(Ta == rocblas_datatype_f64_c && Tx == rocblas_datatype_f64_r
                && Tex == rocblas_datatype_f64_r)
        {
            // dznrm2
            return TEST<rocblas_double_complex, double, double>{}(arg);
        }
    }
    else if(is_rot)
    {
        if(Ta == rocblas_datatype_f32_c && Tx == rocblas_datatype_f32_c
           && Ty == rocblas_datatype_f32_r && Tex == rocblas_datatype_f32_c)
        {
            // rot with complex x/y/compute and real cs
            return TEST<rocblas_float_complex,
                        rocblas_float_complex,
                        float,
                        rocblas_float_complex>{}(arg);
        }
        else if(Ta == rocblas_datatype_f64_c && Tx == rocblas_datatype_f64_c
                && Ty == rocblas_datatype_f64_r && Tex == rocblas_datatype_f64_c)
        {
            // rot with complex x/y/compute and real cs
            return TEST<rocblas_double_complex,
                        rocblas_double_complex,
                        double,
                        rocblas_double_complex>{}(arg);
        }
    }
    else if(is_dot)
    {
        if(Ta == rocblas_datatype_f32_r && Tx == rocblas_datatype_f32_r
           && Ty == rocblas_datatype_f64_r && Tex == rocblas_datatype_f64_r)
        {
            return TEST<float, float, double, double>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

// gemv_batched and gev_strided_batched functions
template <template <typename...> class TEST>
auto rocblas_gemv_batched_and_strided_batched_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type, Tex = arg.compute_type;

    // covers HSH, HSS, TST, TSS precisions for gemv_batched and gev_strided_batched
    if(Ti != Tex && (Ti == rocblas_datatype_f16_r || Ti == rocblas_datatype_bf16_r)
       && (To == Ti || To == rocblas_datatype_f32_r))
    {
        if(Ti == rocblas_datatype_f16_r && Tex == rocblas_datatype_f32_r && To == Ti)
        {
            return TEST<rocblas_half, float, rocblas_half>{}(arg);
        }
        else if(Ti == rocblas_datatype_f16_r && Tex == rocblas_datatype_f32_r && To == Tex)
        {
            return TEST<rocblas_half, float, float>{}(arg);
        }
        else if(Ti == rocblas_datatype_bf16_r && Tex == rocblas_datatype_f32_r && To == Ti)
        {
            return TEST<rocblas_bfloat16, float, rocblas_bfloat16>{}(arg);
        }
        else if(Ti == rocblas_datatype_bf16_r && Tex == rocblas_datatype_f32_r && To == Tex)
        {
            return TEST<rocblas_bfloat16, float, float>{}(arg);
        }
    }
    // covers s, d, c, z precisions for gemv_batched and gev_strided_batched
    else
    {
        return rocblas_simple_dispatch<TEST>(arg); // Ti = Tex = To
    }
    return TEST<void, void, void>{}(arg);
}

// gemm functions
template <template <typename...> class TEST>
auto rocblas_gemm_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type, Tc = arg.compute_type;

    if((rocblas_gemm_flags_fp16_alt_impl & arg.flags)
       && rocblas_internal_get_arch_name() != "gfx90a")
        return TEST<void>{}(arg);

    rocblas_math_mode math_mode = rocblas_math_mode(arg.math_mode);
    if(math_mode != rocblas_default_math
       && !rocblas_internal_tensile_supports_xdl_math_op(math_mode))
        return TEST<void>{}(arg);

    if(arg.b_type == Ti && arg.d_type == To)
    {
        if(Ti != To) // covers HPA: HSS, BSS, I8II
        {
            if(Ti == rocblas_datatype_i8_r && To == rocblas_datatype_i32_r && Tc == To)
            {
                return TEST<int8_t, int32_t, int32_t>{}(arg);
            }
            else if(To == rocblas_datatype_f32_r && Tc == rocblas_datatype_f32_r)
            {
                if(Ti == rocblas_datatype_f16_r)
                {
                    return TEST<rocblas_half, float, float>{}(arg);
                }
                else if(Ti == rocblas_datatype_bf16_r)
                {
                    return TEST<rocblas_bfloat16, float, float>{}(arg);
                }
            }
        }
        else if(Tc != To) // covers HPA: HHS, BBS // Ti=To
        {
            if(To == rocblas_datatype_f16_r && Tc == rocblas_datatype_f32_r) // HHS
            {
                return TEST<rocblas_half, rocblas_half, float>{}(arg);
            }
            else if(To == rocblas_datatype_bf16_r && Tc == rocblas_datatype_f32_r) // BBS
            {
                return TEST<rocblas_bfloat16, rocblas_bfloat16, float>{}(arg);
            }
        }
        else // covers non-HPA: dgemm, sgemm, zgemm, cgemm, hgemm
        {
            return rocblas_simple_dispatch<TEST>(arg); // Ti = To = Tc
        }
    }
    return TEST<void>{}(arg);
}
