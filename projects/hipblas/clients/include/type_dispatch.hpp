/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#ifndef _HIPBLAS_TYPE_DISPATCH_
#define _HIPBLAS_TYPE_DISPATCH_
#include "hipblas.hpp"
#include "utility.h"

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
auto hipblas_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
    case HIPBLAS_R_16F:
        return TEST<hipblasHalf>{}(arg);
    case HIPBLAS_R_16B:
        return TEST<hipblasBfloat16>{}(arg);
    case HIPBLAS_R_32F:
        return TEST<float>{}(arg);
    case HIPBLAS_R_64F:
        return TEST<double>{}(arg);
    //  case hipblas_datatype_f16_c:
    //      return TEST<hipblas_half_complex>{}(arg);
    case HIPBLAS_C_32F:
        return TEST<hipblasComplex>{}(arg);
    case HIPBLAS_C_64F:
        return TEST<hipblasDoubleComplex>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

// BLAS1 functions
template <template <typename...> class TEST>
auto hipblas_blas1_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, Tb = arg.b_type, To = arg.d_type;
    if(Ti == To)
    {
        if(Tb == Ti)
            return hipblas_simple_dispatch<TEST>(arg);
        else
        { // for csscal and zdscal and complex rotg only
            if(Ti == HIPBLAS_C_32F && Tb == HIPBLAS_R_32F)
                return TEST<hipblasComplex, float>{}(arg);
            else if(Ti == HIPBLAS_C_64F && Tb == HIPBLAS_R_64F)
                return TEST<hipblasDoubleComplex, double>{}(arg);
        }
    }
    else if(Ti == HIPBLAS_C_32F && Tb == HIPBLAS_R_32F)
        return TEST<hipblasComplex, float>{}(arg);
    else if(Ti == HIPBLAS_C_64F && Tb == HIPBLAS_R_64F)
        return TEST<hipblasDoubleComplex, double>{}(arg);
    else if(Ti == HIPBLAS_R_32F && Tb == HIPBLAS_R_32F)
        return TEST<float, float>{}(arg);
    else if(Ti == HIPBLAS_R_64F && Tb == HIPBLAS_R_64F)
        return TEST<double, double>{}(arg);
    //  else if(Ti == hipblas_datatype_f16_c && To == HIPBLAS_R_16F)
    //      return TEST<hipblas_half_complex, hipblasHalf>{}(arg);

    return TEST<void>{}(arg);
}

// BLAS1_ex functions
// TODO: Update this when adding these functions to hipblas-bench
template <template <typename...> class TEST>
auto hipblas_blas1_ex_dispatch(const Arguments& arg)
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
        return hipblas_simple_dispatch<TEST>(arg); // Ta == Tx == Ty == Tex
    }
    else if(is_scal && Ta == Tx && Tx == Tex)
    {
        // hscal with f16_r compute (scal doesn't care about Ty)
        return hipblas_simple_dispatch<TEST>(arg);
    }
    else if((is_rot || is_dot || is_axpy) && Ta == Tx && Tx == Ty && Ta == HIPBLAS_R_16F
            && Tex == HIPBLAS_R_32F)
    {
        return TEST<hipblasHalf, hipblasHalf, hipblasHalf, float>{}(arg);
    }
    else if((is_rot || is_dot || is_axpy) && Ta == Tx && Tx == Ty && Ta == HIPBLAS_R_16B
            && Tex == HIPBLAS_R_32F)
    {
        return TEST<hipblasBfloat16, hipblasBfloat16, hipblasBfloat16, float>{}(arg);
    }
    else if(is_axpy && Ta == Tex && Tx == Ty && Tx == HIPBLAS_R_16F && Tex == HIPBLAS_R_32F)
    {
        return TEST<float, hipblasHalf, hipblasHalf, float>{}(arg);
    }
    else if((is_scal || is_nrm2) && Ta == Tx && Ta == HIPBLAS_R_16F && Tex == HIPBLAS_R_32F)
    {
        // half scal, nrm2, axpy
        return TEST<hipblasHalf, hipblasHalf, float>{}(arg);
    }
    else if((is_scal || is_nrm2) && Ta == Tx && Ta == HIPBLAS_R_16B && Tex == HIPBLAS_R_32F)
    {
        // bfloat16 scal, nrm2
        return TEST<hipblasBfloat16, hipblasBfloat16, float>{}(arg);
    }
    else if(is_axpy && Ta == Tex && Tx == Ty && (Tx == HIPBLAS_R_16B || Tx == HIPBLAS_R_16F)
            && Tex == HIPBLAS_R_32F)
    {
        // axpy bfloat16 with float alpha
        return TEST<float, hipblasBfloat16, hipblasBfloat16, float>{}(arg);
    }
    // exclusive functions cases
    else if(is_scal)
    {
        // scal_ex ordering: <alphaType, dataType, exType> opposite order of scal test
        if(Ta == Tex && Tx == HIPBLAS_R_16B && Tex == HIPBLAS_R_32F)
        {
            // scal bfloat16 with float alpha
            return TEST<float, hipblasBfloat16, float>{}(arg);
        }
        else if(Ta == HIPBLAS_R_32F && Tx == HIPBLAS_R_16F && Tex == HIPBLAS_R_32F)
        {
            // scal half with float alpha
            return TEST<float, hipblasHalf, float>{}(arg);
        }
        else if(Ta == HIPBLAS_R_32F && Tx == HIPBLAS_C_32F && Tex == HIPBLAS_C_32F)
        {
            // csscal-like
            return TEST<float, hipblasComplex, hipblasComplex>{}(arg);
        }
        else if(Ta == HIPBLAS_R_64F && Tx == HIPBLAS_C_64F && Tex == HIPBLAS_C_64F)
        {
            // zdscal-like
            return TEST<double, hipblasDoubleComplex, hipblasDoubleComplex>{}(arg);
        }
    }
    else if(is_nrm2)
    {
        if(Ta == HIPBLAS_C_32F && Tx == HIPBLAS_R_32F && Tex == HIPBLAS_R_32F)
        {
            // scnrm2
            return TEST<hipblasComplex, float, float>{}(arg);
        }
        else if(Ta == HIPBLAS_C_64F && Tx == HIPBLAS_R_64F && Tex == HIPBLAS_R_64F)
        {
            // dznrm2
            return TEST<hipblasDoubleComplex, double, double>{}(arg);
        }
    }
    else if(is_rot)
    {
        if(Ta == HIPBLAS_C_32F && Tx == HIPBLAS_C_32F && Ty == HIPBLAS_R_32F
           && Tex == HIPBLAS_C_32F)
        {
            // rot with complex x/y/compute and real cs
            return TEST<hipblasComplex, hipblasComplex, float, hipblasComplex>{}(arg);
        }
        else if(Ta == HIPBLAS_C_64F && Tx == HIPBLAS_C_64F && Ty == HIPBLAS_R_64F
                && Tex == HIPBLAS_C_64F)
        {
            // rot with complex x/y/compute and real cs
            return TEST<hipblasDoubleComplex, hipblasDoubleComplex, double, hipblasDoubleComplex>{}(
                arg);
        }
    }

    return TEST<void>{}(arg);
}

// rot
// giving rot it's own dispatch function so the code is easier to follow
template <template <typename...> class TEST>
auto hipblas_rot_dispatch(const Arguments& arg)
{
    const auto Ta = arg.a_type, Tb = arg.b_type, Tc = arg.c_type;
    if(Ta == Tb && Tb == Tc)
    {
        // srot, drot
        return hipblas_simple_dispatch<TEST>(arg);
    }
    else if(Ta == HIPBLAS_C_32F && Tb == HIPBLAS_R_32F && Tc == Tb)
    {
        // csrot
        return TEST<hipblasComplex, float, float>{}(arg);
    }
    else if(Ta == HIPBLAS_C_64F && Tb == HIPBLAS_R_64F && Tc == Tb)
    {
        // zdrot
        return TEST<hipblasDoubleComplex, double, double>{}(arg);
    }
    else if(Ta == HIPBLAS_C_32F && Tb == HIPBLAS_R_32F && Tc == Ta)
    {
        // crot
        return TEST<hipblasComplex, float, hipblasComplex>{}(arg);
    }
    else if(Ta == HIPBLAS_C_64F && Tb == HIPBLAS_R_64F && Tc == Ta)
    {
        // zrot
        return TEST<hipblasDoubleComplex, double, hipblasDoubleComplex>{}(arg);
    }

    return TEST<void>{}(arg);
}

// gemm functions
template <template <typename...> class TEST>
auto hipblas_gemm_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type, Tc = arg.compute_type;

    if(arg.b_type == Ti && arg.d_type == To)
    {
        if(Ti != To)
        {
            if(Ti == HIPBLAS_R_8I && To == HIPBLAS_R_32I && Tc == To)
                return TEST<int8_t, int32_t, int32_t>{}(arg);
        }
        else if(Tc != To)
        {
            if(To == HIPBLAS_R_16F && Tc == HIPBLAS_R_32F)
            {
                return TEST<hipblasHalf, hipblasHalf, float>{}(arg);
            }
            else if(To == HIPBLAS_R_16B && Tc == HIPBLAS_R_32F)
            {
                return TEST<hipblasBfloat16, hipblasBfloat16, float>{}(arg);
            }
        }
        else
        {
            return hipblas_simple_dispatch<TEST>(arg); // Ti = To = Tc
        }
    }
    return TEST<void>{}(arg);
}

#endif
