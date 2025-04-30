// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once
#ifndef ROCFFT_AGAINST_FFTW
#define ROCFFT_AGAINST_FFTW

#include <math.h>
#include <stdexcept>
#include <vector>

#include "fftw_transform.h"

// Return the precision enum for rocFFT based upon the type.
template <typename Tfloat>
inline fft_precision precision_selector();
template <>
inline fft_precision precision_selector<float>()
{
    return fft_precision_single;
}
template <>
inline fft_precision precision_selector<double>()
{
    return fft_precision_double;
}

extern bool use_fftw_wisdom;

// construct and return an FFTW plan with the specified type,
// precision, and dimensions.  cpu_out is required if we're using
// wisdom, which runs actual FFTs to work out the best plan.
template <typename Tfloat>
static typename fftw_trait<Tfloat>::fftw_plan_type
    fftw_plan_with_precision(const std::vector<fftw_iodim64>& dims,
                             const std::vector<fftw_iodim64>& howmany_dims,
                             const fft_transform_type         transformType,
                             const size_t                     isize,
                             void*                            cpu_in,
                             void*                            cpu_out)
{
    using fftw_complex_type = typename fftw_trait<Tfloat>::fftw_complex_type;

    // NB: Using FFTW_MEASURE implies that the input buffer's data
    // may be destroyed during plan creation.  But if we're wanting
    // to run FFTW in the first place, we must have just created an
    // uninitialized input buffer anyway.

    switch(transformType)
    {
    case fft_transform_type_complex_forward:
        return fftw_plan_guru64_dft<Tfloat>(dims.size(),
                                            dims.data(),
                                            howmany_dims.size(),
                                            howmany_dims.data(),
                                            reinterpret_cast<fftw_complex_type*>(cpu_in),
                                            reinterpret_cast<fftw_complex_type*>(cpu_out),
                                            -1,
                                            use_fftw_wisdom ? FFTW_MEASURE : FFTW_ESTIMATE);
    case fft_transform_type_complex_inverse:
        return fftw_plan_guru64_dft<Tfloat>(dims.size(),
                                            dims.data(),
                                            howmany_dims.size(),
                                            howmany_dims.data(),
                                            reinterpret_cast<fftw_complex_type*>(cpu_in),
                                            reinterpret_cast<fftw_complex_type*>(cpu_out),
                                            1,
                                            use_fftw_wisdom ? FFTW_MEASURE : FFTW_ESTIMATE);
    case fft_transform_type_real_forward:
        return fftw_plan_guru64_r2c<Tfloat>(dims.size(),
                                            dims.data(),
                                            howmany_dims.size(),
                                            howmany_dims.data(),
                                            reinterpret_cast<Tfloat*>(cpu_in),
                                            reinterpret_cast<fftw_complex_type*>(cpu_out),
                                            use_fftw_wisdom ? FFTW_MEASURE : FFTW_ESTIMATE);
    case fft_transform_type_real_inverse:
        return fftw_plan_guru64_c2r<Tfloat>(dims.size(),
                                            dims.data(),
                                            howmany_dims.size(),
                                            howmany_dims.data(),
                                            reinterpret_cast<fftw_complex_type*>(cpu_in),
                                            reinterpret_cast<Tfloat*>(cpu_out),
                                            use_fftw_wisdom ? FFTW_MEASURE : FFTW_ESTIMATE);
    default:
        throw std::runtime_error("Invalid transform type");
    }
}

// construct an FFTW plan, given rocFFT parameters.  output is
// required if planning with wisdom.
template <typename Tfloat>
static typename fftw_trait<Tfloat>::fftw_plan_type
    fftw_plan_via_rocfft(const std::vector<size_t>& length,
                         const std::vector<size_t>& istride,
                         const std::vector<size_t>& ostride,
                         const size_t               nbatch,
                         const size_t               idist,
                         const size_t               odist,
                         const fft_transform_type   transformType,
                         std::vector<hostbuf>&      input,
                         std::vector<hostbuf>&      output)
{
    // Dimension configuration:
    std::vector<fftw_iodim64> dims(length.size());
    for(unsigned int idx = 0; idx < length.size(); ++idx)
    {
        dims[idx].n  = length[idx];
        dims[idx].is = istride[idx];
        dims[idx].os = ostride[idx];
    }

    // Batch configuration:
    std::vector<fftw_iodim64> howmany_dims(1);
    howmany_dims[0].n  = nbatch;
    howmany_dims[0].is = idist;
    howmany_dims[0].os = odist;

    return fftw_plan_with_precision<Tfloat>(dims,
                                            howmany_dims,
                                            transformType,
                                            idist * nbatch,
                                            input.front().data(),
                                            output.empty() ? nullptr : output.front().data());
}

template <typename Tfloat>
void fftw_run(fft_transform_type                          transformType,
              typename fftw_trait<Tfloat>::fftw_plan_type cpu_plan,
              std::vector<hostbuf>&                       cpu_in,
              std::vector<hostbuf>&                       cpu_out)
{
    switch(transformType)
    {
    case fft_transform_type_complex_forward:
    {
        fftw_plan_execute_c2c<Tfloat>(cpu_plan, cpu_in, cpu_out);
        break;
    }
    case fft_transform_type_complex_inverse:
    {
        fftw_plan_execute_c2c<Tfloat>(cpu_plan, cpu_in, cpu_out);
        break;
    }
    case fft_transform_type_real_forward:
    {
        fftw_plan_execute_r2c<Tfloat>(cpu_plan, cpu_in, cpu_out);
        break;
    }
    case fft_transform_type_real_inverse:
    {
        fftw_plan_execute_c2r<Tfloat>(cpu_plan, cpu_in, cpu_out);
        break;
    }
    }
}

// Given a transform type, return the contiguous input type.
inline fft_array_type contiguous_itype(const fft_transform_type transformType)
{
    switch(transformType)
    {
    case fft_transform_type_complex_forward:
    case fft_transform_type_complex_inverse:
        return fft_array_type_complex_interleaved;
    case fft_transform_type_real_forward:
        return fft_array_type_real;
    case fft_transform_type_real_inverse:
        return fft_array_type_hermitian_interleaved;
    default:
        throw std::runtime_error("Invalid transform type");
    }
    return fft_array_type_complex_interleaved;
}

// Given a transform type, return the contiguous output type.
inline fft_array_type contiguous_otype(const fft_transform_type transformType)
{
    switch(transformType)
    {
    case fft_transform_type_complex_forward:
    case fft_transform_type_complex_inverse:
        return fft_array_type_complex_interleaved;
    case fft_transform_type_real_forward:
        return fft_array_type_hermitian_interleaved;
    case fft_transform_type_real_inverse:
        return fft_array_type_real;
    default:
        throw std::runtime_error("Invalid transform type");
    }
    return fft_array_type_complex_interleaved;
}

// Given a precision, return the acceptable tolerance.
inline double type_epsilon(const fft_precision precision)
{
    switch(precision)
    {
    case fft_precision_half:
        return type_epsilon<rocfft_fp16>();
        break;
    case fft_precision_single:
        return type_epsilon<float>();
        break;
    case fft_precision_double:
        return type_epsilon<double>();
        break;
    default:
        throw std::runtime_error("Invalid precision");
    }
}

#endif
