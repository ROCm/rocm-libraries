// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_ENUMS_VS_FFT_ENUMS_H
#define ROCFFT_ENUMS_VS_FFT_ENUMS_H

#include "fft_enums.h"
#include "rocfft/rocfft.h"
#include <stdexcept>

inline fft_status fft_status_from_rocfftparams(const rocfft_status val)
{
    switch(val)
    {
    case rocfft_status_success:
        return fft_status_success;
    case rocfft_status_failure:
        return fft_status_failure;
    case rocfft_status_invalid_arg_value:
        return fft_status_invalid_arg_value;
    case rocfft_status_invalid_dimensions:
        return fft_status_invalid_dimensions;
    case rocfft_status_invalid_array_type:
        return fft_status_invalid_array_type;
    case rocfft_status_invalid_strides:
        return fft_status_invalid_strides;
    case rocfft_status_invalid_distance:
        return fft_status_invalid_distance;
    case rocfft_status_invalid_offset:
        return fft_status_invalid_offset;
    case rocfft_status_invalid_work_buffer:
        return fft_status_invalid_work_buffer;
    default:
        throw std::runtime_error("Invalid status");
    }
}

inline rocfft_precision rocfft_precision_from_fftparams(const fft_precision val)
{
    switch(val)
    {
    case fft_precision_single:
        return rocfft_precision_single;
    case fft_precision_double:
        return rocfft_precision_double;
    case fft_precision_half:
        return rocfft_precision_half;
    default:
        throw std::runtime_error("Invalid precision");
    }
}

inline fft_precision fft_precision_from_rocfft_precision(const rocfft_precision val)
{
    switch(val)
    {
    case rocfft_precision_single:
        return fft_precision_single;
    case rocfft_precision_double:
        return fft_precision_double;
    case rocfft_precision_half:
        return fft_precision_half;
    default:
        throw std::runtime_error("Invalid precision");
    }
}

inline rocfft_array_type rocfft_array_type_from_fftparams(const fft_array_type val)
{
    switch(val)
    {
    case fft_array_type_complex_interleaved:
        return rocfft_array_type_complex_interleaved;
    case fft_array_type_complex_planar:
        return rocfft_array_type_complex_planar;
    case fft_array_type_real:
        return rocfft_array_type_real;
    case fft_array_type_hermitian_interleaved:
        return rocfft_array_type_hermitian_interleaved;
    case fft_array_type_hermitian_planar:
        return rocfft_array_type_hermitian_planar;
    case fft_array_type_unset:
        return rocfft_array_type_unset;
    }
    return rocfft_array_type_unset;
}

inline fft_array_type fft_array_type_from_rocfft_array_type(const rocfft_array_type val)
{
    switch(val)
    {
    case rocfft_array_type_complex_interleaved:
        return fft_array_type_complex_interleaved;
    case rocfft_array_type_complex_planar:
        return fft_array_type_complex_planar;
    case rocfft_array_type_real:
        return fft_array_type_real;
    case rocfft_array_type_hermitian_interleaved:
        return fft_array_type_hermitian_interleaved;
    case rocfft_array_type_hermitian_planar:
        return fft_array_type_hermitian_planar;
    case rocfft_array_type_unset:
        return fft_array_type_unset;
    }
    return fft_array_type_unset;
}

inline rocfft_transform_type rocfft_transform_type_from_fftparams(const fft_transform_type val)
{
    switch(val)
    {
    case fft_transform_type_complex_forward:
        return rocfft_transform_type_complex_forward;
    case fft_transform_type_complex_inverse:
        return rocfft_transform_type_complex_inverse;
    case fft_transform_type_real_forward:
        return rocfft_transform_type_real_forward;
    case fft_transform_type_real_inverse:
        return rocfft_transform_type_real_inverse;
    default:
        throw std::runtime_error("Invalid transform type");
    }
}

inline fft_transform_type
    fft_transform_type_from_rocfft_transform_type(const rocfft_transform_type val)
{
    switch(val)
    {
    case rocfft_transform_type_complex_forward:
        return fft_transform_type_complex_forward;
    case rocfft_transform_type_complex_inverse:
        return fft_transform_type_complex_inverse;
    case rocfft_transform_type_real_forward:
        return fft_transform_type_real_forward;
    case rocfft_transform_type_real_inverse:
        return fft_transform_type_real_inverse;
    default:
        throw std::runtime_error("Invalid transform type");
    }
}

inline rocfft_result_placement
    rocfft_result_placement_from_fftparams(const fft_result_placement val)
{
    switch(val)
    {
    case fft_placement_inplace:
        return rocfft_placement_inplace;
    case fft_placement_notinplace:
        return rocfft_placement_notinplace;
    default:
        throw std::runtime_error("Invalid result placement");
    }
}

inline fft_result_placement
    fft_result_placement_from_rocfft_result_placement(const rocfft_result_placement val)
{
    switch(val)
    {
    case rocfft_placement_inplace:
        return fft_placement_inplace;
    case rocfft_placement_notinplace:
        return fft_placement_notinplace;
    default:
        throw std::runtime_error("Invalid result placement");
    }
}

#endif // ROCFFT_ENUMS_VS_FFT_ENUMS_H
