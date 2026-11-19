// Copyright (C) 2016 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "fft_enums.h"
#include "hostbuf.h"
#include "rocfft_complex.h"
#include "test_params.h"
#include <algorithm>
#include <type_traits>

// Function to return maximum error for float and double types.
//
// Following Schatzman (1996; Accuracy of the Discrete Fourier
// Transform and the Fast Fourier Transform), the shape of relative
// l_2 error vs length should look like
//
//   epsilon * sqrt(log2(length)).
//
// The magic epsilon constants below were chosen so that we get a
// reasonable upper bound for (all of) our tests.
//
// For rocFFT, prime lengths result in the highest error.  As such,
// the epsilons below are perhaps too loose for pow2 lengths; but they
// are appropriate for prime lengths.
template <typename Tfloat>
inline double type_epsilon();
template <>
inline double type_epsilon<rocfft_fp16>()
{
    return half_epsilon;
}
template <>
inline double type_epsilon<float>()
{
    return single_epsilon;
}
template <>
inline double type_epsilon<double>()
{
    return double_epsilon;
}

static constexpr double default_half_epsilon()
{
    return 9.77e-4;
}

static constexpr double default_single_epsilon()
{
    return 3.75e-5;
}

static constexpr double default_double_epsilon()
{
    return 1e-15;
}

// Copies the half-precision input buffer to a single-precision buffer.
static hostbuf half_to_single_copy(const hostbuf& in)
{
    hostbuf out;
    out.alloc(2 * in.size());
    auto in_begin = reinterpret_cast<const rocfft_fp16*>(in.data());
    std::copy_n(in_begin, in.size() / sizeof(rocfft_fp16), reinterpret_cast<float*>(out.data()));
    return out;
}

// converts a wider precision buffer to a narrower precision, in-place
template <typename TfloatIn, typename TfloatOut>
void narrow_precision_inplace(hostbuf& in)
{
    // ensure we're actually shrinking the data
    static_assert(sizeof(TfloatIn) > sizeof(TfloatOut));

    auto readPtr  = reinterpret_cast<const TfloatIn*>(in.data());
    auto writePtr = reinterpret_cast<TfloatOut*>(in.data());
    std::copy_n(readPtr, in.size() / sizeof(TfloatIn), writePtr);
    in.shrink(in.size() / (sizeof(TfloatIn) / sizeof(TfloatOut)));
}

static void single_to_half_inplace(hostbuf& in)
{
    narrow_precision_inplace<float, rocfft_fp16>(in);
}

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
