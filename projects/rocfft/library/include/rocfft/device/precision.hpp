// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/*! @file precision.hpp
 *  @brief Numeric precision tag and precision-to-type traits for the device FFT API.
 */

#pragma once

#include "detail/complex.hpp"

namespace rocfft
{
namespace device
{

/*! @brief Numeric precision of a device FFT. */
enum class Precision
{
    Single, //!< IEEE single precision (32-bit float).
    Double, //!< IEEE double precision (64-bit double).
};

/*! @brief Maps a ::Precision value to its real and complex element types.
 *  @tparam P Precision tag to resolve. Specialized for each ::Precision value.
 */
template <Precision P>
struct precision_traits;

//! @brief ::Precision::Single specialization (float / rocfft_complex<float>).
template <>
struct precision_traits<Precision::Single>
{
    using real_type    = float;                 //!< Real scalar type.
    using complex_type = rocfft_complex<float>; //!< Complex element type.
};

//! @brief ::Precision::Double specialization (double / rocfft_complex<double>).
template <>
struct precision_traits<Precision::Double>
{
    using real_type    = double;                 //!< Real scalar type.
    using complex_type = rocfft_complex<double>; //!< Complex element type.
};

} // namespace device
} // namespace rocfft
