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

/*! @file dispatch.hpp
 *  @brief Compile-time dispatch from (length, precision) to a Stockham configuration.
 *  @details Each specialization records the workgroup size, elements per thread,
 *  and maximum radix for one supported transform, which ::rocfft::device::FFT
 *  uses to select the matching pre-generated Stockham implementation.
 */

#pragma once

#ifndef ROCFFT_DEVICE_DISPATCH_HPP
#define ROCFFT_DEVICE_DISPATCH_HPP

#include "../precision.hpp"

namespace rocfft
{
    namespace device
    {
        namespace detail
        {

            /*! @brief Per-configuration Stockham parameters for a (length, precision) pair.
 *  @details Primary template is intentionally undefined; only supported
 *  configurations are specialized, so an unsupported pair fails to instantiate.
 *  @tparam Length Transform length in complex elements.
 *  @tparam P      Numeric precision; see ::rocfft::device::Precision.
 */
            template <unsigned Length, Precision P>
            struct FFTDispatch;

            // ---- length 32: factors (4,8), max_radix=8 ----

            template <>
            struct FFTDispatch<32, Precision::Single>
            {
                static constexpr unsigned workgroup_size = 8;
                static constexpr unsigned ept            = 4;
                static constexpr unsigned max_radix      = 8;
            };

            template <>
            struct FFTDispatch<32, Precision::Double>
            {
                static constexpr unsigned workgroup_size = 8;
                static constexpr unsigned ept            = 4;
                static constexpr unsigned max_radix      = 8;
            };

            // ---- length 64: factors (2,4,8), max_radix=8 ----

            template <>
            struct FFTDispatch<64, Precision::Single>
            {
                static constexpr unsigned workgroup_size = 32;
                static constexpr unsigned ept            = 2;
                static constexpr unsigned max_radix      = 8;
            };

            template <>
            struct FFTDispatch<64, Precision::Double>
            {
                static constexpr unsigned workgroup_size = 32;
                static constexpr unsigned ept            = 2;
                static constexpr unsigned max_radix      = 8;
            };

            // ---- length 128: factors (16,8), max_radix=16 ----

            template <>
            struct FFTDispatch<128, Precision::Single>
            {
                static constexpr unsigned workgroup_size = 64;
                static constexpr unsigned ept            = 2;
                static constexpr unsigned max_radix      = 16;
            };

            template <>
            struct FFTDispatch<128, Precision::Double>
            {
                static constexpr unsigned workgroup_size = 64;
                static constexpr unsigned ept            = 2;
                static constexpr unsigned max_radix      = 16;
            };

            // ---- length 256: factors (16,16), max_radix=16 ----

            template <>
            struct FFTDispatch<256, Precision::Single>
            {
                static constexpr unsigned workgroup_size = 128;
                static constexpr unsigned ept            = 2;
                static constexpr unsigned max_radix      = 16;
            };

            template <>
            struct FFTDispatch<256, Precision::Double>
            {
                static constexpr unsigned workgroup_size = 128;
                static constexpr unsigned ept            = 2;
                static constexpr unsigned max_radix      = 16;
            };

            // ---- length 512: factors (8,8,8), max_radix=8 ----

            template <>
            struct FFTDispatch<512, Precision::Single>
            {
                static constexpr unsigned workgroup_size = 256;
                static constexpr unsigned ept            = 2;
                static constexpr unsigned max_radix      = 8;
            };

            template <>
            struct FFTDispatch<512, Precision::Double>
            {
                static constexpr unsigned workgroup_size = 128;
                static constexpr unsigned ept            = 4;
                static constexpr unsigned max_radix      = 8;
            };

        } // namespace detail
    } // namespace device
} // namespace rocfft

#endif // ROCFFT_DEVICE_DISPATCH_HPP
