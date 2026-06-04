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

/*! @file twiddle.hpp
 *  @brief Compile-time twiddle table generation for the rocFFT device library.
 *  @details Generates the same twiddle tables as rocFFT's Stockham RTC path,
 *  using the radix-table layout formula:
 *  @code
 *  For each pair of adjacent factors (f_i, f_{i+1}):
 *    product *= f_i
 *    transform_length = product * f_{i+1}
 *    for k in [0, transform_length / f_{i+1}):
 *      theta = -2*pi*k / transform_length
 *      for j in [1, f_{i+1}):
 *        emit (cos(j*theta), sin(j*theta))
 *  @endcode
 */

#pragma once

#ifndef ROCFFT_DEVICE_TWIDDLE_HPP
#define ROCFFT_DEVICE_TWIDDLE_HPP

#include "complex.hpp"

namespace rocfft
{
namespace device
{
namespace detail
{

// Constexpr math helpers.  C++14 constexpr is sufficient for simple loops.
// These use Taylor series for compile-time evaluation.

constexpr double pi_v = 3.14159265358979323846264338327950288;

constexpr double constexpr_fmod(double x, double y)
{
    return x - static_cast<long long>(x / y) * y;
}

constexpr double constexpr_fabs(double x)
{
    return x < 0.0 ? -x : x;
}

constexpr double constexpr_cos(double x)
{
    // Range reduction to [-pi, pi]
    x = constexpr_fmod(x, 2.0 * pi_v);
    if(x > pi_v)
        x -= 2.0 * pi_v;
    if(x < -pi_v)
        x += 2.0 * pi_v;

    // Horner form of Taylor series (18 terms for ~17 digits of precision)
    double x2     = x * x;
    double result = 1.0;
    double term   = 1.0;
    for(int n = 1; n <= 18; ++n)
    {
        term *= -x2 / ((2.0 * n - 1.0) * (2.0 * n));
        result += term;
    }
    return result;
}

constexpr double constexpr_sin(double x)
{
    return constexpr_cos(x - pi_v / 2.0);
}

// Compute the number of twiddle entries for a given factorization.
// This is needed to size the twiddle array at compile time.
template <unsigned N>
constexpr unsigned twiddle_count(const unsigned (&factors)[N])
{
    unsigned count   = 0;
    unsigned product = 1;
    for(unsigned i = 0; i + 1 < N; ++i)
    {
        product *= factors[i];
        unsigned next_radix        = factors[i + 1];
        unsigned transform_length  = product * next_radix;
        unsigned entries_this_pass = (transform_length / next_radix) * (next_radix - 1);
        count += entries_this_pass;
    }
    return count;
}

// Generate twiddle table at compile time.
//
// Usage:
//   constexpr unsigned factors[] = {2, 4, 8};
//   constexpr auto twiddles = make_twiddle_table<float, 3>(factors);
//   // twiddles.data[i] is rocfft_complex<float>
//
template <typename Real, unsigned N>
struct TwiddleTable
{
    static constexpr unsigned                 factors_arr[N] = {};
    static constexpr unsigned                 count          = 0;
    rocfft_complex<Real>                      data[1]; // placeholder
};

template <typename Real, unsigned NumFactors, unsigned Count>
struct TwiddleTableStorage
{
    rocfft_complex<Real> data[Count > 0 ? Count : 1];
};

template <typename Real, unsigned NumFactors, unsigned Count>
constexpr TwiddleTableStorage<Real, NumFactors, Count>
    compute_twiddle_table(const unsigned (&factors)[NumFactors])
{
    TwiddleTableStorage<Real, NumFactors, Count> table{};
    unsigned                                     idx     = 0;
    unsigned                                     product = 1;

    for(unsigned i = 0; i + 1 < NumFactors; ++i)
    {
        product *= factors[i];
        unsigned next_radix       = factors[i + 1];
        unsigned transform_length = product * next_radix;

        for(unsigned k = 0; k < transform_length / next_radix; ++k)
        {
            double theta = -2.0 * pi_v * static_cast<double>(k)
                           / static_cast<double>(transform_length);
            for(unsigned j = 1; j < next_radix; ++j)
            {
                double angle  = static_cast<double>(j) * theta;
                table.data[idx] = rocfft_complex<Real>(
                    static_cast<Real>(constexpr_cos(angle)),
                    static_cast<Real>(constexpr_sin(angle)));
                ++idx;
            }
        }
    }
    return table;
}

} // namespace detail
} // namespace device
} // namespace rocfft

#endif // ROCFFT_DEVICE_TWIDDLE_HPP
