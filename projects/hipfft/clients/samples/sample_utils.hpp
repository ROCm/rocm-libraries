// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights
// reserved.
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

#ifndef SAMPLE_UTILS_HPP_
#define SAMPLE_UTILS_HPP_

#include <complex>
#include <cstddef>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// An array printer with bounds on the number of outputs to show
template <typename Tvalue>
void printarraylimit(const std::vector<Tvalue>& vals,
                     const size_t               Nx,
                     const size_t               Ny,
                     const size_t               printlimit)
{
    const size_t skipmarg = 2;
    const bool   xskip    = Nx > skipmarg && Nx - skipmarg > printlimit;
    const bool   yskip    = Ny > skipmarg && Ny - skipmarg > printlimit;

    bool xskipped = false;
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        if(!xskipped && xskip && xidx > printlimit)
        {
            xskipped = true;
            std::cout << "...\n";
            xidx = Nx - skipmarg;
        }
        bool yskipped = false;
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            if(!yskipped && yskip && yidx > printlimit)
            {
                yskipped = true;
                std::cout << "... ";
                yidx = Ny - skipmarg;
            }
            int pos = xidx * Ny + yidx;
            std::cout << vals[pos] << " ";
        }
        std::cout << "\n";
    }
}

// Make the output of a 2D c2c transform linear.  Useful for visualization purposes.
template <typename Tfloat>
inline void sneakyc2c(std::vector<std::complex<Tfloat>>& cinput,
                      const int                          Nx,
                      const int                          Ny,
                      const int                          direction)
{
    // Implemented only for single and double precision.
    static_assert(std::is_same_v<Tfloat, float> || std::is_same_v<Tfloat, double>,
                  "Tfloat must be a float or double.");

    hipError_t hip_rt;
    using fftctype
        = std::conditional_t<std::is_same_v<Tfloat, float>, hipfftComplex, hipfftDoubleComplex>;
    fftctype* x;

    using ArrayType      = typename std::remove_reference<decltype(cinput)>::type;
    using ValueType      = typename std::remove_extent<ArrayType>::type;
    size_t complex_bytes = sizeof(ValueType) * cinput.size();

    hip_rt = hipMalloc(&x, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    hip_rt = hipMemcpy(x, cinput.data(), complex_bytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
    hipfftHandle plan{};
    auto         hipfft_rt = HIPFFT_SUCCESS;
    hipfft_rt
        = hipfftPlan2d(&plan, Nx, Ny, std::is_same_v<Tfloat, float> ? HIPFFT_C2C : HIPFFT_Z2Z);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan2d failed");
    if constexpr(std::is_same_v<Tfloat, float>)
        hipfft_rt = hipfftExecC2C(plan, x, x, -direction);
    else if constexpr(std::is_same_v<Tfloat, double>)
        hipfft_rt = hipfftExecZ2Z(plan, x, x, -direction);
    else
        throw std::runtime_error("Unsupported precision");

    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftExec failed");
    hip_rt = hipMemcpy(cinput.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    double norm = 1.0 / (Nx * Ny);
    for(auto&& val : cinput)
        val *= norm;

    hipfftDestroy(plan);

    hip_rt = hipFree(x);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");
}

#endif
