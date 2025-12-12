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

#ifndef SAMPLE_UTILS_HPPP_
#define SAMPLE_UTILS_HPPP_

#include <complex>
#include <iostream>
#include <numeric>
#include <vector>
#include <type_traits>

// An array printer that with bounds on the number of outputs to show
template<typename Tvalue>
void printarraylimit(const std::vector<Tvalue> &vals, const size_t Nx, const size_t Ny,
                     const size_t printlimit)
{
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        if(xidx > printlimit)
        {
            std::cout << "...\n";
            xidx = Nx - 1;
        }
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            if(yidx > printlimit)
            {
                std::cout << "... ";
                yidx = Ny - 1;
            }
            int pos = xidx * Ny + yidx;
            std::cout << vals[pos] << " ";
        }
        std::cout << "\n";
    }
}

// An array printer that with bounds on the number of outputs to show
template<typename Tvalue>
void printarraylimit(const std::vector<Tvalue> &vals,
                     const size_t Nx, const size_t Ny, const size_t Nz,
                     const size_t printlimit)
{
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        if(xidx > printlimit)
        {
            std::cout << "...\n";
            xidx = Nx - 1;
        }
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            if(yidx > printlimit)
            {
                std::cout << "...\n";
                yidx = Ny - 1;
            }
            for(size_t zidx = 0; zidx < Nz; ++zidx)
            {
                if(zidx > printlimit)
                {
                    std::cout << "... ";
                    zidx = Nz - 1;
                }
                int pos = (xidx * Ny + yidx) * Nz + zidx;
                std::cout << vals[pos] << " ";
            }
        }
        std::cout << "\n";
    }
}

// Impose the Hermitian-symmetric data format on a complex buffer by doing a round-trip FFT.
template<typename Tfloat>
inline void hsymmetrize(std::vector<std::complex<Tfloat>> &cinput, const int Nx, const int Ny)
{
    const size_t Nyp = Ny / 2 + 1;
    if(cinput.size() < Nx * Nyp)
       throw std::runtime_error("hsyemmetrize has insufficient dimensions");
    
    hipError_t           hip_rt;
    using fftctype =  std::conditional_t<std::is_same_v<Tfloat, float>,
                                         hipfftComplex, hipfftDoubleComplex>;
    fftctype* cbuf;
    
    using ArrayType = typename std::remove_reference<decltype(cinput)>::type;
    using ValueType = typename std::remove_extent<ArrayType>::type;
    size_t complex_bytes = sizeof(ValueType) * Nx * Nyp;

    hip_rt = hipMalloc(&cbuf, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    hip_rt = hipMalloc(&cbuf, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    hip_rt = hipMemcpy(cbuf, cinput.data(), complex_bytes, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed before hsyemmetrize round-trip");
    
    hipfftHandle planr2c{};
    hipfftResult  hipfft_rt = hipfftPlan2d(&planr2c, Nx, Ny, std::is_same_v<Tfloat, float> ? HIPFFT_R2C : HIPFFT_D2Z);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftCreate failed");

    hipfftHandle planc2r{};
    hipfft_rt = hipfftPlan2d(&planc2r, Nx, Ny, std::is_same_v<Tfloat, float> ? HIPFFT_C2R : HIPFFT_Z2D);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftCreate failed");
    
    Tfloat* rbuf;
    size_t real_bytes = sizeof(Tfloat) * Nx * Ny;
    hip_rt = hipMalloc(&rbuf, real_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    
    if constexpr (std::is_same_v<Tfloat, float>)
        hipfft_rt = hipfftExecC2R(planc2r, cbuf, rbuf);
     else if constexpr (std::is_same_v<Tfloat, double>)
        hipfft_rt =  hipfftExecZ2D(planc2r, cbuf, rbuf);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftexec failed");

    if constexpr(std::is_same_v<Tfloat, float>)
        hipfft_rt = hipfftExecR2C(planr2c, rbuf, cbuf);
    else if constexpr(std::is_same_v<Tfloat, double>)
        hipfft_rt = hipfftExecD2Z(planr2c, rbuf, cbuf);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftexec failed");
    
    hip_rt = hipFree(rbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    hipfftDestroy(planr2c);
    hipfftDestroy(planc2r);
    
    hip_rt = hipMemcpy(cinput.data(), cbuf, complex_bytes, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed after hsyemmetrize round-trip");
    
    hip_rt = hipFree(cbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    // Normalize the output:
    const Tfloat denom = 1.0 / (Nx * Ny);
    for(auto &val: cinput)
        val *= denom;
}

// Impose the Hermitian-symmetric data format on a complex buffer by doing a round-trip FFT.
template<typename Tfloat>
inline void hsymmetrize(std::vector<std::complex<Tfloat>> &cinput, const int Nx, const int Ny,
                        const int Nz)
{
    const size_t Nzp = Nz / 2 + 1;
    if(cinput.size() < Nx * Ny * Nzp)
       throw std::runtime_error("hsyemmetrize has insufficient dimensions");
    
    hipError_t           hip_rt;
    using fftctype =  std::conditional_t<std::is_same_v<Tfloat, float>,
                                         hipfftComplex, hipfftDoubleComplex>;
    fftctype* cbuf;
    
    using ArrayType = typename std::remove_reference<decltype(cinput)>::type;
    using ValueType = typename std::remove_extent<ArrayType>::type;
    size_t complex_bytes = sizeof(ValueType) * Nx * Ny * Nzp;

    hip_rt = hipMalloc(&cbuf, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    hip_rt = hipMalloc(&cbuf, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    hip_rt = hipMemcpy(cbuf, cinput.data(), complex_bytes, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed before hsyemmetrize round-trip");
    
    hipfftHandle planr2c{};
    hipfftResult  hipfft_rt = hipfftPlan3d(&planr2c, Nx, Ny, Nz, std::is_same_v<Tfloat, float> ? HIPFFT_R2C : HIPFFT_D2Z);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftCreate failed");

    hipfftHandle planc2r{};
    hipfft_rt = hipfftPlan3d(&planc2r, Nx, Ny, Nz, std::is_same_v<Tfloat, float> ? HIPFFT_C2R : HIPFFT_Z2D);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftCreate failed");
    
    Tfloat* rbuf;
    size_t real_bytes = sizeof(Tfloat) * Nx * Ny * Nz;
    hip_rt = hipMalloc(&rbuf, real_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    
    if constexpr (std::is_same_v<Tfloat, float>)
        hipfft_rt = hipfftExecC2R(planc2r, cbuf, rbuf);
     else if constexpr (std::is_same_v<Tfloat, double>)
        hipfft_rt =  hipfftExecZ2D(planc2r, cbuf, rbuf);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftexec failed");

    if constexpr(std::is_same_v<Tfloat, float>)
        hipfft_rt = hipfftExecR2C(planr2c, rbuf, cbuf);
    else if constexpr(std::is_same_v<Tfloat, double>)
        hipfft_rt = hipfftExecD2Z(planr2c, rbuf, cbuf);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftexec failed");
    
    hip_rt = hipFree(rbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    hipfftDestroy(planr2c);
    hipfftDestroy(planc2r);
    
    hip_rt = hipMemcpy(cinput.data(), cbuf, complex_bytes, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed after hsyemmetrize round-trip");
    
    hip_rt = hipFree(cbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    // Normalize the output:
    const Tfloat denom = 1.0 / (Nx * Ny * Nz);
    for(auto &val: cinput)
        val *= denom;
}

// Make the output of a r2c linear.
template<typename Tfloat>
inline void sneakyr2c(std::vector<Tfloat> &rinput, const int Nx, const int Ny)
{
    const size_t Nyp = Ny / 2 + 1;
    
    std::vector<std::complex<double>> cinput(Nx * Nyp);
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        for(size_t yidx = 0; yidx < Nyp; ++yidx)
        {
            cinput[xidx * Nyp + yidx] = std::complex<double>(xidx,yidx);
        }
    }
    
    hipError_t           hip_rt;
    
    Tfloat* rbuf;
    size_t real_bytes = sizeof(Tfloat) * Nx * Ny;
    hip_rt = hipMalloc(&rbuf, real_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
        
    Tfloat* cbuf;
    size_t complex_bytes = sizeof(std::complex<Tfloat>) * Nx * (Ny/2 + 1);
    hip_rt = hipMalloc(&rbuf, real_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    hipfftHandle planc2r{};
    hipfftResult  hipfft_rt = hipfftPlan2d(&planc2r, Nx, Ny, std::is_same_v<Tfloat, float> ? HIPFFT_C2R : HIPFFT_Z2D);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan2d failed");

    hip_rt = hipMemcpy(cbuf, cinput.data(), complex_bytes, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed before hsyemmetrize round-trip");
    
    if constexpr(std::is_same_v<Tfloat, float>)
        hipfft_rt = hipfftExecR2C(planc2r, cbuf, rbuf);
    else if constexpr(std::is_same_v<Tfloat, double>)
        hipfft_rt = hipfftExecZ2D(planc2r, cbuf, rbuf);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftexec failed");

    hip_rt = hipMemcpy(cinput.data(), rbuf, sizeof(Tfloat) * Nx * Ny, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed after hsyemmetrize round-trip");

    hip_rt = hipFree(cbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");
    
    hip_rt = hipFree(rbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");
    
    hipfftDestroy(planc2r);
    hip_rt = hipFree(cbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");
}

// Make the output of a r2c linear.
template<typename Tfloat>
inline void sneakyr2c(std::vector<Tfloat> &rinput, const int Nx, const int Ny, const int Nz)
{
    const size_t Nzp = Nz / 2 + 1;
    
    std::vector<std::complex<double>> cinput(Nx * Ny * Nzp);
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            for(size_t zidx = 0; zidx < Nzp; ++zidx)
            {
                const size_t pos = (xidx * Ny + yidx) * Nzp + zidx;
                cinput[pos] = std::complex<double>(xidx * Ny + yidx, zidx);
            }
        }
    }
    
    hipError_t           hip_rt;
    
    Tfloat* rbuf;
    size_t real_bytes = sizeof(Tfloat) * Nx * Ny * Nz;
    hip_rt = hipMalloc(&rbuf, real_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
        
    Tfloat* cbuf;
    size_t complex_bytes = sizeof(std::complex<Tfloat>) * Nx * Ny * Nzp;
    hip_rt = hipMalloc(&rbuf, real_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    hipfftHandle planc2r{};
    hipfftResult  hipfft_rt = hipfftPlan3d(&planc2r, Nx, Ny, Nz, std::is_same_v<Tfloat, float> ? HIPFFT_C2R : HIPFFT_Z2D);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan2d failed");

    hip_rt = hipMemcpy(cbuf, cinput.data(), complex_bytes, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed before hsyemmetrize round-trip");
    
    if constexpr(std::is_same_v<Tfloat, float>)
        hipfft_rt = hipfftExecR2C(planc2r, cbuf, rbuf);
    else if constexpr(std::is_same_v<Tfloat, double>)
        hipfft_rt = hipfftExecZ2D(planc2r, cbuf, rbuf);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftexec failed");

    hip_rt = hipMemcpy(cinput.data(), rbuf, sizeof(Tfloat) * Nx * Ny * Nzp, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed after hsyemmetrize round-trip");

    hip_rt = hipFree(cbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");
    
    hip_rt = hipFree(rbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");
    
    hipfftDestroy(planc2r);
    hip_rt = hipFree(cbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");
}

// Make the output of a c2r linear.
template<typename Tfloat>
inline void sneakyc2r(std::vector<std::complex<Tfloat>> &cinput, const int Nx, const int Ny)
{
    if(cinput.size() < Nx * (Ny /2 + 1))
       throw std::runtime_error("sneakyc2r has insufficient dimensions");
   
    std::vector<Tfloat> rvals(Nx * Ny);
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            rvals[xidx * Ny + yidx] = xidx * Nx + yidx;
        }
    }

    hipError_t           hip_rt;
    
    Tfloat* rbuf;
    size_t real_bytes = sizeof(Tfloat) * Nx * Ny;
    hip_rt = hipMalloc(&rbuf, real_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
        
    using fftctype = std::conditional_t<std::is_same_v<Tfloat, float>,
                                        hipfftComplex, hipfftDoubleComplex>;
    fftctype* cbuf;
    size_t complex_bytes = sizeof(std::complex<Tfloat>) * Nx * (Ny / 2 + 1);
    hip_rt = hipMalloc(&cbuf, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    
    hipfftHandle planr2c{};
    hipfftResult hipfft_rt = hipfftPlan2d(&planr2c, Nx, Ny, std::is_same_v<Tfloat, float> ? HIPFFT_R2C : HIPFFT_D2Z);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan2d failed");

    hip_rt = hipMemcpy(rbuf, rvals.data(), real_bytes, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed before sneaky transform");

    if constexpr(std::is_same_v<Tfloat, float>)
        hipfft_rt = hipfftExecR2C(planr2c, rbuf, cbuf);
    else if constexpr(std::is_same_v<Tfloat, double>)
        hipfft_rt = hipfftExecD2Z(planr2c, rbuf, cbuf);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftexec failed");

    hip_rt = hipMemcpy(cinput.data(), cbuf, complex_bytes, hipMemcpyDefault);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed after sneaky transform");

    // Normalize the output:
    const Tfloat denom = 1.0 / (Nx * Ny);
    for(auto &val: cinput)
        val *= denom;
    
    hipfftDestroy(planr2c);
    
    hip_rt = hipFree(rbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("rbuf hipFree failed");

    hip_rt = hipFree(cbuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("cbuf hipFree failed");
}
    
template<typename Tfloat>
inline void sneakyc2c(std::vector<std::complex<Tfloat>> &cinput,
                      const int Nx, const int Ny, const int direction)
{
    hipError_t           hip_rt;
    using fftctype =  std::conditional_t<std::is_same_v<Tfloat, float>,
                                         hipfftComplex, hipfftDoubleComplex>;
    fftctype* x;

    using ArrayType = typename std::remove_reference<decltype(cinput)>::type;
    using ValueType = typename std::remove_extent<ArrayType>::type;
    size_t complex_bytes = sizeof(ValueType) * cinput.size();
    
    //size_t complex_bytes = sizeof(std::remove_extent<decltype(std::remove_reference<cinput>)::value_type>) * cinput.size();

    
    hip_rt = hipMalloc(&x, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    hip_rt = hipMemcpy(x, cinput.data(), complex_bytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
    hipfftHandle plan{};
    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftCreate failed");

    hipfft_rt = hipfftPlan2d(&plan, // plan handle
                             Nx, // transform length
                             Ny, // transform length
                             HIPFFT_Z2Z); // transform type (HIPFFT_C2C for single-precision)
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan2d failed");
    hipfft_rt = hipfftExecZ2Z(plan, x, x, -direction);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftExecZ2Z failed");
    hip_rt = hipMemcpy(cinput.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    double norm = 1.0 / (Nx * Ny);
    for(auto &&val : cinput)
        val *= norm;
        
    hipfftDestroy(plan);

    hip_rt = hipFree(x);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");
}


#endif
