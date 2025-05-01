// Copyright (C) 2019 - 2022 Advanced Micro Devices, Inc. All rights
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

#include <complex>
#include <iostream>
#include <vector>

#include <hipfft/hipfft.h>

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

#include "../hipfft_params.h"

int main()
{
    std::cout << "hipfft 3D double-precision real-to-complex transform\n";

    const size_t Nx = 4;
    const size_t Ny = 5;
    const size_t Nz = 6;

    std::cout << "Nx: " << Nx << "\tNy " << Ny << "\tNz " << Nz << std::endl;

    const size_t Nzcomplex = Nz / 2 + 1;
    const size_t rstride   = Nzcomplex * 2; // Nz for out-of-place

    const size_t real_bytes    = sizeof(double) * Nx * Ny * rstride;
    const size_t complex_bytes = 2 * sizeof(double) * Nx * Ny * Nzcomplex;

    double*    x;
    hipError_t hip_rt;
    hip_rt = hipMalloc(&x, real_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    // Inititalize the data
    std::vector<double> rdata(Nx * Ny * rstride);
    for(size_t i = 0; i < Nx * Ny * rstride; i++)
    {
        rdata[i] = i;
    }
    std::cout << "input:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < rstride; k++)
            {
                auto pos = (i * Ny + j) * rstride + k;
                std::cout << rdata[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    hip_rt = hipMemcpy(x, rdata.data(), real_bytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    // Create plan:
    hipfftHandle plan      = hipfft_params::INVALID_PLAN_HANDLE;
    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");
    hipfft_rt = hipfftPlan3d(&plan, // plan handle
                             Nx,
                             Ny,
                             Nz, // transform lengths
                             HIPFFT_D2Z); // transform type (HIPFFT_R2C for single-precision)
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan3d failed");

    // Execute plan:
    // hipfftExecD2Z: double precision, hipfftExecR2C: single-precision
    hipfft_rt = hipfftExecD2Z(plan, x, (hipfftDoubleComplex*)x);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftExecD2Z failed");

    std::cout << "output:\n";
    std::vector<std::complex<double>> cdata(Nx * Ny * Nz);
    hip_rt = hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nzcomplex; k++)
            {
                auto pos = (i * Ny + j) * Nzcomplex + k;
                std::cout << cdata[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);

    hip_rt = hipFree(x);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    return 0;
}
