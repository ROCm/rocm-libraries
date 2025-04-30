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
    std::cout << "hipfft 2D double-precision real-to-complex transform\n";

    const size_t Nx = 4;
    const size_t Ny = 5;

    std::cout << "Nx: " << Nx << "\tNy: " << Ny << std::endl;

    const size_t Nycomplex = Ny / 2 + 1;
    const size_t rstride   = Nycomplex * 2; // Ny for out-of-place

    std::cout << "Input:\n";
    std::vector<double> rdata(Nx * rstride);
    for(size_t i = 0; i < Nx * rstride; i++)
    {
        rdata[i] = i;
    }
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            auto pos = i * rstride + j;
            std::cout << rdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    double*    x;
    hipError_t hip_rt;
    hip_rt = hipMalloc(&x, rdata.size() * sizeof(decltype(rdata)::value_type));
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    hip_rt = hipMemcpy(
        x, rdata.data(), rdata.size() * sizeof(decltype(rdata)::value_type), hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    // Create plan:
    hipfftHandle plan      = hipfft_params::INVALID_PLAN_HANDLE;
    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");
    hipfft_rt = hipfftPlan2d(&plan, // plan handle
                             Nx, // transform length
                             Ny, // transform length
                             HIPFFT_D2Z); // transform type (HIPFFT_R2C for single-precision)
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlandd failed");

    // Execute plan:
    // hipfftExecD2Z: double precision.  hipfftExecR2C: single-precision
    hipfft_rt = hipfftExecD2Z(plan, x, (hipfftDoubleComplex*)x);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftExecD2Z failed");

    // Copy the output data to the host:
    std::vector<std::complex<double>> cdata(Nx * Nycomplex);
    hip_rt = hipMemcpy(
        cdata.data(), x, cdata.size() * sizeof(decltype(cdata)::value_type), hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    std::cout << "Output:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nycomplex; j++)
        {
            auto pos = i * Nycomplex + j;
            std::cout << cdata[pos] << " ";
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
