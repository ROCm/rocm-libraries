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
    std::cout << "hipfft 1D double-precision real-to-complex transform\n";

    const size_t Nx = 8;

    const size_t Ncomplex = Nx / 2 + 1;

    std::vector<double> rdata(Nx);
    size_t              real_bytes = sizeof(decltype(rdata)::value_type) * rdata.size();
    std::vector<std::complex<decltype(rdata)::value_type>> cdata(Ncomplex);
    size_t complex_bytes = sizeof(std::complex<decltype(rdata)::value_type>) * cdata.size();

    // Create HIP device object
    double*    x;
    hipError_t hip_rt;
    hip_rt = hipMalloc(&x, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    // Inititalize the data
    for(size_t i = 0; i < Nx; i++)
    {
        rdata[i] = i;
    }
    std::cout << "input:\n";
    for(size_t i = 0; i < rdata.size(); i++)
    {
        std::cout << rdata[i] << " ";
    }
    std::cout << std::endl;
    hip_rt = hipMemcpy(x, rdata.data(), real_bytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    // Create the plan
    hipfftHandle plan      = hipfft_params::INVALID_PLAN_HANDLE;
    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");
    hipfft_rt = hipfftPlan1d(&plan, // plan handle
                             Nx, // transform length
                             HIPFFT_D2Z, // transform type (HIPFFT_R2C for single-precision)
                             1); // number of transforms (deprecated)
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan1d failed");

    // Execute plan:
    // hipfftExecD2Z: double precision, hipfftExecR2C: for single-precision
    // Direction is implied by real-to-complex direction
    hipfft_rt = hipfftExecD2Z(plan, x, (hipfftDoubleComplex*)x);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftExecD2Z failed");

    std::cout << "output:\n";
    hip_rt = hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
    for(size_t i = 0; i < cdata.size(); i++)
    {
        std::cout << cdata[i] << " ";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);

    hip_rt = hipFree(x);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    return 0;
}
