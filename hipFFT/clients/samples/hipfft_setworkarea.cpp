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
#include <hipfft/hipfft.h>
#include <iostream>
#include <vector>

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

#include "../hipfft_params.h"

int main()
{
    std::cout << "hipfft 1D single-precision real-to-complex transform showing "
                 "work memory usage\n";

    int major_version;
    hipfftGetProperty(HIPFFT_MAJOR_VERSION, &major_version);
    std::cout << "hipFFT major_version " << major_version << std::endl;

    const size_t N        = 9;
    const size_t Ncomplex = (N / 2 + 1);

    std::vector<float>               rdata(N);
    std::vector<std::complex<float>> cdata(Ncomplex);

    size_t real_bytes    = sizeof(decltype(rdata)::value_type) * rdata.size();
    size_t complex_bytes = sizeof(decltype(cdata)::value_type) * cdata.size();

    hipError_t   hip_rt    = hipSuccess;
    hipfftResult hipfft_rt = HIPFFT_SUCCESS;

    std::cout << "input:\n";
    for(size_t i = 0; i < N; i++)
    {
        rdata[i] = i;
    }
    for(size_t i = 0; i < N; i++)
    {
        std::cout << rdata[i] << " ";
    }
    std::cout << std::endl;

    // Create HIP device object.
    hipfftReal* x;
    hip_rt = hipMalloc(&x, real_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    hipfftComplex* y;
    hip_rt = hipMalloc(&y, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    // Copy input data to device
    hip_rt = hipMemcpy(x, rdata.data(), real_bytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    size_t workSize;
    hipfft_rt = hipfftEstimate1d(N, HIPFFT_R2C, 1, &workSize);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftEstimate1d failed");
    std::cout << "hipfftEstimate 1d workSize: " << workSize << std::endl;

    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    hipfft_rt         = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftCreate failed");
    hipfft_rt = hipfftSetAutoAllocation(plan, 0);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftSetAutoAllocation failed");
    hipfft_rt = hipfftMakePlan1d(plan, N, HIPFFT_R2C, 1, &workSize);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftMakePlan1d failed");

    // Set work buffer
    hipfftComplex* workBuf;
    hip_rt = hipMalloc(&workBuf, workSize);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    hipfft_rt = hipfftSetWorkArea(plan, workBuf);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftSetWorkArea failed");
    hipfft_rt = hipfftGetSize(plan, &workSize);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftGetSize failed");

    std::cout << "hipfftGetSize workSize: " << workSize << std::endl;

    // Execute plan
    hipfft_rt = hipfftExecR2C(plan, x, (hipfftComplex*)y);

    // Copy result back to host
    hip_rt = hipMemcpy(cdata.data(), y, complex_bytes, hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    std::cout << "output:\n";
    for(size_t i = 0; i < Ncomplex; i++)
    {
        std::cout << cdata[i] << " ";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);

    hip_rt = hipFree(x);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    hip_rt = hipFree(workBuf);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    return 0;
}
