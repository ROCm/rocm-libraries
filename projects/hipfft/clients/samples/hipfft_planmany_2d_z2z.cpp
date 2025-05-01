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

int main()
{
    std::cout << "hipfft 2D double-precision complex-to-complex transform using "
                 "advanced interface\n";

    int rank    = 2;
    int n[2]    = {4, 5};
    int howmany = 3;

    // array is contiguous in memory
    int istride = 1;
    // in-place transforms require istride=ostride
    int ostride = istride;

    // we choose to have no padding around our data:
    int inembed[2] = {istride * n[0], istride * n[1]};
    // in-place transforms require inembed=oneembed:
    int onembed[2] = {inembed[0], inembed[1]};

    int idist = inembed[0] * inembed[1];
    int odist = onembed[0] * onembed[1];

    std::cout << "n: " << n[0] << " " << n[1] << "\n"
              << "howmany: " << howmany << "\n"
              << "istride: " << istride << "\tostride: " << ostride << "\n"
              << "inembed: " << inembed[0] << " " << inembed[1] << "\n"
              << "onembed: " << onembed[0] << " " << onembed[1] << "\n"
              << "idist: " << idist << "\todist: " << odist << "\n"
              << std::endl;

    std::vector<std::complex<double>> data(howmany * idist);
    const auto total_bytes = data.size() * sizeof(decltype(data)::value_type);

    std::cout << "input:\n";
    std::fill(data.begin(), data.end(), 0.0);
    for(int ibatch = 0; ibatch < howmany; ++ibatch)
    {
        for(int i = 0; i < n[0]; i++)
        {
            for(int j = 0; j < n[1]; j++)
            {
                const auto pos = ibatch * idist + istride * (i * inembed[1] + j);
                data[pos]      = std::complex<double>(i + ibatch, j);
            }
        }
    }
    for(int ibatch = 0; ibatch < howmany; ++ibatch)
    {
        std::cout << "batch: " << ibatch << "\n";
        for(int i = 0; i < inembed[0]; i++)
        {
            for(int j = 0; j < inembed[1]; j++)
            {
                const auto pos = ibatch * idist + i * inembed[1] + j;
                std::cout << data[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    hipfftHandle hipPlan;
    hipfftResult hipfft_rt;
    hipfft_rt = hipfftPlanMany(
        &hipPlan, rank, n, inembed, istride, idist, onembed, ostride, odist, HIPFFT_Z2Z, howmany);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");

    hipError_t           hip_rt;
    hipfftDoubleComplex* d_in_out;
    hip_rt = hipMalloc((void**)&d_in_out, total_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    hip_rt = hipMemcpy(d_in_out, (void*)data.data(), total_bytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    hipfft_rt = hipfftExecZ2Z(hipPlan, d_in_out, d_in_out, HIPFFT_FORWARD);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to execute plan");

    hip_rt = hipMemcpy((void*)data.data(), d_in_out, total_bytes, hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    std::cout << "output:\n";
    for(int ibatch = 0; ibatch < howmany; ++ibatch)
    {
        std::cout << "batch: " << ibatch << "\n";
        for(int i = 0; i < onembed[0]; i++)
        {
            for(int j = 0; j < onembed[1]; j++)
            {
                const auto pos = ibatch * odist + i * onembed[1] + j;
                std::cout << data[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    hip_rt = hipFree(d_in_out);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    return 0;
}
