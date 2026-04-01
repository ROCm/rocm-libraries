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
    std::cout << "hipfft 3D single-precision real-to-complex transform using "
                 "advanced interface\n";

    int rank    = 3;
    int n[3]    = {4, 5, 6};
    int howmany = 2;

    bool inplace = false;
    
    int n2_complex_elements      = n[2] / 2 + 1;
    int n2_padding_real_elements = inplace ? n2_complex_elements * 2 : n[2]; 

    int istride    = 1;
    int ostride    = 1;
    int inembed[3] = {n[0], n[1], n2_padding_real_elements};
    int onembed[3] = {n[0], n[1], n2_complex_elements};
    int idist      = istride * inembed[0] * inembed[1] * inembed[2];
    int odist      = ostride * onembed[0] * onembed[1] * onembed[2];

    const auto         total_inbytes = howmany * idist * sizeof(float);
    const auto         total_outbytes = inplace ? total_inbytes :
        howmany * odist * sizeof(std::complex<float>);
    
    std::cout << "rank :" << rank << "\n"
              << "n: " << n[0] << " " << n[1] << " " << n[2] << "\n"
              << "howmany: " << howmany << "\n"
              << "istride: " << istride << "\tostride: " << ostride << "\n"
              << "inembed: " << inembed[0] << " " << inembed[1] << " " << inembed[2] << "\n"
              << "onembed: " << onembed[0] << " " << inembed[1] << " " << onembed[2] << "\n"
              << "idist: " << idist << "\todist: " << odist << "\n"
              << "inbytes: " << total_inbytes << "\toutbytes: " << total_outbytes << "\n"
              << std::endl;

    std::cout << "input:\n";
    std::vector<float> indata(howmany * idist);
    std::fill(indata.begin(), indata.end(), 0.0);
    for(int idxb = 0; idxb < howmany; ++idxb)
    {
        for(int idx0 = 0; idx0 < n[0]; ++idx0)
        {
            for(int idx1 = 0; idx1 <  n[1]; ++idx1)
            {
                for(int idx2 = 0; idx2 < n[2]; ++idx2)
                {
                    const auto pos = idxb * idist
                        + istride * (idx2 + inembed[2] * (idx1 + inembed[1] * idx0));
                    indata[pos]      = idx0 + idx1 + idx2 + idxb;
                }
            }
        }
    }
    for(int idxb = 0; idxb < howmany; ++idxb)
    {
        std::cout << "batch: " << idxb << "\n";
        for(int idx0 = 0; idx0 < inembed[0]; ++idx0)
        {
            for(int idx1 = 0; idx1 <  inembed[1]; ++idx1)
            {
                for(int idx2 = 0; idx2 < inembed[2]; ++idx2)
                {
                    const auto pos = idxb * idist
                        + istride * (idx2 + inembed[2] * (idx1 + inembed[1] * idx0));
                    std::cout << indata[pos] << " ";
                }
                
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    hipfftHandle hipForwardPlan;
    hipfftResult hipfft_rt;
    hipfft_rt = hipfftPlanMany(&hipForwardPlan,
                               rank,
                               n,
                               inembed,
                               istride,
                               idist,
                               onembed,
                               ostride,
                               odist,
                               HIPFFT_R2C, // Use HIPFFT_D2Z for double-precsion.
                               howmany);
    std::cout << hipfft_rt << std::endl;
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");

    hipfftReal* gpu_indata;
    hipfftComplex* gpu_outdata;

    hipError_t hip_rt;
    hip_rt = hipMalloc((void**)&gpu_indata, total_inbytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    if(inplace)
    {
        gpu_outdata = nullptr;
    }
    else
    {
        hip_rt = hipMalloc((void**)&gpu_outdata, total_outbytes);
        if(hip_rt != hipSuccess)
            throw std::runtime_error("hipMalloc failed");
    }

    hip_rt = hipMemcpy(gpu_indata, (void*)indata.data(), total_inbytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    if(inplace)
        hipfft_rt = hipfftExecR2C(hipForwardPlan, gpu_indata, (hipfftComplex*)gpu_indata);
    else
        hipfft_rt = hipfftExecR2C(hipForwardPlan, gpu_indata, gpu_outdata);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to execute plan");

    std::vector<std::complex<float>> outdata(howmany * odist, 0.0);
    hip_rt = hipMemcpy((void*)outdata.data(),
                       inplace ? (void*)gpu_indata : (void*)gpu_outdata,
                       total_outbytes, hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    std::cout << "output:\n";
    for(int idxb = 0; idxb < howmany; ++idxb)
    {
        std::cout << "batch: " << idxb << "\n";
        for(int idx0 = 0; idx0 < n[0]; ++idx0)
        {
            for(int idx1 = 0; idx1 <  n[1]; ++idx1)
            {
                for(int idx2 = 0; idx2 < n2_complex_elements; ++idx2)
                {
                    const auto pos = idxb * odist
                        + ostride * (idx2 + onembed[2] * (idx1 + onembed[1] * idx0));
                    std::cout << outdata[pos] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    hipfftDestroy(hipForwardPlan);

    hip_rt = hipFree(gpu_indata);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    if(gpu_outdata)
    {
        hip_rt = hipFree(gpu_outdata);
        if(hip_rt != hipSuccess)
            throw std::runtime_error("hipFree failed");
    }
    
    return 0;
}
