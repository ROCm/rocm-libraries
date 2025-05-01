// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights
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

#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>

#include "../hipfft_params.h"

struct load_cbdata
{
    hipfftDoubleComplex* filter;
    double               scale;
};

__device__ hipfftDoubleComplex load_callback(hipfftDoubleComplex* input,
                                             size_t               offset,
                                             void*                cbdata,
                                             void*                sharedMem)
{
    auto data = static_cast<load_cbdata*>(cbdata);

    // NB: for optimal performance, one may need a custom
    // multiplication operator.
    return hipCmul(hipCmul(input[offset], data->filter[offset]),
                   make_hipDoubleComplex(data->scale, 0));
}

__device__ auto load_callback_dev = load_callback;

int main()
{
    std::cout << "hipfft 1D double-precision complex-to-complex transform with callback\n";

    const int Nx        = 8;
    int       direction = HIPFFT_FORWARD; // forward=-1, backward=1

    std::vector<hipfftDoubleComplex> cdata(Nx), filter(Nx);
    size_t complex_bytes = sizeof(decltype(cdata)::value_type) * cdata.size();

    // Create HIP device object and copy data to device
    // Use hipfftComplex for single-precision
    hipError_t           hip_rt;
    hipfftDoubleComplex *x, *filter_dev;
    hip_rt = hipMalloc(&x, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    hip_rt = hipMalloc(&filter_dev, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    // Initialize the data and filter
    for(size_t i = 0; i < Nx; i++)
    {
        cdata[i].x  = i;
        cdata[i].y  = i;
        filter[i].x = rand() / static_cast<double>(RAND_MAX);
        filter[i].y = 0;
    }
    hip_rt = hipMemcpy(x, cdata.data(), complex_bytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
    hip_rt = hipMemcpy(filter_dev, filter.data(), complex_bytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
    std::cout << "input:\n";
    for(size_t i = 0; i < cdata.size(); i++)
    {
        std::cout << "(" << cdata[i].x << ", " << cdata[i].y << ") ";
    }
    std::cout << std::endl;

    // Create the plan
    hipfftHandle plan      = hipfft_params::INVALID_PLAN_HANDLE;
    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");
    hipfft_rt = hipfftPlan1d(&plan, // plan handle
                             Nx, // transform length
                             HIPFFT_Z2Z, // transform type (HIPFFT_C2C for single-precision)
                             1); // number of transforms
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan1d failed");

    // prepare callback
    load_cbdata cbdata_host;
    cbdata_host.filter = filter_dev;
    cbdata_host.scale  = 1.0 / static_cast<double>(Nx);
    void* cbdata_dev;
    hip_rt = hipMalloc(&cbdata_dev, sizeof(load_cbdata));
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    hip_rt = hipMemcpy(cbdata_dev, &cbdata_host, sizeof(load_cbdata), hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    void* cbptr_host = nullptr;
    hip_rt = hipMemcpyFromSymbol(&cbptr_host, HIP_SYMBOL(load_callback_dev), sizeof(void*));
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpyFromSymbol failed");

    // set callback
    hipfft_rt = hipfftXtSetCallback(plan, &cbptr_host, HIPFFT_CB_LD_COMPLEX_DOUBLE, &cbdata_dev);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtSetCallback failed");

    // Execute plan:
    // hipfftExecZ2Z: double precision, hipfftExecC2C: for single-precision
    hipfft_rt = hipfftExecZ2Z(plan, x, x, direction);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftExecZ2Z failed");

    std::cout << "output:\n";
    hip_rt = hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
    for(size_t i = 0; i < cdata.size(); i++)
    {
        std::cout << "(" << cdata[i].x << ", " << cdata[i].y << ") ";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);

    hip_rt = hipFree(cbdata_dev);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    hip_rt = hipFree(filter_dev);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    hip_rt = hipFree(x);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipFree failed");

    return 0;
}
