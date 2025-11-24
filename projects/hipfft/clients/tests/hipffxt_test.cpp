// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipfft/hipfft.h"
#include "hipfft/hipfftXt.h"

#include <complex>
#include <gtest/gtest.h>


// FIXME: only on cuda?
DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

TEST(hipfftxttest, real_only_inplace)
{
    hipfftHandle plan;

    // FIXME
    
    size_t    ngpus = 2;
    const int Nx    = 1024;
    const int Ny    = 1024;

    auto fftret = HIPFFT_SUCCESS;
    
    fftret =  hipfftPlan2d(&plan,
                             Nx,
                             Ny,
                             HIPFFT_R2C);
    ASSERT_EQ(fftret, HIPFFT_SUCCESS);

    fftret = hipfftDestroy(plan);
    ASSERT_EQ(fftret, HIPFFT_SUCCESS);
}

class hipfftxtc2cinplace : public ::testing::TestWithParam<std::tuple<int, hipfftXtSubFormat>> {};

TEST_P(hipfftxtc2cinplace, formattest)
{
    size_t    ngpus = 2;
    const int Nx    = 1024;
    const int Ny    = 1024;

    auto hipfft_rt = HIPFFT_SUCCESS;

    //std::cout << "Example Test Param: " << GetParam() << std::endl;

    const int direction = std::get<0>(GetParam());
    const hipfftXtSubFormat informat = std::get<1>(GetParam());
    const hipfftXtSubFormat outformat = informat == HIPFFT_XT_FORMAT_INPLACE ? HIPFFT_XT_FORMAT_INPLACE_SHUFFLED : HIPFFT_XT_FORMAT_INPLACE;
    
    hipfftHandle plan;
    
    hipfft_rt =   hipfftCreate(&plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
 
    // We can re-use the same multiple times GPU to get a "multi-gpu" transform.
    std::vector<int> gpus(ngpus);
    std::fill(gpus.begin(), gpus.end(), 0);
    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    std::vector<size_t> workSize(ngpus);
    hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, HIPFFT_Z2Z, workSize.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipLibXtDesc*       inoutdesc = nullptr;
    hipfft_rt                     = hipfftXtMalloc(plan, &inoutdesc, informat);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    std::vector<std::complex<double>> cinput(Nx * Ny);
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            cinput[xidx * Ny + yidx] = std::complex<double>(xidx,yidx);
        }
    }
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(inoutdesc),
                               reinterpret_cast<void*>(cinput.data()),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    
    ASSERT_EQ(inoutdesc->subFormat, informat);

    hipfft_rt = hipfftXtExecDescriptor(plan, inoutdesc, inoutdesc, HIPFFT_FORWARD);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    ASSERT_EQ(inoutdesc->subFormat, outformat);

    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(cinput.data()),
                               reinterpret_cast<void*>(inoutdesc),
                               HIPFFT_COPY_DEVICE_TO_HOST);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    hipfft_rt = hipfftXtFree(inoutdesc);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    hipfft_rt = hipfftDestroy(plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(hipfftxttest,
                         hipfftxtc2cinplace,
                             ::testing::Combine(
                                 ::testing::Values(HIPFFT_FORWARD, HIPFFT_BACKWARD),
                                 ::testing::Values(HIPFFT_XT_FORMAT_INPLACE,
                                                   HIPFFT_XT_FORMAT_INPLACE)
                                 )
    );
                         
TEST(hipfftxttest, c2c_inplace_backward)
{
    size_t    ngpus = 2;
    const int Nx    = 1024;
    const int Ny    = 1024;

    auto hipfft_rt = HIPFFT_SUCCESS;

    hipfftHandle plan;
    
    hipfft_rt =   hipfftCreate(&plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    // We can re-use the same multiple times GPU to get a "multi-gpu" transform.
    std::vector<int> gpus(ngpus);
    std::fill(gpus.begin(), gpus.end(), 0);
    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    std::vector<size_t> workSize(ngpus);
    hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, HIPFFT_Z2Z, workSize.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipLibXtDesc*       inoutdesc = nullptr;
    hipfft_rt                     = hipfftXtMalloc(plan, &inoutdesc, HIPFFT_XT_FORMAT_INPLACE);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    std::vector<std::complex<double>> cinput(Nx * Ny);
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            cinput[xidx * Ny + yidx] = std::complex<double>(xidx,yidx);
        }
    }
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(inoutdesc),
                               reinterpret_cast<void*>(cinput.data()),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    
    ASSERT_EQ(inoutdesc->subFormat, HIPFFT_XT_FORMAT_INPLACE);

    hipfft_rt = hipfftXtExecDescriptor(plan, inoutdesc, inoutdesc, HIPFFT_BACKWARD);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    ASSERT_EQ(inoutdesc->subFormat, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED);

    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(cinput.data()),
                               reinterpret_cast<void*>(inoutdesc),
                               HIPFFT_COPY_DEVICE_TO_HOST);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    hipfft_rt = hipfftXtFree(inoutdesc);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    hipfft_rt = hipfftDestroy(plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}


TEST(hipfftxttest, batch_stuff)
{
    hipfftHandle plan;

    size_t    ngpus = 2;
    const int Nx    = 1024;
    const int Ny    = 1024;
    const int nbatch = 2;


    //FIXME
    
    auto fftret = HIPFFT_SUCCESS;
    
    fftret =  hipfftPlan2d(&plan,
                             Nx,
                             Ny,
                             HIPFFT_C2C);
    ASSERT_EQ(fftret, HIPFFT_SUCCESS);

    fftret = hipfftDestroy(plan);
    ASSERT_EQ(fftret, HIPFFT_SUCCESS);
}
