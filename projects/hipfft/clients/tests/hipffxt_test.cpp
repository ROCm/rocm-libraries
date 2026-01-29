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

// For test parameters (eg verbose)
#include "../../shared/accuracy_test.h"
#include "../hipfft_params.h"

// FIXME: only on cuda?
DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

std::string formatname(const hipfftXtSubFormat format)
{
    switch(format)
    {
    case HIPFFT_XT_FORMAT_INPUT:
        return "HIPFFT_XT_FORMAT_INPUT";
    case HIPFFT_XT_FORMAT_OUTPUT:
        return "HIPFFT_XT_FORMAT_OUTPUT";
    case HIPFFT_XT_FORMAT_INPLACE:
        return "HIPFFT_XT_FORMAT_INPLACE";
    case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
        return "HIPFFT_XT_FORMAT_INPLACE_SHUFFLED";
    case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
        return "HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED";
    case HIPFFT_FORMAT_UNDEFINED:
        return "HIPFFT_FORMAT_UNDEFINED";
    }
}

std::string hipffttype_to_name(const hipfftType txtype )
{
    switch(txtype)
    {
    case HIPFFT_R2C:
        return "HIPFFT_R2C";
    case HIPFFT_C2R:
        return "HIPFFT_C2R";
    case HIPFFT_C2C:
        return "HIPFFT_C2C";
    case HIPFFT_D2Z:
        return "HIPFFT_D2Z";
    case HIPFFT_Z2D:
        return "HIPFFT_Z2D";
    case HIPFFT_Z2Z:
        return "HIPFFT_Z2Z";
    }
}

std::string directionname(const int direction)
{
    switch(direction)
    {
    case HIPFFT_FORWARD:
        return "HIPFFT_FORWARD";
    case HIPFFT_BACKWARD:
        return "HIPFFT_BACKWARD";
    }
}

// Params are direction, format, and batch size.
class hipfftxtdirectionformat : public ::testing::TestWithParam<std::tuple<int, hipfftXtSubFormat,
                                                                           int>>
{};

TEST_P(hipfftxtdirectionformat, c2cinplace)
{
    size_t    ngpus = 2;
    const int Nx    = 1024;
    const int Ny    = 1024;

    auto hipfft_rt = HIPFFT_SUCCESS;

    const int direction = std::get<0>(GetParam());
    const hipfftXtSubFormat informat = std::get<1>(GetParam());
    const int batch = std::get<2>(GetParam());
    
    const hipfftXtSubFormat outformat
        = informat == HIPFFT_XT_FORMAT_INPLACE
        ? HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
        : HIPFFT_XT_FORMAT_INPLACE;

    if(verbose > 0)
    {
        std::cout << "complex-to-complex direction: " << directionname(direction)
                  << " input format: " << formatname(informat)
                  << " output format: " << formatname(outformat)
                  << "\n";
    }
    
    hipfftHandle plan;
    hipfft_rt =   hipfftCreate(&plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
 
    // We can re-use the same multiple times GPU to get a "multi-gpu" transform.
    std::vector<int> gpus(ngpus);
    std::fill(gpus.begin(), gpus.end(), 0);
    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    std::vector<size_t> workSize(ngpus);
    hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, HIPFFT_Z2Z, workSize.data());
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipLibXtDesc*       inoutdesc = nullptr;
    hipfft_rt                     = hipfftXtMalloc(plan, &inoutdesc, informat);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    std::vector<std::complex<double>> input(Nx * Ny);
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            input[xidx * Ny + yidx] = std::complex<double>(xidx,yidx);
        }
    }
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(inoutdesc),
                               reinterpret_cast<void*>(input.data()),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    EXPECT_EQ(inoutdesc->subFormat, informat);

    hipfft_rt = hipfftXtExecDescriptor(plan, inoutdesc, inoutdesc, direction);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    EXPECT_EQ(inoutdesc->subFormat, outformat);
    
    std::vector<std::complex<double>> output(Nx * Ny);
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(output.data()),
                               reinterpret_cast<void*>(inoutdesc),
                               HIPFFT_COPY_DEVICE_TO_HOST);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    hipfft_rt = hipfftXtFree(inoutdesc);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipfft_rt = hipfftDestroy(plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    hipfftxttest,
    hipfftxtdirectionformat,
    ::testing::Combine(
        ::testing::Values(HIPFFT_FORWARD, HIPFFT_BACKWARD),
        ::testing::Values(HIPFFT_XT_FORMAT_INPLACE,
                          HIPFFT_XT_FORMAT_INPLACE_SHUFFLED),
        ::testing::Values(1, 2)
        ),
    [](const testing::TestParamInfo<hipfftxtdirectionformat::ParamType>& info) {
        const int direction = std::get<0>(info.param);
        const hipfftXtSubFormat informat = std::get<1>(info.param);
        std::string name = direction == HIPFFT_FORWARD ? "forward" : "backward";
        name += informat == HIPFFT_XT_FORMAT_INPLACE ? "inplace" : "shuffled";
        name += "batch" + std::to_string(std::get<2>(info.param));
        return name;
    }
    );

TEST_P(hipfftxtdirectionformat, r2cinplace)
{
    size_t    ngpus = 2;
    const int Nx    = 32;
    const int Ny    = 32;

    const int Nyp = Ny / 2 + 1;
    const int Nypp = Ny + 2;
    
    auto hipfft_rt = HIPFFT_SUCCESS;
    
    const int direction = std::get<0>(GetParam());
    const hipfftXtSubFormat informat = std::get<1>(GetParam());
    const int batch = std::get<2>(GetParam());

    // Skip the unhappy paths
    if(direction == HIPFFT_FORWARD && batch == 1 && informat != HIPFFT_XT_FORMAT_INPLACE)
    {
        GTEST_SKIP();
    }
    if(direction == HIPFFT_BACKWARD && batch == 1 && informat != HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
    {
        GTEST_SKIP();
    }
    if(direction == HIPFFT_FORWARD && batch > 1 && informat != HIPFFT_XT_FORMAT_INPLACE)
    {
        GTEST_SKIP();
    }
    if(direction == HIPFFT_BACKWARD && batch > 1)
    {
        GTEST_SKIP();
    }
    if(direction == HIPFFT_BACKWARD)
    {
        GTEST_SKIP();
    }
        
    
    hipfftXtSubFormat outformat;
    if(batch == 1)
    {
        outformat = informat == HIPFFT_XT_FORMAT_INPLACE
            ? HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
            : HIPFFT_XT_FORMAT_INPLACE;
    }
    else
    {
        outformat = informat;
    }

    const hipfftType transform_type  = (direction == HIPFFT_FORWARD) ? HIPFFT_D2Z : HIPFFT_Z2D;
    
    if(verbose > 0)
    {
        std::cout << "hipfftxt format change test\n";
        std::cout << "\tNx: " << Nx << "\n";
        std::cout << "\tNy: " << Ny << "\n";
        std::cout << "\tngpus: " << ngpus << "\n";
        std::cout << "\ttransform_type: " << transform_type << " : "
                  << hipffttype_to_name(transform_type) << "\n";
        std::cout << "\tdirection: " << direction << " : " << directionname(direction)
                  << "\n\tinput subformat: " << informat << " : " << formatname(informat)
                  << "\n\toutput subformat: " << outformat << " : " << formatname(outformat)
                  << "\n";
    }
    
    hipfftHandle plan;
    hipfft_rt =   hipfftCreate(&plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    if(verbose > 1)
    {
        std::cout << "direction: " << directionname(direction)
                  << " informat: " << formatname(informat)
                  << " batch: " << batch << "\n";
    }

    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtSetGPUs failed";
        
    std::vector<size_t> workSize(ngpus);

    if(batch > 1)
    {
        int rank = 2;
        int n[2] = {Nx, Ny};
    
        int n1_complex_elements      = n[1] / 2 + 1;
        int n1_padding_real_elements = n1_complex_elements * 2;

        int istride    = 1;
        int ostride    = istride;
        int inembed[2] = {n[0],
                          direction == HIPFFT_FORWARD
                          ? n1_padding_real_elements
                          : n1_complex_elements};
        int onembed[2] = {n[0],
                          direction == HIPFFT_FORWARD
                          ? n1_complex_elements
                          : n1_padding_real_elements}; 
        int idist      = istride * inembed[0] * inembed[1];
        int odist      = ostride * onembed[0] * onembed[1];

        // NB: it seems that cufftxt will treat the batch=1 hipfftPlanMany case as batched, so the
        // data decomposition is trivial if one calls hipfftPlanMany (even if batch=1).
        hipfft_rt = hipfftPlanMany(&plan,
                                   rank,
                                   n,
                                   inembed,
                                   istride,
                                   idist,
                                   onembed,
                                   ostride,
                                   odist,
                                   transform_type,
                                   batch);
        EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftPlanMany failed";
    }
    else
    {
        hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny,
                                     transform_type,
                                     workSize.data());
        EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftMakePlan2d failed";
    }


    hipLibXtDesc*       inoutdesc = nullptr;
    hipfft_rt                     = hipfftXtMalloc(plan, &inoutdesc, informat);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc failed";

    std::vector<double> real(Nx * Nypp);
    std::vector<std::complex<double>> complex(Nx * Nyp);

    if(direction == HIPFFT_FORWARD)
    {
        for(size_t xidx = 0; xidx < Nx; ++xidx)
        {
            for(size_t yidx = 0; yidx < Ny; ++yidx)
            {
                const size_t pos = xidx * Nypp + yidx;
                const size_t idx = xidx * Ny + yidx;
                real[pos] = idx;
            }
        }
    }
    else
    {
        for(size_t xidx = 0; xidx < Nx; ++xidx)
        {
            for(size_t yidx = 0; yidx < Nyp; ++yidx)
            {
                const size_t pos = xidx * Nyp + yidx;
                complex[pos] = std::complex<double>(xidx, yidx);
            }
        }
    }
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(inoutdesc),
                               direction == HIPFFT_FORWARD
                               ? reinterpret_cast<void*>(real.data())
                               : reinterpret_cast<void*>(complex.data()),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    EXPECT_EQ(inoutdesc->subFormat, informat)
        << "informat not what expected:"
        << " got " << formatname((hipfftXtSubFormat)inoutdesc->subFormat)
        << " expected " << formatname((hipfftXtSubFormat)informat);

    hipfft_rt = hipfftXtExecDescriptor(plan, inoutdesc, inoutdesc, direction);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "exec failed"; 

    EXPECT_EQ(inoutdesc->subFormat, outformat)
        << "outformat not what expected:"
        << " got " << inoutdesc->subFormat << " "
        << formatname((hipfftXtSubFormat)inoutdesc->subFormat)
        << " expected "  << outformat << " "
        << formatname((hipfftXtSubFormat)outformat);
    
    hipfft_rt = hipfftXtMemcpy(plan,
                               direction == HIPFFT_FORWARD
                               ? reinterpret_cast<void*>(complex.data())
                               : reinterpret_cast<void*>(real.data()),
                               reinterpret_cast<void*>(inoutdesc),
                               HIPFFT_COPY_DEVICE_TO_HOST);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
        

    hipfft_rt = hipfftXtFree(inoutdesc);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipfft_rt = hipfftDestroy(plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}
