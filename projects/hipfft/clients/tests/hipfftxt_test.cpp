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

#ifdef __HIP_PLATFORM_NVIDIA__
DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#endif
#include <hip/hip_runtime_api.h>
#ifdef __HIP_PLATFORM_NVIDIA__
DISABLE_WARNING_POP
#endif

std::string transform_type_name(const fft_transform_type transform_type)
{
    switch(transform_type)
    {
    case fft_transform_type_complex_forward:
        return "fft_transform_type_complex_forward";
    case fft_transform_type_complex_inverse:
        return "fft_transform_type_complex_inverse";
    case fft_transform_type_real_forward:
        return "fft_transform_type_real_forward";
    case fft_transform_type_real_inverse:
        return "fft_transform_type_real_inverse";
    default:
        return "Invalid transform value";
    }
}

std::string fft_io_name(const fft_io io)
{
    switch(io)
    {
    case fft_io_in:
        return "fft_io_in";
    case fft_io_out:
        return "fft_io_out";
    default:
        return "Invalid fft_io value";
    }
}

std::string fft_result_placement_name(const fft_result_placement placement)
{
    switch(placement)
    {
    case fft_placement_inplace:
        return "fft_placement_inplace";
    case fft_placement_notinplace:
        return "fft_placement_notinplace";
    default:
        return "Invalid fft_result_placement value";
    }
}

std::string format_name(const int format)
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
    default:
        return "Unknown format";
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

// Params are direction and real/complex
class hipfftxtunit : public ::testing::TestWithParam<std::tuple<int, bool>>
{};

TEST_P(hipfftxtunit, plancreation)
{
    // Test whether we can just make plans.
    
    size_t    ngpus = 2;

    // Just batch=1 for now.

    // FIXME: handle 3D as well.
    
    const int Nx    = 32;
    const int Ny    = 32;

    const int direction = std::get<0>(GetParam());
    const bool realcomplex = std::get<1>(GetParam());
    
    const hipfftType transform_type  = realcomplex ?
        ((direction == HIPFFT_FORWARD) ? HIPFFT_D2Z : HIPFFT_Z2D) : HIPFFT_Z2Z;

    if(verbose > 0)
    {
        std::cout << "hipfftxt plan creation test: " << directionname(direction)
                  << (realcomplex ? " real/complex" : "complex/complex")
                  << "\n";
    }

    auto hipfft_rt = HIPFFT_SUCCESS;
        
    hipfftHandle plan;
    hipfft_rt =   hipfftCreate(&plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    
    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtSetGPUs failed";

    std::vector<size_t> workSize(ngpus);
    hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny,
                                 transform_type,
                                 workSize.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftMakePlan2d failed with return code "
                                         << hipfft_rt << "=" << hipfftResult_string(hipfft_rt);

    hipfft_rt = hipfftDestroy(plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}


INSTANTIATE_TEST_SUITE_P(
    hipfftxttest,
    hipfftxtunit,
    ::testing::Combine(
        ::testing::Values(HIPFFT_FORWARD, HIPFFT_BACKWARD),
        ::testing::Values(true, false)
        ),
    [](const testing::TestParamInfo<hipfftxtunit::ParamType>& info) {
        const int direction = std::get<0>(info.param);
        const int realcomplex = std::get<1>(info.param);
        std::string name = direction == HIPFFT_FORWARD ? "forward" : "backward";
        name += realcomplex ? "rc" : "cc";
        return name;
    }
    );


class hipfftxtunitdesc : public ::testing::TestWithParam<std::tuple<bool, int, hipfftXtSubFormat, size_t>>
{};

// Real/complex, tx direction, subformat, dimension
// FIXME: rename and document
static std::vector<std::tuple<bool, int, hipfftXtSubFormat, size_t>> in_goodlist =
{
    // Real/complex must be inplace 
    {1, HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE, 2},
    {1, HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED, 2},
    {1, HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE, 3},
    {1, HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED, 3},
    // Complex/complex can be in-place or out-of-place
    {0, HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE, 2},
    {0, HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED, 2},
    {0, HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE, 3},
    {0, HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED, 3},

    // It seems that cufftxt only wants to do in-place, even for c2c?
    //{0, HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPUT, 2},
    //{0, HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_OUTPUT, 2}
};

// FIXME: rename this to also mention it does a H2D copy.
TEST_P(hipfftxtunitdesc, desccreation)
{
    const bool realcomplex = std::get<0>(GetParam());
    const int direction = std::get<1>(GetParam());
    const hipfftXtSubFormat format = std::get<2>(GetParam());
    const size_t dimension = std::get<3>(GetParam());

    const int Nx    = 32;
    const int Ny    = 36;
    const int Nz    = 38;
    // Just batch=1 for now.
    
    if(verbose > 0)
    {
        std::cout << "hipfftxt plan creation test: " << directionname(direction)
                  << (realcomplex ? " real/complex" : "complex/complex")
                  << " dimension " << dimension << "\n";
        std::cout << "Nx: " << Nx << " Ny: " << Ny;
        if(dimension == 3)
            std::cout << " Nz: " << Nz;
        std::cout << "\n";
    }

    // FIXME: handle variable number of GPUs
    size_t    ngpus = 2;
    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    
    // TODO: other sizes, batch, etc.
    std::vector<size_t> batches = {1};
    std::vector<size_t> lengths =  {Nx, Ny};
    if(dimension == 3)
        lengths.push_back(Nz);
    std::vector<size_t> batchlengths = batches;
    batchlengths.insert(batchlengths.end(), lengths.begin(), lengths.end());

    // Some facts about the test case:
    const bool forward = (direction == HIPFFT_FORWARD);
    const bool isreal = realcomplex ? ( format == HIPFFT_XT_FORMAT_INPLACE  ) : false;
    const bool isherm = realcomplex ? ( format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED  ) : false;
    const size_t lastdim = batchlengths.size() - 1;
    const bool isinput = format == HIPFFT_XT_FORMAT_INPUT || format == HIPFFT_XT_FORMAT_INPLACE;
    const size_t splitdim = isinput ? 1 : 2;

    if(verbose > 2)
    {
        std::cout << "lastdim: " << lastdim << "\n";
        std::cout << "splitdim: " << splitdim << "\n";
    }
        
    // fft_enums configuration
    const fft_transform_type dft_type = realcomplex
        ? (forward ? fft_transform_type_real_forward : fft_transform_type_real_inverse)
        : (forward ? fft_transform_type_complex_forward : fft_transform_type_complex_inverse);
    const fft_result_placement placement
        = (format == HIPFFT_XT_FORMAT_INPLACE
           || format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
        ? fft_placement_inplace : fft_placement_notinplace;
    const fft_io io = (forward
                       != (format == HIPFFT_XT_FORMAT_INPUT || format == HIPFFT_XT_FORMAT_INPLACE))
        ? fft_io_out : fft_io_in;

    // hipfftxt configuratin:
    const hipfftType transform_type  = realcomplex
        ? (forward ? HIPFFT_D2Z : HIPFFT_Z2D) : HIPFFT_Z2Z;

    // FIXME: document
    auto computedatabatchlengths = [](const bool isherm,
                                   const std::vector<size_t> &batchlengths) -> std::vector<size_t>
    {
        std::vector<size_t> newbatchlengths = batchlengths;
        const size_t lastdim = batchlengths.size() - 1;
        if(isherm)
            newbatchlengths[lastdim] = newbatchlengths[lastdim] / 2 + 1;
        return newbatchlengths;
    };
    
    // Host data configuration:
    const auto host_distances = default_distances(dft_type, placement, io, lengths, batches);
    auto hostdiststrides = host_distances;
    const auto hostdatabatchlengths = computedatabatchlengths(isherm, batchlengths);
    const auto host_strides = default_strides(dft_type, placement, io, lengths);
    hostdiststrides.insert(hostdiststrides.end(), host_strides.begin(), host_strides.end());
    if(verbose > 1)
    {
        std::cout << "dft_type: " << transform_type_name(dft_type) << "\n";
        std::cout << "placement: " << fft_result_placement_name(placement) << "\n";
        std::cout << "io: " << fft_io_name(io) << "\n";
        std::cout << "transform batch/length:";
        for(auto val : batchlengths)
            std::cout << " " << val;
        std::cout << "\n";
        std::cout << "host dist/strides:";
        for(auto val : hostdiststrides)
            std::cout << " " << val;
        std::cout << "\n";
    }    

    // Create the xt plan and descriptor:
    auto hipfft_rt = HIPFFT_SUCCESS;

    hipfftHandle plan;
    hipfft_rt =   hipfftCreate(&plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtSetGPUs failed";

    std::vector<size_t> workSize(ngpus);
    switch(dimension)
    {
    case 2:
        hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny,
                                     transform_type,
                                     workSize.data());
        break;
    case 3:
        hipfft_rt = hipfftMakePlan3d(plan, Nx, Ny, Nz,
                                     transform_type,
                                     workSize.data());
        break;
    default:
        FAIL() << "Test infrastructure only supports 2D and 3D transforms";
    }
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftMakePlan2/3d failed with return code "
                                         << hipfft_rt << "=" << hipfftResult_string(hipfft_rt);
    if(verbose > 2)
        std::cout << "plan created\n";
    
    hipLibXtDesc*       mydesc = nullptr;
    hipfft_rt                     = hipfftXtMalloc(plan, &mydesc, format);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc failed with code "
                                         << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";
    if(verbose > 2)
        std::cout << "descriptor allocated\n";

    for(size_t igpu = 0; igpu < gpus.size(); ++igpu)
    {
        // TODO: handle case where some GPUs don't have data because there isn't enough to go
        // around.  (Particularly for multi-batch cases.)
        ASSERT_NE(mydesc->descriptor->size[igpu], 0) << "gpu buffer size is zero for gpu " << igpu;
    }
    
    auto printhostbuf = [](const char* hostbuf,
                           const bool isreal,
                           std::vector<size_t> batchlengths,
                           const std::vector<size_t> &hostdiststrides) -> void  {
        switch(batchlengths.size())
        {
        case 3:
            // 1 batch + 2D FFT
            for(size_t ibatch = 0; ibatch < batchlengths[0]; ++ibatch)
            {
                for(size_t xidx = 0; xidx < batchlengths[1]; ++xidx)
                {
                    for(size_t yidx = 0; yidx < batchlengths[2]; ++yidx)
                    {
                        const std::vector<size_t> idx = {ibatch, xidx, yidx};
                        const size_t pos = std::inner_product(std::begin(idx),
                                                              std::end(idx),
                                                              std::begin(hostdiststrides), 0);
                        if(isreal) {
                            const auto hostdat = reinterpret_cast<const double*>(hostbuf);
                            if(yidx > 0)
                                std::cout << " ";
                            std::cout << hostdat[pos];
                        }
                        else
                        {
                            const auto hostdat
                                = reinterpret_cast<const std::complex<double>*>(hostbuf);
                            if(yidx > 0)
                                std::cout << " ";
                            std::cout << hostdat[pos];
                        }
                    }
                    std::cout << "\n";
                }
            }
            break;
        case 4:
            // 1 batch + 3D FFT
            for(size_t ibatch = 0; ibatch < batchlengths[0]; ++ibatch)
            {
                for(size_t xidx = 0; xidx < batchlengths[1]; ++xidx)
                {
                    for(size_t yidx = 0; yidx < batchlengths[2]; ++yidx)
                    {
                        for(size_t zidx = 0; zidx < batchlengths[3]; ++zidx)
                        {
                            const std::vector<size_t> idx = {ibatch, xidx, yidx, zidx};
                            const size_t pos = std::inner_product(std::begin(idx),
                                                                  std::end(idx),
                                                                  std::begin(hostdiststrides), 0);
                            if(isreal) {
                                const auto hostdat = reinterpret_cast<const double*>(hostbuf);
                                if(zidx > 0)
                                    std::cout << " ";
                                std::cout << hostdat[pos];
                            }
                            else
                            {
                                const auto hostdat
                                    = reinterpret_cast<const std::complex<double>*>(hostbuf);
                                if(zidx > 0)
                                    std::cout << " ";
                                std::cout << hostdat[pos];
                            }
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
            }
            break;
        default:
            FAIL() << "dimension not handled";
        }
    };
        
    // Initialize desc buffers to zero:
    for(const auto igpu : gpus)
    {
        //const auto device = mydesc->descriptor->GPUs[igpu];
        const auto bufsize = mydesc->descriptor->size[igpu];
        auto devbuf = mydesc->descriptor->data[igpu];
        auto hipret = hipMemset(devbuf, bufsize, 0);
        EXPECT_EQ(hipret, hipSuccess) << "hipMemset failed";
        std::vector<char> hostbufpart(bufsize);
        hipret = hipMemcpy(hostbufpart.data(), devbuf, bufsize, hipMemcpyDeviceToHost);
        EXPECT_EQ(hipret, hipSuccess) << "hipMemcpy failed";
    }

    // Labmda for initializing the host buffer.  We do not care about Hermitian symmetry in 2D/3D,
    // as we are just testing data movement, not transforms.
    auto fillhostbuf = [](std::vector<char> &hostbuf,
                          const bool isreal,
                          std::vector<size_t> batchlengths,
                          const std::vector<size_t> &hostdiststrides) -> void  {
        switch(batchlengths.size())
        {
        case 3:
            // 1 batch + 2D FFT
            for(size_t ibatch = 0; ibatch < batchlengths[0]; ++ibatch)
            {
                for(size_t xidx = 0; xidx < batchlengths[1]; ++xidx)
                {
                    for(size_t yidx = 0; yidx < batchlengths[2]; ++yidx)
                    {
                        const std::vector<size_t> idx = {ibatch, xidx, yidx};
                    
                        const size_t pos = std::inner_product(std::begin(idx),
                                                              std::end(idx),
                                                              std::begin(hostdiststrides), 0);
                        if(isreal) {
                            auto hostdat = reinterpret_cast<double*>(hostbuf.data());
                            hostdat[pos] = xidx + 0.01 * yidx;
                        }
                        else
                        {
                            auto hostdat
                                = reinterpret_cast<std::complex<double>*>(hostbuf.data());
                            hostdat[pos] = std::complex<double>(xidx, yidx);
                        }
                    }
                }
            }
            break;
        case 4:
            // 1 batch + 3D FFT
            for(size_t ibatch = 0; ibatch < batchlengths[0]; ++ibatch)
            {
                for(size_t xidx = 0; xidx < batchlengths[1]; ++xidx)
                {
                    for(size_t yidx = 0; yidx < batchlengths[2]; ++yidx)
                    {
                        for(size_t zidx = 0; zidx < batchlengths[3]; ++zidx)
                        {

                            const std::vector<size_t> idx = {ibatch, xidx, yidx, zidx};
                            
                            const size_t pos = std::inner_product(std::begin(idx),
                                                                  std::end(idx),
                                                                  std::begin(hostdiststrides), 0);
                            if(isreal) {
                                auto hostdat = reinterpret_cast<double*>(hostbuf.data());
                                hostdat[pos] = xidx + 0.01 * yidx +  + 0.0001 * zidx;
                            }
                            else
                            {
                                auto hostdat
                                    = reinterpret_cast<std::complex<double>*>(hostbuf.data());
                                hostdat[pos] = std::complex<double>(xidx + 0.01 * yidx, zidx);
                            }
                        }
                    }
                }
            }
            break;
        default:
            FAIL() << "dimension not handled";
        }
    };

    // FIXME: document.  This is the per-gpu-buffer lengths.
    auto devbatchlength = [splitdim](const size_t ngpus,
                                     const std::vector<size_t> &hostdatabatchlengths,
                                     const size_t igpu) -> std::vector<size_t>
        {
            std::vector<size_t> databatchlengths = hostdatabatchlengths;
            const auto l = databatchlengths[splitdim];
            databatchlengths[splitdim] = l / ngpus + ((igpu < l % ngpus) ? 1 : 0);
            return databatchlengths;
        };
    
    // Return a vector containing {gpu index, batch index, transform indices...}.
    // Batch and transform indices are buffer-local multi-index (ie relative to an index starting at
    // {0, ... , 0} on each brick).
    auto devidx = [splitdim](const size_t ngpus,
                             const std::vector<size_t> &hostidx,
                             const std::vector<size_t> &databatchlengths) -> std::vector<size_t> {
        std::vector<size_t> ret(databatchlengths.size() + 1, 0);
        for(size_t idx = 0; idx < hostidx.size(); ++idx)
        {
            if(idx != splitdim)
                ret[idx + 1] = hostidx[idx];
        }
        const auto l = databatchlengths[splitdim];
        //std::cout << databatchlengths[splitdim] << std::endl;
        const auto b = l / ngpus; // Elements per gpu in splitdim (if no remainder).
        const auto r = l - b * ngpus; // Remainder
        //std::cout << r << std::endl;
        
        const auto a = hostidx[splitdim];
        if(a < r * (b + 1))
        {
            ret[0] = a / (b + 1);
            ret[splitdim + 1] = a - ret[0] * (b + 1);
        }
        else
        {
            ret[0] = r +  (a - r * (b + 1)) / b;
            ret[splitdim + 1] = a - r * (b + 1) - (ret[0] - r) * b;
        }
        
        return ret;
    };
    
    // Fine, let's do a copy test and see what happens.

    // We test the host-to-device copy (since we can easily set up the host data and then check that
    // the buffers have values where we expect them to be).
    const auto copydir = HIPFFT_COPY_HOST_TO_DEVICE;
        
    const size_t nelem = hostdiststrides[0] * hostdatabatchlengths[0];
    const size_t valsize = isreal ? sizeof(double) : sizeof(std::complex<double>);
    std::vector<char> hostbuf(valsize * nelem);
    fillhostbuf(hostbuf, isreal, hostdatabatchlengths, hostdiststrides);

    if(verbose > 4)
    {
        printhostbuf(hostbuf.data(), isreal, hostdatabatchlengths, hostdiststrides);
    }
        
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(mydesc),
                               reinterpret_cast<void*>(hostbuf.data()),
                               copydir);

    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMemcpy H2D"
                                         << " failed with code "
                                         << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";

    if(verbose > 2)
        std::cout << "finished hipfftXtMemcpy\n";

    // A host copy of the individual distributed GPU buffers:
    std::vector<std::vector<char>> hostbufparts(gpus.size());

    // Copy the individual buffers to the host:
    for(const auto igpu : gpus)
    {
        if(verbose > 3)
            std::cout << "buffer " << igpu << " after xtmemcp\n";
        const auto device = mydesc->descriptor->GPUs[igpu];
        const auto bufsize = mydesc->descriptor->size[igpu];
        ASSERT_NE(bufsize, 0) << "gpu buffer size is zero for gpu " << igpu;
        if(verbose > 3)
        {
            std::cout << "device: " << device << "\n";
            std::cout << "buffer size: " << bufsize << "\n";
        }
        hostbufparts[igpu].resize(bufsize);
        auto devbuf = mydesc->descriptor->data[igpu];
        auto hipret = hipMemcpy(hostbufparts[igpu].data(), devbuf, bufsize,
                                hipMemcpyDeviceToHost);
        EXPECT_EQ(hipret, hipSuccess) << "hipMemcpy failed";
    }

    // Each brick gets it own special set of strides.
    std::vector<std::vector<size_t>> brick_batchlengths(gpus.size());
    std::vector<std::vector<size_t>> brick_diststrides(gpus.size());

    for(size_t igpu = 0; igpu < gpus.size(); ++igpu)
    {
        brick_batchlengths[igpu] = devbatchlength(ngpus, hostdatabatchlengths, igpu);
        std::vector<size_t> brick_batches;
        brick_batches.insert(brick_batches.end(),
                             brick_batchlengths[igpu].begin(),
                             brick_batchlengths[igpu].begin() +1);
        std::vector<size_t> brick_lengths;
        brick_lengths.insert(brick_lengths.end(),
                             brick_batchlengths[igpu].begin() + 1,
                             brick_batchlengths[igpu].end());

        std::vector<size_t> brick_distances;
        std::vector<size_t> brick_strides;
        if(isherm)
        {
            brick_distances = default_distances(fft_transform_type_complex_forward,
                                                placement, io,
                                                brick_lengths, brick_batches);
            brick_strides = default_strides(fft_transform_type_complex_forward,
                                            placement, io, brick_lengths);

        }
        else
        {
            brick_distances = default_distances(dft_type, placement, io,
                                                brick_lengths, brick_batches);
            brick_strides = default_strides(dft_type, placement, io, brick_lengths);
        }
                
        brick_diststrides[igpu] = brick_distances;
        brick_diststrides[igpu].insert(brick_diststrides[igpu].end(), brick_strides.begin(),
                                       brick_strides.end());
    }

    if(verbose > 2)
    {
        std::cout << "gpu buffer length and dist/strides:\n";
        for(size_t igpu=0; igpu < gpus.size(); ++igpu)
        {
            std::cout << igpu << "\n";
            std::cout << "\tbrick batch/length:";
            for(const auto val : brick_batchlengths[igpu])
                std::cout << " " << val;
            std::cout << "\n";
            std::cout << "\tbrick dist/stride:";
            for(const auto val : brick_diststrides[igpu])
                std::cout << " " << val;
            std::cout << "\n";

            // FIXME: allow printing of subsection
            // printhostbuf(hostbufparts[igpu].data(), isreal, brick_batchlengths[igpu],
            //              brick_diststrides[igpu]);
        }
    }
        
    // TODO: lambda this?
    // Check all of the host buf values and make sure that they're where we expect them to be:
    switch(batchlengths.size())
    {
    case 3:
        // 1 batch + 2D FFT
        for(size_t xidx = 0; xidx < hostdatabatchlengths[1]; ++xidx)
        {
            for(size_t yidx = 0; yidx < hostdatabatchlengths[2]; ++yidx)
            {
                const std::vector<size_t> hostidx = {0, xidx, yidx};
                const auto bufidx = devidx(ngpus, hostidx, hostdatabatchlengths);

                std::stringstream idxstrs;
                idxstrs << hostidx[0]
                        << " " << hostidx[1]
                        << " " << hostidx[2]
                        << " -> "
                        << bufidx[0]
                        << " " << bufidx[1]
                        << " " << bufidx[2]
                        << " " << bufidx[3]
                        << "\t";
                    
                const size_t hostoffset = std::inner_product(std::begin(hostidx),
                                                             std::end(hostidx),
                                                             std::begin(hostdiststrides), 0);
                const auto igpu = bufidx[0];
                const size_t gpuoffset = std::inner_product(std::begin(bufidx) + 1,
                                                            std::end(bufidx),
                                                            std::begin(brick_diststrides[igpu]),
                                                            0);
                if(isreal)
                {
                    const double* hostbufr = (double*) hostbuf.data();
                    const auto hostval = hostbufr[hostoffset];
                    const double* gpubufr = (double*) hostbufparts[igpu].data();
                    const auto gpuval = gpubufr[gpuoffset];
                    std::stringstream valss;
                    valss << hostoffset << " -> " << hostval << "\t"
                          << gpuoffset << " -> " << gpuval << "\n";
                    if(verbose > 3)
                    {
                        std::cout << idxstrs.str() << valss.str()<< std::flush;
                    }
                    EXPECT_EQ(hostval, gpuval) << idxstrs.str() << valss.str();
                }
                else
                {
                    const std::complex<double>* hostbufr
                        = (std::complex<double>*) hostbuf.data();
                    const auto hostval = hostbufr[hostoffset];
                    const std::complex<double>* gpubufr
                        = (std::complex<double>*) hostbufparts[igpu].data();
                    const auto gpuval = gpubufr[gpuoffset];
                    std::stringstream valss;
                    valss << hostoffset << " -> " << hostval << "\t"
                          << gpuoffset << " -> " << gpuval << "\n";
                    if(verbose > 3)
                    {
                        std::cout << idxstrs.str() << valss.str()<< std::flush;
                    }
                    EXPECT_EQ(hostval, gpuval) << idxstrs.str() << valss.str();
                }
            }
        }
        break;
    case 4:
        // 1 batch + 3D FFT
        for(size_t xidx = 0; xidx < hostdatabatchlengths[1]; ++xidx)
        {
            for(size_t yidx = 0; yidx < hostdatabatchlengths[2]; ++yidx)
            {
                for(size_t zidx = 0; zidx < hostdatabatchlengths[3]; ++zidx)
                {
                    const std::vector<size_t> hostidx = {0, xidx, yidx, zidx};
                    const auto bufidx = devidx(ngpus, hostidx, hostdatabatchlengths);

                    std::stringstream idxstrs;
                    idxstrs << hostidx[0]
                            << " " << hostidx[1]
                            << " " << hostidx[2]
                            << " -> "
                            << bufidx[0]
                            << " " << bufidx[1]
                            << " " << bufidx[2]
                            << " " << bufidx[3]
                            << "\t";
                    
                    const size_t hostoffset = std::inner_product(std::begin(hostidx),
                                                                 std::end(hostidx),
                                                                 std::begin(hostdiststrides), 0);
                    const auto igpu = bufidx[0];
                    const size_t gpuoffset = std::inner_product(std::begin(bufidx) + 1,
                                                                std::end(bufidx),
                                                                std::begin(brick_diststrides[igpu]),
                                                                0);
                    if(isreal)
                    {
                        const double* hostbufr = (double*) hostbuf.data();
                        const auto hostval = hostbufr[hostoffset];
                        const double* gpubufr = (double*) hostbufparts[igpu].data();
                        const auto gpuval = gpubufr[gpuoffset];
                        std::stringstream valss;
                        valss << hostoffset << " -> " << hostval << "\t"
                              << gpuoffset << " -> " << gpuval << "\n";
                        if(verbose > 3)
                        {
                            std::cout << idxstrs.str() << valss.str()<< std::flush;
                        }
                        EXPECT_EQ(hostval, gpuval) << idxstrs.str() << valss.str();
                    }
                    else
                    {
                        const std::complex<double>* hostbufr
                            = (std::complex<double>*) hostbuf.data();
                        const auto hostval = hostbufr[hostoffset];
                        const std::complex<double>* gpubufr
                            = (std::complex<double>*) hostbufparts[igpu].data();
                        const auto gpuval = gpubufr[gpuoffset];
                        std::stringstream valss;
                        valss << hostoffset << " -> " << hostval << "\t"
                              << gpuoffset << " -> " << gpuval << "\n";
                        if(verbose > 3)
                        {
                            std::cout << idxstrs.str() << valss.str()<< std::flush;
                        }
                        EXPECT_EQ(hostval, gpuval) << idxstrs.str() << valss.str();
                    }
                }
            }
        }
        break;
    default:
        FAIL() << "dimension not supported";
    }
    
    hipfft_rt = hipfftXtFree(mydesc);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    
    hipfft_rt = hipfftDestroy(plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    hipfftxttest,
    hipfftxtunitdesc,
    ::testing::ValuesIn(in_goodlist),
    [](const testing::TestParamInfo<hipfftxtunitdesc::ParamType>& info) {
        const bool realcomplex = std::get<0>(info.param);
        const int direction = std::get<1>(info.param);
        const hipfftXtSubFormat format = std::get<2>(info.param);
        const size_t dimension = std::get<3>(info.param);
        std::string name = direction == HIPFFT_FORWARD ? "forward" : "backward";
        name += realcomplex ? "rc" : "cc";
        name += format_name(format);
        name += "dim" + std::to_string(dimension);
        return name;
    }
    );

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

    const auto direction = std::get<0>(GetParam());
    const hipfftXtSubFormat informat = std::get<1>(GetParam());
    const auto batch = std::get<2>(GetParam());
    
    const hipfftXtSubFormat outformat
        = informat == HIPFFT_XT_FORMAT_INPLACE
        ? HIPFFT_XT_FORMAT_INPLACE_SHUFFLED
        : HIPFFT_XT_FORMAT_INPLACE;

    if(verbose > 0)
    {
        std::cout << "complex-to-complex direction: " << directionname(direction)
                  << " input format: " << format_name(informat)
                  << " output format: " << format_name(outformat)
                  << "\n";
    }
    
    hipfftHandle plan;
    hipfft_rt =   hipfftCreate(&plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    // FIXME: deal with not having enough GPUs.
    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    std::vector<size_t> workSize(ngpus);
    hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, HIPFFT_Z2Z, workSize.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)<< "hipfftMakePlan2d failed with code "
                                         << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";
    
    hipLibXtDesc*       inoutdesc = nullptr;
    hipfft_rt                     = hipfftXtMalloc(plan, &inoutdesc, informat);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc failed with code "
                                         << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";
    
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
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMemcpy pre-exec failed with code "
                                         << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";
    
    EXPECT_EQ(inoutdesc->subFormat, informat);

    hipfft_rt = hipfftXtExecDescriptor(plan, inoutdesc, inoutdesc, direction);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtExecDescriptor failed with code "
                                         << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";

    EXPECT_EQ(inoutdesc->subFormat, outformat) << "descriptor subformat is "
                                               << inoutdesc->subFormat
                                               << " (" << format_name(inoutdesc->subFormat) << ")"
                                               << " but we expected " << outformat
                                               << " (" << format_name(outformat) << ")";
    
    std::vector<std::complex<double>> output(Nx * Ny);

    // FIXME: re-enable
    // hipfft_rt = hipfftXtMemcpy(plan,
    //                            reinterpret_cast<void*>(output.data()),
    //                            reinterpret_cast<void*>(inoutdesc),
    //                            HIPFFT_COPY_DEVICE_TO_HOST);
    // EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMemcpy post-exec failed with code "
    //                                      << hipfft_rt
    //                                      << " (" << hipfftResult_string(hipfft_rt) << ")";
    
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
        ::testing::Values(1) // We only cover batch=1 for now.
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
    int ngpus = 2;

    int count = 0;
    ASSERT_EQ(hipGetDeviceCount(&count), hipSuccess) << "hipGetDeviceCount failed";
    if(count < ngpus)
    {
        // We actually use separate GPUs, so skip if we don't have enough GPUs.
        GTEST_SKIP() << "not enough GPUs";
    }
    
    const int Nx    = 32;
    const int Ny    = 32;
    
    const int Nyp = Ny / 2 + 1;
    const int Nypp = 2 * Nyp;
    
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
    if(direction == HIPFFT_BACKWARD && batch > 1 && informat != HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
    {
        GTEST_SKIP();
    }
    if(batch > 1)
    {
        // Running multi-batch transforms seems to lead to failures in subsequent tests for the cuda
        // back-end.
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
    
    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    
    if(verbose > 0)
    {
        std::cout << "hipfftxt format change test\n";
        std::cout << "\tNx: " << Nx << "\n";
        std::cout << "\tNy: " << Ny << "\n";
        std::cout << "\tngpus: " << ngpus << "\n";
        std::cout << "\tgpus:";
        for(const auto igpu: gpus)
            std::cout << " " << igpu;
        std::cout << "\n";
        std::cout << "\ttransform_type: " << transform_type << " : "
                  << hipffttype_to_name(transform_type) << "\n";
        std::cout << "\tdirection: " << direction << " : " << directionname(direction)
                  << "\n\tinput subformat: " << informat << " : " << format_name(informat)
                  << "\n\toutput subformat: " << outformat << " : " << format_name(outformat)
                  << "\n";
    }
    
    hipfftHandle plan;
    hipfft_rt =   hipfftCreate(&plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    if(verbose > 1)
    {
        std::cout << "direction: " << directionname(direction)
                  << " informat: " << format_name(informat)
                  << " batch: " << batch << "\n";
    }

    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtSetGPUs failed";
        
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
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftPlanMany failed with return code "
                                             << hipfft_rt << "=" << hipfftResult_string(hipfft_rt);
    }
    else
    {
        hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny,
                                     transform_type,
                                     workSize.data());
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftMakePlan2d failed with return code "
                                             << hipfft_rt << "=" << hipfftResult_string(hipfft_rt);
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

    
    if(verbose > 1)
        std::cout << "Copying to the devices ...\n";
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(inoutdesc),
                               direction == HIPFFT_FORWARD
                               ? reinterpret_cast<void*>(real.data())
                               : reinterpret_cast<void*>(complex.data()),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMemcpy h2d failed with code "
                                         << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";
    
    EXPECT_EQ(inoutdesc->subFormat, informat)
        << "informat not what expected:"
        << " got " << format_name((hipfftXtSubFormat)inoutdesc->subFormat)
        << " expected " << format_name((hipfftXtSubFormat)informat);

    if(verbose > 1)
        std::cout << "Executing the transform ...\n";
    hipfft_rt = hipfftXtExecDescriptor(plan, inoutdesc, inoutdesc, direction);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtExecDescriptor failed with code "
                                         << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";

    
    EXPECT_EQ(inoutdesc->subFormat, outformat)
        << "outformat not what expected:"
        << " got " << inoutdesc->subFormat << " "
        << format_name((hipfftXtSubFormat)inoutdesc->subFormat)
        << " expected "  << outformat << " "
        << format_name((hipfftXtSubFormat)outformat);

    if(verbose > 1)
        std::cout << "Copying back to host ...\n";
    
    hipfft_rt = hipfftXtMemcpy(plan,
                               direction == HIPFFT_FORWARD
                               ? reinterpret_cast<void*>(complex.data())
                               : reinterpret_cast<void*>(real.data()),
                               reinterpret_cast<void*>(inoutdesc),
                               HIPFFT_COPY_DEVICE_TO_HOST);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMemcpy d2h failed with code "
                                         << hipfft_rt
                                         << " (" << hipfftResult_string(hipfft_rt) << ")";

    hipfft_rt = hipfftXtFree(inoutdesc);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipfft_rt = hipfftDestroy(plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}
