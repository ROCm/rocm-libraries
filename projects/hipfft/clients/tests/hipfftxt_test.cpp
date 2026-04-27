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

static std::string transform_type_name(const fft_transform_type transform_type)
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

static std::string fft_io_name(const fft_io io)
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

static std::string fft_result_placement_name(const fft_result_placement placement)
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

static std::string format_name(const int format)
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

static std::string hipffttype_to_name(const hipfftType txtype)
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

static std::string directionname(const int direction)
{
    switch(direction)
    {
    case HIPFFT_FORWARD:
        return "HIPFFT_FORWARD";
    case HIPFFT_BACKWARD:
        return "HIPFFT_BACKWARD";
    }
}

// We may run tests on all visible devices; query how many devices with this function.
static auto getdevcount()
{
    int        deviceCount = 0;
    const auto ret         = hipGetDeviceCount(&deviceCount);
    if(ret != hipSuccess)
        throw std::runtime_error("hipGetDeviceCount failed");
    return deviceCount;
}

#ifdef __HIP_PLATFORM_AMD__
static const bool rocfft_backend = true;
#else
static const bool rocfft_backend = false;
#endif

// Params are direction and real/complex, is-single-batch
class hipfftxtunit : public ::testing::TestWithParam<std::tuple<int, bool, bool>>
{
};

TEST_P(hipfftxtunit, plancreation)
{
    // Test whether we can just make plans.

    size_t ngpus = getdevcount();
    if(ngpus < 2)
        GTEST_SKIP();

    const int Nx = 32;
    const int Ny = 32;

    const int  direction   = std::get<0>(GetParam());
    const bool realcomplex = std::get<1>(GetParam());
    const bool singlebatch = std::get<2>(GetParam());

    const hipfftType transform_type
        = realcomplex ? ((direction == HIPFFT_FORWARD) ? HIPFFT_D2Z : HIPFFT_Z2D) : HIPFFT_Z2Z;

    if(verbose > 0)
    {
        std::cout << "hipfftxt plan creation test: " << directionname(direction)
                  << (realcomplex ? " real/complex" : "complex/complex") << "\n";
    }

    auto hipfft_rt = HIPFFT_SUCCESS;

    hipfftHandle plan;
    hipfft_rt = hipfftCreate(&plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);

    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtSetGPUs failed";

    std::vector<size_t> workSize(ngpus);
    if(singlebatch)
    {
        hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, transform_type, workSize.data());
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftMakePlan2d failed with return code "
                                             << hipfft_rt << "=" << hipfftResult_string(hipfft_rt);
    }
    else
    {
        std::vector<int> lengths = {Nx, Ny};
        const int        nbatch  = ngpus;
        hipfft_rt                = hipfftMakePlanMany(plan,
                                       lengths.size(),
                                       lengths.data(),
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       0,
                                       0,
                                       transform_type,
                                       nbatch,
                                       workSize.data());
        if(rocfft_backend)
            ASSERT_NE(hipfft_rt, HIPFFT_SUCCESS)
                << "multi-batch multi-gpu transforms should return not implemented";
        else
            ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
                << "hipfftMakePlanMany failed with return code " << hipfft_rt << "="
                << hipfftResult_string(hipfft_rt);
    }

    hipfft_rt = hipfftDestroy(plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(hipfftxttest,
                         hipfftxtunit,
                         ::testing::Combine(::testing::Values(HIPFFT_FORWARD, HIPFFT_BACKWARD),
                                            ::testing::Values(true, false),
                                            ::testing::Bool()),
                         [](const testing::TestParamInfo<hipfftxtunit::ParamType>& info) {
                             const int   direction   = std::get<0>(info.param);
                             const int   realcomplex = std::get<1>(info.param);
                             const int   singlebatch = std::get<2>(info.param);
                             std::string name
                                 = direction == HIPFFT_FORWARD ? "forward" : "backward";
                             name += realcomplex ? "rc" : "cc";
                             name += singlebatch ? "singlebatch" : "multibatch";
                             return name;
                         });

// Data holder struct for combining allowable direction/format combinations.
struct directionformat_t
{
    int               direction;
    hipfftXtSubFormat format;
};

// Real/complex hipfftxt multi-gpu transforms use HIPFFT_XT_FORMAT_INPLACE for the space format, and
// HIPFFT_XT_FORMAT_INPLACE_SHUFFLED for the frequency format.
static std::vector<directionformat_t> real_directionformat
    = {{HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE},
       {HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED}};

// Complex/complex hipfftxt multi-gpu transforms use HIPFFT_XT_FORMAT_INPLACE for the space format,
// and HIPFFT_XT_FORMAT_INPLACE_SHUFFLED for the frequency format.  Out-of-place transforms may use
// HIPFFT_XT_FORMAT_INPUT/HIPFFT_XT_FORMAT_OUTPUT, but we do not currently support this
// functionality.
static std::vector<directionformat_t> complex_directionformat = {
    {HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE},
    {HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED},
    {HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE},
    {HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED},
    //{HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPUT},
    //{HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_OUTPUT},
    //{HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPUT},
    //{HIPFFT_FORWARD, HIPFFT_XT_FORMAT_OUTPUT},
};

// Combine the real/complex and complex/complex direction-format arrays, prepending a bool which is
// true for real/complex tramsforms.
static auto all_directionformat()
{
    std::vector<std::tuple<bool, directionformat_t>> combined;
    for(const auto& val : real_directionformat)
        combined.push_back(std::make_tuple(true, val));
    for(const auto& val : complex_directionformat)
        combined.push_back(std::make_tuple(false, val));
    return combined;
}

// 2D and 3D transforms single-batch multi-gpu FFTs are handled differently than 1D transforms.
static std::vector<size_t> multidims = {2, 3};

// Parameters are real/complex, direction, format, dimension, and number of GPUs.
class hipfftxtunitdesc
    : public ::testing::TestWithParam<std::tuple<bool, int, hipfftXtSubFormat, size_t, int>>
{
};

// Verify that the distributed data decomposition is what we expect.  After distributing the data to
// multiple device buffers via hipfftXtMemcpy, copy the buffers back and verify that the values are
// at the pointer offset where we expect it to be.
TEST_P(hipfftxtunitdesc, xtmemcpytest)
{
    const bool              realcomplex = std::get<0>(GetParam());
    const auto              direction   = std::get<1>(GetParam());
    const hipfftXtSubFormat format      = std::get<2>(GetParam());
    const auto              dimension   = std::get<3>(GetParam());
    const auto              ngpus       = std::get<4>(GetParam());

    const int Nx = 32;
    const int Ny = 36;
    const int Nz = 38;
    // Just batch=1 for now.

    if(verbose > 0)
    {
        std::cout << "hipfftxt plan creation test: " << directionname(direction)
                  << (realcomplex ? " real/complex" : "complex/complex") << " dimension "
                  << dimension << "\n";
        std::cout << "Nx: " << Nx << " Ny: " << Ny;
        if(dimension == 3)
            std::cout << " Nz: " << Nz;
        std::cout << " ngpus: " << ngpus;
        std::cout << "\n";
    }
    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);

    // TODO: other sizes, batch, etc.
    std::vector<size_t> batches = {1};
    std::vector<size_t> lengths = {Nx, Ny};
    if(dimension == 3)
        lengths.push_back(Nz);
    std::vector<size_t> batchlengths = batches;
    batchlengths.insert(batchlengths.end(), lengths.begin(), lengths.end());

    // Some facts about the test case:
    const bool   forward  = (direction == HIPFFT_FORWARD);
    const bool   isreal   = realcomplex ? (format == HIPFFT_XT_FORMAT_INPLACE) : false;
    const bool   isherm   = realcomplex ? (format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED) : false;
    const size_t lastdim  = batchlengths.size() - 1;
    const bool   inspace  = format == HIPFFT_XT_FORMAT_INPUT || format == HIPFFT_XT_FORMAT_INPLACE;
    const size_t splitdim = inspace ? 1 : 2;

    if(verbose > 2)
    {
        std::cout << "lastdim: " << lastdim << "\n";
        std::cout << "splitdim: " << splitdim << "\n";
    }

    // fft_enums configuration
    const fft_transform_type dft_type
        = realcomplex
              ? (forward ? fft_transform_type_real_forward : fft_transform_type_real_inverse)
              : (forward ? fft_transform_type_complex_forward : fft_transform_type_complex_inverse);
    const fft_result_placement placement
        = (format == HIPFFT_XT_FORMAT_INPLACE || format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
              ? fft_placement_inplace
              : fft_placement_notinplace;
    const fft_io io
        = (forward != (format == HIPFFT_XT_FORMAT_INPUT || format == HIPFFT_XT_FORMAT_INPLACE))
              ? fft_io_out
              : fft_io_in;

    // hipfftxt configuratin:
    const hipfftType transform_type
        = realcomplex ? (forward ? HIPFFT_D2Z : HIPFFT_Z2D) : HIPFFT_Z2Z;

    // Compute the data lengths for a (complete transform) buffer.
    // Basically, this function just accounts for Hermitian symmetry.
    auto computedatabatchlengths
        = [](const bool isherm, const std::vector<size_t>& batchlengths) -> std::vector<size_t> {
        std::vector<size_t> newbatchlengths = batchlengths;
        if(isherm)
        {
            const size_t lastdim     = batchlengths.size() - 1;
            newbatchlengths[lastdim] = newbatchlengths[lastdim] / 2 + 1;
        }
        return newbatchlengths;
    };

    // Host data configuration:
    const auto host_distances       = default_distances(dft_type, placement, io, lengths, batches);
    auto       hostdiststrides      = host_distances;
    const auto hostdatabatchlengths = computedatabatchlengths(isherm, batchlengths);
    const auto host_strides         = default_strides(dft_type, placement, io, lengths);
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
    hipfft_rt = hipfftCreate(&plan);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtSetGPUs failed";

    std::vector<size_t> workSize(ngpus);
    switch(dimension)
    {
    case 2:
        hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, transform_type, workSize.data());
        break;
    case 3:
        hipfft_rt = hipfftMakePlan3d(plan, Nx, Ny, Nz, transform_type, workSize.data());
        break;
    default:
        FAIL() << "Test infrastructure only supports 2D and 3D transforms";
    }
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftMakePlan2/3d failed with return code "
                                         << hipfft_rt << "=" << hipfftResult_string(hipfft_rt);
    if(verbose > 2)
        std::cout << "plan created\n";

    hipLibXtDesc* mydesc = nullptr;
    hipfft_rt            = hipfftXtMalloc(plan, &mydesc, format);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc failed with code " << hipfft_rt << " ("
                                         << hipfftResult_string(hipfft_rt) << ")";
    if(verbose > 2)
        std::cout << "descriptor allocated\n";

    for(size_t igpu = 0; igpu < gpus.size(); ++igpu)
    {
        if(verbose > 3)
            std::cout << "buffer " << igpu << " size: " << mydesc->descriptor->size[igpu] << "\n";
        // TODO: handle case where some GPUs don't have data because there isn't enough to go
        // around.  (Particularly for multi-batch cases.)
        ASSERT_NE(mydesc->descriptor->size[igpu], 0) << "gpu buffer size is zero for gpu " << igpu;
    }

    // Host buff printer
    auto printhostbuf = [](const char*                hostbuf,
                           const bool                 isreal,
                           const std::vector<size_t>  batchlengths,
                           const std::vector<size_t>& hostdiststrides) -> void {
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
                        const size_t              pos = std::inner_product(
                            std::begin(idx), std::end(idx), std::begin(hostdiststrides), 0);
                        if(isreal)
                        {
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
                            const size_t              pos = std::inner_product(
                                std::begin(idx), std::end(idx), std::begin(hostdiststrides), 0);
                            if(isreal)
                            {
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
        auto       devbuf  = mydesc->descriptor->data[igpu];
        auto       hipret  = hipMemset(devbuf, 0, bufsize);
        EXPECT_EQ(hipret, hipSuccess) << "hipMemset failed";
        std::vector<char> hostbufpart(bufsize);
        hipret = hipMemcpy(hostbufpart.data(), devbuf, bufsize, hipMemcpyDeviceToHost);
        EXPECT_EQ(hipret, hipSuccess) << "hipMemcpy failed";
    }

    // Labmda for initializing the host buffer.  We do not care about Hermitian symmetry in 2D/3D,
    // as we are just testing data movement, not transforms.
    auto fillhostbuf = [](std::vector<char>&         hostbuf,
                          const bool                 isreal,
                          const std::vector<size_t>  batchlengths,
                          const std::vector<size_t>& hostdiststrides) -> void {
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

                        const size_t pos = std::inner_product(
                            std::begin(idx), std::end(idx), std::begin(hostdiststrides), 0);
                        if(isreal)
                        {
                            auto hostdat = reinterpret_cast<double*>(hostbuf.data());
                            hostdat[pos] = xidx + 0.01 * yidx;
                        }
                        else
                        {
                            auto hostdat = reinterpret_cast<std::complex<double>*>(hostbuf.data());
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

                            const size_t pos = std::inner_product(
                                std::begin(idx), std::end(idx), std::begin(hostdiststrides), 0);
                            if(isreal)
                            {
                                auto hostdat = reinterpret_cast<double*>(hostbuf.data());
                                hostdat[pos] = xidx + 0.01 * yidx + +0.0001 * zidx;
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

    // Compute the per-buffer data length, split in dimension splitdim.  If the data isn't perfectly
    // divisible, then any remainder is distributed between lower-index devices.
    auto devbatchlength = [](const size_t               splitdim,
                             const size_t               ngpus,
                             const std::vector<size_t>& hostdatabatchlengths,
                             const size_t               igpu) -> std::vector<size_t> {
        std::vector<size_t> databatchlengths = hostdatabatchlengths;
        const auto          l                = databatchlengths[splitdim];
        databatchlengths[splitdim]           = l / ngpus + ((igpu < l % ngpus) ? 1 : 0);
        return databatchlengths;
    };

    // Return a vector containing {gpu index, batch index, transform indices...}.
    // Batch and transform indices are buffer-local multi-index (ie relative to an index starting at
    // {0, ... , 0} on each brick).
    auto devidx = [](const size_t               splitdim,
                     const size_t               ngpus,
                     const std::vector<size_t>& hostidx,
                     const std::vector<size_t>& databatchlengths) -> std::vector<size_t> {
        std::vector<size_t> ret(databatchlengths.size() + 1, 0);
        for(size_t idx = 0; idx < hostidx.size(); ++idx)
        {
            if(idx != splitdim)
                ret[idx + 1] = hostidx[idx];
        }
        const auto l = databatchlengths[splitdim];
        const auto b = l / ngpus; // Elements per gpu in splitdim (if no remainder).
        const auto r = l - b * ngpus; // Remainder

        const auto a = hostidx[splitdim];
        if(a < r * (b + 1))
        {
            ret[0]            = a / (b + 1);
            ret[splitdim + 1] = a - ret[0] * (b + 1);
        }
        else
        {
            ret[0]            = r + (a - r * (b + 1)) / b;
            ret[splitdim + 1] = a - r * (b + 1) - (ret[0] - r) * b;
        }

        return ret;
    };

    // Fine, let's do a copy test and see what happens.

    // We test the host-to-device copy (since we can easily set up the host data and then check that
    // the buffers have values where we expect them to be).
    const auto copydir = HIPFFT_COPY_HOST_TO_DEVICE;

    const size_t      nelem   = hostdiststrides[0] * hostdatabatchlengths[0];
    const size_t      valsize = isreal ? sizeof(double) : sizeof(std::complex<double>);
    std::vector<char> hostbuf(valsize * nelem);
    fillhostbuf(hostbuf, isreal, hostdatabatchlengths, hostdiststrides);

    if(verbose > 4)
    {
        printhostbuf(hostbuf.data(), isreal, hostdatabatchlengths, hostdiststrides);
    }

    if(verbose > 2)
        std::cout << "starting hipfftXtMemcpy...\n";
    hipfft_rt = hipfftXtMemcpy(
        plan, reinterpret_cast<void*>(mydesc), reinterpret_cast<void*>(hostbuf.data()), copydir);
    ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
        << "hipfftXtMemcpy H2D"
        << " failed with code " << hipfft_rt << " (" << hipfftResult_string(hipfft_rt) << ")";

    if(verbose > 2)
        std::cout << "finished hipfftXtMemcpy\n";

    // A host copy of the individual distributed GPU buffers:
    std::vector<std::vector<char>> hostbufparts(gpus.size());

    // Copy the individual buffers to the host:
    for(const auto igpu : gpus)
    {
        if(verbose > 3)
        {
            std::cout << "buffer " << igpu << " after xtmemcp\n";
        }
        const auto device  = mydesc->descriptor->GPUs[igpu];
        const auto bufsize = mydesc->descriptor->size[igpu];
        if(verbose > 3)
        {
            std::cout << "device: " << device << "\n";
            std::cout << "buffer size: " << bufsize << "\n";
        }
        ASSERT_NE(bufsize, 0) << "gpu buffer size is zero for gpu " << igpu;
        hostbufparts[igpu].resize(bufsize);
        auto devbuf = mydesc->descriptor->data[igpu];
        auto hipret = hipMemcpy(hostbufparts[igpu].data(), devbuf, bufsize, hipMemcpyDeviceToHost);
        EXPECT_EQ(hipret, hipSuccess) << "hipMemcpy failed";
    }

    // Each brick gets it own special set of strides.
    std::vector<std::vector<size_t>> brick_batchlengths(gpus.size());
    std::vector<std::vector<size_t>> brick_diststrides(gpus.size());

    for(size_t igpu = 0; igpu < gpus.size(); ++igpu)
    {
        brick_batchlengths[igpu] = devbatchlength(splitdim, ngpus, hostdatabatchlengths, igpu);
        std::vector<size_t> brick_batches;
        brick_batches.insert(brick_batches.end(),
                             brick_batchlengths[igpu].begin(),
                             brick_batchlengths[igpu].begin() + 1);
        std::vector<size_t> brick_lengths;
        brick_lengths.insert(brick_lengths.end(),
                             brick_batchlengths[igpu].begin() + 1,
                             brick_batchlengths[igpu].end());

        std::vector<size_t> brick_distances;
        std::vector<size_t> brick_strides;
        if(isherm)
        {
            brick_distances = default_distances(
                fft_transform_type_complex_forward, placement, io, brick_lengths, brick_batches);
            brick_strides
                = default_strides(fft_transform_type_complex_forward, placement, io, brick_lengths);
        }
        else
        {
            brick_distances
                = default_distances(dft_type, placement, io, brick_lengths, brick_batches);
            brick_strides = default_strides(dft_type, placement, io, brick_lengths);
        }

        brick_diststrides[igpu] = brick_distances;
        brick_diststrides[igpu].insert(
            brick_diststrides[igpu].end(), brick_strides.begin(), brick_strides.end());
    }

    if(verbose > 2)
    {
        std::cout << "gpu buffer length and dist/strides:\n";
        for(size_t igpu = 0; igpu < gpus.size(); ++igpu)
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

            // TODO: allow printing of subsection
            // printhostbuf(hostbufparts[igpu].data(), isreal, brick_batchlengths[igpu],
            //              brick_diststrides[igpu]);
        }
    }

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
                const auto bufidx = devidx(splitdim, ngpus, hostidx, hostdatabatchlengths);

                std::stringstream idxstrs;
                idxstrs << hostidx[0] << " " << hostidx[1] << " " << hostidx[2] << " -> "
                        << bufidx[0] << " " << bufidx[1] << " " << bufidx[2] << " " << bufidx[3]
                        << "\t";

                const size_t hostoffset = std::inner_product(
                    std::begin(hostidx), std::end(hostidx), std::begin(hostdiststrides), 0);
                const auto   igpu      = bufidx[0];
                const size_t gpuoffset = std::inner_product(std::begin(bufidx) + 1,
                                                            std::end(bufidx),
                                                            std::begin(brick_diststrides[igpu]),
                                                            0);
                if(isreal)
                {
                    const double*     hostbufr = (double*)hostbuf.data();
                    const auto        hostval  = hostbufr[hostoffset];
                    const double*     gpubufr  = (double*)hostbufparts[igpu].data();
                    const auto        gpuval   = gpubufr[gpuoffset];
                    std::stringstream valss;
                    valss << hostoffset << " -> " << hostval << "\t" << gpuoffset << " -> "
                          << gpuval << "\n";
                    if(verbose > 3)
                        std::cout << idxstrs.str() << valss.str() << std::flush;
                    EXPECT_EQ(hostval, gpuval) << idxstrs.str() << valss.str();
                }
                else
                {
                    const std::complex<double>* hostbufr = (std::complex<double>*)hostbuf.data();
                    const auto                  hostval  = hostbufr[hostoffset];
                    const std::complex<double>* gpubufr
                        = (std::complex<double>*)hostbufparts[igpu].data();
                    const auto        gpuval = gpubufr[gpuoffset];
                    std::stringstream valss;
                    valss << hostoffset << " -> " << hostval << "\t" << gpuoffset << " -> "
                          << gpuval << "\n";
                    if(verbose > 3)
                        std::cout << idxstrs.str() << valss.str() << std::flush;
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
                    const auto bufidx = devidx(splitdim, ngpus, hostidx, hostdatabatchlengths);

                    std::stringstream idxstrs;
                    idxstrs << hostidx[0] << " " << hostidx[1] << " " << hostidx[2] << " -> "
                            << bufidx[0] << " " << bufidx[1] << " " << bufidx[2] << " " << bufidx[3]
                            << "\t";

                    const size_t hostoffset = std::inner_product(
                        std::begin(hostidx), std::end(hostidx), std::begin(hostdiststrides), 0);
                    const auto   igpu      = bufidx[0];
                    const size_t gpuoffset = std::inner_product(std::begin(bufidx) + 1,
                                                                std::end(bufidx),
                                                                std::begin(brick_diststrides[igpu]),
                                                                0);
                    if(isreal)
                    {
                        const double*     hostbufr = (double*)hostbuf.data();
                        const auto        hostval  = hostbufr[hostoffset];
                        const double*     gpubufr  = (double*)hostbufparts[igpu].data();
                        const auto        gpuval   = gpubufr[gpuoffset];
                        std::stringstream valss;
                        valss << hostoffset << " -> " << hostval << "\t" << gpuoffset << " -> "
                              << gpuval << "\n";
                        if(verbose > 3)
                            std::cout << idxstrs.str() << valss.str() << std::flush;
                        EXPECT_EQ(hostval, gpuval) << idxstrs.str() << valss.str();
                    }
                    else
                    {
                        const std::complex<double>* hostbufr
                            = (std::complex<double>*)hostbuf.data();
                        const auto                  hostval = hostbufr[hostoffset];
                        const std::complex<double>* gpubufr
                            = (std::complex<double>*)hostbufparts[igpu].data();
                        const auto        gpuval = gpubufr[gpuoffset];
                        std::stringstream valss;
                        valss << hostoffset << " -> " << hostval << "\t" << gpuoffset << " -> "
                              << gpuval << "\n";
                        if(verbose > 3)
                            std::cout << idxstrs.str() << valss.str() << std::flush;
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
    ::testing::ConvertGenerator(
        ::testing::Combine(::testing::ValuesIn(all_directionformat()),
                           ::testing::ValuesIn(multidims),
#ifdef __HIP_PLATFORM_NVIDIA__
                           ::testing::Range(2, getdevcount() + 1)
#else
                           ::testing::Range(1, getdevcount() + 1)
#endif
                               ),
        [](const std::tuple<std::tuple<bool, directionformat_t>, size_t, int>& t) {
            // This lambda recombines the nested tuples into a flat tuple to
            // make test parametrization simpler.
            auto         rdf         = std::get<0>(t);
            const bool   realcomplex = std::get<0>(rdf);
            auto         df          = std::get<1>(rdf);
            const size_t dim         = std::get<1>(t);
            const int    ngpus       = std::get<2>(t);
            auto         ret = std::make_tuple(realcomplex, df.direction, df.format, dim, ngpus);
            return ret;
        }),
    [](const testing::TestParamInfo<hipfftxtunitdesc::ParamType>& info) {
        const auto realcomplex = std::get<0>(info.param);
        //const auto direction_format  = std::get<1>(info.param);
        const auto  direction = std::get<1>(info.param);
        const auto  format    = std::get<2>(info.param);
        const auto  dimension = std::get<3>(info.param);
        const auto  ngpus     = std::get<4>(info.param);
        std::string name      = realcomplex ? "rc" : "cc";
        name += direction == HIPFFT_FORWARD ? "forward" : "backward";
        name += format_name(format);
        name += "dim" + std::to_string(dimension);
        name += "ngpus" + std::to_string(ngpus);
        return name;
    });

// Parameters are real/complex, direction, format, dimension, and number of GPUs.
class hipfftxtformats : public ::testing::TestWithParam<std::tuple<bool, int, hipfftXtSubFormat>>
{
};

// Test that we support exactly all of the data formats / FFT setups that we have implemented.
TEST_P(hipfftxtformats, supportlistsinglebatch)
{
    size_t ngpus = getdevcount();
#ifdef __HIP_PLATFORM_NVIDIA__
    if(ngpus == 1)
        GTEST_SKIP() << "Need at least 2 gpus for this test";
#endif
    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);

    // Wrost-case minimum size for cuda-backend is 1024.
    const int Nx = 1024;
    const int Ny = 1024;

    auto hipfft_rt = HIPFFT_SUCCESS;

    const bool realcomplex = std::get<0>(GetParam());
    const auto direction   = std::get<1>(GetParam());
    const auto format      = std::get<2>(GetParam());

    if(verbose > 1)
    {
        std::cout << (realcomplex ? "rc" : "cc") << " " << directionname(direction) << " "
                  << format_name(format) << "\n";
    }

#ifdef __HIP_PLATFORM_NVIDIA__
    if(realcomplex && format == HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED)
        GTEST_SKIP(); // Problematic unsupported case, so skip the test.
#endif

    auto good_rdfs = all_directionformat();
    bool goodcase  = false;
    for(const auto& val : good_rdfs)
    {
        if(realcomplex == std::get<0>(val) && std::get<1>(val).direction == direction
           && std::get<1>(val).format == format)
        {
            goodcase = true;
            break;
        }
    }

    hipfftHandle plan;
    hipfft_rt = hipfftCreate(&plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    const auto ffttype
        = realcomplex ? (direction == HIPFFT_FORWARD ? HIPFFT_D2Z : HIPFFT_Z2D) : HIPFFT_Z2Z;
    std::vector<size_t> workSize(ngpus);
    if(verbose > 2)
        std::cout << "creating plan..." << std::flush;
    if(format == HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED)
    {
        const int batchsize = 1;
        hipfft_rt           = hipfftMakePlan1d(plan, Nx, ffttype, batchsize, workSize.data());
        if(realcomplex || rocfft_backend)
        {
            ASSERT_NE(hipfft_rt, HIPFFT_SUCCESS)
                << "hipfftMakePlan1d should have failed for real/complex multi-gpu";
        }
        else
        {
            ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
                << "hipfftMakePlan1d failed with code " << hipfft_rt << " ("
                << hipfftResult_string(hipfft_rt) << ")";
        }
    }
    else
    {
        hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, ffttype, workSize.data());
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftMakePlan2d failed with code " << hipfft_rt
                                             << " (" << hipfftResult_string(hipfft_rt) << ")";
    }
    if(verbose > 2)
        std::cout << " done.\n";

    hipLibXtDesc* indesc = nullptr;
    hipfft_rt            = hipfftXtMalloc(plan, &indesc, format);

    if(goodcase)
    {
        EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc failed with code " << hipfft_rt
                                             << " (" << hipfftResult_string(hipfft_rt) << ")";
    }
    else
    {
        EXPECT_NE(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc passed but should have failed";
    }

    const bool inplace
        = (format == HIPFFT_XT_FORMAT_INPLACE) || (format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED);
    if(!inplace && format != HIPFFT_FORMAT_UNDEFINED)
    {
        hipfftXtSubFormat outformat;
        switch(format)
        {
        case HIPFFT_XT_FORMAT_INPUT:
            outformat = HIPFFT_XT_FORMAT_OUTPUT;
            break;
        case HIPFFT_XT_FORMAT_OUTPUT:
            outformat = HIPFFT_XT_FORMAT_INPUT;
            break;
        case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
            outformat = HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED;
            break;
        case HIPFFT_FORMAT_UNDEFINED:
        case HIPFFT_XT_FORMAT_INPLACE:
        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
        default:
            throw std::runtime_error(
                "Test infrastructure error: input format is not actually an out-of-place format");
        }
        hipLibXtDesc* outdesc = nullptr;
        hipfft_rt             = hipfftXtMalloc(plan, &outdesc, outformat);
        if(goodcase)
        {
            EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc failed with code " << hipfft_rt
                                                 << " (" << hipfftResult_string(hipfft_rt) << ")";
        }
        else
        {
            EXPECT_NE(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc passed but should have failed";
        }

        hipfft_rt = hipfftXtFree(outdesc);
        EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
    }

    hipfft_rt = hipfftXtFree(indesc);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);

    hipfft_rt = hipfftDestroy(plan);
    EXPECT_EQ(hipfft_rt, HIPFFT_SUCCESS);
}

const std::vector<int>               hipfft_directions = {HIPFFT_FORWARD, HIPFFT_BACKWARD};
const std::vector<hipfftXtSubFormat> hipfft_formats    = {HIPFFT_XT_FORMAT_INPUT,
                                                       HIPFFT_XT_FORMAT_OUTPUT,
                                                       HIPFFT_XT_FORMAT_INPLACE,
                                                       HIPFFT_XT_FORMAT_INPLACE_SHUFFLED,
                                                       HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED,
                                                       HIPFFT_FORMAT_UNDEFINED};

INSTANTIATE_TEST_SUITE_P(hipfftxttest,
                         hipfftxtformats,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::ValuesIn(hipfft_directions),
                                            ::testing::ValuesIn(hipfft_formats)),
                         [](const testing::TestParamInfo<hipfftxtformats::ParamType>& info) {
                             const auto  realcomplex = std::get<0>(info.param);
                             const auto  direction   = std::get<1>(info.param);
                             const auto  format      = std::get<2>(info.param);
                             std::string name        = realcomplex ? "rc" : "cc";
                             name += direction == HIPFFT_FORWARD ? "forward" : "backward";
                             name += format_name(format);
                             return name;
                         });
