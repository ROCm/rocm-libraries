// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/fft_enums.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/params_gen.h"
#include "../../shared/reference_fft_data.h"
#include "../../shared/rocfft_hip.h"
#include "../../shared/test_params.h"
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

typedef hip_object_wrapper_t<hipfftHandle, hipfftCreate, hipfftDestroy, HIPFFT_SUCCESS>
    hipfftHandle_wrapper_t;

// hipfftXtMalloc takes (plan, &desc, format) but hip_object_wrapper_t expects TCreate(&obj, ...).
// This adapter reorders the arguments to match.
inline hipfftResult
    hipfftXtMalloc_adapted(hipLibXtDesc** desc, hipfftHandle plan, hipfftXtSubFormat fmt)
{
    return hipfftXtMalloc(plan, desc, fmt);
}
typedef hip_object_wrapper_t<hipLibXtDesc*, hipfftXtMalloc_adapted, hipfftXtFree, HIPFFT_SUCCESS>
    hipfftLibXtDesc_wrapper_t;

static std::string format_name(const hipfftXtSubFormat format)
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

#ifdef __HIP_PLATFORM_AMD__
static constexpr bool rocfft_backend = true;
#else
static constexpr bool rocfft_backend = false;
#endif

// Params are direction and real/complex, is-single-batch
class hipfftxtunit : public ::testing::TestWithParam<std::tuple<int, bool, bool>>
{
};

TEST_P(hipfftxtunit, plancreation)
{
    // Test whether we can just make plans.

    const size_t ngpus = rocfft_scoped_device::device_count();
    if(ngpus < 2)
        GTEST_SKIP();

    // TODO: 3D, single-precision

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
                  << (realcomplex ? " real/complex" : " complex/complex") << "\n";
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
        // Note: implicit default strides and distances enforced if {i,o}nembed are nullptr
        hipfft_rt = hipfftMakePlanMany(plan,
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
        if constexpr(rocfft_backend)
            ASSERT_EQ(hipfft_rt, HIPFFT_NOT_IMPLEMENTED)
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
                                            ::testing::Bool(),
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

// Data holder struct for combining allowable direction / input format combinations.
struct directionformat_t
{
    int               direction;
    hipfftXtSubFormat informat;
};

// Real/complex hipfftxt multi-gpu transforms use HIPFFT_XT_FORMAT_INPLACE for the space format, and
// HIPFFT_XT_FORMAT_INPLACE_SHUFFLED for the frequency format.
static std::vector<directionformat_t> real_directionformat
    = {{HIPFFT_FORWARD, HIPFFT_XT_FORMAT_INPLACE},
       {HIPFFT_BACKWARD, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED}};

// Complex/complex hipfftxt multi-gpu transforms use either  HIPFFT_XT_FORMAT_INPLACE or
// HIPFFT_XT_FORMAT_INPLACE_SHUFFLED for both space and frequency formats.
// Out-of-place transforms may use HIPFFT_XT_FORMAT_INPUT/HIPFFT_XT_FORMAT_OUTPUT, but we do not
// currently support this functionality.
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
// Only testing single-batch for now
// Note: if/when adding some, sort them by decreasing value to leverage caching of reference results
static std::vector<size_t>        test_batch_sizes   = {1};
static std::vector<fft_precision> test_precisions    = {fft_precision_double, fft_precision_single};
static std::vector<std::vector<size_t>> test_lengths = {{32, 36}, {32, 36, 38}};
static std::vector<hipfftXtSubFormat>   test_subformats
    = {HIPFFT_XT_FORMAT_INPLACE, HIPFFT_XT_FORMAT_INPLACE_SHUFFLED};

// Parameters are real/complex, direction, format, dimension, and number of GPUs.
class hipfftxtunitdesc
    : public ::testing::TestWithParam<std::tuple<bool, int, hipfftXtSubFormat, size_t, int>>
{
};

// Compute the data lengths for a (complete transform) buffer.
// Basically, this function just accounts for Hermitian symmetry.
auto computedatabatchlengths(const bool isherm, const std::vector<size_t>& batchlengths)
{
    std::vector<size_t> newbatchlengths = batchlengths;
    if(isherm)
    {
        const size_t lastdim     = batchlengths.size() - 1;
        newbatchlengths[lastdim] = newbatchlengths[lastdim] / 2 + 1;
    }
    return newbatchlengths;
}

// Function for initializing the host buffer.  We do not care about Hermitian symmetry in 2D/3D,
// as we are just testing data movement, not transforms.
template <typename bufT>
void fillhostbuf(std::vector<bufT>&         hostbuf,
                 const bool                 isreal,
                 const std::vector<size_t>& batchlengths,
                 const std::vector<size_t>& hostdiststrides)
{
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
                        auto       hostdat = reinterpret_cast<double*>(hostbuf.data());
                        const auto yscale = std::pow(10.0, -std::ceil(std::log10(batchlengths[2])));
                        hostdat[pos]      = xidx + yscale * yidx;
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
                            auto       hostdat = reinterpret_cast<double*>(hostbuf.data());
                            const auto yscale
                                = std::pow(10.0, -std::ceil(std::log10(batchlengths[2])));
                            const auto zscale
                                = yscale * std::pow(10.0, -std::ceil(std::log10(batchlengths[3])));
                            hostdat[pos] = xidx + yscale * yidx + zscale * zidx;
                        }
                        else
                        {
                            auto hostdat = reinterpret_cast<std::complex<double>*>(hostbuf.data());
                            const auto yscale
                                = std::pow(10.0, -std::ceil(std::log10(batchlengths[2])));
                            hostdat[pos] = std::complex<double>(xidx + yscale * yidx, zidx);
                        }
                    }
                }
            }
        }
        break;
    default:
        FAIL() << "dimension not handled";
    }
}

template <typename bufT>
void fillhostbuf(std::vector<bufT>&         hostbuf,
                 const bool                 isreal,
                 const size_t&              batch,
                 const std::vector<size_t>& lengths,
                 const size_t&              dist,
                 const std::vector<size_t>& strides)
{
    std::vector<size_t> batch_lengths{batch};
    std::vector<size_t> dist_strides{dist};
    std::copy(lengths.begin(), lengths.end(), std::back_inserter(batch_lengths));
    std::copy(strides.begin(), strides.end(), std::back_inserter(dist_strides));
    fillhostbuf<bufT>(hostbuf, isreal, batch_lengths, dist_strides);
}

// Host buff printer
template <typename bufT>
void printhostbuf(const bufT*                hostbuf,
                  const bool                 isreal,
                  const std::vector<size_t>& batchlengths,
                  const std::vector<size_t>& hostdiststrides)
{
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
                        const auto hostdat = reinterpret_cast<const std::complex<double>*>(hostbuf);
                        if(yidx > 0)
                            std::cout << " ";
                        std::cout << hostdat[pos];
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            if(ibatch < batchlengths[0] - 1)
                std::cout << "\n";
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
                        //std::cout << "\n";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            if(ibatch < batchlengths[0] - 1)
                std::cout << "\n";
        }
        break;
    default:
        FAIL() << "dimension not handled";
    }
}

template <typename bufT>
void printhostbuf(const bufT*                hostbuf,
                  const bool                 isreal,
                  const size_t&              batch,
                  const std::vector<size_t>& lengths,
                  const size_t&              dist,
                  const std::vector<size_t>& strides)
{
    std::vector<size_t> batch_lengths{batch};
    std::vector<size_t> dist_strides{dist};
    std::copy(lengths.begin(), lengths.end(), std::back_inserter(batch_lengths));
    std::copy(strides.begin(), strides.end(), std::back_inserter(dist_strides));
    printhostbuf(hostbuf, isreal, batch_lengths, dist_strides);
}

template <typename bufT>
double maxdiffhostbufs(const std::vector<bufT>&   bufa,
                       const std::vector<bufT>&   bufb,
                       const bool                 isreal,
                       const std::vector<size_t>& batchlengths,
                       const std::vector<size_t>& hostdiststrides)
{
    double diff = 0.0;
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
                        const auto vbufa = reinterpret_cast<const double*>(bufa.data());
                        const auto vbufb = reinterpret_cast<const double*>(bufb.data());
                        diff             = std::max(diff, std::abs(vbufa[pos] - vbufb[pos]));
                    }
                    else
                    {
                        const auto vbufa
                            = reinterpret_cast<const std::complex<double>*>(bufa.data());
                        const auto vbufb
                            = reinterpret_cast<const std::complex<double>*>(bufb.data());
                        diff = std::max(diff, std::abs(vbufa[pos] - vbufb[pos]));
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
                            const auto vbufa = reinterpret_cast<const double*>(bufa.data());
                            const auto vbufb = reinterpret_cast<const double*>(bufb.data());
                            diff             = std::max(diff, std::abs(vbufa[pos] - vbufb[pos]));
                        }
                        else
                        {
                            const auto vbufa
                                = reinterpret_cast<const std::complex<double>*>(bufa.data());
                            const auto vbufb
                                = reinterpret_cast<const std::complex<double>*>(bufb.data());
                            diff = std::max(diff, std::abs(vbufa[pos] - vbufb[pos]));
                        }
                    }
                }
            }
        }
        break;
    default:
        throw std::runtime_error("Unhandled dimension");
    }
    return diff;
}

template <typename bufT>
double maxdiffhostbufs(const std::vector<bufT>&   bufa,
                       const std::vector<bufT>&   bufb,
                       const bool                 isreal,
                       const size_t&              batch,
                       const std::vector<size_t>& lengths,
                       const size_t&              dist,
                       const std::vector<size_t>& strides)
{
    std::vector<size_t> batch_lengths{batch};
    std::vector<size_t> dist_strides{dist};
    std::copy(lengths.begin(), lengths.end(), std::back_inserter(batch_lengths));
    std::copy(strides.begin(), strides.end(), std::back_inserter(dist_strides));
    return maxdiffhostbufs(bufa, bufb, isreal, batch_lengths, dist_strides);
}

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
                  << (realcomplex ? " real/complex" : " complex/complex") << " dimension "
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
    const bool   isreal   = realcomplex && (format == HIPFFT_XT_FORMAT_INPLACE);
    const bool   isherm   = realcomplex && (format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED);
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

    // hipfftxt configuratin:
    const hipfftType transform_type
        = realcomplex ? (forward ? HIPFFT_D2Z : HIPFFT_Z2D) : HIPFFT_Z2Z;

    // Host data configuration:
    const auto host_distances = default_distances(dft_type, placement, fft_io_in, lengths, batches);
    auto       hostdiststrides      = host_distances;
    const auto hostdatabatchlengths = computedatabatchlengths(isherm, batchlengths);
    const auto host_strides         = default_strides(dft_type, placement, fft_io_in, lengths);
    hostdiststrides.insert(hostdiststrides.end(), host_strides.begin(), host_strides.end());
    if(verbose > 1)
    {
        std::cout << "dft_type: " << transform_type_name(dft_type) << "\n";
        std::cout << "placement: " << fft_result_placement_name(placement) << "\n";
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
        ASSERT_NE(mydesc->descriptor->size[igpu], size_t{0})
            << "gpu buffer size is zero for gpu " << igpu;
    }

    // Initialize desc buffers to zero:
    for(const auto igpu : gpus)
    {
        const auto           device = mydesc->descriptor->GPUs[igpu];
        rocfft_scoped_device dev(device);

        const auto bufsize = mydesc->descriptor->size[igpu];
        auto       devbuf  = mydesc->descriptor->data[igpu];
        auto       hipret  = hipMemset(devbuf, 0, bufsize);
        EXPECT_EQ(hipret, hipSuccess) << "hipMemset failed";
    }

    // Compute the per-buffer data length, split in dimension splitdim.  If the data isn't perfectly
    // divisible, then any remainder is distributed between lower-index devices.
    auto devdatabatchlength = [](const size_t               splitdim,
                                 const size_t               ngpus,
                                 const std::vector<size_t>& hostdatabatchlengths,
                                 const size_t               igpu) -> std::vector<size_t> {
        std::vector<size_t> databatchlengths = hostdatabatchlengths;
        const auto          l                = databatchlengths[splitdim];
        databatchlengths[splitdim]           = l / ngpus + ((igpu < l % ngpus) ? 1 : 0);
        return databatchlengths;
    };

    // Return a pair vector containing {gpu index, {batch index, transform indices...}}.
    // Batch and transform indices are buffer-local multi-indices (ie relative to an index starting
    // at {0, ... , 0} on each brick).
    auto devidx
        = [](const size_t               splitdim,
             const size_t               ngpus,
             const std::vector<size_t>& hostidx,
             const std::vector<size_t>& databatchlengths) -> std::pair<int, std::vector<size_t>> {
        // The multi-index for the data on the buffer:
        std::vector<size_t> dataidx(databatchlengths.size());
        for(size_t idx = 0; idx < hostidx.size(); ++idx)
        {
            if(idx != splitdim)
                dataidx[idx] = hostidx[idx];
        }
        const auto l = databatchlengths[splitdim];
        const auto b = l / ngpus; // Elements per gpu in splitdim (if no remainder).
        const auto r = l - b * ngpus; // Remainder

        // The buffer index
        int bufidx = 0;

        const auto a = hostidx[splitdim];
        if(a < r * (b + 1))
        {
            bufidx            = a / (b + 1);
            dataidx[splitdim] = a - bufidx * (b + 1);
        }
        else
        {
            bufidx            = r + (a - r * (b + 1)) / b;
            dataidx[splitdim] = a - r * (b + 1) - (bufidx - r) * b;
        }

        return std::make_pair(bufidx, dataidx);
    };

    // Fine, let's do a copy test and see what happens.

    // We test the host-to-device copy (since we can easily set up the host data and then check that
    // the buffers have values where we expect them to be).
    const auto copydir = HIPFFT_COPY_HOST_TO_DEVICE;

    const size_t nelem   = hostdiststrides[0] * hostdatabatchlengths[0];
    const size_t valsize = isreal ? sizeof(double) : sizeof(std::complex<double>);
    const size_t max_align_t_count
        = (valsize * nelem + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
    std::vector<std::max_align_t> hostbuf(max_align_t_count);
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
    std::vector<std::vector<std::max_align_t>> hostbufparts(gpus.size());

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
        rocfft_scoped_device dev(device);
        ASSERT_NE(bufsize, size_t(0)) << "gpu buffer size is zero for gpu " << igpu;
        hostbufparts[igpu].resize(bufsize);
        auto devbuf = mydesc->descriptor->data[igpu];
        auto hipret = hipMemcpy(hostbufparts[igpu].data(), devbuf, bufsize, hipMemcpyDeviceToHost);
        EXPECT_EQ(hipret, hipSuccess) << "hipMemcpy failed";
    }

    // Each brick gets it own special set of strides.
    std::vector<std::vector<size_t>> brick_databatchlengths(gpus.size());
    std::vector<std::vector<size_t>> brick_diststrides(gpus.size());

    for(size_t igpu = 0; igpu < gpus.size(); ++igpu)
    {
        brick_databatchlengths[igpu]
            = devdatabatchlength(splitdim, ngpus, hostdatabatchlengths, igpu);
        std::vector<size_t> brick_batches;
        brick_batches.insert(brick_batches.end(),
                             brick_databatchlengths[igpu].begin(),
                             brick_databatchlengths[igpu].begin() + 1);
        std::vector<size_t> brick_datalengths;
        brick_datalengths.insert(brick_datalengths.end(),
                                 brick_databatchlengths[igpu].begin() + 1,
                                 brick_databatchlengths[igpu].end());

        std::vector<size_t> brick_distances;
        std::vector<size_t> brick_strides;
        if(isherm)
        {
            brick_distances = default_distances(fft_transform_type_complex_forward,
                                                placement,
                                                fft_io_in,
                                                brick_datalengths,
                                                brick_batches);
            brick_strides   = default_strides(
                fft_transform_type_complex_forward, placement, fft_io_in, brick_datalengths);
        }
        else
        {
            brick_distances = default_distances(
                dft_type, placement, fft_io_in, brick_datalengths, brick_batches);
            brick_strides = default_strides(dft_type, placement, fft_io_in, brick_datalengths);
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
            for(const auto val : brick_databatchlengths[igpu])
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
                        << bufidx.second[0] << " " << bufidx.second[1] << " " << bufidx.second[2]
                        << " in buffer " << bufidx.first << "\t";

                const size_t hostoffset = std::inner_product(
                    std::begin(hostidx), std::end(hostidx), std::begin(hostdiststrides), 0);
                const auto   igpu      = bufidx.first;
                const size_t gpuoffset = std::inner_product(std::begin(bufidx.second),
                                                            std::end(bufidx.second),
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
                    idxstrs << hostidx[0] << " " << hostidx[1] << " " << hostidx[2] << " "
                            << hostidx[3] << " -> " << bufidx.second[0] << " " << bufidx.second[1]
                            << " " << bufidx.second[2] << " " << bufidx.second[3] << " in buffer "
                            << bufidx.first << " "
                            << "\t";

                    const size_t hostoffset = std::inner_product(
                        std::begin(hostidx), std::end(hostidx), std::begin(hostdiststrides), 0);
                    const auto   igpu      = bufidx.first;
                    const size_t gpuoffset = std::inner_product(std::begin(bufidx.second),
                                                                std::end(bufidx.second),
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
                           ::testing::Range(2, rocfft_scoped_device::device_count() + 1)
#else
                           ::testing::Range(1, rocfft_scoped_device::device_count() + 1)
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
            auto         ret = std::make_tuple(realcomplex, df.direction, df.informat, dim, ngpus);
            return ret;
        }),
    [](const testing::TestParamInfo<hipfftxtunitdesc::ParamType>& info) {
        const auto  realcomplex = std::get<0>(info.param);
        const auto  direction   = std::get<1>(info.param);
        const auto  format      = std::get<2>(info.param);
        const auto  dimension   = std::get<3>(info.param);
        const auto  ngpus       = std::get<4>(info.param);
        std::string name        = realcomplex ? "rc" : "cc";
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
    const size_t ngpus = rocfft_scoped_device::device_count();
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
           && std::get<1>(val).informat == format)
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
        if(rocfft_backend || realcomplex)
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

struct hipfftxtexec_params
{
    fft_transform_type  dft_type;
    hipfftXtSubFormat   input_desc_format;
    size_t              ngpus;
    size_t              batch;
    std::vector<size_t> transform_lengths;
    fft_precision       precision;

    inline int exec_dir() const
    {
        return is_fwd(dft_type) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
    }

    template <fft_io io>
    inline std::vector<size_t> length_spans() const
    {
        auto ret = transform_lengths;
        if((dft_type == fft_transform_type_real_forward && io == fft_io_out)
           || (dft_type == fft_transform_type_real_inverse && io == fft_io_in))
            ret.back() = ret.back() / 2 + 1;
        return ret;
    }

    template <fft_io io>
    inline bool has_real_data_on() const
    {
        if constexpr(io == fft_io_in)
            return dft_type == fft_transform_type_real_forward;
        else if constexpr(io == fft_io_out)
            return dft_type == fft_transform_type_real_inverse;
        return false;
    }

    inline hipfftType_t hipfft_transform_type() const
    {
        if(precision != fft_precision_single && precision != fft_precision_double)
            throw std::logic_error("invalid precision attempted used in hipfftxtexec test: only "
                                   "single and double are supported");
        const bool single = (precision == fft_precision_single);
        switch(dft_type)
        {
        case fft_transform_type_real_forward:
            return single ? HIPFFT_R2C : HIPFFT_D2Z;
        case fft_transform_type_real_inverse:
            return single ? HIPFFT_C2R : HIPFFT_Z2D;
        case fft_transform_type_complex_forward:
            [[fallthrough]];
        case fft_transform_type_complex_inverse:
            return single ? HIPFFT_C2C : HIPFFT_Z2Z;
        default:
            throw std::logic_error("invalid dft_type");
        }
    }

    inline fft_result_placement placement() const
    {
        return (input_desc_format == HIPFFT_XT_FORMAT_INPLACE
                || input_desc_format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED)
                   ? fft_placement_inplace
                   : fft_placement_notinplace;
    }

    template <fft_io io>
    inline std::vector<size_t> global_strides() const
    {
        return default_strides(dft_type, placement(), io, transform_lengths);
    }
    template <fft_io io>
    inline size_t global_dist() const
    {
        return default_distance(dft_type, placement(), io, transform_lengths, batch);
    }

    template <fft_io io>
    inline size_t global_byte_size() const
    {
        return calc_global_byte_size<io>();
    }

    inline std::string str() const
    {
        std::ostringstream oss;
        oss << (precision == fft_precision_single ? "single" : "double") << "_"
            << transform_type_name(dft_type) << "_" << format_name(input_desc_format) << "_"
            << "batch_" << batch << "_"
            << "lengths_";
        for(auto len : transform_lengths)
            oss << len << "_";
        oss << "ngpus_" << ngpus;

        return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& stream, const hipfftxtexec_params& params)
    {
        stream << "precision: " << (params.precision == fft_precision_single ? "single" : "double")
               << ", "
               << "dft type: " << transform_type_name(params.dft_type) << ", "
               << "input format: " << format_name(params.input_desc_format) << ", "
               << "ngpus: " << params.ngpus << ", "
               << "batch: " << params.batch << ", "
               << "transform lengths: (";
        for(auto it = params.transform_lengths.begin(); it != params.transform_lengths.end(); ++it)
            stream << (it != params.transform_lengths.begin() ? ", " : "") << *it;
        stream << ")";
        return stream;
    }

    fft_params make_params_for_reference_cpu() const
    {
        fft_params tmp;
        tmp.length    = transform_lengths;
        tmp.precision = precision;
        // always do it out-of-place for reference CPU (requirement for reference_fft_data_t construction)
        // but use the same data layout as the test case's global strides/distances (in-place or
        // out-of-place) for direct use in hipfftXtMemcpy of input data.
        tmp.placement      = fft_placement_notinplace;
        tmp.transform_type = dft_type;
        tmp.nbatch         = batch;
        tmp.run_callbacks  = fft_callback_type_none;
        tmp.istride        = global_strides<fft_io_in>();
        tmp.ostride        = global_strides<fft_io_out>();
        tmp.idist          = global_dist<fft_io_in>();
        tmp.odist          = global_dist<fft_io_out>();
        tmp.validate(); // sets itype, otype, isize, osize, etc. from the above
        return tmp;
    }

    // Return true if the input_desc_format is a valid (supported) format for
    // the given dft_type.
    inline bool supported() const
    {
        switch(dft_type)
        {
        case fft_transform_type_real_forward:
            return input_desc_format == HIPFFT_XT_FORMAT_INPLACE;
        case fft_transform_type_real_inverse:
            return input_desc_format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
        case fft_transform_type_complex_forward:
            [[fallthrough]];
        case fft_transform_type_complex_inverse:
            return input_desc_format == HIPFFT_XT_FORMAT_INPLACE
                   || input_desc_format == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
        default:
            throw std::logic_error("unsupported dft_type in hipfftxtexec_params::supported()");
        }
    }

private:
    template <fft_io io, bool nested_call = false>
    inline size_t calc_global_byte_size() const
    {
        const size_t real_size
            = (precision == fft_precision_single) ? sizeof(float) : sizeof(double);
        const size_t elem_size = has_real_data_on<io>() ? real_size : 2 * real_size;
        auto         ret       = std::max(compute_ptrdiff(
                                length_spans<io>(), global_strides<io>(), batch, global_dist<io>()),
                            global_dist<io>() * batch)
                   * elem_size;
        if constexpr(!nested_call)
        {
            if(placement() == fft_placement_inplace)
                ret = std::max(ret, calc_global_byte_size<other<io>(), true>());
        }
        return ret;
    }
};

// Parameters are real/complex, direction, format, dimension, and number of GPUs.
class hipfftxtexec : public ::testing::TestWithParam<hipfftxtexec_params>
{
};

// FIXME: document
TEST_P(hipfftxtexec, hipfftxtexec)
{
    try
    {
        const auto& params = GetParam();
        const auto  rank   = params.transform_lengths.size();

        ASSERT_TRUE(rank == 2 || rank == 3) << "only 2D and 3D use cases supported in this test.";

        // Create FFTW reference for comparison
        reference_fft_data_t reference_results{params.make_params_for_reference_cpu()};
        if(reference_results.needs_computing())
        {
            if(reference_results.needs_input_initialization())
                reference_results.initialize_input(fft_input_generator_host);
            reference_results.launch_async_compute();
        }

        std::vector<int> gpus(params.ngpus);
        std::iota(gpus.begin(), gpus.end(), 0);

        // Create the xt plan and descriptor:
        hipfftHandle_wrapper_t plan;

        auto hipfft_rt = plan.alloc_with_err();
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS);

        hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtSetGPUs failed";

        std::vector<size_t> workSize(params.ngpus);
        switch(rank)
        {
        case 2:
            hipfft_rt = hipfftMakePlan2d(plan,
                                         params.transform_lengths[0],
                                         params.transform_lengths[1],
                                         params.hipfft_transform_type(),
                                         workSize.data());
            break;
        case 3:
            hipfft_rt = hipfftMakePlan3d(plan,
                                         params.transform_lengths[0],
                                         params.transform_lengths[1],
                                         params.transform_lengths[2],
                                         params.hipfft_transform_type(),
                                         workSize.data());
            break;
        default:
            FAIL() << "Test infrastructure only supports 2D and 3D transforms";
        }
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftMakePlan2/3d failed with return code "
                                             << hipfft_rt << "=" << hipfftResult_string(hipfft_rt);
        if(verbose > 2)
            std::cout << "plan created\n";

        hipfftLibXtDesc_wrapper_t mydesc;
        hipfft_rt = mydesc.alloc_with_err(plan, params.input_desc_format);
        if(params.supported())
        {
            ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS) << "hipfftXtMalloc failed with code " << hipfft_rt
                                                 << " (" << hipfftResult_string(hipfft_rt) << ")";
        }
        else
        {
            if constexpr(rocfft_backend)
            {
                ASSERT_EQ(hipfft_rt, HIPFFT_NOT_IMPLEMENTED)
                    << "hipfftXtMalloc did not return HIPFFT_NOT_IMPLEMENTED for a "
                       "supposedly-unimplemented use case (returned code "
                    << hipfft_rt << " = " << hipfftResult_string(hipfft_rt) << ")";
            }
            ASSERT_NE(hipfft_rt, HIPFFT_SUCCESS)
                << "hipfftXtMalloc passed but should have failed for a supposedly-unimplemented "
                   "use case (returned code "
                << hipfft_rt << " = " << hipfftResult_string(hipfft_rt) << ")";
            GTEST_SUCCEED();
            return;
        }
        if(verbose > 2)
            std::cout << "descriptor allocated\n";

        for(size_t igpu = 0; igpu < gpus.size(); ++igpu)
        {
            if(verbose > 3)
                std::cout << "buffer " << igpu << " size: " << (*mydesc).descriptor->size[igpu]
                          << " = " << byte_size_to_str((*mydesc).descriptor->size[igpu]) << "\n";
            // TODO: handle case where some GPUs don't have data because there isn't enough to go
            // around.  (Particularly for multi-batch cases.)
            ASSERT_NE((*mydesc).descriptor->size[igpu], size_t(0))
                << "gpu buffer size is zero for gpu " << igpu;
        }

        if(verbose > 2)
            std::cout << "starting hipfftXtMemcpy...\n";

        hipfft_rt = hipfftXtMemcpy(plan,
                                   mydesc.get_raw(),
                                   reference_results.get_buffers<fft_io_in>().front().data(),
                                   HIPFFT_COPY_HOST_TO_DEVICE);
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
            << "hipfftXtMemcpy H2D"
            << " failed with code " << hipfft_rt << " (" << hipfftResult_string(hipfft_rt) << ")";

        if(verbose > 2)
            std::cout << "finished hipfftXtMemcpy\n";

        // Execute the plan
        hipfft_rt = hipfftXtExecDescriptor(plan, mydesc, mydesc, params.exec_dir());
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
            << "hipfftXtExecDescriptor failed with code " << hipfft_rt << " ("
            << hipfftResult_string(hipfft_rt) << ")";

        std::vector<hostbuf> mgpu_output(1);
        mgpu_output[0].alloc(params.global_byte_size<fft_io_out>());

        hipfft_rt = hipfftXtMemcpy(
            plan, mgpu_output[0].data(), mydesc.get_raw(), HIPFFT_COPY_DEVICE_TO_HOST);
        ASSERT_EQ(hipfft_rt, HIPFFT_SUCCESS)
            << "hipfftXtMemcpy D2H"
            << " failed with code " << hipfft_rt << " (" << hipfftResult_string(hipfft_rt) << ")";

        // Compare multi-GPU output against FFTW reference
        const auto total_length
            = product(params.transform_lengths.begin(), params.transform_lengths.end());
        const auto   cpu_output_norm = reference_results.get_norm<fft_io_out>(params.batch).get();
        const double linf_cutoff
            = type_epsilon(params.precision) * cpu_output_norm.l_inf * log(total_length);

        const auto diff = distance(reference_results.get_buffers<fft_io_out>(),
                                   mgpu_output,
                                   params.length_spans<fft_io_out>(),
                                   params.batch /* may be smaller than ref_cpu_params' */,
                                   params.precision,
                                   reference_results.get_params().otype,
                                   reference_results.get_params().ostride,
                                   reference_results.get_params().odist,
                                   reference_results.get_params().otype,
                                   params.global_strides<fft_io_out>(),
                                   params.global_dist<fft_io_out>(),
                                   nullptr,
                                   linf_cutoff,
                                   {0},
                                   {0});
        if(verbose > 1)
            std::cout << "linf: " << diff.l_inf << " l2: " << diff.l_2 << " cutoff: " << linf_cutoff
                      << "\n";
        EXPECT_LE(diff.l_inf, linf_cutoff) << "l_inf tolerance failure. cutoff: " << linf_cutoff;
    }
    ROCFFT_CATCH_TEST_EXCEPTIONS
}

INSTANTIATE_TEST_SUITE_P(
    hipfftxtexec,
    hipfftxtexec,
    ::testing::ConvertGenerator(
        ::testing::Combine(::testing::ValuesIn(trans_type_range_full),
                           ::testing::ValuesIn(test_precisions),
                           ::testing::ValuesIn(test_batch_sizes),
                           ::testing::ValuesIn(test_lengths),
                           ::testing::ValuesIn(test_subformats),
#ifdef __HIP_PLATFORM_NVIDIA__
                           ::testing::Range(2, rocfft_scoped_device::device_count() + 1)
#else
                           ::testing::Range(1, rocfft_scoped_device::device_count() + 1)
#endif
                               ),
        [](const std::tuple<fft_transform_type,
                            fft_precision,
                            size_t,
                            std::vector<size_t>,
                            hipfftXtSubFormat,
                            int>& t) {
            // This lambda recombines the nested tuples into a flat struct to
            // make test parametrization simpler.
            hipfftxtexec_params ret;
            ret.dft_type          = std::get<0>(t);
            ret.precision         = std::get<1>(t);
            ret.batch             = std::get<2>(t);
            ret.transform_lengths = std::get<3>(t);
            ret.input_desc_format = std::get<4>(t);
            ret.ngpus             = std::get<5>(t);
            return ret;
        }),
    [](const testing::TestParamInfo<hipfftxtexec::ParamType>& info) { return info.param.str(); });
