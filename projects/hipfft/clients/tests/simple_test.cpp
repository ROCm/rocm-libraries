// Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights
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

#include "hipfft/hipfft.h"
#include <fftw3.h>
#include <gtest/gtest.h>
#include <hip/hip_vector_types.h>
#include <vector>

#include "../hipfft_params.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

// Function to return maximum error for float and double types.
template <typename Tfloat>
inline double type_epsilon();
template <>
inline double type_epsilon<float>()
{
    return 1e-6;
}
template <>
inline double type_epsilon<double>()
{
    return 1e-7;
}

TEST(hipfftTest, Create1dPlan)
{
    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    size_t length = 1024;
    ASSERT_EQ(hipfftPlan1d(&plan, length, HIPFFT_C2C, 1), HIPFFT_SUCCESS);

    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
}

TEST(hipfftTest, CreatePlanMany)
{
    int const  rank         = 3;
    int const  nX           = 64;
    int const  nY           = 128;
    int const  nZ           = 23;
    int        n[3]         = {nX, nY, nZ};
    int        inembed[3]   = {nX, nY, nZ};
    int*       inembed_null = nullptr;
    int const  istride      = 1;
    int const  idist        = nX * nY * nZ;
    int        onembed[3]   = {nX, nY, nZ};
    int*       onembed_null = nullptr;
    int const  ostride      = 1;
    int const  odist        = nX * nY * nZ;
    hipfftType type         = HIPFFT_C2C;
    int const  batch        = 1000;
    size_t     workSize;

    // Tests plan creation with null and not null
    // combinations of inembed and onembed.
    //
    // Valid combinations:
    //                      inembed == null && onembed == null
    //                      or
    //                      inembed != null && onembed != null
    //
    // otherwise HIPFFT_INVALID_VALUE should be
    // returned to maintain compatibility with cuFFT

    // inembed == null && onembed == null
    {
        hipfftHandle plan_valid_1 = hipfft_params::INVALID_PLAN_HANDLE;
        ASSERT_EQ(hipfftCreate(&plan_valid_1), HIPFFT_SUCCESS);
        auto ret_hipfft = hipfftMakePlanMany(plan_valid_1,
                                             rank,
                                             (int*)n,
                                             inembed_null,
                                             istride,
                                             idist,
                                             onembed_null,
                                             ostride,
                                             odist,
                                             type,
                                             batch,
                                             &workSize);
        ASSERT_EQ(ret_hipfft, HIPFFT_SUCCESS)
            << "inembed == null && onembed == null failed: " << hipfftResult_string(ret_hipfft);
        ASSERT_EQ(hipfftSetAutoAllocation(plan_valid_1, 0), HIPFFT_SUCCESS);
        ASSERT_EQ(hipfftDestroy(plan_valid_1), HIPFFT_SUCCESS);
    }

    // inembed != null && onembed != null
    {
        hipfftHandle plan_valid_2 = hipfft_params::INVALID_PLAN_HANDLE;
        ASSERT_EQ(hipfftCreate(&plan_valid_2), HIPFFT_SUCCESS);
        auto ret_hipfft = hipfftMakePlanMany(plan_valid_2,
                                             rank,
                                             (int*)n,
                                             inembed,
                                             istride,
                                             idist,
                                             onembed,
                                             ostride,
                                             odist,
                                             type,
                                             batch,
                                             &workSize);
        ASSERT_EQ(ret_hipfft, HIPFFT_SUCCESS)
            << "inembed != null && onembed != null failed: " << hipfftResult_string(ret_hipfft);
        ASSERT_EQ(hipfftSetAutoAllocation(plan_valid_2, 0), HIPFFT_SUCCESS);
        ASSERT_EQ(hipfftDestroy(plan_valid_2), HIPFFT_SUCCESS);
    }

    // inembed != null && onembed == null
    {
        hipfftHandle plan_invalid_1 = hipfft_params::INVALID_PLAN_HANDLE;
        ASSERT_EQ(hipfftCreate(&plan_invalid_1), HIPFFT_SUCCESS);
        auto ret_hipfft = hipfftMakePlanMany(plan_invalid_1,
                                             rank,
                                             (int*)n,
                                             inembed,
                                             istride,
                                             idist,
                                             onembed_null,
                                             ostride,
                                             odist,
                                             type,
                                             batch,
                                             &workSize);
        ASSERT_EQ(ret_hipfft, HIPFFT_INVALID_VALUE)
            << "inembed != null && onembed == null failed: " << hipfftResult_string(ret_hipfft);
        ASSERT_EQ(hipfftDestroy(plan_invalid_1), HIPFFT_SUCCESS);
    }

    // inembed == null && onembed != null
    {
        hipfftHandle plan_invalid_2 = hipfft_params::INVALID_PLAN_HANDLE;
        ASSERT_EQ(hipfftCreate(&plan_invalid_2), HIPFFT_SUCCESS);
        auto ret_hipfft = hipfftMakePlanMany(plan_invalid_2,
                                             rank,
                                             (int*)n,
                                             inembed_null,
                                             istride,
                                             idist,
                                             onembed,
                                             ostride,
                                             odist,
                                             type,
                                             batch,
                                             &workSize);
        ASSERT_EQ(ret_hipfft, HIPFFT_INVALID_VALUE)
            << "inembed == null && onembed != null failed: " << hipfftResult_string(ret_hipfft);
        ASSERT_EQ(hipfftDestroy(plan_invalid_2), HIPFFT_SUCCESS);
    }
}

TEST(hipfftTest, CreatePlanMany64)
{
    int const           rank               = 3;
    long long int const nX                 = 64;
    long long int const nY                 = 128;
    long long int const nZ                 = 23;
    long long int       n[3]               = {nX, nY, nZ};
    long long int       inembed[3]         = {nX, nY, nZ};
    long long int const istride            = 1;
    long long int const idist              = nX * nY * nZ;
    long long int       onembed[3]         = {nX, nY, nZ};
    long long int       onembed_invalid[3] = {nX, nY, -nZ};
    long long int const ostride            = 1;
    long long int const odist              = nX * nY * nZ;
    hipfftType          type               = HIPFFT_C2C;
    long long int const batch              = 1000;
    long long int const batch_invalid      = -2;
    size_t              workSize;

    // Tests the 64-bit version of plan creation
    // with valid/invalid data layouts.

    // First test with a valid data layout
    {
        hipfftHandle plan_valid = hipfft_params::INVALID_PLAN_HANDLE;
        ASSERT_EQ(hipfftCreate(&plan_valid), HIPFFT_SUCCESS);
        auto ret_hipfft = hipfftMakePlanMany64(plan_valid,
                                               rank,
                                               (long long int*)n,
                                               inembed,
                                               istride,
                                               idist,
                                               onembed,
                                               ostride,
                                               odist,
                                               type,
                                               batch,
                                               &workSize);
        ASSERT_EQ(ret_hipfft, HIPFFT_SUCCESS);
        ASSERT_EQ(hipfftSetAutoAllocation(plan_valid, 0), HIPFFT_SUCCESS);
        ASSERT_EQ(hipfftDestroy(plan_valid), HIPFFT_SUCCESS);
    }

    // invalid data layout (n array has a negative entry). only test rocFFT
    // backend, since it's more strict
#ifdef __HIP_PLATFORM_AMD__
    long long int n_invalid[3] = {nX, -nY, nZ};
    {
        hipfftHandle plan_invalid_1 = hipfft_params::INVALID_PLAN_HANDLE;
        ASSERT_EQ(hipfftCreate(&plan_invalid_1), HIPFFT_SUCCESS);
        auto ret_hipfft = hipfftMakePlanMany64(plan_invalid_1,
                                               rank,
                                               (long long int*)n_invalid,
                                               inembed,
                                               istride,
                                               idist,
                                               onembed,
                                               ostride,
                                               odist,
                                               type,
                                               batch,
                                               &workSize);
        ASSERT_EQ(ret_hipfft, HIPFFT_INVALID_VALUE);
        ASSERT_EQ(hipfftSetAutoAllocation(plan_invalid_1, 0), HIPFFT_SUCCESS);
        ASSERT_EQ(hipfftDestroy(plan_invalid_1), HIPFFT_SUCCESS);
    }
#endif

    // invalid data layout (onembed array has a negative entry)
    {
        hipfftHandle plan_invalid_2 = hipfft_params::INVALID_PLAN_HANDLE;
        ASSERT_EQ(hipfftCreate(&plan_invalid_2), HIPFFT_SUCCESS);
        auto ret_hipfft = hipfftMakePlanMany64(plan_invalid_2,
                                               rank,
                                               (long long int*)n,
                                               inembed,
                                               istride,
                                               idist,
                                               onembed_invalid,
                                               ostride,
                                               odist,
                                               type,
                                               batch,
                                               &workSize);
        ASSERT_EQ(ret_hipfft, HIPFFT_INVALID_SIZE);
        ASSERT_EQ(hipfftSetAutoAllocation(plan_invalid_2, 0), HIPFFT_SUCCESS);
        ASSERT_EQ(hipfftDestroy(plan_invalid_2), HIPFFT_SUCCESS);
    }

    // invalid data layout (batch is negative)
    {
        hipfftHandle plan_invalid_3 = hipfft_params::INVALID_PLAN_HANDLE;
        ASSERT_EQ(hipfftCreate(&plan_invalid_3), HIPFFT_SUCCESS);
        auto ret_hipfft = hipfftMakePlanMany64(plan_invalid_3,
                                               rank,
                                               (long long int*)n,
                                               inembed,
                                               istride,
                                               idist,
                                               onembed,
                                               ostride,
                                               odist,
                                               type,
                                               batch_invalid,
                                               &workSize);
        ASSERT_EQ(ret_hipfft, HIPFFT_INVALID_SIZE);
        ASSERT_EQ(hipfftSetAutoAllocation(plan_invalid_3, 0), HIPFFT_SUCCESS);
        ASSERT_EQ(hipfftDestroy(plan_invalid_3), HIPFFT_SUCCESS);
    }
}

TEST(hipfftTest, hipfftGetSizeMany)
{
    int const  rank       = 3;
    int const  nX         = 33;
    int const  nY         = 128;
    int const  nZ         = 100;
    int        n[3]       = {nX, nY, nZ};
    int        inembed[3] = {nX, nY, nZ};
    int const  istride    = 1;
    int const  idist      = nX * nY * nZ;
    int        onembed[3] = {nX, nY, nZ};
    int const  ostride    = 1;
    int const  odist      = nX * nY * nZ;
    hipfftType type       = HIPFFT_C2C;
    int const  batch      = 1;
    size_t     workSize;

    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    auto ret_hipfft = hipfftGetSizeMany(plan,
                                        rank,
                                        (int*)n,
                                        inembed,
                                        istride,
                                        idist,
                                        onembed,
                                        ostride,
                                        odist,
                                        type,
                                        batch,
                                        &workSize);
    ASSERT_EQ(ret_hipfft, HIPFFT_SUCCESS);
    ASSERT_EQ(hipfftSetAutoAllocation(plan, 0), HIPFFT_SUCCESS);
    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
}

TEST(hipfftTest, hipfftGetSizeMany64)
{
    int const           rank       = 3;
    long long int const nX         = 133;
    long long int const nY         = 354;
    long long int const nZ         = 256;
    long long int       n[3]       = {nX, nY, nZ};
    long long int       inembed[3] = {nX, nY, nZ};
    long long int const istride    = 1;
    long long int const idist      = nX * nY * nZ;
    long long int       onembed[3] = {nX, nY, nZ};
    long long int const ostride    = 1;
    long long int const odist      = nX * nY * nZ;
    hipfftType          type       = HIPFFT_C2C;
    long long int const batch      = 2;
    size_t              workSize;

    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    auto ret_hipfft = hipfftGetSizeMany64(plan,
                                          rank,
                                          (long long int*)n,
                                          inembed,
                                          istride,
                                          idist,
                                          onembed,
                                          ostride,
                                          odist,
                                          type,
                                          batch,
                                          &workSize);
    ASSERT_EQ(ret_hipfft, HIPFFT_SUCCESS);
    ASSERT_EQ(hipfftSetAutoAllocation(plan, 0), HIPFFT_SUCCESS);
    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeC2C)
{
    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    size_t n        = 1024;
    size_t workSize = 0;

    ASSERT_EQ(hipfftMakePlan1d(plan, n, HIPFFT_C2C, 1, &workSize), HIPFFT_SUCCESS);

#ifdef __HIP_PLATFORM_AMD__
    // No extra work buffer for C2C
    EXPECT_EQ(workSize, 0);
#endif
    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeR2C)
{
    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    // real forward transform cannot modify input, so we need to pick
    // a sufficiently small N such that rocFFT can fuse
    // post-processing into one kernel and avoid a temp buffer
    size_t n        = 256;
    size_t workSize = 0;

    ASSERT_EQ(hipfftMakePlan1d(plan, n, HIPFFT_R2C, 1, &workSize), HIPFFT_SUCCESS);

#ifdef __HIP_PLATFORM_AMD__
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(n % 2 == 0)
    {
        EXPECT_EQ(workSize, 0);
    }
    else
    {
        EXPECT_EQ(workSize, 2 * n * sizeof(float));
    }
#endif
    EXPECT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeC2R)
{
    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t workSize = 0;

    ASSERT_EQ(hipfftMakePlan1d(plan, n, HIPFFT_C2R, 1, &workSize), HIPFFT_SUCCESS);

#ifdef __HIP_PLATFORM_AMD__
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(n % 2 == 0)
    {
        EXPECT_EQ(workSize, 0);
    }
    else
    {
        EXPECT_EQ(workSize, 2 * n * sizeof(float));
    }
#endif
    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeD2Z)
{
    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    // real forward transform cannot modify input, so we need to pick
    // a sufficiently small N such that rocFFT can fuse
    // post-processing into one kernel and avoid a temp buffer
    size_t n        = 256;
    size_t batch    = 1000;
    size_t workSize = 0;

    ASSERT_EQ(hipfftMakePlan1d(plan, n, HIPFFT_D2Z, batch, &workSize), HIPFFT_SUCCESS);

#ifdef __HIP_PLATFORM_AMD__
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(n % 2 == 0)
    {
        EXPECT_EQ(workSize, 0);
    }
    else
    {
        EXPECT_EQ(workSize, 2 * n * sizeof(double));
    }
#endif

    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeZ2D)
{
    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t batch    = 1000;
    size_t workSize = 0;

    ASSERT_EQ(hipfftMakePlan1d(plan, n, HIPFFT_Z2D, batch, &workSize), HIPFFT_SUCCESS);

#ifdef __HIP_PLATFORM_AMD__
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(n % 2 == 0)
    {
        EXPECT_EQ(workSize, 0);
    }
    else
    {
        EXPECT_EQ(workSize, 2 * n * sizeof(double));
    }
#endif

    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
}

#ifdef __HIP_PLATFORM_AMD__
TEST(hipfftTest, CheckNullWorkBuffer)
{
    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t batch    = 1000;
    size_t workSize = 0;

    ASSERT_EQ(hipfftMakePlan1d(plan, n, HIPFFT_Z2D, batch, &workSize), HIPFFT_SUCCESS);
    EXPECT_EQ(hipfftSetWorkArea(plan, nullptr), HIPFFT_SUCCESS);
    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
}
#endif

TEST(hipfftTest, RunR2C)
{
    const size_t N = 4096;
    float        in[N];
    for(size_t i = 0; i < N; i++)
        in[i] = i + (i % 3) - (i % 7);

    hipfftReal*    d_in;
    hipfftComplex* d_out;
    ASSERT_EQ(hipMalloc(&d_in, N * sizeof(hipfftReal)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_out, (N / 2 + 1) * sizeof(hipfftComplex)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_in, in, N * sizeof(hipfftReal), hipMemcpyHostToDevice), hipSuccess);

    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);
    size_t workSize;
    ASSERT_EQ(hipfftMakePlan1d(plan, N, HIPFFT_R2C, 1, &workSize), HIPFFT_SUCCESS);

    EXPECT_EQ(hipfftExecR2C(plan, d_in, d_out), HIPFFT_SUCCESS);

    std::vector<hipfftComplex> out(N / 2 + 1);
    ASSERT_EQ(hipMemcpy(&out[0], d_out, (N / 2 + 1) * sizeof(hipfftComplex), hipMemcpyDeviceToHost),
              hipSuccess);

    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
    ASSERT_EQ(hipFree(d_in), hipSuccess);
    ASSERT_EQ(hipFree(d_out), hipSuccess);
    ;
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(N % 2 != 0)
    {
        EXPECT_TRUE(workSize != 0);
    }

    double ref_in[N];
    for(size_t i = 0; i < N; i++)
        ref_in[i] = in[i];

    fftw_complex* ref_out;
    fftw_plan     ref_p;

    ref_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    ref_p   = fftw_plan_dft_r2c_1d(N, ref_in, ref_out, FFTW_ESTIMATE);
    fftw_execute(ref_p);

    double maxv  = 0;
    double nrmse = 0; // normalized root mean square error
    for(size_t i = 0; i < (N / 2 + 1); i++)
    {
        // printf("element %d: FFTW result %f, %f; hipFFT result %f, %f \n", (int)i, ref_out[i][0], ref_out[i][1], out[i].x, out[i].y);
        double dr = ref_out[i][0] - out[i].x;
        double di = ref_out[i][1] - out[i].y;
        maxv      = fabs(ref_out[i][0]) > maxv ? fabs(ref_out[i][0]) : maxv;
        maxv      = fabs(ref_out[i][1]) > maxv ? fabs(ref_out[i][1]) : maxv;

        nrmse += ((dr * dr) + (di * di));
    }
    nrmse /= (double)((N / 2 + 1));
    nrmse = sqrt(nrmse);
    nrmse /= maxv;

    EXPECT_TRUE(nrmse < type_epsilon<double>());
    fftw_destroy_plan(ref_p);
    fftw_free(ref_out);
}

// ask for a transform whose parameters are only valid out-of-place.
// since hipFFT generates both in-place and out-place plans up front
// (because it's not told about the placement until exec time), this
// ensures that a failure to create an in-place plan doesn't prevent
// the out-place plan from working.
TEST(hipfftTest, OutplaceOnly)
{
    static const int N_in_const  = 4;
    static const int N_out_const = N_in_const / 2 + 1;
    // mutable sizes for passing to hipFFT
    int   N_in  = N_in_const;
    int   N_out = N_out_const;
    float in[N_in_const];
    for(int i = 0; i < N_in; i++)
        in[i] = i + (i % 3) - (i % 7);

    hipfftReal*    d_in;
    hipfftComplex* d_out;
    ASSERT_EQ(hipMalloc(&d_in, N_in * sizeof(hipfftReal)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_out, N_out * sizeof(hipfftComplex)), hipSuccess);

    ASSERT_EQ(hipMemcpy(d_in, in, N_in * sizeof(hipfftReal), hipMemcpyHostToDevice), hipSuccess);

    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);

    ASSERT_EQ(hipfftPlanMany(&plan, 1, &N_in, &N_in, 1, N_in, &N_out, 1, N_out, HIPFFT_R2C, 1),
              HIPFFT_SUCCESS);

    ASSERT_EQ(plan == hipfft_params::INVALID_PLAN_HANDLE, false);

    ASSERT_EQ(hipfftExecR2C(plan, d_in, d_out), HIPFFT_SUCCESS) << "hipfftExecR2C failed";

    std::vector<hipfftComplex> out(N_out);
    ASSERT_EQ(hipMemcpy(out.data(), d_out, N_out * sizeof(hipfftComplex), hipMemcpyDeviceToHost),
              hipSuccess);

    // in-place transform isn't really *supposed* to work - this
    // might or might not fail but we can at least check that it
    // doesn't blow up.
    //hipfftExecR2C(plan, reinterpret_cast<hipfftReal*>(d_out), d_out);

    ASSERT_EQ(hipfftDestroy(plan), HIPFFT_SUCCESS);
    ASSERT_EQ(hipFree(d_in), hipSuccess);
    ASSERT_EQ(hipFree(d_out), hipSuccess);

    double ref_in[N_in_const];
    for(int i = 0; i < N_in_const; i++)
        ref_in[i] = in[i];

    fftw_complex* ref_out;
    fftw_plan     ref_p;

    ref_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N_out);
    ref_p   = fftw_plan_dft_r2c_1d(N_in, ref_in, ref_out, FFTW_ESTIMATE);
    fftw_execute(ref_p);

    double maxv  = 0;
    double nrmse = 0; // normalized root mean square error
    for(int i = 0; i < N_out; i++)
    {
        // printf("element %d: FFTW result %f, %f; hipFFT result %f, %f \n", (int)i, ref_out[i][0], ref_out[i][1], out[i].x, out[i].y);
        double dr = ref_out[i][0] - out[i].x;
        double di = ref_out[i][1] - out[i].y;
        maxv      = fabs(ref_out[i][0]) > maxv ? fabs(ref_out[i][0]) : maxv;
        maxv      = fabs(ref_out[i][1]) > maxv ? fabs(ref_out[i][1]) : maxv;

        nrmse += ((dr * dr) + (di * di));
    }
    nrmse /= (double)(N_out);
    nrmse = sqrt(nrmse);
    nrmse /= maxv;

    ASSERT_TRUE(nrmse < type_epsilon<double>());
    fftw_destroy_plan(ref_p);
    fftw_free(ref_out);
}
