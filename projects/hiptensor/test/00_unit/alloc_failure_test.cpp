/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

// A HIP allocation failure inside the library used to end the process (exit() in the library's
// error check) or escape the C API as an exception. It must instead be returned as
// HIPTENSOR_STATUS_ALLOC_FAILED, after which the library keeps working.
//
// A complex contraction splits its operands into real and imaginary buffers that the library
// allocates on every execution. Filling the device memory between two executions of the same
// plan makes the first of those allocations fail.

namespace
{
    // ctest reports a test whose output contains "HIPTENSOR_STATUS_ARCH_MISMATCH" as skipped, and
    // gtest prints the expressions of a failed assertion, so that status is named differently.
    constexpr auto kArchMismatch = HIPTENSOR_STATUS_ARCH_MISMATCH;

    // A is M x K, so each of its real and imaginary parts takes M * K floats (4 MiB). That is
    // more than the memory left once the device is filled down to kMinChunkBytes.
    constexpr int64_t     M              = 1024;
    constexpr int64_t     N              = 32;
    constexpr int64_t     K              = 1024;
    constexpr std::size_t kMinChunkBytes = std::size_t(1) << 20;

    // Small integers, so every product and sum is exact in float whatever the summation order.
    hipFloatComplex valueA(int64_t m, int64_t k)
    {
        return make_hipFloatComplex(static_cast<float>((m + 2 * k) % 3 - 1),
                                    static_cast<float>((m + k) % 2));
    }

    hipFloatComplex valueB(int64_t n, int64_t k)
    {
        return make_hipFloatComplex(static_cast<float>((n + k) % 3 - 1),
                                    static_cast<float>((2 * n + k) % 3 - 1));
    }

    // Allocates device memory in decreasing chunks until even kMinChunkBytes cannot be
    // allocated, and frees it on destruction.
    class DeviceMemoryFiller
    {
    public:
        DeviceMemoryFiller()
        {
            std::size_t freeBytes = 0, totalBytes = 0;
            if(hipMemGetInfo(&freeBytes, &totalBytes) != hipSuccess)
            {
                freeBytes = totalBytes = 0;
            }

            for(std::size_t chunk = freeBytes; chunk >= kMinChunkBytes;)
            {
                void* ptr = nullptr;
                if(hipMalloc(&ptr, chunk) == hipSuccess)
                {
                    mChunks.push_back(ptr);
                }
                else
                {
                    chunk /= 2;
                }
            }

            // The failed allocations above set HIP's last error; clear it.
            (void)hipGetLastError();
        }

        ~DeviceMemoryFiller()
        {
            for(auto* ptr : mChunks)
            {
                (void)hipFree(ptr);
            }
        }

        DeviceMemoryFiller(DeviceMemoryFiller const&)            = delete;
        DeviceMemoryFiller& operator=(DeviceMemoryFiller const&) = delete;

    private:
        std::vector<void*> mChunks;
    };
} // namespace

TEST(AllocFailureTest, ComplexContractionReturnsAllocFailedWhenDeviceMemoryIsExhausted)
{
    int             deviceId = 0;
    hipDeviceProp_t props{};
    ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
    ASSERT_EQ(hipGetDeviceProperties(&props, deviceId), hipSuccess);
    if(props.integrated)
    {
        // Device memory is host memory here; filling it would starve the host.
        GTEST_SKIP() << "Not run on an integrated GPU";
    }

    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);

    // Column-major layouts given explicitly.
    int64_t lensA[2]    = {M, K};
    int64_t stridesA[2] = {1, M};
    int64_t lensB[2]    = {N, K};
    int64_t stridesB[2] = {1, N};
    int64_t lensD[2]    = {M, N};
    int64_t stridesD[2] = {1, M};
    int32_t modeA[2]    = {'m', 'k'};
    int32_t modeB[2]    = {'n', 'k'};
    int32_t modeD[2]    = {'m', 'n'};

    hiptensorTensorDescriptor_t descA{}, descB{}, descD{};
    ASSERT_EQ(
        hiptensorCreateTensorDescriptor(handle, &descA, 2, lensA, stridesA, HIPTENSOR_C_32F, 128),
        HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(
        hiptensorCreateTensorDescriptor(handle, &descB, 2, lensB, stridesB, HIPTENSOR_C_32F, 128),
        HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(
        hiptensorCreateTensorDescriptor(handle, &descD, 2, lensD, stridesD, HIPTENSOR_C_32F, 128),
        HIPTENSOR_STATUS_SUCCESS);

    hiptensorOperationDescriptor_t opDesc{};
    auto                           status = hiptensorCreateContraction(handle,
                                             &opDesc,
                                             descA,
                                             modeA,
                                             HIPTENSOR_OP_IDENTITY,
                                             descB,
                                             modeB,
                                             HIPTENSOR_OP_IDENTITY,
                                             nullptr,
                                             nullptr,
                                             HIPTENSOR_OP_IDENTITY,
                                             descD,
                                             modeD,
                                             HIPTENSOR_COMPUTE_DESC_C32F);
    if(status == kArchMismatch)
    {
        hiptensorDestroyTensorDescriptor(descA);
        hiptensorDestroyTensorDescriptor(descB);
        hiptensorDestroyTensorDescriptor(descD);
        hiptensorDestroy(handle);
        GTEST_SKIP() << "Complex contraction is not available on this GPU";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

    hiptensorPlanPreference_t pref{};
    ASSERT_EQ(hiptensorCreatePlanPreference(
                  handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
              HIPTENSOR_STATUS_SUCCESS);

    uint64_t wsSize = 0;
    ASSERT_EQ(
        hiptensorEstimateWorkspaceSize(handle, opDesc, pref, HIPTENSOR_WORKSPACE_DEFAULT, &wsSize),
        HIPTENSOR_STATUS_SUCCESS);

    hiptensorPlan_t plan{};
    ASSERT_EQ(hiptensorCreatePlan(handle, &plan, opDesc, pref, wsSize), HIPTENSOR_STATUS_SUCCESS);

    std::vector<hipFloatComplex> hA(M * K), hB(N * K), hD(M * N);
    for(int64_t k = 0; k < K; ++k)
    {
        for(int64_t m = 0; m < M; ++m)
        {
            hA[m + k * M] = valueA(m, k);
        }
        for(int64_t n = 0; n < N; ++n)
        {
            hB[n + k * N] = valueB(n, k);
        }
    }

    // Independent reference: D(m, n) = sum_k A(m, k) * B(n, k), accumulated exactly in int64.
    std::vector<int64_t> refRe(M * N), refIm(M * N);
    for(int64_t n = 0; n < N; ++n)
    {
        for(int64_t m = 0; m < M; ++m)
        {
            int64_t re = 0, im = 0;
            for(int64_t k = 0; k < K; ++k)
            {
                auto a  = valueA(m, k);
                auto b  = valueB(n, k);
                auto ar = static_cast<int64_t>(hipCrealf(a)),
                     ai = static_cast<int64_t>(hipCimagf(a));
                auto br = static_cast<int64_t>(hipCrealf(b)),
                     bi = static_cast<int64_t>(hipCimagf(b));
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            refRe[m + n * M] = re;
            refIm[m + n * M] = im;
        }
    }

    hipFloatComplex *dA = nullptr, *dB = nullptr, *dD = nullptr;
    void*            workspace = nullptr;
    ASSERT_EQ(hipMalloc(&dA, sizeof(hipFloatComplex) * hA.size()), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, sizeof(hipFloatComplex) * hB.size()), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, sizeof(hipFloatComplex) * hD.size()), hipSuccess);
    if(wsSize > 0)
    {
        ASSERT_EQ(hipMalloc(&workspace, wsSize), hipSuccess);
    }
    ASSERT_EQ(hipMemcpy(dA, hA.data(), sizeof(hipFloatComplex) * hA.size(), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), sizeof(hipFloatComplex) * hB.size(), hipMemcpyHostToDevice),
              hipSuccess);

    hipFloatComplex alpha = make_hipFloatComplex(1.0f, 0.0f);

    auto contractAndCheck = [&](char const* when) {
        ASSERT_EQ(hipMemset(dD, 0, sizeof(hipFloatComplex) * hD.size()), hipSuccess);
        ASSERT_EQ(hiptensorContract(
                      handle, plan, &alpha, dA, dB, nullptr, nullptr, dD, workspace, wsSize, 0),
                  HIPTENSOR_STATUS_SUCCESS)
            << when;
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << when;
        ASSERT_EQ(
            hipMemcpy(hD.data(), dD, sizeof(hipFloatComplex) * hD.size(), hipMemcpyDeviceToHost),
            hipSuccess)
            << when;

        int64_t mismatches = 0;
        for(int64_t i = 0; i < M * N; ++i)
        {
            if(hipCrealf(hD[i]) != static_cast<float>(refRe[i])
               || hipCimagf(hD[i]) != static_cast<float>(refIm[i]))
            {
                if(mismatches++ < 8)
                {
                    ADD_FAILURE() << when << ": D[" << i << "] = (" << hipCrealf(hD[i]) << ", "
                                  << hipCimagf(hD[i]) << "), expected (" << refRe[i] << ", "
                                  << refIm[i] << ")";
                }
            }
        }
        EXPECT_EQ(mismatches, 0) << when;
    };

    // Loads the kernels while memory is available.
    contractAndCheck("before filling device memory");

    {
        DeviceMemoryFiller filler;
        EXPECT_EQ(hiptensorContract(
                      handle, plan, &alpha, dA, dB, nullptr, nullptr, dD, workspace, wsSize, 0),
                  HIPTENSOR_STATUS_ALLOC_FAILED);
        (void)hipDeviceSynchronize();
    }

    // The failure leaves neither a stale HIP error nor a broken plan behind.
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    contractAndCheck("after device memory was released");

    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dD);
    (void)hipFree(workspace);
    hiptensorDestroyPlan(plan);
    hiptensorDestroyPlanPreference(pref);
    hiptensorDestroyOperationDescriptor(opDesc);
    hiptensorDestroyTensorDescriptor(descA);
    hiptensorDestroyTensorDescriptor(descB);
    hiptensorDestroyTensorDescriptor(descD);
    hiptensorDestroy(handle);
}
