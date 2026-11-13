// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Direct checks of the maf1_0 / maf2_0 device fills (hipblaslt_init_device): the GEMM tests compare
// against a CPU reference built from the same buffers, so they cannot see a wrong value, index or
// stream. Each test fills a small device buffer, copies it back and checks the values. No hipBLASLt
// kernels run.

#include <gtest/gtest.h>

#include "hipblaslt_init.hpp"
#include "hipblaslt_test.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace
{
    // Fills `elements` floats on the device with NaN, runs the initializer over them and returns
    // the buffer, so cells it did not write stay NaN.
    std::vector<float> fillFloat(ABC_dims                 abc,
                                 hipblaslt_initialization init,
                                 size_t                   M,
                                 size_t                   N,
                                 size_t                   lda,
                                 size_t                   stride,
                                 size_t                   batch_count,
                                 size_t                   elements)
    {
        float* d = nullptr;
        EXPECT_EQ(hipMalloc(&d, elements * sizeof(float)), hipSuccess);
        EXPECT_EQ(hipMemset(d, 0xFF, elements * sizeof(float)), hipSuccess); // NaN
        hipblaslt_init_device(
            abc, init, false, d, M, N, lda, HIP_R_32F, stride, batch_count, -1, 0);
        std::vector<float> h(elements);
        EXPECT_EQ(hipMemcpy(h.data(), d, h.size() * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        (void)hipFree(d);
        return h;
    }

    // A nonzero batch stride below lda interleaves the batches: M = 2, N = 3, lda = 4, stride = 2
    // puts batch 0 at offsets {0, 1, 4, 5, 8, 9} and batch 1 at {2, 3, 6, 7, 10, 11}. Every batch
    // must be filled, with its own logical index for maf1_0.
    TEST(MafInitDevice_pre_checkin, interleaved_batches_are_all_filled)
    {
        const size_t M = 2, N = 3, lda = 4, stride = 2, batches = 2, total = 12;
        const auto   trig = fillFloat(
            ABC_dims::A, hipblaslt_initialization::maf1_0, M, N, lda, stride, batches, total);
        for(size_t b = 0; b < batches; b++)
            for(size_t j = 0; j < N; j++)
                for(size_t i = 0; i < M; i++)
                    EXPECT_EQ(trig[b * stride + j * lda + i],
                              static_cast<float>(std::sin(double(i + j * M + b * M * N))))
                        << "i " << i << " j " << j << " batch " << b;

        for(auto init :
            {hipblaslt_initialization::maf2_0, hipblaslt_initialization::maf2_0_unit_scale})
        {
            const auto normal = fillFloat(ABC_dims::A, init, M, N, lda, stride, batches, total);
            for(size_t k = 0; k < total; k++)
                EXPECT_FALSE(std::isnan(normal[k]))
                    << hipblaslt_initialization2string(init) << " offset " << k;
        }
    }
} // namespace
