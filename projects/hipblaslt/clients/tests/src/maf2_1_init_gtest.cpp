// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Direct checks of the maf2_1 device fill (hipblaslt_init_device): the GEMM tests compare against a
// CPU reference built from the same buffers, so they cannot see a wrong value or stream. Each test
// fills a small device buffer, copies it back and checks the values. No hipBLASLt kernels run.

#include <gtest/gtest.h>

#include "hipblaslt_init.hpp"
#include "hipblaslt_test.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace
{
    // Fills `elements` floats on the device with NaN, runs the initializer over them and returns
    // the buffer, so cells it did not write stay NaN.
    std::vector<float> fill(ABC_dims                 abc,
                            hipblaslt_initialization init,
                            size_t                   M,
                            size_t                   N,
                            size_t                   lda,
                            size_t                   stride,
                            size_t                   batch_count,
                            size_t                   elements,
                            size_t                   batch_index = 0)
    {
        float* d = nullptr;
        EXPECT_EQ(hipMalloc(&d, elements * sizeof(float)), hipSuccess);
        EXPECT_EQ(hipMemset(d, 0xFF, elements * sizeof(float)), hipSuccess); // NaN
        hipblaslt_init_device(
            abc, init, false, d, M, N, lda, HIP_R_32F, stride, batch_count, -1, batch_index);
        std::vector<float> h(elements);
        EXPECT_EQ(hipMemcpy(h.data(), d, h.size() * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        (void)hipFree(d);
        return h;
    }

    // maf2_1 draws U[-1, 1] from seeded streams: the same matrix repeats exactly, while A, B, C,
    // the batches of one call and pointer-array batches all get different data, and none of them
    // shares maf2_0's stream.
    TEST(Maf2_1InitDevice_pre_checkin, streams_repeat_and_differ)
    {
        const size_t M = 128, N = 256, lda = 128, stride = M * N, batches = 2;
        const size_t total = stride * batches;
        auto         run   = [&](ABC_dims abc, hipblaslt_initialization init, size_t batchIndex) {
            return fill(abc, init, M, N, lda, stride, batches, total, batchIndex);
        };
        const auto a  = run(ABC_dims::A, hipblaslt_initialization::maf2_1, 0);
        const auto a2 = run(ABC_dims::A, hipblaslt_initialization::maf2_1, 0);
        const auto b  = run(ABC_dims::B, hipblaslt_initialization::maf2_1, 0);
        const auto c  = run(ABC_dims::C, hipblaslt_initialization::maf2_1, 0);
        const auto p1 = run(ABC_dims::A, hipblaslt_initialization::maf2_1, 1);
        const auto n  = run(ABC_dims::A, hipblaslt_initialization::maf2_0, 0);
        EXPECT_EQ(a, a2) << "the same matrix must repeat";
        EXPECT_NE(a, b);
        EXPECT_NE(a, c);
        EXPECT_NE(b, c);
        EXPECT_NE(a, p1) << "pointer-array batches must differ";
        EXPECT_FALSE(std::equal(a.begin(), a.begin() + stride, a.begin() + stride))
            << "strided batches must differ";

        // U[-1, 1]: mean 0, variance 1/3. maf2_0's N(0, 1) values from the same position differ.
        double sum = 0, sumSq = 0;
        size_t same = 0;
        for(size_t k = 0; k < total; k++)
        {
            ASSERT_TRUE(a[k] >= -1.f && a[k] <= 1.f) << "offset " << k << " value " << a[k];
            sum += a[k];
            sumSq += double(a[k]) * a[k];
            same += a[k] == n[k];
        }
        const double mean = sum / total;
        EXPECT_NEAR(mean, 0.0, 0.01);
        EXPECT_NEAR(sumSq / total - mean * mean, 1.0 / 3.0, 0.01);
        EXPECT_LT(same, total / 1000) << "maf2_1 must not reuse maf2_0's streams";
    }

    // Every batch of an interleaved layout (nonzero stride below lda) is filled: M = 2, N = 3,
    // lda = 4, stride = 2 puts batch 0 at offsets {0, 1, 4, 5, 8, 9} and batch 1 at the rest.
    TEST(Maf2_1InitDevice_pre_checkin, interleaved_batches_are_all_filled)
    {
        const auto h = fill(ABC_dims::A, hipblaslt_initialization::maf2_1, 2, 3, 4, 2, 2, 12);
        for(size_t k = 0; k < h.size(); k++)
            EXPECT_FALSE(std::isnan(h[k])) << "offset " << k;
    }
} // namespace
