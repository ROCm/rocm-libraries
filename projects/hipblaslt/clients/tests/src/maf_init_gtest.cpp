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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{
    // Fills `elements` values of T on the device with NaN, runs the initializer over them and
    // returns the buffer, so cells it did not write stay NaN.
    template <typename T>
    std::vector<T> fill(ABC_dims                 abc,
                        hipblaslt_initialization init,
                        hipDataType              type,
                        size_t                   M,
                        size_t                   N,
                        size_t                   lda,
                        size_t                   stride,
                        size_t                   batch_count,
                        size_t                   elements,
                        size_t                   batch_index = 0)
    {
        T* d = nullptr;
        EXPECT_EQ(hipMalloc(&d, elements * sizeof(T)), hipSuccess);
        EXPECT_EQ(hipMemset(d, 0xFF, elements * sizeof(T)), hipSuccess); // NaN
        hipblaslt_init_device(
            abc, init, false, d, M, N, lda, type, stride, batch_count, -1, batch_index);
        std::vector<T> h(elements);
        EXPECT_EQ(hipMemcpy(h.data(), d, h.size() * sizeof(T), hipMemcpyDeviceToHost), hipSuccess);
        (void)hipFree(d);
        return h;
    }

    std::vector<float> fillFloat(ABC_dims                 abc,
                                 hipblaslt_initialization init,
                                 size_t                   M,
                                 size_t                   N,
                                 size_t                   lda,
                                 size_t                   stride,
                                 size_t                   batch_count,
                                 size_t                   elements)
    {
        return fill<float>(abc, init, HIP_R_32F, M, N, lda, stride, batch_count, elements);
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

    // maf1_0: element (i, j) of batch b holds sin (A) or cos (B, C) of the logical index
    // i + j * M + b * M * N, so ld padding and the batch stride do not shift it; padding cells
    // inside the filled extent are 0. A pointer-array batch passes its index as batch_index.
    TEST(MafInitDevice_pre_checkin, maf1_0_uses_the_logical_index)
    {
        const size_t M = 3, N = 4, lda = 5, stride = 23, batches = 2;
        const size_t total = stride + M + lda * (N - 1);
        for(auto abc : {ABC_dims::A, ABC_dims::B, ABC_dims::C})
        {
            const auto        h = fill<double>(abc,
                                        hipblaslt_initialization::maf1_0,
                                        HIP_R_64F,
                                        M,
                                        N,
                                        lda,
                                        stride,
                                        batches,
                                        total);
            std::vector<bool> logical(total, false);
            for(size_t b = 0; b < batches; b++)
                for(size_t j = 0; j < N; j++)
                    for(size_t i = 0; i < M; i++)
                    {
                        const size_t o   = b * stride + j * lda + i;
                        const double idx = double(i + j * M + b * M * N);
                        logical[o]       = true;
                        EXPECT_EQ(h[o], abc == ABC_dims::A ? std::sin(idx) : std::cos(idx))
                            << "i " << i << " j " << j << " batch " << b;
                    }
            for(size_t o = 0; o < total; o++)
                if(!logical[o])
                    EXPECT_EQ(h[o], 0.0) << "padding offset " << o;
        }

        // Pointer-array batch 1 of the same shape: one matrix, offset by one batch of indices.
        const auto p1 = fill<double>(ABC_dims::A,
                                     hipblaslt_initialization::maf1_0,
                                     HIP_R_64F,
                                     M,
                                     N,
                                     lda,
                                     0,
                                     1,
                                     M + lda * (N - 1),
                                     /*batch_index=*/1);
        for(size_t j = 0; j < N; j++)
            for(size_t i = 0; i < M; i++)
                EXPECT_EQ(p1[j * lda + i], std::sin(double(i + j * M + M * N)));
    }

    // maf1_0 rounds each double once to BF16. sin(11446) = -0.92382814... lies just past the BF16
    // midpoint -0.923828125, which is where rounding to float first would land; float to BF16 then
    // ties to even (-0.921875). The single rounding gives -0.92578125.
    TEST(MafInitDevice_pre_checkin, maf1_0_rounds_once_to_bf16)
    {
        const size_t M = 11447;
        const auto   h = fill<hip_bfloat16>(
            ABC_dims::A, hipblaslt_initialization::maf1_0, HIP_R_16BF, M, 1, M, 0, 1, M);
        EXPECT_EQ(static_cast<float>(h[11446]), -0.92578125f);
    }

    // maf2_0 draws N(0, 1) from seeded streams: the same matrix repeats exactly, while A, B, C,
    // the batches of one call and pointer-array batches all get different data. Non-block-scaled
    // operands get the same fill from maf2_0_unit_scale.
    TEST(MafInitDevice_pre_checkin, maf2_0_streams_repeat_and_differ)
    {
        const size_t M = 128, N = 256, lda = 128, stride = M * N, batches = 2;
        const size_t total = stride * batches;
        auto         run   = [&](ABC_dims abc, hipblaslt_initialization init, size_t batchIndex) {
            return fill<float>(abc, init, HIP_R_32F, M, N, lda, stride, batches, total, batchIndex);
        };
        const auto a  = run(ABC_dims::A, hipblaslt_initialization::maf2_0, 0);
        const auto a2 = run(ABC_dims::A, hipblaslt_initialization::maf2_0, 0);
        const auto b  = run(ABC_dims::B, hipblaslt_initialization::maf2_0, 0);
        const auto c  = run(ABC_dims::C, hipblaslt_initialization::maf2_0, 0);
        const auto p1 = run(ABC_dims::A, hipblaslt_initialization::maf2_0, 1);
        const auto u  = run(ABC_dims::A, hipblaslt_initialization::maf2_0_unit_scale, 0);
        EXPECT_EQ(a, a2) << "the same matrix must repeat";
        EXPECT_EQ(a, u) << "maf2_0_unit_scale must equal maf2_0 for non-block-scaled operands";
        EXPECT_NE(a, b);
        EXPECT_NE(a, c);
        EXPECT_NE(b, c);
        EXPECT_NE(a, p1) << "pointer-array batches must differ";
        EXPECT_FALSE(std::equal(a.begin(), a.begin() + stride, a.begin() + stride))
            << "strided batches must differ";

        double sum = 0, sumSq = 0;
        for(float v : a)
        {
            ASSERT_TRUE(std::isfinite(v));
            sum += v;
            sumSq += double(v) * v;
        }
        const double mean = sum / total;
        EXPECT_NEAR(mean, 0.0, 0.01);
        EXPECT_NEAR(std::sqrt(sumSq / total - mean * mean), 1.0, 0.01);
    }

    // maf1_0's scalar and vector scale factors (--scaleA/B 1/2) are filled with 0.5.
    TEST(MafInitDevice_pre_checkin, constant_scale_fill)
    {
        std::vector<float> f(7, 1.f);
        hipblaslt_init_constant(f.data(), f.size(), HIP_R_32F, 0.5);
        for(float v : f)
            EXPECT_EQ(v, 0.5f);
        std::vector<double> d(3, 1.0);
        hipblaslt_init_constant(d.data(), d.size(), HIP_R_64F, 0.5);
        for(double v : d)
            EXPECT_EQ(v, 0.5);
        std::vector<hipblasLtHalf> h(5, hipblasLtHalf(1.f));
        hipblaslt_init_constant(h.data(), h.size(), HIP_R_16F, 0.5);
        for(auto v : h)
            EXPECT_EQ(static_cast<float>(v), 0.5f);
    }
} // namespace
