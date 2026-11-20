// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Direct checks of the maf1_0 / maf2_0 host fills (hipblaslt::hostnumerics::initializeMatrix):
// the GEMM tests compare against a CPU reference built from the same buffers, so they cannot see
// a wrong value, index or stream. Each test fills a small host tensor with a hipBLASLt matrix
// layout and checks the values. No hipBLASLt kernels run.

#include <gtest/gtest.h>

#include <hipblaslt/hostnumerics/HipblasltDataInitialization.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace
{
    namespace hn     = roc::hostnumerics;
    namespace client = hipblaslt::hostnumerics;

    // A column-major M x N matrix with leading dimension lda and batch_count batches stride
    // elements apart, as testing_matmul lays them out. The storage starts as NaN, so cells the
    // initializer did not write stay NaN.
    hn::Tensor matrix(
        hn::ScalarType type, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
    {
        hn::Tensor tensor(
            type,
            hn::Layout(hn::Shape{M, N, batch_count}, {1, ptrdiff_t(lda), ptrdiff_t(stride)}, 0));
        std::ranges::fill(tensor.rawEncodedBackingStorage(), std::byte{0xFF});
        return tensor;
    }

    template <typename T>
    T at(hn::Tensor const& tensor, size_t offset)
    {
        T value;
        std::memcpy(
            &value, tensor.rawEncodedBackingStorage().data() + offset * sizeof(T), sizeof(T));
        return value;
    }

    size_t storedElements(hn::Tensor const& tensor)
    {
        return tensor.rawEncodedBackingStorage().size()
               / (hn::scalarTypeInfo(tensor.type()).storageBits / 8);
    }

    void fill(hn::Tensor&              tensor,
              client::MatrixRole       role,
              hipblaslt_initialization init,
              size_t                   batchIndex = 0)
    {
        client::initializeMatrix(tensor, role, init, 0, {.batchIndex = batchIndex});
    }

    // A nonzero batch stride below lda interleaves the batches: M = 2, N = 3, lda = 4, stride = 2
    // puts batch 0 at offsets {0, 1, 4, 5, 8, 9} and batch 1 at {2, 3, 6, 7, 10, 11}. Every batch
    // must be filled, with its own logical index for maf1_0.
    TEST(MafInitHost_pre_checkin, interleaved_batches_are_all_filled)
    {
        const size_t M = 2, N = 3, lda = 4, stride = 2, batches = 2, total = 12;
        auto         trig = matrix(hn::ScalarType::Float32, M, N, lda, stride, batches);
        ASSERT_EQ(storedElements(trig), total);
        fill(trig, client::MatrixRole::A, hipblaslt_initialization::maf1_0);
        for(size_t b = 0; b < batches; b++)
            for(size_t j = 0; j < N; j++)
                for(size_t i = 0; i < M; i++)
                    EXPECT_EQ(at<float>(trig, b * stride + j * lda + i),
                              static_cast<float>(std::sin(double(i + j * M + b * M * N))))
                        << "i " << i << " j " << j << " batch " << b;

        for(auto init :
            {hipblaslt_initialization::maf2_0, hipblaslt_initialization::maf2_0_unit_scale})
        {
            auto normal = matrix(hn::ScalarType::Float32, M, N, lda, stride, batches);
            fill(normal, client::MatrixRole::A, init);
            for(size_t k = 0; k < total; k++)
                EXPECT_FALSE(std::isnan(at<float>(normal, k)))
                    << hipblaslt_initialization2string(init) << " offset " << k;
        }
    }

    // Batches whose stride is below the matrix extent share storage: M = lda = N = 8, stride = 4,
    // batch_count = 2 spans M + lda * (N - 1) + stride = 68 elements, more than lda * N. The host
    // tensor's storage covers exactly that extent, and every fill writes all of it.
    TEST(MafInitHost_pre_checkin, overlapping_batches_fill_their_whole_extent)
    {
        const size_t M = 8, N = 8, lda = 8, stride = 4, batches = 2;
        const size_t total = M + lda * (N - 1) + (batches - 1) * stride;
        for(auto init : {hipblaslt_initialization::maf1_0,
                         hipblaslt_initialization::maf2_0,
                         hipblaslt_initialization::maf2_0_unit_scale})
        {
            auto h = matrix(hn::ScalarType::Float32, M, N, lda, stride, batches);
            ASSERT_EQ(storedElements(h), total);
            fill(h, client::MatrixRole::A, init);
            for(size_t k = 0; k < total; k++)
                EXPECT_FALSE(std::isnan(at<float>(h, k)))
                    << hipblaslt_initialization2string(init) << " offset " << k;
        }
    }

    // maf1_0: element (i, j) of batch b holds sin (A) or cos (B, C) of the logical index
    // i + j * M + b * M * N, so ld padding and the batch stride do not shift it. Padding cells
    // are not part of the matrix and keep their contents. A pointer-array batch passes its index
    // as batchIndex.
    TEST(MafInitHost_pre_checkin, maf1_0_uses_the_logical_index)
    {
        const size_t M = 3, N = 4, lda = 5, stride = 23, batches = 2;
        for(auto role : {client::MatrixRole::A, client::MatrixRole::B, client::MatrixRole::C})
        {
            auto h = matrix(hn::ScalarType::Float64, M, N, lda, stride, batches);
            fill(h, role, hipblaslt_initialization::maf1_0);
            const size_t      total = storedElements(h);
            std::vector<bool> logical(total, false);
            for(size_t b = 0; b < batches; b++)
                for(size_t j = 0; j < N; j++)
                    for(size_t i = 0; i < M; i++)
                    {
                        const size_t o   = b * stride + j * lda + i;
                        const double idx = double(i + j * M + b * M * N);
                        logical[o]       = true;
                        EXPECT_EQ(at<double>(h, o),
                                  role == client::MatrixRole::A ? std::sin(idx) : std::cos(idx))
                            << "i " << i << " j " << j << " batch " << b;
                    }
            for(size_t o = 0; o < total; o++)
                if(!logical[o])
                    EXPECT_TRUE(std::isnan(at<double>(h, o))) << "padding offset " << o;
        }

        // Pointer-array batch 1 of the same shape: one matrix, offset by one batch of indices.
        auto p1 = matrix(hn::ScalarType::Float64, M, N, lda, 0, 1);
        fill(p1, client::MatrixRole::A, hipblaslt_initialization::maf1_0, /*batchIndex=*/1);
        for(size_t j = 0; j < N; j++)
            for(size_t i = 0; i < M; i++)
                EXPECT_EQ(at<double>(p1, j * lda + i), std::sin(double(i + j * M + M * N)));
    }

    // maf1_0 rounds each double once to BF16. sin(11446) = -0.92382814... lies just past the BF16
    // midpoint -0.923828125, which is where rounding to float first would land; float to BF16 then
    // ties to even (-0.921875). The single rounding gives -0.92578125.
    TEST(MafInitHost_pre_checkin, maf1_0_rounds_once_to_bf16)
    {
        const size_t M = 11447;
        auto         h = matrix(hn::ScalarType::BFloat16, M, 1, M, 0, 1);
        fill(h, client::MatrixRole::A, hipblaslt_initialization::maf1_0);
        EXPECT_EQ(h.loadAs<float>({11446, 0, 0}), -0.92578125f);
    }

    // maf2_0 rounds each double draw once to BF16. The FP64 fill of the same shape draws the same
    // values, so every BF16 cell must equal its FP64 cell rounded to nearest even, computed here
    // without going through float. About one draw in 2^17 lands where rounding to float first
    // changes the result; 4M draws hit a few dozen, and must hit one or the check proves nothing.
    TEST(MafInitHost_pre_checkin, maf2_0_rounds_once_to_bf16)
    {
        const size_t M = 2048, N = 2048;
        auto         d = matrix(hn::ScalarType::Float64, M, N, M, 0, 1);
        auto         h = matrix(hn::ScalarType::BFloat16, M, N, M, 0, 1);
        fill(d, client::MatrixRole::A, hipblaslt_initialization::maf2_0);
        fill(h, client::MatrixRole::A, hipblaslt_initialization::maf2_0);
        // Nearest even with 8 significant bits; dividing by a power of two is exact.
        auto bf16 = [](double v) {
            if(v == 0)
                return v;
            const double ulp = std::ldexp(1.0, std::ilogb(v) - 7);
            return std::nearbyint(v / ulp) * ulp;
        };
        size_t wrong = 0, viaFloatDiffers = 0;
        for(size_t j = 0; j < N; j++)
            for(size_t i = 0; i < M; i++)
            {
                const double source = d.loadAs<double>({i, j, 0});
                const double once   = bf16(source);
                if(h.loadAs<double>({i, j, 0}) != once)
                    wrong++;
                if(bf16(static_cast<float>(source)) != once)
                    viaFloatDiffers++;
            }
        EXPECT_EQ(wrong, 0u);
        EXPECT_GT(viaFloatDiffers, 0u);
    }

    // maf2_0 draws N(0, 1) from seeded streams: the same matrix repeats exactly, while A, B, C,
    // the batches of one call and pointer-array batches all get different data. Non-block-scaled
    // operands get the same fill from maf2_0_unit_scale.
    TEST(MafInitHost_pre_checkin, maf2_0_streams_repeat_and_differ)
    {
        const size_t M = 128, N = 256, lda = 128, stride = M * N, batches = 2;
        auto run = [&](client::MatrixRole role, hipblaslt_initialization init, size_t batchIndex) {
            auto tensor = matrix(hn::ScalarType::Float32, M, N, lda, stride, batches);
            fill(tensor, role, init, batchIndex);
            std::vector<float> values(storedElements(tensor));
            for(size_t k = 0; k < values.size(); k++)
                values[k] = at<float>(tensor, k);
            return values;
        };
        const auto a  = run(client::MatrixRole::A, hipblaslt_initialization::maf2_0, 0);
        const auto a2 = run(client::MatrixRole::A, hipblaslt_initialization::maf2_0, 0);
        const auto b  = run(client::MatrixRole::B, hipblaslt_initialization::maf2_0, 0);
        const auto c  = run(client::MatrixRole::C, hipblaslt_initialization::maf2_0, 0);
        const auto p1 = run(client::MatrixRole::A, hipblaslt_initialization::maf2_0, 1);
        const auto u  = run(client::MatrixRole::A, hipblaslt_initialization::maf2_0_unit_scale, 0);
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
        const double mean = sum / a.size();
        EXPECT_NEAR(mean, 0.0, 0.01);
        EXPECT_NEAR(std::sqrt(sumSq / a.size() - mean * mean), 1.0, 0.01);
    }

    // maf1_0's scalar and vector scale factors (--scaleA/B 1/2) are 0.5. testing_matmul fills
    // them with a constant 0.5 recipe; every scale-factor type holds it exactly.
    TEST(MafInitHost_pre_checkin, constant_scale_fill)
    {
        for(auto type : {hn::ScalarType::Float32, hn::ScalarType::Float64, hn::ScalarType::Float16})
        {
            hn::Tensor scales(type, hn::Shape{7});
            hn::generate(
                scales,
                hn::GenerationRecipe::realOnly(
                    hn::GenerationRecipe::constant({.value = 0.5}), 0, hn::LastDimensionFastest{}));
            for(size_t i = 0; i < 7; i++)
                EXPECT_EQ(scales.loadAs<double>({i}), 0.5) << hn::scalarTypeName(type);
        }
    }
} // namespace
