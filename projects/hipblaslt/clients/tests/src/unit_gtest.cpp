// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Preserve the comparison regressions from the former unit.hpp API through
// the tensor-based matmul validator. These tests do not require a GPU.

#include <gtest/gtest.h>
#include <hipblaslt/hostnumerics/MatmulValidation.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace
{
    using namespace roc::hostnumerics;
    using namespace hipblaslt::hostnumerics;

    Tensor matrixTensor(const std::vector<float>& values,
                        size_t                    rows,
                        size_t                    columns,
                        ptrdiff_t                 leadingDimension,
                        ptrdiff_t                 batchStride,
                        size_t                    batches)
    {
        return Tensor::copyEncodedBackingStorage(
            ScalarType::Float32,
            Layout(Shape{rows, columns, batches}, {1, leadingDimension, batchStride}),
            std::as_bytes(std::span(values)));
    }

    bool unitCheck(const Tensor& expected, const Tensor& observed)
    {
        MatmulValidationCase testCase;
        testCase.outputs.emplace_back(expected, observed);
        return validateMatmulOutputs({.compareAllClose = true}, std::span(&testCase, 1)).passed;
    }

    TEST(HostNumericsUnitCheck, ComparesDenseStridedBatches)
    {
        constexpr size_t   rows = 2, columns = 3, stride = rows * columns, batches = 2;
        std::vector<float> expected(stride * batches);
        for(size_t i = 0; i < expected.size(); ++i)
            expected[i] = static_cast<float>(i);
        auto       observed  = expected;
        const auto reference = matrixTensor(expected, rows, columns, rows, stride, batches);
        EXPECT_TRUE(
            unitCheck(reference, matrixTensor(observed, rows, columns, rows, stride, batches)));
        observed.back() += 1.0f;
        EXPECT_FALSE(
            unitCheck(reference, matrixTensor(observed, rows, columns, rows, stride, batches)));
    }

    TEST(HostNumericsUnitCheck, ComparesZeroStrideForMultipleBatches)
    {
        const std::vector<float> expected{1, 2, 3, 4};
        auto                     observed  = expected;
        const auto               reference = matrixTensor(expected, 2, 2, 2, 0, 3);
        EXPECT_TRUE(unitCheck(reference, matrixTensor(observed, 2, 2, 2, 0, 3)));
        observed.back() += 1.0f;
        EXPECT_FALSE(unitCheck(reference, matrixTensor(observed, 2, 2, 2, 0, 3)));
    }

    TEST(HostNumericsUnitCheck, EqualBytesDoNotBypassElementTypes)
    {
        const float value = 1.0f;
        const auto  bytes = std::as_bytes(std::span(&value, 1));
        const auto  expected
            = Tensor::copyEncodedBackingStorage(ScalarType::Float32, Layout(Shape{}, {}), bytes);
        const auto observed
            = Tensor::copyEncodedBackingStorage(ScalarType::BFloat16, Layout(Shape{}, {}), bytes);
        EXPECT_FALSE(unitCheck(expected, observed));
    }

    TEST(HostNumericsUnitCheck, IgnoresColumnPaddingAndBatchGaps)
    {
        constexpr size_t         rows = 2, columns = 2, lda = 3, stride = 8, batches = 2;
        const std::vector<float> expected(stride * batches, 1.0f);
        auto                     observed = expected;
        for(size_t batch = 0; batch < batches; ++batch)
        {
            observed[batch * stride + 2] = 2.0f;
            observed[batch * stride + 5] = 3.0f;
            observed[batch * stride + 6] = 4.0f;
            observed[batch * stride + 7] = 5.0f;
        }
        const auto reference = matrixTensor(expected, rows, columns, lda, stride, batches);
        EXPECT_TRUE(
            unitCheck(reference, matrixTensor(observed, rows, columns, lda, stride, batches)));
        observed[stride + lda] = 6.0f;
        EXPECT_FALSE(
            unitCheck(reference, matrixTensor(observed, rows, columns, lda, stride, batches)));
    }

    TEST(HostNumericsUnitCheck, AcceptsNumericallyEqualEncodings)
    {
        const std::vector<float> expected{0.0f, std::bit_cast<float>(uint32_t{0x7fc00001})};
        const std::vector<float> observed{-0.0f, std::bit_cast<float>(uint32_t{0x7fc00002})};
        ASSERT_NE(std::memcmp(expected.data(), observed.data(), expected.size() * sizeof(float)),
                  0);
        EXPECT_TRUE(unitCheck(matrixTensor(expected, 1, 2, 1, 0, 1),
                              matrixTensor(observed, 1, 2, 1, 0, 1)));
    }

    TEST(HostNumericsUnitCheck, ChecksEveryPointerArrayBatch)
    {
        // The matmul client represents independently allocated batches as tensor pairs.
        MatmulValidationCase     testCase;
        const std::vector<float> first{1, 2}, second{3, 4};
        testCase.outputs.emplace_back(matrixTensor(first, 2, 1, 2, 0, 1),
                                      matrixTensor(first, 2, 1, 2, 0, 1));
        testCase.outputs.emplace_back(matrixTensor(second, 2, 1, 2, 0, 1),
                                      matrixTensor(second, 2, 1, 2, 0, 1));
        EXPECT_TRUE(
            validateMatmulOutputs({.compareAllClose = true}, std::span(&testCase, 1)).passed);
        testCase.outputs.back().second.storeFrom({1, 0, 0}, 5.0f);
        EXPECT_FALSE(
            validateMatmulOutputs({.compareAllClose = true}, std::span(&testCase, 1)).passed);
    }

    TEST(HostNumericsUnitCheck, RejectsScalarMismatch)
    {
        EXPECT_FALSE(unitCheck(Tensor(1.0f), Tensor(2.0f)));
    }

    TEST(HostNumericsUnitCheck, RejectsSinglePointerArrayMismatch)
    {
        MatmulValidationCase testCase;
        testCase.outputs.emplace_back(Tensor(1.0f), Tensor(2.0f));
        EXPECT_FALSE(
            validateMatmulOutputs({.compareAllClose = true}, std::span(&testCase, 1)).passed);
    }

    TEST(HostNumericsUnitCheck, RejectsSpecialValueMismatchWithoutUnitCheck)
    {
        MatmulValidationCase testCase;
        testCase.outputs.emplace_back(Tensor(1.0f), Tensor(std::numeric_limits<float>::infinity()));
        EXPECT_FALSE(validateMatmulOutputs({.compareNorm = true}, std::span(&testCase, 1)).passed);
    }
} // namespace
