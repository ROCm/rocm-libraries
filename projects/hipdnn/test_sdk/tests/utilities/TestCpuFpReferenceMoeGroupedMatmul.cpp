// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using Mode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode;

namespace
{

template <typename Type>
Tensor<Type> createTensor(const std::vector<int64_t>& dims)
{
    return Tensor<Type>(dims, generateStrides(dims));
}

template <typename Type>
void setValues(Tensor<Type>& tensor, const std::vector<float>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = static_cast<Type>(values[i]);
    }
}


template <typename Type>
void expectTensorValues(const Tensor<Type>& tensor, const std::vector<float>& expected)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), expected.size());
    const auto* data = tensor.memory().hostData();
    for(size_t idx = 0; idx < expected.size(); ++idx)
    {
        EXPECT_EQ(data[idx], static_cast<Type>(expected[idx])) << "Mismatch at flat index " << idx;
    }
}

// E=2, K=2, N=2, with W[0] = identity and W[1] = swap, row-major {K*N, N, 1}.
// Both matrices are symmetric, so the flat layout is identical under the
// column-major {K*N, 1, K} layout too -- see ColumnMajorWeightMatchesRowMajor.
template <typename Type>
Tensor<Type> makeIdentitySwapWeight(bool columnMajor = false)
{
    auto weight = columnMajor ? Tensor<Type>({2, 2, 2}, {4, 1, 2})
                              : createTensor<Type>({2, 2, 2});
    setValues(weight, {1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F});
    return weight;
}

template <typename Type>
void runNoneModeGroupsTokensByOffset(bool columnMajorWeight = false)
{
    auto token = createTensor<Type>({1, 4, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto weight = makeIdentitySwapWeight<Type>(columnMajorWeight);
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 2});
    auto output = createTensor<Type>({1, 4, 2});

    CpuFpReferenceMoeGroupedMatmul::forward<Type, Type, Type, float>(
        token, weight, offsets, output, Mode::NONE, 0);

    expectTensorValues(output, {1.0F, 2.0F, 3.0F, 4.0F, 6.0F, 5.0F, 8.0F, 7.0F});
}

} // namespace

/* ============================= Exact-value tests ============================= */

TEST(TestCpuFpReferenceMoeGroupedMatmul, NoneModeGroupsTokensByOffset)
{
    runNoneModeGroupsTokensByOffset<float>();
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, NoneModeGroupsTokensByOffsetHalf)
{
    runNoneModeGroupsTokensByOffset<half>();
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, NoneModeGroupsTokensByOffsetBFloat16)
{
    runNoneModeGroupsTokensByOffset<bfloat16>();
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, NoneModeZeroFillsRowsBeforeFirstOffset)
{
    auto token = createTensor<float>({1, 4, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto weight = makeIdentitySwapWeight<float>();
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {1, 3});
    auto output = createTensor<float>({1, 4, 2});

    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, weight, offsets, output, Mode::NONE, 0);

    expectTensorValues(output, {0.0F, 0.0F, 3.0F, 4.0F, 5.0F, 6.0F, 8.0F, 7.0F});
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, NoneModeEmptyGroupIsSkipped)
{
    auto token = createTensor<float>({1, 4, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto weight = makeIdentitySwapWeight<float>();
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 0});
    auto output = createTensor<float>({1, 4, 2});

    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, weight, offsets, output, Mode::NONE, 0);

    expectTensorValues(output, {2.0F, 1.0F, 4.0F, 3.0F, 6.0F, 5.0F, 8.0F, 7.0F});
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, GatherModeReadsTokenIndex)
{
    auto token = createTensor<float>({1, 3, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    auto weight = makeIdentitySwapWeight<float>();
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 2});
    auto tokenIndex = createTensor<int32_t>({1, 4, 1});
    setValues(tokenIndex, {2, 0, 1, 2});
    auto output = createTensor<float>({1, 4, 2});

    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, weight, offsets, output, Mode::GATHER, 0, &tokenIndex);

    expectTensorValues(output, {5.0F, 6.0F, 1.0F, 2.0F, 4.0F, 3.0F, 6.0F, 5.0F});
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, ScatterModePermutesOutputRows)
{
    auto token = createTensor<float>({1, 4, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto weight = makeIdentitySwapWeight<float>();
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 2});
    auto tokenIndex = createTensor<int32_t>({1, 4, 1});
    setValues(tokenIndex, {0, 1, 1, 0});
    auto tokenKs = createTensor<int32_t>({1, 4, 1});
    setValues(tokenKs, {0, 0, 1, 1});
    auto output = createTensor<float>({1, 4, 2});

    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, weight, offsets, output, Mode::SCATTER, 2, &tokenIndex, &tokenKs);

    expectTensorValues(output, {1.0F, 2.0F, 8.0F, 7.0F, 3.0F, 4.0F, 6.0F, 5.0F});
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, BatchedOffsetsCycleExperts)
{
    auto token = createTensor<float>({1, 4, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto weight = makeIdentitySwapWeight<float>();
    auto offsets = createTensor<int32_t>({4, 1, 1});
    setValues(offsets, {0, 1, 2, 3});
    auto output = createTensor<float>({1, 4, 2});

    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, weight, offsets, output, Mode::NONE, 0);

    expectTensorValues(output, {1.0F, 2.0F, 4.0F, 3.0F, 5.0F, 6.0F, 8.0F, 7.0F});
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, ColumnMajorWeightMatchesRowMajor)
{
    runNoneModeGroupsTokensByOffset<float>(/*columnMajorWeight=*/true);
}

/* ============================= Rejection tests ============================= */

TEST(TestCpuFpReferenceMoeGroupedMatmul, DecreasingOffsetsThrows)
{
    auto token = createTensor<float>({1, 2, 1});
    setValues(token, {1.0F, 2.0F});
    auto weight = createTensor<float>({1, 1, 1});
    setValues(weight, {1.0F});
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {2, 0});
    auto output = createTensor<float>({1, 2, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::NONE, 0)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, FirstTokenOffsetExceedsRowsTotalThrows)
{
    auto token = createTensor<float>({1, 2, 1});
    setValues(token, {1.0F, 2.0F});
    auto weight = createTensor<float>({1, 1, 1});
    setValues(weight, {1.0F});
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 5});
    auto output = createTensor<float>({1, 2, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::NONE, 0)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, GatherTokenIndexEqualToTokenRowsThrows)
{
    auto token = createTensor<float>({1, 2, 1});
    setValues(token, {1.0F, 2.0F});
    auto weight = createTensor<float>({1, 1, 1});
    setValues(weight, {1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto tokenIndex = createTensor<int32_t>({1, 1, 1});
    setValues(tokenIndex, {2});
    auto output = createTensor<float>({1, 1, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::GATHER, 0, &tokenIndex)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, GatherTokenIndexNegativeThrows)
{
    auto token = createTensor<float>({1, 2, 1});
    setValues(token, {1.0F, 2.0F});
    auto weight = createTensor<float>({1, 1, 1});
    setValues(weight, {1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto tokenIndex = createTensor<int32_t>({1, 1, 1});
    setValues(tokenIndex, {-1});
    auto output = createTensor<float>({1, 1, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::GATHER, 0, &tokenIndex)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, ScatterTokenKsEqualToTopKThrows)
{
    auto token = createTensor<float>({1, 1, 1});
    setValues(token, {1.0F});
    auto weight = createTensor<float>({1, 1, 1});
    setValues(weight, {1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto tokenIndex = createTensor<int32_t>({1, 1, 1});
    setValues(tokenIndex, {0});
    auto tokenKs = createTensor<int32_t>({1, 1, 1});
    setValues(tokenKs, {1});
    auto output = createTensor<float>({1, 1, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::SCATTER, 1, &tokenIndex, &tokenKs)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, ScatterDuplicateDestinationsThrows)
{
    auto token = createTensor<float>({1, 2, 1});
    setValues(token, {1.0F, 2.0F});
    auto weight = createTensor<float>({1, 1, 1});
    setValues(weight, {1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto tokenIndex = createTensor<int32_t>({1, 2, 1});
    setValues(tokenIndex, {0, 0});
    auto tokenKs = createTensor<int32_t>({1, 2, 1});
    setValues(tokenKs, {0, 0});
    auto output = createTensor<float>({1, 2, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::SCATTER, 1, &tokenIndex, &tokenKs)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, ScatterTopKExceedsExpertCountThrows)
{
    auto token = createTensor<float>({1, 1, 1});
    setValues(token, {1.0F});
    auto weight = createTensor<float>({1, 1, 1});
    setValues(weight, {1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto tokenIndex = createTensor<int32_t>({1, 1, 1});
    setValues(tokenIndex, {0});
    auto tokenKs = createTensor<int32_t>({1, 1, 1});
    setValues(tokenKs, {0});
    auto output = createTensor<float>({1, 1, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::SCATTER, 2, &tokenIndex, &tokenKs)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, MismatchedHiddenSizeThrows)
{
    auto token = createTensor<float>({1, 1, 2});
    setValues(token, {1.0F, 2.0F});
    auto weight = createTensor<float>({1, 3, 1});
    setValues(weight, {1.0F, 1.0F, 1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto output = createTensor<float>({1, 1, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::NONE, 0)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, MissingTokenIndexForGatherThrows)
{
    auto token = createTensor<float>({1, 1, 1});
    setValues(token, {1.0F});
    auto weight = createTensor<float>({1, 1, 1});
    setValues(weight, {1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto output = createTensor<float>({1, 1, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::GATHER, 0, nullptr)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, MissingTokenKsForScatterThrows)
{
    auto token = createTensor<float>({1, 1, 1});
    setValues(token, {1.0F});
    auto weight = createTensor<float>({1, 1, 1});
    setValues(weight, {1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto tokenIndex = createTensor<int32_t>({1, 1, 1});
    setValues(tokenIndex, {0});
    auto output = createTensor<float>({1, 1, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::SCATTER, 1, &tokenIndex, nullptr)),
                std::runtime_error);
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, RankTwoTokenThrows)
{
    auto token = createTensor<float>({4, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto weight = createTensor<float>({1, 2, 2});
    setValues(weight, {1.0F, 0.0F, 0.0F, 1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto output = createTensor<float>({1, 4, 2});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                     token, weight, offsets, output, Mode::NONE, 0)),
                std::runtime_error);
}

/* ============================= Randomized cross-check ============================= */

TEST(TestCpuFpReferenceMoeGroupedMatmul, LargeRandomMatchesNaiveLoop)
{
    constexpr int64_t experts = 4;
    constexpr int64_t batch = 2;
    constexpr int64_t hiddenK = 17;
    constexpr int64_t outputN = 13;
    constexpr int64_t tokenRows = 64;
    constexpr int32_t topK = 2;
    constexpr int64_t groupCount = batch * experts;
    constexpr int64_t rowsPerGroup = tokenRows / groupCount;

    auto token = createTensor<float>({1, tokenRows, hiddenK});
    auto weight = createTensor<float>({experts, hiddenK, outputN});
    token.fillWithRandomValues(0.0F, 1.0F, 42);
    weight.fillWithRandomValues(0.0F, 1.0F, 43);

    auto offsets = createTensor<int32_t>({groupCount, 1, 1});
    auto tokenIndex = createTensor<int32_t>({1, tokenRows, 1});
    auto tokenKs = createTensor<int32_t>({1, tokenRows, 1});
    for(int64_t g = 0; g < groupCount; ++g)
    {
        offsets.setHostValue(static_cast<int32_t>(g * tokenRows / groupCount), {g, 0, 0});
    }
    for(int64_t r = 0; r < tokenRows; ++r)
    {
        tokenIndex.setHostValue(static_cast<int32_t>(r / topK), {0, r, 0});
        tokenKs.setHostValue(static_cast<int32_t>(r % topK), {0, r, 0});
    }

    auto output = createTensor<float>({1, tokenRows, outputN});
    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, weight, offsets, output, Mode::SCATTER, topK, &tokenIndex, &tokenKs);

    // Independent naive reference: with this deterministic identity-permutation
    // routing, dst == r always, and expert(r) == (r / rowsPerGroup) % experts.
    auto naiveOutput = createTensor<float>({1, tokenRows, outputN});
    for(int64_t r = 0; r < tokenRows; ++r)
    {
        const int64_t expert = (r / rowsPerGroup) % experts;
        for(int64_t n = 0; n < outputN; ++n)
        {
            double acc = 0.0;
            for(int64_t k = 0; k < hiddenK; ++k)
            {
                acc += static_cast<double>(token.getHostValue({0, r, k}))
                       * static_cast<double>(weight.getHostValue({expert, k, n}));
            }
            naiveOutput.setHostValue(static_cast<float>(acc), {0, r, n});
        }
    }

    CpuFpReferenceValidation<float> validator(1e-5F, 1e-5F);
    EXPECT_TRUE(validator.allClose(naiveOutput, output));
}
