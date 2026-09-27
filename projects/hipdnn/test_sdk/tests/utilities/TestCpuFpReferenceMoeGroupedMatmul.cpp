// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

#include <memory>
#include <string>
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

/// Asserts `call` is rejected with a message containing `needle`. The routing
/// bound checks these tests cover also have a read-past-the-end failure mode that
/// happens to throw on garbage data, so asserting only the exception type would
/// not distinguish a diagnosed rejection from an out-of-bounds read.
template <typename Callable>
void expectRejectedWith(Callable&& call, const std::string& needle)
{
    try
    {
        call();
        ADD_FAILURE() << "expected std::runtime_error containing \"" << needle << "\"";
    }
    catch(const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find(needle), std::string::npos) << error.what();
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
// Both matrices are symmetric, so this helper says nothing about weight layout -
// ColumnMajorWeightMatchesRowMajor uses an asymmetric weight for that.
template <typename Type>
Tensor<Type> makeIdentitySwapWeight()
{
    auto weight = createTensor<Type>({2, 2, 2});
    setValues(weight, {1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F});
    return weight;
}

template <typename Type>
void runNoneModeGroupsTokensByOffset()
{
    auto token = createTensor<Type>({1, 4, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto weight = makeIdentitySwapWeight<Type>();
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
    // Row 0 is never routed to, so the zeros asserted below must come from the
    // kernel's zero-fill, not from whatever the allocator happened to hand back.
    output.fillWithSentinelValue();

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

// W = [[1, 2], [3, 4]] for one expert: asymmetric, so the two layouts have
// genuinely different flat contents ({1,2,3,4} row-major vs {1,3,2,4}
// column-major) and the expected values below only hold if the kernel honours
// the stride split. Forcing the row-major branch for both makes the
// column-major case yield {31, 23} instead of {31, 42}.
TEST(TestCpuFpReferenceMoeGroupedMatmul, ColumnMajorWeightMatchesRowMajor)
{
    auto token = createTensor<float>({1, 1, 2});
    setValues(token, {1.0F, 10.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});

    auto rowMajorWeight = createTensor<float>({1, 2, 2});
    setValues(rowMajorWeight, {1.0F, 2.0F, 3.0F, 4.0F});
    auto rowMajorOutput = createTensor<float>({1, 1, 2});
    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, rowMajorWeight, offsets, rowMajorOutput, Mode::NONE, 0);

    auto columnMajorWeight = Tensor<float>({1, 2, 2}, {4, 1, 2});
    setValues(columnMajorWeight, {1.0F, 3.0F, 2.0F, 4.0F});
    auto columnMajorOutput = createTensor<float>({1, 1, 2});
    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, columnMajorWeight, offsets, columnMajorOutput, Mode::NONE, 0);

    expectTensorValues(rowMajorOutput, {31.0F, 42.0F});
    expectTensorValues(columnMajorOutput, {31.0F, 42.0F});
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

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                token, weight, offsets, output, Mode::NONE, 0);
        },
        "FirstTokenOffset[1]");
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

TEST(TestCpuFpReferenceMoeGroupedMatmul, InvalidCoreShapesThrow)
{
    struct ShapeCase
    {
        const char* name;
        std::vector<int64_t> tokenDims;
        std::vector<int64_t> firstTokenOffsetDims;
        std::vector<int64_t> outputDims;
        const char* reason;
    };

    const std::vector<ShapeCase> cases = {
        {"TokenLeadingDimension", {2, 2, 1}, {2, 1, 1}, {1, 2, 1}, "token must have"},
        {"OutputLeadingDimension", {1, 2, 1}, {2, 1, 1}, {2, 2, 1}, "output must have"},
        {"FirstTokenOffsetShape", {1, 2, 1}, {2, 2, 1}, {1, 2, 1}, "first_token_offset"},
        {"OutputWidth", {1, 2, 1}, {2, 1, 1}, {1, 2, 2}, "output width"},
        {"OffsetGroupCount", {1, 2, 1}, {3, 1, 1}, {1, 2, 1}, "multiple of the expert count"},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        auto token = createTensor<float>(testCase.tokenDims);
        auto weight = createTensor<float>({2, 1, 1});
        auto offsets = createTensor<int32_t>(testCase.firstTokenOffsetDims);
        auto output = createTensor<float>(testCase.outputDims);

        expectRejectedWith(
            [&] {
                CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                    token, weight, offsets, output, Mode::NONE, 0);
            },
            testCase.reason);
    }
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, RaggedWeightThrows)
{
    auto token = createTensor<float>({1, 1, 1});
    auto raggedOffset = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{3, 1, 1, 1},
                                                          std::vector<int64_t>{1, 1, 1, 1});
    setValues(*raggedOffset, {0.0F, 1.0F, 1.0F});
    RaggedTensor<float> weight({2, 1, 1}, {1, 1, 1}, 1, raggedOffset);
    auto offsets = createTensor<int32_t>({2, 1, 1});
    auto output = createTensor<float>({1, 1, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                token, weight, offsets, output, Mode::NONE, 0);
        },
        "ragged weight tensor");
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, InvalidRoutingShapesThrow)
{
    struct RoutingShapeCase
    {
        const char* name;
        Mode mode;
        std::vector<int64_t> tokenIndexDims;
        std::vector<int64_t> tokenKsDims;
    };

    const std::vector<RoutingShapeCase> cases = {
        {"GatherTokenIndexRank", Mode::GATHER, {2, 2}, {1, 2, 1}},
        {"GatherTokenIndexLeadingDimension", Mode::GATHER, {2, 2, 1}, {1, 2, 1}},
        {"GatherTokenIndexTrailingDimension", Mode::GATHER, {1, 2, 2}, {1, 2, 1}},
        {"ScatterTokenKsRank", Mode::SCATTER, {1, 2, 1}, {2, 2}},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        auto token = createTensor<float>({1, 2, 1});
        auto weight = createTensor<float>({1, 1, 1});
        auto offsets = createTensor<int32_t>({1, 1, 1});
        auto tokenIndex = createTensor<int32_t>(testCase.tokenIndexDims);
        auto tokenKs = createTensor<int32_t>(testCase.tokenKsDims);
        auto output = createTensor<float>({1, 2, 1});

        expectRejectedWith(
            [&] {
                CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                    token,
                    weight,
                    offsets,
                    output,
                    testCase.mode,
                    testCase.mode == Mode::SCATTER ? 1 : 0,
                    &tokenIndex,
                    testCase.mode == Mode::SCATTER ? &tokenKs : nullptr);
            },
            "must have shape [1, rows, 1]");
    }
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, ScatterRoutingLengthsMustMatch)
{
    auto token = createTensor<float>({1, 2, 1});
    auto weight = createTensor<float>({1, 1, 1});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    auto tokenIndex = createTensor<int32_t>({1, 2, 1});
    auto tokenKs = createTensor<int32_t>({1, 1, 1});
    auto output = createTensor<float>({1, 2, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                token, weight, offsets, output, Mode::SCATTER, 1, &tokenIndex, &tokenKs);
        },
        "must have equal length");
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, GatherOutputRowsMustMatchRoutingRows)
{
    auto token = createTensor<float>({1, 2, 1});
    auto weight = createTensor<float>({1, 1, 1});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    auto tokenIndex = createTensor<int32_t>({1, 2, 1});
    auto output = createTensor<float>({1, 1, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                token, weight, offsets, output, Mode::GATHER, 0, &tokenIndex);
        },
        "output row count to equal the routed row count");
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, NonGatherOutputRowsMustMatchTokenRows)
{
    for(const Mode mode : {Mode::NONE, Mode::SCATTER})
    {
        SCOPED_TRACE(mode == Mode::NONE ? "NONE" : "SCATTER");
        auto token = createTensor<float>({1, 2, 1});
        auto weight = createTensor<float>({1, 1, 1});
        auto offsets = createTensor<int32_t>({1, 1, 1});
        auto tokenIndex = createTensor<int32_t>({1, 2, 1});
        auto tokenKs = createTensor<int32_t>({1, 2, 1});
        auto output = createTensor<float>({1, 1, 1});

        expectRejectedWith(
            [&] {
                CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                    token,
                    weight,
                    offsets,
                    output,
                    mode,
                    mode == Mode::SCATTER ? 1 : 0,
                    mode == Mode::SCATTER ? &tokenIndex : nullptr,
                    mode == Mode::SCATTER ? &tokenKs : nullptr);
            },
            "output row count must equal the token row count");
    }
}

// SCATTER indexes both routing tensors by token row, so a shorter routing tensor
// is an out-of-bounds read, not a smaller workload.
TEST(TestCpuFpReferenceMoeGroupedMatmul, ScatterRoutingShorterThanTokenRowsThrows)
{
    auto token = createTensor<float>({1, 4, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto weight = createTensor<float>({1, 2, 2});
    setValues(weight, {1.0F, 0.0F, 0.0F, 1.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto tokenIndex = createTensor<int32_t>({1, 2, 1});
    setValues(tokenIndex, {0, 1});
    auto tokenKs = createTensor<int32_t>({1, 2, 1});
    setValues(tokenKs, {0, 0});
    auto output = createTensor<float>({1, 4, 2});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                token, weight, offsets, output, Mode::SCATTER, 1, &tokenIndex, &tokenKs);
        },
        "routed row count");
}

// A group's end offset is consumed by the inner loop before the next iteration
// would range-check it, so it must be validated up front.
TEST(TestCpuFpReferenceMoeGroupedMatmul, GatherOffsetPastRoutedRowsThrows)
{
    auto token = createTensor<float>({1, 4, 2});
    setValues(token, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto weight = makeIdentitySwapWeight<float>();
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 64});
    auto tokenIndex = createTensor<int32_t>({1, 4, 1});
    setValues(tokenIndex, {0, 1, 2, 3});
    auto output = createTensor<float>({1, 4, 2});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
                token, weight, offsets, output, Mode::GATHER, 0, &tokenIndex);
        },
        "FirstTokenOffset[1]");
}

/* ============================= Randomized cross-check ============================= */

TEST(TestCpuFpReferenceMoeGroupedMatmul, LargeRandomScatterPermutationMatchesNaiveLoop)
{
    constexpr int64_t EXPERTS = 4;
    constexpr int64_t BATCH = 2;
    constexpr int64_t HIDDEN_K = 17;
    constexpr int64_t OUTPUT_N = 13;
    constexpr int64_t TOKEN_ROWS = 64;
    constexpr int32_t TOP_K = 2;
    constexpr int64_t GROUP_COUNT = BATCH * EXPERTS;
    constexpr int64_t ROWS_PER_GROUP = TOKEN_ROWS / GROUP_COUNT;
    constexpr int64_t PERMUTATION_MULTIPLIER = 13; // Coprime with TOKEN_ROWS.
    constexpr int64_t PERMUTATION_OFFSET = 7;

    auto token = createTensor<float>({1, TOKEN_ROWS, HIDDEN_K});
    auto weight = createTensor<float>({EXPERTS, HIDDEN_K, OUTPUT_N});
    token.fillWithRandomValues(0.0F, 1.0F, 42);
    weight.fillWithRandomValues(0.0F, 1.0F, 43);

    auto offsets = createTensor<int32_t>({GROUP_COUNT, 1, 1});
    auto tokenIndex = createTensor<int32_t>({1, TOKEN_ROWS, 1});
    auto tokenKs = createTensor<int32_t>({1, TOKEN_ROWS, 1});
    for(int64_t g = 0; g < GROUP_COUNT; ++g)
    {
        offsets.setHostValue(static_cast<int32_t>(g * TOKEN_ROWS / GROUP_COUNT), {g, 0, 0});
    }
    for(int64_t r = 0; r < TOKEN_ROWS; ++r)
    {
        const int64_t destination = (r * PERMUTATION_MULTIPLIER + PERMUTATION_OFFSET) % TOKEN_ROWS;
        tokenIndex.setHostValue(static_cast<int32_t>(destination / TOP_K), {0, r, 0});
        tokenKs.setHostValue(static_cast<int32_t>(destination % TOP_K), {0, r, 0});
    }

    auto output = createTensor<float>({1, TOKEN_ROWS, OUTPUT_N});
    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, weight, offsets, output, Mode::SCATTER, TOP_K, &tokenIndex, &tokenKs);

    auto naiveOutput = createTensor<float>({1, TOKEN_ROWS, OUTPUT_N});
    for(int64_t r = 0; r < TOKEN_ROWS; ++r)
    {
        const int64_t destination = (r * PERMUTATION_MULTIPLIER + PERMUTATION_OFFSET) % TOKEN_ROWS;
        const int64_t expert = (r / ROWS_PER_GROUP) % EXPERTS;
        for(int64_t n = 0; n < OUTPUT_N; ++n)
        {
            double acc = 0.0;
            for(int64_t k = 0; k < HIDDEN_K; ++k)
            {
                acc += static_cast<double>(token.getHostValue({0, r, k}))
                       * static_cast<double>(weight.getHostValue({expert, k, n}));
            }
            naiveOutput.setHostValue(static_cast<float>(acc), {0, destination, n});
        }
    }

    const CpuFpReferenceValidation<float> validator(1e-5F, 1e-5F);
    EXPECT_TRUE(validator.allClose(naiveOutput, output));
}

TEST(TestCpuFpReferenceMoeGroupedMatmul, LargeRandomGatherWithLeadingSourceGapMatchesNaiveLoop)
{
    constexpr int64_t EXPERTS = 4;
    constexpr int64_t BATCH = 2;
    constexpr int64_t HIDDEN_K = 17;
    constexpr int64_t OUTPUT_N = 13;
    constexpr int64_t TOKEN_ROWS = 64;
    constexpr int64_t ROUTED_ROWS = 48;
    constexpr int64_t LEADING_SOURCE_GAP = TOKEN_ROWS - ROUTED_ROWS;
    constexpr int64_t GROUP_COUNT = BATCH * EXPERTS;
    constexpr int64_t ROWS_PER_GROUP = ROUTED_ROWS / GROUP_COUNT;

    auto token = createTensor<float>({1, TOKEN_ROWS, HIDDEN_K});
    auto weight = createTensor<float>({EXPERTS, HIDDEN_K, OUTPUT_N});
    token.fillWithRandomValues(0.0F, 1.0F, 44);
    weight.fillWithRandomValues(0.0F, 1.0F, 45);

    auto offsets = createTensor<int32_t>({GROUP_COUNT, 1, 1});
    auto tokenIndex = createTensor<int32_t>({1, ROUTED_ROWS, 1});
    for(int64_t g = 0; g < GROUP_COUNT; ++g)
    {
        offsets.setHostValue(static_cast<int32_t>(g * ROUTED_ROWS / GROUP_COUNT), {g, 0, 0});
    }
    for(int64_t r = 0; r < ROUTED_ROWS; ++r)
    {
        tokenIndex.setHostValue(static_cast<int32_t>(TOKEN_ROWS - 1 - r), {0, r, 0});
    }

    auto output = createTensor<float>({1, ROUTED_ROWS, OUTPUT_N});
    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        token, weight, offsets, output, Mode::GATHER, 0, &tokenIndex);

    auto naiveOutput = createTensor<float>({1, ROUTED_ROWS, OUTPUT_N});
    for(int64_t r = 0; r < ROUTED_ROWS; ++r)
    {
        const int64_t source = TOKEN_ROWS - 1 - r;
        ASSERT_GE(source, LEADING_SOURCE_GAP);
        const int64_t expert = (r / ROWS_PER_GROUP) % EXPERTS;
        for(int64_t n = 0; n < OUTPUT_N; ++n)
        {
            double acc = 0.0;
            for(int64_t k = 0; k < HIDDEN_K; ++k)
            {
                acc += static_cast<double>(token.getHostValue({0, source, k}))
                       * static_cast<double>(weight.getHostValue({expert, k, n}));
            }
            naiveOutput.setHostValue(static_cast<float>(acc), {0, r, n});
        }
    }

    const CpuFpReferenceValidation<float> validator(1e-5F, 1e-5F);
    EXPECT_TRUE(validator.allClose(naiveOutput, output));
}
