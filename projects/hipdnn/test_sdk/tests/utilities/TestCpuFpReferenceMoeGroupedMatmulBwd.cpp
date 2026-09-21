// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmulBwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

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

/// Asserts `call` is rejected with a message containing `needle`. Several of
/// these rejections have a read-past-the-end failure mode that happens to
/// throw on garbage data too, so asserting only the exception type would not
/// distinguish a diagnosed rejection from an out-of-bounds read.
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

// E=2, K=1, N=1, offsets {0, 2} over 4 token rows: expert 0 owns rows 0-1,
// expert 1 owns rows 2-3.
template <typename Type>
void runNoneModeGroupsTokensByOffset()
{
    auto doutput = createTensor<Type>({1, 4, 1});
    setValues(doutput, {1.0F, 2.0F, 3.0F, 4.0F});
    auto token = createTensor<Type>({1, 4, 1});
    setValues(token, {10.0F, 20.0F, 30.0F, 40.0F});
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 2});
    auto dweight = createTensor<Type>({2, 1, 1});

    CpuFpReferenceMoeGroupedMatmulBwd::backward<Type, Type, Type, float>(
        doutput, token, offsets, dweight);

    // expert 0: 1*10 + 2*20 = 50; expert 1: 3*30 + 4*40 = 250
    expectTensorValues(dweight, {50.0F, 250.0F});
}

} // namespace

/* ============================= Exact-value tests ============================= */

using NoneModeGroupsTokensByOffsetTypes = ::testing::Types<float, half, bfloat16>;

template <typename Type>
class CpuFpReferenceMoeGroupedMatmulBwdTyped : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceMoeGroupedMatmulBwdTyped, NoneModeGroupsTokensByOffsetTypes, );

TYPED_TEST(CpuFpReferenceMoeGroupedMatmulBwdTyped, NoneModeGroupsTokensByOffset)
{
    runNoneModeGroupsTokensByOffset<TypeParam>();
}

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, EmptyGroupYieldsZeroRow)
{
    auto doutput = createTensor<float>({1, 4, 1});
    setValues(doutput, {1.0F, 2.0F, 3.0F, 4.0F});
    auto token = createTensor<float>({1, 4, 1});
    setValues(token, {10.0F, 20.0F, 30.0F, 40.0F});
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 0});
    auto dweight = createTensor<float>({2, 1, 1});
    dweight.fillWithSentinelValue();

    CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
        doutput, token, offsets, dweight);

    // expert 0 owns no rows -> must be zero-filled, not the sentinel value.
    // expert 1 owns rows 0-3: 1*10 + 2*20 + 3*30 + 4*40 = 300
    expectTensorValues(dweight, {0.0F, 300.0F});
}

// The mirror of EmptyGroupYieldsZeroRow, with the empty range at the top of the
// token array instead of the bottom: start == end == tokenRows, so a row walk
// that mishandled the bound would run off the end of token/doutput rather than
// before their beginning.
TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, FinalExpertEmptyAtTokenRows)
{
    auto doutput = createTensor<float>({1, 4, 1});
    setValues(doutput, {1.0F, 2.0F, 3.0F, 4.0F});
    auto token = createTensor<float>({1, 4, 1});
    setValues(token, {10.0F, 20.0F, 30.0F, 40.0F});
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 4});
    auto dweight = createTensor<float>({2, 1, 1});
    dweight.fillWithSentinelValue();

    CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
        doutput, token, offsets, dweight);

    // expert 0 owns rows 0-3: 1*10 + 2*20 + 3*30 + 4*40 = 300
    // expert 1 starts one past the last row -> owns none, must be zero-filled.
    expectTensorValues(dweight, {300.0F, 0.0F});
}

// K=2, N=2, asymmetric DWeight: row-major {K*N, N, 1} and column-major {K*N,
// 1, K} have genuinely different flat contents ({31,42,23,34} vs {31,23,42,34}
// for the same logical [[31,42],[23,34]] matrix), so this only holds if the
// kernel honours the stride split.
TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, ColumnMajorDweightMatchesRowMajor)
{
    auto doutput = createTensor<float>({1, 1, 2});
    setValues(doutput, {1.0F, 10.0F});
    auto token = createTensor<float>({1, 1, 2});
    setValues(token, {1.0F, 2.0F});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});

    auto rowMajorDweight = createTensor<float>({1, 2, 2});
    CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
        doutput, token, offsets, rowMajorDweight);

    auto columnMajorDweight = Tensor<float>({1, 2, 2}, {4, 1, 2});
    CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
        doutput, token, offsets, columnMajorDweight);

    // DWeight[k,n] = Token[0,k] * DOutput[0,n]: [[1,10],[2,20]]. Row-major
    // {K*N,N,1} flattens as [k*N+n] -> {1,10,2,20}; column-major {K*N,1,K}
    // flattens as [k+n*K] -> {1,2,10,20}.
    expectTensorValues(rowMajorDweight, {1.0F, 10.0F, 2.0F, 20.0F});
    expectTensorValues(columnMajorDweight, {1.0F, 2.0F, 10.0F, 20.0F});
}

// Token and DOutput are read through hoisted raw pointers rather than
// getHostValue, so their stride[1]/stride[2] split is only exercised by a layout
// where stride[2] != 1. Both inputs here carry the column-major {rows*W, 1,
// rows}, the transpose of the row-major layout every other test in this file
// builds. DWeight stays row-major in both runs so only the input strides vary.
TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, ColumnMajorTokenAndDoutputMatchRowMajor)
{
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});

    // Token rows [[1,2],[3,4]] and DOutput rows [[5,6],[7,8]].
    auto rowMajorToken = createTensor<float>({1, 2, 2});
    setValues(rowMajorToken, {1.0F, 2.0F, 3.0F, 4.0F});
    auto rowMajorDoutput = createTensor<float>({1, 2, 2});
    setValues(rowMajorDoutput, {5.0F, 6.0F, 7.0F, 8.0F});

    // The same logical values under column-major {4, 1, 2}, where element (r, c)
    // lands at flat index r + 2c instead of 2r + c.
    auto columnMajorToken = Tensor<float>({1, 2, 2}, {4, 1, 2});
    setValues(columnMajorToken, {1.0F, 3.0F, 2.0F, 4.0F});
    auto columnMajorDoutput = Tensor<float>({1, 2, 2}, {4, 1, 2});
    setValues(columnMajorDoutput, {5.0F, 7.0F, 6.0F, 8.0F});

    auto dweightFromRowMajor = createTensor<float>({1, 2, 2});
    CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
        rowMajorDoutput, rowMajorToken, offsets, dweightFromRowMajor);

    auto dweightFromColumnMajor = createTensor<float>({1, 2, 2});
    CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
        columnMajorDoutput, columnMajorToken, offsets, dweightFromColumnMajor);

    // DWeight[k,n] = sum_r Token[r,k] * DOutput[r,n]:
    // [0,0] = 1*5 + 3*7 = 26, [0,1] = 1*6 + 3*8 = 30,
    // [1,0] = 2*5 + 4*7 = 38, [1,1] = 2*6 + 4*8 = 44.
    expectTensorValues(dweightFromRowMajor, {26.0F, 30.0F, 38.0F, 44.0F});
    expectTensorValues(dweightFromColumnMajor, {26.0F, 30.0F, 38.0F, 44.0F});
}

/* ============================= Rejection tests ============================= */

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, DecreasingOffsetsThrows)
{
    auto doutput = createTensor<float>({1, 2, 1});
    setValues(doutput, {1.0F, 2.0F});
    auto token = createTensor<float>({1, 2, 1});
    setValues(token, {1.0F, 2.0F});
    auto offsets = createTensor<int32_t>({3, 1, 1});
    setValues(offsets, {0, 2, 1});
    auto dweight = createTensor<float>({3, 1, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
                doutput, token, offsets, dweight);
        },
        "non-decreasing");
}

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, NonZeroFirstOffsetThrows)
{
    auto doutput = createTensor<float>({1, 2, 1});
    setValues(doutput, {1.0F, 2.0F});
    auto token = createTensor<float>({1, 2, 1});
    setValues(token, {1.0F, 2.0F});
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {1, 2});
    auto dweight = createTensor<float>({2, 1, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
                doutput, token, offsets, dweight);
        },
        "FirstTokenOffset[0] must be 0");
}

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, FirstTokenOffsetExceedsRowsTotalThrows)
{
    auto doutput = createTensor<float>({1, 2, 1});
    setValues(doutput, {1.0F, 2.0F});
    auto token = createTensor<float>({1, 2, 1});
    setValues(token, {1.0F, 2.0F});
    auto offsets = createTensor<int32_t>({2, 1, 1});
    setValues(offsets, {0, 5});
    auto dweight = createTensor<float>({2, 1, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
                doutput, token, offsets, dweight);
        },
        "FirstTokenOffset[1]");
}

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, ExpertCountMismatchThrows)
{
    auto doutput = createTensor<float>({1, 2, 1});
    setValues(doutput, {1.0F, 2.0F});
    auto token = createTensor<float>({1, 2, 1});
    setValues(token, {1.0F, 2.0F});
    // 3 offset rows but dweight only describes 2 experts -- backward requires
    // an exact match (no batching), unlike forward's "multiple of" rule.
    auto offsets = createTensor<int32_t>({3, 1, 1});
    setValues(offsets, {0, 1, 2});
    auto dweight = createTensor<float>({2, 1, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
                doutput, token, offsets, dweight);
        },
        "must equal the expert count");
}

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, InvalidCoreShapesThrow)
{
    struct ShapeCase
    {
        const char* name;
        std::vector<int64_t> doutputDims;
        std::vector<int64_t> tokenDims;
        std::vector<int64_t> firstTokenOffsetDims;
        std::vector<int64_t> dweightDims;
        const char* reason;
    };

    const std::vector<ShapeCase> cases = {
        {"DoutputLeadingDimension",
         {2, 2, 1},
         {1, 2, 1},
         {1, 1, 1},
         {1, 1, 1},
         "doutput must have"},
        {"TokenLeadingDimension", {1, 2, 1}, {2, 2, 1}, {1, 1, 1}, {1, 1, 1}, "token must have"},
        {"FirstTokenOffsetShape",
         {1, 2, 1},
         {1, 2, 1},
         {1, 2, 1},
         {1, 1, 1},
         "first_token_offset must have shape"},
        {"HiddenSizeMismatch", {1, 2, 1}, {1, 2, 1}, {1, 1, 1}, {1, 3, 1}, "hidden size"},
        {"OutputWidthMismatch", {1, 2, 1}, {1, 2, 1}, {1, 1, 1}, {1, 1, 2}, "output width"},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        auto doutput = createTensor<float>(testCase.doutputDims);
        auto token = createTensor<float>(testCase.tokenDims);
        auto offsets = createTensor<int32_t>(testCase.firstTokenOffsetDims);
        auto dweight = createTensor<float>(testCase.dweightDims);

        expectRejectedWith(
            [&] {
                CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
                    doutput, token, offsets, dweight);
            },
            testCase.reason);
    }
}

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, DoutputTokenRowCountMismatchThrows)
{
    auto doutput = createTensor<float>({1, 3, 1});
    auto token = createTensor<float>({1, 2, 1});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto dweight = createTensor<float>({1, 1, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
                doutput, token, offsets, dweight);
        },
        "row count");
}

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, RankTwoTensorsThrow)
{
    auto doutput = createTensor<float>({4, 1});
    auto token = createTensor<float>({4, 1});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto dweight = createTensor<float>({1, 1, 1});

    EXPECT_THROW((CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
                     doutput, token, offsets, dweight)),
                 std::runtime_error);
}

// Every non-ragged tensor here is shape-valid, so the only reason to reject is
// the ragged doutput itself.
TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, RaggedDoutputThrows)
{
    auto raggedOffset = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{2, 1, 1, 1},
                                                          std::vector<int64_t>{1, 1, 1, 1});
    setValues(*raggedOffset, {0.0F, 2.0F});
    RaggedTensor<float> doutput({1, 2, 1}, {2, 1, 1}, 1, raggedOffset);
    auto token = createTensor<float>({1, 2, 1});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto dweight = createTensor<float>({1, 1, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
                doutput, token, offsets, dweight);
        },
        "ragged doutput tensor");
}

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, RaggedDweightThrows)
{
    auto doutput = createTensor<float>({1, 1, 1});
    auto token = createTensor<float>({1, 1, 1});
    auto offsets = createTensor<int32_t>({1, 1, 1});
    setValues(offsets, {0});
    auto raggedOffset = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{2, 1, 1, 1},
                                                          std::vector<int64_t>{1, 1, 1, 1});
    setValues(*raggedOffset, {0.0F, 1.0F});
    RaggedTensor<float> dweight({1, 1, 1}, {1, 1, 1}, 1, raggedOffset);

    expectRejectedWith(
        [&] {
            CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
                doutput, token, offsets, dweight);
        },
        "ragged dweight tensor");
}

/* ============================= Randomized cross-check ============================= */

TEST(TestCpuFpReferenceMoeGroupedMatmulBwd, LargeRandomMatchesNaiveLoop)
{
    constexpr int64_t EXPERTS = 4;
    constexpr int64_t HIDDEN_K = 17;
    constexpr int64_t OUTPUT_N = 13;
    constexpr int64_t TOKEN_ROWS = 64;

    auto doutput = createTensor<float>({1, TOKEN_ROWS, OUTPUT_N});
    auto token = createTensor<float>({1, TOKEN_ROWS, HIDDEN_K});
    doutput.fillWithRandomValues(-1.0F, 1.0F, 42);
    token.fillWithRandomValues(-1.0F, 1.0F, 43);

    auto offsets = createTensor<int32_t>({EXPERTS, 1, 1});
    for(int64_t e = 0; e < EXPERTS; ++e)
    {
        offsets.setHostValue(static_cast<int32_t>(e * TOKEN_ROWS / EXPERTS), {e, 0, 0});
    }

    auto dweight = createTensor<float>({EXPERTS, HIDDEN_K, OUTPUT_N});
    CpuFpReferenceMoeGroupedMatmulBwd::backward<float, float, float, float>(
        doutput, token, offsets, dweight);

    auto naiveDweight = createTensor<float>({EXPERTS, HIDDEN_K, OUTPUT_N});
    for(int64_t e = 0; e < EXPERTS; ++e)
    {
        const int64_t start = e * TOKEN_ROWS / EXPERTS;
        const int64_t end = (e + 1 < EXPERTS) ? (e + 1) * TOKEN_ROWS / EXPERTS : TOKEN_ROWS;
        for(int64_t k = 0; k < HIDDEN_K; ++k)
        {
            for(int64_t n = 0; n < OUTPUT_N; ++n)
            {
                double acc = 0.0;
                for(int64_t r = start; r < end; ++r)
                {
                    acc += static_cast<double>(token.getHostValue({0, r, k}))
                           * static_cast<double>(doutput.getHostValue({0, r, n}));
                }
                naiveDweight.setHostValue(static_cast<float>(acc), {e, k, n});
            }
        }
    }

    const float tolerance = matmul::getTolerance<float>();
    const CpuFpReferenceValidation<float> validator(tolerance, tolerance);
    EXPECT_TRUE(validator.allClose(naiveDweight, dweight));
}
