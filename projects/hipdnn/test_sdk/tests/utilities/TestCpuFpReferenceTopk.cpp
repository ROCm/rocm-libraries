// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceTopk.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

namespace
{

constexpr float K_NAN = std::numeric_limits<float>::quiet_NaN();
constexpr float K_INF = std::numeric_limits<float>::infinity();

template <typename Type>
Tensor<Type> makeTopkTensor(const std::vector<int64_t>& dims)
{
    return Tensor<Type>(dims, generateStrides(dims));
}

template <typename Type>
Tensor<Type> makeTopkTensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    return Tensor<Type>(dims, strides);
}

// Advances a row-major multi-index; returns false after the last position.
bool advance(std::vector<int64_t>& index, const std::vector<int64_t>& dims)
{
    for(size_t d = dims.size(); d-- > 0;)
    {
        if(++index[d] < dims[d])
        {
            return true;
        }
        index[d] = 0;
    }
    return false;
}

// Writes `values` in logical row-major order through the tensor's own strides.
template <typename Type>
void setLogicalValues(Tensor<Type>& tensor, const std::vector<float>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    std::vector<int64_t> index(tensor.dims().size(), 0);
    for(const auto value : values)
    {
        tensor.setHostValue(static_cast<Type>(value), index);
        advance(index, tensor.dims());
    }
}

// Reads the tensor in logical row-major order through its own strides.
template <typename Type>
std::vector<float> logicalValues(const Tensor<Type>& tensor)
{
    std::vector<float> result;
    std::vector<int64_t> index(tensor.dims().size(), 0);
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        result.push_back(static_cast<float>(tensor.getHostValue(index)));
        advance(index, tensor.dims());
    }
    return result;
}

template <typename IndexType>
std::vector<int64_t> logicalIndices(const Tensor<IndexType>& tensor)
{
    std::vector<int64_t> result;
    std::vector<int64_t> index(tensor.dims().size(), 0);
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        result.push_back(static_cast<int64_t>(tensor.getHostValue(index)));
        advance(index, tensor.dims());
    }
    return result;
}

} // namespace

template <typename T>
class CpuFpReferenceTopkTyped : public ::testing::Test
{
};

// Every value below is exact in half and bfloat16.
using TopkDataTypes = ::testing::Types<float, half, bfloat16>;
TYPED_TEST_SUITE(CpuFpReferenceTopkTyped, TopkDataTypes, );

TYPED_TEST(CpuFpReferenceTopkTyped, LargestTiesKeepLowerIndexFirst)
{
    auto x = makeTopkTensor<TypeParam>({1, 6});
    setLogicalValues(x, {3, 1, 3, 2, 3, 0});
    auto indices = makeTopkTensor<int64_t>({1, 4});
    auto values = makeTopkTensor<TypeParam>({1, 4});

    CpuFpReferenceTopk::topk(x, indices, &values, 4, true);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{0, 2, 4, 3}));
    EXPECT_EQ(logicalValues(values), (std::vector<float>{3, 3, 3, 2}));
}

TYPED_TEST(CpuFpReferenceTopkTyped, SmallestTiesKeepLowerIndexFirst)
{
    auto x = makeTopkTensor<TypeParam>({1, 6});
    setLogicalValues(x, {3, 1, 3, 1, 3, 0});
    auto indices = makeTopkTensor<int32_t>({1, 3});
    auto values = makeTopkTensor<TypeParam>({1, 3});

    CpuFpReferenceTopk::topk(x, indices, &values, 3, false);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{5, 1, 3}));
    EXPECT_EQ(logicalValues(values), (std::vector<float>{0, 1, 1}));
}

TYPED_TEST(CpuFpReferenceTopkTyped, KOneIsArgmax)
{
    auto x = makeTopkTensor<TypeParam>({2, 4});
    setLogicalValues(x, {1, 5, 5, 2, -1, -3, -1, -2});
    auto indices = makeTopkTensor<int64_t>({2, 1});
    auto values = makeTopkTensor<TypeParam>({2, 1});

    CpuFpReferenceTopk::topk(x, indices, &values, 1, true);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{1, 0}));
    EXPECT_EQ(logicalValues(values), (std::vector<float>{5, -1}));
}

TYPED_TEST(CpuFpReferenceTopkTyped, KEqualToLastDimSortsWholeRowAscending)
{
    auto x = makeTopkTensor<TypeParam>({1, 5});
    setLogicalValues(x, {2, -1, 0, -1, 4});
    auto indices = makeTopkTensor<int32_t>({1, 5});
    auto values = makeTopkTensor<TypeParam>({1, 5});

    CpuFpReferenceTopk::topk(x, indices, &values, 5, false);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{1, 3, 2, 0, 4}));
    EXPECT_EQ(logicalValues(values), (std::vector<float>{-1, -1, 0, 2, 4}));
}

TYPED_TEST(CpuFpReferenceTopkTyped, NanRanksFirstWhenSelectingLargest)
{
    auto x = makeTopkTensor<TypeParam>({1, 5});
    setLogicalValues(x, {1, K_NAN, 3, K_NAN, 2});
    auto indices = makeTopkTensor<int64_t>({1, 3});
    auto values = makeTopkTensor<TypeParam>({1, 3});

    CpuFpReferenceTopk::topk(x, indices, &values, 3, true);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{1, 3, 2}));
    const auto selected = logicalValues(values);
    EXPECT_TRUE(std::isnan(selected[0]));
    EXPECT_TRUE(std::isnan(selected[1]));
    EXPECT_EQ(selected[2], 3.0f);
}

TYPED_TEST(CpuFpReferenceTopkTyped, NanRanksLastWhenSelectingSmallest)
{
    auto x = makeTopkTensor<TypeParam>({1, 5});
    setLogicalValues(x, {1, K_NAN, 3, K_NAN, 2});
    auto indices = makeTopkTensor<int64_t>({1, 5});
    auto values = makeTopkTensor<TypeParam>({1, 5});

    CpuFpReferenceTopk::topk(x, indices, &values, 5, false);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{0, 4, 2, 1, 3}));
    const auto selected = logicalValues(values);
    EXPECT_EQ(selected[0], 1.0f);
    EXPECT_EQ(selected[1], 2.0f);
    EXPECT_EQ(selected[2], 3.0f);
    EXPECT_TRUE(std::isnan(selected[3]));
    EXPECT_TRUE(std::isnan(selected[4]));
}

TYPED_TEST(CpuFpReferenceTopkTyped, MultiDimBatchRows)
{
    // Rows of x [2, 2, 3]: {1, 2, 3}, {6, 5, 4}, {0, 0, 1}, {-1, -2, -3}.
    auto x = makeTopkTensor<TypeParam>({2, 2, 3});
    setLogicalValues(x, {1, 2, 3, 6, 5, 4, 0, 0, 1, -1, -2, -3});
    auto indices = makeTopkTensor<int32_t>({2, 2, 2});
    auto values = makeTopkTensor<TypeParam>({2, 2, 2});

    CpuFpReferenceTopk::topk(x, indices, &values, 2, true);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{2, 1, 0, 1, 2, 0, 0, 1}));
    EXPECT_EQ(logicalValues(values), (std::vector<float>{3, 2, 6, 5, 1, 0, -1, -2}));
}

TEST(TestCpuFpReferenceTopk, Int32AndInt64IndicesAgree)
{
    auto x = makeTopkTensor<float>({3, 4});
    setLogicalValues(x, {0.5f, -1, 7, 7, 2, 2, 2, 2, -4, 9, -8, 1});
    auto indices32 = makeTopkTensor<int32_t>({3, 2});
    auto indices64 = makeTopkTensor<int64_t>({3, 2});

    CpuFpReferenceTopk::topk<float, int32_t>(x, indices32, nullptr, 2, true);
    CpuFpReferenceTopk::topk<float, int64_t>(x, indices64, nullptr, 2, true);

    const std::vector<int64_t> expected{2, 3, 0, 1, 1, 3};
    EXPECT_EQ(logicalIndices(indices32), expected);
    EXPECT_EQ(logicalIndices(indices64), expected);
}

TEST(TestCpuFpReferenceTopk, ValuesOutputIsOptional)
{
    auto x = makeTopkTensor<float>({1, 4});
    setLogicalValues(x, {4, 1, 3, 2});
    auto indices = makeTopkTensor<int64_t>({1, 2});

    CpuFpReferenceTopk::topk<float, int64_t>(x, indices, nullptr, 2, false);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{1, 3}));
}

TEST(TestCpuFpReferenceTopk, SignedZerosTieAndInfinityRanksBelowNan)
{
    auto x = makeTopkTensor<float>({1, 4});
    setLogicalValues(x, {-0.0f, K_INF, 0.0f, K_NAN});
    auto indices = makeTopkTensor<int64_t>({1, 4});
    auto values = makeTopkTensor<float>({1, 4});

    CpuFpReferenceTopk::topk(x, indices, &values, 4, true);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{3, 1, 0, 2}));
    const auto selected = logicalValues(values);
    EXPECT_TRUE(std::isnan(selected[0]));
    EXPECT_EQ(selected[1], K_INF);
    // -0.0 and 0.0 compare equal, so the lower position comes first with its own sign.
    EXPECT_TRUE(std::signbit(selected[2]));
    EXPECT_FALSE(std::signbit(selected[3]));
}

TEST(TestCpuFpReferenceTopk, NonPackedStrides)
{
    // Column-major x [2, 3]: logical rows {3, 1, 2} and {-1, 5, 0}.
    auto x = makeTopkTensor<float>({2, 3}, {1, 2});
    setLogicalValues(x, {3, 1, 2, -1, 5, 0});
    auto indices = makeTopkTensor<int64_t>({2, 2}, {1, 2});
    auto values = makeTopkTensor<float>({2, 2}, {1, 2});

    CpuFpReferenceTopk::topk(x, indices, &values, 2, false);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{1, 2, 0, 2}));
    EXPECT_EQ(logicalValues(values), (std::vector<float>{1, 2, -1, 0}));
}

TEST(TestCpuFpReferenceTopk, RankOneInput)
{
    auto x = makeTopkTensor<float>({4});
    setLogicalValues(x, {2, 8, 8, 1});
    auto indices = makeTopkTensor<int32_t>({2});
    auto values = makeTopkTensor<float>({2});

    CpuFpReferenceTopk::topk(x, indices, &values, 2, true);

    EXPECT_EQ(logicalIndices(indices), (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(logicalValues(values), (std::vector<float>{8, 8}));
}

TEST(TestCpuFpReferenceTopk, RejectsInvalidK)
{
    auto x = makeTopkTensor<float>({2, 4});
    // Tensors cannot have a zero dim, so k = 0 is checked against a [2, 1] indices tensor:
    // the k range check rejects it before any dims comparison.
    auto indicesOne = makeTopkTensor<int64_t>({2, 1});
    auto indicesFive = makeTopkTensor<int64_t>({2, 5});

    EXPECT_THROW((CpuFpReferenceTopk::topk<float, int64_t>(x, indicesOne, nullptr, 0, true)),
                 std::runtime_error);
    EXPECT_THROW((CpuFpReferenceTopk::topk<float, int64_t>(x, indicesFive, nullptr, 5, true)),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceTopk, RejectsMismatchedOutputDims)
{
    auto x = makeTopkTensor<float>({2, 4});
    auto goodIndices = makeTopkTensor<int64_t>({2, 2});
    auto badIndices = makeTopkTensor<int64_t>({2, 3});
    auto badValues = makeTopkTensor<float>({1, 2});

    EXPECT_THROW((CpuFpReferenceTopk::topk<float, int64_t>(x, badIndices, nullptr, 2, true)),
                 std::runtime_error);
    EXPECT_THROW(CpuFpReferenceTopk::topk(x, goodIndices, &badValues, 2, true), std::runtime_error);
}
