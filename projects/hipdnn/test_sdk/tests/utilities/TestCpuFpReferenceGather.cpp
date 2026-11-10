// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceGather.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

namespace
{

template <typename Type>
Tensor<Type> makePackedTensor(const std::vector<int64_t>& dims)
{
    return Tensor<Type>(dims, generateStrides(dims));
}

// Writes `values` in logical row-major order through the tensor's own strides.
template <typename Type>
void setLogicalValues(Tensor<Type>& tensor, const std::vector<float>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    const auto& dims = tensor.dims();
    std::vector<int64_t> index(dims.size(), 0);
    for(const auto value : values)
    {
        tensor.setHostValue(static_cast<Type>(value), index);
        for(size_t d = dims.size(); d-- > 0;)
        {
            if(++index[d] < dims[d])
            {
                break;
            }
            index[d] = 0;
        }
    }
}

template <typename IndexType>
void setIndices(Tensor<IndexType>& tensor, const std::vector<int64_t>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = static_cast<IndexType>(values[i]);
    }
}

// Reads the tensor in logical row-major order through its own strides.
template <typename Type>
std::vector<float> logicalValues(const Tensor<Type>& tensor)
{
    std::vector<float> result;
    const auto& dims = tensor.dims();
    const auto count = tensor.elementCount();
    std::vector<int64_t> index(dims.size(), 0);
    for(size_t i = 0; i < count; ++i)
    {
        result.push_back(static_cast<float>(tensor.getHostValue(index)));
        for(size_t d = dims.size(); d-- > 0;)
        {
            if(++index[d] < dims[d])
            {
                break;
            }
            index[d] = 0;
        }
    }
    return result;
}

// Embedding table [4, 3] with x[v, h] = 10 * v + h (exact in half and bfloat16).
template <typename Type>
Tensor<Type> createEmbeddingTable()
{
    auto table = makePackedTensor<Type>({4, 3});
    setLogicalValues(table, {0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32});
    return table;
}

} // namespace

template <typename T>
class CpuFpReferenceGatherTyped : public ::testing::Test
{
};

using GatherDataTypes = ::testing::Types<float, half, bfloat16>;
TYPED_TEST_SUITE(CpuFpReferenceGatherTyped, GatherDataTypes, );

TYPED_TEST(CpuFpReferenceGatherTyped, EmbeddingLookupRankOneIdsInt32)
{
    auto table = createEmbeddingTable<TypeParam>();
    auto ids = makePackedTensor<int32_t>({3});
    setIndices(ids, {2, 0, 2});
    auto y = makePackedTensor<TypeParam>({3, 3});

    CpuFpReferenceGather::gather(table, ids, y, 0);

    EXPECT_EQ(logicalValues(y), (std::vector<float>{20, 21, 22, 0, 1, 2, 20, 21, 22}));
}

TYPED_TEST(CpuFpReferenceGatherTyped, EmbeddingLookupBatchedIdsInt64)
{
    auto table = createEmbeddingTable<TypeParam>();
    auto ids = makePackedTensor<int64_t>({2, 2});
    setIndices(ids, {3, 1, 0, 3});
    auto y = makePackedTensor<TypeParam>({2, 2, 3});

    CpuFpReferenceGather::gather(table, ids, y, 0);

    EXPECT_EQ(logicalValues(y), (std::vector<float>{30, 31, 32, 10, 11, 12, 0, 1, 2, 30, 31, 32}));
}

TYPED_TEST(CpuFpReferenceGatherTyped, MiddleAxisGather)
{
    // x[a, b, c] = 100 * a + 10 * b + c with dims [2, 3, 2].
    auto x = makePackedTensor<TypeParam>({2, 3, 2});
    setLogicalValues(x, {0, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121});
    auto indices = makePackedTensor<int32_t>({2});
    setIndices(indices, {2, 0});
    auto y = makePackedTensor<TypeParam>({2, 2, 2});

    CpuFpReferenceGather::gather(x, indices, y, 1);

    EXPECT_EQ(logicalValues(y), (std::vector<float>{20, 21, 0, 1, 120, 121, 100, 101}));
}

TEST(TestCpuFpReferenceGather, LastAxisWithRankTwoIndices)
{
    // x [2, 3]; indices [2, 2] along axis 1 -> y [2, 2, 2].
    auto x = makePackedTensor<float>({2, 3});
    setLogicalValues(x, {0, 1, 2, 10, 11, 12});
    auto indices = makePackedTensor<int64_t>({2, 2});
    setIndices(indices, {2, 2, 1, 0});
    auto y = makePackedTensor<float>({2, 2, 2});

    CpuFpReferenceGather::gather(x, indices, y, 1);

    EXPECT_EQ(logicalValues(y), (std::vector<float>{2, 2, 1, 0, 12, 12, 11, 10}));
}

TEST(TestCpuFpReferenceGather, Int32AndInt64IndicesAgree)
{
    auto table = createEmbeddingTable<float>();
    auto ids32 = makePackedTensor<int32_t>({4});
    auto ids64 = makePackedTensor<int64_t>({4});
    setIndices(ids32, {1, 3, 3, 0});
    setIndices(ids64, {1, 3, 3, 0});
    auto y32 = makePackedTensor<float>({4, 3});
    auto y64 = makePackedTensor<float>({4, 3});

    CpuFpReferenceGather::gather(table, ids32, y32, 0);
    CpuFpReferenceGather::gather(table, ids64, y64, 0);

    EXPECT_EQ(logicalValues(y32), logicalValues(y64));
    EXPECT_EQ(logicalValues(y32),
              (std::vector<float>{10, 11, 12, 30, 31, 32, 30, 31, 32, 0, 1, 2}));
}

TEST(TestCpuFpReferenceGather, StridedInputAndOutput)
{
    // Column-major table [4, 3] (strides {1, 4}) and column-major y [2, 3] (strides {1, 2}).
    Tensor<float> table({4, 3}, {1, 4});
    setLogicalValues(table, {0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32});
    auto ids = makePackedTensor<int32_t>({2});
    setIndices(ids, {1, 2});
    Tensor<float> y({2, 3}, {1, 2});

    CpuFpReferenceGather::gather(table, ids, y, 0);

    EXPECT_EQ(logicalValues(y), (std::vector<float>{10, 11, 12, 20, 21, 22}));
    // Physical layout of y follows its column-major strides.
    const auto* yData = y.memory().hostData();
    EXPECT_EQ(yData[0], 10.0f);
    EXPECT_EQ(yData[1], 20.0f);
    EXPECT_EQ(yData[2], 11.0f);
}

TEST(TestCpuFpReferenceGather, IndexEqualToExtentThrowsAndLeavesOutputUntouched)
{
    auto table = createEmbeddingTable<float>();
    auto ids = makePackedTensor<int32_t>({2});
    setIndices(ids, {1, 4});
    auto y = makePackedTensor<float>({2, 3});
    y.fillTensorWithValue(-7.0f);

    EXPECT_THROW(CpuFpReferenceGather::gather(table, ids, y, 0), std::runtime_error);
    EXPECT_EQ(logicalValues(y), (std::vector<float>(6, -7.0f)));
}

TEST(TestCpuFpReferenceGather, NegativeIndexThrows)
{
    auto table = createEmbeddingTable<float>();
    auto ids = makePackedTensor<int64_t>({1});
    setIndices(ids, {-1});
    auto y = makePackedTensor<float>({1, 3});

    EXPECT_THROW(CpuFpReferenceGather::gather(table, ids, y, 0), std::runtime_error);
}

TEST(TestCpuFpReferenceGather, AxisOutOfRangeThrows)
{
    auto table = createEmbeddingTable<float>();
    auto ids = makePackedTensor<int32_t>({1});
    setIndices(ids, {0});
    auto y = makePackedTensor<float>({4, 1});

    EXPECT_THROW(CpuFpReferenceGather::gather(table, ids, y, 2), std::runtime_error);
    EXPECT_THROW(CpuFpReferenceGather::gather(table, ids, y, -1), std::runtime_error);
}

TEST(TestCpuFpReferenceGather, MismatchedOutputDimsThrow)
{
    auto table = createEmbeddingTable<float>();
    auto ids = makePackedTensor<int32_t>({2});
    setIndices(ids, {0, 1});
    auto y = makePackedTensor<float>({3, 3});

    EXPECT_THROW(CpuFpReferenceGather::gather(table, ids, y, 0), std::runtime_error);
}
