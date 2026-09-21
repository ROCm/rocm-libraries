// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include "cpu_graph_executor/MatmulGraphUtils.hpp"
#include "cpu_graph_executor/MatmulTensorBundles.hpp"
#include "cpu_graph_executor/PointwiseGraphUtils.hpp"

#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_sdk_test_utils;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

namespace
{

template <typename Type>
Tensor<Type> createTensor(const std::vector<int64_t>& dims, bool transpose = false)
{
    std::vector<int64_t> strides = generateStrides(dims);
    if(transpose)
    {
        const size_t rank = dims.size();
        strides[rank - 1] = dims[rank - 2];
        strides[rank - 2] = 1;
    }
    return Tensor<Type>(dims, strides);
};

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

} // namespace

/* ============================= Unit tests ============================= */

class TestCpuFpReferenceMatmul : public ::testing::Test
{
};

TEST_F(TestCpuFpReferenceMatmul, IsApplicable)
{
    // CpuFpReferenceMatmul::isApplicable should return true for Matmul node
    {
        const std::vector<int64_t> aDims = {2, 2, 3};
        const std::vector<int64_t> bDims = {2, 3, 4};
        const std::vector<int64_t> cDims = {2, 2, 4};

        MatmulTensorBundle<float> tensorBundle(aDims, bDims, cDims, false, false, 1);
        auto graphTuple = buildMatmulGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

        auto& graph = std::get<0>(graphTuple);
        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrap(
            serializedGraph.data(), serializedGraph.size());
        EXPECT_TRUE(CpuFpReferenceMatmul::isApplicable(graphWrap.getNode(0)));
    }

    // CpuFpReferenceMatmul::isApplicable should return false for any other node type
    {
        const std::vector<int64_t> dims = {1, 3, 4, 4};

        auto graphTuple = buildPointwiseUnaryGraph(dims,
                                                   dims,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                                   1,
                                                   TensorLayout::NCHW);

        auto& graph = std::get<0>(graphTuple);
        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrap(
            serializedGraph.data(), serializedGraph.size());
        EXPECT_FALSE(CpuFpReferenceMatmul::isApplicable(graphWrap.getNode(0)));
    }
}

TEST_F(TestCpuFpReferenceMatmul, ValidateInput)
{
    // Rank mismatching
    {
        auto tensorA = createTensor<float>({2, 2, 3});
        auto tensorB = createTensor<float>({3, 4});
        auto tensorC = createTensor<float>({2, 2, 4});

        EXPECT_THROW(
            (CpuFpReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
            std::invalid_argument);
    }

    // Rank less than 2
    {
        auto tensorA = createTensor<float>({3});
        auto tensorB = createTensor<float>({3});
        auto tensorC = createTensor<float>({3});

        EXPECT_THROW(
            (CpuFpReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
            std::invalid_argument);
    }

    // Mismatched K dimensions
    {
        auto tensorA = createTensor<float>({2, 2, 5});
        auto tensorB = createTensor<float>({2, 2, 5});
        auto tensorC = createTensor<float>({2, 2, 2});

        EXPECT_THROW(
            (CpuFpReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
            std::invalid_argument);
    }

    // Incorrect output shape
    {
        auto tensorA = createTensor<float>({2, 3, 5});
        auto tensorB = createTensor<float>({2, 5, 4});
        auto tensorC = createTensor<float>({2, 4, 3});

        EXPECT_THROW(
            (CpuFpReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
            std::invalid_argument);
    }
}

TEST_F(TestCpuFpReferenceMatmul, ValidateBroadcastableBatchDims)
{
    // Correct case
    {
        auto tensorA = createTensor<float>({2, 1, 2, 3});
        auto tensorB = createTensor<float>({1, 2, 3, 4});
        auto tensorC = createTensor<float>({2, 2, 2, 4});

        EXPECT_NO_THROW(
            (CpuFpReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)));
    }

    // Non-divisible batch dims
    {
        auto tensorA = createTensor<float>({2, 2, 2, 3});
        auto tensorB = createTensor<float>({3, 2, 3, 4});
        auto tensorC = createTensor<float>({2, 2, 3, 4});

        EXPECT_THROW(
            (CpuFpReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
            std::invalid_argument);
    }

    // Incorrect output batch dims
    {
        auto tensorA = createTensor<float>({2, 1, 2, 3});
        auto tensorB = createTensor<float>({1, 2, 3, 4});
        auto tensorC = createTensor<float>({1, 1, 2, 4});

        EXPECT_THROW(
            (CpuFpReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
            std::invalid_argument);
    }
}

/* ====================================================================== */

/* ============================= Func tests ============================= */

template <typename T1, typename T2, typename T3, typename T4>
struct TypePair
{
    using ADataType = T1;
    using BDataType = T2;
    using CDataType = T3;
    using ComputeDataType = T4;
};

using Types = ::testing::Types<TypePair<float, float, float, float>,
                               TypePair<half, half, half, float>,
                               TypePair<bfloat16, bfloat16, bfloat16, float>,
                               TypePair<float, half, float, float>>;

template <class T>
class CpuFpReferenceMatmulBasic : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceMatmulBasic, Types, );

TYPED_TEST(CpuFpReferenceMatmulBasic, Matmul)
{
    auto tensorA = createTensor<typename TypeParam::ADataType>({2, 3});
    auto tensorB = createTensor<typename TypeParam::BDataType>({3, 2});
    auto tensorC = createTensor<typename TypeParam::CDataType>({2, 2});

    // Fill input with sequential values
    const int tensorAElementCount = static_cast<int>(tensorA.elementCount());
    const int tensorBElementCount = static_cast<int>(tensorB.elementCount());
    for(int i = 0; i < tensorAElementCount; ++i)
    {
        tensorA.memory().hostData()[i]
            = safeTestTypeCast<typename TypeParam::ADataType>(static_cast<float>(i + 1));
    }
    for(int i = 0; i < tensorBElementCount; ++i)
    {
        tensorB.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::BDataType>(
            static_cast<float>(i + tensorAElementCount));
    }

    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul<typename TypeParam::ADataType,
                                                             typename TypeParam::BDataType,
                                                             typename TypeParam::CDataType,
                                                             typename TypeParam::ComputeDataType>(
        tensorA, tensorB, tensorC);
    // Expected output for this configuration:
    // output[0, 0] = 1x6 + 2x8 + 3x10 = 52
    // output[0, 1] = 1x7 + 2x9 + 3x11 = 58
    // output[1, 0] = 4x6 + 5x8 + 6x10 = 124
    // output[1, 1] = 4x7 + 5x9 + 6x11 = 139
    expectTensorValues(tensorC, {52.0f, 58.0f, 124.0f, 139.0f});
}

TYPED_TEST(CpuFpReferenceMatmulBasic, MatmulTransposeA)
{
    auto tensorA = createTensor<typename TypeParam::ADataType>({2, 3}, true);
    auto tensorB = createTensor<typename TypeParam::BDataType>({3, 2});
    auto tensorC = createTensor<typename TypeParam::CDataType>({2, 2});

    // Fill input with sequential values
    const int tensorAElementCount = static_cast<int>(tensorA.elementCount());
    const int tensorBElementCount = static_cast<int>(tensorB.elementCount());
    for(int i = 0; i < tensorAElementCount; ++i)
    {
        tensorA.memory().hostData()[i]
            = safeTestTypeCast<typename TypeParam::ADataType>(static_cast<float>(i + 1));
    }
    for(int i = 0; i < tensorBElementCount; ++i)
    {
        tensorB.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::BDataType>(
            static_cast<float>(i + tensorAElementCount));
    }

    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul<typename TypeParam::ADataType,
                                                             typename TypeParam::BDataType,
                                                             typename TypeParam::CDataType,
                                                             typename TypeParam::ComputeDataType>(
        tensorA, tensorB, tensorC);

    // Expected output for this configuration:
    // output[0, 0] = 1x6 + 3x8 + 5x10 = 80
    // output[0, 1] = 1x7 + 3x9 + 5x11 = 89
    // output[1, 0] = 2x6 + 4x8 + 6x10 = 104
    // output[1, 1] = 2x7 + 4x9 + 6x11 = 116
    expectTensorValues(tensorC, {80.0f, 89.0f, 104.0f, 116.0f});
}

TYPED_TEST(CpuFpReferenceMatmulBasic, MatmulTransposeB)
{
    auto tensorA = createTensor<typename TypeParam::ADataType>({2, 3});
    auto tensorB = createTensor<typename TypeParam::BDataType>({3, 2}, true);
    auto tensorC = createTensor<typename TypeParam::CDataType>({2, 2});

    // Fill input with sequential values
    const int tensorAElementCount = static_cast<int>(tensorA.elementCount());
    const int tensorBElementCount = static_cast<int>(tensorB.elementCount());
    for(int i = 0; i < tensorAElementCount; ++i)
    {
        tensorA.memory().hostData()[i]
            = safeTestTypeCast<typename TypeParam::ADataType>(static_cast<float>(i + 1));
    }
    for(int i = 0; i < tensorBElementCount; ++i)
    {
        tensorB.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::BDataType>(
            static_cast<float>(i + tensorAElementCount));
    }

    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul<typename TypeParam::ADataType,
                                                             typename TypeParam::BDataType,
                                                             typename TypeParam::CDataType,
                                                             typename TypeParam::ComputeDataType>(
        tensorA, tensorB, tensorC);

    // Expected output for this configuration:
    // output[0, 0] = 1x6 + 2x7 + 3x8 = 44
    // output[0, 1] = 1x8 + 2x9 + 3x10 = 62
    // output[1, 0] = 4x6 + 5x7 + 6x8 = 107
    // output[1, 1] = 4x8 + 5x9 + 6x10 = 152
    expectTensorValues(tensorC, {44.0f, 62.0f, 107.0f, 152.0f});
}

TYPED_TEST(CpuFpReferenceMatmulBasic, MatmulTransposeBoth)
{
    auto tensorA = createTensor<typename TypeParam::ADataType>({2, 3}, true);
    auto tensorB = createTensor<typename TypeParam::BDataType>({3, 2}, true);
    auto tensorC = createTensor<typename TypeParam::CDataType>({2, 2});

    const int tensorAElementCount = static_cast<int>(tensorA.elementCount());
    const int tensorBElementCount = static_cast<int>(tensorB.elementCount());
    for(int i = 0; i < tensorAElementCount; ++i)
    {
        tensorA.memory().hostData()[i]
            = safeTestTypeCast<typename TypeParam::ADataType>(static_cast<float>(i + 1));
    }
    for(int i = 0; i < tensorBElementCount; ++i)
    {
        tensorB.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::BDataType>(
            static_cast<float>(i + tensorAElementCount));
    }

    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul<typename TypeParam::ADataType,
                                                             typename TypeParam::BDataType,
                                                             typename TypeParam::CDataType,
                                                             typename TypeParam::ComputeDataType>(
        tensorA, tensorB, tensorC);

    // Expected output for this configuration:
    // output[0, 0] = 1x6 + 3x7 + 5x8 = 67
    // output[0, 1] = 1x9 + 3x10 + 5x11 = 94
    // output[1, 0] = 2x6 + 4x7 + 6x8 = 88
    // output[1, 1] = 2x9 + 4x10 + 6x11 = 124
    expectTensorValues(tensorC, {67.0f, 94.0f, 88.0f, 124.0f});
}

TYPED_TEST(CpuFpReferenceMatmulBasic, MatmulBatch3D)
{
    auto tensorA = createTensor<typename TypeParam::ADataType>({2, 2, 3});
    auto tensorB = createTensor<typename TypeParam::BDataType>({2, 3, 2});
    auto tensorC = createTensor<typename TypeParam::CDataType>({2, 2, 2});

    const int tensorAElementCount = static_cast<int>(tensorA.elementCount());
    const int tensorBElementCount = static_cast<int>(tensorB.elementCount());
    const int halfTensorACount = tensorAElementCount / 2;
    for(int i = 0; i < tensorAElementCount; ++i)
    {
        tensorA.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::ADataType>(
            static_cast<float>(i - halfTensorACount));
    }
    for(int i = 0; i < tensorBElementCount; ++i)
    {
        tensorB.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::BDataType>(
            static_cast<float>(i - tensorAElementCount));
    }

    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul<typename TypeParam::ADataType,
                                                             typename TypeParam::BDataType,
                                                             typename TypeParam::CDataType,
                                                             typename TypeParam::ComputeDataType>(
        tensorA, tensorB, tensorC);

    // Batch 0 expected:
    // output[0, 0, 0] = (-6)x(-12) + (-5)x(-10) + (-4)x(-8)  = 154
    // output[0, 0, 1] = (-6)x(-11) + (-5)x(-9)  + (-4)x(-7)  = 139
    // output[0, 1, 0] = (-3)x(-12) + (-2)x(-10) + (-1)x(-8)  = 64
    // output[0, 1, 1] = (-3)x(-11) + (-2)x(-9)  + (-1)x(-7)  = 58
    // Batch 1 expected:
    // output[1, 0, 0] = 0x(-6) + 1x(-4) + 2x(-2) = -8
    // output[1, 0, 1] = 0x(-5) + 1x(-3) + 2x(-1) = -5
    // output[1, 1, 0] = 3x(-6) + 4x(-4) + 5x(-2) = -44
    // output[1, 1, 1] = 3x(-5) + 4x(-3) + 5x(-1) = -32
    expectTensorValues(tensorC, {154.0f, 139.0f, 64.0f, 58.0f, -8.0f, -5.0f, -44.0f, -32.0f});
}

TYPED_TEST(CpuFpReferenceMatmulBasic, Matmul3DBroadcast)
{
    auto tensorA = createTensor<typename TypeParam::ADataType>({2, 2, 3});
    auto tensorB = createTensor<typename TypeParam::BDataType>({1, 3, 2});
    auto tensorC = createTensor<typename TypeParam::CDataType>({2, 2, 2});

    const int tensorAElementCount = static_cast<int>(tensorA.elementCount());
    const int tensorBElementCount = static_cast<int>(tensorB.elementCount());
    const int halfTensorACount = tensorAElementCount / 2;
    for(int i = 0; i < tensorAElementCount; ++i)
    {
        tensorA.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::ADataType>(
            static_cast<float>(i - halfTensorACount));
    }
    for(int i = 0; i < tensorBElementCount; ++i)
    {
        tensorB.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::BDataType>(
            static_cast<float>(i + halfTensorACount));
    }

    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul<typename TypeParam::ADataType,
                                                             typename TypeParam::BDataType,
                                                             typename TypeParam::CDataType,
                                                             typename TypeParam::ComputeDataType>(
        tensorA, tensorB, tensorC);

    // Batch 0 expected:
    // output[0, 0, 0] = (-6)x6 + (-5)x8 + (-4)x10 = -116
    // output[0, 0, 1] = (-6)x7 + (-5)x9 + (-4)x11 = -131
    // output[0, 1, 0] = (-3)x6 + (-2)x8 + (-1)x10 = -44
    // output[0, 1, 1] = (-3)x7 + (-2)x9 + (-1)x11 = -50
    // Batch 1 expected:
    // output[1, 0, 0] = 0x6 + 1x8 + 2x10 = 28
    // output[1, 0, 1] = 0x7 + 1x9 + 2x11 = 31
    // output[1, 1, 0] = 3x6 + 4x8 + 5x10 = 100
    // output[1, 1, 1] = 3x7 + 4x9 + 5x11 = 112
    expectTensorValues(tensorC, {-116.0f, -131.0f, -44.0f, -50.0f, 28.0f, 31.0f, 100.0f, 112.0f});
}

TYPED_TEST(CpuFpReferenceMatmulBasic, MatmulMultipleBatchDims4D)
{
    auto tensorA = createTensor<typename TypeParam::ADataType>({2, 2, 1, 2});
    auto tensorB = createTensor<typename TypeParam::BDataType>({2, 2, 2, 1});
    auto tensorC = createTensor<typename TypeParam::CDataType>({2, 2, 1, 1});

    const int tensorAElementCount = static_cast<int>(tensorA.elementCount());
    const int tensorBElementCount = static_cast<int>(tensorB.elementCount());
    for(int i = 0; i < tensorAElementCount; ++i)
    {
        tensorA.memory().hostData()[i]
            = safeTestTypeCast<typename TypeParam::ADataType>(static_cast<float>(i + 1));
    }
    for(int i = 0; i < tensorBElementCount; ++i)
    {
        tensorB.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::BDataType>(
            static_cast<float>(i + tensorAElementCount));
    }

    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul<typename TypeParam::ADataType,
                                                             typename TypeParam::BDataType,
                                                             typename TypeParam::CDataType,
                                                             typename TypeParam::ComputeDataType>(
        tensorA, tensorB, tensorC);

    // Expected output:
    // output[0, 0] : 1x8 + 2x9   = 26
    // output[0, 1] : 3x10 + 4x11 = 74
    // output[1, 0] : 5x12 + 6x13 = 138
    // output[1, 1] : 7x14 + 8x15 = 218
    expectTensorValues(tensorC, {26.0f, 74.0f, 138.0f, 218.0f});
}

TYPED_TEST(CpuFpReferenceMatmulBasic, Matmul4DBroadcast)
{
    auto tensorA = createTensor<typename TypeParam::ADataType>({2, 1, 2, 2});
    auto tensorB = createTensor<typename TypeParam::BDataType>({1, 3, 2, 1});
    auto tensorC = createTensor<typename TypeParam::CDataType>({2, 3, 2, 1});

    const int tensorAElementCount = static_cast<int>(tensorA.elementCount());
    const int tensorBElementCount = static_cast<int>(tensorB.elementCount());
    const int halfTensorACount = tensorAElementCount / 2;
    for(int i = 0; i < tensorAElementCount; ++i)
    {
        tensorA.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::ADataType>(
            static_cast<float>(i - halfTensorACount));
    }
    for(int i = 0; i < tensorBElementCount; ++i)
    {
        tensorB.memory().hostData()[i] = safeTestTypeCast<typename TypeParam::BDataType>(
            static_cast<float>(i + halfTensorACount));
    }

    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul<typename TypeParam::ADataType,
                                                             typename TypeParam::BDataType,
                                                             typename TypeParam::CDataType,
                                                             typename TypeParam::ComputeDataType>(
        tensorA, tensorB, tensorC);

    // Expected output:
    // output[0, 0]: (-4)x4 + (-3)x5 = -31,  (-2)x4 + (-1)x5 = -13
    // output[0, 1]: (-4)x6 + (-3)x7 = -45,  (-2)x6 + (-1)x7 = -19
    // output[0, 2]: (-4)x8 + (-3)x9 = -59,  (-2)x8 + (-1)x9 = -25
    // output[1, 0]: 0x4  + 1x5      = 5,    2x4  + 3x5      = 23
    // output[1, 1]: 0x6  + 1x7      = 7,    2x6  + 3x7      = 33
    // output[1, 2]: 0x8  + 1x9      = 9,    2x8  + 3x9      = 43
    expectTensorValues(
        tensorC,
        {-31.0f, -13.0f, -45.0f, -19.0f, -59.0f, -25.0f, 5.0f, 23.0f, 7.0f, 33.0f, 9.0f, 43.0f});
}

/* ====================================================================== */
