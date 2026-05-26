// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuReferenceMatmul.hpp>
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

class TestCpuReferenceMatmul : public ::testing::Test
{
};

TEST_F(TestCpuReferenceMatmul, IsApplicable)
{
    {
        const std::vector<int64_t> aDims = {2, 2, 3};
        const std::vector<int64_t> bDims = {2, 3, 4};
        const std::vector<int64_t> cDims = {2, 2, 4};

        MatmulTensorBundle<float> tensorBundle(aDims, bDims, cDims, false, false, 1);
        auto graphTuple = buildMatmulGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

        auto& graph = std::get<0>(graphTuple);
        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());
        EXPECT_TRUE(CpuReferenceMatmul::isApplicable(graphWrap.getNode(0)));
    }

    {
        const std::vector<int64_t> dims = {1, 3, 4, 4};

        auto graphTuple = buildPointwiseUnaryGraph(
            dims, dims,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT,
            hipdnn_frontend::PointwiseMode::RELU_FWD,
            1,
            TensorLayout::NCHW);

        auto& graph = std::get<0>(graphTuple);
        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());
        EXPECT_FALSE(CpuReferenceMatmul::isApplicable(graphWrap.getNode(0)));
    }
}

TEST_F(TestCpuReferenceMatmul, ValidateInput)
{
    {
        auto tensorA = createTensor<float>({2, 2, 3});
        auto tensorB = createTensor<float>({3, 4});
        auto tensorC = createTensor<float>({2, 2, 4});

        EXPECT_THROW((CpuReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
                     std::invalid_argument);
    }

    {
        auto tensorA = createTensor<float>({3});
        auto tensorB = createTensor<float>({3});
        auto tensorC = createTensor<float>({3});

        EXPECT_THROW((CpuReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
                     std::invalid_argument);
    }

    {
        auto tensorA = createTensor<float>({2, 2, 5});
        auto tensorB = createTensor<float>({2, 2, 5});
        auto tensorC = createTensor<float>({2, 2, 2});

        EXPECT_THROW((CpuReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
                     std::invalid_argument);
    }

    {
        auto tensorA = createTensor<float>({2, 3, 5});
        auto tensorB = createTensor<float>({2, 5, 4});
        auto tensorC = createTensor<float>({2, 4, 3});

        EXPECT_THROW((CpuReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
                     std::invalid_argument);
    }
}

TEST_F(TestCpuReferenceMatmul, ValidateBroadcastableBatchDims)
{
    {
        auto tensorA = createTensor<float>({2, 1, 2, 3});
        auto tensorB = createTensor<float>({1, 2, 3, 4});
        auto tensorC = createTensor<float>({2, 2, 2, 4});

        EXPECT_NO_THROW((CpuReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)));
    }

    {
        auto tensorA = createTensor<float>({2, 2, 2, 3});
        auto tensorB = createTensor<float>({3, 2, 3, 4});
        auto tensorC = createTensor<float>({2, 2, 3, 4});

        EXPECT_THROW((CpuReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
                     std::invalid_argument);
    }

    {
        auto tensorA = createTensor<float>({2, 1, 2, 3});
        auto tensorB = createTensor<float>({1, 2, 3, 4});
        auto tensorC = createTensor<float>({1, 1, 2, 4});

        EXPECT_THROW((CpuReferenceMatmul::matmul<float, float, float, float>(tensorA, tensorB, tensorC)),
                     std::invalid_argument);
    }
}

/* ============================= Func tests ============================= */

template <typename T1, typename T2, typename T3, typename T4>
struct TypePair
{
    using ADataType = T1;
    using BDataType = T2;
    using CDataType = T3;
    using ComputeDataType = T4;
};

using Types = ::testing::Types<
    TypePair<float, float, float, float>,
    TypePair<half, half, half, float>,
    TypePair<bfloat16, bfloat16, bfloat16, float>,
    TypePair<float, half, float, float>>;

template <class T>
class CpuReferenceMatmulBasic : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuReferenceMatmulBasic, Types, );

TYPED_TEST(CpuReferenceMatmulBasic, Matmul)
{
    auto tensorA = createTensor<typename TypeParam::ADataType>({2, 3});
    auto tensorB = createTensor<typename TypeParam::BDataType>({3, 2});
    auto tensorC = createTensor<typename TypeParam::CDataType>({2, 2});

    for(int i = 0; i < static_cast<int>(tensorA.elementCount()); ++i)
        tensorA.memory().hostData()[i] =
            safeTestTypeCast<typename TypeParam::ADataType>(static_cast<float>(i + 1));

    for(int i = 0; i < static_cast<int>(tensorB.elementCount()); ++i)
        tensorB.memory().hostData()[i] =
            safeTestTypeCast<typename TypeParam::BDataType>(static_cast<float>(static_cast<size_t>(i) + tensorA.elementCount()));

    CpuReferenceMatmul::matmul<typename TypeParam::ADataType,
                               typename TypeParam::BDataType,
                               typename TypeParam::CDataType,
                               typename TypeParam::ComputeDataType>(tensorA, tensorB, tensorC);

    expectTensorValues(tensorC, {52.0f, 58.0f, 124.0f, 139.0f});
}