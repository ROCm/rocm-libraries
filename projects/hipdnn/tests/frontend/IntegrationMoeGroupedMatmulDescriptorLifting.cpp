// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/MoeGroupedMatmulNode.hpp>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;
using hipdnn_tests::toVec;

namespace
{

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationMoeGroupedMatmulDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard MoeGroupedMatmul graph for round-trip testing.
    static MoeGroupedMatmulAttributes createAttributes(DataType operationComputeDataType
                                                       = DataType::NOT_SET)
    {
        MoeGroupedMatmulAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_mode(MoeGroupedMatmulMode::NONE);
        attrs.set_top_k(0);
        if(operationComputeDataType != DataType::NOT_SET)
        {
            attrs.set_compute_data_type(operationComputeDataType);
        }
        return attrs;
    }

    static std::shared_ptr<TestableGraphLifting> buildGraph(MoeGroupedMatmulAttributes attrs,
                                                            bool includeTokenIndex = false,
                                                            bool includeTokenKs = false)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("MoeGroupedMatmulLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto token = std::make_shared<TensorAttributes>();
        token->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID)
            .set_name("token")
            .set_data_type(DataType::FLOAT);
        token->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES));

        auto weight = std::make_shared<TensorAttributes>();
        weight->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID)
            .set_name("weight")
            .set_data_type(DataType::FLOAT);
        weight->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES));

        auto firstTokenOffset = std::make_shared<TensorAttributes>();
        firstTokenOffset->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID)
            .set_name("first_token_offset")
            .set_data_type(DataType::INT32);
        firstTokenOffset->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));

        std::shared_ptr<TensorAttributes> tokenIndex;
        if(includeTokenIndex)
        {
            tokenIndex = std::make_shared<TensorAttributes>();
            tokenIndex->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID)
                .set_name("token_index")
                .set_data_type(DataType::INT32);
            tokenIndex->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS))
                .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES));
        }

        std::shared_ptr<TensorAttributes> tokenKs;
        if(includeTokenKs)
        {
            tokenKs = std::make_shared<TensorAttributes>();
            tokenKs->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID)
                .set_name("token_ks")
                .set_data_type(DataType::INT32);
            tokenKs->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS))
                .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES));
        }

        auto output = graph->moe_grouped_matmul(
            token, weight, firstTokenOffset, tokenIndex, tokenKs, std::move(attrs));
        output->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID).set_output(true).set_name("output");

        return graph;
    }
};

// Builds a standard MoeGroupedMatmul graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLifting, BasicMoeGroupedMatmulRoundTrip)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

    // Verify token tensor
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->get_uid(),
              K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->get_name(), "token");

    // Verify weight tensor
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->get_uid(),
              K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->get_name(), "weight");

    // Verify first_token_offset tensor
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_uid(),
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_data_type(),
              DataType::INT32);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_name(),
              "first_token_offset");

    // Verify output tensor
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->get_uid(),
              K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->get_name(), "output");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a MoeGroupedMatmulNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_mode(), MoeGroupedMatmulMode::NONE);

    EXPECT_EQ(opNode->attributes.get_top_k(), 0);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// Verifies an operation-level compute type survives descriptor lifting.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLifting, OperationComputeDataTypeSurvivesLifting)
{
    auto originalGraph = buildGraph(createAttributes(DataType::HALF));

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_compute_data_type(), DataType::HALF);
}

// Verifies NONE mode survives lowering and descriptor lifting.
// Mode-irrelevant frontend attributes are intentionally canonicalized away,
// matching cuDNN's mode-gated descriptor packing.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLifting, ModeScenarioNoneCanonicalizesIgnoredRouting)
{
    auto attrs = createAttributes();
    attrs.set_name("test_none_canonicalizes_ignored_routing");
    attrs.set_mode(MoeGroupedMatmulMode::NONE);
    attrs.set_top_k(2);

    auto originalGraph = buildGraph(std::move(attrs), true, true);

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    const auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_mode(), MoeGroupedMatmulMode::NONE);
    EXPECT_EQ(opNode->attributes.get_token_index(), nullptr);
    EXPECT_EQ(opNode->attributes.get_token_ks(), nullptr);
    EXPECT_EQ(opNode->attributes.get_top_k(), 0);
}
// Verifies GATHER mode survives lowering and descriptor lifting.
// Mode-irrelevant frontend attributes are intentionally canonicalized away,
// matching cuDNN's mode-gated descriptor packing.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLifting,
       ModeScenarioGatherCanonicalizesIgnoredScatterAttributes)
{
    auto attrs = createAttributes();
    attrs.set_name("test_gather_canonicalizes_ignored_scatter_attributes");
    attrs.set_mode(MoeGroupedMatmulMode::GATHER);
    attrs.set_top_k(2);

    auto originalGraph = buildGraph(std::move(attrs), true, true);

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    const auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 5u);
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_mode(), MoeGroupedMatmulMode::GATHER);
    ASSERT_NE(opNode->attributes.get_token_index(), nullptr);
    EXPECT_EQ(opNode->attributes.get_token_index()->get_data_type(), DataType::INT32);
    EXPECT_EQ(tensorMap.at(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID).get(),
              opNode->attributes.get_token_index().get());
    EXPECT_EQ(opNode->attributes.get_token_ks(), nullptr);
    EXPECT_EQ(opNode->attributes.get_top_k(), 0);
}
// Verifies SCATTER mode survives lowering and descriptor lifting.
// Mode-irrelevant frontend attributes are intentionally canonicalized away,
// matching cuDNN's mode-gated descriptor packing.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLifting, ModeScenarioScatterPreservesRouting)
{
    auto attrs = createAttributes();
    attrs.set_name("test_scatter_preserves_routing");
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER);
    attrs.set_top_k(2);

    auto originalGraph = buildGraph(std::move(attrs), true, true);

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    const auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 6u);
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_mode(), MoeGroupedMatmulMode::SCATTER);
    ASSERT_NE(opNode->attributes.get_token_index(), nullptr);
    EXPECT_EQ(opNode->attributes.get_token_index()->get_data_type(), DataType::INT32);
    EXPECT_EQ(tensorMap.at(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID).get(),
              opNode->attributes.get_token_index().get());
    ASSERT_NE(opNode->attributes.get_token_ks(), nullptr);
    EXPECT_EQ(opNode->attributes.get_token_ks()->get_data_type(), DataType::INT32);
    EXPECT_EQ(tensorMap.at(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID).get(),
              opNode->attributes.get_token_ks().get());
    EXPECT_EQ(opNode->attributes.get_top_k(), 2);
}

// Exercises JSON serialization and deserialization for each mode, including
// canonicalization of mode-irrelevant frontend attributes.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLifting, JsonRoundTripsAllModeScenarios)
{
    {
        auto attrs = createAttributes();
        attrs.set_name("test_json_none_canonicalizes_ignored_routing");
        attrs.set_mode(MoeGroupedMatmulMode::NONE);
        attrs.set_top_k(2);

        auto originalGraph = buildGraph(std::move(attrs), true, true);

        auto result = originalGraph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        std::string jsonData;
        result = originalGraph->serialize(jsonData);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_FALSE(jsonData.empty());

        auto liftedGraph = std::make_shared<TestableGraphLifting>();
        result = liftedGraph->deserialize(_handle, jsonData);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const auto tensorMap = liftedGraph->getTensorsByUid();
        ASSERT_EQ(tensorMap.size(), 4u);
        auto& subNodes = liftedGraph->getSubNodes();
        ASSERT_EQ(subNodes.size(), 1u);
        auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
        ASSERT_NE(opNode, nullptr);
        EXPECT_EQ(opNode->attributes.get_mode(), MoeGroupedMatmulMode::NONE);
        EXPECT_EQ(opNode->attributes.get_token_index(), nullptr);
        EXPECT_EQ(opNode->attributes.get_token_ks(), nullptr);
        EXPECT_EQ(opNode->attributes.get_top_k(), 0);
    }
    {
        auto attrs = createAttributes();
        attrs.set_name("test_json_gather_canonicalizes_ignored_scatter_attributes");
        attrs.set_mode(MoeGroupedMatmulMode::GATHER);
        attrs.set_top_k(2);

        auto originalGraph = buildGraph(std::move(attrs), true, true);

        auto result = originalGraph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        std::string jsonData;
        result = originalGraph->serialize(jsonData);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_FALSE(jsonData.empty());

        auto liftedGraph = std::make_shared<TestableGraphLifting>();
        result = liftedGraph->deserialize(_handle, jsonData);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const auto tensorMap = liftedGraph->getTensorsByUid();
        ASSERT_EQ(tensorMap.size(), 5u);
        auto& subNodes = liftedGraph->getSubNodes();
        ASSERT_EQ(subNodes.size(), 1u);
        auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
        ASSERT_NE(opNode, nullptr);
        EXPECT_EQ(opNode->attributes.get_mode(), MoeGroupedMatmulMode::GATHER);
        ASSERT_NE(opNode->attributes.get_token_index(), nullptr);
        EXPECT_EQ(opNode->attributes.get_token_ks(), nullptr);
        EXPECT_EQ(opNode->attributes.get_top_k(), 0);
    }
    {
        auto attrs = createAttributes();
        attrs.set_name("test_json_scatter_preserves_routing");
        attrs.set_mode(MoeGroupedMatmulMode::SCATTER);
        attrs.set_top_k(2);

        auto originalGraph = buildGraph(std::move(attrs), true, true);

        auto result = originalGraph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        std::string jsonData;
        result = originalGraph->serialize(jsonData);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_FALSE(jsonData.empty());

        auto liftedGraph = std::make_shared<TestableGraphLifting>();
        result = liftedGraph->deserialize(_handle, jsonData);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const auto tensorMap = liftedGraph->getTensorsByUid();
        ASSERT_EQ(tensorMap.size(), 6u);
        auto& subNodes = liftedGraph->getSubNodes();
        ASSERT_EQ(subNodes.size(), 1u);
        auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
        ASSERT_NE(opNode, nullptr);
        EXPECT_EQ(opNode->attributes.get_mode(), MoeGroupedMatmulMode::SCATTER);
        ASSERT_NE(opNode->attributes.get_token_index(), nullptr);
        ASSERT_NE(opNode->attributes.get_token_ks(), nullptr);
        EXPECT_EQ(opNode->attributes.get_top_k(), 2);
    }
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationMoeGroupedMatmulDescriptorLifting, MoeGroupedMatmulTensorSharingPreserved)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify token tensor sharing
    EXPECT_EQ(opNode->attributes.get_token()->get_uid(), K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID].get(),
              opNode->attributes.get_token().get());
    // Verify weight tensor sharing
    EXPECT_EQ(opNode->attributes.get_weight()->get_uid(), K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID].get(),
              opNode->attributes.get_weight().get());
    // Verify first_token_offset tensor sharing
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_uid(),
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID].get(),
              opNode->attributes.get_first_token_offset().get());
    // Verify output tensor sharing
    EXPECT_EQ(opNode->attributes.get_output()->get_uid(), K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID].get(),
              opNode->attributes.get_output().get());
}

// Builds a MoeGroupedMatmul graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLifting, MoeGroupedMatmulLiftWithoutFinalization)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_mode(), MoeGroupedMatmulMode::NONE);

    EXPECT_EQ(opNode->attributes.get_top_k(), 0);
    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->get_name(), "token");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->get_name(), "weight");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_name(),
              "first_token_offset");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->get_name(), "output");
}

// Builds a MoeGroupedMatmul graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("MoeGroupedMatmulAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto token = std::make_shared<TensorAttributes>();
    token->set_name("token").set_data_type(DataType::FLOAT);
    token->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS))
        .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES));

    auto weight = std::make_shared<TensorAttributes>();
    weight->set_name("weight").set_data_type(DataType::FLOAT);
    weight->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS))
        .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES));

    auto firstTokenOffset = std::make_shared<TensorAttributes>();
    firstTokenOffset->set_name("first_token_offset").set_data_type(DataType::INT32);
    firstTokenOffset->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS))
        .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));

    const std::shared_ptr<TensorAttributes> tokenIndex;

    const std::shared_ptr<TensorAttributes> tokenKs;

    MoeGroupedMatmulAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_mode(MoeGroupedMatmulMode::NONE);
    attrs.set_top_k(0);

    auto output
        = graph->moe_grouped_matmul(token, weight, firstTokenOffset, tokenIndex, tokenKs, attrs);
    output->set_output(true).set_name("output");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

    // Verify all UIDs are positive and distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        EXPECT_GE(uid, 0)
            << "Auto-assigned UID should be non-negative"; // NOLINT(readability-implicit-bool-conversion)
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    ASSERT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "Found duplicate auto-assigned UIDs"; // NOLINT(readability-implicit-bool-conversion)

    // Verify sub-node tensor UIDs are distinct via the node attributes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<MoeGroupedMatmulNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_token(), nullptr);
    nodeUids.insert(opNode->attributes.get_token()->get_uid());
    ASSERT_NE(opNode->attributes.get_weight(), nullptr);
    nodeUids.insert(opNode->attributes.get_weight()->get_uid());
    ASSERT_NE(opNode->attributes.get_first_token_offset(), nullptr);
    nodeUids.insert(opNode->attributes.get_first_token_offset()->get_uid());
    ASSERT_NE(opNode->attributes.get_output(), nullptr);
    nodeUids.insert(opNode->attributes.get_output()->get_uid());
    ASSERT_EQ(nodeUids.size(), 4u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_token()->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS));
    EXPECT_EQ(opNode->attributes.get_token()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES));
    EXPECT_EQ(opNode->attributes.get_token()->get_name(), "token");
    EXPECT_EQ(opNode->attributes.get_weight()->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS));
    EXPECT_EQ(opNode->attributes.get_weight()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES));
    EXPECT_EQ(opNode->attributes.get_weight()->get_name(), "weight");
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_name(), "first_token_offset");
    EXPECT_EQ(opNode->attributes.get_output()->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS));
    EXPECT_EQ(opNode->attributes.get_output()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES));
    EXPECT_EQ(opNode->attributes.get_output()->get_name(), "output");
}

} // namespace
