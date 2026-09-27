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
#include <hipdnn_frontend/node/MoeGroupedMatmulBwdNode.hpp>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulBwdConstants.hpp>
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

// Selects which dweight properties buildGraph() supplies, so the remainder are left for
// infer_properties_node() to derive during lowering.
enum class DweightSetup
{
    ALL_SET,
    STRIDE_INFERRED,
    DIMS_AND_STRIDE_INFERRED,
};

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationMoeGroupedMatmulBwdDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard MoeGroupedMatmulBwd graph for round-trip testing.
    static MoeGroupedMatmulBwdAttributes createAttributes(DataType operationComputeDataType
                                                          = DataType::NOT_SET)
    {
        MoeGroupedMatmulBwdAttributes attrs;
        attrs.set_name("test_op");
        if(operationComputeDataType != DataType::NOT_SET)
        {
            attrs.set_compute_data_type(operationComputeDataType);
        }
        return attrs;
    }

    static std::shared_ptr<TestableGraphLifting> buildGraph(MoeGroupedMatmulBwdAttributes attrs,
                                                            DweightSetup dweightSetup
                                                            = DweightSetup::ALL_SET)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("MoeGroupedMatmulBwdLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dOutput = std::make_shared<TensorAttributes>();
        dOutput->set_uid(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID)
            .set_name("doutput")
            .set_data_type(DataType::FLOAT);
        dOutput->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES));

        auto token = std::make_shared<TensorAttributes>();
        token->set_uid(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID)
            .set_name("token")
            .set_data_type(DataType::FLOAT);
        token->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES));

        auto firstTokenOffset = std::make_shared<TensorAttributes>();
        firstTokenOffset->set_uid(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID)
            .set_name("first_token_offset")
            .set_data_type(DataType::INT32);
        firstTokenOffset->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));

        auto dweight
            = graph->moe_grouped_matmul_bwd(dOutput, token, firstTokenOffset, std::move(attrs));
        dweight->set_uid(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID)
            .set_output(true)
            .set_name("dweight");
        if(dweightSetup != DweightSetup::DIMS_AND_STRIDE_INFERRED)
        {
            dweight->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS));
        }
        if(dweightSetup == DweightSetup::ALL_SET)
        {
            dweight->set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));
        }

        return graph;
    }
};

// Builds a standard MoeGroupedMatmulBwd graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLifting, BasicMoeGroupedMatmulBwdRoundTrip)
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

    // Verify doutput tensor
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->get_uid(),
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->get_data_type(),
              DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->get_name(), "doutput");

    // Verify token tensor
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->get_uid(),
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->get_data_type(),
              DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->get_name(), "token");

    // Verify first_token_offset tensor
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_uid(),
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_data_type(),
              DataType::INT32);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_name(),
              "first_token_offset");

    // Verify dweight tensor
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_uid(),
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_data_type(),
              DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_name(), "dweight");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<MoeGroupedMatmulBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a MoeGroupedMatmulBwdNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // The operation-level compute type was left unset, so fill_from_context() takes the
    // graph-level FLOAT; verify that survives the round-trip.
    EXPECT_EQ(opNode->attributes.get_compute_data_type(), DataType::FLOAT);
}

// Verifies an operation-level compute type survives descriptor lifting.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLifting, OperationComputeDataTypeSurvivesLifting)
{
    auto originalGraph = buildGraph(createAttributes(DataType::HALF));

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<MoeGroupedMatmulBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_compute_data_type(), DataType::HALF);
}

// Exercises JSON serialization and deserialization. NONE mode is the only supported
// scenario for this op, so unlike the forward op's JsonRoundTripsAllModeScenarios there
// are no mode variants to loop over.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLifting, JsonRoundTripsBasicScenario)
{
    auto originalGraph = buildGraph(createAttributes());

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
    auto* opNode = dynamic_cast<MoeGroupedMatmulBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLifting, MoeGroupedMatmulBwdTensorSharingPreserved)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<MoeGroupedMatmulBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify doutput tensor sharing
    EXPECT_EQ(opNode->attributes.get_doutput()->get_uid(),
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID].get(),
              opNode->attributes.get_doutput().get());
    // Verify token tensor sharing
    EXPECT_EQ(opNode->attributes.get_token()->get_uid(), K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID].get(),
              opNode->attributes.get_token().get());
    // Verify first_token_offset tensor sharing
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_uid(),
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID].get(),
              opNode->attributes.get_first_token_offset().get());
    // Verify dweight tensor sharing
    EXPECT_EQ(opNode->attributes.get_dweight()->get_uid(),
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID].get(),
              opNode->attributes.get_dweight().get());
}

// Builds a MoeGroupedMatmulBwd graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLifting, MoeGroupedMatmulBwdLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<MoeGroupedMatmulBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->get_name(), "doutput");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->get_name(), "token");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_name(),
              "first_token_offset");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_name(), "dweight");
}

// Builds a MoeGroupedMatmulBwd graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("MoeGroupedMatmulBwdAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dOutput = std::make_shared<TensorAttributes>();
    dOutput->set_name("doutput").set_data_type(DataType::FLOAT);
    dOutput->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS))
        .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES));

    auto token = std::make_shared<TensorAttributes>();
    token->set_name("token").set_data_type(DataType::FLOAT);
    token->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS))
        .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES));

    auto firstTokenOffset = std::make_shared<TensorAttributes>();
    firstTokenOffset->set_name("first_token_offset").set_data_type(DataType::INT32);
    firstTokenOffset->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS))
        .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));

    MoeGroupedMatmulBwdAttributes attrs;
    attrs.set_name("test_auto_uid");

    auto dweight = graph->moe_grouped_matmul_bwd(dOutput, token, firstTokenOffset, attrs);
    dweight->set_output(true).set_name("dweight");
    dweight->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS))
        .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));

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

    auto* opNode = dynamic_cast<MoeGroupedMatmulBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_doutput(), nullptr);
    nodeUids.insert(opNode->attributes.get_doutput()->get_uid());
    ASSERT_NE(opNode->attributes.get_token(), nullptr);
    nodeUids.insert(opNode->attributes.get_token()->get_uid());
    ASSERT_NE(opNode->attributes.get_first_token_offset(), nullptr);
    nodeUids.insert(opNode->attributes.get_first_token_offset()->get_uid());
    ASSERT_NE(opNode->attributes.get_dweight(), nullptr);
    nodeUids.insert(opNode->attributes.get_dweight()->get_uid());
    ASSERT_EQ(nodeUids.size(), 4u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_doutput()->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS));
    EXPECT_EQ(opNode->attributes.get_doutput()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES));
    EXPECT_EQ(opNode->attributes.get_doutput()->get_name(), "doutput");
    EXPECT_EQ(opNode->attributes.get_token()->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS));
    EXPECT_EQ(opNode->attributes.get_token()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES));
    EXPECT_EQ(opNode->attributes.get_token()->get_name(), "token");
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_name(), "first_token_offset");
    EXPECT_EQ(opNode->attributes.get_dweight()->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS));
    EXPECT_EQ(opNode->attributes.get_dweight()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));
    EXPECT_EQ(opNode->attributes.get_dweight()->get_name(), "dweight");
}

// Builds a MoeGroupedMatmulBwd graph with dweight strides left unset so the full
// lowering pipeline exercises infer_properties_node()'s stride-inference branch, then
// verifies the column-major strides survive the lifting round-trip.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLifting, InferredDweightStridesSurviveRoundTrip)
{
    auto originalGraph = buildGraph(createAttributes(), DweightSetup::STRIDE_INFERRED);

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<MoeGroupedMatmulBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_dweight()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));
}

// Builds a MoeGroupedMatmulBwd graph with dweight dims and strides both left unset, so the
// full lowering pipeline derives dweight's shape from the input tensors as well as its
// strides, then verifies both survive the lifting round-trip.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLifting,
       InferredDweightDimsAndStridesSurviveRoundTrip)
{
    auto originalGraph = buildGraph(createAttributes(), DweightSetup::DIMS_AND_STRIDE_INFERRED);

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<MoeGroupedMatmulBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_dweight()->get_dim(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS));
    EXPECT_EQ(opNode->attributes.get_dweight()->get_stride(),
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));
}

} // namespace
