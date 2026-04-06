// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>
#include <hipdnn_test_sdk/constants/MatmulConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph;
    using Graph::deserialize_via_backend;
    using Graph::fromBackendDescriptor;
    using Graph::get_raw_graph_descriptor;

    const std::vector<std::shared_ptr<INode>>& getSubNodes() const
    {
        return _sub_nodes;
    }
};

class IntegrationMatmulDescriptorLifting : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnDestroy(_handle);
        }
    }

    // Builds a standard matmul graph for round-trip testing
    static std::shared_ptr<TestableGraph> buildMatmulGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("MatmulLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto a = std::make_shared<TensorAttributes>();
        a->set_uid(K_MATMUL_TENSOR_A_UID).set_name("A").set_data_type(DataType::FLOAT);
        a->set_dim(toVec(K_MATMUL_TENSOR_A_DIMS)).set_stride(toVec(K_MATMUL_TENSOR_A_STRIDES));

        auto b = std::make_shared<TensorAttributes>();
        b->set_uid(K_MATMUL_TENSOR_B_UID).set_name("B").set_data_type(DataType::FLOAT);
        b->set_dim(toVec(K_MATMUL_TENSOR_B_DIMS)).set_stride(toVec(K_MATMUL_TENSOR_B_STRIDES));

        MatmulAttributes matmulAttrs;
        matmulAttrs.set_name("matmul_op");

        auto c = graph->matmul(a, b, matmulAttrs);
        c->set_uid(K_MATMUL_TENSOR_C_UID).set_output(true).set_name("C");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard matmul graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationMatmulDescriptorLifting, BasicMatmulRoundTrip)
{
    auto originalGraph = buildMatmulGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors (A, B, C) in lifted graph";

    // Verify A tensor
    ASSERT_NE(tensorMap.count(K_MATMUL_TENSOR_A_UID), 0u);
    auto liftedA = tensorMap[K_MATMUL_TENSOR_A_UID];
    EXPECT_EQ(liftedA->get_uid(), K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(liftedA->get_name(), "A");
    EXPECT_EQ(liftedA->get_dim(), toVec(K_MATMUL_TENSOR_A_DIMS));
    EXPECT_EQ(liftedA->get_stride(), toVec(K_MATMUL_TENSOR_A_STRIDES));
    EXPECT_EQ(liftedA->get_data_type(), DataType::FLOAT);

    // Verify B tensor
    ASSERT_NE(tensorMap.count(K_MATMUL_TENSOR_B_UID), 0u);
    auto liftedB = tensorMap[K_MATMUL_TENSOR_B_UID];
    EXPECT_EQ(liftedB->get_uid(), K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(liftedB->get_name(), "B");
    EXPECT_EQ(liftedB->get_dim(), toVec(K_MATMUL_TENSOR_B_DIMS));
    EXPECT_EQ(liftedB->get_stride(), toVec(K_MATMUL_TENSOR_B_STRIDES));
    EXPECT_EQ(liftedB->get_data_type(), DataType::FLOAT);

    // Verify C tensor
    ASSERT_NE(tensorMap.count(K_MATMUL_TENSOR_C_UID), 0u);
    auto liftedC = tensorMap[K_MATMUL_TENSOR_C_UID];
    EXPECT_EQ(liftedC->get_uid(), K_MATMUL_TENSOR_C_UID);
    EXPECT_EQ(liftedC->get_name(), "C");
    EXPECT_EQ(liftedC->get_dim(), toVec(K_MATMUL_TENSOR_C_DIMS));
    EXPECT_EQ(liftedC->get_stride(), toVec(K_MATMUL_TENSOR_C_STRIDES));
    EXPECT_EQ(liftedC->get_data_type(), DataType::FLOAT);

    // Verify 1 sub-node of the correct type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* matmulNode = dynamic_cast<MatmulNode*>(subNodes[0].get());
    ASSERT_NE(matmulNode, nullptr) << "Expected a MatmulNode";

    // Verify operation name
    EXPECT_EQ(matmulNode->attributes.get_name(), "matmul_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationMatmulDescriptorLifting, MatmulTensorSharingPreserved)
{
    auto originalGraph = buildMatmulGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* matmulNode = dynamic_cast<MatmulNode*>(subNodes[0].get());
    ASSERT_NE(matmulNode, nullptr);

    // Verify UIDs match
    EXPECT_EQ(matmulNode->attributes.get_a()->get_uid(), K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(matmulNode->attributes.get_b()->get_uid(), K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(matmulNode->attributes.get_c()->get_uid(), K_MATMUL_TENSOR_C_UID);

    // Verify tensor names
    EXPECT_EQ(matmulNode->attributes.get_a()->get_name(), "A");
    EXPECT_EQ(matmulNode->attributes.get_b()->get_name(), "B");
    EXPECT_EQ(matmulNode->attributes.get_c()->get_name(), "C");

    // Verify pointer equality: tensor map and node attributes share the same objects
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_A_UID].get(), matmulNode->attributes.get_a().get());
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_B_UID].get(), matmulNode->attributes.get_b().get());
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_C_UID].get(), matmulNode->attributes.get_c().get());
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the round trip and are all distinct.
TEST_F(IntegrationMatmulDescriptorLifting, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidMatmulLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto a = std::make_shared<TensorAttributes>();
    a->set_name("A").set_data_type(DataType::FLOAT);
    a->set_dim(toVec(K_MATMUL_TENSOR_A_DIMS)).set_stride(toVec(K_MATMUL_TENSOR_A_STRIDES));

    auto b = std::make_shared<TensorAttributes>();
    b->set_name("B").set_data_type(DataType::FLOAT);
    b->set_dim(toVec(K_MATMUL_TENSOR_B_DIMS)).set_stride(toVec(K_MATMUL_TENSOR_B_STRIDES));

    MatmulAttributes matmulAttrs;
    matmulAttrs.set_name("auto_uid_matmul_op");

    auto c = graph->matmul(a, b, matmulAttrs);
    c->set_output(true).set_name("C");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors in lifted graph";

    // Collect all UIDs and verify they are distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    EXPECT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "All auto-assigned UIDs must be distinct";

    // Verify the node references tensors with auto-assigned UIDs
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* matmulNode = dynamic_cast<MatmulNode*>(subNodes[0].get());
    ASSERT_NE(matmulNode, nullptr);

    // Verify tensor dims survived the round trip
    auto aUid = matmulNode->attributes.get_a()->get_uid();
    auto bUid = matmulNode->attributes.get_b()->get_uid();
    auto cUid = matmulNode->attributes.get_c()->get_uid();

    EXPECT_NE(aUid, bUid);
    EXPECT_NE(aUid, cUid);
    EXPECT_NE(bUid, cUid);

    EXPECT_EQ(tensorMap[aUid]->get_dim(), toVec(K_MATMUL_TENSOR_A_DIMS));
    EXPECT_EQ(tensorMap[bUid]->get_dim(), toVec(K_MATMUL_TENSOR_B_DIMS));
}

// Builds a matmul graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationMatmulDescriptorLifting, MatmulLiftWithoutFinalization)
{
    auto originalGraph = buildMatmulGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to binary via the frontend
    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create a backend graph descriptor from serialized bytes (no handle, no finalize)
    const detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    ASSERT_TRUE(graphDesc.valid()) << "Failed to create backend graph descriptor";

    // Lift into a new graph via fromBackendDescriptor
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(graphDesc.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* matmulNode = dynamic_cast<MatmulNode*>(subNodes[0].get());
    ASSERT_NE(matmulNode, nullptr);

    // Verify operation name
    EXPECT_EQ(matmulNode->attributes.get_name(), "matmul_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_MATMUL_TENSOR_A_UID), 0u);
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_A_UID]->get_dim(), toVec(K_MATMUL_TENSOR_A_DIMS));
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_A_UID]->get_stride(), toVec(K_MATMUL_TENSOR_A_STRIDES));
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_A_UID]->get_name(), "A");

    ASSERT_NE(tensorMap.count(K_MATMUL_TENSOR_B_UID), 0u);
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_B_UID]->get_dim(), toVec(K_MATMUL_TENSOR_B_DIMS));
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_B_UID]->get_stride(), toVec(K_MATMUL_TENSOR_B_STRIDES));
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_B_UID]->get_name(), "B");

    ASSERT_NE(tensorMap.count(K_MATMUL_TENSOR_C_UID), 0u);
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_C_UID]->get_dim(), toVec(K_MATMUL_TENSOR_C_DIMS));
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_C_UID]->get_stride(), toVec(K_MATMUL_TENSOR_C_STRIDES));
    EXPECT_EQ(tensorMap[K_MATMUL_TENSOR_C_UID]->get_name(), "C");
}

} // namespace
