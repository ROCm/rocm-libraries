// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/SdpaFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_data_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_data_sdk::data_objects::NodeAttributes;
using DiagonalAlignmentSdk = hipdnn_data_sdk::data_objects::DiagonalAlignment;
using AttentionImplementationSdk = hipdnn_data_sdk::data_objects::AttentionImplementation;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationSdpaFpropDescriptorLowering : public ::testing::Test
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

    hipdnnHandle_t _handle = nullptr;
};

// Builds an SDPA graph via the frontend API, lowers it to the backend
// via build_operation_graph_via_descriptors, retrieves the serialized graph,
// and verifies all tensor and operation attributes match the values set
// in the frontend.
TEST_F(IntegrationSdpaFpropDescriptorLowering, SdpaFpropGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestSdpaFpropGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(K_SDPA_TENSOR_Q_UID).set_name("Q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_SDPA_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(K_SDPA_TENSOR_K_UID).set_name("K").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_SDPA_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(K_SDPA_TENSOR_V_UID).set_name("V").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_SDPA_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_TENSOR_V_STRIDES));

    SdpaAttributes sdpaAttrs;
    sdpaAttrs.set_name("sdpa_fprop_op");

    auto [o, stats] = graph->sdpa(q, k, v, std::move(sdpaAttrs));
    o->set_uid(K_SDPA_TENSOR_O_UID).set_output(true).set_name("O");

    // -- Validate and lower --
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // -- Retrieve serialized graph --
    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    // -- Deserialize into GraphT --
    auto graphFb = hipdnn_data_sdk::data_objects::GetGraph(serializedData.data());
    ASSERT_NE(graphFb, nullptr);
    hipdnn_data_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors (Q, K, V, O = 4 tensors) --
    ASSERT_EQ(graphT.tensors.size(), 4u);

    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    // Verify Q tensor
    ASSERT_NE(tensorMap.count(K_SDPA_TENSOR_Q_UID), 0u);
    auto* qT = tensorMap[K_SDPA_TENSOR_Q_UID];
    EXPECT_EQ(qT->name, "Q");
    EXPECT_EQ(qT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(qT->dims, toVec(K_SDPA_TENSOR_Q_DIMS));
    EXPECT_EQ(qT->strides, toVec(K_SDPA_TENSOR_Q_STRIDES));
    EXPECT_FALSE(qT->virtual_);

    // Verify K tensor
    ASSERT_NE(tensorMap.count(K_SDPA_TENSOR_K_UID), 0u);
    auto* kT = tensorMap[K_SDPA_TENSOR_K_UID];
    EXPECT_EQ(kT->name, "K");
    EXPECT_EQ(kT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(kT->dims, toVec(K_SDPA_TENSOR_K_DIMS));
    EXPECT_EQ(kT->strides, toVec(K_SDPA_TENSOR_K_STRIDES));
    EXPECT_FALSE(kT->virtual_);

    // Verify V tensor
    ASSERT_NE(tensorMap.count(K_SDPA_TENSOR_V_UID), 0u);
    auto* vT = tensorMap[K_SDPA_TENSOR_V_UID];
    EXPECT_EQ(vT->name, "V");
    EXPECT_EQ(vT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(vT->dims, toVec(K_SDPA_TENSOR_V_DIMS));
    EXPECT_EQ(vT->strides, toVec(K_SDPA_TENSOR_V_STRIDES));
    EXPECT_FALSE(vT->virtual_);

    // Verify O tensor
    ASSERT_NE(tensorMap.count(K_SDPA_TENSOR_O_UID), 0u);
    auto* oT = tensorMap[K_SDPA_TENSOR_O_UID];
    EXPECT_EQ(oT->name, "O");
    EXPECT_EQ(oT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(oT->dims, toVec(K_SDPA_TENSOR_O_DIMS));
    EXPECT_EQ(oT->strides, toVec(K_SDPA_TENSOR_O_STRIDES));
    EXPECT_FALSE(oT->virtual_);

    // -- Verify SDPA operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::SdpaAttributes);

    auto* sdpa = node->attributes.AsSdpaAttributes();
    ASSERT_NE(sdpa, nullptr);

    EXPECT_EQ(sdpa->q_tensor_uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(sdpa->k_tensor_uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(sdpa->v_tensor_uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(sdpa->o_tensor_uid, K_SDPA_TENSOR_O_UID);

    // Verify default enum values
    EXPECT_EQ(sdpa->diagonal_alignment, DiagonalAlignmentSdk::TOP_LEFT);
    EXPECT_EQ(sdpa->implementation, AttentionImplementationSdk::AUTO);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationSdpaFpropDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidSdpaGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_name("Q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_SDPA_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_name("K").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_SDPA_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_name("V").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_SDPA_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_TENSOR_V_STRIDES));

    SdpaAttributes sdpaAttrs;

    auto [o, stats] = graph->sdpa(q, k, v, std::move(sdpaAttrs));
    o->set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve serialized graph
    auto rawDesc = graph->get_raw_graph_descriptor();
    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    // All tensors should have been auto-assigned unique UIDs (Q, K, V, O = 4)
    ASSERT_EQ(graphT.tensors.size(), 4u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 4u) << "Tensor UIDs are not unique";

    // The SDPA operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* sdpa = graphT.nodes[0]->attributes.AsSdpaAttributes();
    ASSERT_NE(sdpa, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(sdpa->q_tensor_uid) > 0)
        << "Q tensor UID " << sdpa->q_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->k_tensor_uid) > 0)
        << "K tensor UID " << sdpa->k_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->v_tensor_uid) > 0)
        << "V tensor UID " << sdpa->v_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(sdpa->o_tensor_uid) > 0)
        << "O tensor UID " << sdpa->o_tensor_uid << " not found in graph tensors";

    // All four required tensor UIDs referenced by the node should be distinct
    std::unordered_set<int64_t> nodeUids
        = {sdpa->q_tensor_uid, sdpa->k_tensor_uid, sdpa->v_tensor_uid, sdpa->o_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 4u) << "SDPA node tensor UIDs are not distinct";
}

// Verifies that an SDPA graph with stats output generates and preserves
// the stats tensor through the round-trip.
TEST_F(IntegrationSdpaFpropDescriptorLowering, SdpaFpropWithStatsRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestSdpaFpropWithStatsGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(K_SDPA_TENSOR_Q_UID).set_name("Q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_SDPA_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(K_SDPA_TENSOR_K_UID).set_name("K").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_SDPA_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(K_SDPA_TENSOR_V_UID).set_name("V").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_SDPA_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_TENSOR_V_STRIDES));

    SdpaAttributes sdpaAttrs;
    sdpaAttrs.set_name("sdpa_with_stats");
    sdpaAttrs.generate_stats = true;

    auto [o, stats] = graph->sdpa(q, k, v, std::move(sdpaAttrs));
    o->set_uid(K_SDPA_TENSOR_O_UID).set_output(true).set_name("O");
    ASSERT_NE(stats, nullptr) << "Stats tensor should be created when generate_stats is true";
    stats->set_uid(K_SDPA_TENSOR_STATS_UID).set_output(true).set_name("STATS");

    // -- Validate and lower --
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // -- Retrieve serialized graph --
    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    // -- Deserialize into GraphT --
    auto graphFb = hipdnn_data_sdk::data_objects::GetGraph(serializedData.data());
    ASSERT_NE(graphFb, nullptr);
    hipdnn_data_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);

    // -- Verify tensors (Q, K, V, O, STATS = 5 tensors) --
    ASSERT_EQ(graphT.tensors.size(), 5u);

    // -- Verify SDPA operation node attributes --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* sdpa = graphT.nodes[0]->attributes.AsSdpaAttributes();
    ASSERT_NE(sdpa, nullptr);

    EXPECT_EQ(sdpa->q_tensor_uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(sdpa->k_tensor_uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(sdpa->v_tensor_uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(sdpa->o_tensor_uid, K_SDPA_TENSOR_O_UID);
    EXPECT_EQ(sdpa->stats_tensor_uid, K_SDPA_TENSOR_STATS_UID);
    EXPECT_TRUE(sdpa->generate_stats);
}

} // namespace
