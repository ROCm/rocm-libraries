// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/SdpaFwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using DiagonalAlignmentSdk = hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
using AttentionImplementationSdk = hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

// Builds an SDPA graph with standard Q/K/V tensors, validates, lowers via
// descriptors, serializes, and returns the deserialized GraphT.
// The caller configures optional tensors and scalars on sdpaAttrs before calling.
hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserializeSdpaGraph(hipdnnHandle_t handle,
                                                                          SdpaAttributes sdpaAttrs,
                                                                          DataType tensorDataType
                                                                          = DataType::FLOAT)
{
    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;

    const bool wantStats = sdpaAttrs.generate_stats.has_value() && sdpaAttrs.generate_stats.value();

    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestGraph")
        .set_io_data_type(tensorDataType)
        .set_intermediate_data_type(tensorDataType)
        .set_compute_data_type(tensorDataType);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(K_SDPA_TENSOR_Q_UID).set_name("Q").set_data_type(tensorDataType);
    q->set_dim(toVec(K_SDPA_TENSOR_Q_DIMS)).set_stride(toVec(K_SDPA_TENSOR_Q_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(K_SDPA_TENSOR_K_UID).set_name("K").set_data_type(tensorDataType);
    k->set_dim(toVec(K_SDPA_TENSOR_K_DIMS)).set_stride(toVec(K_SDPA_TENSOR_K_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(K_SDPA_TENSOR_V_UID).set_name("V").set_data_type(tensorDataType);
    v->set_dim(toVec(K_SDPA_TENSOR_V_DIMS)).set_stride(toVec(K_SDPA_TENSOR_V_STRIDES));

    auto [o, stats] = graph->sdpa(q, k, v, std::move(sdpaAttrs));
    o->set_uid(K_SDPA_TENSOR_O_UID).set_output(true).set_name("O");

    if(wantStats && stats)
    {
        stats->set_uid(K_SDPA_TENSOR_STATS_UID).set_output(true).set_name("STATS");
    }

    auto result = graph->validate();
    if(result.code != ErrorCode::OK)
    {
        ADD_FAILURE() << "validate() failed: " << result.err_msg;
        return graphT;
    }

    result = graph->build_operation_graph_via_descriptors(handle);
    if(result.code != ErrorCode::OK)
    {
        ADD_FAILURE() << "build_operation_graph_via_descriptors() failed: " << result.err_msg;
        return graphT;
    }

    auto rawDesc = graph->get_raw_graph_descriptor();
    if(rawDesc == nullptr)
    {
        ADD_FAILURE() << "get_raw_graph_descriptor() returned null";
        return graphT;
    }

    size_t serializedSize = 0;
    if(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr)
       != HIPDNN_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "Failed to get serialized size";
        return graphT;
    }

    std::vector<uint8_t> serializedData(serializedSize);
    if(hipdnnBackendGetSerializedBinaryGraph_ext(
           rawDesc, serializedSize, &serializedSize, serializedData.data())
       != HIPDNN_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "Failed to serialize graph";
        return graphT;
    }

    auto graphFb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(serializedData.data());
    if(graphFb == nullptr)
    {
        ADD_FAILURE() << "GetGraph returned null";
        return graphT;
    }
    graphFb->UnPackTo(&graphT);

    return graphT;
}

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationSdpaFwdDescriptorLowering : public ::testing::Test
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
TEST_F(IntegrationSdpaFwdDescriptorLowering, SdpaFwdGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestSdpaFwdGraph")
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
    sdpaAttrs.set_name("sdpa_fwd_op");

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
    auto graphFb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(serializedData.data());
    ASSERT_NE(graphFb, nullptr);
    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors (Q, K, V, O = 4 tensors) --
    ASSERT_EQ(graphT.tensors.size(), 4u);

    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT*>
        tensorMap;
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
TEST_F(IntegrationSdpaFwdDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
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

    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
    hipdnn_flatbuffers_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

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
    const std::unordered_set<int64_t> nodeUids
        = {sdpa->q_tensor_uid, sdpa->k_tensor_uid, sdpa->v_tensor_uid, sdpa->o_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 4u) << "SDPA node tensor UIDs are not distinct";
}

// Verifies that an SDPA graph with stats output generates and preserves
// the stats tensor through the round-trip.
TEST_F(IntegrationSdpaFwdDescriptorLowering, SdpaFwdWithStatsRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestSdpaFwdWithStatsGraph")
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
    auto graphFb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(serializedData.data());
    ASSERT_NE(graphFb, nullptr);
    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
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
    ASSERT_TRUE(sdpa->stats_tensor_uid.has_value());
    EXPECT_EQ(sdpa->stats_tensor_uid.value(), K_SDPA_TENSOR_STATS_UID);
    EXPECT_TRUE(sdpa->generate_stats);

    // Verify inferred stats tensor shape
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT*>
        tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }
    ASSERT_NE(tensorMap.count(K_SDPA_TENSOR_STATS_UID), 0u);
    auto* statsT = tensorMap[K_SDPA_TENSOR_STATS_UID];
    EXPECT_EQ(statsT->dims, toVec(K_SDPA_TENSOR_STATS_DIMS));
    EXPECT_EQ(statsT->data_type, DataTypeSdk::FLOAT);
}

// Exercises packer code paths for optional input tensors (attn_mask, seed,
// offset, dropout_mask), stats output, non-default booleans, optional float
// and int64 scalars, and non-default enum values.
TEST_F(IntegrationSdpaFwdDescriptorLowering, SdpaFwdWithOptionalTensorsAndScalars)
{
    auto attnMask = std::make_shared<TensorAttributes>();
    attnMask->set_uid(K_SDPA_TENSOR_ATTN_MASK_UID).set_name("ATTN_MASK");
    attnMask->set_data_type(DataType::FLOAT);
    attnMask->set_dim(toVec(K_SDPA_TENSOR_ATTN_MASK_DIMS));
    attnMask->set_stride(toVec(K_SDPA_TENSOR_ATTN_MASK_STRIDES));

    auto seed = std::make_shared<TensorAttributes>();
    seed->set_uid(K_SDPA_TENSOR_SEED_UID).set_name("SEED").set_data_type(DataType::FLOAT);
    seed->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    seed->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    auto offset = std::make_shared<TensorAttributes>();
    offset->set_uid(K_SDPA_TENSOR_OFFSET_UID).set_name("OFFSET").set_data_type(DataType::FLOAT);
    offset->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    offset->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    auto dropoutMask = std::make_shared<TensorAttributes>();
    dropoutMask->set_uid(K_SDPA_TENSOR_DROPOUT_MASK_UID).set_name("DROPOUT_MASK");
    dropoutMask->set_data_type(DataType::FLOAT);
    dropoutMask->set_dim(toVec(K_SDPA_TENSOR_ATTN_MASK_DIMS));
    dropoutMask->set_stride(toVec(K_SDPA_TENSOR_ATTN_MASK_STRIDES));

    SdpaAttributes sdpaAttrs;
    sdpaAttrs.set_name("sdpa_with_optionals");
    sdpaAttrs.set_bias(attnMask);
    sdpaAttrs.set_dropout(0.1f, seed, offset);
    sdpaAttrs.set_dropout_mask(dropoutMask);
    sdpaAttrs.set_generate_stats(true);
    sdpaAttrs.set_causal_mask(true);
    sdpaAttrs.set_attn_scale_value(0.125f);
    sdpaAttrs.set_diagonal_band_left_bound(0);
    sdpaAttrs.set_diagonal_band_right_bound(128);
    sdpaAttrs.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
    sdpaAttrs.set_implementation(AttentionImplementation::UNIFIED);

    auto graphT = buildAndDeserializeSdpaGraph(_handle, std::move(sdpaAttrs));
    if(HasFailure())
    {
        return;
    }

    // Q + K + V + O + STATS + attn_mask + seed + offset + dropout_mask = 9
    EXPECT_EQ(graphT.tensors.size(), 9u);

    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_NE(uids.count(K_SDPA_TENSOR_Q_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_K_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_V_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_O_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_STATS_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_ATTN_MASK_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_SEED_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_OFFSET_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_DROPOUT_MASK_UID), 0u);

    // -- Verify SDPA node attributes --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* sdpa = graphT.nodes[0]->attributes.AsSdpaAttributes();
    ASSERT_NE(sdpa, nullptr);

    // Required tensor UIDs
    EXPECT_EQ(sdpa->q_tensor_uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(sdpa->k_tensor_uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(sdpa->v_tensor_uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(sdpa->o_tensor_uid, K_SDPA_TENSOR_O_UID);

    // Optional tensor UIDs
    ASSERT_TRUE(sdpa->attn_mask_tensor_uid.has_value());
    EXPECT_EQ(sdpa->attn_mask_tensor_uid.value(), K_SDPA_TENSOR_ATTN_MASK_UID);

    ASSERT_TRUE(sdpa->seed_tensor_uid.has_value());
    EXPECT_EQ(sdpa->seed_tensor_uid.value(), K_SDPA_TENSOR_SEED_UID);

    ASSERT_TRUE(sdpa->offset_tensor_uid.has_value());
    EXPECT_EQ(sdpa->offset_tensor_uid.value(), K_SDPA_TENSOR_OFFSET_UID);

    ASSERT_TRUE(sdpa->dropout_mask_tensor_uid.has_value());
    EXPECT_EQ(sdpa->dropout_mask_tensor_uid.value(), K_SDPA_TENSOR_DROPOUT_MASK_UID);

    ASSERT_TRUE(sdpa->stats_tensor_uid.has_value());
    EXPECT_EQ(sdpa->stats_tensor_uid.value(), K_SDPA_TENSOR_STATS_UID);

    // Boolean attributes
    ASSERT_TRUE(sdpa->generate_stats.has_value());
    EXPECT_TRUE(sdpa->generate_stats.value());
    EXPECT_TRUE(sdpa->causal_mask);

    // Float scalars
    ASSERT_TRUE(sdpa->dropout_probability.has_value());
    EXPECT_FLOAT_EQ(sdpa->dropout_probability.value(), 0.1f);

    ASSERT_TRUE(sdpa->attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(sdpa->attn_scale_value.value(), 0.125f);

    // Integer scalars
    ASSERT_TRUE(sdpa->left_bound.has_value());
    EXPECT_EQ(sdpa->left_bound.value(), 0);

    ASSERT_TRUE(sdpa->right_bound.has_value());
    EXPECT_EQ(sdpa->right_bound.value(), 128);

    // Enum values
    EXPECT_EQ(sdpa->diagonal_alignment, DiagonalAlignmentSdk::BOTTOM_RIGHT);
    EXPECT_EQ(sdpa->implementation, AttentionImplementationSdk::UNIFIED);
}

// Exercises packer code paths for FP8 descale/scale tensors, amax output
// tensors, the optional int32 max_seq_len_kv scalar, and the conditional
// mma_core_mode enum branch.
TEST_F(IntegrationSdpaFwdDescriptorLowering, SdpaFwdWithFp8Tensors)
{
    auto descaleQ = std::make_shared<TensorAttributes>();
    descaleQ->set_uid(K_SDPA_TENSOR_DESCALE_Q_UID).set_name("DESCALE_Q");
    descaleQ->set_data_type(DataType::FLOAT);
    descaleQ->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    descaleQ->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    auto descaleK = std::make_shared<TensorAttributes>();
    descaleK->set_uid(K_SDPA_TENSOR_DESCALE_K_UID).set_name("DESCALE_K");
    descaleK->set_data_type(DataType::FLOAT);
    descaleK->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    descaleK->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    auto descaleV = std::make_shared<TensorAttributes>();
    descaleV->set_uid(K_SDPA_TENSOR_DESCALE_V_UID).set_name("DESCALE_V");
    descaleV->set_data_type(DataType::FLOAT);
    descaleV->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    descaleV->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    auto descaleS = std::make_shared<TensorAttributes>();
    descaleS->set_uid(K_SDPA_TENSOR_DESCALE_S_UID).set_name("DESCALE_S");
    descaleS->set_data_type(DataType::FLOAT);
    descaleS->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    descaleS->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    auto scaleS = std::make_shared<TensorAttributes>();
    scaleS->set_uid(K_SDPA_TENSOR_SCALE_S_UID).set_name("SCALE_S");
    scaleS->set_data_type(DataType::FLOAT);
    scaleS->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    scaleS->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    auto scaleO = std::make_shared<TensorAttributes>();
    scaleO->set_uid(K_SDPA_TENSOR_SCALE_O_UID).set_name("SCALE_O");
    scaleO->set_data_type(DataType::FLOAT);
    scaleO->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    scaleO->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    auto amaxS = std::make_shared<TensorAttributes>();
    amaxS->set_uid(K_SDPA_TENSOR_AMAX_S_UID).set_name("AMAX_S");
    amaxS->set_data_type(DataType::FLOAT).set_output(true);
    amaxS->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    amaxS->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    auto amaxO = std::make_shared<TensorAttributes>();
    amaxO->set_uid(K_SDPA_TENSOR_AMAX_O_UID).set_name("AMAX_O");
    amaxO->set_data_type(DataType::FLOAT).set_output(true);
    amaxO->set_dim(toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    amaxO->set_stride(toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    SdpaAttributes sdpaAttrs;
    sdpaAttrs.set_name("sdpa_fp8");
    sdpaAttrs.set_descale_q(descaleQ);
    sdpaAttrs.set_descale_k(descaleK);
    sdpaAttrs.set_descale_v(descaleV);
    sdpaAttrs.set_descale_s(descaleS);
    sdpaAttrs.set_scale_s(scaleS);
    sdpaAttrs.set_scale_o(scaleO);
    sdpaAttrs.set_amax_s(amaxS);
    sdpaAttrs.set_amax_o(amaxO);
    sdpaAttrs.set_paged_attention_max_seq_len_kv(256);
    sdpaAttrs.set_mma_core_mode(DataType::HALF);

    auto graphT = buildAndDeserializeSdpaGraph(_handle, std::move(sdpaAttrs), DataType::HALF);
    if(HasFailure())
    {
        return;
    }

    // Q + K + V + O + 6 descale/scale + 2 amax = 12
    EXPECT_EQ(graphT.tensors.size(), 12u);

    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_NE(uids.count(K_SDPA_TENSOR_Q_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_K_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_V_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_O_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_DESCALE_Q_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_DESCALE_K_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_DESCALE_V_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_DESCALE_S_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_SCALE_S_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_SCALE_O_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_AMAX_S_UID), 0u);
    EXPECT_NE(uids.count(K_SDPA_TENSOR_AMAX_O_UID), 0u);

    // -- Verify SDPA node attributes --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* sdpa = graphT.nodes[0]->attributes.AsSdpaAttributes();
    ASSERT_NE(sdpa, nullptr);

    // FP8 tensor UIDs in node attributes
    ASSERT_TRUE(sdpa->descale_q_tensor_uid.has_value());
    EXPECT_EQ(sdpa->descale_q_tensor_uid.value(), K_SDPA_TENSOR_DESCALE_Q_UID);

    ASSERT_TRUE(sdpa->descale_k_tensor_uid.has_value());
    EXPECT_EQ(sdpa->descale_k_tensor_uid.value(), K_SDPA_TENSOR_DESCALE_K_UID);

    ASSERT_TRUE(sdpa->descale_v_tensor_uid.has_value());
    EXPECT_EQ(sdpa->descale_v_tensor_uid.value(), K_SDPA_TENSOR_DESCALE_V_UID);

    ASSERT_TRUE(sdpa->descale_s_tensor_uid.has_value());
    EXPECT_EQ(sdpa->descale_s_tensor_uid.value(), K_SDPA_TENSOR_DESCALE_S_UID);

    ASSERT_TRUE(sdpa->scale_s_tensor_uid.has_value());
    EXPECT_EQ(sdpa->scale_s_tensor_uid.value(), K_SDPA_TENSOR_SCALE_S_UID);

    ASSERT_TRUE(sdpa->scale_o_tensor_uid.has_value());
    EXPECT_EQ(sdpa->scale_o_tensor_uid.value(), K_SDPA_TENSOR_SCALE_O_UID);

    ASSERT_TRUE(sdpa->amax_s_tensor_uid.has_value());
    EXPECT_EQ(sdpa->amax_s_tensor_uid.value(), K_SDPA_TENSOR_AMAX_S_UID);

    ASSERT_TRUE(sdpa->amax_o_tensor_uid.has_value());
    EXPECT_EQ(sdpa->amax_o_tensor_uid.value(), K_SDPA_TENSOR_AMAX_O_UID);

    // Optional int32 scalar
    ASSERT_TRUE(sdpa->max_seq_len_kv.has_value());
    EXPECT_EQ(sdpa->max_seq_len_kv.value(), 256);

    // MMA core mode
    EXPECT_EQ(sdpa->mma_core_mode, DataTypeSdk::HALF);

    // Verify defaults for fields not set
    EXPECT_EQ(sdpa->diagonal_alignment, DiagonalAlignmentSdk::TOP_LEFT);
    EXPECT_EQ(sdpa->implementation, AttentionImplementationSdk::AUTO);
    EXPECT_FALSE(sdpa->generate_stats.has_value());
    EXPECT_FALSE(sdpa->dropout_probability.has_value());
    EXPECT_FALSE(sdpa->attn_scale_value.has_value());
}

} // namespace
