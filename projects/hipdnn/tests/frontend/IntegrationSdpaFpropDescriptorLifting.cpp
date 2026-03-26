// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/SdpaFpropNode.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;

namespace
{

// Exposes protected Graph methods for lifting integration tests
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph;
    using Graph::deserialize_via_backend;
    using Graph::get_raw_graph_descriptor;

    const std::vector<std::shared_ptr<INode>>& getSubNodes() const
    {
        return _sub_nodes;
    }
};

// -- Test constants --
constexpr int64_t K_TEST_Q_UID = 60;
constexpr int64_t K_TEST_K_UID = 61;
constexpr int64_t K_TEST_V_UID = 62;
constexpr int64_t K_TEST_O_UID = 63;
constexpr int64_t K_TEST_ATTN_MASK_UID = 64;
constexpr int64_t K_TEST_SCALE_UID = 65;
constexpr int64_t K_TEST_SEQ_LEN_Q_UID = 66;
constexpr int64_t K_TEST_SEQ_LEN_KV_UID = 67;
constexpr int64_t K_TEST_SEED_UID = 68;
constexpr int64_t K_TEST_OFFSET_UID = 69;
constexpr int64_t K_TEST_DROPOUT_MASK_UID = 70;
constexpr int64_t K_TEST_DROPOUT_SCALE_UID = 71;

constexpr std::array<int64_t, 4> K_TEST_DIMS = {1, 8, 128, 64};
constexpr std::array<int64_t, 4> K_TEST_STRIDES = {65536, 8192, 64, 1};

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationSdpaFpropDescriptorLifting : public ::testing::Test
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

    /// Builds a standard SdpaFprop graph for round-trip testing.
    static std::shared_ptr<TestableGraph> buildGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("SdpaFpropLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        q->set_uid(K_TEST_Q_UID)
            .set_name("q")
            .set_data_type(DataType::FLOAT);
        q->set_dim(toVec(K_TEST_DIMS)).set_stride(toVec(K_TEST_STRIDES));

        auto k = std::make_shared<TensorAttributes>();
        k->set_uid(K_TEST_K_UID)
            .set_name("k")
            .set_data_type(DataType::FLOAT);
        k->set_dim(toVec(K_TEST_DIMS)).set_stride(toVec(K_TEST_STRIDES));

        auto v = std::make_shared<TensorAttributes>();
        v->set_uid(K_TEST_V_UID)
            .set_name("v")
            .set_data_type(DataType::FLOAT);
        v->set_dim(toVec(K_TEST_DIMS)).set_stride(toVec(K_TEST_STRIDES));

        SdpaAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_diagonal_alignment(DiagonalAlignment::TOP_LEFT);

        auto results = graph->sdpa(
            q,
            k,
            v,
            attrs);
        results[0]->set_uid(K_TEST_O_UID)
            .set_output(true)
            .set_name("o");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard SdpaFprop graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationSdpaFpropDescriptorLifting, BasicSdpaFpropRoundTrip)
{
    auto originalGraph = buildGraph();

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
    ASSERT_GE(tensorMap.size(), 4u);

    // Verify q tensor
    ASSERT_NE(tensorMap.count(K_TEST_Q_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_Q_UID]->get_uid(), K_TEST_Q_UID);
    EXPECT_EQ(tensorMap[K_TEST_Q_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_Q_UID]->get_stride(), toVec(K_TEST_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_Q_UID]->get_data_type(), DataType::FLOAT);

    // Verify k tensor
    ASSERT_NE(tensorMap.count(K_TEST_K_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_K_UID]->get_uid(), K_TEST_K_UID);
    EXPECT_EQ(tensorMap[K_TEST_K_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_K_UID]->get_stride(), toVec(K_TEST_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_K_UID]->get_data_type(), DataType::FLOAT);

    // Verify v tensor
    ASSERT_NE(tensorMap.count(K_TEST_V_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_V_UID]->get_uid(), K_TEST_V_UID);
    EXPECT_EQ(tensorMap[K_TEST_V_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_V_UID]->get_stride(), toVec(K_TEST_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_V_UID]->get_data_type(), DataType::FLOAT);

    // Verify o tensor
    ASSERT_NE(tensorMap.count(K_TEST_O_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_O_UID]->get_uid(), K_TEST_O_UID);
    EXPECT_EQ(tensorMap[K_TEST_O_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_O_UID]->get_stride(), toVec(K_TEST_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_O_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<SdpaFpropNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a SdpaFpropNode";

    // Verify diagonal alignment (direct member access, no getter)
    EXPECT_EQ(opNode->attributes.diagonal_alignment, DiagonalAlignment::TOP_LEFT);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationSdpaFpropDescriptorLifting, SdpaFpropTensorSharingPreserved)
{
    auto originalGraph = buildGraph();

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

    auto* opNode = dynamic_cast<SdpaFpropNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify q tensor sharing
    EXPECT_EQ(opNode->attributes.get_q()->get_uid(), K_TEST_Q_UID);
    EXPECT_EQ(tensorMap[K_TEST_Q_UID].get(),
              opNode->attributes.get_q().get());
    // Verify k tensor sharing
    EXPECT_EQ(opNode->attributes.get_k()->get_uid(), K_TEST_K_UID);
    EXPECT_EQ(tensorMap[K_TEST_K_UID].get(),
              opNode->attributes.get_k().get());
    // Verify v tensor sharing
    EXPECT_EQ(opNode->attributes.get_v()->get_uid(), K_TEST_V_UID);
    EXPECT_EQ(tensorMap[K_TEST_V_UID].get(),
              opNode->attributes.get_v().get());
    // Verify o tensor sharing
    EXPECT_EQ(opNode->attributes.get_o()->get_uid(), K_TEST_O_UID);
    EXPECT_EQ(tensorMap[K_TEST_O_UID].get(),
              opNode->attributes.get_o().get());
}

// Builds a SdpaFprop graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationSdpaFpropDescriptorLifting, SdpaFpropLiftWithoutFinalization)
{
    auto originalGraph = buildGraph();

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

    auto* opNode = dynamic_cast<SdpaFpropNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify diagonal alignment
    EXPECT_EQ(opNode->attributes.diagonal_alignment, DiagonalAlignment::TOP_LEFT);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_GE(tensorMap.size(), 4u);

    ASSERT_NE(tensorMap.count(K_TEST_Q_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_Q_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_Q_UID]->get_stride(), toVec(K_TEST_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_K_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_K_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_K_UID]->get_stride(), toVec(K_TEST_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_V_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_V_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_V_UID]->get_stride(), toVec(K_TEST_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_O_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_O_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_O_UID]->get_stride(), toVec(K_TEST_STRIDES));
}

// Builds an SDPA fprop graph with all optional boolean flags, scalar parameters,
// and optional tensors set, lowers via the C-API, lifts back, and verifies
// every attribute survives.
TEST_F(IntegrationSdpaFpropDescriptorLifting, SdpaFpropWithAllOptionalAttributesViaCApi)
{
    auto originalGraph = buildGraph();

    auto& subNodes = originalGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* sdpaNode = dynamic_cast<SdpaFpropNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);

    // Optional input tensors
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_TEST_SCALE_UID).set_name("SCALE");
    scale->set_value(0.125f);

    auto attnMask = std::make_shared<TensorAttributes>();
    attnMask->set_uid(K_TEST_ATTN_MASK_UID)
        .set_name("ATTN_MASK")
        .set_data_type(DataType::FLOAT);
    // ATTN_MASK last dim must equal seq_kv (128) or 1
    constexpr std::array<int64_t, 4> K_ATTN_MASK_DIMS = {1, 1, 128, 128};
    constexpr std::array<int64_t, 4> K_ATTN_MASK_STRIDES = {16384, 16384, 128, 1};
    attnMask->set_dim(toVec(K_ATTN_MASK_DIMS)).set_stride(toVec(K_ATTN_MASK_STRIDES));

    constexpr std::array<int64_t, 4> K_SEQ_LEN_DIMS = {1, 1, 1, 1};
    constexpr std::array<int64_t, 4> K_SEQ_LEN_STRIDES = {1, 1, 1, 1};

    auto seqLenQ = std::make_shared<TensorAttributes>();
    seqLenQ->set_uid(K_TEST_SEQ_LEN_Q_UID)
        .set_name("SEQ_LEN_Q")
        .set_data_type(DataType::INT32);
    seqLenQ->set_dim(toVec(K_SEQ_LEN_DIMS)).set_stride(toVec(K_SEQ_LEN_STRIDES));

    auto seqLenKv = std::make_shared<TensorAttributes>();
    seqLenKv->set_uid(K_TEST_SEQ_LEN_KV_UID)
        .set_name("SEQ_LEN_KV")
        .set_data_type(DataType::INT32);
    seqLenKv->set_dim(toVec(K_SEQ_LEN_DIMS)).set_stride(toVec(K_SEQ_LEN_STRIDES));

    auto seed = std::make_shared<TensorAttributes>();
    seed->set_uid(K_TEST_SEED_UID).set_name("SEED");
    seed->set_value(int32_t{42});

    auto offset = std::make_shared<TensorAttributes>();
    offset->set_uid(K_TEST_OFFSET_UID).set_name("OFFSET");
    offset->set_value(int32_t{0});

    auto dropoutMask = std::make_shared<TensorAttributes>();
    dropoutMask->set_uid(K_TEST_DROPOUT_MASK_UID)
        .set_name("DROPOUT_MASK")
        .set_data_type(DataType::UINT8);
    dropoutMask->set_dim(toVec(K_TEST_DIMS)).set_stride(toVec(K_TEST_STRIDES));

    auto dropoutScale = std::make_shared<TensorAttributes>();
    dropoutScale->set_uid(K_TEST_DROPOUT_SCALE_UID).set_name("DROPOUT_SCALE");
    dropoutScale->set_value(1.0f / (1.0f - 0.1f));

    sdpaNode->attributes.set_attn_scale(scale)
        .set_bias(attnMask)
        .set_seq_len_q(seqLenQ)
        .set_seq_len_kv(seqLenKv)
        .set_dropout(0.1f, seed, offset)
        .set_dropout_mask(dropoutMask)
        .set_dropout_scale(dropoutScale)
        .set_alibi_mask(true)
        .set_padding_mask(true)
        .set_causal_mask(true)
        .set_causal_mask_bottom_right(true)
        .set_generate_stats(true)
        .set_attn_scale_value(0.125f)
        .set_diagonal_band_left_bound(-1)
        .set_diagonal_band_right_bound(1)
        .set_paged_attention_max_seq_len_kv(256)
        .set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify node attributes on the lifted SdpaFpropNode
    auto& liftedNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(liftedNodes.size(), 1u);
    auto* liftedNode = dynamic_cast<SdpaFpropNode*>(liftedNodes[0].get());
    ASSERT_NE(liftedNode, nullptr);

    const auto& attrs = liftedNode->attributes;

    // Optional tensor UIDs on the node
    ASSERT_NE(attrs.get_attn_scale(), nullptr);
    EXPECT_EQ(attrs.get_attn_scale()->get_uid(), K_TEST_SCALE_UID);
    ASSERT_NE(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_bias()->get_uid(), K_TEST_ATTN_MASK_UID);
    ASSERT_NE(attrs.get_seq_len_q(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_q()->get_uid(), K_TEST_SEQ_LEN_Q_UID);
    ASSERT_NE(attrs.get_seq_len_kv(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_kv()->get_uid(), K_TEST_SEQ_LEN_KV_UID);
    ASSERT_NE(attrs.get_seed(), nullptr);
    EXPECT_EQ(attrs.get_seed()->get_uid(), K_TEST_SEED_UID);
    ASSERT_NE(attrs.get_offset(), nullptr);
    EXPECT_EQ(attrs.get_offset()->get_uid(), K_TEST_OFFSET_UID);
    ASSERT_NE(attrs.get_dropout_mask(), nullptr);
    EXPECT_EQ(attrs.get_dropout_mask()->get_uid(), K_TEST_DROPOUT_MASK_UID);
    ASSERT_NE(attrs.get_dropout_scale(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale()->get_uid(), K_TEST_DROPOUT_SCALE_UID);

    // Boolean flags (direct member access -- not optional)
    EXPECT_TRUE(attrs.alibi_mask);
    EXPECT_TRUE(attrs.padding_mask);
    EXPECT_TRUE(attrs.causal_mask);
    EXPECT_TRUE(attrs.causal_mask_bottom_right);

    // Optional bool
    ASSERT_TRUE(attrs.generate_stats.has_value());
    EXPECT_TRUE(attrs.generate_stats.value());

    // Scalar values (direct member access)
    ASSERT_TRUE(attrs.dropout_probability.has_value());
    EXPECT_FLOAT_EQ(attrs.dropout_probability.value(), 0.1f);
    ASSERT_TRUE(attrs.attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(attrs.attn_scale_value.value(), 0.125f);
    ASSERT_TRUE(attrs.left_bound.has_value());
    EXPECT_EQ(attrs.left_bound.value(), -1);
    ASSERT_TRUE(attrs.right_bound.has_value());
    EXPECT_EQ(attrs.right_bound.value(), 1);
    ASSERT_TRUE(attrs.max_seq_len_kv.has_value());
    EXPECT_EQ(attrs.max_seq_len_kv.value(), 256);

    // Diagonal alignment (direct member)
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
}

// Builds an SDPA fprop graph without explicit tensor UIDs, then verifies
// auto-assigned UIDs are unique and correctly referenced after a round-trip.
TEST_F(IntegrationSdpaFpropDescriptorLifting, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidSdpaFpropGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Create tensors WITHOUT explicit UIDs
    auto q = std::make_shared<TensorAttributes>();
    q->set_name("Q").set_data_type(DataType::FLOAT);
    q->set_dim(toVec(K_TEST_DIMS)).set_stride(toVec(K_TEST_STRIDES));

    auto k = std::make_shared<TensorAttributes>();
    k->set_name("K").set_data_type(DataType::FLOAT);
    k->set_dim(toVec(K_TEST_DIMS)).set_stride(toVec(K_TEST_STRIDES));

    auto v = std::make_shared<TensorAttributes>();
    v->set_name("V").set_data_type(DataType::FLOAT);
    v->set_dim(toVec(K_TEST_DIMS)).set_stride(toVec(K_TEST_STRIDES));

    SdpaAttributes sdpaAttrs;

    auto results = graph->sdpa(q, k, v, std::move(sdpaAttrs));
    results[0]->set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // All auto-assigned UIDs should be unique
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_GE(tensorMap.size(), 4u) << "Expected at least 4 tensors with auto-assigned UIDs";

    std::unordered_set<int64_t> uids;
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.insert(uid);
    }
    EXPECT_EQ(uids.size(), tensorMap.size()) << "Auto-assigned tensor UIDs are not unique";

    // Verify the node references resolve to tensors in the map
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* sdpaNode = dynamic_cast<SdpaFpropNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);

    EXPECT_NE(uids.count(sdpaNode->attributes.get_q()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_k()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_v()->get_uid()), 0u);
    EXPECT_NE(uids.count(sdpaNode->attributes.get_o()->get_uid()), 0u);
}

// Serializes an SDPA fprop graph to binary, then deserializes via the backend
// using a handle (full finalization). Verifies the reconstructed graph matches the original.
TEST_F(IntegrationSdpaFpropDescriptorLifting, SdpaFpropDeserializeViaBackendWithHandle)
{
    auto originalGraph = buildGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->deserialize_via_backend(_handle, data);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types and name
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_name(), "SdpaFpropLiftingTestGraph");

    // Verify tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_GE(tensorMap.size(), 4u);

    EXPECT_EQ(tensorMap[K_TEST_Q_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_K_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_V_UID]->get_dim(), toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_O_UID]->get_dim(), toVec(K_TEST_DIMS));

    // Verify the node is an SdpaFpropNode with the correct operation name
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* sdpaNode = dynamic_cast<SdpaFpropNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);
    EXPECT_EQ(sdpaNode->attributes.get_name(), "test_op");
    EXPECT_EQ(sdpaNode->attributes.get_q()->get_uid(), K_TEST_Q_UID);
    EXPECT_EQ(sdpaNode->attributes.get_o()->get_uid(), K_TEST_O_UID);
}

} // namespace
