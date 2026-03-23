// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/SdpaBpropNode.hpp>
#include <hipdnn_test_sdk/constants/SdpaBpropConstants.hpp>
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
    using Graph::get_raw_graph_descriptor;

    const std::vector<std::shared_ptr<INode>>& getSubNodes() const
    {
        return _sub_nodes;
    }
};

// Builds an SDPA backward graph via the frontend, lowers it through the backend C-API
// via build_operation_graph(), then lifts it back with fromBackendDescriptor()
// and verifies the reconstructed graph matches the original.
class IntegrationSdpaBpropLifting : public ::testing::Test
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

    // Builds a standard SDPA backward graph for round-trip testing
    static std::shared_ptr<TestableGraph> buildSdpaBpropGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("LiftingSdpaBpropGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        q->set_uid(K_SDPA_BPROP_TENSOR_Q_UID).set_name("Q").set_data_type(DataType::FLOAT);
        q->set_dim(toVec(K_SDPA_BPROP_TENSOR_Q_DIMS))
            .set_stride(toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));

        auto k = std::make_shared<TensorAttributes>();
        k->set_uid(K_SDPA_BPROP_TENSOR_K_UID).set_name("K").set_data_type(DataType::FLOAT);
        k->set_dim(toVec(K_SDPA_BPROP_TENSOR_K_DIMS))
            .set_stride(toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));

        auto v = std::make_shared<TensorAttributes>();
        v->set_uid(K_SDPA_BPROP_TENSOR_V_UID).set_name("V").set_data_type(DataType::FLOAT);
        v->set_dim(toVec(K_SDPA_BPROP_TENSOR_V_DIMS))
            .set_stride(toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));

        auto o = std::make_shared<TensorAttributes>();
        o->set_uid(K_SDPA_BPROP_TENSOR_O_UID).set_name("O").set_data_type(DataType::FLOAT);
        o->set_dim(toVec(K_SDPA_BPROP_TENSOR_O_DIMS))
            .set_stride(toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));

        auto dO = std::make_shared<TensorAttributes>();
        dO->set_uid(K_SDPA_BPROP_TENSOR_DO_UID).set_name("dO").set_data_type(DataType::FLOAT);
        dO->set_dim(toVec(K_SDPA_BPROP_TENSOR_DO_DIMS))
            .set_stride(toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));

        auto stats = std::make_shared<TensorAttributes>();
        stats->set_uid(K_SDPA_BPROP_TENSOR_STATS_UID)
            .set_name("Stats")
            .set_data_type(DataType::FLOAT);
        stats->set_dim(toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS))
            .set_stride(toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));

        SdpaBackwardAttributes sdpaAttrs;
        sdpaAttrs.set_name("sdpa_bprop_op");

        auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, stats, sdpaAttrs);
        dq->set_uid(K_SDPA_BPROP_TENSOR_DQ_UID).set_output(true).set_name("dQ");
        dk->set_uid(K_SDPA_BPROP_TENSOR_DK_UID).set_output(true).set_name("dK");
        dv->set_uid(K_SDPA_BPROP_TENSOR_DV_UID).set_output(true).set_name("dV");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds an SDPA backward graph, lowers via build_operation_graph(handle), extracts the
// raw descriptor, creates a new graph with fromBackendDescriptor(), and verifies
// tensor dimensions, data types, and graph-level data types.
TEST_F(IntegrationSdpaBpropLifting, SdpaBpropRoundTripViaCApi)
{
    auto originalGraph = buildSdpaBpropGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    // Lift back into a new graph
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID — 9 required tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u) << "Expected 9 tensors in lifted SDPA backward graph";

    // Verify required input tensors
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_Q_UID), 0u);
    auto liftedQ = tensorMap[K_SDPA_BPROP_TENSOR_Q_UID];
    EXPECT_EQ(liftedQ->get_dim(), toVec(K_SDPA_BPROP_TENSOR_Q_DIMS));
    EXPECT_EQ(liftedQ->get_stride(), toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    EXPECT_EQ(liftedQ->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_K_UID), 0u);
    auto liftedK = tensorMap[K_SDPA_BPROP_TENSOR_K_UID];
    EXPECT_EQ(liftedK->get_dim(), toVec(K_SDPA_BPROP_TENSOR_K_DIMS));
    EXPECT_EQ(liftedK->get_stride(), toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    EXPECT_EQ(liftedK->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_V_UID), 0u);
    auto liftedV = tensorMap[K_SDPA_BPROP_TENSOR_V_UID];
    EXPECT_EQ(liftedV->get_dim(), toVec(K_SDPA_BPROP_TENSOR_V_DIMS));
    EXPECT_EQ(liftedV->get_stride(), toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    EXPECT_EQ(liftedV->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_O_UID), 0u);
    auto liftedO = tensorMap[K_SDPA_BPROP_TENSOR_O_UID];
    EXPECT_EQ(liftedO->get_dim(), toVec(K_SDPA_BPROP_TENSOR_O_DIMS));
    EXPECT_EQ(liftedO->get_stride(), toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DO_UID), 0u);
    auto liftedDO = tensorMap[K_SDPA_BPROP_TENSOR_DO_UID];
    EXPECT_EQ(liftedDO->get_dim(), toVec(K_SDPA_BPROP_TENSOR_DO_DIMS));
    EXPECT_EQ(liftedDO->get_stride(), toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_STATS_UID), 0u);
    auto liftedStats = tensorMap[K_SDPA_BPROP_TENSOR_STATS_UID];
    EXPECT_EQ(liftedStats->get_dim(), toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS));
    EXPECT_EQ(liftedStats->get_stride(), toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));

    // Verify required output tensors
    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DQ_UID), 0u);
    auto liftedDQ = tensorMap[K_SDPA_BPROP_TENSOR_DQ_UID];
    EXPECT_EQ(liftedDQ->get_dim(), toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS));
    EXPECT_EQ(liftedDQ->get_stride(), toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DK_UID), 0u);
    auto liftedDK = tensorMap[K_SDPA_BPROP_TENSOR_DK_UID];
    EXPECT_EQ(liftedDK->get_dim(), toVec(K_SDPA_BPROP_TENSOR_DK_DIMS));
    EXPECT_EQ(liftedDK->get_stride(), toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));

    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_DV_UID), 0u);
    auto liftedDV = tensorMap[K_SDPA_BPROP_TENSOR_DV_UID];
    EXPECT_EQ(liftedDV->get_dim(), toVec(K_SDPA_BPROP_TENSOR_DV_DIMS));
    EXPECT_EQ(liftedDV->get_stride(), toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));

    // Verify the lifted graph has 1 SDPA backward operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* sdpaNode = dynamic_cast<SdpaBpropNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr) << "Expected a SdpaBpropNode";
    EXPECT_EQ(sdpaNode->attributes.get_name(), "sdpa_bprop_op");
}

// Verifies that tensors are accessible by UID on the reconstructed graph,
// confirming tensor identity is preserved through the round-trip.
TEST_F(IntegrationSdpaBpropLifting, SdpaBpropTensorSharingPreserved)
{
    auto originalGraph = buildSdpaBpropGraph();

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

    auto* sdpaNode = dynamic_cast<SdpaBpropNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);

    // Verify each node tensor is the same object as in the tensor map (shared_ptr identity)
    EXPECT_EQ(sdpaNode->attributes.get_q()->get_uid(), K_SDPA_BPROP_TENSOR_Q_UID);
    EXPECT_EQ(sdpaNode->attributes.get_k()->get_uid(), K_SDPA_BPROP_TENSOR_K_UID);
    EXPECT_EQ(sdpaNode->attributes.get_v()->get_uid(), K_SDPA_BPROP_TENSOR_V_UID);
    EXPECT_EQ(sdpaNode->attributes.get_o()->get_uid(), K_SDPA_BPROP_TENSOR_O_UID);
    EXPECT_EQ(sdpaNode->attributes.get_do()->get_uid(), K_SDPA_BPROP_TENSOR_DO_UID);
    EXPECT_EQ(sdpaNode->attributes.get_stats()->get_uid(), K_SDPA_BPROP_TENSOR_STATS_UID);
    EXPECT_EQ(sdpaNode->attributes.get_dq()->get_uid(), K_SDPA_BPROP_TENSOR_DQ_UID);
    EXPECT_EQ(sdpaNode->attributes.get_dk()->get_uid(), K_SDPA_BPROP_TENSOR_DK_UID);
    EXPECT_EQ(sdpaNode->attributes.get_dv()->get_uid(), K_SDPA_BPROP_TENSOR_DV_UID);

    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_Q_UID].get(), sdpaNode->attributes.get_q().get());
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_K_UID].get(), sdpaNode->attributes.get_k().get());
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_V_UID].get(), sdpaNode->attributes.get_v().get());
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_O_UID].get(), sdpaNode->attributes.get_o().get());
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_DO_UID].get(), sdpaNode->attributes.get_do().get());
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_STATS_UID].get(),
              sdpaNode->attributes.get_stats().get());
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_DQ_UID].get(), sdpaNode->attributes.get_dq().get());
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_DK_UID].get(), sdpaNode->attributes.get_dk().get());
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_DV_UID].get(), sdpaNode->attributes.get_dv().get());
}

// Builds an SDPA backward graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// the reconstructed graph matches the original.
TEST_F(IntegrationSdpaBpropLifting, SdpaBpropLiftWithoutFinalization)
{
    auto originalGraph = buildSdpaBpropGraph();

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

    // Verify the lifted graph has 1 SDPA backward operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* sdpaNode = dynamic_cast<SdpaBpropNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr) << "Expected a SdpaBpropNode";

    // Verify tensor dims survive the serialization round-trip
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 9u);
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_Q_UID]->get_dim(), toVec(K_SDPA_BPROP_TENSOR_Q_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_Q_UID]->get_stride(),
              toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_K_UID]->get_dim(), toVec(K_SDPA_BPROP_TENSOR_K_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_DQ_UID]->get_dim(), toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_DK_UID]->get_dim(), toVec(K_SDPA_BPROP_TENSOR_DK_DIMS));
    EXPECT_EQ(tensorMap[K_SDPA_BPROP_TENSOR_DV_UID]->get_dim(), toVec(K_SDPA_BPROP_TENSOR_DV_DIMS));
}

// Builds an SDPA backward graph with a pass-by-value attention scale, lowers via the
// C-API, lifts back, and verifies the scale value survives the round-trip.
TEST_F(IntegrationSdpaBpropLifting, SdpaBpropWithAttnScale)
{
    auto originalGraph = buildSdpaBpropGraph();

    // Access the SDPA node and attach the attention scale as a pass-by-value scalar
    auto& subNodes = originalGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* sdpaNode = dynamic_cast<SdpaBpropNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_SDPA_BPROP_TENSOR_SCALE_UID).set_name("SCALE");
    scale->set_value(0.125f); // pass-by-value: embedded in descriptor, not a GPU buffer
    sdpaNode->attributes.set_attn_scale(scale);

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // The scale tensor appears in the lifted graph's tensor list (the C-API round-trip
    // serializes all tensors referenced by the descriptor, including the scale).
    auto tensorMap = liftedGraph->getTensorsByUid();
    EXPECT_EQ(tensorMap.size(), 10u) << "Expected 9 required + scale tensor";

    auto& liftedNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(liftedNodes.size(), 1u);
    auto* liftedNode = dynamic_cast<SdpaBpropNode*>(liftedNodes[0].get());
    ASSERT_NE(liftedNode, nullptr);

    // Verify the scale UID survived (stored as scale_tensor_uid in the FlatBuffer)
    auto liftedScale = liftedNode->attributes.get_attn_scale();
    ASSERT_NE(liftedScale, nullptr) << "Scale tensor should be set after lifting";
    EXPECT_EQ(liftedScale->get_uid(), K_SDPA_BPROP_TENSOR_SCALE_UID);
}

// Builds an SDPA backward graph with an attention mask bias tensor, lowers via the
// C-API, lifts back, and verifies the mask tensor survives the round-trip.
TEST_F(IntegrationSdpaBpropLifting, SdpaBpropWithAttnMask)
{
    auto originalGraph = buildSdpaBpropGraph();

    auto& subNodes = originalGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* sdpaNode = dynamic_cast<SdpaBpropNode*>(subNodes[0].get());
    ASSERT_NE(sdpaNode, nullptr);

    auto attnMask = std::make_shared<TensorAttributes>();
    attnMask->set_uid(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID)
        .set_name("ATTN_MASK")
        .set_data_type(DataType::FLOAT);
    attnMask->set_dim(toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_DIMS))
        .set_stride(toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_STRIDES));
    sdpaNode->attributes.set_bias(attnMask);

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Attn mask is a regular GPU tensor — it appears in the tensor list
    auto tensorMap = liftedGraph->getTensorsByUid();
    EXPECT_EQ(tensorMap.size(), 10u) << "Expected 9 required + 1 attn_mask";

    ASSERT_NE(tensorMap.count(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID), 0u);
    auto liftedMask = tensorMap[K_SDPA_BPROP_TENSOR_ATTN_MASK_UID];
    EXPECT_EQ(liftedMask->get_dim(), toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_DIMS));
    EXPECT_EQ(liftedMask->get_stride(), toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_STRIDES));

    auto& liftedNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(liftedNodes.size(), 1u);
    auto* liftedNode = dynamic_cast<SdpaBpropNode*>(liftedNodes[0].get());
    ASSERT_NE(liftedNode, nullptr);

    auto liftedBias = liftedNode->attributes.get_bias();
    ASSERT_NE(liftedBias, nullptr) << "Attn mask should be set after lifting";
    EXPECT_EQ(liftedBias->get_uid(), K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);
    EXPECT_EQ(liftedBias->get_dim(), toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_DIMS));
    EXPECT_EQ(liftedBias->get_stride(), toVec(K_SDPA_BPROP_TENSOR_ATTN_MASK_STRIDES));
}

// Verifies that the graph name survives the C-API round-trip (lower -> lift).
TEST_F(IntegrationSdpaBpropLifting, GraphNamePreservedThroughCApi)
{
    auto originalGraph = buildSdpaBpropGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_EQ(liftedGraph->get_name(), "LiftingSdpaBpropGraph");
}

} // namespace
