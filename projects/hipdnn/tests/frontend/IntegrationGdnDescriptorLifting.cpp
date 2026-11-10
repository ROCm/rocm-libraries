// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/GdnNode.hpp>
#include <hipdnn_test_sdk/constants/GdnConstants.hpp>
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

// state_indices cannot combine with checkpoints, so the dense-state and pooled-state
// variants are separate graphs.
enum class GdnVariant
{
    MINIMAL, // required operands only
    DENSE_STATE, // initial_state, safe-gate a_log / dt_bias, final_state, state_checkpoints
    POOLED_STATE, // initial_state pool + state_indices, final_state written in place
};

template <typename DimsT, typename StridesT>
std::shared_ptr<TensorAttributes> makeTensor(int64_t uid,
                                             bool assignUids,
                                             const std::string& name,
                                             DataType dataType,
                                             const DimsT& dims,
                                             const StridesT& strides)
{
    auto tensor = std::make_shared<TensorAttributes>();
    if(assignUids)
    {
        tensor->set_uid(uid);
    }
    tensor->set_name(name).set_data_type(dataType);
    tensor->set_dim(toVec(dims)).set_stride(toVec(strides));
    return tensor;
}

void markOutput(const std::shared_ptr<TensorAttributes>& tensor,
                int64_t uid,
                bool assignUids,
                const std::string& name)
{
    if(!tensor)
    {
        return;
    }
    if(assignUids)
    {
        tensor->set_uid(uid);
    }
    tensor->set_output(true).set_name(name);
}

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationGdnDescriptorLifting : public IntegrationTestFixture
{
protected:
    static GdnAttributes createAttributes(DataType operationComputeDataType = DataType::NOT_SET)
    {
        GdnAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_scale(0.25F);
        attrs.set_use_qk_l2norm(true);
        attrs.set_use_beta_sigmoid(true);
        attrs.set_allow_neg_eigval(true);
        attrs.set_batch_invariant(true);
        if(operationComputeDataType != DataType::NOT_SET)
        {
            attrs.set_compute_data_type(operationComputeDataType);
        }
        return attrs;
    }

    static std::shared_ptr<TestableGraphLifting> buildGraph(GdnAttributes attrs,
                                                            GdnVariant variant
                                                            = GdnVariant::DENSE_STATE,
                                                            bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("GdnLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto q = makeTensor(K_GDN_TENSOR_Q_UID,
                            assignUids,
                            "q",
                            DataType::FLOAT,
                            K_GDN_TENSOR_Q_DIMS,
                            K_GDN_TENSOR_Q_STRIDES);
        auto k = makeTensor(K_GDN_TENSOR_K_UID,
                            assignUids,
                            "k",
                            DataType::FLOAT,
                            K_GDN_TENSOR_K_DIMS,
                            K_GDN_TENSOR_K_STRIDES);
        auto v = makeTensor(K_GDN_TENSOR_V_UID,
                            assignUids,
                            "v",
                            DataType::FLOAT,
                            K_GDN_TENSOR_V_DIMS,
                            K_GDN_TENSOR_V_STRIDES);
        auto g = makeTensor(K_GDN_TENSOR_G_UID,
                            assignUids,
                            "g",
                            DataType::FLOAT,
                            K_GDN_TENSOR_G_DIMS,
                            K_GDN_TENSOR_G_STRIDES);
        auto beta = makeTensor(K_GDN_TENSOR_BETA_UID,
                               assignUids,
                               "beta",
                               DataType::FLOAT,
                               K_GDN_TENSOR_BETA_DIMS,
                               K_GDN_TENSOR_BETA_STRIDES);
        auto cuSeqlens = makeTensor(K_GDN_TENSOR_CU_SEQLENS_UID,
                                    assignUids,
                                    "cu_seqlens",
                                    DataType::INT32,
                                    K_GDN_TENSOR_CU_SEQLENS_DIMS,
                                    K_GDN_TENSOR_CU_SEQLENS_STRIDES);

        if(variant != GdnVariant::MINIMAL)
        {
            attrs.set_initial_state(makeTensor(K_GDN_TENSOR_INITIAL_STATE_UID,
                                               assignUids,
                                               "initial_state",
                                               DataType::FLOAT,
                                               K_GDN_TENSOR_INITIAL_STATE_DIMS,
                                               K_GDN_TENSOR_INITIAL_STATE_STRIDES));
            attrs.set_output_final_state(true);
        }
        if(variant == GdnVariant::DENSE_STATE)
        {
            attrs.set_safe_gate(true);
            attrs.set_a_log(makeTensor(K_GDN_TENSOR_A_LOG_UID,
                                       assignUids,
                                       "a_log",
                                       DataType::FLOAT,
                                       K_GDN_TENSOR_A_LOG_DIMS,
                                       K_GDN_TENSOR_A_LOG_STRIDES));
            attrs.set_dt_bias(makeTensor(K_GDN_TENSOR_DT_BIAS_UID,
                                         assignUids,
                                         "dt_bias",
                                         DataType::FLOAT,
                                         K_GDN_TENSOR_DT_BIAS_DIMS,
                                         K_GDN_TENSOR_DT_BIAS_STRIDES));
            attrs.set_checkpoint_every_n_tokens(2);
        }
        if(variant == GdnVariant::POOLED_STATE)
        {
            attrs.set_state_indices(makeTensor(K_GDN_TENSOR_STATE_INDICES_UID,
                                               assignUids,
                                               "state_indices",
                                               DataType::INT32,
                                               K_GDN_TENSOR_STATE_INDICES_DIMS,
                                               K_GDN_TENSOR_STATE_INDICES_STRIDES));
            attrs.set_overwrite_initial_state(true);
        }

        auto [o, finalState, stateCheckpoints]
            = graph->gdn(q, k, v, g, beta, cuSeqlens, std::move(attrs));
        markOutput(o, K_GDN_TENSOR_O_UID, assignUids, "o");
        markOutput(finalState, K_GDN_TENSOR_FINAL_STATE_UID, assignUids, "final_state");
        markOutput(
            stateCheckpoints, K_GDN_TENSOR_STATE_CHECKPOINTS_UID, assignUids, "state_checkpoints");
        return graph;
    }

    static GdnNode* singleGdnNode(TestableGraphLifting& graph)
    {
        auto& subNodes = graph.getSubNodes();
        if(subNodes.size() != 1u)
        {
            return nullptr;
        }
        return dynamic_cast<GdnNode*>(subNodes[0].get());
    }
};

// Builds the dense-state graph, lowers via build_operation_graph(handle), lifts back with
// fromBackendDescriptor(), and validates graph data types, tensors and operation parameters.
TEST_F(IntegrationGdnDescriptorLifting, BasicGdnRoundTrip)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 12u);
    ASSERT_NE(tensorMap.count(K_GDN_TENSOR_Q_UID), 0u);
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_Q_UID]->get_dim(), toVec(K_GDN_TENSOR_Q_DIMS));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_Q_UID]->get_stride(), toVec(K_GDN_TENSOR_Q_STRIDES));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_Q_UID]->get_name(), "q");
    ASSERT_NE(tensorMap.count(K_GDN_TENSOR_CU_SEQLENS_UID), 0u);
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_CU_SEQLENS_UID]->get_data_type(), DataType::INT32);
    ASSERT_NE(tensorMap.count(K_GDN_TENSOR_O_UID), 0u);
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_O_UID]->get_dim(), toVec(K_GDN_TENSOR_O_DIMS));
    ASSERT_NE(tensorMap.count(K_GDN_TENSOR_STATE_CHECKPOINTS_UID), 0u);
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_STATE_CHECKPOINTS_UID]->get_dim(),
              toVec(K_GDN_TENSOR_STATE_CHECKPOINTS_DIMS));

    auto* opNode = singleGdnNode(*liftedGraph);
    ASSERT_NE(opNode, nullptr)
        << "Expected a single GdnNode"; // NOLINT(readability-implicit-bool-conversion)

    const auto& attrs = opNode->attributes;
    ASSERT_TRUE(attrs.get_scale().has_value());
    EXPECT_FLOAT_EQ(attrs.get_scale().value(), 0.25F);
    EXPECT_TRUE(attrs.get_output_final_state());
    EXPECT_TRUE(attrs.get_use_qk_l2norm());
    EXPECT_EQ(attrs.get_checkpoint_every_n_tokens(), 2);
    EXPECT_TRUE(attrs.get_use_beta_sigmoid());
    EXPECT_TRUE(attrs.get_allow_neg_eigval());
    EXPECT_TRUE(attrs.get_safe_gate());
    EXPECT_EQ(attrs.get_gate_domain(), GateDomain::LOG);
    EXPECT_TRUE(attrs.get_batch_invariant());
    EXPECT_FALSE(attrs.get_overwrite_initial_state());
    EXPECT_EQ(attrs.get_name(), "test_op");

    ASSERT_NE(attrs.get_a_log(), nullptr);
    ASSERT_NE(attrs.get_dt_bias(), nullptr);
    ASSERT_NE(attrs.get_initial_state(), nullptr);
    ASSERT_NE(attrs.get_final_state(), nullptr);
    ASSERT_NE(attrs.get_state_checkpoints(), nullptr);
    EXPECT_EQ(attrs.get_state_indices(), nullptr);
}

// The pooled-state variant lifts state_indices and overwrite_initial_state.
TEST_F(IntegrationGdnDescriptorLifting, PooledStateRoundTrip)
{
    auto attrs = createAttributes();
    attrs.set_gate_domain(GateDomain::LINEAR);
    auto originalGraph = buildGraph(std::move(attrs), GdnVariant::POOLED_STATE);

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto* opNode = singleGdnNode(*liftedGraph);
    ASSERT_NE(opNode, nullptr);
    ASSERT_NE(opNode->attributes.get_state_indices(), nullptr);
    EXPECT_EQ(opNode->attributes.get_state_indices()->get_uid(), K_GDN_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(opNode->attributes.get_state_indices()->get_data_type(), DataType::INT32);
    EXPECT_TRUE(opNode->attributes.get_overwrite_initial_state());
    EXPECT_EQ(opNode->attributes.get_gate_domain(), GateDomain::LINEAR);
    EXPECT_EQ(opNode->attributes.get_state_checkpoints(), nullptr);
}

// Verifies an operation-level compute type survives descriptor lifting.
TEST_F(IntegrationGdnDescriptorLifting, OperationComputeDataTypeSurvivesLifting)
{
    auto originalGraph = buildGraph(createAttributes(DataType::HALF));

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto* opNode = singleGdnNode(*liftedGraph);
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_compute_data_type(), DataType::HALF);
}

// After lifting, tensor objects in the node attributes are the same shared_ptr instances
// as in the tensor map.
TEST_F(IntegrationGdnDescriptorLifting, GdnTensorSharingPreserved)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    auto* opNode = singleGdnNode(*liftedGraph);
    ASSERT_NE(opNode, nullptr);

    const auto& attrs = opNode->attributes;
    const std::vector<std::pair<int64_t, std::shared_ptr<TensorAttributes>>> expected = {
        {K_GDN_TENSOR_Q_UID, attrs.get_q()},
        {K_GDN_TENSOR_K_UID, attrs.get_k()},
        {K_GDN_TENSOR_V_UID, attrs.get_v()},
        {K_GDN_TENSOR_G_UID, attrs.get_g()},
        {K_GDN_TENSOR_BETA_UID, attrs.get_beta()},
        {K_GDN_TENSOR_CU_SEQLENS_UID, attrs.get_cu_seqlens()},
        {K_GDN_TENSOR_INITIAL_STATE_UID, attrs.get_initial_state()},
        {K_GDN_TENSOR_A_LOG_UID, attrs.get_a_log()},
        {K_GDN_TENSOR_DT_BIAS_UID, attrs.get_dt_bias()},
        {K_GDN_TENSOR_O_UID, attrs.get_o()},
        {K_GDN_TENSOR_FINAL_STATE_UID, attrs.get_final_state()},
        {K_GDN_TENSOR_STATE_CHECKPOINTS_UID, attrs.get_state_checkpoints()},
    };
    for(const auto& [uid, tensor] : expected)
    {
        ASSERT_NE(tensor, nullptr) << uid;
        EXPECT_EQ(tensor->get_uid(), uid);
        EXPECT_EQ(tensorMap[uid].get(), tensor.get()) << uid;
    }
}

// Serializes to binary, creates a backend descriptor from bytes (no handle, no finalize),
// calls fromBackendDescriptor(), and verifies the fields survive.
TEST_F(IntegrationGdnDescriptorLifting, GdnLiftWithoutFinalization)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    auto* opNode = singleGdnNode(*liftedGraph);
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
    EXPECT_TRUE(opNode->attributes.get_safe_gate());
    EXPECT_EQ(opNode->attributes.get_checkpoint_every_n_tokens(), 2);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 12u);
    ASSERT_NE(tensorMap.count(K_GDN_TENSOR_FINAL_STATE_UID), 0u);
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_FINAL_STATE_UID]->get_dim(),
              toVec(K_GDN_TENSOR_FINAL_STATE_DIMS));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_FINAL_STATE_UID]->get_stride(),
              toVec(K_GDN_TENSOR_FINAL_STATE_STRIDES));
    EXPECT_EQ(tensorMap[K_GDN_TENSOR_FINAL_STATE_UID]->get_name(), "final_state");
}

// An unset scale stays unset; a set scale survives the lifting round-trip.
TEST_F(IntegrationGdnDescriptorLifting, ScalePreservedInLiftingRoundTrip)
{
    GdnAttributes unsetScale;
    unsetScale.set_name("test_no_scale");
    auto unsetGraph = buildGraph(std::move(unsetScale), GdnVariant::MINIMAL);
    auto liftedUnset = liftGraph(*unsetGraph, _handle);
    ASSERT_NE(liftedUnset, nullptr);
    auto* unsetNode = singleGdnNode(*liftedUnset);
    ASSERT_NE(unsetNode, nullptr);
    EXPECT_FALSE(unsetNode->attributes.get_scale().has_value());

    GdnAttributes setScale;
    setScale.set_name("test_scale");
    setScale.set_scale(0.5F);
    auto setGraph = buildGraph(std::move(setScale), GdnVariant::MINIMAL);
    auto liftedSet = liftGraph(*setGraph, _handle);
    ASSERT_NE(liftedSet, nullptr);
    auto* setNode = singleGdnNode(*liftedSet);
    ASSERT_NE(setNode, nullptr);
    ASSERT_TRUE(setNode->attributes.get_scale().has_value());
    EXPECT_FLOAT_EQ(setNode->attributes.get_scale().value(), 0.5F);
}

// Builds the graph without set_uid() on any tensor, lowers, lifts, and verifies all
// auto-assigned UIDs are distinct and survive the round-trip.
TEST_F(IntegrationGdnDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = buildGraph(createAttributes(), GdnVariant::DENSE_STATE, false);

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 12u);

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

    auto* opNode = singleGdnNode(*liftedGraph);
    ASSERT_NE(opNode, nullptr);

    const auto& attrs = opNode->attributes;
    std::set<int64_t> nodeUids;
    for(const auto& tensor : {attrs.get_q(),
                              attrs.get_k(),
                              attrs.get_v(),
                              attrs.get_g(),
                              attrs.get_beta(),
                              attrs.get_cu_seqlens(),
                              attrs.get_initial_state(),
                              attrs.get_a_log(),
                              attrs.get_dt_bias(),
                              attrs.get_o(),
                              attrs.get_final_state(),
                              attrs.get_state_checkpoints()})
    {
        ASSERT_NE(tensor, nullptr);
        nodeUids.insert(tensor->get_uid());
    }
    ASSERT_EQ(nodeUids.size(), 12u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(attrs.get_q()->get_dim(), toVec(K_GDN_TENSOR_Q_DIMS));
    EXPECT_EQ(attrs.get_q()->get_name(), "q");
    EXPECT_EQ(attrs.get_o()->get_dim(), toVec(K_GDN_TENSOR_O_DIMS));
    EXPECT_EQ(attrs.get_o()->get_name(), "o");
}

} // namespace
