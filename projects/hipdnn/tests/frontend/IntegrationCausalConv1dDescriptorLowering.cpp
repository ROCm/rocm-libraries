// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/causal_conv1d_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/CausalConv1dConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using CausalConv1dActivationSdk = hipdnn_flatbuffers_sdk::data_objects::CausalConv1dActivation;

namespace
{

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

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationCausalConv1dDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT. With withState, adds
    /// bias, initial_state, state_indices and requests final_state.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(CausalConv1dAttributes& attrs,
                                                                     bool withState = true,
                                                                     bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("CausalConv1dIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = makeTensor(K_CAUSAL_CONV1D_TENSOR_X_UID,
                            assignUids,
                            "x",
                            DataType::FLOAT,
                            K_CAUSAL_CONV1D_TENSOR_X_DIMS,
                            K_CAUSAL_CONV1D_TENSOR_X_STRIDES);
        auto weight = makeTensor(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                                 assignUids,
                                 "weight",
                                 DataType::FLOAT,
                                 K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS,
                                 K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES);
        auto cuSeqlens = makeTensor(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                                    assignUids,
                                    "cu_seqlens",
                                    DataType::INT32,
                                    K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS,
                                    K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES);

        if(withState)
        {
            attrs.set_bias(makeTensor(K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                                      assignUids,
                                      "bias",
                                      DataType::FLOAT,
                                      K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS,
                                      K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES));
            attrs.set_initial_state(makeTensor(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                                               assignUids,
                                               "initial_state",
                                               DataType::FLOAT,
                                               K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS,
                                               K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES));
            attrs.set_state_indices(makeTensor(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                                               assignUids,
                                               "state_indices",
                                               DataType::INT32,
                                               K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS,
                                               K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES));
            attrs.set_output_final_state(true);
        }

        auto [y, finalState] = graph->causal_conv1d(x, weight, cuSeqlens, attrs);
        if(assignUids)
        {
            y->set_uid(K_CAUSAL_CONV1D_TENSOR_Y_UID);
        }
        y->set_output(true).set_name("y");
        if(finalState)
        {
            if(assignUids)
            {
                finalState->set_uid(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
            }
            finalState->set_output(true).set_name("final_state");
        }

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph with every optional tensor, lowers via
// descriptors, and verifies the deserialized FlatBuffer attributes match.
TEST_F(IntegrationCausalConv1dDescriptorLowering, CausalConv1dLoweringRoundTrip)
{
    CausalConv1dAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_activation(CausalConv1dActivation::SILU);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.tensors.size(), 8u);

    auto tensorMap = buildTensorMap(graphT);
    const auto expectTensor = [&](int64_t uid,
                                  const std::vector<int64_t>& dims,
                                  const std::vector<int64_t>& strides,
                                  DataTypeSdk dataType,
                                  const std::string& name) {
        ASSERT_NE(tensorMap.count(uid), 0u) << name;
        EXPECT_EQ(tensorMap[uid]->dims, dims) << name;
        EXPECT_EQ(tensorMap[uid]->strides, strides) << name;
        EXPECT_EQ(tensorMap[uid]->data_type, dataType) << name;
        EXPECT_EQ(tensorMap[uid]->name, name);
    };
    expectTensor(K_CAUSAL_CONV1D_TENSOR_X_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES),
                 DataTypeSdk::FLOAT,
                 "x");
    expectTensor(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES),
                 DataTypeSdk::FLOAT,
                 "weight");
    expectTensor(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES),
                 DataTypeSdk::INT32,
                 "cu_seqlens");
    expectTensor(K_CAUSAL_CONV1D_TENSOR_Y_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES),
                 DataTypeSdk::FLOAT,
                 "y");
    expectTensor(K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES),
                 DataTypeSdk::FLOAT,
                 "bias");
    expectTensor(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES),
                 DataTypeSdk::FLOAT,
                 "initial_state");
    expectTensor(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES),
                 DataTypeSdk::INT32,
                 "state_indices");
    expectTensor(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES),
                 DataTypeSdk::FLOAT,
                 "final_state");

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::CausalConv1dAttributes);
    EXPECT_EQ(node->name, "test_op");

    auto* opNode = node->attributes.AsCausalConv1dAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->x_tensor_uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    EXPECT_EQ(opNode->weight_tensor_uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    EXPECT_EQ(opNode->cu_seqlens_tensor_uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(opNode->y_tensor_uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    EXPECT_EQ(opNode->bias_tensor_uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    EXPECT_EQ(opNode->initial_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(opNode->state_indices_tensor_uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(opNode->final_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(opNode->activation, CausalConv1dActivationSdk::SILU);
}

// Without optional inputs the optional UIDs stay absent and activation defaults to IDENTITY.
TEST_F(IntegrationCausalConv1dDescriptorLowering, RequiredTensorsOnly)
{
    CausalConv1dAttributes attrs;
    attrs.set_name("test_required_only");

    auto graphT = buildAndDeserialize(attrs, false);

    ASSERT_EQ(graphT.tensors.size(), 4u);
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsCausalConv1dAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_FALSE(opNode->bias_tensor_uid.has_value());
    EXPECT_FALSE(opNode->initial_state_tensor_uid.has_value());
    EXPECT_FALSE(opNode->state_indices_tensor_uid.has_value());
    EXPECT_FALSE(opNode->final_state_tensor_uid.has_value());
    EXPECT_EQ(opNode->activation, CausalConv1dActivationSdk::IDENTITY);
}

// Verifies an operation-level compute type overrides the graph default during lowering.
TEST_F(IntegrationCausalConv1dDescriptorLowering, OperationComputeDataTypeOverride)
{
    CausalConv1dAttributes attrs;
    attrs.set_name("test_operation_compute_type");
    attrs.set_activation(CausalConv1dActivation::SILU);
    attrs.set_compute_data_type(DataType::HALF);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    EXPECT_EQ(graphT.nodes[0]->compute_data_type, DataTypeSdk::HALF);
}

// Verifies that lowering assigns unique UIDs when callers omit them.
TEST_F(IntegrationCausalConv1dDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    CausalConv1dAttributes attrs;
    attrs.set_name("test_auto_uid");

    auto graphT = buildAndDeserialize(attrs, true, false);

    ASSERT_EQ(graphT.tensors.size(), 8u);
    std::unordered_set<int64_t> uids;
    for(const auto& tensor : graphT.tensors)
    {
        uids.insert(tensor->uid);
    }
    ASSERT_EQ(uids.size(), 8u);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsCausalConv1dAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_TRUE(uids.count(opNode->x_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->weight_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->cu_seqlens_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->y_tensor_uid) > 0);
    ASSERT_TRUE(opNode->bias_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(*opNode->bias_tensor_uid) > 0);
    ASSERT_TRUE(opNode->initial_state_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(*opNode->initial_state_tensor_uid) > 0);
    ASSERT_TRUE(opNode->state_indices_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(*opNode->state_indices_tensor_uid) > 0);
    ASSERT_TRUE(opNode->final_state_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(*opNode->final_state_tensor_uid) > 0);
}

} // namespace
