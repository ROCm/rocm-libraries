// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/ReductionConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using ReductionModeSdk = hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationReductionDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(ReductionAttributes& attrs)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("ReductionIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_REDUCTION_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_REDUCTION_TENSOR_X_DIMS))
            .set_stride(toVec(K_REDUCTION_TENSOR_X_STRIDES));

        auto y = graph->reduction(x, attrs);
        y->set_uid(K_REDUCTION_TENSOR_Y_UID).set_output(true).set_name("y");
        y->set_dim(toVec(K_REDUCTION_TENSOR_Y_DIMS))
            .set_stride(toVec(K_REDUCTION_TENSOR_Y_STRIDES));

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationReductionDescriptorLowering, ReductionLoweringRoundTrip)
{
    ReductionAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_mode(ReductionMode::ADD);

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 2u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_REDUCTION_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->dims, toVec(K_REDUCTION_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->strides, toVec(K_REDUCTION_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->name, "x");
    ASSERT_NE(tensorMap.count(K_REDUCTION_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->dims, toVec(K_REDUCTION_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->strides, toVec(K_REDUCTION_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->name, "y");

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::ReductionAttributes);

    auto* opNode = node->attributes.AsReductionAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->in_tensor_uid, K_REDUCTION_TENSOR_X_UID);
    EXPECT_EQ(opNode->out_tensor_uid, K_REDUCTION_TENSOR_Y_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    // Verify mode
    EXPECT_EQ(opNode->mode, ReductionModeSdk::ADD);
}

// Verifies that the optional is_deterministic attribute survives lowering round-trip.
TEST_F(IntegrationReductionDescriptorLowering, IsDeterministicPreservedInRoundTrip)
{
    ReductionAttributes attrs;
    attrs.set_name("test_is_deterministic");
    attrs.set_mode(ReductionMode::ADD);
    attrs.set_is_deterministic(true);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsReductionAttributes();
    ASSERT_NE(opNode, nullptr);

    EXPECT_TRUE(opNode->is_deterministic);
}

// =============================================================================
// Parameterized mode test — verifies all 9 reduction modes survive lowering
// =============================================================================

struct ReductionModeParam
{
    ReductionMode frontend;
    ReductionModeSdk backend;
    std::string name;
};

class IntegrationReductionModeLowering : public IntegrationReductionDescriptorLowering,
                                         public ::testing::WithParamInterface<ReductionModeParam>
{
};

TEST_P(IntegrationReductionModeLowering, ModePreservedInLoweringRoundTrip)
{
    const auto& param = GetParam();

    ReductionAttributes attrs;
    attrs.set_name("test_mode");
    attrs.set_mode(param.frontend);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsReductionAttributes();
    ASSERT_NE(opNode, nullptr);

    EXPECT_EQ(opNode->mode, param.backend);
}

INSTANTIATE_TEST_SUITE_P(
    AllReductionModes,
    IntegrationReductionModeLowering,
    ::testing::Values(ReductionModeParam{ReductionMode::ADD, ReductionModeSdk::ADD, "ADD"},
                      ReductionModeParam{ReductionMode::MUL, ReductionModeSdk::MUL, "MUL"},
                      ReductionModeParam{ReductionMode::MIN, ReductionModeSdk::MIN_OP, "MIN"},
                      ReductionModeParam{ReductionMode::MAX, ReductionModeSdk::MAX_OP, "MAX"},
                      ReductionModeParam{ReductionMode::AMAX, ReductionModeSdk::AMAX, "AMAX"},
                      ReductionModeParam{ReductionMode::AVG, ReductionModeSdk::AVG, "AVG"},
                      ReductionModeParam{ReductionMode::NORM1, ReductionModeSdk::NORM1, "NORM1"},
                      ReductionModeParam{ReductionMode::NORM2, ReductionModeSdk::NORM2, "NORM2"},
                      ReductionModeParam{ReductionMode::MUL_NO_ZEROS,
                                         ReductionModeSdk::MUL_NO_ZEROS,
                                         "MUL_NO_ZEROS"}),
    [](const ::testing::TestParamInfo<ReductionModeParam>& info) { return info.param.name; });

} // namespace
