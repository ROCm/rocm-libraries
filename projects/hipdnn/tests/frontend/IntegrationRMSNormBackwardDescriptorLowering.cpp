// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/RMSNormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

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
class IntegrationRMSNormBackwardDescriptorLowering : public ::testing::Test
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

    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT
        buildAndDeserialize(RMSNormBackwardAttributes attrs)
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("RMSNormBackwardIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_RMSNORMBACKWARD_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_RMSNORMBACKWARD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_RMSNORMBACKWARD_TENSOR_SCALE_UID)
            .set_name("scale")
            .set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));

        attrs.set_compute_dbias(true);
        auto [dx, dscale, dbias] = graph->rmsnorm_backward(dy, x, scale, std::move(attrs));
        dx->set_uid(K_RMSNORMBACKWARD_TENSOR_DX_UID).set_output(true).set_name("DX");
        dscale->set_uid(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID).set_output(true).set_name("DScale");
        dbias->set_uid(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID).set_output(true).set_name("DBias");
        auto result = graph->validate();
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph_via_descriptors(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto rawDesc = graph->get_raw_graph_descriptor();
        EXPECT_NE(rawDesc, nullptr);

        size_t serializedSize = 0;
        EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
                  HIPDNN_STATUS_SUCCESS);

        std::vector<uint8_t> serializedData(serializedSize);
        EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                      rawDesc, serializedSize, &serializedSize, serializedData.data()),
                  HIPDNN_STATUS_SUCCESS);

        hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
        hipdnn_flatbuffers_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);
        return graphT;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationRMSNormBackwardDescriptorLowering, RMSNormBackwardLoweringRoundTrip)
{
    RMSNormBackwardAttributes attrs;
    attrs.set_name("test_op");

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors: dy, x, scale, dx, dscale, dbias = 6 tensors
    ASSERT_EQ(graphT.tensors.size(), 6u);

    // Verify tensor attributes
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT*>
        tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->dims,
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->strides,
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->dims,
              toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->strides,
              toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->dims,
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->strides,
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->dims,
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->strides,
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->dims,
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->strides,
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->dims,
              toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->strides,
              toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID]->data_type, DataTypeSdk::FLOAT);

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);

    auto* opNode = node->attributes.AsRMSNormBackwardAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->dy_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(opNode->x_tensor_uid, K_RMSNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(opNode->scale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(opNode->dx_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(opNode->dscale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(opNode->dbias_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DBIAS_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");
}

} // namespace
