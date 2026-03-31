// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/BlockScaleQuantizeNode.hpp>
#include <hipdnn_test_sdk/constants/BlockScaleQuantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;

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

class IntegrationBlockScaleQuantizeDescriptorLifting : public ::testing::Test
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

    // Builds a standard block scale quantize graph for round-trip testing
    static std::shared_ptr<TestableGraph> buildBsqGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("BsqLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BSQ_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_BSQ_TENSOR_X_DIMS)).set_stride(toVec(K_BSQ_TENSOR_X_STRIDES));

        BlockScaleQuantizeAttributes bsqAttrs;
        bsqAttrs.set_name("bsq_op");
        bsqAttrs.set_block_size(K_BSQ_BLOCK_SIZE);

        auto [y, scale] = graph->block_scale_quantize(x, std::move(bsqAttrs));
        y->set_uid(K_BSQ_TENSOR_Y_UID).set_output(true).set_name("Y");
        scale->set_uid(K_BSQ_TENSOR_SCALE_UID).set_output(true).set_name("Scale");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a block scale quantize graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and BSQ parameters.
TEST_F(IntegrationBlockScaleQuantizeDescriptorLifting, BasicBsqRoundTrip)
{
    auto originalGraph = buildBsqGraph();

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
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors (X, Y, Scale) in lifted graph";

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_BSQ_TENSOR_X_UID), 0u);
    auto liftedX = tensorMap[K_BSQ_TENSOR_X_UID];
    EXPECT_EQ(liftedX->get_uid(), K_BSQ_TENSOR_X_UID);
    EXPECT_EQ(liftedX->get_name(), "X");
    EXPECT_EQ(liftedX->get_dim(), toVec(K_BSQ_TENSOR_X_DIMS));
    EXPECT_EQ(liftedX->get_stride(), toVec(K_BSQ_TENSOR_X_STRIDES));
    EXPECT_EQ(liftedX->get_data_type(), DataType::FLOAT);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_BSQ_TENSOR_Y_UID), 0u);
    auto liftedY = tensorMap[K_BSQ_TENSOR_Y_UID];
    EXPECT_EQ(liftedY->get_uid(), K_BSQ_TENSOR_Y_UID);
    EXPECT_EQ(liftedY->get_name(), "Y");
    EXPECT_EQ(liftedY->get_dim(), toVec(K_BSQ_TENSOR_Y_DIMS));
    EXPECT_EQ(liftedY->get_stride(), toVec(K_BSQ_TENSOR_Y_STRIDES));
    EXPECT_EQ(liftedY->get_data_type(), DataType::FLOAT);

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_BSQ_TENSOR_SCALE_UID), 0u);
    auto liftedScale = tensorMap[K_BSQ_TENSOR_SCALE_UID];
    EXPECT_EQ(liftedScale->get_uid(), K_BSQ_TENSOR_SCALE_UID);
    EXPECT_EQ(liftedScale->get_name(), "Scale");

    // Verify 1 sub-node of the correct type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* bsqNode = dynamic_cast<BlockScaleQuantizeNode*>(subNodes[0].get());
    ASSERT_NE(bsqNode, nullptr) << "Expected a BlockScaleQuantizeNode";

    // Verify BSQ parameters
    EXPECT_EQ(bsqNode->attributes.get_block_size(), K_BSQ_BLOCK_SIZE);
    EXPECT_FALSE(bsqNode->attributes.get_axis().has_value());
    EXPECT_FALSE(bsqNode->attributes.get_transpose());
    EXPECT_EQ(bsqNode->attributes.get_name(), "bsq_op");
}

// Verifies that axis and transpose survive the lifting round-trip.
TEST_F(IntegrationBlockScaleQuantizeDescriptorLifting, BsqWithAxisAndTransposePreserved)
{
    const std::vector<int64_t> xDims = {2, 64, 32, 32};
    const std::vector<int64_t> xStrides = {65536, 1024, 32, 1};
    constexpr int32_t BLOCK_SIZE = 32;
    constexpr int64_t X_UID = 50;
    constexpr int64_t Y_UID = 51;
    constexpr int64_t SCALE_UID = 52;

    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("BsqAxisTransposeLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(xDims).set_stride(xStrides);

    BlockScaleQuantizeAttributes bsqAttrs;
    bsqAttrs.set_name("bsq_transpose_op");
    bsqAttrs.set_block_size(BLOCK_SIZE);
    bsqAttrs.set_axis(1);
    bsqAttrs.set_transpose(true);

    auto [y, scale] = graph->block_scale_quantize(x, std::move(bsqAttrs));
    y->set_uid(Y_UID).set_output(true).set_name("Y");
    scale->set_uid(SCALE_UID).set_output(true).set_name("Scale");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* bsqNode = dynamic_cast<BlockScaleQuantizeNode*>(subNodes[0].get());
    ASSERT_NE(bsqNode, nullptr) << "Expected a BlockScaleQuantizeNode";

    EXPECT_EQ(bsqNode->attributes.get_block_size(), BLOCK_SIZE);
    EXPECT_TRUE(bsqNode->attributes.get_axis().has_value());
    EXPECT_EQ(bsqNode->attributes.get_axis().value(), 1);
    EXPECT_TRUE(bsqNode->attributes.get_transpose());
    EXPECT_EQ(bsqNode->attributes.get_name(), "bsq_transpose_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationBlockScaleQuantizeDescriptorLifting, BsqTensorSharingPreserved)
{
    auto originalGraph = buildBsqGraph();

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

    auto* bsqNode = dynamic_cast<BlockScaleQuantizeNode*>(subNodes[0].get());
    ASSERT_NE(bsqNode, nullptr);

    // Verify UIDs match
    EXPECT_EQ(bsqNode->attributes.get_x()->get_uid(), K_BSQ_TENSOR_X_UID);
    EXPECT_EQ(bsqNode->attributes.get_y()->get_uid(), K_BSQ_TENSOR_Y_UID);
    EXPECT_EQ(bsqNode->attributes.get_scale()->get_uid(), K_BSQ_TENSOR_SCALE_UID);

    // Verify tensor names
    EXPECT_EQ(bsqNode->attributes.get_x()->get_name(), "X");
    EXPECT_EQ(bsqNode->attributes.get_y()->get_name(), "Y");
    EXPECT_EQ(bsqNode->attributes.get_scale()->get_name(), "Scale");

    // Verify pointer equality: tensor map and node attributes share the same objects
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_X_UID].get(), bsqNode->attributes.get_x().get());
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_Y_UID].get(), bsqNode->attributes.get_y().get());
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_SCALE_UID].get(), bsqNode->attributes.get_scale().get());
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the round trip and are all distinct.
TEST_F(IntegrationBlockScaleQuantizeDescriptorLifting, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidBsqLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BSQ_TENSOR_X_DIMS)).set_stride(toVec(K_BSQ_TENSOR_X_STRIDES));

    BlockScaleQuantizeAttributes bsqAttrs;
    bsqAttrs.set_block_size(K_BSQ_BLOCK_SIZE);

    auto [y, scale] = graph->block_scale_quantize(x, std::move(bsqAttrs));
    y->set_output(true).set_name("Y");
    scale->set_output(true).set_name("Scale");

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

    auto* bsqNode = dynamic_cast<BlockScaleQuantizeNode*>(subNodes[0].get());
    ASSERT_NE(bsqNode, nullptr);

    // Verify tensor UIDs are distinct
    auto xUid = bsqNode->attributes.get_x()->get_uid();
    auto yUid = bsqNode->attributes.get_y()->get_uid();
    auto scaleUid = bsqNode->attributes.get_scale()->get_uid();

    EXPECT_NE(xUid, yUid);
    EXPECT_NE(xUid, scaleUid);
    EXPECT_NE(yUid, scaleUid);

    // Verify tensor dims survived the round trip
    EXPECT_EQ(tensorMap[xUid]->get_dim(), toVec(K_BSQ_TENSOR_X_DIMS));
}

// Builds a BSQ graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all BSQ fields survive the FlatBuffer-direct path.
TEST_F(IntegrationBlockScaleQuantizeDescriptorLifting, BsqLiftWithoutFinalization)
{
    auto originalGraph = buildBsqGraph();

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

    auto* bsqNode = dynamic_cast<BlockScaleQuantizeNode*>(subNodes[0].get());
    ASSERT_NE(bsqNode, nullptr);

    // Verify BSQ parameters
    EXPECT_EQ(bsqNode->attributes.get_block_size(), K_BSQ_BLOCK_SIZE);
    EXPECT_FALSE(bsqNode->attributes.get_axis().has_value());
    EXPECT_FALSE(bsqNode->attributes.get_transpose());
    EXPECT_EQ(bsqNode->attributes.get_name(), "bsq_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_BSQ_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_X_UID]->get_dim(), toVec(K_BSQ_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_X_UID]->get_stride(), toVec(K_BSQ_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_X_UID]->get_name(), "X");

    ASSERT_NE(tensorMap.count(K_BSQ_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_Y_UID]->get_dim(), toVec(K_BSQ_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_Y_UID]->get_stride(), toVec(K_BSQ_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_Y_UID]->get_name(), "Y");

    ASSERT_NE(tensorMap.count(K_BSQ_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BSQ_TENSOR_SCALE_UID]->get_name(), "Scale");
}

} // namespace
