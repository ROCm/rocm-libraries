// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/constants/PointwiseConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants::integration;

namespace
{

// Exposes protected Graph methods for testing
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

// Builds a conv fprop graph via the frontend, lowers it through the backend C-API
// via build_operation_graph(), then lifts it back with fromBackendDescriptor()
// and verifies the reconstructed graph matches the original.
class IntegrationGraphLifting : public ::testing::Test
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

    // Builds a standard conv fprop graph for round-trip testing
    static std::shared_ptr<TestableGraph>
        buildConvFpropGraph(DataType computeType = DataType::FLOAT,
                            DataType intermediateType = DataType::FLOAT,
                            DataType ioType = DataType::FLOAT)
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("LiftingTestGraph")
            .set_compute_data_type(computeType)
            .set_intermediate_data_type(intermediateType)
            .set_io_data_type(ioType);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));

        auto w = std::make_shared<TensorAttributes>();
        w->set_uid(K_TENSOR_W_UID).set_name("W").set_data_type(DataType::FLOAT);
        w->set_dim(toVec(K_TENSOR_W_DIMS)).set_stride(toVec(K_TENSOR_W_STRIDES));

        ConvFpropAttributes convAttrs;
        convAttrs.set_name("conv_fprop_op");
        convAttrs.set_pre_padding(toVec(K_CONV_PRE_PADDING));
        convAttrs.set_post_padding(toVec(K_CONV_POST_PADDING));
        convAttrs.set_stride(toVec(K_CONV_STRIDE));
        convAttrs.set_dilation(toVec(K_CONV_DILATION));
        convAttrs.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

        auto y = graph->conv_fprop(x, w, convAttrs);
        y->set_uid(K_TENSOR_Y_UID).set_output(true).set_name("Y");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a conv fprop graph, lowers via build_operation_graph(handle), extracts the
// raw descriptor, creates a new graph with fromBackendDescriptor(), and verifies
// tensor dimensions, data types, convolution parameters, and graph-level data types.
TEST_F(IntegrationGraphLifting, ConvFpropRoundTripViaCApi)
{
    auto originalGraph = buildConvFpropGraph();

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

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors (X, W, Y) in lifted graph";

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    auto liftedX = tensorMap[K_TENSOR_X_UID];
    EXPECT_EQ(liftedX->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(liftedX->get_stride(), toVec(K_TENSOR_X_STRIDES));
    EXPECT_EQ(liftedX->get_data_type(), DataType::FLOAT);

    // Verify W tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_W_UID), 0u);
    auto liftedW = tensorMap[K_TENSOR_W_UID];
    EXPECT_EQ(liftedW->get_dim(), toVec(K_TENSOR_W_DIMS));
    EXPECT_EQ(liftedW->get_stride(), toVec(K_TENSOR_W_STRIDES));
    EXPECT_EQ(liftedW->get_data_type(), DataType::FLOAT);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_Y_UID), 0u);
    auto liftedY = tensorMap[K_TENSOR_Y_UID];
    EXPECT_EQ(liftedY->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedY->get_dim(), toVec(K_TENSOR_Y_DIMS));
    EXPECT_EQ(liftedY->get_stride(), toVec(K_TENSOR_Y_STRIDES));

    // Verify the lifted graph has the correct number of sub-nodes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    // Access the conv fprop node and verify convolution parameters
    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr) << "Expected a ConvolutionFpropNode";

    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_CONV_PRE_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_CONV_POST_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_CONV_DILATION));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);
    EXPECT_EQ(convNode->attributes.get_name(), "conv_fprop_op");
}

// Verifies that tensors are accessible by UID on the reconstructed graph,
// confirming tensor identity is preserved through the round-trip.
TEST_F(IntegrationGraphLifting, ConvFpropTensorSharingPreserved)
{
    auto originalGraph = buildConvFpropGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // All tensors should be accessible by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    EXPECT_NE(tensorMap.count(K_TENSOR_X_UID), 0u) << "X tensor not found by UID";
    EXPECT_NE(tensorMap.count(K_TENSOR_W_UID), 0u) << "W tensor not found by UID";
    EXPECT_NE(tensorMap.count(K_TENSOR_Y_UID), 0u) << "Y tensor not found by UID";

    // Verify the node references the same tensor objects via UID
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr);

    EXPECT_EQ(convNode->attributes.get_x()->get_uid(), K_TENSOR_X_UID);
    EXPECT_EQ(convNode->attributes.get_w()->get_uid(), K_TENSOR_W_UID);
    EXPECT_EQ(convNode->attributes.get_y()->get_uid(), K_TENSOR_Y_UID);

    // Verify tensor objects are shared between the tensorMap and conv node attributes
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID].get(), convNode->attributes.get_x().get());
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID].get(), convNode->attributes.get_w().get());
    EXPECT_EQ(tensorMap[K_TENSOR_Y_UID].get(), convNode->attributes.get_y().get());
}

// Builds a graph with set_preferred_engine_id_ext(42), lowers, lifts, and verifies
// that get_preferred_engine_id_ext() returns 42 on the reconstructed graph.
TEST_F(IntegrationGraphLifting, PreferredEngineIdPreservedThroughCApi)
{
    auto originalGraph = buildConvFpropGraph();
    constexpr int64_t K_PREFERRED_ENGINE_ID = 42;
    originalGraph->set_preferred_engine_id_ext(K_PREFERRED_ENGINE_ID);

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto liftedEngineId = liftedGraph->get_preferred_engine_id_ext();
    ASSERT_TRUE(liftedEngineId.has_value()) << "Preferred engine ID should be set after lifting";
    EXPECT_EQ(liftedEngineId.value(), K_PREFERRED_ENGINE_ID);
}

// Verifies that fromBackendDescriptor(nullptr) returns an error.
TEST_F(IntegrationGraphLifting, NullDescriptorReturnsError)
{
    auto graph = std::make_shared<Graph>();
    auto result = graph->fromBackendDescriptor(nullptr);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "fromBackendDescriptor(nullptr) should return INVALID_VALUE";
}

// Builds a graph with FLOAT compute, HALF intermediate, and BFLOAT16 io data types,
// lowers through the C-API, lifts, and verifies all three are preserved.
TEST_F(IntegrationGraphLifting, DataTypesPreservedThroughCApi)
{
    auto originalGraph = buildConvFpropGraph(DataType::FLOAT, DataType::HALF, DataType::BFLOAT16);

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::BFLOAT16);
}

// Builds a conv fprop graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// the reconstructed graph matches the original.
TEST_F(IntegrationGraphLifting, ConvFpropLiftWithoutFinalization)
{
    auto originalGraph = buildConvFpropGraph();

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

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr);

    // Verify convolution parameters
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_CONV_PRE_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_CONV_POST_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_CONV_DILATION));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_stride(), toVec(K_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_dim(), toVec(K_TENSOR_W_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_stride(), toVec(K_TENSOR_W_STRIDES));
}

// Exercises the deserialize_via_backend() path with a handle (full finalization).
// Builds a conv fprop graph, serializes to binary, then uses deserialize_via_backend()
// with a handle and verifies the reconstructed graph.
TEST_F(IntegrationGraphLifting, DeserializeViaBackendWithHandle)
{
    auto originalGraph = buildConvFpropGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create a new graph and use deserialize_via_backend with handle
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->deserialize_via_backend(_handle, data);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr);

    // Verify convolution parameters
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_CONV_PRE_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_CONV_POST_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_CONV_DILATION));

    // Verify tensor dims
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_dim(), toVec(K_TENSOR_W_DIMS));
}

// Exercises the deserialize_via_backend() path without a handle (no finalization).
// Verifies that the graph can be reconstructed from binary data without
// requiring a hipdnnHandle_t.
TEST_F(IntegrationGraphLifting, DeserializeViaBackendWithoutHandle)
{
    auto originalGraph = buildConvFpropGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create a new graph and use deserialize_via_backend without handle
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->deserialize_via_backend(nullptr, data);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors are present
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);
    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_stride(), toVec(K_TENSOR_X_STRIDES));

    ASSERT_NE(tensorMap.count(K_TENSOR_W_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_dim(), toVec(K_TENSOR_W_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_stride(), toVec(K_TENSOR_W_STRIDES));

    ASSERT_NE(tensorMap.count(K_TENSOR_Y_UID), 0u);

    // Verify conv node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr);
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_CONV_PRE_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
}

// Verifies that fromBackendDescriptor returns an error (not a crash) when given
// a graph descriptor with no operations set.
TEST_F(IntegrationGraphLifting, EmptyGraphDescriptorReturnsError)
{
    // Create a backend graph descriptor with no operations
    hipdnnBackendDescriptor_t desc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &desc),
              HIPDNN_STATUS_SUCCESS);

    // Attempt to lift — should return an error since no operations are set
    auto graph = std::make_shared<TestableGraph>();
    auto result = graph->fromBackendDescriptor(desc);
    EXPECT_NE(result.code, ErrorCode::OK)
        << "fromBackendDescriptor should fail on a descriptor with no operations";

    hipdnnBackendDestroyDescriptor(desc);
}

// Verifies that deserialize_via_backend returns an error (not a crash) when
// given corrupt (garbage) bytes.
TEST_F(IntegrationGraphLifting, DeserializeViaBackendCorruptDataReturnsError)
{
    const std::vector<uint8_t> garbage = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};

    auto graph = std::make_shared<TestableGraph>();
    auto result = graph->deserialize_via_backend(_handle, garbage);
    EXPECT_NE(result.code, ErrorCode::OK) << "deserialize_via_backend should fail on corrupt data";
}

// Verifies that deserialize_via_backend returns an error (not a crash) when
// given an empty data vector.
TEST_F(IntegrationGraphLifting, DeserializeViaBackendEmptyDataReturnsError)
{
    const std::vector<uint8_t> empty;

    auto graph = std::make_shared<TestableGraph>();
    auto result = graph->deserialize_via_backend(_handle, empty);
    EXPECT_NE(result.code, ErrorCode::OK) << "deserialize_via_backend should fail on empty data";
}

// Verifies that the graph name survives the C-API round-trip (lower -> lift).
TEST_F(IntegrationGraphLifting, GraphNamePreservedThroughCApi)
{
    auto originalGraph = buildConvFpropGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_EQ(liftedGraph->get_name(), "LiftingTestGraph");
}

// Exercises the deserialize_via_backend() path and verifies the graph name
// is preserved through the FlatBuffer-direct deserialization path.
TEST_F(IntegrationGraphLifting, GraphNamePreservedThroughDeserializeViaBackend)
{
    auto originalGraph = buildConvFpropGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create a new graph and use deserialize_via_backend without handle
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->deserialize_via_backend(nullptr, data);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_EQ(liftedGraph->get_name(), "LiftingTestGraph");
}

// ============================================================================
// Pointwise Lifting Tests
// ============================================================================

// Builds a binary pointwise (ADD) graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and verifies operation mode, tensors,
// and graph-level data types.
TEST_F(IntegrationGraphLifting, PointwiseBinaryAddRoundTripViaCApi)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PointwiseAddLiftingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    auto in1 = std::make_shared<TensorAttributes>();
    in1->set_uid(K_PW_TENSOR_IN1_UID).set_name("IN1").set_data_type(DataType::FLOAT);
    in1->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("add_op");
    pwAttrs.set_mode(PointwiseMode::ADD);

    auto out0 = graph->pointwise(in0, in1, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    // Lift back into a new graph
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID (in0, in1, out0)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN1_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_OUT0_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 pointwise sub-node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr) << "Expected a PointwiseNode";

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::ADD);
    EXPECT_EQ(pwNode->attributes.get_name(), "add_op");
    EXPECT_EQ(pwNode->attributes.get_input_0()->get_uid(), K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(pwNode->attributes.get_input_1()->get_uid(), K_PW_TENSOR_IN1_UID);
    EXPECT_EQ(pwNode->attributes.get_output_0()->get_uid(), K_PW_TENSOR_OUT0_UID);
}

// Builds a unary pointwise (RELU_FWD) graph with scalar attributes,
// lowers, lifts, and verifies mode + activation parameters.
TEST_F(IntegrationGraphLifting, PointwiseUnaryReluScalarsPreserved)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PointwiseReluLiftingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    constexpr float K_LOWER_CLIP = -1.0F;
    constexpr float K_UPPER_CLIP = 6.0F;
    constexpr float K_LOWER_CLIP_SLOPE = 0.01F;

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("leaky_relu_op");
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    pwAttrs.set_relu_lower_clip(K_LOWER_CLIP);
    pwAttrs.set_relu_upper_clip(K_UPPER_CLIP);
    pwAttrs.set_relu_lower_clip_slope(K_LOWER_CLIP_SLOPE);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

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

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr) << "Expected a PointwiseNode";

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::RELU_FWD);

    ASSERT_TRUE(pwNode->attributes.get_relu_lower_clip().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_relu_lower_clip().value(), K_LOWER_CLIP);

    ASSERT_TRUE(pwNode->attributes.get_relu_upper_clip().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_relu_upper_clip().value(), K_UPPER_CLIP);

    ASSERT_TRUE(pwNode->attributes.get_relu_lower_clip_slope().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_relu_lower_clip_slope().value(), K_LOWER_CLIP_SLOPE);

    // Verify tensor UIDs
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);
    EXPECT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    EXPECT_NE(tensorMap.count(K_PW_TENSOR_OUT0_UID), 0u);
}

// Builds a ternary pointwise (BINARY_SELECT) graph, lowers, lifts,
// and verifies all three inputs and the output are correctly reconstructed.
TEST_F(IntegrationGraphLifting, PointwiseTernarySelectRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PointwiseTernaryLiftingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    auto in1 = std::make_shared<TensorAttributes>();
    in1->set_uid(K_PW_TENSOR_IN1_UID).set_name("IN1").set_data_type(DataType::FLOAT);
    in1->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    auto in2 = std::make_shared<TensorAttributes>();
    in2->set_uid(K_PW_TENSOR_IN2_UID).set_name("IN2").set_data_type(DataType::FLOAT);
    in2->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("select_op");
    pwAttrs.set_mode(PointwiseMode::BINARY_SELECT);

    auto out0 = graph->pointwise(in0, in1, in2, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

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
    ASSERT_EQ(tensorMap.size(), 4u);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN1_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN2_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN2_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN2_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN2_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_OUT0_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_data_type(), DataType::FLOAT);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr) << "Expected a PointwiseNode";

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::BINARY_SELECT);
    EXPECT_EQ(pwNode->attributes.get_input_0()->get_uid(), K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(pwNode->attributes.get_input_1()->get_uid(), K_PW_TENSOR_IN1_UID);
    EXPECT_EQ(pwNode->attributes.get_input_2()->get_uid(), K_PW_TENSOR_IN2_UID);
    EXPECT_EQ(pwNode->attributes.get_output_0()->get_uid(), K_PW_TENSOR_OUT0_UID);
}

// Builds a pointwise graph, serializes to binary, creates a backend descriptor
// from bytes (no handle), calls fromBackendDescriptor(), and verifies the
// pointwise operation survives the FlatBuffer-direct deserialization path.
TEST_F(IntegrationGraphLifting, PointwiseLiftWithoutFinalization)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PointwiseFlatBufferLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    auto in1 = std::make_shared<TensorAttributes>();
    in1->set_uid(K_PW_TENSOR_IN1_UID).set_name("IN1").set_data_type(DataType::FLOAT);
    in1->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("mul_op");
    pwAttrs.set_mode(PointwiseMode::MUL);

    auto out0 = graph->pointwise(in0, in1, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to binary
    auto data = graph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create backend descriptor from bytes (no handle, no finalize)
    const detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    ASSERT_TRUE(graphDesc.valid()) << "Failed to create backend graph descriptor";

    // Lift into a new graph
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(graphDesc.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr) << "Expected a PointwiseNode";

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::MUL);
    EXPECT_EQ(pwNode->attributes.get_input_0()->get_uid(), K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(pwNode->attributes.get_input_1()->get_uid(), K_PW_TENSOR_IN1_UID);
    EXPECT_EQ(pwNode->attributes.get_output_0()->get_uid(), K_PW_TENSOR_OUT0_UID);
}

// Verifies that tensor objects are shared between the tensor map and the
// pointwise node attributes after lifting.
TEST_F(IntegrationGraphLifting, PointwiseTensorSharingPreserved)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PointwiseTensorSharingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("relu_op");
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

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
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr);

    // Verify tensor objects are shared (same pointer) between map and node
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID].get(), pwNode->attributes.get_input_0().get());
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID].get(), pwNode->attributes.get_output_0().get());
}

// Builds a unary pointwise (SWISH_FWD) graph with swish_beta scalar,
// lowers, lifts, and verifies the scalar survives the round trip.
TEST_F(IntegrationGraphLifting, PointwiseSwishBetaPreserved)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PointwiseSwishBetaLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("swish_op");
    pwAttrs.set_mode(PointwiseMode::SWISH_FWD);
    pwAttrs.set_swish_beta(0.5F);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

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

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr) << "Expected a PointwiseNode";

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::SWISH_FWD);
    EXPECT_TRUE(pwNode->attributes.get_swish_beta().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_swish_beta().value(), 0.5F);
}

// Builds a unary pointwise (ELU_FWD) graph with elu_alpha scalar,
// lowers, lifts, and verifies the scalar survives the round trip.
TEST_F(IntegrationGraphLifting, PointwiseEluAlphaPreserved)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PointwiseEluAlphaLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("elu_op");
    pwAttrs.set_mode(PointwiseMode::ELU_FWD);
    pwAttrs.set_elu_alpha(1.0F);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

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

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr) << "Expected a PointwiseNode";

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::ELU_FWD);
    EXPECT_TRUE(pwNode->attributes.get_elu_alpha().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_elu_alpha().value(), 1.0F);
}

// Builds a unary pointwise (SOFTPLUS_FWD) graph with softplus_beta scalar,
// lowers, lifts, and verifies the scalar survives the round trip.
TEST_F(IntegrationGraphLifting, PointwiseSoftplusBetaPreserved)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PointwiseSoftplusBetaLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("softplus_op");
    pwAttrs.set_mode(PointwiseMode::SOFTPLUS_FWD);
    pwAttrs.set_softplus_beta(2.0F);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

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

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr) << "Expected a PointwiseNode";

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::SOFTPLUS_FWD);
    EXPECT_TRUE(pwNode->attributes.get_softplus_beta().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_softplus_beta().value(), 2.0F);
}

// Builds a unary pointwise (GEN_INDEX) graph with axis scalar,
// lowers, lifts, and verifies the scalar survives the round trip.
TEST_F(IntegrationGraphLifting, PointwiseGenIndexAxisPreserved)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PointwiseGenIndexAxisLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("gen_index_op");
    pwAttrs.set_mode(PointwiseMode::GEN_INDEX);
    pwAttrs.set_axis(2);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

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

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr) << "Expected a PointwiseNode";

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::GEN_INDEX);
    EXPECT_TRUE(pwNode->attributes.get_axis().has_value());
    EXPECT_EQ(pwNode->attributes.get_axis().value(), 2);
}

// Builds a conv_fprop + pointwise RELU fusion graph, lowers via the C-API,
// lifts back, and verifies both operation nodes and the shared virtual tensor.
TEST_F(IntegrationGraphLifting, ConvFpropReluFusionRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("ConvReluFusionTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Conv fprop inputs
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_TENSOR_W_UID).set_name("W").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_TENSOR_W_DIMS)).set_stride(toVec(K_TENSOR_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("conv_fprop_op");
    convAttrs.set_pre_padding(toVec(K_CONV_PRE_PADDING));
    convAttrs.set_post_padding(toVec(K_CONV_POST_PADDING));
    convAttrs.set_stride(toVec(K_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_CONV_DILATION));
    convAttrs.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    // Conv output y is a virtual intermediate — no UID, not an output
    auto y = graph->conv_fprop(x, w, convAttrs);

    // Pointwise RELU on the conv output
    PointwiseAttributes reluAttrs;
    reluAttrs.set_name("relu_activation");
    reluAttrs.set_mode(PointwiseMode::RELU_FWD);

    auto reluOut = graph->pointwise(y, reluAttrs);
    reluOut->set_uid(K_PW_RELU_OUT_UID).set_output(true).set_name("relu_out");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    // Lift back into a new graph
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify 2 operation nodes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 2u) << "Expected 2 operation nodes (conv + relu)";

    // First node: ConvolutionFpropNode
    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr) << "Expected first node to be ConvolutionFpropNode";
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_CONV_PRE_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_CONV_POST_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_CONV_DILATION));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);
    EXPECT_EQ(convNode->attributes.get_name(), "conv_fprop_op");

    // Second node: PointwiseNode with RELU_FWD
    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[1].get());
    ASSERT_NE(pwNode, nullptr) << "Expected second node to be PointwiseNode";
    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::RELU_FWD);
    EXPECT_EQ(pwNode->attributes.get_name(), "relu_activation");

    // Verify tensor sharing: conv output and relu input share the same TensorAttributes
    auto convY = convNode->attributes.get_y();
    auto reluIn0 = pwNode->attributes.get_input_0();
    EXPECT_EQ(convY.get(), reluIn0.get())
        << "Conv output and relu input should share the same TensorAttributes object";

    // Verify tensor map contains external tensors (X, W, relu_out) plus the virtual intermediate
    auto tensorMap = liftedGraph->getTensorsByUid();
    EXPECT_NE(tensorMap.count(K_TENSOR_X_UID), 0u) << "X tensor not found";
    EXPECT_NE(tensorMap.count(K_TENSOR_W_UID), 0u) << "W tensor not found";
    EXPECT_NE(tensorMap.count(K_PW_RELU_OUT_UID), 0u) << "relu_out tensor not found";
}

} // namespace
