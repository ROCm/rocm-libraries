// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "GraphTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionFwdOperationDescriptor.hpp"
#include "descriptors/DataTypeConversion.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestGraphDescriptorLifting : public ::testing::Test
{
public:
    // Build a graph via operations, finalize, and return serialized bytes
    std::vector<uint8_t> buildAndSerializeGraph(const std::vector<HipdnnBackendDescriptor*>& ops,
                                                hipdnnDataType_t computeDt = HIPDNN_DATA_FLOAT,
                                                hipdnnDataType_t intermediateDt = HIPDNN_DATA_FLOAT,
                                                hipdnnDataType_t ioDt = HIPDNN_DATA_FLOAT,
                                                std::optional<int64_t> preferredEngineId
                                                = std::nullopt,
                                                const std::string& name = "")
    {
        auto graphWrapper = createDescriptor<GraphDescriptor>();
        auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

        hipdnnHandle_t handle = &_mockHandle;
        graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                HIPDNN_TYPE_HANDLE,
                                1,
                                static_cast<const void*>(&handle));

        graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                static_cast<int64_t>(ops.size()),
                                static_cast<const void*>(ops.data()));

        graphDesc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeDt);
        graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                                HIPDNN_TYPE_DATA_TYPE,
                                1,
                                &intermediateDt);
        graphDesc->setAttribute(
            HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &ioDt);

        if(preferredEngineId.has_value())
        {
            int64_t engineId = preferredEngineId.value();
            graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    &engineId);
        }

        if(!name.empty())
        {
            graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(name.size()),
                                    name.c_str());
        }

        graphDesc->finalize();

        auto serialized = graphDesc->getSerializedGraph();
        return {static_cast<const uint8_t*>(serialized.ptr),
                static_cast<const uint8_t*>(serialized.ptr) + serialized.size};
    }

    // Create a new GraphDescriptor from serialized bytes and finalize
    std::shared_ptr<GraphDescriptor> deserializeAndFinalize(const std::vector<uint8_t>& bytes)
    {
        auto graphWrapper = createDescriptor<GraphDescriptor>();
        auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

        graphDesc->deserializeGraph(bytes.data(), bytes.size());

        hipdnnHandle_t handle = &_mockHandle;
        graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                HIPDNN_TYPE_HANDLE,
                                1,
                                static_cast<const void*>(&handle));
        graphDesc->finalize();

        return graphDesc;
    }

protected:
    mutable MockHandle _mockHandle;
};

// =============================================================================
// Deserialization/Lifting Round-Trip Tests
// =============================================================================

TEST_F(TestGraphDescriptorLifting, DeserializePreservesConvFpropNode)
{
    auto conv = createDefaultConvOp();

    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto serializedBytes = buildAndSerializeGraph(ops);

    // Deserialize and finalize
    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    // Verify via the serialized buffer that the node is preserved
    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::ConvolutionFwdAttributes);
    ASSERT_EQ(graphT->tensors.size(), 3);

    const auto* convAttrs = graphT->nodes[0]->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(convAttrs, nullptr);
    EXPECT_EQ(convAttrs->x_tensor_uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(convAttrs->w_tensor_uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(convAttrs->y_tensor_uid, K_FPROP_TENSOR_Y_UID);
    EXPECT_EQ(convAttrs->pre_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convAttrs->post_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convAttrs->stride, toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(convAttrs->dilation, toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestGraphDescriptorLifting, DeserializePreservesMultipleNodes)
{
    auto conv1 = createDefaultConvOp();

    auto xDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_X2_UID, toVec(K_FPROP_TENSOR_X2_DIMS), toVec(K_FPROP_TENSOR_X2_STRIDES));
    auto wDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_W2_UID, toVec(K_FPROP_TENSOR_W2_DIMS), toVec(K_FPROP_TENSOR_W2_STRIDES));
    auto yDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_Y2_UID, toVec(K_FPROP_TENSOR_Y2_DIMS), toVec(K_FPROP_TENSOR_Y2_STRIDES));
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), yDesc2.get());

    const std::vector<HipdnnBackendDescriptor*> ops = {conv1.convOp.get(), convOp2.get()};
    auto serializedBytes = buildAndSerializeGraph(ops);

    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    // Verify via the serialized buffer
    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_EQ(graphT->nodes.size(), 2);
    ASSERT_EQ(graphT->tensors.size(), 6);
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::ConvolutionFwdAttributes);
    ASSERT_EQ(graphT->nodes[1]->attributes.type, NodeAttributes::ConvolutionFwdAttributes);

    const auto* conv1Attrs = graphT->nodes[0]->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(conv1Attrs, nullptr);
    EXPECT_EQ(conv1Attrs->x_tensor_uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(conv1Attrs->w_tensor_uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(conv1Attrs->y_tensor_uid, K_FPROP_TENSOR_Y_UID);

    const auto* conv2Attrs = graphT->nodes[1]->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(conv2Attrs, nullptr);
    EXPECT_EQ(conv2Attrs->x_tensor_uid, K_FPROP_TENSOR_X2_UID);
    EXPECT_EQ(conv2Attrs->w_tensor_uid, K_FPROP_TENSOR_W2_UID);
    EXPECT_EQ(conv2Attrs->y_tensor_uid, K_FPROP_TENSOR_Y2_UID);
}

TEST_F(TestGraphDescriptorLifting, DeserializePreservesTensorData)
{
    auto conv = createDefaultConvOp();

    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto serializedBytes = buildAndSerializeGraph(ops);

    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    // Verify tensor data via the serialized buffer
    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Find the X tensor by UID and verify its attributes
    auto* xTensor = findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID);
    ASSERT_NE(xTensor, nullptr);
    EXPECT_EQ(xTensor->data_type, DataType::FLOAT);
    EXPECT_EQ(xTensor->dims, toVec(K_FPROP_TENSOR_X_DIMS));
    EXPECT_EQ(xTensor->strides, toVec(K_FPROP_TENSOR_X_STRIDES));

    // Find the W tensor by UID and verify its attributes
    auto* wTensor = findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID);
    ASSERT_NE(wTensor, nullptr);
    EXPECT_EQ(wTensor->data_type, DataType::FLOAT);
    EXPECT_EQ(wTensor->dims, toVec(K_FPROP_TENSOR_W_DIMS));
    EXPECT_EQ(wTensor->strides, toVec(K_FPROP_TENSOR_W_STRIDES));

    // Find the Y tensor by UID and verify its attributes
    auto* yTensor = findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID);
    ASSERT_NE(yTensor, nullptr);
    EXPECT_EQ(yTensor->data_type, DataType::FLOAT);
    EXPECT_EQ(yTensor->dims, toVec(K_FPROP_TENSOR_Y_DIMS));
    EXPECT_EQ(yTensor->strides, toVec(K_FPROP_TENSOR_Y_STRIDES));
}

TEST_F(TestGraphDescriptorLifting, DeserializePreservesGraphLevelAttributes)
{
    auto conv = createDefaultConvOp();

    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto serializedBytes = buildAndSerializeGraph(
        ops, HIPDNN_DATA_HALF, HIPDNN_DATA_BFLOAT16, HIPDNN_DATA_FLOAT, 42);

    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    // Query compute data type
    int64_t computeCount = 0;
    hipdnnDataType_t computeDt = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(liftedGraph->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT,
                                              HIPDNN_TYPE_DATA_TYPE,
                                              1,
                                              &computeCount,
                                              &computeDt));
    EXPECT_EQ(computeDt, HIPDNN_DATA_HALF);

    // Query intermediate data type
    int64_t intermediateCount = 0;
    hipdnnDataType_t intermediateDt = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(liftedGraph->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                                              HIPDNN_TYPE_DATA_TYPE,
                                              1,
                                              &intermediateCount,
                                              &intermediateDt));
    EXPECT_EQ(intermediateDt, HIPDNN_DATA_BFLOAT16);

    // Query IO data type
    int64_t ioCount = 0;
    hipdnnDataType_t ioDt = HIPDNN_DATA_HALF;
    ASSERT_NO_THROW(liftedGraph->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &ioCount, &ioDt));
    EXPECT_EQ(ioDt, HIPDNN_DATA_FLOAT);

    // Query preferred engine ID
    int64_t engineIdCount = 0;
    int64_t engineId = 0;
    ASSERT_NO_THROW(liftedGraph->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT,
                                              HIPDNN_TYPE_INT64,
                                              1,
                                              &engineIdCount,
                                              &engineId));
    EXPECT_EQ(engineIdCount, 1);
    EXPECT_EQ(engineId, 42);
}

TEST_F(TestGraphDescriptorLifting, LiftedGraphSerializesToSameBinary)
{
    auto conv = createDefaultConvOp();

    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto originalBytes = buildAndSerializeGraph(
        ops, HIPDNN_DATA_HALF, HIPDNN_DATA_BFLOAT16, HIPDNN_DATA_FLOAT, 99);

    // Deserialize and finalize (first lift)
    auto liftedGraph = deserializeAndFinalize(originalBytes);

    // Re-serialize the lifted graph
    auto reSerializedData = liftedGraph->getSerializedGraph();
    const std::vector<uint8_t> reSerializedBytes(static_cast<const uint8_t*>(reSerializedData.ptr),
                                                 static_cast<const uint8_t*>(reSerializedData.ptr)
                                                     + reSerializedData.size);

    // Compare binary output
    ASSERT_EQ(originalBytes.size(), reSerializedBytes.size());
    EXPECT_EQ(originalBytes, reSerializedBytes);
}

TEST_F(TestGraphDescriptorLifting, DeserializePreservesNodeTensorUids)
{
    auto conv = createDefaultConvOp();

    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto serializedBytes = buildAndSerializeGraph(ops);

    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    // Verify node and tensor data via the serialized graph buffer
    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify the node has ConvolutionFwdAttributes with correct tensor UIDs
    const auto* convAttrs = graphT->nodes[0]->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(convAttrs, nullptr);
    EXPECT_EQ(convAttrs->x_tensor_uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(convAttrs->w_tensor_uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(convAttrs->y_tensor_uid, K_FPROP_TENSOR_Y_UID);

    // Verify the tensor UIDs in the tensor list
    std::set<int64_t> tensorUids;
    for(const auto& tensor : graphT->tensors)
    {
        tensorUids.insert(tensor->uid);
    }
    EXPECT_EQ(tensorUids.count(K_FPROP_TENSOR_X_UID), 1u);
    EXPECT_EQ(tensorUids.count(K_FPROP_TENSOR_W_UID), 1u);
    EXPECT_EQ(tensorUids.count(K_FPROP_TENSOR_Y_UID), 1u);
    EXPECT_EQ(tensorUids.size(), 3u);
}

TEST_F(TestGraphDescriptorLifting, DeserializePreservesConvolutionParameters)
{
    // Create a conv op with non-default parameters
    auto xDesc = createFinalizedTensor(K_FPROP_TENSOR_X_UID);
    auto wDesc = createFinalizedTensor(
        K_FPROP_TENSOR_W_UID, toVec(K_FPROP_TENSOR_W_DIMS), toVec(K_FPROP_TENSOR_W_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_FPROP_TENSOR_Y_UID, toVec(K_FPROP_TENSOR_Y_DIMS), toVec(K_FPROP_TENSOR_Y_STRIDES));

    auto convWrapper = createDescriptor<ConvolutionFwdOperationDescriptor>();
    auto convDesc = convWrapper->asDescriptor<ConvolutionFwdOperationDescriptor>();

    HipdnnBackendDescriptor* x = xDesc.get();
    HipdnnBackendDescriptor* w = wDesc.get();
    HipdnnBackendDescriptor* y = yDesc.get();

    convDesc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&x));
    convDesc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&w));
    convDesc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&y));

    const std::vector<int64_t> kCustomPrePadding = {2, 3};
    const std::vector<int64_t> kCustomPostPadding = {4, 5};
    const std::vector<int64_t> kCustomStride = {2, 2};
    const std::vector<int64_t> kCustomDilation = {2, 2};

    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, kCustomPrePadding.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, kCustomPostPadding.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, kCustomStride.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, kCustomDilation.data());

    auto computeType = HIPDNN_DATA_HALF;
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    hipdnnConvolutionMode_t convMode = HIPDNN_CROSS_CORRELATION;
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);
    convDesc->finalize();

    const std::vector<HipdnnBackendDescriptor*> ops = {convWrapper.get()};
    auto serializedBytes = buildAndSerializeGraph(ops);

    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    // Verify convolution parameters via the serialized graph buffer
    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_EQ(graphT->nodes.size(), 1);

    const auto* convAttrs = graphT->nodes[0]->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(convAttrs, nullptr);

    EXPECT_EQ(convAttrs->pre_padding, kCustomPrePadding);
    EXPECT_EQ(convAttrs->post_padding, kCustomPostPadding);
    EXPECT_EQ(convAttrs->stride, kCustomStride);
    EXPECT_EQ(convAttrs->dilation, kCustomDilation);
    EXPECT_EQ(convAttrs->conv_mode, ConvMode::CROSS_CORRELATION);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorLifting, DeserializePreservesConvolutionModeConvolution)
{
    // Verify HIPDNN_CONVOLUTION round-trips correctly
    // (contrasting with CROSS_CORRELATION tested in DeserializePreservesConvolutionParameters)
    auto xDesc = createFinalizedTensor(K_FPROP_TENSOR_X_UID);
    auto wDesc = createFinalizedTensor(
        K_FPROP_TENSOR_W_UID, toVec(K_FPROP_TENSOR_W_DIMS), toVec(K_FPROP_TENSOR_W_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_FPROP_TENSOR_Y_UID, toVec(K_FPROP_TENSOR_Y_DIMS), toVec(K_FPROP_TENSOR_Y_STRIDES));

    auto convWrapper = createDescriptor<ConvolutionFwdOperationDescriptor>();
    auto convDesc = convWrapper->asDescriptor<ConvolutionFwdOperationDescriptor>();

    HipdnnBackendDescriptor* x = xDesc.get();
    HipdnnBackendDescriptor* w = wDesc.get();
    HipdnnBackendDescriptor* y = yDesc.get();

    convDesc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&x));
    convDesc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&w));
    convDesc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&y));

    const std::vector<int64_t> kPadding = {1, 1};
    const std::vector<int64_t> kStride = {1, 1};
    const std::vector<int64_t> kDilation = {1, 1};

    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, kPadding.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, kPadding.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, kStride.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, kDilation.data());

    auto computeType = HIPDNN_DATA_FLOAT;
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Use CONVOLUTION mode (not CROSS_CORRELATION)
    hipdnnConvolutionMode_t convMode = HIPDNN_CONVOLUTION;
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);
    convDesc->finalize();

    const std::vector<HipdnnBackendDescriptor*> ops = {convWrapper.get()};
    auto serializedBytes = buildAndSerializeGraph(ops);

    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    // Verify convolution mode via the serialized graph buffer
    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_EQ(graphT->nodes.size(), 1);

    const auto* convAttrs = graphT->nodes[0]->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(convAttrs, nullptr);
    EXPECT_EQ(convAttrs->conv_mode, ConvMode::CONVOLUTION);
}

TEST_F(TestGraphDescriptorLifting, DoubleRoundTrip)
{
    // Build -> Serialize -> Deserialize -> Serialize -> Deserialize -> Verify
    auto conv = createDefaultConvOp();

    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto originalBytes = buildAndSerializeGraph(
        ops, HIPDNN_DATA_HALF, HIPDNN_DATA_BFLOAT16, HIPDNN_DATA_FLOAT, 77);

    // First round-trip
    auto liftedGraph1 = deserializeAndFinalize(originalBytes);
    auto serialized1 = liftedGraph1->getSerializedGraph();
    const std::vector<uint8_t> bytes1(static_cast<const uint8_t*>(serialized1.ptr),
                                      static_cast<const uint8_t*>(serialized1.ptr)
                                          + serialized1.size);

    // Second round-trip
    auto liftedGraph2 = deserializeAndFinalize(bytes1);
    auto serialized2 = liftedGraph2->getSerializedGraph();
    const std::vector<uint8_t> bytes2(static_cast<const uint8_t*>(serialized2.ptr),
                                      static_cast<const uint8_t*>(serialized2.ptr)
                                          + serialized2.size);

    // All three should be identical
    EXPECT_EQ(originalBytes, bytes1);
    EXPECT_EQ(bytes1, bytes2);

    // Verify graph-level attributes on final lift
    int64_t engineIdCount = 0;
    int64_t engineId = 0;
    liftedGraph2->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT,
                               HIPDNN_TYPE_INT64,
                               1,
                               &engineIdCount,
                               &engineId);
    EXPECT_EQ(engineId, 77);

    // Verify the serialized graph still has 1 node
    auto graphT = UnPackGraph(serialized2.ptr);
    EXPECT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::ConvolutionFwdAttributes);

    // Verify graph-level data types from the serialized buffer
    EXPECT_EQ(graphT->compute_data_type, DataType::HALF);
    EXPECT_EQ(graphT->intermediate_data_type, DataType::BFLOAT16);
    EXPECT_EQ(graphT->io_data_type, DataType::FLOAT);
}

// =============================================================================
// Cross-flow Tests (FlatBuffer + C-API interoperability)
// =============================================================================

TEST_F(TestGraphDescriptorLifting, SetOperationsAfterDeserializeSucceeds)
{
    // Build and serialize a single conv op graph
    auto conv1 = createDefaultConvOp();
    const std::vector<HipdnnBackendDescriptor*> ops = {conv1.convOp.get()};
    auto serializedBytes = buildAndSerializeGraph(ops);

    // Create a new GraphDescriptor and deserialize the bytes
    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();
    graphDesc->deserializeGraph(serializedBytes.data(), serializedBytes.size());

    hipdnnHandle_t handle = &_mockHandle;
    ASSERT_NO_THROW(graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                            HIPDNN_TYPE_HANDLE,
                                            1,
                                            static_cast<const void*>(&handle)));

    // Adding operations after deserialization appends to the existing operations
    auto conv2 = createDefaultConvOp(HIPDNN_DATA_HALF);
    HipdnnBackendDescriptor* op2Ptr = conv2.convOp.get();
    ASSERT_NO_THROW(graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            static_cast<const void*>(&op2Ptr)));

    // Verify the operations count includes both deserialized and appended ops
    int64_t elementCount = 0;
    ASSERT_NO_THROW(graphDesc->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 2);
}

TEST_F(TestGraphDescriptorLifting, GetAttributeOpsAfterDeserializeSucceeds)
{
    // Build and serialize a single conv op graph
    auto conv1 = createDefaultConvOp();
    const std::vector<HipdnnBackendDescriptor*> ops = {conv1.convOp.get()};
    auto serializedBytes = buildAndSerializeGraph(ops);

    // Deserialize and finalize (FlatBuffer flow)
    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    // Lazy unpack: getOperations() unpacks the FlatBuffer nodes into _operations
    int64_t elementCount = -1;
    ASSERT_NO_THROW(liftedGraph->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 1);
}

TEST_F(TestGraphDescriptorLifting, DeserializeOnlyFinalize)
{
    // Build and serialize a graph with specific graph-level attributes
    auto conv = createDefaultConvOp();
    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto originalBytes = buildAndSerializeGraph(
        ops, HIPDNN_DATA_HALF, HIPDNN_DATA_BFLOAT16, HIPDNN_DATA_FLOAT, 42);

    // Create new GraphDescriptor, deserialize, set handle, finalize (no operations added)
    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();
    graphDesc->deserializeGraph(originalBytes.data(), originalBytes.size());

    hipdnnHandle_t handle = &_mockHandle;
    ASSERT_NO_THROW(graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                            HIPDNN_TYPE_HANDLE,
                                            1,
                                            static_cast<const void*>(&handle)));
    ASSERT_NO_THROW(graphDesc->finalize());

    // Get the re-serialized bytes
    auto serialized = graphDesc->getSerializedGraph();
    const std::vector<uint8_t> reSerializedBytes(static_cast<const uint8_t*>(serialized.ptr),
                                                 static_cast<const uint8_t*>(serialized.ptr)
                                                     + serialized.size);

    // Re-serializing from unpacked operations should produce identical bytes
    ASSERT_EQ(originalBytes.size(), reSerializedBytes.size());
    EXPECT_EQ(originalBytes, reSerializedBytes);
}

TEST_F(TestGraphDescriptorLifting, FlatBufferFlowFinalizePreservesSerialization)
{
    // Build and serialize a single conv op graph
    auto conv1 = createDefaultConvOp();
    const std::vector<HipdnnBackendDescriptor*> ops = {conv1.convOp.get()};
    auto originalBytes = buildAndSerializeGraph(ops);

    // Deserialize, set handle, finalize (FlatBuffer flow)
    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();
    graphDesc->deserializeGraph(originalBytes.data(), originalBytes.size());

    hipdnnHandle_t handle = &_mockHandle;
    ASSERT_NO_THROW(graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                            HIPDNN_TYPE_HANDLE,
                                            1,
                                            static_cast<const void*>(&handle)));
    ASSERT_NO_THROW(graphDesc->finalize());

    // Get serialized bytes — they should be identical to the original
    auto serialized = graphDesc->getSerializedGraph();
    const std::vector<uint8_t> newBytes(static_cast<const uint8_t*>(serialized.ptr),
                                        static_cast<const uint8_t*>(serialized.ptr)
                                            + serialized.size);
    EXPECT_EQ(originalBytes, newBytes);

    // Verify the serialized graph content
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::ConvolutionFwdAttributes);

    // Spot-check tensor dims
    auto* xTensor = findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID);
    ASSERT_NE(xTensor, nullptr);
    EXPECT_EQ(xTensor->dims, toVec(K_FPROP_TENSOR_X_DIMS));
}

TEST_F(TestGraphDescriptorLifting, GetAttributeWrongTypeForOpsOnCApiFlow)
{
    // Build a graph via the C-API flow
    auto conv = createDefaultConvOp();

    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

    hipdnnHandle_t handle = &_mockHandle;
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                            HIPDNN_TYPE_HANDLE,
                            1,
                            static_cast<const void*>(&handle));

    HipdnnBackendDescriptor* opPtr = conv.convOp.get();
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&opPtr));
    graphDesc->finalize();

    // Call getAttribute with HIPDNN_ATTR_OPERATIONGRAPH_OPS but pass HIPDNN_TYPE_INT64
    // instead of HIPDNN_TYPE_BACKEND_DESCRIPTOR
    int64_t elementCount = 0;
    std::array<int64_t, 1> wrongTypeBuffer = {0};
    ASSERT_THROW_HIPDNN_STATUS(graphDesc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                                       HIPDNN_TYPE_INT64,
                                                       1,
                                                       &elementCount,
                                                       wrongTypeBuffer.data()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptorLifting, GetAttributeRequestedCountTooSmallOnCApiFlow)
{
    // Build a graph with 2 conv ops via the C-API flow
    auto conv1 = createDefaultConvOp();

    auto xDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_X2_UID, toVec(K_FPROP_TENSOR_X2_DIMS), toVec(K_FPROP_TENSOR_X2_STRIDES));
    auto wDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_W2_UID, toVec(K_FPROP_TENSOR_W2_DIMS), toVec(K_FPROP_TENSOR_W2_STRIDES));
    auto yDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_Y2_UID, toVec(K_FPROP_TENSOR_Y2_DIMS), toVec(K_FPROP_TENSOR_Y2_STRIDES));
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), yDesc2.get());

    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

    hipdnnHandle_t handle = &_mockHandle;
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                            HIPDNN_TYPE_HANDLE,
                            1,
                            static_cast<const void*>(&handle));

    HipdnnBackendDescriptor* op1Ptr = conv1.convOp.get();
    HipdnnBackendDescriptor* op2Ptr = convOp2.get();
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&op1Ptr));
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&op2Ptr));
    graphDesc->finalize();

    // Allocate a buffer for 2 ops, but pass requestedElementCount=1
    // with a non-null arrayOfElements
    int64_t elementCount = 0;
    std::array<HipdnnBackendDescriptor*, 2> returnedOps = {nullptr, nullptr};
    ASSERT_THROW_HIPDNN_STATUS(graphDesc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &elementCount,
                                                       returnedOps.data()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptorLifting, DeserializeCorruptedBufferThrows)
{
    // Pass garbage bytes to deserializeGraph(). The FlatBuffer verifier
    // should reject the buffer and throw HIPDNN_STATUS_BAD_PARAM.
    std::vector<uint8_t> garbage = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};

    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();
    ASSERT_THROW_HIPDNN_STATUS(graphDesc->deserializeGraph(garbage.data(), garbage.size()),
                               HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// Tests for getAttribute Without Finalization
// =============================================================================

TEST_F(TestGraphDescriptorLifting, GetAttributeOpsOnFlatBufferFlowWorksWithoutFinalization)
{
    // Build and serialize a conv graph
    auto conv = createDefaultConvOp();
    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto serializedBytes = buildAndSerializeGraph(ops);

    // Create a new GraphDescriptor and deserialize (no handle, no finalize)
    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();
    graphDesc->deserializeGraph(serializedBytes.data(), serializedBytes.size());

    // Lazy unpack works without finalization: unpacks _graph nodes into _operations
    int64_t elementCount = -1;
    ASSERT_NO_THROW(graphDesc->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 1);
}

TEST_F(TestGraphDescriptorLifting, GetAttributeDataTypesWithoutFinalization)
{
    // Build and serialize a conv graph with specific data types
    auto conv = createDefaultConvOp(HIPDNN_DATA_HALF);
    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto serializedBytes
        = buildAndSerializeGraph(ops, HIPDNN_DATA_HALF, HIPDNN_DATA_BFLOAT16, HIPDNN_DATA_FLOAT);

    // Create a new GraphDescriptor and deserialize (no handle, no finalize)
    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();
    graphDesc->deserializeGraph(serializedBytes.data(), serializedBytes.size());

    // Query compute data type without finalization - should succeed
    int64_t computeCount = 0;
    hipdnnDataType_t computeDt = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(graphDesc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT,
                                            HIPDNN_TYPE_DATA_TYPE,
                                            1,
                                            &computeCount,
                                            &computeDt));
    EXPECT_EQ(computeDt, HIPDNN_DATA_HALF);

    // Query intermediate data type without finalization - should succeed
    int64_t intermediateCount = 0;
    hipdnnDataType_t intermediateDt = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(graphDesc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                                            HIPDNN_TYPE_DATA_TYPE,
                                            1,
                                            &intermediateCount,
                                            &intermediateDt));
    EXPECT_EQ(intermediateDt, HIPDNN_DATA_BFLOAT16);

    // Query IO data type without finalization - should succeed
    int64_t ioCount = 0;
    hipdnnDataType_t ioDt = HIPDNN_DATA_HALF;
    ASSERT_NO_THROW(graphDesc->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &ioCount, &ioDt));
    EXPECT_EQ(ioDt, HIPDNN_DATA_FLOAT);
}

TEST_F(TestGraphDescriptorLifting, DeserializePreservesGraphName)
{
    auto conv = createDefaultConvOp();
    const std::string graphName = "TestGraphName";

    const std::vector<HipdnnBackendDescriptor*> ops = {conv.convOp.get()};
    auto bytes = buildAndSerializeGraph(
        ops, HIPDNN_DATA_FLOAT, HIPDNN_DATA_FLOAT, HIPDNN_DATA_FLOAT, std::nullopt, graphName);

    auto liftedGraph = deserializeAndFinalize(bytes);

    // Verify via getAttribute (getString returns size+1 for null terminator)
    int64_t nameCount = 0;
    ASSERT_NO_THROW(liftedGraph->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &nameCount, nullptr));
    ASSERT_EQ(nameCount, static_cast<int64_t>(graphName.size() + 1));

    std::vector<char> nameBuffer(static_cast<size_t>(nameCount));
    int64_t actualCount = 0;
    ASSERT_NO_THROW(liftedGraph->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                                              HIPDNN_TYPE_CHAR,
                                              nameCount,
                                              &actualCount,
                                              nameBuffer.data()));
    EXPECT_STREQ(nameBuffer.data(), graphName.c_str());

    // Also verify via the serialized FlatBuffer
    auto reSerializedData = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(reSerializedData.ptr);
    EXPECT_EQ(graphT->name, graphName);
}
