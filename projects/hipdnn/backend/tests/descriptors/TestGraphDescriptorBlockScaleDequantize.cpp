// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "GraphTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BlockScaleDequantizeOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BlockScaleDequantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized BlockScaleDequantizeOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedBlockScaleDequantizeOp(HipdnnBackendDescriptor* xDesc,
                                          HipdnnBackendDescriptor* scaleDesc,
                                          HipdnnBackendDescriptor* yDesc,
                                          hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<BlockScaleDequantizeOperationDescriptor>();
    auto desc = wrapper->asDescriptor<BlockScaleDequantizeOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));

    std::vector<int32_t> blockSize = {K_BSD_BLOCK_SIZE};
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                       HIPDNN_TYPE_INT32,
                       1,
                       blockSize.data());
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorBlockScaleDequantize : public ::testing::Test
{
public:
    std::shared_ptr<GraphDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<GraphDescriptor>();
    }

    void setHandle() const
    {
        auto desc = getDescriptor();
        hipdnnHandle_t handle = &_mockHandle;
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                           HIPDNN_TYPE_HANDLE,
                           1,
                           static_cast<const void*>(&handle));
    }

    // Build a BSD graph, finalize, and return serialized bytes
    std::vector<uint8_t> buildAndSerializeGraph(HipdnnBackendDescriptor* op)
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
                                1,
                                static_cast<const void*>(&op));
        graphDesc->finalize();

        auto serialized = graphDesc->getSerializedGraph();
        return {static_cast<const uint8_t*>(serialized.ptr),
                static_cast<const uint8_t*>(serialized.ptr) + serialized.size};
    }

    // Deserialize bytes into a new GraphDescriptor and finalize it
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
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    mutable MockHandle _mockHandle;

    void SetUp() override
    {
        _wrapper = createDescriptor<GraphDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
    }
};

// =============================================================================
// Serialization Tests
// =============================================================================

TEST_F(TestGraphDescriptorBlockScaleDequantize, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(
        K_BSD_TENSOR_X_UID, toVec(K_BSD_TENSOR_X_DIMS), toVec(K_BSD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSD_TENSOR_SCALE_UID, toVec(K_BSD_TENSOR_SCALE_DIMS), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSD_TENSOR_Y_UID, toVec(K_BSD_TENSOR_Y_DIMS), toVec(K_BSD_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBlockScaleDequantizeOp(xDesc.get(), scaleDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    // Verify the built graph
    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::BlockScaleDequantizeAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->x_tensor_uid, K_BSD_TENSOR_X_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_BSD_TENSOR_Y_UID);

    // Verify operation data fields survive serialization
    ASSERT_EQ(attrs->block_size.size(), 1);
    EXPECT_EQ(attrs->block_size[0], K_BSD_BLOCK_SIZE);
    EXPECT_FALSE(attrs->is_negative_scale);
}

TEST_F(TestGraphDescriptorBlockScaleDequantize, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(
        K_BSD_TENSOR_X_UID, toVec(K_BSD_TENSOR_X_DIMS), toVec(K_BSD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSD_TENSOR_SCALE_UID, toVec(K_BSD_TENSOR_SCALE_DIMS), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSD_TENSOR_Y_UID, toVec(K_BSD_TENSOR_Y_DIMS), toVec(K_BSD_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBlockScaleDequantizeOp(
        xDesc.get(), scaleDesc.get(), yDesc.get(), HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

// =============================================================================
// Deserialization / Lifting Round-Trip Tests
// =============================================================================

TEST_F(TestGraphDescriptorBlockScaleDequantize, DeserializePreservesNodeAttributes)
{
    auto xDesc = createFinalizedTensor(
        K_BSD_TENSOR_X_UID, toVec(K_BSD_TENSOR_X_DIMS), toVec(K_BSD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSD_TENSOR_SCALE_UID, toVec(K_BSD_TENSOR_SCALE_DIMS), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSD_TENSOR_Y_UID, toVec(K_BSD_TENSOR_Y_DIMS), toVec(K_BSD_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBlockScaleDequantizeOp(xDesc.get(), scaleDesc.get(), yDesc.get());

    auto serializedBytes = buildAndSerializeGraph(opDesc.get());
    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::BlockScaleDequantizeAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->x_tensor_uid, K_BSD_TENSOR_X_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_BSD_TENSOR_Y_UID);
    ASSERT_EQ(attrs->block_size.size(), 1);
    EXPECT_EQ(attrs->block_size[0], K_BSD_BLOCK_SIZE);
    EXPECT_FALSE(attrs->is_negative_scale);
}

TEST_F(TestGraphDescriptorBlockScaleDequantize, DeserializePreservesComputeDataType)
{
    auto xDesc = createFinalizedTensor(
        K_BSD_TENSOR_X_UID, toVec(K_BSD_TENSOR_X_DIMS), toVec(K_BSD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSD_TENSOR_SCALE_UID, toVec(K_BSD_TENSOR_SCALE_DIMS), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSD_TENSOR_Y_UID, toVec(K_BSD_TENSOR_Y_DIMS), toVec(K_BSD_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBlockScaleDequantizeOp(
        xDesc.get(), scaleDesc.get(), yDesc.get(), HIPDNN_DATA_HALF);

    auto serializedBytes = buildAndSerializeGraph(opDesc.get());
    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorBlockScaleDequantize, DeserializePreservesIsNegativeScale)
{
    auto xDesc = createFinalizedTensor(
        K_BSD_TENSOR_X_UID, toVec(K_BSD_TENSOR_X_DIMS), toVec(K_BSD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSD_TENSOR_SCALE_UID, toVec(K_BSD_TENSOR_SCALE_DIMS), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSD_TENSOR_Y_UID, toVec(K_BSD_TENSOR_Y_DIMS), toVec(K_BSD_TENSOR_Y_STRIDES));

    auto wrapper = createDescriptor<BlockScaleDequantizeOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<BlockScaleDequantizeOperationDescriptor>();

    auto* xRaw = xDesc.get();
    auto* scaleRaw = scaleDesc.get();
    auto* yRaw = yDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&xRaw));
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&scaleRaw));
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&yRaw));

    std::vector<int32_t> blockSize = {K_BSD_BLOCK_SIZE};
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                         HIPDNN_TYPE_INT32,
                         1,
                         blockSize.data());

    bool isNegativeScale = true;
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                         HIPDNN_TYPE_BOOLEAN,
                         1,
                         &isNegativeScale);

    hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT;
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                         HIPDNN_TYPE_DATA_TYPE,
                         1,
                         &computeType);
    opDesc->finalize();

    auto serializedBytes = buildAndSerializeGraph(wrapper.get());
    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    auto* attrs = graphT->nodes[0]->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_TRUE(attrs->is_negative_scale);
}

TEST_F(TestGraphDescriptorBlockScaleDequantize, DeserializePreservesTensorData)
{
    auto xDesc = createFinalizedTensor(
        K_BSD_TENSOR_X_UID, toVec(K_BSD_TENSOR_X_DIMS), toVec(K_BSD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSD_TENSOR_SCALE_UID, toVec(K_BSD_TENSOR_SCALE_DIMS), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSD_TENSOR_Y_UID, toVec(K_BSD_TENSOR_Y_DIMS), toVec(K_BSD_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBlockScaleDequantizeOp(xDesc.get(), scaleDesc.get(), yDesc.get());

    auto serializedBytes = buildAndSerializeGraph(opDesc.get());
    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    auto serialized = liftedGraph->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->tensors.size(), 3);

    auto* xTensor = findTensorByUid(*graphT, K_BSD_TENSOR_X_UID);
    ASSERT_NE(xTensor, nullptr);
    EXPECT_EQ(xTensor->dims, toVec(K_BSD_TENSOR_X_DIMS));
    EXPECT_EQ(xTensor->strides, toVec(K_BSD_TENSOR_X_STRIDES));

    auto* scaleTensor = findTensorByUid(*graphT, K_BSD_TENSOR_SCALE_UID);
    ASSERT_NE(scaleTensor, nullptr);
    EXPECT_EQ(scaleTensor->dims, toVec(K_BSD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(scaleTensor->strides, toVec(K_BSD_TENSOR_SCALE_STRIDES));

    auto* yTensor = findTensorByUid(*graphT, K_BSD_TENSOR_Y_UID);
    ASSERT_NE(yTensor, nullptr);
    EXPECT_EQ(yTensor->dims, toVec(K_BSD_TENSOR_Y_DIMS));
    EXPECT_EQ(yTensor->strides, toVec(K_BSD_TENSOR_Y_STRIDES));
}

TEST_F(TestGraphDescriptorBlockScaleDequantize, LiftedGraphSerializesToSameBinary)
{
    auto xDesc = createFinalizedTensor(
        K_BSD_TENSOR_X_UID, toVec(K_BSD_TENSOR_X_DIMS), toVec(K_BSD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSD_TENSOR_SCALE_UID, toVec(K_BSD_TENSOR_SCALE_DIMS), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSD_TENSOR_Y_UID, toVec(K_BSD_TENSOR_Y_DIMS), toVec(K_BSD_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBlockScaleDequantizeOp(xDesc.get(), scaleDesc.get(), yDesc.get());

    auto originalBytes = buildAndSerializeGraph(opDesc.get());
    auto liftedGraph = deserializeAndFinalize(originalBytes);

    auto reSerializedData = liftedGraph->getSerializedGraph();
    const std::vector<uint8_t> reSerializedBytes(static_cast<const uint8_t*>(reSerializedData.ptr),
                                                 static_cast<const uint8_t*>(reSerializedData.ptr)
                                                     + reSerializedData.size);

    ASSERT_EQ(originalBytes.size(), reSerializedBytes.size());
    EXPECT_EQ(originalBytes, reSerializedBytes);
}

TEST_F(TestGraphDescriptorBlockScaleDequantize, GetAttributeOpsAfterDeserializeSucceeds)
{
    auto xDesc = createFinalizedTensor(
        K_BSD_TENSOR_X_UID, toVec(K_BSD_TENSOR_X_DIMS), toVec(K_BSD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSD_TENSOR_SCALE_UID, toVec(K_BSD_TENSOR_SCALE_DIMS), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSD_TENSOR_Y_UID, toVec(K_BSD_TENSOR_Y_DIMS), toVec(K_BSD_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBlockScaleDequantizeOp(xDesc.get(), scaleDesc.get(), yDesc.get());

    auto serializedBytes = buildAndSerializeGraph(opDesc.get());
    auto liftedGraph = deserializeAndFinalize(serializedBytes);

    int64_t elementCount = -1;
    ASSERT_NO_THROW(liftedGraph->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 1);
}

} // namespace
