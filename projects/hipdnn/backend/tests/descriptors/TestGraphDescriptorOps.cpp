// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "GraphTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
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
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <map>
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
// Alternate UIDs for testing attribute-order independence
constexpr int64_t K_ALT_TENSOR_X_UID = 11;
constexpr int64_t K_ALT_TENSOR_W_UID = 12;
constexpr int64_t K_ALT_TENSOR_Y_UID = 13;

// UIDs for the second conv op in SharedTensorDifferentPositions
constexpr int64_t K_SHARED_TENSOR_W_UID = 14;
constexpr int64_t K_SHARED_TENSOR_Y_UID = 15;
} // namespace

class TestGraphDescriptorOps : public ::testing::Test
{
public:
    // Validate a tensor's fields against expected values
    static void verifyTensor(const TensorAttributesT* tensor,
                             int64_t expectedUid,
                             const std::vector<int64_t>& expectedDims,
                             const std::vector<int64_t>& expectedStrides,
                             DataType expectedDataType,
                             bool expectedVirtual = false)
    {
        ASSERT_NE(tensor, nullptr) << "Tensor with UID " << expectedUid << " not found";
        EXPECT_EQ(tensor->uid, expectedUid);
        EXPECT_EQ(tensor->dims, expectedDims);
        EXPECT_EQ(tensor->strides, expectedStrides);
        EXPECT_EQ(tensor->data_type, expectedDataType);
        EXPECT_EQ(tensor->virtual_, expectedVirtual);
    }

    // Validate a node's convolution forward attributes
    static void verifyConvFwdNode(const NodeT& node,
                                  DataType expectedComputeType,
                                  int64_t expectedXUid,
                                  int64_t expectedWUid,
                                  int64_t expectedYUid,
                                  const std::vector<int64_t>& expectedPrePadding,
                                  const std::vector<int64_t>& expectedPostPadding,
                                  const std::vector<int64_t>& expectedStride,
                                  const std::vector<int64_t>& expectedDilation)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::ConvolutionFwdAttributes);

        auto* convAttrs = node.attributes.AsConvolutionFwdAttributes();
        ASSERT_NE(convAttrs, nullptr);

        EXPECT_EQ(convAttrs->x_tensor_uid, expectedXUid);
        EXPECT_EQ(convAttrs->w_tensor_uid, expectedWUid);
        EXPECT_EQ(convAttrs->y_tensor_uid, expectedYUid);
        EXPECT_EQ(convAttrs->pre_padding, expectedPrePadding);
        EXPECT_EQ(convAttrs->post_padding, expectedPostPadding);
        EXPECT_EQ(convAttrs->stride, expectedStride);
        EXPECT_EQ(convAttrs->dilation, expectedDilation);
    }

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
// Build From Operations Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, BuildFromSingleOperation)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
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

    // Verify each tensor has correct fields
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                 K_FPROP_TENSOR_X_UID,
                 toVec(K_FPROP_TENSOR_X_DIMS),
                 toVec(K_FPROP_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                 K_FPROP_TENSOR_W_UID,
                 toVec(K_FPROP_TENSOR_W_DIMS),
                 toVec(K_FPROP_TENSOR_W_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                 K_FPROP_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify the node's convolution attributes and tensor UID references
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestGraphDescriptorOps, BuildFromMultipleOperations)
{
    // First conv: primary tensor set
    auto conv1 = createDefaultConvOp();

    // Second conv: second tensor set
    auto xDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_X2_UID, toVec(K_FPROP_TENSOR_X2_DIMS), toVec(K_FPROP_TENSOR_X2_STRIDES));
    auto wDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_W2_UID, toVec(K_FPROP_TENSOR_W2_DIMS), toVec(K_FPROP_TENSOR_W2_STRIDES));
    auto yDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_Y2_UID, toVec(K_FPROP_TENSOR_Y2_DIMS), toVec(K_FPROP_TENSOR_Y2_STRIDES));
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), yDesc2.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 2> ops = {conv1.convOp.get(), convOp2.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 2);
    ASSERT_EQ(graphT->tensors.size(), 6);

    // Verify all tensors from first conv op
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                 K_FPROP_TENSOR_X_UID,
                 toVec(K_FPROP_TENSOR_X_DIMS),
                 toVec(K_FPROP_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                 K_FPROP_TENSOR_W_UID,
                 toVec(K_FPROP_TENSOR_W_DIMS),
                 toVec(K_FPROP_TENSOR_W_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                 K_FPROP_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify all tensors from second conv op
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X2_UID),
                 K_FPROP_TENSOR_X2_UID,
                 toVec(K_FPROP_TENSOR_X2_DIMS),
                 toVec(K_FPROP_TENSOR_X2_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W2_UID),
                 K_FPROP_TENSOR_W2_UID,
                 toVec(K_FPROP_TENSOR_W2_DIMS),
                 toVec(K_FPROP_TENSOR_W2_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y2_UID),
                 K_FPROP_TENSOR_Y2_UID,
                 toVec(K_FPROP_TENSOR_Y2_DIMS),
                 toVec(K_FPROP_TENSOR_Y2_STRIDES),
                 DataType::FLOAT);

    // Verify first node references primary tensors
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));

    // Verify second node references second tensor set
    verifyConvFwdNode(*graphT->nodes[1],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X2_UID,
                      K_FPROP_TENSOR_W2_UID,
                      K_FPROP_TENSOR_Y2_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestGraphDescriptorOps, TensorDeduplication)
{
    // Two operations sharing the same output tensor (Y from first op)
    auto conv1 = createDefaultConvOp();

    auto xDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_X2_UID, toVec(K_FPROP_TENSOR_X2_DIMS), toVec(K_FPROP_TENSOR_X2_STRIDES));
    auto wDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_W2_UID, toVec(K_FPROP_TENSOR_W2_DIMS), toVec(K_FPROP_TENSOR_W2_STRIDES));

    // Reuse conv1.yDesc (uid 3) as Y for the second op
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), conv1.yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 2> ops = {conv1.convOp.get(), convOp2.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 2);
    // Should have 5 unique tensors, not 6 (tensor Y deduplicated)
    ASSERT_EQ(graphT->tensors.size(), 5);

    // Verify tensor UIDs are unique
    std::set<int64_t> tensorUids;
    for(const auto& tensor : graphT->tensors)
    {
        tensorUids.insert(tensor->uid);
    }
    EXPECT_EQ(tensorUids.size(), 5);

    // Verify each tensor's fields
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                 K_FPROP_TENSOR_X_UID,
                 toVec(K_FPROP_TENSOR_X_DIMS),
                 toVec(K_FPROP_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                 K_FPROP_TENSOR_W_UID,
                 toVec(K_FPROP_TENSOR_W_DIMS),
                 toVec(K_FPROP_TENSOR_W_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                 K_FPROP_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X2_UID),
                 K_FPROP_TENSOR_X2_UID,
                 toVec(K_FPROP_TENSOR_X2_DIMS),
                 toVec(K_FPROP_TENSOR_X2_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W2_UID),
                 K_FPROP_TENSOR_W2_UID,
                 toVec(K_FPROP_TENSOR_W2_DIMS),
                 toVec(K_FPROP_TENSOR_W2_STRIDES),
                 DataType::FLOAT);

    // Verify first node: primary tensors
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));

    // Verify second node also references Y (the shared tensor)
    verifyConvFwdNode(*graphT->nodes[1],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X2_UID,
                      K_FPROP_TENSOR_W2_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestGraphDescriptorOps, ComputeDataTypePreserved)
{
    auto conv = createDefaultConvOp(HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify tensors retain FLOAT data type (tensor data type is independent of compute type)
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                 K_FPROP_TENSOR_X_UID,
                 toVec(K_FPROP_TENSOR_X_DIMS),
                 toVec(K_FPROP_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                 K_FPROP_TENSOR_W_UID,
                 toVec(K_FPROP_TENSOR_W_DIMS),
                 toVec(K_FPROP_TENSOR_W_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                 K_FPROP_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify node compute data type is HALF and all conv attributes are correct
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::HALF,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestGraphDescriptorOps, ConvolutionAttributesPreserved)
{
    auto conv = createDefaultConvOp();

    // Create conv op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<ConvolutionFwdOperationDescriptor>();
    auto convDesc = wrapper->asDescriptor<ConvolutionFwdOperationDescriptor>();

    HipdnnBackendDescriptor* x = conv.xDesc.get();
    HipdnnBackendDescriptor* w = conv.wDesc.get();
    HipdnnBackendDescriptor* y = conv.yDesc.get();

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
    auto dilation = toVec(K_FPROP_CONV_DILATION);

    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, kCustomPrePadding.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, kCustomPostPadding.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, kCustomStride.data());
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());

    auto computeType = HIPDNN_DATA_FLOAT;
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    hipdnnConvolutionMode_t convMode = HIPDNN_CROSS_CORRELATION;
    convDesc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);
    convDesc->finalize();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {wrapper.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify tensors
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                 K_FPROP_TENSOR_X_UID,
                 toVec(K_FPROP_TENSOR_X_DIMS),
                 toVec(K_FPROP_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                 K_FPROP_TENSOR_W_UID,
                 toVec(K_FPROP_TENSOR_W_DIMS),
                 toVec(K_FPROP_TENSOR_W_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                 K_FPROP_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify node with asymmetric padding and non-unit stride
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      kCustomPrePadding,
                      kCustomPostPadding,
                      kCustomStride,
                      toVec(K_FPROP_CONV_DILATION));
}

// =============================================================================
// SetAttribute Order Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, SetOperationsAndHandleAnyOrder)
{
    // Test: operations first, then handle
    {
        auto graphWrapper = createDescriptor<GraphDescriptor>();
        auto desc = graphWrapper->asDescriptor<GraphDescriptor>();

        auto conv = createDefaultConvOp();

        std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
        ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           static_cast<const void*>(ops.data())));

        hipdnnHandle_t handle = &_mockHandle;
        ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                           HIPDNN_TYPE_HANDLE,
                                           1,
                                           static_cast<const void*>(&handle)));

        ASSERT_NO_THROW(desc->finalize());

        auto serialized = desc->getSerializedGraph();
        auto graphT = UnPackGraph(serialized.ptr);

        ASSERT_EQ(graphT->tensors.size(), 3);
        ASSERT_EQ(graphT->nodes.size(), 1);
        verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                     K_FPROP_TENSOR_X_UID,
                     toVec(K_FPROP_TENSOR_X_DIMS),
                     toVec(K_FPROP_TENSOR_X_STRIDES),
                     DataType::FLOAT);
        verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                     K_FPROP_TENSOR_W_UID,
                     toVec(K_FPROP_TENSOR_W_DIMS),
                     toVec(K_FPROP_TENSOR_W_STRIDES),
                     DataType::FLOAT);
        verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                     K_FPROP_TENSOR_Y_UID,
                     toVec(K_FPROP_TENSOR_Y_DIMS),
                     toVec(K_FPROP_TENSOR_Y_STRIDES),
                     DataType::FLOAT);
        verifyConvFwdNode(*graphT->nodes[0],
                          DataType::FLOAT,
                          K_FPROP_TENSOR_X_UID,
                          K_FPROP_TENSOR_W_UID,
                          K_FPROP_TENSOR_Y_UID,
                          toVec(K_FPROP_CONV_PADDING),
                          toVec(K_FPROP_CONV_PADDING),
                          toVec(K_FPROP_CONV_STRIDE),
                          toVec(K_FPROP_CONV_DILATION));
    }

    // Test: handle first, then operations (with alternate UIDs)
    {
        auto graphWrapper = createDescriptor<GraphDescriptor>();
        auto desc = graphWrapper->asDescriptor<GraphDescriptor>();

        auto xDesc = createFinalizedTensor(K_ALT_TENSOR_X_UID);
        auto wDesc = createFinalizedTensor(
            K_ALT_TENSOR_W_UID, toVec(K_FPROP_TENSOR_W_DIMS), toVec(K_FPROP_TENSOR_W_STRIDES));
        auto yDesc = createFinalizedTensor(
            K_ALT_TENSOR_Y_UID, toVec(K_FPROP_TENSOR_Y_DIMS), toVec(K_FPROP_TENSOR_Y_STRIDES));
        auto convOp = createFinalizedConvOp(xDesc.get(), wDesc.get(), yDesc.get());

        hipdnnHandle_t handle = &_mockHandle;
        ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                           HIPDNN_TYPE_HANDLE,
                                           1,
                                           static_cast<const void*>(&handle)));

        std::array<HipdnnBackendDescriptor*, 1> ops = {convOp.get()};
        ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           static_cast<const void*>(ops.data())));

        ASSERT_NO_THROW(desc->finalize());

        auto serialized = desc->getSerializedGraph();
        auto graphT = UnPackGraph(serialized.ptr);

        ASSERT_EQ(graphT->tensors.size(), 3);
        ASSERT_EQ(graphT->nodes.size(), 1);
        verifyTensor(findTensorByUid(*graphT, K_ALT_TENSOR_X_UID),
                     K_ALT_TENSOR_X_UID,
                     toVec(K_FPROP_TENSOR_X_DIMS),
                     toVec(K_FPROP_TENSOR_X_STRIDES),
                     DataType::FLOAT);
        verifyTensor(findTensorByUid(*graphT, K_ALT_TENSOR_W_UID),
                     K_ALT_TENSOR_W_UID,
                     toVec(K_FPROP_TENSOR_W_DIMS),
                     toVec(K_FPROP_TENSOR_W_STRIDES),
                     DataType::FLOAT);
        verifyTensor(findTensorByUid(*graphT, K_ALT_TENSOR_Y_UID),
                     K_ALT_TENSOR_Y_UID,
                     toVec(K_FPROP_TENSOR_Y_DIMS),
                     toVec(K_FPROP_TENSOR_Y_STRIDES),
                     DataType::FLOAT);
        verifyConvFwdNode(*graphT->nodes[0],
                          DataType::FLOAT,
                          K_ALT_TENSOR_X_UID,
                          K_ALT_TENSOR_W_UID,
                          K_ALT_TENSOR_Y_UID,
                          toVec(K_FPROP_CONV_PADDING),
                          toVec(K_FPROP_CONV_PADDING),
                          toVec(K_FPROP_CONV_STRIDE),
                          toVec(K_FPROP_CONV_DILATION));
    }
}

TEST_F(TestGraphDescriptorOps, SetOperationsMultipleBatches)
{
    auto conv1 = createDefaultConvOp();

    auto xDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_X2_UID, toVec(K_FPROP_TENSOR_X2_DIMS), toVec(K_FPROP_TENSOR_X2_STRIDES));
    auto wDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_W2_UID, toVec(K_FPROP_TENSOR_W2_DIMS), toVec(K_FPROP_TENSOR_W2_STRIDES));
    auto yDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_Y2_UID, toVec(K_FPROP_TENSOR_Y2_DIMS), toVec(K_FPROP_TENSOR_Y2_STRIDES));
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), yDesc2.get());

    auto desc = getDescriptor();
    setHandle();

    // Set multiple operations in a single setAttribute call
    std::array<HipdnnBackendDescriptor*, 2> ops = {conv1.convOp.get(), convOp2.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       static_cast<const void*>(ops.data())));

    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    // Both operations should be present in the graph
    ASSERT_EQ(graphT->nodes.size(), 2);
    ASSERT_EQ(graphT->tensors.size(), 6);

    // Verify tensors from first operation
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                 K_FPROP_TENSOR_X_UID,
                 toVec(K_FPROP_TENSOR_X_DIMS),
                 toVec(K_FPROP_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                 K_FPROP_TENSOR_W_UID,
                 toVec(K_FPROP_TENSOR_W_DIMS),
                 toVec(K_FPROP_TENSOR_W_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                 K_FPROP_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify tensors from second operation
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X2_UID),
                 K_FPROP_TENSOR_X2_UID,
                 toVec(K_FPROP_TENSOR_X2_DIMS),
                 toVec(K_FPROP_TENSOR_X2_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W2_UID),
                 K_FPROP_TENSOR_W2_UID,
                 toVec(K_FPROP_TENSOR_W2_DIMS),
                 toVec(K_FPROP_TENSOR_W2_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y2_UID),
                 K_FPROP_TENSOR_Y2_UID,
                 toVec(K_FPROP_TENSOR_Y2_DIMS),
                 toVec(K_FPROP_TENSOR_Y2_STRIDES),
                 DataType::FLOAT);

    // Verify both nodes reference correct tensor UIDs
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
    verifyConvFwdNode(*graphT->nodes[1],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X2_UID,
                      K_FPROP_TENSOR_W2_UID,
                      K_FPROP_TENSOR_Y2_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
}

// =============================================================================
// Error Cases
// =============================================================================

TEST_F(TestGraphDescriptorOps, SetOperationsFailsUnfinalized)
{
    auto desc = getDescriptor();

    // Create unfinalized operation
    auto unfinalizedOp = createDescriptor<ConvolutionFwdOperationDescriptor>();

    std::array<HipdnnBackendDescriptor*, 1> ops = {unfinalizedOp.get()};
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  static_cast<const void*>(ops.data())),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestGraphDescriptorOps, SetOperationsFailsNullDescriptor)
{
    auto desc = getDescriptor();

    std::array<HipdnnBackendDescriptor*, 1> ops = {nullptr};
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  static_cast<const void*>(ops.data())),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestGraphDescriptorOps, SetOperationsFailsWrongType)
{
    auto desc = getDescriptor();

    // Use a TensorDescriptor instead of an operation descriptor
    auto tensorDesc = createFinalizedTensor(K_FPROP_TENSOR_X_UID);

    std::array<HipdnnBackendDescriptor*, 1> ops = {tensorDesc.get()};
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  static_cast<const void*>(ops.data())),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestGraphDescriptorOps, SetOperationsFailsAfterFinalize)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Try to set operations after finalize
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  static_cast<const void*>(ops.data())),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestGraphDescriptorOps, FinalizeFailsWithoutHandle)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    // Don't set handle

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptorOps, FinalizeFailsWithoutOperationsOrGraph)
{
    auto desc = getDescriptor();
    setHandle();
    // Don't set operations or deserialize graph

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// Serialization Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, SerializedGraphVerifiable)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();

    // Verify FlatBuffer is valid
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    // Verify the verified buffer contains the expected graph structure
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_EQ(graphT->tensors.size(), 3);
    ASSERT_EQ(graphT->nodes.size(), 1);
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestGraphDescriptorOps, SerializedGraphUnpackable)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(GetGraph(serialized.ptr), nullptr);

    // Unpack should work and produce correct values
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_NE(graphT, nullptr);
    ASSERT_EQ(graphT->tensors.size(), 3);
    ASSERT_EQ(graphT->nodes.size(), 1);

    // Verify unpacked tensor values match input
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                 K_FPROP_TENSOR_X_UID,
                 toVec(K_FPROP_TENSOR_X_DIMS),
                 toVec(K_FPROP_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                 K_FPROP_TENSOR_W_UID,
                 toVec(K_FPROP_TENSOR_W_DIMS),
                 toVec(K_FPROP_TENSOR_W_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                 K_FPROP_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify unpacked node values match input
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestGraphDescriptorOps, GetSerializedGraphMultipleCalls)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Call getSerializedGraph multiple times
    auto serialized1 = desc->getSerializedGraph();
    auto serialized2 = desc->getSerializedGraph();

    // Should return same data
    EXPECT_EQ(serialized1.ptr, serialized2.ptr);
    EXPECT_EQ(serialized1.size, serialized2.size);

    // Both should unpack to identical graph values
    auto graphT1 = UnPackGraph(serialized1.ptr);
    auto graphT2 = UnPackGraph(serialized2.ptr);

    ASSERT_EQ(graphT1->tensors.size(), 3);
    ASSERT_EQ(graphT2->tensors.size(), 3);
    ASSERT_EQ(graphT1->nodes.size(), 1);
    ASSERT_EQ(graphT2->nodes.size(), 1);

    // Verify both unpacked graphs contain identical tensor data
    for(size_t i = 0; i < graphT1->tensors.size(); ++i)
    {
        EXPECT_EQ(graphT1->tensors[i]->uid, graphT2->tensors[i]->uid);
        EXPECT_EQ(graphT1->tensors[i]->dims, graphT2->tensors[i]->dims);
        EXPECT_EQ(graphT1->tensors[i]->strides, graphT2->tensors[i]->strides);
        EXPECT_EQ(graphT1->tensors[i]->data_type, graphT2->tensors[i]->data_type);
    }

    // Verify both unpacked graphs contain identical node data
    EXPECT_EQ(graphT1->nodes[0]->compute_data_type, graphT2->nodes[0]->compute_data_type);
    EXPECT_EQ(graphT1->nodes[0]->attributes.type, graphT2->nodes[0]->attributes.type);
}

// =============================================================================
// Graph Structure Verification Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, GraphHasCorrectNodeCount)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);

    // Verify node has ConvolutionFwdAttributes and correct tensor UID references
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestGraphDescriptorOps, GraphHasCorrectTensorCount)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify each tensor's full field values (not just UIDs)
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                 K_FPROP_TENSOR_X_UID,
                 toVec(K_FPROP_TENSOR_X_DIMS),
                 toVec(K_FPROP_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                 K_FPROP_TENSOR_W_UID,
                 toVec(K_FPROP_TENSOR_W_DIMS),
                 toVec(K_FPROP_TENSOR_W_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                 K_FPROP_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);
}

// =============================================================================
// Equivalence Tests - Compare FlatBuffer path vs Descriptor path
// =============================================================================

struct ConvEquivalenceParams
{
    std::string name;
    int64_t xUid;
    int64_t wUid;
    int64_t yUid;
    std::vector<int64_t> xDims;
    std::vector<int64_t> xStrides;
    std::vector<int64_t> wDims;
    std::vector<int64_t> wStrides;
    std::vector<int64_t> yDims;
    std::vector<int64_t> yStrides;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    hipdnnDataType_t tensorDataType;
    hipdnnDataType_t computeDataType;
};

// Outside anonymous namespace so ADL finds it for gtest printing
// NOLINTNEXTLINE(misc-use-internal-linkage)
std::ostream& operator<<(std::ostream& os, const ConvEquivalenceParams& p)
{
    return os << p.name;
}

class TestGraphDescriptorEquivalence : public TestGraphDescriptorOps,
                                       public ::testing::WithParamInterface<ConvEquivalenceParams>
{
public:
    // Build a graph via FlatBuffer serialization (the "old" path)
    static flatbuffers::DetachedBuffer
        buildGraphViaFlatBuffer(const TensorAttributesT& xTensor,
                                const TensorAttributesT& wTensor,
                                const TensorAttributesT& yTensor,
                                const ConvolutionFwdAttributesT& convAttrs,
                                DataType computeDataType)
    {
        flatbuffers::FlatBufferBuilder builder;

        // Build tensors
        std::vector<flatbuffers::Offset<TensorAttributes>> tensorOffsets;
        tensorOffsets.push_back(TensorAttributes::Pack(builder, &xTensor));
        tensorOffsets.push_back(TensorAttributes::Pack(builder, &wTensor));
        tensorOffsets.push_back(TensorAttributes::Pack(builder, &yTensor));

        // Build node with conv attributes
        NodeT nodeT;
        nodeT.compute_data_type = computeDataType;
        nodeT.attributes.Set(ConvolutionFwdAttributesT(convAttrs));

        std::vector<flatbuffers::Offset<Node>> nodeOffsets;
        nodeOffsets.push_back(Node::Pack(builder, &nodeT));

        // Build graph
        auto graphOffset = CreateGraphDirect(builder,
                                             nullptr, // name
                                             DataType::UNSET,
                                             DataType::UNSET,
                                             DataType::UNSET,
                                             &tensorOffsets,
                                             &nodeOffsets);
        builder.Finish(graphOffset);
        return builder.Release();
    }

    // Build a graph via descriptors (the "new" path)
    std::unique_ptr<GraphT> buildGraphViaDescriptors(int64_t xUid,
                                                     int64_t wUid,
                                                     int64_t yUid,
                                                     const std::vector<int64_t>& xDims,
                                                     const std::vector<int64_t>& xStrides,
                                                     const std::vector<int64_t>& wDims,
                                                     const std::vector<int64_t>& wStrides,
                                                     const std::vector<int64_t>& yDims,
                                                     const std::vector<int64_t>& yStrides,
                                                     const std::vector<int64_t>& prePadding,
                                                     const std::vector<int64_t>& postPadding,
                                                     const std::vector<int64_t>& stride,
                                                     const std::vector<int64_t>& dilation,
                                                     hipdnnDataType_t tensorDataType,
                                                     hipdnnDataType_t computeDataType)
    {
        // Create tensor descriptors
        auto xDesc = createFinalizedTensor(xUid, xDims, xStrides, tensorDataType);
        auto wDesc = createFinalizedTensor(wUid, wDims, wStrides, tensorDataType);
        auto yDesc = createFinalizedTensor(yUid, yDims, yStrides, tensorDataType);

        // Create conv op descriptor
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
        convDesc->setAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                               HIPDNN_TYPE_INT64,
                               static_cast<int64_t>(prePadding.size()),
                               prePadding.data());
        convDesc->setAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                               HIPDNN_TYPE_INT64,
                               static_cast<int64_t>(postPadding.size()),
                               postPadding.data());
        convDesc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                               HIPDNN_TYPE_INT64,
                               static_cast<int64_t>(stride.size()),
                               stride.data());
        convDesc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                               HIPDNN_TYPE_INT64,
                               static_cast<int64_t>(dilation.size()),
                               dilation.data());
        convDesc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeDataType);
        hipdnnConvolutionMode_t convMode = HIPDNN_CROSS_CORRELATION;
        convDesc->setAttribute(
            HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);
        convDesc->finalize();

        // Build graph via GraphDescriptor
        auto graphWrapper = createDescriptor<GraphDescriptor>();
        auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

        hipdnnHandle_t handle = &_mockHandle;
        graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                HIPDNN_TYPE_HANDLE,
                                1,
                                static_cast<const void*>(&handle));

        std::array<HipdnnBackendDescriptor*, 1> ops = {convWrapper.get()};
        graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                1,
                                static_cast<const void*>(ops.data()));
        graphDesc->finalize();

        auto serialized = graphDesc->getSerializedGraph();
        return UnPackGraph(serialized.ptr);
    }

    void verifyEquivalence(const ConvEquivalenceParams& p)
    {
        // Build via FlatBuffer path
        auto sdkTensorDt = hipdnn_backend::toSdkDataType(p.tensorDataType);
        auto sdkComputeDt = hipdnn_backend::toSdkDataType(p.computeDataType);

        TensorAttributesT xTensor;
        xTensor.uid = p.xUid;
        xTensor.dims = p.xDims;
        xTensor.strides = p.xStrides;
        xTensor.data_type = sdkTensorDt;

        TensorAttributesT wTensor;
        wTensor.uid = p.wUid;
        wTensor.dims = p.wDims;
        wTensor.strides = p.wStrides;
        wTensor.data_type = sdkTensorDt;

        TensorAttributesT yTensor;
        yTensor.uid = p.yUid;
        yTensor.dims = p.yDims;
        yTensor.strides = p.yStrides;
        yTensor.data_type = sdkTensorDt;

        ConvolutionFwdAttributesT convAttrs;
        convAttrs.x_tensor_uid = p.xUid;
        convAttrs.w_tensor_uid = p.wUid;
        convAttrs.y_tensor_uid = p.yUid;
        convAttrs.pre_padding = p.prePadding;
        convAttrs.post_padding = p.postPadding;
        convAttrs.stride = p.stride;
        convAttrs.dilation = p.dilation;

        auto flatbufferBuffer
            = buildGraphViaFlatBuffer(xTensor, wTensor, yTensor, convAttrs, sdkComputeDt);
        auto serializedGraphT = UnPackGraph(flatbufferBuffer.data());

        // Build via descriptor path
        auto descriptorGraphT = buildGraphViaDescriptors(p.xUid,
                                                         p.wUid,
                                                         p.yUid,
                                                         p.xDims,
                                                         p.xStrides,
                                                         p.wDims,
                                                         p.wStrides,
                                                         p.yDims,
                                                         p.yStrides,
                                                         p.prePadding,
                                                         p.postPadding,
                                                         p.stride,
                                                         p.dilation,
                                                         p.tensorDataType,
                                                         p.computeDataType);

        // Verify structural equivalence
        ASSERT_EQ(serializedGraphT->tensors.size(), descriptorGraphT->tensors.size());
        ASSERT_EQ(serializedGraphT->nodes.size(), descriptorGraphT->nodes.size());

        // Compare tensors (order may differ, so compare by UID)
        std::map<int64_t, const TensorAttributesT*> fbTensors;
        for(const auto& t : serializedGraphT->tensors)
        {
            fbTensors[t->uid] = t.get();
        }

        for(const auto& descTensor : descriptorGraphT->tensors)
        {
            auto it = fbTensors.find(descTensor->uid);
            ASSERT_NE(it, fbTensors.end())
                << "Tensor UID " << descTensor->uid << " not found in FlatBuffer graph";

            const auto* fbTensor = it->second;
            EXPECT_EQ(fbTensor->uid, descTensor->uid);
            EXPECT_EQ(fbTensor->dims, descTensor->dims);
            EXPECT_EQ(fbTensor->strides, descTensor->strides);
            EXPECT_EQ(fbTensor->data_type, descTensor->data_type);
        }

        // Compare nodes
        ASSERT_EQ(serializedGraphT->nodes.size(), 1);
        ASSERT_EQ(descriptorGraphT->nodes.size(), 1);

        const auto& fbNode = serializedGraphT->nodes[0];
        const auto& descNode = descriptorGraphT->nodes[0];

        EXPECT_EQ(fbNode->compute_data_type, descNode->compute_data_type);
        EXPECT_EQ(fbNode->attributes.type, descNode->attributes.type);
        EXPECT_EQ(fbNode->attributes.type, NodeAttributes::ConvolutionFwdAttributes);

        const auto* fbConv = fbNode->attributes.AsConvolutionFwdAttributes();
        const auto* descConv = descNode->attributes.AsConvolutionFwdAttributes();

        ASSERT_NE(fbConv, nullptr);
        ASSERT_NE(descConv, nullptr);

        EXPECT_EQ(fbConv->x_tensor_uid, descConv->x_tensor_uid);
        EXPECT_EQ(fbConv->w_tensor_uid, descConv->w_tensor_uid);
        EXPECT_EQ(fbConv->y_tensor_uid, descConv->y_tensor_uid);
        EXPECT_EQ(fbConv->pre_padding, descConv->pre_padding);
        EXPECT_EQ(fbConv->post_padding, descConv->post_padding);
        EXPECT_EQ(fbConv->stride, descConv->stride);
        EXPECT_EQ(fbConv->dilation, descConv->dilation);
    }
};

TEST_P(TestGraphDescriptorEquivalence, ConvOpEquivalence)
{
    verifyEquivalence(GetParam());
}

namespace
{

std::string convEquivalenceParamName(const ::testing::TestParamInfo<ConvEquivalenceParams>& info)
{
    return info.param.name;
}

} // namespace

INSTANTIATE_TEST_SUITE_P(ConvOps,
                         TestGraphDescriptorEquivalence,
                         ::testing::Values(ConvEquivalenceParams{"SingleConvOp",
                                                                 K_FPROP_TENSOR_X_UID,
                                                                 K_FPROP_TENSOR_W_UID,
                                                                 K_FPROP_TENSOR_Y_UID,
                                                                 toVec(K_FPROP_TENSOR_X_DIMS),
                                                                 toVec(K_FPROP_TENSOR_X_STRIDES),
                                                                 toVec(K_FPROP_TENSOR_W_DIMS),
                                                                 toVec(K_FPROP_TENSOR_W_STRIDES),
                                                                 toVec(K_FPROP_TENSOR_Y_DIMS),
                                                                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                                                                 toVec(K_FPROP_CONV_PADDING),
                                                                 toVec(K_FPROP_CONV_PADDING),
                                                                 toVec(K_FPROP_CONV_STRIDE),
                                                                 toVec(K_FPROP_CONV_DILATION),
                                                                 HIPDNN_DATA_FLOAT,
                                                                 HIPDNN_DATA_FLOAT},
                                           ConvEquivalenceParams{"HalfPrecision",
                                                                 10,
                                                                 20,
                                                                 30,
                                                                 {2, 64, 56, 56},
                                                                 {200704, 3136, 56, 1},
                                                                 {128, 64, 3, 3},
                                                                 {576, 9, 3, 1},
                                                                 {2, 128, 56, 56},
                                                                 {401408, 3136, 56, 1},
                                                                 {1, 1},
                                                                 {1, 1},
                                                                 {1, 1},
                                                                 {1, 1},
                                                                 HIPDNN_DATA_HALF,
                                                                 HIPDNN_DATA_HALF},
                                           ConvEquivalenceParams{"NonUnitStrideAndDilation",
                                                                 100,
                                                                 200,
                                                                 300,
                                                                 {1, 256, 28, 28},
                                                                 {200704, 784, 28, 1},
                                                                 {512, 256, 3, 3},
                                                                 {2304, 9, 3, 1},
                                                                 {1, 512, 14, 14},
                                                                 {100352, 196, 14, 1},
                                                                 {2, 2},
                                                                 {2, 2},
                                                                 {2, 2},
                                                                 {2, 2},
                                                                 HIPDNN_DATA_FLOAT,
                                                                 HIPDNN_DATA_FLOAT}),
                         convEquivalenceParamName);

// =============================================================================
// Graph-Level Attribute Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, GraphLevelDataTypesPreserved)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    // Set graph-level data types before finalize
    auto computeDt = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeDt);

    auto intermediateDt = HIPDNN_DATA_BFLOAT16;
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &intermediateDt);

    auto ioDt = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &ioDt);

    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    EXPECT_EQ(graphT->compute_data_type, DataType::HALF);
    EXPECT_EQ(graphT->intermediate_data_type, DataType::BFLOAT16);
    EXPECT_EQ(graphT->io_data_type, DataType::FLOAT);
}

TEST_F(TestGraphDescriptorOps, PreferredEngineIdPreserved)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    int64_t engineId = 42;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT, HIPDNN_TYPE_INT64, 1, &engineId);

    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    EXPECT_EQ(graphT->preferred_engine_id, 42);
}

TEST_F(TestGraphDescriptorOps, GraphLevelDataTypesDefaultToUnset)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    // Finalize without setting any graph-level data types
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    EXPECT_EQ(graphT->compute_data_type, DataType::UNSET);
    EXPECT_EQ(graphT->intermediate_data_type, DataType::UNSET);
    EXPECT_EQ(graphT->io_data_type, DataType::UNSET);
}

TEST_F(TestGraphDescriptorOps, SharedTensorDifferentPositions)
{
    // Same tensor used as Y in first op and X in second op
    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

    auto handle = reinterpret_cast<hipdnnHandle_t>(&_mockHandle);
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                            HIPDNN_TYPE_HANDLE,
                            1,
                            static_cast<const void*>(&handle));

    // Create tensors (Y tensor shared between ops)
    auto xDesc1 = createFinalizedTensor(K_FPROP_TENSOR_X_UID);
    auto wDesc1 = createFinalizedTensor(
        K_FPROP_TENSOR_W_UID, toVec(K_FPROP_TENSOR_W_DIMS), toVec(K_FPROP_TENSOR_W_STRIDES));
    auto sharedTensor = createFinalizedTensor(
        K_FPROP_TENSOR_Y_UID, toVec(K_FPROP_TENSOR_Y_DIMS), toVec(K_FPROP_TENSOR_Y_STRIDES));
    // 64->64 channel weights unique to this test (not in the shared constant set)
    auto wDesc2 = createFinalizedTensor(K_SHARED_TENSOR_W_UID, {64, 64, 3, 3}, {576, 9, 3, 1});
    auto yDesc2 = createFinalizedTensor(
        K_SHARED_TENSOR_Y_UID, toVec(K_FPROP_TENSOR_Y_DIMS), toVec(K_FPROP_TENSOR_Y_STRIDES));

    // Op1: x=1, w=2, y=3
    auto op1 = createFinalizedConvOp(xDesc1.get(), wDesc1.get(), sharedTensor.get());
    // Op2: x=3, w=4, y=5 (shares tensor 3 with op1 in a different position)
    auto op2 = createFinalizedConvOp(sharedTensor.get(), wDesc2.get(), yDesc2.get());

    std::array<HipdnnBackendDescriptor*, 2> opDescs = {op1.get(), op2.get()};
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            2,
                            static_cast<const void*>(opDescs.data()));

    graphDesc->finalize();

    auto serialized = graphDesc->getSerializedGraph();
    ASSERT_NE(GetGraph(serialized.ptr), nullptr);
    auto graphT = UnPackGraph(serialized.ptr);

    // 5 unique tensors, shared tensor is deduplicated
    ASSERT_EQ(graphT->tensors.size(), 5);
    ASSERT_EQ(graphT->nodes.size(), 2);

    // Verify each tensor's fields
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_X_UID),
                 K_FPROP_TENSOR_X_UID,
                 toVec(K_FPROP_TENSOR_X_DIMS),
                 toVec(K_FPROP_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_W_UID),
                 K_FPROP_TENSOR_W_UID,
                 toVec(K_FPROP_TENSOR_W_DIMS),
                 toVec(K_FPROP_TENSOR_W_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_FPROP_TENSOR_Y_UID),
                 K_FPROP_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SHARED_TENSOR_W_UID),
                 K_SHARED_TENSOR_W_UID,
                 {64, 64, 3, 3},
                 {576, 9, 3, 1},
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SHARED_TENSOR_Y_UID),
                 K_SHARED_TENSOR_Y_UID,
                 toVec(K_FPROP_TENSOR_Y_DIMS),
                 toVec(K_FPROP_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify first node: x=1, w=2, y=3 (shared tensor used as Y here)
    verifyConvFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_X_UID,
                      K_FPROP_TENSOR_W_UID,
                      K_FPROP_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));

    // Verify second node: x=3, w=4, y=5 (shared tensor reused as X here)
    verifyConvFwdNode(*graphT->nodes[1],
                      DataType::FLOAT,
                      K_FPROP_TENSOR_Y_UID,
                      K_SHARED_TENSOR_W_UID,
                      K_SHARED_TENSOR_Y_UID,
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_PADDING),
                      toVec(K_FPROP_CONV_STRIDE),
                      toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestGraphDescriptorOps, FinalizeFailsDuplicateTensorUidDifferentDescriptors)
{
    // Create two distinct tensor descriptors with the same UID
    auto xDesc1 = createFinalizedTensor(K_FPROP_TENSOR_X_UID);
    auto xDesc2 = createFinalizedTensor(K_FPROP_TENSOR_X_UID);
    auto wDesc = createFinalizedTensor(
        K_FPROP_TENSOR_W_UID, toVec(K_FPROP_TENSOR_W_DIMS), toVec(K_FPROP_TENSOR_W_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_FPROP_TENSOR_Y_UID, toVec(K_FPROP_TENSOR_Y_DIMS), toVec(K_FPROP_TENSOR_Y_STRIDES));

    // Op1 uses xDesc1, Op2 uses xDesc2 (different object, same UID)
    auto op1 = createFinalizedConvOp(xDesc1.get(), wDesc.get(), yDesc.get());

    auto xDesc2a = createFinalizedTensor(
        K_FPROP_TENSOR_X2_UID, toVec(K_FPROP_TENSOR_X2_DIMS), toVec(K_FPROP_TENSOR_X2_STRIDES));
    auto wDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_W2_UID, toVec(K_FPROP_TENSOR_W2_DIMS), toVec(K_FPROP_TENSOR_W2_STRIDES));
    // Use xDesc2 (same UID as xDesc1, different descriptor object) as Y
    auto op2 = createFinalizedConvOp(xDesc2a.get(), wDesc2.get(), xDesc2.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 2> ops = {op1.get(), op2.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       2,
                       static_cast<const void*>(ops.data()));

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptorOps, SetOperationsRejectsNonOperationDescriptor)
{
    auto graphWrapper = createDescriptor<GraphDescriptor>();
    auto graphDesc = graphWrapper->asDescriptor<GraphDescriptor>();

    // Set handle first
    auto handle = reinterpret_cast<hipdnnHandle_t>(&_mockHandle);
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                            HIPDNN_TYPE_HANDLE,
                            1,
                            static_cast<const void*>(&handle));

    // Create a tensor descriptor (does NOT implement IGraphOperation)
    auto tensorWrapper = createFinalizedTensor(99);
    HipdnnBackendDescriptor* tensorDescPtr = tensorWrapper.get();

    // Attempting to set a non-operation descriptor as an operation should throw
    ASSERT_THROW_HIPDNN_STATUS(graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       static_cast<const void*>(&tensorDescPtr)),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests
// =============================================================================

TEST_F(TestGraphDescriptorOps, GetAttributeReturnsOperationCount)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Query operation count by passing nullptr arrayOfElements
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 1);
}

TEST_F(TestGraphDescriptorOps, GetAttributeReturnsOperationCountMultiple)
{
    auto conv1 = createDefaultConvOp();

    auto xDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_X2_UID, toVec(K_FPROP_TENSOR_X2_DIMS), toVec(K_FPROP_TENSOR_X2_STRIDES));
    auto wDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_W2_UID, toVec(K_FPROP_TENSOR_W2_DIMS), toVec(K_FPROP_TENSOR_W2_STRIDES));
    auto yDesc2 = createFinalizedTensor(
        K_FPROP_TENSOR_Y2_UID, toVec(K_FPROP_TENSOR_Y2_DIMS), toVec(K_FPROP_TENSOR_Y2_STRIDES));
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), yDesc2.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 2> ops = {conv1.convOp.get(), convOp2.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       2,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 2);
}

TEST_F(TestGraphDescriptorOps, GetAttributeReturnsOperations)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Query the actual operations
    int64_t elementCount = 0;
    std::array<HipdnnBackendDescriptor*, 1> returnedOps = {nullptr};
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(returnedOps.data())));
    EXPECT_EQ(elementCount, 1);
    ASSERT_NE(returnedOps[0], nullptr);

    // Verify the returned operation is a conv forward operation
    int64_t opTypeCount = 0;
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    ASSERT_NO_THROW(returnedOps[0]->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType));
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT);

    auto returnedOp0 = std::unique_ptr<HipdnnBackendDescriptor>(returnedOps[0]);
    EXPECT_TRUE(returnedOp0->isFinalized());
}

TEST_F(TestGraphDescriptorOps, GetAttributeReturnsComputeDataType)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    auto computeDt = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeDt);

    desc->finalize();

    // Query compute data type
    int64_t elementCount = 0;
    hipdnnDataType_t queriedType = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &elementCount,
                                       &queriedType));
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(queriedType, HIPDNN_DATA_HALF);
}

TEST_F(TestGraphDescriptorOps, GetAttributeReturnsIntermediateDataType)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    auto intermediateDt = HIPDNN_DATA_BFLOAT16;
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &intermediateDt);

    desc->finalize();

    // Query intermediate data type
    int64_t elementCount = 0;
    hipdnnDataType_t queriedType = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &elementCount,
                                       &queriedType));
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(queriedType, HIPDNN_DATA_BFLOAT16);
}

TEST_F(TestGraphDescriptorOps, GetAttributeReturnsIoDataType)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    auto ioDt = HIPDNN_DATA_DOUBLE;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &ioDt);

    desc->finalize();

    // Query IO data type
    int64_t elementCount = 0;
    hipdnnDataType_t queriedType = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &elementCount,
                                       &queriedType));
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(queriedType, HIPDNN_DATA_DOUBLE);
}

TEST_F(TestGraphDescriptorOps, GetAttributeReturnsPreferredEngineId)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    int64_t engineId = 42;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT, HIPDNN_TYPE_INT64, 1, &engineId);

    desc->finalize();

    // Query preferred engine ID
    int64_t elementCount = 0;
    int64_t queriedEngineId = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &elementCount,
                                       &queriedEngineId));
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(queriedEngineId, 42);
}

TEST_F(TestGraphDescriptorOps, GetAttributePreferredEngineIdCountWhenUnset)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    // Do not set preferred engine ID
    desc->finalize();

    // Query count (should be 0 when not set)
    int64_t elementCount = 99;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT,
                                       HIPDNN_TYPE_INT64,
                                       0,
                                       &elementCount,
                                       nullptr));
    EXPECT_EQ(elementCount, 0);
}

TEST_F(TestGraphDescriptorOps, OperationsPreservedAfterFinalize)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Verify operations are still accessible after finalize + buildGraphFromOperations
    int64_t elementCount = 0;
    std::array<HipdnnBackendDescriptor*, 1> returnedOps = {nullptr};
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(returnedOps.data())));
    EXPECT_EQ(elementCount, 1);
    ASSERT_NE(returnedOps[0], nullptr);

    // Verify the returned operation is a conv forward operation
    int64_t opTypeCount = 0;
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    ASSERT_NO_THROW(returnedOps[0]->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType));
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT);

    auto returnedOp0 = std::unique_ptr<HipdnnBackendDescriptor>(returnedOps[0]);
    EXPECT_TRUE(returnedOp0->isFinalized());
}

TEST_F(TestGraphDescriptorOps, GetAttributeWrongTypeForDataType)
{
    // Build a graph with conv op, set compute data type to HALF, finalize
    auto conv = createDefaultConvOp(HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    auto computeDt = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeDt);

    desc->finalize();

    // Call getAttribute with HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT
    // but pass HIPDNN_TYPE_INT64 instead of HIPDNN_TYPE_DATA_TYPE
    int64_t elementCount = 0;
    int64_t wrongTypeBuffer = 0;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT,
                                                  HIPDNN_TYPE_INT64,
                                                  1,
                                                  &elementCount,
                                                  &wrongTypeBuffer),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptorOps, GetAttributeWrongTypeForPreferredEngineId)
{
    // Build a graph with conv op, set preferred engine id, finalize
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    int64_t engineId = 42;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT, HIPDNN_TYPE_INT64, 1, &engineId);

    desc->finalize();

    // Call getAttribute with HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT
    // but pass HIPDNN_TYPE_DATA_TYPE instead of HIPDNN_TYPE_INT64
    int64_t elementCount = 0;
    hipdnnDataType_t wrongTypeBuffer = HIPDNN_DATA_FLOAT;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT,
                           HIPDNN_TYPE_DATA_TYPE,
                           1,
                           &elementCount,
                           &wrongTypeBuffer),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptorOps, GetAttributeReturnsName)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    const std::string graphName = "MyTestGraph";
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(graphName.size()),
                       graphName.c_str());

    desc->finalize();

    // Query count (getString returns size+1 for null terminator)
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, static_cast<int64_t>(graphName.size() + 1));

    // Query value
    std::vector<char> nameBuffer(static_cast<size_t>(elementCount));
    int64_t actualCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       elementCount,
                                       &actualCount,
                                       nameBuffer.data()));
    EXPECT_STREQ(nameBuffer.data(), graphName.c_str());
}

TEST_F(TestGraphDescriptorOps, GetAttributeNameCountWhenUnset)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    // Do not set name
    desc->finalize();

    // Query count (empty string = just null terminator)
    int64_t elementCount = 99;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 1);
}

TEST_F(TestGraphDescriptorOps, AppendOpsAfterSerialization)
{
    // Set a single op, build the serialized buffer, then append more ops
    auto conv1 = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops1 = {conv1.convOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops1.data())));

    // Build the serialized buffer explicitly
    desc->buildSerializedGraph();
    auto serialized1 = desc->getSerializedGraph();
    ASSERT_NE(serialized1.ptr, nullptr);
    ASSERT_GT(serialized1.size, 0UL);

    // Append a second operation (should succeed and invalidate the cache)
    auto xDesc2 = createFinalizedTensor(
        K_ALT_TENSOR_X_UID, toVec(K_FPROP_TENSOR_X_DIMS), toVec(K_FPROP_TENSOR_X_STRIDES));
    auto wDesc2 = createFinalizedTensor(
        K_ALT_TENSOR_W_UID, toVec(K_FPROP_TENSOR_W_DIMS), toVec(K_FPROP_TENSOR_W_STRIDES));
    auto yDesc2 = createFinalizedTensor(
        K_ALT_TENSOR_Y_UID, toVec(K_FPROP_TENSOR_Y_DIMS), toVec(K_FPROP_TENSOR_Y_STRIDES));
    auto convOp2 = createFinalizedConvOp(xDesc2.get(), wDesc2.get(), yDesc2.get());

    std::array<HipdnnBackendDescriptor*, 1> ops2 = {convOp2.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops2.data())));

    // Verify cache was invalidated by the append
    ASSERT_THROW_HIPDNN_STATUS(desc->getSerializedGraph(), HIPDNN_STATUS_BAD_PARAM);

    // Verify operations count is 2
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    EXPECT_EQ(elementCount, 2);

    // Verify re-serialization reflects both operations
    desc->buildSerializedGraph();
    auto serialized2 = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized2.ptr);
    ASSERT_EQ(graphT->nodes.size(), 2);
    ASSERT_EQ(graphT->tensors.size(), 6);
}

TEST_F(TestGraphDescriptorOps, GetAttributeWrongTypeForName)
{
    auto conv = createDefaultConvOp();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {conv.convOp.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    const std::string graphName = "MyTestGraph";
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(graphName.size()),
                       graphName.c_str());

    desc->finalize();

    // Call getAttribute with HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT
    // but pass HIPDNN_TYPE_INT64 instead of HIPDNN_TYPE_CHAR
    int64_t elementCount = 0;
    int64_t wrongTypeBuffer = 0;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT,
                                                  HIPDNN_TYPE_INT64,
                                                  1,
                                                  &elementCount,
                                                  &wrongTypeBuffer),
                               HIPDNN_STATUS_BAD_PARAM);
}
