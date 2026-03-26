// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/PoolingFwdOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/pooling_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/PoolingFwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized PoolingFwdOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedPoolingFwdOp(HipdnnBackendDescriptor* xDesc,
HipdnnBackendDescriptor* yDesc,
                          hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<PoolingFwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<PoolingFwdOperationDescriptor>();

    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&xDesc));
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&yDesc));

    auto prePadding = toVec(K_POOL_FWD_PRE_PADDING);
    desc->setAttribute(HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, static_cast<int64_t>(prePadding.size()), prePadding.data());

    auto postPadding = toVec(K_POOL_FWD_POST_PADDING);
    desc->setAttribute(HIPDNN_ATTR_POOLING_POST_PADDINGS, HIPDNN_TYPE_INT64, static_cast<int64_t>(postPadding.size()), postPadding.data());

    auto stride = toVec(K_POOL_FWD_STRIDE);
    desc->setAttribute(HIPDNN_ATTR_POOLING_STRIDES, HIPDNN_TYPE_INT64, static_cast<int64_t>(stride.size()), stride.data());

    auto windowSize = toVec(K_POOL_FWD_WINDOW_SIZE);
    desc->setAttribute(HIPDNN_ATTR_POOLING_WINDOW_SIZE, HIPDNN_TYPE_INT64, static_cast<int64_t>(windowSize.size()), windowSize.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorPoolingFwd : public ::testing::Test
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
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, static_cast<const void*>(&handle));
    }

    static const TensorAttributesT* findTensorByUid(const GraphT& graphT, int64_t uid)
    {
        for(const auto& tensor : graphT.tensors)
        {
            if(tensor->uid == uid)
            {
                return tensor.get();
            }
        }
        return nullptr;
    }

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

    static void verifyPoolingFwdNode(const NodeT& node,
                                  DataType expectedComputeType,
                                  int64_t expectedXUid,
                                  int64_t expectedYUid,
                                  const std::vector<int64_t>& expectedPrePadding,
                                  const std::vector<int64_t>& expectedPostPadding,
                                  const std::vector<int64_t>& expectedStride,
                                  const std::vector<int64_t>& expectedWindowSize,
                                  PoolingMode expectedPoolingMode
)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::PoolingFwdAttributes);

        auto* attrs = node.attributes.AsPoolingFwdAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->x_tensor_uid, expectedXUid);
        EXPECT_EQ(attrs->y_tensor_uid, expectedYUid);
        EXPECT_EQ(attrs->pre_padding, expectedPrePadding);
        EXPECT_EQ(attrs->post_padding, expectedPostPadding);
        EXPECT_EQ(attrs->stride, expectedStride);
        EXPECT_EQ(attrs->window_size, expectedWindowSize);
        EXPECT_EQ(attrs->pooling_mode, expectedPoolingMode);
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

TEST_F(TestGraphDescriptorPoolingFwd, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_X_UID, toVec(K_POOL_FWD_TENSOR_X_DIMS), toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_Y_UID, toVec(K_POOL_FWD_TENSOR_Y_DIMS), toVec(K_POOL_FWD_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedPoolingFwdOp(xDesc.get(), yDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    // Verify the built graph
    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 2);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_POOL_FWD_TENSOR_X_UID),
                 K_POOL_FWD_TENSOR_X_UID,
                 toVec(K_POOL_FWD_TENSOR_X_DIMS),
                 toVec(K_POOL_FWD_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_POOL_FWD_TENSOR_Y_UID),
                 K_POOL_FWD_TENSOR_Y_UID,
                 toVec(K_POOL_FWD_TENSOR_Y_DIMS),
                 toVec(K_POOL_FWD_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    verifyPoolingFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_POOL_FWD_TENSOR_X_UID,
                      K_POOL_FWD_TENSOR_Y_UID,
                      toVec(K_POOL_FWD_PRE_PADDING),
                      toVec(K_POOL_FWD_POST_PADDING),
                      toVec(K_POOL_FWD_STRIDE),
                      toVec(K_POOL_FWD_WINDOW_SIZE),
                      PoolingMode::MAX
);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorPoolingFwd, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_X_UID, toVec(K_POOL_FWD_TENSOR_X_DIMS), toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_Y_UID, toVec(K_POOL_FWD_TENSOR_Y_DIMS), toVec(K_POOL_FWD_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedPoolingFwdOp(xDesc.get(), yDesc.get(), HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorPoolingFwd, PoolingFwdAttributesPreserved)
{
    auto xDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_X_UID, toVec(K_POOL_FWD_TENSOR_X_DIMS), toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_Y_UID, toVec(K_POOL_FWD_TENSOR_Y_DIMS), toVec(K_POOL_FWD_TENSOR_Y_STRIDES));

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<PoolingFwdOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<PoolingFwdOperationDescriptor>();

    HipdnnBackendDescriptor* xPtr = xDesc.get();
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&xPtr));
    HipdnnBackendDescriptor* yPtr = yDesc.get();
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&yPtr));

    const std::vector<int64_t> kCustomPrePadding = {1, 1};
    opDesc->setAttribute(
        HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, kCustomPrePadding.data());

    const std::vector<int64_t> kCustomPostPadding = {1, 1};
    opDesc->setAttribute(
        HIPDNN_ATTR_POOLING_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, kCustomPostPadding.data());

    const std::vector<int64_t> kCustomStride = {2, 2};
    opDesc->setAttribute(
        HIPDNN_ATTR_POOLING_STRIDES, HIPDNN_TYPE_INT64, 2, kCustomStride.data());

    const std::vector<int64_t> kCustomWindowSize = {3, 3};
    opDesc->setAttribute(
        HIPDNN_ATTR_POOLING_WINDOW_SIZE, HIPDNN_TYPE_INT64, 2, kCustomWindowSize.data());

    auto poolingMode = HIPDNN_POOLING_MODE_MAX;
    opDesc->setAttribute(HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, 1, &poolingMode);

    auto computeType = HIPDNN_DATA_FLOAT;
    opDesc->setAttribute(HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set operation name
    const std::string opName = "test_poolingfwd";
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                         HIPDNN_TYPE_CHAR,
                         static_cast<int64_t>(opName.size()),
                         opName.c_str());
    opDesc->finalize();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {wrapper.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 2);

    // Verify tensors
    verifyTensor(findTensorByUid(*graphT, K_POOL_FWD_TENSOR_X_UID),
                 K_POOL_FWD_TENSOR_X_UID,
                 toVec(K_POOL_FWD_TENSOR_X_DIMS),
                 toVec(K_POOL_FWD_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_POOL_FWD_TENSOR_Y_UID),
                 K_POOL_FWD_TENSOR_Y_UID,
                 toVec(K_POOL_FWD_TENSOR_Y_DIMS),
                 toVec(K_POOL_FWD_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyPoolingFwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_POOL_FWD_TENSOR_X_UID,
                      K_POOL_FWD_TENSOR_Y_UID,
                      kCustomPrePadding,
                      kCustomPostPadding,
                      kCustomStride,
                      kCustomWindowSize,
                      PoolingMode::MAX
);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_poolingfwd");
}

} // namespace
