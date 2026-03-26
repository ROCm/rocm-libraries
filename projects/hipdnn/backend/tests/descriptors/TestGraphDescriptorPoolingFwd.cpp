// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/PoolingFwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pooling_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/PoolingFwdConstants.hpp>
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

// Helper: create a finalized PoolingFwdOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedPoolingFwdOp(HipdnnBackendDescriptor* xDesc,
                                HipdnnBackendDescriptor* yDesc,
                                HipdnnBackendDescriptor* indexDesc)
{
    auto wrapper = createDescriptor<PoolingFwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<PoolingFwdOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_INDEX_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&indexDesc));

    auto prePadding = toVec(K_POOL_FWD_PRE_PADDING);
    desc->setAttribute(HIPDNN_ATTR_POOLING_PRE_PADDINGS_EXT,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(prePadding.size()),
                       prePadding.data());

    auto postPadding = toVec(K_POOL_FWD_POST_PADDING);
    desc->setAttribute(HIPDNN_ATTR_POOLING_POST_PADDINGS_EXT,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(postPadding.size()),
                       postPadding.data());

    auto stride = toVec(K_POOL_FWD_STRIDE);
    desc->setAttribute(HIPDNN_ATTR_POOLING_STRIDES_EXT,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(stride.size()),
                       stride.data());

    auto window = toVec(K_POOL_FWD_WINDOW);
    desc->setAttribute(HIPDNN_ATTR_POOLING_WINDOW_EXT,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(window.size()),
                       window.data());

    auto poolingMode = HIPDNN_POOLING_MODE_MAX;
    desc->setAttribute(HIPDNN_ATTR_POOLING_MODE_EXT, HIPDNN_TYPE_POOLING_MODE, 1, &poolingMode);

    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
    desc->setAttribute(
        HIPDNN_ATTR_POOLING_PADDING_MODE_EXT, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);

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
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                           HIPDNN_TYPE_HANDLE,
                           1,
                           static_cast<const void*>(&handle));
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
                                     int64_t expectedXUid,
                                     int64_t expectedYUid,
                                     int64_t expectedIndexUid,
                                     const std::vector<int64_t>& expectedPrePadding,
                                     const std::vector<int64_t>& expectedPostPadding,
                                     const std::vector<int64_t>& expectedStride,
                                     const std::vector<int64_t>& expectedWindow,
                                     PoolingMode expectedPoolingMode,
                                     PaddingMode expectedPaddingMode)
    {
        ASSERT_EQ(node.attributes.type, NodeAttributes::PoolingFwdAttributes);

        auto* attrs = node.attributes.AsPoolingFwdAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->x_tensor_uid, expectedXUid);
        EXPECT_EQ(attrs->y_tensor_uid, expectedYUid);
        EXPECT_EQ(attrs->index_tensor_uid, expectedIndexUid);
        EXPECT_EQ(attrs->pre_padding, expectedPrePadding);
        EXPECT_EQ(attrs->post_padding, expectedPostPadding);
        EXPECT_EQ(attrs->stride, expectedStride);
        EXPECT_EQ(attrs->window, expectedWindow);
        EXPECT_EQ(attrs->pooling_mode, expectedPoolingMode);
        EXPECT_EQ(attrs->padding_mode, expectedPaddingMode);
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
    auto xDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_X_UID,
                                       toVec(K_POOL_FWD_TENSOR_X_DIMS),
                                       toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_Y_UID,
                                       toVec(K_POOL_FWD_TENSOR_Y_DIMS),
                                       toVec(K_POOL_FWD_TENSOR_Y_STRIDES));
    auto indexDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_INDEX_UID,
                                           toVec(K_POOL_FWD_TENSOR_INDEX_DIMS),
                                           toVec(K_POOL_FWD_TENSOR_INDEX_STRIDES));
    auto opDesc = createFinalizedPoolingFwdOp(xDesc.get(), yDesc.get(), indexDesc.get());

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
    verifyTensor(findTensorByUid(*graphT, K_POOL_FWD_TENSOR_INDEX_UID),
                 K_POOL_FWD_TENSOR_INDEX_UID,
                 toVec(K_POOL_FWD_TENSOR_INDEX_DIMS),
                 toVec(K_POOL_FWD_TENSOR_INDEX_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    verifyPoolingFwdNode(*graphT->nodes[0],
                         K_POOL_FWD_TENSOR_X_UID,
                         K_POOL_FWD_TENSOR_Y_UID,
                         K_POOL_FWD_TENSOR_INDEX_UID,
                         toVec(K_POOL_FWD_PRE_PADDING),
                         toVec(K_POOL_FWD_POST_PADDING),
                         toVec(K_POOL_FWD_STRIDE),
                         toVec(K_POOL_FWD_WINDOW),
                         PoolingMode::MAX_POOLING,
                         PaddingMode::ZERO_PAD);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorPoolingFwd, PoolingFwdAttributesPreserved)
{
    auto xDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_X_UID,
                                       toVec(K_POOL_FWD_TENSOR_X_DIMS),
                                       toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_Y_UID,
                                       toVec(K_POOL_FWD_TENSOR_Y_DIMS),
                                       toVec(K_POOL_FWD_TENSOR_Y_STRIDES));
    auto indexDesc = createFinalizedTensor(K_POOL_FWD_TENSOR_INDEX_UID,
                                           toVec(K_POOL_FWD_TENSOR_INDEX_DIMS),
                                           toVec(K_POOL_FWD_TENSOR_INDEX_STRIDES));

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<PoolingFwdOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<PoolingFwdOperationDescriptor>();

    HipdnnBackendDescriptor* const xPtr = xDesc.get(); // NOLINT(misc-const-correctness)
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&xPtr));
    HipdnnBackendDescriptor* const yPtr = yDesc.get(); // NOLINT(misc-const-correctness)
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&yPtr));
    HipdnnBackendDescriptor* const indexPtr = indexDesc.get(); // NOLINT(misc-const-correctness)
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_INDEX_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&indexPtr));

    const std::vector<int64_t> kCustomPrePadding = {1, 1};
    opDesc->setAttribute(
        HIPDNN_ATTR_POOLING_PRE_PADDINGS_EXT, HIPDNN_TYPE_INT64, 2, kCustomPrePadding.data());

    const std::vector<int64_t> kCustomPostPadding = {1, 1};
    opDesc->setAttribute(
        HIPDNN_ATTR_POOLING_POST_PADDINGS_EXT, HIPDNN_TYPE_INT64, 2, kCustomPostPadding.data());

    const std::vector<int64_t> kCustomStride = {2, 2};
    opDesc->setAttribute(
        HIPDNN_ATTR_POOLING_STRIDES_EXT, HIPDNN_TYPE_INT64, 2, kCustomStride.data());

    const std::vector<int64_t> kCustomWindow = {3, 3};
    opDesc->setAttribute(
        HIPDNN_ATTR_POOLING_WINDOW_EXT, HIPDNN_TYPE_INT64, 2, kCustomWindow.data());

    auto poolingMode = HIPDNN_POOLING_MODE_MAX;
    opDesc->setAttribute(HIPDNN_ATTR_POOLING_MODE_EXT, HIPDNN_TYPE_POOLING_MODE, 1, &poolingMode);

    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
    opDesc->setAttribute(
        HIPDNN_ATTR_POOLING_PADDING_MODE_EXT, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);

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
    verifyTensor(findTensorByUid(*graphT, K_POOL_FWD_TENSOR_INDEX_UID),
                 K_POOL_FWD_TENSOR_INDEX_UID,
                 toVec(K_POOL_FWD_TENSOR_INDEX_DIMS),
                 toVec(K_POOL_FWD_TENSOR_INDEX_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyPoolingFwdNode(*graphT->nodes[0],
                         K_POOL_FWD_TENSOR_X_UID,
                         K_POOL_FWD_TENSOR_Y_UID,
                         K_POOL_FWD_TENSOR_INDEX_UID,
                         kCustomPrePadding,
                         kCustomPostPadding,
                         kCustomStride,
                         kCustomWindow,
                         PoolingMode::MAX_POOLING,
                         PaddingMode::ZERO_PAD);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_poolingfwd");
}

} // namespace
