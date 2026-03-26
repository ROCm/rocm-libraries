// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/PoolingBwdOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/pooling_bwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;

namespace
{

// Helper: create a finalized PoolingBwdOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedPoolingBwdOp(HipdnnBackendDescriptor* dyDesc,
HipdnnBackendDescriptor* dxDesc,
                          hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<PoolingBwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<PoolingBwdOperationDescriptor>();

    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&dyDesc));
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&dxDesc));

    std::vector<int64_t> prePadding = {1, 1};
    desc->setAttribute(HIPDNN_ATTR_POOLING_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());

    std::vector<int64_t> postPadding = {1, 1};
    desc->setAttribute(HIPDNN_ATTR_POOLING_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());

    std::vector<int64_t> stride = {2, 2};
    desc->setAttribute(HIPDNN_ATTR_POOLING_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());

    std::vector<int64_t> windowSize = {3, 3};
    desc->setAttribute(HIPDNN_ATTR_POOLING_WINDOW_SIZE, HIPDNN_TYPE_INT64, 2, windowSize.data());
    desc->setAttribute(HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorPoolingBwd : public ::testing::Test
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

    static void verifyPoolingBwdNode(const NodeT& node,
                                  DataType expectedComputeType,
                                  int64_t expectedDyUid,
                                  int64_t expectedDxUid,
                                  const std::vector<int64_t>& expectedPrePadding,
                                  const std::vector<int64_t>& expectedPostPadding,
                                  const std::vector<int64_t>& expectedStride,
                                  const std::vector<int64_t>& expectedWindowSize,
                                  PoolingMode expectedPoolingMode
)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::PoolingBwdAttributes);

        auto* attrs = node.attributes.AsPoolingBwdAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->dy_tensor_uid, expectedDyUid);
        EXPECT_EQ(attrs->dx_tensor_uid, expectedDxUid);
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

TEST_F(TestGraphDescriptorPoolingBwd, BuildFromSingleOperation)
{
    auto dyDesc = createFinalizedTensor(42, {1, 3, 16, 16}, {768, 256, 16, 1});
    auto dxDesc = createFinalizedTensor(43, {1, 3, 32, 32}, {3072, 1024, 32, 1});
    auto opDesc = createFinalizedPoolingBwdOp(dyDesc.get(), dxDesc.get());

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
    verifyTensor(findTensorByUid(*graphT, 42),
                 42,
                 {1, 3, 16, 16},
                 {768, 256, 16, 1},
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 43),
                 43,
                 {1, 3, 32, 32},
                 {3072, 1024, 32, 1},
                 DataType::FLOAT);

    // Verify node attributes
    verifyPoolingBwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      42,
                      43,
                      std::vector<int64_t>{1, 1},
                      std::vector<int64_t>{1, 1},
                      std::vector<int64_t>{2, 2},
                      std::vector<int64_t>{3, 3},
                      PoolingMode::MAX
);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorPoolingBwd, ComputeDataTypePreserved)
{
    auto dyDesc = createFinalizedTensor(42, {1, 3, 16, 16}, {768, 256, 16, 1});
    auto dxDesc = createFinalizedTensor(43, {1, 3, 32, 32}, {3072, 1024, 32, 1});
    auto opDesc = createFinalizedPoolingBwdOp(dyDesc.get(), dxDesc.get(), HIPDNN_DATA_HALF);

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

TEST_F(TestGraphDescriptorPoolingBwd, PoolingBwdAttributesPreserved)
{
    auto dyDesc = createFinalizedTensor(42, {1, 3, 16, 16}, {768, 256, 16, 1});
    auto dxDesc = createFinalizedTensor(43, {1, 3, 32, 32}, {3072, 1024, 32, 1});

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<PoolingBwdOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<PoolingBwdOperationDescriptor>();

    HipdnnBackendDescriptor* dyPtr = dyDesc.get();
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&dyPtr));
    HipdnnBackendDescriptor* dxPtr = dxDesc.get();
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&dxPtr));

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
    const std::string opName = "test_poolingbwd";
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
    verifyTensor(findTensorByUid(*graphT, 42),
                 42,
                 {1, 3, 16, 16},
                 {768, 256, 16, 1},
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, 43),
                 43,
                 {1, 3, 32, 32},
                 {3072, 1024, 32, 1},
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyPoolingBwdNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      42,
                      43,
                      kCustomPrePadding,
                      kCustomPostPadding,
                      kCustomStride,
                      kCustomWindowSize,
                      PoolingMode::MAX
);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_poolingbwd");
}

} // namespace
