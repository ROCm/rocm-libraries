// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BlockScaleQuantizeOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BlockScaleQuantizeConstants.hpp>
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

// Helper: create a finalized BlockScaleQuantizeOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedBlockScaleQuantizeOp(HipdnnBackendDescriptor* xDesc,
                                        HipdnnBackendDescriptor* yDesc,
                                        HipdnnBackendDescriptor* scaleDesc,
                                        hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<BlockScaleQuantizeOperationDescriptor>();
    auto desc = wrapper->asDescriptor<BlockScaleQuantizeOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));

    int32_t blockSize = K_BSQ_BLOCK_SIZE;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE_EXT,
                       HIPDNN_TYPE_INT32,
                       1,
                       &blockSize);

    int64_t axis = 1;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT, HIPDNN_TYPE_INT64, 1, &axis);

    bool transpose = true;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &transpose);

    desc->setAttribute(
        HIPDNN_ATTR_BLOCK_SCALE_QUANTIZE_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorBlockScaleQuantize : public ::testing::Test
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

TEST_F(TestGraphDescriptorBlockScaleQuantize, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(
        K_BSQ_TENSOR_X_UID, toVec(K_BSQ_TENSOR_X_DIMS), toVec(K_BSQ_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSQ_TENSOR_Y_UID, toVec(K_BSQ_TENSOR_Y_DIMS), toVec(K_BSQ_TENSOR_Y_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSQ_TENSOR_SCALE_UID, toVec(K_BSQ_TENSOR_SCALE_DIMS), toVec(K_BSQ_TENSOR_SCALE_STRIDES));
    auto opDesc = createFinalizedBlockScaleQuantizeOp(xDesc.get(), yDesc.get(), scaleDesc.get());

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

    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::BlockScaleQuantizeAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsBlockScaleQuantizeAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->x_tensor_uid, K_BSQ_TENSOR_X_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_BSQ_TENSOR_Y_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BSQ_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->block_size, K_BSQ_BLOCK_SIZE);
    EXPECT_TRUE(attrs->axis.has_value());
    EXPECT_EQ(attrs->axis.value(), 1);
    EXPECT_TRUE(attrs->transpose);
}

TEST_F(TestGraphDescriptorBlockScaleQuantize, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(
        K_BSQ_TENSOR_X_UID, toVec(K_BSQ_TENSOR_X_DIMS), toVec(K_BSQ_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BSQ_TENSOR_Y_UID, toVec(K_BSQ_TENSOR_Y_DIMS), toVec(K_BSQ_TENSOR_Y_STRIDES));
    auto scaleDesc = createFinalizedTensor(
        K_BSQ_TENSOR_SCALE_UID, toVec(K_BSQ_TENSOR_SCALE_DIMS), toVec(K_BSQ_TENSOR_SCALE_STRIDES));
    auto opDesc = createFinalizedBlockScaleQuantizeOp(
        xDesc.get(), yDesc.get(), scaleDesc.get(), HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = GetGraph(serialized.ptr)->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

} // namespace
