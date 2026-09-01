// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/PointwiseOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/PointwiseConstants.hpp>
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

// Helper: create a finalized PointwiseOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedPointwiseOp(HipdnnBackendDescriptor* in0Desc,
                               HipdnnBackendDescriptor* out0Desc,
                               HipdnnBackendDescriptor* in1Desc,
                               HipdnnBackendDescriptor* in2Desc,
                               hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<PointwiseOperationDescriptor>();
    auto desc = wrapper->asDescriptor<PointwiseOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&in0Desc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&out0Desc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&in1Desc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&in2Desc));

    auto operation = HIPDNN_POINTWISE_ADD;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

// A finalized virtual tensor, used as an invalid ragged-offset aux.
inline std::unique_ptr<HipdnnBackendDescriptor> createFinalizedVirtualTensor(int64_t uid)
{
    auto wrapper = createDescriptor<TensorDescriptor>();
    auto desc = wrapper->asDescriptor<TensorDescriptor>();
    auto dims = toVec(K_PW_TENSOR_DIMS);
    auto strides = toVec(K_PW_TENSOR_STRIDES);
    auto dataType = HIPDNN_DATA_INT64;
    bool isVirtual = true;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid);
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(dims.size()),
                       dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(strides.size()),
                       strides.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    desc->setAttribute(HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, &isVirtual);
    desc->finalize();
    return wrapper;
}

// A finalized (non-virtual) tensor whose ragged-offset points at auxDesc. A plain
// auxDesc yields a valid primary; an auxDesc that itself carries a ragged offset
// yields the nested case.
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedTensorWithRaggedOffset(int64_t uid, HipdnnBackendDescriptor* auxDesc)
{
    auto wrapper = createDescriptor<TensorDescriptor>();
    auto desc = wrapper->asDescriptor<TensorDescriptor>();
    auto dims = toVec(K_PW_TENSOR_DIMS);
    auto strides = toVec(K_PW_TENSOR_STRIDES);
    auto dataType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid);
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(dims.size()),
                       dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(strides.size()),
                       strides.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    desc->setAttribute(HIPDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&auxDesc));
    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorPointwise : public ::testing::Test
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

TEST_F(TestGraphDescriptorPointwise, BuildFromSingleOperation)
{
    auto in0Desc = createFinalizedTensor(
        K_PW_TENSOR_IN0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
    auto out0Desc = createFinalizedTensor(
        K_PW_TENSOR_OUT0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
    auto in1Desc = createFinalizedTensor(K_PW_TENSOR_IN1_UID);
    auto in2Desc = createFinalizedTensor(K_PW_TENSOR_IN2_UID);
    auto opDesc
        = createFinalizedPointwiseOp(in0Desc.get(), out0Desc.get(), in1Desc.get(), in2Desc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 4);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::PointwiseAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsPointwiseAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->in_0_tensor_uid, K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(attrs->out_0_tensor_uid, K_PW_TENSOR_OUT0_UID);
    EXPECT_EQ(attrs->in_1_tensor_uid, K_PW_TENSOR_IN1_UID);
    EXPECT_EQ(attrs->in_2_tensor_uid, K_PW_TENSOR_IN2_UID);
    EXPECT_FALSE(attrs->axis_tensor_uid.has_value());
}

TEST_F(TestGraphDescriptorPointwise, ComputeDataTypePreserved)
{
    auto in0Desc = createFinalizedTensor(
        K_PW_TENSOR_IN0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
    auto out0Desc = createFinalizedTensor(
        K_PW_TENSOR_OUT0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
    auto in1Desc = createFinalizedTensor(K_PW_TENSOR_IN1_UID);
    auto in2Desc = createFinalizedTensor(K_PW_TENSOR_IN2_UID);
    auto opDesc = createFinalizedPointwiseOp(
        in0Desc.get(), out0Desc.get(), in1Desc.get(), in2Desc.get(), HIPDNN_DATA_HALF);

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

// Aux UIDs 9001/9002 do not collide with K_PW_TENSOR_{IN0,IN1,IN2,OUT0}_UID.
TEST_F(TestGraphDescriptorPointwise, BuildRejectsVirtualRaggedAux)
{
    auto aux = createFinalizedVirtualTensor(9001);
    auto in0 = createFinalizedTensorWithRaggedOffset(K_PW_TENSOR_IN0_UID, aux.get());
    auto out0 = createFinalizedTensor(
        K_PW_TENSOR_OUT0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
    auto in1 = createFinalizedTensor(K_PW_TENSOR_IN1_UID);
    auto in2 = createFinalizedTensor(K_PW_TENSOR_IN2_UID);
    auto op = createFinalizedPointwiseOp(in0.get(), out0.get(), in1.get(), in2.get());

    auto desc = getDescriptor();
    setHandle();
    std::array<HipdnnBackendDescriptor*, 1> ops = {op.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGraphDescriptorPointwise, BuildRejectsNestedRaggedAux)
{
    auto auxInner = createFinalizedTensor(9002);
    auto auxOuter = createFinalizedTensorWithRaggedOffset(9001, auxInner.get());
    auto in0 = createFinalizedTensorWithRaggedOffset(K_PW_TENSOR_IN0_UID, auxOuter.get());
    auto out0 = createFinalizedTensor(
        K_PW_TENSOR_OUT0_UID, toVec(K_PW_TENSOR_DIMS), toVec(K_PW_TENSOR_STRIDES));
    auto in1 = createFinalizedTensor(K_PW_TENSOR_IN1_UID);
    auto in2 = createFinalizedTensor(K_PW_TENSOR_IN2_UID);
    auto op = createFinalizedPointwiseOp(in0.get(), out0.get(), in1.get(), in2.get());

    auto desc = getDescriptor();
    setHandle();
    std::array<HipdnnBackendDescriptor*, 1> ops = {op.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

} // namespace
