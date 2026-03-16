// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionBwdOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/convolution_bwd_attributes_generated.h>
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
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized ConvolutionBwdOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedConvolutionBwdOp(HipdnnBackendDescriptor* dyDesc,
                                    HipdnnBackendDescriptor* wDesc,
                                    HipdnnBackendDescriptor* dxDesc,
                                    hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<ConvolutionBwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<ConvolutionBwdOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DY,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dyDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_W,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&wDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DX,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dxDesc));

    std::vector<int64_t> prePadding = {1, 1};
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, prePadding.data());

    std::vector<int64_t> postPadding = {1, 1};
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, postPadding.data());

    std::vector<int64_t> stride = {1, 1};
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, stride.data());

    std::vector<int64_t> dilation = {1, 1};
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, dilation.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    auto convMode = HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorConvolutionBwd : public ::testing::Test
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

TEST_F(TestGraphDescriptorConvolutionBwd, BuildFromSingleOperation)
{
    auto dyDesc = createFinalizedTensor(10, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto wDesc = createFinalizedTensor(11, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto dxDesc = createFinalizedTensor(12, {1, 3, 32, 32}, {3072, 1024, 32, 1});
    auto opDesc = createFinalizedConvolutionBwdOp(dyDesc.get(), wDesc.get(), dxDesc.get());

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
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::ConvolutionBwdAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsConvolutionBwdAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->dy_tensor_uid, 10);
    EXPECT_EQ(attrs->w_tensor_uid, 11);
    EXPECT_EQ(attrs->dx_tensor_uid, 12);
}

TEST_F(TestGraphDescriptorConvolutionBwd, ComputeDataTypePreserved)
{
    auto dyDesc = createFinalizedTensor(10, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto wDesc = createFinalizedTensor(11, {64, 3, 3, 3}, {27, 9, 3, 1});
    auto dxDesc = createFinalizedTensor(12, {1, 3, 32, 32}, {3072, 1024, 32, 1});
    auto opDesc = createFinalizedConvolutionBwdOp(
        dyDesc.get(), wDesc.get(), dxDesc.get(), HIPDNN_DATA_HALF);

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

} // namespace
