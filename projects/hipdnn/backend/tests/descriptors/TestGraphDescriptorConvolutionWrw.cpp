// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionWrwOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvWgradConstants.hpp>
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

// Helper: create a finalized ConvolutionWrwOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedConvolutionWrwOp(HipdnnBackendDescriptor* xDesc,
                                    HipdnnBackendDescriptor* dyDesc,
                                    HipdnnBackendDescriptor* dwDesc,
                                    hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                    const std::string& name = "")
{
    auto wrapper = createDescriptor<ConvolutionWrwOperationDescriptor>();
    auto desc = wrapper->asDescriptor<ConvolutionWrwOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dyDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dwDesc));

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

    auto convMode = HIPDNN_CROSS_CORRELATION;
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);

    if(!name.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(name.size()),
                           name.c_str());
    }

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorConvolutionWrw : public ::testing::Test
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

TEST_F(TestGraphDescriptorConvolutionWrw, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_X_UID, toVec(K_WGRAD_TENSOR_X_DIMS), toVec(K_WGRAD_TENSOR_X_STRIDES));
    auto dyDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_DY_UID, toVec(K_WGRAD_TENSOR_DY_DIMS), toVec(K_WGRAD_TENSOR_DY_STRIDES));
    auto dwDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_DW_UID, toVec(K_WGRAD_TENSOR_DW_DIMS), toVec(K_WGRAD_TENSOR_DW_STRIDES));
    auto opDesc = createFinalizedConvolutionWrwOp(xDesc.get(), dyDesc.get(), dwDesc.get());

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
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::ConvolutionWrwAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsConvolutionWrwAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->x_tensor_uid, K_WGRAD_TENSOR_X_UID);
    EXPECT_EQ(attrs->dy_tensor_uid, K_WGRAD_TENSOR_DY_UID);
    EXPECT_EQ(attrs->dw_tensor_uid, K_WGRAD_TENSOR_DW_UID);
}

TEST_F(TestGraphDescriptorConvolutionWrw, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_X_UID, toVec(K_WGRAD_TENSOR_X_DIMS), toVec(K_WGRAD_TENSOR_X_STRIDES));
    auto dyDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_DY_UID, toVec(K_WGRAD_TENSOR_DY_DIMS), toVec(K_WGRAD_TENSOR_DY_STRIDES));
    auto dwDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_DW_UID, toVec(K_WGRAD_TENSOR_DW_DIMS), toVec(K_WGRAD_TENSOR_DW_STRIDES));
    auto opDesc = createFinalizedConvolutionWrwOp(
        xDesc.get(), dyDesc.get(), dwDesc.get(), HIPDNN_DATA_HALF);

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

TEST_F(TestGraphDescriptorConvolutionWrw, OperationNamePreservedInSerialization)
{
    auto xDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_X_UID, toVec(K_WGRAD_TENSOR_X_DIMS), toVec(K_WGRAD_TENSOR_X_STRIDES));
    auto dyDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_DY_UID, toVec(K_WGRAD_TENSOR_DY_DIMS), toVec(K_WGRAD_TENSOR_DY_STRIDES));
    auto dwDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_DW_UID, toVec(K_WGRAD_TENSOR_DW_DIMS), toVec(K_WGRAD_TENSOR_DW_STRIDES));
    auto opDesc = createFinalizedConvolutionWrwOp(
        xDesc.get(), dyDesc.get(), dwDesc.get(), HIPDNN_DATA_FLOAT, "test_wrw_op");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_wrw_op");
}

TEST_F(TestGraphDescriptorConvolutionWrw, OperationNameRoundTripThroughLifting)
{
    auto xDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_X_UID, toVec(K_WGRAD_TENSOR_X_DIMS), toVec(K_WGRAD_TENSOR_X_STRIDES));
    auto dyDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_DY_UID, toVec(K_WGRAD_TENSOR_DY_DIMS), toVec(K_WGRAD_TENSOR_DY_STRIDES));
    auto dwDesc = createFinalizedTensor(
        K_WGRAD_TENSOR_DW_UID, toVec(K_WGRAD_TENSOR_DW_DIMS), toVec(K_WGRAD_TENSOR_DW_STRIDES));
    auto opDesc = createFinalizedConvolutionWrwOp(
        xDesc.get(), dyDesc.get(), dwDesc.get(), HIPDNN_DATA_FLOAT, "lift_wrw_name");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Serialize and deserialize via FlatBuffer
    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    // Rebuild from the deserialized graph using NodeFactory
    auto tensorMap = NodeFactory::buildTensorMap(graphT->tensors);
    ASSERT_EQ(graphT->nodes.size(), 1);

    auto rebuilt = NodeFactory::createOperationFromNode(*graphT->nodes[0], tensorMap);
    ASSERT_NE(rebuilt, nullptr);

    auto* graphOp = rebuilt->asGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    auto rebuiltNode = graphOp->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "lift_wrw_name");
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ConvolutionWrwAttributes);
}

} // namespace
