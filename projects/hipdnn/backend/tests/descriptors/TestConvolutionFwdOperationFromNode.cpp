// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionFwdOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;

// =============================================================================
// ConvolutionFwdOperationDescriptor::fromNode() Tests
// =============================================================================

class TestConvolutionFwdOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_FPROP_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = hipdnn_tests::toVec(K_FPROP_TENSOR_X_DIMS);
        xAttrs.strides = hipdnn_tests::toVec(K_FPROP_TENSOR_X_STRIDES);

        _tensorMap[K_FPROP_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);

        TensorAttributesT wAttrs;
        wAttrs.uid = K_FPROP_TENSOR_W_UID;
        wAttrs.data_type = DataType::FLOAT;
        wAttrs.dims = hipdnn_tests::toVec(K_FPROP_TENSOR_W_DIMS);
        wAttrs.strides = hipdnn_tests::toVec(K_FPROP_TENSOR_W_STRIDES);

        _tensorMap[K_FPROP_TENSOR_W_UID] = TensorDescriptor::fromFlatBuffer(wAttrs);

        TensorAttributesT yAttrs;
        yAttrs.uid = K_FPROP_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = hipdnn_tests::toVec(K_FPROP_TENSOR_Y_DIMS);
        yAttrs.strides = hipdnn_tests::toVec(K_FPROP_TENSOR_Y_STRIDES);

        _tensorMap[K_FPROP_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributesT
        createStandardConvolutionFwdAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributesT attrs;
        attrs.x_tensor_uid = K_FPROP_TENSOR_X_UID;
        attrs.w_tensor_uid = K_FPROP_TENSOR_W_UID;
        attrs.y_tensor_uid = K_FPROP_TENSOR_Y_UID;
        attrs.pre_padding = hipdnn_tests::toVec(K_FPROP_CONV_PADDING);
        attrs.post_padding = hipdnn_tests::toVec(K_FPROP_CONV_PADDING);
        attrs.stride = hipdnn_tests::toVec(K_FPROP_CONV_STRIDE);
        attrs.dilation = hipdnn_tests::toVec(K_FPROP_CONV_DILATION);
        attrs.conv_mode = ConvMode::CROSS_CORRELATION;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardConvolutionFwdAttrs());
        return node;
    }
};

TEST_F(TestConvolutionFwdOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_FPROP_TENSOR_X_UID);
}

TEST_F(TestConvolutionFwdOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createStandardNode();

    // NodeFactory::createOperationFromNode delegates to fromNode internally.
    // Verify the delegation produces a valid, correctly-typed descriptor.
    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
    ASSERT_NE(graphOp, nullptr);

    // Verify the factory dispatched to the correct operation type, then static_cast.
    // Cannot use dynamic_pointer_cast: backend tests compile with -fno-rtti.
    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ConvolutionFwdAttributes);
    auto desc = std::static_pointer_cast<ConvolutionFwdOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().w_tensor_uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_FPROP_TENSOR_Y_UID);
    EXPECT_EQ(desc->getData().pre_padding, hipdnn_tests::toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(desc->getData().post_padding, hipdnn_tests::toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(desc->getData().stride, hipdnn_tests::toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(desc->getData().dilation, hipdnn_tests::toVec(K_FPROP_CONV_DILATION));
    EXPECT_EQ(desc->getData().conv_mode, ConvMode::CROSS_CORRELATION);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(desc->getWDesc()->getData().uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_FPROP_TENSOR_Y_UID);
}

TEST_F(TestConvolutionFwdOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestConvolutionFwdOperationFromNode, PreservesConvMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardConvolutionFwdAttrs();
    attrs.conv_mode = ConvMode::CONVOLUTION;
    node.attributes.Set(attrs);
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().conv_mode, ConvMode::CONVOLUTION);
}

TEST_F(TestConvolutionFwdOperationFromNode, PreservesDataFields)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getData().pre_padding, hipdnn_tests::toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(desc->getData().post_padding, hipdnn_tests::toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(desc->getData().stride, hipdnn_tests::toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(desc->getData().dilation, hipdnn_tests::toVec(K_FPROP_CONV_DILATION));
    EXPECT_EQ(desc->getData().conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionFwdOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_FPROP_TENSOR_X_UID);
    ASSERT_NE(desc->getWDesc(), nullptr);
    EXPECT_EQ(desc->getWDesc()->getData().uid, K_FPROP_TENSOR_W_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_FPROP_TENSOR_Y_UID);
}

TEST_F(TestConvolutionFwdOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_FPROP_TENSOR_X_UID]);
    EXPECT_EQ(desc->getWDesc(), _tensorMap[K_FPROP_TENSOR_W_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_FPROP_TENSOR_Y_UID]);
}

TEST_F(TestConvolutionFwdOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, hipdnn_tests::toVec(K_FPROP_TENSOR_X_DIMS));
    EXPECT_EQ(desc->getXDesc()->getData().strides, hipdnn_tests::toVec(K_FPROP_TENSOR_X_STRIDES));

    ASSERT_NE(desc->getWDesc(), nullptr);
    EXPECT_EQ(desc->getWDesc()->getData().uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(desc->getWDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getWDesc()->getData().dims, hipdnn_tests::toVec(K_FPROP_TENSOR_W_DIMS));
    EXPECT_EQ(desc->getWDesc()->getData().strides, hipdnn_tests::toVec(K_FPROP_TENSOR_W_STRIDES));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_FPROP_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, hipdnn_tests::toVec(K_FPROP_TENSOR_Y_DIMS));
    EXPECT_EQ(desc->getYDesc()->getData().strides, hipdnn_tests::toVec(K_FPROP_TENSOR_Y_STRIDES));
}

TEST_F(TestConvolutionFwdOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_FPROP_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestConvolutionFwdOperationFromNode, FailsWithMissingWTensor)
{
    _tensorMap.erase(K_FPROP_TENSOR_W_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestConvolutionFwdOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_FPROP_TENSOR_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestConvolutionFwdOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_FPROP_TENSOR_Y_UID);
}

TEST_F(TestConvolutionFwdOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ConvolutionFwdAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->w_tensor_uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_FPROP_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->pre_padding, hipdnn_tests::toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(rebuiltAttrs->post_padding, hipdnn_tests::toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(rebuiltAttrs->stride, hipdnn_tests::toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(rebuiltAttrs->dilation, hipdnn_tests::toVec(K_FPROP_CONV_DILATION));
    EXPECT_EQ(rebuiltAttrs->conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionFwdOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &prePaddingCount,
                       prePadding.data());
    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, hipdnn_tests::toVec(K_FPROP_CONV_PADDING));

    // Verify post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &postPaddingCount,
                       postPadding.data());
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, hipdnn_tests::toVec(K_FPROP_CONV_PADDING));

    // Verify stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data());
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, hipdnn_tests::toVec(K_FPROP_CONV_STRIDE));

    // Verify dilation
    std::vector<int64_t> dilation(2);
    int64_t dilationCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, &dilationCount, dilation.data());
    ASSERT_EQ(dilationCount, 2);
    EXPECT_EQ(dilation, hipdnn_tests::toVec(K_FPROP_CONV_DILATION));

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify conv_mode
    hipdnnConvolutionMode_t convMode = {};
    int64_t convModeCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                       HIPDNN_TYPE_CONVOLUTION_MODE,
                       1,
                       &convModeCount,
                       &convMode);
    ASSERT_EQ(convMode, HIPDNN_CROSS_CORRELATION);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        xScoped.get(),
        K_FPROP_TENSOR_X_UID,
        HIPDNN_DATA_FLOAT,
        hipdnn_tests::toVec(K_FPROP_TENSOR_X_DIMS),
        hipdnn_tests::toVec(K_FPROP_TENSOR_X_STRIDES));

    // Verify w tensor
    hipdnn_backend::ScopedDescriptor wScoped;
    int64_t wCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &wCount,
                       static_cast<void*>(wScoped.getPtr()));
    ASSERT_EQ(wCount, 1);
    ASSERT_NE(wScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        wScoped.get(),
        K_FPROP_TENSOR_W_UID,
        HIPDNN_DATA_FLOAT,
        hipdnn_tests::toVec(K_FPROP_TENSOR_W_DIMS),
        hipdnn_tests::toVec(K_FPROP_TENSOR_W_STRIDES));

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        yScoped.get(),
        K_FPROP_TENSOR_Y_UID,
        HIPDNN_DATA_FLOAT,
        hipdnn_tests::toVec(K_FPROP_TENSOR_Y_DIMS),
        hipdnn_tests::toVec(K_FPROP_TENSOR_Y_STRIDES));

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT);
}

TEST_F(TestConvolutionFwdOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_convolutionfwd_1";

    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_convolutionfwd_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_convolutionfwd_1");
}

TEST_F(TestConvolutionFwdOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestConvolutionFwdOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestConvolutionFwdOperationFromNode, ToStringIncludesName)
{
    auto node = createStandardNode();
    node.name = "my_fwd_op";

    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);
    auto str = desc->toString();

    EXPECT_NE(str.find("name=my_fwd_op"), std::string::npos);
}
