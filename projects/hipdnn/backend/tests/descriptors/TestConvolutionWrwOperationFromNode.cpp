// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionWrwOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvWgradConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;

// =============================================================================
// ConvolutionWrwOperationDescriptor::fromNode() Tests
// =============================================================================

class TestConvolutionWrwOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_WGRAD_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = hipdnn_tests::toVec(K_WGRAD_TENSOR_X_DIMS);
        xAttrs.strides = hipdnn_tests::toVec(K_WGRAD_TENSOR_X_STRIDES);

        _tensorMap[K_WGRAD_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT dyAttrs;
        dyAttrs.uid = K_WGRAD_TENSOR_DY_UID;
        dyAttrs.data_type = DataType::FLOAT;
        dyAttrs.dims = hipdnn_tests::toVec(K_WGRAD_TENSOR_DY_DIMS);
        dyAttrs.strides = hipdnn_tests::toVec(K_WGRAD_TENSOR_DY_STRIDES);

        _tensorMap[K_WGRAD_TENSOR_DY_UID] = TensorDescriptor::fromFlatBuffer(dyAttrs);
        TensorAttributesT dwAttrs;
        dwAttrs.uid = K_WGRAD_TENSOR_DW_UID;
        dwAttrs.data_type = DataType::FLOAT;
        dwAttrs.dims = hipdnn_tests::toVec(K_WGRAD_TENSOR_DW_DIMS);
        dwAttrs.strides = hipdnn_tests::toVec(K_WGRAD_TENSOR_DW_STRIDES);

        _tensorMap[K_WGRAD_TENSOR_DW_UID] = TensorDescriptor::fromFlatBuffer(dwAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributesT
        createStandardConvolutionWrwAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributesT attrs;
        attrs.x_tensor_uid = K_WGRAD_TENSOR_X_UID;
        attrs.dy_tensor_uid = K_WGRAD_TENSOR_DY_UID;
        attrs.dw_tensor_uid = K_WGRAD_TENSOR_DW_UID;
        attrs.pre_padding = hipdnn_tests::toVec(K_WGRAD_CONV_PADDING);
        attrs.post_padding = hipdnn_tests::toVec(K_WGRAD_CONV_PADDING);
        attrs.stride = hipdnn_tests::toVec(K_WGRAD_CONV_STRIDE);
        attrs.dilation = hipdnn_tests::toVec(K_WGRAD_CONV_DILATION);
        attrs.conv_mode = ConvMode::CROSS_CORRELATION;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardConvolutionWrwAttrs());
        return node;
    }
};

TEST_F(TestConvolutionWrwOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_WGRAD_TENSOR_X_UID);
}

TEST_F(TestConvolutionWrwOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ConvolutionWrwAttributes);
    auto desc = std::static_pointer_cast<ConvolutionWrwOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_WGRAD_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().dy_tensor_uid, K_WGRAD_TENSOR_DY_UID);
    EXPECT_EQ(desc->getData().dw_tensor_uid, K_WGRAD_TENSOR_DW_UID);
    EXPECT_EQ(desc->getData().pre_padding, hipdnn_tests::toVec(K_WGRAD_CONV_PADDING));
    EXPECT_EQ(desc->getData().post_padding, hipdnn_tests::toVec(K_WGRAD_CONV_PADDING));
    EXPECT_EQ(desc->getData().stride, hipdnn_tests::toVec(K_WGRAD_CONV_STRIDE));
    EXPECT_EQ(desc->getData().dilation, hipdnn_tests::toVec(K_WGRAD_CONV_DILATION));
    EXPECT_EQ(desc->getData().conv_mode, ConvMode::CROSS_CORRELATION);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_WGRAD_TENSOR_X_UID);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_WGRAD_TENSOR_DY_UID);
    EXPECT_EQ(desc->getDwDesc()->getData().uid, K_WGRAD_TENSOR_DW_UID);
}

TEST_F(TestConvolutionWrwOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestConvolutionWrwOperationFromNode, PreservesConvMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardConvolutionWrwAttrs();
    attrs.conv_mode = ConvMode::CONVOLUTION;
    node.attributes.Set(attrs);
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().conv_mode, ConvMode::CONVOLUTION);
}

TEST_F(TestConvolutionWrwOperationFromNode, PreservesDataFields)
{
    auto node = createStandardNode();
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getData().pre_padding, hipdnn_tests::toVec(K_WGRAD_CONV_PADDING));
    EXPECT_EQ(desc->getData().post_padding, hipdnn_tests::toVec(K_WGRAD_CONV_PADDING));
    EXPECT_EQ(desc->getData().stride, hipdnn_tests::toVec(K_WGRAD_CONV_STRIDE));
    EXPECT_EQ(desc->getData().dilation, hipdnn_tests::toVec(K_WGRAD_CONV_DILATION));
    EXPECT_EQ(desc->getData().conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionWrwOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_WGRAD_TENSOR_X_UID);
    ASSERT_NE(desc->getDyDesc(), nullptr);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_WGRAD_TENSOR_DY_UID);
    ASSERT_NE(desc->getDwDesc(), nullptr);
    EXPECT_EQ(desc->getDwDesc()->getData().uid, K_WGRAD_TENSOR_DW_UID);
}

TEST_F(TestConvolutionWrwOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_WGRAD_TENSOR_X_UID]);
    EXPECT_EQ(desc->getDyDesc(), _tensorMap[K_WGRAD_TENSOR_DY_UID]);
    EXPECT_EQ(desc->getDwDesc(), _tensorMap[K_WGRAD_TENSOR_DW_UID]);
}

TEST_F(TestConvolutionWrwOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_WGRAD_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, hipdnn_tests::toVec(K_WGRAD_TENSOR_X_DIMS));
    EXPECT_EQ(desc->getXDesc()->getData().strides, hipdnn_tests::toVec(K_WGRAD_TENSOR_X_STRIDES));

    ASSERT_NE(desc->getDyDesc(), nullptr);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_WGRAD_TENSOR_DY_UID);
    EXPECT_EQ(desc->getDyDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDyDesc()->getData().dims, hipdnn_tests::toVec(K_WGRAD_TENSOR_DY_DIMS));
    EXPECT_EQ(desc->getDyDesc()->getData().strides, hipdnn_tests::toVec(K_WGRAD_TENSOR_DY_STRIDES));

    ASSERT_NE(desc->getDwDesc(), nullptr);
    EXPECT_EQ(desc->getDwDesc()->getData().uid, K_WGRAD_TENSOR_DW_UID);
    EXPECT_EQ(desc->getDwDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDwDesc()->getData().dims, hipdnn_tests::toVec(K_WGRAD_TENSOR_DW_DIMS));
    EXPECT_EQ(desc->getDwDesc()->getData().strides, hipdnn_tests::toVec(K_WGRAD_TENSOR_DW_STRIDES));
}

TEST_F(TestConvolutionWrwOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_WGRAD_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestConvolutionWrwOperationFromNode, FailsWithMissingDyTensor)
{
    _tensorMap.erase(K_WGRAD_TENSOR_DY_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestConvolutionWrwOperationFromNode, FailsWithMissingDwTensor)
{
    _tensorMap.erase(K_WGRAD_TENSOR_DW_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestConvolutionWrwOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_WGRAD_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_WGRAD_TENSOR_DY_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_WGRAD_TENSOR_DW_UID);
}

TEST_F(TestConvolutionWrwOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ConvolutionWrwAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsConvolutionWrwAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_WGRAD_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->dy_tensor_uid, K_WGRAD_TENSOR_DY_UID);
    EXPECT_EQ(rebuiltAttrs->dw_tensor_uid, K_WGRAD_TENSOR_DW_UID);
    EXPECT_EQ(rebuiltAttrs->pre_padding, hipdnn_tests::toVec(K_WGRAD_CONV_PADDING));
    EXPECT_EQ(rebuiltAttrs->post_padding, hipdnn_tests::toVec(K_WGRAD_CONV_PADDING));
    EXPECT_EQ(rebuiltAttrs->stride, hipdnn_tests::toVec(K_WGRAD_CONV_STRIDE));
    EXPECT_EQ(rebuiltAttrs->dilation, hipdnn_tests::toVec(K_WGRAD_CONV_DILATION));
    EXPECT_EQ(rebuiltAttrs->conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionWrwOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    // Verify pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &prePaddingCount,
                       prePadding.data());
    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, hipdnn_tests::toVec(K_WGRAD_CONV_PADDING));

    // Verify post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &postPaddingCount,
                       postPadding.data());
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, hipdnn_tests::toVec(K_WGRAD_CONV_PADDING));

    // Verify stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data());
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, hipdnn_tests::toVec(K_WGRAD_CONV_STRIDE));

    // Verify dilation
    std::vector<int64_t> dilation(2);
    int64_t dilationCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, &dilationCount, dilation.data());
    ASSERT_EQ(dilationCount, 2);
    EXPECT_EQ(dilation, hipdnn_tests::toVec(K_WGRAD_CONV_DILATION));

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
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        xScoped.get(),
        K_WGRAD_TENSOR_X_UID,
        HIPDNN_DATA_FLOAT,
        hipdnn_tests::toVec(K_WGRAD_TENSOR_X_DIMS),
        hipdnn_tests::toVec(K_WGRAD_TENSOR_X_STRIDES));

    // Verify dy tensor
    hipdnn_backend::ScopedDescriptor dyScoped;
    int64_t dyCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dyCount,
                       static_cast<void*>(dyScoped.getPtr()));
    ASSERT_EQ(dyCount, 1);
    ASSERT_NE(dyScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        dyScoped.get(),
        K_WGRAD_TENSOR_DY_UID,
        HIPDNN_DATA_FLOAT,
        hipdnn_tests::toVec(K_WGRAD_TENSOR_DY_DIMS),
        hipdnn_tests::toVec(K_WGRAD_TENSOR_DY_STRIDES));

    // Verify dw tensor
    hipdnn_backend::ScopedDescriptor dwScoped;
    int64_t dwCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dwCount,
                       static_cast<void*>(dwScoped.getPtr()));
    ASSERT_EQ(dwCount, 1);
    ASSERT_NE(dwScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        dwScoped.get(),
        K_WGRAD_TENSOR_DW_UID,
        HIPDNN_DATA_FLOAT,
        hipdnn_tests::toVec(K_WGRAD_TENSOR_DW_DIMS),
        hipdnn_tests::toVec(K_WGRAD_TENSOR_DW_STRIDES));

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_WEIGHTS_EXT);
}

TEST_F(TestConvolutionWrwOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_convolutionwrw_1";

    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_convolutionwrw_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_convolutionwrw_1");
}

TEST_F(TestConvolutionWrwOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestConvolutionWrwOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestConvolutionWrwOperationFromNode, ToStringIncludesName)
{
    auto node = createStandardNode();
    node.name = "my_wrw_op";

    auto desc = ConvolutionWrwOperationDescriptor::fromNode(node, _tensorMap);
    auto str = desc->toString();

    EXPECT_NE(str.find("name=my_wrw_op"), std::string::npos);
}
