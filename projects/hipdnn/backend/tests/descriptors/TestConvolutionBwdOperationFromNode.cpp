// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionBwdOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_data_sdk::data_objects;

// =============================================================================
// ConvolutionBwdOperationDescriptor::fromNode() Tests
// =============================================================================

class TestConvolutionBwdOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT dyAttrs;
        dyAttrs.uid = 10;
        dyAttrs.data_type = DataType::FLOAT;
        dyAttrs.dims = {1, 64, 32, 32};
        dyAttrs.strides = {65536, 1024, 32, 1};

        _tensorMap[10] = TensorDescriptor::fromFlatBuffer(dyAttrs);
        TensorAttributesT wAttrs;
        wAttrs.uid = 11;
        wAttrs.data_type = DataType::FLOAT;
        wAttrs.dims = {64, 3, 3, 3};
        wAttrs.strides = {27, 9, 3, 1};

        _tensorMap[11] = TensorDescriptor::fromFlatBuffer(wAttrs);
        TensorAttributesT dxAttrs;
        dxAttrs.uid = 12;
        dxAttrs.data_type = DataType::FLOAT;
        dxAttrs.dims = {1, 3, 32, 32};
        dxAttrs.strides = {3072, 1024, 32, 1};

        _tensorMap[12] = TensorDescriptor::fromFlatBuffer(dxAttrs);
    }

    static hipdnn_data_sdk::data_objects::ConvolutionBwdAttributesT
        createStandardConvolutionBwdAttrs()
    {
        hipdnn_data_sdk::data_objects::ConvolutionBwdAttributesT attrs;
        attrs.dy_tensor_uid = 10;
        attrs.w_tensor_uid = 11;
        attrs.dx_tensor_uid = 12;
        attrs.pre_padding = {1, 1};
        attrs.post_padding = {1, 1};
        attrs.stride = {1, 1};
        attrs.dilation = {1, 1};
        attrs.conv_mode = ConvMode::CROSS_CORRELATION;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardConvolutionBwdAttrs());
        return node;
    }

    // Verifies that a packed tensor descriptor (retrieved via getAttribute) has the
    // expected UID, data_type, dimensions, and strides.
    static void verifyTensorDescriptor(hipdnnBackendDescriptor_t tensorDesc,
                                       int64_t expectedUid,
                                       hipdnnDataType_t expectedDataType,
                                       const std::vector<int64_t>& expectedDims,
                                       const std::vector<int64_t>& expectedStrides)
    {
        int64_t uid = 0;
        int64_t uidCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uidCount, &uid);
        EXPECT_EQ(uid, expectedUid);

        hipdnnDataType_t dataType = {};
        int64_t dtCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &dataType);
        EXPECT_EQ(dataType, expectedDataType);

        int64_t dimCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 0, &dimCount, nullptr);
        ASSERT_EQ(dimCount, static_cast<int64_t>(expectedDims.size()));
        std::vector<int64_t> dims(static_cast<size_t>(dimCount));
        int64_t actualDimCount = 0;
        tensorDesc->getAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS,
                                 HIPDNN_TYPE_INT64,
                                 dimCount,
                                 &actualDimCount,
                                 dims.data());
        EXPECT_EQ(dims, expectedDims);

        int64_t strideCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 0, &strideCount, nullptr);
        ASSERT_EQ(strideCount, static_cast<int64_t>(expectedStrides.size()));
        std::vector<int64_t> strides(static_cast<size_t>(strideCount));
        int64_t actualStrideCount = 0;
        tensorDesc->getAttribute(HIPDNN_ATTR_TENSOR_STRIDES,
                                 HIPDNN_TYPE_INT64,
                                 strideCount,
                                 &actualStrideCount,
                                 strides.data());
        EXPECT_EQ(strides, expectedStrides);
    }
};

TEST_F(TestConvolutionBwdOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DESCRIPTOR);
    EXPECT_EQ(desc->getData().dy_tensor_uid, 10);
}

TEST_F(TestConvolutionBwdOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ConvolutionBwdAttributes);
    auto desc = std::static_pointer_cast<ConvolutionBwdOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().dy_tensor_uid, 10);
    EXPECT_EQ(desc->getData().w_tensor_uid, 11);
    EXPECT_EQ(desc->getData().dx_tensor_uid, 12);
    EXPECT_EQ(desc->getData().pre_padding, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(desc->getData().post_padding, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(desc->getData().stride, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(desc->getData().dilation, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(desc->getData().conv_mode, ConvMode::CROSS_CORRELATION);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, 10);
    EXPECT_EQ(desc->getWDesc()->getData().uid, 11);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, 12);
}

TEST_F(TestConvolutionBwdOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestConvolutionBwdOperationFromNode, PreservesConvMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardConvolutionBwdAttrs();
    attrs.conv_mode = ConvMode::CONVOLUTION;
    node.attributes.Set(attrs);
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().conv_mode, ConvMode::CONVOLUTION);
}

TEST_F(TestConvolutionBwdOperationFromNode, PreservesDataFields)
{
    auto node = createStandardNode();
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getData().pre_padding, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(desc->getData().post_padding, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(desc->getData().stride, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(desc->getData().dilation, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(desc->getData().conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionBwdOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getDyDesc(), nullptr);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, 10);
    ASSERT_NE(desc->getWDesc(), nullptr);
    EXPECT_EQ(desc->getWDesc()->getData().uid, 11);
    ASSERT_NE(desc->getDxDesc(), nullptr);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, 12);
}

TEST_F(TestConvolutionBwdOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getDyDesc(), _tensorMap[10]);
    EXPECT_EQ(desc->getWDesc(), _tensorMap[11]);
    EXPECT_EQ(desc->getDxDesc(), _tensorMap[12]);
}

TEST_F(TestConvolutionBwdOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getDyDesc(), nullptr);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, 10);
    EXPECT_EQ(desc->getDyDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDyDesc()->getData().dims, (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_EQ(desc->getDyDesc()->getData().strides, (std::vector<int64_t>{65536, 1024, 32, 1}));

    ASSERT_NE(desc->getWDesc(), nullptr);
    EXPECT_EQ(desc->getWDesc()->getData().uid, 11);
    EXPECT_EQ(desc->getWDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getWDesc()->getData().dims, (std::vector<int64_t>{64, 3, 3, 3}));
    EXPECT_EQ(desc->getWDesc()->getData().strides, (std::vector<int64_t>{27, 9, 3, 1}));

    ASSERT_NE(desc->getDxDesc(), nullptr);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, 12);
    EXPECT_EQ(desc->getDxDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDxDesc()->getData().dims, (std::vector<int64_t>{1, 3, 32, 32}));
    EXPECT_EQ(desc->getDxDesc()->getData().strides, (std::vector<int64_t>{3072, 1024, 32, 1}));
}

TEST_F(TestConvolutionBwdOperationFromNode, FailsWithMissingDyTensor)
{
    _tensorMap.erase(10);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestConvolutionBwdOperationFromNode, FailsWithMissingWTensor)
{
    _tensorMap.erase(11);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestConvolutionBwdOperationFromNode, FailsWithMissingDxTensor)
{
    _tensorMap.erase(12);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestConvolutionBwdOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, 10);
    EXPECT_EQ(tensors[1]->getData().uid, 11);
    EXPECT_EQ(tensors[2]->getData().uid, 12);
}

TEST_F(TestConvolutionBwdOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ConvolutionBwdAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsConvolutionBwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->dy_tensor_uid, 10);
    EXPECT_EQ(rebuiltAttrs->w_tensor_uid, 11);
    EXPECT_EQ(rebuiltAttrs->dx_tensor_uid, 12);
    EXPECT_EQ(rebuiltAttrs->pre_padding, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(rebuiltAttrs->post_padding, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(rebuiltAttrs->stride, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(rebuiltAttrs->dilation, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(rebuiltAttrs->conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionBwdOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &prePaddingCount,
                       prePadding.data());
    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, (std::vector<int64_t>{1, 1}));

    // Verify post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &postPaddingCount,
                       postPadding.data());
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, (std::vector<int64_t>{1, 1}));

    // Verify stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data());
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, (std::vector<int64_t>{1, 1}));

    // Verify dilation
    std::vector<int64_t> dilation(2);
    int64_t dilationCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, &dilationCount, dilation.data());
    ASSERT_EQ(dilationCount, 2);
    EXPECT_EQ(dilation, (std::vector<int64_t>{1, 1}));

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
    ASSERT_EQ(convMode, HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION);

    // Verify dy tensor
    hipdnn_backend::ScopedDescriptor dyScoped;
    int64_t dyCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DY,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dyCount,
                       static_cast<void*>(dyScoped.getPtr()));
    ASSERT_EQ(dyCount, 1);
    ASSERT_NE(dyScoped.get(), nullptr);
    verifyTensorDescriptor(
        dyScoped.get(), 10, HIPDNN_DATA_FLOAT, {1, 64, 32, 32}, {65536, 1024, 32, 1});

    // Verify w tensor
    hipdnn_backend::ScopedDescriptor wScoped;
    int64_t wCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_W,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &wCount,
                       static_cast<void*>(wScoped.getPtr()));
    ASSERT_EQ(wCount, 1);
    ASSERT_NE(wScoped.get(), nullptr);
    verifyTensorDescriptor(wScoped.get(), 11, HIPDNN_DATA_FLOAT, {64, 3, 3, 3}, {27, 9, 3, 1});

    // Verify dx tensor
    hipdnn_backend::ScopedDescriptor dxScoped;
    int64_t dxCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DX,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dxCount,
                       static_cast<void*>(dxScoped.getPtr()));
    ASSERT_EQ(dxCount, 1);
    ASSERT_NE(dxScoped.get(), nullptr);
    verifyTensorDescriptor(
        dxScoped.get(), 12, HIPDNN_DATA_FLOAT, {1, 3, 32, 32}, {3072, 1024, 32, 1});

    // Verify operation type
    hipdnnOperationType_t opType = HIPDNN_OPERATION_TYPE_NOT_SET;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_DATA);
}

TEST_F(TestConvolutionBwdOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_convolutionbwd_1";

    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_convolutionbwd_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_convolutionbwd_1");
}

TEST_F(TestConvolutionBwdOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestConvolutionBwdOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = ConvolutionBwdOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
