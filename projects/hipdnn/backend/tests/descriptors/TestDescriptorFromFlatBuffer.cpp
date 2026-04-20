// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/ConvolutionFwdOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// TensorDescriptor::fromFlatBuffer() Tests
// =============================================================================

class TestTensorDescriptorFromFlatBuffer : public ::testing::Test
{
protected:
    // Creates a TensorAttributesT with standard test values
    static TensorAttributesT createStandardTensorAttrs(int64_t uid,
                                                       DataType dataType = DataType::FLOAT)
    {
        TensorAttributesT attrs;
        attrs.uid = uid;
        attrs.data_type = dataType;
        attrs.dims = toVec(K_FPROP_TENSOR_X_DIMS);
        attrs.strides = toVec(K_FPROP_TENSOR_X_STRIDES);
        attrs.virtual_ = false;
        return attrs;
    }
};

TEST_F(TestTensorDescriptorFromFlatBuffer, CreatesValidFinalizedDescriptor)
{
    auto attrs = createStandardTensorAttrs(K_FPROP_TENSOR_X_UID);
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_TENSOR_DESCRIPTOR);
    EXPECT_EQ(desc->getData().uid, K_FPROP_TENSOR_X_UID);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, PopulatesAllFieldsCorrectly)
{
    auto attrs = createStandardTensorAttrs(K_FPROP_TENSOR_X_UID);
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getData().uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getData().dims, toVec(K_FPROP_TENSOR_X_DIMS));
    EXPECT_EQ(desc->getData().strides, toVec(K_FPROP_TENSOR_X_STRIDES));
    EXPECT_FALSE(desc->getData().virtual_);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, MoveOverloadPopulatesAllFields)
{
    auto attrs = createStandardTensorAttrs(K_FPROP_TENSOR_X_UID);
    auto desc = TensorDescriptor::fromFlatBuffer(std::move(attrs));

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getData().uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getData().dims, toVec(K_FPROP_TENSOR_X_DIMS));
    EXPECT_EQ(desc->getData().strides, toVec(K_FPROP_TENSOR_X_STRIDES));
    EXPECT_FALSE(desc->getData().virtual_);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, PreservesVirtualFlag)
{
    auto attrs = createStandardTensorAttrs(K_FPROP_TENSOR_X_UID);
    attrs.virtual_ = true;
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    ASSERT_TRUE(desc->getData().virtual_);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, GetAttributeWorksAfterFromFlatBuffer)
{
    auto attrs = createStandardTensorAttrs(K_FPROP_TENSOR_X_UID);
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    // Verify getAttribute returns the correct UID
    int64_t uid = 0;
    int64_t elementCount = 0;
    desc->getAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &elementCount, &uid);
    ASSERT_EQ(uid, K_FPROP_TENSOR_X_UID);
    ASSERT_EQ(elementCount, 1);

    // Verify getAttribute returns the correct data type
    hipdnnDataType_t dataType = {};
    int64_t dtCount = 0;
    desc->getAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &dataType);
    ASSERT_EQ(dataType, HIPDNN_DATA_FLOAT);
    ASSERT_EQ(dtCount, 1);

    // Verify getAttribute returns the correct dims
    std::vector<int64_t> dims(4);
    int64_t dimCount = 0;
    desc->getAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, &dimCount, dims.data());
    ASSERT_EQ(dimCount, 4);
    ASSERT_EQ(dims, toVec(K_FPROP_TENSOR_X_DIMS));
}

TEST_F(TestTensorDescriptorFromFlatBuffer, FailsWithMissingDims)
{
    TensorAttributesT attrs;
    attrs.uid = 1;
    attrs.data_type = DataType::FLOAT;
    // dims left empty
    attrs.strides = {1};

    ASSERT_THROW_HIPDNN_STATUS(TensorDescriptor::fromFlatBuffer(attrs), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, FailsWithMissingStrides)
{
    TensorAttributesT attrs;
    attrs.uid = 1;
    attrs.data_type = DataType::FLOAT;
    attrs.dims = {1};
    // strides left empty

    ASSERT_THROW_HIPDNN_STATUS(TensorDescriptor::fromFlatBuffer(attrs), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, FailsWithUnsetDataType)
{
    TensorAttributesT attrs;
    attrs.uid = 1;
    // data_type defaults to UNSET
    attrs.dims = {1};
    attrs.strides = {1};

    ASSERT_THROW_HIPDNN_STATUS(TensorDescriptor::fromFlatBuffer(attrs), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, PreservesName)
{
    auto attrs = createStandardTensorAttrs(K_FPROP_TENSOR_X_UID);
    attrs.name = "test_tensor";
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    ASSERT_EQ(desc->getData().name, "test_tensor");
}

TEST_F(TestTensorDescriptorFromFlatBuffer, RoundTripWithMultipleDataTypes)
{
    for(auto dt : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE})
    {
        auto attrs = createStandardTensorAttrs(K_FPROP_TENSOR_X_UID, dt);
        auto desc = TensorDescriptor::fromFlatBuffer(attrs);

        ASSERT_EQ(desc->getData().data_type, dt) << "Failed for DataType: " << EnumNameDataType(dt);
        ASSERT_TRUE(desc->isFinalized()) << "Not finalized for DataType: " << EnumNameDataType(dt);
    }
}

// =============================================================================
// ConvolutionFwdOperationDescriptor::fromNode() Tests
// =============================================================================

class TestConvolutionFwdOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        // Build tensor descriptors using fromFlatBuffer
        TensorAttributesT xAttrs;
        xAttrs.uid = K_FPROP_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_FPROP_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_FPROP_TENSOR_X_STRIDES);

        TensorAttributesT wAttrs;
        wAttrs.uid = K_FPROP_TENSOR_W_UID;
        wAttrs.data_type = DataType::FLOAT;
        wAttrs.dims = toVec(K_FPROP_TENSOR_W_DIMS);
        wAttrs.strides = toVec(K_FPROP_TENSOR_W_STRIDES);

        TensorAttributesT yAttrs;
        yAttrs.uid = K_FPROP_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_FPROP_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_FPROP_TENSOR_Y_STRIDES);

        _tensorMap[K_FPROP_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        _tensorMap[K_FPROP_TENSOR_W_UID] = TensorDescriptor::fromFlatBuffer(wAttrs);
        _tensorMap[K_FPROP_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    // Creates a standard ConvolutionFwdAttributesT
    static ConvolutionFwdAttributesT createStandardConvAttrs()
    {
        ConvolutionFwdAttributesT attrs;
        attrs.x_tensor_uid = K_FPROP_TENSOR_X_UID;
        attrs.w_tensor_uid = K_FPROP_TENSOR_W_UID;
        attrs.y_tensor_uid = K_FPROP_TENSOR_Y_UID;
        attrs.pre_padding = toVec(K_FPROP_CONV_PADDING);
        attrs.post_padding = toVec(K_FPROP_CONV_PADDING);
        attrs.stride = toVec(K_FPROP_CONV_STRIDE);
        attrs.dilation = toVec(K_FPROP_CONV_DILATION);
        attrs.conv_mode = ConvMode::CROSS_CORRELATION;
        return attrs;
    }

    // Creates a standard NodeT wrapping ConvolutionFwdAttributes
    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardConvAttrs());
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
    EXPECT_EQ(desc->getData().pre_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(desc->getData().post_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(desc->getData().stride, toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(desc->getData().dilation, toVec(K_FPROP_CONV_DILATION));
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
    auto convAttrs = createStandardConvAttrs();
    convAttrs.conv_mode = ConvMode::CONVOLUTION;
    node.attributes.Set(convAttrs);
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().conv_mode, ConvMode::CONVOLUTION);
}

TEST_F(TestConvolutionFwdOperationFromNode, PreservesPaddingStridesDilation)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getData().pre_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(desc->getData().post_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(desc->getData().stride, toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(desc->getData().dilation, toVec(K_FPROP_CONV_DILATION));
}

TEST_F(TestConvolutionFwdOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getWDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(desc->getWDesc()->getData().uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_FPROP_TENSOR_Y_UID);
}

TEST_F(TestConvolutionFwdOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify the shared_ptrs point to the same objects in the tensor map
    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_FPROP_TENSOR_X_UID]);
    EXPECT_EQ(desc->getWDesc(), _tensorMap[K_FPROP_TENSOR_W_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_FPROP_TENSOR_Y_UID]);
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

    // Build a new node from the descriptor and verify it matches the original
    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ConvolutionFwdAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->w_tensor_uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_FPROP_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->pre_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(rebuiltAttrs->post_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(rebuiltAttrs->stride, toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(rebuiltAttrs->dilation, toVec(K_FPROP_CONV_DILATION));
    EXPECT_EQ(rebuiltAttrs->conv_mode, ConvMode::CROSS_CORRELATION);
}

TEST_F(TestConvolutionFwdOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify getAttribute returns correct pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t elementCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &elementCount,
                       prePadding.data());
    ASSERT_EQ(elementCount, 2);
    EXPECT_EQ(prePadding, toVec(K_FPROP_CONV_PADDING));

    // Verify getAttribute returns correct compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify getAttribute returns correct conv mode
    hipdnnConvolutionMode_t convMode = {};
    int64_t modeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &modeCount, &convMode);
    ASSERT_EQ(convMode, HIPDNN_CROSS_CORRELATION);

    // Verify getAttribute returns correct post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPadCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &postPadCount,
                       postPadding.data());
    ASSERT_EQ(postPadCount, 2);
    EXPECT_EQ(postPadding, toVec(K_FPROP_CONV_PADDING));

    // Verify getAttribute returns correct filter strides
    std::vector<int64_t> filterStrides(2);
    int64_t filterStrideCount = 0;
    desc->getAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                       HIPDNN_TYPE_INT64,
                       2,
                       &filterStrideCount,
                       filterStrides.data());
    ASSERT_EQ(filterStrideCount, 2);
    EXPECT_EQ(filterStrides, toVec(K_FPROP_CONV_STRIDE));

    // Verify getAttribute returns correct dilations
    std::vector<int64_t> dilations(2);
    int64_t dilationCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_CONVOLUTION_DILATIONS, HIPDNN_TYPE_INT64, 2, &dilationCount, dilations.data());
    ASSERT_EQ(dilationCount, 2);
    EXPECT_EQ(dilations, toVec(K_FPROP_CONV_DILATION));

    // Verify getAttribute returns correct X tensor descriptor
    HipdnnBackendDescriptor* xTensorDesc = nullptr;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(&xTensorDesc));
    const std::unique_ptr<HipdnnBackendDescriptor> ownedXDesc(xTensorDesc);
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xTensorDesc, nullptr);
    int64_t xUid = 0;
    int64_t xUidCount = 0;
    xTensorDesc->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &xUidCount, &xUid);
    EXPECT_EQ(xUid, K_FPROP_TENSOR_X_UID);

    // Verify getAttribute returns correct W tensor descriptor
    HipdnnBackendDescriptor* wTensorDesc = nullptr;
    int64_t wCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &wCount,
                       static_cast<void*>(&wTensorDesc));
    const std::unique_ptr<HipdnnBackendDescriptor> ownedWDesc(wTensorDesc);
    ASSERT_EQ(wCount, 1);
    ASSERT_NE(wTensorDesc, nullptr);
    int64_t wUid = 0;
    int64_t wUidCount = 0;
    wTensorDesc->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &wUidCount, &wUid);
    EXPECT_EQ(wUid, K_FPROP_TENSOR_W_UID);

    // Verify getAttribute returns correct Y tensor descriptor
    HipdnnBackendDescriptor* yTensorDesc = nullptr;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(&yTensorDesc));
    const std::unique_ptr<HipdnnBackendDescriptor> ownedYDesc(yTensorDesc);
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yTensorDesc, nullptr);
    int64_t yUid = 0;
    int64_t yUidCount = 0;
    yTensorDesc->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &yUidCount, &yUid);
    EXPECT_EQ(yUid, K_FPROP_TENSOR_Y_UID);

    // Verify getAttribute returns correct operation type
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
    node.name = "conv_fprop_1";

    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    // Query the name size
    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("conv_fprop_1").size() + 1));

    // Retrieve the name
    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "conv_fprop_1");
}

TEST_F(TestConvolutionFwdOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    // node.name is empty by default

    auto desc = ConvolutionFwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1); // just the null terminator
}

TEST_F(TestConvolutionFwdOperationFromNode, NameSetViaAttributeRoundTrips)
{
    auto desc = std::make_shared<ConvolutionFwdOperationDescriptor>();

    // Wrap tensor descriptors into HipdnnBackendDescriptor for the C API boundary
    const std::unique_ptr<HipdnnBackendDescriptor> xPacked(
        HipdnnBackendDescriptor::packDescriptor(_tensorMap[K_FPROP_TENSOR_X_UID]));
    const std::unique_ptr<HipdnnBackendDescriptor> wPacked(
        HipdnnBackendDescriptor::packDescriptor(_tensorMap[K_FPROP_TENSOR_W_UID]));
    const std::unique_ptr<HipdnnBackendDescriptor> yPacked(
        HipdnnBackendDescriptor::packDescriptor(_tensorMap[K_FPROP_TENSOR_Y_UID]));

    // setAttribute takes void* pointing to a HipdnnBackendDescriptor*, so use raw pointers
    auto* xRaw = xPacked.get();
    auto* wRaw = wPacked.get();
    auto* yRaw = yPacked.get();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xRaw));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&wRaw));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yRaw));

    // Set convolution parameters
    std::vector<int64_t> padding = toVec(K_FPROP_CONV_PADDING);
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(padding.size()),
                       padding.data());
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(padding.size()),
                       padding.data());
    std::vector<int64_t> strideVals = toVec(K_FPROP_CONV_STRIDE);
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(strideVals.size()),
                       strideVals.data());
    std::vector<int64_t> dilationVals = toVec(K_FPROP_CONV_DILATION);
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(dilationVals.size()),
                       dilationVals.data());
    auto convMode = static_cast<int64_t>(HIPDNN_CROSS_CORRELATION);
    desc->setAttribute(
        HIPDNN_ATTR_CONVOLUTION_CONV_MODE, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &convMode);
    auto compType = static_cast<int64_t>(HIPDNN_DATA_FLOAT);
    desc->setAttribute(HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compType);

    // Set the name
    const std::string name = "my_conv_op";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(name.size()),
                       name.c_str());

    desc->finalize();

    // Get name size
    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(name.size() + 1));

    // Get name value
    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "my_conv_op");
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
