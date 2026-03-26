// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/PoolingFwdOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/pooling_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/PoolingFwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// PoolingFwdOperationDescriptor::fromNode() Tests
// =============================================================================

class TestPoolingFwdOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_TENSOR_X_STRIDES);

        _tensorMap[K_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = K_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_TENSOR_Y_STRIDES);

        _tensorMap[K_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    static hipdnn_data_sdk::data_objects::PoolingFwdAttributesT createStandardPoolingFwdAttrs()
    {
        hipdnn_data_sdk::data_objects::PoolingFwdAttributesT attrs;
        attrs.x_tensor_uid = K_TENSOR_X_UID;
        attrs.y_tensor_uid = K_TENSOR_Y_UID;
        attrs.pre_padding = toVec(K_PRE_PADDING);
        attrs.post_padding = toVec(K_POST_PADDING);
        attrs.stride = toVec(K_STRIDE);
        attrs.window_size = toVec(K_WINDOW_SIZE);
        attrs.pooling_mode = PoolingMode::MAX;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardPoolingFwdAttrs());
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
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, dimCount, &actualDimCount, dims.data());
        EXPECT_EQ(dims, expectedDims);

        int64_t strideCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 0, &strideCount, nullptr);
        ASSERT_EQ(strideCount, static_cast<int64_t>(expectedStrides.size()));
        std::vector<int64_t> strides(static_cast<size_t>(strideCount));
        int64_t actualStrideCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, strideCount, &actualStrideCount, strides.data());
        EXPECT_EQ(strides, expectedStrides);
    }
};

TEST_F(TestPoolingFwdOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_POOLING_FORWARD_DESCRIPTOR);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_TENSOR_X_UID);
}

TEST_F(TestPoolingFwdOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::PoolingFwdAttributes);
    auto desc = std::static_pointer_cast<PoolingFwdOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_TENSOR_Y_UID);
    EXPECT_EQ(desc->getData().pre_padding, toVec(K_PRE_PADDING));
    EXPECT_EQ(desc->getData().post_padding, toVec(K_POST_PADDING));
    EXPECT_EQ(desc->getData().stride, toVec(K_STRIDE));
    EXPECT_EQ(desc->getData().window_size, toVec(K_WINDOW_SIZE));
    EXPECT_EQ(desc->getData().pooling_mode, PoolingMode::MAX);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_TENSOR_X_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_TENSOR_Y_UID);
}

TEST_F(TestPoolingFwdOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestPoolingFwdOperationFromNode, PreservesPoolingMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardPoolingFwdAttrs();
    attrs.pooling_mode = PoolingMode::AVERAGE;
    node.attributes.Set(attrs);
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().pooling_mode, PoolingMode::AVERAGE);
}

TEST_F(TestPoolingFwdOperationFromNode, PreservesDataFields)
{
    auto node = createStandardNode();
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getData().pre_padding, toVec(K_PRE_PADDING));
    EXPECT_EQ(desc->getData().post_padding, toVec(K_POST_PADDING));
    EXPECT_EQ(desc->getData().stride, toVec(K_STRIDE));
    EXPECT_EQ(desc->getData().window_size, toVec(K_WINDOW_SIZE));
    EXPECT_EQ(desc->getData().pooling_mode, PoolingMode::MAX);
}

TEST_F(TestPoolingFwdOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_TENSOR_X_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_TENSOR_Y_UID);
}

TEST_F(TestPoolingFwdOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_TENSOR_X_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_TENSOR_Y_UID]);
}

TEST_F(TestPoolingFwdOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{1, 3, 32, 32}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{3072, 1024, 32, 1}));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, (std::vector<int64_t>{1, 3, 16, 16}));
    EXPECT_EQ(desc->getYDesc()->getData().strides, (std::vector<int64_t>{768, 256, 16, 1}));

}

TEST_F(TestPoolingFwdOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(PoolingFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestPoolingFwdOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_TENSOR_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(PoolingFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestPoolingFwdOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2);
    EXPECT_EQ(tensors[0]->getData().uid, K_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_TENSOR_Y_UID);
}

TEST_F(TestPoolingFwdOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::PoolingFwdAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsPoolingFwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->pre_padding, toVec(K_PRE_PADDING));
    EXPECT_EQ(rebuiltAttrs->post_padding, toVec(K_POST_PADDING));
    EXPECT_EQ(rebuiltAttrs->stride, toVec(K_STRIDE));
    EXPECT_EQ(rebuiltAttrs->window_size, toVec(K_WINDOW_SIZE));
    EXPECT_EQ(rebuiltAttrs->pooling_mode, PoolingMode::MAX);
}

TEST_F(TestPoolingFwdOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_POOLING_PRE_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &prePaddingCount,
                       prePadding.data());
    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, toVec(K_PRE_PADDING));

    // Verify post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_POOLING_POST_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &postPaddingCount,
                       postPadding.data());
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, toVec(K_POST_PADDING));

    // Verify stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    desc->getAttribute(HIPDNN_ATTR_POOLING_STRIDES,
                       HIPDNN_TYPE_INT64,
                       2,
                       &strideCount,
                       stride.data());
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, toVec(K_STRIDE));

    // Verify window_size
    std::vector<int64_t> windowSize(2);
    int64_t windowSizeCount = 0;
    desc->getAttribute(HIPDNN_ATTR_POOLING_WINDOW_SIZE,
                       HIPDNN_TYPE_INT64,
                       2,
                       &windowSizeCount,
                       windowSize.data());
    ASSERT_EQ(windowSizeCount, 2);
    EXPECT_EQ(windowSize, toVec(K_WINDOW_SIZE));

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_POOLING_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify pooling_mode
    hipdnnPoolingMode_t poolingMode = {};
    int64_t poolingModeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, 1, &poolingModeCount, &poolingMode);
    ASSERT_EQ(poolingMode, HIPDNN_POOLING_MODE_MAX);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(xScoped.get(), K_TENSOR_X_UID, HIPDNN_DATA_FLOAT,
                           {1, 3, 32, 32},
                           {3072, 1024, 32, 1});

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    verifyTensorDescriptor(yScoped.get(), K_TENSOR_Y_UID, HIPDNN_DATA_FLOAT,
                           {1, 3, 16, 16},
                           {768, 256, 16, 1});

    // Verify operation type
    hipdnnOperationType_t opType = HIPDNN_OPERATION_TYPE_NOT_SET;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_POOLING_FORWARD);
}

TEST_F(TestPoolingFwdOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_poolingfwd_1";

    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_poolingfwd_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_poolingfwd_1");
}

TEST_F(TestPoolingFwdOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestPoolingFwdOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = PoolingFwdOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
