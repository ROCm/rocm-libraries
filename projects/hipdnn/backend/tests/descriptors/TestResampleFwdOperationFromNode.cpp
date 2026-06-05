// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ResampleFwdOperationDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ResampleFwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// ResampleFwdOperationDescriptor::fromNode() Tests
// =============================================================================

class TestResampleFwdOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_RESAMPLE_FWD_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES);

        _tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = K_RESAMPLE_FWD_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_RESAMPLE_FWD_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_RESAMPLE_FWD_TENSOR_Y_STRIDES);

        _tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
        TensorAttributesT indexAttrs;
        indexAttrs.uid = K_RESAMPLE_FWD_TENSOR_INDEX_UID;
        indexAttrs.data_type = DataType::FLOAT;
        indexAttrs.dims = toVec(K_RESAMPLE_FWD_TENSOR_INDEX_DIMS);
        indexAttrs.strides = toVec(K_RESAMPLE_FWD_TENSOR_INDEX_STRIDES);

        _tensorMap[K_RESAMPLE_FWD_TENSOR_INDEX_UID] = TensorDescriptor::fromFlatBuffer(indexAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributesT
        createStandardResampleFwdAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributesT attrs;
        attrs.x_tensor_uid = K_RESAMPLE_FWD_TENSOR_X_UID;
        attrs.y_tensor_uid = K_RESAMPLE_FWD_TENSOR_Y_UID;
        attrs.index_tensor_uid = K_RESAMPLE_FWD_TENSOR_INDEX_UID;
        attrs.pre_padding = toVec(K_RESAMPLE_FWD_PRE_PADDING);
        attrs.post_padding = toVec(K_RESAMPLE_FWD_POST_PADDING);
        attrs.stride = toVec(K_RESAMPLE_FWD_STRIDE);
        attrs.window = toVec(K_RESAMPLE_FWD_WINDOW);
        attrs.resample_mode = ResampleMode::MAXPOOL;
        attrs.padding_mode = PaddingMode::ZERO_PAD;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardResampleFwdAttrs());
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

TEST_F(TestResampleFwdOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_RESAMPLE_FWD_TENSOR_X_UID);
}

TEST_F(TestResampleFwdOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ResampleFwdAttributes);
    auto desc = std::static_pointer_cast<ResampleFwdOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    EXPECT_EQ(desc->getData().index_tensor_uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
    EXPECT_EQ(desc->getData().pre_padding, toVec(K_RESAMPLE_FWD_PRE_PADDING));
    EXPECT_EQ(desc->getData().post_padding, toVec(K_RESAMPLE_FWD_POST_PADDING));
    EXPECT_EQ(desc->getData().stride, toVec(K_RESAMPLE_FWD_STRIDE));
    EXPECT_EQ(desc->getData().window, toVec(K_RESAMPLE_FWD_WINDOW));
    EXPECT_EQ(desc->getData().resample_mode, ResampleMode::MAXPOOL);
    EXPECT_EQ(desc->getData().padding_mode, PaddingMode::ZERO_PAD);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    EXPECT_EQ(desc->getIndexDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
}

TEST_F(TestResampleFwdOperationFromNode, PreservesResampleMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardResampleFwdAttrs();
    attrs.resample_mode = ResampleMode::AVGPOOL_EXCLUDE_PADDING;
    node.attributes.Set(attrs);
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().resample_mode, ResampleMode::AVGPOOL_EXCLUDE_PADDING);
}

TEST_F(TestResampleFwdOperationFromNode, PreservesPaddingMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardResampleFwdAttrs();
    attrs.padding_mode = PaddingMode::NEG_INF_PAD;
    node.attributes.Set(attrs);
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().padding_mode, PaddingMode::NEG_INF_PAD);
}

TEST_F(TestResampleFwdOperationFromNode, PreservesDataFields)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getData().pre_padding, toVec(K_RESAMPLE_FWD_PRE_PADDING));
    EXPECT_EQ(desc->getData().post_padding, toVec(K_RESAMPLE_FWD_POST_PADDING));
    EXPECT_EQ(desc->getData().stride, toVec(K_RESAMPLE_FWD_STRIDE));
    EXPECT_EQ(desc->getData().window, toVec(K_RESAMPLE_FWD_WINDOW));
    EXPECT_EQ(desc->getData().resample_mode, ResampleMode::MAXPOOL);
    EXPECT_EQ(desc->getData().padding_mode, PaddingMode::ZERO_PAD);
}

TEST_F(TestResampleFwdOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    ASSERT_NE(desc->getIndexDesc(), nullptr);
    EXPECT_EQ(desc->getIndexDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
}

TEST_F(TestResampleFwdOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]);
    EXPECT_EQ(desc->getIndexDesc(), _tensorMap[K_RESAMPLE_FWD_TENSOR_INDEX_UID]);
}

TEST_F(TestResampleFwdOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{1, 3, 32, 32}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{3072, 1024, 32, 1}));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, (std::vector<int64_t>{1, 3, 16, 16}));
    EXPECT_EQ(desc->getYDesc()->getData().strides, (std::vector<int64_t>{768, 256, 16, 1}));

    ASSERT_NE(desc->getIndexDesc(), nullptr);
    EXPECT_EQ(desc->getIndexDesc()->getData().uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
    EXPECT_EQ(desc->getIndexDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getIndexDesc()->getData().dims, (std::vector<int64_t>{1, 3, 16, 16}));
    EXPECT_EQ(desc->getIndexDesc()->getData().strides, (std::vector<int64_t>{768, 256, 16, 1}));
}

TEST_F(TestResampleFwdOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_RESAMPLE_FWD_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ResampleFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestResampleFwdOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_RESAMPLE_FWD_TENSOR_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ResampleFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestResampleFwdOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardResampleFwdAttrs();
    attrs.index_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getXDesc(), nullptr);
    EXPECT_NE(desc->getYDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getIndexDesc(), nullptr);
}

TEST_F(TestResampleFwdOperationFromNode, FailsWhenOptionalIndexUidSetButTensorMissing)
{
    _tensorMap.erase(K_RESAMPLE_FWD_TENSOR_INDEX_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ResampleFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestResampleFwdOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
}

TEST_F(TestResampleFwdOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ResampleFwdAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsResampleFwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_RESAMPLE_FWD_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->index_tensor_uid, K_RESAMPLE_FWD_TENSOR_INDEX_UID);
    EXPECT_EQ(rebuiltAttrs->pre_padding, toVec(K_RESAMPLE_FWD_PRE_PADDING));
    EXPECT_EQ(rebuiltAttrs->post_padding, toVec(K_RESAMPLE_FWD_POST_PADDING));
    EXPECT_EQ(rebuiltAttrs->stride, toVec(K_RESAMPLE_FWD_STRIDE));
    EXPECT_EQ(rebuiltAttrs->window, toVec(K_RESAMPLE_FWD_WINDOW));
    EXPECT_EQ(rebuiltAttrs->resample_mode, ResampleMode::MAXPOOL);
    EXPECT_EQ(rebuiltAttrs->padding_mode, PaddingMode::ZERO_PAD);
}

TEST_F(TestResampleFwdOperationFromNode, FromNodePreservesGenerateIndex)
{
    auto attrs = createStandardResampleFwdAttrs();
    attrs.generate_index = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().generate_index.has_value());
    EXPECT_EQ(desc->getData().generate_index.value(), true);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsResampleFwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->generate_index.has_value());
    EXPECT_EQ(rebuiltAttrs->generate_index.value(), true);
}

TEST_F(TestResampleFwdOperationFromNode, BuildNodeOmitsUnsetOptionalScalars)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsResampleFwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    EXPECT_FALSE(rebuiltAttrs->generate_index.has_value());
}

TEST_F(TestResampleFwdOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &prePaddingCount,
                       prePadding.data());
    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, toVec(K_RESAMPLE_FWD_PRE_PADDING));

    // Verify post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_RESAMPLE_POST_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &postPaddingCount,
                       postPadding.data());
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, toVec(K_RESAMPLE_FWD_POST_PADDING));

    // Verify stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data());
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, toVec(K_RESAMPLE_FWD_STRIDE));

    // Verify window
    std::vector<int64_t> window(2);
    int64_t windowCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, &windowCount, window.data());
    ASSERT_EQ(windowCount, 2);
    EXPECT_EQ(window, toVec(K_RESAMPLE_FWD_WINDOW));

    // Verify resample_mode
    hipdnnResampleMode_t resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
    int64_t resampleModeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleModeCount, &resampleMode);
    ASSERT_EQ(resampleMode, HIPDNN_RESAMPLE_MAXPOOL);

    // Verify padding_mode
    hipdnnPaddingMode_t paddingMode = HIPDNN_PADDING_ZERO_PAD;
    int64_t paddingModeCount = 0;
    desc->getAttribute(HIPDNN_ATTR_RESAMPLE_PADDING_MODE,
                       HIPDNN_TYPE_PADDING_MODE,
                       1,
                       &paddingModeCount,
                       &paddingMode);
    ASSERT_EQ(paddingMode, HIPDNN_PADDING_ZERO_PAD);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(xScoped.get(),
                           K_RESAMPLE_FWD_TENSOR_X_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 3, 32, 32},
                           {3072, 1024, 32, 1});

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    verifyTensorDescriptor(yScoped.get(),
                           K_RESAMPLE_FWD_TENSOR_Y_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 3, 16, 16},
                           {768, 256, 16, 1});

    // Verify index tensor (optional)
    hipdnn_backend::ScopedDescriptor indexScoped;
    int64_t indexCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &indexCount,
                       static_cast<void*>(indexScoped.getPtr()));
    ASSERT_EQ(indexCount, 1);
    ASSERT_NE(indexScoped.get(), nullptr);
    verifyTensorDescriptor(indexScoped.get(),
                           K_RESAMPLE_FWD_TENSOR_INDEX_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 3, 16, 16},
                           {768, 256, 16, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_RESAMPLE_FWD);

    // Verify compute data type
    hipdnnDataType_t compType = HIPDNN_DATA_HALF;
    int64_t compTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &compTypeCount, &compType);
    ASSERT_EQ(compTypeCount, 1);
    EXPECT_EQ(compType, HIPDNN_DATA_FLOAT);
}

TEST_F(TestResampleFwdOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_resamplefwd_1";

    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_resamplefwd_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_resamplefwd_1");
}

TEST_F(TestResampleFwdOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestResampleFwdOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestResampleFwdOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = ResampleFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getComputeDataType(), DataType::HALF);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->compute_data_type, DataType::HALF);
}
