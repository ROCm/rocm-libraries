// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ResampleBwdOperationDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ResampleBwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_backend::test_utilities::verifyTensorDescriptor;
using hipdnn_tests::toVec;

// =============================================================================
// ResampleBwdOperationDescriptor::fromNode() Tests
// =============================================================================

class TestResampleBwdOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT dyAttrs;
        dyAttrs.uid = K_TENSOR_DY_UID;
        dyAttrs.data_type = DataType::FLOAT;
        dyAttrs.dims = toVec(K_TENSOR_DY_DIMS);
        dyAttrs.strides = toVec(K_TENSOR_DY_STRIDES);

        _tensorMap[K_TENSOR_DY_UID] = TensorDescriptor::fromFlatBuffer(dyAttrs);
        TensorAttributesT dxAttrs;
        dxAttrs.uid = K_TENSOR_DX_UID;
        dxAttrs.data_type = DataType::FLOAT;
        dxAttrs.dims = toVec(K_TENSOR_DX_DIMS);
        dxAttrs.strides = toVec(K_TENSOR_DX_STRIDES);

        _tensorMap[K_TENSOR_DX_UID] = TensorDescriptor::fromFlatBuffer(dxAttrs);
        TensorAttributesT indexAttrs;
        indexAttrs.uid = K_TENSOR_INDEX_UID;
        indexAttrs.data_type = DataType::FLOAT;
        indexAttrs.dims = toVec(K_TENSOR_INDEX_DIMS);
        indexAttrs.strides = toVec(K_TENSOR_INDEX_STRIDES);

        _tensorMap[K_TENSOR_INDEX_UID] = TensorDescriptor::fromFlatBuffer(indexAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributesT
        createStandardResampleBwdAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributesT attrs;
        attrs.dy_tensor_uid = K_TENSOR_DY_UID;
        attrs.dx_tensor_uid = K_TENSOR_DX_UID;
        attrs.index_tensor_uid = K_TENSOR_INDEX_UID;
        attrs.pre_padding = toVec(K_PRE_PADDING);
        attrs.post_padding = toVec(K_POST_PADDING);
        attrs.stride = toVec(K_STRIDE);
        attrs.window = toVec(K_WINDOW);
        attrs.resample_mode = ResampleMode::MAXPOOL;
        attrs.padding_mode = PaddingMode::ZERO_PAD;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardResampleBwdAttrs());
        return node;
    }
};

TEST_F(TestResampleBwdOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR);
    EXPECT_EQ(desc->getData().dy_tensor_uid, K_TENSOR_DY_UID);
}

TEST_F(TestResampleBwdOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    const auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ResampleBwdAttributes);
    auto desc = std::static_pointer_cast<ResampleBwdOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().dy_tensor_uid, K_TENSOR_DY_UID);
    EXPECT_EQ(desc->getData().dx_tensor_uid, K_TENSOR_DX_UID);
    EXPECT_EQ(desc->getData().index_tensor_uid, K_TENSOR_INDEX_UID);
    EXPECT_EQ(desc->getData().pre_padding, toVec(K_PRE_PADDING));
    EXPECT_EQ(desc->getData().post_padding, toVec(K_POST_PADDING));
    EXPECT_EQ(desc->getData().stride, toVec(K_STRIDE));
    EXPECT_EQ(desc->getData().window, toVec(K_WINDOW));
    EXPECT_EQ(desc->getData().resample_mode, ResampleMode::MAXPOOL);
    EXPECT_EQ(desc->getData().padding_mode, PaddingMode::ZERO_PAD);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_TENSOR_DY_UID);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, K_TENSOR_DX_UID);
    EXPECT_EQ(desc->getIndexDesc()->getData().uid, K_TENSOR_INDEX_UID);
}

TEST_F(TestResampleBwdOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestResampleBwdOperationFromNode, BuildNodeOmitsUnsetOptionalScalars)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsResampleBwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    EXPECT_FALSE(rebuiltAttrs->generate_index.has_value());
}

TEST_F(TestResampleBwdOperationFromNode, PreservesResampleMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardResampleBwdAttrs();
    attrs.resample_mode = ResampleMode::AVGPOOL_EXCLUDE_PADDING;
    node.attributes.Set(attrs);
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().resample_mode, ResampleMode::AVGPOOL_EXCLUDE_PADDING);
}

TEST_F(TestResampleBwdOperationFromNode, PreservesPaddingMode)
{
    auto node = createStandardNode();
    auto attrs = createStandardResampleBwdAttrs();
    attrs.padding_mode = PaddingMode::NEG_INF_PAD;
    node.attributes.Set(attrs);
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().padding_mode, PaddingMode::NEG_INF_PAD);
}

TEST_F(TestResampleBwdOperationFromNode, PreservesDataFields)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getData().pre_padding, toVec(K_PRE_PADDING));
    EXPECT_EQ(desc->getData().post_padding, toVec(K_POST_PADDING));
    EXPECT_EQ(desc->getData().stride, toVec(K_STRIDE));
    EXPECT_EQ(desc->getData().window, toVec(K_WINDOW));
    EXPECT_EQ(desc->getData().resample_mode, ResampleMode::MAXPOOL);
    EXPECT_EQ(desc->getData().padding_mode, PaddingMode::ZERO_PAD);
}

TEST_F(TestResampleBwdOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getDyDesc(), nullptr);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_TENSOR_DY_UID);
    ASSERT_NE(desc->getDxDesc(), nullptr);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, K_TENSOR_DX_UID);
    ASSERT_NE(desc->getIndexDesc(), nullptr);
    EXPECT_EQ(desc->getIndexDesc()->getData().uid, K_TENSOR_INDEX_UID);
}

TEST_F(TestResampleBwdOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getDyDesc(), _tensorMap[K_TENSOR_DY_UID]);
    EXPECT_EQ(desc->getDxDesc(), _tensorMap[K_TENSOR_DX_UID]);
    EXPECT_EQ(desc->getIndexDesc(), _tensorMap[K_TENSOR_INDEX_UID]);
}

TEST_F(TestResampleBwdOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getDyDesc(), nullptr);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_TENSOR_DY_UID);
    EXPECT_EQ(desc->getDyDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDyDesc()->getData().dims, (std::vector<int64_t>{1, 3, 16, 16}));
    EXPECT_EQ(desc->getDyDesc()->getData().strides, (std::vector<int64_t>{768, 256, 16, 1}));

    ASSERT_NE(desc->getDxDesc(), nullptr);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, K_TENSOR_DX_UID);
    EXPECT_EQ(desc->getDxDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDxDesc()->getData().dims, (std::vector<int64_t>{1, 3, 32, 32}));
    EXPECT_EQ(desc->getDxDesc()->getData().strides, (std::vector<int64_t>{3072, 1024, 32, 1}));

    ASSERT_NE(desc->getIndexDesc(), nullptr);
    EXPECT_EQ(desc->getIndexDesc()->getData().uid, K_TENSOR_INDEX_UID);
    EXPECT_EQ(desc->getIndexDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getIndexDesc()->getData().dims, (std::vector<int64_t>{1, 3, 16, 16}));
    EXPECT_EQ(desc->getIndexDesc()->getData().strides, (std::vector<int64_t>{768, 256, 16, 1}));
}

TEST_F(TestResampleBwdOperationFromNode, FailsWithMissingDyTensor)
{
    _tensorMap.erase(K_TENSOR_DY_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ResampleBwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestResampleBwdOperationFromNode, FailsWithMissingDxTensor)
{
    _tensorMap.erase(K_TENSOR_DX_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ResampleBwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestResampleBwdOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardResampleBwdAttrs();
    attrs.index_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getDyDesc(), nullptr);
    EXPECT_NE(desc->getDxDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getIndexDesc(), nullptr);
}

TEST_F(TestResampleBwdOperationFromNode, FailsWhenOptionalIndexUidSetButTensorMissing)
{
    _tensorMap.erase(K_TENSOR_INDEX_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(ResampleBwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestResampleBwdOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_TENSOR_DY_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_TENSOR_DX_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_TENSOR_INDEX_UID);
}

TEST_F(TestResampleBwdOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ResampleBwdAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsResampleBwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->dy_tensor_uid, K_TENSOR_DY_UID);
    EXPECT_EQ(rebuiltAttrs->dx_tensor_uid, K_TENSOR_DX_UID);
    EXPECT_EQ(rebuiltAttrs->index_tensor_uid, K_TENSOR_INDEX_UID);
    EXPECT_EQ(rebuiltAttrs->pre_padding, toVec(K_PRE_PADDING));
    EXPECT_EQ(rebuiltAttrs->post_padding, toVec(K_POST_PADDING));
    EXPECT_EQ(rebuiltAttrs->stride, toVec(K_STRIDE));
    EXPECT_EQ(rebuiltAttrs->window, toVec(K_WINDOW));
    EXPECT_EQ(rebuiltAttrs->resample_mode, ResampleMode::MAXPOOL);
    EXPECT_EQ(rebuiltAttrs->padding_mode, PaddingMode::ZERO_PAD);
}

TEST_F(TestResampleBwdOperationFromNode, FromNodePreservesGenerateIndex)
{
    auto attrs = createStandardResampleBwdAttrs();
    attrs.generate_index = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_TRUE(desc->getData().generate_index.has_value());
    EXPECT_EQ(desc->getData().generate_index.value(), true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsResampleBwdAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->generate_index.has_value());
    EXPECT_EQ(rebuiltAttrs->generate_index.value(), true);
}

TEST_F(TestResampleBwdOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify pre_padding
    std::vector<int64_t> prePadding(2);
    int64_t prePaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &prePaddingCount,
                       prePadding.data());
    ASSERT_EQ(prePaddingCount, 2);
    EXPECT_EQ(prePadding, toVec(K_PRE_PADDING));

    // Verify post_padding
    std::vector<int64_t> postPadding(2);
    int64_t postPaddingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_RESAMPLE_POST_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       2,
                       &postPaddingCount,
                       postPadding.data());
    ASSERT_EQ(postPaddingCount, 2);
    EXPECT_EQ(postPadding, toVec(K_POST_PADDING));

    // Verify stride
    std::vector<int64_t> stride(2);
    int64_t strideCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, &strideCount, stride.data());
    ASSERT_EQ(strideCount, 2);
    EXPECT_EQ(stride, toVec(K_STRIDE));

    // Verify window
    std::vector<int64_t> window(2);
    int64_t windowCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, &windowCount, window.data());
    ASSERT_EQ(windowCount, 2);
    EXPECT_EQ(window, toVec(K_WINDOW));

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify resample_mode
    hipdnnResampleMode_t resampleMode = {};
    int64_t resampleModeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleModeCount, &resampleMode);
    ASSERT_EQ(resampleMode, HIPDNN_RESAMPLE_MAXPOOL);

    // Verify padding_mode
    hipdnnPaddingMode_t paddingMode = {};
    int64_t paddingModeCount = 0;
    desc->getAttribute(HIPDNN_ATTR_RESAMPLE_PADDING_MODE,
                       HIPDNN_TYPE_PADDING_MODE,
                       1,
                       &paddingModeCount,
                       &paddingMode);
    ASSERT_EQ(paddingMode, HIPDNN_PADDING_ZERO_PAD);

    // Verify dy tensor
    hipdnn_backend::ScopedDescriptor dyScoped;
    int64_t dyCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dyCount,
                       static_cast<void*>(dyScoped.getPtr()));
    ASSERT_EQ(dyCount, 1);
    ASSERT_NE(dyScoped.get(), nullptr);
    verifyTensorDescriptor(
        dyScoped.get(), K_TENSOR_DY_UID, HIPDNN_DATA_FLOAT, {1, 3, 16, 16}, {768, 256, 16, 1});

    // Verify dx tensor
    hipdnn_backend::ScopedDescriptor dxScoped;
    int64_t dxCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dxCount,
                       static_cast<void*>(dxScoped.getPtr()));
    ASSERT_EQ(dxCount, 1);
    ASSERT_NE(dxScoped.get(), nullptr);
    verifyTensorDescriptor(
        dxScoped.get(), K_TENSOR_DX_UID, HIPDNN_DATA_FLOAT, {1, 3, 32, 32}, {3072, 1024, 32, 1});

    // Verify index tensor (optional)
    hipdnn_backend::ScopedDescriptor indexScoped;
    int64_t indexCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &indexCount,
                       static_cast<void*>(indexScoped.getPtr()));
    ASSERT_EQ(indexCount, 1);
    ASSERT_NE(indexScoped.get(), nullptr);
    verifyTensorDescriptor(indexScoped.get(),
                           K_TENSOR_INDEX_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 3, 16, 16},
                           {768, 256, 16, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_RESAMPLE_BWD);
}

TEST_F(TestResampleBwdOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_resamplebwd_1";

    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_resamplebwd_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_resamplebwd_1");
}

TEST_F(TestResampleBwdOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestResampleBwdOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = ResampleBwdOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
