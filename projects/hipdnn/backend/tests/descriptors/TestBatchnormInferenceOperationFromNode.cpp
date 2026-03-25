// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormInferenceOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <memory>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_data_sdk::data_objects;

// =============================================================================
// BatchnormInferenceOperationDescriptor::fromNode() Tests
// =============================================================================

class TestBatchnormInferenceOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = 70;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = {1, 64, 32, 32};
        xAttrs.strides = {65536, 1024, 32, 1};

        _tensorMap[70] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT meanAttrs;
        meanAttrs.uid = 71;
        meanAttrs.data_type = DataType::FLOAT;
        meanAttrs.dims = {1, 64, 1, 1};
        meanAttrs.strides = {64, 1, 1, 1};

        _tensorMap[71] = TensorDescriptor::fromFlatBuffer(meanAttrs);
        TensorAttributesT invVarianceAttrs;
        invVarianceAttrs.uid = 72;
        invVarianceAttrs.data_type = DataType::FLOAT;
        invVarianceAttrs.dims = {1, 64, 1, 1};
        invVarianceAttrs.strides = {64, 1, 1, 1};

        _tensorMap[72] = TensorDescriptor::fromFlatBuffer(invVarianceAttrs);
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = 73;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = {1, 64, 1, 1};
        scaleAttrs.strides = {64, 1, 1, 1};

        _tensorMap[73] = TensorDescriptor::fromFlatBuffer(scaleAttrs);
        TensorAttributesT biasAttrs;
        biasAttrs.uid = 74;
        biasAttrs.data_type = DataType::FLOAT;
        biasAttrs.dims = {1, 64, 1, 1};
        biasAttrs.strides = {64, 1, 1, 1};

        _tensorMap[74] = TensorDescriptor::fromFlatBuffer(biasAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = 75;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = {1, 64, 32, 32};
        yAttrs.strides = {65536, 1024, 32, 1};

        _tensorMap[75] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    static hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesT
        createStandardBatchnormInferenceAttrs()
    {
        hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesT attrs;
        attrs.x_tensor_uid = 70;
        attrs.mean_tensor_uid = 71;
        attrs.inv_variance_tensor_uid = 72;
        attrs.scale_tensor_uid = 73;
        attrs.bias_tensor_uid = 74;
        attrs.y_tensor_uid = 75;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardBatchnormInferenceAttrs());
        return node;
    }
};

TEST_F(TestBatchnormInferenceOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, 70);
}

TEST_F(TestBatchnormInferenceOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BatchnormInferenceAttributes);
    auto desc = std::static_pointer_cast<BatchnormInferenceOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, 70);
    EXPECT_EQ(desc->getData().mean_tensor_uid, 71);
    EXPECT_EQ(desc->getData().inv_variance_tensor_uid, 72);
    EXPECT_EQ(desc->getData().scale_tensor_uid, 73);
    EXPECT_EQ(desc->getData().bias_tensor_uid, 74);
    EXPECT_EQ(desc->getData().y_tensor_uid, 75);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, 70);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, 71);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid, 72);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, 73);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, 74);
    EXPECT_EQ(desc->getYDesc()->getData().uid, 75);
}

TEST_F(TestBatchnormInferenceOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestBatchnormInferenceOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, 70);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, 71);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid, 72);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, 73);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, 74);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, 75);
}

TEST_F(TestBatchnormInferenceOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[70]);
    EXPECT_EQ(desc->getMeanDesc(), _tensorMap[71]);
    EXPECT_EQ(desc->getInvVarianceDesc(), _tensorMap[72]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[73]);
    EXPECT_EQ(desc->getBiasDesc(), _tensorMap[74]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[75]);
}

TEST_F(TestBatchnormInferenceOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, 70);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{65536, 1024, 32, 1}));

    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, 71);
    EXPECT_EQ(desc->getMeanDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getMeanDesc()->getData().dims, (std::vector<int64_t>{1, 64, 1, 1}));
    EXPECT_EQ(desc->getMeanDesc()->getData().strides, (std::vector<int64_t>{64, 1, 1, 1}));

    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid, 72);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().dims, (std::vector<int64_t>{1, 64, 1, 1}));
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().strides, (std::vector<int64_t>{64, 1, 1, 1}));

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, 73);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims, (std::vector<int64_t>{1, 64, 1, 1}));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides, (std::vector<int64_t>{64, 1, 1, 1}));

    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, 74);
    EXPECT_EQ(desc->getBiasDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBiasDesc()->getData().dims, (std::vector<int64_t>{1, 64, 1, 1}));
    EXPECT_EQ(desc->getBiasDesc()->getData().strides, (std::vector<int64_t>{64, 1, 1, 1}));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, 75);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_EQ(desc->getYDesc()->getData().strides, (std::vector<int64_t>{65536, 1024, 32, 1}));
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(70);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingMeanTensor)
{
    _tensorMap.erase(71);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingInvVarianceTensor)
{
    _tensorMap.erase(72);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(73);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingBiasTensor)
{
    _tensorMap.erase(74);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(75);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    EXPECT_EQ(tensors[0]->getData().uid, 70);
    EXPECT_EQ(tensors[1]->getData().uid, 71);
    EXPECT_EQ(tensors[2]->getData().uid, 72);
    EXPECT_EQ(tensors[3]->getData().uid, 73);
    EXPECT_EQ(tensors[4]->getData().uid, 74);
    EXPECT_EQ(tensors[5]->getData().uid, 75);
}

TEST_F(TestBatchnormInferenceOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BatchnormInferenceAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBatchnormInferenceAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, 70);
    EXPECT_EQ(rebuiltAttrs->mean_tensor_uid, 71);
    EXPECT_EQ(rebuiltAttrs->inv_variance_tensor_uid, 72);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, 73);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, 74);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, 75);
}

TEST_F(TestBatchnormInferenceOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        xScoped.get(), 70, HIPDNN_DATA_FLOAT, {1, 64, 32, 32}, {65536, 1024, 32, 1});

    // Verify mean tensor
    hipdnn_backend::ScopedDescriptor meanScoped;
    int64_t meanCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &meanCount,
                       static_cast<void*>(meanScoped.getPtr()));
    ASSERT_EQ(meanCount, 1);
    ASSERT_NE(meanScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        meanScoped.get(), 71, HIPDNN_DATA_FLOAT, {1, 64, 1, 1}, {64, 1, 1, 1});

    // Verify inv_variance tensor
    hipdnn_backend::ScopedDescriptor invVarianceScoped;
    int64_t invVarianceCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &invVarianceCount,
                       static_cast<void*>(invVarianceScoped.getPtr()));
    ASSERT_EQ(invVarianceCount, 1);
    ASSERT_NE(invVarianceScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        invVarianceScoped.get(), 72, HIPDNN_DATA_FLOAT, {1, 64, 1, 1}, {64, 1, 1, 1});

    // Verify scale tensor
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        scaleScoped.get(), 73, HIPDNN_DATA_FLOAT, {1, 64, 1, 1}, {64, 1, 1, 1});

    // Verify bias tensor
    hipdnn_backend::ScopedDescriptor biasScoped;
    int64_t biasCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &biasCount,
                       static_cast<void*>(biasScoped.getPtr()));
    ASSERT_EQ(biasCount, 1);
    ASSERT_NE(biasScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        biasScoped.get(), 74, HIPDNN_DATA_FLOAT, {1, 64, 1, 1}, {64, 1, 1, 1});

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(
        yScoped.get(), 75, HIPDNN_DATA_FLOAT, {1, 64, 32, 32}, {65536, 1024, 32, 1});

    // Verify operation type
    hipdnnOperationType_t opType = HIPDNN_OPERATION_TYPE_NOT_SET;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE);
}

TEST_F(TestBatchnormInferenceOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_batchnorminference_1";

    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_batchnorminference_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_batchnorminference_1");
}

TEST_F(TestBatchnormInferenceOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestBatchnormInferenceOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestBatchnormInferenceOperationFromNode, ToStringIncludesName)
{
    auto node = createStandardNode();
    node.name = "my_batchnorminference_op";

    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);
    auto str = desc->toString();

    EXPECT_NE(str.find("name=my_batchnorminference_op"), std::string::npos);
}
