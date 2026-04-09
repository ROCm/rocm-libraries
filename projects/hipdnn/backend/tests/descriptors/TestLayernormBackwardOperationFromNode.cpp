// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/LayernormBackwardOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/LayernormBackwardConstants.hpp>
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
// LayernormBackwardOperationDescriptor::fromNode() Tests
// =============================================================================

class TestLayernormBackwardOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT dyAttrs;
        dyAttrs.uid = K_LAYERNORMBACKWARD_TENSOR_DY_UID;
        dyAttrs.data_type = DataType::FLOAT;
        dyAttrs.dims = toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS);
        dyAttrs.strides = toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES);

        _tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID] = TensorDescriptor::fromFlatBuffer(dyAttrs);
        TensorAttributesT xAttrs;
        xAttrs.uid = K_LAYERNORMBACKWARD_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES);

        _tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = K_LAYERNORMBACKWARD_TENSOR_SCALE_UID;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS);
        scaleAttrs.strides = toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES);

        _tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]
            = TensorDescriptor::fromFlatBuffer(scaleAttrs);
        TensorAttributesT meanAttrs;
        meanAttrs.uid = K_LAYERNORMBACKWARD_TENSOR_MEAN_UID;
        meanAttrs.data_type = DataType::FLOAT;
        meanAttrs.dims = toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS);
        meanAttrs.strides = toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES);

        _tensorMap[K_LAYERNORMBACKWARD_TENSOR_MEAN_UID]
            = TensorDescriptor::fromFlatBuffer(meanAttrs);
        TensorAttributesT invVarianceAttrs;
        invVarianceAttrs.uid = K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID;
        invVarianceAttrs.data_type = DataType::FLOAT;
        invVarianceAttrs.dims = toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS);
        invVarianceAttrs.strides = toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES);

        _tensorMap[K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID]
            = TensorDescriptor::fromFlatBuffer(invVarianceAttrs);
        TensorAttributesT epsilonAttrs;
        epsilonAttrs.uid = K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID;
        epsilonAttrs.data_type = DataType::FLOAT;
        epsilonAttrs.dims = toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS);
        epsilonAttrs.strides = toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES);

        _tensorMap[K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID]
            = TensorDescriptor::fromFlatBuffer(epsilonAttrs);
        TensorAttributesT dxAttrs;
        dxAttrs.uid = K_LAYERNORMBACKWARD_TENSOR_DX_UID;
        dxAttrs.data_type = DataType::FLOAT;
        dxAttrs.dims = toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS);
        dxAttrs.strides = toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES);

        _tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID] = TensorDescriptor::fromFlatBuffer(dxAttrs);
        TensorAttributesT dscaleAttrs;
        dscaleAttrs.uid = K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID;
        dscaleAttrs.data_type = DataType::FLOAT;
        dscaleAttrs.dims = toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS);
        dscaleAttrs.strides = toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES);

        _tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]
            = TensorDescriptor::fromFlatBuffer(dscaleAttrs);
        TensorAttributesT dbiasAttrs;
        dbiasAttrs.uid = K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID;
        dbiasAttrs.data_type = DataType::FLOAT;
        dbiasAttrs.dims = toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS);
        dbiasAttrs.strides = toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES);

        _tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]
            = TensorDescriptor::fromFlatBuffer(dbiasAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributesT
        createStandardLayernormBackwardAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributesT attrs;
        attrs.dy_tensor_uid = K_LAYERNORMBACKWARD_TENSOR_DY_UID;
        attrs.x_tensor_uid = K_LAYERNORMBACKWARD_TENSOR_X_UID;
        attrs.scale_tensor_uid = K_LAYERNORMBACKWARD_TENSOR_SCALE_UID;
        attrs.mean_tensor_uid = K_LAYERNORMBACKWARD_TENSOR_MEAN_UID;
        attrs.inv_variance_tensor_uid = K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID;
        attrs.epsilon_tensor_uid = K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID;
        attrs.dx_tensor_uid = K_LAYERNORMBACKWARD_TENSOR_DX_UID;
        attrs.dscale_tensor_uid = K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID;
        attrs.dbias_tensor_uid = K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID;
        attrs.normalized_dim_count = 3;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardLayernormBackwardAttrs());
        return node;
    }
};

TEST_F(TestLayernormBackwardOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_LAYERNORM_BACKWARD_DESCRIPTOR);
    EXPECT_EQ(desc->getData().dy_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
}

TEST_F(TestLayernormBackwardOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::LayernormBackwardAttributes);
    auto desc = std::static_pointer_cast<LayernormBackwardOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().dy_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getData().mean_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    EXPECT_EQ(desc->getData().inv_variance_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getData().epsilon_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    EXPECT_EQ(desc->getData().dx_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(desc->getData().dscale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(desc->getData().dbias_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(desc->getDscaleDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(desc->getDbiasDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid,
              K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
}

TEST_F(TestLayernormBackwardOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestLayernormBackwardOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getDyDesc(), nullptr);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getDxDesc(), nullptr);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    ASSERT_NE(desc->getDscaleDesc(), nullptr);
    EXPECT_EQ(desc->getDscaleDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    ASSERT_NE(desc->getDbiasDesc(), nullptr);
    EXPECT_EQ(desc->getDbiasDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid,
              K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
}

TEST_F(TestLayernormBackwardOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getDyDesc(), _tensorMap[K_LAYERNORMBACKWARD_TENSOR_DY_UID]);
    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_LAYERNORMBACKWARD_TENSOR_X_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_LAYERNORMBACKWARD_TENSOR_SCALE_UID]);
    EXPECT_EQ(desc->getDxDesc(), _tensorMap[K_LAYERNORMBACKWARD_TENSOR_DX_UID]);
    EXPECT_EQ(desc->getDscaleDesc(), _tensorMap[K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID]);
    EXPECT_EQ(desc->getDbiasDesc(), _tensorMap[K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID]);
    EXPECT_EQ(desc->getMeanDesc(), _tensorMap[K_LAYERNORMBACKWARD_TENSOR_MEAN_UID]);
    EXPECT_EQ(desc->getInvVarianceDesc(), _tensorMap[K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID]);
    EXPECT_EQ(desc->getEpsilonDesc(), _tensorMap[K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID]);
}

TEST_F(TestLayernormBackwardOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getDyDesc(), nullptr);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(desc->getDyDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDyDesc()->getData().dims, (std::vector<int64_t>{16, 64, 32, 32}));
    EXPECT_EQ(desc->getDyDesc()->getData().strides, (std::vector<int64_t>{65536, 1024, 32, 1}));

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{16, 64, 32, 32}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{65536, 1024, 32, 1}));

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims, (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides, (std::vector<int64_t>{65536, 1024, 32, 1}));

    ASSERT_NE(desc->getDxDesc(), nullptr);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(desc->getDxDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDxDesc()->getData().dims, (std::vector<int64_t>{16, 64, 32, 32}));
    EXPECT_EQ(desc->getDxDesc()->getData().strides, (std::vector<int64_t>{65536, 1024, 32, 1}));

    ASSERT_NE(desc->getDscaleDesc(), nullptr);
    EXPECT_EQ(desc->getDscaleDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(desc->getDscaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDscaleDesc()->getData().dims, (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_EQ(desc->getDscaleDesc()->getData().strides, (std::vector<int64_t>{65536, 1024, 32, 1}));

    ASSERT_NE(desc->getDbiasDesc(), nullptr);
    EXPECT_EQ(desc->getDbiasDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
    EXPECT_EQ(desc->getDbiasDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDbiasDesc()->getData().dims, (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_EQ(desc->getDbiasDesc()->getData().strides, (std::vector<int64_t>{65536, 1024, 32, 1}));

    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    EXPECT_EQ(desc->getMeanDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getMeanDesc()->getData().dims, (std::vector<int64_t>{16, 1, 1, 1}));
    EXPECT_EQ(desc->getMeanDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid,
              K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().dims, (std::vector<int64_t>{16, 1, 1, 1}));
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getEpsilonDesc()->getData().strides, (std::vector<int64_t>{1}));
}

TEST_F(TestLayernormBackwardOperationFromNode, FailsWithMissingDyTensor)
{
    _tensorMap.erase(K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormBackwardOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_LAYERNORMBACKWARD_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormBackwardOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormBackwardOperationFromNode, FailsWithMissingDxTensor)
{
    _tensorMap.erase(K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormBackwardOperationFromNode, FailsWithMissingDscaleTensor)
{
    _tensorMap.erase(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormBackwardOperationFromNode, FailsWithMissingDbiasTensor)
{
    _tensorMap.erase(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormBackwardOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardLayernormBackwardAttrs();
    attrs.mean_tensor_uid = flatbuffers::nullopt;
    attrs.inv_variance_tensor_uid = flatbuffers::nullopt;
    attrs.epsilon_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getDyDesc(), nullptr);
    EXPECT_NE(desc->getXDesc(), nullptr);
    EXPECT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_NE(desc->getDxDesc(), nullptr);
    EXPECT_NE(desc->getDscaleDesc(), nullptr);
    EXPECT_NE(desc->getDbiasDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationFromNode, FailsWhenOptionalMeanUidSetButTensorMissing)
{
    _tensorMap.erase(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormBackwardOperationFromNode, FailsWhenOptionalInvVarianceUidSetButTensorMissing)
{
    _tensorMap.erase(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormBackwardOperationFromNode, FailsWhenOptionalEpsilonUidSetButTensorMissing)
{
    _tensorMap.erase(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormBackwardOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 9);
    EXPECT_EQ(tensors[0]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    EXPECT_EQ(tensors[6]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(tensors[7]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(tensors[8]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
}

TEST_F(TestLayernormBackwardOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::LayernormBackwardAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsLayernormBackwardAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->dy_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->mean_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    EXPECT_EQ(rebuiltAttrs->inv_variance_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(rebuiltAttrs->epsilon_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    EXPECT_EQ(rebuiltAttrs->dx_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(rebuiltAttrs->dscale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(rebuiltAttrs->dbias_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
}

TEST_F(TestLayernormBackwardOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &dtCount,
                       &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify dy tensor
    hipdnn_backend::ScopedDescriptor dyScoped;
    int64_t dyCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dyCount,
                       static_cast<void*>(dyScoped.getPtr()));
    ASSERT_EQ(dyCount, 1);
    ASSERT_NE(dyScoped.get(), nullptr);
    verifyTensorDescriptor(dyScoped.get(),
                           K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                           HIPDNN_DATA_FLOAT,
                           {16, 64, 32, 32},
                           {65536, 1024, 32, 1});

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(xScoped.get(),
                           K_LAYERNORMBACKWARD_TENSOR_X_UID,
                           HIPDNN_DATA_FLOAT,
                           {16, 64, 32, 32},
                           {65536, 1024, 32, 1});

    // Verify scale tensor
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    verifyTensorDescriptor(scaleScoped.get(),
                           K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 64, 32, 32},
                           {65536, 1024, 32, 1});

    // Verify dx tensor
    hipdnn_backend::ScopedDescriptor dxScoped;
    int64_t dxCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dxCount,
                       static_cast<void*>(dxScoped.getPtr()));
    ASSERT_EQ(dxCount, 1);
    ASSERT_NE(dxScoped.get(), nullptr);
    verifyTensorDescriptor(dxScoped.get(),
                           K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                           HIPDNN_DATA_FLOAT,
                           {16, 64, 32, 32},
                           {65536, 1024, 32, 1});

    // Verify dscale tensor
    hipdnn_backend::ScopedDescriptor dscaleScoped;
    int64_t dscaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dscaleCount,
                       static_cast<void*>(dscaleScoped.getPtr()));
    ASSERT_EQ(dscaleCount, 1);
    ASSERT_NE(dscaleScoped.get(), nullptr);
    verifyTensorDescriptor(dscaleScoped.get(),
                           K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 64, 32, 32},
                           {65536, 1024, 32, 1});

    // Verify dbias tensor
    hipdnn_backend::ScopedDescriptor dbiasScoped;
    int64_t dbiasCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dbiasCount,
                       static_cast<void*>(dbiasScoped.getPtr()));
    ASSERT_EQ(dbiasCount, 1);
    ASSERT_NE(dbiasScoped.get(), nullptr);
    verifyTensorDescriptor(dbiasScoped.get(),
                           K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID,
                           HIPDNN_DATA_FLOAT,
                           {1, 64, 32, 32},
                           {65536, 1024, 32, 1});

    // Verify mean tensor (optional)
    hipdnn_backend::ScopedDescriptor meanScoped;
    int64_t meanCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &meanCount,
                       static_cast<void*>(meanScoped.getPtr()));
    ASSERT_EQ(meanCount, 1);
    ASSERT_NE(meanScoped.get(), nullptr);
    verifyTensorDescriptor(meanScoped.get(),
                           K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                           HIPDNN_DATA_FLOAT,
                           {16, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify inv_variance tensor (optional)
    hipdnn_backend::ScopedDescriptor invVarianceScoped;
    int64_t invVarianceCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &invVarianceCount,
                       static_cast<void*>(invVarianceScoped.getPtr()));
    ASSERT_EQ(invVarianceCount, 1);
    ASSERT_NE(invVarianceScoped.get(), nullptr);
    verifyTensorDescriptor(invVarianceScoped.get(),
                           K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                           HIPDNN_DATA_FLOAT,
                           {16, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify epsilon tensor (optional)
    hipdnn_backend::ScopedDescriptor epsilonScoped;
    int64_t epsilonCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &epsilonCount,
                       static_cast<void*>(epsilonScoped.getPtr()));
    ASSERT_EQ(epsilonCount, 1);
    ASSERT_NE(epsilonScoped.get(), nullptr);
    verifyTensorDescriptor(
        epsilonScoped.get(), K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID, HIPDNN_DATA_FLOAT, {1}, {1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_LAYERNORM_BACKWARD_EXT);
}

TEST_F(TestLayernormBackwardOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_layernormbackward_1";

    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_layernormbackward_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_layernormbackward_1");
}

TEST_F(TestLayernormBackwardOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestLayernormBackwardOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = LayernormBackwardOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
