// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormInferenceVarianceExtOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BnInfVarExtConstants.hpp>
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

// Helper: create a finalized BatchnormInferenceVarianceExtOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedBatchnormInferenceVarianceExtOp(HipdnnBackendDescriptor* xDesc,
                                                   HipdnnBackendDescriptor* meanDesc,
                                                   HipdnnBackendDescriptor* varianceDesc,
                                                   HipdnnBackendDescriptor* scaleDesc,
                                                   HipdnnBackendDescriptor* biasDesc,
                                                   HipdnnBackendDescriptor* yDesc,
                                                   HipdnnBackendDescriptor* epsilonDesc,
                                                   hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                                   const std::string& name = "")
{
    auto wrapper = createDescriptor<BatchnormInferenceVarianceExtOperationDescriptor>();
    auto desc = wrapper->asDescriptor<BatchnormInferenceVarianceExtOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&meanDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&varianceDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&biasDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EPSILON_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&epsilonDesc));
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_VAR_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

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

class TestGraphDescriptorBatchnormInferenceVarianceExt : public ::testing::Test
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

TEST_F(TestGraphDescriptorBatchnormInferenceVarianceExt, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(
        K_BN_INF_VAR_EXT_X_UID, toVec(K_BN_INF_VAR_EXT_X_DIMS), toVec(K_BN_INF_VAR_EXT_X_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_MEAN_UID,
                                          toVec(K_BN_INF_VAR_EXT_MEAN_DIMS),
                                          toVec(K_BN_INF_VAR_EXT_MEAN_STRIDES));
    auto varianceDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_VARIANCE_UID,
                                              toVec(K_BN_INF_VAR_EXT_VARIANCE_DIMS),
                                              toVec(K_BN_INF_VAR_EXT_VARIANCE_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_SCALE_UID,
                                           toVec(K_BN_INF_VAR_EXT_SCALE_DIMS),
                                           toVec(K_BN_INF_VAR_EXT_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_BIAS_UID,
                                          toVec(K_BN_INF_VAR_EXT_BIAS_DIMS),
                                          toVec(K_BN_INF_VAR_EXT_BIAS_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BN_INF_VAR_EXT_Y_UID, toVec(K_BN_INF_VAR_EXT_Y_DIMS), toVec(K_BN_INF_VAR_EXT_Y_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_EPSILON_UID,
                                             toVec(K_BN_INF_VAR_EXT_EPSILON_DIMS),
                                             toVec(K_BN_INF_VAR_EXT_EPSILON_STRIDES));
    auto opDesc = createFinalizedBatchnormInferenceVarianceExtOp(xDesc.get(),
                                                                 meanDesc.get(),
                                                                 varianceDesc.get(),
                                                                 scaleDesc.get(),
                                                                 biasDesc.get(),
                                                                 yDesc.get(),
                                                                 epsilonDesc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 7);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type,
              NodeAttributes::BatchnormInferenceAttributesVarianceExt);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormInferenceAttributesVarianceExt();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->x_tensor_uid, K_BN_INF_VAR_EXT_X_UID);
    EXPECT_EQ(attrs->mean_tensor_uid, K_BN_INF_VAR_EXT_MEAN_UID);
    EXPECT_EQ(attrs->variance_tensor_uid, K_BN_INF_VAR_EXT_VARIANCE_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BN_INF_VAR_EXT_SCALE_UID);
    EXPECT_EQ(attrs->bias_tensor_uid, K_BN_INF_VAR_EXT_BIAS_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_BN_INF_VAR_EXT_Y_UID);
    EXPECT_EQ(attrs->epsilon_tensor_uid, K_BN_INF_VAR_EXT_EPSILON_UID);
}

TEST_F(TestGraphDescriptorBatchnormInferenceVarianceExt, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(
        K_BN_INF_VAR_EXT_X_UID, toVec(K_BN_INF_VAR_EXT_X_DIMS), toVec(K_BN_INF_VAR_EXT_X_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_MEAN_UID,
                                          toVec(K_BN_INF_VAR_EXT_MEAN_DIMS),
                                          toVec(K_BN_INF_VAR_EXT_MEAN_STRIDES));
    auto varianceDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_VARIANCE_UID,
                                              toVec(K_BN_INF_VAR_EXT_VARIANCE_DIMS),
                                              toVec(K_BN_INF_VAR_EXT_VARIANCE_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_SCALE_UID,
                                           toVec(K_BN_INF_VAR_EXT_SCALE_DIMS),
                                           toVec(K_BN_INF_VAR_EXT_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_BIAS_UID,
                                          toVec(K_BN_INF_VAR_EXT_BIAS_DIMS),
                                          toVec(K_BN_INF_VAR_EXT_BIAS_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BN_INF_VAR_EXT_Y_UID, toVec(K_BN_INF_VAR_EXT_Y_DIMS), toVec(K_BN_INF_VAR_EXT_Y_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_EPSILON_UID,
                                             toVec(K_BN_INF_VAR_EXT_EPSILON_DIMS),
                                             toVec(K_BN_INF_VAR_EXT_EPSILON_STRIDES));
    auto opDesc = createFinalizedBatchnormInferenceVarianceExtOp(xDesc.get(),
                                                                 meanDesc.get(),
                                                                 varianceDesc.get(),
                                                                 scaleDesc.get(),
                                                                 biasDesc.get(),
                                                                 yDesc.get(),
                                                                 epsilonDesc.get(),
                                                                 HIPDNN_DATA_HALF);

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

TEST_F(TestGraphDescriptorBatchnormInferenceVarianceExt, OperationNamePreservedInSerialization)
{
    auto xDesc = createFinalizedTensor(
        K_BN_INF_VAR_EXT_X_UID, toVec(K_BN_INF_VAR_EXT_X_DIMS), toVec(K_BN_INF_VAR_EXT_X_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_MEAN_UID,
                                          toVec(K_BN_INF_VAR_EXT_MEAN_DIMS),
                                          toVec(K_BN_INF_VAR_EXT_MEAN_STRIDES));
    auto varianceDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_VARIANCE_UID,
                                              toVec(K_BN_INF_VAR_EXT_VARIANCE_DIMS),
                                              toVec(K_BN_INF_VAR_EXT_VARIANCE_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_SCALE_UID,
                                           toVec(K_BN_INF_VAR_EXT_SCALE_DIMS),
                                           toVec(K_BN_INF_VAR_EXT_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_BIAS_UID,
                                          toVec(K_BN_INF_VAR_EXT_BIAS_DIMS),
                                          toVec(K_BN_INF_VAR_EXT_BIAS_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BN_INF_VAR_EXT_Y_UID, toVec(K_BN_INF_VAR_EXT_Y_DIMS), toVec(K_BN_INF_VAR_EXT_Y_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_EPSILON_UID,
                                             toVec(K_BN_INF_VAR_EXT_EPSILON_DIMS),
                                             toVec(K_BN_INF_VAR_EXT_EPSILON_STRIDES));
    auto opDesc = createFinalizedBatchnormInferenceVarianceExtOp(xDesc.get(),
                                                                 meanDesc.get(),
                                                                 varianceDesc.get(),
                                                                 scaleDesc.get(),
                                                                 biasDesc.get(),
                                                                 yDesc.get(),
                                                                 epsilonDesc.get(),
                                                                 HIPDNN_DATA_FLOAT,
                                                                 "test_bn_inf_var_ext_op");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_bn_inf_var_ext_op");
}

TEST_F(TestGraphDescriptorBatchnormInferenceVarianceExt, OperationNameRoundTripThroughLifting)
{
    auto xDesc = createFinalizedTensor(
        K_BN_INF_VAR_EXT_X_UID, toVec(K_BN_INF_VAR_EXT_X_DIMS), toVec(K_BN_INF_VAR_EXT_X_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_MEAN_UID,
                                          toVec(K_BN_INF_VAR_EXT_MEAN_DIMS),
                                          toVec(K_BN_INF_VAR_EXT_MEAN_STRIDES));
    auto varianceDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_VARIANCE_UID,
                                              toVec(K_BN_INF_VAR_EXT_VARIANCE_DIMS),
                                              toVec(K_BN_INF_VAR_EXT_VARIANCE_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_SCALE_UID,
                                           toVec(K_BN_INF_VAR_EXT_SCALE_DIMS),
                                           toVec(K_BN_INF_VAR_EXT_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_BIAS_UID,
                                          toVec(K_BN_INF_VAR_EXT_BIAS_DIMS),
                                          toVec(K_BN_INF_VAR_EXT_BIAS_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BN_INF_VAR_EXT_Y_UID, toVec(K_BN_INF_VAR_EXT_Y_DIMS), toVec(K_BN_INF_VAR_EXT_Y_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_BN_INF_VAR_EXT_EPSILON_UID,
                                             toVec(K_BN_INF_VAR_EXT_EPSILON_DIMS),
                                             toVec(K_BN_INF_VAR_EXT_EPSILON_STRIDES));
    auto opDesc = createFinalizedBatchnormInferenceVarianceExtOp(xDesc.get(),
                                                                 meanDesc.get(),
                                                                 varianceDesc.get(),
                                                                 scaleDesc.get(),
                                                                 biasDesc.get(),
                                                                 yDesc.get(),
                                                                 epsilonDesc.get(),
                                                                 HIPDNN_DATA_FLOAT,
                                                                 "test_lifting_name");

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

    // Lift with fromNode
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> tensorMap;
    for(const auto& t : graphT->tensors)
    {
        tensorMap[t->uid] = TensorDescriptor::fromFlatBuffer(*t);
    }

    auto liftedOp = NodeFactory::createOperationFromNode(*graphT->nodes[0], tensorMap);
    ASSERT_NE(liftedOp, nullptr);

    // Re-serialize
    auto rebuiltNode = liftedOp->asGraphOperation()->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_lifting_name");
}

} // namespace
