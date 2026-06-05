// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormInferenceOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_test_sdk/constants/BatchnormInferenceConstants.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
namespace
{

// Helper: create a finalized BatchnormInferenceOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedBatchnormInferenceOp(HipdnnBackendDescriptor* xDesc,
                                        HipdnnBackendDescriptor* meanDesc,
                                        HipdnnBackendDescriptor* invVarianceDesc,
                                        HipdnnBackendDescriptor* scaleDesc,
                                        HipdnnBackendDescriptor* biasDesc,
                                        HipdnnBackendDescriptor* yDesc,
                                        hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                        const std::string& name = "")
{
    auto wrapper = createDescriptor<BatchnormInferenceOperationDescriptor>();
    auto desc = wrapper->asDescriptor<BatchnormInferenceOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&meanDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&invVarianceDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&biasDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

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

class TestGraphDescriptorBatchnormInference : public ::testing::Test
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

TEST_F(TestGraphDescriptorBatchnormInference, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(K_BN_INF_TENSOR_X_UID,
                                       {K_BN_INF_SPATIAL_DIMS[0],
                                        K_BN_INF_SPATIAL_DIMS[1],
                                        K_BN_INF_SPATIAL_DIMS[2],
                                        K_BN_INF_SPATIAL_DIMS[3]},
                                       {K_BN_INF_SPATIAL_STRIDES[0],
                                        K_BN_INF_SPATIAL_STRIDES[1],
                                        K_BN_INF_SPATIAL_STRIDES[2],
                                        K_BN_INF_SPATIAL_STRIDES[3]});
    auto meanDesc = createFinalizedTensor(K_BN_INF_TENSOR_MEAN_UID,
                                          {K_BN_INF_CHANNEL_DIMS[0],
                                           K_BN_INF_CHANNEL_DIMS[1],
                                           K_BN_INF_CHANNEL_DIMS[2],
                                           K_BN_INF_CHANNEL_DIMS[3]},
                                          {K_BN_INF_CHANNEL_STRIDES[0],
                                           K_BN_INF_CHANNEL_STRIDES[1],
                                           K_BN_INF_CHANNEL_STRIDES[2],
                                           K_BN_INF_CHANNEL_STRIDES[3]});
    auto invVarianceDesc = createFinalizedTensor(K_BN_INF_TENSOR_INV_VARIANCE_UID,
                                                 {K_BN_INF_CHANNEL_DIMS[0],
                                                  K_BN_INF_CHANNEL_DIMS[1],
                                                  K_BN_INF_CHANNEL_DIMS[2],
                                                  K_BN_INF_CHANNEL_DIMS[3]},
                                                 {K_BN_INF_CHANNEL_STRIDES[0],
                                                  K_BN_INF_CHANNEL_STRIDES[1],
                                                  K_BN_INF_CHANNEL_STRIDES[2],
                                                  K_BN_INF_CHANNEL_STRIDES[3]});
    auto scaleDesc = createFinalizedTensor(K_BN_INF_TENSOR_SCALE_UID,
                                           {K_BN_INF_CHANNEL_DIMS[0],
                                            K_BN_INF_CHANNEL_DIMS[1],
                                            K_BN_INF_CHANNEL_DIMS[2],
                                            K_BN_INF_CHANNEL_DIMS[3]},
                                           {K_BN_INF_CHANNEL_STRIDES[0],
                                            K_BN_INF_CHANNEL_STRIDES[1],
                                            K_BN_INF_CHANNEL_STRIDES[2],
                                            K_BN_INF_CHANNEL_STRIDES[3]});
    auto biasDesc = createFinalizedTensor(K_BN_INF_TENSOR_BIAS_UID,
                                          {K_BN_INF_CHANNEL_DIMS[0],
                                           K_BN_INF_CHANNEL_DIMS[1],
                                           K_BN_INF_CHANNEL_DIMS[2],
                                           K_BN_INF_CHANNEL_DIMS[3]},
                                          {K_BN_INF_CHANNEL_STRIDES[0],
                                           K_BN_INF_CHANNEL_STRIDES[1],
                                           K_BN_INF_CHANNEL_STRIDES[2],
                                           K_BN_INF_CHANNEL_STRIDES[3]});
    auto yDesc = createFinalizedTensor(K_BN_INF_TENSOR_Y_UID,
                                       {K_BN_INF_SPATIAL_DIMS[0],
                                        K_BN_INF_SPATIAL_DIMS[1],
                                        K_BN_INF_SPATIAL_DIMS[2],
                                        K_BN_INF_SPATIAL_DIMS[3]},
                                       {K_BN_INF_SPATIAL_STRIDES[0],
                                        K_BN_INF_SPATIAL_STRIDES[1],
                                        K_BN_INF_SPATIAL_STRIDES[2],
                                        K_BN_INF_SPATIAL_STRIDES[3]});
    auto opDesc = createFinalizedBatchnormInferenceOp(xDesc.get(),
                                                      meanDesc.get(),
                                                      invVarianceDesc.get(),
                                                      scaleDesc.get(),
                                                      biasDesc.get(),
                                                      yDesc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 6);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::BatchnormInferenceAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->x_tensor_uid, K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(attrs->mean_tensor_uid, K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(attrs->inv_variance_tensor_uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->bias_tensor_uid, K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_BN_INF_TENSOR_Y_UID);
}

TEST_F(TestGraphDescriptorBatchnormInference, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(K_BN_INF_TENSOR_X_UID,
                                       {K_BN_INF_SPATIAL_DIMS[0],
                                        K_BN_INF_SPATIAL_DIMS[1],
                                        K_BN_INF_SPATIAL_DIMS[2],
                                        K_BN_INF_SPATIAL_DIMS[3]},
                                       {K_BN_INF_SPATIAL_STRIDES[0],
                                        K_BN_INF_SPATIAL_STRIDES[1],
                                        K_BN_INF_SPATIAL_STRIDES[2],
                                        K_BN_INF_SPATIAL_STRIDES[3]});
    auto meanDesc = createFinalizedTensor(K_BN_INF_TENSOR_MEAN_UID,
                                          {K_BN_INF_CHANNEL_DIMS[0],
                                           K_BN_INF_CHANNEL_DIMS[1],
                                           K_BN_INF_CHANNEL_DIMS[2],
                                           K_BN_INF_CHANNEL_DIMS[3]},
                                          {K_BN_INF_CHANNEL_STRIDES[0],
                                           K_BN_INF_CHANNEL_STRIDES[1],
                                           K_BN_INF_CHANNEL_STRIDES[2],
                                           K_BN_INF_CHANNEL_STRIDES[3]});
    auto invVarianceDesc = createFinalizedTensor(K_BN_INF_TENSOR_INV_VARIANCE_UID,
                                                 {K_BN_INF_CHANNEL_DIMS[0],
                                                  K_BN_INF_CHANNEL_DIMS[1],
                                                  K_BN_INF_CHANNEL_DIMS[2],
                                                  K_BN_INF_CHANNEL_DIMS[3]},
                                                 {K_BN_INF_CHANNEL_STRIDES[0],
                                                  K_BN_INF_CHANNEL_STRIDES[1],
                                                  K_BN_INF_CHANNEL_STRIDES[2],
                                                  K_BN_INF_CHANNEL_STRIDES[3]});
    auto scaleDesc = createFinalizedTensor(K_BN_INF_TENSOR_SCALE_UID,
                                           {K_BN_INF_CHANNEL_DIMS[0],
                                            K_BN_INF_CHANNEL_DIMS[1],
                                            K_BN_INF_CHANNEL_DIMS[2],
                                            K_BN_INF_CHANNEL_DIMS[3]},
                                           {K_BN_INF_CHANNEL_STRIDES[0],
                                            K_BN_INF_CHANNEL_STRIDES[1],
                                            K_BN_INF_CHANNEL_STRIDES[2],
                                            K_BN_INF_CHANNEL_STRIDES[3]});
    auto biasDesc = createFinalizedTensor(K_BN_INF_TENSOR_BIAS_UID,
                                          {K_BN_INF_CHANNEL_DIMS[0],
                                           K_BN_INF_CHANNEL_DIMS[1],
                                           K_BN_INF_CHANNEL_DIMS[2],
                                           K_BN_INF_CHANNEL_DIMS[3]},
                                          {K_BN_INF_CHANNEL_STRIDES[0],
                                           K_BN_INF_CHANNEL_STRIDES[1],
                                           K_BN_INF_CHANNEL_STRIDES[2],
                                           K_BN_INF_CHANNEL_STRIDES[3]});
    auto yDesc = createFinalizedTensor(K_BN_INF_TENSOR_Y_UID,
                                       {K_BN_INF_SPATIAL_DIMS[0],
                                        K_BN_INF_SPATIAL_DIMS[1],
                                        K_BN_INF_SPATIAL_DIMS[2],
                                        K_BN_INF_SPATIAL_DIMS[3]},
                                       {K_BN_INF_SPATIAL_STRIDES[0],
                                        K_BN_INF_SPATIAL_STRIDES[1],
                                        K_BN_INF_SPATIAL_STRIDES[2],
                                        K_BN_INF_SPATIAL_STRIDES[3]});
    auto opDesc = createFinalizedBatchnormInferenceOp(xDesc.get(),
                                                      meanDesc.get(),
                                                      invVarianceDesc.get(),
                                                      scaleDesc.get(),
                                                      biasDesc.get(),
                                                      yDesc.get(),
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

TEST_F(TestGraphDescriptorBatchnormInference, OperationNamePreservedInSerialization)
{
    auto xDesc = createFinalizedTensor(K_BN_INF_TENSOR_X_UID,
                                       {K_BN_INF_SPATIAL_DIMS[0],
                                        K_BN_INF_SPATIAL_DIMS[1],
                                        K_BN_INF_SPATIAL_DIMS[2],
                                        K_BN_INF_SPATIAL_DIMS[3]},
                                       {K_BN_INF_SPATIAL_STRIDES[0],
                                        K_BN_INF_SPATIAL_STRIDES[1],
                                        K_BN_INF_SPATIAL_STRIDES[2],
                                        K_BN_INF_SPATIAL_STRIDES[3]});
    auto meanDesc = createFinalizedTensor(K_BN_INF_TENSOR_MEAN_UID,
                                          {K_BN_INF_CHANNEL_DIMS[0],
                                           K_BN_INF_CHANNEL_DIMS[1],
                                           K_BN_INF_CHANNEL_DIMS[2],
                                           K_BN_INF_CHANNEL_DIMS[3]},
                                          {K_BN_INF_CHANNEL_STRIDES[0],
                                           K_BN_INF_CHANNEL_STRIDES[1],
                                           K_BN_INF_CHANNEL_STRIDES[2],
                                           K_BN_INF_CHANNEL_STRIDES[3]});
    auto invVarianceDesc = createFinalizedTensor(K_BN_INF_TENSOR_INV_VARIANCE_UID,
                                                 {K_BN_INF_CHANNEL_DIMS[0],
                                                  K_BN_INF_CHANNEL_DIMS[1],
                                                  K_BN_INF_CHANNEL_DIMS[2],
                                                  K_BN_INF_CHANNEL_DIMS[3]},
                                                 {K_BN_INF_CHANNEL_STRIDES[0],
                                                  K_BN_INF_CHANNEL_STRIDES[1],
                                                  K_BN_INF_CHANNEL_STRIDES[2],
                                                  K_BN_INF_CHANNEL_STRIDES[3]});
    auto scaleDesc = createFinalizedTensor(K_BN_INF_TENSOR_SCALE_UID,
                                           {K_BN_INF_CHANNEL_DIMS[0],
                                            K_BN_INF_CHANNEL_DIMS[1],
                                            K_BN_INF_CHANNEL_DIMS[2],
                                            K_BN_INF_CHANNEL_DIMS[3]},
                                           {K_BN_INF_CHANNEL_STRIDES[0],
                                            K_BN_INF_CHANNEL_STRIDES[1],
                                            K_BN_INF_CHANNEL_STRIDES[2],
                                            K_BN_INF_CHANNEL_STRIDES[3]});
    auto biasDesc = createFinalizedTensor(K_BN_INF_TENSOR_BIAS_UID,
                                          {K_BN_INF_CHANNEL_DIMS[0],
                                           K_BN_INF_CHANNEL_DIMS[1],
                                           K_BN_INF_CHANNEL_DIMS[2],
                                           K_BN_INF_CHANNEL_DIMS[3]},
                                          {K_BN_INF_CHANNEL_STRIDES[0],
                                           K_BN_INF_CHANNEL_STRIDES[1],
                                           K_BN_INF_CHANNEL_STRIDES[2],
                                           K_BN_INF_CHANNEL_STRIDES[3]});
    auto yDesc = createFinalizedTensor(K_BN_INF_TENSOR_Y_UID,
                                       {K_BN_INF_SPATIAL_DIMS[0],
                                        K_BN_INF_SPATIAL_DIMS[1],
                                        K_BN_INF_SPATIAL_DIMS[2],
                                        K_BN_INF_SPATIAL_DIMS[3]},
                                       {K_BN_INF_SPATIAL_STRIDES[0],
                                        K_BN_INF_SPATIAL_STRIDES[1],
                                        K_BN_INF_SPATIAL_STRIDES[2],
                                        K_BN_INF_SPATIAL_STRIDES[3]});
    auto opDesc = createFinalizedBatchnormInferenceOp(xDesc.get(),
                                                      meanDesc.get(),
                                                      invVarianceDesc.get(),
                                                      scaleDesc.get(),
                                                      biasDesc.get(),
                                                      yDesc.get(),
                                                      HIPDNN_DATA_FLOAT,
                                                      "bn_inf_test");

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

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->name, "bn_inf_test");
}

TEST_F(TestGraphDescriptorBatchnormInference, OperationNameRoundTripThroughLifting)
{
    auto xDesc = createFinalizedTensor(K_BN_INF_TENSOR_X_UID,
                                       {K_BN_INF_SPATIAL_DIMS[0],
                                        K_BN_INF_SPATIAL_DIMS[1],
                                        K_BN_INF_SPATIAL_DIMS[2],
                                        K_BN_INF_SPATIAL_DIMS[3]},
                                       {K_BN_INF_SPATIAL_STRIDES[0],
                                        K_BN_INF_SPATIAL_STRIDES[1],
                                        K_BN_INF_SPATIAL_STRIDES[2],
                                        K_BN_INF_SPATIAL_STRIDES[3]});
    auto meanDesc = createFinalizedTensor(K_BN_INF_TENSOR_MEAN_UID,
                                          {K_BN_INF_CHANNEL_DIMS[0],
                                           K_BN_INF_CHANNEL_DIMS[1],
                                           K_BN_INF_CHANNEL_DIMS[2],
                                           K_BN_INF_CHANNEL_DIMS[3]},
                                          {K_BN_INF_CHANNEL_STRIDES[0],
                                           K_BN_INF_CHANNEL_STRIDES[1],
                                           K_BN_INF_CHANNEL_STRIDES[2],
                                           K_BN_INF_CHANNEL_STRIDES[3]});
    auto invVarianceDesc = createFinalizedTensor(K_BN_INF_TENSOR_INV_VARIANCE_UID,
                                                 {K_BN_INF_CHANNEL_DIMS[0],
                                                  K_BN_INF_CHANNEL_DIMS[1],
                                                  K_BN_INF_CHANNEL_DIMS[2],
                                                  K_BN_INF_CHANNEL_DIMS[3]},
                                                 {K_BN_INF_CHANNEL_STRIDES[0],
                                                  K_BN_INF_CHANNEL_STRIDES[1],
                                                  K_BN_INF_CHANNEL_STRIDES[2],
                                                  K_BN_INF_CHANNEL_STRIDES[3]});
    auto scaleDesc = createFinalizedTensor(K_BN_INF_TENSOR_SCALE_UID,
                                           {K_BN_INF_CHANNEL_DIMS[0],
                                            K_BN_INF_CHANNEL_DIMS[1],
                                            K_BN_INF_CHANNEL_DIMS[2],
                                            K_BN_INF_CHANNEL_DIMS[3]},
                                           {K_BN_INF_CHANNEL_STRIDES[0],
                                            K_BN_INF_CHANNEL_STRIDES[1],
                                            K_BN_INF_CHANNEL_STRIDES[2],
                                            K_BN_INF_CHANNEL_STRIDES[3]});
    auto biasDesc = createFinalizedTensor(K_BN_INF_TENSOR_BIAS_UID,
                                          {K_BN_INF_CHANNEL_DIMS[0],
                                           K_BN_INF_CHANNEL_DIMS[1],
                                           K_BN_INF_CHANNEL_DIMS[2],
                                           K_BN_INF_CHANNEL_DIMS[3]},
                                          {K_BN_INF_CHANNEL_STRIDES[0],
                                           K_BN_INF_CHANNEL_STRIDES[1],
                                           K_BN_INF_CHANNEL_STRIDES[2],
                                           K_BN_INF_CHANNEL_STRIDES[3]});
    auto yDesc = createFinalizedTensor(K_BN_INF_TENSOR_Y_UID,
                                       {K_BN_INF_SPATIAL_DIMS[0],
                                        K_BN_INF_SPATIAL_DIMS[1],
                                        K_BN_INF_SPATIAL_DIMS[2],
                                        K_BN_INF_SPATIAL_DIMS[3]},
                                       {K_BN_INF_SPATIAL_STRIDES[0],
                                        K_BN_INF_SPATIAL_STRIDES[1],
                                        K_BN_INF_SPATIAL_STRIDES[2],
                                        K_BN_INF_SPATIAL_STRIDES[3]});
    auto opDesc = createFinalizedBatchnormInferenceOp(xDesc.get(),
                                                      meanDesc.get(),
                                                      invVarianceDesc.get(),
                                                      scaleDesc.get(),
                                                      biasDesc.get(),
                                                      yDesc.get(),
                                                      HIPDNN_DATA_FLOAT,
                                                      "bn_inf_roundtrip");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Serialize the graph
    auto serialized = desc->getSerializedGraph();
    std::vector<uint8_t> bytes(static_cast<const uint8_t*>(serialized.ptr),
                               static_cast<const uint8_t*>(serialized.ptr) + serialized.size);

    // Deserialize into a new GraphDescriptor (lifting path)
    auto liftedWrapper = createDescriptor<GraphDescriptor>();
    auto liftedDesc = liftedWrapper->asDescriptor<GraphDescriptor>();
    liftedDesc->deserializeGraph(bytes.data(), bytes.size());

    hipdnnHandle_t handle = &_mockHandle;
    liftedDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                             HIPDNN_TYPE_HANDLE,
                             1,
                             static_cast<const void*>(&handle));
    liftedDesc->finalize();

    // Re-serialize and verify name survived the round-trip
    auto reSerialized = liftedDesc->getSerializedGraph();
    auto graphT = UnPackGraph(reSerialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->name, "bn_inf_roundtrip");

    // Verify all tensor UIDs survived
    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->x_tensor_uid, K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(attrs->mean_tensor_uid, K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(attrs->inv_variance_tensor_uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->bias_tensor_uid, K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_BN_INF_TENSOR_Y_UID);
}

} // namespace
