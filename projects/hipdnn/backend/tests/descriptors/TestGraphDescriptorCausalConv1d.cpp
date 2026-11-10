// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/CausalConv1dOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/causal_conv1d_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/CausalConv1dConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized CausalConv1dOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedCausalConv1dOp(HipdnnBackendDescriptor* xDesc,
                                  HipdnnBackendDescriptor* weightDesc,
                                  HipdnnBackendDescriptor* cuSeqlensDesc,
                                  HipdnnBackendDescriptor* yDesc,
                                  HipdnnBackendDescriptor* biasDesc,
                                  HipdnnBackendDescriptor* initialStateDesc,
                                  HipdnnBackendDescriptor* stateIndicesDesc,
                                  HipdnnBackendDescriptor* finalStateDesc,
                                  hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                  const std::string& name = "")
{
    auto wrapper = createDescriptor<CausalConv1dOperationDescriptor>();
    auto desc = wrapper->asDescriptor<CausalConv1dOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&weightDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&cuSeqlensDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&biasDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&initialStateDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&stateIndicesDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&finalStateDesc));

    auto activation = HIPDNN_CAUSAL_CONV1D_ACTIVATION_SILU_EXT;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                       HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                       1,
                       &activation);
    desc->setAttribute(
        HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(!name.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(name.size()),
                           name.data());
    }

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorCausalConv1d : public ::testing::Test
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

    static const TensorAttributesT* findTensorByUid(const GraphT& graphT, int64_t uid)
    {
        for(const auto& tensor : graphT.tensors)
        {
            if(tensor->uid == uid)
            {
                return tensor.get();
            }
        }
        return nullptr;
    }

    static void verifyTensor(const TensorAttributesT* tensor,
                             int64_t expectedUid,
                             const std::vector<int64_t>& expectedDims,
                             const std::vector<int64_t>& expectedStrides,
                             DataType expectedDataType,
                             bool expectedVirtual = false)
    {
        ASSERT_NE(tensor, nullptr) << "Tensor with UID " << expectedUid
                                   << " not found"; // NOLINT(readability-implicit-bool-conversion)
        EXPECT_EQ(tensor->uid, expectedUid);
        EXPECT_EQ(tensor->dims, expectedDims);
        EXPECT_EQ(tensor->strides, expectedStrides);
        EXPECT_EQ(tensor->data_type, expectedDataType);
        EXPECT_EQ(tensor->virtual_, expectedVirtual);
    }

    static void verifyCausalConv1dNode(const NodeT& node,
                                       DataType expectedComputeType,
                                       int64_t expectedXUid,
                                       int64_t expectedWeightUid,
                                       int64_t expectedCuSeqlensUid,
                                       int64_t expectedYUid,
                                       int64_t expectedBiasUid,
                                       int64_t expectedInitialStateUid,
                                       int64_t expectedStateIndicesUid,
                                       int64_t expectedFinalStateUid,
                                       CausalConv1dActivation expectedCausalConv1dActivation)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::CausalConv1dAttributes);

        auto* attrs = node.attributes.AsCausalConv1dAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->x_tensor_uid, expectedXUid);
        EXPECT_EQ(attrs->weight_tensor_uid, expectedWeightUid);
        EXPECT_EQ(attrs->cu_seqlens_tensor_uid, expectedCuSeqlensUid);
        EXPECT_EQ(attrs->y_tensor_uid, expectedYUid);
        EXPECT_EQ(attrs->bias_tensor_uid, expectedBiasUid);
        EXPECT_EQ(attrs->initial_state_tensor_uid, expectedInitialStateUid);
        EXPECT_EQ(attrs->state_indices_tensor_uid, expectedStateIndicesUid);
        EXPECT_EQ(attrs->final_state_tensor_uid, expectedFinalStateUid);
        EXPECT_EQ(attrs->activation, expectedCausalConv1dActivation);
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

TEST_F(TestGraphDescriptorCausalConv1d, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_X_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS),
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto yDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_Y_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto biasDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS),
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto initialStateDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES),
                                HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES),
                                HIPDNN_DATA_INT32);
    auto finalStateDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedCausalConv1dOp(xDesc.get(),
                                                weightDesc.get(),
                                                cuSeqlensDesc.get(),
                                                yDesc.get(),
                                                biasDesc.get(),
                                                initialStateDesc.get(),
                                                stateIndicesDesc.get(),
                                                finalStateDesc.get());

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

    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 8);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_X_UID),
                 K_CAUSAL_CONV1D_TENSOR_X_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID),
                 K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID),
                 K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_Y_UID),
                 K_CAUSAL_CONV1D_TENSOR_Y_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_BIAS_UID),
                 K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID),
                 K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID),
                 K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID),
                 K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    verifyCausalConv1dNode(*graphT->nodes[0],
                           DataType::FLOAT,
                           K_CAUSAL_CONV1D_TENSOR_X_UID,
                           K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                           K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                           K_CAUSAL_CONV1D_TENSOR_Y_UID,
                           K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                           K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                           K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                           K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                           CausalConv1dActivation::SILU);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorCausalConv1d, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_X_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS),
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto yDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_Y_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto biasDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS),
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto initialStateDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES),
                                HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES),
                                HIPDNN_DATA_INT32);
    auto finalStateDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedCausalConv1dOp(xDesc.get(),
                                                weightDesc.get(),
                                                cuSeqlensDesc.get(),
                                                yDesc.get(),
                                                biasDesc.get(),
                                                initialStateDesc.get(),
                                                stateIndicesDesc.get(),
                                                finalStateDesc.get(),
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
    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorCausalConv1d, CausalConv1dAttributesPreserved)
{
    auto xDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_X_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS),
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto yDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_Y_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto biasDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS),
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto initialStateDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES),
                                HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES),
                                HIPDNN_DATA_INT32);
    auto finalStateDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<CausalConv1dOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<CausalConv1dOperationDescriptor>();

    HipdnnBackendDescriptor* xPtr = xDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&xPtr));
    HipdnnBackendDescriptor* weightPtr = weightDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&weightPtr));
    HipdnnBackendDescriptor* cuSeqlensPtr = cuSeqlensDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&cuSeqlensPtr));
    HipdnnBackendDescriptor* yPtr = yDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&yPtr));
    HipdnnBackendDescriptor* biasPtr = biasDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&biasPtr));
    HipdnnBackendDescriptor* initialStatePtr = initialStateDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&initialStatePtr));
    HipdnnBackendDescriptor* stateIndicesPtr = stateIndicesDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&stateIndicesPtr));
    HipdnnBackendDescriptor* finalStatePtr = finalStateDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&finalStatePtr));

    auto activation = HIPDNN_CAUSAL_CONV1D_ACTIVATION_SILU_EXT;
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                         HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                         1,
                         &activation);

    auto computeType = HIPDNN_DATA_FLOAT;
    opDesc->setAttribute(
        HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set operation name
    const std::string opName = "test_causalconv1d";
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                         HIPDNN_TYPE_CHAR,
                         static_cast<int64_t>(opName.size()),
                         opName.c_str());
    opDesc->finalize();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {wrapper.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 8);

    // Verify tensors
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_X_UID),
                 K_CAUSAL_CONV1D_TENSOR_X_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID),
                 K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID),
                 K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_Y_UID),
                 K_CAUSAL_CONV1D_TENSOR_Y_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_BIAS_UID),
                 K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID),
                 K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID),
                 K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID),
                 K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                 toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS),
                 toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyCausalConv1dNode(*graphT->nodes[0],
                           DataType::FLOAT,
                           K_CAUSAL_CONV1D_TENSOR_X_UID,
                           K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                           K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                           K_CAUSAL_CONV1D_TENSOR_Y_UID,
                           K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                           K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                           K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                           K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                           CausalConv1dActivation::SILU);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_causalconv1d");
}

TEST_F(TestGraphDescriptorCausalConv1d, OperationNamePreservedInSerialization)
{
    auto xDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_X_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS),
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto yDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_Y_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto biasDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS),
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto initialStateDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES),
                                HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES),
                                HIPDNN_DATA_INT32);
    auto finalStateDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedCausalConv1dOp(xDesc.get(),
                                                weightDesc.get(),
                                                cuSeqlensDesc.get(),
                                                yDesc.get(),
                                                biasDesc.get(),
                                                initialStateDesc.get(),
                                                stateIndicesDesc.get(),
                                                finalStateDesc.get(),
                                                HIPDNN_DATA_FLOAT,
                                                "test_causalconv1d_name");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->name, "test_causalconv1d_name");
}

TEST_F(TestGraphDescriptorCausalConv1d, OperationNameRoundTripThroughLifting)
{
    auto xDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_X_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS),
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto yDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_Y_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto biasDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS),
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto initialStateDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES),
                                HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc
        = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS),
                                toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES),
                                HIPDNN_DATA_INT32);
    auto finalStateDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedCausalConv1dOp(xDesc.get(),
                                                weightDesc.get(),
                                                cuSeqlensDesc.get(),
                                                yDesc.get(),
                                                biasDesc.get(),
                                                initialStateDesc.get(),
                                                stateIndicesDesc.get(),
                                                finalStateDesc.get(),
                                                HIPDNN_DATA_FLOAT,
                                                "test_causalconv1d_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_causalconv1d_lifting");
}

} // namespace
