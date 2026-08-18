// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/MoeGroupedMatmulOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulConstants.hpp>
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

// Helper: create a finalized MoeGroupedMatmulOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedMoeGroupedMatmulOp(HipdnnBackendDescriptor* tokenDesc,
                                      HipdnnBackendDescriptor* weightDesc,
                                      HipdnnBackendDescriptor* firstTokenOffsetDesc,
                                      HipdnnBackendDescriptor* tokenIndexDesc,
                                      HipdnnBackendDescriptor* tokenKsDesc,
                                      HipdnnBackendDescriptor* outputDesc,
                                      hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                      const std::string& name = "")
{
    auto wrapper = createDescriptor<MoeGroupedMatmulOperationDescriptor>();
    auto desc = wrapper->asDescriptor<MoeGroupedMatmulOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&tokenDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&weightDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&firstTokenOffsetDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&tokenIndexDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&tokenKsDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&outputDesc));

    auto topK = static_cast<int32_t>(2);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);

    auto mode = HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                       HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                       1,
                       &mode);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

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

class TestGraphDescriptorMoeGroupedMatmul : public ::testing::Test
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

    static void verifyMoeGroupedMatmulNode(const NodeT& node,
                                           DataType expectedComputeType,
                                           int64_t expectedTokenUid,
                                           int64_t expectedWeightUid,
                                           int64_t expectedFirstTokenOffsetUid,
                                           int64_t expectedTokenIndexUid,
                                           int64_t expectedTokenKsUid,
                                           int64_t expectedOutputUid,
                                           MoeGroupedMatmulMode expectedMoeGroupedMatmulMode,
                                           int32_t expectedTopK)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::MoeGroupedMatmulAttributes);

        auto* attrs = node.attributes.AsMoeGroupedMatmulAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->token_tensor_uid, expectedTokenUid);
        EXPECT_EQ(attrs->weight_tensor_uid, expectedWeightUid);
        EXPECT_EQ(attrs->first_token_offset_tensor_uid, expectedFirstTokenOffsetUid);
        EXPECT_EQ(attrs->token_index_tensor_uid, expectedTokenIndexUid);
        EXPECT_EQ(attrs->token_ks_tensor_uid, expectedTokenKsUid);
        EXPECT_EQ(attrs->output_tensor_uid, expectedOutputUid);
        EXPECT_EQ(attrs->mode, expectedMoeGroupedMatmulMode);
        EXPECT_EQ(attrs->top_k, expectedTopK);
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

TEST_F(TestGraphDescriptorMoeGroupedMatmul, BuildFromSingleOperation)
{
    auto tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenIndexDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenKsDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto outputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedMoeGroupedMatmulOp(tokenDesc.get(),
                                                    weightDesc.get(),
                                                    firstTokenOffsetDesc.get(),
                                                    tokenIndexDesc.get(),
                                                    tokenKsDesc.get(),
                                                    outputDesc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 6);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    verifyMoeGroupedMatmulNode(*graphT->nodes[0],
                               DataType::FLOAT,
                               K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                               MoeGroupedMatmulMode::SCATTER,
                               static_cast<int32_t>(2));

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorMoeGroupedMatmul, ComputeDataTypePreserved)
{
    auto tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenIndexDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenKsDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto outputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedMoeGroupedMatmulOp(tokenDesc.get(),
                                                    weightDesc.get(),
                                                    firstTokenOffsetDesc.get(),
                                                    tokenIndexDesc.get(),
                                                    tokenKsDesc.get(),
                                                    outputDesc.get(),
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

TEST_F(TestGraphDescriptorMoeGroupedMatmul, MoeGroupedMatmulAttributesPreserved)
{
    auto tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenIndexDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenKsDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto outputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES),
                                            HIPDNN_DATA_FLOAT);

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<MoeGroupedMatmulOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<MoeGroupedMatmulOperationDescriptor>();

    HipdnnBackendDescriptor* tokenPtr = tokenDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&tokenPtr));
    HipdnnBackendDescriptor* weightPtr = weightDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&weightPtr));
    HipdnnBackendDescriptor* firstTokenOffsetPtr = firstTokenOffsetDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&firstTokenOffsetPtr));
    HipdnnBackendDescriptor* tokenIndexPtr = tokenIndexDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&tokenIndexPtr));
    HipdnnBackendDescriptor* tokenKsPtr = tokenKsDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&tokenKsPtr));
    HipdnnBackendDescriptor* outputPtr = outputDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&outputPtr));

    auto mode = HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                         HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                         1,
                         &mode);

    auto topK = static_cast<int32_t>(2);
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);

    auto computeType = HIPDNN_DATA_FLOAT;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set operation name
    const std::string opName = "test_moegroupedmatmul";
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
    ASSERT_EQ(graphT->tensors.size(), 6);

    // Verify tensors
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID),
                 K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyMoeGroupedMatmulNode(*graphT->nodes[0],
                               DataType::FLOAT,
                               K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                               K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                               MoeGroupedMatmulMode::SCATTER,
                               static_cast<int32_t>(2));

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_moegroupedmatmul");
}

TEST_F(TestGraphDescriptorMoeGroupedMatmul, OperationNamePreservedInSerialization)
{
    auto tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenIndexDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenKsDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto outputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedMoeGroupedMatmulOp(tokenDesc.get(),
                                                    weightDesc.get(),
                                                    firstTokenOffsetDesc.get(),
                                                    tokenIndexDesc.get(),
                                                    tokenKsDesc.get(),
                                                    outputDesc.get(),
                                                    HIPDNN_DATA_FLOAT,
                                                    "test_moegroupedmatmul_name");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_moegroupedmatmul_name");
}

TEST_F(TestGraphDescriptorMoeGroupedMatmul, OperationNameRoundTripThroughLifting)
{
    auto tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto weightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenIndexDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenKsDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto outputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedMoeGroupedMatmulOp(tokenDesc.get(),
                                                    weightDesc.get(),
                                                    firstTokenOffsetDesc.get(),
                                                    tokenIndexDesc.get(),
                                                    tokenKsDesc.get(),
                                                    outputDesc.get(),
                                                    HIPDNN_DATA_FLOAT,
                                                    "test_moegroupedmatmul_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_moegroupedmatmul_lifting");
}

} // namespace
