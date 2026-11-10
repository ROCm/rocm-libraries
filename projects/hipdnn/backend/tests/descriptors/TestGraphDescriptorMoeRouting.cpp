// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/MoeRoutingOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_routing_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MoeRoutingConstants.hpp>
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

// Helper: create a finalized MoeRoutingOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedMoeRoutingOp(HipdnnBackendDescriptor* topkIdsDesc,
                                HipdnnBackendDescriptor* firstTokenOffsetDesc,
                                HipdnnBackendDescriptor* tokenIndexDesc,
                                HipdnnBackendDescriptor* tokenKsDesc,
                                [[maybe_unused]] hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                const std::string& name = "")
{
    auto wrapper = createDescriptor<MoeRoutingOperationDescriptor>();
    auto desc = wrapper->asDescriptor<MoeRoutingOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&topkIdsDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&firstTokenOffsetDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&tokenIndexDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&tokenKsDesc));

    auto numExperts = static_cast<int64_t>(4);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_ROUTING_NUM_EXPERTS_EXT, HIPDNN_TYPE_INT64, 1, &numExperts);

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

class TestGraphDescriptorMoeRouting : public ::testing::Test
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

    static void verifyMoeRoutingNode(const NodeT& node,
                                     int64_t expectedTopkIdsUid,
                                     int64_t expectedFirstTokenOffsetUid,
                                     int64_t expectedTokenIndexUid,
                                     int64_t expectedTokenKsUid,
                                     int64_t expectedNumExperts)
    {
        ASSERT_EQ(node.attributes.type, NodeAttributes::MoeRoutingAttributes);

        auto* attrs = node.attributes.AsMoeRoutingAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->topk_ids_tensor_uid, expectedTopkIdsUid);
        EXPECT_EQ(attrs->first_token_offset_tensor_uid, expectedFirstTokenOffsetUid);
        EXPECT_EQ(attrs->token_index_tensor_uid, expectedTokenIndexUid);
        EXPECT_EQ(attrs->token_ks_tensor_uid, expectedTokenKsUid);
        EXPECT_EQ(attrs->num_experts, expectedNumExperts);
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

TEST_F(TestGraphDescriptorMoeRouting, BuildFromSingleOperation)
{
    auto topkIdsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenIndexDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS),
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES),
                                                HIPDNN_DATA_INT32);
    auto tokenKsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto opDesc = createFinalizedMoeRoutingOp(
        topkIdsDesc.get(), firstTokenOffsetDesc.get(), tokenIndexDesc.get(), tokenKsDesc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 4);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID),
                 K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                 toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS),
                 toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID),
                 K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                 toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                 toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID),
                 K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                 toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS),
                 toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID),
                 K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                 toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS),
                 toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES),
                 DataType::INT32);

    // Verify node attributes
    verifyMoeRoutingNode(*graphT->nodes[0],
                         K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                         K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                         K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                         K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                         static_cast<int64_t>(4));

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorMoeRouting, MoeRoutingAttributesPreserved)
{
    auto topkIdsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenIndexDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS),
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES),
                                                HIPDNN_DATA_INT32);
    auto tokenKsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<MoeRoutingOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<MoeRoutingOperationDescriptor>();

    HipdnnBackendDescriptor* topkIdsPtr = topkIdsDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&topkIdsPtr));
    HipdnnBackendDescriptor* firstTokenOffsetPtr = firstTokenOffsetDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&firstTokenOffsetPtr));
    HipdnnBackendDescriptor* tokenIndexPtr = tokenIndexDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&tokenIndexPtr));
    HipdnnBackendDescriptor* tokenKsPtr = tokenKsDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&tokenKsPtr));

    auto numExperts = static_cast<int64_t>(4);
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_ROUTING_NUM_EXPERTS_EXT, HIPDNN_TYPE_INT64, 1, &numExperts);

    // Set operation name
    const std::string opName = "test_moerouting";
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
    ASSERT_EQ(graphT->tensors.size(), 4);

    // Verify tensors
    verifyTensor(findTensorByUid(*graphT, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID),
                 K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                 toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS),
                 toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID),
                 K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                 toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                 toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID),
                 K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                 toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS),
                 toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID),
                 K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                 toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS),
                 toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES),
                 DataType::INT32);

    // Verify node with non-default attribute values
    verifyMoeRoutingNode(*graphT->nodes[0],
                         K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                         K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                         K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                         K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                         static_cast<int64_t>(4));

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_moerouting");
}

TEST_F(TestGraphDescriptorMoeRouting, OperationNamePreservedInSerialization)
{
    auto topkIdsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenIndexDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS),
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES),
                                                HIPDNN_DATA_INT32);
    auto tokenKsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto opDesc = createFinalizedMoeRoutingOp(topkIdsDesc.get(),
                                              firstTokenOffsetDesc.get(),
                                              tokenIndexDesc.get(),
                                              tokenKsDesc.get(),
                                              HIPDNN_DATA_FLOAT,
                                              "test_moerouting_name");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_moerouting_name");
}

TEST_F(TestGraphDescriptorMoeRouting, OperationNameRoundTripThroughLifting)
{
    auto topkIdsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto tokenIndexDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS),
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES),
                                                HIPDNN_DATA_INT32);
    auto tokenKsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
    auto opDesc = createFinalizedMoeRoutingOp(topkIdsDesc.get(),
                                              firstTokenOffsetDesc.get(),
                                              tokenIndexDesc.get(),
                                              tokenKsDesc.get(),
                                              HIPDNN_DATA_FLOAT,
                                              "test_moerouting_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_moerouting_lifting");
}

} // namespace
