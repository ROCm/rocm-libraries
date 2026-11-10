// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/SoftmaxOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/softmax_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SoftmaxConstants.hpp>
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

// Helper: create a finalized SoftmaxOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedSoftmaxOp(HipdnnBackendDescriptor* pDesc,
                             HipdnnBackendDescriptor* sDesc,
                             HipdnnBackendDescriptor* statsDesc,
                             hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                             const std::string& name = "")
{
    auto wrapper = createDescriptor<SoftmaxOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SoftmaxOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&pDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&sDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&statsDesc));
    desc->setAttribute(HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    auto statsUseLog2 = true;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2, HIPDNN_TYPE_BOOLEAN, 1, &statsUseLog2);

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

class TestGraphDescriptorSoftmax : public ::testing::Test
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

    static void verifySoftmaxNode(const NodeT& node,
                                  DataType expectedComputeType,
                                  int64_t expectedPUid,
                                  int64_t expectedSUid,
                                  int64_t expectedStatsUid,
                                  bool expectedStatsUseLog2)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::SoftmaxAttributes);

        auto* attrs = node.attributes.AsSoftmaxAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->p_tensor_uid, expectedPUid);
        EXPECT_EQ(attrs->s_tensor_uid, expectedSUid);
        EXPECT_EQ(attrs->stats_tensor_uid, expectedStatsUid);
        EXPECT_EQ(attrs->stats_use_log2, expectedStatsUseLog2);
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

TEST_F(TestGraphDescriptorSoftmax, BuildFromSingleOperation)
{
    auto pDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_P_UID,
                                       toVec(K_SOFTMAX_TENSOR_P_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_P_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto sDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_S_UID,
                                       toVec(K_SOFTMAX_TENSOR_S_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_S_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto statsDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_STATS_UID,
                                           toVec(K_SOFTMAX_TENSOR_STATS_DIMS),
                                           toVec(K_SOFTMAX_TENSOR_STATS_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedSoftmaxOp(pDesc.get(), sDesc.get(), statsDesc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_SOFTMAX_TENSOR_P_UID),
                 K_SOFTMAX_TENSOR_P_UID,
                 toVec(K_SOFTMAX_TENSOR_P_DIMS),
                 toVec(K_SOFTMAX_TENSOR_P_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SOFTMAX_TENSOR_S_UID),
                 K_SOFTMAX_TENSOR_S_UID,
                 toVec(K_SOFTMAX_TENSOR_S_DIMS),
                 toVec(K_SOFTMAX_TENSOR_S_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SOFTMAX_TENSOR_STATS_UID),
                 K_SOFTMAX_TENSOR_STATS_UID,
                 toVec(K_SOFTMAX_TENSOR_STATS_DIMS),
                 toVec(K_SOFTMAX_TENSOR_STATS_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    verifySoftmaxNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_SOFTMAX_TENSOR_P_UID,
                      K_SOFTMAX_TENSOR_S_UID,
                      K_SOFTMAX_TENSOR_STATS_UID,
                      true);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorSoftmax, ComputeDataTypePreserved)
{
    auto pDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_P_UID,
                                       toVec(K_SOFTMAX_TENSOR_P_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_P_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto sDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_S_UID,
                                       toVec(K_SOFTMAX_TENSOR_S_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_S_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto statsDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_STATS_UID,
                                           toVec(K_SOFTMAX_TENSOR_STATS_DIMS),
                                           toVec(K_SOFTMAX_TENSOR_STATS_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto opDesc
        = createFinalizedSoftmaxOp(pDesc.get(), sDesc.get(), statsDesc.get(), HIPDNN_DATA_HALF);

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

TEST_F(TestGraphDescriptorSoftmax, SoftmaxAttributesPreserved)
{
    auto pDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_P_UID,
                                       toVec(K_SOFTMAX_TENSOR_P_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_P_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto sDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_S_UID,
                                       toVec(K_SOFTMAX_TENSOR_S_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_S_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto statsDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_STATS_UID,
                                           toVec(K_SOFTMAX_TENSOR_STATS_DIMS),
                                           toVec(K_SOFTMAX_TENSOR_STATS_STRIDES),
                                           HIPDNN_DATA_FLOAT);

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<SoftmaxOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<SoftmaxOperationDescriptor>();

    HipdnnBackendDescriptor* pPtr = pDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&pPtr));
    HipdnnBackendDescriptor* sPtr = sDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&sPtr));
    HipdnnBackendDescriptor* statsPtr = statsDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&statsPtr));

    auto statsUseLog2 = true;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2, HIPDNN_TYPE_BOOLEAN, 1, &statsUseLog2);

    auto computeType = HIPDNN_DATA_FLOAT;
    opDesc->setAttribute(HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set operation name
    const std::string opName = "test_softmax";
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
    ASSERT_EQ(graphT->tensors.size(), 3);

    // Verify tensors
    verifyTensor(findTensorByUid(*graphT, K_SOFTMAX_TENSOR_P_UID),
                 K_SOFTMAX_TENSOR_P_UID,
                 toVec(K_SOFTMAX_TENSOR_P_DIMS),
                 toVec(K_SOFTMAX_TENSOR_P_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SOFTMAX_TENSOR_S_UID),
                 K_SOFTMAX_TENSOR_S_UID,
                 toVec(K_SOFTMAX_TENSOR_S_DIMS),
                 toVec(K_SOFTMAX_TENSOR_S_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SOFTMAX_TENSOR_STATS_UID),
                 K_SOFTMAX_TENSOR_STATS_UID,
                 toVec(K_SOFTMAX_TENSOR_STATS_DIMS),
                 toVec(K_SOFTMAX_TENSOR_STATS_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifySoftmaxNode(*graphT->nodes[0],
                      DataType::FLOAT,
                      K_SOFTMAX_TENSOR_P_UID,
                      K_SOFTMAX_TENSOR_S_UID,
                      K_SOFTMAX_TENSOR_STATS_UID,
                      true);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_softmax");
}

TEST_F(TestGraphDescriptorSoftmax, OperationNamePreservedInSerialization)
{
    auto pDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_P_UID,
                                       toVec(K_SOFTMAX_TENSOR_P_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_P_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto sDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_S_UID,
                                       toVec(K_SOFTMAX_TENSOR_S_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_S_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto statsDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_STATS_UID,
                                           toVec(K_SOFTMAX_TENSOR_STATS_DIMS),
                                           toVec(K_SOFTMAX_TENSOR_STATS_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedSoftmaxOp(
        pDesc.get(), sDesc.get(), statsDesc.get(), HIPDNN_DATA_FLOAT, "test_softmax_name");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_softmax_name");
}

TEST_F(TestGraphDescriptorSoftmax, OperationNameRoundTripThroughLifting)
{
    auto pDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_P_UID,
                                       toVec(K_SOFTMAX_TENSOR_P_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_P_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto sDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_S_UID,
                                       toVec(K_SOFTMAX_TENSOR_S_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_S_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto statsDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_STATS_UID,
                                           toVec(K_SOFTMAX_TENSOR_STATS_DIMS),
                                           toVec(K_SOFTMAX_TENSOR_STATS_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedSoftmaxOp(
        pDesc.get(), sDesc.get(), statsDesc.get(), HIPDNN_DATA_FLOAT, "test_softmax_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_softmax_lifting");
}

} // namespace
