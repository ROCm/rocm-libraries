// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/MoeGroupedMatmulBwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulBwdConstants.hpp>
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

// Helper: create a finalized MoeGroupedMatmulBwdOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedMoeGroupedMatmulBwdOp(HipdnnBackendDescriptor* doutputDesc,
                                         HipdnnBackendDescriptor* tokenDesc,
                                         HipdnnBackendDescriptor* firstTokenOffsetDesc,
                                         HipdnnBackendDescriptor* dweightDesc,
                                         hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                         const std::string& name = "")
{
    auto wrapper = createDescriptor<MoeGroupedMatmulBwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<MoeGroupedMatmulBwdOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&doutputDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&tokenDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&firstTokenOffsetDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dweightDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);

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

class TestGraphDescriptorMoeGroupedMatmulBwd : public ::testing::Test
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

TEST_F(TestGraphDescriptorMoeGroupedMatmulBwd, BuildFromSingleOperation)
{
    auto doutputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
    auto tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto dweightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedMoeGroupedMatmulBwdOp(
        doutputDesc.get(), tokenDesc.get(), firstTokenOffsetDesc.get(), dweightDesc.get());

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
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID),
                 K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID,
                 toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID),
                 K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID,
                 toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID),
                 K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
                 toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID),
                 K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID,
                 toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS),
                 toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::MoeGroupedMatmulBwdAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsMoeGroupedMatmulBwdAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::FLOAT);

    // Verify tensor UID references
    EXPECT_EQ(attrs->doutput_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    EXPECT_EQ(attrs->token_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    EXPECT_EQ(attrs->first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(attrs->dweight_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorMoeGroupedMatmulBwd, ComputeDataTypePreserved)
{
    auto doutputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
    auto tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto dweightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedMoeGroupedMatmulBwdOp(doutputDesc.get(),
                                                       tokenDesc.get(),
                                                       firstTokenOffsetDesc.get(),
                                                       dweightDesc.get(),
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

TEST_F(TestGraphDescriptorMoeGroupedMatmulBwd, OperationNamePreservedInSerialization)
{
    auto doutputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
    auto tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto dweightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedMoeGroupedMatmulBwdOp(doutputDesc.get(),
                                                       tokenDesc.get(),
                                                       firstTokenOffsetDesc.get(),
                                                       dweightDesc.get(),
                                                       HIPDNN_DATA_FLOAT,
                                                       "test_moegroupedmatmulbwd_name");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_moegroupedmatmulbwd_name");
}

TEST_F(TestGraphDescriptorMoeGroupedMatmulBwd, OperationNameRoundTripThroughLifting)
{
    auto doutputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
    auto tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto firstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_INT32);
    auto dweightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedMoeGroupedMatmulBwdOp(doutputDesc.get(),
                                                       tokenDesc.get(),
                                                       firstTokenOffsetDesc.get(),
                                                       dweightDesc.get(),
                                                       HIPDNN_DATA_FLOAT,
                                                       "test_moegroupedmatmulbwd_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_moegroupedmatmulbwd_lifting");
}

} // namespace
