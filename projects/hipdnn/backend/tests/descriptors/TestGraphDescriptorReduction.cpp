// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/ReductionOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ReductionConstants.hpp>
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

// Helper: create a finalized ReductionOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedReductionOp(HipdnnBackendDescriptor* xDesc,
                               HipdnnBackendDescriptor* yDesc,
                               hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                               const std::string& name = "",
                               bool isDeterministic = false)
{
    auto wrapper = createDescriptor<ReductionOperationDescriptor>();
    auto desc = wrapper->asDescriptor<ReductionOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_YDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));

    auto mode = HIPDNN_REDUCE_TENSOR_ADD;
    desc->setAttribute(
        HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &mode);
    desc->setAttribute(HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(isDeterministic)
    {
        bool trueVal = true;
        desc->setAttribute(
            HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    }

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

class TestGraphDescriptorReduction : public ::testing::Test
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
        ASSERT_NE(tensor, nullptr) << "Tensor with UID " << expectedUid << " not found";
        EXPECT_EQ(tensor->uid, expectedUid);
        EXPECT_EQ(tensor->dims, expectedDims);
        EXPECT_EQ(tensor->strides, expectedStrides);
        EXPECT_EQ(tensor->data_type, expectedDataType);
        EXPECT_EQ(tensor->virtual_, expectedVirtual);
    }

    static void verifyReductionNode(const NodeT& node,
                                    DataType expectedComputeType,
                                    int64_t expectedXUid,
                                    int64_t expectedYUid,
                                    ReductionMode expectedReductionMode,
                                    bool expectedIsDeterministic = false)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::ReductionAttributes);

        auto* attrs = node.attributes.AsReductionAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->in_tensor_uid, expectedXUid);
        EXPECT_EQ(attrs->out_tensor_uid, expectedYUid);
        EXPECT_EQ(attrs->mode, expectedReductionMode);
        EXPECT_EQ(attrs->is_deterministic, expectedIsDeterministic);
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

TEST_F(TestGraphDescriptorReduction, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(K_REDUCTION_TENSOR_X_UID,
                                       toVec(K_REDUCTION_TENSOR_X_DIMS),
                                       toVec(K_REDUCTION_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_REDUCTION_TENSOR_Y_UID,
                                       toVec(K_REDUCTION_TENSOR_Y_DIMS),
                                       toVec(K_REDUCTION_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedReductionOp(xDesc.get(), yDesc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 2);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_REDUCTION_TENSOR_X_UID),
                 K_REDUCTION_TENSOR_X_UID,
                 toVec(K_REDUCTION_TENSOR_X_DIMS),
                 toVec(K_REDUCTION_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_REDUCTION_TENSOR_Y_UID),
                 K_REDUCTION_TENSOR_Y_UID,
                 toVec(K_REDUCTION_TENSOR_Y_DIMS),
                 toVec(K_REDUCTION_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    verifyReductionNode(*graphT->nodes[0],
                        DataType::FLOAT,
                        K_REDUCTION_TENSOR_X_UID,
                        K_REDUCTION_TENSOR_Y_UID,
                        ReductionMode::ADD);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorReduction, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(K_REDUCTION_TENSOR_X_UID,
                                       toVec(K_REDUCTION_TENSOR_X_DIMS),
                                       toVec(K_REDUCTION_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_REDUCTION_TENSOR_Y_UID,
                                       toVec(K_REDUCTION_TENSOR_Y_DIMS),
                                       toVec(K_REDUCTION_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedReductionOp(xDesc.get(), yDesc.get(), HIPDNN_DATA_HALF);

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

TEST_F(TestGraphDescriptorReduction, ReductionAttributesPreserved)
{
    auto xDesc = createFinalizedTensor(K_REDUCTION_TENSOR_X_UID,
                                       toVec(K_REDUCTION_TENSOR_X_DIMS),
                                       toVec(K_REDUCTION_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_REDUCTION_TENSOR_Y_UID,
                                       toVec(K_REDUCTION_TENSOR_Y_DIMS),
                                       toVec(K_REDUCTION_TENSOR_Y_STRIDES));

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<ReductionOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<ReductionOperationDescriptor>();

    HipdnnBackendDescriptor* xPtr = xDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&xPtr));
    HipdnnBackendDescriptor* yPtr = yDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_REDUCTION_YDESC,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&yPtr));

    auto mode = HIPDNN_REDUCE_TENSOR_ADD;
    opDesc->setAttribute(
        HIPDNN_ATTR_REDUCTION_OPERATOR, HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &mode);

    auto computeType = HIPDNN_DATA_FLOAT;
    opDesc->setAttribute(HIPDNN_ATTR_REDUCTION_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set operation name
    const std::string opName = "test_reduction";
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
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 2);

    // Verify tensors
    verifyTensor(findTensorByUid(*graphT, K_REDUCTION_TENSOR_X_UID),
                 K_REDUCTION_TENSOR_X_UID,
                 toVec(K_REDUCTION_TENSOR_X_DIMS),
                 toVec(K_REDUCTION_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_REDUCTION_TENSOR_Y_UID),
                 K_REDUCTION_TENSOR_Y_UID,
                 toVec(K_REDUCTION_TENSOR_Y_DIMS),
                 toVec(K_REDUCTION_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyReductionNode(*graphT->nodes[0],
                        DataType::FLOAT,
                        K_REDUCTION_TENSOR_X_UID,
                        K_REDUCTION_TENSOR_Y_UID,
                        ReductionMode::ADD);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_reduction");
}

TEST_F(TestGraphDescriptorReduction, OperationNamePreservedInSerialization)
{
    auto xDesc = createFinalizedTensor(K_REDUCTION_TENSOR_X_UID,
                                       toVec(K_REDUCTION_TENSOR_X_DIMS),
                                       toVec(K_REDUCTION_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_REDUCTION_TENSOR_Y_UID,
                                       toVec(K_REDUCTION_TENSOR_Y_DIMS),
                                       toVec(K_REDUCTION_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedReductionOp(
        xDesc.get(), yDesc.get(), HIPDNN_DATA_FLOAT, "test_reduction_name");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_reduction_name");
}

TEST_F(TestGraphDescriptorReduction, OperationNameRoundTripThroughLifting)
{
    auto xDesc = createFinalizedTensor(K_REDUCTION_TENSOR_X_UID,
                                       toVec(K_REDUCTION_TENSOR_X_DIMS),
                                       toVec(K_REDUCTION_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_REDUCTION_TENSOR_Y_UID,
                                       toVec(K_REDUCTION_TENSOR_Y_DIMS),
                                       toVec(K_REDUCTION_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedReductionOp(
        xDesc.get(), yDesc.get(), HIPDNN_DATA_FLOAT, "test_reduction_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_reduction_lifting");
}

TEST_F(TestGraphDescriptorReduction, IsDeterministicPreservedInSerialization)
{
    auto xDesc = createFinalizedTensor(K_REDUCTION_TENSOR_X_UID,
                                       toVec(K_REDUCTION_TENSOR_X_DIMS),
                                       toVec(K_REDUCTION_TENSOR_X_STRIDES));
    auto yDesc = createFinalizedTensor(K_REDUCTION_TENSOR_Y_UID,
                                       toVec(K_REDUCTION_TENSOR_Y_DIMS),
                                       toVec(K_REDUCTION_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedReductionOp(xDesc.get(), yDesc.get(), HIPDNN_DATA_FLOAT, "", true);

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

    auto reSerialized = liftedDesc->getSerializedGraph();
    auto graphT = UnPackGraph(reSerialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    auto* opNode = graphT->nodes[0]->attributes.AsReductionAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_TRUE(opNode->is_deterministic);
}

} // namespace
