// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "descriptors/TopkOperationDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/topk_attributes_generated.h>
#include <hipdnn_test_sdk/constants/TopkConstants.hpp>
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

// Helper: create a finalized TopkOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedTopkOp(HipdnnBackendDescriptor* xDesc,
                          HipdnnBackendDescriptor* indicesDesc,
                          HipdnnBackendDescriptor* valuesDesc,
                          const std::string& name = "")
{
    auto wrapper = createDescriptor<TopkOperationDescriptor>();
    auto desc = wrapper->asDescriptor<TopkOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&indicesDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&valuesDesc));

    auto k = static_cast<int64_t>(4);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_K_EXT, HIPDNN_TYPE_INT64, 1, &k);
    auto largest = false;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_LARGEST_EXT, HIPDNN_TYPE_BOOLEAN, 1, &largest);

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

class TestGraphDescriptorTopk : public ::testing::Test
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

    static void verifyTopkNode(const NodeT& node,
                               int64_t expectedXUid,
                               int64_t expectedIndicesUid,
                               int64_t expectedValuesUid,
                               int64_t expectedK,
                               bool expectedLargest)
    {
        ASSERT_EQ(node.attributes.type, NodeAttributes::TopkAttributes);

        auto* attrs = node.attributes.AsTopkAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->x_tensor_uid, expectedXUid);
        EXPECT_EQ(attrs->indices_tensor_uid, expectedIndicesUid);
        EXPECT_EQ(attrs->values_tensor_uid, expectedValuesUid);
        EXPECT_EQ(attrs->k, expectedK);
        EXPECT_EQ(attrs->largest, expectedLargest);
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

TEST_F(TestGraphDescriptorTopk, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(K_TOPK_TENSOR_X_UID,
                                       toVec(K_TOPK_TENSOR_X_DIMS),
                                       toVec(K_TOPK_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto indicesDesc = createFinalizedTensor(K_TOPK_TENSOR_INDICES_UID,
                                             toVec(K_TOPK_TENSOR_INDICES_DIMS),
                                             toVec(K_TOPK_TENSOR_INDICES_STRIDES),
                                             HIPDNN_DATA_INT64);
    auto valuesDesc = createFinalizedTensor(K_TOPK_TENSOR_VALUES_UID,
                                            toVec(K_TOPK_TENSOR_VALUES_DIMS),
                                            toVec(K_TOPK_TENSOR_VALUES_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedTopkOp(xDesc.get(), indicesDesc.get(), valuesDesc.get());

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
    verifyTensor(findTensorByUid(*graphT, K_TOPK_TENSOR_X_UID),
                 K_TOPK_TENSOR_X_UID,
                 toVec(K_TOPK_TENSOR_X_DIMS),
                 toVec(K_TOPK_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_TOPK_TENSOR_INDICES_UID),
                 K_TOPK_TENSOR_INDICES_UID,
                 toVec(K_TOPK_TENSOR_INDICES_DIMS),
                 toVec(K_TOPK_TENSOR_INDICES_STRIDES),
                 DataType::INT64);
    verifyTensor(findTensorByUid(*graphT, K_TOPK_TENSOR_VALUES_UID),
                 K_TOPK_TENSOR_VALUES_UID,
                 toVec(K_TOPK_TENSOR_VALUES_DIMS),
                 toVec(K_TOPK_TENSOR_VALUES_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    verifyTopkNode(*graphT->nodes[0],
                   K_TOPK_TENSOR_X_UID,
                   K_TOPK_TENSOR_INDICES_UID,
                   K_TOPK_TENSOR_VALUES_UID,
                   static_cast<int64_t>(4),
                   false);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorTopk, TopkAttributesPreserved)
{
    auto xDesc = createFinalizedTensor(K_TOPK_TENSOR_X_UID,
                                       toVec(K_TOPK_TENSOR_X_DIMS),
                                       toVec(K_TOPK_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto indicesDesc = createFinalizedTensor(K_TOPK_TENSOR_INDICES_UID,
                                             toVec(K_TOPK_TENSOR_INDICES_DIMS),
                                             toVec(K_TOPK_TENSOR_INDICES_STRIDES),
                                             HIPDNN_DATA_INT64);
    auto valuesDesc = createFinalizedTensor(K_TOPK_TENSOR_VALUES_UID,
                                            toVec(K_TOPK_TENSOR_VALUES_DIMS),
                                            toVec(K_TOPK_TENSOR_VALUES_STRIDES),
                                            HIPDNN_DATA_FLOAT);

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<TopkOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<TopkOperationDescriptor>();

    HipdnnBackendDescriptor* xPtr = xDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&xPtr));
    HipdnnBackendDescriptor* indicesPtr = indicesDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&indicesPtr));
    HipdnnBackendDescriptor* valuesPtr = valuesDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&valuesPtr));

    auto k = static_cast<int64_t>(4);
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_K_EXT, HIPDNN_TYPE_INT64, 1, &k);

    auto largest = false;
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_LARGEST_EXT, HIPDNN_TYPE_BOOLEAN, 1, &largest);

    // Set operation name
    const std::string opName = "test_topk";
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
    verifyTensor(findTensorByUid(*graphT, K_TOPK_TENSOR_X_UID),
                 K_TOPK_TENSOR_X_UID,
                 toVec(K_TOPK_TENSOR_X_DIMS),
                 toVec(K_TOPK_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_TOPK_TENSOR_INDICES_UID),
                 K_TOPK_TENSOR_INDICES_UID,
                 toVec(K_TOPK_TENSOR_INDICES_DIMS),
                 toVec(K_TOPK_TENSOR_INDICES_STRIDES),
                 DataType::INT64);
    verifyTensor(findTensorByUid(*graphT, K_TOPK_TENSOR_VALUES_UID),
                 K_TOPK_TENSOR_VALUES_UID,
                 toVec(K_TOPK_TENSOR_VALUES_DIMS),
                 toVec(K_TOPK_TENSOR_VALUES_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyTopkNode(*graphT->nodes[0],
                   K_TOPK_TENSOR_X_UID,
                   K_TOPK_TENSOR_INDICES_UID,
                   K_TOPK_TENSOR_VALUES_UID,
                   static_cast<int64_t>(4),
                   false);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_topk");
}

TEST_F(TestGraphDescriptorTopk, OperationNamePreservedInSerialization)
{
    auto xDesc = createFinalizedTensor(K_TOPK_TENSOR_X_UID,
                                       toVec(K_TOPK_TENSOR_X_DIMS),
                                       toVec(K_TOPK_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto indicesDesc = createFinalizedTensor(K_TOPK_TENSOR_INDICES_UID,
                                             toVec(K_TOPK_TENSOR_INDICES_DIMS),
                                             toVec(K_TOPK_TENSOR_INDICES_STRIDES),
                                             HIPDNN_DATA_INT64);
    auto valuesDesc = createFinalizedTensor(K_TOPK_TENSOR_VALUES_UID,
                                            toVec(K_TOPK_TENSOR_VALUES_DIMS),
                                            toVec(K_TOPK_TENSOR_VALUES_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto opDesc
        = createFinalizedTopkOp(xDesc.get(), indicesDesc.get(), valuesDesc.get(), "test_topk_name");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_topk_name");
}

TEST_F(TestGraphDescriptorTopk, OperationNameRoundTripThroughLifting)
{
    auto xDesc = createFinalizedTensor(K_TOPK_TENSOR_X_UID,
                                       toVec(K_TOPK_TENSOR_X_DIMS),
                                       toVec(K_TOPK_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto indicesDesc = createFinalizedTensor(K_TOPK_TENSOR_INDICES_UID,
                                             toVec(K_TOPK_TENSOR_INDICES_DIMS),
                                             toVec(K_TOPK_TENSOR_INDICES_STRIDES),
                                             HIPDNN_DATA_INT64);
    auto valuesDesc = createFinalizedTensor(K_TOPK_TENSOR_VALUES_UID,
                                            toVec(K_TOPK_TENSOR_VALUES_DIMS),
                                            toVec(K_TOPK_TENSOR_VALUES_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedTopkOp(
        xDesc.get(), indicesDesc.get(), valuesDesc.get(), "test_topk_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_topk_lifting");
}

} // namespace
