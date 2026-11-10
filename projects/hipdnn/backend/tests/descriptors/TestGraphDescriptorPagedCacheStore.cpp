// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/PagedCacheStoreOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/paged_cache_store_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/PagedCacheStoreConstants.hpp>
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

// Helper: create a finalized PagedCacheStoreOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedPagedCacheStoreOp(HipdnnBackendDescriptor* xDesc,
                                     HipdnnBackendDescriptor* slotMappingDesc,
                                     HipdnnBackendDescriptor* containerDesc,
                                     HipdnnBackendDescriptor* scaleDesc,
                                     const std::string& name = "")
{
    auto wrapper = createDescriptor<PagedCacheStoreOperationDescriptor>();
    auto desc = wrapper->asDescriptor<PagedCacheStoreOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&slotMappingDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&containerDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));

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

class TestGraphDescriptorPagedCacheStore : public ::testing::Test
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

TEST_F(TestGraphDescriptorPagedCacheStore, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_X_UID,
                                       toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS),
                                       toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto slotMappingDesc
        = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID,
                                toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS),
                                toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES),
                                HIPDNN_DATA_INT32);
    auto containerDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID,
                                               toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS),
                                               toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES),
                                               HIPDNN_DATA_FLOAT);
    auto scaleDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_SCALE_UID,
                                           toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS),
                                           toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedPagedCacheStoreOp(
        xDesc.get(), slotMappingDesc.get(), containerDesc.get(), scaleDesc.get());

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
    verifyTensor(findTensorByUid(*graphT, K_PAGED_CACHE_STORE_TENSOR_X_UID),
                 K_PAGED_CACHE_STORE_TENSOR_X_UID,
                 toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS),
                 toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID),
                 K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID,
                 toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS),
                 toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID),
                 K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID,
                 toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS),
                 toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID),
                 K_PAGED_CACHE_STORE_TENSOR_SCALE_UID,
                 toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS),
                 toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::PagedCacheStoreAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsPagedCacheStoreAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->x_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    EXPECT_EQ(attrs->slot_mapping_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    EXPECT_EQ(attrs->container_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorPagedCacheStore, OperationNamePreservedInSerialization)
{
    auto xDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_X_UID,
                                       toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS),
                                       toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto slotMappingDesc
        = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID,
                                toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS),
                                toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES),
                                HIPDNN_DATA_INT32);
    auto containerDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID,
                                               toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS),
                                               toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES),
                                               HIPDNN_DATA_FLOAT);
    auto scaleDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_SCALE_UID,
                                           toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS),
                                           toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedPagedCacheStoreOp(xDesc.get(),
                                                   slotMappingDesc.get(),
                                                   containerDesc.get(),
                                                   scaleDesc.get(),
                                                   "test_pagedcachestore_name");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_pagedcachestore_name");
}

TEST_F(TestGraphDescriptorPagedCacheStore, OperationNameRoundTripThroughLifting)
{
    auto xDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_X_UID,
                                       toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS),
                                       toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto slotMappingDesc
        = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID,
                                toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS),
                                toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES),
                                HIPDNN_DATA_INT32);
    auto containerDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID,
                                               toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS),
                                               toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES),
                                               HIPDNN_DATA_FLOAT);
    auto scaleDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_SCALE_UID,
                                           toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS),
                                           toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_STRIDES),
                                           HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedPagedCacheStoreOp(xDesc.get(),
                                                   slotMappingDesc.get(),
                                                   containerDesc.get(),
                                                   scaleDesc.get(),
                                                   "test_pagedcachestore_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_pagedcachestore_lifting");
}

} // namespace
