// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/ResampleBwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ResampleBwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <optional>
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

// Helper: create a finalized ResampleBwdOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedResampleBwdOp(HipdnnBackendDescriptor* dyDesc,
                                 HipdnnBackendDescriptor* dxDesc,
                                 HipdnnBackendDescriptor* indexDesc = nullptr,
                                 hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                 const std::string& name = "",
                                 std::optional<bool> generateIndex = std::nullopt)
{
    auto wrapper = createDescriptor<ResampleBwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<ResampleBwdOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dyDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dxDesc));
    if(indexDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&indexDesc));
    }

    auto prePadding = toVec(K_PRE_PADDING);
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(prePadding.size()),
                       prePadding.data());

    auto postPadding = toVec(K_POST_PADDING);
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_POST_PADDINGS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(postPadding.size()),
                       postPadding.data());

    auto stride = toVec(K_STRIDE);
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(stride.size()),
                       stride.data());

    auto window = toVec(K_WINDOW);
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(window.size()),
                       window.data());

    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode);

    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
    desc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);

    desc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(!name.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(name.size()),
                           name.data());
    }

    if(generateIndex.has_value())
    {
        bool val = generateIndex.value();
        desc->setAttribute(HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT, HIPDNN_TYPE_BOOLEAN, 1, &val);
    }

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorResampleBwd : public ::testing::Test
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

    static void verifyResampleBwdNode(const NodeT& node,
                                      DataType expectedComputeType,
                                      int64_t expectedDyUid,
                                      int64_t expectedDxUid,
                                      int64_t expectedIndexUid,
                                      const std::vector<int64_t>& expectedPrePadding,
                                      const std::vector<int64_t>& expectedPostPadding,
                                      const std::vector<int64_t>& expectedStride,
                                      const std::vector<int64_t>& expectedWindow,
                                      ResampleMode expectedResampleMode,
                                      PaddingMode expectedPaddingMode)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::ResampleBwdAttributes);

        auto* attrs = node.attributes.AsResampleBwdAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->dy_tensor_uid, expectedDyUid);
        EXPECT_EQ(attrs->dx_tensor_uid, expectedDxUid);
        EXPECT_EQ(attrs->index_tensor_uid, expectedIndexUid);
        EXPECT_EQ(attrs->pre_padding, expectedPrePadding);
        EXPECT_EQ(attrs->post_padding, expectedPostPadding);
        EXPECT_EQ(attrs->stride, expectedStride);
        EXPECT_EQ(attrs->window, expectedWindow);
        EXPECT_EQ(attrs->resample_mode, expectedResampleMode);
        EXPECT_EQ(attrs->padding_mode, expectedPaddingMode);
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

TEST_F(TestGraphDescriptorResampleBwd, BuildFromSingleOperation)
{
    auto dyDesc = createFinalizedTensor(
        K_TENSOR_DY_UID, toVec(K_TENSOR_DY_DIMS), toVec(K_TENSOR_DY_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_TENSOR_DX_UID, toVec(K_TENSOR_DX_DIMS), toVec(K_TENSOR_DX_STRIDES));
    auto indexDesc = createFinalizedTensor(
        K_TENSOR_INDEX_UID, toVec(K_TENSOR_INDEX_DIMS), toVec(K_TENSOR_INDEX_STRIDES));
    auto opDesc = createFinalizedResampleBwdOp(dyDesc.get(), dxDesc.get(), indexDesc.get());

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
    verifyTensor(findTensorByUid(*graphT, K_TENSOR_DY_UID),
                 K_TENSOR_DY_UID,
                 toVec(K_TENSOR_DY_DIMS),
                 toVec(K_TENSOR_DY_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_TENSOR_DX_UID),
                 K_TENSOR_DX_UID,
                 toVec(K_TENSOR_DX_DIMS),
                 toVec(K_TENSOR_DX_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_TENSOR_INDEX_UID),
                 K_TENSOR_INDEX_UID,
                 toVec(K_TENSOR_INDEX_DIMS),
                 toVec(K_TENSOR_INDEX_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    verifyResampleBwdNode(*graphT->nodes[0],
                          DataType::FLOAT,
                          K_TENSOR_DY_UID,
                          K_TENSOR_DX_UID,
                          K_TENSOR_INDEX_UID,
                          toVec(K_PRE_PADDING),
                          toVec(K_POST_PADDING),
                          toVec(K_STRIDE),
                          toVec(K_WINDOW),
                          ResampleMode::MAXPOOL,
                          PaddingMode::ZERO_PAD);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorResampleBwd, ComputeDataTypePreserved)
{
    auto dyDesc = createFinalizedTensor(
        K_TENSOR_DY_UID, toVec(K_TENSOR_DY_DIMS), toVec(K_TENSOR_DY_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_TENSOR_DX_UID, toVec(K_TENSOR_DX_DIMS), toVec(K_TENSOR_DX_STRIDES));
    auto indexDesc = createFinalizedTensor(
        K_TENSOR_INDEX_UID, toVec(K_TENSOR_INDEX_DIMS), toVec(K_TENSOR_INDEX_STRIDES));
    auto opDesc = createFinalizedResampleBwdOp(
        dyDesc.get(), dxDesc.get(), indexDesc.get(), HIPDNN_DATA_HALF);

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

TEST_F(TestGraphDescriptorResampleBwd, ResampleBwdAttributesPreserved)
{
    auto dyDesc = createFinalizedTensor(
        K_TENSOR_DY_UID, toVec(K_TENSOR_DY_DIMS), toVec(K_TENSOR_DY_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_TENSOR_DX_UID, toVec(K_TENSOR_DX_DIMS), toVec(K_TENSOR_DX_STRIDES));
    auto indexDesc = createFinalizedTensor(
        K_TENSOR_INDEX_UID, toVec(K_TENSOR_INDEX_DIMS), toVec(K_TENSOR_INDEX_STRIDES));

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<ResampleBwdOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<ResampleBwdOperationDescriptor>();

    HipdnnBackendDescriptor* dyPtr = dyDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DY,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&dyPtr));
    HipdnnBackendDescriptor* dxPtr = dxDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DX,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&dxPtr));
    HipdnnBackendDescriptor* indexPtr = indexDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_INDEX,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&indexPtr));

    const std::vector<int64_t> kCustomPrePadding = {1, 1};
    opDesc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, HIPDNN_TYPE_INT64, 2, kCustomPrePadding.data());

    const std::vector<int64_t> kCustomPostPadding = {1, 1};
    opDesc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, HIPDNN_TYPE_INT64, 2, kCustomPostPadding.data());

    const std::vector<int64_t> kCustomStride = {2, 2};
    opDesc->setAttribute(HIPDNN_ATTR_RESAMPLE_STRIDES, HIPDNN_TYPE_INT64, 2, kCustomStride.data());

    const std::vector<int64_t> kCustomWindow = {3, 3};
    opDesc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, HIPDNN_TYPE_INT64, 2, kCustomWindow.data());

    auto resampleMode = HIPDNN_RESAMPLE_MAXPOOL;
    opDesc->setAttribute(HIPDNN_ATTR_RESAMPLE_MODE, HIPDNN_TYPE_RESAMPLE_MODE, 1, &resampleMode);

    auto paddingMode = HIPDNN_PADDING_ZERO_PAD;
    opDesc->setAttribute(
        HIPDNN_ATTR_RESAMPLE_PADDING_MODE, HIPDNN_TYPE_PADDING_MODE, 1, &paddingMode);

    auto computeType = HIPDNN_DATA_FLOAT;
    opDesc->setAttribute(HIPDNN_ATTR_RESAMPLE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set operation name
    const std::string opName = "test_resamplebwd";
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
    verifyTensor(findTensorByUid(*graphT, K_TENSOR_DY_UID),
                 K_TENSOR_DY_UID,
                 toVec(K_TENSOR_DY_DIMS),
                 toVec(K_TENSOR_DY_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_TENSOR_DX_UID),
                 K_TENSOR_DX_UID,
                 toVec(K_TENSOR_DX_DIMS),
                 toVec(K_TENSOR_DX_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_TENSOR_INDEX_UID),
                 K_TENSOR_INDEX_UID,
                 toVec(K_TENSOR_INDEX_DIMS),
                 toVec(K_TENSOR_INDEX_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyResampleBwdNode(*graphT->nodes[0],
                          DataType::FLOAT,
                          K_TENSOR_DY_UID,
                          K_TENSOR_DX_UID,
                          K_TENSOR_INDEX_UID,
                          kCustomPrePadding,
                          kCustomPostPadding,
                          kCustomStride,
                          kCustomWindow,
                          ResampleMode::MAXPOOL,
                          PaddingMode::ZERO_PAD);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_resamplebwd");
}

TEST_F(TestGraphDescriptorResampleBwd, OperationNamePreservedInSerialization)
{
    auto dyDesc = createFinalizedTensor(
        K_TENSOR_DY_UID, toVec(K_TENSOR_DY_DIMS), toVec(K_TENSOR_DY_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_TENSOR_DX_UID, toVec(K_TENSOR_DX_DIMS), toVec(K_TENSOR_DX_STRIDES));
    auto indexDesc = createFinalizedTensor(
        K_TENSOR_INDEX_UID, toVec(K_TENSOR_INDEX_DIMS), toVec(K_TENSOR_INDEX_STRIDES));
    auto opDesc = createFinalizedResampleBwdOp(
        dyDesc.get(), dxDesc.get(), indexDesc.get(), HIPDNN_DATA_FLOAT, "test_resamplebwd_name");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_resamplebwd_name");
}

TEST_F(TestGraphDescriptorResampleBwd, OperationNameRoundTripThroughLifting)
{
    auto dyDesc = createFinalizedTensor(
        K_TENSOR_DY_UID, toVec(K_TENSOR_DY_DIMS), toVec(K_TENSOR_DY_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_TENSOR_DX_UID, toVec(K_TENSOR_DX_DIMS), toVec(K_TENSOR_DX_STRIDES));
    auto indexDesc = createFinalizedTensor(
        K_TENSOR_INDEX_UID, toVec(K_TENSOR_INDEX_DIMS), toVec(K_TENSOR_INDEX_STRIDES));
    auto opDesc = createFinalizedResampleBwdOp(
        dyDesc.get(), dxDesc.get(), indexDesc.get(), HIPDNN_DATA_FLOAT, "test_resamplebwd_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_resamplebwd_lifting");
}

TEST_F(TestGraphDescriptorResampleBwd, BuildFromOperationWithoutOptionalTensors)
{
    auto dyDesc = createFinalizedTensor(
        K_TENSOR_DY_UID, toVec(K_TENSOR_DY_DIMS), toVec(K_TENSOR_DY_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_TENSOR_DX_UID, toVec(K_TENSOR_DX_DIMS), toVec(K_TENSOR_DX_STRIDES));

    // Create op WITHOUT index tensor
    auto opDesc = createFinalizedResampleBwdOp(dyDesc.get(), dxDesc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 2);

    auto* attrs = graphT->nodes[0]->attributes.AsResampleBwdAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->dy_tensor_uid, K_TENSOR_DY_UID);
    EXPECT_EQ(attrs->dx_tensor_uid, K_TENSOR_DX_UID);
    EXPECT_FALSE(attrs->index_tensor_uid.has_value());
}

TEST_F(TestGraphDescriptorResampleBwd, GenerateIndexPreservedInSerialization)
{
    auto dyDesc = createFinalizedTensor(
        K_TENSOR_DY_UID, toVec(K_TENSOR_DY_DIMS), toVec(K_TENSOR_DY_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_TENSOR_DX_UID, toVec(K_TENSOR_DX_DIMS), toVec(K_TENSOR_DX_STRIDES));
    auto indexDesc = createFinalizedTensor(
        K_TENSOR_INDEX_UID, toVec(K_TENSOR_INDEX_DIMS), toVec(K_TENSOR_INDEX_STRIDES));
    auto opDesc = createFinalizedResampleBwdOp(
        dyDesc.get(), dxDesc.get(), indexDesc.get(), HIPDNN_DATA_FLOAT, "", true);

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

    auto* attrs = graphT->nodes[0]->attributes.AsResampleBwdAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_TRUE(attrs->generate_index.has_value());
    EXPECT_EQ(attrs->generate_index.value(), true);
}

} // namespace
