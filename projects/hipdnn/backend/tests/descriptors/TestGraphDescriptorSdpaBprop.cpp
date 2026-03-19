// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/SdpaBpropOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SdpaBpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized SdpaBpropOperationDescriptor
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedSdpaBpropOp(HipdnnBackendDescriptor* qDesc,
                               HipdnnBackendDescriptor* kDesc,
                               HipdnnBackendDescriptor* vDesc,
                               HipdnnBackendDescriptor* oDesc,
                               HipdnnBackendDescriptor* doDesc,
                               HipdnnBackendDescriptor* statsDesc,
                               HipdnnBackendDescriptor* dqDesc,
                               HipdnnBackendDescriptor* dkDesc,
                               HipdnnBackendDescriptor* dvDesc,
                               HipdnnBackendDescriptor* scaleDesc,
                               HipdnnBackendDescriptor* attnMaskDesc,
                               HipdnnBackendDescriptor* seqLenQDesc,
                               HipdnnBackendDescriptor* seqLenKvDesc,
                               HipdnnBackendDescriptor* seedDesc,
                               HipdnnBackendDescriptor* offsetDesc,
                               HipdnnBackendDescriptor* dropoutMaskDesc,
                               HipdnnBackendDescriptor* dropoutScaleDesc,
                               HipdnnBackendDescriptor* dropoutScaleInvDesc,
                               HipdnnBackendDescriptor* dbiasDesc,
                               hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<SdpaBpropOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaBpropOperationDescriptor>();

    // Required input tensors
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&qDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&kDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&vDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&oDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&doDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&statsDesc));

    // Required output tensors
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dqDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dkDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dvDesc));

    // Optional tensors
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_ATTN_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&attnMaskDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&seqLenQDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_KV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&seqLenKvDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEED_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&seedDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_OFFSET_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&offsetDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dropoutMaskDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dropoutScaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_INV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dropoutScaleInvDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DBIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dbiasDesc));
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedSdpaBpropOpRequiredOnly(HipdnnBackendDescriptor* qDesc,
                                           HipdnnBackendDescriptor* kDesc,
                                           HipdnnBackendDescriptor* vDesc,
                                           HipdnnBackendDescriptor* oDesc,
                                           HipdnnBackendDescriptor* doDesc,
                                           HipdnnBackendDescriptor* statsDesc,
                                           HipdnnBackendDescriptor* dqDesc,
                                           HipdnnBackendDescriptor* dkDesc,
                                           HipdnnBackendDescriptor* dvDesc,
                                           hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<SdpaBpropOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaBpropOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&qDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&kDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&vDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&oDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&doDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&statsDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dqDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dkDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dvDesc));
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorSdpaBprop : public ::testing::Test
{
public:
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

TEST_F(TestGraphDescriptorSdpaBprop, BuildFromSingleOperation)
{
    auto qDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
    auto doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
    auto statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
    auto dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
    auto dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
    auto dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SCALE_UID);
    auto attnMaskDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);
    auto seqLenQDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID);
    auto seqLenKvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID);
    auto seedDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEED_UID);
    auto offsetDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_OFFSET_UID);
    auto dropoutMaskDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID);
    auto dropoutScaleDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID);
    auto dropoutScaleInvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID);
    auto dbiasDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DBIAS_UID);

    auto opWrapper = createFinalizedSdpaBpropOp(qDesc.get(),
                                                kDesc.get(),
                                                vDesc.get(),
                                                oDesc.get(),
                                                doDesc.get(),
                                                statsDesc.get(),
                                                dqDesc.get(),
                                                dkDesc.get(),
                                                dvDesc.get(),
                                                scaleDesc.get(),
                                                attnMaskDesc.get(),
                                                seqLenQDesc.get(),
                                                seqLenKvDesc.get(),
                                                seedDesc.get(),
                                                offsetDesc.get(),
                                                dropoutMaskDesc.get(),
                                                dropoutScaleDesc.get(),
                                                dropoutScaleInvDesc.get(),
                                                dbiasDesc.get());

    auto graphDesc = getDescriptor();
    setHandle();

    auto* opDescPtr = opWrapper.get();
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&opDescPtr));

    ASSERT_NO_THROW(graphDesc->finalize());
    ASSERT_TRUE(graphDesc->isFinalized());

    auto serialized = graphDesc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_NE(graphT, nullptr);
    ASSERT_EQ(graphT->nodes.size(), 1u);

    auto& node = graphT->nodes[0];
    ASSERT_EQ(node->attributes.type, NodeAttributes::SdpaBackwardAttributes);

    auto* attrs = node->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->q_tensor_uid, K_SDPA_BPROP_TENSOR_Q_UID);
    EXPECT_EQ(attrs->k_tensor_uid, K_SDPA_BPROP_TENSOR_K_UID);
    EXPECT_EQ(attrs->v_tensor_uid, K_SDPA_BPROP_TENSOR_V_UID);
    EXPECT_EQ(attrs->o_tensor_uid, K_SDPA_BPROP_TENSOR_O_UID);
    EXPECT_EQ(attrs->do_tensor_uid, K_SDPA_BPROP_TENSOR_DO_UID);
    EXPECT_EQ(attrs->stats_tensor_uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    EXPECT_EQ(attrs->dq_tensor_uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    EXPECT_EQ(attrs->dk_tensor_uid, K_SDPA_BPROP_TENSOR_DK_UID);
    EXPECT_EQ(attrs->dv_tensor_uid, K_SDPA_BPROP_TENSOR_DV_UID);

    // Check required tensor in the tensor list
    EXPECT_GE(graphT->tensors.size(), 9u); // at least the required tensors
}

TEST_F(TestGraphDescriptorSdpaBprop, BuildFromRequiredOnlyOperation)
{
    auto qDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
    auto doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
    auto statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
    auto dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
    auto dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
    auto dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));

    auto opWrapper = createFinalizedSdpaBpropOpRequiredOnly(qDesc.get(),
                                                            kDesc.get(),
                                                            vDesc.get(),
                                                            oDesc.get(),
                                                            doDesc.get(),
                                                            statsDesc.get(),
                                                            dqDesc.get(),
                                                            dkDesc.get(),
                                                            dvDesc.get());

    auto graphDesc = getDescriptor();
    setHandle();

    auto* opDescPtr = opWrapper.get();
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&opDescPtr));

    ASSERT_NO_THROW(graphDesc->finalize());
    ASSERT_TRUE(graphDesc->isFinalized());

    auto serialized = graphDesc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_NE(graphT, nullptr);
    ASSERT_EQ(graphT->nodes.size(), 1u);
    ASSERT_EQ(graphT->tensors.size(), 9u); // exactly the required tensors
}

} // namespace
