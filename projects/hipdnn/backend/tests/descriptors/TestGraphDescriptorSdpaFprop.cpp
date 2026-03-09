// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/SdpaFpropOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized SdpaFpropOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedSdpaFpropOp(HipdnnBackendDescriptor* qDesc,
                               HipdnnBackendDescriptor* kDesc,
                               HipdnnBackendDescriptor* vDesc,
                               HipdnnBackendDescriptor* oDesc,
                               HipdnnBackendDescriptor* attnMaskDesc,
                               HipdnnBackendDescriptor* scaleDesc,
                               HipdnnBackendDescriptor* seqLenQDesc,
                               HipdnnBackendDescriptor* seqLenKvDesc,
                               HipdnnBackendDescriptor* seedDesc,
                               HipdnnBackendDescriptor* offsetDesc,
                               HipdnnBackendDescriptor* dropoutMaskDesc,
                               HipdnnBackendDescriptor* dropoutScaleDesc,
                               HipdnnBackendDescriptor* pageTableKDesc,
                               HipdnnBackendDescriptor* pageTableVDesc,
                               HipdnnBackendDescriptor* blockMaskDesc,
                               HipdnnBackendDescriptor* sinkTokenDesc,
                               HipdnnBackendDescriptor* descaleQDesc,
                               HipdnnBackendDescriptor* descaleKDesc,
                               HipdnnBackendDescriptor* descaleVDesc,
                               HipdnnBackendDescriptor* descaleSDesc,
                               HipdnnBackendDescriptor* scaleSDesc,
                               HipdnnBackendDescriptor* scaleODesc,
                               HipdnnBackendDescriptor* statsDesc,
                               HipdnnBackendDescriptor* maxDesc,
                               HipdnnBackendDescriptor* sumExpDesc,
                               HipdnnBackendDescriptor* rngDumpDesc,
                               HipdnnBackendDescriptor* amaxSDesc,
                               HipdnnBackendDescriptor* amaxODesc,
                               hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<SdpaFpropOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaFpropOperationDescriptor>();

    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &qDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &kDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &vDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &oDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &attnMaskDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &seqLenQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &seqLenKvDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &seedDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &offsetDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dropoutMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dropoutScaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &pageTableKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &pageTableVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &blockMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &sinkTokenDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &descaleQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &descaleKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &descaleVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &descaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleODesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &statsDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &maxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &sumExpDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &rngDumpDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &amaxSDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &amaxODesc);
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorSdpaFprop : public ::testing::Test
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
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
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

TEST_F(TestGraphDescriptorSdpaFprop, BuildFromSingleOperation)
{
    auto qDesc = createFinalizedTensor(40, {2, 4, 128, 64}, {32768, 8192, 64, 1});
    auto kDesc = createFinalizedTensor(41, {2, 4, 128, 64}, {32768, 8192, 64, 1});
    auto vDesc = createFinalizedTensor(42, {2, 4, 128, 64}, {32768, 8192, 64, 1});
    auto oDesc = createFinalizedTensor(43, {2, 4, 128, 64}, {32768, 8192, 64, 1});
    auto attnMaskDesc = createFinalizedTensor(5);
    auto scaleDesc = createFinalizedTensor(6);
    auto seqLenQDesc = createFinalizedTensor(7);
    auto seqLenKvDesc = createFinalizedTensor(8);
    auto seedDesc = createFinalizedTensor(9);
    auto offsetDesc = createFinalizedTensor(10);
    auto dropoutMaskDesc = createFinalizedTensor(11);
    auto dropoutScaleDesc = createFinalizedTensor(12);
    auto pageTableKDesc = createFinalizedTensor(13);
    auto pageTableVDesc = createFinalizedTensor(14);
    auto blockMaskDesc = createFinalizedTensor(15);
    auto sinkTokenDesc = createFinalizedTensor(16);
    auto descaleQDesc = createFinalizedTensor(17);
    auto descaleKDesc = createFinalizedTensor(18);
    auto descaleVDesc = createFinalizedTensor(19);
    auto descaleSDesc = createFinalizedTensor(20);
    auto scaleSDesc = createFinalizedTensor(21);
    auto scaleODesc = createFinalizedTensor(22);
    auto statsDesc = createFinalizedTensor(23);
    auto maxDesc = createFinalizedTensor(24);
    auto sumExpDesc = createFinalizedTensor(25);
    auto rngDumpDesc = createFinalizedTensor(26);
    auto amaxSDesc = createFinalizedTensor(27);
    auto amaxODesc = createFinalizedTensor(28);
    auto opDesc = createFinalizedSdpaFpropOp(qDesc.get(),
                                             kDesc.get(),
                                             vDesc.get(),
                                             oDesc.get(),
                                             attnMaskDesc.get(),
                                             scaleDesc.get(),
                                             seqLenQDesc.get(),
                                             seqLenKvDesc.get(),
                                             seedDesc.get(),
                                             offsetDesc.get(),
                                             dropoutMaskDesc.get(),
                                             dropoutScaleDesc.get(),
                                             pageTableKDesc.get(),
                                             pageTableVDesc.get(),
                                             blockMaskDesc.get(),
                                             sinkTokenDesc.get(),
                                             descaleQDesc.get(),
                                             descaleKDesc.get(),
                                             descaleVDesc.get(),
                                             descaleSDesc.get(),
                                             scaleSDesc.get(),
                                             scaleODesc.get(),
                                             statsDesc.get(),
                                             maxDesc.get(),
                                             sumExpDesc.get(),
                                             rngDumpDesc.get(),
                                             amaxSDesc.get(),
                                             amaxODesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    // Verify the built graph
    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 28);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::SdpaAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsSdpaAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->q_tensor_uid, 40);
    EXPECT_EQ(attrs->k_tensor_uid, 41);
    EXPECT_EQ(attrs->v_tensor_uid, 42);
    EXPECT_EQ(attrs->o_tensor_uid, 43);
    EXPECT_EQ(attrs->attn_mask_tensor_uid, 5);
    EXPECT_EQ(attrs->scale_tensor_uid, 6);
    EXPECT_EQ(attrs->seq_len_q_tensor_uid, 7);
    EXPECT_EQ(attrs->seq_len_kv_tensor_uid, 8);
    EXPECT_EQ(attrs->seed_tensor_uid, 9);
    EXPECT_EQ(attrs->offset_tensor_uid, 10);
    EXPECT_EQ(attrs->dropout_mask_tensor_uid, 11);
    EXPECT_EQ(attrs->dropout_scale_tensor_uid, 12);
    EXPECT_EQ(attrs->page_table_k_tensor_uid, 13);
    EXPECT_EQ(attrs->page_table_v_tensor_uid, 14);
    EXPECT_EQ(attrs->block_mask_tensor_uid, 15);
    EXPECT_EQ(attrs->sink_token_tensor_uid, 16);
    EXPECT_EQ(attrs->descale_q_tensor_uid, 17);
    EXPECT_EQ(attrs->descale_k_tensor_uid, 18);
    EXPECT_EQ(attrs->descale_v_tensor_uid, 19);
    EXPECT_EQ(attrs->descale_s_tensor_uid, 20);
    EXPECT_EQ(attrs->scale_s_tensor_uid, 21);
    EXPECT_EQ(attrs->scale_o_tensor_uid, 22);
    EXPECT_EQ(attrs->stats_tensor_uid, 23);
    EXPECT_EQ(attrs->max_tensor_uid, 24);
    EXPECT_EQ(attrs->sum_exp_tensor_uid, 25);
    EXPECT_EQ(attrs->rng_dump_tensor_uid, 26);
    EXPECT_EQ(attrs->amax_s_tensor_uid, 27);
    EXPECT_EQ(attrs->amax_o_tensor_uid, 28);
}

TEST_F(TestGraphDescriptorSdpaFprop, ComputeDataTypePreserved)
{
    auto qDesc = createFinalizedTensor(40, {2, 4, 128, 64}, {32768, 8192, 64, 1});
    auto kDesc = createFinalizedTensor(41, {2, 4, 128, 64}, {32768, 8192, 64, 1});
    auto vDesc = createFinalizedTensor(42, {2, 4, 128, 64}, {32768, 8192, 64, 1});
    auto oDesc = createFinalizedTensor(43, {2, 4, 128, 64}, {32768, 8192, 64, 1});
    auto attnMaskDesc = createFinalizedTensor(5);
    auto scaleDesc = createFinalizedTensor(6);
    auto seqLenQDesc = createFinalizedTensor(7);
    auto seqLenKvDesc = createFinalizedTensor(8);
    auto seedDesc = createFinalizedTensor(9);
    auto offsetDesc = createFinalizedTensor(10);
    auto dropoutMaskDesc = createFinalizedTensor(11);
    auto dropoutScaleDesc = createFinalizedTensor(12);
    auto pageTableKDesc = createFinalizedTensor(13);
    auto pageTableVDesc = createFinalizedTensor(14);
    auto blockMaskDesc = createFinalizedTensor(15);
    auto sinkTokenDesc = createFinalizedTensor(16);
    auto descaleQDesc = createFinalizedTensor(17);
    auto descaleKDesc = createFinalizedTensor(18);
    auto descaleVDesc = createFinalizedTensor(19);
    auto descaleSDesc = createFinalizedTensor(20);
    auto scaleSDesc = createFinalizedTensor(21);
    auto scaleODesc = createFinalizedTensor(22);
    auto statsDesc = createFinalizedTensor(23);
    auto maxDesc = createFinalizedTensor(24);
    auto sumExpDesc = createFinalizedTensor(25);
    auto rngDumpDesc = createFinalizedTensor(26);
    auto amaxSDesc = createFinalizedTensor(27);
    auto amaxODesc = createFinalizedTensor(28);
    auto opDesc = createFinalizedSdpaFpropOp(qDesc.get(),
                                             kDesc.get(),
                                             vDesc.get(),
                                             oDesc.get(),
                                             attnMaskDesc.get(),
                                             scaleDesc.get(),
                                             seqLenQDesc.get(),
                                             seqLenKvDesc.get(),
                                             seedDesc.get(),
                                             offsetDesc.get(),
                                             dropoutMaskDesc.get(),
                                             dropoutScaleDesc.get(),
                                             pageTableKDesc.get(),
                                             pageTableVDesc.get(),
                                             blockMaskDesc.get(),
                                             sinkTokenDesc.get(),
                                             descaleQDesc.get(),
                                             descaleKDesc.get(),
                                             descaleVDesc.get(),
                                             descaleSDesc.get(),
                                             scaleSDesc.get(),
                                             scaleODesc.get(),
                                             statsDesc.get(),
                                             maxDesc.get(),
                                             sumExpDesc.get(),
                                             rngDumpDesc.get(),
                                             amaxSDesc.get(),
                                             amaxODesc.get(),
                                             HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = GetGraph(serialized.ptr)->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

} // namespace
