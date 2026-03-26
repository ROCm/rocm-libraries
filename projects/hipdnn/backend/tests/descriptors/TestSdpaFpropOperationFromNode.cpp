// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/SdpaFpropOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_data_sdk::data_objects;

// =============================================================================
// SdpaFpropOperationDescriptor::fromNode() Tests
// =============================================================================

class TestSdpaFpropOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT qAttrs;
        qAttrs.uid = 60;
        qAttrs.data_type = DataType::FLOAT;
        qAttrs.dims = {1, 8, 128, 64};
        qAttrs.strides = {65536, 8192, 64, 1};

        _tensorMap[60] = TensorDescriptor::fromFlatBuffer(qAttrs);
        TensorAttributesT kAttrs;
        kAttrs.uid = 61;
        kAttrs.data_type = DataType::FLOAT;
        kAttrs.dims = {1, 8, 128, 64};
        kAttrs.strides = {65536, 8192, 64, 1};

        _tensorMap[61] = TensorDescriptor::fromFlatBuffer(kAttrs);
        TensorAttributesT vAttrs;
        vAttrs.uid = 62;
        vAttrs.data_type = DataType::FLOAT;
        vAttrs.dims = {1, 8, 128, 64};
        vAttrs.strides = {65536, 8192, 64, 1};

        _tensorMap[62] = TensorDescriptor::fromFlatBuffer(vAttrs);
        TensorAttributesT oAttrs;
        oAttrs.uid = 63;
        oAttrs.data_type = DataType::FLOAT;
        oAttrs.dims = {1, 8, 128, 64};
        oAttrs.strides = {65536, 8192, 64, 1};

        _tensorMap[63] = TensorDescriptor::fromFlatBuffer(oAttrs);
        TensorAttributesT attnMaskAttrs;
        attnMaskAttrs.uid = 64;
        attnMaskAttrs.data_type = DataType::FLOAT;
        attnMaskAttrs.dims = {1, 1, 128, 128};
        attnMaskAttrs.strides = {16384, 16384, 128, 1};

        _tensorMap[64] = TensorDescriptor::fromFlatBuffer(attnMaskAttrs);
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = 65;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = {1, 1, 1, 1};
        scaleAttrs.strides = {1, 1, 1, 1};

        _tensorMap[65] = TensorDescriptor::fromFlatBuffer(scaleAttrs);
        TensorAttributesT seqLenQAttrs;
        seqLenQAttrs.uid = 66;
        seqLenQAttrs.data_type = DataType::FLOAT;
        seqLenQAttrs.dims = {1};
        seqLenQAttrs.strides = {1};

        _tensorMap[66] = TensorDescriptor::fromFlatBuffer(seqLenQAttrs);
        TensorAttributesT seqLenKvAttrs;
        seqLenKvAttrs.uid = 67;
        seqLenKvAttrs.data_type = DataType::FLOAT;
        seqLenKvAttrs.dims = {1};
        seqLenKvAttrs.strides = {1};

        _tensorMap[67] = TensorDescriptor::fromFlatBuffer(seqLenKvAttrs);
        TensorAttributesT seedAttrs;
        seedAttrs.uid = 68;
        seedAttrs.data_type = DataType::FLOAT;
        seedAttrs.dims = {1, 1, 1, 1};
        seedAttrs.strides = {1, 1, 1, 1};

        _tensorMap[68] = TensorDescriptor::fromFlatBuffer(seedAttrs);
        TensorAttributesT offsetAttrs;
        offsetAttrs.uid = 69;
        offsetAttrs.data_type = DataType::FLOAT;
        offsetAttrs.dims = {1, 1, 1, 1};
        offsetAttrs.strides = {1, 1, 1, 1};

        _tensorMap[69] = TensorDescriptor::fromFlatBuffer(offsetAttrs);
        TensorAttributesT dropoutMaskAttrs;
        dropoutMaskAttrs.uid = 70;
        dropoutMaskAttrs.data_type = DataType::FLOAT;
        dropoutMaskAttrs.dims = {1, 8, 128, 128};
        dropoutMaskAttrs.strides = {131072, 16384, 128, 1};

        _tensorMap[70] = TensorDescriptor::fromFlatBuffer(dropoutMaskAttrs);
        TensorAttributesT dropoutScaleAttrs;
        dropoutScaleAttrs.uid = 71;
        dropoutScaleAttrs.data_type = DataType::FLOAT;
        dropoutScaleAttrs.dims = {1, 1, 1, 1};
        dropoutScaleAttrs.strides = {1, 1, 1, 1};

        _tensorMap[71] = TensorDescriptor::fromFlatBuffer(dropoutScaleAttrs);
        TensorAttributesT pageTableKAttrs;
        pageTableKAttrs.uid = 72;
        pageTableKAttrs.data_type = DataType::FLOAT;
        pageTableKAttrs.dims = {1};
        pageTableKAttrs.strides = {1};

        _tensorMap[72] = TensorDescriptor::fromFlatBuffer(pageTableKAttrs);
        TensorAttributesT pageTableVAttrs;
        pageTableVAttrs.uid = 73;
        pageTableVAttrs.data_type = DataType::FLOAT;
        pageTableVAttrs.dims = {1};
        pageTableVAttrs.strides = {1};

        _tensorMap[73] = TensorDescriptor::fromFlatBuffer(pageTableVAttrs);
        TensorAttributesT blockMaskAttrs;
        blockMaskAttrs.uid = 74;
        blockMaskAttrs.data_type = DataType::FLOAT;
        blockMaskAttrs.dims = {1};
        blockMaskAttrs.strides = {1};

        _tensorMap[74] = TensorDescriptor::fromFlatBuffer(blockMaskAttrs);
        TensorAttributesT sinkTokenAttrs;
        sinkTokenAttrs.uid = 75;
        sinkTokenAttrs.data_type = DataType::FLOAT;
        sinkTokenAttrs.dims = {1};
        sinkTokenAttrs.strides = {1};

        _tensorMap[75] = TensorDescriptor::fromFlatBuffer(sinkTokenAttrs);
        TensorAttributesT descaleQAttrs;
        descaleQAttrs.uid = 76;
        descaleQAttrs.data_type = DataType::FLOAT;
        descaleQAttrs.dims = {1, 1, 1, 1};
        descaleQAttrs.strides = {1, 1, 1, 1};

        _tensorMap[76] = TensorDescriptor::fromFlatBuffer(descaleQAttrs);
        TensorAttributesT descaleKAttrs;
        descaleKAttrs.uid = 77;
        descaleKAttrs.data_type = DataType::FLOAT;
        descaleKAttrs.dims = {1, 1, 1, 1};
        descaleKAttrs.strides = {1, 1, 1, 1};

        _tensorMap[77] = TensorDescriptor::fromFlatBuffer(descaleKAttrs);
        TensorAttributesT descaleVAttrs;
        descaleVAttrs.uid = 78;
        descaleVAttrs.data_type = DataType::FLOAT;
        descaleVAttrs.dims = {1, 1, 1, 1};
        descaleVAttrs.strides = {1, 1, 1, 1};

        _tensorMap[78] = TensorDescriptor::fromFlatBuffer(descaleVAttrs);
        TensorAttributesT descaleSAttrs;
        descaleSAttrs.uid = 79;
        descaleSAttrs.data_type = DataType::FLOAT;
        descaleSAttrs.dims = {1, 1, 1, 1};
        descaleSAttrs.strides = {1, 1, 1, 1};

        _tensorMap[79] = TensorDescriptor::fromFlatBuffer(descaleSAttrs);
        TensorAttributesT scaleSAttrs;
        scaleSAttrs.uid = 80;
        scaleSAttrs.data_type = DataType::FLOAT;
        scaleSAttrs.dims = {1, 1, 1, 1};
        scaleSAttrs.strides = {1, 1, 1, 1};

        _tensorMap[80] = TensorDescriptor::fromFlatBuffer(scaleSAttrs);
        TensorAttributesT scaleOAttrs;
        scaleOAttrs.uid = 81;
        scaleOAttrs.data_type = DataType::FLOAT;
        scaleOAttrs.dims = {1, 1, 1, 1};
        scaleOAttrs.strides = {1, 1, 1, 1};

        _tensorMap[81] = TensorDescriptor::fromFlatBuffer(scaleOAttrs);
        TensorAttributesT statsAttrs;
        statsAttrs.uid = 82;
        statsAttrs.data_type = DataType::FLOAT;
        statsAttrs.dims = {1, 8, 128, 1};
        statsAttrs.strides = {1024, 128, 1, 1};

        _tensorMap[82] = TensorDescriptor::fromFlatBuffer(statsAttrs);
        TensorAttributesT maxOutputAttrs;
        maxOutputAttrs.uid = 83;
        maxOutputAttrs.data_type = DataType::FLOAT;
        maxOutputAttrs.dims = {1, 8, 128, 1};
        maxOutputAttrs.strides = {1024, 128, 1, 1};

        _tensorMap[83] = TensorDescriptor::fromFlatBuffer(maxOutputAttrs);
        TensorAttributesT sumExpAttrs;
        sumExpAttrs.uid = 84;
        sumExpAttrs.data_type = DataType::FLOAT;
        sumExpAttrs.dims = {1, 8, 128, 1};
        sumExpAttrs.strides = {1024, 128, 1, 1};

        _tensorMap[84] = TensorDescriptor::fromFlatBuffer(sumExpAttrs);
        TensorAttributesT rngDumpAttrs;
        rngDumpAttrs.uid = 85;
        rngDumpAttrs.data_type = DataType::FLOAT;
        rngDumpAttrs.dims = {1, 8, 128, 128};
        rngDumpAttrs.strides = {131072, 16384, 128, 1};

        _tensorMap[85] = TensorDescriptor::fromFlatBuffer(rngDumpAttrs);
        TensorAttributesT amaxSAttrs;
        amaxSAttrs.uid = 86;
        amaxSAttrs.data_type = DataType::FLOAT;
        amaxSAttrs.dims = {1, 1, 1, 1};
        amaxSAttrs.strides = {1, 1, 1, 1};

        _tensorMap[86] = TensorDescriptor::fromFlatBuffer(amaxSAttrs);
        TensorAttributesT amaxOAttrs;
        amaxOAttrs.uid = 87;
        amaxOAttrs.data_type = DataType::FLOAT;
        amaxOAttrs.dims = {1, 1, 1, 1};
        amaxOAttrs.strides = {1, 1, 1, 1};

        _tensorMap[87] = TensorDescriptor::fromFlatBuffer(amaxOAttrs);
    }

    static hipdnn_data_sdk::data_objects::SdpaAttributesT createStandardSdpaFpropAttrs()
    {
        hipdnn_data_sdk::data_objects::SdpaAttributesT attrs;
        attrs.q_tensor_uid = 60;
        attrs.k_tensor_uid = 61;
        attrs.v_tensor_uid = 62;
        attrs.o_tensor_uid = 63;
        attrs.attn_mask_tensor_uid = 64;
        attrs.scale_tensor_uid = 65;
        attrs.seq_len_q_tensor_uid = 66;
        attrs.seq_len_kv_tensor_uid = 67;
        attrs.seed_tensor_uid = 68;
        attrs.offset_tensor_uid = 69;
        attrs.dropout_mask_tensor_uid = 70;
        attrs.dropout_scale_tensor_uid = 71;
        attrs.page_table_k_tensor_uid = 72;
        attrs.page_table_v_tensor_uid = 73;
        attrs.block_mask_tensor_uid = 74;
        attrs.sink_token_tensor_uid = 75;
        attrs.descale_q_tensor_uid = 76;
        attrs.descale_k_tensor_uid = 77;
        attrs.descale_v_tensor_uid = 78;
        attrs.descale_s_tensor_uid = 79;
        attrs.scale_s_tensor_uid = 80;
        attrs.scale_o_tensor_uid = 81;
        attrs.stats_tensor_uid = 82;
        attrs.max_tensor_uid = 83;
        attrs.sum_exp_tensor_uid = 84;
        attrs.rng_dump_tensor_uid = 85;
        attrs.amax_s_tensor_uid = 86;
        attrs.amax_o_tensor_uid = 87;
        attrs.diagonal_alignment = DiagonalAlignment::TOP_LEFT;
        attrs.implementation = AttentionImplementation::AUTO;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardSdpaFpropAttrs());
        return node;
    }

    // Verifies that a packed tensor descriptor (retrieved via getAttribute) has the
    // expected UID, data_type, dimensions, and strides.
    static void verifyTensorDescriptor(hipdnnBackendDescriptor_t tensorDesc,
                                       int64_t expectedUid,
                                       hipdnnDataType_t expectedDataType,
                                       const std::vector<int64_t>& expectedDims,
                                       const std::vector<int64_t>& expectedStrides)
    {
        int64_t uid = 0;
        int64_t uidCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uidCount, &uid);
        EXPECT_EQ(uid, expectedUid);

        hipdnnDataType_t dataType = {};
        int64_t dtCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &dataType);
        EXPECT_EQ(dataType, expectedDataType);

        int64_t dimCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 0, &dimCount, nullptr);
        ASSERT_EQ(dimCount, static_cast<int64_t>(expectedDims.size()));
        std::vector<int64_t> dims(static_cast<size_t>(dimCount));
        int64_t actualDimCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, dimCount, &actualDimCount, dims.data());
        EXPECT_EQ(dims, expectedDims);

        int64_t strideCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 0, &strideCount, nullptr);
        ASSERT_EQ(strideCount, static_cast<int64_t>(expectedStrides.size()));
        std::vector<int64_t> strides(static_cast<size_t>(strideCount));
        int64_t actualStrideCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, strideCount, &actualStrideCount, strides.data());
        EXPECT_EQ(strides, expectedStrides);
    }
};

TEST_F(TestSdpaFpropOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_SDPA_FPROP_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().q_tensor_uid, 60);
}

TEST_F(TestSdpaFpropOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createStandardNode();

    // NodeFactory::createOperationFromNode delegates to fromNode internally.
    // Verify the delegation produces a valid, correctly-typed descriptor.
    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
    ASSERT_NE(graphOp, nullptr);

    // Verify the factory dispatched to the correct operation type, then static_cast.
    // Cannot use dynamic_pointer_cast: backend tests compile with -fno-rtti.
    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SdpaAttributes);
    auto desc = std::static_pointer_cast<SdpaFpropOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().q_tensor_uid, 60);
    EXPECT_EQ(desc->getData().k_tensor_uid, 61);
    EXPECT_EQ(desc->getData().v_tensor_uid, 62);
    EXPECT_EQ(desc->getData().o_tensor_uid, 63);
    EXPECT_EQ(desc->getData().attn_mask_tensor_uid, 64);
    EXPECT_EQ(desc->getData().scale_tensor_uid, 65);
    EXPECT_EQ(desc->getData().seq_len_q_tensor_uid, 66);
    EXPECT_EQ(desc->getData().seq_len_kv_tensor_uid, 67);
    EXPECT_EQ(desc->getData().seed_tensor_uid, 68);
    EXPECT_EQ(desc->getData().offset_tensor_uid, 69);
    EXPECT_EQ(desc->getData().dropout_mask_tensor_uid, 70);
    EXPECT_EQ(desc->getData().dropout_scale_tensor_uid, 71);
    EXPECT_EQ(desc->getData().page_table_k_tensor_uid, 72);
    EXPECT_EQ(desc->getData().page_table_v_tensor_uid, 73);
    EXPECT_EQ(desc->getData().block_mask_tensor_uid, 74);
    EXPECT_EQ(desc->getData().sink_token_tensor_uid, 75);
    EXPECT_EQ(desc->getData().descale_q_tensor_uid, 76);
    EXPECT_EQ(desc->getData().descale_k_tensor_uid, 77);
    EXPECT_EQ(desc->getData().descale_v_tensor_uid, 78);
    EXPECT_EQ(desc->getData().descale_s_tensor_uid, 79);
    EXPECT_EQ(desc->getData().scale_s_tensor_uid, 80);
    EXPECT_EQ(desc->getData().scale_o_tensor_uid, 81);
    EXPECT_EQ(desc->getData().stats_tensor_uid, 82);
    EXPECT_EQ(desc->getData().max_tensor_uid, 83);
    EXPECT_EQ(desc->getData().sum_exp_tensor_uid, 84);
    EXPECT_EQ(desc->getData().rng_dump_tensor_uid, 85);
    EXPECT_EQ(desc->getData().amax_s_tensor_uid, 86);
    EXPECT_EQ(desc->getData().amax_o_tensor_uid, 87);
    EXPECT_EQ(desc->getData().diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(desc->getData().implementation, AttentionImplementation::AUTO);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getQDesc()->getData().uid, 60);
    EXPECT_EQ(desc->getKDesc()->getData().uid, 61);
    EXPECT_EQ(desc->getVDesc()->getData().uid, 62);
    EXPECT_EQ(desc->getODesc()->getData().uid, 63);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().uid, 64);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, 65);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().uid, 66);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().uid, 67);
    EXPECT_EQ(desc->getSeedDesc()->getData().uid, 68);
    EXPECT_EQ(desc->getOffsetDesc()->getData().uid, 69);
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().uid, 70);
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().uid, 71);
    EXPECT_EQ(desc->getPageTableKDesc()->getData().uid, 72);
    EXPECT_EQ(desc->getPageTableVDesc()->getData().uid, 73);
    EXPECT_EQ(desc->getBlockMaskDesc()->getData().uid, 74);
    EXPECT_EQ(desc->getSinkTokenDesc()->getData().uid, 75);
    EXPECT_EQ(desc->getDescaleQDesc()->getData().uid, 76);
    EXPECT_EQ(desc->getDescaleKDesc()->getData().uid, 77);
    EXPECT_EQ(desc->getDescaleVDesc()->getData().uid, 78);
    EXPECT_EQ(desc->getDescaleSDesc()->getData().uid, 79);
    EXPECT_EQ(desc->getScaleSDesc()->getData().uid, 80);
    EXPECT_EQ(desc->getScaleODesc()->getData().uid, 81);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, 82);
    EXPECT_EQ(desc->getMaxDesc()->getData().uid, 83);
    EXPECT_EQ(desc->getSumExpDesc()->getData().uid, 84);
    EXPECT_EQ(desc->getRngDumpDesc()->getData().uid, 85);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().uid, 86);
    EXPECT_EQ(desc->getAmaxODesc()->getData().uid, 87);
}

TEST_F(TestSdpaFpropOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestSdpaFpropOperationFromNode, PreservesDiagonalAlignment)
{
    auto node = createStandardNode();
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    node.attributes.Set(attrs);
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
}

TEST_F(TestSdpaFpropOperationFromNode, PreservesAttentionImplementation)
{
    auto node = createStandardNode();
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.implementation = AttentionImplementation::COMPOSITE;
    node.attributes.Set(attrs);
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().implementation, AttentionImplementation::COMPOSITE);
}

TEST_F(TestSdpaFpropOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getQDesc(), nullptr);
    EXPECT_EQ(desc->getQDesc()->getData().uid, 60);
    ASSERT_NE(desc->getKDesc(), nullptr);
    EXPECT_EQ(desc->getKDesc()->getData().uid, 61);
    ASSERT_NE(desc->getVDesc(), nullptr);
    EXPECT_EQ(desc->getVDesc()->getData().uid, 62);
    ASSERT_NE(desc->getODesc(), nullptr);
    EXPECT_EQ(desc->getODesc()->getData().uid, 63);
    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().uid, 64);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, 65);
    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().uid, 66);
    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().uid, 67);
    ASSERT_NE(desc->getSeedDesc(), nullptr);
    EXPECT_EQ(desc->getSeedDesc()->getData().uid, 68);
    ASSERT_NE(desc->getOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getOffsetDesc()->getData().uid, 69);
    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().uid, 70);
    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().uid, 71);
    ASSERT_NE(desc->getPageTableKDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableKDesc()->getData().uid, 72);
    ASSERT_NE(desc->getPageTableVDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableVDesc()->getData().uid, 73);
    ASSERT_NE(desc->getBlockMaskDesc(), nullptr);
    EXPECT_EQ(desc->getBlockMaskDesc()->getData().uid, 74);
    ASSERT_NE(desc->getSinkTokenDesc(), nullptr);
    EXPECT_EQ(desc->getSinkTokenDesc()->getData().uid, 75);
    ASSERT_NE(desc->getDescaleQDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleQDesc()->getData().uid, 76);
    ASSERT_NE(desc->getDescaleKDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleKDesc()->getData().uid, 77);
    ASSERT_NE(desc->getDescaleVDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleVDesc()->getData().uid, 78);
    ASSERT_NE(desc->getDescaleSDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleSDesc()->getData().uid, 79);
    ASSERT_NE(desc->getScaleSDesc(), nullptr);
    EXPECT_EQ(desc->getScaleSDesc()->getData().uid, 80);
    ASSERT_NE(desc->getScaleODesc(), nullptr);
    EXPECT_EQ(desc->getScaleODesc()->getData().uid, 81);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, 82);
    ASSERT_NE(desc->getMaxDesc(), nullptr);
    EXPECT_EQ(desc->getMaxDesc()->getData().uid, 83);
    ASSERT_NE(desc->getSumExpDesc(), nullptr);
    EXPECT_EQ(desc->getSumExpDesc()->getData().uid, 84);
    ASSERT_NE(desc->getRngDumpDesc(), nullptr);
    EXPECT_EQ(desc->getRngDumpDesc()->getData().uid, 85);
    ASSERT_NE(desc->getAmaxSDesc(), nullptr);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().uid, 86);
    ASSERT_NE(desc->getAmaxODesc(), nullptr);
    EXPECT_EQ(desc->getAmaxODesc()->getData().uid, 87);
}

TEST_F(TestSdpaFpropOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getQDesc(), _tensorMap[60]);
    EXPECT_EQ(desc->getKDesc(), _tensorMap[61]);
    EXPECT_EQ(desc->getVDesc(), _tensorMap[62]);
    EXPECT_EQ(desc->getODesc(), _tensorMap[63]);
    EXPECT_EQ(desc->getAttnMaskDesc(), _tensorMap[64]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[65]);
    EXPECT_EQ(desc->getSeqLenQDesc(), _tensorMap[66]);
    EXPECT_EQ(desc->getSeqLenKvDesc(), _tensorMap[67]);
    EXPECT_EQ(desc->getSeedDesc(), _tensorMap[68]);
    EXPECT_EQ(desc->getOffsetDesc(), _tensorMap[69]);
    EXPECT_EQ(desc->getDropoutMaskDesc(), _tensorMap[70]);
    EXPECT_EQ(desc->getDropoutScaleDesc(), _tensorMap[71]);
    EXPECT_EQ(desc->getPageTableKDesc(), _tensorMap[72]);
    EXPECT_EQ(desc->getPageTableVDesc(), _tensorMap[73]);
    EXPECT_EQ(desc->getBlockMaskDesc(), _tensorMap[74]);
    EXPECT_EQ(desc->getSinkTokenDesc(), _tensorMap[75]);
    EXPECT_EQ(desc->getDescaleQDesc(), _tensorMap[76]);
    EXPECT_EQ(desc->getDescaleKDesc(), _tensorMap[77]);
    EXPECT_EQ(desc->getDescaleVDesc(), _tensorMap[78]);
    EXPECT_EQ(desc->getDescaleSDesc(), _tensorMap[79]);
    EXPECT_EQ(desc->getScaleSDesc(), _tensorMap[80]);
    EXPECT_EQ(desc->getScaleODesc(), _tensorMap[81]);
    EXPECT_EQ(desc->getStatsDesc(), _tensorMap[82]);
    EXPECT_EQ(desc->getMaxDesc(), _tensorMap[83]);
    EXPECT_EQ(desc->getSumExpDesc(), _tensorMap[84]);
    EXPECT_EQ(desc->getRngDumpDesc(), _tensorMap[85]);
    EXPECT_EQ(desc->getAmaxSDesc(), _tensorMap[86]);
    EXPECT_EQ(desc->getAmaxODesc(), _tensorMap[87]);
}

TEST_F(TestSdpaFpropOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getQDesc(), nullptr);
    EXPECT_EQ(desc->getQDesc()->getData().uid, 60);
    EXPECT_EQ(desc->getQDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getQDesc()->getData().dims, (std::vector<int64_t>{1, 8, 128, 64}));
    EXPECT_EQ(desc->getQDesc()->getData().strides, (std::vector<int64_t>{65536, 8192, 64, 1}));

    ASSERT_NE(desc->getKDesc(), nullptr);
    EXPECT_EQ(desc->getKDesc()->getData().uid, 61);
    EXPECT_EQ(desc->getKDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getKDesc()->getData().dims, (std::vector<int64_t>{1, 8, 128, 64}));
    EXPECT_EQ(desc->getKDesc()->getData().strides, (std::vector<int64_t>{65536, 8192, 64, 1}));

    ASSERT_NE(desc->getVDesc(), nullptr);
    EXPECT_EQ(desc->getVDesc()->getData().uid, 62);
    EXPECT_EQ(desc->getVDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getVDesc()->getData().dims, (std::vector<int64_t>{1, 8, 128, 64}));
    EXPECT_EQ(desc->getVDesc()->getData().strides, (std::vector<int64_t>{65536, 8192, 64, 1}));

    ASSERT_NE(desc->getODesc(), nullptr);
    EXPECT_EQ(desc->getODesc()->getData().uid, 63);
    EXPECT_EQ(desc->getODesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getODesc()->getData().dims, (std::vector<int64_t>{1, 8, 128, 64}));
    EXPECT_EQ(desc->getODesc()->getData().strides, (std::vector<int64_t>{65536, 8192, 64, 1}));

    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().uid, 64);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().dims, (std::vector<int64_t>{1, 1, 128, 128}));
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().strides, (std::vector<int64_t>{16384, 16384, 128, 1}));

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, 65);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().uid, 66);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().uid, 67);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getSeedDesc(), nullptr);
    EXPECT_EQ(desc->getSeedDesc()->getData().uid, 68);
    EXPECT_EQ(desc->getSeedDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getSeedDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getSeedDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getOffsetDesc()->getData().uid, 69);
    EXPECT_EQ(desc->getOffsetDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getOffsetDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getOffsetDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().uid, 70);
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().dims, (std::vector<int64_t>{1, 8, 128, 128}));
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().strides, (std::vector<int64_t>{131072, 16384, 128, 1}));

    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().uid, 71);
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getPageTableKDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableKDesc()->getData().uid, 72);
    EXPECT_EQ(desc->getPageTableKDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getPageTableKDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getPageTableKDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getPageTableVDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableVDesc()->getData().uid, 73);
    EXPECT_EQ(desc->getPageTableVDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getPageTableVDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getPageTableVDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getBlockMaskDesc(), nullptr);
    EXPECT_EQ(desc->getBlockMaskDesc()->getData().uid, 74);
    EXPECT_EQ(desc->getBlockMaskDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBlockMaskDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getBlockMaskDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getSinkTokenDesc(), nullptr);
    EXPECT_EQ(desc->getSinkTokenDesc()->getData().uid, 75);
    EXPECT_EQ(desc->getSinkTokenDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getSinkTokenDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getSinkTokenDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getDescaleQDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleQDesc()->getData().uid, 76);
    EXPECT_EQ(desc->getDescaleQDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDescaleQDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getDescaleQDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getDescaleKDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleKDesc()->getData().uid, 77);
    EXPECT_EQ(desc->getDescaleKDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDescaleKDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getDescaleKDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getDescaleVDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleVDesc()->getData().uid, 78);
    EXPECT_EQ(desc->getDescaleVDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDescaleVDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getDescaleVDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getDescaleSDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleSDesc()->getData().uid, 79);
    EXPECT_EQ(desc->getDescaleSDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDescaleSDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getDescaleSDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getScaleSDesc(), nullptr);
    EXPECT_EQ(desc->getScaleSDesc()->getData().uid, 80);
    EXPECT_EQ(desc->getScaleSDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleSDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getScaleSDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getScaleODesc(), nullptr);
    EXPECT_EQ(desc->getScaleODesc()->getData().uid, 81);
    EXPECT_EQ(desc->getScaleODesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleODesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getScaleODesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getStatsDesc(), nullptr);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, 82);
    EXPECT_EQ(desc->getStatsDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getStatsDesc()->getData().dims, (std::vector<int64_t>{1, 8, 128, 1}));
    EXPECT_EQ(desc->getStatsDesc()->getData().strides, (std::vector<int64_t>{1024, 128, 1, 1}));

    ASSERT_NE(desc->getMaxDesc(), nullptr);
    EXPECT_EQ(desc->getMaxDesc()->getData().uid, 83);
    EXPECT_EQ(desc->getMaxDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getMaxDesc()->getData().dims, (std::vector<int64_t>{1, 8, 128, 1}));
    EXPECT_EQ(desc->getMaxDesc()->getData().strides, (std::vector<int64_t>{1024, 128, 1, 1}));

    ASSERT_NE(desc->getSumExpDesc(), nullptr);
    EXPECT_EQ(desc->getSumExpDesc()->getData().uid, 84);
    EXPECT_EQ(desc->getSumExpDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getSumExpDesc()->getData().dims, (std::vector<int64_t>{1, 8, 128, 1}));
    EXPECT_EQ(desc->getSumExpDesc()->getData().strides, (std::vector<int64_t>{1024, 128, 1, 1}));

    ASSERT_NE(desc->getRngDumpDesc(), nullptr);
    EXPECT_EQ(desc->getRngDumpDesc()->getData().uid, 85);
    EXPECT_EQ(desc->getRngDumpDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getRngDumpDesc()->getData().dims, (std::vector<int64_t>{1, 8, 128, 128}));
    EXPECT_EQ(desc->getRngDumpDesc()->getData().strides, (std::vector<int64_t>{131072, 16384, 128, 1}));

    ASSERT_NE(desc->getAmaxSDesc(), nullptr);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().uid, 86);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getAmaxSDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

    ASSERT_NE(desc->getAmaxODesc(), nullptr);
    EXPECT_EQ(desc->getAmaxODesc()->getData().uid, 87);
    EXPECT_EQ(desc->getAmaxODesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getAmaxODesc()->getData().dims, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(desc->getAmaxODesc()->getData().strides, (std::vector<int64_t>{1, 1, 1, 1}));

}

TEST_F(TestSdpaFpropOperationFromNode, FailsWithMissingQTensor)
{
    _tensorMap.erase(60);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SdpaFpropOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSdpaFpropOperationFromNode, FailsWithMissingKTensor)
{
    _tensorMap.erase(61);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SdpaFpropOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSdpaFpropOperationFromNode, FailsWithMissingVTensor)
{
    _tensorMap.erase(62);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SdpaFpropOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSdpaFpropOperationFromNode, FailsWithMissingOTensor)
{
    _tensorMap.erase(63);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SdpaFpropOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSdpaFpropOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.attn_mask_tensor_uid = flatbuffers::nullopt;
    attrs.scale_tensor_uid = flatbuffers::nullopt;
    attrs.seq_len_q_tensor_uid = flatbuffers::nullopt;
    attrs.seq_len_kv_tensor_uid = flatbuffers::nullopt;
    attrs.seed_tensor_uid = flatbuffers::nullopt;
    attrs.offset_tensor_uid = flatbuffers::nullopt;
    attrs.dropout_mask_tensor_uid = flatbuffers::nullopt;
    attrs.dropout_scale_tensor_uid = flatbuffers::nullopt;
    attrs.page_table_k_tensor_uid = flatbuffers::nullopt;
    attrs.page_table_v_tensor_uid = flatbuffers::nullopt;
    attrs.block_mask_tensor_uid = flatbuffers::nullopt;
    attrs.sink_token_tensor_uid = flatbuffers::nullopt;
    attrs.descale_q_tensor_uid = flatbuffers::nullopt;
    attrs.descale_k_tensor_uid = flatbuffers::nullopt;
    attrs.descale_v_tensor_uid = flatbuffers::nullopt;
    attrs.descale_s_tensor_uid = flatbuffers::nullopt;
    attrs.scale_s_tensor_uid = flatbuffers::nullopt;
    attrs.scale_o_tensor_uid = flatbuffers::nullopt;
    attrs.stats_tensor_uid = flatbuffers::nullopt;
    attrs.max_tensor_uid = flatbuffers::nullopt;
    attrs.sum_exp_tensor_uid = flatbuffers::nullopt;
    attrs.rng_dump_tensor_uid = flatbuffers::nullopt;
    attrs.amax_s_tensor_uid = flatbuffers::nullopt;
    attrs.amax_o_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getQDesc(), nullptr);
    EXPECT_NE(desc->getKDesc(), nullptr);
    EXPECT_NE(desc->getVDesc(), nullptr);
    EXPECT_NE(desc->getODesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getAttnMaskDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_EQ(desc->getSeedDesc(), nullptr);
    EXPECT_EQ(desc->getOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableKDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableVDesc(), nullptr);
    EXPECT_EQ(desc->getBlockMaskDesc(), nullptr);
    EXPECT_EQ(desc->getSinkTokenDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleQDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleKDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleVDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleSDesc(), nullptr);
    EXPECT_EQ(desc->getScaleSDesc(), nullptr);
    EXPECT_EQ(desc->getScaleODesc(), nullptr);
    EXPECT_EQ(desc->getStatsDesc(), nullptr);
    EXPECT_EQ(desc->getMaxDesc(), nullptr);
    EXPECT_EQ(desc->getSumExpDesc(), nullptr);
    EXPECT_EQ(desc->getRngDumpDesc(), nullptr);
    EXPECT_EQ(desc->getAmaxSDesc(), nullptr);
    EXPECT_EQ(desc->getAmaxODesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalAttnMaskNullWhenTensorMissing)
{
    _tensorMap.erase(64);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getAttnMaskDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalScaleNullWhenTensorMissing)
{
    _tensorMap.erase(65);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getScaleDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalSeqLenQNullWhenTensorMissing)
{
    _tensorMap.erase(66);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalSeqLenKvNullWhenTensorMissing)
{
    _tensorMap.erase(67);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalSeedNullWhenTensorMissing)
{
    _tensorMap.erase(68);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getSeedDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalOffsetNullWhenTensorMissing)
{
    _tensorMap.erase(69);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getOffsetDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalDropoutMaskNullWhenTensorMissing)
{
    _tensorMap.erase(70);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalDropoutScaleNullWhenTensorMissing)
{
    _tensorMap.erase(71);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalPageTableKNullWhenTensorMissing)
{
    _tensorMap.erase(72);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getPageTableKDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalPageTableVNullWhenTensorMissing)
{
    _tensorMap.erase(73);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getPageTableVDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalBlockMaskNullWhenTensorMissing)
{
    _tensorMap.erase(74);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getBlockMaskDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalSinkTokenNullWhenTensorMissing)
{
    _tensorMap.erase(75);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getSinkTokenDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalDescaleQNullWhenTensorMissing)
{
    _tensorMap.erase(76);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getDescaleQDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalDescaleKNullWhenTensorMissing)
{
    _tensorMap.erase(77);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getDescaleKDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalDescaleVNullWhenTensorMissing)
{
    _tensorMap.erase(78);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getDescaleVDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalDescaleSNullWhenTensorMissing)
{
    _tensorMap.erase(79);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getDescaleSDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalScaleSNullWhenTensorMissing)
{
    _tensorMap.erase(80);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getScaleSDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalScaleONullWhenTensorMissing)
{
    _tensorMap.erase(81);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getScaleODesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalStatsNullWhenTensorMissing)
{
    _tensorMap.erase(82);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getStatsDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalMaxOutputNullWhenTensorMissing)
{
    _tensorMap.erase(83);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getMaxDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalSumExpNullWhenTensorMissing)
{
    _tensorMap.erase(84);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getSumExpDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalRngDumpNullWhenTensorMissing)
{
    _tensorMap.erase(85);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getRngDumpDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalAmaxSNullWhenTensorMissing)
{
    _tensorMap.erase(86);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getAmaxSDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, OptionalAmaxONullWhenTensorMissing)
{
    _tensorMap.erase(87);
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getAmaxODesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 28);
    EXPECT_EQ(tensors[0]->getData().uid, 60);
    EXPECT_EQ(tensors[1]->getData().uid, 61);
    EXPECT_EQ(tensors[2]->getData().uid, 62);
    EXPECT_EQ(tensors[3]->getData().uid, 63);
    EXPECT_EQ(tensors[4]->getData().uid, 64);
    EXPECT_EQ(tensors[5]->getData().uid, 65);
    EXPECT_EQ(tensors[6]->getData().uid, 66);
    EXPECT_EQ(tensors[7]->getData().uid, 67);
    EXPECT_EQ(tensors[8]->getData().uid, 68);
    EXPECT_EQ(tensors[9]->getData().uid, 69);
    EXPECT_EQ(tensors[10]->getData().uid, 70);
    EXPECT_EQ(tensors[11]->getData().uid, 71);
    EXPECT_EQ(tensors[12]->getData().uid, 72);
    EXPECT_EQ(tensors[13]->getData().uid, 73);
    EXPECT_EQ(tensors[14]->getData().uid, 74);
    EXPECT_EQ(tensors[15]->getData().uid, 75);
    EXPECT_EQ(tensors[16]->getData().uid, 76);
    EXPECT_EQ(tensors[17]->getData().uid, 77);
    EXPECT_EQ(tensors[18]->getData().uid, 78);
    EXPECT_EQ(tensors[19]->getData().uid, 79);
    EXPECT_EQ(tensors[20]->getData().uid, 80);
    EXPECT_EQ(tensors[21]->getData().uid, 81);
    EXPECT_EQ(tensors[22]->getData().uid, 82);
    EXPECT_EQ(tensors[23]->getData().uid, 83);
    EXPECT_EQ(tensors[24]->getData().uid, 84);
    EXPECT_EQ(tensors[25]->getData().uid, 85);
    EXPECT_EQ(tensors[26]->getData().uid, 86);
    EXPECT_EQ(tensors[27]->getData().uid, 87);
}

TEST_F(TestSdpaFpropOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SdpaAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->q_tensor_uid, 60);
    EXPECT_EQ(rebuiltAttrs->k_tensor_uid, 61);
    EXPECT_EQ(rebuiltAttrs->v_tensor_uid, 62);
    EXPECT_EQ(rebuiltAttrs->o_tensor_uid, 63);
    EXPECT_EQ(rebuiltAttrs->attn_mask_tensor_uid, 64);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, 65);
    EXPECT_EQ(rebuiltAttrs->seq_len_q_tensor_uid, 66);
    EXPECT_EQ(rebuiltAttrs->seq_len_kv_tensor_uid, 67);
    EXPECT_EQ(rebuiltAttrs->seed_tensor_uid, 68);
    EXPECT_EQ(rebuiltAttrs->offset_tensor_uid, 69);
    EXPECT_EQ(rebuiltAttrs->dropout_mask_tensor_uid, 70);
    EXPECT_EQ(rebuiltAttrs->dropout_scale_tensor_uid, 71);
    EXPECT_EQ(rebuiltAttrs->page_table_k_tensor_uid, 72);
    EXPECT_EQ(rebuiltAttrs->page_table_v_tensor_uid, 73);
    EXPECT_EQ(rebuiltAttrs->block_mask_tensor_uid, 74);
    EXPECT_EQ(rebuiltAttrs->sink_token_tensor_uid, 75);
    EXPECT_EQ(rebuiltAttrs->descale_q_tensor_uid, 76);
    EXPECT_EQ(rebuiltAttrs->descale_k_tensor_uid, 77);
    EXPECT_EQ(rebuiltAttrs->descale_v_tensor_uid, 78);
    EXPECT_EQ(rebuiltAttrs->descale_s_tensor_uid, 79);
    EXPECT_EQ(rebuiltAttrs->scale_s_tensor_uid, 80);
    EXPECT_EQ(rebuiltAttrs->scale_o_tensor_uid, 81);
    EXPECT_EQ(rebuiltAttrs->stats_tensor_uid, 82);
    EXPECT_EQ(rebuiltAttrs->max_tensor_uid, 83);
    EXPECT_EQ(rebuiltAttrs->sum_exp_tensor_uid, 84);
    EXPECT_EQ(rebuiltAttrs->rng_dump_tensor_uid, 85);
    EXPECT_EQ(rebuiltAttrs->amax_s_tensor_uid, 86);
    EXPECT_EQ(rebuiltAttrs->amax_o_tensor_uid, 87);
    EXPECT_EQ(rebuiltAttrs->diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(rebuiltAttrs->implementation, AttentionImplementation::AUTO);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesGenerateStats)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.generate_stats = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().generate_stats.has_value());
    EXPECT_EQ(desc->getData().generate_stats.value(), true);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->generate_stats.has_value());
    EXPECT_EQ(rebuiltAttrs->generate_stats.value(), true);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesAlibiMask)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.alibi_mask = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().alibi_mask);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->alibi_mask);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesPaddingMask)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.padding_mask = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().padding_mask);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->padding_mask);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesCausalMask)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.causal_mask = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().causal_mask);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->causal_mask);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesCausalMaskBottomRight)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.causal_mask_bottom_right = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().causal_mask_bottom_right);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->causal_mask_bottom_right);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesDropoutProbability)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.dropout_probability = 0.5F;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().dropout_probability.has_value());
    EXPECT_FLOAT_EQ(desc->getData().dropout_probability.value(), 0.5F);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->dropout_probability.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->dropout_probability.value(), 0.5F);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesAttnScaleValue)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.attn_scale_value = 0.5F;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(desc->getData().attn_scale_value.value(), 0.5F);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->attn_scale_value.value(), 0.5F);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesLeftBound)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.left_bound = 2;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().left_bound.has_value());
    EXPECT_EQ(desc->getData().left_bound.value(), 2);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->left_bound.has_value());
    EXPECT_EQ(rebuiltAttrs->left_bound.value(), 2);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesRightBound)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.right_bound = 2;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().right_bound.has_value());
    EXPECT_EQ(desc->getData().right_bound.value(), 2);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->right_bound.has_value());
    EXPECT_EQ(rebuiltAttrs->right_bound.value(), 2);
}

TEST_F(TestSdpaFpropOperationFromNode, FromNodePreservesMaxSeqLenKv)
{
    auto attrs = createStandardSdpaFpropAttrs();
    attrs.max_seq_len_kv = 2;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().max_seq_len_kv.has_value());
    EXPECT_EQ(desc->getData().max_seq_len_kv.value(), 2);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->max_seq_len_kv.has_value());
    EXPECT_EQ(rebuiltAttrs->max_seq_len_kv.value(), 2);
}

TEST_F(TestSdpaFpropOperationFromNode, BuildNodeOmitsUnsetOptionalScalars)
{
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    EXPECT_FALSE(rebuiltAttrs->generate_stats.has_value());
    EXPECT_FALSE(rebuiltAttrs->alibi_mask);
    EXPECT_FALSE(rebuiltAttrs->padding_mask);
    EXPECT_FALSE(rebuiltAttrs->causal_mask);
    EXPECT_FALSE(rebuiltAttrs->causal_mask_bottom_right);
    EXPECT_FALSE(rebuiltAttrs->dropout_probability.has_value());
    EXPECT_FALSE(rebuiltAttrs->attn_scale_value.has_value());
    EXPECT_FALSE(rebuiltAttrs->left_bound.has_value());
    EXPECT_FALSE(rebuiltAttrs->right_bound.has_value());
    EXPECT_FALSE(rebuiltAttrs->max_seq_len_kv.has_value());
}

TEST_F(TestSdpaFpropOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify diagonal_alignment
    hipdnnDiagonalAlignment_t diagonalAlignment = {};
    int64_t diagonalAlignmentCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT, HIPDNN_TYPE_DIAGONAL_ALIGNMENT, 1, &diagonalAlignmentCount, &diagonalAlignment);
    ASSERT_EQ(diagonalAlignment, HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT);

    // Verify implementation
    hipdnnAttentionImplementation_t implementation = {};
    int64_t implementationCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT, HIPDNN_TYPE_ATTENTION_IMPLEMENTATION, 1, &implementationCount, &implementation);
    ASSERT_EQ(implementation, HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT);

    // Verify q tensor
    hipdnn_backend::ScopedDescriptor qScoped;
    int64_t qCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &qCount,
                       static_cast<void*>(qScoped.getPtr()));
    ASSERT_EQ(qCount, 1);
    ASSERT_NE(qScoped.get(), nullptr);
    verifyTensorDescriptor(qScoped.get(), 60, HIPDNN_DATA_FLOAT,
                           {1, 8, 128, 64},
                           {65536, 8192, 64, 1});

    // Verify k tensor
    hipdnn_backend::ScopedDescriptor kScoped;
    int64_t kCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &kCount,
                       static_cast<void*>(kScoped.getPtr()));
    ASSERT_EQ(kCount, 1);
    ASSERT_NE(kScoped.get(), nullptr);
    verifyTensorDescriptor(kScoped.get(), 61, HIPDNN_DATA_FLOAT,
                           {1, 8, 128, 64},
                           {65536, 8192, 64, 1});

    // Verify v tensor
    hipdnn_backend::ScopedDescriptor vScoped;
    int64_t vCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &vCount,
                       static_cast<void*>(vScoped.getPtr()));
    ASSERT_EQ(vCount, 1);
    ASSERT_NE(vScoped.get(), nullptr);
    verifyTensorDescriptor(vScoped.get(), 62, HIPDNN_DATA_FLOAT,
                           {1, 8, 128, 64},
                           {65536, 8192, 64, 1});

    // Verify o tensor
    hipdnn_backend::ScopedDescriptor oScoped;
    int64_t oCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &oCount,
                       static_cast<void*>(oScoped.getPtr()));
    ASSERT_EQ(oCount, 1);
    ASSERT_NE(oScoped.get(), nullptr);
    verifyTensorDescriptor(oScoped.get(), 63, HIPDNN_DATA_FLOAT,
                           {1, 8, 128, 64},
                           {65536, 8192, 64, 1});

    // Verify attn_mask tensor (optional)
    hipdnn_backend::ScopedDescriptor attnMaskScoped;
    int64_t attnMaskCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &attnMaskCount,
                       static_cast<void*>(attnMaskScoped.getPtr()));
    ASSERT_EQ(attnMaskCount, 1);
    ASSERT_NE(attnMaskScoped.get(), nullptr);
    verifyTensorDescriptor(attnMaskScoped.get(), 64, HIPDNN_DATA_FLOAT,
                           {1, 1, 128, 128},
                           {16384, 16384, 128, 1});

    // Verify scale tensor (optional)
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    verifyTensorDescriptor(scaleScoped.get(), 65, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify seq_len_q tensor (optional)
    hipdnn_backend::ScopedDescriptor seqLenQScoped;
    int64_t seqLenQCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &seqLenQCount,
                       static_cast<void*>(seqLenQScoped.getPtr()));
    ASSERT_EQ(seqLenQCount, 1);
    ASSERT_NE(seqLenQScoped.get(), nullptr);
    verifyTensorDescriptor(seqLenQScoped.get(), 66, HIPDNN_DATA_FLOAT,
                           {1},
                           {1});

    // Verify seq_len_kv tensor (optional)
    hipdnn_backend::ScopedDescriptor seqLenKvScoped;
    int64_t seqLenKvCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &seqLenKvCount,
                       static_cast<void*>(seqLenKvScoped.getPtr()));
    ASSERT_EQ(seqLenKvCount, 1);
    ASSERT_NE(seqLenKvScoped.get(), nullptr);
    verifyTensorDescriptor(seqLenKvScoped.get(), 67, HIPDNN_DATA_FLOAT,
                           {1},
                           {1});

    // Verify seed tensor (optional)
    hipdnn_backend::ScopedDescriptor seedScoped;
    int64_t seedCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &seedCount,
                       static_cast<void*>(seedScoped.getPtr()));
    ASSERT_EQ(seedCount, 1);
    ASSERT_NE(seedScoped.get(), nullptr);
    verifyTensorDescriptor(seedScoped.get(), 68, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify offset tensor (optional)
    hipdnn_backend::ScopedDescriptor offsetScoped;
    int64_t offsetCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &offsetCount,
                       static_cast<void*>(offsetScoped.getPtr()));
    ASSERT_EQ(offsetCount, 1);
    ASSERT_NE(offsetScoped.get(), nullptr);
    verifyTensorDescriptor(offsetScoped.get(), 69, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify dropout_mask tensor (optional)
    hipdnn_backend::ScopedDescriptor dropoutMaskScoped;
    int64_t dropoutMaskCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dropoutMaskCount,
                       static_cast<void*>(dropoutMaskScoped.getPtr()));
    ASSERT_EQ(dropoutMaskCount, 1);
    ASSERT_NE(dropoutMaskScoped.get(), nullptr);
    verifyTensorDescriptor(dropoutMaskScoped.get(), 70, HIPDNN_DATA_FLOAT,
                           {1, 8, 128, 128},
                           {131072, 16384, 128, 1});

    // Verify dropout_scale tensor (optional)
    hipdnn_backend::ScopedDescriptor dropoutScaleScoped;
    int64_t dropoutScaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dropoutScaleCount,
                       static_cast<void*>(dropoutScaleScoped.getPtr()));
    ASSERT_EQ(dropoutScaleCount, 1);
    ASSERT_NE(dropoutScaleScoped.get(), nullptr);
    verifyTensorDescriptor(dropoutScaleScoped.get(), 71, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify page_table_k tensor (optional)
    hipdnn_backend::ScopedDescriptor pageTableKScoped;
    int64_t pageTableKCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &pageTableKCount,
                       static_cast<void*>(pageTableKScoped.getPtr()));
    ASSERT_EQ(pageTableKCount, 1);
    ASSERT_NE(pageTableKScoped.get(), nullptr);
    verifyTensorDescriptor(pageTableKScoped.get(), 72, HIPDNN_DATA_FLOAT,
                           {1},
                           {1});

    // Verify page_table_v tensor (optional)
    hipdnn_backend::ScopedDescriptor pageTableVScoped;
    int64_t pageTableVCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &pageTableVCount,
                       static_cast<void*>(pageTableVScoped.getPtr()));
    ASSERT_EQ(pageTableVCount, 1);
    ASSERT_NE(pageTableVScoped.get(), nullptr);
    verifyTensorDescriptor(pageTableVScoped.get(), 73, HIPDNN_DATA_FLOAT,
                           {1},
                           {1});

    // Verify block_mask tensor (optional)
    hipdnn_backend::ScopedDescriptor blockMaskScoped;
    int64_t blockMaskCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &blockMaskCount,
                       static_cast<void*>(blockMaskScoped.getPtr()));
    ASSERT_EQ(blockMaskCount, 1);
    ASSERT_NE(blockMaskScoped.get(), nullptr);
    verifyTensorDescriptor(blockMaskScoped.get(), 74, HIPDNN_DATA_FLOAT,
                           {1},
                           {1});

    // Verify sink_token tensor (optional)
    hipdnn_backend::ScopedDescriptor sinkTokenScoped;
    int64_t sinkTokenCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &sinkTokenCount,
                       static_cast<void*>(sinkTokenScoped.getPtr()));
    ASSERT_EQ(sinkTokenCount, 1);
    ASSERT_NE(sinkTokenScoped.get(), nullptr);
    verifyTensorDescriptor(sinkTokenScoped.get(), 75, HIPDNN_DATA_FLOAT,
                           {1},
                           {1});

    // Verify descale_q tensor (optional)
    hipdnn_backend::ScopedDescriptor descaleQScoped;
    int64_t descaleQCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &descaleQCount,
                       static_cast<void*>(descaleQScoped.getPtr()));
    ASSERT_EQ(descaleQCount, 1);
    ASSERT_NE(descaleQScoped.get(), nullptr);
    verifyTensorDescriptor(descaleQScoped.get(), 76, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify descale_k tensor (optional)
    hipdnn_backend::ScopedDescriptor descaleKScoped;
    int64_t descaleKCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &descaleKCount,
                       static_cast<void*>(descaleKScoped.getPtr()));
    ASSERT_EQ(descaleKCount, 1);
    ASSERT_NE(descaleKScoped.get(), nullptr);
    verifyTensorDescriptor(descaleKScoped.get(), 77, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify descale_v tensor (optional)
    hipdnn_backend::ScopedDescriptor descaleVScoped;
    int64_t descaleVCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &descaleVCount,
                       static_cast<void*>(descaleVScoped.getPtr()));
    ASSERT_EQ(descaleVCount, 1);
    ASSERT_NE(descaleVScoped.get(), nullptr);
    verifyTensorDescriptor(descaleVScoped.get(), 78, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify descale_s tensor (optional)
    hipdnn_backend::ScopedDescriptor descaleSScoped;
    int64_t descaleSCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &descaleSCount,
                       static_cast<void*>(descaleSScoped.getPtr()));
    ASSERT_EQ(descaleSCount, 1);
    ASSERT_NE(descaleSScoped.get(), nullptr);
    verifyTensorDescriptor(descaleSScoped.get(), 79, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify scale_s tensor (optional)
    hipdnn_backend::ScopedDescriptor scaleSScoped;
    int64_t scaleSCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleSCount,
                       static_cast<void*>(scaleSScoped.getPtr()));
    ASSERT_EQ(scaleSCount, 1);
    ASSERT_NE(scaleSScoped.get(), nullptr);
    verifyTensorDescriptor(scaleSScoped.get(), 80, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify scale_o tensor (optional)
    hipdnn_backend::ScopedDescriptor scaleOScoped;
    int64_t scaleOCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleOCount,
                       static_cast<void*>(scaleOScoped.getPtr()));
    ASSERT_EQ(scaleOCount, 1);
    ASSERT_NE(scaleOScoped.get(), nullptr);
    verifyTensorDescriptor(scaleOScoped.get(), 81, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify stats tensor (optional)
    hipdnn_backend::ScopedDescriptor statsScoped;
    int64_t statsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &statsCount,
                       static_cast<void*>(statsScoped.getPtr()));
    ASSERT_EQ(statsCount, 1);
    ASSERT_NE(statsScoped.get(), nullptr);
    verifyTensorDescriptor(statsScoped.get(), 82, HIPDNN_DATA_FLOAT,
                           {1, 8, 128, 1},
                           {1024, 128, 1, 1});

    // Verify max_output tensor (optional)
    hipdnn_backend::ScopedDescriptor maxOutputScoped;
    int64_t maxOutputCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &maxOutputCount,
                       static_cast<void*>(maxOutputScoped.getPtr()));
    ASSERT_EQ(maxOutputCount, 1);
    ASSERT_NE(maxOutputScoped.get(), nullptr);
    verifyTensorDescriptor(maxOutputScoped.get(), 83, HIPDNN_DATA_FLOAT,
                           {1, 8, 128, 1},
                           {1024, 128, 1, 1});

    // Verify sum_exp tensor (optional)
    hipdnn_backend::ScopedDescriptor sumExpScoped;
    int64_t sumExpCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &sumExpCount,
                       static_cast<void*>(sumExpScoped.getPtr()));
    ASSERT_EQ(sumExpCount, 1);
    ASSERT_NE(sumExpScoped.get(), nullptr);
    verifyTensorDescriptor(sumExpScoped.get(), 84, HIPDNN_DATA_FLOAT,
                           {1, 8, 128, 1},
                           {1024, 128, 1, 1});

    // Verify rng_dump tensor (optional)
    hipdnn_backend::ScopedDescriptor rngDumpScoped;
    int64_t rngDumpCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &rngDumpCount,
                       static_cast<void*>(rngDumpScoped.getPtr()));
    ASSERT_EQ(rngDumpCount, 1);
    ASSERT_NE(rngDumpScoped.get(), nullptr);
    verifyTensorDescriptor(rngDumpScoped.get(), 85, HIPDNN_DATA_FLOAT,
                           {1, 8, 128, 128},
                           {131072, 16384, 128, 1});

    // Verify amax_s tensor (optional)
    hipdnn_backend::ScopedDescriptor amaxSScoped;
    int64_t amaxSCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &amaxSCount,
                       static_cast<void*>(amaxSScoped.getPtr()));
    ASSERT_EQ(amaxSCount, 1);
    ASSERT_NE(amaxSScoped.get(), nullptr);
    verifyTensorDescriptor(amaxSScoped.get(), 86, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

    // Verify amax_o tensor (optional)
    hipdnn_backend::ScopedDescriptor amaxOScoped;
    int64_t amaxOCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &amaxOCount,
                       static_cast<void*>(amaxOScoped.getPtr()));
    ASSERT_EQ(amaxOCount, 1);
    ASSERT_NE(amaxOScoped.get(), nullptr);
    verifyTensorDescriptor(amaxOScoped.get(), 87, HIPDNN_DATA_FLOAT,
                           {1, 1, 1, 1},
                           {1, 1, 1, 1});

}

TEST_F(TestSdpaFpropOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_sdpafprop_1";

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_sdpafprop_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_sdpafprop_1");
}

TEST_F(TestSdpaFpropOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestSdpaFpropOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = SdpaFpropOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
