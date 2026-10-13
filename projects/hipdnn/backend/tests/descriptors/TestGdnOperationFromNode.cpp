// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GdnOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/gdn_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/GdnConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_backend::test_utilities::verifyTensorDescriptor;
using hipdnn_tests::toVec;

// =============================================================================
// GdnOperationDescriptor::fromNode() Tests
// =============================================================================

class TestGdnOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT qAttrs;
        qAttrs.uid = K_GDN_TENSOR_Q_UID;
        qAttrs.data_type = DataType::FLOAT;
        qAttrs.dims = toVec(K_GDN_TENSOR_Q_DIMS);
        qAttrs.strides = toVec(K_GDN_TENSOR_Q_STRIDES);

        _tensorMap[K_GDN_TENSOR_Q_UID] = TensorDescriptor::fromFlatBuffer(qAttrs);
        TensorAttributesT kAttrs;
        kAttrs.uid = K_GDN_TENSOR_K_UID;
        kAttrs.data_type = DataType::FLOAT;
        kAttrs.dims = toVec(K_GDN_TENSOR_K_DIMS);
        kAttrs.strides = toVec(K_GDN_TENSOR_K_STRIDES);

        _tensorMap[K_GDN_TENSOR_K_UID] = TensorDescriptor::fromFlatBuffer(kAttrs);
        TensorAttributesT vAttrs;
        vAttrs.uid = K_GDN_TENSOR_V_UID;
        vAttrs.data_type = DataType::FLOAT;
        vAttrs.dims = toVec(K_GDN_TENSOR_V_DIMS);
        vAttrs.strides = toVec(K_GDN_TENSOR_V_STRIDES);

        _tensorMap[K_GDN_TENSOR_V_UID] = TensorDescriptor::fromFlatBuffer(vAttrs);
        TensorAttributesT gAttrs;
        gAttrs.uid = K_GDN_TENSOR_G_UID;
        gAttrs.data_type = DataType::FLOAT;
        gAttrs.dims = toVec(K_GDN_TENSOR_G_DIMS);
        gAttrs.strides = toVec(K_GDN_TENSOR_G_STRIDES);

        _tensorMap[K_GDN_TENSOR_G_UID] = TensorDescriptor::fromFlatBuffer(gAttrs);
        TensorAttributesT betaAttrs;
        betaAttrs.uid = K_GDN_TENSOR_BETA_UID;
        betaAttrs.data_type = DataType::FLOAT;
        betaAttrs.dims = toVec(K_GDN_TENSOR_BETA_DIMS);
        betaAttrs.strides = toVec(K_GDN_TENSOR_BETA_STRIDES);

        _tensorMap[K_GDN_TENSOR_BETA_UID] = TensorDescriptor::fromFlatBuffer(betaAttrs);
        TensorAttributesT cuSeqlensAttrs;
        cuSeqlensAttrs.uid = K_GDN_TENSOR_CU_SEQLENS_UID;
        cuSeqlensAttrs.data_type = DataType::INT32;
        cuSeqlensAttrs.dims = toVec(K_GDN_TENSOR_CU_SEQLENS_DIMS);
        cuSeqlensAttrs.strides = toVec(K_GDN_TENSOR_CU_SEQLENS_STRIDES);

        _tensorMap[K_GDN_TENSOR_CU_SEQLENS_UID] = TensorDescriptor::fromFlatBuffer(cuSeqlensAttrs);
        TensorAttributesT initialStateAttrs;
        initialStateAttrs.uid = K_GDN_TENSOR_INITIAL_STATE_UID;
        initialStateAttrs.data_type = DataType::FLOAT;
        initialStateAttrs.dims = toVec(K_GDN_TENSOR_INITIAL_STATE_DIMS);
        initialStateAttrs.strides = toVec(K_GDN_TENSOR_INITIAL_STATE_STRIDES);

        _tensorMap[K_GDN_TENSOR_INITIAL_STATE_UID]
            = TensorDescriptor::fromFlatBuffer(initialStateAttrs);
        TensorAttributesT aLogAttrs;
        aLogAttrs.uid = K_GDN_TENSOR_A_LOG_UID;
        aLogAttrs.data_type = DataType::FLOAT;
        aLogAttrs.dims = toVec(K_GDN_TENSOR_A_LOG_DIMS);
        aLogAttrs.strides = toVec(K_GDN_TENSOR_A_LOG_STRIDES);

        _tensorMap[K_GDN_TENSOR_A_LOG_UID] = TensorDescriptor::fromFlatBuffer(aLogAttrs);
        TensorAttributesT dtBiasAttrs;
        dtBiasAttrs.uid = K_GDN_TENSOR_DT_BIAS_UID;
        dtBiasAttrs.data_type = DataType::FLOAT;
        dtBiasAttrs.dims = toVec(K_GDN_TENSOR_DT_BIAS_DIMS);
        dtBiasAttrs.strides = toVec(K_GDN_TENSOR_DT_BIAS_STRIDES);

        _tensorMap[K_GDN_TENSOR_DT_BIAS_UID] = TensorDescriptor::fromFlatBuffer(dtBiasAttrs);
        TensorAttributesT stateIndicesAttrs;
        stateIndicesAttrs.uid = K_GDN_TENSOR_STATE_INDICES_UID;
        stateIndicesAttrs.data_type = DataType::INT32;
        stateIndicesAttrs.dims = toVec(K_GDN_TENSOR_STATE_INDICES_DIMS);
        stateIndicesAttrs.strides = toVec(K_GDN_TENSOR_STATE_INDICES_STRIDES);

        _tensorMap[K_GDN_TENSOR_STATE_INDICES_UID]
            = TensorDescriptor::fromFlatBuffer(stateIndicesAttrs);
        TensorAttributesT oAttrs;
        oAttrs.uid = K_GDN_TENSOR_O_UID;
        oAttrs.data_type = DataType::FLOAT;
        oAttrs.dims = toVec(K_GDN_TENSOR_O_DIMS);
        oAttrs.strides = toVec(K_GDN_TENSOR_O_STRIDES);

        _tensorMap[K_GDN_TENSOR_O_UID] = TensorDescriptor::fromFlatBuffer(oAttrs);
        TensorAttributesT finalStateAttrs;
        finalStateAttrs.uid = K_GDN_TENSOR_FINAL_STATE_UID;
        finalStateAttrs.data_type = DataType::FLOAT;
        finalStateAttrs.dims = toVec(K_GDN_TENSOR_FINAL_STATE_DIMS);
        finalStateAttrs.strides = toVec(K_GDN_TENSOR_FINAL_STATE_STRIDES);

        _tensorMap[K_GDN_TENSOR_FINAL_STATE_UID]
            = TensorDescriptor::fromFlatBuffer(finalStateAttrs);
        TensorAttributesT stateCheckpointsAttrs;
        stateCheckpointsAttrs.uid = K_GDN_TENSOR_STATE_CHECKPOINTS_UID;
        stateCheckpointsAttrs.data_type = DataType::FLOAT;
        stateCheckpointsAttrs.dims = toVec(K_GDN_TENSOR_STATE_CHECKPOINTS_DIMS);
        stateCheckpointsAttrs.strides = toVec(K_GDN_TENSOR_STATE_CHECKPOINTS_STRIDES);

        _tensorMap[K_GDN_TENSOR_STATE_CHECKPOINTS_UID]
            = TensorDescriptor::fromFlatBuffer(stateCheckpointsAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::GdnAttributesT createStandardGdnAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::GdnAttributesT attrs;
        attrs.q_tensor_uid = K_GDN_TENSOR_Q_UID;
        attrs.k_tensor_uid = K_GDN_TENSOR_K_UID;
        attrs.v_tensor_uid = K_GDN_TENSOR_V_UID;
        attrs.g_tensor_uid = K_GDN_TENSOR_G_UID;
        attrs.beta_tensor_uid = K_GDN_TENSOR_BETA_UID;
        attrs.cu_seqlens_tensor_uid = K_GDN_TENSOR_CU_SEQLENS_UID;
        attrs.initial_state_tensor_uid = K_GDN_TENSOR_INITIAL_STATE_UID;
        attrs.a_log_tensor_uid = K_GDN_TENSOR_A_LOG_UID;
        attrs.dt_bias_tensor_uid = K_GDN_TENSOR_DT_BIAS_UID;
        attrs.state_indices_tensor_uid = K_GDN_TENSOR_STATE_INDICES_UID;
        attrs.o_tensor_uid = K_GDN_TENSOR_O_UID;
        attrs.final_state_tensor_uid = K_GDN_TENSOR_FINAL_STATE_UID;
        attrs.state_checkpoints_tensor_uid = K_GDN_TENSOR_STATE_CHECKPOINTS_UID;
        attrs.scale = 0.35355339F;
        attrs.output_final_state = true;
        attrs.use_qk_l2norm = true;
        attrs.checkpoint_every_n_tokens = 4;
        attrs.use_beta_sigmoid = true;
        attrs.allow_neg_eigval = true;
        attrs.safe_gate = true;
        attrs.gate_domain = GateDomain::LOG;
        attrs.batch_invariant = true;
        attrs.overwrite_initial_state = false;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardGdnAttrs());
        return node;
    }
};

TEST_F(TestGdnOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_GDN_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().q_tensor_uid, K_GDN_TENSOR_Q_UID);
}

TEST_F(TestGdnOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    const auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::GdnAttributes);
    auto desc = std::static_pointer_cast<GdnOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().q_tensor_uid, K_GDN_TENSOR_Q_UID);
    EXPECT_EQ(desc->getData().k_tensor_uid, K_GDN_TENSOR_K_UID);
    EXPECT_EQ(desc->getData().v_tensor_uid, K_GDN_TENSOR_V_UID);
    EXPECT_EQ(desc->getData().g_tensor_uid, K_GDN_TENSOR_G_UID);
    EXPECT_EQ(desc->getData().beta_tensor_uid, K_GDN_TENSOR_BETA_UID);
    EXPECT_EQ(desc->getData().cu_seqlens_tensor_uid, K_GDN_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(desc->getData().initial_state_tensor_uid, K_GDN_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(desc->getData().a_log_tensor_uid, K_GDN_TENSOR_A_LOG_UID);
    EXPECT_EQ(desc->getData().dt_bias_tensor_uid, K_GDN_TENSOR_DT_BIAS_UID);
    EXPECT_EQ(desc->getData().state_indices_tensor_uid, K_GDN_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(desc->getData().o_tensor_uid, K_GDN_TENSOR_O_UID);
    EXPECT_EQ(desc->getData().final_state_tensor_uid, K_GDN_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(desc->getData().state_checkpoints_tensor_uid, K_GDN_TENSOR_STATE_CHECKPOINTS_UID);
    EXPECT_EQ(desc->getData().gate_domain, GateDomain::LOG);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getQDesc()->getData().uid, K_GDN_TENSOR_Q_UID);
    EXPECT_EQ(desc->getKDesc()->getData().uid, K_GDN_TENSOR_K_UID);
    EXPECT_EQ(desc->getVDesc()->getData().uid, K_GDN_TENSOR_V_UID);
    EXPECT_EQ(desc->getGDesc()->getData().uid, K_GDN_TENSOR_G_UID);
    EXPECT_EQ(desc->getBetaDesc()->getData().uid, K_GDN_TENSOR_BETA_UID);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().uid, K_GDN_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(desc->getODesc()->getData().uid, K_GDN_TENSOR_O_UID);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().uid, K_GDN_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(desc->getALogDesc()->getData().uid, K_GDN_TENSOR_A_LOG_UID);
    EXPECT_EQ(desc->getDtBiasDesc()->getData().uid, K_GDN_TENSOR_DT_BIAS_UID);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().uid, K_GDN_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().uid, K_GDN_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(desc->getStateCheckpointsDesc()->getData().uid, K_GDN_TENSOR_STATE_CHECKPOINTS_UID);
}

TEST_F(TestGdnOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestGdnOperationFromNode, PreservesGateDomain)
{
    auto node = createStandardNode();
    auto attrs = createStandardGdnAttrs();
    attrs.gate_domain = GateDomain::LINEAR;
    node.attributes.Set(attrs);
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().gate_domain, GateDomain::LINEAR);
}

TEST_F(TestGdnOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getQDesc(), nullptr);
    EXPECT_EQ(desc->getQDesc()->getData().uid, K_GDN_TENSOR_Q_UID);
    ASSERT_NE(desc->getKDesc(), nullptr);
    EXPECT_EQ(desc->getKDesc()->getData().uid, K_GDN_TENSOR_K_UID);
    ASSERT_NE(desc->getVDesc(), nullptr);
    EXPECT_EQ(desc->getVDesc()->getData().uid, K_GDN_TENSOR_V_UID);
    ASSERT_NE(desc->getGDesc(), nullptr);
    EXPECT_EQ(desc->getGDesc()->getData().uid, K_GDN_TENSOR_G_UID);
    ASSERT_NE(desc->getBetaDesc(), nullptr);
    EXPECT_EQ(desc->getBetaDesc()->getData().uid, K_GDN_TENSOR_BETA_UID);
    ASSERT_NE(desc->getCuSeqlensDesc(), nullptr);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().uid, K_GDN_TENSOR_CU_SEQLENS_UID);
    ASSERT_NE(desc->getODesc(), nullptr);
    EXPECT_EQ(desc->getODesc()->getData().uid, K_GDN_TENSOR_O_UID);
    ASSERT_NE(desc->getInitialStateDesc(), nullptr);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().uid, K_GDN_TENSOR_INITIAL_STATE_UID);
    ASSERT_NE(desc->getALogDesc(), nullptr);
    EXPECT_EQ(desc->getALogDesc()->getData().uid, K_GDN_TENSOR_A_LOG_UID);
    ASSERT_NE(desc->getDtBiasDesc(), nullptr);
    EXPECT_EQ(desc->getDtBiasDesc()->getData().uid, K_GDN_TENSOR_DT_BIAS_UID);
    ASSERT_NE(desc->getStateIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().uid, K_GDN_TENSOR_STATE_INDICES_UID);
    ASSERT_NE(desc->getFinalStateDesc(), nullptr);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().uid, K_GDN_TENSOR_FINAL_STATE_UID);
    ASSERT_NE(desc->getStateCheckpointsDesc(), nullptr);
    EXPECT_EQ(desc->getStateCheckpointsDesc()->getData().uid, K_GDN_TENSOR_STATE_CHECKPOINTS_UID);
}

TEST_F(TestGdnOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getQDesc(), _tensorMap[K_GDN_TENSOR_Q_UID]);
    EXPECT_EQ(desc->getKDesc(), _tensorMap[K_GDN_TENSOR_K_UID]);
    EXPECT_EQ(desc->getVDesc(), _tensorMap[K_GDN_TENSOR_V_UID]);
    EXPECT_EQ(desc->getGDesc(), _tensorMap[K_GDN_TENSOR_G_UID]);
    EXPECT_EQ(desc->getBetaDesc(), _tensorMap[K_GDN_TENSOR_BETA_UID]);
    EXPECT_EQ(desc->getCuSeqlensDesc(), _tensorMap[K_GDN_TENSOR_CU_SEQLENS_UID]);
    EXPECT_EQ(desc->getODesc(), _tensorMap[K_GDN_TENSOR_O_UID]);
    EXPECT_EQ(desc->getInitialStateDesc(), _tensorMap[K_GDN_TENSOR_INITIAL_STATE_UID]);
    EXPECT_EQ(desc->getALogDesc(), _tensorMap[K_GDN_TENSOR_A_LOG_UID]);
    EXPECT_EQ(desc->getDtBiasDesc(), _tensorMap[K_GDN_TENSOR_DT_BIAS_UID]);
    EXPECT_EQ(desc->getStateIndicesDesc(), _tensorMap[K_GDN_TENSOR_STATE_INDICES_UID]);
    EXPECT_EQ(desc->getFinalStateDesc(), _tensorMap[K_GDN_TENSOR_FINAL_STATE_UID]);
    EXPECT_EQ(desc->getStateCheckpointsDesc(), _tensorMap[K_GDN_TENSOR_STATE_CHECKPOINTS_UID]);
}

TEST_F(TestGdnOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getQDesc(), nullptr);
    EXPECT_EQ(desc->getQDesc()->getData().uid, K_GDN_TENSOR_Q_UID);
    EXPECT_EQ(desc->getQDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getQDesc()->getData().dims, (std::vector<int64_t>{8, 2, 8}));
    EXPECT_EQ(desc->getQDesc()->getData().strides, (std::vector<int64_t>{16, 8, 1}));

    ASSERT_NE(desc->getKDesc(), nullptr);
    EXPECT_EQ(desc->getKDesc()->getData().uid, K_GDN_TENSOR_K_UID);
    EXPECT_EQ(desc->getKDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getKDesc()->getData().dims, (std::vector<int64_t>{8, 2, 8}));
    EXPECT_EQ(desc->getKDesc()->getData().strides, (std::vector<int64_t>{16, 8, 1}));

    ASSERT_NE(desc->getVDesc(), nullptr);
    EXPECT_EQ(desc->getVDesc()->getData().uid, K_GDN_TENSOR_V_UID);
    EXPECT_EQ(desc->getVDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getVDesc()->getData().dims, (std::vector<int64_t>{8, 2, 8}));
    EXPECT_EQ(desc->getVDesc()->getData().strides, (std::vector<int64_t>{16, 8, 1}));

    ASSERT_NE(desc->getGDesc(), nullptr);
    EXPECT_EQ(desc->getGDesc()->getData().uid, K_GDN_TENSOR_G_UID);
    EXPECT_EQ(desc->getGDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getGDesc()->getData().dims, (std::vector<int64_t>{8, 2}));
    EXPECT_EQ(desc->getGDesc()->getData().strides, (std::vector<int64_t>{2, 1}));

    ASSERT_NE(desc->getBetaDesc(), nullptr);
    EXPECT_EQ(desc->getBetaDesc()->getData().uid, K_GDN_TENSOR_BETA_UID);
    EXPECT_EQ(desc->getBetaDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBetaDesc()->getData().dims, (std::vector<int64_t>{8, 2}));
    EXPECT_EQ(desc->getBetaDesc()->getData().strides, (std::vector<int64_t>{2, 1}));

    ASSERT_NE(desc->getCuSeqlensDesc(), nullptr);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().uid, K_GDN_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().dims, (std::vector<int64_t>{3}));
    EXPECT_EQ(desc->getCuSeqlensDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getODesc(), nullptr);
    EXPECT_EQ(desc->getODesc()->getData().uid, K_GDN_TENSOR_O_UID);
    EXPECT_EQ(desc->getODesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getODesc()->getData().dims, (std::vector<int64_t>{8, 2, 8}));
    EXPECT_EQ(desc->getODesc()->getData().strides, (std::vector<int64_t>{16, 8, 1}));

    ASSERT_NE(desc->getInitialStateDesc(), nullptr);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().uid, K_GDN_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInitialStateDesc()->getData().dims, (std::vector<int64_t>{2, 2, 8, 8}));
    EXPECT_EQ(desc->getInitialStateDesc()->getData().strides,
              (std::vector<int64_t>{128, 64, 8, 1}));

    ASSERT_NE(desc->getALogDesc(), nullptr);
    EXPECT_EQ(desc->getALogDesc()->getData().uid, K_GDN_TENSOR_A_LOG_UID);
    EXPECT_EQ(desc->getALogDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getALogDesc()->getData().dims, (std::vector<int64_t>{2}));
    EXPECT_EQ(desc->getALogDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getDtBiasDesc(), nullptr);
    EXPECT_EQ(desc->getDtBiasDesc()->getData().uid, K_GDN_TENSOR_DT_BIAS_UID);
    EXPECT_EQ(desc->getDtBiasDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDtBiasDesc()->getData().dims, (std::vector<int64_t>{2}));
    EXPECT_EQ(desc->getDtBiasDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getStateIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().uid, K_GDN_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().dims, (std::vector<int64_t>{2}));
    EXPECT_EQ(desc->getStateIndicesDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getFinalStateDesc(), nullptr);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().uid, K_GDN_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getFinalStateDesc()->getData().dims, (std::vector<int64_t>{2, 2, 8, 8}));
    EXPECT_EQ(desc->getFinalStateDesc()->getData().strides, (std::vector<int64_t>{128, 64, 8, 1}));

    ASSERT_NE(desc->getStateCheckpointsDesc(), nullptr);
    EXPECT_EQ(desc->getStateCheckpointsDesc()->getData().uid, K_GDN_TENSOR_STATE_CHECKPOINTS_UID);
    EXPECT_EQ(desc->getStateCheckpointsDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getStateCheckpointsDesc()->getData().dims, (std::vector<int64_t>{4, 2, 8, 8}));
    EXPECT_EQ(desc->getStateCheckpointsDesc()->getData().strides,
              (std::vector<int64_t>{128, 64, 8, 1}));
}

// The parameter is the UID of the one required tensor to drop from the tensor map.
class TestGdnOperationFromNodeMissingTensor : public TestGdnOperationFromNode,
                                              public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestGdnOperationFromNodeMissingTensor, FailsWithMissingTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(GdnOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(RequiredTensors,
                         TestGdnOperationFromNodeMissingTensor,
                         ::testing::Values(K_GDN_TENSOR_Q_UID,
                                           K_GDN_TENSOR_K_UID,
                                           K_GDN_TENSOR_V_UID,
                                           K_GDN_TENSOR_G_UID,
                                           K_GDN_TENSOR_BETA_UID,
                                           K_GDN_TENSOR_CU_SEQLENS_UID,
                                           K_GDN_TENSOR_O_UID));

TEST_F(TestGdnOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardGdnAttrs();
    attrs.initial_state_tensor_uid = flatbuffers::nullopt;
    attrs.a_log_tensor_uid = flatbuffers::nullopt;
    attrs.dt_bias_tensor_uid = flatbuffers::nullopt;
    attrs.state_indices_tensor_uid = flatbuffers::nullopt;
    attrs.final_state_tensor_uid = flatbuffers::nullopt;
    attrs.state_checkpoints_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getQDesc(), nullptr);
    EXPECT_NE(desc->getKDesc(), nullptr);
    EXPECT_NE(desc->getVDesc(), nullptr);
    EXPECT_NE(desc->getGDesc(), nullptr);
    EXPECT_NE(desc->getBetaDesc(), nullptr);
    EXPECT_NE(desc->getCuSeqlensDesc(), nullptr);
    EXPECT_NE(desc->getODesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getInitialStateDesc(), nullptr);
    EXPECT_EQ(desc->getALogDesc(), nullptr);
    EXPECT_EQ(desc->getDtBiasDesc(), nullptr);
    EXPECT_EQ(desc->getStateIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getFinalStateDesc(), nullptr);
    EXPECT_EQ(desc->getStateCheckpointsDesc(), nullptr);

    const auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 7u);
    EXPECT_EQ(tensors[0], desc->getQDesc());
    EXPECT_EQ(tensors[1], desc->getKDesc());
    EXPECT_EQ(tensors[2], desc->getVDesc());
    EXPECT_EQ(tensors[3], desc->getGDesc());
    EXPECT_EQ(tensors[4], desc->getBetaDesc());
    EXPECT_EQ(tensors[5], desc->getCuSeqlensDesc());
    EXPECT_EQ(tensors[6], desc->getODesc());
}

TEST_F(TestGdnOperationFromNode, FailsWhenOptionalInitialStateUidSetButTensorMissing)
{
    _tensorMap.erase(K_GDN_TENSOR_INITIAL_STATE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(GdnOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestGdnOperationFromNode, FailsWhenOptionalALogUidSetButTensorMissing)
{
    _tensorMap.erase(K_GDN_TENSOR_A_LOG_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(GdnOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestGdnOperationFromNode, FailsWhenOptionalDtBiasUidSetButTensorMissing)
{
    _tensorMap.erase(K_GDN_TENSOR_DT_BIAS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(GdnOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestGdnOperationFromNode, FailsWhenOptionalStateIndicesUidSetButTensorMissing)
{
    _tensorMap.erase(K_GDN_TENSOR_STATE_INDICES_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(GdnOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestGdnOperationFromNode, FailsWhenOptionalFinalStateUidSetButTensorMissing)
{
    _tensorMap.erase(K_GDN_TENSOR_FINAL_STATE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(GdnOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestGdnOperationFromNode, FailsWhenOptionalStateCheckpointsUidSetButTensorMissing)
{
    _tensorMap.erase(K_GDN_TENSOR_STATE_CHECKPOINTS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(GdnOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestGdnOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 13);
    EXPECT_EQ(tensors[0]->getData().uid, K_GDN_TENSOR_Q_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_GDN_TENSOR_K_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_GDN_TENSOR_V_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_GDN_TENSOR_G_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_GDN_TENSOR_BETA_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_GDN_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(tensors[6]->getData().uid, K_GDN_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(tensors[7]->getData().uid, K_GDN_TENSOR_A_LOG_UID);
    EXPECT_EQ(tensors[8]->getData().uid, K_GDN_TENSOR_DT_BIAS_UID);
    EXPECT_EQ(tensors[9]->getData().uid, K_GDN_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(tensors[10]->getData().uid, K_GDN_TENSOR_O_UID);
    EXPECT_EQ(tensors[11]->getData().uid, K_GDN_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(tensors[12]->getData().uid, K_GDN_TENSOR_STATE_CHECKPOINTS_UID);
}

TEST_F(TestGdnOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::GdnAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->q_tensor_uid, K_GDN_TENSOR_Q_UID);
    EXPECT_EQ(rebuiltAttrs->k_tensor_uid, K_GDN_TENSOR_K_UID);
    EXPECT_EQ(rebuiltAttrs->v_tensor_uid, K_GDN_TENSOR_V_UID);
    EXPECT_EQ(rebuiltAttrs->g_tensor_uid, K_GDN_TENSOR_G_UID);
    EXPECT_EQ(rebuiltAttrs->beta_tensor_uid, K_GDN_TENSOR_BETA_UID);
    EXPECT_EQ(rebuiltAttrs->cu_seqlens_tensor_uid, K_GDN_TENSOR_CU_SEQLENS_UID);
    EXPECT_EQ(rebuiltAttrs->initial_state_tensor_uid, K_GDN_TENSOR_INITIAL_STATE_UID);
    EXPECT_EQ(rebuiltAttrs->a_log_tensor_uid, K_GDN_TENSOR_A_LOG_UID);
    EXPECT_EQ(rebuiltAttrs->dt_bias_tensor_uid, K_GDN_TENSOR_DT_BIAS_UID);
    EXPECT_EQ(rebuiltAttrs->state_indices_tensor_uid, K_GDN_TENSOR_STATE_INDICES_UID);
    EXPECT_EQ(rebuiltAttrs->o_tensor_uid, K_GDN_TENSOR_O_UID);
    EXPECT_EQ(rebuiltAttrs->final_state_tensor_uid, K_GDN_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(rebuiltAttrs->state_checkpoints_tensor_uid, K_GDN_TENSOR_STATE_CHECKPOINTS_UID);
    EXPECT_EQ(rebuiltAttrs->gate_domain, GateDomain::LOG);
}

TEST_F(TestGdnOperationFromNode, FromNodePreservesScale)
{
    auto attrs = createStandardGdnAttrs();
    attrs.scale = 0.5F;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().scale.has_value());
    EXPECT_FLOAT_EQ(desc->getData().scale.value(), 0.5F);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->scale.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->scale.value(), 0.5F);
}

TEST_F(TestGdnOperationFromNode, FromNodePreservesOutputFinalState)
{
    auto attrs = createStandardGdnAttrs();
    attrs.output_final_state = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getData().output_final_state, true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->output_final_state, true);
}

TEST_F(TestGdnOperationFromNode, FromNodePreservesUseQkL2norm)
{
    auto attrs = createStandardGdnAttrs();
    attrs.use_qk_l2norm = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getData().use_qk_l2norm, true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->use_qk_l2norm, true);
}

TEST_F(TestGdnOperationFromNode, FromNodePreservesUseBetaSigmoid)
{
    auto attrs = createStandardGdnAttrs();
    attrs.use_beta_sigmoid = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getData().use_beta_sigmoid, true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->use_beta_sigmoid, true);
}

TEST_F(TestGdnOperationFromNode, FromNodePreservesAllowNegEigval)
{
    auto attrs = createStandardGdnAttrs();
    attrs.allow_neg_eigval = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getData().allow_neg_eigval, true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->allow_neg_eigval, true);
}

TEST_F(TestGdnOperationFromNode, FromNodePreservesSafeGate)
{
    auto attrs = createStandardGdnAttrs();
    attrs.safe_gate = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getData().safe_gate, true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->safe_gate, true);
}

TEST_F(TestGdnOperationFromNode, FromNodePreservesBatchInvariant)
{
    auto attrs = createStandardGdnAttrs();
    attrs.batch_invariant = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getData().batch_invariant, true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->batch_invariant, true);
}

TEST_F(TestGdnOperationFromNode, FromNodePreservesOverwriteInitialState)
{
    auto attrs = createStandardGdnAttrs();
    attrs.overwrite_initial_state = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getData().overwrite_initial_state, true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->overwrite_initial_state, true);
}

TEST_F(TestGdnOperationFromNode, BuildNodeOmitsUnsetOptionalScalars)
{
    auto node = createStandardNode();
    // createStandardNode() sets every scalar; clear the optional ones for this test.
    node.attributes.AsGdnAttributes()->scale.reset();
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGdnAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    EXPECT_FALSE(rebuiltAttrs->scale.has_value());
}

TEST_F(TestGdnOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_GDN_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify gate_domain
    hipdnnGateDomain_t gateDomain = {};
    int64_t gateDomainCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_GATE_DOMAIN_EXT,
                       HIPDNN_TYPE_GATE_DOMAIN_EXT,
                       1,
                       &gateDomainCount,
                       &gateDomain);
    ASSERT_EQ(gateDomain, HIPDNN_GATE_DOMAIN_LOG_EXT);

    // Verify q tensor
    hipdnn_backend::ScopedDescriptor qScoped;
    int64_t qCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_Q_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &qCount,
                       static_cast<void*>(qScoped.getPtr()));
    ASSERT_EQ(qCount, 1);
    ASSERT_NE(qScoped.get(), nullptr);
    verifyTensorDescriptor(
        qScoped.get(), K_GDN_TENSOR_Q_UID, HIPDNN_DATA_FLOAT, {8, 2, 8}, {16, 8, 1});

    // Verify k tensor
    hipdnn_backend::ScopedDescriptor kScoped;
    int64_t kCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_K_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &kCount,
                       static_cast<void*>(kScoped.getPtr()));
    ASSERT_EQ(kCount, 1);
    ASSERT_NE(kScoped.get(), nullptr);
    verifyTensorDescriptor(
        kScoped.get(), K_GDN_TENSOR_K_UID, HIPDNN_DATA_FLOAT, {8, 2, 8}, {16, 8, 1});

    // Verify v tensor
    hipdnn_backend::ScopedDescriptor vScoped;
    int64_t vCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_V_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &vCount,
                       static_cast<void*>(vScoped.getPtr()));
    ASSERT_EQ(vCount, 1);
    ASSERT_NE(vScoped.get(), nullptr);
    verifyTensorDescriptor(
        vScoped.get(), K_GDN_TENSOR_V_UID, HIPDNN_DATA_FLOAT, {8, 2, 8}, {16, 8, 1});

    // Verify g tensor
    hipdnn_backend::ScopedDescriptor gScoped;
    int64_t gCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_G_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &gCount,
                       static_cast<void*>(gScoped.getPtr()));
    ASSERT_EQ(gCount, 1);
    ASSERT_NE(gScoped.get(), nullptr);
    verifyTensorDescriptor(gScoped.get(), K_GDN_TENSOR_G_UID, HIPDNN_DATA_FLOAT, {8, 2}, {2, 1});

    // Verify beta tensor
    hipdnn_backend::ScopedDescriptor betaScoped;
    int64_t betaCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_BETA_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &betaCount,
                       static_cast<void*>(betaScoped.getPtr()));
    ASSERT_EQ(betaCount, 1);
    ASSERT_NE(betaScoped.get(), nullptr);
    verifyTensorDescriptor(
        betaScoped.get(), K_GDN_TENSOR_BETA_UID, HIPDNN_DATA_FLOAT, {8, 2}, {2, 1});

    // Verify cu_seqlens tensor
    hipdnn_backend::ScopedDescriptor cuSeqlensScoped;
    int64_t cuSeqlensCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_CU_SEQLENS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &cuSeqlensCount,
                       static_cast<void*>(cuSeqlensScoped.getPtr()));
    ASSERT_EQ(cuSeqlensCount, 1);
    ASSERT_NE(cuSeqlensScoped.get(), nullptr);
    verifyTensorDescriptor(
        cuSeqlensScoped.get(), K_GDN_TENSOR_CU_SEQLENS_UID, HIPDNN_DATA_INT32, {3}, {1});

    // Verify o tensor
    hipdnn_backend::ScopedDescriptor oScoped;
    int64_t oCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_O_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &oCount,
                       static_cast<void*>(oScoped.getPtr()));
    ASSERT_EQ(oCount, 1);
    ASSERT_NE(oScoped.get(), nullptr);
    verifyTensorDescriptor(
        oScoped.get(), K_GDN_TENSOR_O_UID, HIPDNN_DATA_FLOAT, {8, 2, 8}, {16, 8, 1});

    // Verify initial_state tensor (optional)
    hipdnn_backend::ScopedDescriptor initialStateScoped;
    int64_t initialStateCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_INITIAL_STATE_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &initialStateCount,
                       static_cast<void*>(initialStateScoped.getPtr()));
    ASSERT_EQ(initialStateCount, 1);
    ASSERT_NE(initialStateScoped.get(), nullptr);
    verifyTensorDescriptor(initialStateScoped.get(),
                           K_GDN_TENSOR_INITIAL_STATE_UID,
                           HIPDNN_DATA_FLOAT,
                           {2, 2, 8, 8},
                           {128, 64, 8, 1});

    // Verify a_log tensor (optional)
    hipdnn_backend::ScopedDescriptor aLogScoped;
    int64_t aLogCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_A_LOG_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &aLogCount,
                       static_cast<void*>(aLogScoped.getPtr()));
    ASSERT_EQ(aLogCount, 1);
    ASSERT_NE(aLogScoped.get(), nullptr);
    verifyTensorDescriptor(aLogScoped.get(), K_GDN_TENSOR_A_LOG_UID, HIPDNN_DATA_FLOAT, {2}, {1});

    // Verify dt_bias tensor (optional)
    hipdnn_backend::ScopedDescriptor dtBiasScoped;
    int64_t dtBiasCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_DT_BIAS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dtBiasCount,
                       static_cast<void*>(dtBiasScoped.getPtr()));
    ASSERT_EQ(dtBiasCount, 1);
    ASSERT_NE(dtBiasScoped.get(), nullptr);
    verifyTensorDescriptor(
        dtBiasScoped.get(), K_GDN_TENSOR_DT_BIAS_UID, HIPDNN_DATA_FLOAT, {2}, {1});

    // Verify state_indices tensor (optional)
    hipdnn_backend::ScopedDescriptor stateIndicesScoped;
    int64_t stateIndicesCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_STATE_INDICES_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &stateIndicesCount,
                       static_cast<void*>(stateIndicesScoped.getPtr()));
    ASSERT_EQ(stateIndicesCount, 1);
    ASSERT_NE(stateIndicesScoped.get(), nullptr);
    verifyTensorDescriptor(
        stateIndicesScoped.get(), K_GDN_TENSOR_STATE_INDICES_UID, HIPDNN_DATA_INT32, {2}, {1});

    // Verify final_state tensor (optional)
    hipdnn_backend::ScopedDescriptor finalStateScoped;
    int64_t finalStateCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_FINAL_STATE_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &finalStateCount,
                       static_cast<void*>(finalStateScoped.getPtr()));
    ASSERT_EQ(finalStateCount, 1);
    ASSERT_NE(finalStateScoped.get(), nullptr);
    verifyTensorDescriptor(finalStateScoped.get(),
                           K_GDN_TENSOR_FINAL_STATE_UID,
                           HIPDNN_DATA_FLOAT,
                           {2, 2, 8, 8},
                           {128, 64, 8, 1});

    // Verify state_checkpoints tensor (optional)
    hipdnn_backend::ScopedDescriptor stateCheckpointsScoped;
    int64_t stateCheckpointsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GDN_STATE_CHECKPOINTS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &stateCheckpointsCount,
                       static_cast<void*>(stateCheckpointsScoped.getPtr()));
    ASSERT_EQ(stateCheckpointsCount, 1);
    ASSERT_NE(stateCheckpointsScoped.get(), nullptr);
    verifyTensorDescriptor(stateCheckpointsScoped.get(),
                           K_GDN_TENSOR_STATE_CHECKPOINTS_UID,
                           HIPDNN_DATA_FLOAT,
                           {4, 2, 8, 8},
                           {128, 64, 8, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_GDN_EXT);
}

TEST_F(TestGdnOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_gdn_1";

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_gdn_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_gdn_1");
}

TEST_F(TestGdnOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestGdnOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = GdnOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
