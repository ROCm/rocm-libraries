// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/GdnAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestGdnAttributes ---

TEST(TestGdnAttributes, CreateGdnAttributes)
{
    GdnAttributes attrs;

    // Set all tensors
    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_uid(2000);
    attrs.set_q(qTensor);
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_uid(2001);
    attrs.set_k(kTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_uid(2002);
    attrs.set_v(vTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_uid(2003);
    attrs.set_g(gTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_uid(2004);
    attrs.set_beta(betaTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_uid(2005);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto initialStateTensor = std::make_shared<TensorAttributes>();
    initialStateTensor->set_uid(2006);
    attrs.set_initial_state(initialStateTensor);
    auto aLogTensor = std::make_shared<TensorAttributes>();
    aLogTensor->set_uid(2007);
    attrs.set_a_log(aLogTensor);
    auto dtBiasTensor = std::make_shared<TensorAttributes>();
    dtBiasTensor->set_uid(2008);
    attrs.set_dt_bias(dtBiasTensor);
    auto stateIndicesTensor = std::make_shared<TensorAttributes>();
    stateIndicesTensor->set_uid(2009);
    attrs.set_state_indices(stateIndicesTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_uid(2010);
    attrs.set_o(oTensor);
    auto finalStateTensor = std::make_shared<TensorAttributes>();
    finalStateTensor->set_uid(2011);
    attrs.set_final_state(finalStateTensor);
    auto stateCheckpointsTensor = std::make_shared<TensorAttributes>();
    stateCheckpointsTensor->set_uid(2012);
    attrs.set_state_checkpoints(stateCheckpointsTensor);

    // Set data fields
    attrs.set_scale(0.35355339F);
    attrs.set_output_final_state(true);
    attrs.set_use_qk_l2norm(true);
    attrs.set_checkpoint_every_n_tokens(4);
    attrs.set_use_beta_sigmoid(true);
    attrs.set_allow_neg_eigval(true);
    attrs.set_safe_gate(true);
    attrs.set_gate_domain(GateDomain::LOG);
    attrs.set_batch_invariant(true);
    attrs.set_overwrite_initial_state(false);

    // Verify tensor getters
    EXPECT_NE(attrs.get_q(), nullptr);
    EXPECT_EQ(attrs.get_q()->get_uid(), 2000);
    EXPECT_NE(attrs.get_k(), nullptr);
    EXPECT_EQ(attrs.get_k()->get_uid(), 2001);
    EXPECT_NE(attrs.get_v(), nullptr);
    EXPECT_EQ(attrs.get_v()->get_uid(), 2002);
    EXPECT_NE(attrs.get_g(), nullptr);
    EXPECT_EQ(attrs.get_g()->get_uid(), 2003);
    EXPECT_NE(attrs.get_beta(), nullptr);
    EXPECT_EQ(attrs.get_beta()->get_uid(), 2004);
    EXPECT_NE(attrs.get_cu_seqlens(), nullptr);
    EXPECT_EQ(attrs.get_cu_seqlens()->get_uid(), 2005);
    EXPECT_NE(attrs.get_initial_state(), nullptr);
    EXPECT_EQ(attrs.get_initial_state()->get_uid(), 2006);
    EXPECT_NE(attrs.get_a_log(), nullptr);
    EXPECT_EQ(attrs.get_a_log()->get_uid(), 2007);
    EXPECT_NE(attrs.get_dt_bias(), nullptr);
    EXPECT_EQ(attrs.get_dt_bias()->get_uid(), 2008);
    EXPECT_NE(attrs.get_state_indices(), nullptr);
    EXPECT_EQ(attrs.get_state_indices()->get_uid(), 2009);
    EXPECT_NE(attrs.get_o(), nullptr);
    EXPECT_EQ(attrs.get_o()->get_uid(), 2010);
    EXPECT_NE(attrs.get_final_state(), nullptr);
    EXPECT_EQ(attrs.get_final_state()->get_uid(), 2011);
    EXPECT_NE(attrs.get_state_checkpoints(), nullptr);
    EXPECT_EQ(attrs.get_state_checkpoints()->get_uid(), 2012);

    // Verify data field getters
    EXPECT_EQ(attrs.get_scale(), 0.35355339F);
    EXPECT_EQ(attrs.get_output_final_state(), true);
    EXPECT_EQ(attrs.get_use_qk_l2norm(), true);
    EXPECT_EQ(attrs.get_checkpoint_every_n_tokens(), 4);
    EXPECT_EQ(attrs.get_use_beta_sigmoid(), true);
    EXPECT_EQ(attrs.get_allow_neg_eigval(), true);
    EXPECT_EQ(attrs.get_safe_gate(), true);
    EXPECT_EQ(attrs.get_batch_invariant(), true);
    EXPECT_EQ(attrs.get_overwrite_initial_state(), false);
}

TEST(TestGdnAttributes, DefaultValues)
{
    const GdnAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_q(), nullptr);
    EXPECT_EQ(attrs.get_k(), nullptr);
    EXPECT_EQ(attrs.get_v(), nullptr);
    EXPECT_EQ(attrs.get_g(), nullptr);
    EXPECT_EQ(attrs.get_beta(), nullptr);
    EXPECT_EQ(attrs.get_cu_seqlens(), nullptr);
    EXPECT_EQ(attrs.get_initial_state(), nullptr);
    EXPECT_EQ(attrs.get_a_log(), nullptr);
    EXPECT_EQ(attrs.get_dt_bias(), nullptr);
    EXPECT_EQ(attrs.get_state_indices(), nullptr);
    EXPECT_EQ(attrs.get_o(), nullptr);
    EXPECT_EQ(attrs.get_final_state(), nullptr);
    EXPECT_EQ(attrs.get_state_checkpoints(), nullptr);

    // Vector fields should be empty by default
    EXPECT_FALSE(attrs.get_scale().has_value());
    EXPECT_EQ(attrs.get_output_final_state(), 0);
    EXPECT_EQ(attrs.get_use_qk_l2norm(), 0);
    EXPECT_EQ(attrs.get_checkpoint_every_n_tokens(), 0);
    EXPECT_EQ(attrs.get_use_beta_sigmoid(), 0);
    EXPECT_EQ(attrs.get_allow_neg_eigval(), 0);
    EXPECT_EQ(attrs.get_safe_gate(), 0);
    EXPECT_EQ(attrs.get_gate_domain(), GateDomain::LOG);
    EXPECT_EQ(attrs.get_batch_invariant(), 0);
    EXPECT_EQ(attrs.get_overwrite_initial_state(), 0);
}

TEST(TestGdnAttributes, SetQMove)
{
    GdnAttributes attrs;

    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_uid(2000).set_name("MovedQTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = qTensor.get();

    attrs.set_q(std::move(qTensor));

    // After move, original should be nullptr
    EXPECT_EQ(qTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_q();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetKMove)
{
    GdnAttributes attrs;

    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_uid(2001).set_name("MovedKTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = kTensor.get();

    attrs.set_k(std::move(kTensor));

    // After move, original should be nullptr
    EXPECT_EQ(kTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_k();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetVMove)
{
    GdnAttributes attrs;

    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_uid(2002).set_name("MovedVTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = vTensor.get();

    attrs.set_v(std::move(vTensor));

    // After move, original should be nullptr
    EXPECT_EQ(vTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_v();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetGMove)
{
    GdnAttributes attrs;

    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_uid(2003).set_name("MovedGTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = gTensor.get();

    attrs.set_g(std::move(gTensor));

    // After move, original should be nullptr
    EXPECT_EQ(gTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_g();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetBetaMove)
{
    GdnAttributes attrs;

    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_uid(2004)
        .set_name("MovedBetaTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = betaTensor.get();

    attrs.set_beta(std::move(betaTensor));

    // After move, original should be nullptr
    EXPECT_EQ(betaTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_beta();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetCuSeqlensMove)
{
    GdnAttributes attrs;

    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_uid(2005)
        .set_name("MovedCuSeqlensTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = cuSeqlensTensor.get();

    attrs.set_cu_seqlens(std::move(cuSeqlensTensor));

    // After move, original should be nullptr
    EXPECT_EQ(cuSeqlensTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_cu_seqlens();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetInitialStateMove)
{
    GdnAttributes attrs;

    auto initialStateTensor = std::make_shared<TensorAttributes>();
    initialStateTensor->set_uid(2006)
        .set_name("MovedInitialStateTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = initialStateTensor.get();

    attrs.set_initial_state(std::move(initialStateTensor));

    // After move, original should be nullptr
    EXPECT_EQ(initialStateTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_initial_state();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetALogMove)
{
    GdnAttributes attrs;

    auto aLogTensor = std::make_shared<TensorAttributes>();
    aLogTensor->set_uid(2007)
        .set_name("MovedALogTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = aLogTensor.get();

    attrs.set_a_log(std::move(aLogTensor));

    // After move, original should be nullptr
    EXPECT_EQ(aLogTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_a_log();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetDtBiasMove)
{
    GdnAttributes attrs;

    auto dtBiasTensor = std::make_shared<TensorAttributes>();
    dtBiasTensor->set_uid(2008)
        .set_name("MovedDtBiasTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dtBiasTensor.get();

    attrs.set_dt_bias(std::move(dtBiasTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dtBiasTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dt_bias();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetStateIndicesMove)
{
    GdnAttributes attrs;

    auto stateIndicesTensor = std::make_shared<TensorAttributes>();
    stateIndicesTensor->set_uid(2009)
        .set_name("MovedStateIndicesTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = stateIndicesTensor.get();

    attrs.set_state_indices(std::move(stateIndicesTensor));

    // After move, original should be nullptr
    EXPECT_EQ(stateIndicesTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_state_indices();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetOMove)
{
    GdnAttributes attrs;

    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_uid(2010).set_name("MovedOTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = oTensor.get();

    attrs.set_o(std::move(oTensor));

    // After move, original should be nullptr
    EXPECT_EQ(oTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_o();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetFinalStateMove)
{
    GdnAttributes attrs;

    auto finalStateTensor = std::make_shared<TensorAttributes>();
    finalStateTensor->set_uid(2011)
        .set_name("MovedFinalStateTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = finalStateTensor.get();

    attrs.set_final_state(std::move(finalStateTensor));

    // After move, original should be nullptr
    EXPECT_EQ(finalStateTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_final_state();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetStateCheckpointsMove)
{
    GdnAttributes attrs;

    auto stateCheckpointsTensor = std::make_shared<TensorAttributes>();
    stateCheckpointsTensor->set_uid(2012)
        .set_name("MovedStateCheckpointsTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = stateCheckpointsTensor.get();

    attrs.set_state_checkpoints(std::move(stateCheckpointsTensor));

    // After move, original should be nullptr
    EXPECT_EQ(stateCheckpointsTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_state_checkpoints();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestGdnAttributes, SetTensorsConstRef)
{
    GdnAttributes attrs;

    // Create tensors
    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_uid(2000).set_name("QConstRef");
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_uid(2001).set_name("KConstRef");
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_uid(2002).set_name("VConstRef");
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_uid(2003).set_name("GConstRef");
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_uid(2004).set_name("BetaConstRef");
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_uid(2005).set_name("CuSeqlensConstRef");
    auto initialStateTensor = std::make_shared<TensorAttributes>();
    initialStateTensor->set_uid(2006).set_name("InitialStateConstRef");
    auto aLogTensor = std::make_shared<TensorAttributes>();
    aLogTensor->set_uid(2007).set_name("ALogConstRef");
    auto dtBiasTensor = std::make_shared<TensorAttributes>();
    dtBiasTensor->set_uid(2008).set_name("DtBiasConstRef");
    auto stateIndicesTensor = std::make_shared<TensorAttributes>();
    stateIndicesTensor->set_uid(2009).set_name("StateIndicesConstRef");
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_uid(2010).set_name("OConstRef");
    auto finalStateTensor = std::make_shared<TensorAttributes>();
    finalStateTensor->set_uid(2011).set_name("FinalStateConstRef");
    auto stateCheckpointsTensor = std::make_shared<TensorAttributes>();
    stateCheckpointsTensor->set_uid(2012).set_name("StateCheckpointsConstRef");

    // Set using const reference (copy)
    attrs.set_q(qTensor);
    attrs.set_k(kTensor);
    attrs.set_v(vTensor);
    attrs.set_g(gTensor);
    attrs.set_beta(betaTensor);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    attrs.set_initial_state(initialStateTensor);
    attrs.set_a_log(aLogTensor);
    attrs.set_dt_bias(dtBiasTensor);
    attrs.set_state_indices(stateIndicesTensor);
    attrs.set_o(oTensor);
    attrs.set_final_state(finalStateTensor);
    attrs.set_state_checkpoints(stateCheckpointsTensor);

    // Original tensors should still be valid
    EXPECT_NE(qTensor, nullptr);
    EXPECT_NE(kTensor, nullptr);
    EXPECT_NE(vTensor, nullptr);
    EXPECT_NE(gTensor, nullptr);
    EXPECT_NE(betaTensor, nullptr);
    EXPECT_NE(cuSeqlensTensor, nullptr);
    EXPECT_NE(initialStateTensor, nullptr);
    EXPECT_NE(aLogTensor, nullptr);
    EXPECT_NE(dtBiasTensor, nullptr);
    EXPECT_NE(stateIndicesTensor, nullptr);
    EXPECT_NE(oTensor, nullptr);
    EXPECT_NE(finalStateTensor, nullptr);
    EXPECT_NE(stateCheckpointsTensor, nullptr);
}

TEST(TestGdnAttributes, LogicalAndStrictEquality)
{
    GdnAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_scale(0.35355339F);
    attr1.set_output_final_state(true);
    attr1.set_use_qk_l2norm(true);
    attr1.set_checkpoint_every_n_tokens(4);
    attr1.set_use_beta_sigmoid(true);
    attr1.set_allow_neg_eigval(true);
    attr1.set_safe_gate(true);
    attr1.set_gate_domain(GateDomain::LOG);
    attr1.set_batch_invariant(true);
    attr1.set_overwrite_initial_state(false);
    auto q1 = std::make_shared<TensorAttributes>();
    q1->set_uid(1).set_name("Q").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_q(q1);
    auto k1 = std::make_shared<TensorAttributes>();
    k1->set_uid(2).set_name("K").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_k(k1);
    auto v1 = std::make_shared<TensorAttributes>();
    v1->set_uid(3).set_name("V").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_v(v1);
    auto g1 = std::make_shared<TensorAttributes>();
    g1->set_uid(4).set_name("G").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_g(g1);
    auto beta1 = std::make_shared<TensorAttributes>();
    beta1->set_uid(5).set_name("Beta").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_beta(beta1);
    auto cuSeqlens1 = std::make_shared<TensorAttributes>();
    cuSeqlens1->set_uid(6).set_name("CuSeqlens").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_cu_seqlens(cuSeqlens1);
    auto initialState1 = std::make_shared<TensorAttributes>();
    initialState1->set_uid(7)
        .set_name("InitialState")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_initial_state(initialState1);
    auto aLog1 = std::make_shared<TensorAttributes>();
    aLog1->set_uid(8).set_name("ALog").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_a_log(aLog1);
    auto dtBias1 = std::make_shared<TensorAttributes>();
    dtBias1->set_uid(9).set_name("DtBias").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_dt_bias(dtBias1);
    auto stateIndices1 = std::make_shared<TensorAttributes>();
    stateIndices1->set_uid(10)
        .set_name("StateIndices")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_state_indices(stateIndices1);
    auto o1 = std::make_shared<TensorAttributes>();
    o1->set_uid(11).set_name("O").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_o(o1);
    auto finalState1 = std::make_shared<TensorAttributes>();
    finalState1->set_uid(12).set_name("FinalState").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_final_state(finalState1);
    auto stateCheckpoints1 = std::make_shared<TensorAttributes>();
    stateCheckpoints1->set_uid(13)
        .set_name("StateCheckpoints")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_state_checkpoints(stateCheckpoints1);

    GdnAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_scale(0.35355339F);
    attr2.set_output_final_state(true);
    attr2.set_use_qk_l2norm(true);
    attr2.set_checkpoint_every_n_tokens(4);
    attr2.set_use_beta_sigmoid(true);
    attr2.set_allow_neg_eigval(true);
    attr2.set_safe_gate(true);
    attr2.set_gate_domain(GateDomain::LOG);
    attr2.set_batch_invariant(true);
    attr2.set_overwrite_initial_state(false);
    auto q2 = std::make_shared<TensorAttributes>();
    q2->set_uid(1).set_name("Q").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_q(q2);
    auto k2 = std::make_shared<TensorAttributes>();
    k2->set_uid(2).set_name("K").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_k(k2);
    auto v2 = std::make_shared<TensorAttributes>();
    v2->set_uid(3).set_name("V").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_v(v2);
    auto g2 = std::make_shared<TensorAttributes>();
    g2->set_uid(4).set_name("G").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_g(g2);
    auto beta2 = std::make_shared<TensorAttributes>();
    beta2->set_uid(5).set_name("Beta").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_beta(beta2);
    auto cuSeqlens2 = std::make_shared<TensorAttributes>();
    cuSeqlens2->set_uid(6).set_name("CuSeqlens").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_cu_seqlens(cuSeqlens2);
    auto initialState2 = std::make_shared<TensorAttributes>();
    initialState2->set_uid(7)
        .set_name("InitialState")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_initial_state(initialState2);
    auto aLog2 = std::make_shared<TensorAttributes>();
    aLog2->set_uid(8).set_name("ALog").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_a_log(aLog2);
    auto dtBias2 = std::make_shared<TensorAttributes>();
    dtBias2->set_uid(9).set_name("DtBias").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_dt_bias(dtBias2);
    auto stateIndices2 = std::make_shared<TensorAttributes>();
    stateIndices2->set_uid(10)
        .set_name("StateIndices")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_state_indices(stateIndices2);
    auto o2 = std::make_shared<TensorAttributes>();
    o2->set_uid(11).set_name("O").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_o(o2);
    auto finalState2 = std::make_shared<TensorAttributes>();
    finalState2->set_uid(12).set_name("FinalState").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_final_state(finalState2);
    auto stateCheckpoints2 = std::make_shared<TensorAttributes>();
    stateCheckpoints2->set_uid(13)
        .set_name("StateCheckpoints")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_state_checkpoints(stateCheckpoints2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchQ = std::make_shared<TensorAttributes>();
    structuralMismatchQ->set_uid(9999).set_name("MismatchedQ");
    attr2.set_q(structuralMismatchQ);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_q(q2); // Revert

    // scale mismatch: semantic, must fail both checks
    attr2.set_scale(1.0);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_scale(0.35355339F); // Revert

    // output_final_state mismatch: semantic, must fail both checks
    attr2.set_output_final_state(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_output_final_state(true); // Revert

    // use_qk_l2norm mismatch: semantic, must fail both checks
    attr2.set_use_qk_l2norm(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_use_qk_l2norm(true); // Revert

    // checkpoint_every_n_tokens mismatch: semantic, must fail both checks
    attr2.set_checkpoint_every_n_tokens(5);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_checkpoint_every_n_tokens(4); // Revert

    // use_beta_sigmoid mismatch: semantic, must fail both checks
    attr2.set_use_beta_sigmoid(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_use_beta_sigmoid(true); // Revert

    // allow_neg_eigval mismatch: semantic, must fail both checks
    attr2.set_allow_neg_eigval(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_allow_neg_eigval(true); // Revert

    // safe_gate mismatch: semantic, must fail both checks
    attr2.set_safe_gate(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_safe_gate(true); // Revert

    // gate_domain mismatch: semantic, must fail both checks
    attr2.set_gate_domain(GateDomain::LINEAR);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_gate_domain(GateDomain::LOG); // Revert

    // batch_invariant mismatch: semantic, must fail both checks
    attr2.set_batch_invariant(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_batch_invariant(true); // Revert

    // overwrite_initial_state mismatch: semantic, must fail both checks
    attr2.set_overwrite_initial_state(true);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_overwrite_initial_state(false); // Revert

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchQ = std::make_shared<TensorAttributes>();
    logicalMatchQ
        ->set_uid(555) // Diverges from attr1's q1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's q1 ("Q")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_q(logicalMatchQ);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
