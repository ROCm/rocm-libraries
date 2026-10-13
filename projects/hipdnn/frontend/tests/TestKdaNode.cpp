// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/KdaAttributes.hpp>
#include <hipdnn_frontend/node/KdaNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

KdaAttributes createValidAttributes()
{
    KdaAttributes attrs;

    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_dim({8, 2, 8});
    qTensor->set_stride({16, 8, 1});
    qTensor->set_data_type(DataType::FLOAT);
    attrs.set_q(qTensor);
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_dim({8, 2, 8});
    kTensor->set_stride({16, 8, 1});
    kTensor->set_data_type(DataType::FLOAT);
    attrs.set_k(kTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_dim({8, 2, 8});
    vTensor->set_stride({16, 8, 1});
    vTensor->set_data_type(DataType::FLOAT);
    attrs.set_v(vTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_dim({8, 2, 8});
    gTensor->set_stride({16, 8, 1});
    gTensor->set_data_type(DataType::FLOAT);
    attrs.set_g(gTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_dim({8, 2});
    betaTensor->set_stride({2, 1});
    betaTensor->set_data_type(DataType::FLOAT);
    attrs.set_beta(betaTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_dim({8, 2, 8});
    oTensor->set_stride({16, 8, 1});
    oTensor->set_data_type(DataType::FLOAT);
    attrs.set_o(oTensor);

    // checkpoint_every_n_tokens > 0 requires the state_checkpoints output:
    // 8 tokens / 4 + 2 sequences = 4 rows.
    attrs.set_checkpoint_every_n_tokens(4);
    auto stateCheckpointsTensor = std::make_shared<TensorAttributes>();
    stateCheckpointsTensor->set_dim({4, 2, 8, 8});
    stateCheckpointsTensor->set_stride({128, 64, 8, 1});
    stateCheckpointsTensor->set_data_type(DataType::FLOAT);
    attrs.set_state_checkpoints(stateCheckpointsTensor);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestKdaNode, GetNodeTypeReturnsKda)
{
    const GraphAttributes graphAttrs;
    const KdaNode node(KdaAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::KDA);
}

// --- PreValidateNode (success case) ---

TEST(TestKdaNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestKdaNode, PreValidateNodeMissingQTensor)
{
    KdaAttributes attrs;

    // Set all required tensors except q
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_dim({8, 2, 8});
    kTensor->set_stride({16, 8, 1});
    kTensor->set_data_type(DataType::FLOAT);
    attrs.set_k(kTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_dim({8, 2, 8});
    vTensor->set_stride({16, 8, 1});
    vTensor->set_data_type(DataType::FLOAT);
    attrs.set_v(vTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_dim({8, 2, 8});
    gTensor->set_stride({16, 8, 1});
    gTensor->set_data_type(DataType::FLOAT);
    attrs.set_g(gTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_dim({8, 2});
    betaTensor->set_stride({2, 1});
    betaTensor->set_data_type(DataType::FLOAT);
    attrs.set_beta(betaTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_dim({8, 2, 8});
    oTensor->set_stride({16, 8, 1});
    oTensor->set_data_type(DataType::FLOAT);
    attrs.set_o(oTensor);

    attrs.set_checkpoint_every_n_tokens(4);

    // q tensor is missing
    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestKdaNode, PreValidateNodeMissingKTensor)
{
    KdaAttributes attrs;

    // Set all required tensors except k
    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_dim({8, 2, 8});
    qTensor->set_stride({16, 8, 1});
    qTensor->set_data_type(DataType::FLOAT);
    attrs.set_q(qTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_dim({8, 2, 8});
    vTensor->set_stride({16, 8, 1});
    vTensor->set_data_type(DataType::FLOAT);
    attrs.set_v(vTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_dim({8, 2, 8});
    gTensor->set_stride({16, 8, 1});
    gTensor->set_data_type(DataType::FLOAT);
    attrs.set_g(gTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_dim({8, 2});
    betaTensor->set_stride({2, 1});
    betaTensor->set_data_type(DataType::FLOAT);
    attrs.set_beta(betaTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_dim({8, 2, 8});
    oTensor->set_stride({16, 8, 1});
    oTensor->set_data_type(DataType::FLOAT);
    attrs.set_o(oTensor);

    attrs.set_checkpoint_every_n_tokens(4);

    // k tensor is missing
    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestKdaNode, PreValidateNodeMissingVTensor)
{
    KdaAttributes attrs;

    // Set all required tensors except v
    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_dim({8, 2, 8});
    qTensor->set_stride({16, 8, 1});
    qTensor->set_data_type(DataType::FLOAT);
    attrs.set_q(qTensor);
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_dim({8, 2, 8});
    kTensor->set_stride({16, 8, 1});
    kTensor->set_data_type(DataType::FLOAT);
    attrs.set_k(kTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_dim({8, 2, 8});
    gTensor->set_stride({16, 8, 1});
    gTensor->set_data_type(DataType::FLOAT);
    attrs.set_g(gTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_dim({8, 2});
    betaTensor->set_stride({2, 1});
    betaTensor->set_data_type(DataType::FLOAT);
    attrs.set_beta(betaTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_dim({8, 2, 8});
    oTensor->set_stride({16, 8, 1});
    oTensor->set_data_type(DataType::FLOAT);
    attrs.set_o(oTensor);

    attrs.set_checkpoint_every_n_tokens(4);

    // v tensor is missing
    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestKdaNode, PreValidateNodeMissingGTensor)
{
    KdaAttributes attrs;

    // Set all required tensors except g
    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_dim({8, 2, 8});
    qTensor->set_stride({16, 8, 1});
    qTensor->set_data_type(DataType::FLOAT);
    attrs.set_q(qTensor);
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_dim({8, 2, 8});
    kTensor->set_stride({16, 8, 1});
    kTensor->set_data_type(DataType::FLOAT);
    attrs.set_k(kTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_dim({8, 2, 8});
    vTensor->set_stride({16, 8, 1});
    vTensor->set_data_type(DataType::FLOAT);
    attrs.set_v(vTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_dim({8, 2});
    betaTensor->set_stride({2, 1});
    betaTensor->set_data_type(DataType::FLOAT);
    attrs.set_beta(betaTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_dim({8, 2, 8});
    oTensor->set_stride({16, 8, 1});
    oTensor->set_data_type(DataType::FLOAT);
    attrs.set_o(oTensor);

    attrs.set_checkpoint_every_n_tokens(4);

    // g tensor is missing
    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestKdaNode, PreValidateNodeMissingBetaTensor)
{
    KdaAttributes attrs;

    // Set all required tensors except beta
    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_dim({8, 2, 8});
    qTensor->set_stride({16, 8, 1});
    qTensor->set_data_type(DataType::FLOAT);
    attrs.set_q(qTensor);
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_dim({8, 2, 8});
    kTensor->set_stride({16, 8, 1});
    kTensor->set_data_type(DataType::FLOAT);
    attrs.set_k(kTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_dim({8, 2, 8});
    vTensor->set_stride({16, 8, 1});
    vTensor->set_data_type(DataType::FLOAT);
    attrs.set_v(vTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_dim({8, 2, 8});
    gTensor->set_stride({16, 8, 1});
    gTensor->set_data_type(DataType::FLOAT);
    attrs.set_g(gTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_dim({8, 2, 8});
    oTensor->set_stride({16, 8, 1});
    oTensor->set_data_type(DataType::FLOAT);
    attrs.set_o(oTensor);

    attrs.set_checkpoint_every_n_tokens(4);

    // beta tensor is missing
    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestKdaNode, PreValidateNodeMissingCuSeqlensTensor)
{
    KdaAttributes attrs;

    // Set all required tensors except cu_seqlens
    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_dim({8, 2, 8});
    qTensor->set_stride({16, 8, 1});
    qTensor->set_data_type(DataType::FLOAT);
    attrs.set_q(qTensor);
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_dim({8, 2, 8});
    kTensor->set_stride({16, 8, 1});
    kTensor->set_data_type(DataType::FLOAT);
    attrs.set_k(kTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_dim({8, 2, 8});
    vTensor->set_stride({16, 8, 1});
    vTensor->set_data_type(DataType::FLOAT);
    attrs.set_v(vTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_dim({8, 2, 8});
    gTensor->set_stride({16, 8, 1});
    gTensor->set_data_type(DataType::FLOAT);
    attrs.set_g(gTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_dim({8, 2});
    betaTensor->set_stride({2, 1});
    betaTensor->set_data_type(DataType::FLOAT);
    attrs.set_beta(betaTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_dim({8, 2, 8});
    oTensor->set_stride({16, 8, 1});
    oTensor->set_data_type(DataType::FLOAT);
    attrs.set_o(oTensor);

    attrs.set_checkpoint_every_n_tokens(4);

    // cu_seqlens tensor is missing
    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestKdaNode, PreValidateNodeMissingOTensor)
{
    KdaAttributes attrs;

    // Set all required tensors except o
    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_dim({8, 2, 8});
    qTensor->set_stride({16, 8, 1});
    qTensor->set_data_type(DataType::FLOAT);
    attrs.set_q(qTensor);
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_dim({8, 2, 8});
    kTensor->set_stride({16, 8, 1});
    kTensor->set_data_type(DataType::FLOAT);
    attrs.set_k(kTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_dim({8, 2, 8});
    vTensor->set_stride({16, 8, 1});
    vTensor->set_data_type(DataType::FLOAT);
    attrs.set_v(vTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_dim({8, 2, 8});
    gTensor->set_stride({16, 8, 1});
    gTensor->set_data_type(DataType::FLOAT);
    attrs.set_g(gTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_dim({8, 2});
    betaTensor->set_stride({2, 1});
    betaTensor->set_data_type(DataType::FLOAT);
    attrs.set_beta(betaTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);

    attrs.set_checkpoint_every_n_tokens(4);

    // o tensor is missing
    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestKdaNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestKdaNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    KdaNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    // Stub implementation: verify the method can be called without error
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- GatherHipdnnTensors ---

TEST(TestKdaNode, GatherHipdnnTensor)
{
    KdaAttributes attrs;

    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_uid(2100).set_name("QTensor");
    attrs.set_q(qTensor);
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_uid(2101).set_name("KTensor");
    attrs.set_k(kTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_uid(2102).set_name("VTensor");
    attrs.set_v(vTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_uid(2103).set_name("GTensor");
    attrs.set_g(gTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_uid(2104).set_name("BetaTensor");
    attrs.set_beta(betaTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_uid(2105).set_name("CuSeqlensTensor");
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_uid(2110).set_name("OTensor");
    attrs.set_o(oTensor);

    attrs.set_checkpoint_every_n_tokens(4);

    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(qTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(kTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(vTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(gTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(betaTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(cuSeqlensTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(oTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 7u);
}

// --- cuDNN flag-consistency rules (cudnn/linear_attention/graph_analyzer.py) ---

TEST(TestKdaNode, PreValidateRejectsGateLowerBoundWithoutSafeGate)
{
    auto attrs = createValidAttributes();
    attrs.set_gate_lower_bound(-5.0F);

    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestKdaNode, PreValidateRejectsOverwriteInitialStateWithoutFinalState)
{
    auto attrs = createValidAttributes();
    auto initialState = std::make_shared<TensorAttributes>();
    initialState->set_dim({2, 2, 8, 8});
    initialState->set_stride({128, 64, 8, 1});
    initialState->set_data_type(DataType::FLOAT);
    attrs.set_initial_state(initialState);
    attrs.set_overwrite_initial_state(true);

    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestKdaNode, PreValidateRejectsStateIndicesWithCheckpoints)
{
    auto attrs = createValidAttributes(); // checkpoints every 4 tokens
    auto initialState = std::make_shared<TensorAttributes>();
    initialState->set_dim({2, 2, 8, 8});
    initialState->set_stride({128, 64, 8, 1});
    initialState->set_data_type(DataType::FLOAT);
    attrs.set_initial_state(initialState);
    auto stateIndices = std::make_shared<TensorAttributes>();
    stateIndices->set_dim({2});
    stateIndices->set_stride({1});
    stateIndices->set_data_type(DataType::INT32);
    attrs.set_state_indices(stateIndices);

    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestKdaNode, PreValidateRejectsFinalStateWithoutOutputFinalState)
{
    auto attrs = createValidAttributes(); // output_final_state is false
    auto finalState = std::make_shared<TensorAttributes>();
    finalState->set_dim({2, 2, 8, 8});
    finalState->set_stride({128, 64, 8, 1});
    finalState->set_data_type(DataType::FLOAT);
    attrs.set_final_state(finalState);

    const GraphAttributes graphAttributes;
    const KdaNode node(std::move(attrs), graphAttributes);

    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

// NOLINTEND(misc-const-correctness)
