// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GdnAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/node/GdnNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

GdnAttributes createValidAttributes()
{
    GdnAttributes attrs;

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
    gTensor->set_dim({8, 2});
    gTensor->set_stride({2, 1});
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

TEST(TestGdnNode, GetNodeTypeReturnsGdn)
{
    const GraphAttributes graphAttrs;
    const GdnNode node(GdnAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::GDN);
}

// --- PreValidateNode (success case) ---

TEST(TestGdnNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestGdnNode, PreValidateNodeMissingQTensor)
{
    GdnAttributes attrs;

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
    gTensor->set_dim({8, 2});
    gTensor->set_stride({2, 1});
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
    const GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGdnNode, PreValidateNodeMissingKTensor)
{
    GdnAttributes attrs;

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
    gTensor->set_dim({8, 2});
    gTensor->set_stride({2, 1});
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
    const GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGdnNode, PreValidateNodeMissingVTensor)
{
    GdnAttributes attrs;

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
    gTensor->set_dim({8, 2});
    gTensor->set_stride({2, 1});
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
    const GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGdnNode, PreValidateNodeMissingGTensor)
{
    GdnAttributes attrs;

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
    const GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGdnNode, PreValidateNodeMissingBetaTensor)
{
    GdnAttributes attrs;

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
    gTensor->set_dim({8, 2});
    gTensor->set_stride({2, 1});
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
    const GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGdnNode, PreValidateNodeMissingCuSeqlensTensor)
{
    GdnAttributes attrs;

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
    gTensor->set_dim({8, 2});
    gTensor->set_stride({2, 1});
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
    const GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGdnNode, PreValidateNodeMissingOTensor)
{
    GdnAttributes attrs;

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
    gTensor->set_dim({8, 2});
    gTensor->set_stride({2, 1});
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
    const GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGdnNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestGdnNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    GdnNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    // Stub implementation: verify the method can be called without error
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- GatherHipdnnTensors ---

TEST(TestGdnNode, GatherHipdnnTensor)
{
    GdnAttributes attrs;

    auto qTensor = std::make_shared<TensorAttributes>();
    qTensor->set_uid(2000).set_name("QTensor");
    attrs.set_q(qTensor);
    auto kTensor = std::make_shared<TensorAttributes>();
    kTensor->set_uid(2001).set_name("KTensor");
    attrs.set_k(kTensor);
    auto vTensor = std::make_shared<TensorAttributes>();
    vTensor->set_uid(2002).set_name("VTensor");
    attrs.set_v(vTensor);
    auto gTensor = std::make_shared<TensorAttributes>();
    gTensor->set_uid(2003).set_name("GTensor");
    attrs.set_g(gTensor);
    auto betaTensor = std::make_shared<TensorAttributes>();
    betaTensor->set_uid(2004).set_name("BetaTensor");
    attrs.set_beta(betaTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_uid(2005).set_name("CuSeqlensTensor");
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto oTensor = std::make_shared<TensorAttributes>();
    oTensor->set_uid(2010).set_name("OTensor");
    attrs.set_o(oTensor);

    attrs.set_checkpoint_every_n_tokens(4);

    const GraphAttributes graphAttributes;
    const GdnNode node(std::move(attrs), graphAttributes);

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

// NOLINTEND(misc-const-correctness)
