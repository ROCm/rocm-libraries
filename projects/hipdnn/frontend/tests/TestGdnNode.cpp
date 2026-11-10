// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GdnAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/node/GdnNode.hpp>

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

namespace
{

// T = 4 tokens, N = 2 sequences, H = HK = HV = HO = 2, K = 3, V = 4.
constexpr int64_t K_T = 4;
constexpr int64_t K_N = 2;
constexpr int64_t K_HO = 2;
constexpr int64_t K_K = 3;
constexpr int64_t K_V = 4;

std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims,
                                             DataType dataType = DataType::FLOAT)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim(dims).set_data_type(dataType);
    return tensor;
}

GdnAttributes createValidAttributes()
{
    GdnAttributes attrs;
    attrs.set_q(makeTensor({K_T, K_HO, K_K}));
    attrs.set_k(makeTensor({K_T, K_HO, K_K}));
    attrs.set_v(makeTensor({K_T, K_HO, K_V}));
    attrs.set_g(makeTensor({K_T, K_HO}));
    attrs.set_beta(makeTensor({K_T, K_HO}));
    attrs.set_cu_seqlens(makeTensor({K_N + 1}, DataType::INT32));
    attrs.set_o(std::make_shared<TensorAttributes>());
    return attrs;
}

Error preValidate(GdnAttributes attrs)
{
    const GraphAttributes graphAttributes;
    const GdnNode node(std::move(attrs), graphAttributes);
    return node.pre_validate_node();
}

Error validateAndInfer(GdnNode& node)
{
    HIPDNN_CHECK_ERROR(node.pre_validate_node());
    return node.infer_properties_node();
}

void expectInvalid(const std::function<void(GdnAttributes&)>& mutate)
{
    auto attrs = createValidAttributes();
    mutate(attrs);
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::INVALID_VALUE);
}

} // namespace

TEST(TestGdnNode, GetNodeTypeReturnsGdn)
{
    const GraphAttributes graphAttrs;
    const GdnNode node(GdnAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::GDN);
}

TEST(TestGdnNode, PreValidateNodeAcceptsMinimalNode)
{
    const auto error = preValidate(createValidAttributes());
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();
}

TEST(TestGdnNode, PreValidateNodeRejectsEachMissingRequiredTensor)
{
    const std::vector<std::function<void(GdnAttributes&)>> clearers = {
        [](GdnAttributes& a) { a.set_q(nullptr); },
        [](GdnAttributes& a) { a.set_k(nullptr); },
        [](GdnAttributes& a) { a.set_v(nullptr); },
        [](GdnAttributes& a) { a.set_g(nullptr); },
        [](GdnAttributes& a) { a.set_beta(nullptr); },
        [](GdnAttributes& a) { a.set_cu_seqlens(nullptr); },
        [](GdnAttributes& a) { a.set_o(nullptr); },
    };
    for(const auto& clear : clearers)
    {
        auto attrs = createValidAttributes();
        clear(attrs);
        EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::ATTRIBUTE_NOT_SET);
    }
}

TEST(TestGdnNode, PreValidateNodeRejectsBadRanks)
{
    expectInvalid([](GdnAttributes& a) { a.set_q(makeTensor({K_T, K_HO * K_K})); });
    expectInvalid([](GdnAttributes& a) { a.set_k(makeTensor({K_T, K_HO, K_K, 1})); });
    expectInvalid([](GdnAttributes& a) { a.set_v(makeTensor({K_T * K_HO, K_V})); });
    expectInvalid([](GdnAttributes& a) { a.set_g(makeTensor({K_T, K_HO, 1})); });
    expectInvalid([](GdnAttributes& a) { a.set_beta(makeTensor({K_T})); });
    expectInvalid(
        [](GdnAttributes& a) { a.set_cu_seqlens(makeTensor({K_N + 1, 1}, DataType::INT32)); });
}

TEST(TestGdnNode, PreValidateNodeRejectsMismatchedShapes)
{
    // Token counts must agree.
    expectInvalid([](GdnAttributes& a) { a.set_k(makeTensor({K_T + 1, K_HO, K_K})); });
    expectInvalid([](GdnAttributes& a) { a.set_v(makeTensor({K_T - 1, K_HO, K_V})); });
    // k head dim must equal q's K.
    expectInvalid([](GdnAttributes& a) { a.set_k(makeTensor({K_T, K_HO, K_K + 1})); });
    // g and beta are [T, HO].
    expectInvalid([](GdnAttributes& a) { a.set_g(makeTensor({K_T, K_HO + 1})); });
    expectInvalid([](GdnAttributes& a) { a.set_beta(makeTensor({K_T + 1, K_HO})); });
    // o, when given, is [T, HO, V].
    expectInvalid([](GdnAttributes& a) { a.set_o(makeTensor({K_T, K_HO, K_K})); });
}

TEST(TestGdnNode, PreValidateNodeEnforcesHeadSharing)
{
    // HK must be H or HV.
    expectInvalid([](GdnAttributes& a) { a.set_k(makeTensor({K_T, 3, K_K})); });
    // H = 2, HV = 3: HO = 3 is not divisible by H.
    expectInvalid([](GdnAttributes& a) {
        a.set_v(makeTensor({K_T, 3, K_V}));
        a.set_g(makeTensor({K_T, 3}));
        a.set_beta(makeTensor({K_T, 3}));
    });

    // GQA: H = HK = 1 query/key head shared by HV = HO = 2 value heads.
    auto attrs = createValidAttributes();
    attrs.set_q(makeTensor({K_T, 1, K_K}));
    attrs.set_k(makeTensor({K_T, 1, K_K}));
    const auto error = preValidate(std::move(attrs));
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();

    // HK = HV with H = 1 < HV is also allowed.
    auto attrsKv = createValidAttributes();
    attrsKv.set_q(makeTensor({K_T, 1, K_K}));
    const auto errorKv = preValidate(std::move(attrsKv));
    EXPECT_EQ(errorKv.code, ErrorCode::OK) << errorKv.get_message();
}

TEST(TestGdnNode, PreValidateNodeRejectsBadCuSeqlens)
{
    expectInvalid([](GdnAttributes& a) { a.set_cu_seqlens(makeTensor({1}, DataType::INT32)); });
    expectInvalid([](GdnAttributes& a) { a.set_cu_seqlens(makeTensor({K_N + 1})); });

    auto attrs = createValidAttributes();
    attrs.set_cu_seqlens(makeTensor({K_N + 1}, DataType::INT64));
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::OK);
}

TEST(TestGdnNode, PreValidateNodeEnforcesGateRules)
{
    expectInvalid([](GdnAttributes& a) {
        a.set_safe_gate(true);
        a.set_gate_domain(GateDomain::LINEAR);
    });
    expectInvalid([](GdnAttributes& a) { a.set_a_log(makeTensor({K_HO})); });
    expectInvalid([](GdnAttributes& a) { a.set_dt_bias(makeTensor({K_HO})); });
    expectInvalid([](GdnAttributes& a) {
        a.set_safe_gate(true);
        a.set_a_log(makeTensor({K_HO + 1}));
    });
    expectInvalid([](GdnAttributes& a) {
        a.set_safe_gate(true);
        a.set_dt_bias(makeTensor({1, K_HO}));
    });
    expectInvalid([](GdnAttributes& a) { a.set_allow_neg_eigval(true); });

    auto attrs = createValidAttributes();
    attrs.set_safe_gate(true).set_a_log(makeTensor({K_HO})).set_dt_bias(makeTensor({K_HO}));
    attrs.set_use_beta_sigmoid(true).set_allow_neg_eigval(true);
    const auto error = preValidate(std::move(attrs));
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();
}

TEST(TestGdnNode, PreValidateNodeEnforcesStateRules)
{
    // state_indices needs initial_state, is [N] INT32, and excludes checkpoints.
    expectInvalid(
        [](GdnAttributes& a) { a.set_state_indices(makeTensor({K_N}, DataType::INT32)); });
    expectInvalid([](GdnAttributes& a) {
        a.set_initial_state(makeTensor({3, K_HO, K_V, K_K}));
        a.set_state_indices(makeTensor({K_N + 1}, DataType::INT32));
    });
    expectInvalid([](GdnAttributes& a) {
        a.set_initial_state(makeTensor({3, K_HO, K_V, K_K}));
        a.set_state_indices(makeTensor({K_N}, DataType::INT64));
    });
    expectInvalid([](GdnAttributes& a) {
        a.set_initial_state(makeTensor({3, K_HO, K_V, K_K}));
        a.set_state_indices(makeTensor({K_N}, DataType::INT32));
        a.set_checkpoint_every_n_tokens(2);
        a.set_state_checkpoints(std::make_shared<TensorAttributes>());
    });
    // overwrite_initial_state needs initial_state and final_state.
    expectInvalid([](GdnAttributes& a) {
        a.set_overwrite_initial_state(true);
        a.set_output_final_state(true);
        a.set_final_state(std::make_shared<TensorAttributes>());
    });
    expectInvalid([](GdnAttributes& a) {
        a.set_overwrite_initial_state(true);
        a.set_initial_state(makeTensor({K_N, K_HO, K_V, K_K}));
    });
    // output_final_state <=> final_state.
    expectInvalid([](GdnAttributes& a) { a.set_output_final_state(true); });
    expectInvalid(
        [](GdnAttributes& a) { a.set_final_state(std::make_shared<TensorAttributes>()); });
    // checkpoint_every_n_tokens > 0 <=> state_checkpoints, and n >= 0.
    expectInvalid([](GdnAttributes& a) { a.set_checkpoint_every_n_tokens(2); });
    expectInvalid(
        [](GdnAttributes& a) { a.set_state_checkpoints(std::make_shared<TensorAttributes>()); });
    expectInvalid([](GdnAttributes& a) { a.set_checkpoint_every_n_tokens(-1); });
}

TEST(TestGdnNode, PreValidateNodeEnforcesStateShapes)
{
    expectInvalid([](GdnAttributes& a) { a.set_initial_state(makeTensor({K_N, K_HO, K_V})); });
    expectInvalid(
        [](GdnAttributes& a) { a.set_initial_state(makeTensor({K_N + 1, K_HO, K_V, K_K})); });
    // K-major [N, HO, K, V] dims are rejected; K-major layouts are expressed through strides.
    expectInvalid([](GdnAttributes& a) { a.set_initial_state(makeTensor({K_N, K_HO, K_K, K_V})); });
    expectInvalid([](GdnAttributes& a) {
        a.set_output_final_state(true);
        a.set_final_state(makeTensor({K_N, K_HO + 1, K_V, K_K}));
    });
    // Pool rows of final_state follow initial_state.
    expectInvalid([](GdnAttributes& a) {
        a.set_initial_state(makeTensor({5, K_HO, K_V, K_K}));
        a.set_state_indices(makeTensor({K_N}, DataType::INT32));
        a.set_output_final_state(true);
        a.set_final_state(makeTensor({4, K_HO, K_V, K_K}));
    });
    // C = max(T / n + N, 1) = 4 / 2 + 2 = 4 rows.
    expectInvalid([](GdnAttributes& a) {
        a.set_checkpoint_every_n_tokens(2);
        a.set_state_checkpoints(makeTensor({3, K_HO, K_V, K_K}));
    });

    auto attrs = createValidAttributes();
    attrs.set_checkpoint_every_n_tokens(2);
    attrs.set_state_checkpoints(makeTensor({4, K_HO, K_V, K_K}));
    attrs.set_initial_state(makeTensor({K_N, K_HO, K_V, K_K}));
    attrs.set_output_final_state(true);
    attrs.set_final_state(makeTensor({K_N, K_HO, K_V, K_K}));
    const auto error = preValidate(std::move(attrs));
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();
}

TEST(TestGdnNode, InferPropertiesNodeFillsOutputs)
{
    auto attrs = createValidAttributes();
    attrs.get_v()->set_stride({K_HO * K_V, K_V, 1});
    attrs.set_output_final_state(true);
    attrs.set_final_state(std::make_shared<TensorAttributes>());
    attrs.set_checkpoint_every_n_tokens(3);
    attrs.set_state_checkpoints(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttributes;
    graphAttributes.set_io_data_type(DataType::HALF);
    GdnNode node(std::move(attrs), graphAttributes);

    const auto error = validateAndInfer(node);
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();

    const auto o = node.attributes.get_o();
    EXPECT_EQ(o->get_dim(), (std::vector<int64_t>{K_T, K_HO, K_V}));
    EXPECT_EQ(o->get_stride(), (std::vector<int64_t>{K_HO * K_V, K_V, 1}));

    // Without an initial state, the final state is FLOAT [N, HO, V, K].
    const auto finalState = node.attributes.get_final_state();
    EXPECT_EQ(finalState->get_dim(), (std::vector<int64_t>{K_N, K_HO, K_V, K_K}));
    EXPECT_EQ(finalState->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(finalState->get_stride(),
              (std::vector<int64_t>{K_HO * K_V * K_K, K_V * K_K, K_K, 1}));

    // Checkpoints: C = max(4 / 3 + 2, 1) = 3 rows in the io (q) dtype.
    const auto checkpoints = node.attributes.get_state_checkpoints();
    EXPECT_EQ(checkpoints->get_dim(), (std::vector<int64_t>{3, K_HO, K_V, K_K}));
    EXPECT_EQ(checkpoints->get_data_type(), DataType::FLOAT);
}

TEST(TestGdnNode, InferPropertiesNodeFinalStateFollowsInitialStatePool)
{
    auto attrs = createValidAttributes();
    attrs.set_initial_state(makeTensor({5, K_HO, K_V, K_K}, DataType::BFLOAT16));
    attrs.set_state_indices(makeTensor({K_N}, DataType::INT32));
    attrs.set_output_final_state(true);
    attrs.set_final_state(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    GdnNode node(std::move(attrs), graphAttributes);
    const auto error = validateAndInfer(node);
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();

    const auto finalState = node.attributes.get_final_state();
    EXPECT_EQ(finalState->get_dim(), (std::vector<int64_t>{5, K_HO, K_V, K_K}));
    EXPECT_EQ(finalState->get_data_type(), DataType::BFLOAT16);
}

TEST(TestGdnNode, InferPropertiesNodeRejectsBadDataTypes)
{
    const std::vector<std::function<void(GdnAttributes&)>> mutators = {
        [](GdnAttributes& a) { a.get_k()->set_data_type(DataType::HALF); },
        [](GdnAttributes& a) { a.get_v()->set_data_type(DataType::BFLOAT16); },
        [](GdnAttributes& a) {
            a.get_q()->set_data_type(DataType::HALF);
            a.get_k()->set_data_type(DataType::HALF);
            a.get_v()->set_data_type(DataType::HALF);
            a.get_beta()->set_data_type(DataType::BFLOAT16);
        },
        [](GdnAttributes& a) { a.get_g()->set_data_type(DataType::INT32); },
        [](GdnAttributes& a) {
            a.set_initial_state(makeTensor({K_N, K_HO, K_V, K_K}, DataType::INT32));
        },
        [](GdnAttributes& a) {
            a.set_initial_state(makeTensor({K_N, K_HO, K_V, K_K}));
            a.set_output_final_state(true);
            a.set_final_state(makeTensor({K_N, K_HO, K_V, K_K}, DataType::BFLOAT16));
        },
    };
    for(const auto& mutate : mutators)
    {
        auto attrs = createValidAttributes();
        mutate(attrs);
        const GraphAttributes graphAttributes;
        GdnNode node(std::move(attrs), graphAttributes);
        EXPECT_EQ(validateAndInfer(node).code, ErrorCode::INVALID_VALUE);
    }

    // beta may be FLOAT with a HALF io dtype.
    auto attrs = createValidAttributes();
    attrs.get_q()->set_data_type(DataType::HALF);
    attrs.get_k()->set_data_type(DataType::HALF);
    attrs.get_v()->set_data_type(DataType::HALF);
    const GraphAttributes graphAttributes;
    GdnNode node(std::move(attrs), graphAttributes);
    const auto error = validateAndInfer(node);
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();
}

TEST(TestGdnNode, GatherHipdnnTensor)
{
    auto attrs = createValidAttributes();
    auto initialState = makeTensor({K_N, K_HO, K_V, K_K});
    attrs.set_initial_state(initialState);

    const GraphAttributes graphAttributes;
    const GdnNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(node.attributes.get_q()) != allTensors.end());
    EXPECT_TRUE(allTensors.find(node.attributes.get_o()) != allTensors.end());
    EXPECT_TRUE(allTensors.find(initialState) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 8u);
}

// NOLINTEND(misc-const-correctness)
