// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/SdpaFwdNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{
// Helper: create a rank-4 tensor with given dims [batch, heads, seq, head_dim]
std::shared_ptr<TensorAttributes>
    makeTensor4D(int64_t batch, int64_t heads, int64_t seq, int64_t headDim)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim({batch, heads, seq, headDim});
    return t;
}

SdpaAttributes makeMinimalAttrs(const std::shared_ptr<TensorAttributes>& q,
                                const std::shared_ptr<TensorAttributes>& k,
                                const std::shared_ptr<TensorAttributes>& v)
{
    SdpaAttributes attrs;
    attrs.set_q(q);
    attrs.set_k(k);
    attrs.set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    return attrs;
}
} // namespace

TEST(TestSdpaFwdNode, PreValidateSucceedsMinimal)
{
    // batch=2, num_heads=8, seq_q=16, head_dim=64
    // num_kv_heads=8, seq_kv=32
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsGQA)
{
    // num_heads=8, num_kv_heads=2 — 8 % 2 == 0, GQA valid
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 2, 32, 64);
    auto v = makeTensor4D(2, 2, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsMQA)
{
    // MQA: num_kv_heads=1
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 1, 32, 64);
    auto v = makeTensor4D(2, 1, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsGQADifferentKVHeads)
{
    // Q=32 heads, K=8, V=4 — independent GQA broadcast
    auto q = makeTensor4D(2, 32, 128, 64);
    auto k = makeTensor4D(2, 8, 128, 64);
    auto v = makeTensor4D(2, 4, 128, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsInvalidGQAVHeads)
{
    // Q=8 heads, K=2 (valid), V=3 (8%3!=0)
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 2, 32, 64);
    auto v = makeTensor4D(2, 3, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsZeroKvHeads)
{
    // K has 0 heads — should fail positivity check before divisibility
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 0, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsZeroVHeads)
{
    // V has 0 heads — should fail positivity check
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 0, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsInvalidGQAKHeads)
{
    // Q=8 heads, K=3 (8%3!=0), V=4 (8%4==0) — K invalid, V valid
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 3, 32, 64);
    auto v = makeTensor4D(2, 4, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsMissingQ)
{
    SdpaAttributes attrs;
    attrs.set_k(makeTensor4D(2, 8, 32, 64));
    attrs.set_v(makeTensor4D(2, 8, 32, 64));
    attrs.set_o(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaFwdNode, PreValidateFailsMissingK)
{
    SdpaAttributes attrs;
    attrs.set_q(makeTensor4D(2, 8, 16, 64));
    attrs.set_v(makeTensor4D(2, 8, 32, 64));
    attrs.set_o(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaFwdNode, PreValidateFailsMissingV)
{
    SdpaAttributes attrs;
    attrs.set_q(makeTensor4D(2, 8, 16, 64));
    attrs.set_k(makeTensor4D(2, 8, 32, 64));
    attrs.set_o(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaFwdNode, PreValidateFailsRankLessThan4)
{
    // Q is rank-3, should fail
    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({2, 8, 64}); // rank-3
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsRankGreaterThan4)
{
    // Q is rank-5, should fail
    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({1, 2, 8, 16, 64}); // rank-5
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateSucceedsDifferentHeadDimV)
{
    // V head_dim=32 differs from Q/K head_dim=64 — valid, headDimV is independent
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 32);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsHeadDimMismatch)
{
    // K head_dim=32, Q head_dim=64 — mismatch
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 32);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsOutputShapeMismatch)
{
    // O shape [2, 8, 16, 99] but headDimV=64 — mismatch
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaAttributes attrs;
    attrs.set_q(q);
    attrs.set_k(k);
    attrs.set_v(v);
    auto o = makeTensor4D(2, 8, 16, 99); // wrong head_dim
    attrs.set_o(o);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateSucceedsCorrectOutputShape)
{
    // O shape matches [batch, num_heads, seq_q, headDimV]
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 32); // headDimV=32

    SdpaAttributes attrs;
    attrs.set_q(q);
    attrs.set_k(k);
    attrs.set_v(v);
    auto o = makeTensor4D(2, 8, 16, 32); // correct: matches headDimV
    attrs.set_o(o);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsSeqKvMismatch)
{
    // K seq_kv=32, V seq_kv=16 — mismatch
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 16, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsInvalidGQA)
{
    // num_heads=8, num_kv_heads=3 — 8 % 3 != 0
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 3, 32, 64);
    auto v = makeTensor4D(2, 3, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateSucceedsWithAttnMask)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    // Valid mask: [2, 8, 16, 32] — exact match
    auto mask = std::make_shared<TensorAttributes>();
    mask->set_dim({2, 8, 16, 32});
    attrs.set_bias(mask);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsWithBroadcastAttnMask)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    // Broadcast mask: last two dims are 1
    auto mask = std::make_shared<TensorAttributes>();
    mask->set_dim({1, 1});
    attrs.set_bias(mask);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsAttnMaskRankTooLarge)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    // Rank-5 mask is invalid
    auto mask = std::make_shared<TensorAttributes>();
    mask->set_dim({1, 2, 8, 16, 32});
    attrs.set_bias(mask);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsAttnScaleTensorAndValueBothSet)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    // Rule 6: attn_scale tensor and attn_scale_value are mutually exclusive.
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1}).set_is_pass_by_value(true);
    attrs.set_attn_scale(scale);
    attrs.attn_scale_value = 1.0f;

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, InferPropertiesSetsOutputShape)
{
    // headDimV=64, so O shape should be [2, 8, 16, 64]
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    auto o = attrs.get_o();

    const GraphAttributes graphAttrs;
    SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto dims = o->get_dim();
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 8);
    EXPECT_EQ(dims[2], 16);
    EXPECT_EQ(dims[3], 64); // headDimV
}

TEST(TestSdpaFwdNode, InferPropertiesSetsOutputShapeDifferentHeadDimV)
{
    // headDimV=32 (different from headDimQK=64), O shape should be [2, 8, 16, 32]
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 32); // headDimV=32
    auto attrs = makeMinimalAttrs(q, k, v);
    auto o = attrs.get_o();

    const GraphAttributes graphAttrs;
    SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto dims = o->get_dim();
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 8);
    EXPECT_EQ(dims[2], 16);
    EXPECT_EQ(dims[3], 32); // headDimV, not headDimQK
}

TEST(TestSdpaFwdNode, InferPropertiesSetsStatsShape)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    auto stats = std::make_shared<TensorAttributes>();
    attrs.set_stats(stats);

    const GraphAttributes graphAttrs;
    SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Stats shape: [batch, num_heads, seq_q, 1]
    auto dims = stats->get_dim();
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 8);
    EXPECT_EQ(dims[2], 16);
    EXPECT_EQ(dims[3], 1);

    // Stats strides should be set
    EXPECT_FALSE(stats->get_stride().empty());
}

TEST(TestSdpaFwdNode, InferPropertiesSetsOutputStrides)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    auto o = attrs.get_o();

    const GraphAttributes graphAttrs;
    SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto strides = o->get_stride();
    ASSERT_EQ(strides.size(), 4u);
    // Row-major strides for [2, 8, 16, 64]: [8*16*64, 16*64, 64, 1]
    EXPECT_EQ(strides[3], 1);
    EXPECT_EQ(strides[2], 64);
    EXPECT_EQ(strides[1], 16 * 64);
    EXPECT_EQ(strides[0], 8 * 16 * 64);
}

TEST(TestSdpaFwdNode, InferPropertiesPreservesExplicitOutputShape)
{
    // If O dims are already set, they should not be overwritten
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaAttributes attrs;
    attrs.set_q(q);
    attrs.set_k(k);
    attrs.set_v(v);

    auto o = std::make_shared<TensorAttributes>();
    o->set_dim({2, 8, 16, 64});
    o->set_stride({8192, 1024, 64, 1});
    attrs.set_o(o);

    const GraphAttributes graphAttrs;
    SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Dims should remain unchanged
    EXPECT_EQ(o->get_dim(), (std::vector<int64_t>{2, 8, 16, 64}));
    // Strides should remain unchanged (they were already set)
    EXPECT_EQ(o->get_stride(), (std::vector<int64_t>{8192, 1024, 64, 1}));
}

TEST(TestSdpaFwdNode, PreValidateFailsBatchMismatchQK)
{
    // K batch=4 differs from Q batch=2
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(4, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsBatchMismatchQV)
{
    // V batch=4 differs from Q batch=2
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(4, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, GetNodeTypeReturnsSdpaFwd)
{
    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(SdpaAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::SDPA_FWD);
}

//==============================================================================
// Fail-loudly drain of unsupported cuDNN-compat setters
//==============================================================================

TEST(TestSdpaFwdNode, PreValidateFailsUnsupportedScoreMod)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_score_mod([](int) { return 0; });

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("score modifier"), std::string::npos) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateDrainRunsBeforeShapeChecks)
{
    // Q is rank-3 (shape-invalid) AND an unsupported setter is present. The drain
    // must run first, so the surfaced message is the unsupported one.
    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({2, 8, 16});
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_score_mod([](int) { return 0; });

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("score modifier"), std::string::npos) << err.err_msg;
    EXPECT_EQ(err.err_msg.find("rank"), std::string::npos) << err.err_msg;
}

//==============================================================================
// Paged K/V and variable sequence lengths (cuDNN paged-attention contract)
//==============================================================================

namespace
{
std::shared_ptr<TensorAttributes> makeInt32Tensor(const std::vector<int64_t>& dims)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim(dims).set_data_type(DataType::INT32);
    return t;
}

// Paged decode: B=2, Hq=8, Hkv=2, Sq=1, D=64; containers of 10 blocks of 16 tokens;
// max_seq_len_kv=64, so the page tables are [2, 1, 4, 1].
struct PagedGraph
{
    static constexpr int64_t BATCH = 2;
    static constexpr int64_t BLOCK_SIZE = 16;
    static constexpr int64_t NUM_BLOCKS = 10;
    static constexpr int32_t MAX_SEQ_LEN_KV = 64;
    static constexpr int64_t MAX_BLOCKS_PER_SEQ = 4;

    std::shared_ptr<TensorAttributes> q = makeTensor4D(BATCH, 8, 1, 64);
    std::shared_ptr<TensorAttributes> k = makeTensor4D(NUM_BLOCKS, 2, BLOCK_SIZE, 64);
    std::shared_ptr<TensorAttributes> v = makeTensor4D(NUM_BLOCKS, 2, BLOCK_SIZE, 64);
    std::shared_ptr<TensorAttributes> pageTableK
        = makeInt32Tensor({BATCH, 1, MAX_BLOCKS_PER_SEQ, 1});
    std::shared_ptr<TensorAttributes> pageTableV
        = makeInt32Tensor({BATCH, 1, MAX_BLOCKS_PER_SEQ, 1});
    std::shared_ptr<TensorAttributes> seqLenKv = makeInt32Tensor({BATCH, 1, 1, 1});

    SdpaAttributes attrs() const
    {
        auto a = makeMinimalAttrs(q, k, v);
        a.set_paged_attention_k_table(pageTableK);
        a.set_paged_attention_v_table(pageTableV);
        a.set_seq_len_kv(seqLenKv);
        a.set_paged_attention_max_seq_len_kv(MAX_SEQ_LEN_KV);
        return a;
    }
};

Error preValidate(SdpaAttributes&& attrs)
{
    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    return node.pre_validate_node();
}
} // namespace

TEST(TestSdpaFwdNode, PreValidateSucceedsPagedDecode)
{
    // K/V dim 0 is num_blocks (10), not the batch (2): the batch check must not apply.
    const PagedGraph g;
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsPagedSharedPageTable)
{
    PagedGraph g;
    g.pageTableV = g.pageTableK;
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsPagedPartialLastBlock)
{
    // max_seq_len_kv=50 with block_size 16 needs ceil(50/16) = 4 blocks per sequence.
    const PagedGraph g;
    auto attrs = g.attrs();
    attrs.set_paged_attention_max_seq_len_kv(50);
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsPagedRaggedQWithSeqLenQ)
{
    PagedGraph g;
    g.q = makeTensor4D(PagedGraph::BATCH, 8, 7, 64);
    g.q->set_ragged_offset(makeInt32Tensor({PagedGraph::BATCH + 1, 1, 1, 1}));
    auto attrs = g.attrs();
    attrs.set_seq_len_q(makeInt32Tensor({PagedGraph::BATCH, 1, 1, 1}));
    attrs.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT).set_diagonal_band_right_bound(0);
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedRaggedQWithoutSeqLenQ)
{
    PagedGraph g;
    g.q = makeTensor4D(PagedGraph::BATCH, 8, 7, 64);
    g.q->set_ragged_offset(makeInt32Tensor({PagedGraph::BATCH + 1, 1, 1, 1}));
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedOnlyKTable)
{
    const PagedGraph g;
    auto attrs = g.attrs();
    attrs.set_paged_attention_v_table(std::shared_ptr<TensorAttributes>{});
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedOnlyVTable)
{
    const PagedGraph g;
    auto attrs = g.attrs();
    attrs.set_paged_attention_k_table(std::shared_ptr<TensorAttributes>{});
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsPageTableWrongRank)
{
    PagedGraph g;
    g.pageTableK = makeInt32Tensor({PagedGraph::BATCH, PagedGraph::MAX_BLOCKS_PER_SEQ});
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPageTableWrongDataType)
{
    PagedGraph g;
    g.pageTableV = makeTensor4D(PagedGraph::BATCH, 1, PagedGraph::MAX_BLOCKS_PER_SEQ, 1);
    g.pageTableV->set_data_type(DataType::INT64);
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPageTableBatchMismatch)
{
    PagedGraph g;
    g.pageTableK = makeInt32Tensor({PagedGraph::BATCH + 1, 1, PagedGraph::MAX_BLOCKS_PER_SEQ, 1});
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsPageTableNonUnitDims)
{
    PagedGraph g;
    g.pageTableK = makeInt32Tensor({PagedGraph::BATCH, 2, PagedGraph::MAX_BLOCKS_PER_SEQ, 1});
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsRaggedPageTable)
{
    // Packed (ragged) page tables are a cuDNN feature hipDNN does not support.
    const PagedGraph g;
    g.pageTableK->set_ragged_offset(makeInt32Tensor({PagedGraph::BATCH + 1, 1, 1, 1}));
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedMissingMaxSeqLenKv)
{
    const PagedGraph g;
    auto attrs = g.attrs();
    attrs.max_seq_len_kv.reset();
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedNonPositiveMaxSeqLenKv)
{
    const PagedGraph g;
    auto attrs = g.attrs();
    attrs.set_paged_attention_max_seq_len_kv(0);
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedMaxSeqLenKvNeedsMoreBlocks)
{
    // 65 tokens need 5 blocks of 16, but the tables hold 4.
    const PagedGraph g;
    auto attrs = g.attrs();
    attrs.set_paged_attention_max_seq_len_kv(65);
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedMaxSeqLenKvNeedsFewerBlocks)
{
    // 48 tokens need 3 blocks of 16; cuDNN requires the table width to match exactly.
    const PagedGraph g;
    auto attrs = g.attrs();
    attrs.set_paged_attention_max_seq_len_kv(48);
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedMissingSeqLenKv)
{
    const PagedGraph g;
    auto attrs = g.attrs();
    attrs.set_seq_len_kv(std::shared_ptr<TensorAttributes>{});
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedBlockSizeMismatch)
{
    PagedGraph g;
    g.v = makeTensor4D(PagedGraph::NUM_BLOCKS, 2, 2 * PagedGraph::BLOCK_SIZE, 64);
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsMaxSeqLenKvWithoutPaging)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_paged_attention_max_seq_len_kv(32);
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsSeqLensWithoutPaging)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_seq_len_q(makeInt32Tensor({2, 1, 1, 1}));
    attrs.set_seq_len_kv(makeInt32Tensor({2, 1, 1, 1}));
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsRankOneSeqLensWithoutPaging)
{
    // The [B, 1, 1, 1] shape applies to every SDPA forward graph, not only paged ones.
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_seq_len_q(makeInt32Tensor({2}));
    attrs.set_seq_len_kv(makeInt32Tensor({2}));
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsSeqLenKvWrongShape)
{
    PagedGraph g;
    g.seqLenKv = makeInt32Tensor({PagedGraph::BATCH});
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsSeqLenKvBatchMismatch)
{
    PagedGraph g;
    g.seqLenKv = makeInt32Tensor({PagedGraph::BATCH + 1, 1, 1, 1});
    auto err = preValidate(g.attrs());
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsSeqLenQWrongDataType)
{
    const PagedGraph g;
    auto attrs = g.attrs();
    auto seqLenQ = makeTensor4D(PagedGraph::BATCH, 1, 1, 1);
    seqLenQ->set_data_type(DataType::FLOAT);
    attrs.set_seq_len_q(seqLenQ);
    auto err = preValidate(std::move(attrs));
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidatePagedAttnMaskUsesMaxSeqLenKv)
{
    // With paged K/V the mask's KV extent is max_seq_len_kv (64), not the block size (16).
    const PagedGraph g;
    auto ok = g.attrs();
    auto mask = std::make_shared<TensorAttributes>();
    mask->set_dim({PagedGraph::BATCH, 8, 1, PagedGraph::MAX_SEQ_LEN_KV});
    ok.set_bias(mask);
    auto err = preValidate(std::move(ok));
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto bad = g.attrs();
    auto blockMask = std::make_shared<TensorAttributes>();
    blockMask->set_dim({PagedGraph::BATCH, 8, 1, PagedGraph::BLOCK_SIZE});
    bad.set_bias(blockMask);
    err = preValidate(std::move(bad));
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

namespace
{
constexpr int64_t SINK_HEADS = 8;

SdpaAttributes
    makeSinkAttrs(DataType ioType, DataType sinkType, const std::vector<int64_t>& sinkDims)
{
    auto q = makeTensor4D(2, SINK_HEADS, 16, 64);
    auto k = makeTensor4D(2, 2, 32, 64);
    auto v = makeTensor4D(2, 2, 32, 64);
    q->set_data_type(ioType);
    k->set_data_type(ioType);
    v->set_data_type(ioType);
    auto attrs = makeMinimalAttrs(q, k, v);
    auto sink = std::make_shared<TensorAttributes>();
    sink->set_dim(sinkDims);
    sink->set_data_type(sinkType);
    attrs.set_sink_token(sink);
    return attrs;
}

Error preValidateSink(DataType ioType, DataType sinkType, const std::vector<int64_t>& sinkDims)
{
    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(makeSinkAttrs(ioType, sinkType, sinkDims), graphAttrs);
    return node.pre_validate_node();
}
} // namespace

TEST(TestSdpaFwdNode, PreValidateSucceedsFloatSink)
{
    auto err = preValidateSink(DataType::BFLOAT16, DataType::FLOAT, {1, SINK_HEADS, 1, 1});
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsSinkInQDataType)
{
    auto err = preValidateSink(DataType::BFLOAT16, DataType::BFLOAT16, {1, SINK_HEADS, 1, 1});
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
    err = preValidateSink(DataType::HALF, DataType::HALF, {1, SINK_HEADS, 1, 1});
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsSinkWrongRank)
{
    auto err = preValidateSink(DataType::BFLOAT16, DataType::FLOAT, {SINK_HEADS});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
    err = preValidateSink(DataType::BFLOAT16, DataType::FLOAT, {1, SINK_HEADS, 1});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsSinkWrongDims)
{
    auto err = preValidateSink(DataType::BFLOAT16, DataType::FLOAT, {2, SINK_HEADS, 1, 1});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
    err = preValidateSink(DataType::BFLOAT16, DataType::FLOAT, {1, SINK_HEADS, 4, 1});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
    err = preValidateSink(DataType::BFLOAT16, DataType::FLOAT, {1, 1, 1, SINK_HEADS});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsSinkHeadMismatch)
{
    // One sink per query head: the KV head count (2) is not accepted.
    auto err = preValidateSink(DataType::BFLOAT16, DataType::FLOAT, {1, 2, 1, 1});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsSinkHalfWithBfloat16Q)
{
    auto err = preValidateSink(DataType::BFLOAT16, DataType::HALF, {1, SINK_HEADS, 1, 1});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsSinkInt32)
{
    auto err = preValidateSink(DataType::BFLOAT16, DataType::INT32, {1, SINK_HEADS, 1, 1});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsSinkMatchingFloat8Q)
{
    // The Q-dtype extension covers HALF/BFLOAT16 only.
    auto err = preValidateSink(DataType::FP8_E4M3, DataType::FP8_E4M3, {1, SINK_HEADS, 1, 1});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateSinkUsesGraphIoDataType)
{
    // Unset tensor types resolve to the graph's io_data_type, as fill_from_context does.
    GraphAttributes graphAttrs;
    graphAttrs.set_io_data_type(DataType::BFLOAT16);
    const SdpaFwdNode ok(makeSinkAttrs(DataType::NOT_SET, DataType::NOT_SET, {1, SINK_HEADS, 1, 1}),
                         graphAttrs);
    auto err = ok.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    const SdpaFwdNode bad(makeSinkAttrs(DataType::NOT_SET, DataType::HALF, {1, SINK_HEADS, 1, 1}),
                          graphAttrs);
    err = bad.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}
