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
