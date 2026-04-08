// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/SdpaBwdNode.hpp>

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

// Build a minimal SdpaBackwardAttributes with Q, K, V, O, dO, Stats, dQ, dK, dV set
SdpaBackwardAttributes makeMinimalAttrs(const std::shared_ptr<TensorAttributes>& q,
                                        const std::shared_ptr<TensorAttributes>& k,
                                        const std::shared_ptr<TensorAttributes>& v)
{
    SdpaBackwardAttributes attrs;
    attrs.set_q(q);
    attrs.set_k(k);
    attrs.set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());
    return attrs;
}
} // namespace

//==============================================================================
// pre_validate_node tests
//==============================================================================

TEST(TestSdpaBwdNode, PreValidateSucceedsMinimal)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateSucceedsGQA)
{
    // num_heads=8, num_kv_heads=2 — GQA valid (8 % 2 == 0)
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 2, 32, 64);
    auto v = makeTensor4D(2, 2, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateSucceedsMQA)
{
    // MQA: num_kv_heads=1
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 1, 32, 64);
    auto v = makeTensor4D(2, 1, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateFailsZeroKvHeads)
{
    // K has 0 heads — should fail positivity check before divisibility
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 0, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsZeroVHeads)
{
    // V has 0 heads — should fail positivity check
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 0, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsInvalidGQAKHeads)
{
    // Q=8 heads, K=3 (8%3!=0), V=4 (8%4==0) — K invalid, V valid
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 3, 32, 64);
    auto v = makeTensor4D(2, 4, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsMissingQ)
{
    SdpaBackwardAttributes attrs;
    attrs.set_k(makeTensor4D(2, 8, 32, 64));
    attrs.set_v(makeTensor4D(2, 8, 32, 64));
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBwdNode, PreValidateFailsMissingK)
{
    SdpaBackwardAttributes attrs;
    attrs.set_q(makeTensor4D(2, 8, 16, 64));
    attrs.set_v(makeTensor4D(2, 8, 32, 64));
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBwdNode, PreValidateFailsMissingV)
{
    SdpaBackwardAttributes attrs;
    attrs.set_q(makeTensor4D(2, 8, 16, 64));
    attrs.set_k(makeTensor4D(2, 8, 32, 64));
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBwdNode, PreValidateFailsMissingDo)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    // dO intentionally not set
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBwdNode, PreValidateFailsMissingStats)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    // stats intentionally not set
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBwdNode, PreValidateFailsMissingDq)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    // dq intentionally not set
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBwdNode, PreValidateFailsRankLessThan4)
{
    // Q is rank-3
    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({2, 8, 64}); // rank-3
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsRankGreaterThan4)
{
    // K is rank-5
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = std::make_shared<TensorAttributes>();
    k->set_dim({1, 2, 8, 32, 64}); // rank-5
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsBatchMismatchQK)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(4, 8, 32, 64); // batch=4 != 2
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsBatchMismatchQV)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(4, 8, 32, 64); // batch=4 != 2
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsHeadDimMismatch)
{
    // K head_dim=32, Q head_dim=64 — must match
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 32); // head_dim mismatch
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsSeqKvMismatch)
{
    // K seq_kv=32, V seq_kv=16 — must match
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 16, 64); // seq_kv mismatch
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateSucceedsGQADifferentKVHeads)
{
    // K num_kv_heads=2, V num_kv_heads=4 — both divide Q num_heads=8
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 2, 32, 64);
    auto v = makeTensor4D(2, 4, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateFailsInvalidGQAVHeads)
{
    // Q=8 heads, K=2 (valid), V=3 (8%3!=0)
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 2, 32, 64);
    auto v = makeTensor4D(2, 3, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsInvalidGQA)
{
    // num_heads=8, num_kv_heads=3 — 8 % 3 != 0
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 3, 32, 64);
    auto v = makeTensor4D(2, 3, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateSuccessDoMatchesOShape)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    auto o = makeTensor4D(2, 8, 16, 64);
    attrs.set_o(o);
    auto dOut = makeTensor4D(2, 8, 16, 64); // matches O shape
    attrs.set_do(dOut);
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateFailsDoShapeMismatchWithO)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    auto o = makeTensor4D(2, 8, 16, 64);
    attrs.set_o(o);
    auto dOut = makeTensor4D(2, 8, 16, 99); // head_dim mismatch
    attrs.set_do(dOut);
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

//==============================================================================
// dropout_mask validation tests
//==============================================================================

TEST(TestSdpaBwdNode, PreValidateSucceedsDropoutMaskExactShape)
{
    // dropout_mask shape (B, H_q, S_q, S_kv) — exact match
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_dropout_mask(makeTensor4D(2, 8, 16, 32));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateSucceedsDropoutMaskBroadcastBatch)
{
    // dropout_mask dim[0]=1 broadcasts over batch
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_dropout_mask(makeTensor4D(1, 8, 16, 32));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateSucceedsDropoutMaskBroadcastHeads)
{
    // dropout_mask dim[1]=1 broadcasts over num_heads
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_dropout_mask(makeTensor4D(2, 1, 16, 32));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateSucceedsDropoutMaskBroadcastBothOuter)
{
    // dropout_mask (1, 1, S_q, S_kv) — both batch and heads broadcast
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_dropout_mask(makeTensor4D(1, 1, 16, 32));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateFailsDropoutMaskWrongRank)
{
    // dropout_mask must be rank-4; rank-3 is invalid
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    auto mask = std::make_shared<TensorAttributes>();
    mask->set_dim({8, 16, 32}); // rank-3
    attrs.set_dropout_mask(mask);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsDropoutMaskBatchMismatch)
{
    // dim[0]=3 is neither batch(2) nor 1
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_dropout_mask(makeTensor4D(3, 8, 16, 32));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsDropoutMaskHeadsMismatch)
{
    // dim[1]=4 is neither num_heads(8) nor 1
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_dropout_mask(makeTensor4D(2, 4, 16, 32));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsDropoutMaskSeqQMismatch)
{
    // dim[2] must exactly equal seq_q(16), not 1
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_dropout_mask(makeTensor4D(2, 8, 8, 32)); // wrong seq_q

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsDropoutMaskSeqKvMismatch)
{
    // dim[3] must exactly equal seq_kv(32), not 1
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_dropout_mask(makeTensor4D(2, 8, 16, 16)); // wrong seq_kv

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

//==============================================================================
// padding_mask validation tests
//==============================================================================

namespace
{
// Helper: create a (B, 1, 1, 1) INT32 tensor for seq_len
std::shared_ptr<TensorAttributes> makeSeqLenTensor(int64_t batch)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim({batch, 1, 1, 1});
    t->set_data_type(DataType::INT32);
    return t;
}
} // namespace

TEST(TestSdpaBwdNode, PreValidateSucceedsPaddingMaskWithValidSeqLens)
{
    // padding_mask=true with valid (B,1,1,1) INT32 seq_len tensors
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_padding_mask(true);
    attrs.set_seq_len_q(makeSeqLenTensor(2));
    attrs.set_seq_len_kv(makeSeqLenTensor(2));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateSucceedsNoPaddingMaskNoSeqLens)
{
    // padding_mask=false (default) — seq_len tensors not required
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    // no seq_len_q / seq_len_kv

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateFailsPaddingMaskMissingSeqLenQ)
{
    // padding_mask=true but seq_len_q not set
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_padding_mask(true);
    attrs.set_seq_len_kv(makeSeqLenTensor(2));
    // seq_len_q intentionally not set

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBwdNode, PreValidateFailsPaddingMaskMissingSeqLenKv)
{
    // padding_mask=true but seq_len_kv not set
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_padding_mask(true);
    attrs.set_seq_len_q(makeSeqLenTensor(2));
    // seq_len_kv intentionally not set

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBwdNode, PreValidateFailsPaddingMaskSeqLenQWrongRank)
{
    // seq_len_q must be rank-4; rank-1 is invalid
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_padding_mask(true);

    auto seqLenQ = std::make_shared<TensorAttributes>();
    seqLenQ->set_dim({2}); // rank-1
    seqLenQ->set_data_type(DataType::INT32);
    attrs.set_seq_len_q(seqLenQ);
    attrs.set_seq_len_kv(makeSeqLenTensor(2));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsPaddingMaskSeqLenQBatchMismatch)
{
    // seq_len_q dim[0] must equal batch (2), not 1
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_padding_mask(true);
    attrs.set_seq_len_q(makeSeqLenTensor(4)); // batch mismatch
    attrs.set_seq_len_kv(makeSeqLenTensor(2));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsPaddingMaskSeqLenQNonOneDim)
{
    // seq_len_q dim[1]/[2]/[3] must be 1
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_padding_mask(true);

    auto seqLenQ = std::make_shared<TensorAttributes>();
    seqLenQ->set_dim({2, 8, 1, 1}); // dim[1] should be 1
    seqLenQ->set_data_type(DataType::INT32);
    attrs.set_seq_len_q(seqLenQ);
    attrs.set_seq_len_kv(makeSeqLenTensor(2));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsPaddingMaskSeqLenQWrongDataType)
{
    // seq_len_q must be INT32; FLOAT is invalid
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_padding_mask(true);

    auto seqLenQ = std::make_shared<TensorAttributes>();
    seqLenQ->set_dim({2, 1, 1, 1});
    seqLenQ->set_data_type(DataType::FLOAT); // wrong dtype
    attrs.set_seq_len_q(seqLenQ);
    attrs.set_seq_len_kv(makeSeqLenTensor(2));

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsPaddingMaskSeqLenKvWrongDataType)
{
    // seq_len_kv must be INT32; HALF is invalid
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_padding_mask(true);
    attrs.set_seq_len_q(makeSeqLenTensor(2));

    auto seqLenKv = std::make_shared<TensorAttributes>();
    seqLenKv->set_dim({2, 1, 1, 1});
    seqLenKv->set_data_type(DataType::HALF); // wrong dtype
    attrs.set_seq_len_kv(seqLenKv);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

//==============================================================================
// infer_properties_node tests
//==============================================================================

TEST(TestSdpaBwdNode, InferPropertiesSetsDqShape)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    auto dq = attrs.get_dq();

    const GraphAttributes graphAttrs;
    SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // dQ should have same shape as Q: [2, 8, 16, 64]
    auto dims = dq->get_dim();
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 8);
    EXPECT_EQ(dims[2], 16);
    EXPECT_EQ(dims[3], 64);
}

TEST(TestSdpaBwdNode, InferPropertiesSetsDkShape)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 4, 32, 64); // GQA: num_kv_heads=4
    auto v = makeTensor4D(2, 4, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    auto dk = attrs.get_dk();

    const GraphAttributes graphAttrs;
    SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // dK should have same shape as K: [2, 4, 32, 64]
    auto dims = dk->get_dim();
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 4);
    EXPECT_EQ(dims[2], 32);
    EXPECT_EQ(dims[3], 64);
}

TEST(TestSdpaBwdNode, InferPropertiesSetsDvShape)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 32); // headDimV=32 differs from headDimQK=64
    auto attrs = makeMinimalAttrs(q, k, v);
    auto dv = attrs.get_dv();

    const GraphAttributes graphAttrs;
    SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // dV should have same shape as V: [2, 8, 32, 32]
    auto dims = dv->get_dim();
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 8);
    EXPECT_EQ(dims[2], 32);
    EXPECT_EQ(dims[3], 32);
}

TEST(TestSdpaBwdNode, InferPropertiesSetsStrides)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    auto dq = attrs.get_dq();

    const GraphAttributes graphAttrs;
    SdpaBwdNode node(std::move(attrs), graphAttrs);
    node.infer_properties_node();

    // Row-major strides for [2, 8, 16, 64]: [8*16*64, 16*64, 64, 1]
    auto strides = dq->get_stride();
    ASSERT_EQ(strides.size(), 4u);
    EXPECT_EQ(strides[3], 1);
    EXPECT_EQ(strides[2], 64);
    EXPECT_EQ(strides[1], 16 * 64);
    EXPECT_EQ(strides[0], 8 * 16 * 64);
}

TEST(TestSdpaBwdNode, InferPropertiesPreservesExplicitShape)
{
    // If dQ already has dims set, they should not be overwritten
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());

    auto dq = std::make_shared<TensorAttributes>();
    dq->set_dim({2, 8, 16, 64});
    dq->set_stride({8192, 1024, 64, 1});
    attrs.set_dq(dq);
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    SdpaBwdNode node(std::move(attrs), graphAttrs);
    node.infer_properties_node();

    EXPECT_EQ(dq->get_dim(), (std::vector<int64_t>{2, 8, 16, 64}));
    EXPECT_EQ(dq->get_stride(), (std::vector<int64_t>{8192, 1024, 64, 1}));
}

TEST(TestSdpaBwdNode, GetNodeTypeReturnsSdpaBwd)
{
    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(SdpaBackwardAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::SDPA_BWD);
}
