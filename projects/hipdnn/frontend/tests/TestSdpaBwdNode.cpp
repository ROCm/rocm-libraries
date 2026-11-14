// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <map>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/SdpaBwdNode.hpp>
#include <unordered_map>
#include <vector>

#include "fake_backend/MockHipdnnBackend.hpp"
#include <optional>

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

TEST(TestSdpaBwdNode, PreValidateChecksDiagonalBandBounds)
{
    // left_bound must be positive or -1 (unbounded); right_bound must be >= -1.
    const auto validate = [](std::optional<int64_t> left, std::optional<int64_t> right) {
        auto attrs = makeMinimalAttrs(
            makeTensor4D(2, 8, 16, 64), makeTensor4D(2, 8, 32, 64), makeTensor4D(2, 8, 32, 64));
        if(left.has_value())
        {
            attrs.set_diagonal_band_left_bound(*left);
        }
        if(right.has_value())
        {
            attrs.set_diagonal_band_right_bound(*right);
        }
        const GraphAttributes graphAttrs;
        const SdpaBwdNode node(std::move(attrs), graphAttrs);
        return node.pre_validate_node().code;
    };
    EXPECT_EQ(validate(1, 0), error_code_t::OK);
    EXPECT_EQ(validate(-1, 0), error_code_t::OK);
    EXPECT_EQ(validate(std::nullopt, -1), error_code_t::OK);
    EXPECT_EQ(validate(0, 0), error_code_t::INVALID_VALUE);
    EXPECT_EQ(validate(-2, std::nullopt), error_code_t::INVALID_VALUE);
    EXPECT_EQ(validate(std::nullopt, -2), error_code_t::INVALID_VALUE);
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

TEST(TestSdpaBwdNode, PreValidateFailsAttnScaleTensorAndValueBothSet)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    // Rule 10: attn_scale tensor and attn_scale_value are mutually exclusive.
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1}).set_is_pass_by_value(true);
    attrs.set_attn_scale(scale);
    attrs.attn_scale_value = 1.0f;

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

//==============================================================================
// Unsupported cuDNN-compat setters (score_mod only) and accepted new fields
//==============================================================================

TEST(TestSdpaBwdNode, PreValidateDrainRunsBeforeShapeChecks)
{
    // Q is rank-3 (shape-invalid) AND an unsupported setter is present. The drain
    // must run first, so the surfaced message is the unsupported one.
    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({2, 8, 16});
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_score_mod([](auto, auto t) { return t; });

    const GraphAttributes graphAttrs;
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
    EXPECT_NE(err.err_msg.find("score modifier"), std::string::npos) << err.err_msg;
    EXPECT_EQ(err.err_msg.find("rank"), std::string::npos) << err.err_msg;
}

TEST(TestSdpaBwdNode, PreValidateFailsEachUnsupportedSetter)
{
    const auto expectUnsupported
        = [](const std::function<void(SdpaBackwardAttributes&)>& applyUnsupported,
             const std::string& expectedFragment) {
              auto q = makeTensor4D(2, 8, 16, 64);
              auto k = makeTensor4D(2, 8, 32, 64);
              auto v = makeTensor4D(2, 8, 32, 64);
              auto attrs = makeMinimalAttrs(q, k, v);
              applyUnsupported(attrs);

              const GraphAttributes graphAttrs;
              const SdpaBwdNode node(std::move(attrs), graphAttrs);
              auto err = node.pre_validate_node();
              EXPECT_EQ(err.code, error_code_t::INVALID_VALUE) << expectedFragment;
              EXPECT_NE(err.err_msg.find(expectedFragment), std::string::npos) << err.err_msg;
          };

    expectUnsupported(
        [](SdpaBackwardAttributes& a) { a.set_score_mod([](auto, auto t) { return t; }); },
        "score modifier");
    expectUnsupported(
        [](SdpaBackwardAttributes& a) { a.set_score_mod_bprop([](auto, auto t) { return t; }); },
        "score-modifier backprop");
}

//==============================================================================
// Philox seed / offset (cuDNN): INT32 or INT64, exactly one element
//==============================================================================

namespace
{
std::shared_ptr<TensorAttributes> makeTyped(const std::vector<int64_t>& dims, DataType type)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim(dims).set_data_type(type);
    return t;
}

// GQA graph (H_q = 8) with a configurable I/O data type.
error_code_t validateBwd(const std::function<void(SdpaBackwardAttributes&)>& configure,
                         DataType io = DataType::HALF)
{
    auto attrs = makeMinimalAttrs(
        makeTensor4D(2, 8, 16, 64), makeTensor4D(2, 2, 32, 64), makeTensor4D(2, 2, 32, 64));
    configure(attrs);
    GraphAttributes graphAttrs;
    graphAttrs.set_io_data_type(io);
    const SdpaBwdNode node(std::move(attrs), graphAttrs);
    return node.pre_validate_node().code;
}

error_code_t validateSeedOffset(const std::shared_ptr<TensorAttributes>& seed,
                                const std::shared_ptr<TensorAttributes>& offset)
{
    return validateBwd([&](SdpaBackwardAttributes& a) { a.set_dropout(0.1f, seed, offset); });
}
} // namespace

TEST(TestSdpaBwdNode, PreValidateAcceptsSeedOffsetSetValueInt32)
{
    auto seed = std::make_shared<TensorAttributes>();
    seed->set_value(int32_t{42});
    auto offset = std::make_shared<TensorAttributes>();
    offset->set_value(int32_t{0});
    EXPECT_EQ(validateSeedOffset(seed, offset), error_code_t::OK);
}

TEST(TestSdpaBwdNode, PreValidateAcceptsSeedOffsetInt64Device1111)
{
    EXPECT_EQ(validateSeedOffset(makeTyped({1, 1, 1, 1}, DataType::INT64),
                                 makeTyped({1, 1, 1, 1}, DataType::INT64)),
              error_code_t::OK);
}

TEST(TestSdpaBwdNode, PreValidateAcceptsSeedOffsetInt64Device1)
{
    EXPECT_EQ(validateSeedOffset(makeTyped({1}, DataType::INT64), makeTyped({1}, DataType::INT32)),
              error_code_t::OK);
}

TEST(TestSdpaBwdNode, PreValidateFailsSeedFloat)
{
    EXPECT_EQ(validateSeedOffset(makeTyped({1}, DataType::FLOAT), makeTyped({1}, DataType::INT64)),
              error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsSeedUnsetDtypeResolvesToIoType)
{
    // An unset dtype becomes the graph I/O type (HALF), which is neither INT32 nor INT64.
    EXPECT_EQ(
        validateSeedOffset(makeTyped({1}, DataType::NOT_SET), makeTyped({1}, DataType::INT64)),
        error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsSeedTwoElements)
{
    EXPECT_EQ(validateSeedOffset(makeTyped({2}, DataType::INT64), makeTyped({1}, DataType::INT64)),
              error_code_t::INVALID_VALUE);
    EXPECT_EQ(validateSeedOffset(makeTyped({1}, DataType::INT64),
                                 makeTyped({1, 2, 1, 1}, DataType::INT64)),
              error_code_t::INVALID_VALUE)
        << "offset follows the same single-element rule";
}

TEST(TestSdpaBwdNode, PreValidateFailsOffsetWrongType)
{
    EXPECT_EQ(validateSeedOffset(makeTyped({1}, DataType::INT64), makeTyped({1}, DataType::HALF)),
              error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsSeedRank5)
{
    EXPECT_EQ(validateSeedOffset(makeTyped({1, 1, 1, 1, 1}, DataType::INT64),
                                 makeTyped({1}, DataType::INT64)),
              error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsSeedOffsetEmptyDims)
{
    EXPECT_EQ(validateSeedOffset(makeTyped({}, DataType::INT64), makeTyped({1}, DataType::INT64)),
              error_code_t::INVALID_VALUE);
    EXPECT_EQ(validateSeedOffset(makeTyped({1}, DataType::INT64), makeTyped({}, DataType::INT64)),
              error_code_t::INVALID_VALUE);
}

//==============================================================================
// Sink / dSink: [1, H_q, 1, 1], FLOAT, or Q's data type set on the tensor itself
//==============================================================================

TEST(TestSdpaBwdNode, PreValidateSucceedsSinkAndDsinkFloat)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
              }),
              error_code_t::OK);
}

TEST(TestSdpaBwdNode, PreValidateSucceedsSinkAndDsinkIoDtype)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::HALF));
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::HALF));
              }),
              error_code_t::OK);
}

TEST(TestSdpaBwdNode, PreValidateFailsUntypedSinkInA16BitGraph)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::NOT_SET));
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::HALF));
              }),
              error_code_t::INVALID_VALUE)
        << "an untyped sink takes the io type, which must be FLOAT";
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::HALF));
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::NOT_SET));
              }),
              error_code_t::INVALID_VALUE)
        << "an untyped dSink takes the io type, which must be FLOAT";
    EXPECT_EQ(validateBwd(
                  [](SdpaBackwardAttributes& a) {
                      a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::NOT_SET));
                      a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::NOT_SET));
                  },
                  DataType::FLOAT),
              error_code_t::OK)
        << "untyped in a FLOAT graph: both are FLOAT";
}

// A sink or dSink type supplied by fill_from_context counts as untyped.
TEST(TestSdpaBwdNode, PreValidateSinkTypeFilledFromTheGraphIsUntyped)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_io_data_type(DataType::HALF);
    auto filled = makeTyped({1, 8, 1, 1}, DataType::NOT_SET);
    filled->fill_from_context(graphAttrs);
    ASSERT_EQ(filled->get_data_type(), DataType::HALF);
    EXPECT_EQ(validateBwd([&](SdpaBackwardAttributes& a) {
                  a.set_sink_token(filled);
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::HALF));
              }),
              error_code_t::INVALID_VALUE)
        << "sink HALF came from the graph";
    EXPECT_EQ(validateBwd([&](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::HALF));
                  a.set_dsink_token(filled);
              }),
              error_code_t::INVALID_VALUE)
        << "dSink HALF came from the graph";
}

TEST(TestSdpaBwdNode, PreValidateFailsSinkAndDsinkWhenNoDataTypeResolves)
{
    EXPECT_EQ(validateBwd(
                  [](SdpaBackwardAttributes& a) {
                      a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::NOT_SET));
                  },
                  DataType::NOT_SET),
              error_code_t::INVALID_VALUE)
        << "sink and Q are both unset: no data type to compare against";
    EXPECT_EQ(validateBwd(
                  [](SdpaBackwardAttributes& a) {
                      a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::NOT_SET));
                      a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::NOT_SET));
                  },
                  DataType::NOT_SET),
              error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateSucceedsSinkAndDsinkInQExplicitDtype)
{
    const auto configure = [](SdpaBackwardAttributes& a) {
        a.get_q()->set_data_type(DataType::BFLOAT16);
        a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::BFLOAT16));
        a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::BFLOAT16));
    };
    EXPECT_EQ(validateBwd(configure, DataType::HALF), error_code_t::OK)
        << "the referent is Q's own type, not the graph io type";
    EXPECT_EQ(validateBwd(configure, DataType::NOT_SET), error_code_t::OK);
}

TEST(TestSdpaBwdNode, PreValidateFailsSinkInIoDtypeWhenQHasExplicitDtype)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.get_q()->set_data_type(DataType::BFLOAT16);
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::HALF));
              }),
              error_code_t::INVALID_VALUE)
        << "equal to the io type (HALF) but not to Q's explicit type";
}

TEST(TestSdpaBwdNode, PreValidateFailsSinkWrongDims)
{
    for(const auto& dims : {std::vector<int64_t>{1, 2, 1, 1},
                            std::vector<int64_t>{2, 8, 1, 1},
                            std::vector<int64_t>{1, 8, 2, 1},
                            std::vector<int64_t>{1, 8, 1, 2},
                            std::vector<int64_t>{8},
                            std::vector<int64_t>{1, 8, 1, 1, 1},
                            std::vector<int64_t>{1, 8, 1}})
    {
        EXPECT_EQ(validateBwd([&](SdpaBackwardAttributes& a) {
                      a.set_sink_token(makeTyped(dims, DataType::FLOAT));
                  }),
                  error_code_t::INVALID_VALUE);
    }
}

TEST(TestSdpaBwdNode, PreValidateFailsSinkDtypeNeitherFloatNorIo)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::INT32));
              }),
              error_code_t::INVALID_VALUE);
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::BFLOAT16));
              }),
              error_code_t::INVALID_VALUE)
        << "a half-precision type other than the io type";
}

TEST(TestSdpaBwdNode, PreValidateFailsDsinkWrongDims)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
                  a.set_dsink_token(makeTyped({1, 2, 1, 1}, DataType::FLOAT));
              }),
              error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsDsinkDtypeNeitherFloatNorIo)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::INT32));
              }),
              error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsDsinkDtypeDiffersFromSink)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::HALF));
              }),
              error_code_t::INVALID_VALUE);
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::HALF));
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
              }),
              error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBwdNode, PreValidateFailsDsinkWithoutSinkAttributeNotSet)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
              }),
              error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBwdNode, PreValidateSucceedsSinkWithoutDsink)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
              }),
              error_code_t::OK);
}

TEST(TestSdpaBwdNode, PreValidateSucceedsSinkWithRaggedQ)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.get_q()->set_ragged_offset(makeTyped({3, 1, 1, 1}, DataType::INT32));
                  a.set_sink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
                  a.set_dsink_token(makeTyped({1, 8, 1, 1}, DataType::FLOAT));
              }),
              error_code_t::OK);
}

//==============================================================================
// rng_dump, deterministic, max_total_seq_len
//==============================================================================

TEST(TestSdpaBwdNode, PreValidateSucceedsRngDumpWithoutDropout)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.set_rng_dump(makeTyped({2, 8, 16, 32}, DataType::FLOAT));
              }),
              error_code_t::OK)
        << "cuDNN ignores a dump without dropout";
}

TEST(TestSdpaBwdNode, PreValidateFailsMaxTotalSeqLenWithoutRagged)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) { a.set_max_total_seq_len_q(128); }),
              error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) { a.set_max_total_seq_len_kv(128); }),
              error_code_t::GRAPH_NOT_SUPPORTED);
}

TEST(TestSdpaBwdNode, PreValidateSucceedsMaxTotalSeqLenWithRaggedQ)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.get_q()->set_ragged_offset(makeTyped({3, 1, 1, 1}, DataType::INT32));
                  a.set_max_total_seq_len_q(128).set_max_total_seq_len_kv(256);
              }),
              error_code_t::OK);
}

TEST(TestSdpaBwdNode, PreValidateSucceedsMaxTotalSeqLenWithRaggedOnlyO)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.get_o()->set_ragged_offset(makeTyped({3, 1, 1, 1}, DataType::INT32));
                  a.set_max_total_seq_len_q(128);
              }),
              error_code_t::OK)
        << "any of Q, K, V, O counts as packed";
}

TEST(TestSdpaBwdNode, PreValidateSucceedsMaxTotalSeqLenNonPositive)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.get_k()->set_ragged_offset(makeTyped({3, 1, 1, 1}, DataType::INT32));
                  a.set_max_total_seq_len_q(0).set_max_total_seq_len_kv(-5);
              }),
              error_code_t::OK)
        << "no positivity check, as cuDNN";
}

TEST(TestSdpaBwdNode, PreValidateSucceedsMaxTotalSeqLenWithRaggedOnlyV)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) {
                  a.get_v()->set_ragged_offset(makeTyped({3, 1, 1, 1}, DataType::INT32));
                  a.set_max_total_seq_len_kv(128);
              }),
              error_code_t::OK)
        << "V alone counts as packed";
}

TEST(TestSdpaBwdNode, PreValidateSucceedsDeterministicTrueAndFalse)
{
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) { a.set_deterministic_algorithm(true); }),
              error_code_t::OK);
    EXPECT_EQ(validateBwd([](SdpaBackwardAttributes& a) { a.set_deterministic_algorithm(false); }),
              error_code_t::OK);
}

//==============================================================================
// Packer / unpacker round trip through a recording backend (no device)
//==============================================================================

namespace
{
// Backend fake that stores every SetAttribute per (descriptor, attribute) and answers
// GetAttribute from that store; an attribute never set reports zero elements.
class TestSdpaBwdPackUnpack : public ::testing::Test
{
protected:
    void SetUp() override
    {
        using ::testing::_;

        _backend = std::make_shared<::testing::NiceMock<Mock_hipdnn_backend>>();
        hipdnn_frontend::detail::IHipdnnBackend::setInstance(_backend);

        ON_CALL(*_backend, backendCreateDescriptor(_, _))
            .WillByDefault([this](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(
                    &_descriptorPool[_nextDescriptor++]);
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_backend, backendDestroyDescriptor(_))
            .WillByDefault(::testing::Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_backend, backendFinalize(_))
            .WillByDefault(::testing::Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_backend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t desc,
                                  hipdnnBackendAttributeName_t name,
                                  hipdnnBackendAttributeType_t type,
                                  int64_t count,
                                  const void* elements) {
                Stored stored;
                stored.count = count;
                if(elements != nullptr && count > 0)
                {
                    stored.bytes.resize(static_cast<size_t>(count) * elementSize(type));
                    std::memcpy(stored.bytes.data(), elements, stored.bytes.size());
                }
                _store[key(desc, name)] = std::move(stored);
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_backend, backendGetAttribute(_, _, _, _, _, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t desc,
                                  hipdnnBackendAttributeName_t name,
                                  hipdnnBackendAttributeType_t type,
                                  int64_t requested,
                                  int64_t* count,
                                  void* elements) {
                const auto found = _store.find(key(desc, name));
                if(found == _store.end() || found->second.bytes.empty())
                {
                    *count = 0;
                    return HIPDNN_STATUS_SUCCESS;
                }
                const auto n = std::min(requested, found->second.count);
                if(elements == nullptr)
                {
                    *count = found->second.count;
                    return HIPDNN_STATUS_SUCCESS;
                }
                std::memcpy(elements,
                            found->second.bytes.data(),
                            static_cast<size_t>(n) * elementSize(type));
                *count = n;
                return HIPDNN_STATUS_SUCCESS;
            });
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::resetInstance();
        _backend.reset();
    }

    static std::shared_ptr<TensorAttributes>
        makeUidTensor(int64_t uid, const std::vector<int64_t>& dims, DataType type)
    {
        std::vector<int64_t> strides(dims.size(), 1);
        for(size_t i = dims.size(); i-- > 1;)
        {
            strides[i - 1] = strides[i] * dims[i];
        }
        auto t = std::make_shared<TensorAttributes>();
        t->set_uid(uid).set_dim(dims).set_stride(strides).set_data_type(type);
        return t;
    }

private:
    struct Stored
    {
        int64_t count = 0;
        std::vector<std::byte> bytes;
    };

    static size_t elementSize(hipdnnBackendAttributeType_t type)
    {
        switch(type)
        {
        case HIPDNN_TYPE_BACKEND_DESCRIPTOR:
            return sizeof(hipdnnBackendDescriptor_t);
        case HIPDNN_TYPE_BOOLEAN:
            return sizeof(bool);
        case HIPDNN_TYPE_INT64:
            return sizeof(int64_t);
        case HIPDNN_TYPE_CHAR:
            return 1;
        default: // FLOAT and the enum types
            return sizeof(int32_t);
        }
    }

    static std::pair<std::uintptr_t, int64_t> key(hipdnnBackendDescriptor_t desc,
                                                  hipdnnBackendAttributeName_t name)
    {
        return {reinterpret_cast<std::uintptr_t>(desc), static_cast<int64_t>(name)};
    }

    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>> _backend;
    std::array<char, 256> _descriptorPool{};
    size_t _nextDescriptor = 0;
    std::map<std::pair<std::uintptr_t, int64_t>, Stored> _store;
};
} // namespace

TEST_F(TestSdpaBwdPackUnpack, SinkDsinkRngDumpDeterministicAndMaxTotalSeqLenSurviveRoundTrip)
{
    SdpaBackwardAttributes attrs;
    attrs.set_q(makeUidTensor(1, {2, 8, 16, 64}, DataType::FLOAT));
    attrs.set_k(makeUidTensor(2, {2, 8, 32, 64}, DataType::FLOAT));
    attrs.set_v(makeUidTensor(3, {2, 8, 32, 64}, DataType::FLOAT));
    attrs.set_o(makeUidTensor(4, {2, 8, 16, 64}, DataType::FLOAT));
    attrs.set_do(makeUidTensor(5, {2, 8, 16, 64}, DataType::FLOAT));
    attrs.set_stats(makeUidTensor(6, {2, 8, 16, 1}, DataType::FLOAT));
    attrs.set_dq(makeUidTensor(7, {2, 8, 16, 64}, DataType::FLOAT));
    attrs.set_dk(makeUidTensor(8, {2, 8, 32, 64}, DataType::FLOAT));
    attrs.set_dv(makeUidTensor(9, {2, 8, 32, 64}, DataType::FLOAT));
    attrs.set_sink_token(makeUidTensor(10, {1, 8, 1, 1}, DataType::FLOAT));
    attrs.set_dsink_token(makeUidTensor(11, {1, 8, 1, 1}, DataType::FLOAT));
    attrs.set_rng_dump(makeUidTensor(12, {2, 8, 16, 32}, DataType::FLOAT));
    attrs.set_deterministic_algorithm(true);
    attrs.set_max_total_seq_len_q(128).set_max_total_seq_len_kv(256);
    attrs.set_compute_data_type(DataType::FLOAT);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode packed(std::move(attrs), graphAttrs);
    std::unordered_map<int64_t, hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> operations;
    const auto packError = packed.create_operation(tensorDescs, operations);
    ASSERT_EQ(packError.code, error_code_t::OK) << packError.err_msg;
    ASSERT_EQ(operations.size(), 1U);

    SdpaBwdNode unpacked(SdpaBackwardAttributes{}, graphAttrs);
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    const auto unpackError = unpacked.unpack_from_descriptor(operations[0].get(), tensorMap);
    ASSERT_EQ(unpackError.code, error_code_t::OK) << unpackError.err_msg;

    const auto& out = unpacked.attributes;
    ASSERT_NE(out.get_sink_token(), nullptr);
    EXPECT_EQ(out.get_sink_token()->get_uid(), 10);
    EXPECT_EQ(out.get_sink_token()->get_dim(), (std::vector<int64_t>{1, 8, 1, 1}));
    ASSERT_NE(out.get_dsink_token(), nullptr);
    EXPECT_EQ(out.get_dsink_token()->get_uid(), 11);
    ASSERT_NE(out.get_rng_dump(), nullptr);
    EXPECT_EQ(out.get_rng_dump()->get_uid(), 12);
    EXPECT_TRUE(out.is_deterministic_algorithm);
    EXPECT_EQ(out.max_total_seq_len_q, std::optional<int64_t>{128});
    EXPECT_EQ(out.max_total_seq_len_kv, std::optional<int64_t>{256});
}

TEST_F(TestSdpaBwdPackUnpack, UnsetNewFieldsStayUnsetAfterRoundTrip)
{
    SdpaBackwardAttributes attrs;
    attrs.set_q(makeUidTensor(1, {2, 8, 16, 64}, DataType::FLOAT));
    attrs.set_k(makeUidTensor(2, {2, 8, 32, 64}, DataType::FLOAT));
    attrs.set_v(makeUidTensor(3, {2, 8, 32, 64}, DataType::FLOAT));
    attrs.set_o(makeUidTensor(4, {2, 8, 16, 64}, DataType::FLOAT));
    attrs.set_do(makeUidTensor(5, {2, 8, 16, 64}, DataType::FLOAT));
    attrs.set_stats(makeUidTensor(6, {2, 8, 16, 1}, DataType::FLOAT));
    attrs.set_dq(makeUidTensor(7, {2, 8, 16, 64}, DataType::FLOAT));
    attrs.set_dk(makeUidTensor(8, {2, 8, 32, 64}, DataType::FLOAT));
    attrs.set_dv(makeUidTensor(9, {2, 8, 32, 64}, DataType::FLOAT));
    attrs.set_compute_data_type(DataType::FLOAT);

    const GraphAttributes graphAttrs;
    const SdpaBwdNode packed(std::move(attrs), graphAttrs);
    std::unordered_map<int64_t, hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> operations;
    ASSERT_EQ(packed.create_operation(tensorDescs, operations).code, error_code_t::OK);

    SdpaBwdNode unpacked(SdpaBackwardAttributes{}, graphAttrs);
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    ASSERT_EQ(unpacked.unpack_from_descriptor(operations[0].get(), tensorMap).code,
              error_code_t::OK);

    const auto& out = unpacked.attributes;
    EXPECT_EQ(out.get_sink_token(), nullptr);
    EXPECT_EQ(out.get_dsink_token(), nullptr);
    EXPECT_EQ(out.get_rng_dump(), nullptr);
    EXPECT_FALSE(out.is_deterministic_algorithm);
    EXPECT_FALSE(out.max_total_seq_len_q.has_value());
    EXPECT_FALSE(out.max_total_seq_len_kv.has_value());
}
