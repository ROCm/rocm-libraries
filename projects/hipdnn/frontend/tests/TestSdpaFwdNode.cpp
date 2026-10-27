// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/SdpaFwdNode.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

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

TEST(TestSdpaFwdNode, PreValidateChecksDiagonalBandBounds)
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
        const SdpaFwdNode node(std::move(attrs), graphAttrs);
        return node.pre_validate_node().code;
    };
    EXPECT_EQ(validate(1, 0), error_code_t::OK);
    EXPECT_EQ(validate(-1, 0), error_code_t::OK);
    EXPECT_EQ(validate(std::nullopt, -1), error_code_t::OK);
    EXPECT_EQ(validate(0, 0), error_code_t::INVALID_VALUE);
    EXPECT_EQ(validate(-2, std::nullopt), error_code_t::INVALID_VALUE);
    EXPECT_EQ(validate(std::nullopt, -2), error_code_t::INVALID_VALUE);
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

// Paged K/V (cuDNN layout): containers [num_blocks, H, block_size, D] with page tables
// [B, 1, table_size, 1]. The container's leading dim is a block count and dim 2 a block size,
// so neither is compared with Q's batch or between K and V. The node applies cuDNN's paged rules
// (INT32 tables of width ceil(max_seq_len_kv / block_size), both seq_len tensors); whether an
// engine supports paging is the engine's decision.
namespace
{
std::shared_ptr<TensorAttributes> makePageTable(int64_t batch, int64_t tableSize)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim({batch, 1, tableSize, 1}).set_data_type(DataType::INT32);
    return t;
}

std::shared_ptr<TensorAttributes> makeSeqLen(int64_t batch)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim({batch, 1, 1, 1}).set_data_type(DataType::INT32);
    return t;
}

// The minimal attributes plus the seq_len tensors that paging needs.
SdpaAttributes makePaddedAttrs(const std::shared_ptr<TensorAttributes>& q,
                               const std::shared_ptr<TensorAttributes>& k,
                               const std::shared_ptr<TensorAttributes>& v)
{
    auto attrs = makeMinimalAttrs(q, k, v);
    attrs.set_seq_len_q(makeSeqLen(q->get_dim()[0]));
    attrs.set_seq_len_kv(makeSeqLen(q->get_dim()[0]));
    return attrs;
}
} // namespace

TEST(TestSdpaFwdNode, PreValidateSucceedsPagedKvWithBlockCountUnlikeBatch)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(37, 2, 16, 64); // 37 blocks of 16 keys
    auto v = makeTensor4D(37, 2, 16, 64);
    auto attrs = makePaddedAttrs(q, k, v);
    attrs.set_paged_attention_k_table(makePageTable(2, 8));
    attrs.set_paged_attention_v_table(makePageTable(2, 8));
    attrs.set_paged_attention_max_seq_len_kv(128); // ceil(128 / 16) = 8 table entries

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateSucceedsPagedKWithDenseV)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(9, 2, 4, 64); // paged container, block size 4
    auto v = makeTensor4D(2, 2, 32, 64); // dense V, 32 keys
    auto attrs = makePaddedAttrs(q, k, v);
    attrs.set_paged_attention_k_table(makePageTable(2, 8));

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaFwdNode, PreValidateFailsPagedDenseSideBatchMismatch)
{
    // K is paged, but the dense V still has to carry Q's batch.
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(9, 2, 4, 64);
    auto v = makeTensor4D(3, 2, 32, 64);
    auto attrs = makePaddedAttrs(q, k, v);
    attrs.set_paged_attention_k_table(makePageTable(2, 8));

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidateFailsPageTableBatchMismatch)
{
    // Every batch needs a page-table row.
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(9, 2, 4, 64);
    auto v = makeTensor4D(9, 2, 4, 64);
    auto attrs = makePaddedAttrs(q, k, v);
    attrs.set_paged_attention_k_table(makePageTable(2, 8));
    attrs.set_paged_attention_v_table(makePageTable(3, 8));

    const GraphAttributes graphAttrs;
    const SdpaFwdNode node(std::move(attrs), graphAttrs);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaFwdNode, PreValidatePagedBiasMustMatchMaxSeqLenKv)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(9, 2, 4, 64);
    auto v = makeTensor4D(9, 2, 4, 64);
    for(const auto& [biasKv, maxSeqLenKv, ok] :
        {std::tuple{int64_t{30}, std::optional<int32_t>{30}, true},
         std::tuple{int64_t{31}, std::optional<int32_t>{30}, false},
         std::tuple{int64_t{1}, std::optional<int32_t>{30}, true},
         // Without max_seq_len_kv the key length is not known to the node: the engine checks.
         std::tuple{int64_t{31}, std::optional<int32_t>{}, true}})
    {
        auto attrs = makePaddedAttrs(q, k, v);
        attrs.set_paged_attention_k_table(makePageTable(2, 8));
        attrs.set_paged_attention_v_table(makePageTable(2, 8));
        if(maxSeqLenKv.has_value())
        {
            attrs.set_paged_attention_max_seq_len_kv(maxSeqLenKv.value());
        }
        attrs.set_bias(makeTensor4D(2, 8, 16, biasKv));

        const GraphAttributes graphAttrs;
        const SdpaFwdNode node(std::move(attrs), graphAttrs);
        EXPECT_EQ(node.pre_validate_node().code == error_code_t::OK, ok)
            << "bias last dim " << biasKv;
    }
}

// cuDNN's paged rules, one field at a time from an accepted graph: unpacked INT32 [B, 1, M, 1]
// tables with M > 0, M == ceil(max_seq_len_kv / block_size) when max_seq_len_kv is given, a
// positive block size, both seq_len tensors, and no max_seq_len_kv without paging.
TEST(TestSdpaFwdNode, PreValidatePagedRulesFollowCudnn)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(9, 2, 4, 64); // block size 4
    auto v = makeTensor4D(9, 2, 4, 64);
    const auto validate = [](SdpaAttributes attrs) {
        const GraphAttributes graphAttrs;
        const SdpaFwdNode node(std::move(attrs), graphAttrs);
        return node.pre_validate_node().code;
    };
    const auto paged = [&](int64_t tableSize, std::optional<int32_t> maxSeqLenKv) {
        auto attrs = makePaddedAttrs(q, k, v);
        attrs.set_paged_attention_k_table(makePageTable(2, tableSize));
        attrs.set_paged_attention_v_table(makePageTable(2, tableSize));
        if(maxSeqLenKv.has_value())
        {
            attrs.set_paged_attention_max_seq_len_kv(maxSeqLenKv.value());
        }
        return attrs;
    };

    EXPECT_EQ(validate(paged(8, 30)), error_code_t::OK);
    EXPECT_EQ(validate(paged(8, std::nullopt)), error_code_t::OK) << "the engine infers the extent";
    EXPECT_EQ(validate(paged(9, 30)), error_code_t::INVALID_VALUE) << "table wider than ceil(30/4)";
    EXPECT_EQ(validate(paged(7, 30)), error_code_t::INVALID_VALUE) << "table narrower";
    EXPECT_EQ(validate(paged(8, 0)), error_code_t::INVALID_VALUE) << "non-positive extent";

    auto floatTable = paged(8, 30);
    auto floatK = makePageTable(2, 8);
    floatK->set_data_type(DataType::FLOAT);
    floatTable.set_paged_attention_k_table(floatK);
    EXPECT_EQ(validate(floatTable), error_code_t::INVALID_VALUE) << "INT32 tables";

    auto rank3Table = paged(8, 30);
    auto rank3 = std::make_shared<TensorAttributes>();
    rank3->set_dim({2, 1, 8}).set_data_type(DataType::INT32);
    rank3Table.set_paged_attention_v_table(rank3);
    EXPECT_EQ(validate(rank3Table), error_code_t::INVALID_VALUE) << "rank-4 tables";

    for(const auto& dims : {std::vector<int64_t>{2, 2, 8, 1}, std::vector<int64_t>{2, 1, 8, 2}})
    {
        auto wideTable = paged(8, 30);
        auto table = std::make_shared<TensorAttributes>();
        table->set_dim(dims).set_data_type(DataType::INT32);
        wideTable.set_paged_attention_k_table(table);
        EXPECT_EQ(validate(wideTable), error_code_t::INVALID_VALUE)
            << "tables are [B, 1, M, 1], got dims " << dims[1] << ", " << dims[3];
    }
    EXPECT_EQ(validate(paged(0, std::nullopt)), error_code_t::INVALID_VALUE)
        << "at least one table entry";

    auto packedTable = paged(8, 30);
    auto packedK = makePageTable(2, 8);
    packedK->set_ragged_offset(makeSeqLen(3));
    packedTable.set_paged_attention_k_table(packedK);
    EXPECT_EQ(validate(packedTable), error_code_t::INVALID_VALUE) << "tables are not packed";

    auto zeroBlocks = makePaddedAttrs(q, makeTensor4D(9, 2, 0, 64), v);
    zeroBlocks.set_paged_attention_k_table(makePageTable(2, 8));
    zeroBlocks.set_paged_attention_v_table(makePageTable(2, 8));
    EXPECT_EQ(validate(zeroBlocks), error_code_t::INVALID_VALUE) << "positive block size";

    auto noSeqLens = paged(8, 30);
    noSeqLens.set_seq_len_q(nullptr);
    EXPECT_EQ(validate(noSeqLens), error_code_t::ATTRIBUTE_NOT_SET) << "paging needs padding";
    auto noSeqLenKv = paged(8, 30);
    noSeqLenKv.set_seq_len_kv(nullptr);
    EXPECT_EQ(validate(noSeqLenKv), error_code_t::ATTRIBUTE_NOT_SET) << "paging needs seq_len_kv";

    auto unpaged = makeMinimalAttrs(q, makeTensor4D(2, 2, 32, 64), makeTensor4D(2, 2, 32, 64));
    unpaged.set_paged_attention_max_seq_len_kv(32);
    EXPECT_EQ(validate(unpaged), error_code_t::INVALID_VALUE) << "max_seq_len_kv needs paging";
}

// cuDNN's seq_len_q / seq_len_kv: INT32 [B, 1, 1, 1].
TEST(TestSdpaFwdNode, PreValidateSeqLenShapeAndType)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    const auto validate = [&](const std::shared_ptr<TensorAttributes>& seqLenQ,
                              const std::shared_ptr<TensorAttributes>& seqLenKv) {
        auto attrs = makePaddedAttrs(q, k, v);
        attrs.set_seq_len_q(seqLenQ);
        attrs.set_seq_len_kv(seqLenKv);
        const GraphAttributes graphAttrs;
        const SdpaFwdNode node(std::move(attrs), graphAttrs);
        return node.pre_validate_node().code;
    };
    const auto withDims = [](const std::vector<int64_t>& dims) {
        auto t = std::make_shared<TensorAttributes>();
        t->set_dim(dims).set_data_type(DataType::INT32);
        return t;
    };
    const auto valid = makeSeqLen(2);
    EXPECT_EQ(validate(valid, valid), error_code_t::OK);

    // Each invalid tensor is rejected as seq_len_q and as seq_len_kv.
    auto int64Len = makeSeqLen(2);
    int64Len->set_data_type(DataType::INT64);
    for(const auto& [invalid, rule] : {std::pair{makeSeqLen(3), "batch"},
                                       std::pair{withDims({2}), "rank"},
                                       std::pair{withDims({2, 2, 1, 1}), "dim 1"},
                                       std::pair{withDims({2, 1, 2, 1}), "dim 2"},
                                       std::pair{withDims({2, 1, 1, 2}), "dim 3"},
                                       std::pair{int64Len, "INT32"}})
    {
        EXPECT_EQ(validate(invalid, valid), error_code_t::INVALID_VALUE) << "seq_len_q " << rule;
        EXPECT_EQ(validate(valid, invalid), error_code_t::INVALID_VALUE) << "seq_len_kv " << rule;
    }
}

// cuDNN's sink: FLOAT [1, H_q, 1, 1] (an unset dtype takes the graph io type).
TEST(TestSdpaFwdNode, PreValidateSinkShapeAndType)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 2, 32, 64);
    auto v = makeTensor4D(2, 2, 32, 64);
    const auto validate = [&](const std::vector<int64_t>& dims, DataType type, DataType io) {
        auto attrs = makeMinimalAttrs(q, k, v);
        auto sink = std::make_shared<TensorAttributes>();
        sink->set_dim(dims).set_data_type(type);
        attrs.set_sink_token(sink);
        GraphAttributes graphAttrs;
        graphAttrs.set_io_data_type(io);
        const SdpaFwdNode node(std::move(attrs), graphAttrs);
        return node.pre_validate_node().code;
    };
    EXPECT_EQ(validate({1, 8, 1, 1}, DataType::FLOAT, DataType::HALF), error_code_t::OK);
    EXPECT_EQ(validate({1, 2, 1, 1}, DataType::FLOAT, DataType::HALF), error_code_t::INVALID_VALUE)
        << "one logit per query head";
    for(const auto& dims : {std::vector<int64_t>{2, 8, 1, 1},
                            std::vector<int64_t>{1, 8, 2, 1},
                            std::vector<int64_t>{1, 8, 1, 2}})
    {
        EXPECT_EQ(validate(dims, DataType::FLOAT, DataType::HALF), error_code_t::INVALID_VALUE)
            << "dims other than H_q are 1, got " << dims[0] << ", " << dims[2] << ", " << dims[3];
    }
    EXPECT_EQ(validate({8}, DataType::FLOAT, DataType::HALF), error_code_t::INVALID_VALUE)
        << "rank 4";
    EXPECT_EQ(validate({1, 8, 1, 1}, DataType::HALF, DataType::HALF), error_code_t::INVALID_VALUE)
        << "FLOAT only, as cuDNN";
    EXPECT_EQ(validate({1, 8, 1, 1}, DataType::NOT_SET, DataType::BFLOAT16),
              error_code_t::INVALID_VALUE)
        << "an unset dtype becomes the io type";
    EXPECT_EQ(validate({1, 8, 1, 1}, DataType::NOT_SET, DataType::FLOAT), error_code_t::OK);
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
