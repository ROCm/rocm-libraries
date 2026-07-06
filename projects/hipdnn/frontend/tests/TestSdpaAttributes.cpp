// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>

using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend;

namespace
{
std::shared_ptr<TensorAttributes> makeTensor(int64_t uid)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_uid(uid);
    return t;
}
} // namespace

TEST(TestSdpaAttributes, DefaultValues)
{
    const SdpaAttributes attrs;

    // Required I/O tensors
    EXPECT_EQ(attrs.get_q(), nullptr);
    EXPECT_EQ(attrs.get_k(), nullptr);
    EXPECT_EQ(attrs.get_v(), nullptr);
    EXPECT_EQ(attrs.get_o(), nullptr);

    // Optional input tensors
    EXPECT_EQ(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_attn_scale(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_q(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_kv(), nullptr);
    EXPECT_EQ(attrs.get_seed(), nullptr);
    EXPECT_EQ(attrs.get_offset(), nullptr);
    EXPECT_EQ(attrs.get_dropout_mask(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale(), nullptr);
    EXPECT_EQ(attrs.get_page_table_k(), nullptr);
    EXPECT_EQ(attrs.get_page_table_v(), nullptr);
    EXPECT_EQ(attrs.get_block_mask(), nullptr);
    EXPECT_EQ(attrs.get_sink_token(), nullptr);
    EXPECT_EQ(attrs.get_descale_q(), nullptr);
    EXPECT_EQ(attrs.get_descale_k(), nullptr);
    EXPECT_EQ(attrs.get_descale_v(), nullptr);
    EXPECT_EQ(attrs.get_descale_s(), nullptr);
    EXPECT_EQ(attrs.get_scale_s(), nullptr);
    EXPECT_EQ(attrs.get_scale_o(), nullptr);

    // Optional output tensors
    EXPECT_EQ(attrs.get_stats(), nullptr);
    EXPECT_EQ(attrs.get_max(), nullptr);
    EXPECT_EQ(attrs.get_sum_exp(), nullptr);
    EXPECT_EQ(attrs.get_rng_dump(), nullptr);
    EXPECT_EQ(attrs.get_amax_s(), nullptr);
    EXPECT_EQ(attrs.get_amax_o(), nullptr);

    // Boolean flags
    EXPECT_FALSE(attrs.generate_stats.has_value());
    EXPECT_FALSE(attrs.alibi_mask);
    EXPECT_FALSE(attrs.padding_mask);
    EXPECT_FALSE(attrs.causal_mask);
    EXPECT_FALSE(attrs.causal_mask_bottom_right);

    // Scalar attributes
    EXPECT_FALSE(attrs.dropout_probability.has_value());
    EXPECT_FALSE(attrs.attn_scale_value.has_value());
    EXPECT_FALSE(attrs.left_bound.has_value());
    EXPECT_FALSE(attrs.right_bound.has_value());
    EXPECT_FALSE(attrs.max_seq_len_kv.has_value());

    // Enum defaults
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(attrs.mma_core_mode, DataType::NOT_SET);
    EXPECT_EQ(attrs.implementation, AttentionImplementation::AUTO);
}

TEST(TestSdpaAttributes, SetRequiredTensors)
{
    SdpaAttributes attrs;

    auto q = makeTensor(1);
    auto k = makeTensor(2);
    auto v = makeTensor(3);
    auto o = makeTensor(4);

    attrs.set_q(q).set_k(k).set_v(v).set_o(o);

    EXPECT_EQ(attrs.get_q(), q);
    EXPECT_EQ(attrs.get_k(), k);
    EXPECT_EQ(attrs.get_v(), v);
    EXPECT_EQ(attrs.get_o(), o);

    // Unset optional tensors remain null
    EXPECT_EQ(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_attn_scale(), nullptr);
    EXPECT_EQ(attrs.get_stats(), nullptr);
}

TEST(TestSdpaAttributes, SetInputTensors)
{
    SdpaAttributes attrs;

    auto attnMask = makeTensor(10);
    auto scale = makeTensor(11);
    auto seqLenQ = makeTensor(12);
    auto seqLenKv = makeTensor(13);
    auto seed = makeTensor(14);
    auto offset = makeTensor(15);
    auto dropoutMask = makeTensor(16);
    auto dropoutScale = makeTensor(17);
    auto pageTableK = makeTensor(18);
    auto pageTableV = makeTensor(19);
    auto blockMask = makeTensor(20);
    auto sinkToken = makeTensor(21);
    auto descaleQ = makeTensor(22);
    auto descaleK = makeTensor(23);
    auto descaleV = makeTensor(24);
    auto descaleS = makeTensor(25);
    auto scaleS = makeTensor(26);
    auto scaleO = makeTensor(27);

    attrs.set_bias(attnMask)
        .set_attn_scale(scale)
        .set_seq_len_q(seqLenQ)
        .set_seq_len_kv(seqLenKv)
        .set_seed(seed)
        .set_offset(offset)
        .set_dropout_mask(dropoutMask)
        .set_dropout_scale(dropoutScale)
        .set_page_table_k(pageTableK)
        .set_page_table_v(pageTableV)
        .set_block_mask(blockMask)
        .set_sink_token(sinkToken)
        .set_descale_q(descaleQ)
        .set_descale_k(descaleK)
        .set_descale_v(descaleV)
        .set_descale_s(descaleS)
        .set_scale_s(scaleS)
        .set_scale_o(scaleO);

    EXPECT_EQ(attrs.get_bias(), attnMask);
    EXPECT_EQ(attrs.get_attn_scale(), scale);
    EXPECT_EQ(attrs.get_seq_len_q(), seqLenQ);
    EXPECT_EQ(attrs.get_seq_len_kv(), seqLenKv);
    EXPECT_EQ(attrs.get_seed(), seed);
    EXPECT_EQ(attrs.get_offset(), offset);
    EXPECT_EQ(attrs.get_dropout_mask(), dropoutMask);
    EXPECT_EQ(attrs.get_dropout_scale(), dropoutScale);
    EXPECT_EQ(attrs.get_page_table_k(), pageTableK);
    EXPECT_EQ(attrs.get_page_table_v(), pageTableV);
    EXPECT_EQ(attrs.get_block_mask(), blockMask);
    EXPECT_EQ(attrs.get_sink_token(), sinkToken);
    EXPECT_EQ(attrs.get_descale_q(), descaleQ);
    EXPECT_EQ(attrs.get_descale_k(), descaleK);
    EXPECT_EQ(attrs.get_descale_v(), descaleV);
    EXPECT_EQ(attrs.get_descale_s(), descaleS);
    EXPECT_EQ(attrs.get_scale_s(), scaleS);
    EXPECT_EQ(attrs.get_scale_o(), scaleO);
}

TEST(TestSdpaAttributes, SetOutputTensors)
{
    SdpaAttributes attrs;

    auto stats = makeTensor(100);
    auto max = makeTensor(101);
    auto sumExp = makeTensor(102);
    auto rngDump = makeTensor(103);
    auto amaxS = makeTensor(104);
    auto amaxO = makeTensor(105);

    attrs.set_stats(stats)
        .set_max(max)
        .set_sum_exp(sumExp)
        .set_rng_dump(rngDump)
        .set_amax_s(amaxS)
        .set_amax_o(amaxO);

    EXPECT_EQ(attrs.get_stats(), stats);
    EXPECT_EQ(attrs.get_max(), max);
    EXPECT_EQ(attrs.get_sum_exp(), sumExp);
    EXPECT_EQ(attrs.get_rng_dump(), rngDump);
    EXPECT_EQ(attrs.get_amax_s(), amaxS);
    EXPECT_EQ(attrs.get_amax_o(), amaxO);

    // Unrelated tensors remain null
    EXPECT_EQ(attrs.get_o(), nullptr);
}

TEST(TestSdpaAttributes, SetDropout)
{
    SdpaAttributes attrs;

    auto seed = makeTensor(50);
    auto offset = makeTensor(51);
    attrs.set_dropout(0.1f, seed, offset);

    ASSERT_TRUE(attrs.dropout_probability.has_value());
    EXPECT_FLOAT_EQ(*attrs.dropout_probability, 0.1f);
    EXPECT_EQ(attrs.get_seed(), seed);
    EXPECT_EQ(attrs.get_offset(), offset);
}

TEST(TestSdpaAttributes, SetBooleanFlags)
{
    SdpaAttributes attrs;

    attrs.set_generate_stats(true);
    ASSERT_TRUE(attrs.generate_stats.has_value());
    EXPECT_TRUE(*attrs.generate_stats);

    attrs.set_generate_stats(false);
    ASSERT_TRUE(attrs.generate_stats.has_value());
    EXPECT_FALSE(*attrs.generate_stats);

    attrs.set_alibi_mask(true);
    EXPECT_TRUE(attrs.alibi_mask);

    attrs.set_padding_mask(true);
    EXPECT_TRUE(attrs.padding_mask);

    attrs.set_causal_mask(true);
    EXPECT_TRUE(attrs.causal_mask);

    attrs.set_causal_mask_bottom_right(true);
    EXPECT_TRUE(attrs.causal_mask_bottom_right);
}

TEST(TestSdpaAttributes, SetScalarAttributes)
{
    SdpaAttributes attrs;

    attrs.set_attn_scale_value(0.5f);
    ASSERT_TRUE(attrs.attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(*attrs.attn_scale_value, 0.5f);

    attrs.set_diagonal_band_left_bound(-3);
    ASSERT_TRUE(attrs.left_bound.has_value());
    EXPECT_EQ(*attrs.left_bound, -3);

    attrs.set_diagonal_band_right_bound(7);
    ASSERT_TRUE(attrs.right_bound.has_value());
    EXPECT_EQ(*attrs.right_bound, 7);

    attrs.set_paged_attention_max_seq_len_kv(512);
    ASSERT_TRUE(attrs.max_seq_len_kv.has_value());
    EXPECT_EQ(*attrs.max_seq_len_kv, 512);
}

TEST(TestSdpaAttributes, SetEnumAttributes)
{
    SdpaAttributes attrs;

    attrs.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);

    attrs.set_mma_core_mode(DataType::HALF);
    EXPECT_EQ(attrs.mma_core_mode, DataType::HALF);

    attrs.set_implementation(AttentionImplementation::COMPOSITE);
    EXPECT_EQ(attrs.implementation, AttentionImplementation::COMPOSITE);

    attrs.set_implementation(AttentionImplementation::UNIFIED);
    EXPECT_EQ(attrs.implementation, AttentionImplementation::UNIFIED);
}
