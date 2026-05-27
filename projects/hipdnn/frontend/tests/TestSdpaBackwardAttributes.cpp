// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>

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

TEST(TestSdpaBackwardAttributes, DefaultValues)
{
    const SdpaBackwardAttributes attrs;

    // Required I/O tensors should be null
    EXPECT_EQ(attrs.get_q(), nullptr);
    EXPECT_EQ(attrs.get_k(), nullptr);
    EXPECT_EQ(attrs.get_v(), nullptr);
    EXPECT_EQ(attrs.get_o(), nullptr);
    EXPECT_EQ(attrs.get_do(), nullptr);
    EXPECT_EQ(attrs.get_stats(), nullptr);
    EXPECT_EQ(attrs.get_dq(), nullptr);
    EXPECT_EQ(attrs.get_dk(), nullptr);
    EXPECT_EQ(attrs.get_dv(), nullptr);

    // Optional input tensors
    EXPECT_EQ(attrs.get_attn_scale(), nullptr);
    EXPECT_EQ(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_q(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_kv(), nullptr);
    EXPECT_EQ(attrs.get_seed(), nullptr);
    EXPECT_EQ(attrs.get_offset(), nullptr);
    EXPECT_EQ(attrs.get_dropout_mask(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale_inv(), nullptr);

    // Optional output tensors
    EXPECT_EQ(attrs.get_dbias(), nullptr);

    // Boolean flags
    EXPECT_FALSE(attrs.alibi_mask);
    EXPECT_FALSE(attrs.padding_mask);
    EXPECT_FALSE(attrs.causal_mask);
    EXPECT_FALSE(attrs.causal_mask_bottom_right);

    // Scalar attributes
    EXPECT_FALSE(attrs.dropout_probability.has_value());
    EXPECT_FALSE(attrs.attn_scale_value.has_value());
    EXPECT_FALSE(attrs.left_bound.has_value());
    EXPECT_FALSE(attrs.right_bound.has_value());

    // Enum defaults
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::TOP_LEFT);
}

TEST(TestSdpaBackwardAttributes, SetRequiredTensors)
{
    SdpaBackwardAttributes attrs;

    auto q = makeTensor(1);
    auto k = makeTensor(2);
    auto v = makeTensor(3);
    auto o = makeTensor(4);
    auto dOut = makeTensor(5);
    auto stats = makeTensor(6);
    auto dq = makeTensor(7);
    auto dk = makeTensor(8);
    auto dv = makeTensor(9);

    attrs.set_q(q)
        .set_k(k)
        .set_v(v)
        .set_o(o)
        .set_do(dOut)
        .set_stats(stats)
        .set_dq(dq)
        .set_dk(dk)
        .set_dv(dv);

    EXPECT_EQ(attrs.get_q(), q);
    EXPECT_EQ(attrs.get_k(), k);
    EXPECT_EQ(attrs.get_v(), v);
    EXPECT_EQ(attrs.get_o(), o);
    EXPECT_EQ(attrs.get_do(), dOut);
    EXPECT_EQ(attrs.get_stats(), stats);
    EXPECT_EQ(attrs.get_dq(), dq);
    EXPECT_EQ(attrs.get_dk(), dk);
    EXPECT_EQ(attrs.get_dv(), dv);

    // Unset optional tensors remain null
    EXPECT_EQ(attrs.get_attn_scale(), nullptr);
    EXPECT_EQ(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_dbias(), nullptr);
}

TEST(TestSdpaBackwardAttributes, SetOptionalInputTensors)
{
    SdpaBackwardAttributes attrs;

    auto scale = makeTensor(10);
    auto attnMask = makeTensor(11);
    auto seqLenQ = makeTensor(12);
    auto seqLenKv = makeTensor(13);
    auto seed = makeTensor(14);
    auto offset = makeTensor(15);
    auto dropoutMask = makeTensor(16);
    auto dropoutScale = makeTensor(17);
    auto dropoutScaleInv = makeTensor(18);
    auto dbias = makeTensor(19);

    attrs.set_attn_scale(scale)
        .set_bias(attnMask)
        .set_seq_len_q(seqLenQ)
        .set_seq_len_kv(seqLenKv)
        .set_seed(seed)
        .set_offset(offset)
        .set_dropout_mask(dropoutMask)
        .set_dropout_scale(dropoutScale)
        .set_dropout_scale_inv(dropoutScaleInv)
        .set_dbias(dbias);

    EXPECT_EQ(attrs.get_attn_scale(), scale);
    EXPECT_EQ(attrs.get_bias(), attnMask);
    EXPECT_EQ(attrs.get_seq_len_q(), seqLenQ);
    EXPECT_EQ(attrs.get_seq_len_kv(), seqLenKv);
    EXPECT_EQ(attrs.get_seed(), seed);
    EXPECT_EQ(attrs.get_offset(), offset);
    EXPECT_EQ(attrs.get_dropout_mask(), dropoutMask);
    EXPECT_EQ(attrs.get_dropout_scale(), dropoutScale);
    EXPECT_EQ(attrs.get_dropout_scale_inv(), dropoutScaleInv);
    EXPECT_EQ(attrs.get_dbias(), dbias);
}

TEST(TestSdpaBackwardAttributes, SetBooleanFlags)
{
    SdpaBackwardAttributes attrs;

    attrs.set_alibi_mask(true);
    EXPECT_TRUE(attrs.alibi_mask);

    attrs.set_padding_mask(true);
    EXPECT_TRUE(attrs.padding_mask);

    attrs.set_causal_mask(true);
    EXPECT_TRUE(attrs.causal_mask);

    attrs.set_causal_mask_bottom_right(true);
    EXPECT_TRUE(attrs.causal_mask_bottom_right);

    // Reset to false
    attrs.set_alibi_mask(false);
    EXPECT_FALSE(attrs.alibi_mask);
}

TEST(TestSdpaBackwardAttributes, SetDropout)
{
    SdpaBackwardAttributes attrs;

    auto seed = makeTensor(50);
    auto offset = makeTensor(51);
    attrs.set_dropout(0.1f, seed, offset);

    ASSERT_TRUE(attrs.dropout_probability.has_value());
    EXPECT_FLOAT_EQ(*attrs.dropout_probability, 0.1f);
    EXPECT_EQ(attrs.get_seed(), seed);
    EXPECT_EQ(attrs.get_offset(), offset);
}

TEST(TestSdpaBackwardAttributes, SetScalarAttributes)
{
    SdpaBackwardAttributes attrs;

    attrs.set_attn_scale_value(0.5f);
    ASSERT_TRUE(attrs.attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(*attrs.attn_scale_value, 0.5f);

    attrs.set_diagonal_band_left_bound(-3);
    ASSERT_TRUE(attrs.left_bound.has_value());
    EXPECT_EQ(*attrs.left_bound, -3);

    attrs.set_diagonal_band_right_bound(7);
    ASSERT_TRUE(attrs.right_bound.has_value());
    EXPECT_EQ(*attrs.right_bound, 7);
}

TEST(TestSdpaBackwardAttributes, SetEnumAttributes)
{
    SdpaBackwardAttributes attrs;

    attrs.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);

    attrs.set_diagonal_alignment(DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::TOP_LEFT);
}
