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
    SdpaBackwardAttributes attrs;

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

TEST(TestSdpaBackwardAttributes, PackAttributesAllFields)
{
    SdpaBackwardAttributes attrs;

    // Required tensors
    attrs.set_q(makeTensor(1))
        .set_k(makeTensor(2))
        .set_v(makeTensor(3))
        .set_o(makeTensor(4))
        .set_do(makeTensor(5))
        .set_stats(makeTensor(6))
        .set_dq(makeTensor(7))
        .set_dk(makeTensor(8))
        .set_dv(makeTensor(9));

    // Optional input tensors
    attrs.set_attn_scale(makeTensor(10))
        .set_bias(makeTensor(11))
        .set_seq_len_q(makeTensor(12))
        .set_seq_len_kv(makeTensor(13))
        .set_seed(makeTensor(14))
        .set_offset(makeTensor(15))
        .set_dropout_mask(makeTensor(16))
        .set_dropout_scale(makeTensor(17))
        .set_dropout_scale_inv(makeTensor(18))
        .set_dbias(makeTensor(19));

    // Boolean flags
    attrs.set_alibi_mask(true)
        .set_padding_mask(true)
        .set_causal_mask(true)
        .set_causal_mask_bottom_right(true);

    // Scalar attributes
    attrs.dropout_probability = 0.3f;
    attrs.set_attn_scale_value(0.125f)
        .set_diagonal_band_left_bound(-2)
        .set_diagonal_band_right_bound(2);

    // Enum
    attrs.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);

    flatbuffers::FlatBufferBuilder builder;
    auto packed = attrs.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::SdpaBackwardAttributes>(buf);

    // Required tensors
    EXPECT_EQ(fb->q_tensor_uid(), 1);
    EXPECT_EQ(fb->k_tensor_uid(), 2);
    EXPECT_EQ(fb->v_tensor_uid(), 3);
    EXPECT_EQ(fb->o_tensor_uid(), 4);
    EXPECT_EQ(fb->do_tensor_uid(), 5);
    EXPECT_EQ(fb->stats_tensor_uid(), 6);
    EXPECT_EQ(fb->dq_tensor_uid(), 7);
    EXPECT_EQ(fb->dk_tensor_uid(), 8);
    EXPECT_EQ(fb->dv_tensor_uid(), 9);

    // Optional input tensors
    ASSERT_TRUE(fb->scale_tensor_uid().has_value());
    EXPECT_EQ(*fb->scale_tensor_uid(), 10);
    ASSERT_TRUE(fb->attn_mask_tensor_uid().has_value());
    EXPECT_EQ(*fb->attn_mask_tensor_uid(), 11);
    ASSERT_TRUE(fb->seq_len_q_tensor_uid().has_value());
    EXPECT_EQ(*fb->seq_len_q_tensor_uid(), 12);
    ASSERT_TRUE(fb->seq_len_kv_tensor_uid().has_value());
    EXPECT_EQ(*fb->seq_len_kv_tensor_uid(), 13);
    ASSERT_TRUE(fb->seed_tensor_uid().has_value());
    EXPECT_EQ(*fb->seed_tensor_uid(), 14);
    ASSERT_TRUE(fb->offset_tensor_uid().has_value());
    EXPECT_EQ(*fb->offset_tensor_uid(), 15);
    ASSERT_TRUE(fb->dropout_mask_tensor_uid().has_value());
    EXPECT_EQ(*fb->dropout_mask_tensor_uid(), 16);
    ASSERT_TRUE(fb->dropout_scale_tensor_uid().has_value());
    EXPECT_EQ(*fb->dropout_scale_tensor_uid(), 17);
    ASSERT_TRUE(fb->dropout_scale_inv_tensor_uid().has_value());
    EXPECT_EQ(*fb->dropout_scale_inv_tensor_uid(), 18);
    ASSERT_TRUE(fb->dbias_tensor_uid().has_value());
    EXPECT_EQ(*fb->dbias_tensor_uid(), 19);

    // Boolean flags
    EXPECT_TRUE(fb->alibi_mask());
    EXPECT_TRUE(fb->padding_mask());
    EXPECT_TRUE(fb->causal_mask());
    EXPECT_TRUE(fb->causal_mask_bottom_right());

    // Scalar attributes
    ASSERT_TRUE(fb->dropout_probability().has_value());
    EXPECT_FLOAT_EQ(*fb->dropout_probability(), 0.3f);
    ASSERT_TRUE(fb->attn_scale_value().has_value());
    EXPECT_FLOAT_EQ(*fb->attn_scale_value(), 0.125f);
    ASSERT_TRUE(fb->left_bound().has_value());
    EXPECT_EQ(*fb->left_bound(), -2);
    ASSERT_TRUE(fb->right_bound().has_value());
    EXPECT_EQ(*fb->right_bound(), 2);

    // Enum
    EXPECT_EQ(fb->diagonal_alignment(),
              hipdnn_data_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT);
}

TEST(TestSdpaBackwardAttributes, PackAttributesNoOptionals)
{
    SdpaBackwardAttributes attrs;
    attrs.set_q(makeTensor(1))
        .set_k(makeTensor(2))
        .set_v(makeTensor(3))
        .set_o(makeTensor(4))
        .set_do(makeTensor(5))
        .set_stats(makeTensor(6))
        .set_dq(makeTensor(7))
        .set_dk(makeTensor(8))
        .set_dv(makeTensor(9));

    flatbuffers::FlatBufferBuilder builder;
    auto packed = attrs.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::SdpaBackwardAttributes>(buf);

    // Required tensors
    EXPECT_EQ(fb->q_tensor_uid(), 1);
    EXPECT_EQ(fb->dv_tensor_uid(), 9);

    // All optional tensor UIDs absent
    EXPECT_FALSE(fb->scale_tensor_uid().has_value());
    EXPECT_FALSE(fb->attn_mask_tensor_uid().has_value());
    EXPECT_FALSE(fb->seq_len_q_tensor_uid().has_value());
    EXPECT_FALSE(fb->seq_len_kv_tensor_uid().has_value());
    EXPECT_FALSE(fb->seed_tensor_uid().has_value());
    EXPECT_FALSE(fb->offset_tensor_uid().has_value());
    EXPECT_FALSE(fb->dropout_mask_tensor_uid().has_value());
    EXPECT_FALSE(fb->dropout_scale_tensor_uid().has_value());
    EXPECT_FALSE(fb->dropout_scale_inv_tensor_uid().has_value());
    EXPECT_FALSE(fb->dbias_tensor_uid().has_value());

    // Boolean flags at defaults
    EXPECT_FALSE(fb->alibi_mask());
    EXPECT_FALSE(fb->padding_mask());
    EXPECT_FALSE(fb->causal_mask());
    EXPECT_FALSE(fb->causal_mask_bottom_right());

    // Scalar attributes absent
    EXPECT_FALSE(fb->dropout_probability().has_value());
    EXPECT_FALSE(fb->attn_scale_value().has_value());
    EXPECT_FALSE(fb->left_bound().has_value());
    EXPECT_FALSE(fb->right_bound().has_value());

    // Enum at default
    EXPECT_EQ(fb->diagonal_alignment(), hipdnn_data_sdk::data_objects::DiagonalAlignment::TOP_LEFT);
}

TEST(TestSdpaBackwardAttributes, FromFlatBufferRoundtrip)
{
    SdpaBackwardAttributes original;
    original.set_q(makeTensor(1))
        .set_k(makeTensor(2))
        .set_v(makeTensor(3))
        .set_o(makeTensor(4))
        .set_do(makeTensor(5))
        .set_stats(makeTensor(6))
        .set_dq(makeTensor(7))
        .set_dk(makeTensor(8))
        .set_dv(makeTensor(9));
    original.set_attn_scale(makeTensor(10))
        .set_bias(makeTensor(11))
        .set_seq_len_q(makeTensor(12))
        .set_seq_len_kv(makeTensor(13))
        .set_seed(makeTensor(14))
        .set_offset(makeTensor(15))
        .set_dropout_mask(makeTensor(16))
        .set_dropout_scale(makeTensor(17))
        .set_dropout_scale_inv(makeTensor(18))
        .set_dbias(makeTensor(19));
    original.set_alibi_mask(true)
        .set_padding_mask(true)
        .set_causal_mask(true)
        .set_causal_mask_bottom_right(true);
    original.dropout_probability = 0.2f;
    original.set_attn_scale_value(0.25f)
        .set_diagonal_band_left_bound(-1)
        .set_diagonal_band_right_bound(3)
        .set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);

    flatbuffers::FlatBufferBuilder builder;
    auto packed = original.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::SdpaBackwardAttributes>(buf);

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    for(int64_t uid = 1; uid <= 19; ++uid)
    {
        tensorMap[uid] = makeTensor(uid);
    }

    auto attrs = SdpaBackwardAttributes::fromFlatBuffer(fb, tensorMap);

    // Required tensors
    ASSERT_NE(attrs.get_q(), nullptr);
    EXPECT_EQ(attrs.get_q()->get_uid(), 1);
    ASSERT_NE(attrs.get_k(), nullptr);
    EXPECT_EQ(attrs.get_k()->get_uid(), 2);
    ASSERT_NE(attrs.get_v(), nullptr);
    EXPECT_EQ(attrs.get_v()->get_uid(), 3);
    ASSERT_NE(attrs.get_o(), nullptr);
    EXPECT_EQ(attrs.get_o()->get_uid(), 4);
    ASSERT_NE(attrs.get_do(), nullptr);
    EXPECT_EQ(attrs.get_do()->get_uid(), 5);
    ASSERT_NE(attrs.get_stats(), nullptr);
    EXPECT_EQ(attrs.get_stats()->get_uid(), 6);
    ASSERT_NE(attrs.get_dq(), nullptr);
    EXPECT_EQ(attrs.get_dq()->get_uid(), 7);
    ASSERT_NE(attrs.get_dk(), nullptr);
    EXPECT_EQ(attrs.get_dk()->get_uid(), 8);
    ASSERT_NE(attrs.get_dv(), nullptr);
    EXPECT_EQ(attrs.get_dv()->get_uid(), 9);

    // Optional input tensors
    ASSERT_NE(attrs.get_attn_scale(), nullptr);
    EXPECT_EQ(attrs.get_attn_scale()->get_uid(), 10);
    ASSERT_NE(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_bias()->get_uid(), 11);
    ASSERT_NE(attrs.get_seq_len_q(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_q()->get_uid(), 12);
    ASSERT_NE(attrs.get_seq_len_kv(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_kv()->get_uid(), 13);
    ASSERT_NE(attrs.get_seed(), nullptr);
    EXPECT_EQ(attrs.get_seed()->get_uid(), 14);
    ASSERT_NE(attrs.get_offset(), nullptr);
    EXPECT_EQ(attrs.get_offset()->get_uid(), 15);
    ASSERT_NE(attrs.get_dropout_mask(), nullptr);
    EXPECT_EQ(attrs.get_dropout_mask()->get_uid(), 16);
    ASSERT_NE(attrs.get_dropout_scale(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale()->get_uid(), 17);
    ASSERT_NE(attrs.get_dropout_scale_inv(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale_inv()->get_uid(), 18);
    ASSERT_NE(attrs.get_dbias(), nullptr);
    EXPECT_EQ(attrs.get_dbias()->get_uid(), 19);

    // Boolean flags
    EXPECT_TRUE(attrs.alibi_mask);
    EXPECT_TRUE(attrs.padding_mask);
    EXPECT_TRUE(attrs.causal_mask);
    EXPECT_TRUE(attrs.causal_mask_bottom_right);

    // Scalar attributes
    ASSERT_TRUE(attrs.dropout_probability.has_value());
    EXPECT_FLOAT_EQ(*attrs.dropout_probability, 0.2f);
    ASSERT_TRUE(attrs.attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(*attrs.attn_scale_value, 0.25f);
    ASSERT_TRUE(attrs.left_bound.has_value());
    EXPECT_EQ(*attrs.left_bound, -1);
    ASSERT_TRUE(attrs.right_bound.has_value());
    EXPECT_EQ(*attrs.right_bound, 3);

    // Enum
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
}

TEST(TestSdpaBackwardAttributes, FromFlatBufferNoOptionals)
{
    SdpaBackwardAttributes original;
    original.set_q(makeTensor(1))
        .set_k(makeTensor(2))
        .set_v(makeTensor(3))
        .set_o(makeTensor(4))
        .set_do(makeTensor(5))
        .set_stats(makeTensor(6))
        .set_dq(makeTensor(7))
        .set_dk(makeTensor(8))
        .set_dv(makeTensor(9));

    flatbuffers::FlatBufferBuilder builder;
    auto packed = original.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::SdpaBackwardAttributes>(buf);

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    for(int64_t uid = 1; uid <= 9; ++uid)
    {
        tensorMap[uid] = makeTensor(uid);
    }

    auto attrs = SdpaBackwardAttributes::fromFlatBuffer(fb, tensorMap);

    // Required tensors present
    ASSERT_NE(attrs.get_q(), nullptr);
    EXPECT_EQ(attrs.get_q()->get_uid(), 1);
    ASSERT_NE(attrs.get_dv(), nullptr);
    EXPECT_EQ(attrs.get_dv()->get_uid(), 9);

    // All optional tensors absent
    EXPECT_EQ(attrs.get_attn_scale(), nullptr);
    EXPECT_EQ(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_q(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_kv(), nullptr);
    EXPECT_EQ(attrs.get_seed(), nullptr);
    EXPECT_EQ(attrs.get_offset(), nullptr);
    EXPECT_EQ(attrs.get_dropout_mask(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale_inv(), nullptr);
    EXPECT_EQ(attrs.get_dbias(), nullptr);

    // Boolean flags at defaults
    EXPECT_FALSE(attrs.alibi_mask);
    EXPECT_FALSE(attrs.padding_mask);
    EXPECT_FALSE(attrs.causal_mask);
    EXPECT_FALSE(attrs.causal_mask_bottom_right);

    // Scalar attributes absent
    EXPECT_FALSE(attrs.dropout_probability.has_value());
    EXPECT_FALSE(attrs.attn_scale_value.has_value());
    EXPECT_FALSE(attrs.left_bound.has_value());
    EXPECT_FALSE(attrs.right_bound.has_value());

    // Enum at default
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::TOP_LEFT);
}
