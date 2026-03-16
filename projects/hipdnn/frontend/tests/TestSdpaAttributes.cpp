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
    SdpaAttributes attrs;

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

TEST(TestSdpaAttributes, PackAttributes)
{
    SdpaAttributes attrs;
    // Required tensors
    attrs.set_q(makeTensor(1)).set_k(makeTensor(2)).set_v(makeTensor(3)).set_o(makeTensor(4));
    // Optional input tensors
    attrs.set_bias(makeTensor(5))
        .set_attn_scale(makeTensor(6))
        .set_seq_len_q(makeTensor(7))
        .set_seq_len_kv(makeTensor(8))
        .set_seed(makeTensor(9))
        .set_offset(makeTensor(10))
        .set_dropout_mask(makeTensor(11))
        .set_dropout_scale(makeTensor(12))
        .set_page_table_k(makeTensor(13))
        .set_page_table_v(makeTensor(14))
        .set_block_mask(makeTensor(15))
        .set_sink_token(makeTensor(16))
        .set_descale_q(makeTensor(17))
        .set_descale_k(makeTensor(18))
        .set_descale_v(makeTensor(19))
        .set_descale_s(makeTensor(20))
        .set_scale_s(makeTensor(21))
        .set_scale_o(makeTensor(22));
    // Optional output tensors
    attrs.set_stats(makeTensor(23))
        .set_max(makeTensor(24))
        .set_sum_exp(makeTensor(25))
        .set_rng_dump(makeTensor(26))
        .set_amax_s(makeTensor(27))
        .set_amax_o(makeTensor(28));
    // Boolean flags
    attrs.set_generate_stats(true)
        .set_alibi_mask(true)
        .set_padding_mask(true)
        .set_causal_mask(true)
        .set_causal_mask_bottom_right(true);
    // Scalar attributes (dropout_probability has no dedicated setter)
    attrs.dropout_probability = 0.3f;
    attrs.set_attn_scale_value(0.125f)
        .set_diagonal_band_left_bound(-2)
        .set_diagonal_band_right_bound(2)
        .set_paged_attention_max_seq_len_kv(128);
    // Enum attributes
    attrs.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT)
        .set_mma_core_mode(DataType::FLOAT)
        .set_implementation(AttentionImplementation::UNIFIED);

    flatbuffers::FlatBufferBuilder builder;
    auto packed = attrs.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::SdpaAttributes>(buf);

    // Required tensors
    EXPECT_EQ(fb->q_tensor_uid(), 1);
    EXPECT_EQ(fb->k_tensor_uid(), 2);
    EXPECT_EQ(fb->v_tensor_uid(), 3);
    EXPECT_EQ(fb->o_tensor_uid(), 4);

    // Optional input tensors
    ASSERT_TRUE(fb->attn_mask_tensor_uid().has_value());
    EXPECT_EQ(*fb->attn_mask_tensor_uid(), 5);
    ASSERT_TRUE(fb->scale_tensor_uid().has_value());
    EXPECT_EQ(*fb->scale_tensor_uid(), 6);
    ASSERT_TRUE(fb->seq_len_q_tensor_uid().has_value());
    EXPECT_EQ(*fb->seq_len_q_tensor_uid(), 7);
    ASSERT_TRUE(fb->seq_len_kv_tensor_uid().has_value());
    EXPECT_EQ(*fb->seq_len_kv_tensor_uid(), 8);
    ASSERT_TRUE(fb->seed_tensor_uid().has_value());
    EXPECT_EQ(*fb->seed_tensor_uid(), 9);
    ASSERT_TRUE(fb->offset_tensor_uid().has_value());
    EXPECT_EQ(*fb->offset_tensor_uid(), 10);
    ASSERT_TRUE(fb->dropout_mask_tensor_uid().has_value());
    EXPECT_EQ(*fb->dropout_mask_tensor_uid(), 11);
    ASSERT_TRUE(fb->dropout_scale_tensor_uid().has_value());
    EXPECT_EQ(*fb->dropout_scale_tensor_uid(), 12);
    ASSERT_TRUE(fb->page_table_k_tensor_uid().has_value());
    EXPECT_EQ(*fb->page_table_k_tensor_uid(), 13);
    ASSERT_TRUE(fb->page_table_v_tensor_uid().has_value());
    EXPECT_EQ(*fb->page_table_v_tensor_uid(), 14);
    ASSERT_TRUE(fb->block_mask_tensor_uid().has_value());
    EXPECT_EQ(*fb->block_mask_tensor_uid(), 15);
    ASSERT_TRUE(fb->sink_token_tensor_uid().has_value());
    EXPECT_EQ(*fb->sink_token_tensor_uid(), 16);
    ASSERT_TRUE(fb->descale_q_tensor_uid().has_value());
    EXPECT_EQ(*fb->descale_q_tensor_uid(), 17);
    ASSERT_TRUE(fb->descale_k_tensor_uid().has_value());
    EXPECT_EQ(*fb->descale_k_tensor_uid(), 18);
    ASSERT_TRUE(fb->descale_v_tensor_uid().has_value());
    EXPECT_EQ(*fb->descale_v_tensor_uid(), 19);
    ASSERT_TRUE(fb->descale_s_tensor_uid().has_value());
    EXPECT_EQ(*fb->descale_s_tensor_uid(), 20);
    ASSERT_TRUE(fb->scale_s_tensor_uid().has_value());
    EXPECT_EQ(*fb->scale_s_tensor_uid(), 21);
    ASSERT_TRUE(fb->scale_o_tensor_uid().has_value());
    EXPECT_EQ(*fb->scale_o_tensor_uid(), 22);

    // Optional output tensors
    ASSERT_TRUE(fb->stats_tensor_uid().has_value());
    EXPECT_EQ(*fb->stats_tensor_uid(), 23);
    ASSERT_TRUE(fb->max_tensor_uid().has_value());
    EXPECT_EQ(*fb->max_tensor_uid(), 24);
    ASSERT_TRUE(fb->sum_exp_tensor_uid().has_value());
    EXPECT_EQ(*fb->sum_exp_tensor_uid(), 25);
    ASSERT_TRUE(fb->rng_dump_tensor_uid().has_value());
    EXPECT_EQ(*fb->rng_dump_tensor_uid(), 26);
    ASSERT_TRUE(fb->amax_s_tensor_uid().has_value());
    EXPECT_EQ(*fb->amax_s_tensor_uid(), 27);
    ASSERT_TRUE(fb->amax_o_tensor_uid().has_value());
    EXPECT_EQ(*fb->amax_o_tensor_uid(), 28);

    // Boolean flags
    ASSERT_TRUE(fb->generate_stats().has_value());
    EXPECT_TRUE(*fb->generate_stats());
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
    ASSERT_TRUE(fb->max_seq_len_kv().has_value());
    EXPECT_EQ(*fb->max_seq_len_kv(), 128);

    // Enum attributes
    EXPECT_EQ(fb->diagonal_alignment(),
              hipdnn_data_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT);
    EXPECT_EQ(fb->mma_core_mode(), hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(fb->implementation(),
              hipdnn_data_sdk::data_objects::AttentionImplementation::UNIFIED);
}

TEST(TestSdpaAttributes, PackAttributesNoOptionals)
{
    SdpaAttributes attrs;
    attrs.set_q(makeTensor(1)).set_k(makeTensor(2)).set_v(makeTensor(3)).set_o(makeTensor(4));

    flatbuffers::FlatBufferBuilder builder;
    auto packed = attrs.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::SdpaAttributes>(buf);

    EXPECT_EQ(fb->q_tensor_uid(), 1);
    EXPECT_EQ(fb->k_tensor_uid(), 2);
    EXPECT_EQ(fb->v_tensor_uid(), 3);
    EXPECT_EQ(fb->o_tensor_uid(), 4);

    // All optional tensor UIDs absent
    EXPECT_FALSE(fb->attn_mask_tensor_uid().has_value());
    EXPECT_FALSE(fb->scale_tensor_uid().has_value());
    EXPECT_FALSE(fb->seq_len_q_tensor_uid().has_value());
    EXPECT_FALSE(fb->seq_len_kv_tensor_uid().has_value());
    EXPECT_FALSE(fb->seed_tensor_uid().has_value());
    EXPECT_FALSE(fb->offset_tensor_uid().has_value());
    EXPECT_FALSE(fb->dropout_mask_tensor_uid().has_value());
    EXPECT_FALSE(fb->dropout_scale_tensor_uid().has_value());
    EXPECT_FALSE(fb->page_table_k_tensor_uid().has_value());
    EXPECT_FALSE(fb->page_table_v_tensor_uid().has_value());
    EXPECT_FALSE(fb->block_mask_tensor_uid().has_value());
    EXPECT_FALSE(fb->sink_token_tensor_uid().has_value());
    EXPECT_FALSE(fb->descale_q_tensor_uid().has_value());
    EXPECT_FALSE(fb->descale_k_tensor_uid().has_value());
    EXPECT_FALSE(fb->descale_v_tensor_uid().has_value());
    EXPECT_FALSE(fb->descale_s_tensor_uid().has_value());
    EXPECT_FALSE(fb->scale_s_tensor_uid().has_value());
    EXPECT_FALSE(fb->scale_o_tensor_uid().has_value());
    EXPECT_FALSE(fb->stats_tensor_uid().has_value());
    EXPECT_FALSE(fb->max_tensor_uid().has_value());
    EXPECT_FALSE(fb->sum_exp_tensor_uid().has_value());
    EXPECT_FALSE(fb->rng_dump_tensor_uid().has_value());
    EXPECT_FALSE(fb->amax_s_tensor_uid().has_value());
    EXPECT_FALSE(fb->amax_o_tensor_uid().has_value());

    // Boolean flags at defaults
    EXPECT_FALSE(fb->generate_stats().has_value());
    EXPECT_FALSE(fb->alibi_mask());
    EXPECT_FALSE(fb->padding_mask());
    EXPECT_FALSE(fb->causal_mask());
    EXPECT_FALSE(fb->causal_mask_bottom_right());

    // Scalar attributes absent
    EXPECT_FALSE(fb->dropout_probability().has_value());
    EXPECT_FALSE(fb->attn_scale_value().has_value());
    EXPECT_FALSE(fb->left_bound().has_value());
    EXPECT_FALSE(fb->right_bound().has_value());
    EXPECT_FALSE(fb->max_seq_len_kv().has_value());

    // Enum attributes at defaults
    EXPECT_EQ(fb->diagonal_alignment(), hipdnn_data_sdk::data_objects::DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(fb->mma_core_mode(), hipdnn_data_sdk::data_objects::DataType::UNSET);
    EXPECT_EQ(fb->implementation(), hipdnn_data_sdk::data_objects::AttentionImplementation::AUTO);
}

TEST(TestSdpaAttributes, FromFlatBufferRoundtrip)
{
    // Build attrs with all fields set, then verify fromFlatBuffer reconstructs them correctly
    SdpaAttributes original;
    original.set_q(makeTensor(1)).set_k(makeTensor(2)).set_v(makeTensor(3)).set_o(makeTensor(4));
    original.set_bias(makeTensor(5))
        .set_attn_scale(makeTensor(6))
        .set_seq_len_q(makeTensor(7))
        .set_seq_len_kv(makeTensor(8))
        .set_seed(makeTensor(9))
        .set_offset(makeTensor(10))
        .set_dropout_mask(makeTensor(11))
        .set_dropout_scale(makeTensor(12))
        .set_page_table_k(makeTensor(13))
        .set_page_table_v(makeTensor(14))
        .set_block_mask(makeTensor(15))
        .set_sink_token(makeTensor(16))
        .set_descale_q(makeTensor(17))
        .set_descale_k(makeTensor(18))
        .set_descale_v(makeTensor(19))
        .set_descale_s(makeTensor(20))
        .set_scale_s(makeTensor(21))
        .set_scale_o(makeTensor(22));
    original.set_stats(makeTensor(23))
        .set_max(makeTensor(24))
        .set_sum_exp(makeTensor(25))
        .set_rng_dump(makeTensor(26))
        .set_amax_s(makeTensor(27))
        .set_amax_o(makeTensor(28));
    original.set_generate_stats(true)
        .set_alibi_mask(true)
        .set_padding_mask(true)
        .set_causal_mask(true)
        .set_causal_mask_bottom_right(true);
    original.dropout_probability = 0.2f;
    original.set_attn_scale_value(0.25f)
        .set_diagonal_band_left_bound(-1)
        .set_diagonal_band_right_bound(3)
        .set_paged_attention_max_seq_len_kv(64);
    original.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT)
        .set_mma_core_mode(DataType::HALF)
        .set_implementation(AttentionImplementation::COMPOSITE);

    flatbuffers::FlatBufferBuilder builder;
    auto packed = original.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::SdpaAttributes>(buf);

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    for(int64_t uid = 1; uid <= 28; ++uid)
    {
        tensorMap[uid] = makeTensor(uid);
    }

    auto attrs = SdpaAttributes::fromFlatBuffer(fb, tensorMap);

    // Required tensors
    ASSERT_NE(attrs.get_q(), nullptr);
    EXPECT_EQ(attrs.get_q()->get_uid(), 1);
    ASSERT_NE(attrs.get_k(), nullptr);
    EXPECT_EQ(attrs.get_k()->get_uid(), 2);
    ASSERT_NE(attrs.get_v(), nullptr);
    EXPECT_EQ(attrs.get_v()->get_uid(), 3);
    ASSERT_NE(attrs.get_o(), nullptr);
    EXPECT_EQ(attrs.get_o()->get_uid(), 4);

    // Optional input tensors
    ASSERT_NE(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_bias()->get_uid(), 5);
    ASSERT_NE(attrs.get_attn_scale(), nullptr);
    EXPECT_EQ(attrs.get_attn_scale()->get_uid(), 6);
    ASSERT_NE(attrs.get_seq_len_q(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_q()->get_uid(), 7);
    ASSERT_NE(attrs.get_seq_len_kv(), nullptr);
    EXPECT_EQ(attrs.get_seq_len_kv()->get_uid(), 8);
    ASSERT_NE(attrs.get_seed(), nullptr);
    EXPECT_EQ(attrs.get_seed()->get_uid(), 9);
    ASSERT_NE(attrs.get_offset(), nullptr);
    EXPECT_EQ(attrs.get_offset()->get_uid(), 10);
    ASSERT_NE(attrs.get_dropout_mask(), nullptr);
    EXPECT_EQ(attrs.get_dropout_mask()->get_uid(), 11);
    ASSERT_NE(attrs.get_dropout_scale(), nullptr);
    EXPECT_EQ(attrs.get_dropout_scale()->get_uid(), 12);
    ASSERT_NE(attrs.get_page_table_k(), nullptr);
    EXPECT_EQ(attrs.get_page_table_k()->get_uid(), 13);
    ASSERT_NE(attrs.get_page_table_v(), nullptr);
    EXPECT_EQ(attrs.get_page_table_v()->get_uid(), 14);
    ASSERT_NE(attrs.get_block_mask(), nullptr);
    EXPECT_EQ(attrs.get_block_mask()->get_uid(), 15);
    ASSERT_NE(attrs.get_sink_token(), nullptr);
    EXPECT_EQ(attrs.get_sink_token()->get_uid(), 16);
    ASSERT_NE(attrs.get_descale_q(), nullptr);
    EXPECT_EQ(attrs.get_descale_q()->get_uid(), 17);
    ASSERT_NE(attrs.get_descale_k(), nullptr);
    EXPECT_EQ(attrs.get_descale_k()->get_uid(), 18);
    ASSERT_NE(attrs.get_descale_v(), nullptr);
    EXPECT_EQ(attrs.get_descale_v()->get_uid(), 19);
    ASSERT_NE(attrs.get_descale_s(), nullptr);
    EXPECT_EQ(attrs.get_descale_s()->get_uid(), 20);
    ASSERT_NE(attrs.get_scale_s(), nullptr);
    EXPECT_EQ(attrs.get_scale_s()->get_uid(), 21);
    ASSERT_NE(attrs.get_scale_o(), nullptr);
    EXPECT_EQ(attrs.get_scale_o()->get_uid(), 22);

    // Optional output tensors
    ASSERT_NE(attrs.get_stats(), nullptr);
    EXPECT_EQ(attrs.get_stats()->get_uid(), 23);
    ASSERT_NE(attrs.get_max(), nullptr);
    EXPECT_EQ(attrs.get_max()->get_uid(), 24);
    ASSERT_NE(attrs.get_sum_exp(), nullptr);
    EXPECT_EQ(attrs.get_sum_exp()->get_uid(), 25);
    ASSERT_NE(attrs.get_rng_dump(), nullptr);
    EXPECT_EQ(attrs.get_rng_dump()->get_uid(), 26);
    ASSERT_NE(attrs.get_amax_s(), nullptr);
    EXPECT_EQ(attrs.get_amax_s()->get_uid(), 27);
    ASSERT_NE(attrs.get_amax_o(), nullptr);
    EXPECT_EQ(attrs.get_amax_o()->get_uid(), 28);

    // Boolean flags
    ASSERT_TRUE(attrs.generate_stats.has_value());
    EXPECT_TRUE(*attrs.generate_stats);
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
    ASSERT_TRUE(attrs.max_seq_len_kv.has_value());
    EXPECT_EQ(*attrs.max_seq_len_kv, 64);

    // Enum attributes
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
    EXPECT_EQ(attrs.mma_core_mode, DataType::HALF);
    EXPECT_EQ(attrs.implementation, AttentionImplementation::COMPOSITE);
}

TEST(TestSdpaAttributes, FromFlatBufferNoOptionals)
{
    // Pack a minimal attrs (required fields only), then verify fromFlatBuffer leaves all
    // optional fields at their defaults
    SdpaAttributes original;
    original.set_q(makeTensor(1)).set_k(makeTensor(2)).set_v(makeTensor(3)).set_o(makeTensor(4));

    flatbuffers::FlatBufferBuilder builder;
    auto packed = original.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::SdpaAttributes>(buf);

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    for(int64_t uid = 1; uid <= 4; ++uid)
    {
        tensorMap[uid] = makeTensor(uid);
    }

    auto attrs = SdpaAttributes::fromFlatBuffer(fb, tensorMap);

    ASSERT_NE(attrs.get_q(), nullptr);
    EXPECT_EQ(attrs.get_q()->get_uid(), 1);
    ASSERT_NE(attrs.get_k(), nullptr);
    EXPECT_EQ(attrs.get_k()->get_uid(), 2);
    ASSERT_NE(attrs.get_v(), nullptr);
    EXPECT_EQ(attrs.get_v()->get_uid(), 3);
    ASSERT_NE(attrs.get_o(), nullptr);
    EXPECT_EQ(attrs.get_o()->get_uid(), 4);

    // All optional tensors absent
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
    EXPECT_EQ(attrs.get_stats(), nullptr);
    EXPECT_EQ(attrs.get_max(), nullptr);
    EXPECT_EQ(attrs.get_sum_exp(), nullptr);
    EXPECT_EQ(attrs.get_rng_dump(), nullptr);
    EXPECT_EQ(attrs.get_amax_s(), nullptr);
    EXPECT_EQ(attrs.get_amax_o(), nullptr);

    // Boolean flags at defaults
    EXPECT_FALSE(attrs.generate_stats.has_value());
    EXPECT_FALSE(attrs.alibi_mask);
    EXPECT_FALSE(attrs.padding_mask);
    EXPECT_FALSE(attrs.causal_mask);
    EXPECT_FALSE(attrs.causal_mask_bottom_right);

    // Scalar attributes absent
    EXPECT_FALSE(attrs.dropout_probability.has_value());
    EXPECT_FALSE(attrs.attn_scale_value.has_value());
    EXPECT_FALSE(attrs.left_bound.has_value());
    EXPECT_FALSE(attrs.right_bound.has_value());
    EXPECT_FALSE(attrs.max_seq_len_kv.has_value());

    // Enum attributes at defaults
    EXPECT_EQ(attrs.diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(attrs.mma_core_mode, DataType::NOT_SET);
    EXPECT_EQ(attrs.implementation, AttentionImplementation::AUTO);
}
