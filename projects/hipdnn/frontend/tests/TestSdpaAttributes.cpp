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
        .set_paged_attention_k_table(pageTableK)
        .set_paged_attention_v_table(pageTableV)
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
        .set_logit_max(max)
        .set_score_sum_exp(sumExp)
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

    attrs.set_attn_scale(0.5f);
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

TEST(TestSdpaAttributes, LogicalAndStrictEquality)
{
    hipdnn_frontend::graph::SdpaAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_generate_stats(true);
    attr1.set_alibi_mask(false);
    attr1.set_padding_mask(false);
    attr1.set_causal_mask(true);
    attr1.set_causal_mask_bottom_right(false);
    attr1.set_dropout_probability(0.1f);
    attr1.set_attn_scale(0.125f);
    attr1.set_diagonal_band_left_bound(5);
    attr1.set_diagonal_band_right_bound(0);
    attr1.set_paged_attention_max_seq_len_kv(4096);
    attr1.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    attr1.set_mma_core_mode(hipdnn_frontend::DataType::FLOAT);
    attr1.set_implementation(hipdnn_frontend::AttentionImplementation::AUTO);

    auto q1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    q1->set_uid(1).set_name("Q").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_q(q1);

    auto k1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    k1->set_uid(2).set_name("K").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_k(k1);

    auto v1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    v1->set_uid(3).set_name("V").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_v(v1);

    auto o1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    o1->set_uid(4).set_name("O").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_o(o1);

    hipdnn_frontend::graph::SdpaAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_generate_stats(true);
    attr2.set_alibi_mask(false);
    attr2.set_padding_mask(false);
    attr2.set_causal_mask(true);
    attr2.set_causal_mask_bottom_right(false);
    attr2.set_dropout_probability(0.1f);
    attr2.set_attn_scale(0.125f);
    attr2.set_diagonal_band_left_bound(5);
    attr2.set_diagonal_band_right_bound(0);
    attr2.set_paged_attention_max_seq_len_kv(4096);
    attr2.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    attr2.set_mma_core_mode(hipdnn_frontend::DataType::FLOAT);
    attr2.set_implementation(hipdnn_frontend::AttentionImplementation::AUTO);

    auto q2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    q2->set_uid(1).set_name("Q").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_q(q2);

    auto k2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    k2->set_uid(2).set_name("K").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_k(k2);

    auto v2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    v2->set_uid(3).set_name("V").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_v(v2);

    auto o2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    o2->set_uid(4).set_name("O").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_o(o2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchQ = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    structuralMismatchQ->set_uid(99).set_name("MismatchedQ");
    attr2.set_q(structuralMismatchQ);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_q(q2); // Revert

    // generate_stats mismatch: semantic, must fail both checks
    attr2.set_generate_stats(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_generate_stats(true); // Revert

    // alibi_mask mismatch
    attr2.set_alibi_mask(true);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_alibi_mask(false); // Revert

    // padding_mask mismatch
    attr2.set_padding_mask(true);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_padding_mask(false); // Revert

    // causal_mask mismatch (deprecated field, still live state)
    attr2.set_causal_mask(false);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_causal_mask(true); // Revert

    // causal_mask_bottom_right mismatch (deprecated field, still live state)
    attr2.set_causal_mask_bottom_right(true);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_causal_mask_bottom_right(false); // Revert

    // dropout_probability mismatch
    attr2.set_dropout_probability(0.2f);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_dropout_probability(0.1f); // Revert

    // attn_scale_value mismatch
    attr2.set_attn_scale(0.25f);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_attn_scale(0.125f); // Revert

    // left_bound mismatch
    attr2.set_diagonal_band_left_bound(10);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_diagonal_band_left_bound(5); // Revert

    // right_bound mismatch
    attr2.set_diagonal_band_right_bound(3);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_diagonal_band_right_bound(0); // Revert

    // max_seq_len_kv mismatch
    attr2.set_paged_attention_max_seq_len_kv(2048);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_paged_attention_max_seq_len_kv(4096); // Revert

    // diagonal_alignment mismatch: semantic, must fail both checks
    attr2.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT); // Revert

    // implementation mismatch: semantic, must fail both checks
    attr2.set_implementation(hipdnn_frontend::AttentionImplementation::UNIFIED);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_implementation(hipdnn_frontend::AttentionImplementation::AUTO); // Revert

    // mma_core_mode mismatch
    attr2.set_mma_core_mode(hipdnn_frontend::DataType::HALF);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_mma_core_mode(hipdnn_frontend::DataType::FLOAT); // Revert

    // Unset-vs-unset optional fields: two attrs with the same fields left
    // unset should still compare equal
    hipdnn_frontend::graph::SdpaAttributes sparse1;
    sparse1.set_implementation(hipdnn_frontend::AttentionImplementation::AUTO);
    hipdnn_frontend::graph::SdpaAttributes sparse2;
    sparse2.set_implementation(hipdnn_frontend::AttentionImplementation::AUTO);
    EXPECT_TRUE(sparse1 == sparse2);
    EXPECT_TRUE(sparse1.logicallyEquals(sparse2));

    // Set-vs-unset should differ
    sparse2.set_attn_scale(0.1f);
    EXPECT_FALSE(sparse1 == sparse2);
    EXPECT_FALSE(sparse1.logicallyEquals(sparse2));

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchQ = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    logicalMatchQ
        ->set_uid(555) // Diverges from attr1's q1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's q1 ("Q")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_q(logicalMatchQ);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}

//==============================================================================
// cuDNN source-compatibility setters (native surface)
//==============================================================================

TEST(TestSdpaAttributes, NativeSetAttnScaleFloatPopulatesValueNotTensor)
{
    SdpaAttributes attrs;
    attrs.set_attn_scale(0.25f);

    ASSERT_TRUE(attrs.attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(*attrs.attn_scale_value, 0.25f);
    // The float overload must not populate the Attn_scale tensor input.
    EXPECT_EQ(attrs.get_attn_scale(), nullptr);
}

TEST(TestSdpaAttributes, NativeSetLogitMaxRoundTrip)
{
    SdpaAttributes attrs;
    auto logitMax = makeTensor(60);
    attrs.set_logit_max(logitMax);
    EXPECT_EQ(attrs.get_max(), logitMax);
}

TEST(TestSdpaAttributes, NativeSetScoreSumExpRoundTrip)
{
    SdpaAttributes attrs;
    auto sumExp = makeTensor(61);
    attrs.set_score_sum_exp(sumExp);
    EXPECT_EQ(attrs.get_sum_exp(), sumExp);
}

TEST(TestSdpaAttributes, NativeSetSlidingWindowLengthMapsToLeftBound)
{
    SdpaAttributes attrs;
    attrs.set_sliding_window_length(128);
    ASSERT_TRUE(attrs.left_bound.has_value());
    EXPECT_EQ(*attrs.left_bound, 128);
}

TEST(TestSdpaAttributes, NativePagedAttentionTablesRoundTrip)
{
    SdpaAttributes attrs;
    auto kTable = makeTensor(62);
    auto vTable = makeTensor(63);
    attrs.set_paged_attention_k_table(kTable);
    attrs.set_paged_attention_v_table(vTable);
    EXPECT_EQ(attrs.get_page_table_k(), kTable);
    EXPECT_EQ(attrs.get_page_table_v(), vTable);
}

TEST(TestSdpaAttributes, NativeSetMmaCoreMode)
{
    SdpaAttributes attrs;
    EXPECT_EQ(attrs.mma_core_mode, DataType::NOT_SET);
    attrs._set_mma_core_mode(DataType::HALF);
    EXPECT_EQ(attrs.mma_core_mode, DataType::HALF);
}

TEST(TestSdpaAttributes, NativeFusedDropoutTwoArg)
{
    SdpaAttributes attrs;
    auto mask = makeTensor(64);
    auto scale = makeTensor(65);
    attrs.set_dropout(mask, scale);
    EXPECT_EQ(attrs.get_dropout_mask(), mask);
    EXPECT_EQ(attrs.get_dropout_scale(), scale);
}

TEST(TestSdpaAttributes, DeprecatedSetIsInferenceMapsToGenerateStats)
{
    SdpaAttributes inferAttrs;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    inferAttrs.set_is_inference(true);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    ASSERT_TRUE(inferAttrs.generate_stats.has_value());
    EXPECT_FALSE(*inferAttrs.generate_stats);

    SdpaAttributes trainAttrs;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    trainAttrs.set_is_inference(false);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    ASSERT_TRUE(trainAttrs.generate_stats.has_value());
    EXPECT_TRUE(*trainAttrs.generate_stats);
}

TEST(TestSdpaAttributes, NativeSetScoreModRecordsUnsupported)
{
    SdpaAttributes attrs;
    attrs.set_score_mod([](int) { return 0; });

    ASSERT_TRUE(attrs.hasUnsupportedUsage());
    EXPECT_NE(attrs.getUnsupportedReason().find("score modifier"), std::string::npos);
    // Nothing applied: no stats-related output tensor was created as a side effect.
    EXPECT_EQ(attrs.get_max(), nullptr);
    EXPECT_EQ(attrs.get_sum_exp(), nullptr);
}

TEST(TestSdpaAttributes, NativeSetUnfuseFmaSetsHintNotUnsupported)
{
    SdpaAttributes attrs;
    attrs.set_unfuse_fma(true);
    EXPECT_TRUE(attrs.unfuse_fma_hint);
    EXPECT_FALSE(attrs.hasUnsupportedUsage());

    attrs.set_unfuse_fma(false);
    EXPECT_FALSE(attrs.unfuse_fma_hint);
    EXPECT_FALSE(attrs.hasUnsupportedUsage());
}
