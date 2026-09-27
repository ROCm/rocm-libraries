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

    attrs.set_attn_scale(0.5f);
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

TEST(TestSdpaBackwardAttributes, LogicalAndStrictEquality)
{
    hipdnn_frontend::graph::SdpaBackwardAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_alibi_mask(false);
    attr1.set_padding_mask(false);
    attr1.set_causal_mask(true);
    attr1.set_causal_mask_bottom_right(false);
    attr1.dropout_probability = 0.1f;
    attr1.set_attn_scale(0.125f);
    attr1.set_diagonal_band_left_bound(5);
    attr1.set_diagonal_band_right_bound(0);
    attr1.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT);

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

    auto do1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    do1->set_uid(5).set_name("dO").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_do(do1);

    auto stats1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    stats1->set_uid(6).set_name("Stats").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_stats(stats1);

    auto dq1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dq1->set_uid(7).set_name("dQ").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_dq(dq1);

    auto dk1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dk1->set_uid(8).set_name("dK").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_dk(dk1);

    auto dv1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dv1->set_uid(9).set_name("dV").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_dv(dv1);

    hipdnn_frontend::graph::SdpaBackwardAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_alibi_mask(false);
    attr2.set_padding_mask(false);
    attr2.set_causal_mask(true);
    attr2.set_causal_mask_bottom_right(false);
    attr2.dropout_probability = 0.1f;
    attr2.set_attn_scale(0.125f);
    attr2.set_diagonal_band_left_bound(5);
    attr2.set_diagonal_band_right_bound(0);
    attr2.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT);

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

    auto do2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    do2->set_uid(5).set_name("dO").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_do(do2);

    auto stats2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    stats2->set_uid(6).set_name("Stats").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_stats(stats2);

    auto dq2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dq2->set_uid(7).set_name("dQ").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_dq(dq2);

    auto dk2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dk2->set_uid(8).set_name("dK").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_dk(dk2);

    auto dv2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dv2->set_uid(9).set_name("dV").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_dv(dv2);

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

    // alibi_mask mismatch: semantic, must fail both checks
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
    // NOTE: SdpaBackwardAttributes has no standalone set_dropout_probability
    // (unlike SdpaAttributes) — only set_dropout(prob, seed, offset), so the
    // public field is set directly here to avoid touching seed/offset tensors.
    attr2.dropout_probability = 0.2f;
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.dropout_probability = 0.1f; // Revert

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

    // diagonal_alignment mismatch: semantic, must fail both checks
    attr2.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT); // Revert

    // Unset-vs-unset optional fields: two attrs with the same fields left
    // unset should still compare equal
    hipdnn_frontend::graph::SdpaBackwardAttributes sparse1;
    sparse1.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    hipdnn_frontend::graph::SdpaBackwardAttributes sparse2;
    sparse2.set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
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

TEST(TestSdpaBackwardAttributes, NativeSetAttnScaleFloatPopulatesValueNotTensor)
{
    SdpaBackwardAttributes attrs;
    attrs.set_attn_scale(0.25f);
    ASSERT_TRUE(attrs.attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(*attrs.attn_scale_value, 0.25f);
    EXPECT_EQ(attrs.get_attn_scale(), nullptr);
}

TEST(TestSdpaBackwardAttributes, NativeSetSlidingWindowLengthMapsToLeftBound)
{
    SdpaBackwardAttributes attrs;
    attrs.set_sliding_window_length(128);
    ASSERT_TRUE(attrs.left_bound.has_value());
    EXPECT_EQ(*attrs.left_bound, 128);
}

TEST(TestSdpaBackwardAttributes, NativeFusedDropoutThreeArg)
{
    SdpaBackwardAttributes attrs;
    auto mask = makeTensor(64);
    auto scale = makeTensor(65);
    auto scaleInv = makeTensor(66);
    attrs.set_dropout(mask, scale, scaleInv);
    EXPECT_EQ(attrs.get_dropout_mask(), mask);
    EXPECT_EQ(attrs.get_dropout_scale(), scale);
    EXPECT_EQ(attrs.get_dropout_scale_inv(), scaleInv);
}

TEST(TestSdpaBackwardAttributes, NativeUnsupportedScoreMod)
{
    SdpaBackwardAttributes attrs;
    attrs.set_score_mod([](int) { return 0; });
    ASSERT_TRUE(attrs.hasUnsupportedUsage());
    EXPECT_NE(attrs.getUnsupportedReason().find("score modifier"), std::string::npos);
}

TEST(TestSdpaBackwardAttributes, NativeUnsupportedScoreModBprop)
{
    SdpaBackwardAttributes attrs;
    attrs.set_score_mod_bprop([](int) { return 0; });
    ASSERT_TRUE(attrs.hasUnsupportedUsage());
    EXPECT_NE(attrs.getUnsupportedReason().find("score-modifier backprop"), std::string::npos);
}

TEST(TestSdpaBackwardAttributes, NativeUnsupportedMaxTotalSeqLenQ)
{
    SdpaBackwardAttributes attrs;
    attrs.set_max_total_seq_len_q(4096);
    ASSERT_TRUE(attrs.hasUnsupportedUsage());
    EXPECT_NE(attrs.getUnsupportedReason().find("max_total_seq_len_q"), std::string::npos);
}

TEST(TestSdpaBackwardAttributes, NativeUnsupportedMaxTotalSeqLenKv)
{
    SdpaBackwardAttributes attrs;
    attrs.set_max_total_seq_len_kv(4096);
    ASSERT_TRUE(attrs.hasUnsupportedUsage());
    EXPECT_NE(attrs.getUnsupportedReason().find("max_total_seq_len_kv"), std::string::npos);
}

TEST(TestSdpaBackwardAttributes, NativeUnsupportedRngDump)
{
    SdpaBackwardAttributes attrs;
    attrs.set_rng_dump(makeTensor(70));
    ASSERT_TRUE(attrs.hasUnsupportedUsage());
    EXPECT_NE(attrs.getUnsupportedReason().find("RNG dump"), std::string::npos);
}

TEST(TestSdpaBackwardAttributes, NativeUnsupportedSinkToken)
{
    SdpaBackwardAttributes attrs;
    attrs.set_sink_token(makeTensor(71));
    ASSERT_TRUE(attrs.hasUnsupportedUsage());
    EXPECT_NE(attrs.getUnsupportedReason().find("sink token"), std::string::npos);
}

TEST(TestSdpaBackwardAttributes, NativeUnsupportedDsinkToken)
{
    SdpaBackwardAttributes attrs;
    attrs.set_dsink_token(makeTensor(72));
    ASSERT_TRUE(attrs.hasUnsupportedUsage());
    EXPECT_NE(attrs.getUnsupportedReason().find("sink-token gradient"), std::string::npos);
}

TEST(TestSdpaBackwardAttributes, NativeDeterministicAlgorithmTrueUnsupportedFalseIgnored)
{
    SdpaBackwardAttributes falseAttrs;
    falseAttrs.set_deterministic_algorithm(false);
    EXPECT_FALSE(falseAttrs.hasUnsupportedUsage());

    SdpaBackwardAttributes trueAttrs;
    trueAttrs.set_deterministic_algorithm(true);
    ASSERT_TRUE(trueAttrs.hasUnsupportedUsage());
    EXPECT_NE(trueAttrs.getUnsupportedReason().find("Deterministic"), std::string::npos);
}

TEST(TestSdpaBackwardAttributes, UnsupportedReasonFirstWinsLatch)
{
    // First unsupported setter wins: max_total_seq_len_q recorded, score mod not.
    SdpaBackwardAttributes seqFirst;
    seqFirst.set_max_total_seq_len_q(1).set_score_mod([](int) { return 0; });
    ASSERT_TRUE(seqFirst.hasUnsupportedUsage());
    EXPECT_NE(seqFirst.getUnsupportedReason().find("max_total_seq_len_q"), std::string::npos);
    EXPECT_EQ(seqFirst.getUnsupportedReason().find("score modifier"), std::string::npos);

    // Reverse order: score modifier recorded first and retained.
    SdpaBackwardAttributes scoreFirst;
    scoreFirst.set_score_mod([](int) { return 0; }).set_max_total_seq_len_q(1);
    ASSERT_TRUE(scoreFirst.hasUnsupportedUsage());
    EXPECT_NE(scoreFirst.getUnsupportedReason().find("score modifier"), std::string::npos);
    EXPECT_EQ(scoreFirst.getUnsupportedReason().find("max_total_seq_len_q"), std::string::npos);
}
