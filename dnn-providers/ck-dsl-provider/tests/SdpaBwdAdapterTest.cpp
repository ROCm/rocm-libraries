// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cmath>
#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>
#include <vector>

#include "adapters/sdpa/SdpaBwdAdapter.hpp"
#include "adapters/sdpa/SdpaBwdSpec.hpp"

namespace {

using ck_dsl_provider::SdpaBwdAdapter;
using ck_dsl_provider::SdpaBwdSpec;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using DataType = data_objects::DataType;
using SdpaBackwardAttributes = data_objects::SdpaBackwardAttributes;

/// BSHD physical strides for a logical [B, H, S, D] tensor. Heads are
/// interleaved within each sequence position, so:
///   batch (strides[0]) = S*H*D   (== seqlen * token-stride)
///   head  (strides[1]) = D
///   token (strides[2]) = H*D
///   d     (strides[3]) = 1
/// The bwd kernel consumes token = strides[2], head = strides[1],
/// requires unit d-stride, and folds the batch offset as
/// batch_idx*seqlen*token -- so the batch stride must equal
/// seqlen*token. The adapter enforces all three; only BSHD-strided
/// tensors are accepted.
std::vector<std::int64_t> bshdStrides(int H, int S, int D) {
    return {static_cast<std::int64_t>(S) * H * D, D, static_cast<std::int64_t>(H) * D, 1};
}

/// Knobs the hand-built fixture flips, one at a time, to exercise the
/// bwd adapter's validation surface. Everything defaults to "valid": a
/// baseline [B=2, Hq=Hkv=8, Sq=Skv=16, D=64] problem with FP16 Q/K/V/O/dO,
/// FLOAT dQ/dK/dV/stats, contiguous BSHD layout, head-major contiguous
/// stats, and no mask. A test sets exactly the field(s) it wants under
/// test.
///
/// **Why hand-built rather than the SDK helper.** The test-SDK
/// ``createValidSdpaBwdGraph`` builds dQ/dK/dV with the SAME dtype as
/// Q/K/V (HALF by default), but the bwd adapter REQUIRES dQ/dK/dV to be
/// FLOAT (f32 accumulators). The helper therefore cannot produce a graph
/// the adapter accepts, so the fixture constructs each tensor + the
/// attributes table by hand with the correct per-role dtype.
struct SdpaBwdAttrConfig {
    std::int32_t B{2};
    std::int32_t Hq{8};
    std::int32_t Hkv{8};
    std::int32_t Sq{16};
    std::int32_t Skv{16};
    std::int32_t D{64};

    // Per-role dtype overrides (default to the accepted dtypes).
    DataType qDtype{DataType::HALF};
    DataType dqDtype{DataType::FLOAT};
    DataType dkDtype{DataType::FLOAT};
    DataType dvDtype{DataType::FLOAT};
    DataType statsDtype{DataType::FLOAT};

    // Mask / feature flags.
    bool causal{false};
    bool alibi{false};
    bool padding{false};
    bool causalBottomRight{false};

    // Optional uids the adapter rejects when present.
    flatbuffers::Optional<std::int64_t> attnMaskUid{flatbuffers::nullopt};
    flatbuffers::Optional<std::int64_t> scaleUid{flatbuffers::nullopt};
    flatbuffers::Optional<std::int64_t> seqLenQUid{flatbuffers::nullopt};
    flatbuffers::Optional<std::int64_t> seedUid{flatbuffers::nullopt};
    flatbuffers::Optional<std::int64_t> dbiasUid{flatbuffers::nullopt};
    flatbuffers::Optional<float> dropoutProbability{flatbuffers::nullopt};
    flatbuffers::Optional<float> attnScaleValue{flatbuffers::nullopt};

    // Stride overrides. When non-empty, used verbatim instead of the
    // derived BSHD strides, so a test can inject a non-unit head-dim
    // stride, a BHSD layout, a mismatched gradient head stride, etc.
    std::vector<std::int64_t> qStridesOverride{};
    std::vector<std::int64_t> dqStridesOverride{};
    std::vector<std::int64_t> statsStridesOverride{};
};

/// Build a single SDPA-backward node + its nine-tensor map by hand.
/// Mirrors SdpaAdapterTest's SdpaAttrFixture standalone-buffer idiom:
/// each tensor and the attributes table live in their own finished
/// FlatBuffer, rooted via GetRoot. Q/K/V/O/dO are FP16; dQ/dK/dV/stats
/// are FLOAT. UID order matches the SDK helper: q=1, k=2, v=3, o=4,
/// do=5, stats=6, dq=7, dk=8, dv=9.
struct SdpaBwdAttrFixture {
    std::vector<flatbuffers::FlatBufferBuilder> tensorBuilders;
    flatbuffers::FlatBufferBuilder attrBuilder;
    const SdpaBackwardAttributes* attr{nullptr};
    SdpaBwdAdapter::TensorMap tensorMap;

    const data_objects::TensorAttributes* finishTensor(std::int64_t uid, const std::string& name,
                                                       DataType dtype,
                                                       const std::vector<std::int64_t>& dims,
                                                       const std::vector<std::int64_t>& strides) {
        tensorBuilders.emplace_back();
        flatbuffers::FlatBufferBuilder& builder = tensorBuilders.back();
        auto attrOffset = data_objects::CreateTensorAttributesDirect(
            builder, uid, name.c_str(), dtype, &strides, &dims, /*virtual=*/false);
        builder.Finish(attrOffset);
        return flatbuffers::GetRoot<data_objects::TensorAttributes>(builder.GetBufferPointer());
    }

    explicit SdpaBwdAttrFixture(const SdpaBwdAttrConfig& cfg) {
        // tensorBuilders must not reallocate after roots are taken; nine
        // tensors are pushed below (plus the stats override path uses one
        // of these slots).
        tensorBuilders.reserve(16);

        const std::vector<std::int64_t> qoDims{cfg.B, cfg.Hq, cfg.Sq, cfg.D};
        const std::vector<std::int64_t> kvDims{cfg.B, cfg.Hkv, cfg.Skv, cfg.D};
        const std::vector<std::int64_t> qoStrides = cfg.qStridesOverride.empty()
                                                        ? bshdStrides(cfg.Hq, cfg.Sq, cfg.D)
                                                        : cfg.qStridesOverride;
        const std::vector<std::int64_t> kvStrides = bshdStrides(cfg.Hkv, cfg.Skv, cfg.D);
        const std::vector<std::int64_t> dqStrides = cfg.dqStridesOverride.empty()
                                                        ? bshdStrides(cfg.Hq, cfg.Sq, cfg.D)
                                                        : cfg.dqStridesOverride;

        // stats: rank-4 [B, Hq, Sq, 1], head-major contiguous strides
        // {Hq*Sq, Sq, 1, 1}.
        const std::vector<std::int64_t> statsDims{cfg.B, cfg.Hq, cfg.Sq, 1};
        const std::vector<std::int64_t> statsStrides =
            cfg.statsStridesOverride.empty()
                ? std::vector<std::int64_t>{static_cast<std::int64_t>(cfg.Hq) * cfg.Sq, cfg.Sq, 1,
                                            1}
                : cfg.statsStridesOverride;

        tensorMap[1] = finishTensor(1, "q", cfg.qDtype, qoDims, qoStrides);
        tensorMap[2] = finishTensor(2, "k", cfg.qDtype, kvDims, kvStrides);
        tensorMap[3] = finishTensor(3, "v", cfg.qDtype, kvDims, kvStrides);
        tensorMap[4] = finishTensor(4, "o", cfg.qDtype, qoDims, qoStrides);
        tensorMap[5] = finishTensor(5, "do", cfg.qDtype, qoDims, qoStrides);
        tensorMap[6] = finishTensor(6, "stats", cfg.statsDtype, statsDims, statsStrides);
        tensorMap[7] = finishTensor(7, "dq", cfg.dqDtype, qoDims, dqStrides);
        tensorMap[8] = finishTensor(8, "dk", cfg.dkDtype, kvDims, kvStrides);
        tensorMap[9] = finishTensor(9, "dv", cfg.dvDtype, kvDims, kvStrides);

        auto attrOffset = data_objects::CreateSdpaBackwardAttributes(
            attrBuilder, /*q_tensor_uid=*/1, /*k_tensor_uid=*/2, /*v_tensor_uid=*/3,
            /*o_tensor_uid=*/4, /*do_tensor_uid=*/5, /*stats_tensor_uid=*/6, /*dq_tensor_uid=*/7,
            /*dk_tensor_uid=*/8, /*dv_tensor_uid=*/9, /*scale_tensor_uid=*/cfg.scaleUid,
            /*attn_mask_tensor_uid=*/cfg.attnMaskUid, /*seq_len_q_tensor_uid=*/cfg.seqLenQUid,
            /*seq_len_kv_tensor_uid=*/flatbuffers::nullopt, /*seed_tensor_uid=*/cfg.seedUid,
            /*offset_tensor_uid=*/flatbuffers::nullopt,
            /*dropout_mask_tensor_uid=*/flatbuffers::nullopt,
            /*dropout_scale_tensor_uid=*/flatbuffers::nullopt,
            /*dropout_scale_inv_tensor_uid=*/flatbuffers::nullopt,
            /*dbias_tensor_uid=*/cfg.dbiasUid, /*alibi_mask=*/cfg.alibi,
            /*padding_mask=*/cfg.padding, /*causal_mask=*/cfg.causal,
            /*causal_mask_bottom_right=*/cfg.causalBottomRight,
            /*dropout_probability=*/cfg.dropoutProbability,
            /*attn_scale_value=*/cfg.attnScaleValue);
        attrBuilder.Finish(attrOffset);
        attr = flatbuffers::GetRoot<SdpaBackwardAttributes>(attrBuilder.GetBufferPointer());
    }
};

// ---- Accept cases -------------------------------------------------------

TEST(TestSdpaBwdAdapter, BuildSpecForDefaultShape) {
    SdpaBwdAttrConfig cfg;
    SdpaBwdAttrFixture fx(cfg);
    SdpaBwdSpec spec = SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap);

    EXPECT_EQ(spec.problem.B, 2);
    EXPECT_EQ(spec.problem.Hq, 8);
    EXPECT_EQ(spec.problem.Hkv, 8);
    EXPECT_EQ(spec.problem.Sq, 16);
    EXPECT_EQ(spec.problem.Skv, 16);
    EXPECT_EQ(spec.problem.D, 64);

    // BSHD: token = strides[2] = H*D = 512, head = strides[1] = D = 64.
    EXPECT_EQ(spec.problem.stride_q_token, 512);
    EXPECT_EQ(spec.problem.stride_q_head, 64);
    EXPECT_EQ(spec.problem.stride_k_token, 512);
    EXPECT_EQ(spec.problem.stride_k_head, 64);
    EXPECT_EQ(spec.problem.stride_v_token, 512);
    EXPECT_EQ(spec.problem.stride_v_head, 64);
    EXPECT_EQ(spec.problem.stride_do_token, 512);
    EXPECT_EQ(spec.problem.stride_do_head, 64);
    EXPECT_EQ(spec.problem.stride_dq_token, 512);
    EXPECT_EQ(spec.problem.stride_dk_token, 512);
    EXPECT_EQ(spec.problem.stride_dv_token, 512);

    EXPECT_EQ(spec.dtype, "f16");
    EXPECT_EQ(spec.mask_mode, "none");

    // Default scale: 1/sqrt(D) folded into log2 space, plus the raw
    // 1/sqrt(D) carried separately.
    constexpr float kLog2E = 1.44269504088896340736f;
    const float invSqrtD = 1.0f / std::sqrt(64.0f);
    EXPECT_NEAR(spec.problem.scale_inv, invSqrtD, 1e-6);
    EXPECT_NEAR(spec.problem.scale_log2, invSqrtD * kLog2E, 1e-6);
}

TEST(TestSdpaBwdAdapter, AcceptsCausalMask) {
    SdpaBwdAttrConfig cfg;
    cfg.causal = true;
    SdpaBwdAttrFixture fx(cfg);
    SdpaBwdSpec spec = SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap);
    EXPECT_EQ(spec.mask_mode, "causal");
}

TEST(TestSdpaBwdAdapter, AcceptsGqa) {
    // Hq=8, Hkv=2 (ratio 4), D=64.
    SdpaBwdAttrConfig cfg;
    cfg.Hkv = 2;
    SdpaBwdAttrFixture fx(cfg);
    SdpaBwdSpec spec = SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap);
    EXPECT_EQ(spec.problem.Hq, 8);
    EXPECT_EQ(spec.problem.Hkv, 2);
    EXPECT_EQ(spec.problem.Hq % spec.problem.Hkv, 0);
}

TEST(TestSdpaBwdAdapter, UsesExplicitAttnScaleValue) {
    SdpaBwdAttrConfig cfg;
    cfg.attnScaleValue = flatbuffers::Optional<float>(0.25f);
    SdpaBwdAttrFixture fx(cfg);
    SdpaBwdSpec spec = SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap);
    constexpr float kLog2E = 1.44269504088896340736f;
    EXPECT_NEAR(spec.problem.scale_inv, 0.25f, 1e-6);
    EXPECT_NEAR(spec.problem.scale_log2, 0.25f * kLog2E, 1e-6);
}

// ---- dtype rejects ------------------------------------------------------

TEST(TestSdpaBwdAdapter, RejectsBf16Inputs) {
    // bf16 Q/K/V/O/dO -- the bwd path is FP16-only.
    SdpaBwdAttrConfig cfg;
    cfg.qDtype = DataType::BFLOAT16;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsHalfGradients) {
    // dQ/dK/dV must be FLOAT (f32 accumulators); HALF is rejected. This
    // is exactly the constraint the SDK helper's default graph violates.
    SdpaBwdAttrConfig cfg;
    cfg.dqDtype = DataType::HALF;
    cfg.dkDtype = DataType::HALF;
    cfg.dvDtype = DataType::HALF;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsNonFloatStats) {
    // stats carries the natural-log LSE in f32; HALF is rejected.
    SdpaBwdAttrConfig cfg;
    cfg.statsDtype = DataType::HALF;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---- shape rejects ------------------------------------------------------

TEST(TestSdpaBwdAdapter, RejectsHeadSize32) {
    // D=32 is accepted by the forward path but rejected by the bwd kernel
    // (head_size must be >= WARP_SIZE and a multiple of 64).
    SdpaBwdAttrConfig cfg;
    cfg.D = 32;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsSeqlenQNotMultipleOf16) {
    // Sq=20 (not %16). Skv stays a multiple of 16 so the only failure is
    // the seqlen_q check.
    SdpaBwdAttrConfig cfg;
    cfg.Sq = 20;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsNonDivisibleGqa) {
    // Hq=8, Hkv=3 -> 8 % 3 != 0.
    SdpaBwdAttrConfig cfg;
    cfg.Hkv = 3;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---- layout rejects -----------------------------------------------------

TEST(TestSdpaBwdAdapter, RejectsNonUnitHeadDimStride) {
    // BSHD shape but with a non-unit d-stride on Q. All Q strides scaled
    // by 2 to keep the layout internally consistent; the kernel requires
    // a unit head-dim stride.
    SdpaBwdAttrConfig cfg;
    cfg.qStridesOverride = {16384, 128, 1024, 2};
    cfg.dqStridesOverride = {16384, 128, 1024, 2};
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, BhsdLayoutRejectedForBatchGt1) {
    // Contiguous BHSD strides {H*S*D, S*D, D, 1} = {8192, 1024, 64, 1}
    // for [2, 8, 16, 64]. With B>1 the batch stride (8192) != seqlen *
    // sequence stride (16 * 1024), so the adapter rejects.
    SdpaBwdAttrConfig cfg;
    cfg.qStridesOverride = {8192, 1024, 64, 1};
    cfg.dqStridesOverride = {8192, 1024, 64, 1};
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsMismatchedGradientHeadStride) {
    // dQ uses a head stride (strides[1]) that differs from Q's. The
    // kernel reuses the input head stride for the gradient write, so a
    // mismatch is rejected. Build a dQ layout that is otherwise valid
    // BSHD (unit d-stride, batch == seqlen*token) but doubles the
    // per-head spacing: dims [2,8,16,64] with token=512 -> batch must be
    // 16*512=8192, head set to 128 (!= Q's 64).
    SdpaBwdAttrConfig cfg;
    cfg.dqStridesOverride = {8192, 128, 512, 1};
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsNonContiguousStats) {
    // stats must be contiguous head-major {Hq*Sq, Sq, 1, 1}. Double the
    // head stride so it is no longer Sq; the LSE-prep kernel reads stats
    // as a flat contiguous buffer, so this is rejected.
    SdpaBwdAttrConfig cfg;
    cfg.statsStridesOverride = {static_cast<std::int64_t>(cfg.Hq) * cfg.Sq * 2,
                                static_cast<std::int64_t>(cfg.Sq) * 2, 1, 1};
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---- mask / feature rejects --------------------------------------------

TEST(TestSdpaBwdAdapter, RejectsAlibiMask) {
    SdpaBwdAttrConfig cfg;
    cfg.alibi = true;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsPaddingMask) {
    SdpaBwdAttrConfig cfg;
    cfg.padding = true;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsBottomRightCausalMask) {
    SdpaBwdAttrConfig cfg;
    cfg.causalBottomRight = true;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsAttnMask) {
    SdpaBwdAttrConfig cfg;
    cfg.attnMaskUid = flatbuffers::Optional<std::int64_t>(20);
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsScaleTensor) {
    SdpaBwdAttrConfig cfg;
    cfg.scaleUid = flatbuffers::Optional<std::int64_t>(20);
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsVariableLengthSequences) {
    SdpaBwdAttrConfig cfg;
    cfg.seqLenQUid = flatbuffers::Optional<std::int64_t>(20);
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsDropoutTensors) {
    SdpaBwdAttrConfig cfg;
    cfg.seedUid = flatbuffers::Optional<std::int64_t>(20);
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsDropoutProbability) {
    SdpaBwdAttrConfig cfg;
    cfg.dropoutProbability = flatbuffers::Optional<float>(0.1f);
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsDbiasOutput) {
    SdpaBwdAttrConfig cfg;
    cfg.dbiasUid = flatbuffers::Optional<std::int64_t>(20);
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaBwdAdapter, RejectsZeroSeqlen) {
    // Sq=0 trips the positivity guard before the % 16 check (0 % 16 == 0).
    SdpaBwdAttrConfig cfg;
    cfg.Sq = 0;
    SdpaBwdAttrFixture fx(cfg);
    EXPECT_THROW(SdpaBwdAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

}  // namespace
