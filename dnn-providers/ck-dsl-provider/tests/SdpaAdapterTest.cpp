// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <pybind11/embed.h>

#include <cmath>
#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <string>
#include <vector>

#include "CkDslContainer.hpp"
#include "adapters/sdpa/SdpaAdapter.hpp"
#include "adapters/sdpa/SdpaPayload.hpp"
#include "adapters/sdpa/SdpaSpec.hpp"

namespace py = pybind11;

namespace {

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::SdpaAdapter;
using ck_dsl_provider::SdpaSpec;
using ck_dsl_provider::sdpaSpecToPayload;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using DataType = data_objects::DataType;
using SdpaAttributes = data_objects::SdpaAttributes;

/// BSHD physical strides for a logical [B, H, S, D] tensor. Heads are
/// interleaved within each sequence position, so:
///   batch (strides[0]) = S*H*D   (== seqlen * token-stride)
///   head  (strides[1]) = D
///   token (strides[2]) = H*D
///   d     (strides[3]) = 1
/// The kernel consumes token = strides[2], head = strides[1], requires
/// unit d-stride, and folds the batch offset as batch_idx*seqlen*token --
/// so the batch stride must equal seqlen*token. The adapter enforces all
/// three; only BSHD-strided tensors are accepted.
std::vector<std::int64_t> bshdStrides(int H, int S, int D) {
    return {static_cast<std::int64_t>(S) * H * D, D, static_cast<std::int64_t>(H) * D, 1};
}

/// Pull the single SDPA node's attributes + the tensor map out of a
/// GraphWrapper built by the test SDK helper -- the same surfaces the
/// plan builder hands the adapter at runtime.
SdpaSpec buildSpecFromGraph(const flatbuffer_utilities::GraphWrapper& graph) {
    const auto& sdpaAttr = graph.getNodeWrapper(0).attributesAs<SdpaAttributes>();
    return SdpaAdapter::buildSpec(sdpaAttr, graph.getTensorMap());
}

/// Valid SDPA-fwd graph: B=2, Hq=Hkv=8, Sq=Skv=16, D=64, FP16, no mask,
/// BSHD strides. For these dims BSHD = {8192, 64, 512, 1}: batch =
/// 16*8*64 = 8192, head = 64, token = 8*64 = 512, d = 1.
TEST(TestSdpaAdapter, BuildSpecForDefaultShape) {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    SdpaSpec spec = buildSpecFromGraph(graph);

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
    EXPECT_EQ(spec.problem.stride_o_token, 512);
    EXPECT_EQ(spec.problem.stride_o_head, 64);

    EXPECT_EQ(spec.dtype, "f16");
    EXPECT_EQ(spec.mask_mode, "none");

    // Default scale: 1/sqrt(D) folded into log2 space.
    const float expectedScaleLog2 = (1.0f / std::sqrt(64.0f)) * static_cast<float>(M_LOG2E);
    EXPECT_NEAR(spec.problem.scale_log2, expectedScaleLog2, 1e-5);
}

TEST(TestSdpaAdapter, AcceptsCausalMask) {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF, /*withAttnMask=*/false, /*withScale=*/false,
        /*withStats=*/false, /*alibiMask=*/false, /*paddingMask=*/false, /*causalMask=*/true);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    SdpaSpec spec = buildSpecFromGraph(graph);
    EXPECT_EQ(spec.mask_mode, "causal");
}

TEST(TestSdpaAdapter, AcceptsGqa) {
    // Hq=8, Hkv=2 (ratio 4), D=64, Sq=Skv=16. BSHD strides:
    //   Q/O (H=8): {8192, 64, 512, 1}
    //   K/V (H=2): {2048, 64, 128, 1}  (batch=16*2*64, token=2*64=128)
    const auto qoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    const auto kvStrides = bshdStrides(/*H=*/2, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qoStrides,
        /*kDims=*/{2, 2, 16, 64}, /*kStrides=*/kvStrides,
        /*vDims=*/{2, 2, 16, 64}, /*vStrides=*/kvStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qoStrides,
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    SdpaSpec spec = buildSpecFromGraph(graph);
    EXPECT_EQ(spec.problem.Hq, 8);
    EXPECT_EQ(spec.problem.Hkv, 2);
    EXPECT_EQ(spec.problem.Hq % spec.problem.Hkv, 0);
}

// ---- Reject cases -------------------------------------------------------

TEST(TestSdpaAdapter, RejectsBf16Dtype) {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::BFLOAT16);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsSeqlenQNotMultipleOf16) {
    // Sq=20 (not %16). Keep BSHD strides so the only failure is the
    // seqlen check, not the layout or a cross-tensor mismatch.
    const auto qoStrides = bshdStrides(/*H=*/8, /*S=*/20, /*D=*/64);
    const auto kvStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 20, 64}, /*qStrides=*/qoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/kvStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/kvStrides,
        /*oDims=*/{2, 8, 20, 64}, /*oStrides=*/qoStrides,
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsUnsupportedHeadSize) {
    // D=48 is not in {32, 64, 128, 192, 256}. BSHD strides for D=48.
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/48);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 48}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 48}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 48}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 48}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsNonDivisibleGqa) {
    // Hq=8, Hkv=3 -> 8 % 3 != 0. BSHD strides per tensor's head count.
    const auto qoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    const auto kvStrides = bshdStrides(/*H=*/3, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qoStrides,
        /*kDims=*/{2, 3, 16, 64}, /*kStrides=*/kvStrides,
        /*vDims=*/{2, 3, 16, 64}, /*vStrides=*/kvStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qoStrides,
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsNonUnitHeadDimStride) {
    // BSHD shape but with a non-unit d-stride (strides[3]=2). All other
    // strides are scaled by 2 to keep the layout internally consistent;
    // the kernel requires unit head-dim stride, so the adapter rejects.
    const std::vector<std::int64_t> qkvoStrides{16384, 128, 1024, 2};
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, BhsdLayoutRejectedForBatchGt1) {
    // Contiguous BHSD strides {H*S*D, S*D, D, 1} = {8192, 1024, 64, 1}.
    // With B>1 this violates the batch-contiguity contract (batch stride
    // must equal seqlen*token = 16*1024 != 8192), so the adapter rejects.
    const std::vector<std::int64_t> bhsdStrides{8192, 1024, 64, 1};
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/bhsdStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/bhsdStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/bhsdStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/bhsdStrides,
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsAttnMask) {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF, /*withAttnMask=*/true);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsScaleTensor) {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF, /*withAttnMask=*/false, /*withScale=*/true);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsStatsOutput) {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF, /*withAttnMask=*/false, /*withScale=*/false,
        /*withStats=*/true);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsAlibiMask) {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF, /*withAttnMask=*/false, /*withScale=*/false,
        /*withStats=*/false, /*alibiMask=*/true);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsPaddingMask) {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF, /*withAttnMask=*/false, /*withScale=*/false,
        /*withStats=*/false, /*alibiMask=*/false, /*paddingMask=*/true);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_THROW(buildSpecFromGraph(graph), hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---- Hand-built SdpaAttributes fixture ----------------------------------

/// Knobs the hand-built fixture can flip, one at a time, to exercise the
/// adapter's scalar/optional-uid validation surface. Everything defaults
/// to "valid": no extra tensors, no scalars, contiguous BSHD layout. A
/// test sets exactly the field it wants under test.
struct SdpaAttrConfig {
    // Tensor geometry (BSHD strides are derived from these). Defaults are
    // the valid baseline [B=2, Hq=Hkv=8, Sq=Skv=16, D=64].
    std::int32_t B{2};
    std::int32_t Hq{8};
    std::int32_t Hkv{8};
    std::int32_t Sq{16};
    std::int32_t Skv{16};
    std::int32_t D{64};

    // Optional scalars / uids the M1 forward adapter rejects when present.
    flatbuffers::Optional<float> attnScaleValue{flatbuffers::nullopt};
    flatbuffers::Optional<float> dropoutProbability{flatbuffers::nullopt};
    flatbuffers::Optional<std::int64_t> descaleQUid{flatbuffers::nullopt};
    flatbuffers::Optional<std::int64_t> amaxSUid{flatbuffers::nullopt};
    flatbuffers::Optional<std::int64_t> rngDumpUid{flatbuffers::nullopt};
};

/// Build a single SDPA node + its Q/K/V/O tensor map by hand so we can
/// set scalars / optional uids the SDK helper does not expose (e.g.
/// ``attn_scale_value``, fp8 descales, amax outputs, rng dump, dropout
/// probability). Mirrors the conv ``ConvGraphFixture`` standalone-buffer
/// idiom: each tensor and the attributes table live in their own finished
/// FlatBuffer, rooted via GetRoot. Tensors use BSHD strides so the only
/// thing under test is the flipped feature.
struct SdpaAttrFixture {
    flatbuffers::FlatBufferBuilder qBuilder;
    flatbuffers::FlatBufferBuilder kBuilder;
    flatbuffers::FlatBufferBuilder vBuilder;
    flatbuffers::FlatBufferBuilder oBuilder;
    flatbuffers::FlatBufferBuilder attrBuilder;
    const data_objects::TensorAttributes* q{nullptr};
    const data_objects::TensorAttributes* k{nullptr};
    const data_objects::TensorAttributes* v{nullptr};
    const data_objects::TensorAttributes* o{nullptr};
    const SdpaAttributes* attr{nullptr};
    SdpaAdapter::TensorMap tensorMap;

    static const data_objects::TensorAttributes* finishTensor(
        flatbuffers::FlatBufferBuilder& builder, std::int64_t uid, const std::string& name,
        const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& strides) {
        auto attrOffset = data_objects::CreateTensorAttributesDirect(
            builder, uid, name.c_str(), DataType::HALF, &strides, &dims, /*virtual=*/false);
        builder.Finish(attrOffset);
        return flatbuffers::GetRoot<data_objects::TensorAttributes>(builder.GetBufferPointer());
    }

    explicit SdpaAttrFixture(const SdpaAttrConfig& cfg) {
        const std::vector<std::int64_t> qoDims{cfg.B, cfg.Hq, cfg.Sq, cfg.D};
        const std::vector<std::int64_t> kvDims{cfg.B, cfg.Hkv, cfg.Skv, cfg.D};
        const std::vector<std::int64_t> qoStrides = bshdStrides(cfg.Hq, cfg.Sq, cfg.D);
        const std::vector<std::int64_t> kvStrides = bshdStrides(cfg.Hkv, cfg.Skv, cfg.D);
        q = finishTensor(qBuilder, /*uid=*/1, "q", qoDims, qoStrides);
        k = finishTensor(kBuilder, /*uid=*/2, "k", kvDims, kvStrides);
        v = finishTensor(vBuilder, /*uid=*/3, "v", kvDims, kvStrides);
        o = finishTensor(oBuilder, /*uid=*/4, "o", qoDims, qoStrides);
        tensorMap[1] = q;
        tensorMap[2] = k;
        tensorMap[3] = v;
        tensorMap[4] = o;

        auto attrOffset = data_objects::CreateSdpaAttributes(
            attrBuilder, /*q_tensor_uid=*/1, /*k_tensor_uid=*/2, /*v_tensor_uid=*/3,
            /*o_tensor_uid=*/4, /*attn_mask_tensor_uid=*/flatbuffers::nullopt,
            /*scale_tensor_uid=*/flatbuffers::nullopt,
            /*seq_len_q_tensor_uid=*/flatbuffers::nullopt,
            /*seq_len_kv_tensor_uid=*/flatbuffers::nullopt,
            /*seed_tensor_uid=*/flatbuffers::nullopt,
            /*offset_tensor_uid=*/flatbuffers::nullopt,
            /*dropout_mask_tensor_uid=*/flatbuffers::nullopt,
            /*dropout_scale_tensor_uid=*/flatbuffers::nullopt,
            /*page_table_k_tensor_uid=*/flatbuffers::nullopt,
            /*page_table_v_tensor_uid=*/flatbuffers::nullopt,
            /*block_mask_tensor_uid=*/flatbuffers::nullopt,
            /*sink_token_tensor_uid=*/flatbuffers::nullopt,
            /*descale_q_tensor_uid=*/cfg.descaleQUid,
            /*descale_k_tensor_uid=*/flatbuffers::nullopt,
            /*descale_v_tensor_uid=*/flatbuffers::nullopt,
            /*descale_s_tensor_uid=*/flatbuffers::nullopt,
            /*scale_s_tensor_uid=*/flatbuffers::nullopt,
            /*scale_o_tensor_uid=*/flatbuffers::nullopt,
            /*stats_tensor_uid=*/flatbuffers::nullopt,
            /*max_tensor_uid=*/flatbuffers::nullopt,
            /*sum_exp_tensor_uid=*/flatbuffers::nullopt,
            /*rng_dump_tensor_uid=*/cfg.rngDumpUid,
            /*amax_s_tensor_uid=*/cfg.amaxSUid,
            /*amax_o_tensor_uid=*/flatbuffers::nullopt,
            /*generate_stats=*/flatbuffers::nullopt, /*alibi_mask=*/false, /*padding_mask=*/false,
            /*causal_mask=*/false, /*causal_mask_bottom_right=*/false,
            /*dropout_probability=*/cfg.dropoutProbability,
            /*attn_scale_value=*/cfg.attnScaleValue);
        attrBuilder.Finish(attrOffset);
        attr = flatbuffers::GetRoot<SdpaAttributes>(attrBuilder.GetBufferPointer());
    }
};

TEST(TestSdpaAdapter, UsesExplicitAttnScaleValue) {
    SdpaAttrConfig cfg;
    cfg.attnScaleValue = flatbuffers::Optional<float>(0.25f);
    SdpaAttrFixture fx(cfg);
    SdpaSpec spec = SdpaAdapter::buildSpec(*fx.attr, fx.tensorMap);
    EXPECT_NEAR(spec.problem.scale_log2, 0.25f * static_cast<float>(M_LOG2E), 1e-5);
}

// ---- Feature rejects via the hand-built fixture -------------------------

TEST(TestSdpaAdapter, RejectsFp8DescaleTensor) {
    // An fp8 descale uid is set -- the M1 forward path is FP16-only and
    // rejects any descale tensor.
    SdpaAttrConfig cfg;
    cfg.descaleQUid = flatbuffers::Optional<std::int64_t>(5);
    SdpaAttrFixture fx(cfg);
    EXPECT_THROW(SdpaAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsAmaxOutput) {
    SdpaAttrConfig cfg;
    cfg.amaxSUid = flatbuffers::Optional<std::int64_t>(5);
    SdpaAttrFixture fx(cfg);
    EXPECT_THROW(SdpaAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsRngDump) {
    SdpaAttrConfig cfg;
    cfg.rngDumpUid = flatbuffers::Optional<std::int64_t>(5);
    SdpaAttrFixture fx(cfg);
    EXPECT_THROW(SdpaAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsDropoutProbability) {
    SdpaAttrConfig cfg;
    cfg.dropoutProbability = flatbuffers::Optional<float>(0.1f);
    SdpaAttrFixture fx(cfg);
    EXPECT_THROW(SdpaAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaAdapter, RejectsZeroSeqlen) {
    // Sq=0 trips the adapter's positivity guard (which precedes the % 16
    // check) before any kernel could be built.
    SdpaAttrConfig cfg;
    cfg.Sq = 0;
    SdpaAttrFixture fx(cfg);
    EXPECT_THROW(SdpaAdapter::buildSpec(*fx.attr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---- Payload round-trip (needs the embedded interpreter) ----------------

/// Payload conversion allocates Python objects, so the embedded
/// interpreter must be up. Constructing CkDslContainer runs
/// Py_Initialize before we touch any py::* call.
class TestSdpaPayload : public ::testing::Test {
   protected:
    void SetUp() override {
        _container = std::make_unique<CkDslContainer>();
    }

    std::unique_ptr<CkDslContainer> _container;
};

TEST_F(TestSdpaPayload, PayloadDictForDefaultShape) {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    SdpaSpec spec = buildSpecFromGraph(graph);

    py::gil_scoped_acquire gil;
    py::dict payload = sdpaSpecToPayload(spec);

    ASSERT_TRUE(payload.contains("batch"));
    ASSERT_TRUE(payload.contains("shape"));
    ASSERT_TRUE(payload.contains("dtype"));
    ASSERT_TRUE(payload.contains("mask_mode"));
    ASSERT_TRUE(payload.contains("seqlen_q"));
    ASSERT_TRUE(payload.contains("seqlen_k"));

    EXPECT_EQ(payload["batch"].cast<int>(), 2);
    EXPECT_EQ(payload["dtype"].cast<std::string>(), "f16");
    EXPECT_EQ(payload["mask_mode"].cast<std::string>(), "none");
    EXPECT_EQ(payload["seqlen_q"].cast<int>(), 16);
    EXPECT_EQ(payload["seqlen_k"].cast<int>(), 16);

    auto shape = payload["shape"].cast<py::dict>();
    EXPECT_EQ(shape["head_size"].cast<int>(), 64);
    EXPECT_EQ(shape["num_query_heads"].cast<int>(), 8);
    EXPECT_EQ(shape["num_kv_heads"].cast<int>(), 8);
}

}  // namespace
