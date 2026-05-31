// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <pybind11/embed.h>

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <memory>
#include <string>
#include <vector>

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "TestUtils.hpp"
#include "adapters/sdpa/SdpaBwdAdapter.hpp"
#include "adapters/sdpa/SdpaBwdPayload.hpp"
#include "adapters/sdpa/SdpaBwdSpec.hpp"
#include "engines/sdpa/SdpaBwdPlan.hpp"
#include "engines/sdpa/SdpaBwdPlanBuilder.hpp"
#include "python/CompileServiceBridge.hpp"
#include "runtime/JitCache.hpp"

namespace {

namespace py = pybind11;

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::CkDslContext;
using ck_dsl_provider::SdpaBwdAdapter;
using ck_dsl_provider::SdpaBwdPlan;
using ck_dsl_provider::SdpaBwdPlanBuilder;
using ck_dsl_provider::SdpaBwdSpec;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using DataType = data_objects::DataType;
using SdpaBackwardAttributes = data_objects::SdpaBackwardAttributes;

/// BSHD physical strides for a logical [B, H, S, D] tensor -- the layout
/// the FMHA kernels require: batch = S*H*D (== seqlen*token), head = D,
/// token = H*D, d = 1.
std::vector<std::int64_t> bshdStrides(int H, int S, int D) {
    return {static_cast<std::int64_t>(S) * H * D, D, static_cast<std::int64_t>(H) * D, 1};
}

/// Build a complete single-node SDPA-backward graph FlatBuffer the
/// GraphWrapper can parse -- the same surface the SDK hands the plan
/// builder at runtime.
///
/// **Why not the SDK helper.** ``createValidSdpaBwdGraph`` builds
/// dQ/dK/dV with the same dtype as Q/K/V (HALF by default), but the bwd
/// adapter REQUIRES dQ/dK/dV FLOAT (f32 accumulators). The helper cannot
/// produce a graph the adapter accepts, so this builds the graph by hand
/// with FP16 Q/K/V/O/dO and FLOAT stats/dQ/dK/dV. UID order matches the
/// helper: q=1, k=2, v=3, o=4, do=5, stats=6, dq=7, dk=8, dv=9.
flatbuffers::FlatBufferBuilder makeValidSdpaBwdGraph(int B = 2, int Hq = 8, int Hkv = 8,
                                                     int Sq = 16, int Skv = 16, int D = 64,
                                                     bool causal = false) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensorAttributes;

    const std::vector<std::int64_t> qoDims{B, Hq, Sq, D};
    const std::vector<std::int64_t> kvDims{B, Hkv, Skv, D};
    const std::vector<std::int64_t> qoStrides = bshdStrides(Hq, Sq, D);
    const std::vector<std::int64_t> kvStrides = bshdStrides(Hkv, Skv, D);

    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/1, "q", DataType::HALF, &qoStrides, &qoDims));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/2, "k", DataType::HALF, &kvStrides, &kvDims));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/3, "v", DataType::HALF, &kvStrides, &kvDims));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/4, "o", DataType::HALF, &qoStrides, &qoDims));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/5, "do", DataType::HALF, &qoStrides, &qoDims));

    // stats: [B, Hq, Sq, 1] FLOAT, head-major contiguous.
    const std::vector<std::int64_t> statsDims{B, Hq, Sq, 1};
    const std::vector<std::int64_t> statsStrides{static_cast<std::int64_t>(Hq) * Sq, Sq, 1, 1};
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/6, "stats", DataType::FLOAT, &statsStrides, &statsDims));

    // dQ/dK/dV: FLOAT gradient accumulators mirroring Q/K/V dims+strides.
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/7, "dq", DataType::FLOAT, &qoStrides, &qoDims));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/8, "dk", DataType::FLOAT, &kvStrides, &kvDims));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/9, "dv", DataType::FLOAT, &kvStrides, &kvDims));

    auto sdpaBwdAttributes = data_objects::CreateSdpaBackwardAttributes(
        builder, /*q=*/1, /*k=*/2, /*v=*/3, /*o=*/4, /*do=*/5, /*stats=*/6, /*dq=*/7, /*dk=*/8,
        /*dv=*/9, /*scale_tensor_uid=*/flatbuffers::nullopt,
        /*attn_mask_tensor_uid=*/flatbuffers::nullopt,
        /*seq_len_q_tensor_uid=*/flatbuffers::nullopt,
        /*seq_len_kv_tensor_uid=*/flatbuffers::nullopt, /*seed_tensor_uid=*/flatbuffers::nullopt,
        /*offset_tensor_uid=*/flatbuffers::nullopt,
        /*dropout_mask_tensor_uid=*/flatbuffers::nullopt,
        /*dropout_scale_tensor_uid=*/flatbuffers::nullopt,
        /*dropout_scale_inv_tensor_uid=*/flatbuffers::nullopt,
        /*dbias_tensor_uid=*/flatbuffers::nullopt, /*alibi_mask=*/false, /*padding_mask=*/false,
        /*causal_mask=*/causal);

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(
        builder, "sdpa_bwd", DataType::HALF, data_objects::NodeAttributes::SdpaBackwardAttributes,
        sdpaBwdAttributes.Union()));

    auto graphOffset =
        data_objects::CreateGraphDirect(builder, "test", DataType::FLOAT, DataType::HALF,
                                        DataType::BFLOAT16, &tensorAttributes, &nodes);
    builder.Finish(graphOffset);
    return builder;
}

/// A valid bwd spec built straight off the default graph; the arch-gate
/// test only needs the spec (not the FlatBuffer) to drive the bridge
/// validator.
SdpaBwdSpec makeValidSpec() {
    auto fbBuilder = makeValidSdpaBwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    const auto& attr = graph.getNodeWrapper(0).attributesAs<SdpaBackwardAttributes>();
    return SdpaBwdAdapter::buildSpec(attr, graph.getTensorMap());
}

/// Host-only base: needs the container so the bridge + interpreter are
/// up, but does NOT need a GPU. Exercises the structural applicability
/// gate, the workspace formula, and the bridge arch predicate -- all of
/// which are pure data predicates. The plan builder is constructed
/// directly against the container's bridge with a test-owned cache so
/// size() assertions are deterministic.
class SdpaBwdPlanBuilderHost : public ::testing::Test {
   protected:
    void SetUp() override {
        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        _cache = std::make_unique<ck_dsl_provider::JitCache>();
        _planBuilder =
            std::make_unique<SdpaBwdPlanBuilder>(_container->compileServiceBridge(), *_cache);
    }

    SdpaBwdPlanBuilder& builder() {
        return *_planBuilder;
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<ck_dsl_provider::JitCache> _cache;
    std::unique_ptr<SdpaBwdPlanBuilder> _planBuilder;
};

TEST_F(SdpaBwdPlanBuilderHost, IsApplicableFalseForFwdSdpaGraph) {
    // A forward-SDPA graph's single node is an SdpaAttributes node, not
    // SdpaBackwardAttributes; the bwd builder declines on the structural
    // gate (no device needed).
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_FALSE(builder().isApplicable(*_handle, graph));
}

TEST_F(SdpaBwdPlanBuilderHost, IsApplicableFalseForConvGraph) {
    // A conv-fwd graph has no SDPA-backward node; the bwd builder
    // declines on the structural gate.
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        /*xDims=*/{8, 64, 56, 56}, /*xStrides=*/{64 * 56 * 56, 1, 56 * 64, 64},
        /*wDims=*/{64, 64, 3, 3}, /*wStrides=*/{64 * 3 * 3, 1, 3 * 64, 64},
        /*yDims=*/{8, 64, 56, 56}, /*yStrides=*/{64 * 56 * 56, 1, 56 * 64, 64},
        /*convPrePadding=*/{1, 1}, /*convPostPadding=*/{1, 1}, /*convStrides=*/{1, 1},
        /*convDilation=*/{1, 1}, /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_FALSE(builder().isApplicable(*_handle, graph));
}

TEST_F(SdpaBwdPlanBuilderHost, IsApplicableFalseForHalfGradients) {
    // The SDK helper builds dQ/dK/dV as HALF, which the adapter rejects;
    // isApplicable downgrades the throw to a structural decline. No
    // device needed.
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/bshdStrides(8, 16, 64),
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/bshdStrides(8, 16, 64),
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/bshdStrides(8, 16, 64),
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/bshdStrides(8, 16, 64),
        /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_FALSE(builder().isApplicable(*_handle, graph));
}

TEST_F(SdpaBwdPlanBuilderHost, GetMaxWorkspaceSizeMatchesFormula) {
    // getMaxWorkspaceSize only builds the spec (a pure adapter walk) to
    // size the M_saved + L_saved scratch: 2 * B * Sq * Hq * sizeof(float).
    // No device, no interpreter -- host-only.
    constexpr int kB = 2, kHq = 8, kSq = 16;
    auto fbBuilder = makeValidSdpaBwdGraph(kB, kHq, /*Hkv=*/8, kSq, /*Skv=*/16, /*D=*/64);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    ck_dsl_provider::CkDslSettings settings;
    const std::size_t expected = static_cast<std::size_t>(2) * kB * kSq * kHq * sizeof(float);
    EXPECT_EQ(builder().getMaxWorkspaceSize(*_handle, graph, settings), expected);
}

// The arch gate isApplicable consults runs through the bridge into the
// DSL's is_valid_spec -- a pure data predicate (no GPU, no comgr), so
// this is host-only. It proves the target arch is threaded into the
// applicability decision and that the gate keys off the real arch: the
// bwd FMHA path is a gfx950 vertical, so a valid spec is applicable on
// gfx950, inapplicable on gfx942 (forward-only today), and inapplicable
// on an unknown arch.
TEST_F(SdpaBwdPlanBuilderHost, BridgeIsApplicableIsArchAware) {
    SdpaBwdSpec spec = makeValidSpec();

    auto& bridge = _container->compileServiceBridge();
    const char* op = SdpaBwdPlanBuilder::opKind();

    py::gil_scoped_acquire gil;
    py::dict payload = ck_dsl_provider::sdpaBwdSpecToPayload(spec);
    auto verdictForArch = [&](const char* arch) { return bridge.isApplicable(op, payload, arch); };

    auto onGfx950 = verdictForArch("gfx950");
    EXPECT_TRUE(onGfx950.first) << "expected applicable on gfx950; reason: " << onGfx950.second;
    // The FMHA backward kernel is valid on any CDNA wave64 arch with the
    // f16 MFMA atom (head_size % wave_size == 0), which includes gfx942 --
    // its applicability is decided by the DSL's is_valid_spec, not pinned
    // to gfx950.
    auto onGfx942 = verdictForArch("gfx942");
    EXPECT_TRUE(onGfx942.first) << "expected applicable on gfx942; reason: " << onGfx942.second;
    EXPECT_FALSE(verdictForArch("gfx777").first) << "unknown arch must be inapplicable";
}

/// GPU-gated: buildPlan triggers the JitCache loader twice (once for the
/// bwd kernel, once for the LSE-prep kernel), each compiling a real
/// kernel via the bridge + Python compile service. Requires a gfx950
/// device.
class SdpaBwdPlanBuilderGpu : public SdpaBwdPlanBuilderHost {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("SdpaBwdPlanBuilderGpu");
        SdpaBwdPlanBuilderHost::SetUp();
    }
};

TEST_F(SdpaBwdPlanBuilderGpu, BuildPlanCachesOnSecondCall) {
    auto fbBuilder = makeValidSdpaBwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);

    auto& planBuilder = builder();

    // First call: cold cache, compiles BOTH kernels (bwd + LSE-prep), so
    // the cache gains two entries.
    EXPECT_EQ(planBuilder.cacheForTesting().size(), 0u);
    CkDslContext ctx1;
    planBuilder.buildPlan(*_handle, graph, engineConfig, ctx1);

    ASSERT_TRUE(ctx1.hasValidPlan());
    auto* concretePlan1 = dynamic_cast<SdpaBwdPlan*>(&ctx1.plan());
    ASSERT_NE(concretePlan1, nullptr) << "plan must be a SdpaBwdPlan";
    EXPECT_EQ(planBuilder.cacheForTesting().size(), 2u)
        << "first buildPlan compiles two kernels (bwd + prep)";

    EXPECT_FALSE(concretePlan1->bwdModuleForTesting().kernelName().empty());
    EXPECT_FALSE(concretePlan1->prepModuleForTesting().kernelName().empty());

    // Tensor UIDs from makeValidSdpaBwdGraph: q=1, dq=7, dk=8, dv=9.
    EXPECT_EQ(concretePlan1->qUidForTesting(), 1);
    EXPECT_EQ(concretePlan1->dqUidForTesting(), 7);
    EXPECT_EQ(concretePlan1->dkUidForTesting(), 8);
    EXPECT_EQ(concretePlan1->dvUidForTesting(), 9);

    // Second call: cache hit for both kernels. The cache must not grow,
    // and both plans must reference the SAME HipModule instances.
    CkDslContext ctx2;
    planBuilder.buildPlan(*_handle, graph, engineConfig, ctx2);

    ASSERT_TRUE(ctx2.hasValidPlan());
    auto* concretePlan2 = dynamic_cast<SdpaBwdPlan*>(&ctx2.plan());
    ASSERT_NE(concretePlan2, nullptr);

    EXPECT_EQ(planBuilder.cacheForTesting().size(), 2u) << "cache must not grow on hit";
    EXPECT_EQ(&concretePlan1->bwdModuleForTesting(), &concretePlan2->bwdModuleForTesting());
    EXPECT_EQ(&concretePlan1->prepModuleForTesting(), &concretePlan2->prepModuleForTesting());
}

}  // namespace
