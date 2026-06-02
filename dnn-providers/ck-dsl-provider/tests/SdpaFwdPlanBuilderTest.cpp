// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <pybind11/embed.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <memory>
#include <string>

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "TestUtils.hpp"
#include "adapters/sdpa/SdpaAdapter.hpp"
#include "adapters/sdpa/SdpaPayload.hpp"
#include "adapters/sdpa/SdpaSpec.hpp"
#include "engines/sdpa/SdpaFwdPlan.hpp"
#include "engines/sdpa/SdpaFwdPlanBuilder.hpp"
#include "python/CompileServiceBridge.hpp"
#include "runtime/JitCache.hpp"

namespace {

namespace py = pybind11;

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::CkDslContext;
using ck_dsl_provider::SdpaAdapter;
using ck_dsl_provider::SdpaFwdPlan;
using ck_dsl_provider::SdpaFwdPlanBuilder;
using ck_dsl_provider::SdpaSpec;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using DataType = data_objects::DataType;
using SdpaAttributes = data_objects::SdpaAttributes;

/// BSHD physical strides for a logical [B, H, S, D] tensor. The kernel
/// requires this layout: batch = S*H*D (== seqlen*token), head = D,
/// token = H*D, d = 1. The adapter rejects anything else, so the
/// "valid" graph must be BSHD-strided.
std::vector<std::int64_t> bshdStrides(int H, int S, int D) {
    return {static_cast<std::int64_t>(S) * H * D, D, static_cast<std::int64_t>(H) * D, 1};
}

/// Default valid SDPA-fwd graph (B=2, Hq=Hkv=8, Sq=Skv=16, D=64, FP16,
/// no mask) with BSHD strides {8192, 64, 512, 1} -- the same GraphWrapper
/// the SDK hands the plan builder at runtime.
flatbuffers::FlatBufferBuilder makeValidSdpaFwdGraph() {
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    // causalMask=true: the unified paged kernel applies causal masking
    // unconditionally, so the capability gate (Phase 2d) declines a
    // non-causal graph. A "valid" SDPA-fwd graph for this provider must
    // therefore request top-left causal masking.
    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF, /*withAttnMask=*/false, /*withScale=*/false,
        /*withStats=*/false, /*alibiMask=*/false, /*paddingMask=*/false, /*causalMask=*/true);
}

/// A valid spec built straight off the default graph; the arch-gate
/// tests only need the spec (not the FlatBuffer) to drive the bridge
/// validator.
SdpaSpec makeValidSpec() {
    auto fbBuilder = makeValidSdpaFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    const auto& attr = graph.getNodeWrapper(0).attributesAs<SdpaAttributes>();
    return SdpaAdapter::buildSpec(attr, graph.getTensorMap());
}

/// Host-only base: needs the container so the bridge + interpreter are
/// up, but does NOT need a GPU. Exercises isApplicable() only. The plan
/// builder is constructed directly against the container's bridge with a
/// test-owned cache so size() assertions are deterministic.
class SdpaFwdPlanBuilderHost : public ::testing::Test {
   protected:
    void SetUp() override {
        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        _cache = std::make_unique<ck_dsl_provider::JitCache>();
        _planBuilder =
            std::make_unique<SdpaFwdPlanBuilder>(_container->compileServiceBridge(), *_cache);
    }

    SdpaFwdPlanBuilder& builder() {
        return *_planBuilder;
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<ck_dsl_provider::JitCache> _cache;
    std::unique_ptr<SdpaFwdPlanBuilder> _planBuilder;
};

TEST_F(SdpaFwdPlanBuilderHost, IsApplicableFalseForConvGraph) {
    // A conv-fwd graph has no SDPA node; the SDPA builder declines.
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        /*xDims=*/{8, 64, 56, 56}, /*xStrides=*/{64 * 56 * 56, 1, 56 * 64, 64},
        /*wDims=*/{64, 64, 3, 3}, /*wStrides=*/{64 * 3 * 3, 1, 3 * 64, 64},
        /*yDims=*/{8, 64, 56, 56}, /*yStrides=*/{64 * 56 * 56, 1, 56 * 64, 64},
        /*convPrePadding=*/{1, 1}, /*convPostPadding=*/{1, 1}, /*convStrides=*/{1, 1},
        /*convDilation=*/{1, 1}, /*dataType=*/DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_FALSE(builder().isApplicable(*_handle, graph));
}

TEST_F(SdpaFwdPlanBuilderHost, IsApplicableFalseForBf16Sdpa) {
    // BSHD strides so the only disqualifier is the dtype.
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::BFLOAT16);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_FALSE(builder().isApplicable(*_handle, graph));
}

TEST_F(SdpaFwdPlanBuilderHost, IsApplicableFalseForUnsupportedFeature) {
    // Additive attn_mask is rejected by the adapter, so isApplicable
    // declines (the throw is caught + downgraded to false). BSHD strides
    // so the only disqualifier is the unsupported feature.
    const auto qkvoStrides = bshdStrides(/*H=*/8, /*S=*/16, /*D=*/64);
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        /*qDims=*/{2, 8, 16, 64}, /*qStrides=*/qkvoStrides,
        /*kDims=*/{2, 8, 16, 64}, /*kStrides=*/qkvoStrides,
        /*vDims=*/{2, 8, 16, 64}, /*vStrides=*/qkvoStrides,
        /*oDims=*/{2, 8, 16, 64}, /*oStrides=*/qkvoStrides,
        /*dataType=*/DataType::HALF, /*withAttnMask=*/true);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_FALSE(builder().isApplicable(*_handle, graph));
}

TEST_F(SdpaFwdPlanBuilderHost, GetMaxWorkspaceSizeIsZero) {
    auto fbBuilder = makeValidSdpaFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    ck_dsl_provider::CkDslSettings settings;
    EXPECT_EQ(builder().getMaxWorkspaceSize(*_handle, graph, settings), 0u);
}

// The arch gate isApplicable consults runs through the bridge into the
// DSL's is_valid_spec -- a pure data predicate (no GPU, no comgr), so
// this is host-only. It proves the target arch is threaded into the
// applicability decision and that the gate keys off the real arch rather
// than a hardcoded one: the M1 FMHA-forward path is a gfx950 vertical,
// so a valid spec is applicable on gfx950 and inapplicable on an unknown
// arch.
TEST_F(SdpaFwdPlanBuilderHost, BridgeIsApplicableIsArchAware) {
    SdpaSpec spec = makeValidSpec();

    auto& bridge = _container->compileServiceBridge();
    const char* op = SdpaFwdPlanBuilder::opKind();

    py::gil_scoped_acquire gil;
    py::dict payload = ck_dsl_provider::sdpaSpecToPayload(spec);
    auto verdictForArch = [&](const char* arch) { return bridge.isApplicable(op, payload, arch); };

    auto onGfx950 = verdictForArch("gfx950");
    EXPECT_TRUE(onGfx950.first) << "expected applicable on gfx950; reason: " << onGfx950.second;
    EXPECT_FALSE(verdictForArch("gfx777").first) << "unknown arch must be inapplicable";
}

/// GPU-gated: buildPlan triggers the JitCache loader, which compiles the
/// FMHA-forward kernel via the bridge + Python compile service. The
/// second buildPlan on the same graph must hit the cache.
class SdpaFwdPlanBuilderGpu : public SdpaFwdPlanBuilderHost {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("SdpaFwdPlanBuilderGpu");
        SdpaFwdPlanBuilderHost::SetUp();
    }
};

TEST_F(SdpaFwdPlanBuilderGpu, BuildPlanCachesOnSecondCall) {
    auto fbBuilder = makeValidSdpaFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);

    auto& planBuilder = builder();

    // First call: cache miss, compiles the real FMHA kernel.
    EXPECT_EQ(planBuilder.cacheForTesting().size(), 0u);
    CkDslContext ctx1;
    planBuilder.buildPlan(*_handle, graph, engineConfig, ctx1);

    ASSERT_TRUE(ctx1.hasValidPlan());
    auto* concretePlan1 = dynamic_cast<SdpaFwdPlan*>(&ctx1.plan());
    ASSERT_NE(concretePlan1, nullptr) << "plan must be a SdpaFwdPlan";
    EXPECT_EQ(planBuilder.cacheForTesting().size(), 1u);

    EXPECT_FALSE(concretePlan1->moduleForTesting().kernelName().empty());

    // Tensor UIDs from createValidSdpaFwdGraph: q=1, k=2, v=3, o=4.
    EXPECT_EQ(concretePlan1->qUidForTesting(), 1);
    EXPECT_EQ(concretePlan1->oUidForTesting(), 4);

    // Second call: cache hit. Cache must not grow, and both plans must
    // reference the SAME HipModule (the cache returns the same
    // shared_ptr on hit).
    CkDslContext ctx2;
    planBuilder.buildPlan(*_handle, graph, engineConfig, ctx2);

    ASSERT_TRUE(ctx2.hasValidPlan());
    auto* concretePlan2 = dynamic_cast<SdpaFwdPlan*>(&ctx2.plan());
    ASSERT_NE(concretePlan2, nullptr);

    EXPECT_EQ(planBuilder.cacheForTesting().size(), 1u) << "cache must not grow on hit";
    EXPECT_EQ(&concretePlan1->moduleForTesting(), &concretePlan2->moduleForTesting());
}

}  // namespace
