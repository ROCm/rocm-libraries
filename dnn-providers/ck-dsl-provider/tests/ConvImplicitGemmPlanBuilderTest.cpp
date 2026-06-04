// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <memory>
#include <optional>
#include <string>

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "TestUtils.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmAdapter.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmPayload.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmSpec.hpp"
#include "engines/conv_implicit_gemm/CkDslConvImplicitGemmEngine.hpp"
#include "engines/conv_implicit_gemm/ConvImplicitGemmPlan.hpp"
#include "engines/conv_implicit_gemm/ConvImplicitGemmPlanBuilder.hpp"
#include "python/CompileServiceBridge.hpp"
#include "runtime/DeviceArch.hpp"
#include "runtime/HipModule.hpp"
#include "runtime/JitCache.hpp"
#include "runtime/KernelArtifact.hpp"

namespace {

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::CkDslContext;
using ck_dsl_provider::ConvImplicitGemmPlan;
using ck_dsl_provider::ConvImplicitGemmPlanBuilder;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Build the example conv-fwd graph via the test SDK helper, plus
/// the handful of supporting tensors with HALF dtype and NHWC strides.
/// The plan-builder consumes the same GraphWrapper the SDK produces
/// at runtime so we exercise the real IGraph traversal path.
flatbuffers::FlatBufferBuilder makeExampleConvFwdGraph() {
    return hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        /*xDims=*/{8, 64, 56, 56},
        /*xStrides=*/{64 * 56 * 56, 1, 56 * 64, 64},
        /*wDims=*/{64, 64, 3, 3},
        /*wStrides=*/{64 * 3 * 3, 1, 3 * 64, 64},
        /*yDims=*/{8, 64, 56, 56},
        /*yStrides=*/{64 * 56 * 56, 1, 56 * 64, 64},
        /*convPrePadding=*/{1, 1},
        /*convPostPadding=*/{1, 1},
        /*convStrides=*/{1, 1},
        /*convDilation=*/{1, 1},
        /*dataType=*/data_objects::DataType::HALF);
}

/// Hand-built spec matching the example graph shape (N8 C64 H56 W56,
/// K64 R3 S3, stride/pad/dilation 1). Codegen knobs keep their example
/// constexpr defaults -- the same spec ``buildSpec`` produces from the
/// example graph. Used by the arch-gate tests, which only need the
/// spec (not the FlatBuffer graph) to drive the bridge validator.
ck_dsl_provider::ConvImplicitGemmSpec makeExampleSpec() {
    ck_dsl_provider::ConvImplicitGemmSpec spec;
    ck_dsl_provider::ConvProblem& p = spec.problem;
    p.N = 8;
    p.Hi = 56;
    p.Wi = 56;
    p.C = 64;
    p.K = 64;
    p.R = 3;
    p.S = 3;
    p.sH = 1;
    p.sW = 1;
    p.pH = 1;
    p.pW = 1;
    p.dH = 1;
    p.dW = 1;
    return spec;
}

/// The example shape with the **cross-arch example config**: the
/// 16x16x16 atom (an MFMA op on gfx942/gfx950, a WMMA op on gfx1151)
/// with the ``mem`` pipeline and ``default`` epilogue -- the one config
/// the DSL validates on all three M1 targets. ``wave_size`` is the only
/// field that must track the hardware (32 on the wave32 RDNA target
/// gfx1151, 64 on the CDNA targets gfx942/gfx950). This config is a
/// TEST concern: production keeps the gfx950-tuned DSL dataclass default
/// (see ConvImplicitGemmSpec), so the example we exercise across arches
/// lives here rather than leaking into the provider.
ck_dsl_provider::ConvImplicitGemmSpec makeExampleSpecForArch(const std::string& arch) {
    ck_dsl_provider::ConvImplicitGemmSpec spec = makeExampleSpec();
    // Only the atom and wave size differ from the defaults; ``pipeline``
    // (``mem``) and ``epilogue`` (``default``) are already the spec
    // defaults inherited from makeExampleSpec().
    spec.warp_tile_m = 16;
    spec.warp_tile_n = 16;
    spec.warp_tile_k = 16;
    spec.wave_size = (arch == "gfx1151") ? 32 : 64;
    return spec;
}

/// Build an unsupported graph (FLOAT dtype) for the host-only
/// isApplicable=false case. The plan builder's adapter rejects
/// FLOAT, so the SDK can skip this engine and look elsewhere.
flatbuffers::FlatBufferBuilder makeUnsupportedConvFwdGraph() {
    return hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        {8, 64, 56, 56}, {64 * 56 * 56, 1, 56 * 64, 64}, {64, 64, 3, 3},
        {64 * 3 * 3, 1, 3 * 64, 64}, {8, 64, 56, 56}, {64 * 56 * 56, 1, 56 * 64, 64}, {1, 1},
        {1, 1}, {1, 1}, {1, 1}, data_objects::DataType::FLOAT);
}

/// Host-only base: needs the container so the bridge + interpreter
/// are up, but does NOT need a GPU. Exercises isApplicable() only.
/// We construct a plan builder directly against the container's
/// bridge rather than reaching into the registered engine, so the
/// test cleanly owns its cache state across test cases.
class ConvImplicitGemmPlanBuilderHost : public ::testing::Test {
   protected:
    void SetUp() override {
        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        // Test-owned cache so size() assertions are deterministic
        // across test cases; the container's process-static cache
        // would carry state in from earlier runs.
        _cache = std::make_unique<ck_dsl_provider::JitCache>();
        _planBuilder = std::make_unique<ConvImplicitGemmPlanBuilder>(
            _container->compileServiceBridge(), *_cache);
    }

    ConvImplicitGemmPlanBuilder& builder() {
        return *_planBuilder;
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<ck_dsl_provider::JitCache> _cache;
    std::unique_ptr<ConvImplicitGemmPlanBuilder> _planBuilder;
};

TEST_F(ConvImplicitGemmPlanBuilderHost, IsApplicableReflectsDeviceArch) {
    // isApplicable is arch-aware: a structurally-valid FP16 conv is only
    // applicable if the M1 default (example) knobs are valid on the
    // device's arch. Rather than hardcode a per-arch expectation, assert
    // the builder's end-to-end verdict (graph -> adapter -> device-arch
    // detection -> bridge) agrees with the authoritative validator for
    // the detected arch -- proving it keys off the real device.
    auto fbBuilder = makeExampleConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    const bool got = builder().isApplicable(*_handle, graph);

    std::optional<std::string> arch = ck_dsl_provider::detectDeviceArch(_handle->getStream());
    if (!arch.has_value()) {
        // No HIP device visible (host-only CI): the conv kernel cannot
        // run here, so isApplicable declines rather than claiming a graph
        // it could never validate against a real device arch.
        EXPECT_FALSE(got) << "with no visible device, isApplicable should decline";
        return;
    }

    // Mirror what the builder does internally: select the per-arch
    // codegen config for the detected device, then ask the authoritative
    // validator about that exact spec. An arch the adapter has no config
    // for is one the builder declines outright.
    ck_dsl_provider::ConvImplicitGemmSpec spec = makeExampleSpec();
    if (!ck_dsl_provider::ConvImplicitGemmAdapter::applyArchCodegenConfig(spec, *arch)) {
        EXPECT_FALSE(got) << "no codegen config for arch " << *arch
                          << "; isApplicable should decline";
        return;
    }
    const ck_dsl_provider::PayloadDict payload =
        ck_dsl_provider::convImplicitGemmSpecToPayload(spec);
    const bool expected = _container->compileServiceBridge()
                              .isApplicable(ConvImplicitGemmPlanBuilder::opKind(), payload, *arch)
                              .first;

    EXPECT_EQ(got, expected) << "isApplicable must match is_valid_spec for device arch " << *arch;
}

TEST_F(ConvImplicitGemmPlanBuilderHost, IsApplicableReturnsFalseForFloatDtype) {
    auto fbBuilder = makeUnsupportedConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_FALSE(builder().isApplicable(*_handle, graph));
}

TEST_F(ConvImplicitGemmPlanBuilderHost, GetMaxWorkspaceSizeIsZero) {
    auto fbBuilder = makeExampleConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    ck_dsl_provider::CkDslSettings settings;
    EXPECT_EQ(builder().getMaxWorkspaceSize(*_handle, graph, settings), 0u);
}

TEST_F(ConvImplicitGemmPlanBuilderHost, GetCustomKnobsIsEmpty) {
    auto fbBuilder = makeExampleConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_TRUE(builder().getCustomKnobs(*_handle, graph).empty());
}

// The arch gate that isApplicable consults runs through the bridge into
// the DSL's is_valid_spec -- a pure data predicate (no GPU, no comgr),
// so this is host-only. It proves the target arch is threaded into the
// applicability decision and that the gate matches the validator the
// compile path enforces: the example conv spec is valid on gfx950 but
// not on gfx1151 (wave32 rejects the wave64 MFMA path), not on gfx942
// (the 32x32x16 f16 atom is absent there), and not on an unknown arch.
TEST_F(ConvImplicitGemmPlanBuilderHost, BridgeIsApplicableIsArchAware) {
    ck_dsl_provider::ConvImplicitGemmSpec spec = makeExampleSpec();

    auto& bridge = _container->compileServiceBridge();
    const char* op = ConvImplicitGemmPlanBuilder::opKind();

    // arch is a separate argument (an orthogonal compile target, not a
    // spec field) -- the same shape the plan builder uses in production.
    const ck_dsl_provider::PayloadDict payload =
        ck_dsl_provider::convImplicitGemmSpecToPayload(spec);
    auto verdictForArch = [&](const char* arch) { return bridge.isApplicable(op, payload, arch); };

    auto onGfx950 = verdictForArch("gfx950");
    EXPECT_TRUE(onGfx950.first) << "expected applicable on gfx950; reason: " << onGfx950.second;
    EXPECT_FALSE(verdictForArch("gfx1151").first)
        << "example spec should be inapplicable on gfx1151";
    EXPECT_FALSE(verdictForArch("gfx942").first) << "example spec should be inapplicable on gfx942";
    EXPECT_FALSE(verdictForArch("gfx777").first) << "unknown arch must be inapplicable";
}

// M1 multi-arch goal (compile coverage): the example shape compiles for
// every target arch with the cross-arch example config. comgr
// cross-compiles without the matching device, so this runs on any box
// (no GPU needed) and gives real gfx942/gfx950/gfx1151 coverage from a
// single machine. One instantiation per arch (via TEST_P) so each gets
// its own name and pass/fail -- a failure on one arch doesn't mask the
// others.
class ConvImplicitGemmExampleCompile : public ConvImplicitGemmPlanBuilderHost,
                                       public ::testing::WithParamInterface<std::string> {};

TEST_P(ConvImplicitGemmExampleCompile, CompilesExampleShape) {
    const std::string arch = GetParam();
    ck_dsl_provider::ConvImplicitGemmSpec spec = makeExampleSpecForArch(arch);

    auto& bridge = _container->compileServiceBridge();
    const char* op = ConvImplicitGemmPlanBuilder::opKind();

    const ck_dsl_provider::PayloadDict payload =
        ck_dsl_provider::convImplicitGemmSpecToPayload(spec);

    auto verdict = bridge.isApplicable(op, payload, arch);
    EXPECT_TRUE(verdict.first) << arch << " applicability: " << verdict.second;

    ck_dsl_provider::KernelArtifact artifact = bridge.compile(op, payload, arch);
    EXPECT_NE(artifact.isa.find(arch), std::string::npos)
        << "compiled ISA '" << artifact.isa << "' does not target " << arch;
    EXPECT_FALSE(artifact.hsaco.empty()) << arch << ": empty HSACO";
}

INSTANTIATE_TEST_SUITE_P(Arches, ConvImplicitGemmExampleCompile,
                         ::testing::Values("gfx942", "gfx950", "gfx1151"),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             return info.param;  // gfx token is a valid test-name suffix
                         });

// M1 multi-arch goal (execution coverage): run the example shape on
// WHATEVER supported device is present, compiling the cross-arch example
// config for that device's arch. This drives the bridge + HipModule +
// plan directly with a test-owned config (the production buildPlan path
// is covered by ConvImplicitGemmPlanBuilderGpu). Skips cleanly with no
// GPU or on an arch outside the M1 set -- the "tests that skip on
// unsupported platforms" half of the multi-arch coverage. With zero
// input and zero weight the convolution output is zero everywhere.
TEST_F(ConvImplicitGemmPlanBuilderHost, ExecutesExampleShapeOnPresentDevice) {
    std::string arch;
    CK_DSL_PROVIDER_SKIP_IF_UNSUPPORTED_ARCH("ExecutesExampleShapeOnPresentDevice", arch);

    ck_dsl_provider::ConvImplicitGemmSpec spec = makeExampleSpecForArch(arch);

    ck_dsl_provider::KernelArtifact artifact;
    {
        const ck_dsl_provider::PayloadDict payload =
            ck_dsl_provider::convImplicitGemmSpecToPayload(spec);
        artifact = _container->compileServiceBridge().compile(ConvImplicitGemmPlanBuilder::opKind(),
                                                              payload, arch);
    }
    ASSERT_NE(artifact.isa.find(arch), std::string::npos) << "isa=" << artifact.isa;

    auto module = std::make_shared<ck_dsl_provider::HipModule>(artifact);

    // Buffer-rsrc byte sizes from the example geometry (FP16).
    const auto& p = spec.problem;
    constexpr std::int64_t kFp16 = 2;
    const auto xBytes = static_cast<std::int32_t>(std::int64_t(p.N) * p.Hi * p.Wi * p.C * kFp16);
    const auto wBytes = static_cast<std::int32_t>(std::int64_t(p.K) * p.R * p.S * p.C * kFp16);
    const auto yBytes =
        static_cast<std::int32_t>(std::int64_t(p.N) * p.Ho() * p.Wo() * p.K * kFp16);

    ConvImplicitGemmPlan plan(module, /*xUid=*/1, /*wUid=*/2, /*yUid=*/3, xBytes, wBytes, yBytes);

    void* dX = nullptr;
    void* dW = nullptr;
    void* dY = nullptr;
    ASSERT_EQ(hipMalloc(&dX, static_cast<std::size_t>(xBytes)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW, static_cast<std::size_t>(wBytes)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dY, static_cast<std::size_t>(yBytes)), hipSuccess);
    ASSERT_EQ(hipMemset(dX, 0, static_cast<std::size_t>(xBytes)), hipSuccess);
    ASSERT_EQ(hipMemset(dW, 0, static_cast<std::size_t>(wBytes)), hipSuccess);
    ASSERT_EQ(hipMemset(dY, 0xab, static_cast<std::size_t>(yBytes)),
              hipSuccess);  // launch sentinel

    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, dX}, {2, dW}, {3, dY}};
    EXPECT_NO_THROW(plan.execute(*_handle, buffers.data(),
                                 static_cast<std::uint32_t>(buffers.size()),
                                 /*workspace=*/nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::uint16_t firstHalf = 0xffff;
    ASSERT_EQ(hipMemcpy(&firstHalf, dY, sizeof(firstHalf), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(firstHalf, 0u) << "expected zero output for zero input + zero weight on " << arch;

    EXPECT_EQ(hipFree(dX), hipSuccess);
    EXPECT_EQ(hipFree(dW), hipSuccess);
    EXPECT_EQ(hipFree(dY), hipSuccess);
}

/// GPU-gated: buildPlan triggers the JitCache loader, which compiles
/// the implicit-GEMM conv kernel via the bridge + Python compile
/// service. Second buildPlan on the same graph must hit the cache.
///
/// Runs on any DSL-supported device (gfx942/gfx950/gfx1151): buildPlan
/// goes through the production adapter, which now selects a valid
/// per-arch codegen config via applyArchCodegenConfig. The detected
/// arch is captured in ``_arch`` so the launch-metadata assertions can
/// account for the per-arch wave size.
class ConvImplicitGemmPlanBuilderGpu : public ConvImplicitGemmPlanBuilderHost {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_UNSUPPORTED_ARCH("ConvImplicitGemmPlanBuilderGpu", _arch);
        ConvImplicitGemmPlanBuilderHost::SetUp();
    }

    std::string _arch;
};

TEST_F(ConvImplicitGemmPlanBuilderGpu, BuildPlanCachesOnSecondCall) {
    auto fbBuilder = makeExampleConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);  // empty config

    auto& planBuilder = builder();

    // First call: cache miss, compiles the real conv kernel
    // (multi-second on a cold comgr).
    CkDslContext ctx1;
    auto firstStart = std::chrono::steady_clock::now();
    planBuilder.buildPlan(*_handle, graph, engineConfig, ctx1);
    auto firstElapsed = std::chrono::steady_clock::now() - firstStart;

    ASSERT_TRUE(ctx1.hasValidPlan());
    auto* concretePlan1 = dynamic_cast<ConvImplicitGemmPlan*>(&ctx1.plan());
    ASSERT_NE(concretePlan1, nullptr) << "plan must be a ConvImplicitGemmPlan";
    EXPECT_EQ(planBuilder.cacheForTesting().size(), 1u);

    // Confirm the loaded kernel matches the example naming convention
    // emitted by build_implicit_gemm_conv.
    auto kernelName = concretePlan1->moduleForTesting().kernelName();
    EXPECT_NE(kernelName.find("ck_dsl_conv_igemm"), std::string::npos)
        << "unexpected kernel name: " << kernelName;
    EXPECT_NE(kernelName.find("N8H56W56C64"), std::string::npos)
        << "kernel name missing example shape token: " << kernelName;

    // Tensor UIDs from createValidConvFwdGraph: x=1, w=2, y=3.
    EXPECT_EQ(concretePlan1->xUidForTesting(), 1);
    EXPECT_EQ(concretePlan1->yUidForTesting(), 3);

    // Launch metadata cross-check against plan §4. The grid is
    // arch-independent (tiles are 64x64 regardless of the MMA atom):
    //   grid = (num_pid_n, num_pid_m, 1) = (ceil(64/64), ceil(8*56*56/64), 1) = (1, 392, 1)
    // The block size tracks the per-arch wave size the adapter selects:
    //   block.x = warp_m * warp_n * wave_size = 4 * wave_size
    //           = 256 on the wave64 CDNA targets, 128 on wave32 gfx1151.
    const std::uint32_t waveSize = (_arch == "gfx1151") ? 32u : 64u;
    const std::uint32_t expectedBlockX = 4u * waveSize;
    EXPECT_EQ(concretePlan1->moduleForTesting().grid().x, 1u);
    EXPECT_EQ(concretePlan1->moduleForTesting().grid().y, 392u);
    EXPECT_EQ(concretePlan1->moduleForTesting().grid().z, 1u);
    EXPECT_EQ(concretePlan1->moduleForTesting().block().x, expectedBlockX);
    EXPECT_EQ(concretePlan1->moduleForTesting().argSchema().size(), 6u);

    // Second call: cache hit. The cost should drop by orders of
    // magnitude since no compile / no HipModule load runs.
    CkDslContext ctx2;
    auto secondStart = std::chrono::steady_clock::now();
    planBuilder.buildPlan(*_handle, graph, engineConfig, ctx2);
    auto secondElapsed = std::chrono::steady_clock::now() - secondStart;

    ASSERT_TRUE(ctx2.hasValidPlan());
    auto* concretePlan2 = dynamic_cast<ConvImplicitGemmPlan*>(&ctx2.plan());
    ASSERT_NE(concretePlan2, nullptr);

    EXPECT_EQ(planBuilder.cacheForTesting().size(), 1u) << "cache must not grow on hit";

    // Both plans should reference the SAME HipModule -- the cache
    // returns the same shared_ptr on hit. (Plan instances are
    // distinct because each buildPlan call creates a fresh
    // unique_ptr<IPlan> for the new context.)
    EXPECT_EQ(&concretePlan1->moduleForTesting(), &concretePlan2->moduleForTesting());

    auto firstMs = std::chrono::duration_cast<std::chrono::milliseconds>(firstElapsed).count();
    auto secondMs = std::chrono::duration_cast<std::chrono::milliseconds>(secondElapsed).count();
    EXPECT_LT(secondMs, 50) << "cache hit took " << secondMs
                            << " ms; expected <50 ms (first compile took " << firstMs << " ms)";
}

TEST_F(ConvImplicitGemmPlanBuilderGpu, PlanExecuteLaunches) {
    // I-8: plan.execute() now packs args and launches the real conv
    // kernel against device buffers. This test allocates X/W/Y at the
    // example shape, zero-initialises them, runs execute() on the
    // default stream, and synchronises. Output correctness against
    // CpuFpReferenceConvolution is the I-10 integration test; for I-8
    // we only verify the launch path returns hipSuccess.
    auto fbBuilder = makeExampleConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);

    CkDslContext ctx;
    builder().buildPlan(*_handle, graph, engineConfig, ctx);
    ASSERT_TRUE(ctx.hasValidPlan());
    auto* concretePlan = dynamic_cast<ConvImplicitGemmPlan*>(&ctx.plan());
    ASSERT_NE(concretePlan, nullptr);

    // Example shape: X = 8*64*56*56 fp16 = 3.21 MB, W = 64*64*3*3
    // fp16 = 73.7 KB, Y = 8*64*56*56 fp16 = 3.21 MB. The plan-builder
    // computed these same byte counts and embedded them in the plan
    // (xBytesForTesting cross-checks one of them).
    constexpr std::size_t kXBytes = 8 * 64 * 56 * 56 * 2;
    constexpr std::size_t kWBytes = 64 * 64 * 3 * 3 * 2;
    constexpr std::size_t kYBytes = 8 * 64 * 56 * 56 * 2;
    EXPECT_EQ(concretePlan->xBytesForTesting(), static_cast<std::int32_t>(kXBytes));

    void* dX = nullptr;
    void* dW = nullptr;
    void* dY = nullptr;
    ASSERT_EQ(hipMalloc(&dX, kXBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW, kWBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dY, kYBytes), hipSuccess);
    ASSERT_EQ(hipMemset(dX, 0, kXBytes), hipSuccess);
    ASSERT_EQ(hipMemset(dW, 0, kWBytes), hipSuccess);
    ASSERT_EQ(hipMemset(dY, 0xab, kYBytes), hipSuccess);  // sentinel for "did the launch write?"

    // Tensor UIDs from createValidConvFwdGraph: x=1, w=2, y=3.
    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {
        {1, dX},
        {2, dW},
        {3, dY},
    };

    EXPECT_NO_THROW(ctx.plan().execute(*_handle, deviceBuffers.data(),
                                       static_cast<std::uint32_t>(deviceBuffers.size()),
                                       /*workspace=*/nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Spot-check the output was actually written -- the input is zero
    // and the weight is zero, so the convolution output is zero
    // everywhere. The sentinel byte (0xab) must be gone.
    std::uint16_t firstHalf = 0;
    ASSERT_EQ(hipMemcpy(&firstHalf, dY, sizeof(firstHalf), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(firstHalf, 0u) << "expected zero output for zero input + zero weight; got 0x"
                             << std::hex << firstHalf;

    EXPECT_EQ(hipFree(dX), hipSuccess);
    EXPECT_EQ(hipFree(dW), hipSuccess);
    EXPECT_EQ(hipFree(dY), hipSuccess);
}

TEST_F(ConvImplicitGemmPlanBuilderHost, ExecuteRejectsMissingDeviceBuffer) {
    // The uid-lookup throws before any HIP call, but buildPlan still
    // compiles the kernel for the present device first, so this needs a
    // DSL-supported device. The production adapter selects a valid
    // per-arch codegen config (applyArchCodegenConfig), so the path runs
    // on any of gfx942/gfx950/gfx1151.
    std::string arch;
    CK_DSL_PROVIDER_SKIP_IF_UNSUPPORTED_ARCH(
        "ConvImplicitGemmPlanBuilderHost.ExecuteRejectsMissingDeviceBuffer", arch);

    auto fbBuilder = makeExampleConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);

    CkDslContext ctx;
    builder().buildPlan(*_handle, graph, engineConfig, ctx);

    // Only X present in the buffer array. W (uid=2) lookup throws.
    void* dX = nullptr;
    ASSERT_EQ(hipMalloc(&dX, 1), hipSuccess);
    std::vector<hipdnnPluginDeviceBuffer_t> incomplete = {{1, dX}};
    EXPECT_THROW(
        ctx.plan().execute(*_handle, incomplete.data(),
                           static_cast<std::uint32_t>(incomplete.size()), /*workspace=*/nullptr),
        hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_EQ(hipFree(dX), hipSuccess);
}

}  // namespace
