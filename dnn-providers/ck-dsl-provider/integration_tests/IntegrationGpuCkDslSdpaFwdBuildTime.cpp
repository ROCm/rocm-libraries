// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Build-time / JIT-latency probe for the unified SDPA-forward path. Times
// SdpaFwdPlanBuilder::buildPlan = capability gate + dispatcher scoring +
// comgr JIT compile of the gfx950 kernel + HipModule load, separating:
//   * first call          -- one-time process warmup (LightGBM model load
//                            + comgr/embedded-CPython init + first compile)
//   * steady cold compile -- per NEW shape, once the process is warm
//   * JitCache hit         -- repeat shape: module cached by signature, so
//                            only the dispatcher re-scoring runs
//
// This is the first-use latency a deployment pays (the per-launch kernel
// time is measured separately in IntegrationGpuCkDslSdpaFwdPerf). gfx950
// only (buildPlan JIT-compiles for the live device, which the DSL kernel
// supports only on gfx950).

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <iostream>
#include <memory>
#include <vector>

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "engines/sdpa/SdpaFwdPlanBuilder.hpp"
#include "python/CompileServiceBridge.hpp"
#include "tests/TestUtils.hpp"

namespace {

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::CkDslContext;
using ck_dsl_provider::SdpaFwdPlanBuilder;

std::vector<std::int64_t> bshd(int H, int S, int D) {
    return {static_cast<std::int64_t>(S) * H * D, D, static_cast<std::int64_t>(H) * D, 1};
}

// One buildPlan (causal, fp16, BSHD), returning its wall time in ms.
double buildOnceMs(SdpaFwdPlanBuilder& pb, ::CkDslHandle& h, int B, int Hq, int Hkv, int S, int D) {
    const std::vector<std::int64_t> q{B, Hq, S, D};
    const std::vector<std::int64_t> kv{B, Hkv, S, D};
    auto fb = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        q, bshd(Hq, S, D), kv, bshd(Hkv, S, D), kv, bshd(Hkv, S, D), q, bshd(Hq, S, D),
        data_objects::DataType::HALF, false, false, false, false, false, /*causalMask=*/true);
    flatbuffer_utilities::GraphWrapper graph(fb.GetBufferPointer(), fb.GetSize());
    flatbuffer_utilities::EngineConfigWrapper cfg(nullptr, 0);
    CkDslContext ctx;
    const auto t0 = std::chrono::steady_clock::now();
    pb.buildPlan(h, graph, cfg, ctx);
    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_TRUE(ctx.hasValidPlan());
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

class IntegrationGpuCkDslSdpaFwdBuildTimeGpu : public ::testing::Test {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("IntegrationGpuCkDslSdpaFwdBuildTimeGpu");
        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        _planBuilder = std::make_unique<SdpaFwdPlanBuilder>(_container->compileServiceBridge(),
                                                            _container->jitCache());
    }
    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<SdpaFwdPlanBuilder> _planBuilder;
};

TEST_F(IntegrationGpuCkDslSdpaFwdBuildTimeGpu, ColdVsCached) {
    SdpaFwdPlanBuilder& pb = *_planBuilder;
    ::CkDslHandle& h = *_handle;

    // Shape A first call: one-time warmup (model load + comgr + first compile).
    const double aFirst = buildOnceMs(pb, h, 1, 32, 8, 2048, 128);
    const double aCached = buildOnceMs(pb, h, 1, 32, 8, 2048, 128);  // JitCache hit
    // Distinct shapes -> steady-state cold compile (warmup already paid).
    const double bCold = buildOnceMs(pb, h, 1, 32, 8, 4096, 128);
    const double bCached = buildOnceMs(pb, h, 1, 32, 8, 4096, 128);
    const double cCold = buildOnceMs(pb, h, 1, 32, 8, 8192, 128);

    std::cout << "[BuildTime] first_call_incl_warmup_ms=" << aFirst << " (S2048)\n"
              << "[BuildTime] steady_cold_compile_ms: S4096=" << bCold << " S8192=" << cCold << "\n"
              << "[BuildTime] jitcache_hit_ms (re-score only): S2048=" << aCached
              << " S4096=" << bCached << std::endl;
}

}  // namespace
