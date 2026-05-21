// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <memory>

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "engines/conv_implicit_gemm/CkDslConvImplicitGemmEngine.hpp"
#include "engines/conv_implicit_gemm/ConvImplicitGemmPlan.hpp"
#include "engines/conv_implicit_gemm/ConvImplicitGemmPlanBuilder.hpp"
#include "python/CompileServiceBridge.hpp"

namespace {

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::CkDslContext;
using ck_dsl_provider::ConvImplicitGemmPlan;
using ck_dsl_provider::ConvImplicitGemmPlanBuilder;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Build the bake-off conv-fwd graph via the test SDK helper, plus
/// the handful of supporting tensors with HALF dtype and NHWC strides.
/// The plan-builder consumes the same GraphWrapper the SDK produces
/// at runtime so we exercise the real IGraph traversal path.
flatbuffers::FlatBufferBuilder makeBakeOffConvFwdGraph() {
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
        _planBuilder =
            std::make_unique<ConvImplicitGemmPlanBuilder>(_container->compileServiceBridge());
    }

    ConvImplicitGemmPlanBuilder& builder() {
        return *_planBuilder;
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<ConvImplicitGemmPlanBuilder> _planBuilder;
};

TEST_F(ConvImplicitGemmPlanBuilderHost, IsApplicableReturnsTrueForSupportedGraph) {
    auto fbBuilder = makeBakeOffConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_TRUE(builder().isApplicable(*_handle, graph));
}

TEST_F(ConvImplicitGemmPlanBuilderHost, IsApplicableReturnsFalseForFloatDtype) {
    auto fbBuilder = makeUnsupportedConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_FALSE(builder().isApplicable(*_handle, graph));
}

TEST_F(ConvImplicitGemmPlanBuilderHost, GetMaxWorkspaceSizeIsZero) {
    auto fbBuilder = makeBakeOffConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    ck_dsl_provider::CkDslSettings settings;
    EXPECT_EQ(builder().getMaxWorkspaceSize(*_handle, graph, settings), 0u);
}

TEST_F(ConvImplicitGemmPlanBuilderHost, GetCustomKnobsIsEmpty) {
    auto fbBuilder = makeBakeOffConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    EXPECT_TRUE(builder().getCustomKnobs(*_handle, graph).empty());
}

/// GPU-gated: buildPlan triggers the JitCache loader, which compiles
/// the implicit-GEMM conv kernel via the bridge + Python compile
/// service. Second buildPlan on the same graph must hit the cache.
class ConvImplicitGemmPlanBuilderGpu : public ConvImplicitGemmPlanBuilderHost {
   protected:
    void SetUp() override {
        int deviceCount = 0;
        hipError_t err = hipGetDeviceCount(&deviceCount);
        if (err != hipSuccess || deviceCount == 0) {
            GTEST_SKIP() << "no HIP-visible device (deviceCount=" << deviceCount
                         << ", hipError=" << static_cast<int>(err) << ")";
        }
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ConvImplicitGemmPlanBuilderHost::SetUp();
    }
};

TEST_F(ConvImplicitGemmPlanBuilderGpu, BuildPlanCachesOnSecondCall) {
    auto fbBuilder = makeBakeOffConvFwdGraph();
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

    // Confirm the loaded kernel matches the bake-off naming convention
    // emitted by build_implicit_gemm_conv (see PREP_FINDINGS P-5).
    auto kernelName = concretePlan1->moduleForTesting().kernelName();
    EXPECT_NE(kernelName.find("ck_dsl_conv_igemm"), std::string::npos)
        << "unexpected kernel name: " << kernelName;
    EXPECT_NE(kernelName.find("N8H56W56C64"), std::string::npos)
        << "kernel name missing bake-off shape token: " << kernelName;

    // Tensor UIDs from createValidConvFwdGraph: x=1, w=2, y=3.
    EXPECT_EQ(concretePlan1->xUidForTesting(), 1);
    EXPECT_EQ(concretePlan1->yUidForTesting(), 3);

    // Launch metadata cross-check against plan §4:
    //   grid = (num_pid_n, num_pid_m, 1) = (ceil(64/64), ceil(8*56*56/64), 1) = (1, 392, 1)
    //   block = (warp_m * warp_n * wave_size, 1, 1) = (256, 1, 1)
    EXPECT_EQ(concretePlan1->moduleForTesting().grid().x, 1u);
    EXPECT_EQ(concretePlan1->moduleForTesting().grid().y, 392u);
    EXPECT_EQ(concretePlan1->moduleForTesting().grid().z, 1u);
    EXPECT_EQ(concretePlan1->moduleForTesting().block().x, 256u);
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

TEST_F(ConvImplicitGemmPlanBuilderGpu, PlanExecuteIsStub) {
    // Until I-8, calling execute() on the plan should fail loudly so
    // anyone hooking up the integration path before I-8 lands sees a
    // clear "not implemented" instead of an opaque crash.
    auto fbBuilder = makeBakeOffConvFwdGraph();
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);

    CkDslContext ctx;
    builder().buildPlan(*_handle, graph, engineConfig, ctx);
    ASSERT_TRUE(ctx.hasValidPlan());

    EXPECT_THROW(ctx.plan().execute(*_handle, /*deviceBuffers=*/nullptr, /*numDeviceBuffers=*/0,
                                    /*workspace=*/nullptr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

}  // namespace
