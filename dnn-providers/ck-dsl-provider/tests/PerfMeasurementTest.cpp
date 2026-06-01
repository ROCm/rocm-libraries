// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <memory>
#include <vector>

#include "CkDslContainer.hpp"
#include "TestUtils.hpp"
#include "perf/PerfMeasurement.hpp"
#include "python/CompileServiceBridge.hpp"
#include "runtime/HipModule.hpp"
#include "runtime/KernelArtifact.hpp"
#include "runtime/LaunchAbi.hpp"

namespace {

using ck_dsl_provider::ArgValue;
using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::computePerfStats;
using ck_dsl_provider::HipModule;
using ck_dsl_provider::KernelArtifact;
using ck_dsl_provider::LaunchAbi;
using ck_dsl_provider::PerfMeasurement;
using ck_dsl_provider::PerfResult;
using ck_dsl_provider::PerfStats;

// Host-only: stats math doesn't touch HIP.
TEST(TestPerfMeasurement, ComputeStatsOddCount) {
    // Five samples; median is the middle element after partial sort.
    PerfStats s = computePerfStats({30.0, 10.0, 50.0, 20.0, 40.0});
    EXPECT_DOUBLE_EQ(s.minUs, 10.0);
    EXPECT_DOUBLE_EQ(s.medianUs, 30.0);
}

TEST(TestPerfMeasurement, ComputeStatsEvenCount) {
    // Six samples; median is the mean of the two central elements.
    // Sorted: {5, 10, 15, 25, 30, 40} -> median = (15+25)/2 = 20.
    PerfStats s = computePerfStats({25.0, 5.0, 30.0, 10.0, 40.0, 15.0});
    EXPECT_DOUBLE_EQ(s.minUs, 5.0);
    EXPECT_DOUBLE_EQ(s.medianUs, 20.0);
}

TEST(TestPerfMeasurement, ComputeStatsSingleSample) {
    PerfStats s = computePerfStats({42.5});
    EXPECT_DOUBLE_EQ(s.minUs, 42.5);
    EXPECT_DOUBLE_EQ(s.medianUs, 42.5);
}

TEST(TestPerfMeasurement, ComputeStatsRejectsEmpty) {
    EXPECT_THROW(computePerfStats({}), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestPerfMeasurement, RejectsZeroTimedIters) {
    EXPECT_THROW(PerfMeasurement(/*warmup=*/0, /*timed=*/0),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestPerfMeasurement, DefaultsMatchP7) {
    PerfMeasurement pm;
    EXPECT_EQ(pm.warmupIters(), 5u);
    EXPECT_EQ(pm.timedIters(), 50u);
}

/// GPU-gated: run the helper against the trivial elementwise-copy
/// smoke kernel (no MFMA, single block, single wave). Asserts the
/// timing fields are populated coherently rather than tying to a
/// specific number -- per P-7 the helper is logging-only, no
/// perf-target assertions.
class PerfMeasurementGpu : public ::testing::Test {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("PerfMeasurementGpu");

        _container = std::make_unique<CkDslContainer>();
        _artifact =
            std::make_unique<KernelArtifact>(_container->compileServiceBridge().compileSmoke());
        _module = std::make_unique<HipModule>(*_artifact);

        ASSERT_EQ(hipMalloc(&_dA, sizeof(std::uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&_dC, sizeof(std::uint16_t)), hipSuccess);
        std::uint16_t one = 0x3C00;  // fp16 1.0
        std::uint16_t zero = 0;
        ASSERT_EQ(hipMemcpy(_dA, &one, sizeof(one), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(_dC, &zero, sizeof(zero), hipMemcpyHostToDevice), hipSuccess);

        _packed = LaunchAbi::pack(_module->argSchema(), {ArgValue::pointer(_dA),
                                                         ArgValue::pointer(_dC), ArgValue::i32(1)});
    }

    void TearDown() override {
        if (_dA != nullptr) {
            EXPECT_EQ(hipFree(_dA), hipSuccess);
        }
        if (_dC != nullptr) {
            EXPECT_EQ(hipFree(_dC), hipSuccess);
        }
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<KernelArtifact> _artifact;
    std::unique_ptr<HipModule> _module;
    std::uint16_t* _dA{nullptr};
    std::uint16_t* _dC{nullptr};
    std::vector<std::byte> _packed;
};

TEST_F(PerfMeasurementGpu, MeasuresSmokeKernel) {
    // Tight iter counts so the test runs in <500 ms even when the
    // device is heavily loaded; the helper is exercised end-to-end
    // regardless.
    PerfMeasurement pm(/*warmup=*/2, /*timed=*/7);

    std::atomic<int> launchCount{0};
    auto launchFn = [&]() {
        launchCount.fetch_add(1, std::memory_order_relaxed);
        _module->launch(*_artifact, _packed, /*stream=*/nullptr);
    };

    PerfResult result = pm.measure(launchFn, /*flops=*/0.0, /*stream=*/nullptr);

    EXPECT_EQ(result.warmupIters, 2u);
    EXPECT_EQ(result.timedIters, 7u);
    EXPECT_EQ(launchCount.load(), 2 + 7);

    // Timing sanity: positive, min <= median, both well under 100 ms
    // for a 1-element fp16 copy kernel even on a contended device.
    EXPECT_GT(result.minUs, 0.0);
    EXPECT_GE(result.medianUs, result.minUs);
    EXPECT_LT(result.medianUs, 100000.0);

    // flops=0 -> tflops==0 (no division by zero, no spurious value).
    EXPECT_DOUBLE_EQ(result.tflops, 0.0);

    // Log goes through HIPDNN_PLUGIN_LOG_INFO -- main.cpp routes that
    // through the test-sdk log recorder so it doesn't pollute stdout
    // but is exercised for the format-string regressions a TEST_THAT
    // formatter check would otherwise miss.
    pm.log("perf_measurement_smoke", result);
}

TEST_F(PerfMeasurementGpu, ComputesTflopsWhenFlopsProvided) {
    PerfMeasurement pm(/*warmup=*/1, /*timed=*/3);
    auto launchFn = [&]() { _module->launch(*_artifact, _packed, /*stream=*/nullptr); };

    // Use a large arbitrary flops value so the resulting tflops is
    // numerically distinguishable from zero even at sub-ms launch
    // times. We don't assert a specific value -- it depends on the
    // device and the GPU's current state -- just that the formula
    // wired through correctly.
    constexpr double kArbitraryFlops = 1.0e9;
    PerfResult result = pm.measure(launchFn, kArbitraryFlops, /*stream=*/nullptr);

    EXPECT_GT(result.tflops, 0.0);

    // Cross-check: tflops == kArbitraryFlops / (medianUs * 1e-6) / 1e12.
    const double expected = kArbitraryFlops / (result.medianUs * 1.0e-6) / 1.0e12;
    EXPECT_NEAR(result.tflops, expected, 1.0e-9);
}

}  // namespace
