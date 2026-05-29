// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <memory>
#include <sstream>
#include <vector>

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "engines/conv_implicit_gemm/ConvImplicitGemmPlan.hpp"
#include "engines/conv_implicit_gemm/ConvImplicitGemmPlanBuilder.hpp"
#include "perf/PerfMeasurement.hpp"
#include "python/CompileServiceBridge.hpp"
#include "tests/TestUtils.hpp"

namespace {

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace utilities = hipdnn_data_sdk::utilities;
using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::CkDslContext;
using ck_dsl_provider::ConvImplicitGemmPlan;
using ck_dsl_provider::ConvImplicitGemmPlanBuilder;
using ck_dsl_provider::PerfMeasurement;
using ck_dsl_provider::PerfResult;
using hipdnn_data_sdk::types::half;
using hipdnn_test_sdk::utilities::CpuFpReferenceConvolution;

/// I-10 capstone: end-to-end M1 integration test.
///
/// Per plan §1 the test:
///   1. Builds a single-op conv-fwd graph (bake-off shape: N=8,
///      56x56x64 -> 64, 3x3, stride 1, pad 1, FP16, NHWC).
///   2. Runs it through the JIT pipeline (engine, plan-builder,
///      adapter, bridge, compile_service, JitCache, HipModule).
///   3. Validates the output against CpuFpReferenceConvolution::fprop
///      from the test SDK within tolerance.
///   4. Logs achieved kernel time and TFLOPS via PerfMeasurement.
///
/// **Adaptation from plan §1:** the test bypasses the hipDNN frontend
/// API and the backend's .so-loading plugin path. Both surfaces are
/// architecturally additive on top of what the unit-test suite
/// already proves -- the plan-builder + plan-execute path here is
/// the exact same code the backend would call after dlopen. The
/// frontend-API integration lands as M1.5 (or as part of I-11) once
/// the .so installs cleanly into a hipDNN that can find it.
///
/// Tensor layout convention (PREP_FINDINGS P-6 + miopen-provider
/// precedent): host-side tensors carry logical NCHW dims for X/Y and
/// logical KCRS for W, with physical NHWC strides on top. The DSL
/// kernel reads/writes the same NHWC memory layout; the CPU reference
/// iterates logical dims and resolves via strides, so a direct
/// element-wise compare over the packed NHWC buffers walks the same
/// logical positions in the same order.
class IntegrationGpuCkDslConvFp16Gpu : public ::testing::Test {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("IntegrationGpuCkDslConvFp16Gpu");

        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        _planBuilder = std::make_unique<ConvImplicitGemmPlanBuilder>(
            _container->compileServiceBridge(), _container->jitCache());
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<ConvImplicitGemmPlanBuilder> _planBuilder;
};

TEST_F(IntegrationGpuCkDslConvFp16Gpu, BakeOffConv) {
    // Bake-off shape from plan §4.
    constexpr std::int64_t kN = 8;
    constexpr std::int64_t kC = 64;
    constexpr std::int64_t kHi = 56;
    constexpr std::int64_t kWi = 56;
    constexpr std::int64_t kK = 64;
    constexpr std::int64_t kR = 3;
    constexpr std::int64_t kS = 3;
    // Ho = (Hi + 2*pH - dH*(R-1) - 1)/sH + 1 = (56 + 2 - 2 - 1)/1 + 1 = 56.
    constexpr std::int64_t kHo = 56;
    constexpr std::int64_t kWo = 56;

    // FB graph -- exact same shape as ConvImplicitGemmPlanBuilderTest's
    // ``makeBakeOffConvFwdGraph``. Tensor UIDs from
    // createValidConvFwdGraph: x=1, w=2, y=3.
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        /*xDims=*/{kN, kC, kHi, kWi},
        /*xStrides=*/{kC * kHi * kWi, 1, kWi * kC, kC},
        /*wDims=*/{kK, kC, kR, kS},
        /*wStrides=*/{kC * kR * kS, 1, kS * kC, kC},
        /*yDims=*/{kN, kK, kHo, kWo},
        /*yStrides=*/{kK * kHo * kWo, 1, kWo * kK, kK},
        /*convPrePadding=*/{1, 1},
        /*convPostPadding=*/{1, 1},
        /*convStrides=*/{1, 1},
        /*convDilation=*/{1, 1},
        /*dataType=*/data_objects::DataType::HALF);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    // Host-side tensors. NHWC strides via TensorLayout::NHWC; W gets
    // the same channel-last stride order applied to logical KCRS so
    // physical memory is KRSC. ``Tensor<half>`` owns both host and
    // device storage and syncs on demand via MigratableMemory.
    const utilities::TensorLayout& nhwc = utilities::TensorLayout::NHWC;
    utilities::Tensor<half> tensorX({kN, kC, kHi, kWi}, nhwc);
    utilities::Tensor<half> tensorW({kK, kC, kR, kS},
                                    utilities::generateStrides({kK, kC, kR, kS}, nhwc.strideOrder));
    utilities::Tensor<half> tensorYGpu({kN, kK, kHo, kWo}, nhwc);
    utilities::Tensor<half> tensorYCpu({kN, kK, kHo, kWo}, nhwc);

    // Seed both inputs. Small range so the K_gemm=576 accumulation
    // stays in a numerically friendly part of FP16 (max accumulator
    // value is bounded by |x|*|w|*K = 0.1*0.1*576 = 5.76). Random
    // distributions still exercise every codepath the kernel takes;
    // adjusting the range only reduces the tail accumulator error.
    constexpr unsigned kSeedX = 0x4242u;
    constexpr unsigned kSeedW = 0x5555u;
    tensorX.fillWithRandomValues(half(-0.1f), half(0.1f), kSeedX);
    tensorW.fillWithRandomValues(half(-0.1f), half(0.1f), kSeedW);

    // Build the plan. This compiles the kernel on a cold cache
    // (multi-second the first time per provider session).
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);
    CkDslContext ctx;
    _planBuilder->buildPlan(*_handle, graph, engineConfig, ctx);
    ASSERT_TRUE(ctx.hasValidPlan());

    // Drive H->D copies by reading ``deviceData()``. The output
    // tensor's device buffer is left uninitialised -- the kernel
    // writes every element via NHWC strides.
    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {
        {1, tensorX.memory().deviceData()},
        {2, tensorW.memory().deviceData()},
        {3, tensorYGpu.memory().deviceData()},
    };

    // Single execution for correctness; subsequent launches for
    // perf measurement reuse the same plan + buffers.
    ASSERT_NO_THROW(ctx.plan().execute(*_handle, deviceBuffers.data(),
                                       static_cast<std::uint32_t>(deviceBuffers.size()),
                                       /*workspace=*/nullptr));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // CPU reference. Operates on the host-side TensorBase via the
    // logical NCHW dims + NHWC strides -- iterates {n, c, h, w} and
    // resolves to memory via inner_product(indices, strides) per
    // P-6's stride math. Output Y_cpu is written into a freshly-
    // allocated host tensor with matching strides.
    CpuFpReferenceConvolution::fprop<half, half, half, float>(tensorX, tensorW, tensorYCpu,
                                                              /*strides=*/{1, 1},
                                                              /*dilations=*/{1, 1},
                                                              /*padding=*/{1, 1});

    // Force D->H on the GPU output so subsequent host reads see the
    // kernel's writes. markDeviceModified is what tells the migration
    // layer "device has the canonical version, copy when hostData is
    // requested next."
    tensorYGpu.memory().markDeviceModified();
    const half* gpuOut = tensorYGpu.memory().hostData();
    const half* cpuOut = tensorYCpu.memory().hostData();

    // Tolerance bound (per plan §1 + PREP_FINDINGS): expected error
    // for K_gemm=576 random-uniform fp16 accumulations is roughly
    // sqrt(K_gemm) * fp16_eps * |max_input * max_weight| =
    // 24 * 1e-3 * 0.01 = 2.4e-4 typical. We use a generous 5e-2
    // absolute tolerance to accommodate accumulation tail behaviour
    // without making the test brittle to minor codegen reshufflings.
    constexpr float kAbsTol = 5.0e-2f;
    std::size_t mismatches = 0;
    std::size_t firstMismatchIdx = 0;
    float worstError = 0.0f;
    float worstGpu = 0.0f;
    float worstCpu = 0.0f;

    // ``memory().count()`` is the element span -- packed NHWC is
    // exactly N*K*Ho*Wo elements with no gap. Both tensors share the
    // same layout so a linear walk visits matching logical positions
    // in matching order.
    const std::size_t numElements = tensorYGpu.memory().count();
    for (std::size_t i = 0; i < numElements; ++i) {
        const float gpu = static_cast<float>(gpuOut[i]);
        const float cpu = static_cast<float>(cpuOut[i]);
        const float diff = std::fabs(gpu - cpu);
        if (diff > worstError) {
            worstError = diff;
            worstGpu = gpu;
            worstCpu = cpu;
        }
        if (diff > kAbsTol) {
            if (mismatches == 0) {
                firstMismatchIdx = i;
            }
            ++mismatches;
        }
    }

    EXPECT_EQ(mismatches, 0u) << "found " << mismatches << " elements outside the " << kAbsTol
                              << " tolerance ("
                              << static_cast<double>(mismatches) /
                                     static_cast<double>(numElements) * 100.0
                              << "%); first mismatch at linear index " << firstMismatchIdx
                              << "; worst diff " << worstError << " (gpu=" << worstGpu
                              << ", cpu=" << worstCpu << ")";

    // Perf measurement (no perf-target assertion, log only per Q9).
    // FLOPS formula from plan §4: 2 * N * Ho * Wo * K * C * R * S.
    constexpr double kFlops = 2.0 * static_cast<double>(kN) * static_cast<double>(kHo) *
                              static_cast<double>(kWo) * static_cast<double>(kK) *
                              static_cast<double>(kC) * static_cast<double>(kR) *
                              static_cast<double>(kS);
    PerfMeasurement pm;
    auto launchFn = [&]() {
        ctx.plan().execute(*_handle, deviceBuffers.data(),
                           static_cast<std::uint32_t>(deviceBuffers.size()),
                           /*workspace=*/nullptr);
    };
    PerfResult result = pm.measure(launchFn, kFlops, _handle->getStream());

    // The perf line goes through the plugin logger so the test
    // harness's recorder captures it. Also stamp the worst element
    // diff in the result message so a passing test still leaves a
    // breadcrumb of the numerical agreement quality.
    pm.log("conv_implicit_gemm_bake_off_N8H56W56C64_K64R3S3", result);

    std::ostringstream summary;
    summary << "IntegrationGpuCkDslConvFp16Gpu.BakeOffConv: numerical agreement "
            << "(worst abs diff = " << worstError << " < tol = " << kAbsTol
            << "), perf min_us = " << result.minUs << ", median_us = " << result.medianUs
            << ", tflops = " << result.tflops;
    // testing::Test::RecordProperty surfaces in --gtest_output=xml; for
    // ad-hoc console runs the message is appended to the test output
    // on PASS too via SCOPED_TRACE-style note.
    RecordProperty("ck_dsl_perf_summary", summary.str());
}

}  // namespace
