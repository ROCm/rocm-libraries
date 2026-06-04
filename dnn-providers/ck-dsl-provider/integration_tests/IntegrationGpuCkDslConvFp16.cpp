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
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <memory>
#include <sstream>
#include <string>
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

/// One forward-convolution problem the parameterized integration test
/// drives end to end. Spatial fields are per-dimension so non-square
/// kernels, padding, strides, and dilations can each be exercised.
struct ConvCase {
    const char* name;
    std::int64_t n, c, hi, wi;  // input  (logical NCHW)
    std::int64_t k, r, s;       // weight (logical KCRS)
    std::int64_t strideH, strideW;
    std::int64_t padH, padW;  // symmetric (pre == post) per spatial dim
    std::int64_t dilH, dilW;
};

/// Standard forward-conv output extent for one spatial dimension.
std::int64_t convOutputDim(std::int64_t in, std::int64_t pad, std::int64_t dil, std::int64_t k,
                           std::int64_t stride) {
    return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

/// End-to-end M1 integration coverage for the implicit-GEMM conv path.
///
/// Each case:
///   1. Builds a single-op conv-fwd graph for the parameterized shape.
///   2. Runs it through the JIT pipeline (engine, plan-builder,
///      adapter, bridge, compile_service, JitCache, HipModule).
///   3. Validates the output against CpuFpReferenceConvolution::fprop
///      from the test SDK within tolerance.
///   4. Logs achieved kernel time and TFLOPS via PerfMeasurement.
///
/// The shape set spans tile-aligned variants (M = N*Ho*Wo, GEMM-N = K,
/// and GEMM-K = C*R*S each a multiple of the kernel's 64-wide tile) and
/// partial-tile probes where one or more of those dimensions is not a
/// multiple of 64 -- the latter directly exercise the last-tile
/// boundary handling the tile-aligned example shape never touches.
///
/// Runs on whatever DSL-supported device is present (gfx942 / gfx950 /
/// gfx1151): the production plan builder detects the device arch and the
/// adapter's ``applyArchCodegenConfig`` selects a valid per-arch codegen
/// config, so the same graph compiles and runs on each. The test skips
/// cleanly on a host with no supported device.
///
/// **Adaptation from plan §1:** the test bypasses the hipDNN frontend
/// API and the backend's .so-loading plugin path. Both surfaces are
/// architecturally additive on top of what the unit-test suite
/// already proves -- the plan-builder + plan-execute path here is
/// the exact same code the backend would call after dlopen.
///
/// Tensor layout convention (miopen-provider precedent): host-side
/// tensors carry logical NCHW dims for X/Y and
/// logical KCRS for W, with physical NHWC strides on top. The DSL
/// kernel reads/writes the same NHWC memory layout; the CPU reference
/// iterates logical dims and resolves via strides, so a direct
/// element-wise compare over the packed NHWC buffers walks the same
/// logical positions in the same order.
class IntegrationGpuCkDslConvFp16Gpu : public ::testing::TestWithParam<ConvCase> {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_UNSUPPORTED_ARCH("IntegrationGpuCkDslConvFp16Gpu", _arch);

        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        _planBuilder = std::make_unique<ConvImplicitGemmPlanBuilder>(
            _container->compileServiceBridge(), _container->jitCache());
    }

    std::string _arch;
    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<ConvImplicitGemmPlanBuilder> _planBuilder;
};

TEST_P(IntegrationGpuCkDslConvFp16Gpu, Conv) {
    const ConvCase& cse = GetParam();
    const std::int64_t kN = cse.n;
    const std::int64_t kC = cse.c;
    const std::int64_t kHi = cse.hi;
    const std::int64_t kWi = cse.wi;
    const std::int64_t kK = cse.k;
    const std::int64_t kR = cse.r;
    const std::int64_t kS = cse.s;
    const std::int64_t kHo = convOutputDim(kHi, cse.padH, cse.dilH, kR, cse.strideH);
    const std::int64_t kWo = convOutputDim(kWi, cse.padW, cse.dilW, kS, cse.strideW);
    ASSERT_GT(kHo, 0) << "shape '" << cse.name << "' yields non-positive Ho=" << kHo;
    ASSERT_GT(kWo, 0) << "shape '" << cse.name << "' yields non-positive Wo=" << kWo;

    // FB graph. Strides are the NHWC physical layout expressed over the
    // logical NCHW (X/Y) and KCRS (W) dim order: the channel stride is
    // 1, the W stride is C, the H stride is W*C, the N/K stride is the
    // full per-image span. Tensor UIDs from createValidConvFwdGraph:
    // x=1, w=2, y=3.
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        /*xDims=*/{kN, kC, kHi, kWi},
        /*xStrides=*/{kC * kHi * kWi, 1, kWi * kC, kC},
        /*wDims=*/{kK, kC, kR, kS},
        /*wStrides=*/{kC * kR * kS, 1, kS * kC, kC},
        /*yDims=*/{kN, kK, kHo, kWo},
        /*yStrides=*/{kK * kHo * kWo, 1, kWo * kK, kK},
        /*convPrePadding=*/{cse.padH, cse.padW},
        /*convPostPadding=*/{cse.padH, cse.padW},
        /*convStrides=*/{cse.strideH, cse.strideW},
        /*convDilation=*/{cse.dilH, cse.dilW},
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

    // Seed both inputs. Small range so the K_gemm = C*R*S accumulation
    // stays in a numerically friendly part of FP16 (the accumulator is
    // bounded by |x|*|w|*K_gemm = 0.1*0.1*K_gemm). Random distributions
    // still exercise every codepath the kernel takes; adjusting the
    // range only reduces the tail accumulator error.
    constexpr unsigned kSeedX = 0x4242u;
    constexpr unsigned kSeedW = 0x5555u;
    tensorX.fillWithRandomValues(half(-0.1f), half(0.1f), kSeedX);
    tensorW.fillWithRandomValues(half(-0.1f), half(0.1f), kSeedW);

    // Build the plan. This compiles the kernel on a cold cache
    // (multi-second the first time per unique shape).
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
    CpuFpReferenceConvolution::fprop<half, half, half, float>(
        tensorX, tensorW, tensorYCpu,
        /*strides=*/{cse.strideH, cse.strideW},
        /*dilations=*/{cse.dilH, cse.dilW},
        /*padding=*/{cse.padH, cse.padW});

    // Force D->H on the GPU output so the comparison below sees the
    // kernel's writes. markDeviceModified tells the migration layer
    // "device has the canonical version, copy when hostData is requested
    // next"; reading hostData() once forces that copy before the tensor
    // diff walks the buffer via its host view.
    tensorYGpu.memory().markDeviceModified();
    (void)tensorYGpu.memory().hostData();

    // Tolerance bound (per plan §1): expected error
    // for K_gemm random-uniform fp16 accumulations is roughly
    // sqrt(K_gemm) * fp16_eps * |max_input * max_weight|, which for the
    // shapes here stays well under 1e-3. We use a generous 5e-2
    // absolute tolerance to accommodate accumulation tail behaviour
    // without making the test brittle to minor codegen reshufflings.
    //
    // computeTensorDiff (hipdnn_test_sdk) is the shared comparison
    // oracle: it walks both tensors over their logical dims via host
    // TensorViews and reports the mismatch count + worst offenders, so
    // the provider does not hand-roll its own element loop.
    constexpr float kAbsTol = 5.0e-2f;
    constexpr float kRelTol = 0.0f;
    auto diff = hipdnn_test_sdk::utilities::computeTensorDiff<half>(tensorYCpu, tensorYGpu, kAbsTol,
                                                                    kRelTol);

    if (diff.mismatchCount != 0u) {
        std::ostringstream diffMsg;
        hipdnn_test_sdk::utilities::printTensorDiffSummary(diffMsg, std::string("Y/") + cse.name,
                                                           diff);
        ADD_FAILURE() << "shape '" << cse.name << "' on " << _arch << ": " << diff.mismatchCount
                      << " of " << diff.totalElements
                      << " elements outside tolerance (atol=" << kAbsTol << ")\n"
                      << diffMsg.str();
    }
    const float worstError = diff.maxAbsDiff;

    // Perf measurement (no perf-target assertion, log only per Q9).
    // FLOPS formula from plan §4: 2 * N * Ho * Wo * K * C * R * S.
    const double kFlops = 2.0 * static_cast<double>(kN) * static_cast<double>(kHo) *
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
    pm.log(std::string("conv_implicit_gemm_") + cse.name, result);

    std::ostringstream summary;
    summary << "IntegrationGpuCkDslConvFp16Gpu.Conv/" << cse.name << ": numerical agreement "
            << "(worst abs diff = " << worstError << " < tol = " << kAbsTol
            << "), perf min_us = " << result.minUs << ", median_us = " << result.medianUs
            << ", tflops = " << result.tflops;
    // testing::Test::RecordProperty surfaces in --gtest_output=xml; for
    // ad-hoc console runs the message is appended to the test output
    // on PASS too via SCOPED_TRACE-style note.
    RecordProperty("ck_dsl_perf_summary", summary.str());
}

// Shape set. Cases 1-6 keep M = N*Ho*Wo, GEMM-N = K, and GEMM-K = C*R*S
// each a multiple of the kernel's 64-wide tile (expected to pass). Cases
// 7-10 leave one or more of those dimensions partial to probe last-tile
// boundary handling the tile-aligned example shape never exercises.
const std::vector<ConvCase> kConvCases = {
    // name                  N   C  Hi  Wi    K  R  S   sH sW  pH pW  dH dW
    {"Example", 8, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1},
    {"Stride2", 8, 64, 56, 56, 64, 3, 3, 2, 2, 1, 1, 1, 1},
    {"OneByOne", 8, 64, 56, 56, 64, 1, 1, 1, 1, 0, 0, 1, 1},
    {"BigChannels128", 8, 128, 56, 56, 128, 3, 3, 1, 1, 1, 1, 1, 1},
    {"NonSquare3x1", 8, 64, 56, 56, 64, 3, 1, 1, 1, 1, 0, 1, 1},
    {"Dilation2", 8, 64, 56, 56, 64, 3, 3, 1, 1, 2, 2, 2, 2},
    {"PartialGemmN_K96", 8, 64, 56, 56, 96, 3, 3, 1, 1, 1, 1, 1, 1},
    {"PartialGemmK_C48", 8, 48, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1},
    {"PartialGemmM_1x7x7", 1, 64, 7, 7, 64, 3, 3, 1, 1, 1, 1, 1, 1},
    {"AllPartial", 1, 48, 7, 7, 96, 3, 3, 1, 1, 1, 1, 1, 1},
};

INSTANTIATE_TEST_SUITE_P(Shapes, IntegrationGpuCkDslConvFp16Gpu, ::testing::ValuesIn(kConvCases),
                         [](const ::testing::TestParamInfo<ConvCase>& info) {
                             return std::string(info.param.name);
                         });

}  // namespace
