// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "engines/sdpa/SdpaFwdPlan.hpp"
#include "engines/sdpa/SdpaFwdPlanBuilder.hpp"
#include "perf/PerfMeasurement.hpp"
#include "python/CompileServiceBridge.hpp"
#include "tests/TestUtils.hpp"

namespace {

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace utilities = hipdnn_data_sdk::utilities;
using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::CkDslContext;
using ck_dsl_provider::PerfMeasurement;
using ck_dsl_provider::PerfResult;
using ck_dsl_provider::SdpaFwdPlanBuilder;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/// One LARGE forward-SDPA shape the perf harness times end to end. Every
/// case here is causal (the unified paged kernel is causal-only) and
/// satisfies the Phase-2d capability gate: D in {64, 128, 256}, GQA ratio
/// Hq/Hkv in 1..16, Sq/Skv multiples of 16. This is the PERF-ONLY twin of
/// the SdpaCase in IntegrationGpuCkDslSdpaFwdFp16.cpp -- same fields, but
/// these shapes are prefill-scale (Sq=Skv in the thousands) so a CPU
/// reference compare is intentionally omitted (prohibitively slow); we
/// only want a TFLOPS signal on gfx950.
struct SdpaPerfCase {
    const char* name;
    data_objects::DataType dtype;
    int B, Hq, Hkv, Sq, Skv, D;
};

/// BSHD physical strides for a logical [B, H, S, D] tensor -- the layout
/// the FMHA kernel requires (heads interleaved within each sequence
/// position): batch = S*H*D, head (dim 1) = D, token/seq (dim 2) = H*D,
/// d = 1. The kernel folds the batch offset as batch_idx*seqlen*token (no
/// batch-stride arg), so batch must equal seqlen*token. Identical to the
/// helper in the correctness test.
std::vector<std::int64_t> bshdStrides(int H, int S, int D) {
    return {static_cast<std::int64_t>(S) * H * D, D, static_cast<std::int64_t>(H) * D, 1};
}

/// Element-type-templated body for one perf case. ``ElemT`` is the
/// host/device storage type (half or bfloat16); ``cse.dtype`` must match
/// (HALF -> half, BFLOAT16 -> bfloat16).
///
/// Each case:
///   1. Builds a single-op causal SDPA-fwd graph for the parameterized
///      shape.
///   2. Runs buildPlan -- this drives the capability gate, dispatcher
///      scoring, and JIT compile of the real gfx950 kernel.
///   3. Allocates Q/K/V/O device tensors (small random fill; no D->H
///      readback) and a plan-sized device workspace.
///   4. Times execute() via PerfMeasurement::measure with the
///      full (non-causal-adjusted) FMHA-forward FLOPS formula, matching
///      the correctness test's denominator so the external PyTorch
///      comparison uses the same basis.
///   5. Logs min/median us + TFLOPS via pm.log; no correctness assertion.
template <typename ElemT>
void runSdpaPerfCase(const SdpaPerfCase& cse, ::CkDslHandle& handle,
                     SdpaFwdPlanBuilder& planBuilder) {
    const int kB = cse.B;
    const int kHq = cse.Hq;
    const int kHkv = cse.Hkv;
    const int kSq = cse.Sq;
    const int kSkv = cse.Skv;
    const int kD = cse.D;

    const std::vector<std::int64_t> qDims{kB, kHq, kSq, kD};
    const std::vector<std::int64_t> kDims{kB, kHkv, kSkv, kD};
    const std::vector<std::int64_t> vDims{kB, kHkv, kSkv, kD};
    const std::vector<std::int64_t> oDims{kB, kHq, kSq, kD};

    const std::vector<std::int64_t> qStrides = bshdStrides(kHq, kSq, kD);
    const std::vector<std::int64_t> kStrides = bshdStrides(kHkv, kSkv, kD);
    const std::vector<std::int64_t> vStrides = bshdStrides(kHkv, kSkv, kD);
    const std::vector<std::int64_t> oStrides = bshdStrides(kHq, kSq, kD);

    // Causal FB graph. Tensor UIDs from createValidSdpaFwdGraph: q=1, k=2,
    // v=3, o=4. The graph dtype matches the host/device element type.
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims, qStrides, kDims, kStrides, vDims, vStrides, oDims, oStrides,
        /*dataType=*/cse.dtype, /*withAttnMask=*/false, /*withScale=*/false,
        /*withStats=*/false, /*alibiMask=*/false, /*paddingMask=*/false, /*causalMask=*/true);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    // Host/device tensors with logical [B, H, S, D] dims + BSHD strides.
    // No D->H readback is needed -- this is perf only.
    utilities::Tensor<ElemT> tensorQ(qDims, qStrides);
    utilities::Tensor<ElemT> tensorK(kDims, kStrides);
    utilities::Tensor<ElemT> tensorV(vDims, vStrides);
    utilities::Tensor<ElemT> tensorO(oDims, oStrides);

    // Small range so the softmax accumulation stays numerically friendly
    // in the 16-bit float range.
    constexpr unsigned kSeedQ = 0x4242u;
    constexpr unsigned kSeedK = 0x5555u;
    constexpr unsigned kSeedV = 0x6363u;
    tensorQ.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedQ);
    tensorK.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedK);
    tensorV.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedV);

    // Build the plan. This runs the capability gate + dispatcher scoring +
    // JIT compile of the real gfx950 kernel (multi-second on a cold cache
    // per unique shape). If the gate declines the shape, buildPlan throws
    // and the case fails -- that is a finding, not pre-trimmed here.
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);
    CkDslContext ctx;
    planBuilder.buildPlan(handle, graph, engineConfig, ctx);
    ASSERT_TRUE(ctx.hasValidPlan()) << "case '" << cse.name << "': buildPlan produced no plan";

    // execute() REQUIRES a non-null workspace (marshalled i32 arrays the
    // kernel reads). Size it from the plan; free it before returning.
    const std::size_t wsBytes = ctx.plan().getWorkspaceSize(handle);
    void* workspace = nullptr;
    if (wsBytes > 0) {
        ASSERT_EQ(hipMalloc(&workspace, wsBytes), hipSuccess);
    }

    // Reading deviceData() drives the H->D copies for the inputs; the
    // output device buffer is written by the kernel.
    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {
        {1, tensorQ.memory().deviceData()},
        {2, tensorK.memory().deviceData()},
        {3, tensorV.memory().deviceData()},
        {4, tensorO.memory().deviceData()},
    };

    // One warm execute outside the timing loop -- surfaces any execute
    // failure as a thrown exception, and primes the device buffers.
    ASSERT_NO_THROW(ctx.plan().execute(
        handle, deviceBuffers.data(), static_cast<std::uint32_t>(deviceBuffers.size()), workspace));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // FMHA-forward FLOPS: two GEMMs (QK^T and PV), each 2*B*Hq*Sq*Skv*D.
    // Kept identical to the correctness test (full, non-causal-adjusted)
    // so the external PyTorch comparison shares the same denominator.
    const double kFlops = 4.0 * static_cast<double>(kB) * static_cast<double>(kHq) *
                          static_cast<double>(kSq) * static_cast<double>(kSkv) *
                          static_cast<double>(kD);

    PerfMeasurement pm;
    auto launchFn = [&ctx, &handle, &deviceBuffers, workspace]() {
        ctx.plan().execute(handle, deviceBuffers.data(),
                           static_cast<std::uint32_t>(deviceBuffers.size()), workspace);
    };
    PerfResult result = pm.measure(launchFn, kFlops, handle.getStream());
    // pm.log emits the perf summary (min/median us, tflops) under
    // HIPDNN_LOG_LEVEL=info. RecordProperty is intentionally NOT used (it
    // is a Test-fixture method, unavailable in this free helper); the
    // logged line carries the same data.
    pm.log(std::string("sdpa_perf_") + cse.name, result);

    if (workspace != nullptr) {
        ASSERT_EQ(hipFree(workspace), hipSuccess);
    }
}

/// PERF-ONLY gfx950 harness for the unified SDPA-forward kernel over
/// realistic LARGE (prefill-scale) shapes. No CPU correctness compare --
/// the large Sq makes the reference prohibitively slow; we only collect
/// timings (median us + TFLOPS). Skips on non-gfx950.
class IntegrationGpuCkDslSdpaFwdPerfGpu : public ::testing::TestWithParam<SdpaPerfCase> {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("IntegrationGpuCkDslSdpaFwdPerfGpu");

        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        _planBuilder = std::make_unique<SdpaFwdPlanBuilder>(_container->compileServiceBridge(),
                                                            _container->jitCache());
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<SdpaFwdPlanBuilder> _planBuilder;
};

TEST_P(IntegrationGpuCkDslSdpaFwdPerfGpu, Perf) {
    const SdpaPerfCase& cse = GetParam();
    switch (cse.dtype) {
        case data_objects::DataType::HALF:
            runSdpaPerfCase<half>(cse, *_handle, *_planBuilder);
            break;
        case data_objects::DataType::BFLOAT16:
            runSdpaPerfCase<bfloat16>(cse, *_handle, *_planBuilder);
            break;
        default:
            FAIL() << "unsupported dtype for case '" << cse.name << "'";
    }
}

// Realistic LARGE prefill shapes. All causal, BSHD, D in {64, 128, 256}.
// Coverage spans: GQA vs MHA, S2048 vs S4096, B1 vs B4, all three head
// sizes, and both kernel dtypes. Names + dims are mirrored by the external
// PyTorch comparison script -- keep them exact.
const std::vector<SdpaPerfCase> kSdpaPerfCases = {
    // name                              dtype                          B Hq Hkv  Sq  Skv  D
    {"Fp16_Prefill_GQA_S2048_D128", data_objects::DataType::HALF, 1, 32, 8, 2048, 2048, 128},
    {"Fp16_Prefill_GQA_S4096_D128", data_objects::DataType::HALF, 1, 32, 8, 4096, 4096, 128},
    {"Fp16_Prefill_MHA_S2048_D128", data_objects::DataType::HALF, 1, 32, 32, 2048, 2048, 128},
    {"Fp16_Prefill_GQA_B4_S2048_D128", data_objects::DataType::HALF, 4, 32, 8, 2048, 2048, 128},
    {"Fp16_Prefill_GQA_S2048_D64", data_objects::DataType::HALF, 1, 32, 8, 2048, 2048, 64},
    {"Fp16_Prefill_GQA_S2048_D256", data_objects::DataType::HALF, 1, 32, 8, 2048, 2048, 256},
    {"Bf16_Prefill_GQA_S2048_D128", data_objects::DataType::BFLOAT16, 1, 32, 8, 2048, 2048, 128},
    // In-family with the gfx950 heuristic's training set (~bf16/d64/h64kv8;
    // GQA ratio 8 = Hq64/Hkv8). S2016 makes the dense-degenerate block_size
    // resolve to 32 (2016 = 63*32, not %64), matching the trained b32.
    {"Bf16_InFamily_GQA8_D64_S2048", data_objects::DataType::BFLOAT16, 1, 64, 8, 2048, 2048, 64},
    {"Bf16_InFamily_GQA8_D64_S2016_B32", data_objects::DataType::BFLOAT16, 1, 64, 8, 2016, 2016,
     64},
};

INSTANTIATE_TEST_SUITE_P(Shapes, IntegrationGpuCkDslSdpaFwdPerfGpu,
                         ::testing::ValuesIn(kSdpaPerfCases),
                         [](const ::testing::TestParamInfo<SdpaPerfCase>& info) {
                             return std::string(info.param.name);
                         });

}  // namespace
