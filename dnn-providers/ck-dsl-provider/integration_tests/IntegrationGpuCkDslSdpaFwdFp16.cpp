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
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <memory>
#include <optional>
#include <sstream>
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
using hipdnn_test_sdk::utilities::CpuFpReferenceSdpa;

/// One forward-SDPA problem the parameterized integration test drives
/// end to end. GQA is expressed via Hkv < Hq (Hq % Hkv == 0). Sq/Skv
/// must be multiples of 16 and D one of the kernel-supported sizes
/// (64, 128, 256). The unified paged kernel is causal-only, so every
/// case here is causal -- the Phase-2d capability gate declines a
/// non-causal request (buildPlan would throw).
struct SdpaCase {
    int B, Hq, Hkv, Sq, Skv, D;
    bool causal;
    data_objects::DataType dtype;
    const char* name;
};

/// BSHD physical strides for a logical [B, H, S, D] tensor -- the layout
/// the FMHA kernel requires (heads interleaved within each sequence
/// position): batch = S*H*D (== seqlen*token), head (dim 1) = D,
/// token/seq (dim 2) = H*D, d = 1. The kernel folds the batch offset as
/// batch_idx*seqlen*token (no batch-stride arg), so batch must equal
/// seqlen*token; the adapter rejects any other layout.
std::vector<std::int64_t> bshdStrides(int H, int S, int D) {
    return {static_cast<std::int64_t>(S) * H * D, D, static_cast<std::int64_t>(H) * D, 1};
}

/// Element-type-templated body for one forward-SDPA case. ``ElemT`` is
/// the host/device storage type (half or bfloat16); ``cse.dtype`` must
/// match (HALF -> half, BFLOAT16 -> bfloat16). The flatbuffer graph,
/// host tensors, kernel I/O dtype, and CPU reference are all driven by
/// ``ElemT`` so a single implementation covers fp16 and bf16.
///
/// Each case:
///   1. Builds a single-op SDPA-fwd graph for the parameterized shape.
///   2. Runs it through the JIT pipeline (plan-builder, adapter, bridge,
///      compile_service, JitCache, HipModule).
///   3. Validates the output against CpuFpReferenceSdpa::forward from the
///      test SDK within tolerance.
///   4. Logs achieved kernel time and TFLOPS via PerfMeasurement.
///
/// Tensor layout: host-side tensors carry logical [B, H, S, D] dims with
/// BSHD physical strides (heads interleaved within each sequence
/// position). The DSL kernel reads/writes the same memory; the CPU
/// reference iterates logical dims and resolves via the tensor's strides,
/// so a direct element-wise compare over the buffers walks the same
/// logical positions in the same order.
template <typename ElemT>
void runSdpaCase(const SdpaCase& cse, ::CkDslHandle& handle, SdpaFwdPlanBuilder& planBuilder) {
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

    // FB graph. Tensor UIDs from createValidSdpaFwdGraph: q=1, k=2, v=3,
    // o=4. The graph dtype matches the host/device element type.
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims, qStrides, kDims, kStrides, vDims, vStrides, oDims, oStrides,
        /*dataType=*/cse.dtype, /*withAttnMask=*/false, /*withScale=*/false,
        /*withStats=*/false, /*alibiMask=*/false, /*paddingMask=*/false, /*causalMask=*/cse.causal);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    // Host tensors with logical [B, H, S, D] dims + BSHD strides.
    // ``Tensor<ElemT>`` owns both host and device storage and syncs on
    // demand.
    utilities::Tensor<ElemT> tensorQ(qDims, qStrides);
    utilities::Tensor<ElemT> tensorK(kDims, kStrides);
    utilities::Tensor<ElemT> tensorV(vDims, vStrides);
    utilities::Tensor<ElemT> tensorOGpu(oDims, oStrides);
    utilities::Tensor<ElemT> tensorOCpu(oDims, oStrides);

    // Small range so the softmax accumulation stays in a numerically
    // friendly part of the 16-bit float range.
    constexpr unsigned kSeedQ = 0x4242u;
    constexpr unsigned kSeedK = 0x5555u;
    constexpr unsigned kSeedV = 0x6363u;
    tensorQ.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedQ);
    tensorK.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedK);
    tensorV.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedV);

    // Build the plan. Compiles the kernel on a cold cache (multi-second
    // the first time per unique shape).
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);
    CkDslContext ctx;
    planBuilder.buildPlan(handle, graph, engineConfig, ctx);
    ASSERT_TRUE(ctx.hasValidPlan());

    // execute() is now wired and REQUIRES a non-null workspace: it holds
    // the 3 marshalled i32 arrays the kernel reads. Size it from the plan
    // and allocate once; free it before the test body returns.
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
        {4, tensorOGpu.memory().deviceData()},
    };

    ASSERT_NO_THROW(ctx.plan().execute(
        handle, deviceBuffers.data(), static_cast<std::uint32_t>(deviceBuffers.size()), workspace));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // CPU reference. nullopt scale -> the reference uses 1/sqrt(D),
    // matching the adapter's default scale. float accumulation for both
    // fp16 and bf16 host tensors.
    CpuFpReferenceSdpa::forward<ElemT, ElemT, ElemT, ElemT, float>(
        tensorQ, tensorK, tensorV, tensorOCpu, /*attnScaleValue=*/std::nullopt,
        /*attnMask=*/nullptr, /*causalMask=*/cse.causal, /*lse=*/nullptr);

    // Force D->H on the GPU output so host reads see the kernel's writes.
    tensorOGpu.memory().markDeviceModified();
    const ElemT* gpuOut = tensorOGpu.memory().hostData();
    const ElemT* cpuOut = tensorOCpu.memory().hostData();

    // 16-bit SDPA tolerance: softmax(QK^T)V over a small input range. bf16
    // has ~3 fewer mantissa bits than fp16, so use a slightly looser bound
    // for it while keeping fp16 tight.
    const float kAbsTol = (cse.dtype == data_objects::DataType::BFLOAT16) ? 4.0e-2f : 1.0e-2f;
    std::size_t mismatches = 0;
    std::size_t firstMismatchIdx = 0;
    float worstError = 0.0f;
    float worstGpu = 0.0f;
    float worstCpu = 0.0f;

    const std::size_t numElements = tensorOGpu.memory().count();
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

    EXPECT_EQ(mismatches, 0u) << "shape '" << cse.name << "': found " << mismatches
                              << " elements outside the " << kAbsTol << " tolerance ("
                              << static_cast<double>(mismatches) /
                                     static_cast<double>(numElements) * 100.0
                              << "%); first mismatch at linear index " << firstMismatchIdx
                              << "; worst diff " << worstError << " (gpu=" << worstGpu
                              << ", cpu=" << worstCpu << ")";

    // Perf measurement (no perf-target assertion, log only). FMHA-forward
    // FLOPS: two GEMMs (QK^T and PV), each 2*B*Hq*Sq*Skv*D.
    const double kFlops = 4.0 * static_cast<double>(kB) * static_cast<double>(kHq) *
                          static_cast<double>(kSq) * static_cast<double>(kSkv) *
                          static_cast<double>(kD);
    PerfMeasurement pm;
    auto launchFn = [&ctx, &handle, &deviceBuffers, workspace]() {
        ctx.plan().execute(handle, deviceBuffers.data(),
                           static_cast<std::uint32_t>(deviceBuffers.size()), workspace);
    };
    PerfResult result = pm.measure(launchFn, kFlops, handle.getStream());
    // pm.log emits the perf summary (min/median us, tflops) to the test
    // log. (RecordProperty is a Test-fixture method and is not available
    // here in the free helper; the logged line carries the same data.)
    pm.log(std::string("sdpa_fmha_fwd_") + cse.name, result);

    if (workspace != nullptr) {
        ASSERT_EQ(hipFree(workspace), hipSuccess);
    }
}

/// End-to-end M1 integration coverage for the FMHA-forward path.
///
/// **Adaptation:** the test bypasses the hipDNN frontend API and the
/// backend's .so-loading plugin path -- both are architecturally
/// additive on top of the plan-builder + plan-execute path exercised
/// here, which is the exact same code the backend would call after
/// dlopen.
class IntegrationGpuCkDslSdpaFwdFp16Gpu : public ::testing::TestWithParam<SdpaCase> {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("IntegrationGpuCkDslSdpaFwdFp16Gpu");

        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        _planBuilder = std::make_unique<SdpaFwdPlanBuilder>(_container->compileServiceBridge(),
                                                            _container->jitCache());
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<SdpaFwdPlanBuilder> _planBuilder;
};

TEST_P(IntegrationGpuCkDslSdpaFwdFp16Gpu, Sdpa) {
    const SdpaCase& cse = GetParam();
    switch (cse.dtype) {
        case data_objects::DataType::HALF:
            runSdpaCase<half>(cse, *_handle, *_planBuilder);
            break;
        case data_objects::DataType::BFLOAT16:
            runSdpaCase<bfloat16>(cse, *_handle, *_planBuilder);
            break;
        default:
            FAIL() << "unsupported dtype for case '" << cse.name << "'";
    }
}

// Supported-matrix subset. Every case is causal -- the unified paged
// kernel (and the Phase-2d capability gate) is causal-only, so a
// non-causal request is declined by buildPlan. Coverage spans:
//   - both kernel dtypes (fp16, bf16),
//   - all three head_size values D in {64, 128, 256},
//   - a GQA case (Hkv < Hq, Hq % Hkv == 0).
// Sq/Skv are multiples of 16.
//
// NOTE: sliding-window (left_bound > 0) and attention-sink correctness
// cases are intentionally omitted. CpuFpReferenceSdpa models scale +
// top-left causal + LSE only -- it has no sliding-window band or sink
// token, so it cannot serve as a numerical oracle for those features.
// Adding sliding-window/sinks numerical coverage requires extending the
// CPU reference first (a gfx950-session follow-up).
const std::vector<SdpaCase> kSdpaCases = {
    // B  Hq Hkv  Sq  Skv   D  causal dtype                          name
    {2, 8, 8, 64, 64, 64, true, data_objects::DataType::HALF, "Fp16CausalD64"},
    {2, 8, 8, 64, 64, 128, true, data_objects::DataType::HALF, "Fp16CausalD128"},
    {2, 8, 8, 64, 64, 256, true, data_objects::DataType::HALF, "Fp16CausalD256"},
    {2, 8, 2, 64, 64, 128, true, data_objects::DataType::HALF, "Fp16CausalGqaD128"},
    {2, 8, 8, 64, 64, 64, true, data_objects::DataType::BFLOAT16, "Bf16CausalD64"},
    {2, 8, 8, 64, 64, 128, true, data_objects::DataType::BFLOAT16, "Bf16CausalD128"},
    {2, 8, 8, 64, 64, 256, true, data_objects::DataType::BFLOAT16, "Bf16CausalD256"},
    {2, 8, 2, 64, 64, 128, true, data_objects::DataType::BFLOAT16, "Bf16CausalGqaD128"},
};

INSTANTIATE_TEST_SUITE_P(Shapes, IntegrationGpuCkDslSdpaFwdFp16Gpu, ::testing::ValuesIn(kSdpaCases),
                         [](const ::testing::TestParamInfo<SdpaCase>& info) {
                             return std::string(info.param.name);
                         });

}  // namespace
