// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cmath>
#include <cstdint>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "engines/sdpa/SdpaBwdPlanBuilder.hpp"
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
using ck_dsl_provider::SdpaBwdPlanBuilder;
using hipdnn_data_sdk::types::half;
using hipdnn_test_sdk::utilities::CpuFpReferenceSdpa;
using DataType = data_objects::DataType;

/// One backward-SDPA problem the parameterized integration test drives
/// end to end. GQA is expressed via Hkv < Hq (Hq % Hkv == 0). Sq/Skv
/// must be multiples of 16 and D one of {64, 128, 192, 256}.
struct SdpaBwdCase {
    int B, Hq, Hkv, Sq, Skv, D;
    bool causal;
    const char* name;
};

/// BSHD physical strides for a logical [B, H, S, D] tensor -- the layout
/// the FMHA kernels require: batch = S*H*D (== seqlen*token), head = D,
/// token = H*D, d = 1.
std::vector<std::int64_t> bshdStrides(int H, int S, int D) {
    return {static_cast<std::int64_t>(S) * H * D, D, static_cast<std::int64_t>(H) * D, 1};
}

/// Build a complete single-node SDPA-backward graph FlatBuffer with FP16
/// Q/K/V/O/dO and FLOAT stats/dQ/dK/dV. UID order: q=1, k=2, v=3, o=4,
/// do=5, stats=6, dq=7, dk=8, dv=9.
///
/// **Why not the SDK helper.** ``createValidSdpaBwdGraph`` builds
/// dQ/dK/dV with the same dtype as Q/K/V (HALF), but the bwd adapter
/// REQUIRES dQ/dK/dV FLOAT (f32 accumulators), so the helper cannot
/// produce a graph the adapter accepts. This builds the graph by hand.
flatbuffers::FlatBufferBuilder makeSdpaBwdGraph(const SdpaBwdCase& cse) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensorAttributes;

    const std::vector<std::int64_t> qoDims{cse.B, cse.Hq, cse.Sq, cse.D};
    const std::vector<std::int64_t> kvDims{cse.B, cse.Hkv, cse.Skv, cse.D};
    const std::vector<std::int64_t> qoStrides = bshdStrides(cse.Hq, cse.Sq, cse.D);
    const std::vector<std::int64_t> kvStrides = bshdStrides(cse.Hkv, cse.Skv, cse.D);

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

    const std::vector<std::int64_t> statsDims{cse.B, cse.Hq, cse.Sq, 1};
    const std::vector<std::int64_t> statsStrides{static_cast<std::int64_t>(cse.Hq) * cse.Sq, cse.Sq,
                                                 1, 1};
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, /*uid=*/6, "stats", DataType::FLOAT, &statsStrides, &statsDims));

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
        /*causal_mask=*/cse.causal);

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

/// End-to-end M1 integration coverage for the FMHA-backward path.
///
/// Each case:
///   1. Builds a single-op SDPA-backward graph for the parameterized
///      shape (FP16 Q/K/V/O/dO, FLOAT stats/dQ/dK/dV, BSHD strides).
///   2. Computes O and the natural-log LSE on host via the forward CPU
///      reference; the LSE seeds the kernel's ``stats`` input (the bwd
///      kernel consumes the forward LSE rather than recomputing softmax).
///   3. Runs the bwd graph through the JIT pipeline (plan-builder,
///      adapter, bridge, compile_service, JitCache, two HipModules),
///      allocating the M/L scratch workspace the prep kernel needs.
///   4. Validates dQ/dK/dV against CpuFpReferenceSdpa::backward (fed the
///      SAME host LSE + causal flag) within tolerance.
///   5. Logs achieved kernel time and TFLOPS via PerfMeasurement.
///
/// **Adaptation:** like the forward integration test, this bypasses the
/// hipDNN frontend API and the backend's .so-loading plugin path -- both
/// are architecturally additive on top of the plan-builder +
/// plan-execute path exercised here, which is the exact same code the
/// backend would call after dlopen.
class IntegrationGpuCkDslSdpaBwdFp16Gpu : public ::testing::TestWithParam<SdpaBwdCase> {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("IntegrationGpuCkDslSdpaBwdFp16Gpu");

        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        _planBuilder = std::make_unique<SdpaBwdPlanBuilder>(_container->compileServiceBridge(),
                                                            _container->jitCache());
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<SdpaBwdPlanBuilder> _planBuilder;
};

TEST_P(IntegrationGpuCkDslSdpaBwdFp16Gpu, Sdpa) {
    const SdpaBwdCase& cse = GetParam();
    const int kB = cse.B;
    const int kHq = cse.Hq;
    const int kHkv = cse.Hkv;
    const int kSq = cse.Sq;
    const int kSkv = cse.Skv;
    const int kD = cse.D;

    const std::vector<std::int64_t> qDims{kB, kHq, kSq, kD};
    const std::vector<std::int64_t> kDims{kB, kHkv, kSkv, kD};
    const std::vector<std::int64_t> oDims{kB, kHq, kSq, kD};
    const std::vector<std::int64_t> statsDims{kB, kHq, kSq};  // rank-3 [B, Hq, Sq]

    const std::vector<std::int64_t> qStrides = bshdStrides(kHq, kSq, kD);
    const std::vector<std::int64_t> kStrides = bshdStrides(kHkv, kSkv, kD);
    const std::vector<std::int64_t> oStrides = bshdStrides(kHq, kSq, kD);
    const std::vector<std::int64_t> statsStrides{static_cast<std::int64_t>(kHq) * kSq, kSq, 1};

    auto fbBuilder = makeSdpaBwdGraph(cse);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    // Host tensors. Q/K/V/O/dO are FP16 with BSHD strides; dQ/dK/dV are
    // FLOAT accumulators; stats is the FLOAT natural-log LSE [B, Hq, Sq].
    utilities::Tensor<half> tensorQ(qDims, qStrides);
    utilities::Tensor<half> tensorK(kDims, kStrides);
    utilities::Tensor<half> tensorV(kDims, kStrides);
    utilities::Tensor<half> tensorO(oDims, oStrides);   // forward output (host-computed)
    utilities::Tensor<half> tensorDO(oDims, oStrides);  // upstream gradient
    utilities::Tensor<float> tensorStats(statsDims, statsStrides);

    utilities::Tensor<float> tensorDQGpu(qDims, qStrides);
    utilities::Tensor<float> tensorDKGpu(kDims, kStrides);
    utilities::Tensor<float> tensorDVGpu(kDims, kStrides);
    utilities::Tensor<float> tensorDQCpu(qDims, qStrides);
    utilities::Tensor<float> tensorDKCpu(kDims, kStrides);
    utilities::Tensor<float> tensorDVCpu(kDims, kStrides);

    // Small range so softmax accumulation stays in a numerically friendly
    // part of FP16.
    constexpr unsigned kSeedQ = 0x4242u;
    constexpr unsigned kSeedK = 0x5555u;
    constexpr unsigned kSeedV = 0x6363u;
    constexpr unsigned kSeedDO = 0x7171u;
    tensorQ.fillWithRandomValues(half(-0.1f), half(0.1f), kSeedQ);
    tensorK.fillWithRandomValues(half(-0.1f), half(0.1f), kSeedK);
    tensorV.fillWithRandomValues(half(-0.1f), half(0.1f), kSeedV);
    tensorDO.fillWithRandomValues(half(-0.1f), half(0.1f), kSeedDO);

    // Forward reference to produce O and the natural-log LSE. nullopt
    // scale -> 1/sqrt(D), matching the adapter's default. The LSE seeds
    // BOTH the kernel's stats input and the backward reference, so the
    // two share an identical softmax normaliser.
    CpuFpReferenceSdpa::forward<half, half, half, half, float>(
        tensorQ, tensorK, tensorV, tensorO, /*attnScaleValue=*/std::nullopt,
        /*attnMask=*/nullptr, /*causalMask=*/cse.causal, /*lse=*/&tensorStats);

    // Build the plan. Compiles two kernels on a cold cache.
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);
    CkDslContext ctx;
    _planBuilder->buildPlan(*_handle, graph, engineConfig, ctx);
    ASSERT_TRUE(ctx.hasValidPlan());

    // Allocate the M/L scratch workspace the LSE-prep kernel writes.
    const std::size_t workspaceBytes = ctx.plan().getWorkspaceSize(*_handle);
    void* workspace = nullptr;
    if (workspaceBytes > 0) {
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
    }

    // Reading deviceData() drives the H->D copies for the inputs; the
    // gradient device buffers are written by the kernel. O is a graph
    // input the kernel ignores, but provide it for completeness.
    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {
        {1, tensorQ.memory().deviceData()},     {2, tensorK.memory().deviceData()},
        {3, tensorV.memory().deviceData()},     {4, tensorO.memory().deviceData()},
        {5, tensorDO.memory().deviceData()},    {6, tensorStats.memory().deviceData()},
        {7, tensorDQGpu.memory().deviceData()}, {8, tensorDKGpu.memory().deviceData()},
        {9, tensorDVGpu.memory().deviceData()},
    };

    ASSERT_NO_THROW(ctx.plan().execute(*_handle, deviceBuffers.data(),
                                       static_cast<std::uint32_t>(deviceBuffers.size()),
                                       workspace));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Backward reference. Pass the same host LSE + causal flag so the
    // softmax recomputation matches the kernel's.
    CpuFpReferenceSdpa::backward<half, half, half, half, half, float, float, float, float>(
        tensorQ, tensorK, tensorV, tensorO, tensorDO, tensorDQCpu, tensorDKCpu, tensorDVCpu,
        /*attnScaleValue=*/std::nullopt, /*lse=*/&tensorStats, /*attnMask=*/nullptr,
        /*causalMask=*/cse.causal);

    // Force D->H so host reads see the kernel's writes. markDeviceModified()
    // flags the host copy stale; the actual sync happens on the next
    // NON-const hostData() access. compareGrad below reads through a
    // ``const Tensor&`` (const hostData() cannot trigger the copy and would
    // throw "host memory is out of date"), so drive the D->H sync here on the
    // non-const tensors first.
    tensorDQGpu.memory().markDeviceModified();
    tensorDKGpu.memory().markDeviceModified();
    tensorDVGpu.memory().markDeviceModified();
    (void)tensorDQGpu.memory().hostData();
    (void)tensorDKGpu.memory().hostData();
    (void)tensorDVGpu.memory().hostData();

    // FP16 inputs + f32 accumulation: bwd gradients accumulate across the
    // kv sequence, so use a slightly looser tolerance than the forward
    // path.
    constexpr float kAbsTol = 5.0e-2f;

    auto compareGrad =
        [&](const char* gradName, const utilities::Tensor<float>& gpu,
            const utilities::Tensor<float>& cpu) {
            const float* gpuData = gpu.memory().hostData();
            const float* cpuData = cpu.memory().hostData();
            const std::size_t numElements = gpu.memory().count();
            std::size_t mismatches = 0;
            std::size_t firstMismatchIdx = 0;
            float worstError = 0.0f;
            float worstGpu = 0.0f;
            float worstCpu = 0.0f;
            for (std::size_t i = 0; i < numElements; ++i) {
                const float g = gpuData[i];
                const float c = cpuData[i];
                const float diff = std::fabs(g - c);
                if (diff > worstError) {
                    worstError = diff;
                    worstGpu = g;
                    worstCpu = c;
                }
                if (diff > kAbsTol) {
                    if (mismatches == 0) {
                        firstMismatchIdx = i;
                    }
                    ++mismatches;
                }
            }
            EXPECT_EQ(mismatches, 0u)
                << "shape '" << cse.name << "' grad '" << gradName << "': found " << mismatches
                << " elements outside the " << kAbsTol << " tolerance ("
                << static_cast<double>(mismatches) / static_cast<double>(numElements) * 100.0
                << "%); first mismatch at linear index " << firstMismatchIdx << "; worst diff "
                << worstError << " (gpu=" << worstGpu << ", cpu=" << worstCpu << ")";
            return worstError;
        };

    const float worstDQ = compareGrad("dQ", tensorDQGpu, tensorDQCpu);
    const float worstDK = compareGrad("dK", tensorDKGpu, tensorDKCpu);
    const float worstDV = compareGrad("dV", tensorDVGpu, tensorDVCpu);
    const float worstError = std::max({worstDQ, worstDK, worstDV});

    // Perf measurement (no perf-target assertion, log only). FMHA-backward
    // FLOPS: roughly 5 GEMMs of 2*B*Hq*Sq*Skv*D each (dS, dQ, dK, dV, and
    // the recomputed P), so ~10 * B*Hq*Sq*Skv*D.
    const double kFlops = 10.0 * static_cast<double>(kB) * static_cast<double>(kHq) *
                          static_cast<double>(kSq) * static_cast<double>(kSkv) *
                          static_cast<double>(kD);
    PerfMeasurement pm;
    auto launchFn = [&]() {
        ctx.plan().execute(*_handle, deviceBuffers.data(),
                           static_cast<std::uint32_t>(deviceBuffers.size()), workspace);
    };
    PerfResult result = pm.measure(launchFn, kFlops, _handle->getStream());
    pm.log(std::string("sdpa_fmha_bwd_") + cse.name, result);

    std::ostringstream summary;
    summary << "IntegrationGpuCkDslSdpaBwdFp16Gpu.Sdpa/" << cse.name << ": numerical agreement "
            << "(worst abs diff = " << worstError << " < tol = " << kAbsTol
            << "), perf min_us = " << result.minUs << ", median_us = " << result.medianUs
            << ", tflops = " << result.tflops;
    RecordProperty("ck_dsl_perf_summary", summary.str());

    if (workspace != nullptr) {
        ASSERT_EQ(hipFree(workspace), hipSuccess);
    }
}

// Shape set. D=128 (the M1 forward coverage gap). A no-mask MHA case, a
// causal MHA case, a GQA case, and a causal+GQA case. Sq/Skv multiples
// of 16.
const std::vector<SdpaBwdCase> kSdpaBwdCases = {
    // B  Hq Hkv  Sq  Skv   D  causal name
    {2, 8, 8, 64, 64, 128, false, "NoMask"},
    {2, 8, 8, 64, 64, 128, true, "Causal"},
    {2, 8, 2, 64, 64, 128, false, "Gqa"},
    {2, 8, 2, 64, 64, 128, true, "CausalGqa"},
};

INSTANTIATE_TEST_SUITE_P(Shapes, IntegrationGpuCkDslSdpaBwdFp16Gpu,
                         ::testing::ValuesIn(kSdpaBwdCases),
                         [](const ::testing::TestParamInfo<SdpaBwdCase>& info) {
                             return std::string(info.param.name);
                         });

}  // namespace
