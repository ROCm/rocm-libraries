// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

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
using DataType = data_objects::DataType;

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

/// Warm + timed execute() of an already-built plan, logging TFLOPS. Shared
/// by the dense, paged, and varlen harnesses so the timing logic lives in
/// one place. ``deviceBuffers`` must already carry every uid the plan
/// resolves (Q/K/V/O plus any page-table / seq-len buffers); ``workspace``
/// must be plan-sized (or null when getWorkspaceSize is 0). ``flops`` is
/// the case's FLOPS basis (the caller picks the right denominator -- full
/// B*S^2 for dense/paged, the actual-token sum for varlen). No correctness
/// assertion: these are perf probes.
void timeExecuteAndLog(const std::string& caseName, ck_dsl_provider::CkDslContext& ctx,
                       ::CkDslHandle& handle,
                       std::vector<hipdnnPluginDeviceBuffer_t>& deviceBuffers, void* workspace,
                       double flops) {
    // One warm execute outside the timing loop -- surfaces any execute
    // failure as a thrown exception, and primes the device buffers.
    ASSERT_NO_THROW(ctx.plan().execute(
        handle, deviceBuffers.data(), static_cast<std::uint32_t>(deviceBuffers.size()), workspace));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    PerfMeasurement pm;
    auto launchFn = [&ctx, &handle, &deviceBuffers, workspace]() {
        ctx.plan().execute(handle, deviceBuffers.data(),
                           static_cast<std::uint32_t>(deviceBuffers.size()), workspace);
    };
    PerfResult result = pm.measure(launchFn, flops, handle.getStream());
    // pm.log emits the perf summary (min/median us, tflops) under
    // HIPDNN_LOG_LEVEL=info. RecordProperty is intentionally NOT used (it
    // is a Test-fixture method, unavailable in this free helper); the
    // logged line carries the same data.
    pm.log(std::string("sdpa_perf_") + caseName, result);
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

    // FMHA-forward FLOPS: two GEMMs (QK^T and PV), each 2*B*Hq*Sq*Skv*D.
    // Kept identical to the correctness test (full, non-causal-adjusted)
    // so the external PyTorch comparison shares the same denominator.
    const double kFlops = 4.0 * static_cast<double>(kB) * static_cast<double>(kHq) *
                          static_cast<double>(kSq) * static_cast<double>(kSkv) *
                          static_cast<double>(kD);

    timeExecuteAndLog(cse.name, ctx, handle, deviceBuffers, workspace, kFlops);

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
    // Saturation sweep: push S and B up to see if TFLOPS plateaus.
    {"Fp16_Prefill_GQA_S8192_D128", data_objects::DataType::HALF, 1, 32, 8, 8192, 8192, 128},
    {"Fp16_Prefill_GQA_B8_S2048_D128", data_objects::DataType::HALF, 8, 32, 8, 2048, 2048, 128},
    {"Fp16_Prefill_GQA_B4_S4096_D128", data_objects::DataType::HALF, 4, 32, 8, 4096, 4096, 128},
};

INSTANTIATE_TEST_SUITE_P(Shapes, IntegrationGpuCkDslSdpaFwdPerfGpu,
                         ::testing::ValuesIn(kSdpaPerfCases),
                         [](const ::testing::TestParamInfo<SdpaPerfCase>& info) {
                             return std::string(info.param.name);
                         });

// ---- Paged / varlen perf probes -----------------------------------------
//
// These exercise the real-paged and varlen launch paths the dense harness
// above never touches. They are PERF-ONLY (no CPU oracle): there is no
// reference for the gather/varlen layouts in this POC, so we only collect
// a TFLOPS signal. Each builds a single-buffer SDPA graph BY HAND (the
// shared createValidSdpaFwdGraph helper does not expose the page-table /
// seq-len UIDs), mirroring the SdpaAttrFixture in SdpaAdapterTest.cpp.

// One representative GQA shape, shared by every paged/varlen probe so the
// numbers compare against the dense Fp16_Prefill_GQA_B4_S2048_D128 case.
constexpr int kVarB = 4;
constexpr int kVarHq = 32;
constexpr int kVarHkv = 8;
constexpr int kVarS = 2048;
constexpr int kVarD = 128;
// block_size 64: maxBlocksPerSeq = ceil(2048/64) = 32, and with
// max_seq_len_kv = 2048 the adapter derives ceil(2048/32) = 64. So the
// page table is [B, 32] and the block-table stride is 32.
constexpr int kVarBlockSize = 64;

// UIDs for the optional variant tensors (Q/K/V/O are 1..4). Matches the
// SdpaAttrFixture contract so the adapter / plan resolve them.
constexpr std::int64_t kPerfPageKUid = 11;
constexpr std::int64_t kPerfPageVUid = 12;
constexpr std::int64_t kPerfSeqQUid = 7;
constexpr std::int64_t kPerfSeqKvUid = 8;

/// ceil(a / b) for positive ints.
int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

/// Append one int32 tensor descriptor (uid/name/dims/strides) to the
/// growing tensor-attribute list of a graph builder.
void pushInt32Tensor(flatbuffers::FlatBufferBuilder& builder,
                     std::vector<flatbuffers::Offset<data_objects::TensorAttributes>>& tensors,
                     std::int64_t uid, const char* name, const std::vector<std::int64_t>& dims,
                     const std::vector<std::int64_t>& strides) {
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder, uid, name,
                                                                 DataType::INT32, &strides, &dims));
}

/// Build a single-buffer paged and/or varlen SDPA-fwd graph. Q/K/V/O use
/// BSHD strides; page tables ([B, maxBlocksPerSeq]) and/or seq-len tensors
/// ([B]) are added when requested, and their UIDs wired into the
/// SdpaAttributes so the adapter selects the paged / varlen path. Causal,
/// dtype-parameterized. ``maxSeqLenKv`` drives the adapter's block-size
/// derivation on the paged path.
flatbuffers::FlatBufferBuilder buildPagedVarlenSdpaGraph(DataType dtype, bool withPaged,
                                                         bool withVarlen, int maxBlocksPerSeq,
                                                         int maxSeqLenKv) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;

    const std::vector<std::int64_t> qDims{kVarB, kVarHq, kVarS, kVarD};
    const std::vector<std::int64_t> kvDims{kVarB, kVarHkv, kVarS, kVarD};
    const std::vector<std::int64_t> qStrides = bshdStrides(kVarHq, kVarS, kVarD);
    const std::vector<std::int64_t> kvStrides = bshdStrides(kVarHkv, kVarS, kVarD);

    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder, 1, "q", dtype, &qStrides, &qDims));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder, 2, "k", dtype, &kvStrides, &kvDims));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder, 3, "v", dtype, &kvStrides, &kvDims));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder, 4, "o", dtype, &qStrides, &qDims));

    flatbuffers::Optional<std::int64_t> pageKUid = flatbuffers::nullopt;
    flatbuffers::Optional<std::int64_t> pageVUid = flatbuffers::nullopt;
    if (withPaged) {
        // [num_seqs, maxBlocksPerSeq] row-major int32 block table.
        const std::vector<std::int64_t> dims{kVarB, maxBlocksPerSeq};
        const std::vector<std::int64_t> strides{maxBlocksPerSeq, 1};
        pushInt32Tensor(builder, tensors, kPerfPageKUid, "page_k", dims, strides);
        pushInt32Tensor(builder, tensors, kPerfPageVUid, "page_v", dims, strides);
        pageKUid = flatbuffers::Optional<std::int64_t>(kPerfPageKUid);
        pageVUid = flatbuffers::Optional<std::int64_t>(kPerfPageVUid);
    }

    flatbuffers::Optional<std::int64_t> seqQUid = flatbuffers::nullopt;
    flatbuffers::Optional<std::int64_t> seqKvUid = flatbuffers::nullopt;
    if (withVarlen) {
        const std::vector<std::int64_t> dims{kVarB};
        const std::vector<std::int64_t> strides{1};
        pushInt32Tensor(builder, tensors, kPerfSeqQUid, "seq_q", dims, strides);
        pushInt32Tensor(builder, tensors, kPerfSeqKvUid, "seq_kv", dims, strides);
        seqQUid = flatbuffers::Optional<std::int64_t>(kPerfSeqQUid);
        seqKvUid = flatbuffers::Optional<std::int64_t>(kPerfSeqKvUid);
    }

    flatbuffers::Optional<std::int32_t> maxSeqLenKvOpt = flatbuffers::nullopt;
    if (withPaged) {
        maxSeqLenKvOpt = flatbuffers::Optional<std::int32_t>(maxSeqLenKv);
    }

    auto sdpaAttributes = data_objects::CreateSdpaAttributes(
        builder, /*q_tensor_uid=*/1, /*k_tensor_uid=*/2, /*v_tensor_uid=*/3, /*o_tensor_uid=*/4,
        /*attn_mask_tensor_uid=*/flatbuffers::nullopt, /*scale_tensor_uid=*/flatbuffers::nullopt,
        /*seq_len_q_tensor_uid=*/seqQUid, /*seq_len_kv_tensor_uid=*/seqKvUid,
        /*seed_tensor_uid=*/flatbuffers::nullopt, /*offset_tensor_uid=*/flatbuffers::nullopt,
        /*dropout_mask_tensor_uid=*/flatbuffers::nullopt,
        /*dropout_scale_tensor_uid=*/flatbuffers::nullopt,
        /*page_table_k_tensor_uid=*/pageKUid, /*page_table_v_tensor_uid=*/pageVUid,
        /*block_mask_tensor_uid=*/flatbuffers::nullopt,
        /*sink_token_tensor_uid=*/flatbuffers::nullopt,
        /*descale_q_tensor_uid=*/flatbuffers::nullopt,
        /*descale_k_tensor_uid=*/flatbuffers::nullopt,
        /*descale_v_tensor_uid=*/flatbuffers::nullopt,
        /*descale_s_tensor_uid=*/flatbuffers::nullopt, /*scale_s_tensor_uid=*/flatbuffers::nullopt,
        /*scale_o_tensor_uid=*/flatbuffers::nullopt, /*stats_tensor_uid=*/flatbuffers::nullopt,
        /*max_tensor_uid=*/flatbuffers::nullopt, /*sum_exp_tensor_uid=*/flatbuffers::nullopt,
        /*rng_dump_tensor_uid=*/flatbuffers::nullopt, /*amax_s_tensor_uid=*/flatbuffers::nullopt,
        /*amax_o_tensor_uid=*/flatbuffers::nullopt, /*generate_stats=*/flatbuffers::nullopt,
        /*alibi_mask=*/false, /*padding_mask=*/false, /*causal_mask=*/true,
        /*causal_mask_bottom_right=*/false, /*dropout_probability=*/flatbuffers::nullopt,
        /*attn_scale_value=*/flatbuffers::nullopt, /*left_bound=*/flatbuffers::nullopt,
        /*right_bound=*/flatbuffers::nullopt, /*max_seq_len_kv=*/maxSeqLenKvOpt);

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder, "sdpa_fwd", dtype,
                                                   data_objects::NodeAttributes::SdpaAttributes,
                                                   sdpaAttributes.Union()));

    auto graphOffset = data_objects::CreateGraphDirect(
        builder, "test", DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, &tensors, &nodes);
    builder.Finish(graphOffset);
    return builder;
}

/// Variant selector for the hand-built probes.
enum class PagedVarlenKind { PagedIdentity, PagedScatter, VarlenMixed };

/// Build the host block-table contents for the paged probes. ``identity``
/// yields contiguous physical indices [0, 1, 2, ...]; otherwise a
/// DETERMINISTIC reverse permutation of the same index set (a real
/// gather pattern, reproducible run-to-run -- no RNG). Layout is
/// [num_seqs, stride] row-major; total = num_seqs * stride blocks.
std::vector<std::int32_t> makeBlockTable(int numSeqs, int stride, bool identity) {
    const int total = numSeqs * stride;
    std::vector<std::int32_t> table(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i) {
        // identity: i ; reverse permutation: (total - 1 - i). Both are
        // bijections onto [0, total), so every physical block index is
        // valid against a KV cache sized for ``total`` blocks.
        table[static_cast<std::size_t>(i)] = identity ? i : (total - 1 - i);
    }
    return table;
}

/// Build the per-sequence Q/KV lengths for the varlen probe: a
/// DETERMINISTIC mixed distribution cycling {S, S/2, S, S/4} across the
/// sequences (clamped to >= 1). Q and KV share the schedule.
std::vector<std::int32_t> makeMixedLens(int numSeqs, int maxS) {
    const std::int32_t pattern[4] = {maxS, maxS / 2, maxS, maxS / 4};
    std::vector<std::int32_t> lens(static_cast<std::size_t>(numSeqs));
    for (int i = 0; i < numSeqs; ++i) {
        const std::int32_t v = pattern[i % 4];
        lens[static_cast<std::size_t>(i)] = v > 0 ? v : 1;
    }
    return lens;
}

/// Element-type-templated body for one paged/varlen perf probe. Builds the
/// hand-rolled graph, runs buildPlan (capability gate + scoring + JIT),
/// allocates Q/K/V/O plus the page-table and/or seq-len device buffers,
/// sizes the workspace from the plan, and times execute(). PERF-ONLY: no
/// correctness compare (there is no oracle for the gather/varlen layout).
template <typename ElemT>
void runPagedVarlenProbe(const char* caseName, DataType dtype, PagedVarlenKind kind,
                         ::CkDslHandle& handle, SdpaFwdPlanBuilder& planBuilder) {
    const bool withPaged =
        kind == PagedVarlenKind::PagedIdentity || kind == PagedVarlenKind::PagedScatter;
    const bool withVarlen = kind == PagedVarlenKind::VarlenMixed;

    const int maxBlocksPerSeq = ceilDiv(kVarS, kVarBlockSize);  // = stride
    auto fbBuilder =
        buildPagedVarlenSdpaGraph(dtype, withPaged, withVarlen, maxBlocksPerSeq, kVarS);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    const std::vector<std::int64_t> qDims{kVarB, kVarHq, kVarS, kVarD};
    const std::vector<std::int64_t> kvDims{kVarB, kVarHkv, kVarS, kVarD};
    const std::vector<std::int64_t> qStrides = bshdStrides(kVarHq, kVarS, kVarD);
    const std::vector<std::int64_t> kvStrides = bshdStrides(kVarHkv, kVarS, kVarD);

    // Q/K/V are padded-to-max S contiguous tensors. For varlen this is a
    // PERF PROBE, not a packed layout: cu_seqlens carries the real lengths
    // but the buffers themselves stay padded-to-max (Sq = Skv = kVarS).
    // K/V double as the paged KV cache: sized for B*S tokens = num_seqs *
    // stride blocks, so any physical block index from makeBlockTable lands
    // in bounds.
    utilities::Tensor<ElemT> tensorQ(qDims, qStrides);
    utilities::Tensor<ElemT> tensorK(kvDims, kvStrides);
    utilities::Tensor<ElemT> tensorV(kvDims, kvStrides);
    utilities::Tensor<ElemT> tensorO(qDims, qStrides);

    constexpr unsigned kSeedQ = 0x4242u;
    constexpr unsigned kSeedK = 0x5555u;
    constexpr unsigned kSeedV = 0x6363u;
    tensorQ.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedQ);
    tensorK.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedK);
    tensorV.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedV);

    // Build the plan (drives the paged / varlen capability gate).
    flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr, 0);
    CkDslContext ctx;
    planBuilder.buildPlan(handle, graph, engineConfig, ctx);
    ASSERT_TRUE(ctx.hasValidPlan()) << "case '" << caseName << "': buildPlan produced no plan";

    const std::size_t wsBytes = ctx.plan().getWorkspaceSize(handle);
    void* workspace = nullptr;
    if (wsBytes > 0) {
        ASSERT_EQ(hipMalloc(&workspace, wsBytes), hipSuccess);
    }

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {
        {1, tensorQ.memory().deviceData()},
        {2, tensorK.memory().deviceData()},
        {3, tensorV.memory().deviceData()},
        {4, tensorO.memory().deviceData()},
    };

    // The paged path binds the graph's Page_table_K buffer DIRECTLY to the
    // block_tables slot, so the int32 block table must be a real device
    // buffer (uid 11). Page_table_V (uid 12) is bound only structurally by
    // the adapter (single-table kernel), but we register it so the buffer
    // array is complete. Both carry the same physical-index table.
    std::unique_ptr<utilities::Tensor<std::int32_t>> tensorPageK;
    std::unique_ptr<utilities::Tensor<std::int32_t>> tensorPageV;
    if (withPaged) {
        const bool identity = kind == PagedVarlenKind::PagedIdentity;
        const std::vector<std::int32_t> table = makeBlockTable(kVarB, maxBlocksPerSeq, identity);
        const std::vector<std::int64_t> dims{kVarB, maxBlocksPerSeq};
        const std::vector<std::int64_t> strides{maxBlocksPerSeq, 1};
        tensorPageK = std::make_unique<utilities::Tensor<std::int32_t>>(dims, strides);
        tensorPageV = std::make_unique<utilities::Tensor<std::int32_t>>(dims, strides);
        tensorPageK->fillWithData(table.data(), table.size() * sizeof(std::int32_t));
        tensorPageV->fillWithData(table.data(), table.size() * sizeof(std::int32_t));
        deviceBuffers.push_back({kPerfPageKUid, tensorPageK->memory().deviceData()});
        deviceBuffers.push_back({kPerfPageVUid, tensorPageV->memory().deviceData()});
    }

    // The varlen path D2H-copies the seq_len_q / seq_len_kv buffers, so
    // they must be real int32 device buffers (uids 7, 8) carrying the
    // per-sequence lengths. Q and KV share the mixed schedule.
    std::unique_ptr<utilities::Tensor<std::int32_t>> tensorSeqQ;
    std::unique_ptr<utilities::Tensor<std::int32_t>> tensorSeqKv;
    std::vector<std::int32_t> qLens;
    std::vector<std::int32_t> kLens;
    if (withVarlen) {
        qLens = makeMixedLens(kVarB, kVarS);
        kLens = qLens;  // same schedule for Q and KV
        const std::vector<std::int64_t> dims{kVarB};
        const std::vector<std::int64_t> strides{1};
        tensorSeqQ = std::make_unique<utilities::Tensor<std::int32_t>>(dims, strides);
        tensorSeqKv = std::make_unique<utilities::Tensor<std::int32_t>>(dims, strides);
        tensorSeqQ->fillWithData(qLens.data(), qLens.size() * sizeof(std::int32_t));
        tensorSeqKv->fillWithData(kLens.data(), kLens.size() * sizeof(std::int32_t));
        deviceBuffers.push_back({kPerfSeqQUid, tensorSeqQ->memory().deviceData()});
        deviceBuffers.push_back({kPerfSeqKvUid, tensorSeqKv->memory().deviceData()});
    }

    // FLOPS basis. Paged probes run uniform full-length sequences, so they
    // share the dense denominator (full, non-causal-adjusted): two GEMMs
    // each 2*B*Hq*S*S*D. The varlen probe instead sums the ACTUAL per-
    // sequence token work (4*Hq*q_len*k_len*D), since the kernel only
    // attends the cu_seqlens-declared tokens -- B*S^2 would overstate it.
    double flops = 0.0;
    if (withVarlen) {
        for (int s = 0; s < kVarB; ++s) {
            flops += 4.0 * static_cast<double>(kVarHq) *
                     static_cast<double>(qLens[static_cast<std::size_t>(s)]) *
                     static_cast<double>(kLens[static_cast<std::size_t>(s)]) *
                     static_cast<double>(kVarD);
        }
    } else {
        flops = 4.0 * static_cast<double>(kVarB) * static_cast<double>(kVarHq) *
                static_cast<double>(kVarS) * static_cast<double>(kVarS) *
                static_cast<double>(kVarD);
    }

    timeExecuteAndLog(caseName, ctx, handle, deviceBuffers, workspace, flops);

    if (workspace != nullptr) {
        ASSERT_EQ(hipFree(workspace), hipSuccess);
    }
}

/// PERF-ONLY gfx950 harness for the paged / varlen launch paths. Same
/// container/handle setup as the dense fixture; skips on non-gfx950.
class IntegrationGpuCkDslSdpaFwdPagedVarlenGpu : public ::testing::Test {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("IntegrationGpuCkDslSdpaFwdPagedVarlenGpu");

        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        _planBuilder = std::make_unique<SdpaFwdPlanBuilder>(_container->compileServiceBridge(),
                                                            _container->jitCache());
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::unique_ptr<SdpaFwdPlanBuilder> _planBuilder;
};

// (a) Paged with an identity/contiguous block table -- proves the paged
//     passthrough plumbing; perf should track dense parity.
TEST_F(IntegrationGpuCkDslSdpaFwdPagedVarlenGpu, Paged_Identity) {
    runPagedVarlenProbe<half>("Paged_Identity", DataType::HALF, PagedVarlenKind::PagedIdentity,
                              *_handle, *_planBuilder);
}

// (b) Paged with a shuffled (reverse) block table -- a real paged-gather
//     perf signal vs the identity case.
TEST_F(IntegrationGpuCkDslSdpaFwdPagedVarlenGpu, Paged_Scatter) {
    runPagedVarlenProbe<half>("Paged_Scatter", DataType::HALF, PagedVarlenKind::PagedScatter,
                              *_handle, *_planBuilder);
}

// (c) Varlen with a mixed per-sequence length distribution -- exercises
//     the seq-len D2H + cu_seqlens marshalling; FLOPS counts actual tokens.
TEST_F(IntegrationGpuCkDslSdpaFwdPagedVarlenGpu, Varlen_Mixed) {
    runPagedVarlenProbe<bfloat16>("Varlen_Mixed", DataType::BFLOAT16, PagedVarlenKind::VarlenMixed,
                                  *_handle, *_planBuilder);
}

}  // namespace
