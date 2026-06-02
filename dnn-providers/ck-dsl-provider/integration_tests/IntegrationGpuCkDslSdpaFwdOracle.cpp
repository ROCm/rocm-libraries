// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <pybind11/embed.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CkDslContainer.hpp"
#include "CkDslHandle.hpp"
#include "adapters/sdpa/SdpaAdapter.hpp"
#include "adapters/sdpa/SdpaCandidateSelector.hpp"
#include "adapters/sdpa/SdpaPayload.hpp"
#include "adapters/sdpa/SdpaPerfKnobs.hpp"
#include "adapters/sdpa/SdpaScorer.hpp"
#include "adapters/sdpa/SdpaSpec.hpp"
#include "engines/sdpa/SdpaFwdPlan.hpp"
#include "engines/sdpa/SdpaFwdPlanBuilder.hpp"
#include "perf/PerfMeasurement.hpp"
#include "python/CompileServiceBridge.hpp"
#include "runtime/DeviceArch.hpp"
#include "runtime/HipModule.hpp"
#include "runtime/KernelArtifact.hpp"
#include "tests/TestUtils.hpp"

namespace py = pybind11;

namespace {

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace utilities = hipdnn_data_sdk::utilities;
using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::HipModule;
using ck_dsl_provider::KernelArtifact;
using ck_dsl_provider::PerfMeasurement;
using ck_dsl_provider::PerfResult;
using ck_dsl_provider::SdpaAdapter;
using ck_dsl_provider::SdpaFwdPlan;
using ck_dsl_provider::SdpaFwdPlanBuilder;
using ck_dsl_provider::SdpaPerfKnobs;
using ck_dsl_provider::SdpaScorer;
using ck_dsl_provider::SdpaSelectionProblem;
using ck_dsl_provider::SdpaSpec;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/// One LARGE forward-SDPA shape the oracle sweep brute-forces. All causal,
/// BSHD. The flagship case is the S8192 GQA shape where the heuristic's
/// TFLOPS plateaued at ~0.31x of a roofline expectation -- the sweep
/// answers whether that plateau is the heuristic's pick or a config-space
/// ceiling. ``dtype`` selects fp16 vs bf16 (bf16/d64/gqa8 is the model's
/// in-family regime, where a trained-faithful query should rank best).
struct OracleCase {
    const char* name;
    int B, Hq, Hkv, Sq, Skv, D;
    data_objects::DataType dtype;
};

/// BSHD physical strides for a logical [B, H, S, D] tensor -- identical to
/// the helper in the perf/correctness tests. batch = S*H*D, head = D,
/// token/seq = H*D, d = 1.
std::vector<std::int64_t> bshdStrides(int H, int S, int D) {
    return {static_cast<std::int64_t>(S) * H * D, D, static_cast<std::int64_t>(H) * D, 1};
}

/// Largest of {64, 32, 16} that divides Skv; fall back to 16. This
/// REPLICATES ``chooseDegenerateBlockSize`` from SdpaFwdPlanBuilder.cpp's
/// anonymous namespace (not accessible from a test TU). For Skv=8192 this
/// resolves to 64; for Skv=2048 it also resolves to 64.
std::int32_t chooseDegenerateBlockSizeLocal(std::int32_t skv) {
    for (const std::int32_t candidate : {64, 32, 16}) {
        if (skv > 0 && skv % candidate == 0) {
            return candidate;
        }
    }
    return 16;
}

/// Normalise "f16" -> "fp16" (the kernel/scorer spelling); "bf16" passes
/// through. REPLICATES ``normalizeScoringDtype`` from the plan builder.
std::string normalizeScoringDtypeLocal(const std::string& specDtype) {
    if (specDtype == "f16") {
        return "fp16";
    }
    return specDtype;
}

/// Build the ``SdpaSelectionProblem`` from a block-size-finalised spec.
/// This REPLICATES ``buildSelectionProblem`` from SdpaFwdPlanBuilder.cpp's
/// anonymous namespace verbatim (use_paged_kv=true because the unified
/// kernel is always paged, mask_type=1 top-left causal, bias_type=0).
SdpaSelectionProblem buildSelectionProblemLocal(const SdpaSpec& spec) {
    SdpaSelectionProblem selProblem;
    selProblem.batch = spec.problem.B;
    selProblem.num_query_heads = spec.problem.Hq;
    selProblem.num_kv_heads = spec.problem.Hkv;
    selProblem.seqlen_q = spec.problem.Sq;
    selProblem.seqlen_k = spec.problem.Skv;
    selProblem.head_size = spec.problem.D;
    selProblem.block_size = spec.block_size;
    selProblem.dtype = normalizeScoringDtypeLocal(spec.dtype);
    selProblem.use_paged_kv = true;
    selProblem.use_sinks = spec.use_sinks;
    selProblem.sliding_window = spec.sliding_window;
    selProblem.mask_type = 1;
    selProblem.bias_type = 0;
    selProblem.skip_min_seqlen_q = false;
    return selProblem;
}

/// Format a knob combo's scored axes + schedule flags into a compact
/// human-readable tuple for the per-config log + summary lines.
std::string formatKnobs(const SdpaPerfKnobs& k) {
    std::ostringstream os;
    os << "nw=" << k.num_warps << ",mw=" << k.block_m_per_warp << ",t=" << k.tile_size
       << ",mfma32=" << (k.use_mfma_32x32 ? 1 : 0)
       << ",trqk=" << (k.use_transposed_qk_32x32 ? 1 : 0)
       << ",regpv=" << (k.use_register_pv ? 1 : 0)
       << ",earlyv=" << (k.use_early_v_schedule ? 1 : 0);
    return os.str();
}

/// Equality over the scored knob axes + schedule flags. ``SdpaPerfKnobs``
/// has no ``operator==``; the sweep compares only the fields the
/// enumerator varies (the scored axes + curated schedule flags) so the
/// heuristic's pick can be matched back to its swept TFLOPS.
bool knobsScoredEqual(const SdpaPerfKnobs& a, const SdpaPerfKnobs& b) {
    return a.num_warps == b.num_warps && a.block_m_per_warp == b.block_m_per_warp &&
           a.tile_size == b.tile_size && a.use_mfma_32x32 == b.use_mfma_32x32 &&
           a.use_transposed_qk_32x32 == b.use_transposed_qk_32x32 &&
           a.use_register_pv == b.use_register_pv &&
           a.use_early_v_schedule == b.use_early_v_schedule;
}

/// Compile + time ONE candidate config end to end on the device and return
/// its achieved TFLOPS (median-based). Returns std::nullopt when the
/// candidate fails to compile, build a module, allocate workspace, or
/// launch -- the caller logs + skips it without aborting the sweep.
///
/// ``container``/``handle``/``arch`` provide the compile bridge + stream +
/// target arch. ``deviceBuffers`` is the shared {Q,K,V,O} buffer array
/// allocated once by the caller. ``flops`` is the full FMHA-fwd FLOPS for
/// the shape. ``mUs`` is filled with the median microseconds on success.
std::optional<double> timeCandidate(CkDslContainer& container, ::CkDslHandle& handle,
                                    const std::string& arch, const SdpaSpec& baseSpec,
                                    const SdpaPerfKnobs& candidate,
                                    const std::vector<hipdnnPluginDeviceBuffer_t>& deviceBuffers,
                                    double flops, double& mUs) {
    SdpaSpec cfgSpec = baseSpec;
    cfgSpec.knobs = candidate;

    // Compile this exact config. sdpaSpecToPayload + the bridge both need
    // the GIL (the payload allocates py objects). Wrap in try/catch so a
    // config the DSL declines to build is a logged skip, not a sweep abort.
    KernelArtifact artifact;
    try {
        py::gil_scoped_acquire gil;
        py::dict payload = ck_dsl_provider::sdpaSpecToPayload(cfgSpec);
        artifact =
            container.compileServiceBridge().compile(SdpaFwdPlanBuilder::opKind(), payload, arch);
    } catch (const std::exception& e) {
        HIPDNN_PLUGIN_LOG_INFO("[Oracle] candidate ("
                               << formatKnobs(candidate)
                               << ") compile FAILED, skipping: " << e.what());
        return std::nullopt;
    }

    std::shared_ptr<HipModule> module;
    try {
        module = std::make_shared<HipModule>(artifact);
    } catch (const std::exception& e) {
        HIPDNN_PLUGIN_LOG_INFO("[Oracle] candidate ("
                               << formatKnobs(candidate)
                               << ") module load FAILED, skipping: " << e.what());
        return std::nullopt;
    }

    // Dense degenerate path: blockSize is the finalised dense block size;
    // isPaged/isVarlen/useSinks all false (mirrors buildPlan's dense ctor
    // call). Tensor UIDs match createValidSdpaFwdGraph: q=1, k=2, v=3, o=4.
    SdpaFwdPlan plan(module, /*qUid=*/1, /*kUid=*/2, /*vUid=*/3, /*oUid=*/4,
                     baseSpec.problem.scale_log2, baseSpec.problem.Sq, baseSpec.problem.Skv,
                     baseSpec.problem.stride_q_token, baseSpec.problem.stride_q_head,
                     baseSpec.problem.stride_k_token, baseSpec.problem.stride_k_head,
                     baseSpec.problem.stride_v_token, baseSpec.problem.stride_v_head,
                     baseSpec.problem.stride_o_token, baseSpec.problem.stride_o_head,
                     baseSpec.problem.B, baseSpec.block_size, /*isPaged=*/false, /*isVarlen=*/false,
                     /*useSinks=*/false, /*pageTableUid=*/-1, /*seqLenQUid=*/-1,
                     /*seqLenKvUid=*/-1, /*sinkUid=*/-1);

    const std::size_t wsBytes = plan.getWorkspaceSize(handle);
    void* workspace = nullptr;
    if (wsBytes > 0) {
        if (hipMalloc(&workspace, wsBytes) != hipSuccess) {
            HIPDNN_PLUGIN_LOG_INFO("[Oracle] candidate (" << formatKnobs(candidate)
                                                          << ") workspace hipMalloc FAILED, "
                                                             "skipping");
            return std::nullopt;
        }
    }

    std::optional<double> tflops;
    try {
        // One warm launch outside the timing loop; surfaces a launch fault
        // as a throw so we skip rather than time a broken config.
        plan.execute(handle, deviceBuffers.data(), static_cast<std::uint32_t>(deviceBuffers.size()),
                     workspace);
        if (hipDeviceSynchronize() != hipSuccess) {
            throw std::runtime_error("hipDeviceSynchronize after warm launch failed");
        }

        PerfMeasurement pm;
        auto launchFn = [&plan, &handle, &deviceBuffers, workspace]() {
            plan.execute(handle, deviceBuffers.data(),
                         static_cast<std::uint32_t>(deviceBuffers.size()), workspace);
        };
        PerfResult result = pm.measure(launchFn, flops, handle.getStream());
        mUs = result.medianUs;
        tflops = result.tflops;
    } catch (const std::exception& e) {
        HIPDNN_PLUGIN_LOG_INFO("[Oracle] candidate ("
                               << formatKnobs(candidate)
                               << ") launch/measure FAILED, skipping: " << e.what());
        tflops = std::nullopt;
    }

    if (workspace != nullptr) {
        (void)hipFree(workspace);
    }
    return tflops;
}

/// Brute-force the entire enumerated candidate config space for one shape:
/// compile + time EACH candidate, report the max TFLOPS (the ORACLE best),
/// and compare it to the config the dispatcher heuristic actually selects.
template <typename ElemT>
void runOracleSweepImpl(const OracleCase& cse, CkDslContainer& container, ::CkDslHandle& handle,
                        const std::string& arch) {
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

    // 1. Build a single-op causal fp16 SDPA-fwd graph -> GraphWrapper ->
    //    the SdpaAttributes node + tensor map -> the spec. UIDs from
    //    createValidSdpaFwdGraph: q=1, k=2, v=3, o=4.
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims, qStrides, kDims, kStrides, vDims, vStrides, oDims, oStrides,
        /*dataType=*/cse.dtype, /*withAttnMask=*/false, /*withScale=*/false,
        /*withStats=*/false, /*alibiMask=*/false, /*paddingMask=*/false, /*causalMask=*/true);
    flatbuffer_utilities::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    ASSERT_EQ(graph.nodeCount(), 1u) << "case '" << cse.name << "': expected a one-node graph";
    const auto& node = graph.getNodeWrapper(0);
    ASSERT_EQ(node.attributesType(), data_objects::NodeAttributes::SdpaAttributes)
        << "case '" << cse.name << "': node is not SdpaAttributes";
    const auto& sdpaAttr = node.attributesAs<data_objects::SdpaAttributes>();

    SdpaSpec spec = SdpaAdapter::buildSpec(sdpaAttr, graph.getTensorMap());

    // 2. Finalise the dense block_size exactly as buildPlan does (the
    //    adapter leaves it 0 on the dense path). Skv=8192 -> 64.
    if (!spec.is_paged && spec.block_size == 0) {
        spec.block_size = chooseDegenerateBlockSizeLocal(spec.problem.Skv);
    }
    spec.is_paged = false;
    spec.is_varlen = false;

    // 3. Build the selection problem inline (replicated from the plan
    //    builder's anonymous-namespace helper).
    const SdpaSelectionProblem selProblem = buildSelectionProblemLocal(spec);

    // 4. Enumerate every buildable candidate for this problem.
    const std::vector<SdpaPerfKnobs> candidates = ck_dsl_provider::enumerateCandidates(selProblem);
    ASSERT_FALSE(candidates.empty())
        << "case '" << cse.name << "': enumerateCandidates returned no buildable combos";

    // 5. Allocate Q/K/V/O device tensors ONCE (small random fill; no
    //    readback -- this is perf only). Reading deviceData() drives the
    //    H->D copy for inputs; O is written by the kernel.
    utilities::Tensor<ElemT> tensorQ(qDims, qStrides);
    utilities::Tensor<ElemT> tensorK(kDims, kStrides);
    utilities::Tensor<ElemT> tensorV(vDims, vStrides);
    utilities::Tensor<ElemT> tensorO(oDims, oStrides);
    constexpr unsigned kSeedQ = 0x4242u;
    constexpr unsigned kSeedK = 0x5555u;
    constexpr unsigned kSeedV = 0x6363u;
    tensorQ.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedQ);
    tensorK.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedK);
    tensorV.fillWithRandomValues(ElemT(-0.1f), ElemT(0.1f), kSeedV);

    const std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {
        {1, tensorQ.memory().deviceData()},
        {2, tensorK.memory().deviceData()},
        {3, tensorV.memory().deviceData()},
        {4, tensorO.memory().deviceData()},
    };

    // FMHA-forward FLOPS: two GEMMs (QK^T and PV), each 2*B*Hq*Sq*Skv*D.
    // Full, non-causal-adjusted -- matches the perf/correctness tests so
    // the numbers are comparable across harnesses.
    const double kFlops = 4.0 * static_cast<double>(kB) * static_cast<double>(kHq) *
                          static_cast<double>(kSq) * static_cast<double>(kSkv) *
                          static_cast<double>(kD);

    HIPDNN_PLUGIN_LOG_INFO("[Oracle] " << cse.name << " sweeping " << candidates.size()
                                       << " candidates (block_size=" << spec.block_size
                                       << ", D=" << kD << ", Sq=Skv=" << kSq << ")");

    // 6. Sweep: compile + time EVERY candidate. Track the best (max
    //    TFLOPS). Per-config log line carries the knobs + tflops + median.
    double bestTflops = -1.0;
    double bestUs = 0.0;
    SdpaPerfKnobs bestKnobs{};
    std::size_t sweptOk = 0;
    std::size_t sweptFail = 0;
    for (const SdpaPerfKnobs& candidate : candidates) {
        double medianUs = 0.0;
        std::optional<double> tflops = timeCandidate(container, handle, arch, spec, candidate,
                                                     deviceBuffers, kFlops, medianUs);
        if (!tflops.has_value()) {
            ++sweptFail;
            continue;
        }
        ++sweptOk;
        HIPDNN_PLUGIN_LOG_INFO("[Oracle] cfg (" << formatKnobs(candidate) << ") tflops=" << *tflops
                                                << " median_us=" << medianUs);
        if (*tflops > bestTflops) {
            bestTflops = *tflops;
            bestUs = medianUs;
            bestKnobs = candidate;
        }
    }

    ASSERT_GT(sweptOk, 0u) << "case '" << cse.name
                           << "': every candidate failed to compile/launch (" << sweptFail
                           << " failures); cannot determine an oracle best";

    // 7. The heuristic's pick. Construct the scorer (loads the gfx950
    //    LightGBM model) and select over the SAME candidate set. Then
    //    match the picked knobs back to a swept result by scored-field
    //    equality; if it was a swept FAILURE (didn't compile/launch), time
    //    it explicitly here so the comparison still has a number.
    SdpaScorer scorer;
    const SdpaPerfKnobs picked = ck_dsl_provider::selectPerfKnobs(selProblem, candidates, scorer);
    if (!scorer.isLoaded()) {
        HIPDNN_PLUGIN_LOG_INFO(
            "[Oracle] WARNING: scorer model not loaded -- selectPerfKnobs used the ANALYTIC "
            "fallback, not the ML heuristic");
    }

    double heuristicTflops = -1.0;
    double heuristicUs = 0.0;
    bool matched = false;
    // Re-run the sweep bookkeeping to find the picked combo's TFLOPS. The
    // per-config results were only logged, not stored, so re-time the
    // picked combo directly (one extra compile+time) -- this both covers
    // the "picked combo failed during the sweep" case and avoids holding
    // every PerfResult in memory.
    {
        double medianUs = 0.0;
        std::optional<double> tflops =
            timeCandidate(container, handle, arch, spec, picked, deviceBuffers, kFlops, medianUs);
        if (tflops.has_value()) {
            heuristicTflops = *tflops;
            heuristicUs = medianUs;
            matched = true;
        }
    }
    if (!matched) {
        HIPDNN_PLUGIN_LOG_INFO("[Oracle] WARNING: heuristic-picked combo ("
                               << formatKnobs(picked)
                               << ") failed to compile/launch; heuristic tflops unavailable");
    }

    // Sanity: the picked combo must be one the enumerator emitted.
    bool pickInCandidates = false;
    for (const SdpaPerfKnobs& candidate : candidates) {
        if (knobsScoredEqual(candidate, picked)) {
            pickInCandidates = true;
            break;
        }
    }
    EXPECT_TRUE(pickInCandidates) << "case '" << cse.name << "': heuristic pick ("
                                  << formatKnobs(picked)
                                  << ") is not in the enumerated candidate set";

    // 7b. The ANALYTIC fallback's pick (the non-ML baseline). Timing it
    //     here closes the dispatcher-vs-analytic A/B: does the ML heuristic
    //     beat the explicit analytic ordering on real hardware? Same
    //     compile+time path as the heuristic pick.
    const SdpaPerfKnobs analytic = ck_dsl_provider::selectAnalyticFallback(selProblem, candidates);
    double analyticTflops = -1.0;
    double analyticUs = 0.0;
    bool analyticMatched = false;
    {
        double medianUs = 0.0;
        std::optional<double> tflops =
            timeCandidate(container, handle, arch, spec, analytic, deviceBuffers, kFlops, medianUs);
        if (tflops.has_value()) {
            analyticTflops = *tflops;
            analyticUs = medianUs;
            analyticMatched = true;
        }
    }

    // 8. SUMMARY line. ratio = oracle/heuristic (>1 means the heuristic
    //    left perf on the table; ~1 means the plateau is a config-space /
    //    kernel ceiling, not the heuristic's fault).
    std::ostringstream summary;
    summary << "[Oracle] " << cse.name << " candidates=" << candidates.size()
            << " swept_ok=" << sweptOk << " swept_fail=" << sweptFail << " best=" << bestTflops
            << " tflops @(" << formatKnobs(bestKnobs) << ") median_us=" << bestUs;
    if (matched) {
        const double ratio = heuristicTflops > 0.0 ? (bestTflops / heuristicTflops) : 0.0;
        summary << " heuristic=" << heuristicTflops << " tflops @(" << formatKnobs(picked)
                << ") median_us=" << heuristicUs << " oracle/heuristic=" << ratio << "x";
    } else {
        summary << " heuristic=UNAVAILABLE @(" << formatKnobs(picked) << ")";
    }
    if (analyticMatched) {
        const double ratioA = analyticTflops > 0.0 ? (bestTflops / analyticTflops) : 0.0;
        const double hVsA = (analyticTflops > 0.0 && heuristicTflops > 0.0)
                                ? (heuristicTflops / analyticTflops)
                                : 0.0;
        summary << " analytic=" << analyticTflops << " tflops @(" << formatKnobs(analytic)
                << ") median_us=" << analyticUs << " oracle/analytic=" << ratioA
                << "x heuristic/analytic=" << hVsA << "x";
    } else {
        summary << " analytic=UNAVAILABLE @(" << formatKnobs(analytic) << ")";
    }
    HIPDNN_PLUGIN_LOG_INFO(summary.str());
    // Also to stdout so the summary survives even with plugin logging off.
    std::cout << summary.str() << std::endl;
}

/// Dispatch the templated sweep on the case dtype (HALF -> half,
/// BFLOAT16 -> bfloat16), mirroring the correctness test's dispatcher.
void runOracleSweep(const OracleCase& cse, CkDslContainer& container, ::CkDslHandle& handle,
                    const std::string& arch) {
    switch (cse.dtype) {
        case data_objects::DataType::HALF:
            runOracleSweepImpl<half>(cse, container, handle, arch);
            break;
        case data_objects::DataType::BFLOAT16:
            runOracleSweepImpl<bfloat16>(cse, container, handle, arch);
            break;
        default:
            FAIL() << "unsupported dtype for oracle case '" << cse.name << "'";
    }
}

/// gfx950-gated oracle perf-sweep fixture. Brings up the embedded
/// interpreter (CkDslContainer) + a handle once per test.
class IntegrationGpuCkDslSdpaFwdOracleGpu : public ::testing::Test {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950("IntegrationGpuCkDslSdpaFwdOracleGpu");
        _container = std::make_unique<CkDslContainer>();
        _handle = std::make_unique<::CkDslHandle>();
        std::optional<std::string> arch = ck_dsl_provider::detectDeviceArch(_handle->getStream());
        ASSERT_TRUE(arch.has_value()) << "a device is present but its arch could not be detected";
        _arch = *arch;
    }

    std::unique_ptr<CkDslContainer> _container;
    std::unique_ptr<::CkDslHandle> _handle;
    std::string _arch;
};

// Flagship: the S8192 GQA fp16 D128 case (the ~0.31x-ratio plateau).
TEST_F(IntegrationGpuCkDslSdpaFwdOracleGpu, OracleConfigSweep) {
    const OracleCase cse{
        "Fp16_GQA_S8192_D128", /*B=*/1,      /*Hq=*/32, /*Hkv=*/8,
        /*Sq=*/8192,           /*Skv=*/8192, /*D=*/128, data_objects::DataType::HALF};
    runOracleSweep(cse, *_container, *_handle, _arch);
}

// Contrast: a smaller S2048 GQA fp16 D128 shape, same sweep machinery.
TEST_F(IntegrationGpuCkDslSdpaFwdOracleGpu, OracleConfigSweepS2048) {
    const OracleCase cse{
        "Fp16_GQA_S2048_D128", /*B=*/1,      /*Hq=*/32, /*Hkv=*/8,
        /*Sq=*/2048,           /*Skv=*/2048, /*D=*/128, data_objects::DataType::HALF};
    runOracleSweep(cse, *_container, *_handle, _arch);
}

// IN-FAMILY: bf16 / D64 / Hq64-Hkv8 (GQA ratio 8) -- the regime the gfx950
// fwd model was trained on. After the trained-faithful scoring query, this
// is where the heuristic pick should track oracle-best most closely.
TEST_F(IntegrationGpuCkDslSdpaFwdOracleGpu, OracleConfigSweepInFamilyBf16S2048) {
    const OracleCase cse{"Bf16_InFamily_GQA8_D64_S2048",
                         /*B=*/2,
                         /*Hq=*/64,
                         /*Hkv=*/8,
                         /*Sq=*/2048,
                         /*Skv=*/2048,
                         /*D=*/64,
                         data_objects::DataType::BFLOAT16};
    runOracleSweep(cse, *_container, *_handle, _arch);
}

// IN-FAMILY at large S: does the in-distribution pick still hold at S8192?
TEST_F(IntegrationGpuCkDslSdpaFwdOracleGpu, OracleConfigSweepInFamilyBf16S8192) {
    const OracleCase cse{"Bf16_InFamily_GQA8_D64_S8192",
                         /*B=*/2,
                         /*Hq=*/64,
                         /*Hkv=*/8,
                         /*Sq=*/8192,
                         /*Skv=*/8192,
                         /*D=*/64,
                         data_objects::DataType::BFLOAT16};
    runOracleSweep(cse, *_container, *_handle, _arch);
}

}  // namespace
