// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaFwdPlanBuilder.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <pybind11/embed.h>

#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../adapters/sdpa/SdpaAdapter.hpp"
#include "../../adapters/sdpa/SdpaCandidateSelector.hpp"
#include "../../adapters/sdpa/SdpaPayload.hpp"
#include "../../adapters/sdpa/SdpaPerfKnobs.hpp"
#include "../../adapters/sdpa/SdpaScorer.hpp"
#include "../../adapters/sdpa/SdpaSpec.hpp"
#include "../../graph/GraphSignature.hpp"
#include "../../python/CompileServiceBridge.hpp"
#include "../../runtime/DeviceArch.hpp"
#include "SdpaFwdPlan.hpp"

namespace py = pybind11;

namespace ck_dsl_provider {

namespace {

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

using TensorMap = SdpaAdapter::TensorMap;

/// Pull the single SdpaAttributes out of a one-node graph, or throw if
/// the graph isn't shaped that way. On the commit path (buildPlan) a
/// failure here is exceptional; the applicability gate goes through the
/// non-throwing tryBuildSpec below instead of catching this directly.
const data_objects::SdpaAttributes& getSingleSdpaFwdNode(
    const flatbuffer_utilities::IGraph& opGraph) {
    if (opGraph.nodeCount() != 1) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaFwdPlanBuilder: graph must have exactly one node; got " +
                std::to_string(opGraph.nodeCount()));
    }
    const auto& node = opGraph.getNodeWrapper(0);
    if (node.attributesType() != data_objects::NodeAttributes::SdpaAttributes) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("SdpaFwdPlanBuilder: node '") + node.name() +
                "' must be SdpaAttributes; got " +
                std::string(data_objects::EnumNameNodeAttributes(node.attributesType())));
    }
    return node.attributesAs<data_objects::SdpaAttributes>();
}

/// Non-throwing structural + dtype + shape + mask gate for
/// ``isApplicable``. Returns the spec when the graph is an SDPA this
/// provider can model, or ``std::nullopt`` (with ``reason`` set) when it
/// is not -- a multi-node graph, a non-SDPA node, an unsupported dtype /
/// shape / mask / feature, etc. "Not for us" is the normal, expected
/// answer to an applicability probe, so it is reported as a value rather
/// than an exception: the adapter's throws are caught here, once,
/// instead of at the call site.
std::optional<SdpaSpec> tryBuildSpec(const flatbuffer_utilities::IGraph& opGraph,
                                     std::string& reason) {
    try {
        const auto& sdpaAttr = getSingleSdpaFwdNode(opGraph);
        return SdpaAdapter::buildSpec(sdpaAttr, opGraph.getTensorMap());
    } catch (const hipdnn_plugin_sdk::HipdnnPluginException& e) {
        reason = e.what();
        return std::nullopt;
    }
}

/// Pick a concrete paged-KV block size for the dense (non-paged)
/// degenerate path. The unified kernel is always paged; a dense graph
/// runs the one-block-per-sequence layout, but the scorer (and the 2c
/// marshalling) still need a real block size in {16, 32, 64}. Choose the
/// largest of {64, 32, 16} that divides ``Skv`` so a single block tiles
/// the sequence cleanly; fall back to 16 (the supports() floor) when
/// none divides. Folded into the cache key via ``spec.block_size`` so a
/// scored dense plan caches distinctly from an unscored one.
std::int32_t chooseDegenerateBlockSize(std::int32_t skv) {
    for (const std::int32_t candidate : {64, 32, 16}) {
        if (skv > 0 && skv % candidate == 0) {
            return candidate;
        }
    }
    return 16;
}
void finalizeDenseBlockSize(SdpaSpec& spec) {
    if (!spec.is_paged && spec.block_size == 0) {
        spec.block_size = chooseDegenerateBlockSize(spec.problem.Skv);
    }
}


/// Normalise the spec's dtype spelling ("f16") to the kernel's
/// ("fp16") that ``SdpaSelectionProblem`` and the heuristic expect.
/// "bf16" passes through unchanged.
std::string normalizeScoringDtype(const std::string& specDtype) {
    if (specDtype == "f16") {
        return "fp16";
    }
    return specDtype;
}

/// Build the selection problem the candidate enumerator + scorer read,
/// from the (already block-size-finalised) spec.
SdpaSelectionProblem buildSelectionProblem(const SdpaSpec& spec) {
    SdpaSelectionProblem selProblem;
    selProblem.batch = spec.problem.B;
    selProblem.num_query_heads = spec.problem.Hq;
    selProblem.num_kv_heads = spec.problem.Hkv;
    selProblem.seqlen_q = spec.problem.Sq;
    selProblem.seqlen_k = spec.problem.Skv;
    selProblem.head_size = spec.problem.D;
    selProblem.block_size = spec.block_size;
    selProblem.dtype = normalizeScoringDtype(spec.dtype);

    // The unified kernel is ALWAYS paged (real paged graph or the dense
    // degenerate one-block-per-sequence layout), so use_paged_kv is the
    // honest feature value for the model regardless of spec.is_paged.
    selProblem.use_paged_kv = true;
    selProblem.use_sinks = spec.use_sinks;
    selProblem.sliding_window = spec.sliding_window;

    // The capability gate guarantees causal masking for this provider;
    // top-left causal == fmha mask_enum int 1.
    selProblem.mask_type = 1;
    selProblem.bias_type = 0;
    selProblem.skip_min_seqlen_q = false;
    return selProblem;
}

}  // namespace

SdpaFwdPlanBuilder::SdpaFwdPlanBuilder(CompileServiceBridge& bridge, JitCache& cache)
    : _bridge(bridge), _cache(cache) {}

bool SdpaFwdPlanBuilder::isApplicable(const ::CkDslHandle& handle,
                                      const flatbuffer_utilities::IGraph& opGraph) const {
    // Structural + dtype + shape + mask gate first: multi-node /
    // non-SDPA / dtype / shape / unsupported-feature rejects need
    // neither a device nor the Python interpreter. "Not for us" is a
    // normal verdict, returned as nullopt -- no exception drives this
    // path.
    std::string reason;
    std::optional<SdpaSpec> spec = tryBuildSpec(opGraph, reason);
    if (!spec.has_value()) {
        HIPDNN_PLUGIN_LOG_INFO("SdpaFwdPlanBuilder::isApplicable rejected graph: " << reason);
        return false;
    }

    // Dense graphs reach the unified paged kernel through a degenerate
    // one-block-per-sequence layout. Finalise that implicit block_size before
    // calling the DSL applicability predicate so isApplicable and buildPlan
    // validate the same payload.
    finalizeDenseBlockSize(*spec);

    // Arch + DSL-validity gate. A spec valid on gfx950 can be invalid on
    // another arch, so validate against the device arch via the DSL's
    // own predicate -- so isApplicable can never report a spec buildPlan
    // would then reject.
    //
    // Unlike the structural gate, these steps touch the device and the
    // embedded interpreter, which fail only in genuinely exceptional
    // ways (a GPU present but unreadable, or a Python/bridge fault). This
    // gate must answer only true/false and never propagate, so the
    // backstop below converts any such fault into a logged decline.
    try {
        std::optional<std::string> arch = detectDeviceArch(handle.getStream());
        if (!arch.has_value()) {
            // No visible device: the kernel cannot run here, so we
            // decline rather than claim a graph we never validated
            // against a real arch.
            HIPDNN_PLUGIN_LOG_INFO(
                "SdpaFwdPlanBuilder::isApplicable declining: no HIP device is visible");
            return false;
        }
        // arch is an orthogonal compile target, passed alongside the
        // spec payload (mirroring the DSL) rather than baked into it.
        py::gil_scoped_acquire gil;
        py::dict payload = sdpaSpecToPayload(*spec);
        std::pair<bool, std::string> verdict = _bridge.isApplicable(opKind(), payload, *arch);
        if (!verdict.first) {
            HIPDNN_PLUGIN_LOG_INFO("SdpaFwdPlanBuilder::isApplicable rejected graph for arch "
                                   << *arch << ": " << verdict.second);
        }
        return verdict.first;
    } catch (const DeviceArchDetectionError& e) {
        // A device is present but its arch can't be read -- an
        // environment fault. Surface it loudly; we still decline (fail
        // closed).
        HIPDNN_PLUGIN_LOG_ERROR(
            "SdpaFwdPlanBuilder::isApplicable could not determine the device "
            "architecture; declining: "
            << e.what());
        return false;
    } catch (const std::exception& e) {
        // Unexpected bridge / interpreter fault. Honor the never-throw
        // contract: log and decline.
        HIPDNN_PLUGIN_LOG_ERROR(
            "SdpaFwdPlanBuilder::isApplicable declining after an unexpected error: " << e.what());
        return false;
    }
}

std::size_t SdpaFwdPlanBuilder::getMaxWorkspaceSize(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& /*opGraph*/,
    const CkDslSettings& /*executionSettings*/) const {
    // The FMHA-forward kernel allocates its scratch in static LDS; no
    // host-managed workspace is needed for M1. If a future variant needs
    // an external buffer (e.g. a global-memory scratchpad for split-K
    // reductions) it surfaces here.
    return 0;
}

void SdpaFwdPlanBuilder::initializeExecutionSettings(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& /*opGraph*/,
    const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    CkDslSettings& /*executionSettings*/) const {
    // No knobs in M1; the settings struct is unused on this path.
}

void SdpaFwdPlanBuilder::buildPlan(const ::CkDslHandle& handle,
                                   const flatbuffer_utilities::IGraph& opGraph,
                                   const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
                                   CkDslContext& executionContext) const {
    const auto& sdpaAttr = getSingleSdpaFwdNode(opGraph);
    const auto& tensorMap = opGraph.getTensorMap();

    // Single adapter walk: the spec drives both the cache key derivation
    // and the loader closure. Captured by value into the closure so the
    // loader sees the exact same spec used to build the plan -- no risk
    // of drift from re-deriving the spec on the compile path.
    SdpaSpec spec = SdpaAdapter::buildSpec(sdpaAttr, tensorMap);

    // arch is an orthogonal compile target (not a spec field, mirroring
    // the DSL), threaded separately into the cache key (different arches
    // must not alias), the loader, and the bridge. Detection is
    // mandatory: a plan exists only to launch a kernel on a real device,
    // so we never guess a default (that would silently miscompile for
    // the wrong target). detectDeviceArch already throws when a device
    // is present but its arch can't be read; the check below covers the
    // remaining case of no visible device.
    std::optional<std::string> detectedArch = detectDeviceArch(handle.getStream());
    if (!detectedArch.has_value()) {
        throw DeviceArchDetectionError(
            "SdpaFwdPlanBuilder::buildPlan: no HIP device is visible, so the target "
            "architecture cannot be determined; a GPU is required to build a plan");
    }
    const std::string arch = *detectedArch;

    // --- Scorer-driven perf-knob selection ---------------------------
    // Run BEFORE computeForSpec and the loader: both read spec, and the
    // chosen knobs (plus the finalised block_size) are folded into the
    // cache key. Mutating spec here keeps the key, the loader's payload,
    // and the eventual marshalling (Task 2c) in lock-step.

    // 1. Finalise block_size for the dense path so the scorer and the
    //    cache key see a concrete value. Real paged graphs already carry
    //    a block_size from the adapter; leave those as-is.
    finalizeDenseBlockSize(spec);

    // 2. Build the selection problem from the finalised spec.
    const SdpaSelectionProblem selProblem = buildSelectionProblem(spec);

    // 3. Enumerate buildable candidates. Post-gate, at least one combo
    //    must exist; an empty set is a genuine fault on the commit path.
    const std::vector<SdpaPerfKnobs> candidates = enumerateCandidates(selProblem);
    if (candidates.empty()) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaFwdPlanBuilder::buildPlan: no buildable FMHA-forward kernel "
            "configuration for this problem (block_size=" +
                std::to_string(spec.block_size) + ", head_size=" + std::to_string(spec.problem.D) +
                "); the "
                "applicability gate should have rejected it earlier");
    }

    // 4. Select the perf knobs. The scorer is loaded once for the
    //    process (the gfx950 model is ~11 MB): a function-local static
    //    that is shared across every buildPlan call. When the model
    //    failed to load, selectPerfKnobs degrades to the analytic
    //    fallback over the same candidate set. The returned knobs carry
    //    the problem-driven sinks / sliding-window lanes (the enumerator
    //    copied them from selProblem).
    //    Thread-safety: C++11 guarantees thread-safe static init, and
    //    predict() is const. This assumes LightGBM's booster predict is
    //    reentrant under concurrent buildPlan calls -- true for the
    //    single-threaded host path; CONFIRM before any multi-threaded
    //    plan-finding (Phase 4) or guard the predict with a mutex.
    static const SdpaScorer kScorer;
    spec.knobs = selectPerfKnobs(selProblem, candidates, kScorer);

    SignatureHash key = GraphSignature::computeForSpec(opKind(), spec, arch);

    auto loader = [spec, arch, this]() -> KernelArtifact {
        py::gil_scoped_acquire gil;
        py::dict payload = sdpaSpecToPayload(spec);
        return _bridge.compile(opKind(), payload, arch);
    };

    std::shared_ptr<HipModule> module = _cache.getOrLoad(key, loader);

    // Opt-in stats (LSE) output. The unified paged kernel emits no LSE
    // (the adapter capability gate declines any stats request, so
    // spec.generate_stats is always false here and the loaded module's
    // 18-slot ABI carries no LSE_out slot). The flag + UID are still
    // passed through to the plan ctor for source compatibility; the ctor
    // rejects a stats-on plan on this path.
    const bool hasStats = spec.generate_stats;
    const std::int64_t statsUid = hasStats ? sdpaAttr.stats_tensor_uid().value() : -1;

    // The SDPA sink_token tensor uid is 20 in the graph contract; pass it
    // when sinks are in effect so execute() can bind the sink pointer
    // (otherwise -1, sink pointer is null). A literal is acceptable for
    // the POC -- the dense path the launch wires today never enables
    // sinks, so this lane is structural until the real-paged path lands.
    constexpr std::int64_t kSdpaSinkTokenUid = 20;
    const std::int64_t sinkUid = spec.use_sinks ? kSdpaSinkTokenUid : -1;

    // Runtime-input tensor UIDs execute() resolves to device buffers for
    // the paged / varlen launch paths. Mirroring the sinkUid lane: read
    // the optional uid off the attributes when the spec selects that path,
    // else -1. The paged path binds the graph's Page_table_K buffer to the
    // block_tables slot directly; the varlen path D2H-copies the
    // seq_len_q / seq_len_kv buffers to recover the per-sequence lengths.
    const std::int64_t pageTableUid =
        spec.is_paged ? sdpaAttr.page_table_k_tensor_uid().value() : -1;
    const std::int64_t seqLenQUid = spec.is_varlen ? sdpaAttr.seq_len_q_tensor_uid().value() : -1;
    const std::int64_t seqLenKvUid = spec.is_varlen ? sdpaAttr.seq_len_kv_tensor_uid().value() : -1;

    // The plan needs the module + tensor UIDs + the launch-time scalars +
    // the marshalling-path lanes (batch / block_size / paged / varlen /
    // sinks) execute() reads to build the 18-slot ABI. HipModule carries
    // the launch metadata (grid, block, ldsBytes, argSchema) captured from
    // the artifact at load time, so execute() can pack args and launch
    // without re-reading the cache.
    auto plan = std::make_unique<SdpaFwdPlan>(
        std::move(module), sdpaAttr.q_tensor_uid(), sdpaAttr.k_tensor_uid(),
        sdpaAttr.v_tensor_uid(), sdpaAttr.o_tensor_uid(), spec.problem.scale_log2, spec.problem.Sq,
        spec.problem.Skv, spec.problem.stride_q_token, spec.problem.stride_q_head,
        spec.problem.stride_k_token, spec.problem.stride_k_head, spec.problem.stride_v_token,
        spec.problem.stride_v_head, spec.problem.stride_o_token, spec.problem.stride_o_head,
        spec.problem.B, spec.block_size, spec.is_paged, spec.is_varlen, spec.use_sinks,
        pageTableUid, seqLenQUid, seqLenKvUid, sinkUid, hasStats, statsUid);

    executionContext.setPlan(std::move(plan));
}

std::vector<data_objects::KnobT> SdpaFwdPlanBuilder::getCustomKnobs(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& /*opGraph*/) const {
    return {};
}

}  // namespace ck_dsl_provider
