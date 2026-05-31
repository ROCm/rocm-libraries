// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaBwdPlanBuilder.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <pybind11/embed.h>

#include <cstddef>
#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "../../adapters/sdpa/SdpaBwdAdapter.hpp"
#include "../../adapters/sdpa/SdpaBwdPayload.hpp"
#include "../../adapters/sdpa/SdpaBwdSpec.hpp"
#include "../../graph/GraphSignature.hpp"
#include "../../python/CompileServiceBridge.hpp"
#include "../../runtime/DeviceArch.hpp"
#include "SdpaBwdPlan.hpp"

namespace py = pybind11;

namespace ck_dsl_provider {

namespace {

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

using TensorMap = SdpaBwdAdapter::TensorMap;

/// Pull the single SdpaBackwardAttributes out of a one-node graph, or
/// throw if the graph isn't shaped that way. On the commit path
/// (buildPlan) a failure here is exceptional; the applicability gate goes
/// through the non-throwing tryBuildSpec below instead of catching this
/// directly.
const data_objects::SdpaBackwardAttributes& getSingleSdpaBwdNode(
    const flatbuffer_utilities::IGraph& opGraph) {
    if (opGraph.nodeCount() != 1) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaBwdPlanBuilder: graph must have exactly one node; got " +
                std::to_string(opGraph.nodeCount()));
    }
    const auto& node = opGraph.getNodeWrapper(0);
    if (node.attributesType() != data_objects::NodeAttributes::SdpaBackwardAttributes) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("SdpaBwdPlanBuilder: node '") + node.name() +
                "' must be SdpaBackwardAttributes; got " +
                std::string(data_objects::EnumNameNodeAttributes(node.attributesType())));
    }
    return node.attributesAs<data_objects::SdpaBackwardAttributes>();
}

/// Non-throwing structural + dtype + shape + mask gate for
/// ``isApplicable``. Returns the spec when the graph is an SDPA-backward
/// this provider can model, or ``std::nullopt`` (with ``reason`` set)
/// when it is not. "Not for us" is the normal, expected answer to an
/// applicability probe, so it is reported as a value rather than an
/// exception: the adapter's throws are caught here, once, instead of at
/// the call site.
std::optional<SdpaBwdSpec> tryBuildSpec(const flatbuffer_utilities::IGraph& opGraph,
                                        std::string& reason) {
    try {
        const auto& sdpaAttr = getSingleSdpaBwdNode(opGraph);
        return SdpaBwdAdapter::buildSpec(sdpaAttr, opGraph.getTensorMap());
    } catch (const hipdnn_plugin_sdk::HipdnnPluginException& e) {
        reason = e.what();
        return std::nullopt;
    }
}

}  // namespace

SdpaBwdPlanBuilder::SdpaBwdPlanBuilder(CompileServiceBridge& bridge, JitCache& cache)
    : _bridge(bridge), _cache(cache) {}

bool SdpaBwdPlanBuilder::isApplicable(const ::CkDslHandle& handle,
                                      const flatbuffer_utilities::IGraph& opGraph) const {
    // Structural + dtype + shape + mask gate first: multi-node /
    // non-SDPA / dtype / shape / unsupported-feature rejects need
    // neither a device nor the Python interpreter. "Not for us" is a
    // normal verdict, returned as nullopt -- no exception drives this
    // path.
    std::string reason;
    std::optional<SdpaBwdSpec> spec = tryBuildSpec(opGraph, reason);
    if (!spec.has_value()) {
        HIPDNN_PLUGIN_LOG_INFO("SdpaBwdPlanBuilder::isApplicable rejected graph: " << reason);
        return false;
    }

    // Arch + DSL-validity gate. A spec valid on gfx950 can be invalid on
    // another arch, so validate against the device arch via the DSL's
    // own predicate -- so isApplicable can never report a spec buildPlan
    // would then reject.
    //
    // Unlike the structural gate, these steps touch the device and the
    // embedded interpreter, which fail only in genuinely exceptional
    // ways. This gate must answer only true/false and never propagate, so
    // the backstop below converts any such fault into a logged decline.
    try {
        std::optional<std::string> arch = detectDeviceArch(handle.getStream());
        if (!arch.has_value()) {
            // No visible device: the kernel cannot run here, so we
            // decline rather than claim a graph we never validated
            // against a real arch.
            HIPDNN_PLUGIN_LOG_INFO(
                "SdpaBwdPlanBuilder::isApplicable declining: no HIP device is visible");
            return false;
        }
        // arch is an orthogonal compile target, passed alongside the
        // spec payload (mirroring the DSL) rather than baked into it. The
        // bwd kernel is the gating op; the LSE-prep kernel shares the
        // same arch and a strictly smaller shape, so a bwd-applicable
        // arch is prep-applicable too.
        py::gil_scoped_acquire gil;
        py::dict payload = sdpaBwdSpecToPayload(*spec);
        std::pair<bool, std::string> verdict = _bridge.isApplicable(opKind(), payload, *arch);
        if (!verdict.first) {
            HIPDNN_PLUGIN_LOG_INFO("SdpaBwdPlanBuilder::isApplicable rejected graph for arch "
                                   << *arch << ": " << verdict.second);
        }
        return verdict.first;
    } catch (const DeviceArchDetectionError& e) {
        // A device is present but its arch can't be read -- an
        // environment fault. Surface it loudly; we still decline (fail
        // closed).
        HIPDNN_PLUGIN_LOG_ERROR(
            "SdpaBwdPlanBuilder::isApplicable could not determine the device "
            "architecture; declining: "
            << e.what());
        return false;
    } catch (const std::exception& e) {
        // Unexpected bridge / interpreter fault. Honor the never-throw
        // contract: log and decline.
        HIPDNN_PLUGIN_LOG_ERROR(
            "SdpaBwdPlanBuilder::isApplicable declining after an unexpected error: " << e.what());
        return false;
    }
}

std::size_t SdpaBwdPlanBuilder::getMaxWorkspaceSize(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& opGraph,
    const CkDslSettings& /*executionSettings*/) const {
    // The bwd path needs a host-managed scratch for the two
    // per-(B, Hq, Sq) reductions (M_saved + L_saved) the LSE-prep kernel
    // writes. Build the spec to read the shape, then size the scratch the
    // same way SdpaBwdPlan::getWorkspaceSize does.
    const auto& sdpaAttr = getSingleSdpaBwdNode(opGraph);
    SdpaBwdSpec spec = SdpaBwdAdapter::buildSpec(sdpaAttr, opGraph.getTensorMap());
    return static_cast<std::size_t>(2) * static_cast<std::size_t>(spec.problem.B) *
           static_cast<std::size_t>(spec.problem.Sq) * static_cast<std::size_t>(spec.problem.Hq) *
           sizeof(float);
}

void SdpaBwdPlanBuilder::initializeExecutionSettings(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& /*opGraph*/,
    const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    CkDslSettings& /*executionSettings*/) const {
    // No knobs; the settings struct is unused on this path.
}

void SdpaBwdPlanBuilder::buildPlan(const ::CkDslHandle& handle,
                                   const flatbuffer_utilities::IGraph& opGraph,
                                   const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
                                   CkDslContext& executionContext) const {
    const auto& sdpaAttr = getSingleSdpaBwdNode(opGraph);
    const auto& tensorMap = opGraph.getTensorMap();

    // Single adapter walk: the spec drives both cache-key derivations and
    // both loader closures. Captured by value into the closures so each
    // loader sees the exact same spec used to build the plan -- no risk
    // of drift from re-deriving the spec on the compile path.
    SdpaBwdSpec spec = SdpaBwdAdapter::buildSpec(sdpaAttr, tensorMap);

    // arch is an orthogonal compile target (not a spec field, mirroring
    // the DSL), threaded separately into the cache keys (different arches
    // must not alias), the loaders, and the bridge. Detection is
    // mandatory: a plan exists only to launch a kernel on a real device,
    // so we never guess a default. detectDeviceArch already throws when a
    // device is present but its arch can't be read; the check below covers
    // the remaining case of no visible device.
    std::optional<std::string> detectedArch = detectDeviceArch(handle.getStream());
    if (!detectedArch.has_value()) {
        throw DeviceArchDetectionError(
            "SdpaBwdPlanBuilder::buildPlan: no HIP device is visible, so the target "
            "architecture cannot be determined; a GPU is required to build a plan");
    }
    const std::string arch = *detectedArch;

    // Backward kernel module.
    SignatureHash bwdKey = GraphSignature::computeForSpec(opKind(), spec, arch);
    auto bwdLoader = [spec, arch, this]() -> KernelArtifact {
        py::gil_scoped_acquire gil;
        py::dict payload = sdpaBwdSpecToPayload(spec);
        return _bridge.compile(opKind(), payload, arch);
    };
    std::shared_ptr<HipModule> bwdModule = _cache.getOrLoad(bwdKey, bwdLoader);

    // LSE-prep kernel module. Cached under a separate key folding only
    // version/opKind/arch + B/Hq/Sq, since the prep kernel is independent
    // of head_size and the kv sequence length.
    SignatureHash prepKey = GraphSignature::computeForSdpaLsePrep(prepOpKind(), spec, arch);
    auto prepLoader = [spec, arch, this]() -> KernelArtifact {
        py::gil_scoped_acquire gil;
        py::dict payload = sdpaLsePrepSpecToPayload(spec);
        return _bridge.compile(prepOpKind(), payload, arch);
    };
    std::shared_ptr<HipModule> prepModule = _cache.getOrLoad(prepKey, prepLoader);

    // The plan needs both modules + tensor UIDs + dims + scales +
    // strides. Each HipModule carries its launch metadata (grid, block,
    // ldsBytes, argSchema) captured from the artifact at load time, so
    // execute() can pack args and launch without re-reading the cache.
    auto plan = std::make_unique<SdpaBwdPlan>(
        std::move(bwdModule), std::move(prepModule), sdpaAttr.q_tensor_uid(),
        sdpaAttr.k_tensor_uid(), sdpaAttr.v_tensor_uid(), sdpaAttr.do_tensor_uid(),
        sdpaAttr.stats_tensor_uid(), sdpaAttr.dq_tensor_uid(), sdpaAttr.dk_tensor_uid(),
        sdpaAttr.dv_tensor_uid(), spec.problem.B, spec.problem.Hq, spec.problem.Hkv,
        spec.problem.Sq, spec.problem.Skv, spec.problem.D, spec.problem.scale_log2,
        spec.problem.scale_inv, spec.problem.stride_q_token, spec.problem.stride_q_head,
        spec.problem.stride_k_token, spec.problem.stride_k_head, spec.problem.stride_v_token,
        spec.problem.stride_v_head, spec.problem.stride_do_token, spec.problem.stride_do_head,
        spec.problem.stride_dq_token, spec.problem.stride_dk_token, spec.problem.stride_dv_token);

    executionContext.setPlan(std::move(plan));
}

std::vector<data_objects::KnobT> SdpaBwdPlanBuilder::getCustomKnobs(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& /*opGraph*/) const {
    return {};
}

}  // namespace ck_dsl_provider
