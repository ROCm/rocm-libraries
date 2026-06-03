// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmPlanBuilder.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "../../adapters/conv_implicit_gemm/ConvImplicitGemmAdapter.hpp"
#include "../../adapters/conv_implicit_gemm/ConvImplicitGemmPayload.hpp"
#include "../../adapters/conv_implicit_gemm/ConvImplicitGemmSpec.hpp"
#include "../../graph/GraphSignature.hpp"
#include "../../python/CompileServiceBridge.hpp"
#include "../../runtime/DeviceArch.hpp"
#include "ConvImplicitGemmPlan.hpp"

namespace ck_dsl_provider {

namespace {

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

using TensorMap = ConvImplicitGemmAdapter::TensorMap;

/// Pull the single ConvolutionFwdAttributes out of a one-node graph,
/// or throw if the graph isn't shaped that way. Mirrors the
/// miopen-provider pattern. On the commit path (buildPlan) a failure
/// here is exceptional; the applicability gate goes through the
/// non-throwing tryBuildSpec below instead of catching this directly.
const data_objects::ConvolutionFwdAttributes& getSingleConvFwdNode(
    const flatbuffer_utilities::IGraph& opGraph) {
    if (opGraph.nodeCount() != 1) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ConvImplicitGemmPlanBuilder: graph must have exactly one node; got " +
                std::to_string(opGraph.nodeCount()));
    }
    const auto& node = opGraph.getNodeWrapper(0);
    if (node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("ConvImplicitGemmPlanBuilder: node '") + node.name() +
                "' must be ConvolutionFwdAttributes; got " +
                std::string(data_objects::EnumNameNodeAttributes(node.attributesType())));
    }
    return node.attributesAs<data_objects::ConvolutionFwdAttributes>();
}

/// Non-throwing structural + dtype + layout gate for ``isApplicable``.
/// Returns the spec when the graph is a conv this provider can model, or
/// ``std::nullopt`` (with ``reason`` set) when it is not -- a multi-node
/// graph, a non-conv node, a non-FP16 / wrong-layout / wrong-rank tensor,
/// etc. "Not for us" is the normal, expected answer to an applicability
/// probe, so it is reported as a value rather than an exception: the
/// adapter's throws are caught here, once, instead of at the call site.
///
/// The non-throwing guarantee is scoped to ``HipdnnPluginException``,
/// which is the only type the adapter raises (``buildSpec`` rejects
/// solely via ``throwBadParam``). Should the adapter ever throw another
/// type (``std::bad_alloc``, a future validation path), it escapes here
/// rather than being misclassified as "not for us"; on the applicability
/// path ``isApplicable``'s ``std::exception`` backstop still catches it
/// and fails closed.
std::optional<ConvImplicitGemmSpec> tryBuildSpec(const flatbuffer_utilities::IGraph& opGraph,
                                                 std::string& reason) {
    try {
        const auto& convAttr = getSingleConvFwdNode(opGraph);
        return ConvImplicitGemmAdapter::buildSpec(convAttr, opGraph.getTensorMap());
    } catch (const hipdnn_plugin_sdk::HipdnnPluginException& e) {
        reason = e.what();
        return std::nullopt;
    }
}

}  // namespace

ConvImplicitGemmPlanBuilder::ConvImplicitGemmPlanBuilder(CompileServiceBridge& bridge,
                                                         JitCache& cache)
    : _bridge(bridge), _cache(cache) {}

bool ConvImplicitGemmPlanBuilder::isApplicable(const ::CkDslHandle& handle,
                                               const flatbuffer_utilities::IGraph& opGraph) const {
    // Structural + dtype + layout gate first: multi-node / non-conv /
    // dtype / dims / layout rejects need neither a device nor the Python
    // interpreter. "Not for us" is a normal verdict, returned as nullopt
    // -- no exception drives this path.
    std::string reason;
    std::optional<ConvImplicitGemmSpec> spec = tryBuildSpec(opGraph, reason);
    if (!spec.has_value()) {
        HIPDNN_PLUGIN_LOG_INFO(
            "ConvImplicitGemmPlanBuilder::isApplicable rejected graph: " << reason);
        return false;
    }

    // Arch + DSL-validity gate. A spec valid on gfx950 can be invalid on
    // another arch (a wave32 target rejects the wave64 MFMA path; an atom
    // present on gfx950 may be absent on gfx942), so validate against the
    // device arch via the DSL's is_valid_spec -- the SAME predicate
    // build_implicit_gemm_conv enforces at compile time -- so isApplicable
    // can never report a spec buildPlan would then reject.
    //
    // Unlike the structural gate, these steps touch the device and the
    // embedded interpreter, which fail only in genuinely exceptional ways
    // (a GPU present but unreadable, or a Python/bridge fault). This gate
    // must answer only true/false and never propagate, so the backstop
    // below converts any such fault into a logged decline.
    try {
        std::optional<std::string> arch = detectDeviceArch(handle.getStream());
        if (!arch.has_value()) {
            // No visible device: the kernel cannot run here, so we decline
            // rather than claim a graph we never validated against a real arch.
            HIPDNN_PLUGIN_LOG_INFO(
                "ConvImplicitGemmPlanBuilder::isApplicable declining: no HIP device is visible");
            return false;
        }
        // Select the arch-dependent codegen knobs (warp-tile atom + wave
        // size) for the detected device before asking the validator: the
        // same config the compile path will use, so the applicability
        // verdict matches what buildPlan would produce. An arch with no
        // known config means the provider cannot target this device.
        if (!ConvImplicitGemmAdapter::applyArchCodegenConfig(*spec, *arch)) {
            HIPDNN_PLUGIN_LOG_INFO(
                "ConvImplicitGemmPlanBuilder::isApplicable declining: no codegen config for arch "
                << *arch);
            return false;
        }

        // arch is an orthogonal compile target, passed alongside the
        // spec payload (mirroring the DSL) rather than baked into it. The bridge
        // serialises interpreter access internally (no GIL in MicroPython).
        PayloadDict payload = convImplicitGemmSpecToPayload(*spec);
        std::pair<bool, std::string> verdict = _bridge.isApplicable(opKind(), payload, *arch);
        if (!verdict.first) {
            HIPDNN_PLUGIN_LOG_INFO(
                "ConvImplicitGemmPlanBuilder::isApplicable rejected graph for arch "
                << *arch << ": " << verdict.second);
        }
        return verdict.first;
    } catch (const DeviceArchDetectionError& e) {
        // A device is present but its arch can't be read -- an environment
        // fault. Surface it loudly; we still decline (fail closed).
        HIPDNN_PLUGIN_LOG_ERROR(
            "ConvImplicitGemmPlanBuilder::isApplicable could not determine the device "
            "architecture; declining: "
            << e.what());
        return false;
    } catch (const std::exception& e) {
        // Unexpected bridge / interpreter fault. Honor the never-throw
        // contract: log and decline.
        HIPDNN_PLUGIN_LOG_ERROR(
            "ConvImplicitGemmPlanBuilder::isApplicable declining after an unexpected error: "
            << e.what());
        return false;
    }
}

std::size_t ConvImplicitGemmPlanBuilder::getMaxWorkspaceSize(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& /*opGraph*/,
    const CkDslSettings& /*executionSettings*/) const {
    // The implicit-GEMM kernel allocates its scratch in static LDS;
    // no host-managed workspace is needed for M1. If a future variant
    // needs an external buffer (e.g. a global-memory scratchpad for
    // multi-block reductions) it surfaces here.
    return 0;
}

void ConvImplicitGemmPlanBuilder::initializeExecutionSettings(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& /*opGraph*/,
    const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    CkDslSettings& /*executionSettings*/) const {
    // No knobs in M1; the settings struct is unused on this path.
}

void ConvImplicitGemmPlanBuilder::buildPlan(
    const ::CkDslHandle& handle, const flatbuffer_utilities::IGraph& opGraph,
    const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    CkDslContext& executionContext) const {
    const auto& convAttr = getSingleConvFwdNode(opGraph);
    const auto& tensorMap = opGraph.getTensorMap();

    // Single adapter walk: the spec drives both the cache key
    // derivation and the loader closure. Captured by value into the
    // closure so the loader sees the exact same spec the byte-size
    // computation below uses -- no risk of drift from re-deriving the
    // spec on the compile path.
    ConvImplicitGemmSpec spec = ConvImplicitGemmAdapter::buildSpec(convAttr, tensorMap);

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
            "ConvImplicitGemmPlanBuilder::buildPlan: no HIP device is visible, so the target "
            "architecture cannot be determined; a GPU is required to build a plan");
    }
    const std::string arch = *detectedArch;

    // Select the arch-dependent codegen knobs (warp-tile atom + wave
    // size) for the detected device. buildSpec leaves these at the
    // struct defaults; this picks the per-arch config the DSL validates
    // for ``arch`` so the kernel is built for the device it will run on.
    // An unknown arch is fatal here -- isApplicable should already have
    // declined, so reaching buildPlan on an untargetable device is an
    // exceptional, fail-closed condition.
    if (!ConvImplicitGemmAdapter::applyArchCodegenConfig(spec, arch)) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ConvImplicitGemmPlanBuilder::buildPlan: no codegen config for device arch '" + arch +
                "'; the CK DSL provider cannot target this device");
    }

    SignatureHash key = GraphSignature::computeForSpec(opKind(), spec, arch);

    auto loader = [spec, arch, this]() -> KernelArtifact {
        PayloadDict payload = convImplicitGemmSpecToPayload(spec);
        return _bridge.compile(opKind(), payload, arch);
    };

    std::shared_ptr<HipModule> module = _cache.getOrLoad(key, loader);

    // Buffer-rsrc byte sizes for the kernel's free OOB-clamping args
    // (A_bytes / B_bytes / D_bytes). Computed from the spec's
    // ConvProblem geometry rather than re-walking the tensor map:
    //   X (NHWC fp16): N * Hi * Wi * C * 2
    //   W (KRSC fp16): K * R  * S  * C * 2
    //   Y (NHWK fp16): N * Ho * Wo * K * 2
    // The kernel's signature is i32 for these; the example shape
    // produces values well under 2^31 (~3.2 MB for X/Y, ~73 KB for W).
    // Only FP16 is supported today, so the byte multiplier is fixed;
    // when the adapter starts surfacing a dtype this will become a
    // dtype-aware lookup.
    constexpr std::int64_t kFp16BytesPerElement = 2;
    const auto& p = spec.problem;
    std::int64_t xBytes64 =
        static_cast<std::int64_t>(p.N) * p.Hi * p.Wi * p.C * kFp16BytesPerElement;
    std::int64_t wBytes64 = static_cast<std::int64_t>(p.K) * p.R * p.S * p.C * kFp16BytesPerElement;
    std::int64_t yBytes64 =
        static_cast<std::int64_t>(p.N) * p.Ho() * p.Wo() * p.K * kFp16BytesPerElement;
    if (xBytes64 > std::numeric_limits<std::int32_t>::max() ||
        wBytes64 > std::numeric_limits<std::int32_t>::max() ||
        yBytes64 > std::numeric_limits<std::int32_t>::max()) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ConvImplicitGemmPlanBuilder: tensor byte sizes exceed int32_t "
            "(the kernel signature is i32 for A_bytes/B_bytes/D_bytes); "
            "shapes this large need an ABI widening before they can be supported");
    }

    // The plan needs the module + tensor UIDs + buffer-rsrc byte
    // sizes. HipModule carries the launch metadata (grid, block,
    // ldsBytes, argSchema) captured from the artifact at load time,
    // so execute() can pack args and launch without re-reading the
    // cache.
    auto plan = std::make_unique<ConvImplicitGemmPlan>(
        std::move(module), convAttr.x_tensor_uid(), convAttr.w_tensor_uid(),
        convAttr.y_tensor_uid(), static_cast<std::int32_t>(xBytes64),
        static_cast<std::int32_t>(wBytes64), static_cast<std::int32_t>(yBytes64));

    executionContext.setPlan(std::move(plan));
}

std::vector<data_objects::KnobT> ConvImplicitGemmPlanBuilder::getCustomKnobs(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& /*opGraph*/) const {
    return {};
}

}  // namespace ck_dsl_provider
