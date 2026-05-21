// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmPlanBuilder.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <pybind11/embed.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "../../adapters/conv_implicit_gemm/ConvImplicitGemmAdapter.hpp"
#include "../../adapters/conv_implicit_gemm/ConvImplicitGemmPayload.hpp"
#include "../../adapters/conv_implicit_gemm/ConvImplicitGemmSpec.hpp"
#include "../../graph/GraphSignature.hpp"
#include "../../python/CompileServiceBridge.hpp"
#include "ConvImplicitGemmPlan.hpp"

namespace py = pybind11;

namespace ck_dsl_provider {

namespace {

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

using TensorMap = ConvImplicitGemmAdapter::TensorMap;

/// Pull the single ConvolutionFwdAttributes out of a one-node graph,
/// or throw if the graph isn't shaped that way. Mirrors the
/// miopen-provider pattern; throws are converted to a false return by
/// the isApplicable caller.
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

}  // namespace

ConvImplicitGemmPlanBuilder::ConvImplicitGemmPlanBuilder(CompileServiceBridge& bridge)
    : _bridge(bridge) {}

bool ConvImplicitGemmPlanBuilder::isApplicable(const ::CkDslHandle& /*handle*/,
                                               const flatbuffer_utilities::IGraph& opGraph) const {
    // Cheap structural check first so we don't pay for adapter
    // validation on obviously-wrong graphs (multi-node, non-conv).
    try {
        const auto& convAttr = getSingleConvFwdNode(opGraph);
        // Full adapter validation: dtype, dims, spatial size, etc.
        // The adapter's reject paths cover everything M1 cares about.
        (void)ConvImplicitGemmAdapter::buildSpec(convAttr, opGraph.getTensorMap());
        return true;
    } catch (const std::exception& e) {
        HIPDNN_PLUGIN_LOG_INFO(
            "ConvImplicitGemmPlanBuilder::isApplicable rejected graph: " << e.what());
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
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& opGraph,
    const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    CkDslContext& executionContext) const {
    const auto& convAttr = getSingleConvFwdNode(opGraph);
    const auto& tensorMap = opGraph.getTensorMap();

    // Build the spec once -- both the cache key derivation (via the
    // same input attrs) and the loader (which payloads it) need the
    // same view. We don't share the spec object between them because
    // the loader runs lazily inside JitCache::getOrLoad and may not
    // run at all on a cache hit.
    ConvImplicitGemmSpec spec = ConvImplicitGemmAdapter::buildSpec(convAttr, tensorMap);

    SignatureHash key = GraphSignature::computeForConvFwd(opKind(), convAttr, tensorMap);

    auto loader = [&]() -> KernelArtifact {
        // Re-derive the spec inside the loader so a future change to
        // the adapter (e.g. autotuning that mutates spec by graph) is
        // visible on the compile path. For M1 the second buildSpec is
        // cheap and runs at most once per signature.
        ConvImplicitGemmSpec specForCompile =
            ConvImplicitGemmAdapter::buildSpec(convAttr, tensorMap);
        py::gil_scoped_acquire gil;
        py::dict payload = convImplicitGemmSpecToPayload(specForCompile);
        return _bridge.compile(opKind(), payload);
    };

    std::shared_ptr<HipModule> module = _cache.getOrLoad(key, loader);

    // The plan only needs the module + tensor UIDs: HipModule carries
    // the launch metadata (grid, block, ldsBytes, argSchema) captured
    // from the artifact at load time, so execute() can pack args and
    // launch without re-reading the cache.
    auto plan =
        std::make_unique<ConvImplicitGemmPlan>(std::move(module), convAttr.x_tensor_uid(),
                                               convAttr.w_tensor_uid(), convAttr.y_tensor_uid());

    executionContext.setPlan(std::move(plan));

    // `spec` is built above for both the cache loader and as a
    // forward-compat hook for I-9's perf-measurement (which will read
    // the FLOPS-derived shape from the spec). For I-7 we silence the
    // unused warning explicitly so the intent is clear.
    (void)spec;
}

std::vector<data_objects::KnobT> ConvImplicitGemmPlanBuilder::getCustomKnobs(
    const ::CkDslHandle& /*handle*/, const flatbuffer_utilities::IGraph& /*opGraph*/) const {
    return {};
}

}  // namespace ck_dsl_provider
