// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmPlanBuilder.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <pybind11/embed.h>

#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <limits>
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

ConvImplicitGemmPlanBuilder::ConvImplicitGemmPlanBuilder(CompileServiceBridge& bridge,
                                                         JitCache& cache)
    : _bridge(bridge), _cache(cache) {}

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

    // Single adapter walk: the spec drives both the cache key
    // derivation and the loader closure. Captured by value into the
    // closure so the loader sees the exact same spec the byte-size
    // computation below uses -- no risk of drift from re-deriving the
    // spec on the compile path.
    ConvImplicitGemmSpec spec = ConvImplicitGemmAdapter::buildSpec(convAttr, tensorMap);

    SignatureHash key = GraphSignature::computeForSpec(opKind(), spec);

    auto loader = [spec, this]() -> KernelArtifact {
        py::gil_scoped_acquire gil;
        py::dict payload = convImplicitGemmSpecToPayload(spec);
        return _bridge.compile(opKind(), payload);
    };

    std::shared_ptr<HipModule> module = _cache.getOrLoad(key, loader);

    // Buffer-rsrc byte sizes for the kernel's free OOB-clamping args
    // (A_bytes / B_bytes / D_bytes). Computed from the spec's
    // ConvProblem geometry rather than re-walking the tensor map:
    //   X (NHWC fp16): N * Hi * Wi * C * 2
    //   W (KRSC fp16): K * R  * S  * C * 2
    //   Y (NHWK fp16): N * Ho * Wo * K * 2
    // The kernel's signature is i32 for these; the bake-off shape
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
