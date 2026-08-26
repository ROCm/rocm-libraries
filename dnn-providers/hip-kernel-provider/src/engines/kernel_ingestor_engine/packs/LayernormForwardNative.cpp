// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormFwdPlan.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file LayernormForwardNative.cpp
 * @brief The layernorm forward engine's native half.
 *
 * The opposite shape to the ASM SDPA pack, and worth having both. That engine had a real
 * variant table and the conversion moved it wholesale into descriptor data. This one has
 * no table at all: there is exactly one kernel, its block size is a constant, and its
 * outer, inner and stride extents are computed from the graph's dims and stride order at
 * plan build. So its KMD is empty, its pack has one entry, and it lists no kernel-scoped
 * matcher, because with one kernel and no metadata there is nothing to prune on.
 *
 * What that leaves is an engine whose descriptors carry its *identity* and none of its
 * behaviour. The value is real but narrower than the SDPA case: the engine appears,
 * disappears and is disabled by files, and its native surface is three functions rather
 * than an IEngine, IPlanBuilder and IPlan triple. Nothing about how it selects or
 * launches became data, because it never had a choice to describe.
 *
 * Preparation delegates to the existing LayernormFwdParams and LayernormFwdPlan rather
 * than restating them: the derivation is the engine, and a second copy of it against the
 * same kernel is how the two drift.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.layernorm_fwd.graph_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.layernorm_fwd.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.layernorm_fwd.dispatch";

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/**
 * @brief Graph-scoped applicability: one fp32-compute layernorm node over a tensor
 *        configuration the validator accepts.
 *
 * The whole of this engine's selection logic. It binds no tokens: preparation rebuilds
 * its parameters from the node, which is the one place that derivation lives.
 */
std::optional<BoundTokens> layernormForwardGraphMatches(const MatchContext& context)
{
    // Execute-time override shapes can diverge from the dims the kernel is compiled for,
    // and those dims are baked into the launch (RFC 0008 §4.6).
    if(context.graph.getGraph().is_override_shape_enabled() || context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::LayernormAttributes
       || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }

    // The validator is the existing engine's tensor-configuration check, reused rather
    // than restated. It reports a refusal by throwing, which a matcher must not do.
    try
    {
        layernorm::LayernormValidator validator(context.graph.getTensorMap());
        validator.checkTensorConfigSupported(
            node.attributesAs<data_objects::LayernormAttributes>());
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO("layernorm forward declines this graph: " << refusal.what());
        return std::nullopt;
    }
    return BoundTokens{};
}

/// One kernel, so ranking has nothing to order. A heuristic is still required.
double layernormForwardScore(const MatchContext& /*context*/,
                             const BoundTokens& /*bound*/,
                             const KernelDefinition& /*kernel*/)
{
    return 0.0;
}

/**
 * @brief The full HIP properties of @p deviceId.
 *
 * The ingestor's own DeviceProperties is a deliberately narrow three-field view -- arch
 * name, warp size, CU count -- which is everything matching needs and less than a kernel
 * compile does. LayernormFwdPlan's compile() takes hipDeviceProp_t, so the full record is
 * queried here rather than widening what every matcher is handed.
 *
 * @throws HipdnnPluginException if the device cannot be queried.
 */
hipDeviceProp_t fullDeviceProperties(DeviceId deviceId)
{
    hipDeviceProp_t properties{};
    if(deviceId == NO_DEVICE || hipGetDeviceProperties(&properties, deviceId) != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "layernorm forward dispatch could not query properties for device "
                + std::to_string(deviceId));
    }
    return properties;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Owns the compiled plan the existing engine would have built.
class PreparedLayernormForward : public PreparedDispatch
{
public:
    explicit PreparedLayernormForward(std::unique_ptr<layernorm::LayernormFwdPlan> plan)
        : _plan(std::move(plan))
    {
    }

    const layernorm::LayernormFwdPlan& plan() const
    {
        return *_plan;
    }

private:
    std::unique_ptr<layernorm::LayernormFwdPlan> _plan;
};

/**
 * @brief The native dispatch behind this engine's UDD.
 *
 * A thin adapter onto LayernormFwdPlan: the derivation of extents, element types and
 * launch geometry from the graph is that class's compile(), and duplicating it here to
 * fit the ingestor's shape would be two implementations of one kernel's launch.
 */
class LayernormForwardDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    explicit LayernormForwardDispatchHandler(const compilation::IKernelCompiler& kernelCompiler)
        : _kernelCompiler(kernelCompiler)
    {
    }

    /// The kernel reduces in LDS and writes its outputs in place.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        const auto& node = context.graph.getNodeWrapper(0);
        layernorm::LayernormFwdParams params(node.attributesAs<data_objects::LayernormAttributes>(),
                                             context.graph.getTensorMap());

        auto plan = std::make_unique<layernorm::LayernormFwdPlan>(std::move(params));
        plan->compile(_kernelCompiler, fullDeviceProperties(context.deviceId));
        return std::make_unique<PreparedLayernormForward>(std::move(plan));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        dynamic_cast<const PreparedLayernormForward&>(prepared).plan().execute(
            handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
};

/// This engine's dispatch handler. Process-lifetime: the registry holds a non-owning
/// pointer while a Container is created and destroyed per handle.
const LayernormForwardDispatchHandler& layernormForwardDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const LayernormForwardDispatchHandler s_dispatchHandler(s_kernelCompiler);
    return s_dispatchHandler;
}

} // namespace

void registerLayernormForwardSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &layernormForwardGraphMatches);
    scope.add(std::string(SCORE_SYMBOL), &layernormForwardScore);
    scope.add(std::string(DISPATCH_SYMBOL), &layernormForwardDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
