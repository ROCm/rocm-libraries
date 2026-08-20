// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleFwdPlan.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file ResampleNative.cpp
 * @brief The resample forward engine's native half.
 *
 * The same shape as the layernorm pack: runtime-compiled, no variant table. The
 * builder does branch on resample_mode and padding_mode, but not to pick a different
 * source file or entry point -- both are read off the graph and forwarded as compile
 * macros (HIP_PLUGIN_RESAMPLE_MODE, HIP_PLUGIN_RESAMPLE_PADDING_MODE) alongside every
 * dim, stride, and window value, the same "derive at plan build, compile once" shape
 * as layernorm's normalized dimension and extents. The `if constexpr` branches on mode
 * live inside the kernel source, not in which kernel gets selected, so there is nothing
 * here for a KMD field to key: one kernel, one metadata tuple, empty schema.
 *
 * Preparation delegates to the existing ResampleFwdParams and ResampleFwdPlan rather
 * than restating them: the derivation is the engine, and a second copy of it against
 * the same kernel is how the two drift.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.resample_fwd.graph_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.resample_fwd.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.resample_fwd.dispatch";

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/**
 * @brief Graph-scoped applicability: one fp32-compute resample node over a tensor
 *        configuration the validator accepts.
 *
 * The whole of this engine's selection logic. It binds no tokens: preparation rebuilds
 * its parameters from the node, which is the one place that derivation lives.
 */
std::optional<BoundTokens> resampleForwardGraphMatches(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ResampleFwdAttributes
       || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }

    if(!context.graph.hasOnlySupportedAttributes(std::set<data_objects::NodeAttributes>{
           data_objects::NodeAttributes::ResampleFwdAttributes}))
    {
        return std::nullopt;
    }

    // The validator is the existing engine's tensor-configuration check, reused rather
    // than restated. It reports a refusal by throwing, which a matcher must not do.
    try
    {
        resample::ResampleValidator validator(context.graph.getTensorMap());
        validator.checkTensorConfigSupported(
            node.attributesAs<data_objects::ResampleFwdAttributes>());
    }
    catch(const std::exception& refusal)
    {
        HIPDNN_PLUGIN_LOG_INFO("resample forward declines this graph: " << refusal.what());
        return std::nullopt;
    }
    return BoundTokens{};
}

/// One kernel, so ranking has nothing to order. A heuristic is still required.
double resampleForwardScore(const MatchContext& /*context*/,
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
 * compile does. ResampleFwdPlan's compile() takes hipDeviceProp_t, so the full record is
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
            "resample forward dispatch could not query properties for device "
                + std::to_string(deviceId));
    }
    return properties;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Owns the compiled plan the existing engine would have built.
class PreparedResampleForward : public PreparedDispatch
{
public:
    explicit PreparedResampleForward(std::unique_ptr<resample::ResampleFwdPlan> plan)
        : _plan(std::move(plan))
    {
    }

    const resample::ResampleFwdPlan& plan() const
    {
        return *_plan;
    }

private:
    std::unique_ptr<resample::ResampleFwdPlan> _plan;
};

/**
 * @brief The native dispatch behind this engine's UDD.
 *
 * A thin adapter onto ResampleFwdPlan: the derivation of mode, padding, extents and
 * launch geometry from the graph is that class's compile(), and duplicating it here to
 * fit the ingestor's shape would be two implementations of one kernel's launch.
 */
class ResampleForwardDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    explicit ResampleForwardDispatchHandler(const compilation::IKernelCompiler& kernelCompiler)
        : _kernelCompiler(kernelCompiler)
    {
    }

    /// The kernel writes its output (and, for maxpool, its index) directly; no scratch.
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
        resample::ResampleFwdParams params(node.attributesAs<data_objects::ResampleFwdAttributes>(),
                                           context.graph.getTensorMap(),
                                           node.computeDataType());

        auto plan = std::make_unique<resample::ResampleFwdPlan>(std::move(params));
        plan->compile(_kernelCompiler, fullDeviceProperties(context.deviceId));
        return std::make_unique<PreparedResampleForward>(std::move(plan));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        dynamic_cast<const PreparedResampleForward&>(prepared).plan().execute(
            handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
};

/// This engine's dispatch handler. Process-lifetime: the registry holds a non-owning
/// pointer while a Container is created and destroyed per handle.
const ResampleForwardDispatchHandler& resampleForwardDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const ResampleForwardDispatchHandler s_dispatchHandler(s_kernelCompiler);
    return s_dispatchHandler;
}

} // namespace

void registerResampleSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &resampleForwardGraphMatches);
    scope.add(std::string(SCORE_SYMBOL), &resampleForwardScore);
    scope.add(std::string(DISPATCH_SYMBOL), &resampleForwardDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
