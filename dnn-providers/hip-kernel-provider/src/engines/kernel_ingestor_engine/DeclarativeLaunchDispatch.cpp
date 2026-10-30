// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledLaunch.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IRunnableKernel.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file DeclarativeLaunchDispatch.cpp
 * @brief The one dispatch handler behind every declarative UDD: it launches the
 *        formulas a pack authors as data, with no native code of the pack's own.
 *
 * The loader gives a declarative UDD the reserved symbol `hipdnn.declarative_launch`, and
 * the state manager attaches the UDD's compiled launch to each kernel of the pack. This
 * handler evaluates that launch at prepare() and launches the kernel with the argument
 * list the formulas resolved to. Kernels are prebuilt only: an embedded_source kernel is
 * refused, because nothing here supplies source-compilation inputs.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

/// The compiled launch of @p kernel.
///
/// @throws HipdnnPluginException with INVALID_VALUE when @p kernel carries none, which
///         means its pack's UDD is not declarative but names this handler anyway.
const CompiledLaunch& requireLaunch(const KernelDefinition& kernel)
{
    if(kernel.launch == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            describeDescriptor("kernel", kernel.name, kernel.kernelId)
                + " reached the declarative launch handler without a compiled launch; only a "
                  "kernel of a pack whose UDD is declarative can be launched by '"
                + std::string(DECLARATIVE_LAUNCH_SYMBOL) + "'");
    }
    return *kernel.launch;
}

/// The argument list @p launch passes, as a kernel signature: what a saved kernel's
/// recorded signature is checked against at restore. Matches CompiledLaunch::signature()
/// for the launch that resolved it.
std::vector<KernelArgument> signatureOf(const ResolvedLaunch& launch)
{
    std::vector<KernelArgument> signature;
    signature.reserve(launch.arguments.size());
    for(const auto& argument : launch.arguments)
    {
        signature.push_back(KernelArgument{
            argument.kind == LaunchArgumentKind::SCALAR ? "by_value" : "global_buffer",
            argument.size(),
            0,
            ""});
    }
    return signature;
}

/// The kernel code plus the launch its formulas resolved to. Owns the scalar argument
/// bytes the launch points at, and nothing that points back into the plan.
class PreparedDeclarativeLaunch : public IngestorPreparedDispatch
{
public:
    PreparedDeclarativeLaunch(IngestorKernelCode code, ResolvedLaunch launch)
        : _code(std::move(code))
        , _launch(std::move(launch))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    /// The kernel for the device this dispatch is running on. Resolved here rather than
    /// at prepare() because a plan outlives the handle it was built from.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const ResolvedLaunch& launch() const
    {
        return _launch;
    }

private:
    IngestorKernelCode _code;
    ResolvedLaunch _launch;
};

// Applies the resolved geometry to `code` and builds the prepared dispatch. A plan built
// from a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePreparedDeclarativeLaunch(IngestorKernelCode code,
                                                                ResolvedLaunch launch)
{
    code.setBlockSize(launch.block[0], launch.block[1], launch.block[2]);
    code.setGridSize(launch.grid[0], launch.grid[1], launch.grid[2]);
    code.setSharedMemBytes(launch.sharedMemBytes);
    return std::make_unique<PreparedDeclarativeLaunch>(std::move(code), std::move(launch));
}

/**
 * @brief Launches a kernel from its pack's declarative UDD.
 *
 * Every formula is evaluated once, at prepare(). execute() only maps tensor uids to
 * buffers and launches, so nothing mutates once prepared and concurrent execution is
 * safe. A formula that fails for a shape throws BAD_PARAM, which plan building treats as
 * "this kernel does not serve this graph".
 */
class DeclarativeLaunchDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kpackLoader Must outlive this handler; both are process-lifetime.
    explicit DeclarativeLaunchDispatchHandler(const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        return requireLaunch(kernel).workspaceBytes(context, bound, kernel);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto& compiled = requireLaunch(kernel);
        auto launch = compiled.resolve(context, bound, kernel);
        auto code = buildIngestorKernelCode(_kpackLoader, context, kernel, compiled.signature());
        return makePreparedDeclarativeLaunch(std::move(code), std::move(launch));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedLaunch = dynamic_cast<const PreparedDeclarativeLaunch&>(prepared);
        const auto& arguments = preparedLaunch.launch().arguments;
        const size_t count = arguments.size();

        // One allocation: the first half holds each pointer argument's value, the second
        // half the kernel parameter array, which points at those values or at the scalar
        // bytes the prepared dispatch owns.
        std::vector<void*> storage(2 * count, nullptr);
        void** values = storage.data();
        void** params = storage.data() + count;
        for(size_t i = 0; i < count; ++i)
        {
            const auto& argument = arguments[i];
            switch(argument.kind)
            {
            case LaunchArgumentKind::POINTER:
                values[i] = hipdnn_plugin_sdk::findDeviceBuffer(
                                argument.tensorUid, deviceBuffers, numDeviceBuffers)
                                .ptr;
                params[i] = static_cast<void*>(&values[i]);
                break;
            case LaunchArgumentKind::WORKSPACE:
                values[i] = workspace;
                params[i] = static_cast<void*>(&values[i]);
                break;
            case LaunchArgumentKind::SCALAR:
                // The kernel reads the bytes; the launch API takes a non-const pointer.
                params[i] = const_cast<std::byte*>(argument.bytes.data());
                break;
            default:
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "declarative launch argument " + std::to_string(i) + " has unknown kind "
                        + std::to_string(static_cast<int64_t>(argument.kind)));
            }
        }

        preparedLaunch.kernelForStream(handle.getStream())
            .launchWithParams(handle.getStream(), params);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& preparedLaunch = dynamic_cast<const PreparedDeclarativeLaunch&>(prepared);
        return SavedLaunchInputs{std::string(DECLARATIVE_LAUNCH_SYMBOL_V1),
                                 saveResolvedLaunch(preparedLaunch.launch())};
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        requireLaunchContract(inputs, DECLARATIVE_LAUNCH_SYMBOL_V1);

        // Read first: the signature the saved kernel is checked against derives from the
        // resolved arguments.
        ResolvedLaunch launch;
        try
        {
            launch = restoreResolvedLaunch(inputs.values);
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
        {
            if(error.getStatus() != HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
            {
                throw;
            }
            serialization::refuseIngestorPlan(
                serialization::IngestorPlanRefusal::INCOMPATIBLE,
                "the launch values of dispatch '" + inputs.dispatchSymbol
                    + "' do not form a declarative launch: " + error.getMessage());
        }

        const auto signature = signatureOf(launch);
        auto restored = restoreIngestorLaunch<ResolvedLaunch>(
            inputs,
            std::move(code),
            signature,
            [&launch](const SavedLaunchInputs& /*inputs*/) { return std::move(launch); },
            deviceOrdinal);
        return makePreparedDeclarativeLaunch(std::move(restored.code), std::move(restored.launch));
    }

    bool supportsSaving() const override
    {
        return true;
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& declarativeLaunchKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetDeclarativeLaunchModuleCache()
{
    declarativeLaunchKpackModuleCache().clear();
}

namespace
{

/// The handler, process-lifetime: the registry holds a non-owning pointer to it, but a
/// provider's Container is created and destroyed per handle, so it (and the loader it
/// holds) must outlive every Container.
const DeclarativeLaunchDispatchHandler& declarativeLaunchDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(declarativeLaunchKpackModuleCache());
    static const DeclarativeLaunchDispatchHandler s_dispatchHandler(s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerDeclarativeLaunchSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(DECLARATIVE_LAUNCH_SYMBOL), &declarativeLaunchDispatchHandler());
    scope.add(std::string(DECLARATIVE_LAUNCH_SYMBOL_V1), &declarativeLaunchDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
