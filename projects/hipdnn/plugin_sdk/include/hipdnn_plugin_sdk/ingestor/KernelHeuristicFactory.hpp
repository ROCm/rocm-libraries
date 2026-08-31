// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

/// @file KernelHeuristicFactory.hpp
/// @brief Builds the IKernelHeuristic a UHD descriptor names.
///
/// Separate from IKernelHeuristic.hpp because the factory has to know every
/// implementation, and one of them -- UhdKernelHeuristic -- derives from the interface.
/// Keeping them in one header would make the include cycle order-dependent.
namespace hipdnn_plugin_sdk::ingestor
{

/// @param describedBy Engine named in the warning when @p descriptor is nullopt.
/// @throws std::runtime_error if a NATIVE descriptor names an unregistered symbol.
inline std::shared_ptr<IKernelHeuristic>
    makeKernelHeuristic(const std::optional<HeuristicDescriptor>& descriptor,
                        const std::string& describedBy = {})
{
    if(!descriptor.has_value())
    {
        // Warn, not fail: an engine with no model still selects deterministically. The
        // warning is the point -- it separates an engine that declares its order from
        // one still waiting on a UHD, which otherwise look identical from the outside.
        HIPDNN_PLUGIN_LOG_WARN("ingestor: " << (describedBy.empty() ? "engine" : describedBy)
                                            << " ships no heuristic; kernels rank by priority, "
                                               "then descriptor id");
        return std::make_shared<UnrankedKernelHeuristic>();
    }

    const auto named = describeDescriptor("heuristic", descriptor->name, descriptor->id);

    switch(descriptor->adapter)
    {
    case UhdAdapter::STATIC_ORDER:
        // The declared order made explicit. Distinct from shipping no heuristic only in
        // that the author said so, which is why this one does not warn.
        return std::make_shared<UnrankedKernelHeuristic>();

    case UhdAdapter::NATIVE:
        return std::make_shared<NativeKernelHeuristic>(descriptor->nativeSymbol, named);

    case UhdAdapter::TREE_DATA:
    case UhdAdapter::TABLE:
        // Where NATIVE throws, a model degrades. An unregistered symbol is a build fact
        // and the engine could never score, so there is nothing to fall back to.
        //
        // Reaching here with a *missing* artifact means no loader pre-flighted this
        // descriptor -- DescriptorLoader drops that engine before the factory sees it.
        // What is left is a file that exists and will not come up: a truncated download,
        // a model built against a different schema, a features_hash disagreeing with the
        // signature. Those are recoverable in the only sense that matters at plan build,
        // so RFC 0019 §5 applies and the engine keeps selecting by declared order.
        if(auto heuristic = UhdKernelHeuristic::tryCreate(*descriptor, named))
        {
            return heuristic;
        }
        HIPDNN_PLUGIN_LOG_ERROR("ingestor: " << named
                                             << " could not be brought up; kernels rank by "
                                                "priority, then descriptor id");
        return std::make_shared<UnrankedKernelHeuristic>();

    // Unreachable: uhdAdapterFromString rejects anything not in the enum, and a
    // descriptor built in memory gets the default. Present because -Wswitch-default
    // requires an arm even for a closed enum.
    default:
        return std::make_shared<UnrankedKernelHeuristic>();
    }
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
