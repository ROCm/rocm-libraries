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
/// @param knobs The UED's declared knobs, checked against the model's `$kernel.*` axes
///              (RFC 0019 §6.3 check 2).
inline std::shared_ptr<IKernelHeuristic>
    makeKernelHeuristic(const std::optional<HeuristicDescriptor>& descriptor,
                        const std::string& describedBy = {},
                        const std::vector<std::string>& knobs = {},
                        const std::map<std::string, HeuristicDescriptor>& byArch = {})
{
    if(!descriptor.has_value())
    {
        // No `default` model, but the UED may still name models per architecture. RFC 0019
        // §8.3's first step is the exact gcnArchName, so those have to be reachable -- and they
        // were not: this returned before ever looking at byArch, discarding the whole map and
        // ranking by declared order even on the architectures the engine had a model for.
        //
        // The arch is unknown here, by construction (see DescriptorLoader), so this builds a
        // resolver that consults the map at first rank(), when a device exists.
        if(!byArch.empty())
        {
            return UhdKernelHeuristic::makeArchResolver(byArch, describedBy, knobs);
        }

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
    case UhdAdapter::CUSTOM_LIBRARY:
        // Where NATIVE throws, a model degrades. An unregistered symbol is a build fact
        // and the engine could never score, so there is nothing to fall back to; an
        // unloadable model is a deployment fact, and RFC 0019 §5 wants the engine still
        // selecting, by declared order.
        //
        // `knobs` and `byArch` carry RFC 0019 §6.3 check 2 and §8.3 respectively: the
        // first rejects a model whose axes are not the engine's exposed knobs, the second
        // lets a UHD re-resolve against the running device when this descriptor does not
        // describe it.
        if(auto heuristic = UhdKernelHeuristic::tryCreate(*descriptor, named, knobs, byArch))
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
