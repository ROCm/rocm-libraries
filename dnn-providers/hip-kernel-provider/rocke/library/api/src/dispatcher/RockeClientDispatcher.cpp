// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/RockeClientDispatcher.hpp"

#include <map>
#include <string>
#include <utility>

#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "RockeClientHandle.hpp"
#include "dispatcher/SdpaGraphAdapter.hpp"
#include "dispatcher/SelectionConstraints.hpp"

namespace rocke_client::dispatcher
{

namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

// Bare gfx arch string (e.g. "gfx942") for the stream's device, or "" when no
// device is resolvable (e.g. host-only unit tests). Only ever called inside
// selectInstance's try/catch, so a std::bad_alloc from the small string build
// is handled there rather than escaping the noexcept selection path.
std::string deviceArch(hipStream_t stream)
{
    int device = 0;
    if(hipStreamGetDevice(stream, &device) != hipSuccess)
    {
        return {};
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
    {
        return {};
    }
    std::string arch = props.gcnArchName;
    const auto colon = arch.find(':'); // strip "gfx942:sramecc+:xnack-"
    if(colon != std::string::npos)
    {
        arch.resize(colon);
    }
    return arch;
}

// Emit a selection-failure warning without letting the log path throw.
void logSelectionFailure(const char* reason) noexcept
{
    try
    {
        HIPDNN_PLUGIN_LOG_WARN("rocke-client dispatcher selection failed: " << reason);
    }
    // NOLINTNEXTLINE(bugprone-empty-catch) -- a failed log must never escape noexcept
    catch(...)
    {
    }
}

} // namespace

// ---- Constructors -----------------------------------------------------------

RockeClientDispatcher::RockeClientDispatcher() = default;

RockeClientDispatcher::RockeClientDispatcher(AotCatalog catalog)
    : _injectedCatalog(std::move(catalog))
{
}

// ---- Private: per-arch catalog ---------------------------------------------

const AotCatalog& RockeClientDispatcher::catalogForArch(const std::string& arch) const
{
    const std::lock_guard<std::mutex> lock(_catalogMutex);

    // Fast path: already loaded (or found absent) for this arch.
    auto it = _catalogsByArch.find(arch);
    if(it != _catalogsByArch.end())
    {
        return it->second;
    }

    // First access for this arch: load (or use injected catalog).
    // try_emplace inserts a default AotCatalog{} first so the map entry is
    // stable before we populate it; subsequent calls return immediately above.
    const auto emplaceResult = _catalogsByArch.try_emplace(arch);
    auto& newIt = emplaceResult.first;
    if(_injectedCatalog.has_value())
    {
        newIt->second = *_injectedCatalog;
    }
    else
    {
        newIt->second = AotCatalog::loadForDevice(arch);
    }
    return newIt->second;
}

// ---- Private: core selection ------------------------------------------------

std::optional<AotInstance> RockeClientDispatcher::selectFromCatalog(const AotCatalog& catalog,
                                                                    const SdpaProblem& problem)
{
    const auto candidates = catalog.candidatesFor(problem.op, problem.arch);
    if(candidates.empty())
    {
        return std::nullopt;
    }

    const std::map<std::string, AttrValue> attributes = problem.attributes();
    for(const AotInstance& instance : candidates)
    {
        if(satisfies(instance, problem, attributes))
        {
            // First match wins (stable catalog order).
            // TODO(heuristics): when >1 instances match and a trained per-arch
            // attention model is available, break ties with the model score instead.
            return instance;
        }
    }
    return std::nullopt;
}

// ---- Public selection API ---------------------------------------------------

std::optional<AotInstance> RockeClientDispatcher::select(const SdpaProblem& problem) const
{
    // Uses the catalog keyed on problem.arch.
    // This is the test seam: unit tests inject a catalog via the ctor; the
    // injected catalog is returned for any arch query without HIP calls.
    return selectFromCatalog(catalogForArch(problem.arch), problem);
}

std::optional<AotInstance>
    RockeClientDispatcher::selectForArch(const std::string& arch,
                                         const fb::IGraph& graph) const noexcept
{
    // Test seam: bypasses HIP stream device detection.
    try
    {
        std::optional<SdpaProblem> problem = translate(graph);
        if(!problem.has_value())
        {
            return std::nullopt;
        }
        problem->arch = arch;
        return selectFromCatalog(catalogForArch(arch), *problem);
    }
    catch(const std::exception& e)
    {
        logSelectionFailure(e.what());
        return std::nullopt;
    }
    catch(...)
    {
        logSelectionFailure("unknown error");
        return std::nullopt;
    }
}

std::optional<AotInstance>
    RockeClientDispatcher::selectInstance(const RockeClientHandle& handle,
                                          const fb::IGraph& graph) const noexcept
{
    // deviceArch builds a string / calls HIP; guard it so nothing escapes this
    // noexcept function (selectForArch is itself noexcept, but the device arch
    // resolution preceding it is not).
    try
    {
        hipStream_t stream = handle.getStream();
        const std::string arch = deviceArch(stream);

        std::optional<SdpaProblem> problem = translate(graph);
        if(!problem.has_value())
        {
            return std::nullopt;
        }
        problem->arch = arch;

        return selectFromCatalog(catalogForArch(arch), *problem);
    }
    catch(...)
    {
        return std::nullopt;
    }
}

bool RockeClientDispatcher::isApplicable(const RockeClientHandle& handle,
                                         const fb::IGraph& graph) const noexcept
{
    return selectInstance(handle, graph).has_value();
}

} // namespace rocke_client::dispatcher
