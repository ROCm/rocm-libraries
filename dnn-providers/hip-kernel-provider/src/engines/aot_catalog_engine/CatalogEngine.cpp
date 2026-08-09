// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CatalogEngine.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>

#include "launch/ModuleLoader.hpp"
#include "launch/PluginError.hpp"
#include "ops/GemmAdapter.hpp"
#include "ops/RmsNormAdapter.hpp"
#include "ops/SdpaAdapter.hpp"
#include "plans/CatalogPlan.hpp"

namespace aot_catalog_engine
{

CatalogEngine::CatalogEngine()
{
    // Register the op adapters this engine understands. Adding a new op kind
    // (sdpa, conv) is one push_back here plus a new adapter class.
    _adapters.push_back(std::make_unique<ops::GemmAdapter>());
    _adapters.push_back(std::make_unique<ops::RmsNormAdapter>());
    _adapters.push_back(std::make_unique<ops::SdpaAdapter>());
}

int64_t CatalogEngine::staticId()
{
    return hipdnn_data_sdk::utilities::AOT_CATALOG_ENGINE_ID;
}

int64_t CatalogEngine::id() const
{
    return staticId();
}

const catalog::Catalog& CatalogEngine::catalogForArch(const std::string& arch) const
{
    const std::lock_guard<std::mutex> lock(_catalogMutex);
    auto it = _catalogs.find(arch);
    if(it == _catalogs.end())
    {
        it = _catalogs
                 .emplace(arch,
                          catalog::Catalog::loadForDevice(catalog::defaultCatalogDir(), arch))
                 .first;
    }
    return it->second;
}

std::optional<CatalogEngine::Match> CatalogEngine::matchGraph(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    std::string arch;
    try
    {
        arch = hip_kernel_provider_common::getDeviceString(handle.getStream());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("aot-catalog: could not query device arch: " << e.what());
        return std::nullopt;
    }

    const catalog::Catalog& catalog = catalogForArch(arch);
    if(catalog.empty())
    {
        return std::nullopt;
    }

    for(const auto& adapter : _adapters)
    {
        std::optional<catalog::ProblemShape> problem = adapter->decode(opGraph);
        if(!problem.has_value())
        {
            continue;
        }

        std::vector<catalog::Catalog::Candidate> candidates
            = catalog.candidatesFor(adapter->opKind(), *problem);
        if(candidates.empty())
        {
            continue;
        }

        // Carry ALL applicable candidates so the plan can measure-and-cache the
        // fastest on the first execute (Phase 2).
        return Match{adapter.get(), std::move(*problem), std::move(candidates)};
    }

    return std::nullopt;
}

bool CatalogEngine::isApplicable(
    Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return matchGraph(handle, opGraph).has_value();
}

void CatalogEngine::getDetails(
    Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    auto engineDetails
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetailsDirect(builder, id(), nullptr);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    auto* dataPtr = detachedBuffer->data();
    handle.storeEngineDetailsDetachedBuffer(dataPtr, std::move(detachedBuffer));
}

size_t CatalogEngine::getMaxWorkspaceSize(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const
{
    const std::optional<Match> match = matchGraph(handle, opGraph);
    if(!match.has_value())
    {
        return 0;
    }
    // The winner isn't known until the first execute times the candidates, so
    // reserve the max any candidate needs (each uses <= its own <= max).
    size_t maxBytes = 0;
    for(const catalog::Catalog::Candidate& candidate : match->candidates)
    {
        maxBytes = std::max(maxBytes, candidate.kernel->workspaceBytes);
    }
    return maxBytes;
}

void CatalogEngine::initializeExecutionContext(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    Context& executionContext) const
{
    executionContext.setExecutionSettings(Settings{});

    const std::optional<Match> match = matchGraph(handle, opGraph);
    if(!match.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
            "aot-catalog: no catalog kernel matches this graph");
    }

    const ops::IOpAdapter& adapter = *match->adapter;

    // Bind modules to the device backing the handle's stream (hipModuleLoad loads
    // into the current device's context). Done once for all candidates.
    hipDevice_t deviceId = 0;
    if(const hipError_t err = hipStreamGetDevice(handle.getStream(), &deviceId); err != hipSuccess)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         std::string("aot-catalog: hipStreamGetDevice failed: ")
                             + hipGetErrorString(err));
    }

    // Switching the current device is thread-global HIP state; restore the caller's
    // device on every exit path (including throws) so we do not leave it changed on
    // a multi-GPU host. Kernel launches later target the stream's device regardless
    // of the current device, so restoring here after module load is safe.
    struct CurrentDeviceGuard
    {
        int prior = 0;
        bool valid = false;
        ~CurrentDeviceGuard()
        {
            if(valid)
            {
                (void)hipSetDevice(prior);
            }
        }
    } deviceGuard;
    deviceGuard.valid = (hipGetDevice(&deviceGuard.prior) == hipSuccess);

    if(const hipError_t err = hipSetDevice(deviceId); err != hipSuccess)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         std::string("aot-catalog: hipSetDevice failed: ")
                             + hipGetErrorString(err));
    }

    // Build one PlanCandidate per applicable kernel: load its module and resolve
    // its bindings/grid via the adapter. The plan measures-and-caches the fastest.
    std::vector<PlanCandidate> planCandidates;
    planCandidates.reserve(match->candidates.size());
    for(const catalog::Catalog::Candidate& candidate : match->candidates)
    {
        const catalog::KernelEntry& kernel = *candidate.kernel;

        std::optional<launch::HipModuleGuard> module
            = launch::loadKernelModule(kernel.coPath, kernel.symbol);
        if(!module.has_value())
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: failed to load kernel '" + kernel.symbol + "' from "
                                 + kernel.coPath);
        }

        catalog::LaunchBindings bindings = adapter.buildBindings(opGraph, match->problem, kernel);
        launch::SymbolTable gridSymbols = adapter.gridSymbols(match->problem, kernel);

        planCandidates.push_back(PlanCandidate{std::move(*module),
                                               kernel.launch,
                                               std::move(bindings),
                                               std::move(gridSymbols),
                                               kernel.workspaceBytes,
                                               kernel.symbol});
    }

    // Key the tuning decision on the family + decoded problem (family name
    // already encodes arch+dtype, so keys never collide across families).
    const std::string key
        = catalog::problemKey(match->candidates.front().family->name, match->problem);

    executionContext.setPlan(
        std::make_unique<CatalogPlan>(std::move(planCandidates), &_tuneCache, key));
}

} // namespace aot_catalog_engine
