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

#include "catalog/AotDebug.hpp"
#include "catalog/Selection.hpp"
#include "launch/ModuleLoader.hpp"
#include "launch/PluginError.hpp"
#include "ops/ActivationAdapter.hpp"
#include "ops/ConvFpropAdapter.hpp"
#include "ops/GemmAdapter.hpp"
#include "ops/LayerNormAdapter.hpp"
#include "ops/RmsNormAdapter.hpp"
#include "ops/SdpaAdapter.hpp"
#include "plans/CatalogPlan.hpp"

namespace aot_catalog_engine
{

namespace
{

// HIPDNN_AOT_DEBUG-only: an op adapter decoded the graph but no kernel matched.
// Explain per kernel why, so a KA sees exactly which constraint (or missing
// shape key) filtered their kernel out. Guarded by aotDebugEnabled() at the call
// site, so the family walk only happens when debugging.
void debugExplainNoCandidates(const std::string& opKind,
                              const catalog::ProblemShape& problem,
                              const catalog::Catalog& catalog)
{
    AOT_DEBUG("op '" << opKind << "' decoded (shape: " << catalog::describeShape(problem)
                     << ") but NO catalog kernel matched:");
    size_t familyCount = 0;
    for(const auto& family : catalog.families())
    {
        if(family.opKind != opKind)
        {
            continue;
        }
        ++familyCount;
        for(size_t i = 0; i < family.kernels.size(); ++i)
        {
            const std::string reason
                = catalog::explainMismatch(family.kernels[i].constraints, problem);
            AOT_DEBUG("  family '" << family.name << "' kernel[" << i << "] '"
                                   << family.kernels[i].symbol
                                   << "': " << (reason.empty() ? "matches (unexpected)" : reason));
        }
    }
    if(familyCount == 0)
    {
        AOT_DEBUG("  no family with op_kind '" << opKind
                                               << "' is loaded for this arch (check family.json "
                                                  "'op_kind' and that the family built).");
    }
}

// Symbol table for evaluating a kernel's workspace expression: the adapter's
// grid symbols (M, N, K, ...) plus `elem_size` when the problem's dtype is
// known. Referencing `elem_size` for an unmapped dtype fails closed in
// evalWorkspace as "undefined symbol" rather than silently using a wrong width.
launch::SymbolTable workspaceSymbols(const ops::IOpAdapter& adapter,
                                     const catalog::ProblemShape& problem,
                                     const catalog::KernelEntry& kernel)
{
    launch::SymbolTable symbols = adapter.gridSymbols(problem, kernel);
    if(const std::optional<int64_t> elemSize = catalog::elementSizeBytes(problem))
    {
        symbols.emplace("elem_size", *elemSize);
    }
    return symbols;
}

} // namespace

CatalogEngine::CatalogEngine()
{
    // Register the op adapters this engine understands. Adding a new op kind is
    // one push_back here plus a new adapter class.
    _adapters.push_back(std::make_unique<ops::GemmAdapter>());
    _adapters.push_back(std::make_unique<ops::RmsNormAdapter>());
    _adapters.push_back(std::make_unique<ops::LayerNormAdapter>());
    _adapters.push_back(std::make_unique<ops::ActivationAdapter>());
    _adapters.push_back(std::make_unique<ops::SdpaAdapter>());
    _adapters.push_back(std::make_unique<ops::ConvFpropAdapter>());
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
        // Resolve here (not inside loadForDevice) so the debug header can report
        // HOW the root was found; the per-family details print inside loadForDevice.
        const catalog::CatalogDirResolution resolution = catalog::resolveCatalogDir();
        AOT_DEBUG("resolving catalog for arch " << arch << ": root=" << resolution.dir << " ("
                                                << catalog::catalogDirSourceName(resolution.source)
                                                << ")");
        it = _catalogs.emplace(arch, catalog::Catalog::loadForDevice(resolution.dir, arch)).first;
    }
    return it->second;
}

std::optional<CatalogEngine::Match> CatalogEngine::matchGraph(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    std::string arch;
    try
    {
        arch = hip_kernel_provider_common::getDeviceString(handle.getStream());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("aot-catalog: could not query device arch: " << e.what());
        AOT_DEBUG("could not query device arch -> declining: " << e.what());
        return std::nullopt;
    }

    const catalog::Catalog& catalog = catalogForArch(arch);
    if(catalog.empty())
    {
        AOT_DEBUG("catalog for arch " << arch
                                      << " is empty -> declining (no family loaded; see the "
                                         "resolution/load lines above for the cause).");
        return std::nullopt;
    }

    bool anyDecoded = false;
    for(const auto& adapter : _adapters)
    {
        std::optional<catalog::ProblemShape> problem = adapter->decode(opGraph);
        if(!problem.has_value())
        {
            continue;
        }
        anyDecoded = true;

        std::vector<catalog::Catalog::Candidate> candidates
            = catalog.candidatesFor(adapter->opKind(), *problem);
        if(candidates.empty())
        {
            if(aotDebugEnabled())
            {
                debugExplainNoCandidates(adapter->opKind(), *problem, catalog);
            }
            continue;
        }

        // Carry ALL applicable candidates so the plan can measure-and-cache the
        // fastest on the first execute (Phase 2).
        return Match{adapter.get(), std::move(*problem), std::move(candidates)};
    }

    if(!anyDecoded)
    {
        AOT_DEBUG("no op adapter decoded this graph -> declining (op is not one of "
                  "gemm/rmsnorm/sdpa, or its attributes are unsupported).");
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
    // reserve the max any candidate needs (each uses <= its own <= max). Each
    // candidate's workspace is an expression over its grid symbols + elem_size,
    // evaluated against this problem (a bare integer is the degenerate case).
    size_t maxBytes = 0;
    for(const catalog::Catalog::Candidate& candidate : match->candidates)
    {
        const launch::SymbolTable symbols
            = workspaceSymbols(*match->adapter, match->problem, *candidate.kernel);
        const int64_t bytes = launch::evalWorkspace(candidate.kernel->workspace, symbols);
        maxBytes = std::max(maxBytes, static_cast<size_t>(bytes));
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
        // The workspace symbol table (grid symbols + elem_size) also drives grid
        // eval at launch; the extra elem_size symbol is harmless there. Evaluate
        // the workspace expression now so the plan caches a concrete byte count.
        launch::SymbolTable gridSymbols = workspaceSymbols(adapter, match->problem, kernel);
        const auto workspaceBytes
            = static_cast<size_t>(launch::evalWorkspace(kernel.workspace, gridSymbols));

        planCandidates.push_back(PlanCandidate{std::move(*module),
                                               kernel.launch,
                                               std::move(bindings),
                                               std::move(gridSymbols),
                                               workspaceBytes,
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
