// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "catalog/Catalog.hpp"
#include "catalog/TuneCache.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "ops/IOpAdapter.hpp"

// The AOT catalog engine: a deliberately short-term, throwaway path that loads
// loose rocKE-authored .co/HSACO kernels described by data-only family.json
// files and dispatches hipDNN op-graphs to them. Kernel authors add or update
// kernels by dropping a .co plus editing JSON -- no C++ change for a supported
// op. See docs/CATALOG_FORMAT.md.
//
// Selection is measure-and-cache (Phase 2): a matched graph carries every
// applicable kernel into the plan, and the first execute() for a given problem
// times them on the real device buffers, caches the fastest, and launches it.
namespace aot_catalog_engine
{

using IEngine = hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>;

class CatalogEngine : public IEngine
{
public:
    CatalogEngine();

    static int64_t staticId();

    static const char* engineName()
    {
        return hipdnn_data_sdk::utilities::AOT_CATALOG_ENGINE_NAME;
    }

    int64_t id() const override;

    bool isApplicable(
        Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(Handle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    size_t getMaxWorkspaceSize(const Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void initializeExecutionContext(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Context& executionContext) const override;

private:
    // A decoded, catalog-matched graph: which adapter owns it, the decoded
    // problem shape, and every applicable (family, kernel) candidate. All
    // candidates share one family (the first adapter with a non-empty match).
    struct Match
    {
        const ops::IOpAdapter* adapter = nullptr;
        catalog::ProblemShape problem;
        std::vector<catalog::Catalog::Candidate> candidates;
    };

    // Query the device arch, decode the graph with each adapter, and return the
    // first adapter whose op has one or more applicable catalog kernels, with
    // ALL of them. nullopt if the graph is unsupported or the catalog is empty.
    std::optional<Match>
        matchGraph(const Handle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const;

    // Lazily load (once) and return the catalog for `arch`.
    const catalog::Catalog& catalogForArch(const std::string& arch) const;

    std::vector<std::unique_ptr<ops::IOpAdapter>> _adapters;

    mutable std::mutex _catalogMutex;
    mutable std::map<std::string, catalog::Catalog> _catalogs;

    // Process-lifetime measure-and-cache store, shared across all plans this
    // engine builds (persisted to JSON per its own env/temp resolution).
    mutable catalog::TuneCache _tuneCache;
};

} // namespace aot_catalog_engine
