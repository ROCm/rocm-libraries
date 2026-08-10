// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// The one, op-agnostic plan for the AOT catalog engine. Holds one or more loaded
// .co candidates (module + static launch metadata + adapter-resolved runtime
// bindings/grid-symbols) for a single decoded problem, and at execute() time
// packs the kernarg buffer via LaunchAbi and fires hipModuleLaunchKernel.
//
// Phase 2 measure-and-cache selection: when a plan carries more than one
// applicable candidate, the first execute() for a given problem times every
// candidate on the real device buffers, records the fastest symbol in the
// TuneCache, and always launches the winner last so the caller sees its output.
// Subsequent executes (same problem key) skip straight to the cached winner.
// Our kernels have pure-overwrite output semantics (each fully recomputes the
// output from unchanged inputs), so timing candidates on the real output buffer
// is safe.
//
// Every op (matmul, rmsnorm now; sdpa/conv later) shares this plan -- the
// op-specific knowledge lives entirely in the adapter that built the bindings.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "catalog/CatalogTypes.hpp"
#include "catalog/TuneCache.hpp"
#include "core/Handle.hpp"
#include "launch/LaunchAbi.hpp"
#include "launch/ModuleLoader.hpp"

namespace aot_catalog_engine
{

// One executable candidate: a loaded module plus everything LaunchAbi needs to
// pack args and evaluate the grid for it.
struct PlanCandidate
{
    launch::HipModuleGuard module;
    catalog::LaunchMetadata launch;
    catalog::LaunchBindings bindings;
    launch::SymbolTable gridSymbols;
    size_t workspaceBytes = 0;
    std::string symbol;
};

class CatalogPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    // Single-candidate plan (no tuning): keeps the direct-substrate parity tests
    // and any single-kernel family launching exactly as before.
    CatalogPlan(launch::HipModuleGuard module,
                catalog::LaunchMetadata launchMetadata,
                catalog::LaunchBindings bindings,
                launch::SymbolTable gridSymbols,
                size_t workspaceBytes,
                std::string kernelName);

    // Multi-candidate plan: measure-and-cache selection across `candidates`
    // keyed by `problemKey` in `cache`. `cache` may be null (always re-tune).
    CatalogPlan(std::vector<PlanCandidate> candidates,
                catalog::TuneCache* cache,
                std::string problemKey);

    ~CatalogPlan() override = default;

    // Move-only: HipModuleGuard owns the module(s).
    CatalogPlan(const CatalogPlan&) = delete;
    CatalogPlan& operator=(const CatalogPlan&) = delete;
    CatalogPlan(CatalogPlan&&) noexcept = default;
    CatalogPlan& operator=(CatalogPlan&&) noexcept = default;

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    // Pack args + eval grid + hipModuleLaunchKernel for a single candidate.
    // Throws PluginError on a launch failure. Static: depends only on its args.
    static void launchCandidate(const PlanCandidate& candidate,
                                const Handle& handle,
                                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                uint32_t numDeviceBuffers,
                                void* workspace);

    // Time every candidate on the real buffers, record the fastest in the cache,
    // and return its index. Candidates that error while timing are skipped.
    size_t tuneAndSelect(const Handle& handle,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         void* workspace) const;

    std::vector<PlanCandidate> _candidates;
    catalog::TuneCache* _cache = nullptr; // not owned
    std::string _problemKey;
};

} // namespace aot_catalog_engine
