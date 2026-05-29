// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <functional>
#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <memory>
#include <vector>

#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "CkDslSettings.hpp"

namespace ck_dsl_provider {

class CompileServiceBridge;
class JitCache;

/// Type alias for engine pointers used during registration.
using CkDslEnginePtr =
    std::unique_ptr<hipdnn_plugin_sdk::IEngine<::CkDslHandle, CkDslSettings, CkDslContext>>;

/// Container class that owns engine instantiations for the CK DSL
/// provider plugin.
///
/// Lifetime: hipDNN's ``SharedContainerManager`` keeps a
/// ``std::weak_ptr`` to this container; the strong reference lives on
/// each plugin handle. The container is therefore alive for as long as
/// at least one handle exists, and is reconstructed when a handle is
/// created after the previous generation's last handle was released.
/// Long-lived process-wide state (the embedded interpreter; the JIT
/// module cache) must NOT live on the container, since that would
/// throw it away on handle-generation cycling. Such state lives on the
/// container as a reference to a process-static side table, defined in
/// the implementation file.
///
/// Engine set is declared once in the implementation file
/// (``engineDefinitions()``) as a ``(id, factory)`` table read by both
/// ``copyEngineIds`` (for the SDK's engine-listing C ABI) and
/// ``createEngine`` (for the per-handle construction path). Sibling
/// per-op engines join that table without changing this class.
class CkDslContainer {
   public:
    CkDslContainer();
    ~CkDslContainer() noexcept;

    CkDslContainer(const CkDslContainer&) = delete;
    CkDslContainer& operator=(const CkDslContainer&) = delete;
    CkDslContainer(CkDslContainer&&) = delete;
    CkDslContainer& operator=(CkDslContainer&&) = delete;

    /// Copy engine IDs into a buffer.
    /// If maxEngines == 0: does not copy, only queries total count.
    /// If maxEngines > 0: copies up to maxEngines IDs into engineIds
    /// and sets numEngines to the number copied.
    /// Returns the total number of available engines.
    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    hipdnn_plugin_sdk::EngineManager<::CkDslHandle, CkDslSettings, CkDslContext>&
    getEngineManager();

    /// Access to the per-process Python compile-service bridge. The
    /// bridge is constructed in the container ctor after the embedded
    /// interpreter is up; it owns the cached compile_service module
    /// import and the GIL plumbing used by the JIT path. Throws if the
    /// container was not fully constructed.
    CompileServiceBridge& compileServiceBridge();

    /// Access to the process-wide JIT module cache. Shared across
    /// every container generation so kernels compiled by an earlier
    /// generation are still hits after a handle-cycle.
    JitCache& jitCache();

   private:
    /// Per-engine constructor invoked from the container ctor. Engines
    /// take non-owning references to the container's
    /// ``CompileServiceBridge`` and to the process-wide ``JitCache``;
    /// they outlive both for the duration of the container's lifetime.
    CkDslEnginePtr createEngine(int64_t id) const;

    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<::CkDslHandle, CkDslSettings, CkDslContext>>
        _engineManager;
    std::unique_ptr<CompileServiceBridge> _compileServiceBridge;
    // Non-owning reference into a process-static JitCache; ownership is
    // documented in the .cpp.
    JitCache* _jitCache{nullptr};
};

}  // namespace ck_dsl_provider
