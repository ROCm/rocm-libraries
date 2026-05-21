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

/// Type alias for engine pointers used during registration.
using CkDslEnginePtr =
    std::unique_ptr<hipdnn_plugin_sdk::IEngine<::CkDslHandle, CkDslSettings, CkDslContext>>;

/// Container class that owns engine instantiations for the CK DSL
/// provider plugin. Constructed once per plugin handle (shared across
/// handles via std::shared_ptr).
///
/// For I-1 it registers a single engine
/// (CkDslConvImplicitGemmEngine). Sibling per-op engines
/// (CkDslGemmEngine, CkDslAttentionEngine, ...) join the
/// s_engineDefinitions vector in M5 without refactoring this class.
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

   private:
    struct EngineDefinition {
        int64_t id;
        std::function<CkDslEnginePtr()> createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<::CkDslHandle, CkDslSettings, CkDslContext>>
        _engineManager;
};

}  // namespace ck_dsl_provider
