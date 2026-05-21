// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CkDslContainer.hpp"

#include <algorithm>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/conv_implicit_gemm/CkDslConvImplicitGemmEngine.hpp"
#include "python/EmbeddedInterpreter.hpp"

namespace ck_dsl_provider {

// Register every engine name exposed by this plugin. The 2-argument
// form keeps the identifier readable in code while pinning the
// wire-visible string the SDK hashes into an int64_t engine ID. The
// string value must never change once shipped: it is hashed (FNV-1a)
// and any rename produces a different engine ID, breaking selection
// in any host already configured to pick this engine.
HIPDNN_REGISTER_ENGINE(CK_DSL_CONV_IMPLICIT_GEMM_ENGINE, "ck_dsl_conv_implicit_gemm_engine")

const std::vector<CkDslContainer::EngineDefinition>& CkDslContainer::getEngineDefinitions() {
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        {CK_DSL_CONV_IMPLICIT_GEMM_ENGINE_ID,
         []() -> CkDslEnginePtr {
             return std::make_unique<CkDslConvImplicitGemmEngine>(
                 CK_DSL_CONV_IMPLICIT_GEMM_ENGINE_ID);
         }},
    };

    return s_engineDefinitions;
}

uint32_t CkDslContainer::copyEngineIds(int64_t* engineIds, uint32_t maxEngines,
                                       uint32_t& numEngines) {
    const auto& engineDefinitions = getEngineDefinitions();
    auto totalEngines = static_cast<uint32_t>(engineDefinitions.size());

    if (maxEngines == 0) {
        numEngines = totalEngines;
        return totalEngines;
    }

    auto enginesToCopy = std::min(maxEngines, totalEngines);
    for (uint32_t i = 0; i < enginesToCopy; ++i) {
        engineIds[i] = engineDefinitions[i].id;
    }

    numEngines = enginesToCopy;

    return totalEngines;
}

CkDslContainer::CkDslContainer() {
    HIPDNN_PLUGIN_LOG_INFO("Creating CkDslContainer");

    // The CK DSL provider drives its compile pipeline through an
    // embedded CPython interpreter (plan §3.2). This is the natural
    // per-process initialisation point: hipDNN's SharedContainerManager
    // ensures the container is constructed exactly once per process
    // (see EnginePluginImpl.inl), regardless of how many handles the
    // host creates. ensureInitialized() is also idempotent and
    // thread-safe, so repeated calls in pathological host code are
    // harmless. The plan v0.9 step I-2 wording ("CkDslHandle
    // constructs") is imprecise on this point; the container ctor is
    // the right hook for the per-process singleton.
    ck_dsl_provider::EmbeddedInterpreter::ensureInitialized();

    _engineManager = std::make_unique<
        hipdnn_plugin_sdk::EngineManager<::CkDslHandle, CkDslSettings, CkDslContext>>();

    for (const auto& engineDefinition : getEngineDefinitions()) {
        _engineManager->addEngine(engineDefinition.createEngine());
    }
}

CkDslContainer::~CkDslContainer() noexcept {
    try {
        HIPDNN_PLUGIN_LOG_INFO("Destroying CkDslContainer");
    } catch (...)  // NOLINT(bugprone-empty-catch)
    {
    }
}

hipdnn_plugin_sdk::EngineManager<::CkDslHandle, CkDslSettings, CkDslContext>&
CkDslContainer::getEngineManager() {
    return *_engineManager;
}

}  // namespace ck_dsl_provider
