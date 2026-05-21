// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CkDslContainer.hpp"

#include <algorithm>
#include <array>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>

#include "engines/conv_implicit_gemm/CkDslConvImplicitGemmEngine.hpp"
#include "python/CompileServiceBridge.hpp"
#include "python/EmbeddedInterpreter.hpp"

namespace ck_dsl_provider {

// Register every engine name exposed by this plugin. The 2-argument
// form keeps the identifier readable in code while pinning the
// wire-visible string the SDK hashes into an int64_t engine ID. The
// string value must never change once shipped: it is hashed (FNV-1a)
// and any rename produces a different engine ID, breaking selection
// in any host already configured to pick this engine.
HIPDNN_REGISTER_ENGINE(CK_DSL_CONV_IMPLICIT_GEMM_ENGINE, "ck_dsl_conv_implicit_gemm_engine")

namespace {

// One entry per registered engine. Wrapped in a function-local static
// so the array is initialised on first call rather than at global
// static-init time -- HIPDNN_REGISTER_ENGINE fills in the engine ID
// via a runtime static initialiser, and a namespace-level constexpr
// or const array would capture the pre-init zero. The
// function-local static defers initialisation until after all global
// constructors have run, by which time the macro has populated the
// ID.
const std::array<int64_t, 1>& engineIds() {
    static const std::array<int64_t, 1> ids = {CK_DSL_CONV_IMPLICIT_GEMM_ENGINE_ID};
    return ids;
}

}  // namespace

CkDslEnginePtr CkDslContainer::createEngine(int64_t id) const {
    if (id == CK_DSL_CONV_IMPLICIT_GEMM_ENGINE_ID) {
        return std::make_unique<CkDslConvImplicitGemmEngine>(id, *_compileServiceBridge);
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "CkDslContainer::createEngine: no engine registered for id=" + std::to_string(id));
}

uint32_t CkDslContainer::copyEngineIds(int64_t* engineIdsOut, uint32_t maxEngines,
                                       uint32_t& numEngines) {
    const auto& ids = engineIds();
    auto totalEngines = static_cast<uint32_t>(ids.size());

    if (maxEngines == 0) {
        numEngines = totalEngines;
        return totalEngines;
    }

    auto enginesToCopy = std::min(maxEngines, totalEngines);
    for (uint32_t i = 0; i < enginesToCopy; ++i) {
        engineIdsOut[i] = ids[i];
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

    // Bring up the Python compile-service bridge before any engines are
    // registered. Per plan §3.1 the bridge is the single boundary
    // through which the JIT pipeline calls into ck_dsl; constructing it
    // here means the import + sys.path injection happen exactly once
    // per process and any failure surfaces with a clear container-ctor
    // stack trace rather than deep inside an engine call.
    _compileServiceBridge = std::make_unique<CompileServiceBridge>();

    _engineManager = std::make_unique<
        hipdnn_plugin_sdk::EngineManager<::CkDslHandle, CkDslSettings, CkDslContext>>();

    for (int64_t id : engineIds()) {
        _engineManager->addEngine(createEngine(id));
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

CompileServiceBridge& CkDslContainer::compileServiceBridge() {
    if (!_compileServiceBridge) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED,
            "CkDslContainer::compileServiceBridge() called before construction completed");
    }
    return *_compileServiceBridge;
}

}  // namespace ck_dsl_provider
