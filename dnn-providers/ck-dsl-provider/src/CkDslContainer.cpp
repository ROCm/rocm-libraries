// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CkDslContainer.hpp"

#include <algorithm>
#include <functional>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>
#include <vector>

#include "engines/conv_implicit_gemm/CkDslConvImplicitGemmEngine.hpp"
#include "python/CompileServiceBridge.hpp"
#include "python/EmbeddedInterpreter.hpp"
#include "runtime/JitCache.hpp"

namespace ck_dsl_provider {

// Register every engine name exposed by this plugin. The 2-argument
// form keeps the identifier readable in code while pinning the
// wire-visible string the SDK hashes into an int64_t engine ID. The
// string value must never change once shipped: it is hashed (FNV-1a)
// and any rename produces a different engine ID, breaking selection
// in any host already configured to pick this engine.
HIPDNN_REGISTER_ENGINE(CK_DSL_CONV_IMPLICIT_GEMM_ENGINE, "ck_dsl_conv_implicit_gemm_engine")

namespace {

/// Single source of truth for the engine set: id + factory together.
/// ``copyEngineIds`` reads the ids; ``createEngine`` reads the
/// factories; both walk this same vector. Adding a sibling engine is
/// one entry here -- no edits to ``copyEngineIds`` or the createEngine
/// switch (there is no switch any more).
struct EngineDefinition {
    std::int64_t id;
    std::function<CkDslEnginePtr(std::int64_t, CompileServiceBridge&, JitCache&)> factory;
};

const std::vector<EngineDefinition>& engineDefinitions() {
    // Function-local static so the vector is initialised on first call
    // rather than at global static-init time: HIPDNN_REGISTER_ENGINE
    // populates CK_DSL_..._ENGINE_ID via a runtime static initialiser,
    // and a namespace-scope const vector would capture the pre-init
    // zero. Function-local static defers initialisation until after
    // all global constructors have run.
    static const std::vector<EngineDefinition> defs = {
        {CK_DSL_CONV_IMPLICIT_GEMM_ENGINE_ID,
         [](std::int64_t id, CompileServiceBridge& bridge, JitCache& cache) -> CkDslEnginePtr {
             return std::make_unique<CkDslConvImplicitGemmEngine>(id, bridge, cache);
         }},
    };
    return defs;
}

/// Process-wide JIT cache shared across container generations.
/// ``SharedContainerManager`` reconstructs the container when a new
/// handle arrives after the previous generation's last handle was
/// released; if the JitCache lived on the container, every previously
/// compiled kernel would be thrown away on that cycle. Hosting it as
/// a function-local static keeps the cache alive for the entire
/// process lifetime, which is the correct scope for "we already
/// compiled this kernel once."
JitCache& processJitCache() {
    static JitCache cache;
    return cache;
}

}  // namespace

CkDslEnginePtr CkDslContainer::createEngine(std::int64_t id) const {
    for (const auto& def : engineDefinitions()) {
        if (def.id == id) {
            return def.factory(id, *_compileServiceBridge, *_jitCache);
        }
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "CkDslContainer::createEngine: no engine registered for id=" + std::to_string(id));
}

uint32_t CkDslContainer::copyEngineIds(int64_t* engineIdsOut, uint32_t maxEngines,
                                       uint32_t& numEngines) {
    const auto& defs = engineDefinitions();
    auto totalEngines = static_cast<uint32_t>(defs.size());

    if (maxEngines == 0) {
        numEngines = totalEngines;
        return totalEngines;
    }

    auto enginesToCopy = std::min(maxEngines, totalEngines);
    for (uint32_t i = 0; i < enginesToCopy; ++i) {
        engineIdsOut[i] = defs[i].id;
    }

    numEngines = enginesToCopy;

    return totalEngines;
}

CkDslContainer::CkDslContainer() {
    HIPDNN_PLUGIN_LOG_INFO("Creating CkDslContainer");

    // The CK DSL provider drives its compile pipeline through an
    // embedded MicroPython interpreter. ensureInitialized() is idempotent
    // and thread-safe, and the interpreter is intentionally never
    // deinitialised (mp_embed_deinit would tear down the GC heap +
    // runtime state for the whole process), so the initialisation is a
    // per-process action even though SharedContainerManager may
    // reconstruct this container across handle generations.
    ck_dsl_provider::EmbeddedInterpreter::ensureInitialized();

    // Bring up the compile-service bridge before any engines are
    // registered. The bridge is the single boundary through which the
    // JIT pipeline calls into the frozen ck_dsl; constructing it here
    // means the module import happens exactly once per container lifetime
    // and any failure surfaces with a clear container-ctor stack trace
    // rather than deep inside an engine call.
    _compileServiceBridge = std::make_unique<CompileServiceBridge>();

    // Bind to the process-wide JIT cache. The cache outlives the
    // container; references handed to engines remain valid only for the
    // container's lifetime (the engines die with it), but the cache
    // entries themselves persist into the next container generation.
    _jitCache = &processJitCache();

    _engineManager = std::make_unique<
        hipdnn_plugin_sdk::EngineManager<::CkDslHandle, CkDslSettings, CkDslContext>>();

    for (const auto& def : engineDefinitions()) {
        _engineManager->addEngine(createEngine(def.id));
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

JitCache& CkDslContainer::jitCache() {
    if (_jitCache == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED,
            "CkDslContainer::jitCache() called before construction completed");
    }
    return *_jitCache;
}

}  // namespace ck_dsl_provider
