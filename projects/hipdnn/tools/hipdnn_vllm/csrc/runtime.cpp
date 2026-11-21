// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "runtime.h"

#include "stats.h"

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifndef HIPDNN_VLLM_BUILD_BACKEND_VERSION
#error "HIPDNN_VLLM_BUILD_BACKEND_VERSION must be defined by the build (setup.py)"
#endif

// _C resolves the backend at runtime (the bundled copy, by explicit path) and
// must never bind a libhipdnn_backend symbol at link time.
#ifndef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND
#error "hipdnn_vllm._C must be built with HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND (setup.py)"
#endif

namespace fe = hipdnn_frontend;

namespace hipdnn_vllm
{
namespace
{

std::mutex& processMutex()
{
    static std::mutex s_mutex;
    return s_mutex;
}

bool& configuredFlag()
{
    static bool s_configured = false;
    return s_configured;
}

// Indexed by device ordinal. Runtimes live for the process: handles are not
// destroyed at exit, because the HIP runtime may already be torn down then.
std::unordered_map<int, Runtime*>& runtimes()
{
    static std::unordered_map<int, Runtime*> s_map;
    return s_map;
}

// Every backend entry point goes through the frontend's runtime-load wrapper,
// or for the _ext queries it does not wrap, through the module's resolved
// backend handle: never through a link-time symbol.
fe::detail::IHipdnnBackend& backend()
{
    return *fe::detail::hipdnnBackend();
}

template <typename Fn>
Fn requireBackendSymbol(std::atomic<void*>& cache, const char* name)
{
    auto fn = fe::detail::resolveBackendSymbol<Fn>(cache, name);
    if(fn == nullptr)
    {
        throw std::runtime_error(std::string("hipdnn_vllm: backend entry point ") + name
                                 + " is not available from " + backendLibraryPath());
    }
    return fn;
}

std::string lastBackendError()
{
    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> buffer{};
    backend().getLastErrorString(buffer.data(), buffer.size());
    return {buffer.data()};
}

void throwOnStatus(hipdnnStatus_t status, const char* what)
{
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string("hipdnn_vllm: ") + what + " failed: "
                                 + backend().getErrorString(status) + " (" + lastBackendError()
                                 + ")");
    }
}

uint64_t nowNs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

Decline fromError(Stage stage, const fe::Error& err)
{
    return Decline{.stage = stage, .detail = {}, .message = err.get_message()};
}

// The backend and its providers report failures through their own status, but
// every failed HIP call inside them also sets the calling thread's HIP last
// error. Left there, the next torch kernel launch check (hipGetLastError)
// raises it from unrelated code: an ASM_SDPA engine without its code objects
// fails plan finalization with hipErrorFileNotFound, and the next torch op then
// raised "file not found". So every backend call is bracketed:
// an error the caller already had is left for the caller, an error raised
// inside the bracket is consumed here and recorded with the outcome.
class HipErrorScope
{
public:
    HipErrorScope()
        : _callerPending(hipPeekAtLastError() != hipSuccess)
    {
    }

    // hipSuccess, or the HIP error raised since construction (now cleared).
    hipError_t take() const
    {
        return _callerPending ? hipSuccess : hipGetLastError();
    }

private:
    bool _callerPending;
};

std::string leakedHipError(hipError_t error)
{
    return std::string("HIP error left by the backend: ") + hipGetErrorName(error);
}

void appendMessage(std::string& message, const std::string& extra)
{
    message += message.empty() ? extra : "; " + extra;
}

bool streamCapturing(hipStream_t stream)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    return hipStreamIsCapturing(stream, &status) == hipSuccess
           && status == hipStreamCaptureStatusActive;
}

// The answer for a key with no plan while the stream is capturing. Not cached:
// the miss says nothing about the signature, and the next eager call builds it.
std::shared_ptr<PlanEntry> captureMiss(const std::string& op)
{
    auto entry = std::make_shared<PlanEntry>();
    entry->op = op;
    entry->decline = Decline{.stage = Stage::CAPTURE_INVARIANT,
                             .detail = {},
                             .message = "no plan for this signature was built before capture "
                                        "(run it eagerly first); plans are never built while "
                                        "capturing"};
    const std::string reason = entry->decline->reason();
    Stats& stats = Stats::instance();
    entry->eagerCell = stats.cell(op, ROUTE_FALLBACK, reason, false);
    entry->captureCell = stats.cell(op, ROUTE_FALLBACK, reason, true);
    return entry;
}

} // namespace

std::string buildBackendVersion()
{
    return HIPDNN_VLLM_BUILD_BACKEND_VERSION;
}

bool setBackendDirectory(const std::string& directory)
{
    return fe::setBackendLibraryPath_ext(std::filesystem::path(directory));
}

std::string backendLibraryPath()
{
    return fe::detail::resolveBackendLibraryPath().string();
}

std::string backendSymbolOrigin(const std::string& symbol)
{
    void* address = fe::detail::resolveSymbol(symbol.c_str());
    if(address == nullptr)
    {
        return {};
    }
    Dl_info info{};
    if(dladdr(address, &info) == 0 || info.dli_fname == nullptr)
    {
        return {};
    }
    return std::filesystem::weakly_canonical(info.dli_fname).string();
}

std::string loadedBackendVersion()
{
    if(fe::detail::backendLibraryHandle() == nullptr)
    {
        return {};
    }
    static std::atomic<void*> s_cache{nullptr};
    auto fn = requireBackendSymbol<decltype(&hipdnnVersionString_ext)>(s_cache,
                                                                         "hipdnnVersionString_ext");
    const char* version = fn();
    return version != nullptr ? std::string(version) : std::string();
}

void checkBackendVersion()
{
    const std::string built = buildBackendVersion();
    const std::string loaded = loadedBackendVersion();
    if(built != loaded)
    {
        throw std::runtime_error(
            "hipdnn_vllm: _C was built against libhipdnn_backend " + built
            + " but the loaded libhipdnn_backend reports " + loaded
            + ". Install the hipdnn-vllm wheel built for this rocm-sdk version.");
    }
}

void configurePlugins(PluginMode mode, const std::vector<std::string>& paths)
{
    const std::lock_guard<std::mutex> lock(processMutex());
    if(!runtimes().empty())
    {
        throw std::runtime_error(
            "hipdnn_vllm: plugin paths must be configured before the first hipDNN handle");
    }
    std::vector<const char*> cpaths;
    cpaths.reserve(paths.size());
    for(const auto& path : paths)
    {
        cpaths.push_back(path.c_str());
    }
    const auto backendMode = mode == PluginMode::ABSOLUTE ? HIPDNN_PLUGIN_LOADING_ABSOLUTE
                                                          : HIPDNN_PLUGIN_LOADING_ADDITIVE;
    throwOnStatus(backend().setEnginePluginPathsExt(
                      cpaths.size(), cpaths.empty() ? nullptr : cpaths.data(), backendMode),
                  "hipdnnSetEnginePluginPaths_ext");
    configuredFlag() = true;
}

bool pluginsConfigured()
{
    const std::lock_guard<std::mutex> lock(processMutex());
    return configuredFlag();
}

PlanCache& planCache()
{
    // Never destroyed: cached plans own backend descriptors, and destroying them
    // from a static destructor runs after the backend and its plugins (and the
    // HIP runtime) may already be torn down, which corrupts the heap at exit.
    static auto* s_cache = new PlanCache(4096);
    return *s_cache;
}

Runtime& Runtime::forDevice(int device)
{
    const std::lock_guard<std::mutex> lock(processMutex());
    auto& map = runtimes();
    auto it = map.find(device);
    if(it != map.end())
    {
        return *it->second;
    }
    if(!configuredFlag())
    {
        throw std::runtime_error("hipdnn_vllm: Runtime requested before the bootstrap "
                                 "configured plugin paths (call hipdnn_vllm._bootstrap.load())");
    }
    // Intentionally leaked; see runtimes().
    auto* runtime = new Runtime(device);
    map.emplace(device, runtime);
    return *runtime;
}

bool Runtime::anyCreated()
{
    const std::lock_guard<std::mutex> lock(processMutex());
    return !runtimes().empty();
}

Runtime::Runtime(int device)
    : _device(device)
{
    const HipErrorScope hipErrors;
    const hipdnnStatus_t created = backend().create(&_handle);
    // Provider initialization may leave a HIP error behind, even on success.
    (void)hipErrors.take();
    throwOnStatus(created, "hipdnnCreate");
    // Engine choice is hipDNN's job, but the configured order changes which
    // engine a key resolves to, so it is part of every plan key.
    const char* order = std::getenv("HIPDNN_HEUR_FALLBACK_ENGINE_ORDER");
    _keyPrefix = "dev=" + std::to_string(device)
                 + ";order=" + (order != nullptr ? std::string(order) : std::string()) + ";";
}

Runtime::~Runtime()
{
    if(_handle != nullptr)
    {
        backend().destroy(_handle);
    }
}

void Runtime::bindStreamLocked(hipStream_t stream)
{
    if(_streamBound && stream == _stream)
    {
        return;
    }
    throwOnStatus(backend().setStream(_handle, stream), "hipdnnSetStream");
    _stream = stream;
    _streamBound = true;
}

std::shared_ptr<PlanEntry> Runtime::buildEntry(const std::string& op, const GraphBuilder& builder)
{
    auto entry = std::make_shared<PlanEntry>();
    entry->op = op;
    const uint64_t start = nowNs();
    const HipErrorScope hipErrors;

    auto finish = [&](std::optional<Decline> decline) {
        // A plan the backend built stands even if a failed attempt on the way
        // (e.g. the top-ranked engine) left a HIP error; a decline records it.
        const hipError_t leaked = hipErrors.take();
        if(leaked != hipSuccess && decline)
        {
            appendMessage(decline->message, leakedHipError(leaked));
        }
        Stats& stats = Stats::instance();
        stats.planBuilt(nowNs() - start, decline.has_value());
        const std::string reason = decline ? decline->reason() : std::string();
        const char* route = decline ? ROUTE_FALLBACK : ROUTE_HIPDNN;
        entry->eagerCell = stats.cell(op, route, reason, false);
        entry->captureCell = stats.cell(op, route, reason, true);
        entry->decline = std::move(decline);
        return entry;
    };

    BuildResult built = builder();
    if(auto* decline = std::get_if<Decline>(&built))
    {
        return finish(std::move(*decline));
    }
    GraphDef def = std::get<GraphDef>(std::move(built));
    fe::graph::Graph& graph = *def.graph;

    fe::Error err = graph.validate();
    if(err.is_bad())
    {
        return finish(fromError(Stage::NO_BUILD, err));
    }
    err = graph.build_operation_graph(_handle);
    if(err.is_bad())
    {
        return finish(fromError(Stage::NO_BUILD, err));
    }
    err = graph.create_execution_plans({fe::HeuristicMode::FALLBACK});
    if(err.is_bad())
    {
        return finish(fromError(Stage::NO_ENGINE, err));
    }
    if(graph.get_execution_plan_count() == 0)
    {
        return finish(Decline{.stage = Stage::NO_ENGINE, .detail = {}, .message = "empty engine list"});
    }
    err = graph.check_support();
    if(err.is_bad())
    {
        return finish(fromError(Stage::CHECK_SUPPORT, err));
    }
    err = graph.build_plans(fe::BuildPlanPolicy::HEURISTICS_CHOICE);
    if(err.is_bad())
    {
        // The top-ranked engine failed to compile; let the next one serve.
        const fe::Error all = graph.build_plans(fe::BuildPlanPolicy::ALL);
        if(all.is_bad())
        {
            return finish(Decline{.stage = Stage::BUILD_PLANS, .detail = {}, .message = err.get_message()});
        }
    }

    auto plan = std::make_shared<BuiltPlan>();
    err = graph.get_workspace_size(plan->workspaceBytes);
    if(err.is_bad())
    {
        return finish(fromError(Stage::BUILD_PLANS, err));
    }
    std::string engineName;
    if(graph.get_plan_name(engineName).is_good())
    {
        plan->engineName = std::move(engineName);
    }
    plan->slotUids = std::move(def.slotUids);
    for(const int64_t uid : plan->slotUids)
    {
        plan->variantPack.emplace(uid, nullptr);
    }
    plan->graph = std::move(def.graph);
    entry->plan = std::move(plan);
    return finish(std::nullopt);
}

std::shared_ptr<PlanEntry> Runtime::getOrBuild(const std::string& op,
                                               const std::string& key,
                                               const GraphBuilder& builder,
                                               hipStream_t stream)
{
    PlanCache& cache = planCache();
    const bool capture = streamCapturing(stream);
    if(auto hit = cache.get(key))
    {
        Stats::instance().planCacheHit();
        if(capture && hit->plan)
        {
            cache.pin(hit);
        }
        return hit;
    }
    if(capture)
    {
        return captureMiss(op);
    }
    const std::lock_guard<std::mutex> lock(_mutex);
    // Another thread may have built the key while this one waited for the lock.
    if(auto hit = cache.get(key))
    {
        Stats::instance().planCacheHit();
        return hit;
    }
    std::shared_ptr<PlanEntry> entry = buildEntry(op, builder);
    cache.put(key, entry);
    return entry;
}

std::optional<Decline> Runtime::execute(PlanEntry& entry,
                                        const void* const* slotPtrs,
                                        size_t numSlots,
                                        void* workspace,
                                        hipStream_t stream)
{
    const uint64_t start = nowNs();
    BuiltPlan& plan = *entry.plan;
    if(numSlots != plan.slotUids.size())
    {
        throw std::logic_error("hipdnn_vllm: execute slot count does not match the plan");
    }

    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &capture) != hipSuccess)
    {
        capture = hipStreamCaptureStatusNone;
    }

    fe::Error err;
    hipError_t leaked = hipSuccess;
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        bindStreamLocked(stream);
        for(size_t i = 0; i < numSlots; ++i)
        {
            plan.variantPack[plan.slotUids[i]] = const_cast<void*>(slotPtrs[i]);
        }
        const HipErrorScope hipErrors;
        err = plan.graph->execute(_handle, plan.variantPack, workspace);
        leaked = hipErrors.take();
    }
    // A HIP failure inside a launch the backend reported as good means the
    // output may not have been written: decline so the caller's fallback runs.
    if(err.is_bad() || leaked != hipSuccess)
    {
        Decline decline{.stage = Stage::EXECUTE, .detail = {}, .message = {}};
        if(err.is_bad())
        {
            decline.message = err.get_message();
        }
        if(leaked != hipSuccess)
        {
            appendMessage(decline.message, leakedHipError(leaked));
        }
        Stats::instance().count(entry.op,
                                ROUTE_FALLBACK,
                                decline.reason(),
                                capture == hipStreamCaptureStatusActive);
        return decline;
    }
    (capture == hipStreamCaptureStatusActive ? entry.captureCell : entry.eagerCell)
        ->fetch_add(1, std::memory_order_relaxed);
    Stats::instance().executed(nowNs() - start);
    return std::nullopt;
}

std::vector<EngineInfo> Runtime::engines() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    size_t count = 0;
    static std::atomic<void*> s_count{nullptr};
    static std::atomic<void*> s_info{nullptr};
    auto getCount = requireBackendSymbol<decltype(&hipdnnGetEngineCount_ext)>(
        s_count, "hipdnnGetEngineCount_ext");
    auto getInfo = requireBackendSymbol<decltype(&hipdnnGetEngineInfo_ext)>(
        s_info, "hipdnnGetEngineInfo_ext");
    throwOnStatus(getCount(_handle, &count), "hipdnnGetEngineCount_ext");
    std::vector<EngineInfo> out;
    out.reserve(count);
    for(size_t i = 0; i < count; ++i)
    {
        size_t nameLen = 0;
        size_t pluginLen = 0;
        size_t versionLen = 0;
        size_t typeLen = 0;
        throwOnStatus(getInfo(_handle,
                              i,
                              nullptr,
                              nullptr,
                              &nameLen,
                              nullptr,
                              &pluginLen,
                              nullptr,
                              &versionLen,
                              nullptr,
                              &typeLen),
                      "hipdnnGetEngineInfo_ext(size)");
        std::string name(nameLen, '\0');
        std::string plugin(pluginLen, '\0');
        std::string version(versionLen, '\0');
        std::string type(typeLen, '\0');
        int64_t id = 0;
        throwOnStatus(getInfo(_handle,
                              i,
                              &id,
                              name.data(),
                              &nameLen,
                              plugin.data(),
                              &pluginLen,
                              version.data(),
                              &versionLen,
                              type.data(),
                              &typeLen),
                      "hipdnnGetEngineInfo_ext");
        auto trim = [](std::string& s) { s.resize(s.find('\0') == std::string::npos ? s.size() : s.find('\0')); };
        trim(name);
        trim(plugin);
        trim(version);
        trim(type);
        out.push_back(EngineInfo{.id = id,
                                 .name = std::move(name),
                                 .plugin = std::move(plugin),
                                 .version = std::move(version),
                                 .type = std::move(type)});
    }
    return out;
}

std::vector<std::string> Runtime::loadedPluginPaths() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    size_t count = 0;
    size_t maxLen = 0;
    throwOnStatus(backend().getLoadedEnginePluginPathsExt(_handle, &count, nullptr, &maxLen),
                  "hipdnnGetLoadedEnginePluginPaths_ext(size)");
    std::vector<std::string> storage(count, std::string(maxLen, '\0'));
    std::vector<char*> ptrs;
    ptrs.reserve(count);
    for(auto& s : storage)
    {
        ptrs.push_back(s.data());
    }
    throwOnStatus(
        backend().getLoadedEnginePluginPathsExt(_handle, &count, ptrs.data(), &maxLen),
        "hipdnnGetLoadedEnginePluginPaths_ext");
    storage.resize(count);
    for(auto& s : storage)
    {
        s.resize(s.find('\0') == std::string::npos ? s.size() : s.find('\0'));
    }
    return storage;
}

} // namespace hipdnn_vllm
