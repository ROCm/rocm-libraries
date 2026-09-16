// hipDNN N-API addon.
//
// Bridges the Electron main process to the hipDNN frontend C++ API. The renderer
// hands over the Studio graph as JSON (the src/graph/model.ts shape — a format we
// fully own). This addon parses it and drives the *public* frontend builder
// (`Graph::tensor / conv_fprop / pointwise / matmul / resample_fwd → build →
// execute`), then returns structured results (matching src/engine/types.ts).
//
// Building the graph through the documented builder — rather than hipDNN's
// internal from_json — means we never depend on the backend's private JSON
// schema. Errors from build()/execute() are hipDNN's real, structured errors.
//
// Build: node-gyp (see ../binding.gyp). Requires the ROCm/hipDNN SDK headers +
// hipdnn_backend/amdhip64 import libraries and the HIP runtime.

#include <napi.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph_translate.hpp"
#include <hip/hip_runtime.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <nlohmann/json.hpp>

namespace
{

using json = nlohmann::json;
using hipdnn_frontend::DataType;
using hipdnn_frontend::Error;
using hipdnn_frontend::ErrorCode;
using hipdnn_frontend::HipdnnHandlePtr;
using hipdnn_frontend::graph::Graph;
using studio::BuildInputError;
using studio::TensorPtr;

// A built graph the renderer holds a token to. Owns its handle so the compiled
// plan stays valid until release().
struct BuiltGraph
{
    HipdnnHandlePtr handle;
    Graph graph;
    int64_t workspaceSize = 0;
};

std::unordered_map<std::string, std::unique_ptr<BuiltGraph>> g_registry;
uint64_t g_nextToken = 0;
int64_t g_nextUid = 0;

// ── Log capture ────────────────────────────────────────────────────────
// hipDNN invokes the user log callback (possibly on a worker thread), so the
// sink is mutex-guarded. build()/execute() clear it, run, then drain it into
// the result. Registration happens once, lazily.
struct CapturedLine
{
    std::string severity;
    std::string message;
};
std::mutex g_logMutex;
std::vector<CapturedLine> g_logSink;
bool g_captureEnabled = false;
bool g_logCallbackRegistered = false;
int g_logHandleTag = 0; // address used as the callback's unique userHandle

const char* severityName(hipdnnSeverity_t sev)
{
    switch(sev)
    {
    case HIPDNN_SEV_INFO:
        return "INFO";
    case HIPDNN_SEV_WARN:
        return "WARN";
    case HIPDNN_SEV_ERROR:
        return "ERROR";
    case HIPDNN_SEV_FATAL:
        return "FATAL";
    default:
        return "INFO";
    }
}

void logCallback(hipdnnUserLogCallbackHandle_t, hipdnnSeverity_t sev, const char* message)
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    if(!g_captureEnabled)
        return;
    g_logSink.push_back({severityName(sev), message ? message : ""});
}

void ensureLogCallback()
{
    if(g_logCallbackRegistered)
        return;
    // Register at INFO so the global level (set via setLogLevel) is what actually
    // gates output; the callback then receives everything the global level emits.
    hipdnn_frontend::setUserLogCallback(
        logCallback, HIPDNN_SEV_INFO, hipdnn_frontend::LogCallbackMode::SYNC, &g_logHandleTag);
    g_logCallbackRegistered = true;
}

// Swap out the captured lines (thread-safe) and start/stop capture around a call.
void beginCapture()
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_logSink.clear();
    g_captureEnabled = true;
}
std::vector<CapturedLine> endCapture()
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_captureEnabled = false;
    return std::move(g_logSink);
}

Napi::Array capturedToJs(Napi::Env env, const std::vector<CapturedLine>& lines)
{
    Napi::Array arr = Napi::Array::New(env, lines.size());
    for(std::size_t i = 0; i < lines.size(); ++i)
    {
        Napi::Object o = Napi::Object::New(env);
        o.Set("severity", lines[i].severity);
        o.Set("message", lines[i].message);
        arr.Set(i, o);
    }
    return arr;
}

// RAII: enable capture for a build/execute call and, on scope exit, attach the
// drained lines to the result object under "captured".
struct CaptureScope
{
    Napi::Env env;
    Napi::Object& result;
    CaptureScope(Napi::Env e, Napi::Object& r)
        : env(e)
        , result(r)
    {
        ensureLogCallback();
        beginCapture();
    }
    ~CaptureScope()
    {
        result.Set("captured", capturedToJs(env, endCapture()));
    }
};

std::string errorCodeName(ErrorCode code)
{
    return hipdnn_frontend::to_string(code);
}

Napi::Object makeError(Napi::Env env, const std::string& code, const std::string& message)
{
    Napi::Object err = Napi::Object::New(env);
    err.Set("code", code);
    err.Set("message", message);
    return err;
}

Napi::Array logArray(Napi::Env env, const std::vector<std::string>& lines)
{
    Napi::Array arr = Napi::Array::New(env, lines.size());
    for(std::size_t i = 0; i < lines.size(); ++i)
        arr.Set(i, lines[i]);
    return arr;
}

// ── info() ─────────────────────────────────────────────────────────────
Napi::Value Info(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::Object out = Napi::Object::New(env);
    int deviceCount = 0;
    const hipError_t hipErr = hipGetDeviceCount(&deviceCount);
    const bool haveGpu = (hipErr == hipSuccess && deviceCount > 0);
    out.Set("available", haveGpu);
    out.Set("backend", std::string(haveGpu ? "hipDNN frontend" : "hipDNN (no GPU device found)"));
    if(haveGpu)
    {
        hipDeviceProp_t props{};
        if(hipGetDeviceProperties(&props, 0) == hipSuccess)
            out.Set("device", std::string(props.name));
    }
    return out;
}

// ── Shared graph setup ─────────────────────────────────────────────────
void setFailure(Napi::Env env,
                Napi::Object& result,
                const std::string& code,
                const std::string& message,
                const std::vector<std::string>& log)
{
    result.Set("ok", false);
    result.Set("error", makeError(env, code, message));
    result.Set("log", logArray(env, log));
}

// Parse the Studio JSON, open a hipDNN handle and translate the nodes into a
// frontend graph. On failure `result` carries the structured error and the
// return value is null.
std::unique_ptr<BuiltGraph> prepareGraph(Napi::Env env,
                                         const Napi::CallbackInfo& info,
                                         Napi::Object& result,
                                         std::vector<std::string>& log)
{
    if(info.Length() < 1 || !info[0].IsString())
    {
        setFailure(env, result, "INVALID_VALUE", "Expected a JSON graph string.", log);
        return nullptr;
    }

    json root;
    try
    {
        root = json::parse(info[0].As<Napi::String>().Utf8Value());
    }
    catch(const std::exception& e)
    {
        setFailure(env, result, "INVALID_VALUE", std::string("Invalid JSON: ") + e.what(), log);
        return nullptr;
    }

    auto built = std::make_unique<BuiltGraph>();
    const Error err = hipdnn_frontend::createHipdnnHandle(built->handle);
    if(err.is_bad())
    {
        setFailure(env, result, errorCodeName(err.get_code()), err.get_message(), log);
        return nullptr;
    }

    const DataType ioDtype = studio::pickIoDtype(root);
    built->graph.set_io_data_type(ioDtype)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);
    if(root.contains("name") && root["name"].is_string())
    {
        built->graph.set_name(root["name"].get<std::string>());
    }

    try
    {
        std::unordered_map<std::string, TensorPtr> outputByNode;
        studio::translateGraph(root, built->graph, ioDtype, outputByNode);
    }
    catch(const BuildInputError& e)
    {
        setFailure(env, result, e.code, e.message, log);
        return nullptr;
    }
    catch(const std::exception& e)
    {
        setFailure(env, result, "INVALID_VALUE", std::string("Malformed graph: ") + e.what(), log);
        return nullptr;
    }
    log.push_back("Constructed frontend graph.");
    return built;
}

// The name a loaded engine carries. The backend answers for plugin-supplied
// engines, which the frontend's static registry knows nothing about; that
// registry, and finally the hexadecimal id, are the fallbacks.
std::string engineNameFor(hipdnnHandle_t handle, int64_t id)
{
    size_t length = 0;
    if(handle != nullptr
       && hipdnn_frontend::detail::hipdnnBackend()->getEngineNameByIdExt(
              handle, id, nullptr, &length)
              == HIPDNN_STATUS_SUCCESS
       && length > 0)
    {
        std::vector<char> name(length);
        if(hipdnn_frontend::detail::hipdnnBackend()->getEngineNameByIdExt(
               handle, id, name.data(), &length)
           == HIPDNN_STATUS_SUCCESS)
            return {name.data()};
    }
    return hipdnn_frontend::detail::resolveEngineName(id);
}

// Engine ids are 64-bit name hashes, so they cross the JS boundary as decimal
// strings — a double would silently round them. The name is what the renderer
// shows and pins builds to; the id is kept for display and report rows.
Napi::Object engineToJs(Napi::Env env, hipdnnHandle_t handle, int64_t id)
{
    Napi::Object eng = Napi::Object::New(env);
    eng.Set("id", std::to_string(id));
    eng.Set("name", engineNameFor(handle, id));
    return eng;
}

// Heuristic-ranked engine candidates for a graph whose operation graph is built.
// Index 0 is the heuristics' own pick. A query failure yields an empty list.
Napi::Array rankedEnginesToJs(Napi::Env env, hipdnnHandle_t handle, Graph& graph)
{
    std::vector<int64_t> ids;
    if(graph.get_ranked_engine_ids(ids).is_bad())
        ids.clear();
    Napi::Array arr = Napi::Array::New(env, ids.size());
    for(std::size_t i = 0; i < ids.size(); ++i)
        arr.Set(i, engineToJs(env, handle, ids[i]));
    return arr;
}

// Read options.engineName. Absent or empty means hipDNN's heuristics choose.
std::optional<std::string> readEngineOption(const Napi::CallbackInfo& info)
{
    if(info.Length() < 2 || !info[1].IsObject())
        return std::nullopt;
    const Napi::Value nameValue = info[1].As<Napi::Object>().Get("engineName");
    if(!nameValue.IsString())
        return std::nullopt;
    std::string name = nameValue.As<Napi::String>().Utf8Value();
    if(name.empty())
        return std::nullopt;
    return name;
}

void applyPreferredEngine(BuiltGraph& built,
                          const std::optional<std::string>& engineName,
                          std::vector<std::string>& log)
{
    if(!engineName)
        return;
    built.graph.set_preferred_engine_id_ext(*engineName);
    log.push_back("Requested engine " + *engineName + ".");
}

// Compile the plan and fill the JS result: workspace, the engine hipDNN chose,
// the ranked candidates and hipDNN's canonical JSON for the built graph. Takes
// ownership — on success the graph lands in the registry under a token.
Napi::Value compileAndReport(Napi::Env env,
                             Napi::Object& result,
                             std::vector<std::string>& log,
                             std::unique_ptr<BuiltGraph> built)
{
    const Error err = built->graph.build(*built->handle);
    if(err.is_bad())
    {
        setFailure(env, result, errorCodeName(err.get_code()), err.get_message(), log);
        return result;
    }
    log.push_back("Built execution plan.");
    built->graph.get_workspace_size(built->workspaceSize);

    hipdnnHandle_t handle = *built->handle;
    Napi::Array engines = rankedEnginesToJs(env, handle, built->graph);

    int64_t engineId = -1;
    const bool haveEngineId = built->graph.get_execution_plan_engine_id(engineId).is_good();

    std::string canonicalJson;
    const bool haveJson = built->graph.serialize(canonicalJson).is_good();

    const std::string token = "plan_" + std::to_string(++g_nextToken);
    const int64_t ws = built->workspaceSize;
    g_registry[token] = std::move(built);

    result.Set("ok", true);
    result.Set("handle", token);
    result.Set("workspaceSize", Napi::Number::New(env, static_cast<double>(ws)));
    if(haveEngineId)
        result.Set("selectedEngine", engineToJs(env, handle, engineId));
    if(haveJson)
        result.Set("serializedGraph", canonicalJson);
    result.Set("engines", engines);
    result.Set("log", logArray(env, log));
    return result;
}

// ── build(graphJson, options) ──────────────────────────────────────────
// options.engineName pins the engine; omitted means the heuristics choose. A
// pinned engine that isn't applicable to this graph is ignored by hipDNN, so
// the caller compares selectedEngine against its request.
Napi::Value Build(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::Object result = Napi::Object::New(env);
    std::vector<std::string> log;
    CaptureScope capture(env, result);

    const std::optional<std::string> preferredEngine = readEngineOption(info);

    std::unique_ptr<BuiltGraph> built = prepareGraph(env, info, result, log);
    if(!built)
        return result;

    applyPreferredEngine(*built, preferredEngine, log);
    return compileAndReport(env, result, log, std::move(built));
}

// ── buildHipdnnJson(hipdnnJson, options) ───────────────────────────────
// Compile a graph straight from hipDNN's canonical JSON (what build() returns
// as serializedGraph) rather than the Studio format. The resulting plan runs
// through the same execute()/release() path; the Studio canvas isn't involved.
Napi::Value BuildHipdnnJson(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::Object result = Napi::Object::New(env);
    std::vector<std::string> log;
    CaptureScope capture(env, result);

    if(info.Length() < 1 || !info[0].IsString())
    {
        setFailure(env, result, "INVALID_VALUE", "Expected a hipDNN JSON string.", log);
        return result;
    }

    const std::optional<std::string> preferredEngine = readEngineOption(info);

    auto built = std::make_unique<BuiltGraph>();
    const Error handleErr = hipdnn_frontend::createHipdnnHandle(built->handle);
    if(handleErr.is_bad())
    {
        setFailure(env, result, errorCodeName(handleErr.get_code()), handleErr.get_message(), log);
        return result;
    }

    const Error err
        = built->graph.deserialize(*built->handle, info[0].As<Napi::String>().Utf8Value());
    if(err.is_bad())
    {
        setFailure(env, result, errorCodeName(err.get_code()), err.get_message(), log);
        return result;
    }
    log.push_back("Loaded hipDNN JSON graph.");

    applyPreferredEngine(*built, preferredEngine, log);
    return compileAndReport(env, result, log, std::move(built));
}

// ── serializeGraph(graphJson) ──────────────────────────────────────────
// hipDNN's canonical JSON for a Studio graph, without compiling a plan:
// Graph::serialize() lowers to a backend descriptor on demand. The output is
// what deserialize()/buildHipdnnJson() accept.
Napi::Value SerializeGraph(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::Object result = Napi::Object::New(env);
    std::vector<std::string> log;
    CaptureScope capture(env, result);

    std::unique_ptr<BuiltGraph> built = prepareGraph(env, info, result, log);
    if(!built)
        return result;

    std::string canonicalJson;
    const Error err = built->graph.serialize(canonicalJson);
    if(err.is_bad())
    {
        setFailure(env, result, errorCodeName(err.get_code()), err.get_message(), log);
        return result;
    }
    log.push_back("Serialized graph to hipDNN JSON.");

    result.Set("ok", true);
    result.Set("serializedGraph", canonicalJson);
    result.Set("log", logArray(env, log));
    return result;
}

// ── listEngines(graphJson) ─────────────────────────────────────────────
// Heuristic-ranked engines applicable to this graph, without compiling a plan.
Napi::Value ListEngines(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::Object result = Napi::Object::New(env);
    std::vector<std::string> log;
    CaptureScope capture(env, result);

    std::unique_ptr<BuiltGraph> built = prepareGraph(env, info, result, log);
    if(!built)
        return result;

    const Error err = built->graph.build_operation_graph(*built->handle);
    if(err.is_bad())
    {
        setFailure(env, result, errorCodeName(err.get_code()), err.get_message(), log);
        return result;
    }

    std::vector<int64_t> ids;
    const Error rankErr = built->graph.get_ranked_engine_ids(ids);
    if(rankErr.is_bad())
    {
        setFailure(env, result, errorCodeName(rankErr.get_code()), rankErr.get_message(), log);
        return result;
    }

    Napi::Array engines = Napi::Array::New(env, ids.size());
    for(std::size_t i = 0; i < ids.size(); ++i)
        engines.Set(i, engineToJs(env, *built->handle, ids[i]));

    result.Set("ok", true);
    result.Set("engines", engines);
    result.Set("log", logArray(env, log));
    return result;
}

// ── execute(handle, options) ───────────────────────────────────────────
Napi::Value Execute(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::Object result = Napi::Object::New(env);
    std::vector<std::string> log;
    CaptureScope capture(env, result);

    if(info.Length() < 1 || !info[0].IsString())
    {
        result.Set("ok", false);
        result.Set("error", makeError(env, "INVALID_VALUE", "execute() expects a handle string."));
        result.Set("log", logArray(env, log));
        return result;
    }
    auto it = g_registry.find(info[0].As<Napi::String>().Utf8Value());
    if(it == g_registry.end())
    {
        result.Set("ok", false);
        result.Set("error",
                   makeError(env, "INVALID_VALUE", "Unknown build handle. Rebuild the graph."));
        result.Set("log", logArray(env, log));
        return result;
    }
    BuiltGraph& built = *it->second;

    std::vector<void*> allocations;
    std::unordered_map<int64_t, void*> variantPack;
    auto cleanup = [&]() {
        for(void* p : allocations)
            hipFree(p);
    };

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> byteDist(0, 255);

    const auto tensors = built.graph.getTensorsByUid();
    for(const auto& kv : tensors)
    {
        const int64_t uid = kv.first;
        const TensorPtr& tensor = kv.second;
        if(!tensor || tensor->get_is_virtual())
            continue;
        const int64_t elements = studio::storageElements(tensor->get_dim(), tensor->get_stride());
        const std::size_t bytes
            = static_cast<std::size_t>(elements) * studio::dtypeSize(tensor->get_data_type());
        if(bytes == 0)
        {
            cleanup();
            result.Set("ok", false);
            result.Set("error",
                       makeError(env,
                                 "INVALID_VALUE",
                                 "Tensor uid " + std::to_string(uid) + " has zero size."));
            result.Set("log", logArray(env, log));
            return result;
        }
        void* dptr = nullptr;
        if(hipMalloc(&dptr, bytes) != hipSuccess)
        {
            cleanup();
            result.Set("ok", false);
            result.Set("error",
                       makeError(env, "HIPDNN_BACKEND_ERROR", "hipMalloc failed for a tensor."));
            result.Set("log", logArray(env, log));
            return result;
        }
        allocations.push_back(dptr);
        std::vector<uint8_t> host(bytes);
        for(auto& b : host)
            b = static_cast<uint8_t>(byteDist(rng));
        hipMemcpy(dptr, host.data(), bytes, hipMemcpyHostToDevice);
        variantPack[uid] = dptr;
    }

    void* workspace = nullptr;
    if(built.workspaceSize > 0
       && hipMalloc(&workspace, static_cast<std::size_t>(built.workspaceSize)) != hipSuccess)
    {
        cleanup();
        result.Set("ok", false);
        result.Set("error",
                   makeError(env, "HIPDNN_BACKEND_ERROR", "hipMalloc failed for workspace."));
        result.Set("log", logArray(env, log));
        return result;
    }

    const auto start = std::chrono::steady_clock::now();
    Error err = built.graph.execute(*built.handle, variantPack, workspace);
    hipDeviceSynchronize();
    const auto end = std::chrono::steady_clock::now();

    if(workspace)
        hipFree(workspace);
    cleanup();

    if(err.is_bad())
    {
        result.Set("ok", false);
        result.Set("error", makeError(env, errorCodeName(err.get_code()), err.get_message()));
        result.Set("log", logArray(env, log));
        return result;
    }

    log.push_back("Executed on GPU with randomized inputs.");
    result.Set("ok", true);
    result.Set(
        "elapsedMs",
        Napi::Number::New(env, std::chrono::duration<double, std::milli>(end - start).count()));
    result.Set("log", logArray(env, log));
    return result;
}

// ── release(handle) ────────────────────────────────────────────────────
Napi::Value Release(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    if(info.Length() >= 1 && info[0].IsString())
    {
        g_registry.erase(info[0].As<Napi::String>().Utf8Value());
    }
    return env.Undefined();
}

// ── setLogLevel(level) ─────────────────────────────────────────────────
Napi::Value SetLogLevel(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    const std::string level = (info.Length() >= 1 && info[0].IsString())
                                  ? info[0].As<Napi::String>().Utf8Value()
                                  : "off";
    hipdnnSeverity_t sev = HIPDNN_SEV_OFF;
    if(level == "info")
        sev = HIPDNN_SEV_INFO;
    else if(level == "warn")
        sev = HIPDNN_SEV_WARN;
    else if(level == "error")
        sev = HIPDNN_SEV_ERROR;
    else
        sev = HIPDNN_SEV_OFF;
    ensureLogCallback();
    hipdnn_frontend::setGlobalLogLevel(sev);
    return env.Undefined();
}

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set("info", Napi::Function::New(env, Info));
    exports.Set("build", Napi::Function::New(env, Build));
    exports.Set("buildHipdnnJson", Napi::Function::New(env, BuildHipdnnJson));
    exports.Set("serializeGraph", Napi::Function::New(env, SerializeGraph));
    exports.Set("listEngines", Napi::Function::New(env, ListEngines));
    exports.Set("execute", Napi::Function::New(env, Execute));
    exports.Set("release", Napi::Function::New(env, Release));
    exports.Set("setLogLevel", Napi::Function::New(env, SetLogLevel));
    return exports;
}

} // namespace

NODE_API_MODULE(hipdnn_engine, Init)
