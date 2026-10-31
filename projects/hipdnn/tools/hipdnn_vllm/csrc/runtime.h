// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Per-(process, device) hipDNN runtime.
//
// * Plugin paths are configured once per process, before the first handle.
// * One Runtime per device owns one hipdnnHandle_t. The handle's stream is
//   re-bound only when the caller's stream differs (pointer compare); the
//   re-bind is a host-only call, so it is legal before a captured launch.
// * Plans are built by a GraphBuilder callback and cached process-wide (the
//   key carries the device ordinal), including declines.
// * HIP-graph capture: no plan is ever built while the caller's stream is
//   capturing (graph build, module load and allocation would run inside the
//   capture). A miss under capture is an uncached CAPTURE_INVARIANT decline, so
//   the next eager call still builds the plan. A plan fetched under capture is
//   pinned for the life of the process, because the captured graph replays its
//   launches after the LRU may have dropped it.
//
// This file has no torch dependency; graph_api.h adds the at::Tensor layer.

#pragma once

#include "decline.h"
#include "plan_cache.h"

#include <hip/hip_runtime_api.h>
#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace hipdnn_vllm
{

// What a graph builder returns: the graph plus the uid bound to each execute
// slot, or a plugin-side Decline (UNSUPPORTED_FEATURE:* pre-filters).
struct GraphDef
{
    std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
    std::vector<int64_t> slotUids;
};
using BuildResult = std::variant<GraphDef, Decline>;
using GraphBuilder = std::function<BuildResult()>;

enum class PluginMode
{
    ADDITIVE,
    ABSOLUTE,
};

struct EngineInfo
{
    int64_t id;
    std::string name;
    std::string plugin;
    std::string version;
    std::string type;
};

// --- process-wide -----------------------------------------------------------

// The bundled backend's version, read by setup.py when _C was built.
std::string buildBackendVersion();
// The version reported by the backend _C resolved (empty if none loaded).
std::string loadedBackendVersion();
// Throws std::runtime_error when the two differ.
void checkBackendVersion();

// _C loads its backend at runtime (frontend runtime-load mode). Points that
// resolution at an absolute directory holding libhipdnn_backend.so; false once
// resolution has run. Must precede every other backend call.
bool setBackendDirectory(const std::string& directory);
// Path of the backend _C resolved (resolves on first call; empty on failure).
std::string backendLibraryPath();
// Canonical path of the object defining @p symbol as _C resolves it (dladdr).
std::string backendSymbolOrigin(const std::string& symbol);

// Must run before the first Runtime. Throws if a handle already exists or the
// backend rejects the paths.
void configurePlugins(PluginMode mode, const std::vector<std::string>& paths);
bool pluginsConfigured();

PlanCache& planCache();

// --- per device -------------------------------------------------------------

class Runtime
{
public:
    // The Runtime for `device`, created on first use (HIP device must be set
    // to `device` by the caller, e.g. via a torch DeviceGuard).
    static Runtime& forDevice(int device);
    static bool anyCreated();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    ~Runtime();

    int device() const
    {
        return _device;
    }
    hipdnnHandle_t handle() const
    {
        return _handle;
    }
    const std::string& keyPrefix() const
    {
        return _keyPrefix;
    }

    // Get the cached entry for `key` (which must already include keyPrefix())
    // or build it with `builder`. `op` labels the stats cells. `stream` is the
    // stream the plan will run on; see the capture rules above.
    std::shared_ptr<PlanEntry> getOrBuild(const std::string& op,
                                          const std::string& key,
                                          const GraphBuilder& builder,
                                          hipStream_t stream);

    // Bind `stream`, write the slot pointers and launch. Returns a Decline
    // (stage EXECUTE) when the backend rejects the launch.
    std::optional<Decline> execute(PlanEntry& entry,
                                   const void* const* slotPtrs,
                                   size_t numSlots,
                                   void* workspace,
                                   hipStream_t stream);

    std::vector<EngineInfo> engines() const;
    std::vector<std::string> loadedPluginPaths() const;

private:
    explicit Runtime(int device);
    void bindStreamLocked(hipStream_t stream);
    std::shared_ptr<PlanEntry> buildEntry(const std::string& op, const GraphBuilder& builder);

    int _device;
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    bool _streamBound = false;
    std::string _keyPrefix;
    // hipdnnHandle_t is not thread-safe (docs/Design.md).
    mutable std::mutex _mutex;
};

} // namespace hipdnn_vllm
