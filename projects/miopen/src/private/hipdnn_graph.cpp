// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_graph.hpp"
#include "hipdnn_types.hpp"

#include "lru_cache.hpp"
#include "miopen_impl.h"

#include <hipdnn_frontend.hpp>

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace miopen {
namespace wrapper {
namespace hipdnn {

namespace {

namespace fe = hipdnn_frontend;

struct PlanKeyHash
{
    size_t operator()(const PlanKey& key) const
    {
        size_t seed = std::hash<const void*>{}(key.handle);
        for(const int64_t value : key.problem)
            seed ^= static_cast<size_t>(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

// Everything hipDNN needs that is tied to one MIOpen handle. The workspace is
// reused across calls rather than reallocated: the hipDNN handle runs on the
// MIOpen handle's stream, so successive forwarded calls on that handle are
// already ordered against each other.
struct HandleState
{
    fe::HipdnnHandlePtr hipdnnHandle;
    hipStream_t stream   = nullptr;
    void* workspace      = nullptr;
    size_t workspaceSize = 0;
    std::mutex mutex;

    ~HandleState()
    {
        if(workspace != nullptr)
            static_cast<void>(hipFree(workspace));
    }

    bool EnsureWorkspace(size_t bytes)
    {
        if(bytes <= workspaceSize)
            return true;
        void* grown = nullptr;
        if(hipMalloc(&grown, bytes) != hipSuccess)
            return false;
        // Work queued by an earlier call may still be using the old buffer.
        // hipFree happens to wait for it, but that is not a documented promise.
        if(workspace != nullptr)
        {
            static_cast<void>(hipStreamSynchronize(stream));
            static_cast<void>(hipFree(workspace));
        }
        workspace     = grown;
        workspaceSize = bytes;
        return true;
    }
};

std::mutex& HandleMapMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<miopenHandle_t, std::unique_ptr<HandleState>>& HandleMap()
{
    static std::unordered_map<miopenHandle_t, std::unique_ptr<HandleState>> handles;
    return handles;
}

// Created on first forwarded call rather than in miopenCreate, so a process that
// never forwards never pays for hipdnnCreate.
HandleState* AcquireHandleState(miopenHandle_t handle)
{
    hipStream_t stream = nullptr;
    if(miopenGetStream_impl(handle, &stream) != miopenStatusSuccess)
        return nullptr;

    const std::lock_guard<std::mutex> lock(HandleMapMutex());
    auto& slot = HandleMap()[handle];
    if(slot == nullptr)
    {
        auto [created, error] = fe::createHipdnnHandle(stream);
        if(!error.is_good() || created == nullptr)
        {
            HandleMap().erase(handle);
            return nullptr;
        }
        slot               = std::make_unique<HandleState>();
        slot->hipdnnHandle = std::move(created);
        slot->stream       = stream;
    }
    else if(slot->stream != stream)
    {
        // miopenSetStream can move a handle to a different stream at any point.
        if(!fe::setHipdnnHandleStream(slot->hipdnnHandle, stream).is_good())
            return nullptr;
        slot->stream = stream;
    }
    return slot.get();
}

using GraphPtr = std::shared_ptr<fe::graph::Graph>;

std::mutex& PlanMutex()
{
    static std::mutex mutex;
    return mutex;
}

// Capped because a workload with changing shapes (varying batch size or input
// size) makes a new key per shape, and each built graph holds backend state for
// as long as it is cached. Evicting is always safe: a miss just rebuilds, and a
// call still running an evicted graph holds its own reference to it.
constexpr size_t kPlanCacheCapacity = 128;

LruCache<PlanKey, GraphPtr, PlanKeyHash>& PlanCache()
{
    static LruCache<PlanKey, GraphPtr, PlanKeyHash> plans(kPlanCacheCapacity);
    return plans;
}

struct LastForwardedError
{
    bool failed = false;
    miopenStatus_t status{};
    std::string message;
};

// Per-thread so that one thread's forwarded failure cannot be attributed to
// another thread's miopenGetErrorString call.
LastForwardedError& LastError()
{
    static thread_local LastForwardedError last;
    return last;
}

miopenStatus_t RecordHipdnnFailure(const fe::Error& error)
{
    return RecordFailure(TranslateHipdnnError(error), error.get_message());
}

miopenStatus_t RecordSuccess()
{
    LastError().failed = false;
    return miopenStatusSuccess;
}

// Held across build() on purpose. A build can take seconds, but it happens once
// per distinct problem and serializing it is far simpler than letting two
// threads race to build the same graph.
std::pair<GraphPtr, miopenStatus_t>
AcquireGraph(const PlanKey& key, const PopulateGraphFn& populate, hipdnnHandle_t hipdnnHandle)
{
    const std::lock_guard<std::mutex> lock(PlanMutex());

    if(const GraphPtr* cached = PlanCache().Find(key))
        return {*cached, miopenStatusSuccess};

    GraphPtr graph = std::make_shared<fe::graph::Graph>();
    if(!populate(*graph))
        return {nullptr,
                RecordFailure(miopenStatusUnsupportedOp,
                              "could not build a hipDNN graph for this problem")};

    const fe::Error error = graph->build(hipdnnHandle);
    if(!error.is_good())
        return {nullptr, RecordHipdnnFailure(error)};

    PlanCache().Insert(key, graph);
    return {graph, miopenStatusSuccess};
}

miopenStatus_t RunGraph(HandleState& state, const GraphPtr& graph, VariantPack& variantPack)
{
    const std::lock_guard<std::mutex> lock(state.mutex);

    int64_t workspaceSize = 0;
    if(const fe::Error error = graph->get_workspace_size(workspaceSize); !error.is_good())
        return RecordHipdnnFailure(error);

    if(!state.EnsureWorkspace(static_cast<size_t>(workspaceSize)))
        return RecordFailure(miopenStatusAllocFailed, "hipDNN workspace allocation failed");

    const fe::Error error = graph->execute(*state.hipdnnHandle, variantPack, state.workspace);
    if(!error.is_good())
        return RecordHipdnnFailure(error);

    return RecordSuccess();
}

} // namespace

miopenStatus_t RecordFailure(miopenStatus_t status, std::string message)
{
    LastError() = LastForwardedError{true, status, std::move(message)};
    return status;
}

miopenStatus_t RunCachedGraph(miopenHandle_t handle,
                              const PlanKey& key,
                              const PopulateGraphFn& populate,
                              VariantPack& variantPack)
{
    HandleState* state = AcquireHandleState(handle);
    if(state == nullptr)
        return RecordFailure(miopenStatusInternalError, "could not create a hipDNN handle");

    auto [graph, status] = AcquireGraph(key, populate, *state->hipdnnHandle);
    if(graph == nullptr)
        return status;

    return RunGraph(*state, graph, variantPack);
}

bool IsAvailable()
{
    static const bool available = [] {
        // Creating a handle makes the frontend load the backend, refuse it if its major
        // version differs, and then run hipdnnCreate, so this one call catches every way
        // the backend can be unusable. hipDNN always says why a load failed, but explains
        // a refused version only when its logging is on, hence the hint. The frontend's
        // message goes last because it ends in an empty "Backend error: " when no
        // backend was loaded.
        auto [handle, error] = fe::createHipdnnHandle();
        if(error.is_good() && handle != nullptr)
            return true;
        std::cerr << "[MIOpen] hipDNN forwarding is unavailable (if hipDNN gives no reason, set "
                     "HIPDNN_LOG_LEVEL=error): "
                  << error.get_message() << "\n";
        return false;
    }();
    return available;
}

void ReleaseHandle(miopenHandle_t handle)
{
    {
        const std::lock_guard<std::mutex> lock(PlanMutex());
        PlanCache().EraseIf([handle](const PlanKey& key) { return key.handle == handle; });
    }

    const std::lock_guard<std::mutex> lock(HandleMapMutex());
    HandleMap().erase(handle);
}

const char* PrefixedErrorString(miopenStatus_t status, const char* nativeMessage)
{
    const LastForwardedError& last = LastError();
    if(!last.failed || last.status != status || nativeMessage == nullptr)
        return nullptr;

    // miopenGetErrorString returns a bare const char* the caller does not own,
    // so the prefixed text has to outlive this call without being leaked.
    static thread_local std::string prefixed;
    prefixed = "[hipDNN-forwarded] " + std::string(nativeMessage);
    if(!last.message.empty())
        prefixed += ": " + last.message;
    return prefixed.c_str();
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
