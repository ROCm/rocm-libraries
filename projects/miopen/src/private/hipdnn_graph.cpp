// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_graph.hpp"
#include "hipdnn_types.hpp"

#include "lru_cache.hpp"
#include "miopen_impl.h"
#include "routing.hpp"

#include <hipdnn_frontend.hpp>

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

// Everything hipDNN needs that is tied to one MIOpen handle. Reusing one
// workspace across calls is safe only because the calls are ordered: they share
// a stream, and AcquireHandleState orders them when the stream changes.
struct HandleState
{
    fe::HipdnnHandlePtr hipdnnHandle;
    hipStream_t stream = nullptr;
    // Marks the end of the last call's work. A call on a new stream waits on this
    // instead of the old stream, which the caller may have destroyed.
    hipEvent_t lastWork  = nullptr;
    void* workspace      = nullptr;
    size_t workspaceSize = 0;
    std::mutex mutex;

    ~HandleState()
    {
        if(workspace != nullptr)
            static_cast<void>(hipFree(workspace));
        if(lastWork != nullptr)
            static_cast<void>(hipEventDestroy(lastWork));
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

class ScopedDevice
{
public:
    ScopedDevice()                               = default;
    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

    ~ScopedDevice()
    {
        if(previous_ >= 0)
            static_cast<void>(hipSetDevice(previous_));
    }

    bool Set(int device)
    {
        int current = 0;
        if(hipGetDevice(&current) != hipSuccess)
            return false;
        if(current == device)
            return true;
        if(hipSetDevice(device) != hipSuccess)
            return false;
        previous_ = current;
        return true;
    }

private:
    int previous_ = -1;
};

// Created on first forwarded call rather than in miopenCreate, so a process that
// never forwards never pays for hipdnnCreate.
std::pair<HandleState*, miopenStatus_t> AcquireHandleState(miopenHandle_t handle,
                                                           hipStream_t stream)
{
    const std::lock_guard<std::mutex> lock(HandleMapMutex());
    auto& slot = HandleMap()[handle];
    if(slot == nullptr)
    {
        auto [created, error] = fe::createHipdnnHandle(stream);
        if(!error.is_good() || created == nullptr)
        {
            HandleMap().erase(handle);
            return {nullptr,
                    RecordFailure(miopenStatusInternalError, "could not create a hipDNN handle")};
        }
        auto state = std::make_unique<HandleState>();
        if(hipEventCreateWithFlags(&state->lastWork, hipEventDisableTiming) != hipSuccess)
        {
            HandleMap().erase(handle);
            return {nullptr,
                    RecordFailure(miopenStatusInternalError,
                                  "could not create the HIP event that orders hipDNN work")};
        }
        state->hipdnnHandle = std::move(created);
        state->stream       = stream;
        slot                = std::move(state);
    }
    else if(slot->stream != stream)
    {
        // Work still queued on the old stream may be using the shared workspace.
        if(hipStreamWaitEvent(stream, slot->lastWork, 0) != hipSuccess)
            return {nullptr,
                    RecordFailure(miopenStatusInternalError,
                                  "could not make the MIOpen handle's new stream wait for the "
                                  "hipDNN work queued on its old stream")};
        if(!fe::setHipdnnHandleStream(slot->hipdnnHandle, stream).is_good())
            return {nullptr,
                    RecordFailure(miopenStatusInternalError,
                                  "could not move the hipDNN handle to the MIOpen handle's new "
                                  "stream")};
        slot->stream = stream;
    }
    return {slot.get(), miopenStatusSuccess};
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
};

// Per-thread so that one thread's forwarded failure cannot be attributed to
// another thread's miopenGetErrorString call.
LastForwardedError& LastError()
{
    static thread_local LastForwardedError last;
    return last;
}

// Pointers into the set stay valid as it grows, because its elements are never
// moved.
struct InternedStrings
{
    std::mutex mutex;
    std::unordered_set<std::string> strings;
};

miopenStatus_t RecordHipdnnFailure(const fe::Error& error)
{
    return RecordFailure(TranslateHipdnnError(error), error.get_message());
}

miopenStatus_t RecordSuccess()
{
    ClearForwardedFailure();
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

// One graph of a sequence, with the variant pack it runs on and the function
// that adds the scratch pointers to that pack.
struct GraphRun
{
    GraphPtr graph;
    VariantPack* variantPack;
    const BindScratchFn* bindScratch;
};

// Runs `runs` in order. They share one workspace, big enough for any of them,
// and one scratch after it, so each run sees the scratch at the same address.
miopenStatus_t RunGraphs(HandleState& state,
                         const std::vector<GraphRun>& runs,
                         void* callerWorkspace,
                         size_t callerWorkspaceSize,
                         size_t scratchBytes)
{
    const std::lock_guard<std::mutex> lock(state.mutex);

    size_t needed = 0;
    for(const GraphRun& run : runs)
    {
        int64_t runNeeded = 0;
        if(const fe::Error error = run.graph->get_workspace_size(runNeeded); !error.is_good())
            return RecordHipdnnFailure(error);
        needed = std::max(needed, static_cast<size_t>(runNeeded));
    }

    void* workspace = callerWorkspace;
    void* scratch   = nullptr;
    if(scratchBytes > 0)
    {
        // The scratch follows the graph's workspace in the handle's buffer.
        constexpr size_t alignment = 256;
        const size_t offset        = (needed + alignment - 1) / alignment * alignment;
        if(!state.EnsureWorkspace(offset + scratchBytes))
            return RecordFailure(miopenStatusAllocFailed, "hipDNN scratch allocation failed");
        workspace = state.workspace;
        scratch   = static_cast<char*>(state.workspace) + offset;
    }
    else if(needed > 0 && (callerWorkspace == nullptr || callerWorkspaceSize < needed))
    {
        if(!state.EnsureWorkspace(needed))
            return RecordFailure(miopenStatusAllocFailed, "hipDNN workspace allocation failed");
        workspace = state.workspace;
    }

    for(const GraphRun& run : runs)
    {
        if(scratch != nullptr && run.bindScratch != nullptr && *run.bindScratch)
            (*run.bindScratch)(*run.variantPack, scratch);
        const fe::Error error =
            run.graph->execute(*state.hipdnnHandle, *run.variantPack, workspace);
        // Recorded even when execute failed, because it may have queued work first.
        const bool recorded = hipEventRecord(state.lastWork, state.stream) == hipSuccess;
        if(!error.is_good())
            return RecordHipdnnFailure(error);
        if(!recorded)
            return RecordFailure(miopenStatusInternalError,
                                 "could not record the end of the hipDNN work on the MIOpen "
                                 "handle's stream");
    }

    return RecordSuccess();
}

// One graph to look up or build, then run.
struct GraphRequest
{
    const PlanKey* key;
    const PopulateGraphFn* populate;
    VariantPack* variantPack;
    const BindScratchFn* bindScratch;
};

miopenStatus_t RunCachedRequests(miopenHandle_t handle,
                                 const std::vector<GraphRequest>& requests,
                                 void* workspace,
                                 size_t workspaceSize,
                                 size_t scratchBytes)
{
    hipStream_t stream = nullptr;
    if(miopenGetStream_impl(handle, &stream) != miopenStatusSuccess)
        return RecordFailure(miopenStatusInternalError,
                             "could not read the MIOpen handle's stream");

    // MIOpen makes the handle's device current before each launch. Doing the same
    // here puts the workspace, event and hipDNN handle on that device. A null
    // stream reports the current device, so a handle on the null stream runs on
    // whichever device the caller has current.
    hipDevice_t device = 0;
    ScopedDevice scopedDevice;
    if(hipStreamGetDevice(stream, &device) != hipSuccess || !scopedDevice.Set(device))
        return RecordFailure(miopenStatusInternalError,
                             "could not make the MIOpen handle's device current");

    auto [state, acquired] = AcquireHandleState(handle, stream);
    if(state == nullptr)
        return acquired;

    // Every graph is built before any runs, so a sequence that cannot be built
    // does no work.
    std::vector<GraphRun> runs;
    runs.reserve(requests.size());
    for(const GraphRequest& request : requests)
    {
        auto [graph, status] = AcquireGraph(*request.key, *request.populate, *state->hipdnnHandle);
        if(graph == nullptr)
            return status;
        runs.push_back({graph, request.variantPack, request.bindScratch});
    }

    return RunGraphs(*state, runs, workspace, workspaceSize, scratchBytes);
}

} // namespace

miopenStatus_t RecordFailure(miopenStatus_t status, std::string_view reason)
{
    LastError() = LastForwardedError{true, status};
    if(ErrorLoggingEnabled())
    {
        std::string line = "[MIOpen] [hipDNN-forwarded] ";
        line += miopenGetErrorString_impl(status);
        line += ": ";
        line += reason;
        line += '\n';
        std::cerr << line;
    }
    return status;
}

miopenStatus_t RecordCurrentException() noexcept
{
    try
    {
        try
        {
            throw;
        }
        catch(const std::exception& e)
        {
            return RecordFailure(miopenStatusUnknownError,
                                 std::string("unexpected exception: ") + e.what());
        }
        catch(...)
        {
            return RecordFailure(miopenStatusUnknownError, "unexpected exception");
        }
    }
    catch(...)
    {
        // Building the reason ran out of memory, so record the status alone.
        LastError() = LastForwardedError{true, miopenStatusUnknownError};
        return miopenStatusUnknownError;
    }
}

void ClearForwardedFailure() noexcept { LastError().failed = false; }

bool LastForwardedFailure(miopenStatus_t* status) noexcept
{
    const LastForwardedError& last = LastError();
    if(!last.failed)
        return false;
    *status = last.status;
    return true;
}

miopenStatus_t RunCachedGraph(miopenHandle_t handle,
                              const PlanKey& key,
                              const PopulateGraphFn& populate,
                              VariantPack& variantPack,
                              void* workspace,
                              size_t workspaceSize,
                              const GraphScratch& scratch)
{
    return RunCachedRequests(handle,
                             {{&key, &populate, &variantPack, &scratch.bind}},
                             workspace,
                             workspaceSize,
                             scratch.bytes);
}

miopenStatus_t
RunCachedGraphs(miopenHandle_t handle, std::vector<GraphStep>& steps, size_t scratchBytes)
{
    std::vector<GraphRequest> requests;
    requests.reserve(steps.size());
    for(GraphStep& step : steps)
        requests.push_back({&step.key, &step.populate, &step.variantPack, &step.bindScratch});
    return RunCachedRequests(handle, requests, nullptr, 0, scratchBytes);
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
        if(ErrorLoggingEnabled())
            std::cerr << "[MIOpen] hipDNN forwarding is unavailable (if hipDNN gives no reason, "
                         "set HIPDNN_LOG_LEVEL=error): " +
                             error.get_message() + "\n";
        return false;
    }();
    return available;
}

void ReleaseHandle(miopenHandle_t handle) noexcept
{
    try
    {
        {
            const std::lock_guard<std::mutex> lock(PlanMutex());
            PlanCache().EraseIf([handle](const PlanKey& key) { return key.handle == handle; });
        }

        const std::lock_guard<std::mutex> lock(HandleMapMutex());
        HandleMap().erase(handle);
    }
    catch(...)
    {
        // miopenDestroy must still succeed, so the state is left allocated.
        std::cerr << "[MIOpen] could not release the hipDNN state of a destroyed MIOpen handle\n";
    }
}

const char* PrefixedErrorString(miopenStatus_t status, const char* nativeMessage) noexcept
{
    const LastForwardedError& last = LastError();
    if(!last.failed || last.status != status || nativeMessage == nullptr)
        return nullptr;

    try
    {
        // Never freed, so a returned pointer stays valid for the whole process, as
        // MIOpen's own strings do. It holds at most one string per status.
        static auto* const prefixed = new InternedStrings;
        const std::lock_guard<std::mutex> lock(prefixed->mutex);
        return prefixed->strings.insert("[hipDNN-forwarded] " + std::string(nativeMessage))
            .first->c_str();
    }
    catch(...)
    {
        return nullptr;
    }
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
