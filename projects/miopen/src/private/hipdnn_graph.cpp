// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_graph.hpp"
#include "hipdnn_types.hpp"

#include "erase_outside_lock.hpp"
#include "insert_outside_lock.hpp"
#include "lru_cache.hpp"
#include "miopen_impl.h"
#include "routing.hpp"

#include <hipdnn_frontend.hpp>

#include <hip/hip_runtime_api.h>

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

// Creating a handle runs the plugin's code, and so does destroying one that is not kept. For a
// plugin built against the public libMIOpen.so that code calls miopenDestroy, which takes
// HandleMapMutex, so neither happens while the lock is held.
std::pair<std::unique_ptr<HandleState>, miopenStatus_t> CreateHandleState(hipStream_t stream)
{
    auto [created, error] = fe::createHipdnnHandle(stream);
    if(!error.is_good() || created == nullptr)
        return {nullptr,
                RecordFailure(miopenStatusInternalError, "could not create a hipDNN handle")};
    auto state = std::make_unique<HandleState>();
    if(hipEventCreateWithFlags(&state->lastWork, hipEventDisableTiming) != hipSuccess)
        return {nullptr,
                RecordFailure(miopenStatusInternalError,
                              "could not create the HIP event that orders hipDNN work")};
    state->hipdnnHandle = std::move(created);
    state->stream       = stream;
    return {std::move(state), miopenStatusSuccess};
}

// Created on first forwarded call rather than in miopenCreate, so a process that
// never forwards never pays for hipdnnCreate.
std::pair<HandleState*, miopenStatus_t> AcquireHandleState(miopenHandle_t handle,
                                                           hipStream_t stream)
{
    // Two threads using one MIOpen handle at once can both miss and both create a state;
    // the one that loses is dropped and the loop finds the winner's.
    for(;;)
    {
        {
            const std::lock_guard<std::mutex> lock(HandleMapMutex());
            const auto found = HandleMap().find(handle);
            if(found != HandleMap().end())
            {
                HandleState& state = *found->second;
                if(state.stream != stream)
                {
                    // Work still queued on the old stream may be using the shared workspace.
                    if(hipStreamWaitEvent(stream, state.lastWork, 0) != hipSuccess)
                        return {nullptr,
                                RecordFailure(
                                    miopenStatusInternalError,
                                    "could not make the MIOpen handle's new stream wait for the "
                                    "hipDNN work queued on its old stream")};
                    if(!fe::setHipdnnHandleStream(state.hipdnnHandle, stream).is_good())
                        return {nullptr,
                                RecordFailure(miopenStatusInternalError,
                                              "could not move the hipDNN handle to the MIOpen "
                                              "handle's new stream")};
                    state.stream = stream;
                }
                return {&state, miopenStatusSuccess};
            }
        }

        auto [created, status] = CreateHandleState(stream);
        if(created == nullptr)
            return {nullptr, status};
        HandleState* const raw = created.get();
        if(InsertDestroyingLoserOutsideLock(
               HandleMap(), HandleMapMutex(), handle, std::move(created)))
            return {raw, miopenStatusSuccess};
    }
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

struct ReentryState
{
    int depth = 0;
    // The entry point most recently rejected in the outermost scope; null means none.
    const char* rejected = nullptr;
};

ReentryState& Reentry()
{
    static thread_local ReentryState state;
    return state;
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

std::optional<miopenStatus_t> FailureOfHipdnnResult(const fe::Error& error)
{
    if(error.is_good())
        return std::nullopt;
    return RecordHipdnnFailure(error);
}

std::pair<GraphPtr, miopenStatus_t>
AcquireGraph(const PlanKey& key, const PopulateGraphFn& populate, hipdnnHandle_t hipdnnHandle)
{
    return AcquireGraphWith<fe::graph::Graph>(
        PlanCache(),
        PlanMutex(),
        key,
        populate,
        [&](fe::graph::Graph& graph) { return graph.build(hipdnnHandle); },
        FailureOfHipdnnResult);
}

miopenStatus_t RunGraph(HandleState& state,
                        const GraphPtr& graph,
                        VariantPack& variantPack,
                        void* callerWorkspace,
                        size_t callerWorkspaceSize)
{
    return RunGraphWith(
        state.mutex,
        callerWorkspace,
        callerWorkspaceSize,
        [&](int64_t& needed) { return graph->get_workspace_size(needed); },
        [&](size_t bytes) -> void* {
            return state.EnsureWorkspace(bytes) ? state.workspace : nullptr;
        },
        [&](void* workspace) {
            return graph->execute(*state.hipdnnHandle, variantPack, workspace);
        },
        [&] { return hipEventRecord(state.lastWork, state.stream) == hipSuccess; },
        FailureOfHipdnnResult);
}

} // namespace

miopenStatus_t
RecordFailureTo(miopenStatus_t status, std::string_view reason, bool log, std::ostream& os)
{
    LastError() = LastForwardedError{true, status};
    if(log)
    {
        std::string line = "[MIOpen] [hipDNN-forwarded] ";
        line += miopenGetErrorString_impl(status);
        line += ": ";
        line += reason;
        line += '\n';
        os << line;
    }
    return status;
}

miopenStatus_t RecordFailure(miopenStatus_t status, std::string_view reason)
{
    return RecordFailureTo(status, reason, ErrorLoggingEnabled(), std::cerr);
}

miopenStatus_t ForwardingNotImplemented(const char* entryPoint)
{
    return RecordFailure(miopenStatusNotImplemented,
                         std::string(entryPoint) +
                             " is in the forwarding set but has no hipDNN implementation");
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

ScopedHipdnnCall::ScopedHipdnnCall() noexcept
{
    ReentryState& state = Reentry();
    if(state.depth == 0)
        state.rejected = nullptr;
    ++state.depth;
}

ScopedHipdnnCall::~ScopedHipdnnCall() { --Reentry().depth; }

bool ScopedHipdnnCall::ReentryDetected() const noexcept { return Reentry().rejected != nullptr; }

bool InHipdnnCall() noexcept { return Reentry().depth > 0; }

miopenStatus_t RejectReentrantCallTo(const char* entryPoint, bool trace, std::ostream& os) noexcept
{
    Reentry().rejected = entryPoint != nullptr ? entryPoint : "<null>";
    try
    {
        if(trace)
            ReportRouteDecision(entryPoint,
                                ForwardingMode::Enabled,
                                {Route::Rejected, RouteReason::ReentrantFromHipdnn},
                                os);
    }
    catch(...)
    {
        // A lost trace line must not turn the rejection into an exception.
    }
    return miopenStatusInternalError;
}

miopenStatus_t RejectReentrantCall(const char* entryPoint) noexcept
{
    return RejectReentrantCallTo(entryPoint, RouteTraceEnabled(), std::cerr);
}

miopenStatus_t RecordReentryFailureTo(bool log, std::ostream& os) noexcept
{
#ifdef _WIN32
    constexpr const char* publicLibrary  = "MIOpen.dll";
    constexpr const char* privateLibrary = "MIOpen_private.dll";
#else
    constexpr const char* publicLibrary  = "libMIOpen.so";
    constexpr const char* privateLibrary = "libMIOpen_private.so";
#endif
    try
    {
        const char* entryPoint = Reentry().rejected;
        return RecordFailureTo(
            miopenStatusInternalError,
            std::string("a hipDNN plugin called the public MIOpen entry point ") +
                (entryPoint != nullptr ? entryPoint : "<unknown>") +
                " while hipDNN was running a forwarded call. The plugin was built against " +
                publicLibrary + ", not " + privateLibrary +
                ", so it would call back into hipDNN forever. Rebuild or reinstall the plugin "
                "from the same build as MIOpen.",
            log,
            os);
    }
    catch(...)
    {
        // Building the reason ran out of memory, so record the status alone.
        LastError() = LastForwardedError{true, miopenStatusInternalError};
        return miopenStatusInternalError;
    }
}

miopenStatus_t RecordReentryFailure() noexcept
{
    return RecordReentryFailureTo(ErrorLoggingEnabled(), std::cerr);
}

miopenStatus_t RunCachedGraph(miopenHandle_t handle,
                              const PlanKey& key,
                              const PopulateGraphFn& populate,
                              VariantPack& variantPack,
                              void* workspace,
                              size_t workspaceSize)
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

    auto [graph, status] = AcquireGraph(key, populate, *state->hipdnnHandle);
    if(graph == nullptr)
        return status;

    return RunGraph(*state, graph, variantPack, workspace, workspaceSize);
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

        EraseDestroyingOutsideLock(HandleMap(), HandleMapMutex(), handle);
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
