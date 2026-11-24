// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_graph.hpp"
#include "hipdnn_types.hpp"

#include "erase_outside_lock.hpp"
#include "insert_outside_lock.hpp"
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
#include <optional>
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
        size_t seed = 0;
        for(const int64_t value : key.problem)
            seed ^= static_cast<size_t>(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= static_cast<size_t>(key.workspaceLimit) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

using GraphPtr = std::shared_ptr<fe::graph::Graph>;

// One handle's built graphs, one per problem. Not capped: a cached graph holds only host
// bookkeeping (its loaded kernels live in the engine plugins' shared caches), and a cap
// below a workload's working set would rebuild graphs on every call, since a training
// step visits its problems in the same order every step.
class PlanMap
{
public:
    const GraphPtr* Find(const PlanKey& key) const
    {
        const auto found = plans_.find(key);
        return found == plans_.end() ? nullptr : &found->second;
    }
    void Insert(const PlanKey& key, GraphPtr graph)
    {
        plans_.insert_or_assign(key, std::move(graph));
    }

private:
    std::unordered_map<PlanKey, GraphPtr, PlanKeyHash> plans_;
};

// Everything hipDNN needs that is tied to one MIOpen handle. Work on the handle is
// ordered by its stream alone: every call runs in the caller's workspace, so
// after a stream switch nothing the wrapper owns is shared with work still queued on the
// old stream, and ordering across the switch is the caller's.
struct HandleState
{
    fe::HipdnnHandlePtr hipdnnHandle;
    hipStream_t stream = nullptr;
    std::mutex mutex;
    // Declared after the hipDNN handle so the graphs built with it are destroyed first.
    PlanMap plans;
    std::mutex plansMutex;
};

std::mutex& HandleMapMutex()
{
    static std::mutex mutex;
    return mutex;
}

// Never destroyed, and with it the graphs each handle's state holds. Destroying a hipDNN
// handle runs the engine plugins' teardown, and the MIOpen provider's destroys its MIOpen
// handle, which unloads HIP modules. At process exit the HIP runtime has already torn its state
// down by the time a static destructor here would run, so that unload frees freed memory. The OS
// reclaims what is left at exit; a handle released through miopenDestroy is still torn down then,
// while HIP is alive.
std::unordered_map<miopenHandle_t, std::unique_ptr<HandleState>>& HandleMap()
{
    static auto* const handles =
        new std::unordered_map<miopenHandle_t, std::unique_ptr<HandleState>>;
    return *handles;
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
    auto state          = std::make_unique<HandleState>();
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
AcquireGraph(HandleState& state, const PlanKey& key, const PopulateGraphFn& populate)
{
    return AcquireGraphWith<fe::graph::Graph>(
        state.plans,
        state.plansMutex,
        key,
        populate,
        [&](fe::graph::Graph& graph) { return graph.build(*state.hipdnnHandle); },
        FailureOfHipdnnResult);
}

// The plugin SDK's global knob that bounds the workspace an engine's plan may use; an engine
// that offers it chooses its kernel within the bound when the plan runs.
constexpr const char* kWorkspaceSizeLimitKnob = "global.workspace_size_limit";

// Builds the best plan whose workspace is within `limit`, walking hipDNN's ranked engine configs:
// a config whose workspace is within the limit is used as it is; one that is not, but whose
// engine offers the workspace-limit knob, is used with the knob set to the limit, so the engine
// picks its kernel within it (the MIOpen provider runs MIOpen's Find with that workspace). The
// engine filters set by `populate` still apply. Returns the last failure when no config fits.
fe::Error BuildWithinWorkspace(fe::graph::Graph& graph, hipdnnHandle_t handle, int64_t limit)
{
    fe::Error result = graph.build_operation_graph(handle);
    std::vector<fe::EngineConfigInfo> configs;
    if(result.is_good())
        result = graph.get_engine_configs(handle, configs);
    if(result.is_bad())
        return result;

    result = {fe::ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
              "no hipDNN engine fits the " + std::to_string(limit) + "-byte workspace given"};
    for(const auto& config : configs)
    {
        std::vector<fe::KnobSetting> settings;
        if(config.estimatedWorkspaceSize > limit)
        {
            const bool limitable =
                std::any_of(config.knobs.begin(), config.knobs.end(), [](const auto& knob) {
                    return knob.knobId() == kWorkspaceSizeLimitKnob;
                });
            if(!limitable)
                continue;
            settings.emplace_back(kWorkspaceSizeLimitKnob, limit);
        }
        fe::Error attempt = graph.create_execution_plan_ext(config.engineId, settings);
        if(attempt.is_good())
            attempt = graph.check_support();
        if(attempt.is_good())
            attempt = graph.build_plans();
        int64_t needed = 0;
        if(attempt.is_good())
            attempt = graph.get_workspace_size(needed);
        if(attempt.is_good() && needed > limit)
            attempt = {fe::ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                       "the plan needs " + std::to_string(needed) + " bytes of workspace"};
        result = attempt;
        if(result.is_good())
            break;
    }
    return result;
}

miopenStatus_t RunGraph(HandleState& state,
                        const GraphPtr& graph,
                        VariantPack& variantPack,
                        size_t needed,
                        void* callerWorkspace,
                        size_t callerWorkspaceSize)
{
    return RunGraphWith(
        state.mutex,
        needed,
        callerWorkspace,
        callerWorkspaceSize,
        [&](void* workspace) {
            return graph->execute(*state.hipdnnHandle, variantPack, workspace);
        },
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

// The part every forwarded call shares: make the handle's device current, then
// find or create the handle's hipDNN state and the problem's built graph.
// `scopedDevice` keeps that device current until the caller returns.
static miopenStatus_t AcquireHandleAndGraph(miopenHandle_t handle,
                                            const PlanKey& key,
                                            const PopulateGraphFn& populate,
                                            ScopedDevice& scopedDevice,
                                            HandleState*& state,
                                            GraphPtr& graph)
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
    if(hipStreamGetDevice(stream, &device) != hipSuccess || !scopedDevice.Set(device))
        return RecordFailure(miopenStatusInternalError,
                             "could not make the MIOpen handle's device current");

    auto [acquiredState, acquired] = AcquireHandleState(handle, stream);
    if(acquiredState == nullptr)
        return acquired;
    state = acquiredState;

    auto [acquiredGraph, status] = AcquireGraph(*state, key, populate);
    if(acquiredGraph == nullptr)
        return status;
    graph = std::move(acquiredGraph);
    return miopenStatusSuccess;
}

// The plan's workspace in bytes.
static miopenStatus_t PlanWorkspace(const GraphPtr& graph, size_t& workspaceSize)
{
    int64_t needed     = 0;
    const auto queried = CallInsideHipdnn([&] { return graph->get_workspace_size(needed); });
    if(const auto failure = FailureOfGuardedCall(queried, FailureOfHipdnnResult))
        return *failure;
    workspaceSize = needed > 0 ? static_cast<size_t>(needed) : 0;
    return miopenStatusSuccess;
}

// The best plan whose workspace fits `limit` bytes: hipDNN's best plan when it fits, otherwise
// the first of hipDNN's ranked engine configs that can run within the limit
// (BuildWithinWorkspace), cached under the limit. A caller may give less workspace than the size
// query reported; then the best choice that fits is wanted, and the call fails only when nothing
// fits. No limit (nullopt) is hipDNN's best plan.
static miopenStatus_t AcquireFittingGraph(miopenHandle_t handle,
                                          const PlanKey& key,
                                          const PopulateGraphFn& populate,
                                          std::optional<size_t> limit,
                                          ScopedDevice& scopedDevice,
                                          HandleState*& state,
                                          GraphPtr& graph,
                                          size_t& workspaceSize)
{
    if(const auto status = AcquireHandleAndGraph(handle, key, populate, scopedDevice, state, graph);
       status != miopenStatusSuccess)
        return status;
    if(const auto status = PlanWorkspace(graph, workspaceSize); status != miopenStatusSuccess)
        return status;
    if(!limit || workspaceSize <= *limit)
        return miopenStatusSuccess;

    PlanKey bounded        = key;
    bounded.workspaceLimit = static_cast<int64_t>(*limit);
    const int64_t bound    = bounded.workspaceLimit;
    HandleState& owner     = *state;
    auto [fitting, status] = AcquireGraphWith<fe::graph::Graph>(
        owner.plans,
        owner.plansMutex,
        bounded,
        populate,
        [&](fe::graph::Graph& g) { return BuildWithinWorkspace(g, *owner.hipdnnHandle, bound); },
        FailureOfHipdnnResult);
    if(fitting == nullptr)
        return RecordFailure(miopenStatusUnknownError,
                             "No suitable algorithm was found to execute the required "
                             "convolution: no hipDNN engine fits the " +
                                 std::to_string(*limit) + "-byte workspace given");
    graph = std::move(fitting);
    return PlanWorkspace(graph, workspaceSize);
}

miopenStatus_t RunCachedGraph(miopenHandle_t handle,
                              const PlanKey& key,
                              const PopulateGraphFn& populate,
                              VariantPack& variantPack,
                              void* workspace,
                              size_t workspaceSize)
{
    ScopedDevice scopedDevice;
    HandleState* state = nullptr;
    GraphPtr graph;
    size_t needed      = 0;
    const size_t given = workspace == nullptr ? 0 : workspaceSize;
    if(const auto status =
           AcquireFittingGraph(handle, key, populate, given, scopedDevice, state, graph, needed);
       status != miopenStatusSuccess)
        return status;

    return RunGraph(*state, graph, variantPack, needed, workspace, workspaceSize);
}

miopenStatus_t CachedGraphWorkspaceSize(miopenHandle_t handle,
                                        const PlanKey& key,
                                        const PopulateGraphFn& populate,
                                        std::optional<size_t> limit,
                                        size_t& workspaceSize)
{
    ScopedDevice scopedDevice;
    HandleState* state = nullptr;
    GraphPtr graph;
    if(const auto status = AcquireFittingGraph(
           handle, key, populate, limit, scopedDevice, state, graph, workspaceSize);
       status != miopenStatusSuccess)
        return status;
    ClearForwardedFailure();
    return miopenStatusSuccess;
}

miopenStatus_t CachedGraphWorkspaceRange(miopenHandle_t handle,
                                         const PlanKey& key,
                                         const PopulateGraphFn& populate,
                                         size_t& minWorkspaceSize,
                                         size_t& maxWorkspaceSize)
{
    ScopedDevice scopedDevice;
    HandleState* state = nullptr;
    GraphPtr graph;
    size_t best = 0;
    if(const auto status = AcquireFittingGraph(
           handle, key, populate, std::nullopt, scopedDevice, state, graph, best);
       status != miopenStatusSuccess)
        return status;

    // Each engine's config bounds the workspace of every plan built from it, so a limit at
    // an engine's estimate admits that engine. The smallest limit that yields a plan is the
    // smallest workspace any engine can run in; at or above it a plan always fits.
    std::vector<fe::EngineConfigInfo> configs;
    {
        fe::graph::Graph probe;
        if(!populate(probe))
            return RecordFailure(miopenStatusUnsupportedOp,
                                 "could not build a hipDNN graph for this problem");
        const auto listed = CallInsideHipdnn([&]() -> fe::Error {
            if(auto error = probe.build_operation_graph(*state->hipdnnHandle); error.is_bad())
                return error;
            return probe.get_engine_configs(*state->hipdnnHandle, configs);
        });
        if(const auto failure = FailureOfGuardedCall(listed, FailureOfHipdnnResult))
            return *failure;
    }
    std::vector<size_t> candidates{0};
    for(const auto& config : configs)
    {
        if(config.estimatedWorkspaceSize > 0 &&
           static_cast<size_t>(config.estimatedWorkspaceSize) < best)
            candidates.push_back(static_cast<size_t>(config.estimatedWorkspaceSize));
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    minWorkspaceSize = best;
    for(const size_t candidate : candidates)
    {
        if(candidate >= best)
            break;
        GraphPtr fitting;
        size_t needed = 0;
        if(AcquireFittingGraph(
               handle, key, populate, candidate, scopedDevice, state, fitting, needed) ==
           miopenStatusSuccess)
        {
            minWorkspaceSize = needed;
            break;
        }
    }
    maxWorkspaceSize = best;
    ClearForwardedFailure();
    return miopenStatusSuccess;
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
        // The handle's built graphs go with its state.
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
