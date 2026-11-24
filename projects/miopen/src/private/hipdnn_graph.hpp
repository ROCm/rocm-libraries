// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The part of the hipDNN forwarding that does not depend on the operation. The
// first half is what the rest of the wrapper calls: probing the backend,
// releasing a handle's hipDNN state and prefixing forwarded error messages. The
// second half is what each operation's file (hipdnn_conv.cpp, hipdnn_batchnorm.cpp) uses: it
// describes its problem as a PlanKey and supplies the function that builds the
// graph for it, and RunCachedGraph does the rest.
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <miopen/miopen.h>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// Declared rather than included so that the wrapper's stub files and routing.cpp,
// which only need the first half of this header, do not pull in the hipDNN
// frontend.
namespace hipdnn_frontend {
namespace graph {
class Graph;
} // namespace graph
} // namespace hipdnn_frontend

namespace miopen {
namespace wrapper {
namespace hipdnn {

// True when hipDNN can serve a forwarded call. Loads the backend on first use, so
// a process that forwards nothing should never reach it. The answer is cached for
// the process, and a false one also prints one line to stderr saying why, unless
// MIOPEN_LOG_LEVEL hides errors.
bool IsAvailable();

// Drops whatever hipDNN state was created for this MIOpen handle. Called from
// the miopenDestroy stub on both routes, because a handle can be destroyed after
// MIOPEN_DISABLE_HIPDNN_FOR took its entry points back off the hipDNN path.
// Like miopenDestroy, it must not run while another call is using the handle:
// calls use the handle's state without holding the map lock.
void ReleaseHandle(miopenHandle_t handle) noexcept;

// Replacement text for miopenGetErrorString when the last wrapped call on this
// thread was forwarded and failed with `status`, or null otherwise. The text is
// fixed per status and lives for the whole process, like MIOpen's own strings,
// so a caller may keep the pointer. The failure's reason, when recorded, goes to
// stderr instead.
//
// This lets a user reading a log tell a forwarded failure from the same status
// raised by MIOpen itself; programs can ask miopenGetLastForwardedError instead.
const char* PrefixedErrorString(miopenStatus_t status, const char* nativeMessage) noexcept;

// Every stub that MIOpen serves calls this, so a later MIOpen failure with the
// same status is not reported as forwarded.
void ClearForwardedFailure() noexcept;

// Writes the status of this thread's last forwarded failure and returns true, or
// returns false when the last wrapped call was not a failed forwarded one.
bool LastForwardedFailure(miopenStatus_t* status) noexcept;

// The hipDNN path for every entry point that has no forwarding of its own yet;
// each is swapped for a real implementation as it lands. Until then, an entry
// point added to the forwarding set fails loudly here, with its failure recorded
// like any other forwarded one, instead of silently running MIOpen.
miopenStatus_t ForwardingNotImplemented(const char* entryPoint);

// Marks this thread as inside a hipDNN build or execute call. A plugin built against the
// public libMIOpen.so binds to this wrapper, so its call to a forwarded entry point would be
// forwarded back into hipDNN, which calls the plugin again, without end (RFC 0001 section 4.7).
//
// Thread-local, so a plugin that hands the call to a thread of its own is not seen. Only
// build and execute are marked, which is where the provider calls forwarded entry points. A
// plugin also runs during workspace queries and handle creation and destruction, and a
// forwarded entry point called from there is not rejected.
class ScopedHipdnnCall
{
public:
    ScopedHipdnnCall() noexcept;
    ~ScopedHipdnnCall();
    ScopedHipdnnCall(const ScopedHipdnnCall&)            = delete;
    ScopedHipdnnCall& operator=(const ScopedHipdnnCall&) = delete;

    // True if RejectReentrantCall ran on this thread since the outermost scope opened.
    // Check it after hipDNN returns: the plugin may have swallowed the error it was given.
    bool ReentryDetected() const noexcept;
};

// True while a ScopedHipdnnCall is open on this thread.
bool InHipdnnCall() noexcept;

// What a forwarded entry point does instead of forwarding while InHipdnnCall() is true.
// Remembers the entry point for ReentryDetected, writes a route trace line if MIOPEN_LOG_LEVEL
// asks for one, and returns miopenStatusInternalError. It records no failure and prints no
// error line: the call that opened the scope reports once, through RecordReentryFailure.
miopenStatus_t RejectReentrantCall(const char* entryPoint) noexcept;

// Records that a hipDNN plugin re-entered the wrapper as a forwarded failure and returns
// miopenStatusInternalError. Call it when ReentryDetected() is true, whatever hipDNN returned.
miopenStatus_t RecordReentryFailure() noexcept;

// The two functions above with explicit output, so tests can run both the logging and the
// silent case: `trace` and `log` choose whether the trace line and the error line go to `os`.
miopenStatus_t RejectReentrantCallTo(const char* entryPoint, bool trace, std::ostream& os) noexcept;
miopenStatus_t RecordReentryFailureTo(bool log, std::ostream& os) noexcept;

// What a hipDNN build or execute call returned, and whether a forwarded entry point was
// rejected while it ran.
template <typename Result>
struct GuardedResult
{
    Result result;
    bool reentered;
};

// Runs `call`, a hipDNN build or execute, with this thread marked as inside hipDNN. The
// scope is opened here, not at each call site, so no call into hipDNN can go without one.
template <typename Call>
auto CallInsideHipdnn(Call&& call) -> GuardedResult<std::decay_t<decltype(call())>>
{
    const ScopedHipdnnCall scope;
    auto result = call();
    return {std::move(result), scope.ReentryDetected()};
}

// The failure to report for `guarded`, or nullopt when the call worked. `hipdnnFailure`
// maps what hipDNN returned to its failure, or nullopt. A re-entry takes precedence over
// it: a plugin that ignored the error it was given makes hipDNN report success for an
// output nobody wrote.
template <typename Result, typename HipdnnFailure>
std::optional<miopenStatus_t> FailureOfGuardedCall(const GuardedResult<Result>& guarded,
                                                   HipdnnFailure&& hipdnnFailure)
{
    if(guarded.reentered)
        return RecordReentryFailure();
    return hipdnnFailure(guarded.result);
}

// One value per kind of graph the wrapper builds. Kept in one list so that the
// values stay distinct as operations are added: it is the first element of
// every PlanKey.
enum class GraphKind : int64_t
{
    ConvFprop,
    ConvDgrad,
    ConvWgrad,
    ConvBiasActivation,
    BatchnormInference,
};

// Identifies one cached graph. `problem` is everything that tells two problems
// apart, flattened to integers by the operation. It starts with the GraphKind,
// and every variable-length list in it is preceded by its length, so two
// different problems never flatten to the same sequence.
struct PlanKey
{
    std::vector<int64_t> problem;
    // The largest workspace the plan may need, or -1 for no limit: hipDNN's best plan.
    int64_t workspaceLimit = -1;

    bool operator==(const PlanKey& other) const
    {
        return problem == other.problem && workspaceLimit == other.workspaceLimit;
    }
};

// Records a forwarded failure for miopenGetErrorString and
// miopenGetLastForwardedError to report, writes `reason` to stderr unless
// MIOPEN_LOG_LEVEL hides errors, and returns `status`.
miopenStatus_t RecordFailure(miopenStatus_t status, std::string_view reason);

// RecordFailure with explicit output: the error line goes to `os` if `log` is set.
miopenStatus_t
RecordFailureTo(miopenStatus_t status, std::string_view reason, bool log, std::ostream& os);

// Records the exception being handled as a forwarded miopenStatusUnknownError,
// the status MIOpen returns for one, and returns it. Call only from a catch
// block.
miopenStatus_t RecordCurrentException() noexcept;

// Finds or builds the graph for `key`. `populate` fills in a new graph, `build(graph)`
// finalizes it in hipDNN, and `failureOf` turns what `build` returned into a recorded
// failure, or nullopt. They are parameters so that tests can stand in for hipDNN.
//
// The mutex is held across the build on purpose: a build can take seconds, but it happens
// once per problem, and serializing it is simpler than letting two threads race to build the
// same graph.
template <typename Graph, typename Cache, typename Build, typename FailureOf>
std::pair<std::shared_ptr<Graph>, miopenStatus_t>
AcquireGraphWith(Cache& cache,
                 std::mutex& mutex,
                 const PlanKey& key,
                 const std::function<bool(Graph&)>& populate,
                 Build&& build,
                 FailureOf&& failureOf)
{
    const std::lock_guard<std::mutex> lock(mutex);

    if(const auto* cached = cache.Find(key))
        return {*cached, miopenStatusSuccess};

    auto graph = std::make_shared<Graph>();
    if(!populate(*graph))
        return {nullptr,
                RecordFailure(miopenStatusUnsupportedOp,
                              "could not build a hipDNN graph for this problem")};

    const auto built = CallInsideHipdnn([&] { return build(*graph); });
    // Checked before caching: a graph built during a re-entry is not worth reusing.
    if(const auto failure = FailureOfGuardedCall(built, failureOf))
        return {nullptr, *failure};

    cache.Insert(key, graph);
    return {graph, miopenStatusSuccess};
}

// Runs a built graph under `mutex`. `needed` is the plan's workspace in bytes, read when
// the plan was acquired; `execute` runs the graph on the given workspace, and `failureOf`
// turns what hipDNN returned into a recorded failure, or nullopt.
template <typename Execute, typename FailureOf>
miopenStatus_t RunGraphWith(std::mutex& mutex,
                            size_t needed,
                            void* callerWorkspace,
                            size_t callerWorkspaceSize,
                            Execute&& execute,
                            FailureOf&& failureOf)
{
    const std::lock_guard<std::mutex> lock(mutex);

    // The workspace is always the caller's: the plan was chosen to fit it (RunCachedGraph), as
    // MIOpen chooses a solver that fits, and the wrapper never allocates one of its own.
    const size_t given = callerWorkspace == nullptr ? 0 : callerWorkspaceSize;
    if(given < needed)
        return RecordFailure(miopenStatusUnknownError,
                             "Not enough workspace for the hipDNN plan: " + std::to_string(needed) +
                                 " bytes needed, " + std::to_string(given) + " given");

    const auto executed = CallInsideHipdnn([&] { return execute(callerWorkspace); });
    if(const auto failure = FailureOfGuardedCall(executed, failureOf))
        return *failure;

    ClearForwardedFailure();
    return miopenStatusSuccess;
}

using PopulateGraphFn = std::function<bool(hipdnn_frontend::graph::Graph&)>;
using VariantPack     = std::unordered_map<int64_t, void*>;

// Runs the graph for `key` on `handle`'s stream. On a cache miss, `populate`
// fills in a new graph, which is then built and cached. The plan is the best one
// whose workspace fits the caller's (hipDNN's best plan when it fits), and runs in
// the caller's workspace; when no engine fits, the call fails, as MIOpen's does.
// The wrapper never allocates a workspace. Failures are recorded with RecordFailure.
miopenStatus_t RunCachedGraph(miopenHandle_t handle,
                              const PlanKey& key,
                              const PopulateGraphFn& populate,
                              VariantPack& variantPack,
                              void* workspace,
                              size_t workspaceSize);

// The workspace of the plan for `key` that fits `limit` bytes (no limit: hipDNN's best
// plan), building and caching it on first use exactly as RunCachedGraph does, without
// running it. Serves the forwarded Find, workspace-size and solution queries, so the size
// they report is the one the forwarded execution, given that workspace, will use. Fails as
// MIOpen does when no engine fits.
miopenStatus_t CachedGraphWorkspaceSize(miopenHandle_t handle,
                                        const PlanKey& key,
                                        const PopulateGraphFn& populate,
                                        std::optional<size_t> limit,
                                        size_t& workspaceSize);

// The smallest workspace any engine can run the problem in (at or above it a plan always
// fits), and hipDNN's best plan's workspace, which is the most a caller ever needs.
miopenStatus_t CachedGraphWorkspaceRange(miopenHandle_t handle,
                                         const PlanKey& key,
                                         const PopulateGraphFn& populate,
                                         size_t& minWorkspaceSize,
                                         size_t& maxWorkspaceSize);

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
