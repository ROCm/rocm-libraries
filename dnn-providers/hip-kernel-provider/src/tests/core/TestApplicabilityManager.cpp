// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/ApplicabilityManager.hpp"
#include "core/ApplicabilityPool.hpp"
#include "core/Container.hpp"
#include "core/Handle.hpp"

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/ingestor/StagedApplicability.hpp>
#endif

using namespace hip_kernel_provider::core;

namespace
{

using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;
using hipdnn_test_sdk::utilities::SharedLogRecorder;

/// Every task in this file finishes in milliseconds when the code under test is right. The
/// waits are bounded only so that a regression fails instead of hanging the binary.
constexpr auto RENDEZVOUS_TIMEOUT = std::chrono::seconds(5);

/// Runs a batch of @p taskCount on @p pool, calling @p extra for each index, and reports which
/// thread ran each one.
std::vector<std::thread::id> runOnPool(ApplicabilityPool& pool,
                                       std::size_t taskCount,
                                       std::size_t maxWorkers,
                                       bool& ran,
                                       const std::function<void(std::size_t)>& extra = {})
{
    std::vector<std::thread::id> threads(taskCount);
    ran = pool.tryRun(taskCount, maxWorkers, [&](std::size_t index) {
        threads[index] = std::this_thread::get_id();
        if(extra)
        {
            extra(index);
        }
    });
    return threads;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------------------------

TEST(TestApplicabilityKnobs, MaxThreadsDefaultsToOneWhenUnsetOrEmpty)
{
    EXPECT_EQ(parseMaxThreads(""), 1u);

    const ScopedEnvironmentVariableSetter guard(APPLICABILITY_MAX_THREADS_ENV_NAME);
    hipdnn_data_sdk::utilities::unsetEnv(APPLICABILITY_MAX_THREADS_ENV_NAME);
    EXPECT_EQ(maxThreadsFromEnv(), 1u);
}

TEST(TestApplicabilityKnobs, MaxThreadsAutoAndZeroBothMeanAuto)
{
    EXPECT_EQ(parseMaxThreads("0"), 0u);
    EXPECT_EQ(parseMaxThreads("auto"), 0u);
    EXPECT_EQ(parseMaxThreads("AUTO"), 0u);
    EXPECT_EQ(parseMaxThreads(" auto "), 0u);

    const ScopedEnvironmentVariableSetter guard(APPLICABILITY_MAX_THREADS_ENV_NAME, "auto");
    EXPECT_EQ(maxThreadsFromEnv(), 0u);
}

TEST(TestApplicabilityKnobs, MaxThreadsTakesAnExplicitCount)
{
    const auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    EXPECT_EQ(parseMaxThreads("3"), 3u);
    EXPECT_EQ(parseMaxThreads("1"), 1u);

    const ScopedEnvironmentVariableSetter guard(APPLICABILITY_MAX_THREADS_ENV_NAME, "3");
    EXPECT_EQ(maxThreadsFromEnv(), 3u);

    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_WARN), 0u) << recorder.getRecordedLogsAsString();
}

TEST(TestApplicabilityKnobs, MaxThreadsJunkFallsBackToOffWithAWarning)
{
    for(const std::string raw : {"junk", "-2", "3x", "99999999999999999999999999"})
    {
        SCOPED_TRACE(raw);
        const auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

        EXPECT_EQ(parseMaxThreads(raw), 1u);
        EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, raw))
            << recorder.getRecordedLogsAsString();
    }

    const ScopedEnvironmentVariableSetter guard(APPLICABILITY_MAX_THREADS_ENV_NAME, "junk");
    EXPECT_EQ(maxThreadsFromEnv(), 1u);
}

TEST(TestApplicabilityKnobs, MinOffloadVariantsDefaultsAndParses)
{
    const auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    EXPECT_EQ(parseMinOffloadVariants(""), 200u);
    EXPECT_EQ(parseMinOffloadVariants("0"), 0u);
    EXPECT_EQ(parseMinOffloadVariants("1000000000"), 1000000000u);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_WARN), 0u) << recorder.getRecordedLogsAsString();

    const ScopedEnvironmentVariableSetter guard(APPLICABILITY_MIN_OFFLOAD_VARIANTS_ENV_NAME);
    hipdnn_data_sdk::utilities::unsetEnv(APPLICABILITY_MIN_OFFLOAD_VARIANTS_ENV_NAME);
    EXPECT_EQ(minOffloadVariantsFromEnv(), 200u);
}

TEST(TestApplicabilityKnobs, MinOffloadVariantsJunkFallsBackToTheDefaultWithAWarning)
{
    for(const std::string raw : {"many", "-1", "12abc"})
    {
        SCOPED_TRACE(raw);
        const auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

        EXPECT_EQ(parseMinOffloadVariants(raw), 200u);
        EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, raw))
            << recorder.getRecordedLogsAsString();
    }
}

TEST(TestApplicabilityKnobs, AutoIsCappedByPendingCpusAndNine)
{
    EXPECT_EQ(resolveTotalThreads(0, 20, 64), 9u);
    EXPECT_EQ(resolveTotalThreads(0, 3, 64), 3u);
    EXPECT_EQ(resolveTotalThreads(0, 20, 4), 4u);
    EXPECT_GE(resolveTotalThreads(0, 20, 0), 1u);
}

TEST(TestApplicabilityKnobs, ExplicitCountIsCappedByPendingOnly)
{
    EXPECT_EQ(resolveTotalThreads(2, 20, 64), 2u);
    EXPECT_EQ(resolveTotalThreads(5, 3, 64), 3u);
    EXPECT_EQ(resolveTotalThreads(16, 20, 4), 16u);
    EXPECT_EQ(resolveTotalThreads(1, 20, 64), 1u);
}

// ---------------------------------------------------------------------------------------------
// Pool and budget
// ---------------------------------------------------------------------------------------------

TEST(TestApplicabilityPool, BudgetGrantsWhatItHasAndNeverMore)
{
    WorkerBudget budget{3};

    EXPECT_EQ(budget.tryAcquire(2), 2u);
    EXPECT_EQ(budget.available(), 1u);
    EXPECT_EQ(budget.tryAcquire(5), 1u);
    EXPECT_EQ(budget.tryAcquire(1), 0u);

    budget.release(3);
    EXPECT_EQ(budget.available(), 3u);
}

TEST(TestApplicabilityPool, RunsEveryIndexExactlyOnceAndIndexZeroOnTheCaller)
{
    WorkerBudget budget{4};
    const auto baseline = ApplicabilityPool::liveThreads();
    {
        ApplicabilityPool pool{budget};
        constexpr std::size_t TASKS = 64;
        std::vector<std::atomic<int>> runs(TASKS);

        bool ran = false;
        const auto threads = runOnPool(pool, TASKS, 3, ran, [&](std::size_t index) {
            runs[index].fetch_add(1);
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        });

        ASSERT_TRUE(ran);
        for(std::size_t i = 0; i < TASKS; ++i)
        {
            EXPECT_EQ(runs[i].load(), 1) << "index " << i;
        }
        EXPECT_EQ(threads[0], std::this_thread::get_id());
        EXPECT_LE(pool.threadCount(), 3u);

        // The workers' grant is returned once the call has drained.
        EXPECT_EQ(budget.available(), 4u);
    }
    EXPECT_EQ(ApplicabilityPool::liveThreads(), baseline);
}

TEST(TestApplicabilityPool, WorkersPickUpTasksWhileTheCallerIsBusy)
{
    WorkerBudget budget{2};
    ApplicabilityPool pool{budget};
    const auto caller = std::this_thread::get_id();

    std::mutex mutex;
    std::condition_variable cv;
    bool workerRan = false;

    bool ran = false;
    const auto threads = runOnPool(pool, 3, 2, ran, [&](std::size_t index) {
        if(index == 0)
        {
            // Hold the caller until a worker has run something, so the outcome cannot
            // depend on the caller draining the queue first.
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_for(lock, RENDEZVOUS_TIMEOUT, [&] { return workerRan; });
        }
        else if(std::this_thread::get_id() != caller)
        {
            const std::lock_guard<std::mutex> lock(mutex);
            workerRan = true;
            cv.notify_all();
        }
    });

    ASSERT_TRUE(ran);
    EXPECT_TRUE(std::any_of(
        threads.begin(), threads.end(), [&](std::thread::id id) { return id != caller; }));
    EXPECT_GE(pool.threadCount(), 1u);
}

TEST(TestApplicabilityPool, RunsNothingWhenNoWorkerCanBeHad)
{
    WorkerBudget empty{0};
    ApplicabilityPool pool{empty};
    std::atomic<int> runs{0};

    EXPECT_FALSE(pool.tryRun(4, 3, [&](std::size_t) { runs.fetch_add(1); }));
    EXPECT_EQ(runs.load(), 0);
    EXPECT_EQ(pool.threadCount(), 0u);
}

TEST(TestApplicabilityPool, RunsNothingForFewerThanTwoTasksOrNoWorkers)
{
    WorkerBudget budget{4};
    ApplicabilityPool pool{budget};
    std::atomic<int> runs{0};

    EXPECT_FALSE(pool.tryRun(1, 3, [&](std::size_t) { runs.fetch_add(1); }));
    EXPECT_FALSE(pool.tryRun(4, 0, [&](std::size_t) { runs.fetch_add(1); }));
    EXPECT_EQ(runs.load(), 0);
    EXPECT_EQ(pool.threadCount(), 0u);
    EXPECT_EQ(budget.available(), 4u);
}

TEST(TestApplicabilityPool, DestructionJoinsEveryWorker)
{
    WorkerBudget budget{4};
    const auto baseline = ApplicabilityPool::liveThreads();
    {
        ApplicabilityPool pool{budget};
        bool ran = false;
        runOnPool(pool, 8, 3, ran, [](std::size_t) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        });
        ASSERT_TRUE(ran);
        ASSERT_GE(pool.threadCount(), 1u);

        // A warm pool keeps its workers between calls.
        EXPECT_EQ(ApplicabilityPool::liveThreads(), baseline + pool.threadCount());
    }
    EXPECT_EQ(ApplicabilityPool::liveThreads(), baseline);
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// ---------------------------------------------------------------------------------------------
// Manager over fake engines
// ---------------------------------------------------------------------------------------------

namespace
{

using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
using hipdnn_plugin_sdk::HipdnnPluginException;
using hipdnn_plugin_sdk::ingestor::ApplicabilityScope;
using hipdnn_plugin_sdk::ingestor::IStagedApplicability;
using hipdnn_plugin_sdk::ingestor::PendingApplicability;
using hipdnn_plugin_sdk::ingestor::StagedApplicability;

using PlainEngine = hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>;

enum class Failure
{
    NONE,
    IN_BEGIN,
    IN_FINISH,
};

struct Thrown
{
    hipdnnPluginStatus_t status;
    std::string message;
};

/// What every fake engine reports into, shared across one test's engines so a finish can
/// wait for a sibling's finish on another thread.
class Journal
{
public:
    explicit Journal(std::thread::id caller)
        : _caller(caller)
    {
    }

    void recordFinish(int64_t engineId)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _finishes.emplace_back(engineId, std::this_thread::get_id());
        _cv.notify_all();
    }

    void recordHandWritten(int64_t engineId)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _handWritten.emplace_back(engineId, std::this_thread::get_id());
    }

    /// Blocks until some finish has run on a thread other than the current one.
    void awaitFinishOnAnotherThread()
    {
        const auto self = std::this_thread::get_id();
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait_for(lock, RENDEZVOUS_TIMEOUT, [&] {
            return std::any_of(_finishes.begin(), _finishes.end(), [&](const auto& entry) {
                return entry.second != self;
            });
        });
    }

    std::vector<std::pair<int64_t, std::thread::id>> finishes() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _finishes;
    }

    std::vector<std::pair<int64_t, std::thread::id>> handWritten() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _handWritten;
    }

    std::thread::id caller() const
    {
        return _caller;
    }

    bool anyFinishOffCaller() const
    {
        const auto all = finishes();
        return std::any_of(
            all.begin(), all.end(), [&](const auto& entry) { return entry.second != _caller; });
    }

    std::size_t finishCount(int64_t engineId) const
    {
        const auto all = finishes();
        return static_cast<std::size_t>(std::count_if(
            all.begin(), all.end(), [&](const auto& entry) { return entry.first == engineId; }));
    }

    std::thread::id finishThread(int64_t engineId) const
    {
        for(const auto& entry : finishes())
        {
            if(entry.first == engineId)
            {
                return entry.second;
            }
        }
        return {};
    }

    void clear()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _finishes.clear();
        _handWritten.clear();
    }

private:
    std::thread::id _caller;
    mutable std::mutex _mutex;
    std::condition_variable _cv;
    std::vector<std::pair<int64_t, std::thread::id>> _finishes;
    std::vector<std::pair<int64_t, std::thread::id>> _handWritten;
};

/// The non-applicability half of the engine interface, which no test here reaches.
class FakeEngineBase : public PlainEngine
{
public:
    explicit FakeEngineBase(int64_t engineId)
        : _id(engineId)
    {
    }

    int64_t id() const override
    {
        return _id;
    }

    void getDetails(Handle& /*handle*/,
                    const IGraph& /*graph*/,
                    hipdnnPluginConstData_t& /*details*/) const override
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "not reached");
    }

    size_t getMaxWorkspaceSize(
        const Handle& /*handle*/,
        const IGraph& /*graph*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*config*/)
        const override
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "not reached");
    }

    void initializeExecutionContext(
        const Handle& /*handle*/,
        const IGraph& /*graph*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*config*/,
        Context& /*context*/) const override
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "not reached");
    }

private:
    int64_t _id;
};

/// A hand-written engine: plain isApplicable only.
class FakeHandWrittenEngine : public FakeEngineBase
{
public:
    FakeHandWrittenEngine(int64_t engineId, bool verdict, Journal& journal)
        : FakeEngineBase(engineId)
        , _verdict(verdict)
        , _journal(journal)
    {
    }

    bool isApplicable(Handle& /*handle*/, const IGraph& /*graph*/) const override
    {
        _journal.recordHandWritten(id());
        return _verdict;
    }

private:
    bool _verdict;
    Journal& _journal;
};

struct StagedSpec
{
    int64_t id = 0;
    bool decided = false;
    bool verdict = false;
    std::size_t predictedVariants = 0;
    Failure failure = Failure::NONE;
    hipdnnPluginStatus_t failureStatus = HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    std::string failureMessage;
    /// Hold this finish until a sibling's finish has run on another thread.
    bool awaitSibling = false;
};

/// An ingestor-style engine. Its plain isApplicable is begin-then-finish on the caller, which
/// is the invariant the real engines keep, so the sequential reference and the staged path
/// agree by construction and every difference is the manager's.
///
/// The spec is shared with the test, which may change it between calls -- never during one --
/// once the manager's iteration order is known.
class FakeStagedEngine : public FakeEngineBase, public IStagedApplicability<Handle>
{
public:
    FakeStagedEngine(std::shared_ptr<StagedSpec> spec, Journal& journal)
        : FakeEngineBase(spec->id)
        , _spec(std::move(spec))
        , _journal(journal)
    {
    }

    bool isApplicable(Handle& handle, const IGraph& graph) const override
    {
        ApplicabilityScope scope;
        auto staged = beginApplicability(handle, graph, scope);
        if(staged.decided)
        {
            return staged.verdict;
        }
        return finishApplicability(std::move(staged.pending));
    }

    StagedApplicability beginApplicability(Handle& /*handle*/,
                                           const IGraph& /*graph*/,
                                           ApplicabilityScope& /*scope*/) const override
    {
        if(_spec->failure == Failure::IN_BEGIN)
        {
            throw HipdnnPluginException(_spec->failureStatus, _spec->failureMessage);
        }

        StagedApplicability staged;
        if(_spec->decided)
        {
            staged.decided = true;
            staged.verdict = _spec->verdict;
            return staged;
        }

        auto state = std::make_unique<State>();
        state->predictedVariants = _spec->predictedVariants;
        state->verdict = _spec->verdict;
        state->owner = this;
        staged.pending = PendingApplicability(std::move(state));
        return staged;
    }

    bool finishApplicability(PendingApplicability&& pending) const override
    {
        const auto released = pending.release();
        // The SDK's RTTI-free tag, as GenericPlanBuilder uses it.
        const auto* state = released != nullptr && released->owner == this
                                ? static_cast<const State*>(released.get())
                                : nullptr;

        _journal.recordFinish(id());
        if(_spec->awaitSibling)
        {
            _journal.awaitFinishOnAnotherThread();
        }
        // Long enough for the other tasks of the call to be in flight at the same time.
        std::this_thread::sleep_for(std::chrono::microseconds(300));

        if(_spec->failure == Failure::IN_FINISH)
        {
            throw HipdnnPluginException(_spec->failureStatus, _spec->failureMessage);
        }
        return state != nullptr && state->verdict;
    }

private:
    struct State : PendingApplicability::State
    {
        bool verdict = false;
    };

    std::shared_ptr<StagedSpec> _spec;
    Journal& _journal;
};

/// The outcome of one getApplicableEngineIds call: the ids, or what it threw.
struct Outcome
{
    std::vector<int64_t> ids;
    bool threw = false;
    bool threwPluginException = false;
    Thrown thrown{HIPDNN_PLUGIN_STATUS_SUCCESS, {}};
};

class TestApplicabilityManager : public ::testing::Test
{
protected:
    TestApplicabilityManager()
        : _graphBuffer(hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph())
        , _graph(_graphBuffer.GetBufferPointer(), _graphBuffer.GetSize())
        , _journal(std::this_thread::get_id())
    {
    }

    void addHandWritten(ApplicabilityManager& manager, int64_t engineId, bool verdict)
    {
        manager.addEngine(std::make_unique<FakeHandWrittenEngine>(engineId, verdict, _journal));
    }

    /// Registers a staged fake and returns its spec, for adjusting between calls.
    std::shared_ptr<StagedSpec> addStaged(ApplicabilityManager& manager, StagedSpec spec)
    {
        auto shared = std::make_shared<StagedSpec>(std::move(spec));
        manager.addEngine(std::make_unique<FakeStagedEngine>(shared, _journal));
        return shared;
    }

    static StagedSpec pending(int64_t engineId, bool verdict, std::size_t predicted)
    {
        StagedSpec spec;
        spec.id = engineId;
        spec.verdict = verdict;
        spec.predictedVariants = predicted;
        return spec;
    }

    static StagedSpec decided(int64_t engineId, bool verdict)
    {
        StagedSpec spec;
        spec.id = engineId;
        spec.decided = true;
        spec.verdict = verdict;
        return spec;
    }

    /// One call under the given knobs. An empty knob value means unset.
    Outcome run(ApplicabilityManager& manager,
                const std::string& maxThreads,
                const std::string& minOffload)
    {
        const ScopedEnvironmentVariableSetter threadsKnob(APPLICABILITY_MAX_THREADS_ENV_NAME);
        const ScopedEnvironmentVariableSetter offloadKnob(
            APPLICABILITY_MIN_OFFLOAD_VARIANTS_ENV_NAME);
        setOrUnset(APPLICABILITY_MAX_THREADS_ENV_NAME, maxThreads);
        setOrUnset(APPLICABILITY_MIN_OFFLOAD_VARIANTS_ENV_NAME, minOffload);

        Outcome outcome;
        try
        {
            outcome.ids = manager.getApplicableEngineIds(_handle, _graph);
        }
        catch(const HipdnnPluginException& e)
        {
            outcome.threw = true;
            outcome.threwPluginException = true;
            outcome.thrown = {e.getStatus(), e.what()};
        }
        catch(const std::exception& e)
        {
            outcome.threw = true;
            outcome.thrown = {HIPDNN_PLUGIN_STATUS_SUCCESS, e.what()};
        }
        return outcome;
    }

    /// The manager's own iteration order, filtered to @p wanted: the order every
    /// getApplicableEngineIds result must come back in.
    static std::vector<int64_t> inIterationOrder(const ApplicabilityManager& manager,
                                                 const std::vector<int64_t>& wanted)
    {
        std::vector<int64_t> ordered;
        for(const auto engineId : manager.getAllEngineIds())
        {
            if(std::find(wanted.begin(), wanted.end(), engineId) != wanted.end())
            {
                ordered.push_back(engineId);
            }
        }
        return ordered;
    }

    bool allFinishesOnCaller() const
    {
        return !_journal.anyFinishOffCaller();
    }

    flatbuffers::FlatBufferBuilder _graphBuffer;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    Handle _handle;
    Journal _journal;

private:
    static void setOrUnset(const char* name, const std::string& value)
    {
        if(value.empty())
        {
            hipdnn_data_sdk::utilities::unsetEnv(name);
        }
        else
        {
            hipdnn_data_sdk::utilities::setEnv(name, value.c_str());
        }
    }
};

constexpr int64_t HAND_YES = 0x1001;
constexpr int64_t HAND_NO = 0x1002;
constexpr int64_t DECIDED_YES = 0x2001;
constexpr int64_t DECIDED_NO = 0x2002;
constexpr int64_t PENDING_A = 0x3001;
constexpr int64_t PENDING_B = 0x3002;
constexpr int64_t PENDING_C = 0x3003;
constexpr int64_t PENDING_D = 0x3004;
constexpr int64_t PENDING_E = 0x3005;

} // namespace

TEST_F(TestApplicabilityManager, ThreadedResultsMatchSequentialAcrossEveryKnob)
{
    WorkerBudget budget{8};
    ApplicabilityPool pool{budget};
    ApplicabilityManager manager{pool};

    addHandWritten(manager, HAND_YES, true);
    addHandWritten(manager, HAND_NO, false);
    addStaged(manager, decided(DECIDED_YES, true));
    addStaged(manager, decided(DECIDED_NO, false));
    // Σ − max = 760: offloaded at T = 0 and 200, inline at 10^9.
    const auto largest = addStaged(manager, pending(PENDING_A, true, 900));
    addStaged(manager, pending(PENDING_B, false, 400));
    addStaged(manager, pending(PENDING_C, true, 300));
    addStaged(manager, pending(PENDING_D, true, 50));
    addStaged(manager, pending(PENDING_E, false, 10));
    ASSERT_EQ(manager.stagedEngineCount(), 7u);

    const auto expected
        = inIterationOrder(manager, {HAND_YES, DECIDED_YES, PENDING_A, PENDING_C, PENDING_D});

    const auto sequential = run(manager, "1", "");
    ASSERT_FALSE(sequential.threw) << sequential.thrown.message;
    ASSERT_EQ(sequential.ids, expected);

    for(const std::string maxThreads : {"1", "2", "4", "auto"})
    {
        for(const std::string minOffload : {"0", "200", "1000000000"})
        {
            SCOPED_TRACE(std::string("MAX_THREADS=")
                             .append(maxThreads)
                             .append(" MIN_OFFLOAD_VARIANTS=")
                             .append(minOffload));
            // Auto never exceeds the CPUs, so on a one-CPU host it means no worker at all.
            const bool gateOpens = maxThreads != "1" && minOffload != "1000000000"
                                   && (maxThreads != "auto" || affinityCpuCount() > 1);
            // Where the gate opens, hold the caller's engine until a worker has finished one,
            // so the threaded path is the one compared rather than a caller that drained the
            // whole batch before any worker woke.
            largest->awaitSibling = gateOpens;
            _journal.clear();

            const auto outcome = run(manager, maxThreads, minOffload);

            ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
            EXPECT_EQ(outcome.ids, sequential.ids);
            EXPECT_EQ(_journal.anyFinishOffCaller(), gateOpens);
            for(const auto& [engineId, thread] : _journal.handWritten())
            {
                EXPECT_EQ(thread, _journal.caller()) << "hand-written engine " << engineId;
            }
        }
    }
    EXPECT_EQ(budget.available(), 8u);
}

TEST_F(TestApplicabilityManager, StaysInlineWithFewerThanTwoPending)
{
    WorkerBudget budget{8};
    ApplicabilityPool pool{budget};
    ApplicabilityManager manager{pool};

    addStaged(manager, decided(DECIDED_YES, true));
    addStaged(manager, decided(DECIDED_NO, false));
    addStaged(manager, pending(PENDING_A, true, 1000000));

    const auto outcome = run(manager, "4", "0");

    ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
    EXPECT_EQ(outcome.ids, inIterationOrder(manager, {DECIDED_YES, PENDING_A}));
    EXPECT_TRUE(allFinishesOnCaller());
    EXPECT_EQ(pool.threadCount(), 0u);
    // A decided engine has nothing left to finish.
    EXPECT_EQ(_journal.finishCount(DECIDED_YES), 0u);
    EXPECT_EQ(_journal.finishCount(DECIDED_NO), 0u);
    EXPECT_EQ(_journal.finishCount(PENDING_A), 1u);
}

TEST_F(TestApplicabilityManager, StaysInlineWhenOffloadedVariantsAreBelowTheThreshold)
{
    WorkerBudget budget{8};
    ApplicabilityPool pool{budget};
    ApplicabilityManager manager{pool};

    // Σ − max = 200.
    addStaged(manager, pending(PENDING_A, true, 100));
    addStaged(manager, pending(PENDING_B, true, 100));
    addStaged(manager, pending(PENDING_C, false, 100));

    const auto outcome = run(manager, "4", "201");

    ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
    EXPECT_EQ(outcome.ids, inIterationOrder(manager, {PENDING_A, PENDING_B}));
    EXPECT_TRUE(allFinishesOnCaller());
    EXPECT_EQ(pool.threadCount(), 0u);
}

TEST_F(TestApplicabilityManager, OffloadsWhenOffloadedVariantsReachTheThreshold)
{
    WorkerBudget budget{8};
    ApplicabilityPool pool{budget};
    ApplicabilityManager manager{pool};

    // Σ − max = 200, exactly T: the gate is inclusive.
    for(const auto engineId : {PENDING_A, PENDING_B, PENDING_C})
    {
        auto spec = pending(engineId, engineId != PENDING_C, 100);
        spec.awaitSibling = true;
        addStaged(manager, std::move(spec));
    }

    const auto outcome = run(manager, "4", "200");

    ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
    EXPECT_EQ(outcome.ids, inIterationOrder(manager, {PENDING_A, PENDING_B}));
    EXPECT_TRUE(_journal.anyFinishOffCaller());
}

TEST_F(TestApplicabilityManager, StaysInlineWhenMaxThreadsIsOne)
{
    WorkerBudget budget{8};
    ApplicabilityPool pool{budget};
    ApplicabilityManager manager{pool};

    addStaged(manager, pending(PENDING_A, true, 1000000));
    addStaged(manager, pending(PENDING_B, true, 1000000));
    addStaged(manager, pending(PENDING_C, false, 1000000));

    for(const std::string maxThreads : {"1", ""})
    {
        SCOPED_TRACE("MAX_THREADS='" + maxThreads + "'");
        _journal.clear();

        const auto outcome = run(manager, maxThreads, "0");

        ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
        EXPECT_EQ(outcome.ids, inIterationOrder(manager, {PENDING_A, PENDING_B}));
        EXPECT_TRUE(allFinishesOnCaller());
        EXPECT_EQ(pool.threadCount(), 0u);
    }
}

TEST_F(TestApplicabilityManager, RunsTheLargestPendingEngineOnTheCaller)
{
    WorkerBudget budget{8};
    ApplicabilityPool pool{budget};
    ApplicabilityManager manager{pool};

    // The largest is added in the middle so neither first nor last position is mistaken
    // for "largest".
    addStaged(manager, pending(PENDING_A, true, 300));
    auto largest = pending(PENDING_B, true, 5000);
    largest.awaitSibling = true;
    addStaged(manager, std::move(largest));
    addStaged(manager, pending(PENDING_C, true, 300));
    addHandWritten(manager, HAND_YES, true);
    addStaged(manager, decided(DECIDED_NO, false));

    const auto outcome = run(manager, "4", "0");

    ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
    EXPECT_EQ(outcome.ids, inIterationOrder(manager, {PENDING_A, PENDING_B, PENDING_C, HAND_YES}));
    EXPECT_EQ(_journal.finishThread(PENDING_B), _journal.caller());
    EXPECT_TRUE(_journal.anyFinishOffCaller());
    EXPECT_GE(pool.threadCount(), 1u);
    for(const auto engineId : {PENDING_A, PENDING_B, PENDING_C})
    {
        EXPECT_EQ(_journal.finishCount(engineId), 1u) << "engine " << engineId;
    }
    EXPECT_EQ(_journal.finishCount(DECIDED_NO), 0u);
    for(const auto& [engineId, thread] : _journal.handWritten())
    {
        EXPECT_EQ(thread, _journal.caller()) << "hand-written engine " << engineId;
    }
    // Workers return their slots when the call drains.
    EXPECT_EQ(budget.available(), 8u);
}

TEST_F(TestApplicabilityManager, RethrowsTheLowestIndexFailureThreadedOrNot)
{
    // Two throwers in finish, and each in turn the largest, so "lowest index" cannot be
    // passed by "whichever ran on the caller" or "whichever finished first".
    for(const bool lowerIsLargest : {true, false})
    {
        SCOPED_TRACE(lowerIsLargest ? "lower index is the largest" : "higher index is the largest");

        WorkerBudget budget{8};
        ApplicabilityPool pool{budget};
        ApplicabilityManager manager{pool};
        _journal.clear();

        addHandWritten(manager, HAND_YES, true);
        addStaged(manager, pending(PENDING_C, true, 700));
        auto specA = addStaged(manager, pending(PENDING_A, true, 0));
        auto specB = addStaged(manager, pending(PENDING_B, true, 0));

        // The iteration order is the manager's to choose; read it back and assign the
        // failures by position.
        const auto order = inIterationOrder(manager, {PENDING_A, PENDING_B});
        ASSERT_EQ(order.size(), 2u);
        const auto& lower = order[0] == PENDING_A ? specA : specB;
        const auto& higher = order[0] == PENDING_A ? specB : specA;

        lower->failure = Failure::IN_FINISH;
        lower->failureStatus = HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
        lower->failureMessage = "lower index failed";
        higher->failure = Failure::IN_FINISH;
        higher->failureStatus = HIPDNN_PLUGIN_STATUS_ALLOC_FAILED;
        higher->failureMessage = "higher index failed";
        const auto& largest = lowerIsLargest ? lower : higher;
        const auto& smaller = lowerIsLargest ? higher : lower;
        largest->predictedVariants = 5000;
        smaller->predictedVariants = 400;

        const auto sequential = run(manager, "1", "");
        largest->awaitSibling = true;
        const auto threaded = run(manager, "4", "0");

        for(const auto* outcome : {&sequential, &threaded})
        {
            ASSERT_TRUE(outcome->threw);
            EXPECT_TRUE(outcome->threwPluginException);
            EXPECT_EQ(outcome->thrown.message, "lower index failed");
            EXPECT_EQ(outcome->thrown.status, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        }
        EXPECT_TRUE(_journal.anyFinishOffCaller());
        // A failed call still returns its workers' slots.
        EXPECT_EQ(budget.available(), 8u);
    }
}

TEST_F(TestApplicabilityManager, LowerIndexFinishFailureBeatsHigherIndexBeginFailure)
{
    WorkerBudget budget{8};
    ApplicabilityPool pool{budget};
    ApplicabilityManager manager{pool};

    addStaged(manager, pending(PENDING_C, true, 700));
    addStaged(manager, pending(PENDING_D, true, 700));
    auto specA = addStaged(manager, pending(PENDING_A, true, 600));
    auto specB = addStaged(manager, pending(PENDING_B, true, 600));

    const auto order = inIterationOrder(manager, {PENDING_A, PENDING_B});
    ASSERT_EQ(order.size(), 2u);
    const auto& lower = order[0] == PENDING_A ? specA : specB;
    const auto& higher = order[0] == PENDING_A ? specB : specA;

    // The lower index fails late, in its variant stage; the higher index fails early, in
    // begin on the caller. Sequentially the lower one is reached first, so it must win
    // even though the higher one's failure is raised before any finish runs.
    lower->failure = Failure::IN_FINISH;
    lower->failureStatus = HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    lower->failureMessage = "variant stage failed";
    higher->failure = Failure::IN_BEGIN;
    higher->failureStatus = HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    higher->failureMessage = "graph stage failed";

    const auto sequential = run(manager, "1", "");
    const auto threaded = run(manager, "4", "0");

    for(const auto* outcome : {&sequential, &threaded})
    {
        ASSERT_TRUE(outcome->threw);
        EXPECT_TRUE(outcome->threwPluginException);
        EXPECT_EQ(outcome->thrown.message, "variant stage failed");
        EXPECT_EQ(outcome->thrown.status, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    }
    EXPECT_EQ(budget.available(), 8u);
}

TEST_F(TestApplicabilityManager, ExhaustedBudgetRunsInlineWithTheSameResult)
{
    WorkerBudget budget{4};
    ApplicabilityPool pool{budget};
    ApplicabilityManager manager{pool};

    addHandWritten(manager, HAND_NO, false);
    addStaged(manager, pending(PENDING_A, true, 5000));
    addStaged(manager, pending(PENDING_B, false, 5000));
    addStaged(manager, pending(PENDING_C, true, 5000));
    addStaged(manager, pending(PENDING_D, true, 5000));
    const auto expected = inIterationOrder(manager, {PENDING_A, PENDING_C, PENDING_D});

    // Another call holds every worker slot.
    const auto held = budget.tryAcquire(4);
    ASSERT_EQ(held, 4u);

    const auto outcome = run(manager, "4", "0");

    ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
    EXPECT_EQ(outcome.ids, expected);
    EXPECT_TRUE(allFinishesOnCaller());
    EXPECT_EQ(pool.threadCount(), 0u);
    EXPECT_EQ(budget.available(), 0u);

    budget.release(held);
}

TEST_F(TestApplicabilityManager, ZeroCapacityBudgetRunsInlineWithTheSameResult)
{
    WorkerBudget budget{0};
    ApplicabilityPool pool{budget};
    ApplicabilityManager manager{pool};

    addStaged(manager, pending(PENDING_A, true, 5000));
    addStaged(manager, pending(PENDING_B, false, 5000));
    addStaged(manager, pending(PENDING_C, true, 5000));

    const auto outcome = run(manager, "auto", "0");

    ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
    EXPECT_EQ(outcome.ids, inIterationOrder(manager, {PENDING_A, PENDING_C}));
    EXPECT_TRUE(allFinishesOnCaller());
    EXPECT_EQ(pool.threadCount(), 0u);
}

TEST_F(TestApplicabilityManager, DestroyingThePoolOwnerJoinsEveryWorker)
{
    const auto baseline = ApplicabilityPool::liveThreads();
    {
        WorkerBudget budget{8};
        ApplicabilityPool pool{budget};
        ApplicabilityManager manager{pool};
        for(const auto engineId : {PENDING_A, PENDING_B, PENDING_C, PENDING_D})
        {
            auto spec = pending(engineId, true, 5000);
            spec.awaitSibling = true;
            addStaged(manager, std::move(spec));
        }

        const auto outcome = run(manager, "4", "0");

        ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
        ASSERT_TRUE(_journal.anyFinishOffCaller());
        ASSERT_GE(pool.threadCount(), 1u);
        EXPECT_GT(ApplicabilityPool::liveThreads(), baseline);
    }
    EXPECT_EQ(ApplicabilityPool::liveThreads(), baseline);
}

TEST_F(TestApplicabilityManager, DestroyingTheContainerJoinsEveryWorker)
{
    // The container also holds the build's real engines, whose applicability checks may
    // resolve the device.
    SKIP_IF_NO_DEVICES();

    const auto baseline = ApplicabilityPool::liveThreads();
    {
        Container container;
        auto& manager = container.getEngineManager();
        for(const auto engineId : {PENDING_A, PENDING_B, PENDING_C, PENDING_D})
        {
            auto spec = pending(engineId, true, 1000000);
            spec.awaitSibling = true;
            addStaged(manager, std::move(spec));
        }

        const auto outcome = run(manager, "4", "0");

        ASSERT_FALSE(outcome.threw) << outcome.thrown.message;
        for(const auto engineId : {PENDING_A, PENDING_B, PENDING_C, PENDING_D})
        {
            EXPECT_NE(std::find(outcome.ids.begin(), outcome.ids.end(), engineId),
                      outcome.ids.end())
                << "engine " << engineId;
        }
        ASSERT_TRUE(_journal.anyFinishOffCaller());
        EXPECT_GT(ApplicabilityPool::liveThreads(), baseline);
    }
    EXPECT_EQ(ApplicabilityPool::liveThreads(), baseline);
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
