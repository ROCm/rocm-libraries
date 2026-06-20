// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ClientRunScheduler.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    struct NullProblemInputs final : ProblemInputs
    {
    };

    struct FakeHardware final : Hardware
    {
        size_t id() const override
        {
            return 0;
        }

        std::string description() const override
        {
            return "fake-hardware";
        }

        std::string archName() const override
        {
            return "gfx000";
        }
    };

    KernelInvocation makeKernel(std::string name)
    {
        KernelInvocation kernel;
        kernel.kernelName = std::move(name);
        return kernel;
    }

    std::vector<std::shared_ptr<ContractionProblem>> makeProblems(size_t count)
    {
        std::vector<std::shared_ptr<ContractionProblem>> problems;
        problems.reserve(count);
        for(size_t i = 0; i < count; ++i)
        {
            problems.push_back(std::make_shared<ContractionProblemGemm>());
        }
        return problems;
    }

    std::vector<std::string>
        extractEventsWithPrefix(std::vector<std::string> const& events, std::string const& prefix)
    {
        std::vector<std::string> rv;
        for(auto const& event : events)
        {
            if(event.rfind(prefix, 0) == 0)
                rv.push_back(event.substr(prefix.size()));
        }
        return rv;
    }

    size_t indexOfEvent(std::vector<std::string> const& events, std::string const& value)
    {
        auto it = std::find(events.begin(), events.end(), value);
        if(it == events.end())
            return events.size();
        return static_cast<size_t>(std::distance(events.begin(), it));
    }

    std::vector<size_t> indicesOfEvent(std::vector<std::string> const& events,
                                       std::string const&              value)
    {
        std::vector<size_t> indices;
        for(size_t i = 0; i < events.size(); ++i)
        {
            if(events[i] == value)
                indices.push_back(i);
        }
        return indices;
    }

    size_t countExactEvent(std::vector<std::string> const& events, std::string const& value)
    {
        return static_cast<size_t>(std::count(events.begin(), events.end(), value));
    }

    struct CountQueryRecord
    {
        size_t value           = 0;
        bool   seenPreSolution = false;
        bool   seenPostWarmup  = false;
    };

    bool hasCountQuery(std::vector<CountQueryRecord> const& records,
                       size_t                               value,
                       bool                                 seenPreSolution,
                       bool                                 seenPostWarmup)
    {
        return std::any_of(records.begin(),
                           records.end(),
                           [&](CountQueryRecord const& record) {
                               return record.value == value
                                      && record.seenPreSolution == seenPreSolution
                                      && record.seenPostWarmup == seenPostWarmup;
                           });
    }

    class RecordingRunListener final : public RunListener
    {
    public:
        explicit RecordingRunListener(std::vector<std::string>& events)
            : m_events(events)
        {
        }

        bool needMoreBenchmarkRuns() const override
        {
            if(m_benchmarkRunsRemaining <= 0)
                return false;
            --m_benchmarkRunsRemaining;
            return true;
        }

        void preBenchmarkRun() override
        {
            m_events.push_back("preBenchmarkRun");
        }

        void postBenchmarkRun() override
        {
            m_events.push_back("postBenchmarkRun");
        }

        void preProblem(ContractionProblem* const problem) override
        {
            m_events.push_back("preProblem");
            if(setCurrentProblemFn)
                setCurrentProblemFn(problem);
            if(resetSolutionSourceOnPreProblem && resetSolutionSourceFn)
                resetSolutionSourceFn();
        }

        void postProblem() override
        {
            m_events.push_back("postProblem");
        }

        void preSolution(ContractionSolution* const) override
        {
            m_events.push_back("preSolution");
            seenPreSolution = true;
            if(resetSolutionRunsOnPreSolution)
                m_solutionRunsRemaining = static_cast<int>(solutionRunsPerSolution);
            if(changeWarmupRunsAfterPreSolution)
                m_warmupRuns = warmupRunsAfterPreSolution;
        }

        void postSolution() override
        {
            m_events.push_back("postSolution");
        }

        bool needMoreRunsInSolution() const override
        {
            if(m_solutionRunsRemaining <= 0)
                return false;
            --m_solutionRunsRemaining;
            return true;
        }

        size_t numWarmupRuns() override
        {
            if(recordCountQueries)
            {
                warmupCountQueries.push_back(
                    CountQueryRecord{m_warmupRuns, seenPreSolution, seenPostWarmup});
            }
            return m_warmupRuns;
        }

        void setNumWarmupRuns(size_t count) override
        {
            m_warmupRuns = count;
        }

        void preWarmup() override
        {
            m_events.push_back("preWarmup");
        }

        void postWarmup(TimingEvents const&, TimingEvents const&, hipStream_t const&) override
        {
            m_events.push_back("postWarmup");
            if(changeSyncsAndEnqueuesAfterPostWarmup)
            {
                m_syncs    = syncsAfterPostWarmup;
                m_enqueues = enqueuesAfterPostWarmup;
            }
            seenPostWarmup = true;
        }

        void validateWarmups(std::shared_ptr<ProblemInputs>,
                             TimingEvents const&,
                             TimingEvents const&) override
        {
            m_events.push_back("validateWarmups");
        }

        size_t numSyncs() override
        {
            if(recordCountQueries)
            {
                syncCountQueries.push_back(
                    CountQueryRecord{m_syncs, seenPreSolution, seenPostWarmup});
            }
            return m_syncs;
        }

        void setNumSyncs(size_t count) override
        {
            m_syncs = count;
        }

        void preSyncs() override
        {
            m_events.push_back("preSyncs");
        }

        void postSyncs() override
        {
            m_events.push_back("postSyncs");
        }

        size_t numEnqueuesPerSync() override
        {
            if(recordCountQueries)
            {
                enqueueCountQueries.push_back(
                    CountQueryRecord{m_enqueues, seenPreSolution, seenPostWarmup});
            }
            return m_enqueues;
        }

        void setNumEnqueuesPerSync(size_t count) override
        {
            m_enqueues = count;
        }

        void preEnqueues(hipStream_t const&) override
        {
            m_events.push_back("preEnqueues");
        }

        void postEnqueues(TimingEvents const&, TimingEvents const&, hipStream_t const&) override
        {
            m_events.push_back("postEnqueues");
        }

        void validateEnqueues(std::shared_ptr<ProblemInputs>,
                              TimingEvents const&,
                              TimingEvents const&) override
        {
            m_events.push_back("validateEnqueues");
        }

        void finalizeReport() override {}

        int error() const override
        {
            return m_error;
        }

        int    m_error                 = 0;
        size_t m_warmupRuns            = 0;
        size_t m_syncs                 = 0;
        size_t m_enqueues              = 0;
        bool   recordCountQueries      = false;
        bool   seenPreSolution         = false;
        bool   seenPostWarmup          = false;
        std::vector<CountQueryRecord> warmupCountQueries;
        std::vector<CountQueryRecord> syncCountQueries;
        std::vector<CountQueryRecord> enqueueCountQueries;
        bool   changeWarmupRunsAfterPreSolution      = false;
        size_t warmupRunsAfterPreSolution            = 0;
        bool   changeSyncsAndEnqueuesAfterPostWarmup = false;
        size_t syncsAfterPostWarmup                  = 0;
        size_t enqueuesAfterPostWarmup               = 0;
        bool   resetSolutionSourceOnPreProblem       = false;
        bool   resetSolutionRunsOnPreSolution        = false;
        size_t solutionRunsPerSolution               = 1;
        mutable int m_benchmarkRunsRemaining = 1;
        mutable int m_solutionRunsRemaining  = 1;
        std::function<void(ContractionProblem const*)> setCurrentProblemFn;
        std::function<void()>                         resetSolutionSourceFn;

    private:
        std::vector<std::string>& m_events;
    };

    class RecordingReporter final : public RunReporter
    {
    public:
        explicit RecordingReporter(std::vector<std::string>& events)
            : m_events(events)
        {
        }

        void reportProblemIndex(int idx) override
        {
            m_events.push_back("reportProblemIndex:" + std::to_string(idx));
            reportedProblemIndices.push_back(idx);
        }

        void reportProblemProgress(std::string const& text) override
        {
            m_events.push_back("reportProblemProgress:" + text);
            reportedProblemProgress.push_back(text);
        }

        void reportInvalid() override
        {
            m_events.push_back("reportInvalid");
            ++invalidCount;
        }

        void logError(std::string const& message) override
        {
            m_events.push_back("logError:" + message);
            errorMessages.push_back(message);
        }

        int                        invalidCount = 0;
        std::vector<std::string>   errorMessages;
        std::vector<int>           reportedProblemIndices;
        std::vector<std::string>   reportedProblemProgress;

    private:
        std::vector<std::string>& m_events;
    };

    class RecordingDataCoordinator final : public RunDataCoordinator
    {
    public:
        explicit RecordingDataCoordinator(std::vector<std::string>& events)
            : m_events(events)
        {
        }

        void setCurrentProblem(ContractionProblem const* problem)
        {
            currentProblemIndex = -1;
            if(!recordProblemContext || problems == nullptr || problem == nullptr)
                return;

            for(size_t i = 0; i < problems->size(); ++i)
            {
                if(problems->at(i).get() == problem)
                {
                    currentProblemIndex = static_cast<int>(i);
                    break;
                }
            }
        }

        void resetPreparedSlotsForProblem() override
        {
            m_events.push_back("resetPreparedSlotsForProblem");
            recordProblemEvent("reset");
        }

        std::shared_ptr<ProblemInputs> prepareGPUInputs(ContractionProblem const*) override
        {
            m_events.push_back("prepareGPUInputs");
            recordProblemEvent("prepare");
            return std::make_shared<NullProblemInputs>();
        }

        std::vector<std::shared_ptr<ProblemInputs>> prepareRotatingGPUOutput(
            int32_t                        maxRotatingBufferNum,
            ContractionProblem const*,
            std::shared_ptr<ProblemInputs> inputs,
            hipStream_t) override
        {
            m_events.push_back("prepareRotatingGPUOutput:" + std::to_string(maxRotatingBufferNum));
            recordProblemRotateEvent(maxRotatingBufferNum);
            std::vector<std::shared_ptr<ProblemInputs>> rv(std::max<size_t>(1, rotatingSlots),
                                                           std::move(inputs));
            return rv;
        }

        void waitForPreparedSlot(hipStream_t) override
        {
            m_events.push_back("waitForPreparedSlot");
        }

        void primeNextInputSlot(ContractionProblem const*) override
        {
            m_events.push_back("primeNextInputSlot");
            recordProblemEvent("prime");
        }

        size_t rotatingSlots = 1;
        bool   recordProblemContext = false;
        std::vector<std::shared_ptr<ContractionProblem>> const* problems = nullptr;
        std::vector<std::string> problemEvents;
        int currentProblemIndex = -1;

    private:
        void recordProblemEvent(std::string const& prefix)
        {
            if(currentProblemIndex >= 0)
                problemEvents.push_back(prefix + ":p" + std::to_string(currentProblemIndex));
        }

        void recordProblemRotateEvent(int32_t maxRotatingBufferNum)
        {
            if(currentProblemIndex >= 0)
            {
                problemEvents.push_back("rotate:p" + std::to_string(currentProblemIndex) + ":"
                                        + std::to_string(maxRotatingBufferNum));
            }
        }

        std::vector<std::string>& m_events;
    };

    class RecordingSolution final : public ContractionSolution
    {
    public:
        explicit RecordingSolution(std::vector<std::string>& events)
            : m_events(events)
        {
        }

        std::string name() const override
        {
            return "recording-solution";
        }

        std::string description() const override
        {
            return "recording-solution";
        }

        bool isFallbackForHW(Hardware const&) const override
        {
            return false;
        }

        std::vector<KernelInvocation> solve(ContractionProblem const&,
                                            ProblemInputs const&,
                                            Hardware const&,
                                            void*,
                                            size_t,
                                            hipStream_t) const override
        {
            m_events.push_back("solve");
            if(throwOnSolve)
                throw std::runtime_error("solve failed");
            return nextKernels();
        }

        std::vector<KernelInvocation> solveTensileGPU(ContractionProblem const&,
                                                      ProblemInputs const&,
                                                      Hardware const&,
                                                      void** dUA,
                                                      void** dUAHost,
                                                      void*,
                                                      size_t,
                                                      hipStream_t) const override
        {
            m_events.push_back("solveTensileGPU");
            if(throwOnSolve)
                throw std::runtime_error("solve failed");
            auto const allocationId = assignedUserArgs.size();
            dUASentinels.push_back(
                std::make_unique<int>(static_cast<int>(allocationId * 2 + 1)));
            dUAHostSentinels.push_back(
                std::make_unique<int>(static_cast<int>(allocationId * 2 + 2)));
            void* dUAValue     = dUASentinels.back().get();
            void* dUAHostValue = dUAHostSentinels.back().get();
            if(dUA)
                *dUA = dUAValue;
            if(dUAHost)
                *dUAHost = dUAHostValue;
            assignedUserArgs.emplace_back(dUAValue, dUAHostValue);
            return nextKernels();
        }

        void relaseDeviceUserArgs(void* dUA, void* dUAHost) override
        {
            m_events.push_back("releaseDeviceUserArgs");
            releasedUserArgs.emplace_back(dUA, dUAHost);
        }

        std::vector<std::vector<KernelInvocation>> kernelsPerSolveCall;
        bool                                        throwOnSolve = false;
        mutable std::vector<std::pair<void*, void*>> assignedUserArgs;
        mutable std::vector<std::pair<void*, void*>> releasedUserArgs;

    private:
        std::vector<KernelInvocation> nextKernels() const
        {
            if(kernelsPerSolveCall.empty())
                return {};

            auto idx = std::min(solveCallIndex++, kernelsPerSolveCall.size() - 1);
            return kernelsPerSolveCall[idx];
        }

        std::vector<std::string>&               m_events;
        mutable size_t                          solveCallIndex = 0;
        mutable std::vector<std::unique_ptr<int>> dUASentinels;
        mutable std::vector<std::unique_ptr<int>> dUAHostSentinels;
    };

    class RecordingSolutionSource final : public RunSolutionSource
    {
    public:
        bool moreSolutionsInProblem() const override
        {
            return nextSolution < solutions.size();
        }

        std::shared_ptr<ContractionSolution> getSolution() override
        {
            return solutions.at(nextSolution++);
        }

        bool runCurrentSolution() override
        {
            return runCurrentSolutionResult;
        }

        void resetForProblem()
        {
            nextSolution = 0;
        }

        std::vector<std::shared_ptr<ContractionSolution>> solutions;
        bool                                              runCurrentSolutionResult = true;

    private:
        mutable size_t nextSolution = 0;
    };

    class RecordingKernelLauncher final : public RunKernelLauncher
    {
    public:
        explicit RecordingKernelLauncher(std::vector<std::string>& events)
            : m_events(events)
        {
        }

        int numRotationModules() override
        {
            return rotationModules;
        }

        void selectRotationCopy(int idx) override
        {
            rotationSelections.push_back(idx);
            m_events.push_back("selectRotationCopy:" + std::to_string(idx));
        }

        hipError_t loadCodeObjectFileExtraCopies(std::string const& path,
                                                 int               extraCopies) override
        {
            extraCopyLoads.emplace_back(path, extraCopies);
            m_events.push_back("loadCodeObjectFileExtraCopies:" + path + ":"
                               + std::to_string(extraCopies));
            rotationModules = std::max(rotationModules, extraCopies + 1);
            return hipSuccess;
        }

        hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                 hipStream_t,
                                 std::vector<hipEvent_t> const&,
                                 std::vector<hipEvent_t> const&) override
        {
            m_events.push_back("launchWarmup:" + kernelLabel(kernels));
            return takeLaunchResult(warmupLaunchResult, warmupLaunchResultConsumeOnce);
        }

        hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                 hipStream_t,
                                 hipEvent_t,
                                 hipEvent_t) override
        {
            m_events.push_back("launchBenchmark:" + kernelLabel(kernels));
            return takeLaunchResult(benchmarkLaunchResult, benchmarkLaunchResultConsumeOnce);
        }

        int numRotationModulesValue() const
        {
            return rotationModules;
        }

        int                                            rotationModules = 1;
        std::vector<int>                               rotationSelections;
        std::vector<std::pair<std::string, int>>       extraCopyLoads;
        hipError_t                                     warmupLaunchResult = hipSuccess;
        hipError_t                                     benchmarkLaunchResult = hipSuccess;
        bool                                           warmupLaunchResultConsumeOnce = false;
        bool                                           benchmarkLaunchResultConsumeOnce = false;

    private:
        static hipError_t takeLaunchResult(hipError_t& result, bool consumeOnce)
        {
            auto rv = result;
            if(consumeOnce && rv != hipSuccess)
                result = hipSuccess;
            return rv;
        }

        static std::string kernelLabel(std::vector<KernelInvocation> const& kernels)
        {
            if(kernels.empty())
                return "empty";
            if(kernels.front().kernelName.empty())
                return "unnamed";
            return kernels.front().kernelName;
        }

        std::vector<std::string>& m_events;
    };

    struct SchedulerHarness
    {
        SchedulerHarness()
            : solution(std::make_shared<RecordingSolution>(events))
        {
            problems = makeProblems(1);
            config.lastProblemIdx = static_cast<int>(problems.size() - 1);
            config.icacheFlushArgs = {false};
            solutionSource.solutions.push_back(solution);
            data.problems = &problems;
            listeners.setCurrentProblemFn = [this](ContractionProblem const* problem) {
                data.setCurrentProblem(problem);
            };
            listeners.resetSolutionSourceFn = [this] { solutionSource.resetForProblem(); };
            callbacks.flushGridSizeFn = [this] {
                ++flushGridSizeCalls;
                return uint32_t{17};
            };
            callbacks.flushIcacheFn = [this](uint32_t flushGridSize, hipStream_t) {
                events.push_back("flushIcache:" + std::to_string(flushGridSize));
            };
            callbacks.deviceSynchronizeFn = [this] { events.push_back("deviceSynchronize"); };
            callbacks.setIcacheFlushTimeUsFn = [this](float timeUs) {
                flushTimeUs.push_back(timeUs);
            };
        }

        ClientRunScheduler makeScheduler()
        {
            return ClientRunScheduler(config,
                                      ClientRunSchedulerDependencies{
                                          &problems,
                                          &listeners,
                                          &reporter,
                                          &data,
                                          &solutionSource,
                                          &launcher,
                                          &hardware,
                                          nullptr,
                                          callbacks});
        }

        void setProblems(std::vector<std::shared_ptr<ContractionProblem>> newProblems)
        {
            problems = std::move(newProblems);
            config.lastProblemIdx = problems.empty() ? -1
                                                     : static_cast<int>(problems.size() - 1);
            data.problems = &problems;
            data.setCurrentProblem(nullptr);
        }

        std::vector<std::string>      events;
        RecordingRunListener          listeners{events};
        RecordingReporter             reporter{events};
        RecordingDataCoordinator      data{events};
        RecordingKernelLauncher       launcher{events};
        FakeHardware                  hardware{};
        std::vector<std::shared_ptr<ContractionProblem>> problems;
        std::shared_ptr<RecordingSolution> solution;
        RecordingSolutionSource       solutionSource;
        ClientRunSchedulerConfig      config;
        ClientRunSchedulerCallbacks   callbacks;
        std::vector<float>            flushTimeUs;
        int                           flushGridSizeCalls = 0;
        void*                         dUA     = nullptr;
        void*                         dUAHost = nullptr;
    };

    class ClientRunSchedulerTest : public ::testing::Test
    {
    protected:
        SchedulerHarness harness;
    };
} // namespace

TEST_F(ClientRunSchedulerTest, NoBenchmarkValidationRunExecutesWarmupAndSubmitsRingResets)
{
    harness.listeners.m_warmupRuns = 1;
    harness.listeners.m_syncs      = 0;
    harness.listeners.m_enqueues   = 0;
    harness.listeners.m_benchmarkRunsRemaining = 1;
    harness.listeners.m_solutionRunsRemaining   = 1;
    harness.config.runKernels = true;
    harness.config.gpuTimer   = false;
    harness.solution->kernelsPerSolveCall = {{}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);

    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchBenchmark:").empty());
    EXPECT_EQ(extractEventsWithPrefix(harness.events, "launchWarmup:"),
              (std::vector<std::string>{"empty"}));
    EXPECT_EQ(extractEventsWithPrefix(harness.events, "primeNextInputSlot"),
              (std::vector<std::string>{"", ""}));

    auto solveIdx          = indexOfEvent(harness.events, "solve");
    auto waitForPreparedSlotIdx = indexOfEvent(harness.events, "waitForPreparedSlot");
    auto warmupLaunchIdx   = indexOfEvent(harness.events, "launchWarmup:empty");
    auto validateWarmupsIdx = indexOfEvent(harness.events, "validateWarmups");
    auto firstResetIdx     = indexOfEvent(harness.events, "primeNextInputSlot");
    auto postSolutionIdx   = indexOfEvent(harness.events, "postSolution");

    ASSERT_NE(solveIdx, harness.events.size());
    ASSERT_NE(waitForPreparedSlotIdx, harness.events.size());
    ASSERT_NE(warmupLaunchIdx, harness.events.size());
    ASSERT_NE(validateWarmupsIdx, harness.events.size());
    ASSERT_NE(firstResetIdx, harness.events.size());
    ASSERT_NE(postSolutionIdx, harness.events.size());

    auto secondReset = std::find(harness.events.begin() + firstResetIdx + 1,
                                 harness.events.end(),
                                 "primeNextInputSlot");
    ASSERT_NE(secondReset, harness.events.end());

    EXPECT_LT(solveIdx, waitForPreparedSlotIdx);
    EXPECT_LT(waitForPreparedSlotIdx, warmupLaunchIdx);
    EXPECT_LT(warmupLaunchIdx, validateWarmupsIdx);
    EXPECT_LT(validateWarmupsIdx, firstResetIdx);
    EXPECT_LT(static_cast<size_t>(std::distance(harness.events.begin(), secondReset)),
              postSolutionIdx);

    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "preBenchmarkRun"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "reportProblemIndex:0"),
              harness.events.end());
    EXPECT_NE(
        std::find(harness.events.begin(), harness.events.end(), "reportProblemProgress:0/0"),
        harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "preProblem"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "resetPreparedSlotsForProblem"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "prepareGPUInputs"),
              harness.events.end());
    EXPECT_NE(
        std::find(harness.events.begin(), harness.events.end(), "prepareRotatingGPUOutput:1"),
        harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "deviceSynchronize"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "preSolution"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "preWarmup"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postWarmup"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "preSyncs"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postSyncs"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postProblem"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postBenchmarkRun"),
              harness.events.end());
    EXPECT_TRUE(harness.launcher.rotationSelections.empty());
}

TEST_F(ClientRunSchedulerTest, WaitForPreparedSlotPrecedesWarmupAndBenchmarkLaunches)
{
    harness.listeners.m_warmupRuns = 2;
    harness.listeners.m_syncs      = 1;
    harness.listeners.m_enqueues   = 2;
    harness.config.runKernels     = true;
    harness.config.gpuTimer       = false;
    harness.data.rotatingSlots     = 2;
    harness.launcher.rotationModules = 2;
    harness.solution->kernelsPerSolveCall = {{}, {}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);

    EXPECT_EQ(extractEventsWithPrefix(harness.events, "launchWarmup:"),
              (std::vector<std::string>{"empty", "empty"}));
    EXPECT_EQ(extractEventsWithPrefix(harness.events, "launchBenchmark:"),
              (std::vector<std::string>{"empty", "empty"}));
    EXPECT_EQ(harness.launcher.rotationSelections, (std::vector<int>{0, 1}));

    auto waitForPreparedSlotIdx = indexOfEvent(harness.events, "waitForPreparedSlot");
    auto firstWarmupIdx   = indexOfEvent(harness.events, "launchWarmup:empty");
    auto firstBenchIdx    = indexOfEvent(harness.events, "launchBenchmark:empty");
    auto firstRotateIdx   = indexOfEvent(harness.events, "selectRotationCopy:0");
    auto postWarmupIdx    = indexOfEvent(harness.events, "postWarmup");

    ASSERT_NE(waitForPreparedSlotIdx, harness.events.size());
    ASSERT_NE(firstWarmupIdx, harness.events.size());
    ASSERT_NE(firstBenchIdx, harness.events.size());
    ASSERT_NE(firstRotateIdx, harness.events.size());
    ASSERT_NE(postWarmupIdx, harness.events.size());

    EXPECT_LT(waitForPreparedSlotIdx, firstWarmupIdx);
    EXPECT_LT(waitForPreparedSlotIdx, firstBenchIdx);
    EXPECT_LT(postWarmupIdx, firstRotateIdx);
}

TEST_F(ClientRunSchedulerTest, RequeriesListenerCountsAfterPreSolutionAndPostWarmup)
{
    harness.listeners.m_warmupRuns                    = 1;
    harness.listeners.m_syncs                         = 1;
    harness.listeners.m_enqueues                      = 1;
    harness.listeners.recordCountQueries              = true;
    harness.listeners.changeWarmupRunsAfterPreSolution = true;
    harness.listeners.warmupRunsAfterPreSolution      = 2;
    harness.listeners.changeSyncsAndEnqueuesAfterPostWarmup = true;
    harness.listeners.syncsAfterPostWarmup                = 0;
    harness.listeners.enqueuesAfterPostWarmup             = 0;
    harness.config.runKernels                             = true;
    harness.config.gpuTimer                               = false;
    harness.solution->kernelsPerSolveCall                 = {{}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(extractEventsWithPrefix(harness.events, "launchWarmup:"),
              (std::vector<std::string>{"empty", "empty"}));
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchBenchmark:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "preEnqueues").empty());
    EXPECT_TRUE(hasCountQuery(harness.listeners.warmupCountQueries, 1u, false, false));
    EXPECT_TRUE(hasCountQuery(harness.listeners.warmupCountQueries, 2u, true, false));
    EXPECT_TRUE(hasCountQuery(harness.listeners.syncCountQueries, 1u, false, false));
    EXPECT_TRUE(hasCountQuery(harness.listeners.syncCountQueries, 0u, true, true));
    EXPECT_TRUE(hasCountQuery(harness.listeners.enqueueCountQueries, 1u, false, false));
    EXPECT_TRUE(hasCountQuery(harness.listeners.enqueueCountQueries, 0u, true, true));
    auto primeIndices  = indicesOfEvent(harness.events, "primeNextInputSlot");
    auto postSyncIdx   = indexOfEvent(harness.events, "postSyncs");
    auto postSolutionIdx = indexOfEvent(harness.events, "postSolution");
    ASSERT_EQ(primeIndices.size(), 2u);
    ASSERT_NE(postSyncIdx, harness.events.size());
    ASSERT_NE(postSolutionIdx, harness.events.size());
    EXPECT_LT(postSyncIdx, primeIndices.front());
    EXPECT_LT(primeIndices.back(), postSolutionIdx);
    EXPECT_EQ(harness.flushGridSizeCalls, 1);
}

TEST_F(ClientRunSchedulerTest, BenchmarkLoopUsesRotatingKernelIndexModuloInputArray)
{
    harness.listeners.m_warmupRuns = 0;
    harness.listeners.m_syncs      = 2;
    harness.listeners.m_enqueues   = 3;
    harness.config.runKernels     = true;
    harness.config.gpuTimer       = false;
    harness.data.rotatingSlots     = 2;
    harness.launcher.rotationModules = 2;
    harness.solution->kernelsPerSolveCall = {{makeKernel("slot0")}, {makeKernel("slot1")}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);

    EXPECT_EQ(harness.launcher.rotationSelections,
              (std::vector<int>{0, 1, 0, 1, 0, 1}));
    EXPECT_EQ(extractEventsWithPrefix(harness.events, "launchBenchmark:"),
              (std::vector<std::string>{"slot0", "slot1", "slot0", "slot1", "slot0", "slot1"}));
}

TEST_F(ClientRunSchedulerTest, BenchmarkPathDoesNotUseNoBenchmarkResetHook)
{
    harness.listeners.m_warmupRuns     = 0;
    harness.listeners.m_syncs          = 1;
    harness.listeners.m_enqueues       = 1;
    harness.config.runKernels          = true;
    harness.config.gpuTimer            = false;
    harness.solution->kernelsPerSolveCall = {{makeKernel("bench")}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(extractEventsWithPrefix(harness.events, "launchBenchmark:"),
              (std::vector<std::string>{"bench"}));
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "primeNextInputSlot").empty());
    auto preSyncsIdx        = indexOfEvent(harness.events, "preSyncs");
    auto preEnqueuesIdx     = indexOfEvent(harness.events, "preEnqueues");
    auto selectRotationIdx  = indexOfEvent(harness.events, "selectRotationCopy:0");
    auto launchBenchmarkIdx = indexOfEvent(harness.events, "launchBenchmark:bench");
    auto postEnqueuesIdx    = indexOfEvent(harness.events, "postEnqueues");
    auto validateEnqueuesIdx = indexOfEvent(harness.events, "validateEnqueues");
    auto postSyncsIdx       = indexOfEvent(harness.events, "postSyncs");

    ASSERT_NE(preSyncsIdx, harness.events.size());
    ASSERT_NE(preEnqueuesIdx, harness.events.size());
    ASSERT_NE(selectRotationIdx, harness.events.size());
    ASSERT_NE(launchBenchmarkIdx, harness.events.size());
    ASSERT_NE(postEnqueuesIdx, harness.events.size());
    ASSERT_NE(validateEnqueuesIdx, harness.events.size());
    ASSERT_NE(postSyncsIdx, harness.events.size());

    EXPECT_LT(preSyncsIdx, preEnqueuesIdx);
    EXPECT_LT(preEnqueuesIdx, selectRotationIdx);
    EXPECT_LT(selectRotationIdx, launchBenchmarkIdx);
    EXPECT_LT(launchBenchmarkIdx, postEnqueuesIdx);
    EXPECT_LT(postEnqueuesIdx, validateEnqueuesIdx);
    EXPECT_LT(validateEnqueuesIdx, postSyncsIdx);
}

TEST_F(ClientRunSchedulerTest,
       UserArgsBenchmarkRunRoutesThroughSolveTensileGPUAndReleasesAllRotatingSlots)
{
    harness.listeners.m_warmupRuns = 0;
    harness.listeners.m_syncs      = 1;
    harness.listeners.m_enqueues   = 1;
    harness.listeners.resetSolutionSourceOnPreProblem = true;
    harness.listeners.resetSolutionRunsOnPreSolution  = true;
    harness.listeners.solutionRunsPerSolution         = 1;
    harness.data.recordProblemContext                 = true;
    harness.data.rotatingSlots                        = 2;
    harness.config.runKernels                         = true;
    harness.config.gpuTimer                           = false;
    harness.config.useUserArgs                        = true;
    harness.solution->kernelsPerSolveCall             = {{makeKernel("slot0")},
                                                          {makeKernel("slot1")}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(countExactEvent(harness.events, "solveTensileGPU"), 2u);
    EXPECT_EQ(countExactEvent(harness.events, "solve"), 0u);
    EXPECT_EQ(harness.solution->assignedUserArgs.size(), 2u);
    EXPECT_EQ(harness.solution->releasedUserArgs.size(), 2u);
    EXPECT_NE(harness.solution->assignedUserArgs[0].first,
              harness.solution->assignedUserArgs[1].first);
    EXPECT_NE(harness.solution->assignedUserArgs[0].second,
              harness.solution->assignedUserArgs[1].second);
    EXPECT_EQ(harness.solution->assignedUserArgs, harness.solution->releasedUserArgs);
    EXPECT_EQ(harness.reporter.reportedProblemIndices, (std::vector<int>{0}));
    EXPECT_EQ(harness.reporter.reportedProblemProgress, (std::vector<std::string>{"0/0"}));

    auto solveIndices       = indicesOfEvent(harness.events, "solveTensileGPU");
    auto waitIdx            = indexOfEvent(harness.events, "waitForPreparedSlot");
    auto launchIdx          = indexOfEvent(harness.events, "launchBenchmark:slot0");
    auto releaseIndices     = indicesOfEvent(harness.events, "releaseDeviceUserArgs");
    auto postSolutionIdx    = indexOfEvent(harness.events, "postSolution");

    ASSERT_EQ(solveIndices.size(), 2u);
    ASSERT_NE(waitIdx, harness.events.size());
    ASSERT_NE(launchIdx, harness.events.size());
    ASSERT_EQ(releaseIndices.size(), 2u);
    ASSERT_NE(postSolutionIdx, harness.events.size());

    EXPECT_LT(solveIndices[1], waitIdx);
    EXPECT_LT(waitIdx, launchIdx);
    EXPECT_LT(launchIdx, releaseIndices.front());
    EXPECT_LT(releaseIndices.back(), postSolutionIdx);
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "primeNextInputSlot").empty());
}

TEST_F(ClientRunSchedulerTest,
       UserArgsNoBenchmarkRunReleasesAllRotatingSlotArgsBeforeRingResetSubmission)
{
    harness.listeners.m_warmupRuns = 0;
    harness.listeners.m_syncs      = 0;
    harness.listeners.m_enqueues   = 0;
    harness.listeners.resetSolutionSourceOnPreProblem = true;
    harness.listeners.resetSolutionRunsOnPreSolution  = true;
    harness.listeners.solutionRunsPerSolution         = 1;
    harness.data.recordProblemContext                 = true;
    harness.data.rotatingSlots                        = 2;
    harness.config.runKernels                         = true;
    harness.config.gpuTimer                           = false;
    harness.config.useUserArgs                        = true;
    harness.solution->kernelsPerSolveCall             = {{makeKernel("slot0")},
                                                          {makeKernel("slot1")}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(countExactEvent(harness.events, "solveTensileGPU"), 2u);
    EXPECT_EQ(countExactEvent(harness.events, "solve"), 0u);
    EXPECT_EQ(harness.solution->assignedUserArgs.size(), 2u);
    EXPECT_EQ(harness.solution->releasedUserArgs.size(), 2u);
    EXPECT_EQ(harness.solution->assignedUserArgs, harness.solution->releasedUserArgs);

    auto releaseIndices = indicesOfEvent(harness.events, "releaseDeviceUserArgs");
    auto primeIndices    = indicesOfEvent(harness.events, "primeNextInputSlot");
    auto postSolutionIdx = indexOfEvent(harness.events, "postSolution");

    ASSERT_EQ(releaseIndices.size(), 2u);
    ASSERT_EQ(primeIndices.size(), 2u);
    ASSERT_NE(postSolutionIdx, harness.events.size());

    EXPECT_LT(releaseIndices.back(), primeIndices.front());
    EXPECT_LT(primeIndices.back(), postSolutionIdx);
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchBenchmark:").empty());
    EXPECT_EQ(harness.reporter.reportedProblemIndices, (std::vector<int>{0}));
    EXPECT_EQ(harness.reporter.reportedProblemProgress, (std::vector<std::string>{"0/0"}));
}

TEST_F(ClientRunSchedulerTest, UserArgsSolveFailureDoesNotReleaseAndReportsInvalid)
{
    harness.listeners.m_warmupRuns = 0;
    harness.listeners.m_syncs      = 1;
    harness.listeners.m_enqueues   = 1;
    harness.config.runKernels      = true;
    harness.config.gpuTimer        = false;
    harness.config.useUserArgs     = true;
    harness.solution->throwOnSolve  = true;

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(countExactEvent(harness.events, "solveTensileGPU"), 1u);
    EXPECT_EQ(countExactEvent(harness.events, "solve"), 0u);
    EXPECT_TRUE(harness.solution->assignedUserArgs.empty());
    EXPECT_TRUE(harness.solution->releasedUserArgs.empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "waitForPreparedSlot").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchWarmup:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchBenchmark:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "primeNextInputSlot").empty());
    EXPECT_EQ(countExactEvent(harness.events, "reportInvalid"), 1u);
    EXPECT_EQ(harness.reporter.invalidCount, 1);
    ASSERT_EQ(harness.reporter.errorMessages.size(), 1u);
    EXPECT_NE(harness.reporter.errorMessages.front().find("solve failed"),
              std::string::npos);
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postSolution"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postProblem"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postBenchmarkRun"),
              harness.events.end());
}

TEST_F(ClientRunSchedulerTest, UserArgsLaunchFailureReleasesAllocatedArgsAndReportsInvalid)
{
    harness.listeners.m_warmupRuns = 0;
    harness.listeners.m_syncs      = 1;
    harness.listeners.m_enqueues   = 1;
    harness.listeners.resetSolutionSourceOnPreProblem = true;
    harness.listeners.resetSolutionRunsOnPreSolution  = true;
    harness.listeners.solutionRunsPerSolution         = 1;
    harness.data.recordProblemContext                 = true;
    harness.data.rotatingSlots                        = 2;
    harness.config.runKernels                         = true;
    harness.config.gpuTimer                           = false;
    harness.config.useUserArgs                        = true;
    harness.solution->kernelsPerSolveCall             = {{makeKernel("slot0")},
                                                          {makeKernel("slot1")}};
    harness.launcher.benchmarkLaunchResult            = hipErrorInvalidValue;

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(countExactEvent(harness.events, "solveTensileGPU"), 2u);
    EXPECT_EQ(countExactEvent(harness.events, "solve"), 0u);
    EXPECT_EQ(harness.solution->assignedUserArgs.size(), 2u);
    EXPECT_EQ(harness.solution->releasedUserArgs.size(), 2u);
    EXPECT_EQ(harness.solution->assignedUserArgs, harness.solution->releasedUserArgs);

    auto launchIdx       = indexOfEvent(harness.events, "launchBenchmark:slot0");
    auto releaseIndices   = indicesOfEvent(harness.events, "releaseDeviceUserArgs");
    auto reportInvalidIdx = indexOfEvent(harness.events, "reportInvalid");

    ASSERT_NE(launchIdx, harness.events.size());
    ASSERT_EQ(releaseIndices.size(), 2u);
    ASSERT_NE(reportInvalidIdx, harness.events.size());

    EXPECT_LT(launchIdx, releaseIndices.front());
    EXPECT_LT(releaseIndices.back(), reportInvalidIdx);
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "primeNextInputSlot").empty());
    EXPECT_EQ(countExactEvent(harness.events, "reportInvalid"), 1u);
    EXPECT_EQ(harness.reporter.invalidCount, 1);
    ASSERT_EQ(harness.reporter.errorMessages.size(), 1u);
    EXPECT_NE(harness.reporter.errorMessages.front().find("Exception occurred:"),
              std::string::npos);
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postSolution"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postProblem"),
              harness.events.end());
    EXPECT_NE(std::find(harness.events.begin(), harness.events.end(), "postBenchmarkRun"),
              harness.events.end());
}

TEST_F(ClientRunSchedulerTest, SkipsFlushGridCallbackWhenNoBenchmarkRuns)
{
    harness.config.runKernels             = false;
    harness.config.gpuTimer               = false;
    harness.listeners.m_benchmarkRunsRemaining = 0;

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(harness.flushGridSizeCalls, 0);
    EXPECT_TRUE(harness.events.empty());
}

TEST_F(ClientRunSchedulerTest, SubmitsTwoPrimeNextInputSlotCallsAfterSuccessfulExecutedRun)
{
    harness.listeners.m_warmupRuns = 0;
    harness.listeners.m_syncs      = 0;
    harness.listeners.m_enqueues   = 0;
    harness.config.runKernels     = true;
    harness.config.gpuTimer       = false;
    harness.solution->kernelsPerSolveCall = {{makeKernel("slot0")}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);

    EXPECT_EQ(extractEventsWithPrefix(harness.events, "primeNextInputSlot"),
              (std::vector<std::string>{"", ""}));
    auto firstReset = indexOfEvent(harness.events, "primeNextInputSlot");
    ASSERT_NE(firstReset, harness.events.size());
    auto secondReset = std::find(harness.events.begin() + firstReset + 1,
                                 harness.events.end(),
                                 "primeNextInputSlot");
    ASSERT_NE(secondReset, harness.events.end());
    auto postSolutionIdx = indexOfEvent(harness.events, "postSolution");
    ASSERT_NE(postSolutionIdx, harness.events.size());
    EXPECT_LT(static_cast<size_t>(std::distance(harness.events.begin(), secondReset)),
              postSolutionIdx);
}

TEST_F(ClientRunSchedulerTest, ResetSubmissionHappensBeforePostSolutionOrNextPrepare)
{
    harness.listeners.m_warmupRuns          = 0;
    harness.listeners.m_syncs               = 0;
    harness.listeners.m_enqueues            = 0;
    harness.listeners.m_solutionRunsRemaining = 2;
    harness.config.runKernels               = true;
    harness.config.gpuTimer                 = false;
    harness.solution->kernelsPerSolveCall   = {{}, {}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);

    auto prepareIndices = indicesOfEvent(harness.events, "prepareGPUInputs");
    auto resetIndices   = indicesOfEvent(harness.events, "primeNextInputSlot");
    auto postSolutionIdx = indexOfEvent(harness.events, "postSolution");

    ASSERT_EQ(prepareIndices.size(), 3u);
    ASSERT_EQ(resetIndices.size(), 4u);
    ASSERT_NE(postSolutionIdx, harness.events.size());

    EXPECT_LT(prepareIndices[0], prepareIndices[1]);
    EXPECT_LT(prepareIndices[1], resetIndices[0]);
    EXPECT_LT(resetIndices[0], resetIndices[1]);
    EXPECT_LT(resetIndices[1], prepareIndices[2]);
    EXPECT_LT(prepareIndices[2], resetIndices[2]);
    EXPECT_LT(resetIndices[2], resetIndices[3]);
    EXPECT_LT(resetIndices[3], postSolutionIdx);
}

TEST_F(ClientRunSchedulerTest, SkipsKernelFlowWhenSolutionRejected)
{
    harness.config.runKernels             = true;
    harness.config.gpuTimer               = false;
    harness.solutionSource.runCurrentSolutionResult = false;
    harness.listeners.m_warmupRuns        = 1;
    harness.listeners.m_syncs             = 1;
    harness.listeners.m_enqueues          = 1;

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "solve").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "waitForPreparedSlot").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchWarmup:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchBenchmark:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "primeNextInputSlot").empty());
    auto preSolutionIdx    = indexOfEvent(harness.events, "preSolution");
    auto postSolutionIdx   = indexOfEvent(harness.events, "postSolution");
    auto postProblemIdx    = indexOfEvent(harness.events, "postProblem");
    auto postBenchmarkIdx  = indexOfEvent(harness.events, "postBenchmarkRun");

    ASSERT_NE(preSolutionIdx, harness.events.size());
    ASSERT_NE(postSolutionIdx, harness.events.size());
    ASSERT_NE(postProblemIdx, harness.events.size());
    ASSERT_NE(postBenchmarkIdx, harness.events.size());

    EXPECT_LT(preSolutionIdx, postSolutionIdx);
    EXPECT_LT(postSolutionIdx, postProblemIdx);
    EXPECT_LT(postProblemIdx, postBenchmarkIdx);
}

TEST_F(ClientRunSchedulerTest, RuntimeErrorReportsInvalidAndPostsSolution)
{
    harness.config.runKernels = true;
    harness.config.gpuTimer   = false;
    harness.solution->throwOnSolve = true;

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(harness.reporter.invalidCount, 1);
    ASSERT_EQ(harness.reporter.errorMessages.size(), 1u);
    EXPECT_NE(harness.reporter.errorMessages.front().find("solve failed"), std::string::npos);
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "primeNextInputSlot").empty());
    EXPECT_EQ(extractEventsWithPrefix(harness.events, "reportInvalid"),
              (std::vector<std::string>{""}));
    EXPECT_TRUE(std::find(harness.events.begin(), harness.events.end(), "postSolution")
                != harness.events.end());
    EXPECT_TRUE(std::find(harness.events.begin(), harness.events.end(), "postProblem")
                != harness.events.end());
    EXPECT_TRUE(std::find(harness.events.begin(), harness.events.end(), "postBenchmarkRun")
                != harness.events.end());
}

TEST_F(ClientRunSchedulerTest, SelectionOnlyDoesNotRequireBenchmarkTimer)
{
    harness.config.runKernels          = false;
    harness.config.gpuTimer            = false;
    harness.config.icacheFlushArgs      = {true};
    harness.config.icacheFlushTimeUs    = 123.0f;
    harness.listeners.m_warmupRuns     = 1;
    harness.listeners.m_syncs          = 1;
    harness.listeners.m_enqueues       = 1;
    harness.solutionSource.runCurrentSolutionResult = true;

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    ASSERT_EQ(harness.flushTimeUs.size(), 1u);
    EXPECT_FLOAT_EQ(harness.flushTimeUs[0], 123.0f);
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "solve").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "waitForPreparedSlot").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchWarmup:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchBenchmark:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "primeNextInputSlot").empty());
}

TEST_F(ClientRunSchedulerTest, ExitOnErrorReturnsCappedError)
{
    harness.config.runKernels = false;
    harness.config.gpuTimer   = false;
    harness.config.exitOnError = true;
    harness.listeners.m_error = 300;
    harness.solutionSource.runCurrentSolutionResult = false;

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_TRUE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 255);
    EXPECT_TRUE(std::find(harness.events.begin(), harness.events.end(), "postProblem")
                == harness.events.end());
    EXPECT_TRUE(std::find(harness.events.begin(), harness.events.end(), "postBenchmarkRun")
                == harness.events.end());
}

TEST_F(ClientRunSchedulerTest, NonZeroFirstProblemMultiProblemRunResetsAndPrimesPerProblem)
{
    harness.setProblems(makeProblems(4));
    harness.config.firstProblemIdx = 1;
    harness.config.lastProblemIdx  = 3;
    harness.listeners.m_warmupRuns = 0;
    harness.listeners.m_syncs      = 0;
    harness.listeners.m_enqueues   = 0;
    harness.listeners.resetSolutionSourceOnPreProblem = true;
    harness.listeners.resetSolutionRunsOnPreSolution  = true;
    harness.listeners.solutionRunsPerSolution         = 1;
    harness.data.recordProblemContext                 = true;
    harness.data.rotatingSlots                        = 2;
    harness.config.runKernels                         = true;
    harness.config.gpuTimer                           = false;

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(harness.reporter.reportedProblemIndices, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(harness.reporter.reportedProblemProgress,
              (std::vector<std::string>{"1/3", "2/3", "3/3"}));
    EXPECT_EQ(harness.data.problemEvents,
              (std::vector<std::string>{"reset:p1",
                                        "prepare:p1",
                                        "rotate:p1:0",
                                        "prepare:p1",
                                        "prime:p1",
                                        "prime:p1",
                                        "reset:p2",
                                        "prepare:p2",
                                        "rotate:p2:0",
                                        "prepare:p2",
                                        "prime:p2",
                                        "prime:p2",
                                        "reset:p3",
                                        "prepare:p3",
                                        "rotate:p3:0",
                                        "prepare:p3",
                                        "prime:p3",
                                        "prime:p3"}));
    EXPECT_EQ(countExactEvent(harness.events, "primeNextInputSlot"), 6u);
}

TEST_F(ClientRunSchedulerTest, TwoBenchmarkPassesRepeatSchedulerOuterLoop)
{
    harness.listeners.m_benchmarkRunsRemaining = 2;
    harness.listeners.m_warmupRuns             = 0;
    harness.listeners.m_syncs                  = 1;
    harness.listeners.m_enqueues               = 1;
    harness.listeners.resetSolutionSourceOnPreProblem = true;
    harness.listeners.resetSolutionRunsOnPreSolution  = true;
    harness.listeners.solutionRunsPerSolution         = 1;
    harness.config.runKernels                         = true;
    harness.config.gpuTimer                           = false;
    harness.solution->kernelsPerSolveCall             = {{makeKernel("bench")}};

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    EXPECT_EQ(countExactEvent(harness.events, "preBenchmarkRun"), 2u);
    EXPECT_EQ(countExactEvent(harness.events, "postBenchmarkRun"), 2u);
    EXPECT_EQ(harness.reporter.reportedProblemIndices, (std::vector<int>{0, 0}));
    EXPECT_EQ(harness.reporter.reportedProblemProgress, (std::vector<std::string>{"0/0", "0/0"}));
    EXPECT_EQ(countExactEvent(harness.events, "solve"), 2u);
    EXPECT_EQ(extractEventsWithPrefix(harness.events, "launchBenchmark:"),
              (std::vector<std::string>{"bench", "bench"}));
    EXPECT_EQ(harness.flushGridSizeCalls, 2);
}

TEST_F(ClientRunSchedulerTest, AutoIcacheRotationLoadsExtraCopiesOnceAcrossOuterLoop)
{
    harness.setProblems(makeProblems(3));
    harness.config.firstProblemIdx    = 1;
    harness.config.lastProblemIdx     = 2;
    harness.listeners.m_benchmarkRunsRemaining = 2;
    harness.config.runKernels                = false;
    harness.config.gpuTimer                  = false;
    harness.config.icacheFlushArgs           = {false, true};
    harness.config.icacheFlushTimeUs         = 123.0f;
    harness.config.icacheRotateCopies        = -1;
    harness.config.icacheRotateSizeKB        = 64;
    harness.config.codeObjectFilenames       = {"kernel.co"};
    harness.data.rotatingSlots               = 2;
    harness.launcher.rotationModules         = 1;
    harness.callbacks.kernelHotPathSizeFn    = [](std::string const&) {
        return std::uintmax_t{32768};
    };

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    ASSERT_EQ(harness.launcher.extraCopyLoads.size(), 1u);
    EXPECT_EQ(harness.launcher.extraCopyLoads[0],
              std::make_pair(std::string("kernel.co"), 3));
    EXPECT_EQ(harness.launcher.rotationModules, 4);
    EXPECT_EQ(harness.flushTimeUs,
              (std::vector<float>{0.f, 123.f, 0.f, 123.f}));
}

TEST_F(ClientRunSchedulerTest, AutoIcacheRotationComputesExtraCopyCount)
{
    harness.config.runKernels            = false;
    harness.config.gpuTimer              = false;
    harness.config.icacheRotateCopies    = -1;
    harness.config.icacheRotateSizeKB    = 64;
    harness.config.codeObjectFilenames    = {"kernel.co"};
    harness.data.rotatingSlots           = 4;
    harness.launcher.rotationModules     = 1;
    harness.callbacks.kernelHotPathSizeFn = [](std::string const&) {
        return std::uintmax_t{65536};
    };

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    ASSERT_EQ(harness.launcher.extraCopyLoads.size(), 1u);
    EXPECT_EQ(harness.launcher.extraCopyLoads[0],
              std::make_pair(std::string("kernel.co"), 3));
    EXPECT_EQ(harness.launcher.rotationModules, 4);
}

TEST_F(ClientRunSchedulerTest, AutoIcacheRotationLoadsExtraCopiesOnce)
{
    harness.config.runKernels            = false;
    harness.config.gpuTimer              = false;
    harness.config.icacheRotateCopies    = -1;
    harness.config.icacheRotateSizeKB    = 64;
    harness.config.codeObjectFilenames    = {"kernel.co"};
    harness.data.rotatingSlots           = 2;
    harness.launcher.rotationModules     = 1;
    harness.callbacks.kernelHotPathSizeFn = [](std::string const&) {
        return std::uintmax_t{32768};
    };

    auto scheduler = harness.makeScheduler();
    auto result    = scheduler.run(harness.dUA, harness.dUAHost);

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);
    ASSERT_EQ(harness.launcher.extraCopyLoads.size(), 1u);
    EXPECT_EQ(harness.launcher.extraCopyLoads[0],
              std::make_pair(std::string("kernel.co"), 3));
    EXPECT_EQ(harness.launcher.rotationModules, 4);
}
