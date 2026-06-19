// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ClientRunScheduler.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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

        void preProblem(ContractionProblem* const) override
        {
            m_events.push_back("preProblem");
        }

        void postProblem() override
        {
            m_events.push_back("postProblem");
        }

        void preSolution(ContractionSolution* const) override
        {
            m_events.push_back("preSolution");
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
        }

        void validateWarmups(std::shared_ptr<ProblemInputs>,
                             TimingEvents const&,
                             TimingEvents const&) override
        {
            m_events.push_back("validateWarmups");
        }

        size_t numSyncs() override
        {
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
        bool   changeWarmupRunsAfterPreSolution      = false;
        size_t warmupRunsAfterPreSolution            = 0;
        bool   changeSyncsAndEnqueuesAfterPostWarmup = false;
        size_t syncsAfterPostWarmup                  = 0;
        size_t enqueuesAfterPostWarmup               = 0;
        mutable int m_benchmarkRunsRemaining = 1;
        mutable int m_solutionRunsRemaining  = 1;

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
        }

        void reportProblemProgress(std::string const& text) override
        {
            m_events.push_back("reportProblemProgress:" + text);
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

        void cancelAsyncReset() override
        {
            m_events.push_back("cancelAsyncReset");
        }

        std::shared_ptr<ProblemInputs> prepareGPUInputs(ContractionProblem const*) override
        {
            m_events.push_back("prepareGPUInputs");
            return std::make_shared<NullProblemInputs>();
        }

        std::vector<std::shared_ptr<ProblemInputs>> prepareRotatingGPUOutput(
            int32_t                        maxRotatingBufferNum,
            ContractionProblem const*,
            std::shared_ptr<ProblemInputs> inputs,
            hipStream_t) override
        {
            m_events.push_back("prepareRotatingGPUOutput:" + std::to_string(maxRotatingBufferNum));
            std::vector<std::shared_ptr<ProblemInputs>> rv(std::max<size_t>(1, rotatingSlots),
                                                           std::move(inputs));
            return rv;
        }

        void waitCopyDone(hipStream_t) override
        {
            m_events.push_back("waitCopyDone");
        }

        void beginAsyncReset(ContractionProblem const*) override
        {
            m_events.push_back("beginAsyncReset");
        }

        size_t rotatingSlots = 1;

    private:
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
                                                      void**,
                                                      void**,
                                                      void*,
                                                      size_t,
                                                      hipStream_t) const override
        {
            m_events.push_back("solveTensileGPU");
            if(throwOnSolve)
                throw std::runtime_error("solve failed");
            return nextKernels();
        }

        void relaseDeviceUserArgs(void*, void*) override
        {
            m_events.push_back("releaseDeviceUserArgs");
        }

        std::vector<std::vector<KernelInvocation>> kernelsPerSolveCall;
        bool                                        throwOnSolve = false;

    private:
        std::vector<KernelInvocation> nextKernels() const
        {
            if(kernelsPerSolveCall.empty())
                return {};

            auto idx = std::min(solveCallIndex++, kernelsPerSolveCall.size() - 1);
            return kernelsPerSolveCall[idx];
        }

        std::vector<std::string>& m_events;
        mutable size_t            solveCallIndex = 0;
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
            return hipSuccess;
        }

        hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                 hipStream_t,
                                 hipEvent_t,
                                 hipEvent_t) override
        {
            m_events.push_back("launchBenchmark:" + kernelLabel(kernels));
            return hipSuccess;
        }

        int numRotationModulesValue() const
        {
            return rotationModules;
        }

        int                                            rotationModules = 1;
        std::vector<int>                               rotationSelections;
        std::vector<std::pair<std::string, int>>       extraCopyLoads;

    private:
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

TEST_F(ClientRunSchedulerTest, OneUntimedRunPreservesDataAndListenerOrder)
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

    EXPECT_EQ(harness.events,
              (std::vector<std::string>{"preBenchmarkRun",
                                        "reportProblemIndex:0",
                                        "reportProblemProgress:0/0",
                                        "preProblem",
                                        "cancelAsyncReset",
                                        "prepareGPUInputs",
                                        "prepareRotatingGPUOutput:1",
                                        "deviceSynchronize",
                                        "preSolution",
                                        "prepareGPUInputs",
                                        "solve",
                                        "waitCopyDone",
                                        "preWarmup",
                                        "launchWarmup:empty",
                                        "validateWarmups",
                                        "postWarmup",
                                        "preSyncs",
                                        "postSyncs",
                                        "beginAsyncReset",
                                        "beginAsyncReset",
                                        "postSolution",
                                        "postProblem",
                                        "postBenchmarkRun"}));
    EXPECT_TRUE(harness.launcher.rotationSelections.empty());
}

TEST_F(ClientRunSchedulerTest, WaitCopyDonePrecedesWarmupAndBenchmarkLaunches)
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

    auto waitCopyDoneIdx = indexOfEvent(harness.events, "waitCopyDone");
    auto firstWarmupIdx   = indexOfEvent(harness.events, "launchWarmup:empty");
    auto firstBenchIdx    = indexOfEvent(harness.events, "launchBenchmark:empty");
    auto firstRotateIdx   = indexOfEvent(harness.events, "selectRotationCopy:0");
    auto postWarmupIdx    = indexOfEvent(harness.events, "postWarmup");

    ASSERT_NE(waitCopyDoneIdx, harness.events.size());
    ASSERT_NE(firstWarmupIdx, harness.events.size());
    ASSERT_NE(firstBenchIdx, harness.events.size());
    ASSERT_NE(firstRotateIdx, harness.events.size());
    ASSERT_NE(postWarmupIdx, harness.events.size());

    EXPECT_LT(waitCopyDoneIdx, firstWarmupIdx);
    EXPECT_LT(waitCopyDoneIdx, firstBenchIdx);
    EXPECT_LT(postWarmupIdx, firstRotateIdx);
}

TEST_F(ClientRunSchedulerTest, RequeriesListenerCountsAfterPreSolutionAndPostWarmup)
{
    harness.listeners.m_warmupRuns                    = 1;
    harness.listeners.m_syncs                         = 1;
    harness.listeners.m_enqueues                      = 1;
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
    EXPECT_EQ(
        harness.events,
        (std::vector<std::string>{"preBenchmarkRun",
                                  "reportProblemIndex:0",
                                  "reportProblemProgress:0/0",
                                  "preProblem",
                                  "cancelAsyncReset",
                                  "prepareGPUInputs",
                                  "prepareRotatingGPUOutput:1",
                                  "deviceSynchronize",
                                  "preSolution",
                                  "prepareGPUInputs",
                                  "solve",
                                  "waitCopyDone",
                                  "preWarmup",
                                  "launchWarmup:empty",
                                  "validateWarmups",
                                  "launchWarmup:empty",
                                  "postWarmup",
                                  "preSyncs",
                                  "postSyncs",
                                  "beginAsyncReset",
                                  "beginAsyncReset",
                                  "postSolution",
                                  "postProblem",
                                  "postBenchmarkRun"}));
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchBenchmark:").empty());
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

TEST_F(ClientRunSchedulerTest, SubmitsTwoAsyncResetsAfterSuccessfulExecutedRun)
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

    EXPECT_EQ(extractEventsWithPrefix(harness.events, "beginAsyncReset"),
              (std::vector<std::string>{"", ""}));
    auto firstReset = indexOfEvent(harness.events, "beginAsyncReset");
    ASSERT_NE(firstReset, harness.events.size());
    auto secondReset = std::find(harness.events.begin() + firstReset + 1,
                                 harness.events.end(),
                                 "beginAsyncReset");
    ASSERT_NE(secondReset, harness.events.end());
    auto postSolutionIdx = indexOfEvent(harness.events, "postSolution");
    ASSERT_NE(postSolutionIdx, harness.events.size());
    EXPECT_LT(static_cast<size_t>(std::distance(harness.events.begin(), secondReset)),
              postSolutionIdx);
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
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "waitCopyDone").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchWarmup:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchBenchmark:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "beginAsyncReset").empty());
    EXPECT_EQ(harness.events,
              (std::vector<std::string>{"preBenchmarkRun",
                                        "reportProblemIndex:0",
                                        "reportProblemProgress:0/0",
                                        "preProblem",
                                        "cancelAsyncReset",
                                        "prepareGPUInputs",
                                        "prepareRotatingGPUOutput:1",
                                        "deviceSynchronize",
                                        "preSolution",
                                        "postSolution",
                                        "postProblem",
                                        "postBenchmarkRun"}));
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
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "beginAsyncReset").empty());
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
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "waitCopyDone").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchWarmup:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "launchBenchmark:").empty());
    EXPECT_TRUE(extractEventsWithPrefix(harness.events, "beginAsyncReset").empty());
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
