// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "BenchmarkTimer.hpp"
#include "ClientProblemFactory.hpp"
#include "ClientRunScheduler.hpp"
#include "ClientRunSchedulerAdapters.hpp"
#include "DataInitialization.hpp"
#include "DataInitializationTestUtils.hpp"
#include "HipCopyEngine.hpp"
#include "HipStreamGuard.hpp"
#include "MetaRunListener.hpp"
#include "MetaResultReporter.hpp"
#include "PerformanceReporter.hpp"
#include "RecordingCopyEngine.hpp"
#include "ReferenceValidator.hpp"
#include "ResultReporter.hpp"
#include "RunListener.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    using TensileLite::testing::HipStreamGuard;
    using TensileLite::testing::RecordingCopyEngine;
    using TensileLite::testing::buildRingArgs;
    using TensileLite::testing::detail::setDataInitArg;

    class ForwardingCopyEngine final : public CopyEngine
    {
    public:
        using CallType          = RecordingCopyEngine::CallType;
        using Call              = RecordingCopyEngine::Call;
        using CopySubmissionMode = CopyEngine::CopySubmissionMode;

        hipStream_t stream() const noexcept override
        {
            return m_delegate.stream();
        }

        void copy(void*             dst,
                  void const*       src,
                  size_t            bytes,
                  hipMemcpyKind     kind,
                  hipStream_t       stream,
                  CopySubmissionMode mode) override
        {
            calls.push_back({CallType::Copy, dst, src, bytes, kind, stream, mode});
            m_delegate.copy(dst, src, bytes, kind, stream, mode);
        }

        void synchronize(hipStream_t stream) override
        {
            calls.push_back({CallType::Synchronize,
                             nullptr,
                             nullptr,
                             0,
                             hipMemcpyHostToHost,
                             stream,
                             CopySubmissionMode::Sync});
            m_delegate.synchronize(stream);
        }

        void synchronizeDefaultStream() override
        {
            calls.push_back({CallType::SynchronizeDefaultStream,
                             nullptr,
                             nullptr,
                             0,
                             hipMemcpyHostToHost,
                             stream(),
                             CopySubmissionMode::Sync});
            m_delegate.synchronizeDefaultStream();
        }

        void recordCopyDone(size_t slot) override
        {
            calls.push_back({CallType::RecordCopyDone,
                             nullptr,
                             nullptr,
                             0,
                             hipMemcpyHostToHost,
                             stream(),
                             CopySubmissionMode::Sync,
                             slot});
            m_delegate.recordCopyDone(slot);
        }

        void waitForCopyDone(size_t slot, hipStream_t computeStream) override
        {
            calls.push_back({CallType::WaitForCopyDone,
                             nullptr,
                             nullptr,
                             0,
                             hipMemcpyHostToHost,
                             nullptr,
                             CopySubmissionMode::Sync,
                             slot,
                             computeStream});
            m_delegate.waitForCopyDone(slot, computeStream);
        }

        std::vector<Call> calls;

    private:
        HipCopyEngine m_delegate{3};
    };

    class InspectableDataInitialization final : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;

        bool ringPolicyAllowed() const
        {
            return m_ringPolicy.allowed;
        }

        size_t activeBufferCount() const
        {
            return m_ring.activeBufferCount();
        }

        bool altSlotsReady() const
        {
            return m_altSlotsReady;
        }

        size_t activeRingSlot() const
        {
            return m_ring.activeSlot();
        }

        std::shared_ptr<ProblemInputs> cachedGPUInputs() const
        {
            return m_cachedGPUInputs;
        }

        auto const& slotState(size_t slot) const
        {
            return m_gpuInputSlots.at(slot);
        }
    };

    class RecordingRunReporter final : public RunReporter
    {
    public:
        void reportProblemIndex(int idx) override
        {
            problemIndices.push_back(idx);
        }

        void reportProblemProgress(std::string const& text) override
        {
            problemProgress.push_back(text);
        }

        void reportInvalid() override
        {
            ++invalidCount;
        }

        void logError(std::string const& message) override
        {
            errorMessages.push_back(message);
        }

        std::vector<int>         problemIndices;
        std::vector<std::string>  problemProgress;
        std::vector<std::string>  errorMessages;
        int                       invalidCount = 0;
    };

    class RecordingResultReporter final : public ResultReporter
    {
    public:
        void reportValue_string(std::string const& key, std::string const& value) override
        {
            if(key == ResultKey::Validation)
                validationReports.push_back(value);
        }

        void reportValue_uint(std::string const&, uint64_t) override {}
        void reportValue_int(std::string const&, int64_t) override {}
        void reportValue_double(std::string const&, double) override {}
        void reportValue_sizes(std::string const&, std::vector<size_t> const&) override {}
        void reportValue_vecOfSizes(std::string const&,
                                    std::vector<std::vector<size_t>> const&) override {}
        void finalizeReport() override {}

        std::vector<std::string> validationReports;
    };

    class NoopSolution final : public ContractionSolution
    {
    public:
        explicit NoopSolution(std::string label)
        {
            solutionName = std::move(label);
            kernelName   = solutionName;
            sizeMapping.macroTile    = TensileLite::dim3{32, 32, 1};
            sizeMapping.workGroupSize = TensileLite::dim3{64, 1, 1};
            sizeMapping.threadTile    = TensileLite::dim3{1, 1, 1};
            sizeMapping.depthU        = 1;
            sizeMapping.globalSplitU  = 1;
            sizeMapping.LocalSplitU   = 1;
            sizeMapping.waveNum       = 1;
            ideals[32]                = 1.0;
        }

        std::vector<KernelInvocation>
            solve(ContractionProblem const&,
                  ProblemInputs const&,
                  Hardware const&,
                  void*,
                  size_t,
                  hipStream_t) const override
        {
            return {};
        }

        std::vector<KernelInvocation>
            solveTensileGPU(ContractionProblem const&,
                            ProblemInputs const&,
                            Hardware const&,
                            void**,
                            void**,
                            void*,
                            size_t,
                            hipStream_t) const override
        {
            return {};
        }

        void relaseDeviceUserArgs(void*, void*) override {}
    };

    class RecordingKernelLauncher final : public RunKernelLauncher
    {
    public:
        int numRotationModules() override
        {
            return rotationModules;
        }

        void selectRotationCopy(int idx) override
        {
            rotationSelections.push_back(idx);
        }

        hipError_t loadCodeObjectFileExtraCopies(std::string const&, int) override
        {
            return hipSuccess;
        }

        hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                 hipStream_t,
                                 std::vector<hipEvent_t> const&,
                                 std::vector<hipEvent_t> const&) override
        {
            warmupLaunches.push_back(kernelLabel(kernels));
            return hipSuccess;
        }

        hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                 hipStream_t,
                                 hipEvent_t,
                                 hipEvent_t) override
        {
            benchmarkLaunches.push_back(kernelLabel(kernels));
            return hipSuccess;
        }

        std::vector<int>         rotationSelections;
        std::vector<std::string> warmupLaunches;
        std::vector<std::string> benchmarkLaunches;
        int                      rotationModules = 1;

    private:
        static std::string kernelLabel(std::vector<KernelInvocation> const& kernels)
        {
            if(kernels.empty())
                return "empty";
            if(kernels.front().kernelName.empty())
                return "unnamed";
            return kernels.front().kernelName;
        }
    };

    class RecordingSolutionSource final : public RunSolutionSource
    {
    public:
        explicit RecordingSolutionSource(
            std::vector<std::shared_ptr<ContractionSolution>> solutions)
            : m_solutions(std::move(solutions))
        {
        }

        bool moreSolutionsInProblem() const override
        {
            return m_nextSolution < m_solutions.size();
        }

        std::shared_ptr<ContractionSolution> getSolution() override
        {
            return m_solutions.at(m_nextSolution++);
        }

        bool runCurrentSolution() override
        {
            return true;
        }

    private:
        std::vector<std::shared_ptr<ContractionSolution>> m_solutions;
        size_t                                            m_nextSolution = 0;
    };

    ::testing::AssertionResult hasHipDevice()
    {
        int        deviceCount = 0;
        hipError_t err         = hipGetDeviceCount(&deviceCount);
        if(err != hipSuccess)
        {
            return ::testing::AssertionFailure()
                   << "hipGetDeviceCount failed: " << hipGetErrorString(err);
        }

        if(deviceCount <= 0)
            return ::testing::AssertionFailure() << "No HIP devices available";

        return ::testing::AssertionSuccess();
    }

    Client::po::variables_map buildCompositionArgs()
    {
        auto args = buildRingArgs({{32, 32, 32}}, 1);

        setDataInitArg(args, "init-alpha", std::any(InitMode::Zero));
        setDataInitArg(args, "init-beta", std::any(InitMode::Zero));
        setDataInitArg(args, "use-gpu-timer", std::any(false));
        setDataInitArg(args, "sync-after-warmups", std::any(false));
        setDataInitArg(args, "sleep-percent", std::any(0));
        setDataInitArg(args, "skip-slow-solution-ratio", std::any(0.0f));
        setDataInitArg(args, "prob-sol-map", std::any(std::map<int, int>{}));
        setDataInitArg(args, "print-tensor-scale-alpha-vec", std::any(false));

        return args;
    }

    size_t findCopyEvent(std::vector<ForwardingCopyEngine::Call> const& calls,
                         ForwardingCopyEngine::CallType                 type,
                         size_t                                        slot)
    {
        auto it = std::find_if(calls.begin(), calls.end(), [&](ForwardingCopyEngine::Call const& call) {
            return call.type == type && call.slot == slot;
        });
        if(it == calls.end())
            return calls.size();

        return static_cast<size_t>(std::distance(calls.begin(), it));
    }
} // namespace

TEST(ClientRunSchedulerProductionComposition,
     ValidationOnlyNoBenchmarkConsumesPreparedRingSlot)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto args = buildCompositionArgs();
    ClientProblemFactory factory(args);
    auto const&          problems = factory.problems();
    ASSERT_EQ(problems.size(), 1u);

    AMDGPU hardware(AMDGPU::Processor::gfx950, 120, "test-gpu");
    HipStreamGuard computeStream(hipStreamNonBlocking);

    auto copyEngine = std::make_shared<ForwardingCopyEngine>();
    auto dataInit    = std::make_shared<InspectableDataInitialization>(args, factory, copyEngine);

    MetaRunListener listeners;
    listeners.addListener(dataInit);
    listeners.addListener(std::make_shared<ReferenceValidator>(args, dataInit));
    auto benchmarkTimer = std::make_shared<BenchmarkTimer>(args, hardware, 0.0f);
    listeners.addListener(benchmarkTimer);

    auto recordingReporter = std::make_shared<RecordingResultReporter>();
    auto reporterStack     = std::make_shared<MetaResultReporter>();
    reporterStack->addReporter(std::make_shared<PerformanceReporter>(0, 0.0, 0.0, 0.0, 0.0));
    reporterStack->addReporter(recordingReporter);
    listeners.setReporter(reporterStack);

    RecordingRunReporter runReporter;
    RecordingSolutionSource solutionSource({std::make_shared<NoopSolution>("prime-slot-0"),
                                            std::make_shared<NoopSolution>("prime-slot-1")});

    RecordingKernelLauncher launcher;
    launcher.rotationModules = 1;

    ClientRunSchedulerConfig schedulerConfig;
    schedulerConfig.firstProblemIdx = 0;
    schedulerConfig.lastProblemIdx  = 0;
    schedulerConfig.runKernels      = true;
    schedulerConfig.gpuTimer        = false;
    schedulerConfig.icacheFlushArgs = {false};

    ClientRunSchedulerCallbacks callbacks;
    callbacks.flushGridSizeFn = [] { return uint32_t{0}; };
    callbacks.deviceSynchronizeFn = [] { HIP_CHECK_EXC(hipDeviceSynchronize()); };

    void* dUA     = nullptr;
    void* dUAHost = nullptr;

    SchedulerDataCoordinatorAdapter dataCoordinator(dataInit);
    ClientRunScheduler scheduler(
        schedulerConfig,
        ClientRunSchedulerDependencies{&problems,
                                       &listeners,
                                       &runReporter,
                                       &dataCoordinator,
                                       &solutionSource,
                                       &launcher,
                                       &hardware,
                                       computeStream.get(),
                                       callbacks});

    auto result = scheduler.run(dUA, dUAHost);
    computeStream.synchronize();

    EXPECT_FALSE(result.exitedEarly);
    EXPECT_EQ(result.returnCode, 0);

    EXPECT_EQ(runReporter.problemIndices, (std::vector<int>{0}));
    EXPECT_EQ(runReporter.problemProgress, (std::vector<std::string>{"0/0"}));
    EXPECT_EQ(runReporter.invalidCount, 0);
    EXPECT_TRUE(runReporter.errorMessages.empty());

    EXPECT_EQ(recordingReporter->validationReports,
              (std::vector<std::string>{"PASSED", "PASSED"}));

    EXPECT_EQ(launcher.warmupLaunches, (std::vector<std::string>{"empty", "empty"}));
    EXPECT_TRUE(launcher.benchmarkLaunches.empty());
    EXPECT_TRUE(launcher.rotationSelections.empty());

    EXPECT_TRUE(dataInit->ringPolicyAllowed());
    EXPECT_EQ(dataInit->activeBufferCount(), 3u);
    EXPECT_TRUE(dataInit->altSlotsReady());
    EXPECT_EQ(dataInit->activeRingSlot(), 1u);
    EXPECT_NE(dataInit->cachedGPUInputs(), nullptr);
    EXPECT_EQ(dataInit->cachedGPUInputs(), dataInit->slotState(1).cachedInputs);
    EXPECT_NE(dataInit->slotState(0).cachedInputs, nullptr);
    EXPECT_NE(dataInit->slotState(1).cachedInputs, nullptr);
    EXPECT_NE(dataInit->slotState(2).cachedInputs, nullptr);

    auto const recordCopyDoneIdx = findCopyEvent(copyEngine->calls,
                                                 ForwardingCopyEngine::CallType::RecordCopyDone,
                                                 1);
    auto const waitForCopyDoneIdx = findCopyEvent(copyEngine->calls,
                                                  ForwardingCopyEngine::CallType::WaitForCopyDone,
                                                  1);

    ASSERT_NE(recordCopyDoneIdx, copyEngine->calls.size());
    ASSERT_NE(waitForCopyDoneIdx, copyEngine->calls.size());
    EXPECT_LT(recordCopyDoneIdx, waitForCopyDoneIdx);
    EXPECT_EQ(copyEngine->calls[recordCopyDoneIdx].slot, 1u);
    EXPECT_EQ(copyEngine->calls[waitForCopyDoneIdx].slot, 1u);
    EXPECT_EQ(copyEngine->calls[waitForCopyDoneIdx].computeStream, computeStream.get());
}
