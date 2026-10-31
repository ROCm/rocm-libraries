// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "MetaRunListener.hpp"
#include "ResultReporter.hpp"
#include "SynchronizerValidator.hpp"

#include <Tensile/ContractionSolution.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace TensileLite::Client;

namespace
{
    // Captures every reportValue_string call (the FAILED verdict is reported
    // through this overload) and every logged message.
    class FakeResultReporter : public ResultReporter
    {
    public:
        std::vector<std::pair<std::string, std::string>> stringReports;
        std::string                                      logged;

        void reportValue_string(std::string const& key, std::string const& value) override
        {
            stringReports.emplace_back(key, value);
        }
        void reportValue_uint(std::string const&, uint64_t) override {}
        void reportValue_int(std::string const&, int64_t) override {}
        void reportValue_double(std::string const&, double) override {}
        void reportValue_sizes(std::string const&, std::vector<size_t> const&) override {}
        void reportValue_vecOfSizes(std::string const&,
                                    std::vector<std::vector<size_t>> const&) override
        {
        }
        void finalizeReport() override {}
        bool logAtLevel(LogLevel) override
        {
            return true;
        }
        void logMessage(LogLevel, std::string const& message) override
        {
            logged += message;
        }
    };

    // Exposes the protected reporting state so the test can drive
    // preSolution()/postSolution() without a GPU-backed dirty buffer.
    class TestableSynchronizerValidator : public SynchronizerValidator
    {
    public:
        using SynchronizerValidator::SynchronizerValidator;
        void markFailed()
        {
            m_failedInSolution = true;
        }
        // The listener is passive, so the gate is not visible through
        // needMoreRunsInSolution; read it directly.
        bool mayUseSynchronizer() const
        {
            return m_mayUseSynchronizer;
        }
        bool isActive() const
        {
            return active();
        }
    };

    po::variables_map enabledArgs()
    {
        po::variables_map vm;
        vm["check-synchronizer"].value() = true;
        return vm;
    }

    po::variables_map disabledArgs()
    {
        po::variables_map vm;
        vm["check-synchronizer"].value() = false;
        return vm;
    }

    // The gate reads only sizeMapping. ContractionSolution is non-copyable, so
    // fill one in place rather than returning it.
    using TensileLite::TileProcessingStrategy;

    void setSolution(TensileLite::ContractionSolution& s,
                     TileProcessingStrategy            strategy,
                     int                               globalAccumulation,
                     int                               streamKAtomic = 0)
    {
        s.sizeMapping.tileProcessingStrategy = strategy;
        s.sizeMapping.globalAccumulation     = globalAccumulation;
        s.sizeMapping.streamKAtomic          = streamKAtomic;
    }

    using TensileLite::CustomArgSemantic;

    // A custom kernel with no tile-processing strategy, accumulation mode,
    // workspace type or amaxD, so only its declared args can enable the scan.
    void setCustomKernel(TensileLite::ContractionSolution&     s,
                         std::vector<CustomArgSemantic> const& semantics,
                         bool                                  generated = false)
    {
        setSolution(s, TileProcessingStrategy::None, 0);
        s.customKernel.name      = "custom_kernel";
        s.customKernel.generated = generated;
        for(auto semantic : semantics)
            s.customKernel.args.push_back({TensileLite::CustomArgType::address, semantic});
    }

    bool customKernelIsChecked(std::vector<CustomArgSemantic> const& semantics,
                               bool                                  generated = false)
    {
        TestableSynchronizerValidator    validator(enabledArgs());
        TensileLite::ContractionSolution solution;
        setCustomKernel(solution, semantics, generated);
        validator.preSolution(&solution);
        return validator.mayUseSynchronizer();
    }

}

// Passive, so it cannot turn a zero-launch codegen config (validate 0, syncs 0)
// into an execution one.
TEST(SynchronizerValidatorReporting, ValidatorNeverDrivesARun)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::StreamK, 0); // a consumer, so this is not the gate talking

    validator.preSolution(&solution);
    ASSERT_TRUE(validator.mayUseSynchronizer());
    EXPECT_FALSE(validator.needMoreRunsInSolution());
    EXPECT_EQ(validator.numWarmupRuns(), 0u);
}

// Switched off, a consumer solution is still inert.
TEST(SynchronizerValidatorReporting, DisabledValidatorChecksNothing)
{
    TestableSynchronizerValidator    validator(disabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::StreamK, 0);

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.mayUseSynchronizer());
    EXPECT_FALSE(validator.isActive());
}

// StreamK uses the buffer as its work-queue / fixup Flags.
TEST(SynchronizerValidatorReporting, StreamKSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::StreamK, 0);

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.mayUseSynchronizer());
}

// GSU MultipleBufferSingleKernel is the other consumer and is not StreamK, so
// gating on StreamK alone would silently drop gsu_mbsk.yaml's coverage.
TEST(SynchronizerValidatorReporting, MbskSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::None, 3);

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.mayUseSynchronizer());
}

// AdaptiveGemmGSUA kernels switch to MBSK whenever the pointer is non-null,
// which the grouped user-args launch always passes, even for a static MB mode.
TEST(SynchronizerValidatorReporting, AdaptiveGsuaSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::None, 2);
    solution.sizeMapping.adaptiveGemmGSUA = 1;

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.mayUseSynchronizer());
}

// Unknown solution means unknown answer; scan rather than skip.
TEST(SynchronizerValidatorReporting, UnknownSolutionIsChecked)
{
    TestableSynchronizerValidator validator(enabledArgs());

    validator.preSolution(nullptr);
    EXPECT_TRUE(validator.mayUseSynchronizer());
}

// amaxD is the third consumer: the dispatcher appends the buffer as AmaxSync
// whenever outputAmaxD is set, independent of tile-processing strategy and globalAccumulation.
TEST(SynchronizerValidatorReporting, AmaxDSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::None, 0);
    solution.problemType.outputAmaxD = true;

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.mayUseSynchronizer());
}

// Custom StreamK kernels read and reset the buffer through AddressSynchronizer
// although they set no tile-processing strategy.
TEST(SynchronizerValidatorReporting, CustomStreamKSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::None, 0);
    solution.customKernel.workspaceType = TensileLite::CustomWorkspaceType::StreamK;

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.mayUseSynchronizer());
}

// Atomic StreamK reduces in place; the dispatcher never appends Flags for it.
TEST(SynchronizerValidatorReporting, AtomicStreamKSolutionIsSkipped)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::StreamK, 0, /*streamKAtomic=*/1);

    validator.preSolution(&solution);
    EXPECT_FALSE(validator.mayUseSynchronizer());
}

// Persistent DataParallel kernels have neither the workspace nor the Flags
// argument, so the buffer the check reads is never passed.
TEST(SynchronizerValidatorReporting, DataParallelSolutionIsSkipped)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::DataParallel, 0);

    validator.preSolution(&solution);
    EXPECT_FALSE(validator.mayUseSynchronizer());
}

// Everything else never receives the buffer, so a scan could only come back
// clean; skipping is what keeps the check free on those runs.
TEST(SynchronizerValidatorReporting, NonConsumerSolutionIsSkipped)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::None, 0);

    validator.preSolution(&solution);
    EXPECT_FALSE(validator.mayUseSynchronizer());
}

// The custom-arg dispatch passes the buffer as AddressFlags (non-parallel
// reduction), AddressSynchronizer and AmaxSync, independent of the solution's
// strategy, workspace type and amaxD flag.
TEST(SynchronizerValidatorReporting, CustomAddressFlagsArgIsChecked)
{
    EXPECT_TRUE(customKernelIsChecked({CustomArgSemantic::AddressA, CustomArgSemantic::AddressFlags}));
}

TEST(SynchronizerValidatorReporting, CustomAddressSynchronizerArgIsChecked)
{
    EXPECT_TRUE(customKernelIsChecked({CustomArgSemantic::AddressSynchronizer}));
}

TEST(SynchronizerValidatorReporting, CustomAmaxSyncArgIsChecked)
{
    EXPECT_TRUE(customKernelIsChecked({CustomArgSemantic::AddressAmaxOut,
                                       CustomArgSemantic::AmaxWS,
                                       CustomArgSemantic::AmaxSync}));
}

// A custom Synchronizer arg is null unless the resolved accumulation is MBSK,
// which requires a static globalAccumulation of 3.
TEST(SynchronizerValidatorReporting, CustomSynchronizerArgFollowsMbsk)
{
    EXPECT_FALSE(customKernelIsChecked({CustomArgSemantic::Synchronizer}));

    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setCustomKernel(solution, {CustomArgSemantic::Synchronizer});
    solution.sizeMapping.globalAccumulation = 3;
    validator.preSolution(&solution);
    EXPECT_TRUE(validator.mayUseSynchronizer());
}

// GSUSync is a constant, not the buffer.
TEST(SynchronizerValidatorReporting, CustomGsuSyncArgIsSkipped)
{
    EXPECT_FALSE(customKernelIsChecked({CustomArgSemantic::GSUSync}));
}

TEST(SynchronizerValidatorReporting, CustomUnrelatedArgsAreSkipped)
{
    EXPECT_FALSE(customKernelIsChecked({CustomArgSemantic::AddressA,
                                        CustomArgSemantic::AddressB,
                                        CustomArgSemantic::AddressWorkspace,
                                        CustomArgSemantic::DebugBuffer}));
}

// Generated kernels take singleCallArgs, not the custom-arg dispatch, so the
// args they declare do not enable the scan.
TEST(SynchronizerValidatorReporting, GeneratedKernelArgsAreSkipped)
{
    EXPECT_FALSE(customKernelIsChecked(
        {CustomArgSemantic::AddressFlags, CustomArgSemantic::AddressSynchronizer,
         CustomArgSemantic::AmaxSync, CustomArgSemantic::Synchronizer},
        /*generated=*/true));
}

TEST(SynchronizerValidatorReporting, CleanSolutionReportsNothing)
{
    TestableSynchronizerValidator validator(enabledArgs());
    auto                          reporter = std::make_shared<FakeResultReporter>();
    validator.setReporter(reporter);

    validator.preSolution(nullptr);
    validator.postSolution();

    EXPECT_EQ(validator.error(), 0);
    EXPECT_TRUE(reporter->stringReports.empty());
}

TEST(SynchronizerValidatorReporting, DirtySolutionReportsFailureOnce)
{
    TestableSynchronizerValidator validator(enabledArgs());
    auto                          reporter = std::make_shared<FakeResultReporter>();
    validator.setReporter(reporter);

    validator.preSolution(nullptr);
    validator.markFailed();
    validator.postSolution();

    EXPECT_EQ(validator.error(), 1);
    ASSERT_EQ(reporter->stringReports.size(), 1u);
    EXPECT_EQ(reporter->stringReports[0].first, ResultKey::Validation);
    EXPECT_EQ(reporter->stringReports[0].second, "FAILED");

    // preSolution() resets the flag, so the next (clean) solution does not
    // re-report the previous one's failure.
    validator.preSolution(nullptr);
    validator.postSolution();
    EXPECT_EQ(validator.error(), 1);
    EXPECT_EQ(reporter->stringReports.size(), 1u);
}

namespace
{
    enum class TransferFailure { None, Readback, Clear };

    class HostBufferValidator : public SynchronizerValidator
    {
    public:
        HostBufferValidator() : SynchronizerValidator(enabledArgs()) {}
        TransferFailure failure = TransferFailure::None;
        int             reads   = 0;

    protected:
        uint8_t* readBuffer(void* device, size_t) override
        {
            reads++;
            if(failure == TransferFailure::Readback)
                throw std::runtime_error("readback failed");
            return static_cast<uint8_t*>(device);
        }
        void clearBuffer(void* device, size_t bytes) override
        {
            if(failure == TransferFailure::Clear)
                throw std::runtime_error("clear failed");
            std::fill_n(static_cast<uint8_t*>(device), bytes, uint8_t{0});
        }
    };

    // Models the reference listener's successful verdict at postSolution.
    class PassingReferenceListener : public FakeResultReporter
    {
        void setReporter(std::shared_ptr<ResultReporter> reporter) override
        {
            RunListener::setReporter(reporter);
        }
        void postSolution() override
        {
            m_reporter->report(ResultKey::Validation, "PASSED");
        }
    };

    // main's listener calls for one problem, with the Synchronizer in host memory.
    // As in main, the validators get the per-problem inputs (prepared before the
    // solution loop: the GuardPageFront buffer), while each run iteration's
    // kernels use that iteration's inputs: the same buffer, or under
    // GuardPageAll a different one (GuardPageBack) in the second iteration.
    struct SolutionLoop
    {
        using Inputs = std::shared_ptr<TensileLite::ProblemInputs>;

        std::shared_ptr<HostBufferValidator>            validator;
        std::shared_ptr<FakeResultReporter>             reporter;
        MetaRunListener                                 listeners;
        TensileLite::ContractionProblemGemm             problem;
        std::vector<uint32_t>                           buffer;
        std::vector<uint32_t>                           backBuffer;
        std::shared_ptr<TensileLite::ContractionInputs> inputs;
        std::shared_ptr<TensileLite::ContractionInputs> backInputs;
        TimingEvents                                    events;
        hipStream_t                                     stream = nullptr;

        SolutionLoop()
            : validator(std::make_shared<HostBufferValidator>())
            , reporter(std::make_shared<FakeResultReporter>())
            , buffer(4, 0)
            , backBuffer(4, 0)
            , inputs(std::make_shared<TensileLite::ContractionInputs>())
            , backInputs(std::make_shared<TensileLite::ContractionInputs>())
            , events(0, 0)
        {
            // Match main: postSolution visits these in reverse order.
            listeners.addListener(validator);
            listeners.addListener(std::make_shared<PassingReferenceListener>());
            listeners.setReporter(reporter);
            problem.setSynchronizer(rocisa::DataType::Int32, buffer.size());
            inputs->Synchronizer     = buffer.data();
            backInputs->Synchronizer = backBuffer.data();
            listeners.preProblem(&problem);
        }

        // After the input reset, before the iteration's launches.
        void beginIteration(bool guardPageBack = false)
        {
            validator->beginRunIteration({guardPageBack ? Inputs(backInputs) : Inputs(inputs)});
        }
        // Warmup 0, then the validators.
        void warmup()
        {
            listeners.preWarmup();
            listeners.validateWarmups(inputs, events, events);
        }
        void enqueues()
        {
            listeners.preEnqueues(stream);
            listeners.validateEnqueues(inputs, events, events);
        }
        void endIteration()
        {
            validator->checkRunIteration();
        }

        std::string lastVerdict() const
        {
            return reporter->stringReports.empty() ? "" : reporter->stringReports.back().second;
        }
        long failedVerdicts() const
        {
            return std::count_if(reporter->stringReports.begin(),
                                 reporter->stringReports.end(),
                                 [](auto const& r) { return r.second == "FAILED"; });
        }
    };

    class SynchronizerValidatorLifecycle : public ::testing::TestWithParam<TransferFailure> {};
}

// Also: a dirty buffer gives FAILED although the reference check passed.
TEST_P(SynchronizerValidatorLifecycle, FailureSurvivesReferenceVerdictAndNextSolutionIsClean)
{
    SolutionLoop loop;
    loop.buffer[1] = 1;
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.validator->failure = GetParam();

    // main's catch handles the throw, then calls postSolution.
    if(GetParam() == TransferFailure::None)
        EXPECT_NO_THROW(loop.endIteration());
    else
        EXPECT_THROW(loop.endIteration(), std::runtime_error);
    loop.listeners.postSolution();

    EXPECT_EQ(loop.listeners.error(), 1);
    ASSERT_EQ(loop.reporter->stringReports.size(), 2u);
    EXPECT_EQ(loop.reporter->stringReports.back(),
              std::make_pair(std::string(ResultKey::Validation), std::string("FAILED")));
    EXPECT_EQ(loop.buffer[1], GetParam() == TransferFailure::None ? 0u : 1u);
    // A failed check is not repeated in postSolution.
    EXPECT_EQ(loop.validator->reads, 1);

    // A later successful check must not inherit the earlier solution's verdict.
    loop.validator->failure = TransferFailure::None;
    std::fill(loop.buffer.begin(), loop.buffer.end(), 0);
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.endIteration();
    loop.listeners.postSolution();
    EXPECT_EQ(loop.listeners.error(), 1);
    ASSERT_EQ(loop.reporter->stringReports.size(), 3u);
    EXPECT_EQ(loop.reporter->stringReports.back().second, "PASSED");
}

INSTANTIATE_TEST_SUITE_P(Transfers,
                        SynchronizerValidatorLifecycle,
                        ::testing::Values(TransferFailure::None,
                                          TransferFailure::Readback,
                                          TransferFailure::Clear));

// Residue left after warmup 0 (by a later warmup or a benchmark launch) fails
// the solution that left it, not the next one.
TEST(SynchronizerValidatorSolutionLoop, LateResidueIsAttributedToItsOwnSolution)
{
    SolutionLoop loop;
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.buffer[2] = 7;
    loop.enqueues();
    loop.endIteration();
    loop.listeners.postSolution();

    EXPECT_EQ(loop.lastVerdict(), "FAILED");
    EXPECT_EQ(loop.listeners.error(), 1);
    EXPECT_EQ(loop.buffer[2], 0u);
    EXPECT_NE(loop.reporter->logged.find("Synchronizer left dirty after run iteration (gemm 0): "
                                         "1/4 ints nonzero, first at int offset 2"),
              std::string::npos);

    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.enqueues();
    loop.endIteration();
    loop.listeners.postSolution();
    EXPECT_EQ(loop.lastVerdict(), "PASSED");
    EXPECT_EQ(loop.listeners.error(), 1);
}

// Guard 1: no readback for a solution that launched nothing (rejected,
// never run, or an iteration without warmups and enqueues), nor for one that
// cannot use the Synchronizer.
TEST(SynchronizerValidatorSolutionLoop, NoCheckWithoutALaunch)
{
    SolutionLoop loop;
    loop.buffer[0] = 1;

    loop.listeners.preSolution(nullptr);
    loop.listeners.postSolution();

    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.endIteration();
    loop.listeners.postSolution();

    TensileLite::ContractionSolution solution;
    setSolution(solution, TileProcessingStrategy::None, 0);
    loop.listeners.preSolution(&solution);
    loop.beginIteration();
    loop.warmup();
    loop.enqueues();
    loop.endIteration();
    loop.listeners.postSolution();

    EXPECT_EQ(loop.validator->reads, 0);
    EXPECT_EQ(loop.listeners.error(), 0);
    EXPECT_EQ(loop.failedVerdicts(), 0);
    EXPECT_EQ(loop.buffer[0], 1u);
}

// An exception ended the iteration before its check (main reports INVALID).
// With a usable device postSolution checks the launches; its FAILED is
// reported after INVALID and the reference verdict, so it is the verdict.
TEST(SynchronizerValidatorSolutionLoop, ResidueAfterAnInvalidLaunchIsFailed)
{
    SolutionLoop loop;
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.buffer[3] = 1;
    loop.listeners.preEnqueues(loop.stream);
    loop.reporter->report(ResultKey::Validation, "INVALID");
    loop.listeners.postSolution();

    EXPECT_EQ(loop.validator->reads, 1);
    ASSERT_EQ(loop.reporter->stringReports.size(), 3u);
    EXPECT_EQ(loop.reporter->stringReports[0].second, "INVALID");
    EXPECT_EQ(loop.lastVerdict(), "FAILED");
    EXPECT_EQ(loop.listeners.error(), 1);
    EXPECT_EQ(loop.buffer[3], 0u);
    EXPECT_NE(loop.reporter->logged.find("after interrupted run iteration"), std::string::npos);

    // A clean buffer adds no verdict to the INVALID one.
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.reporter->report(ResultKey::Validation, "INVALID");
    loop.listeners.postSolution();
    EXPECT_EQ(loop.validator->reads, 2);
    EXPECT_EQ(loop.failedVerdicts(), 1);
    EXPECT_EQ(loop.listeners.error(), 1);
}

// Guard 2: on a lost device the check's own HIP error only fails the solution;
// postSolution never throws.
TEST(SynchronizerValidatorSolutionLoop, DeviceLostFailsWithoutThrowing)
{
    SolutionLoop loop;
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.reporter->report(ResultKey::Validation, "INVALID");
    loop.validator->failure = TransferFailure::Readback;

    EXPECT_NO_THROW(loop.listeners.postSolution());
    EXPECT_EQ(loop.validator->reads, 1);
    EXPECT_EQ(loop.lastVerdict(), "FAILED");
    EXPECT_EQ(loop.listeners.error(), 1);
    EXPECT_NE(loop.reporter->logged.find("Synchronizer check failed: readback failed"),
              std::string::npos);
}

// GuardPageAll runs two iterations, the second one on a different buffer
// (GuardPageBack): each iteration's launches are checked at the buffer they
// used, and the solution gets one verdict.
TEST(SynchronizerValidatorSolutionLoop, GuardPageAllIterationsAreCheckedAtTheirOwnBuffer)
{
    SolutionLoop loop;
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.buffer[0] = 1;
    loop.enqueues();
    loop.endIteration();
    loop.beginIteration(/*guardPageBack=*/true);
    loop.warmup();
    loop.enqueues();
    loop.endIteration();
    loop.listeners.postSolution();

    EXPECT_EQ(loop.validator->reads, 2);
    EXPECT_EQ(loop.failedVerdicts(), 1);
    EXPECT_EQ(loop.lastVerdict(), "FAILED");
    EXPECT_EQ(loop.listeners.error(), 1);
    EXPECT_EQ(loop.buffer[0], 0u);

    // Residue left in the second iteration's buffer.
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.endIteration();
    loop.beginIteration(/*guardPageBack=*/true);
    loop.warmup();
    loop.backBuffer[2] = 1;
    loop.enqueues();
    loop.endIteration();
    loop.listeners.postSolution();

    EXPECT_EQ(loop.validator->reads, 4);
    EXPECT_EQ(loop.failedVerdicts(), 2);
    EXPECT_EQ(loop.listeners.error(), 2);
    EXPECT_EQ(loop.backBuffer[2], 0u);
    EXPECT_NE(loop.reporter->logged.find("first at int offset 2"), std::string::npos);

    // Both iterations dirty: one diagnostic each, still one verdict.
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.buffer[0] = 1;
    loop.endIteration();
    loop.beginIteration(/*guardPageBack=*/true);
    loop.warmup();
    loop.backBuffer[1] = 1;
    loop.endIteration();
    loop.listeners.postSolution();

    EXPECT_EQ(loop.validator->reads, 6);
    EXPECT_EQ(loop.failedVerdicts(), 3);
    EXPECT_EQ(loop.listeners.error(), 3);
    EXPECT_EQ(loop.buffer, std::vector<uint32_t>(4, 0));
    EXPECT_EQ(loop.backBuffer, std::vector<uint32_t>(4, 0));
}

// An exception ends GuardPageAll's second iteration: postSolution checks the
// buffer that iteration used, not the one the validators were given.
TEST(SynchronizerValidatorSolutionLoop, InterruptedSecondIterationIsCheckedAtItsOwnBuffer)
{
    SolutionLoop loop;
    loop.listeners.preSolution(nullptr);
    loop.beginIteration();
    loop.warmup();
    loop.endIteration();
    loop.beginIteration(/*guardPageBack=*/true);
    loop.warmup();
    loop.listeners.preEnqueues(loop.stream);
    loop.backBuffer[1] = 1;
    loop.reporter->report(ResultKey::Validation, "INVALID");
    loop.listeners.postSolution();

    EXPECT_EQ(loop.validator->reads, 2);
    EXPECT_EQ(loop.lastVerdict(), "FAILED");
    EXPECT_EQ(loop.listeners.error(), 1);
    EXPECT_EQ(loop.backBuffer[1], 0u);
}

// Rotating-buffer inputs are built once per problem, so under GuardPageAll
// they keep the first iteration's buffer while the reset inputs move: every
// distinct buffer is read, each once.
TEST(SynchronizerValidatorSolutionLoop, EachDistinctBufferIsReadOnce)
{
    SolutionLoop loop;
    loop.listeners.preSolution(nullptr);
    loop.validator->beginRunIteration({loop.inputs, loop.inputs, loop.inputs});
    loop.warmup();
    loop.endIteration();
    loop.listeners.postSolution();
    EXPECT_EQ(loop.validator->reads, 1);

    loop.listeners.preSolution(nullptr);
    loop.validator->beginRunIteration({loop.backInputs, loop.inputs, loop.inputs});
    loop.warmup();
    loop.buffer[3] = 1;
    loop.endIteration();
    loop.listeners.postSolution();
    EXPECT_EQ(loop.validator->reads, 3);
    EXPECT_EQ(loop.lastVerdict(), "FAILED");
    EXPECT_EQ(loop.buffer[3], 0u);
}
