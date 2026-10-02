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
    // Captures every reportValue_string call; the FAILED verdict is reported
    // through this overload.
    class FakeResultReporter : public ResultReporter
    {
    public:
        std::vector<std::pair<std::string, std::string>> stringReports;

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

    protected:
        uint8_t* readBuffer(void* device, size_t) override
        {
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

    class SynchronizerValidatorLifecycle : public ::testing::TestWithParam<TransferFailure> {};
}

TEST_P(SynchronizerValidatorLifecycle, FailureSurvivesReferenceVerdictAndNextSolutionIsClean)
{
    auto validator = std::make_shared<HostBufferValidator>();
    auto reporter = std::make_shared<FakeResultReporter>();
    MetaRunListener listeners;
    // Match main: postSolution visits these in reverse order.
    listeners.addListener(validator);
    listeners.addListener(std::make_shared<PassingReferenceListener>());
    listeners.setReporter(reporter);

    TensileLite::ContractionProblemGemm problem;
    problem.setSynchronizer(rocisa::DataType::Int32, 4);
    std::vector<uint32_t> buffer{0, 1, 0, 0};
    auto inputs = std::make_shared<TensileLite::ContractionInputs>();
    inputs->Synchronizer = buffer.data();
    TimingEvents events(0, 0);
    listeners.preProblem(&problem);
    listeners.preSolution(nullptr);
    validator->failure = GetParam();

    if(GetParam() == TransferFailure::None)
        EXPECT_NO_THROW(listeners.validateWarmups(inputs, events, events));
    else
        EXPECT_THROW(listeners.validateWarmups(inputs, events, events), std::runtime_error);
    listeners.postSolution();

    EXPECT_EQ(listeners.error(), 1);
    ASSERT_EQ(reporter->stringReports.size(), 2u);
    EXPECT_EQ(reporter->stringReports.back(),
              std::make_pair(std::string(ResultKey::Validation), std::string("FAILED")));
    EXPECT_EQ(buffer[1], GetParam() == TransferFailure::None ? 0u : 1u);

    // A later successful check must not inherit the earlier solution's verdict.
    validator->failure = TransferFailure::None;
    std::fill(buffer.begin(), buffer.end(), 0);
    listeners.preSolution(nullptr);
    listeners.validateWarmups(inputs, events, events);
    listeners.postSolution();
    EXPECT_EQ(listeners.error(), 1);
    ASSERT_EQ(reporter->stringReports.size(), 3u);
    EXPECT_EQ(reporter->stringReports.back().second, "PASSED");
}

INSTANTIATE_TEST_SUITE_P(Transfers,
                        SynchronizerValidatorLifecycle,
                        ::testing::Values(TransferFailure::None,
                                          TransferFailure::Readback,
                                          TransferFailure::Clear));
