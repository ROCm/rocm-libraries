// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ResultReporter.hpp"
#include "SynchronizerValidator.hpp"

#include <Tensile/ContractionSolution.hpp>

#include <memory>
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
        void markDirty()
        {
            m_dirtyInSolution = true;
        }
        // The listener is passive, so the gate is not visible through
        // needMoreRunsInSolution; read it directly.
        bool usesSynchronizer() const
        {
            return m_usesSynchronizer;
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
    void setSolution(TensileLite::ContractionSolution& s,
                     int                               streamK,
                     int                               globalAccumulation,
                     int                               streamKAtomic      = 0,
                     int                               streamKForceDPOnly = 0)
    {
        s.sizeMapping.streamK            = streamK;
        s.sizeMapping.globalAccumulation = globalAccumulation;
        s.sizeMapping.streamKAtomic      = streamKAtomic;
        s.sizeMapping.streamKForceDPOnly = streamKForceDPOnly;
    }

}

// Passive, so it cannot turn a zero-launch codegen config (validate 0, syncs 0)
// into an execution one.
TEST(SynchronizerValidatorReporting, ValidatorNeverDrivesARun)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 3, 0); // a consumer, so this is not the gate talking

    validator.preSolution(&solution);
    ASSERT_TRUE(validator.usesSynchronizer());
    EXPECT_FALSE(validator.needMoreRunsInSolution());
    EXPECT_EQ(validator.numWarmupRuns(), 0u);
}

// Switched off, a consumer solution is still inert.
TEST(SynchronizerValidatorReporting, DisabledValidatorChecksNothing)
{
    TestableSynchronizerValidator    validator(disabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 3, 0);

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.usesSynchronizer());
    EXPECT_FALSE(validator.isActive());
}

// StreamK uses the buffer as its work-queue / fixup Flags.
TEST(SynchronizerValidatorReporting, StreamKSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 3, 0);

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.usesSynchronizer());
}

// GSU MultipleBufferSingleKernel is the other consumer and is not StreamK, so
// gating on StreamK alone would silently drop gsu_mbsk.yaml's coverage.
TEST(SynchronizerValidatorReporting, MbskSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 0, 3);

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.usesSynchronizer());
}

// Unknown solution means unknown answer; scan rather than skip.
TEST(SynchronizerValidatorReporting, UnknownSolutionIsChecked)
{
    TestableSynchronizerValidator validator(enabledArgs());

    validator.preSolution(nullptr);
    EXPECT_TRUE(validator.usesSynchronizer());
}

// amaxD is the third consumer: the dispatcher appends the buffer as AmaxSync
// whenever outputAmaxD is set, independent of streamK and globalAccumulation.
TEST(SynchronizerValidatorReporting, AmaxDSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 0, 0);
    solution.problemType.outputAmaxD = true;

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.usesSynchronizer());
}

// Atomic StreamK reduces in place; the dispatcher never appends Flags for it.
TEST(SynchronizerValidatorReporting, AtomicStreamKSolutionIsSkipped)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 3, 0, /*streamKAtomic=*/1);

    validator.preSolution(&solution);
    EXPECT_FALSE(validator.usesSynchronizer());
}

// StreamKForceDPOnly kernels drop AddressWS/AddressFlags from the SGPR define
// entirely, so the buffer argument the check reads is never passed.
TEST(SynchronizerValidatorReporting, ForceDPOnlyStreamKSolutionIsSkipped)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 3, 0, /*streamKAtomic=*/0, /*streamKForceDPOnly=*/1);

    validator.preSolution(&solution);
    EXPECT_FALSE(validator.usesSynchronizer());
}

// Everything else never receives the buffer, so a scan could only come back
// clean; skipping is what keeps the check free on those runs.
TEST(SynchronizerValidatorReporting, NonConsumerSolutionIsSkipped)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 0, 0);

    validator.preSolution(&solution);
    EXPECT_FALSE(validator.usesSynchronizer());
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
    validator.markDirty();
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
