// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ResultReporter.hpp"
#include "SynchronizerValidator.hpp"
#include "TimingEvents.hpp"

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
    };

    po::variables_map enabledArgs()
    {
        po::variables_map vm;
        vm["check-streamk-sync"].value() = true;
        return vm;
    }

    po::variables_map disabledArgs()
    {
        po::variables_map vm;
        vm["check-streamk-sync"].value() = false;
        return vm;
    }

    // Only sizeMapping is read by the consumer gate, so a default-constructed
    // solution with those fields set is enough. ContractionSolution is
    // non-copyable, so this fills one in place rather than returning it.
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

    // validateWarmups ignores the events; TimingEvents(0, 0) creates none, so
    // this needs no GPU.
    void driveWarmup(SynchronizerValidator& validator)
    {
        TimingEvents events(0, 0);
        validator.validateWarmups(nullptr, events, events);
    }
}

// With NumElementsToValidate 0 and SyncsPerBenchmark 0, ReferenceValidator and
// BenchmarkTimer both decline to run the solution. The validator must drive the
// loop itself or its warmup -- and so the whole check -- never happens.
TEST(SynchronizerValidatorReporting, EnabledValidatorRequestsARunInSolution)
{
    TestableSynchronizerValidator validator(enabledArgs());

    validator.preSolution(nullptr);
    EXPECT_TRUE(validator.needMoreRunsInSolution());
}

// One pass only, so it never extends a loop another listener is driving.
TEST(SynchronizerValidatorReporting, RunIsRequestedOncePerSolution)
{
    TestableSynchronizerValidator validator(enabledArgs());

    validator.preSolution(nullptr);
    ASSERT_TRUE(validator.needMoreRunsInSolution());
    driveWarmup(validator);
    EXPECT_FALSE(validator.needMoreRunsInSolution());

    // ...and it asks again for the next solution.
    validator.preSolution(nullptr);
    EXPECT_TRUE(validator.needMoreRunsInSolution());
}

TEST(SynchronizerValidatorReporting, DisabledValidatorRequestsNothing)
{
    TestableSynchronizerValidator validator(disabledArgs());

    validator.preSolution(nullptr);
    EXPECT_FALSE(validator.needMoreRunsInSolution());
    EXPECT_EQ(validator.numWarmupRuns(), 0u);
}

// StreamK uses the buffer as its work-queue / fixup Flags.
TEST(SynchronizerValidatorReporting, StreamKSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 3, 0);

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.needMoreRunsInSolution());
    EXPECT_EQ(validator.numWarmupRuns(), 1u);
}

// GSU MultipleBufferSingleKernel is the other consumer, and it is not StreamK
// -- gating on StreamK alone would drop gsu_mbsk.yaml's coverage silently.
TEST(SynchronizerValidatorReporting, MbskSolutionIsChecked)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 0, 3);

    validator.preSolution(&solution);
    EXPECT_TRUE(validator.needMoreRunsInSolution());
    EXPECT_EQ(validator.numWarmupRuns(), 1u);
}

// Atomic StreamK reduces in place; the dispatcher never appends Flags for it.
TEST(SynchronizerValidatorReporting, AtomicStreamKSolutionIsSkipped)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 3, 0, /*streamKAtomic=*/1);

    validator.preSolution(&solution);
    EXPECT_FALSE(validator.needMoreRunsInSolution());
}

// StreamKForceDPOnly kernels drop AddressWS/AddressFlags from the SGPR define
// entirely, so the buffer argument the check reads is never passed.
TEST(SynchronizerValidatorReporting, ForceDPOnlyStreamKSolutionIsSkipped)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 3, 0, /*streamKAtomic=*/0, /*streamKForceDPOnly=*/1);

    validator.preSolution(&solution);
    EXPECT_FALSE(validator.needMoreRunsInSolution());
}

// Everything else never receives the buffer, so the scan could only come back
// clean. Skipping it is what keeps the check free on those runs.
TEST(SynchronizerValidatorReporting, NonConsumerSolutionIsSkipped)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 0, 0);

    validator.preSolution(&solution);
    EXPECT_FALSE(validator.needMoreRunsInSolution());
    EXPECT_EQ(validator.numWarmupRuns(), 0u);
}

// A skipped solution must not leave the gate closed: numWarmupRuns is consulted
// once per problem before any solution is picked, and a 0 there would size the
// rotating buffers below what a consumer solution in that problem needs.
TEST(SynchronizerValidatorReporting, PreProblemReopensTheGate)
{
    TestableSynchronizerValidator    validator(enabledArgs());
    TensileLite::ContractionSolution solution;
    setSolution(solution, 0, 0);

    validator.preSolution(&solution);
    ASSERT_EQ(validator.numWarmupRuns(), 0u);

    validator.preProblem(nullptr);
    EXPECT_EQ(validator.numWarmupRuns(), 1u);
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
