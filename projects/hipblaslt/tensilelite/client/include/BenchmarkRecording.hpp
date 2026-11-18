// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef TENSILELITE_CLIENT_BENCHMARK_RECORDING_HPP
#define TENSILELITE_CLIENT_BENCHMARK_RECORDING_HPP

#include "BenchmarkDatabase.hpp"
#include "DatabaseReporter.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

namespace TensileLite
{
    namespace Client
    {
        // Thrown when the device became unusable after a kernel fault. It carries the exit code
        // the client would have returned.
        class DeviceLostError : public std::runtime_error
        {
        public:
            explicit DeviceLostError(int32_t exitCode);

            int32_t exitCode() const;

        private:
            int32_t m_exitCode;
        };

        // Scope of a database session. end() is explicit so that a failure propagates; the
        // destructor ends the session only on a normal scope exit, so unwinding leaves it open
        // for a rerun to see as a crash.
        class BenchmarkSession
        {
        public:
            BenchmarkSession(std::shared_ptr<BenchmarkDatabase> database,
                             int64_t                            pid,
                             int64_t                            seed,
                             IndirectFailurePolicy              policy);
            ~BenchmarkSession() noexcept;

            BenchmarkSession(BenchmarkSession const&)            = delete;
            BenchmarkSession& operator=(BenchmarkSession const&) = delete;
            BenchmarkSession(BenchmarkSession&&)                 = delete;
            BenchmarkSession& operator=(BenchmarkSession&&)      = delete;

            // Ends the session once; later calls do nothing. May throw.
            void end();

        private:
            std::shared_ptr<BenchmarkDatabase> m_database;
            int32_t                            m_uncaughtExceptions;
            bool                               m_ended = false;
        };

        enum class PairRecording
        {
            // No database.
            Disabled,
            // The database already holds the pair, so it is not run.
            SkippedByDatabase,
            // The pair is pending and its run is recorded.
            Record,
        };

        PairRecording classifyPair(BenchmarkDatabase const* database,
                                   int64_t                  problemIndex,
                                   int64_t                  solutionIndex);

        // Brackets the recording of one pair. An unfinished pair (an exception between the
        // construction and finish) simply stays Unknown, as after a crash.
        class PairRecorder
        {
        public:
            // Commits the Unknown marker before the launch.
            PairRecorder(DatabaseReporter& reporter, int64_t problemIndex, int64_t solutionIndex);

            PairRecorder(PairRecorder const&)            = delete;
            PairRecorder& operator=(PairRecorder const&) = delete;

            // Records the verdict of the pair.
            void finish();

        private:
            DatabaseReporter& m_reporter;
            int64_t           m_problemIndex;
            int64_t           m_solutionIndex;
        };

        // On Linux, asks the kernel to SIGKILL this process when its parent thread exits. Returns
        // false if the parent had already exited before the request took effect (a parent that
        // died before this process started cannot be detected) and true otherwise, always true
        // off Linux. Throws if the request fails.
        bool dieWithParent();

        // Handles the errors thrown once the database is open: a failing end() or endRun leaves
        // the session open and the pair Unknown for a rerun. Open and mismatch errors are
        // reported earlier by main, and a failing beginSession leaves no session. Other
        // exceptions propagate unchanged.
        int32_t runCatchingBenchmarkFailures(std::function<int32_t()> const& client);
    }
}

#endif // TENSILELITE_CLIENT_BENCHMARK_RECORDING_HPP
