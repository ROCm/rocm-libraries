// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef TENSILELITE_CLIENT_BENCHMARK_DATABASE_HPP
#define TENSILELITE_CLIENT_BENCHMARK_DATABASE_HPP

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "SqliteDatabase.hpp"

namespace TensileLite
{
    namespace Client
    {
        // Thrown for a misuse of the benchmark database or an inconsistency in it.
        class BenchmarkDatabaseError : public std::runtime_error
        {
        public:
            using std::runtime_error::runtime_error;
        };

        // Thrown when an existing database was created for a different run.
        class BenchmarkDatabaseMismatch : public BenchmarkDatabaseError
        {
        public:
            using BenchmarkDatabaseError::BenchmarkDatabaseError;
        };

        enum class RunStatus : int32_t
        {
            NotBenchmarked = 0,
            Unknown        = 1,
            Pass           = 2,
            Failure        = 3,
            // The solution crashed, faulted or hung on another problem.
            IndirectlyFailed = 4,
            // The solution failed verification on another problem.
            IndirectVerificationFailure = 5,
            // The prediction iterator did not select the pair. It is not pending, and it is
            // neither a failure nor a pass.
            NotSelected = 6,
        };

        // Which failures spread to the remaining pairs of the same solution.
        struct IndirectFailurePolicy
        {
            bool crashes              = true;
            bool verificationFailures = true;
        };

        // A solution is identified by its library index, name and complete serialized spec.
        struct SolutionRecord
        {
            int64_t     libraryIndex;
            std::string name;
            std::string spec;
            std::string specFormat;
        };

        struct InitConfigRecord
        {
            int64_t problemIndex;
        };

        struct IterationRecord
        {
            int64_t syncIndex;
            double  timeUs;
            int64_t enqueues;
        };

        using KeyValues = std::vector<std::pair<std::string, std::string>>;

        // Crash-resilient record of which (init config, solution) pairs have been benchmarked.
        class BenchmarkDatabase
        {
        public:
            // Opens or creates the database. A file without tables is populated with the given
            // solutions, the client configuration shared by all init configs, the init configs
            // and one NotBenchmarked run per pair. An existing one must hold the same solutions
            // (index, name and byte-identical spec), the byte-identical client configuration and
            // the same set of problem indices. A file with tables and another schema version
            // is left unmodified. Both cases throw BenchmarkDatabaseMismatch.
            BenchmarkDatabase(std::string const&                   path,
                              std::vector<SolutionRecord> const&   solutions,
                              std::string const&                   clientConfig,
                              std::vector<InitConfigRecord> const& initConfigs);

            BenchmarkDatabase(BenchmarkDatabase const&)            = delete;
            BenchmarkDatabase& operator=(BenchmarkDatabase const&) = delete;

            void setMeta(std::string const& key, std::string const& value);

            // Marks the remaining pairs of every solution left Unknown by a dead session as
            // IndirectlyFailed, if the policy asks for it. Pairs marked under a disabled kind
            // are reset to NotBenchmarked. The caller must ensure a single writer per database
            // file: a concurrent client would have its Unknown pairs marked as indirectly failed.
            int64_t beginSession(int64_t                     pid,
                                 int64_t                     initSeed,
                                 IndirectFailurePolicy const& policy = IndirectFailurePolicy());
            void    endSession();

            void setProblemValues(int64_t problemIndex, KeyValues const& values);

            // Marks the pair Unknown in a durably committed transaction (synchronous=FULL, so
            // it is flushed to disk) before the run starts. A crash or a node reset before
            // endRun leaves the pair Unknown.
            void beginRun(int64_t problemIndex, int64_t solutionIndex);

            // Records the final status, values and iterations of an Unknown pair atomically.
            // Requires a session. Besides misuse, the only failures are I/O or sqlite errors
            // (for example a full disk); on failure the pair stays Unknown.
            void endRun(int64_t                             problemIndex,
                        int64_t                             solutionIndex,
                        RunStatus                           status,
                        std::string const&                  validation,
                        KeyValues const&                    values,
                        std::vector<IterationRecord> const& iterations);

            // Marks every NotBenchmarked pair of the problem as NotSelected. Call it only after
            // the problem finished: a problem cut short leaves its remaining pairs
            // NotBenchmarked, so that a resumed run selects them again. Requires a session.
            void markNotSelected(int64_t problemIndex);

            bool      isPending(int64_t problemIndex, int64_t solutionIndex) const;
            bool      hasPendingRuns(int64_t problemIndex) const;
            RunStatus runStatus(int64_t problemIndex, int64_t solutionIndex) const;

        private:
            // Mutable because read-only queries still prepare statements.
            mutable SqliteConnection m_db;
            std::optional<int64_t>   m_sessionId;
            IndirectFailurePolicy    m_policy;
            // Set when the previous session died: verification may then fail spuriously.
            bool m_tainted = false;
        };
    }
}

#endif // TENSILELITE_CLIENT_BENCHMARK_DATABASE_HPP
