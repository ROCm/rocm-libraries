// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef TENSILELITE_CLIENT_DATABASE_REPORTER_HPP
#define TENSILELITE_CLIENT_DATABASE_REPORTER_HPP

#include <Tensile/ContractionProblem.hpp>

#include "BenchmarkDatabase.hpp"
#include "ResultReporter.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // Collects the reported values of each problem and solution and stores them in the
        // benchmark database when the caller brackets a pair with beginRun and endRun. A problem
        // starts at its ProblemIndex report; values reported before it, like the problem count
        // and run number, are dropped. Values reported before the first preSolution of a problem
        // are problem values and those after are run values. A problem's values are written at
        // its first beginRun.
        class DatabaseReporter : public ResultReporter
        {
        public:
            explicit DatabaseReporter(std::shared_ptr<BenchmarkDatabase> database);

            // Marks the pair Unknown before its first launch; stores the problem values the
            // first time a problem runs.
            void beginRun(int64_t problemIndex, int64_t solutionIndex);

            // Stores the collected values and iterations and the final status.
            void endRun(int64_t problemIndex, int64_t solutionIndex);

            virtual void reportValue_string(std::string const& key,
                                            std::string const& value) override;
            virtual void reportValue_uint(std::string const& key, uint64_t value) override;
            virtual void reportValue_int(std::string const& key, int64_t value) override;
            virtual void reportValue_double(std::string const& key, double value) override;
            virtual void reportValue_sizes(std::string const&         key,
                                           std::vector<size_t> const& value) override;
            virtual void
                reportValue_vecOfSizes(std::string const&                      key,
                                       std::vector<std::vector<size_t>> const& value) override;

            virtual void preSolution(ContractionSolution* const solution) override;
            virtual void finalizeReport() override;

        private:
            void store(std::string const& key, std::string const& value);
            void storeProblemSizes(std::vector<size_t> const& sizes);

            std::shared_ptr<BenchmarkDatabase> m_database;

            std::map<std::string, std::string> m_problemValues;
            std::map<std::string, std::string> m_runValues;
            std::vector<IterationRecord>       m_iterations;
            int64_t                            m_pendingEnqueues     = 0;
            bool                               m_problemValuesStored = false;
            bool                               m_inSolution          = false;
        };
    }
}

#endif // TENSILELITE_CLIENT_DATABASE_REPORTER_HPP
