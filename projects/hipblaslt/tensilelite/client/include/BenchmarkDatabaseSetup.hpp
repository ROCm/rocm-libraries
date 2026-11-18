// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef TENSILELITE_CLIENT_BENCHMARK_DATABASE_SETUP_HPP
#define TENSILELITE_CLIENT_BENCHMARK_DATABASE_SETUP_HPP

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>

#include "BenchmarkDatabase.hpp"
#include "ProgramOptions.hpp"
#include "SolutionIterator.hpp"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // Options that change how a run is recorded or scheduled, not what it computes, and so
        // are left out of the client configuration that a resumed database must match.
        std::set<std::string> optionsExcludedFromDatabaseConfig();

        // A solution serialized in the best format this build can write.
        struct SolutionSpec
        {
            std::string bytes;
            std::string format;
        };

        // The complete solution as msgpack whenever the host can write it, else as YAML.
        SolutionSpec serializeSolution(ContractionSolution const& solution);

        // Throws when `args` holds an option that cannot be combined with `benchmark-db`.
        void checkSupportedOptions(po::variables_map const& args);

        // Stores the options that decide how the results CSV is written from the database.
        void recordCsvMeta(BenchmarkDatabase& database,
                           PerformanceMetric  metric,
                           bool               exportExtraColumns,
                           bool               mergeSameProblems);

        // Opens (creating and populating, or validating) the database named by
        // `benchmark-db`; returns nullptr when the option is empty. `clientConfig` is the
        // serialized client configuration that must match for a database to be resumed.
        std::shared_ptr<BenchmarkDatabase>
            openBenchmarkDatabase(po::variables_map const&                             args,
                                  std::string const&                                   clientConfig,
                                  std::vector<int> const&                              problemIdxs,
                                  MasterSolutionLibrary<ContractionProblemGemm> const& library,
                                  SolutionIterator const&                              solutionIterator);
    }
}

#endif // TENSILELITE_CLIENT_BENCHMARK_DATABASE_SETUP_HPP
