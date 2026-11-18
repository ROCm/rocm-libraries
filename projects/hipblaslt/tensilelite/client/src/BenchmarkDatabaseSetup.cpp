// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BenchmarkDatabaseSetup.hpp"

#include <Tensile/FusedA2A.hpp>

#ifdef TENSILELITE_HAVE_MSGPACK_WRITER
#include <Tensile/msgpack/Writing.hpp>
#elif defined(TENSILE_YAML)
#include <Tensile/llvm/Writing.hpp>
#endif

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

std::set<std::string> Client::optionsExcludedFromDatabaseConfig()
{
    return {"help",
            "config-file",
            "benchmark-db",
            "indirect-failures",
            "indirect-verification-failures"};
}

SolutionSpec Client::serializeSolution([[maybe_unused]] ContractionSolution const& solution)
{
#ifdef TENSILELITE_HAVE_MSGPACK_WRITER
    std::vector<uint8_t> bytes = MessagePackWriteSolutionData(solution);
    return {std::string(bytes.begin(), bytes.end()), "msgpack"};
#elif defined(TENSILE_YAML)
    return {LLVMWriteSolutionData(solution), "yaml"};
#else
    throw std::runtime_error("benchmark-db needs a build that can serialize solutions");
#endif
}

// Rejects option combinations that the database mode does not support, before any database
// file is created.
void Client::checkSupportedOptions(po::variables_map const& args)
{
#if !defined(TENSILELITE_HAVE_MSGPACK_WRITER) && !defined(TENSILE_YAML)
    // A client built against an installed host has no solution writer, so fail before the
    // database file is created.
    throw std::runtime_error(
        "benchmark-db needs a client built with the msgpack or YAML solution writer");
#endif
    if(args["num-benchmarks"].as<int>() > 1)
        throw std::runtime_error("benchmark-db cannot be combined with num-benchmarks greater than 1");
    if(args["fused-a2a"].as<FusedA2A>() != FusedA2A::Disabled)
        throw std::runtime_error("benchmark-db cannot be combined with fused-a2a");
    if(args["selection-only"].as<bool>())
        throw std::runtime_error("benchmark-db cannot be combined with selection-only");
    if(args["icache-flush-args"].as<std::vector<bool>>().size() > 1)
        throw std::runtime_error("benchmark-db supports a single icache-flush-args value");
}

static std::vector<SolutionRecord>
    makeSolutionRecords(MasterSolutionLibrary<ContractionProblemGemm> const& library,
                        AllSolutionsIterator const&                          iterator)
{
    std::vector<SolutionRecord> records;
    for(auto const& [index, solution] : library.solutions)
    {
        if(index < iterator.firstSolutionIdx() || index > iterator.lastSolutionIdx())
            continue;
        SolutionSpec spec = serializeSolution(*solution);
        records.push_back(
            SolutionRecord{index, solution->name(), std::move(spec.bytes), std::move(spec.format)});
    }
    return records;
}

void Client::recordCsvMeta(BenchmarkDatabase& database,
                           PerformanceMetric  metric,
                           bool               exportExtraColumns,
                           bool               mergeSameProblems)
{
    std::ostringstream metricText;
    metricText << metric;
    database.setMeta("performance_metric", metricText.str());
    database.setMeta("csv_export_extra_cols", exportExtraColumns ? "true" : "false");
    database.setMeta("csv_merge_same_problems", mergeSameProblems ? "true" : "false");
}

std::shared_ptr<BenchmarkDatabase>
    Client::openBenchmarkDatabase(po::variables_map const&                             args,
                                  std::string const&                                   clientConfig,
                                  std::vector<int> const&                              problemIdxs,
                                  MasterSolutionLibrary<ContractionProblemGemm> const& library,
                                  SolutionIterator const&                              solutionIterator)
{
    std::string path = args["benchmark-db"].as<std::string>();
    if(path.empty())
        return nullptr;

    checkSupportedOptions(args);

    auto const* allSolutions = dynamic_cast<AllSolutionsIterator const*>(&solutionIterator);
    if(!allSolutions)
        throw std::runtime_error("benchmark-db needs the all-solutions run, not best-solution");

    std::vector<InitConfigRecord> configs;
    for(int problemIdx : problemIdxs)
        configs.push_back(InitConfigRecord{problemIdx});

    std::shared_ptr<BenchmarkDatabase> database = std::make_shared<BenchmarkDatabase>(
        path, makeSolutionRecords(library, *allSolutions), clientConfig, configs);

    recordCsvMeta(*database,
                  args["performance-metric"].as<PerformanceMetric>(),
                  args["csv-export-extra-cols"].as<bool>(),
                  args["csv-merge-same-problems"].as<bool>());
    return database;
}
