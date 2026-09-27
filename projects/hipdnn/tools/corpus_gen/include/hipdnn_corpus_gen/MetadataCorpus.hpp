// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/DeclaredOracle.hpp>
#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
#include <hipdnn_corpus_gen/GraphSize.hpp>
#include <hipdnn_corpus_gen/OperationDirectory.hpp>
#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <hipdnn_frontend.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

/// @file MetadataCorpus.hpp
/// @brief Generating an engine's problems from declarations alone (RFC 0019.13 §4, §5).
///
/// The whole path, with nothing hand-written per problem: a metadata file declares an
/// operation's parameters and how they build; the exploration walks that space once, the same
/// way for every operation; each candidate is built into a graph and offered to the engine;
/// what the engine accepts is the corpus.
///
/// Presenting *one* problem is not the deliverable -- anybody can write a graph by hand. The
/// deliverable is that the problems are produced automatically across the range the engine
/// serves, which is why the only inputs here are a directory of declarations and an engine.
namespace hipdnn_corpus_gen
{

/// One operation's corpus, and what it cost to find.
struct MetadataOperationCorpus
{
    std::string operation;
    std::string metadataPath;
    ProblemCorpus corpus;

    /// Problems whose graph could not be built at all -- a metadata bug rather than an engine
    /// refusal, and counted separately so the two are never confused.
    int64_t buildFailures = 0;

    /// First build failure seen, since one message is worth more than a count.
    std::string firstBuildError;
};

/// Where an engine query's time goes. Applicability is a yes/no question and should be cheap;
/// these say which stage is not.
struct OracleTiming
{
    int64_t queries = 0;
    double buildSeconds = 0.0; ///< declaration -> graph bytes
    double loadSeconds = 0.0; ///< frontend deserialize + build_operation_graph
    double askSeconds = 0.0; ///< get_ranked_engine_ids
};

/// @brief An oracle that asks @p engineId about the graph @p metadata builds for a point.
///
/// The one place a declaration meets a live engine. Everything upstream is data; everything
/// downstream is a measurement.
///
/// @p handle must be a live handle; pass `nullptr` to @ref makeCorpusOracle instead, which is
/// the branch that names no engine.
///
/// The declared half -- does it build, does it fit @p maxBytes -- is @ref buildAdmissible, not
/// a copy of it, so this oracle and the device-free @ref makeDeclaredOracle cannot drift into
/// disagreeing about what a declaration can express.
inline ProblemOracle makeMetadataOracle(hipdnnHandle_t handle,
                                        int64_t engineId,
                                        const OperationMetadata& metadata,
                                        int64_t* buildFailures = nullptr,
                                        std::string* firstBuildError = nullptr,
                                        int64_t maxBytes = 0,
                                        OracleTiming* timing = nullptr)
{
    const BuildTally tally{buildFailures, firstBuildError};
    return [handle, engineId, &metadata, tally, maxBytes, timing](const ProblemPoint& point) -> bool {
        using Clock = std::chrono::steady_clock;
        auto mark = Clock::now();
        const auto lap = [&mark](double OracleTiming::*stage, OracleTiming* into) {
            const auto now = Clock::now();
            if(into != nullptr)
            {
                into->*stage += std::chrono::duration<double>(now - mark).count();
            }
            mark = now;
        };
        if(timing != nullptr)
        {
            ++timing->queries;
        }
        const auto built = buildAdmissible(metadata, point, maxBytes, tally);
        lap(&OracleTiming::buildSeconds, timing);
        if(!built.has_value())
        {
            return false;
        }

        try
        {
            hipdnn_frontend::graph::Graph graph;
            const auto restored = graph.deserialize(handle, *built);
            if(!restored.is_good())
            {
                // Distinct from an engine refusal for the same reason a build failure is: a
                // graph the frontend will not read is broken for every point, and folding it
                // into "declined" reports an engine that serves nothing.
                tally.note("deserialize: " + restored.get_message());
                return false;
            }

            const auto finalized = graph.build_operation_graph(handle);
            lap(&OracleTiming::loadSeconds, timing);
            if(!finalized.is_good())
            {
                tally.note("build_operation_graph: " + finalized.get_message());
                return false;
            }

            std::vector<int64_t> applicable;
            const auto asked = graph.get_ranked_engine_ids(applicable);
            lap(&OracleTiming::askSeconds, timing);
            if(!asked.is_good())
            {
                return false;
            }
            return std::find(applicable.begin(), applicable.end(), engineId) != applicable.end();
        }
        catch(...)
        {
            return false;
        }
    };
}

/// @brief The oracle for a run, which may or may not have named an engine.
///
/// A null @p handle is the no-engine case, not an error: the corpus is then every point the
/// declaration can express and benchmark. That is what a deterministic engine needs measured,
/// and it is produced without a device -- so naming an engine narrows a corpus rather than
/// enabling one.
inline ProblemOracle makeCorpusOracle(hipdnnHandle_t handle,
                                      int64_t engineId,
                                      const OperationMetadata& metadata,
                                      int64_t* buildFailures = nullptr,
                                      std::string* firstBuildError = nullptr,
                                      int64_t maxBytes = 0,
                                      OracleTiming* timing = nullptr)
{
    if(handle == nullptr)
    {
        return makeDeclaredOracle(metadata, buildFailures, firstBuildError, maxBytes);
    }
    return makeMetadataOracle(handle, engineId, metadata, buildFailures, firstBuildError,
                              maxBytes, timing);
}

/// @brief Generates the problem corpus for @p engineId across every declared operation.
///
/// This is requirement 3: not one problem, but the range, produced without anyone writing a
/// problem down. An operation the engine declines contributes nothing and says so.
///
/// @p handle may be null; see @ref makeCorpusOracle.
///
/// @p keep, when set, is asked about a point before the engine is: a point it refuses never
/// costs an oracle call, and never counts toward a combination's target. Applied afterwards
/// instead, a filter spends the search on points it then discards and the corpus comes back
/// short by exactly the filtered fraction. It receives the operation's name alongside the point.
using CorpusFilter = std::function<bool(const std::string&, const ProblemPoint&)>;

inline std::vector<MetadataOperationCorpus>
    generateCorpus(hipdnnHandle_t handle,
                   int64_t engineId,
                   const MetadataSet& declarations,
                   const ExplorationRequest& request,
                   int64_t maxBytes = 0,
                   const CorpusFilter& keep = {})
{
    std::vector<MetadataOperationCorpus> results;

    for(const auto& entry : declarations.operations)
    {
        MetadataOperationCorpus result;
        result.metadataPath = entry.first;
        result.operation = entry.second.operation;

        const auto oracle = makeCorpusOracle(handle,
                                             engineId,
                                             entry.second,
                                             &result.buildFailures,
                                             &result.firstBuildError,
                                             maxBytes);

        const auto& operation = entry.second.operation;
        const ProblemOracle admits = [&](const ProblemPoint& point) {
            return (!keep || keep(operation, point)) && oracle(point);
        };

        result.corpus = exploreProblemSpace(entry.second, request, admits);
        results.push_back(std::move(result));
    }
    return results;
}

} // namespace hipdnn_corpus_gen
