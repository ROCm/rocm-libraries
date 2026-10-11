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
namespace hipdnn_corpus_gen
{

/// One operation's corpus, and what it cost to find.
struct MetadataOperationCorpus
{
    std::string operation;
    std::string metadataPath;
    ProblemCorpus corpus;

    /// Graphs that could not be built at all: a metadata bug, counted apart from engine refusals.
    int64_t buildFailures = 0;

    /// First build failure seen.
    std::string firstBuildError;
};

/// Where an engine query's time goes.
struct OracleTiming
{
    int64_t queries = 0;
    double buildSeconds = 0.0; ///< declaration -> graph bytes
    double loadSeconds = 0.0; ///< frontend deserialize + build_operation_graph
    double askSeconds = 0.0; ///< get_ranked_engine_ids
};

/// @brief An oracle that asks @p engineId about the graph @p metadata builds for a point.
///
/// @p handle must be live; for the no-engine case use @ref makeCorpusOracle with `nullptr`.
/// The declared checks (builds, fits @p maxBytes) reuse @ref buildAdmissible so this oracle
/// and @ref makeDeclaredOracle cannot disagree.
inline ProblemOracle makeMetadataOracle(hipdnnHandle_t handle,
                                        int64_t engineId,
                                        const OperationMetadata& metadata,
                                        int64_t* buildFailures = nullptr,
                                        std::string* firstBuildError = nullptr,
                                        int64_t maxBytes = 0,
                                        OracleTiming* timing = nullptr)
{
    const BuildTally tally{buildFailures, firstBuildError};
    return [handle, engineId, &metadata, tally, maxBytes, timing](
               const ProblemPoint& point) -> bool {
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
                // Counted as a build failure, not an engine refusal: an unreadable graph is
                // broken for every point.
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
/// A null @p handle means no engine: the corpus is then every point the declaration can
/// express, produced without a device.
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
    return makeMetadataOracle(
        handle, engineId, metadata, buildFailures, firstBuildError, maxBytes, timing);
}

/// Pre-oracle filter, given the operation name and the point. A refused point costs no oracle
/// call and does not count toward a combination's target.
using CorpusFilter = std::function<bool(const std::string&, const ProblemPoint&)>;

/// @brief Generates the problem corpus for @p engineId across every declared operation.
///
/// @p handle may be null; see @ref makeCorpusOracle.
inline std::vector<MetadataOperationCorpus> generateCorpus(hipdnnHandle_t handle,
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
