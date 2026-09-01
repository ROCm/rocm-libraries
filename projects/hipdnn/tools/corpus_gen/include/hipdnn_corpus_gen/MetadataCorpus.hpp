// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
#include <hipdnn_corpus_gen/GraphSize.hpp>
#include <hipdnn_corpus_gen/OperationDirectory.hpp>
#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <hipdnn_frontend.hpp>

#include <algorithm>
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

/// @brief An oracle that asks @p engineId about the graph @p metadata builds for a point.
///
/// The one place a declaration meets a live engine. Everything upstream is data; everything
/// downstream is a measurement.
///
/// @p maxBytes is the benchmarking ceiling (see GraphSize.hpp): a problem whose tensors do not
/// fit cannot be timed, so it cannot enter a corpus at any budget.
inline ProblemOracle makeMetadataOracle(hipdnnHandle_t handle,
                                        int64_t engineId,
                                        const OperationMetadata& metadata,
                                        int64_t* buildFailures = nullptr,
                                        std::string* firstBuildError = nullptr,
                                        int64_t maxBytes = 0)
{
    return [handle, engineId, &metadata, buildFailures, firstBuildError, maxBytes](
               const ProblemPoint& point) -> bool {
        const auto built = buildGraphFor(metadata, point);
        if(!built.ok())
        {
            // Distinguished from an engine refusal on purpose. A declaration that cannot build
            // is broken for every point, and would otherwise read as an engine that serves
            // almost nothing -- the search would report a tiny region in good faith.
            if(buildFailures != nullptr)
            {
                ++*buildFailures;
            }
            if(firstBuildError != nullptr && firstBuildError->empty())
            {
                *firstBuildError = built.error;
            }
            return false;
        }

        if(maxBytes > 0 && graphBytes(built.bytes) > maxBytes)
        {
            // Not a refusal by the engine and not a broken declaration: a problem too large to
            // benchmark. Silent, because it is neither party's fault and counting it would
            // drown the counts that mean something.
            return false;
        }

        try
        {
            hipdnn_frontend::graph::Graph graph;
            const auto restored = graph.deserialize(handle, built.bytes);
            if(!restored.is_good())
            {
                // Distinct from an engine refusal for the same reason a build failure is: a
                // graph the frontend will not read is broken for every point, and folding it
                // into "declined" reports an engine that serves nothing.
                if(buildFailures != nullptr)
                {
                    ++*buildFailures;
                }
                if(firstBuildError != nullptr && firstBuildError->empty())
                {
                    *firstBuildError = "deserialize: " + restored.get_message();
                }
                return false;
            }

            const auto finalized = graph.build_operation_graph(handle);
            if(!finalized.is_good())
            {
                if(buildFailures != nullptr)
                {
                    ++*buildFailures;
                }
                if(firstBuildError != nullptr && firstBuildError->empty())
                {
                    *firstBuildError = "build_operation_graph: " + finalized.get_message();
                }
                return false;
            }

            std::vector<int64_t> applicable;
            if(!graph.get_ranked_engine_ids(applicable).is_good())
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

/// @brief Generates the problem corpus for @p engineId across every declared operation.
///
/// This is requirement 3: not one problem, but the range, produced without anyone writing a
/// problem down. An operation the engine declines contributes nothing and says so.
inline std::vector<MetadataOperationCorpus>
    generateCorpus(hipdnnHandle_t handle,
                   int64_t engineId,
                   const MetadataSet& declarations,
                   const ExplorationRequest& request,
                   int64_t maxBytes = 0)
{
    std::vector<MetadataOperationCorpus> results;

    for(const auto& entry : declarations.operations)
    {
        MetadataOperationCorpus result;
        result.metadataPath = entry.first;
        result.operation = entry.second.operation;

        const auto oracle = makeMetadataOracle(handle,
                                               engineId,
                                               entry.second,
                                               &result.buildFailures,
                                               &result.firstBuildError,
                                               maxBytes);

        result.corpus = exploreProblemSpace(entry.second, request, oracle);
        results.push_back(std::move(result));
    }
    return results;
}

} // namespace hipdnn_corpus_gen
