// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
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
/// Total bytes the graph's tensors occupy, from the serialized graph itself.
///
/// The benchmarking ceiling §4.3.2 describes: a problem whose tensors do not fit cannot be
/// timed, so it cannot enter a corpus at any budget. Computed rather than declared, because it
/// is a property of the device and the dtype rather than of the operation -- and because no
/// per-dimension window can express it. Without it the search faithfully proposes convolutions
/// that are applicable, enormous, and take minutes each; a corpus of eighty such problems does
/// not finish.
inline int64_t graphBytes(const GraphBytes& bytes)
{
    const auto* graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(bytes.data());
    if(graph == nullptr || graph->tensors() == nullptr)
    {
        return 0;
    }

    int64_t total = 0;
    for(const auto* tensor : *graph->tensors())
    {
        const auto* dims = tensor->dims();
        if(dims == nullptr)
        {
            continue;
        }
        int64_t elements = 1;
        for(const auto dim : *dims)
        {
            if(dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim)
            {
                return std::numeric_limits<int64_t>::max();
            }
            elements *= dim;
        }
        // Four bytes is an upper bound for every element type a corpus currently uses and an
        // underestimate for none of them below fp64, which keeps this a ceiling rather than a
        // guess that could let a too-large problem through.
        total += elements * 4;
    }
    return total;
}

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

/// Every `*.opmeta.json` in @p directory, parsed. Files that fail validation are reported
/// rather than skipped: a declaration that does not load is a hole in the corpus.
struct MetadataSet
{
    std::vector<std::pair<std::string, OperationMetadata>> operations;
    std::vector<std::string> errors;
};

inline MetadataSet loadOperationDirectory(const std::filesystem::path& directory)
{
    MetadataSet set;
    if(!std::filesystem::is_directory(directory))
    {
        set.errors.push_back("not a directory: " + directory.string());
        return set;
    }

    // Sorted, so a corpus generated twice visits its operations in one order.
    std::vector<std::filesystem::path> files;
    for(const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if(entry.path().extension() == ".json"
           && entry.path().string().find(".opmeta.") != std::string::npos)
        {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for(const auto& file : files)
    {
        std::ifstream stream(file);
        if(!stream)
        {
            set.errors.push_back("cannot read " + file.string());
            continue;
        }

        try
        {
            auto parsed = parseOperationMetadata(nlohmann::json::parse(stream));
            if(!parsed.ok())
            {
                for(const auto& error : parsed.errors)
                {
                    set.errors.push_back(file.filename().string() + ": " + error);
                }
                continue;
            }
            set.operations.emplace_back(file.string(), std::move(*parsed.metadata));
        }
        catch(const std::exception& error)
        {
            set.errors.push_back(file.filename().string() + ": " + error.what());
        }
    }
    return set;
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
