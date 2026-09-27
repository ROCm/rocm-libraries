// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
#include <hipdnn_corpus_gen/GraphSize.hpp>
#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <cstdint>
#include <optional>
#include <string>

/// @file DeclaredOracle.hpp
/// @brief What a declaration alone can decide about a problem, with no device in the room.
///
/// Two questions are answered here and nowhere else: does this point build into a graph, and
/// does that graph fit the benchmarking ceiling. Neither needs a handle, so neither belongs in
/// `MetadataCorpus.hpp` -- this header lives in the `hipdnn_corpus_gen` INTERFACE library,
/// which links no backend and is tested with no GPU.
///
/// `makeMetadataOracle` is this plus one more question (does the named engine accept the
/// graph). It is written as this plus that question rather than as a second copy of it: a
/// corpus built with an engine and a corpus built without one must agree about what a
/// declaration can express, and they only agree for certain if it is the same code deciding.
namespace hipdnn_corpus_gen
{

/// @brief Where a build failure is recorded, when anyone is counting.
///
/// A build failure is not an engine refusal and must never be reported as one: a declaration
/// that cannot build is broken for *every* point, so folding the two together reports an
/// engine that serves almost nothing, in good faith and wrongly. Both members may be null --
/// an exploration that does not care still runs the same admission.
struct BuildTally
{
    int64_t* failures = nullptr;

    /// First failure seen, since one message is worth more than a count.
    std::string* firstError = nullptr;

    void note(const std::string& message) const
    {
        if(failures != nullptr)
        {
            ++*failures;
        }
        if(firstError != nullptr && firstError->empty())
        {
            *firstError = message;
        }
    }
};

/// @brief The graph @p metadata builds for @p point, if a declaration alone admits it.
///
/// @p maxBytes is the benchmarking ceiling (see GraphSize.hpp): a problem whose tensors do not
/// fit cannot be timed, so it cannot enter a corpus at any budget. Zero disables the ceiling.
///
/// An oversized graph is refused silently, unlike a build failure: it is neither the
/// declaration's fault nor an engine's, and counting it would drown the counts that mean
/// something.
inline std::optional<GraphBytes> buildAdmissible(const OperationMetadata& metadata,
                                                 const ProblemPoint& point,
                                                 int64_t maxBytes    = 0,
                                                 const BuildTally& tally = {})
{
    const auto built = buildGraphFor(metadata, point);
    if(!built.ok())
    {
        tally.note(built.error);
        return std::nullopt;
    }

    if(maxBytes > 0 && graphBytes(built.bytes) > maxBytes)
    {
        return std::nullopt;
    }
    return built.bytes;
}

/// @brief An oracle that asks only what @p metadata declares.
///
/// The corpus this produces is every point the operation can express and benchmark, which is
/// what a deterministic engine needs measured and what any engine can be offered later. It is
/// also the only oracle available when no engine was named -- and naming one is optional
/// precisely because this exists.
inline ProblemOracle makeDeclaredOracle(const OperationMetadata& metadata,
                                        int64_t* buildFailures     = nullptr,
                                        std::string* firstBuildError = nullptr,
                                        int64_t maxBytes           = 0)
{
    const BuildTally tally{buildFailures, firstBuildError};
    return [&metadata, tally, maxBytes](const ProblemPoint& point) -> bool {
        return buildAdmissible(metadata, point, maxBytes, tally).has_value();
    };
}

} // namespace hipdnn_corpus_gen
