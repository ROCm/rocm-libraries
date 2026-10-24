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
/// @brief What a declaration alone can decide about a problem, without a device.
///
/// Device-free (INTERFACE library, no backend); `makeMetadataOracle` reuses this so corpora
/// built with and without an engine agree on what a declaration can express.
namespace hipdnn_corpus_gen
{

/// @brief Where a build failure is recorded, when anyone is counting.
///
/// Build failures are counted apart from engine refusals: a declaration that cannot build is
/// broken for every point. Both members may be null.
struct BuildTally
{
    int64_t* failures = nullptr;

    /// First failure seen.
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
/// @p maxBytes is the benchmarking ceiling (see GraphSize.hpp); zero disables it. An oversized
/// graph is refused without counting as a build failure.
inline std::optional<GraphBytes> buildAdmissible(const OperationMetadata& metadata,
                                                 const ProblemPoint& point,
                                                 int64_t maxBytes = 0,
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
/// Admits every point the operation can express and benchmark; used when no engine is named.
inline ProblemOracle makeDeclaredOracle(const OperationMetadata& metadata,
                                        int64_t* buildFailures = nullptr,
                                        std::string* firstBuildError = nullptr,
                                        int64_t maxBytes = 0)
{
    const BuildTally tally{buildFailures, firstBuildError};
    return [&metadata, tally, maxBytes](const ProblemPoint& point) -> bool {
        return buildAdmissible(metadata, point, maxBytes, tally).has_value();
    };
}

} // namespace hipdnn_corpus_gen
