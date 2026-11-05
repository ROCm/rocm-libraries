// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <hipdnn_frontend/Error.hpp>

#include <nlohmann/json.hpp>

#include "harness/bundle/IGraphEngineRunner.hpp"
#include "harness/bundle/LoadedEngine.hpp"

namespace hipdnn_integration_tests::bundle
{

struct IntegrationTestBundle;

/// The most candidates one page of the engine's catalog holds
/// (hipdnn_frontend::detail::getEngineCandidates accepts a limit in [1, 10000]).
inline constexpr int64_t CANDIDATE_PAGE_LIMIT = 10000;

/// What listing one case's candidates came to.
enum class ListingStatus
{
    OK, ///< the engine enumerated its catalog; every page was read
    NOT_ENUMERABLE, ///< the engine cannot enumerate (or does not take the graph): the catalog is
    ///< one entry, its heuristic pick
    LISTING_ERROR, ///< the graph did not load or the enumeration broke; nothing can be said of the catalog
};

const char* toString(ListingStatus status);

/// One catalog candidate: the id the engine gave it and its complete knob tuple, a JSON object
/// of knob name to value that `metadata.engine_variant.knobs` takes back verbatim.
struct ListedCandidate
{
    std::string id;
    nlohmann::json knobs = nlohmann::json::object();
};

/// The candidate list of one case on the engine under test.
struct CaseListing
{
    std::string test; ///< "Suite.Test", the name --gtest_filter selects on
    std::string bundle;
    std::string engine;
    ListingStatus status = ListingStatus::LISTING_ERROR;
    /// The catalog size the engine reported; equal to candidates.size() for an OK listing.
    uint64_t total = 0;
    std::vector<ListedCandidate> candidates;
    /// Why the listing is not OK. Empty for OK.
    std::string reason;
};

/// A case the run would execute, as the listing needs it.
struct ListingCase
{
    std::string test;
    std::string bundle;
    /// Null when the bundle failed to load; `loadError` then says why.
    std::shared_ptr<const IntegrationTestBundle> loaded;
    std::string loadError;
};

/// Whether `error` is the engine saying it cannot enumerate its candidates: a plugin answers
/// NOT_APPLICABLE, which the frontend reports as "Engine does not support matched-catalog
/// enumeration". Anything else is the enumeration breaking.
bool isEnumerationUnsupported(const hipdnn_frontend::Error& error);

/// Lists every candidate `engine` can build for `listed`'s graph, walking every page of the
/// catalog. Builds and executes nothing. Never throws: every failure is a status.
CaseListing listCaseCandidates(IGraphEngineRunner& runner,
                               const ListingCase& listed,
                               const LoadedEngine& engine);

/// One JSON object, no trailing newline:
/// `{"test", "bundle", "engine", "status", "total", "candidates": [{"id", "knobs"}], "reason"}`.
/// `reason` is empty for an OK listing.
std::string listingRecordJson(const CaseListing& listing);

/// Writes one line per case in `cases`, in order, flushed per line, and returns how many of the
/// listings ended in LISTING_ERROR.
size_t writeCandidateListing(std::ostream& out,
                             IGraphEngineRunner& runner,
                             const std::vector<ListingCase>& cases,
                             const LoadedEngine& engine);

} // namespace hipdnn_integration_tests::bundle
