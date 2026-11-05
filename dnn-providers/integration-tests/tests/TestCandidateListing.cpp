// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// --list-candidates: the candidate catalog of the engine under test, one JSON line per case.
// Driven through the runner seam with a mocked enumeration, so no device, handle or engine
// plugin is involved and nothing is ever built or executed.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_frontend/EngineQueryTypes.hpp>
#include <hipdnn_frontend/Error.hpp>

#include <nlohmann/json.hpp>

#include "HarnessTestSupport.hpp"
#include "harness/BundleMetadata.hpp"
#include "harness/EngineNotApplicableError.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/bundle/CandidateListing.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "mocks/MockGraphEngineRunner.hpp"

using hipdnn_frontend::EngineCandidate;
using hipdnn_frontend::EngineCandidatePage;
using hipdnn_frontend::Error;
using hipdnn_frontend::ErrorCode;
using hipdnn_integration_tests::EngineNotApplicableError;
using namespace hipdnn_integration_tests::bundle;
using namespace hipdnn_integration_tests::bundle::testing_support;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

using ::testing::_;

const LoadedEngine ENGINE{7, "ENGINE_UNDER_TEST"};

const char* const UNSUPPORTED_MESSAGE = "Engine does not support matched-catalog enumeration";

/// What an engine with `total` candidates answers for `limit` entries from `offset`: candidate
/// i is "c<i>" with the one integer knob HIPDNN_KNOB_INDEX = i.
EngineCandidatePage servePage(uint64_t total, int64_t offset, int64_t limit)
{
    EngineCandidatePage page;
    page.engineId = ENGINE.id;
    page.totalCount = total;
    page.offset = static_cast<uint64_t>(offset);
    const auto end = std::min<uint64_t>(total, static_cast<uint64_t>(offset + limit));
    for(auto i = static_cast<uint64_t>(offset); i < end; ++i)
    {
        EngineCandidate candidate;
        candidate.id = "c" + std::to_string(i);
        candidate.variant.engineId = ENGINE.id;
        candidate.variant.knobSettings.emplace("HIPDNN_KNOB_INDEX", static_cast<int64_t>(i));
        page.candidates.push_back(std::move(candidate));
    }
    if(end < total)
    {
        page.nextOffset = end;
    }
    return page;
}

ListingCase loadedCase(const std::string& test = "Suite.case_0",
                       const std::string& bundle = "dir/sweep.json#case_0")
{
    return {test, bundle, std::make_shared<IntegrationTestBundle>(), std::string()};
}

} // namespace

class TestCandidateListing : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ON_CALL(_runner, openGraph(_, _))
            .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
                return acceptedSession();
            });
    }

    /// The engine enumerates `total` candidates; every request it gets is kept.
    void enumerates(uint64_t total)
    {
        ON_CALL(_runner, candidatePage(_, _, _, _, _))
            .WillByDefault([this, total](GraphSession&,
                                         const LoadedEngine&,
                                         int64_t offset,
                                         int64_t limit,
                                         EngineCandidatePage& page) {
                _requests.emplace_back(offset, limit);
                page = servePage(total, offset, limit);
                return Error{};
            });
    }

    ::testing::NiceMock<MockGraphEngineRunner> _runner;
    std::vector<std::pair<int64_t, int64_t>> _requests;
};

// Bug caught: a listing that stops after the first page, which silently drops every candidate
// past the page size from the sweep.
TEST_F(TestCandidateListing, EveryPageOfTheCatalogIsWalked)
{
    constexpr uint64_t TOTAL = 2 * static_cast<uint64_t>(CANDIDATE_PAGE_LIMIT) + 5;
    enumerates(TOTAL);

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    ASSERT_EQ(listing.status, ListingStatus::OK) << listing.reason;
    EXPECT_EQ(listing.total, TOTAL);
    ASSERT_EQ(listing.candidates.size(), TOTAL);
    const std::vector<std::pair<int64_t, int64_t>> expectedRequests{
        {0, CANDIDATE_PAGE_LIMIT},
        {CANDIDATE_PAGE_LIMIT, CANDIDATE_PAGE_LIMIT},
        {2 * CANDIDATE_PAGE_LIMIT, CANDIDATE_PAGE_LIMIT}};
    EXPECT_EQ(_requests, expectedRequests);

    // In catalog order, each with its own knobs, across the page seams and at the end.
    for(const auto index : {uint64_t{0},
                            static_cast<uint64_t>(CANDIDATE_PAGE_LIMIT) - 1,
                            static_cast<uint64_t>(CANDIDATE_PAGE_LIMIT),
                            TOTAL - 1})
    {
        EXPECT_EQ(listing.candidates[index].id, "c" + std::to_string(index));
        EXPECT_EQ(listing.candidates[index].knobs,
                  (nlohmann::json{{"HIPDNN_KNOB_INDEX", static_cast<int64_t>(index)}}));
    }
    EXPECT_EQ(listing.engine, ENGINE.name);
    EXPECT_EQ(listing.test, "Suite.case_0");
    EXPECT_EQ(listing.bundle, "dir/sweep.json#case_0");
}

// Bug caught: an off-by-one at the page boundary reading a nonexistent extra page, or none.
TEST_F(TestCandidateListing, ACatalogOfExactlyOnePageIsOnePage)
{
    enumerates(static_cast<uint64_t>(CANDIDATE_PAGE_LIMIT));

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    ASSERT_EQ(listing.status, ListingStatus::OK) << listing.reason;
    EXPECT_EQ(listing.candidates.size(), static_cast<size_t>(CANDIDATE_PAGE_LIMIT));
    EXPECT_EQ(_requests.size(), 1u);
}

// An engine that enumerates and has nothing for this graph is a real, empty catalog.
TEST_F(TestCandidateListing, AnEmptyCatalogIsOkAndEmpty)
{
    enumerates(0);

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::OK) << listing.reason;
    EXPECT_EQ(listing.total, 0u);
    EXPECT_TRUE(listing.candidates.empty());
}

// Bug caught: an engine that cannot enumerate reported as an error (the whole sweep would fail
// on it) or as an empty catalog (it would be swept zero times): it is one default entry.
TEST_F(TestCandidateListing, AnEngineThatCannotEnumerateIsNotEnumerable)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault(
            [](GraphSession&, const LoadedEngine&, int64_t, int64_t, EngineCandidatePage&) {
                return Error{ErrorCode::HIPDNN_BACKEND_ERROR, UNSUPPORTED_MESSAGE};
            });

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::NOT_ENUMERABLE);
    EXPECT_TRUE(listing.candidates.empty());
    EXPECT_NE(listing.reason.find("matched-catalog enumeration"), std::string::npos)
        << listing.reason;
}

// The same answer arriving as a throw instead of a returned error.
TEST_F(TestCandidateListing, AnEngineThatThrowsNotApplicableIsNotEnumerable)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault(
            [](GraphSession&, const LoadedEngine&, int64_t, int64_t, EngineCandidatePage&)
                -> Error { throw EngineNotApplicableError("not applicable"); });

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::NOT_ENUMERABLE);
    EXPECT_EQ(listing.reason, "not applicable");
}

// Bug caught: any enumeration failure read as "cannot enumerate", which would hide a broken
// engine behind a quiet one-entry catalog.
TEST_F(TestCandidateListing, ABrokenEnumerationIsAnErrorAndNotADefault)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault(
            [](GraphSession&, const LoadedEngine&, int64_t, int64_t, EngineCandidatePage&) {
                return Error{ErrorCode::HIPDNN_BACKEND_ERROR, "Invalid candidate page buffer"};
            });

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::LISTING_ERROR);
    EXPECT_EQ(listing.reason, "Invalid candidate page buffer");
}

// Bug caught: an engine that served page one and then refused read as non-enumerable, which
// would sweep a truncated catalog's first page as if it were everything.
TEST_F(TestCandidateListing, AnEngineThatRefusesALaterPageIsAnErrorWithNoPartialCatalog)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault([](GraphSession&,
                          const LoadedEngine&,
                          int64_t offset,
                          int64_t limit,
                          EngineCandidatePage& page) {
            if(offset > 0)
            {
                return Error{ErrorCode::HIPDNN_BACKEND_ERROR, UNSUPPORTED_MESSAGE};
            }
            page = servePage(3 * static_cast<uint64_t>(limit), offset, limit);
            return Error{};
        });

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::LISTING_ERROR);
    EXPECT_TRUE(listing.candidates.empty());
    EXPECT_EQ(listing.total, 0u);
}

// Bug caught: a catalog that changed under the walk (pages of two different catalogs stitched
// together) taken as one listing.
TEST_F(TestCandidateListing, ACatalogThatChangesSizeMidWalkIsAnError)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault([](GraphSession&,
                          const LoadedEngine&,
                          int64_t offset,
                          int64_t limit,
                          EngineCandidatePage& page) {
            const uint64_t total = offset == 0 ? 3 * static_cast<uint64_t>(limit)
                                               : 3 * static_cast<uint64_t>(limit) + 1;
            page = servePage(total, offset, limit);
            return Error{};
        });

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::LISTING_ERROR);
    EXPECT_NE(listing.reason.find("changed"), std::string::npos) << listing.reason;
}

// Bug caught: an engine whose next page offset does not move, which would never end the walk.
TEST_F(TestCandidateListing, AWalkThatDoesNotAdvanceEndsInAnError)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault([](GraphSession&,
                          const LoadedEngine&,
                          int64_t offset,
                          int64_t limit,
                          EngineCandidatePage& page) {
            page = servePage(100, offset, limit);
            page.nextOffset = static_cast<uint64_t>(offset);
            return Error{};
        });

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::LISTING_ERROR);
    EXPECT_NE(listing.reason.find("advance"), std::string::npos) << listing.reason;
}

// Bug caught: a candidate listed twice across a page seam, which would be swept (and its
// failures counted) twice.
TEST_F(TestCandidateListing, AnIdThatRepeatsAcrossPagesIsAnError)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault([](GraphSession&,
                          const LoadedEngine&,
                          int64_t offset,
                          int64_t limit,
                          EngineCandidatePage& page) {
            page = servePage(2 * static_cast<uint64_t>(limit), 0, limit);
            page.offset = static_cast<uint64_t>(offset);
            page.nextOffset = offset == 0 ? std::optional<uint64_t>(static_cast<uint64_t>(limit))
                                          : std::nullopt;
            return Error{};
        });

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::LISTING_ERROR);
    EXPECT_NE(listing.reason.find("twice"), std::string::npos) << listing.reason;
}

// Bug caught: a page of fewer candidates than the engine said it has, taken as the catalog.
TEST_F(TestCandidateListing, AShortCatalogIsAnError)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault([](GraphSession&,
                          const LoadedEngine&,
                          int64_t offset,
                          int64_t limit,
                          EngineCandidatePage& page) {
            page = servePage(5, offset, limit);
            page.candidates.pop_back();
            page.nextOffset.reset();
            return Error{};
        });

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::LISTING_ERROR);
}

// Bug caught: a graph the engine declines enumerated anyway, or reported as an error: the
// engine answered "not mine", so the case stays in the sweep as one default entry and a run of
// it reports the decline.
TEST_F(TestCandidateListing, AGraphTheEngineDoesNotTakeIsNotEnumerableAndIsNotQueried)
{
    ON_CALL(_runner, openGraph(_, _))
        .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
            return declinedSession();
        });
    EXPECT_CALL(_runner, candidatePage(_, _, _, _, _)).Times(0);

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::NOT_ENUMERABLE);
    EXPECT_NE(listing.reason.find(ENGINE.name), std::string::npos) << listing.reason;
}

TEST_F(TestCandidateListing, AGraphThatDoesNotLoadIsAnErrorAndIsNotQueried)
{
    ON_CALL(_runner, openGraph(_, _))
        .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
            return buildErrorSession("from_binary said no");
        });
    EXPECT_CALL(_runner, candidatePage(_, _, _, _, _)).Times(0);

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::LISTING_ERROR);
    EXPECT_NE(listing.reason.find("from_binary said no"), std::string::npos) << listing.reason;
}

// Bug caught: a case that was written and cannot load missing from the listing, which reads as
// a case that was never there.
TEST_F(TestCandidateListing, ABundleThatFailedToLoadIsListedAsAnError)
{
    EXPECT_CALL(_runner, openGraph(_, _)).Times(0);
    EXPECT_CALL(_runner, candidatePage(_, _, _, _, _)).Times(0);
    const ListingCase failed{"Suite.broken", "dir/broken.json", nullptr, "Failed to load bundle"};

    const auto listing = listCaseCandidates(_runner, failed, ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::LISTING_ERROR);
    EXPECT_EQ(listing.reason, "Failed to load bundle");
    EXPECT_EQ(listing.test, "Suite.broken");
}

// Bug caught: the listing building or running something: it must only enumerate.
TEST_F(TestCandidateListing, ListingBuildsAndExecutesNothing)
{
    enumerates(3);
    EXPECT_CALL(_runner, buildPlans(_, _)).Times(0);
    EXPECT_CALL(_runner, execute(_, _, _)).Times(0);
    EXPECT_CALL(_runner, benchmark(_, _, _)).Times(0);

    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);

    EXPECT_EQ(listing.status, ListingStatus::OK);
}

// Bug caught: a record the driver cannot read: knobs lost or retyped, a missing field, or the
// status spelled another way.
TEST_F(TestCandidateListing, TheRecordCarriesEveryFieldTheDriverReads)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault([](GraphSession&,
                          const LoadedEngine&,
                          int64_t offset,
                          int64_t limit,
                          EngineCandidatePage& page) {
            page = servePage(1, offset, limit);
            auto& knobs = page.candidates.front().variant.knobSettings;
            knobs.clear();
            knobs.emplace("HIPDNN_KNOB_TILE_M", int64_t{128});
            knobs.emplace("HIPDNN_KNOB_SCALE", 0.5);
            knobs.emplace("HIPDNN_KNOB_WHOLE", 2.0);
            knobs.emplace("HIPDNN_KNOB_MODE", std::string("fast"));
            return Error{};
        });

    const auto json = nlohmann::json::parse(
        listingRecordJson(listCaseCandidates(_runner, loadedCase(), ENGINE)));

    EXPECT_EQ(json.at("test"), "Suite.case_0");
    EXPECT_EQ(json.at("bundle"), "dir/sweep.json#case_0");
    EXPECT_EQ(json.at("engine"), ENGINE.name);
    EXPECT_EQ(json.at("status"), "ok");
    EXPECT_EQ(json.at("total"), 1);
    EXPECT_EQ(json.at("reason"), "");
    ASSERT_EQ(json.at("candidates").size(), 1u);
    const auto& candidate = json.at("candidates").at(0);
    EXPECT_EQ(candidate.at("id"), "c0");
    const auto& knobs = candidate.at("knobs");
    EXPECT_TRUE(knobs.at("HIPDNN_KNOB_TILE_M").is_number_integer());
    EXPECT_EQ(knobs.at("HIPDNN_KNOB_TILE_M"), 128);
    EXPECT_TRUE(knobs.at("HIPDNN_KNOB_SCALE").is_number_float());
    EXPECT_TRUE(knobs.at("HIPDNN_KNOB_WHOLE").is_number_float());
    EXPECT_EQ(knobs.at("HIPDNN_KNOB_MODE"), "fast");
}

// Bug caught: a listed candidate that does not select itself when a case forces it: the knobs
// the listing prints must read back, in their own types, as the tuple the engine enumerated.
TEST_F(TestCandidateListing, AListedCandidateReadsBackAsTheSameKnobTuple)
{
    using KnobTuple = std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant>;
    const KnobTuple enumerated{{"HIPDNN_KNOB_TILE_M", int64_t{128}},
                               {"HIPDNN_KNOB_SCALE", 2.0},
                               {"HIPDNN_KNOB_MODE", std::string("fast")}};
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault([&enumerated](GraphSession&,
                                     const LoadedEngine&,
                                     int64_t offset,
                                     int64_t limit,
                                     EngineCandidatePage& page) {
            page = servePage(1, offset, limit);
            page.candidates.front().variant.knobSettings = enumerated;
            return Error{};
        });
    const auto listing = listCaseCandidates(_runner, loadedCase(), ENGINE);
    ASSERT_EQ(listing.status, ListingStatus::OK) << listing.reason;

    const auto line = nlohmann::json::parse(listingRecordJson(listing));
    const nlohmann::json metadata{{"format_version", 1},
                                  {"engine_variant",
                                   {{"id", line.at("candidates").at(0).at("id")},
                                    {"knobs", line.at("candidates").at(0).at("knobs")}}}};
    const auto meta = hipdnn_integration_tests::parseBundleMetadataJson(metadata);

    ASSERT_TRUE(meta.engineVariant.has_value());
    KnobTuple readBack;
    for(const auto& knob : meta.engineVariant->knobs)
    {
        readBack.emplace(knob.knobId(), knob.value());
    }
    EXPECT_TRUE(readBack == enumerated);
}

// Bug caught: a non-OK record carrying a stale partial catalog, or no reason.
TEST_F(TestCandidateListing, ANonOkRecordHasNoCandidatesAndSaysWhy)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault(
            [](GraphSession&, const LoadedEngine&, int64_t, int64_t, EngineCandidatePage&) {
                return Error{ErrorCode::HIPDNN_BACKEND_ERROR, UNSUPPORTED_MESSAGE};
            });

    const auto json = nlohmann::json::parse(
        listingRecordJson(listCaseCandidates(_runner, loadedCase(), ENGINE)));

    EXPECT_EQ(json.at("status"), "not_enumerable");
    EXPECT_EQ(json.at("total"), 0);
    EXPECT_TRUE(json.at("candidates").empty());
    EXPECT_FALSE(json.at("reason").get<std::string>().empty());
}

// Bug caught: a record dropped or merged: one line per case, in order, so a case that listed an
// error does not take its neighbours with it.
TEST_F(TestCandidateListing, TheListingIsOneLinePerCaseInOrderAndCountsTheErrors)
{
    enumerates(2);
    const std::vector<ListingCase> cases{
        loadedCase("Suite.a", "dir/a.json"),
        ListingCase{"Suite.b", "dir/b.json", nullptr, "Failed to load bundle"},
        loadedCase("Suite.c", "dir/c.json#case_1")};
    std::ostringstream out;

    const auto errors = writeCandidateListing(out, _runner, cases, ENGINE);

    EXPECT_EQ(errors, 1u);
    std::istringstream lines(out.str());
    std::string line;
    std::vector<nlohmann::json> records;
    while(std::getline(lines, line))
    {
        records.push_back(nlohmann::json::parse(line));
    }
    ASSERT_EQ(records.size(), 3u);
    EXPECT_EQ(records[0].at("test"), "Suite.a");
    EXPECT_EQ(records[0].at("status"), "ok");
    EXPECT_EQ(records[0].at("candidates").size(), 2u);
    EXPECT_EQ(records[1].at("test"), "Suite.b");
    EXPECT_EQ(records[1].at("status"), "error");
    EXPECT_EQ(records[2].at("test"), "Suite.c");
    EXPECT_EQ(records[2].at("bundle"), "dir/c.json#case_1");
    EXPECT_EQ(records[2].at("status"), "ok");
}

// Bug caught: an engine message that is not valid UTF-8 aborting the line, which loses the case.
TEST_F(TestCandidateListing, AMessageThatIsNotUtf8StillWritesTheRecord)
{
    ON_CALL(_runner, candidatePage(_, _, _, _, _))
        .WillByDefault(
            [](GraphSession&, const LoadedEngine&, int64_t, int64_t, EngineCandidatePage&) {
                return Error{ErrorCode::HIPDNN_BACKEND_ERROR, std::string("bad \xff byte")};
            });

    const auto line = listingRecordJson(listCaseCandidates(_runner, loadedCase(), ENGINE));

    EXPECT_EQ(nlohmann::json::parse(line).at("status"), "error");
}

// What the listing is given: the bundles registration loaded and the ones it could not.
TEST(TestToListingCases, LoadedAndFailedBundlesBothBecomeCases)
{
    detail::LoadedBundle loaded;
    loaded.suiteName = "Suite";
    loaded.testName = "case_0";
    loaded.jsonPath = "dir/sweep.json#case_0";
    loaded.bundle = std::make_shared<IntegrationTestBundle>();
    const detail::FailedLoad failed{
        "Suite", "case_1", "Failed to load bundle: bad metadata", "dir/sweep.json#case_1"};

    const auto cases = toListingCases({loaded}, {failed});

    ASSERT_EQ(cases.size(), 2u);
    EXPECT_EQ(cases[0].test, "Suite.case_0");
    EXPECT_EQ(cases[0].bundle, "dir/sweep.json#case_0");
    EXPECT_EQ(cases[0].loaded, loaded.bundle);
    EXPECT_EQ(cases[1].test, "Suite.case_1");
    EXPECT_EQ(cases[1].bundle, "dir/sweep.json#case_1");
    EXPECT_EQ(cases[1].loaded, nullptr);
    EXPECT_EQ(cases[1].loadError, "Failed to load bundle: bad metadata");
}

// NOLINTEND(readability-identifier-naming)
