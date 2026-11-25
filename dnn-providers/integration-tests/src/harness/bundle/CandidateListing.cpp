// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/CandidateListing.hpp"

#include <exception>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <variant>

#include "harness/EngineNotApplicableError.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::bundle
{

const char* toString(ListingStatus status)
{
    switch(status)
    {
    case ListingStatus::OK:
        return "ok";
    case ListingStatus::NOT_ENUMERABLE:
        return "not_enumerable";
    case ListingStatus::LISTING_ERROR:
        return "error";
    default:
        return "unknown";
    }
}

bool isEnumerationUnsupported(const hipdnn_frontend::Error& error)
{
    return error.get_message().find("does not support matched-catalog enumeration")
           != std::string::npos;
}

namespace
{

nlohmann::json
    knobsJson(const std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant>& knobs)
{
    auto result = nlohmann::json::object();
    for(const auto& entry : knobs)
    {
        const auto& name = entry.first;
        std::visit([&result, &name](const auto& held) { result[name] = held; }, entry.second);
    }
    return result;
}

// Ends `listing` as something other than OK: the status and the reason, with no partial
// catalog left behind to be mistaken for a whole one.
CaseListing ended(CaseListing& listing, ListingStatus status, std::string reason)
{
    listing.status = status;
    listing.reason = std::move(reason);
    listing.total = 0;
    listing.candidates.clear();
    return std::move(listing);
}

// The walk proper; throws what the runner throws, which listCaseCandidates() turns into a
// status.
CaseListing walkCatalog(IGraphEngineRunner& runner,
                        const ListingCase& listed,
                        const LoadedEngine& engine,
                        CaseListing listing)
{
    if(listed.loaded == nullptr)
    {
        return ended(listing, ListingStatus::LISTING_ERROR, listed.loadError);
    }

    auto session = runner.openGraph(*listed.loaded, engine);
    if(session.buildFailed)
    {
        return ended(
            listing, ListingStatus::LISTING_ERROR, "the graph did not load: " + session.buildError);
    }
    if(!session.engines.accepted)
    {
        // Not an error: the engine answers "not mine", which a run of this case reports as a
        // decline, so the case stays in the sweep as the engine's single default entry.
        const auto& status = session.engines.status;
        const auto why
            = status.is_good() ? std::string("it is not in the ranked list") : status.get_message();
        return ended(listing,
                     ListingStatus::NOT_ENUMERABLE,
                     "engine " + engine.name + " does not take this graph: " + why);
    }

    std::set<std::string> seenIds;
    int64_t offset = 0;
    bool firstPage = true;
    while(true)
    {
        hipdnn_frontend::EngineCandidatePage page;
        const auto error
            = runner.candidatePage(session, engine, offset, CANDIDATE_PAGE_LIMIT, page);
        if(!error.is_good())
        {
            // Only the first page can say "cannot enumerate": an engine that served page one
            // and then refused has broken, not declined.
            const bool unsupported = firstPage && isEnumerationUnsupported(error);
            return ended(listing,
                         unsupported ? ListingStatus::NOT_ENUMERABLE : ListingStatus::LISTING_ERROR,
                         error.get_message());
        }

        if(firstPage)
        {
            listing.total = page.totalCount;
        }
        else if(page.totalCount != listing.total)
        {
            return ended(listing,
                         ListingStatus::LISTING_ERROR,
                         "the catalog changed while it was being listed (total "
                             + std::to_string(listing.total) + " then "
                             + std::to_string(page.totalCount) + ")");
        }
        firstPage = false;

        for(const auto& candidate : page.candidates)
        {
            if(!seenIds.insert(candidate.id).second)
            {
                return ended(listing,
                             ListingStatus::LISTING_ERROR,
                             "candidate id \"" + candidate.id + "\" appears twice");
            }
            listing.candidates.push_back({candidate.id, knobsJson(candidate.variant.knobSettings)});
        }

        if(!page.nextOffset.has_value())
        {
            break;
        }
        if(*page.nextOffset <= static_cast<uint64_t>(offset))
        {
            return ended(listing,
                         ListingStatus::LISTING_ERROR,
                         "the catalog page at offset " + std::to_string(offset)
                             + " did not advance");
        }
        offset = static_cast<int64_t>(*page.nextOffset);
    }

    if(listing.candidates.size() != listing.total)
    {
        return ended(listing,
                     ListingStatus::LISTING_ERROR,
                     "the engine reported " + std::to_string(listing.total)
                         + " candidates and listed " + std::to_string(listing.candidates.size()));
    }

    listing.status = ListingStatus::OK;
    return listing;
}

} // namespace

CaseListing listCaseCandidates(IGraphEngineRunner& runner,
                               const ListingCase& listed,
                               const LoadedEngine& engine)
{
    CaseListing listing;
    listing.test = listed.test;
    listing.bundle = listed.bundle;
    listing.engine = engine.name;

    try
    {
        return walkCatalog(runner, listed, engine, listing);
    }
    // A plugin that throws "not applicable" instead of returning it is the same answer.
    catch(const EngineNotApplicableError& e)
    {
        return ended(listing, ListingStatus::NOT_ENUMERABLE, e.what());
    }
    catch(const std::exception& e)
    {
        return ended(listing, ListingStatus::LISTING_ERROR, e.what());
    }
}

std::string listingRecordJson(const CaseListing& listing)
{
    auto candidates = nlohmann::json::array();
    for(const auto& candidate : listing.candidates)
    {
        candidates.push_back({{"id", candidate.id}, {"knobs", candidate.knobs}});
    }

    nlohmann::json json = nlohmann::json::object();
    json["test"] = listing.test;
    json["bundle"] = listing.bundle;
    json["engine"] = listing.engine;
    json["status"] = toString(listing.status);
    json["total"] = listing.total;
    json["candidates"] = std::move(candidates);
    json["reason"] = listing.reason;
    // Invalid UTF-8 in an engine's message must not stop the line from being written: a lost
    // record is a case with no listing.
    return json.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

size_t writeCandidateListing(std::ostream& out,
                             IGraphEngineRunner& runner,
                             const std::vector<ListingCase>& cases,
                             const LoadedEngine& engine)
{
    size_t errors = 0;
    for(const auto& listed : cases)
    {
        const auto listing = listCaseCandidates(runner, listed, engine);
        if(listing.status == ListingStatus::LISTING_ERROR)
        {
            ++errors;
        }
        out << listingRecordJson(listing) << '\n';
        out.flush();
    }
    return errors;
}

} // namespace hipdnn_integration_tests::bundle
