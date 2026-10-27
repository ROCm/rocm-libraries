// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <nlohmann/json.hpp>

#include <map>
#include <string>

/// @file EngineCoverage.hpp
/// @brief Per-engine coverage overrides from operations/engines.json.
///
/// A `pack` engine serves exactly its pack's shapes, so shape search is skipped for it.
namespace hipdnn_corpus_gen
{

enum class EngineCoverage
{
    SEARCH, ///< The default: the pack and model shapes, then a search for the rest.
    PACK ///< Exactly the pack's shapes; nothing outside them is served.
};

struct EngineCoverageEntry
{
    EngineCoverage coverage = EngineCoverage::SEARCH;
    std::string reason;
};

using EngineCoverageTable = std::map<std::string, EngineCoverageEntry>;

/// @brief Parses `{"engines": {"<name>": {"coverage": "pack"|"search", "reason": "..."}}}`.
///
/// Rejects an unknown coverage value and a `pack` entry with no reason.
inline bool parseEngineCoverage(const nlohmann::json& document,
                                EngineCoverageTable& table,
                                std::string& error)
{
    const auto engines = document.find("engines");
    if(engines == document.end() || !engines->is_object())
    {
        error = "no \"engines\" object";
        return false;
    }
    for(const auto& [name, body] : engines->items())
    {
        if(!body.is_object())
        {
            error = "engine '" + name + "' is not an object";
            return false;
        }
        EngineCoverageEntry entry;
        const auto coverage = body.value("coverage", std::string("search"));
        if(coverage == "pack")
        {
            entry.coverage = EngineCoverage::PACK;
        }
        else if(coverage != "search")
        {
            error = "engine '" + name + "' has coverage '";
            error += coverage;
            error += R"('; expected "pack" or "search")";
            return false;
        }
        entry.reason = body.value("reason", std::string());
        if(entry.coverage == EngineCoverage::PACK && entry.reason.empty())
        {
            error = "engine '" + name + "' claims pack coverage with no reason";
            return false;
        }
        table[name] = std::move(entry);
    }
    return true;
}

} // namespace hipdnn_corpus_gen
