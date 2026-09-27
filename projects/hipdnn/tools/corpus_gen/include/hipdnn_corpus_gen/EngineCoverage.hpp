// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <nlohmann/json.hpp>

#include <map>
#include <string>

/// @file EngineCoverage.hpp
/// @brief What shape generation knows about an engine's coverage that the engine does not say.
///
/// An engine answers "do you serve this graph" and nothing wider: no query returns the set of
/// graphs it serves, and a natively-coded matcher does not declare which fields it compares.
/// For an engine whose kernels bake every shape field in -- `hipkernel:Gfx942AttentionDense`,
/// whose matcher demands equality on all of them -- that set is exactly its pack, and searching
/// for more can only rediscover pack shapes. Until an engine can state that itself, it is
/// recorded here, with the reason, and generation skips the search for it.
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
/// An unknown coverage value, or a `pack` entry with no reason, is refused: an entry is a
/// claim that stops a search, and a claim nobody can check is exactly what this file exists
/// to avoid.
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
            error = "engine '" + name + "' has coverage '" + coverage
                    + "'; expected \"pack\" or \"search\"";
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
