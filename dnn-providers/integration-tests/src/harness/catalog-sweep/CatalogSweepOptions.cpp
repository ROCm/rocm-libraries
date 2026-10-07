// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/catalog-sweep/CatalogSweepOptions.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <system_error>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace hipdnn_integration_tests::catalog_sweep
{

namespace
{

constexpr const char* K_FORCE_BENCHMARKING_ENV = "HIPDNN_FORCE_BENCHMARKING";
constexpr const char* K_CACHE_DIR_ENV = "HIPDNN_CACHE_DIR";

std::string_view trim(std::string_view text)
{
    const auto isSpace = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    while(!text.empty() && isSpace(text.front()))
    {
        text.remove_prefix(1);
    }
    while(!text.empty() && isSpace(text.back()))
    {
        text.remove_suffix(1);
    }
    return text;
}

std::size_t parseCount(std::string_view digits, std::string_view whole)
{
    std::size_t value = 0;
    const std::string text(digits);
    const auto* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    const bool allDigits = !digits.empty() && std::all_of(digits.begin(), digits.end(), [](char c) {
        return c >= '0' && c <= '9';
    });
    if(!allDigits || ec != std::errc{} || ptr != end)
    {
        throw std::invalid_argument("--catalog-shard expects <i>/<n> with decimal integers, got '"
                                    + std::string(whole) + "'");
    }
    return value;
}

/// Whole-string glob match: '*' is any run (including none), '?' is exactly one character.
bool globMatches(std::string_view pattern, std::string_view text)
{
    std::size_t p = 0;
    std::size_t t = 0;
    std::size_t starAt = std::string_view::npos;
    std::size_t resumeText = 0;

    while(t < text.size())
    {
        if(p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t]))
        {
            ++p;
            ++t;
        }
        else if(p < pattern.size() && pattern[p] == '*')
        {
            starAt = p++;
            resumeText = t;
        }
        else if(starAt != std::string_view::npos)
        {
            p = starAt + 1;
            t = ++resumeText;
        }
        else
        {
            return false;
        }
    }
    while(p < pattern.size() && pattern[p] == '*')
    {
        ++p;
    }
    return p == pattern.size();
}

/// 64-bit FNV-1a. Fixed by definition, so every host and every build agrees on which
/// shard owns a graph -- std::hash gives no such promise.
std::uint64_t fnv1a(std::string_view text)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for(const char c : text)
    {
        hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

ShardSpec parseShard(std::string_view text)
{
    const auto slash = text.find('/');
    if(slash == std::string_view::npos)
    {
        throw std::invalid_argument("--catalog-shard expects <i>/<n>, got '" + std::string(text)
                                    + "'");
    }

    ShardSpec shard;
    shard.index = parseCount(text.substr(0, slash), text);
    shard.count = parseCount(text.substr(slash + 1), text);

    if(shard.count == 0)
    {
        throw std::invalid_argument("--catalog-shard shard count must be at least 1, got '"
                                    + std::string(text) + "'");
    }
    if(shard.index >= shard.count)
    {
        throw std::invalid_argument("--catalog-shard index must be below the count (0-based), got '"
                                    + std::string(text) + "'");
    }
    return shard;
}

bool shardOwns(const ShardSpec& shard, std::string_view graphName)
{
    return fnv1a(graphName) % shard.count == shard.index;
}

CatalogFilter CatalogFilter::parse(std::string_view expression)
{
    if(trim(expression).empty())
    {
        throw std::invalid_argument("--catalog-filter expression is empty");
    }

    CatalogFilter filter;
    std::size_t begin = 0;
    while(begin <= expression.size())
    {
        auto end = expression.find(',', begin);
        if(end == std::string_view::npos)
        {
            end = expression.size();
        }
        const auto text = trim(expression.substr(begin, end - begin));
        begin = end + 1;

        const std::string clauseText(text);
        const auto opAt = text.find_first_of("=~!");
        if(opAt == std::string_view::npos || opAt == 0)
        {
            throw std::invalid_argument("--catalog-filter clause '" + clauseText
                                        + "' is not <field><op><value>; ops are =, !=, ~, !~");
        }

        Clause clause;
        const auto field = trim(text.substr(0, opAt));
        if(field == "graph")
        {
            clause.field = Field::GRAPH;
        }
        else if(field == "engine")
        {
            clause.field = Field::ENGINE;
        }
        else
        {
            throw std::invalid_argument("--catalog-filter clause '" + clauseText
                                        + "' names an unknown field '" + std::string(field)
                                        + "'; fields are graph, engine");
        }

        auto rest = text.substr(opAt);
        if(rest.front() == '!')
        {
            clause.negated = true;
            rest.remove_prefix(1);
        }
        if(rest.empty() || (rest.front() != '=' && rest.front() != '~'))
        {
            throw std::invalid_argument("--catalog-filter clause '" + clauseText
                                        + "' has a bad operator; ops are =, !=, ~, !~");
        }
        clause.match = rest.front() == '=' ? Match::GLOB : Match::SUBSTRING;
        rest.remove_prefix(1);

        std::size_t valueBegin = 0;
        while(valueBegin <= rest.size())
        {
            auto valueEnd = rest.find('|', valueBegin);
            if(valueEnd == std::string_view::npos)
            {
                valueEnd = rest.size();
            }
            const auto pattern = trim(rest.substr(valueBegin, valueEnd - valueBegin));
            valueBegin = valueEnd + 1;
            if(pattern.empty())
            {
                throw std::invalid_argument("--catalog-filter clause '" + clauseText
                                            + "' has an empty pattern");
            }
            clause.patterns.emplace_back(pattern);
        }

        filter._clauses.push_back(std::move(clause));
    }
    return filter;
}

bool CatalogFilter::clauseHolds(const Clause& clause, std::string_view value)
{
    const bool anyMatches
        = std::any_of(clause.patterns.begin(), clause.patterns.end(), [&](const std::string& p) {
              return clause.match == Match::GLOB ? globMatches(p, value)
                                                 : value.find(p) != std::string_view::npos;
          });
    return anyMatches != clause.negated;
}

bool CatalogFilter::matches(std::string_view graph, std::string_view engine) const
{
    return std::all_of(_clauses.begin(), _clauses.end(), [&](const Clause& clause) {
        return clauseHolds(clause, clause.field == Field::GRAPH ? graph : engine);
    });
}

bool CatalogFilter::matchesGraph(std::string_view graph) const
{
    return std::all_of(_clauses.begin(), _clauses.end(), [&](const Clause& clause) {
        return clause.field != Field::GRAPH || clauseHolds(clause, graph);
    });
}

CatalogSweepResolution resolveCatalogSweepOptions(const CatalogSweepRequest& request)
{
    CatalogSweepResolution resolution;

    if(!request.sweep)
    {
        std::string stray;
        if(request.reportPath.has_value())
        {
            stray = "--catalog-report";
        }
        else if(request.filter.has_value())
        {
            stray = "--catalog-filter";
        }
        else if(request.shard.has_value())
        {
            stray = "--catalog-shard";
        }

        if(!stray.empty())
        {
            resolution.error = stray
                               + " only applies to a sweep; add --sweep-knob-catalog (it is "
                                 "refused rather than ignored, so a run is never mistaken for "
                                 "a sweep)";
        }
        return resolution;
    }

    if(!request.reportPath.has_value() || request.reportPath->empty())
    {
        resolution.error = "--sweep-knob-catalog requires --catalog-report <path>: a sweep whose "
                           "per-entry results are not recorded has verified nothing";
        return resolution;
    }

    try
    {
        CatalogSweepOptions options;
        options.reportPath = *request.reportPath;
        if(request.filter.has_value())
        {
            options.filter = CatalogFilter::parse(*request.filter);
        }
        if(request.shard.has_value())
        {
            options.shard = parseShard(*request.shard);
        }
        resolution.options = std::move(options);
    }
    catch(const std::invalid_argument& e)
    {
        resolution.error = e.what();
    }
    return resolution;
}

EnvLookup processEnvironment()
{
    return [](const std::string& name) -> std::optional<std::string> {
        // A value no environment can hold, so "set to empty" and "absent" stay distinct.
        static const std::string s_absent("\x01<absent>");
        auto value = hipdnn_data_sdk::utilities::getEnv(name.c_str(), s_absent.c_str());
        if(value == s_absent)
        {
            return std::nullopt;
        }
        return value;
    };
}

std::optional<std::string> checkSweepEnvironment(const EnvLookup& env)
{
    if(const auto forced = env(K_FORCE_BENCHMARKING_ENV); forced.has_value())
    {
        return std::string(K_FORCE_BENCHMARKING_ENV) + " is set (to '" + *forced
               + "'); unset it. It overrides global.benchmarking for every plan, so the sweep "
                 "would not be building the configurations the catalog lists.";
    }

    const auto cacheDir = env(K_CACHE_DIR_ENV);
    if(!cacheDir.has_value() || cacheDir->empty())
    {
        return std::string(K_CACHE_DIR_ENV)
               + " is not set. Point it at a fresh directory (absent or empty) so no engine "
                 "ranking cached by an earlier run influences the sweep.";
    }

    const std::filesystem::path dir(*cacheDir);
    std::error_code ec;
    const auto status = std::filesystem::status(dir, ec);
    if(ec || !std::filesystem::exists(status))
    {
        return std::nullopt; // absent: fresh by definition, hipDNN creates it on first use
    }
    if(!std::filesystem::is_directory(status))
    {
        return std::string(K_CACHE_DIR_ENV) + " names '" + *cacheDir
               + "', which exists and is not a directory.";
    }
    if(!std::filesystem::is_empty(dir, ec) || ec)
    {
        return std::string(K_CACHE_DIR_ENV) + " names '" + *cacheDir
               + "', which is not empty. Use a fresh directory (absent or empty) so no cached "
                 "engine ranking from an earlier run influences the sweep.";
    }
    return std::nullopt;
}

} // namespace hipdnn_integration_tests::catalog_sweep
