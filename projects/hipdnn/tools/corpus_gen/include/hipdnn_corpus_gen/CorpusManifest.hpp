// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/CorpusOutput.hpp>
#include <hipdnn_corpus_gen/GraphIdentity.hpp>
#include <hipdnn_corpus_gen/PoolAssembly.hpp>
#include <hipdnn_corpus_gen/RegimeLabel.hpp>

#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

/// @file CorpusManifest.hpp
/// @brief What was generated, where each graph came from, and the key it is measured under.
///
/// Two manifests, one content, as the retired Python assembler wrote them.
/// `manifest.json` is the audit record. `manifest.csv` is the same rows in the form
/// `uhd_gen evaluate --regime-column` consumes -- keyed on `benchmark`, which is the identity
/// the collected corpus reports for the same graph, so the two join directly. Without that
/// column RFC 0019.13 §11.2's per-regime table reports UNAVAILABLE, which is the state the
/// Python was written to end and which this port must not reintroduce.
///
/// The port is op-general in one place: the Python emitted a column per SDPA field
/// (`dtype`, `batch`, `heads_q`, ...) and a column per hardcoded regime facet (`phase`,
/// `context`, `grouping`). Here the parameter columns come from the point itself via
/// `asQueryColumns` -- already the spelling `uhd_gen`'s feature hash expects -- and the facet
/// columns from the declaration's `regime_label` axes. Nothing here knows an operation's name.
namespace hipdnn_corpus_gen
{

namespace detail
{

/// A CSV field, quoted only when it has to be.
///
/// `origin` is free text ("a pack file and its kernel count, a model name, a draw index") and
/// `name` is composed from labels, so neither can be assumed comma-free. The `q.*` values
/// cannot contain a comma -- they are numbers, `true`/`false`, or declared enum identifiers --
/// but they go through the same function rather than relying on that as an invariant nothing
/// checks.
inline std::string asCsvField(const std::string& text)
{
    if(text.find_first_of(",\"\n\r") == std::string::npos)
    {
        return text;
    }

    std::string quoted = "\"";
    for(const char character : text)
    {
        if(character == '"')
        {
            quoted += '"';
        }
        quoted += character;
    }
    quoted += "\"";
    return quoted;
}

} // namespace detail

/// One graph, as both manifests record it.
struct ManifestEntry
{
    /// The pool membership: the problem, which source supplied it, where inside that source,
    /// and the regime it was ordered under.
    PoolEntry entry;

    /// The identity the measured corpus will carry for this graph. See @ref graphIdentity;
    /// this must be the id written into the graph document, not a second opinion about it.
    std::string benchmark;

    /// The graph document's own name, which is how an L2 collection -- which mints its own
    /// ids -- joins back to this row.
    std::string name;

    /// Corpus-relative, e.g. `graphs/sdpa_fwd_0001.fb`.
    std::string file;

    /// The graph's tensor footprint, as the byte budget measured it.
    int64_t bytes = 0;

    /// Which operation's declaration this problem was drawn from, and that declaration's
    /// facets. Per entry and not per run, because one corpus may cover several operations:
    /// their parameters do not agree, and neither do their regime axes, so a single set of
    /// either would describe one of them and misdescribe the rest.
    std::string operation;
    std::vector<RegimeAxis> regimeAxes;
};

/// Everything the manifest records that is not per-graph.
struct ManifestContext
{
    /// Which generation produced this corpus. Recorded because the C++ sampler applies the
    /// declared mixture as per-combination quotas where the Python applied it as per-draw
    /// weights: both honour the shares, and neither reproduces the other's bytes at a seed.
    std::string tool = "corpus_gen";

    /// The operations covered, in the order they were generated. A label for the run; the
    /// authoritative per-row answer is @ref ManifestEntry::operation.
    std::vector<std::string> operations;

    uint64_t seed = 0;

    /// What was asked for, against which @ref ManifestEntry count is the shortfall. Reported,
    /// never filled: a corpus of 664 problems from an engine that serves 664 is complete.
    int64_t requested = 0;

    /// Per source, from `PoolAssembly::select`.
    std::map<std::string, int64_t> allocation;
    std::map<std::string, int64_t> duplicatesDropped;

    /// Files the corpus was derived from. Digested here so a rerun that quietly read a
    /// different pack is visible in a diff of the manifest.
    std::vector<std::filesystem::path> inputs;

    /// Per-source notes -- what each pool held and what it dropped. Free-form because the
    /// sources are not alike; see `PoolEntry::origin`.
    nlohmann::json reports = nlohmann::json::object();
};

namespace detail
{

inline std::string digestOf(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open())
    {
        return "";
    }
    const std::string bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    return hipdnn_plugin_sdk::uhd::sha256(bytes);
}

} // namespace detail

/// @brief The `manifest.json` document.
///
/// Top-level keys are the Python's verbatim, because `uhd_gen/reproduce/score_predictions.py`
/// and `compare_engines.py` index `manifest["graphs"]` and read `benchmark`, `name` and
/// `regime` off each row. Those three surviving the port unchanged is the contract; renaming
/// any of them is a silent break, since both scripts fall back to reporting the graph id.
inline nlohmann::json corpusManifest(const std::vector<ManifestEntry>& entries,
                                     const ManifestContext& context)
{
    nlohmann::json records = nlohmann::json::array();
    std::map<std::string, int64_t> mix;
    std::map<std::string, int64_t> regimes;

    for(const auto& entry : entries)
    {
        nlohmann::json record;
        record["benchmark"] = entry.benchmark;
        record["file"]      = entry.file;
        record["name"]      = entry.name;
        record["source"]    = entry.entry.source;
        record["origin"]    = entry.entry.origin;
        record["regime"]    = entry.entry.regime;

        // The facets beside the joined label, so a report can group by one of them without
        // parsing the other. Derived from the point rather than split out of `regime`, whose
        // separator is also legal inside a declared label.
        for(const auto& facet : regimeFacets(entry.regimeAxes, entry.entry.point))
        {
            record[facet.first] = facet.second;
        }

        record["op"] = entry.operation;
        for(const auto& parameter : entry.entry.point)
        {
            record["q." + parameter.first] = asText(parameter.second);
        }
        record["bytes"] = entry.bytes;

        records.push_back(record);
        ++mix[entry.entry.source];
        ++regimes[entry.entry.regime];
    }

    nlohmann::json inputs = nlohmann::json::array();
    for(const auto& path : context.inputs)
    {
        inputs.push_back({{"path", path.string()}, {"sha256", detail::digestOf(path)}});
    }

    nlohmann::json manifest;
    manifest["tool"]              = context.tool;
    manifest["operations"]        = context.operations;
    manifest["seed"]              = context.seed;
    manifest["requested"]         = context.requested;
    manifest["emitted"]           = static_cast<int64_t>(entries.size());
    manifest["mix"]               = mix;
    manifest["allocation"]        = context.allocation;
    manifest["duplicates_dropped"] = context.duplicatesDropped;
    manifest["regimes"]           = regimes;
    manifest["inputs"]            = inputs;
    manifest["reports"]           = context.reports;
    manifest["graphs"]            = records;
    manifest["note"]
        = "`benchmark` is the content-derived id written into each graph document, and is the "
          "`benchmark` column of the corpus collected from it -- join manifest.csv on it to "
          "give `uhd_gen evaluate --regime-column regime` the column RFC 0019.13 section 11.2 "
          "requires. An L2 collection mints its own graph ids; join on `name` there, which the "
          "graph document carries.";
    return manifest;
}

namespace detail
{

/// Every facet name and every `q.*` name any entry carries, first-seen order preserved.
///
/// A union rather than the first entry's columns, because a corpus may cover more than one
/// operation and theirs do not agree. First-seen rather than sorted so that the common case --
/// one operation -- still emits its parameters in declaration order, which is the order every
/// other file this tool writes uses.
inline std::vector<std::string> unionOfColumns(const std::vector<ManifestEntry>& entries,
                                               bool facets)
{
    std::vector<std::string> columns;
    const auto note = [&columns](const std::string& column) {
        if(std::find(columns.begin(), columns.end(), column) == columns.end())
        {
            columns.push_back(column);
        }
    };

    for(const auto& entry : entries)
    {
        if(facets)
        {
            for(const auto& facet : regimeFacets(entry.regimeAxes, entry.entry.point))
            {
                note(facet.first);
            }
        }
        else
        {
            for(const auto& parameter : entry.entry.point)
            {
                note("q." + parameter.first);
            }
        }
    }
    return columns;
}

/// The value @p entry has in each of @p columns, empty where it has none.
inline std::vector<std::string> valuesFor(const std::vector<std::string>& columns,
                                          const std::vector<std::pair<std::string, std::string>>& held)
{
    std::vector<std::string> values(columns.size());
    for(const auto& one : held)
    {
        const auto found = std::find(columns.begin(), columns.end(), one.first);
        if(found != columns.end())
        {
            values[static_cast<size_t>(found - columns.begin())] = one.second;
        }
    }
    return values;
}

} // namespace detail

/// @brief The `manifest.csv` header and rows, in one ordered pass.
///
/// Column order follows the Python's: identity, then what the graph is, then where it came
/// from, then the problem itself, then the file. The `q.*` block sits exactly where the SDPA
/// field columns did.
///
/// Header and rows are generated from the same column list for the same reason
/// `asQueryColumns` is -- a header that disagrees with its rows transposes two features and
/// trains a model on the wrong ones. Where the Python could assume one operation's fields, this
/// takes the union across entries and leaves a cell empty where an entry has no such parameter:
/// a row of a two-operation corpus is not wrong about its own columns just because the other
/// operation has some it does not.
/// Takes no @ref ManifestContext: every column it writes is now per-entry, which is the point
/// of the union above.
inline std::string corpusManifestCsv(const std::vector<ManifestEntry>& entries)
{
    const auto facetColumns = detail::unionOfColumns(entries, true);
    const auto queryColumns = detail::unionOfColumns(entries, false);

    std::string text = "benchmark,name,regime";
    for(const auto& column : facetColumns)
    {
        text += "," + detail::asCsvField(column);
    }
    text += ",source,origin,op";
    for(const auto& column : queryColumns)
    {
        text += "," + detail::asCsvField(column);
    }
    text += ",bytes,file\n";

    for(const auto& entry : entries)
    {
        std::vector<std::pair<std::string, std::string>> parameters;
        parameters.reserve(entry.entry.point.size());
        for(const auto& parameter : entry.entry.point)
        {
            parameters.emplace_back("q." + parameter.first, asText(parameter.second));
        }

        text += detail::asCsvField(entry.benchmark);
        text += "," + detail::asCsvField(entry.name);
        text += "," + detail::asCsvField(entry.entry.regime);
        for(const auto& value :
            detail::valuesFor(facetColumns, regimeFacets(entry.regimeAxes, entry.entry.point)))
        {
            text += "," + detail::asCsvField(value);
        }
        text += "," + detail::asCsvField(entry.entry.source);
        text += "," + detail::asCsvField(entry.entry.origin);
        text += "," + detail::asCsvField(entry.operation);
        for(const auto& value : detail::valuesFor(queryColumns, parameters))
        {
            text += "," + detail::asCsvField(value);
        }
        text += "," + std::to_string(entry.bytes);
        text += "," + detail::asCsvField(entry.file);
        text += "\n";
    }
    return text;
}

/// @brief Writes both manifests into @p root, and returns the JSON one.
inline nlohmann::json writeCorpusManifest(const std::filesystem::path& root,
                                          const std::vector<ManifestEntry>& entries,
                                          const ManifestContext& context)
{
    const auto manifest = corpusManifest(entries, context);

    std::ofstream json(root / "manifest.json");
    json << manifest.dump(2) << "\n";

    std::ofstream csv(root / "manifest.csv");
    csv << corpusManifestCsv(entries);

    return manifest;
}

} // namespace hipdnn_corpus_gen
