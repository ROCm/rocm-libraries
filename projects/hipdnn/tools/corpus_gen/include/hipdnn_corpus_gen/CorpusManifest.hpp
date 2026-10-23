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
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

/// @file CorpusManifest.hpp
/// @brief What was generated, where each graph came from, and the key it is measured under.
/// `manifest.json` is the audit record; `manifest.csv` holds the same rows keyed on `benchmark`
/// for `uhd_gen evaluate --regime-column` (RFC 0019.13 §11.2).
namespace hipdnn_corpus_gen
{

namespace detail
{

/// A CSV field, quoted only when it has to be.
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
    PoolEntry entry;

    /// The graph's content key (@ref graphContentKey): the key a measurement of it is stored
    /// and deduplicated under, which `hipdnn_bench` reports as `graph_key`.
    std::string benchmark;

    /// The graph document's name.
    std::string name;

    /// Corpus-relative, e.g. `graphs/sdpa_fwd_0001.fb`.
    std::string file;

    /// Tensor footprint, as the byte budget measured it.
    int64_t bytes = 0;

    /// Per entry, not per run: one corpus may cover several operations with different axes.
    std::string operation;
    std::vector<RegimeAxis> regimeAxes;
};

/// Everything the manifest records that is not per-graph.
struct ManifestContext
{
    std::string tool = "corpus_gen";

    /// In generation order; the per-row authority is @ref ManifestEntry::operation.
    std::vector<std::string> operations;

    uint64_t seed = 0;

    /// Requested entry count. A shortfall is reported, never padded.
    int64_t requested = 0;

    /// Per source, from `PoolAssembly::select`.
    std::map<std::string, int64_t> allocation;
    std::map<std::string, int64_t> duplicatesDropped;

    /// Input files; digested so a rerun on different inputs shows in a manifest diff.
    std::vector<std::filesystem::path> inputs;

    /// Free-form per-source notes: what each pool held and dropped.
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
/// Key names are a contract with `uhd_gen/reproduce/compare_engines.py` (`graphs`, `benchmark`,
/// `name`, `regime`); renaming one breaks it silently.
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
        record["file"] = entry.file;
        record["name"] = entry.name;
        record["source"] = entry.entry.source;
        record["origin"] = entry.entry.origin;
        record["regime"] = entry.entry.regime;

        // Facets come from the point, not split from `regime`: its separator may appear in
        // a label.
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
    manifest["tool"] = context.tool;
    manifest["operations"] = context.operations;
    manifest["seed"] = context.seed;
    manifest["requested"] = context.requested;
    manifest["emitted"] = static_cast<int64_t>(entries.size());
    manifest["mix"] = mix;
    manifest["allocation"] = context.allocation;
    manifest["duplicates_dropped"] = context.duplicatesDropped;
    manifest["regimes"] = regimes;
    manifest["inputs"] = inputs;
    manifest["reports"] = context.reports;
    manifest["graphs"] = records;
    manifest["note"]
        = "`benchmark` is each graph's content key, which hipdnn_bench reports as `graph_key` "
          "and uhd_gen stores as the `benchmark` column of every collection made from it -- "
          "join manifest.csv on it to give `uhd_gen evaluate --regime-column regime` the column "
          "RFC 0019.13 section 11.2 requires.";
    return manifest;
}

namespace detail
{

/// Union of facet (or `q.*`) columns across entries, in first-seen order so a
/// single-operation corpus keeps declaration order.
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
inline std::vector<std::string>
    valuesFor(const std::vector<std::string>& columns,
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
/// Header and rows share one column list; cells are empty where an entry lacks a column.
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

/// @brief The content keys of every graph the corpora at @p manifests hold, hashed from the
/// graph files the manifests list, so a corpus is held out by what it holds whatever its
/// manifest recorded. Throws when a manifest or one of its graph files cannot be read.
inline std::set<std::string> excludedGraphKeys(const std::vector<std::filesystem::path>& manifests)
{
    std::set<std::string> keys;
    for(const auto& path : manifests)
    {
        std::ifstream file(path);
        if(!file)
        {
            throw std::runtime_error("cannot read " + path.string());
        }
        nlohmann::json manifest;
        file >> manifest;
        for(const auto& graph : manifest.value("graphs", nlohmann::json::array()))
        {
            const auto graphPath = path.parent_path() / graph.at("file").get<std::string>();
            std::ifstream bytes(graphPath, std::ios::binary);
            if(!bytes)
            {
                throw std::runtime_error("cannot read " + graphPath.string()
                                         + ", which keys a graph of " + path.string());
            }
            const std::vector<uint8_t> content((std::istreambuf_iterator<char>(bytes)),
                                               std::istreambuf_iterator<char>());
            keys.insert(graphContentKey(content.data(), content.size()));
        }
    }
    return keys;
}

/// @brief Writes both manifests into @p root, and returns the JSON one.
inline nlohmann::json writeCorpusManifest(const std::filesystem::path& root,
                                          const std::vector<ManifestEntry>& entries,
                                          const ManifestContext& context)
{
    const auto manifest = corpusManifest(entries, context);

    // A truncated manifest reads as a smaller corpus, so a failed write is an error.
    const auto write = [](const std::filesystem::path& path, const std::string& text) {
        std::ofstream out(path);
        out << text;
        out.close();
        if(!out)
        {
            throw std::runtime_error("Failed to write " + path.string());
        }
    };
    write(root / "manifest.json", manifest.dump(2) + "\n");
    write(root / "manifest.csv", corpusManifestCsv(entries));

    return manifest;
}

} // namespace hipdnn_corpus_gen
