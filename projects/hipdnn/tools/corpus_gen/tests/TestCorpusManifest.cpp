// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/CorpusManifest.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/// @file TestCorpusManifest.cpp
/// @brief The record that makes a corpus joinable to its measurements.
///
/// `uhd_gen/reproduce/score_predictions.py` and `compare_engines.py` read `benchmark`, `name`
/// and `regime` from `manifest["graphs"]`, so those keys are asserted by name.

using namespace hipdnn_corpus_gen;

namespace
{

std::vector<RegimeAxis> shippedSdpaAxes()
{
    const std::string path
        = std::string(HIPDNN_CORPUS_GEN_OPERATIONS_DIR) + "/sdpa_fwd.opmeta.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());
    return parsed.ok() ? parsed.metadata->regimeLabel : std::vector<RegimeAxis>{};
}

ProblemPoint sdpaPoint(int64_t seqlenQ, int64_t seqlenK)
{
    return ProblemPoint{{"batch", int64_t{1}},
                        {"heads", int64_t{32}},
                        {"heads_kv", int64_t{32}},
                        {"seqlen_q", seqlenQ},
                        {"seqlen_k", seqlenK},
                        {"head_dim", int64_t{128}},
                        {"is_causal", true},
                        {"dtype", std::string("bf16")}};
}

ManifestEntry sdpaEntry(const std::string& name,
                        int64_t seqlenQ,
                        int64_t seqlenK,
                        const std::string& source,
                        const std::string& origin,
                        const std::string& regime)
{
    ManifestEntry entry;
    entry.entry.point = sdpaPoint(seqlenQ, seqlenK);
    entry.entry.source = source;
    entry.entry.origin = origin;
    entry.entry.regime = regime;
    entry.benchmark = graphIdentity(reinterpret_cast<const uint8_t*>(name.data()), name.size());
    entry.name = name;
    entry.file = "graphs/" + name + ".fb";
    entry.bytes = 4096;
    entry.operation = "sdpa_fwd";
    entry.regimeAxes = shippedSdpaAxes();
    return entry;
}

ManifestContext sdpaContext()
{
    ManifestContext context;
    context.operations = {"sdpa_fwd"};
    context.seed = 7;
    context.requested = 4;
    context.allocation = {{"kernel", 1}, {"model", 1}, {"sweep", 0}};
    return context;
}

std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for(size_t i = 0; i < line.size(); ++i)
    {
        const char character = line[i];
        if(quoted)
        {
            if(character == '"' && i + 1 < line.size() && line[i + 1] == '"')
            {
                field += '"';
                ++i;
            }
            else if(character == '"')
            {
                quoted = false;
            }
            else
            {
                field += character;
            }
        }
        else if(character == '"')
        {
            quoted = true;
        }
        else if(character == ',')
        {
            fields.push_back(field);
            field.clear();
        }
        else
        {
            field += character;
        }
    }
    fields.push_back(field);
    return fields;
}

std::vector<std::vector<std::string>> parseCsv(const std::string& text)
{
    std::vector<std::vector<std::string>> rows;
    std::istringstream stream(text);
    std::string line;
    while(std::getline(stream, line))
    {
        rows.push_back(splitCsvLine(line));
    }
    return rows;
}

} // namespace

TEST(TestCorpusManifest, TheThreeColumnsTheReproduceScriptsIndexSurviveThePort)
{
    // A rename would not error in the scripts; they fall back to raw graph ids.
    const auto manifest = corpusManifest(
        {sdpaEntry("a", 2048, 2048, "sweep", "draw 0", "prefill_short")}, sdpaContext());

    ASSERT_EQ(manifest["graphs"].size(), 1u);
    const auto& row = manifest["graphs"][0];
    EXPECT_TRUE(row.contains("benchmark"));
    EXPECT_EQ(row["name"], "a");
    EXPECT_EQ(row["regime"], "prefill_short");
}

TEST(TestCorpusManifest, TheProblemColumnsComeFromThePointRatherThanFromAnOperationsFieldNames)
{
    // Columns are the point's parameters under the `q.` prefix uhd_gen's feature hash expects.
    const auto manifest = corpusManifest(
        {sdpaEntry("a", 1, 4096, "kernel", "pack.kdp.json", "decode_long")}, sdpaContext());

    const auto& row = manifest["graphs"][0];
    EXPECT_EQ(row["q.batch"], "1");
    EXPECT_EQ(row["q.seqlen_q"], "1");
    EXPECT_EQ(row["q.seqlen_k"], "4096");
    EXPECT_EQ(row["q.dtype"], "bf16");
    EXPECT_EQ(row["q.is_causal"], "true");
    EXPECT_FALSE(row.contains("heads_q")) << "an SDPA field name leaked into the port";
    EXPECT_FALSE(row.contains("dtype")) << "an SDPA field name leaked into the port";
}

TEST(TestCorpusManifest, EachDeclaredFacetGetsItsOwnColumnBesideTheJoinedLabel)
{
    // Facets cannot be split back out of `regime`: a label may contain the separator.
    const auto manifest
        = corpusManifest({sdpaEntry("a", 1, 4096, "kernel", "pack", "decode_long")}, sdpaContext());

    const auto& row = manifest["graphs"][0];
    EXPECT_EQ(row["phase"], "decode");
    EXPECT_EQ(row["context"], "long");
    EXPECT_EQ(row["regime"], "decode_long");
}

TEST(TestCorpusManifest, AnOperationDeclaringNoPopulationsGetsNoFacetColumns)
{
    auto entry = sdpaEntry("a", 1, 4096, "kernel", "pack", "");
    entry.regimeAxes = {};

    const auto manifest = corpusManifest({entry}, sdpaContext());
    const auto& row = manifest["graphs"][0];
    EXPECT_FALSE(row.contains("phase"));
    EXPECT_EQ(row["regime"], "");

    const auto rows = parseCsv(corpusManifestCsv({entry}));
    EXPECT_EQ(rows.front()[2], "regime");
    EXPECT_EQ(rows.front()[3], "source");
}

TEST(TestCorpusManifest, TheCsvHeaderAndItsRowsAgreeColumnForColumn)
{
    // A mismatched header would silently transpose features.
    const std::vector<ManifestEntry> entries{
        sdpaEntry("a", 1, 4096, "kernel", "pack.kdp.json", "decode_long"),
        sdpaEntry("b", 2048, 2048, "sweep", "draw 3", "prefill_short")};

    const auto rows = parseCsv(corpusManifestCsv(entries));
    ASSERT_EQ(rows.size(), 3u);
    for(const auto& row : rows)
    {
        EXPECT_EQ(row.size(), rows.front().size());
    }

    const auto& header = rows.front();
    const auto column = [&header](const std::string& name) {
        return static_cast<size_t>(std::find(header.begin(), header.end(), name) - header.begin());
    };
    ASSERT_LT(column("q.seqlen_k"), header.size());
    EXPECT_EQ(rows[1][column("q.seqlen_k")], "4096");
    EXPECT_EQ(rows[2][column("q.seqlen_k")], "2048");
    EXPECT_EQ(rows[1][column("phase")], "decode");
    EXPECT_EQ(rows[2][column("phase")], "prefill");
    EXPECT_EQ(rows[1][column("op")], "sdpa_fwd");
}

TEST(TestCorpusManifest, TheCsvAndTheJsonCarryTheSameValuesForEveryRow)
{
    // The two manifests come from separate traversals.
    const auto context = sdpaContext();
    const std::vector<ManifestEntry> entries{
        sdpaEntry("a", 1, 4096, "kernel", "pack.kdp.json", "decode_long"),
        sdpaEntry("b", 2048, 2048, "sweep", "draw 3", "prefill_short")};

    const auto manifest = corpusManifest(entries, context);
    const auto rows = parseCsv(corpusManifestCsv(entries));
    ASSERT_EQ(rows.size(), manifest["graphs"].size() + 1);

    const auto& header = rows.front();
    for(size_t i = 0; i < manifest["graphs"].size(); ++i)
    {
        const auto& record = manifest["graphs"][i];
        for(size_t column = 0; column < header.size(); ++column)
        {
            if(header[column] == "bytes")
            {
                EXPECT_EQ(rows[i + 1][column], std::to_string(record["bytes"].get<int64_t>()));
                continue;
            }
            ASSERT_TRUE(record.contains(header[column])) << header[column];
            EXPECT_EQ(rows[i + 1][column], record[header[column]].get<std::string>())
                << header[column];
        }
    }
}

TEST(TestCorpusManifest, AnOriginCarryingACommaDoesNotShiftTheRow)
{
    // `origin` is free text, so it is the column an unquoted writer would break.
    auto entry = sdpaEntry("a", 1, 4096, "kernel", "pack.kdp.json, 6 kernels", "decode_long");
    const auto rows = parseCsv(corpusManifestCsv({entry}));

    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[1].size(), rows[0].size());
    const auto& header = rows.front();
    const auto column
        = static_cast<size_t>(std::find(header.begin(), header.end(), "origin") - header.begin());
    EXPECT_EQ(rows[1][column], "pack.kdp.json, 6 kernels");
}

TEST(TestCorpusManifest, TheTotalsReportWhatWasEmittedAndWhatWasAskedForSeparately)
{
    // Shortfall is reported, never filled (RFC 0019.13).
    auto context = sdpaContext();
    context.requested = 4;

    const auto manifest
        = corpusManifest({sdpaEntry("a", 1, 4096, "kernel", "pack", "decode_long"),
                          sdpaEntry("b", 2048, 2048, "sweep", "draw 3", "prefill_short")},
                         context);

    EXPECT_EQ(manifest["requested"], 4);
    EXPECT_EQ(manifest["emitted"], 2);
    EXPECT_EQ(manifest["mix"]["kernel"], 1);
    EXPECT_EQ(manifest["mix"]["sweep"], 1);
    EXPECT_EQ(manifest["regimes"]["decode_long"], 1);
    EXPECT_EQ(manifest["tool"], "corpus_gen");
    ASSERT_EQ(manifest["operations"].size(), 1u);
    EXPECT_EQ(manifest["operations"][0], "sdpa_fwd");
    EXPECT_EQ(manifest["seed"], 7);
}

TEST(TestCorpusManifest, TwoOperationsShareOneHeaderAndLeaveEachOthersColumnsEmpty)
{
    // One manifest may cover several operations: the header is the union of their columns and
    // cells a row cannot fill are empty.
    auto other = sdpaEntry("b", 2048, 2048, "sweep", "draw 3", "");
    other.operation = "conv_fwd";
    other.regimeAxes = {};
    other.entry.point = ProblemPoint{{"channels", int64_t{64}}};

    const auto rows = parseCsv(
        corpusManifestCsv({sdpaEntry("a", 1, 4096, "kernel", "pack", "decode_long"), other}));

    ASSERT_EQ(rows.size(), 3u);
    const auto& header = rows.front();
    const auto column = [&header](const std::string& name) {
        return static_cast<size_t>(std::find(header.begin(), header.end(), name) - header.begin());
    };
    ASSERT_LT(column("q.seqlen_k"), header.size());
    ASSERT_LT(column("q.channels"), header.size());

    EXPECT_EQ(rows[1][column("op")], "sdpa_fwd");
    EXPECT_EQ(rows[1][column("q.seqlen_k")], "4096");
    EXPECT_EQ(rows[1][column("q.channels")], "");
    EXPECT_EQ(rows[2][column("op")], "conv_fwd");
    EXPECT_EQ(rows[2][column("q.seqlen_k")], "");
    EXPECT_EQ(rows[2][column("q.channels")], "64");

    // Facet columns are not borrowed by an operation that never declared them.
    ASSERT_LT(column("phase"), header.size());
    EXPECT_EQ(rows[1][column("phase")], "decode");
    EXPECT_EQ(rows[2][column("phase")], "");
}
