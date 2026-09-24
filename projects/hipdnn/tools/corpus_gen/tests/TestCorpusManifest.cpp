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
/// What is being protected here is a contract with code outside this tree:
/// `uhd_gen/reproduce/score_predictions.py:43-45` and `compare_engines.py:46-49` index
/// `manifest["graphs"]` and read `benchmark`, `name` and `regime` off each row. The port from
/// the retired Python assembler is only correct if those survive it, so they are
/// asserted by name rather than left to a round-trip test that would pass under any renaming.

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
    entry.entry.point  = sdpaPoint(seqlenQ, seqlenK);
    entry.entry.source = source;
    entry.entry.origin = origin;
    entry.entry.regime = regime;
    entry.benchmark    = graphIdentity(reinterpret_cast<const uint8_t*>(name.data()), name.size());
    entry.name         = name;
    entry.file         = "graphs/" + name + ".fb";
    entry.bytes        = 4096;
    entry.operation    = "sdpa_fwd";
    entry.regimeAxes   = shippedSdpaAxes();
    return entry;
}

ManifestContext sdpaContext()
{
    ManifestContext context;
    context.operations = {"sdpa_fwd"};
    context.seed       = 7;
    context.requested  = 4;
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
    // score_predictions.py:43-45 and compare_engines.py:46-49 read exactly these off each row
    // of manifest["graphs"]. Renaming any of them does not fail -- both scripts fall back to
    // reporting the raw graph id -- so the break would be a quietly worse report, not an error.
    const auto manifest
        = corpusManifest({sdpaEntry("a", 2048, 2048, "sweep", "draw 0", "prefill_short")},
                         sdpaContext());

    ASSERT_EQ(manifest["graphs"].size(), 1u);
    const auto& row = manifest["graphs"][0];
    EXPECT_TRUE(row.contains("benchmark"));
    EXPECT_EQ(row["name"], "a");
    EXPECT_EQ(row["regime"], "prefill_short");
}

TEST(TestCorpusManifest, TheProblemColumnsComeFromThePointRatherThanFromAnOperationsFieldNames)
{
    // The Python emitted `dtype`, `batch`, `heads_q`, ... because it knew it was writing SDPA.
    // Here they are the point's own parameters under the `q.` prefix uhd_gen's feature hash
    // expects, which is the single edit that makes the manifest op-general.
    const auto manifest
        = corpusManifest({sdpaEntry("a", 1, 4096, "kernel", "pack.kdp.json", "decode_long")},
                         sdpaContext());

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
    // `phase` and `context` were columns in the Python because those three facet names were
    // compiled in. Splitting them back out of `regime` is not available: a declared label may
    // itself contain the separator.
    const auto manifest
        = corpusManifest({sdpaEntry("a", 1, 4096, "kernel", "pack", "decode_long")},
                         sdpaContext());

    const auto& row = manifest["graphs"][0];
    EXPECT_EQ(row["phase"], "decode");
    EXPECT_EQ(row["context"], "long");
    EXPECT_EQ(row["regime"], "decode_long");
}

TEST(TestCorpusManifest, AnOperationDeclaringNoPopulationsGetsNoFacetColumns)
{
    // Coverage is "whatever has a declaration", so a declaration without a regime_label block
    // must still produce a manifest -- with no invented column and no invented label.
    auto entry        = sdpaEntry("a", 1, 4096, "kernel", "pack", "");
    entry.regimeAxes  = {};

    const auto manifest = corpusManifest({entry}, sdpaContext());
    const auto& row     = manifest["graphs"][0];
    EXPECT_FALSE(row.contains("phase"));
    EXPECT_EQ(row["regime"], "");

    const auto rows = parseCsv(corpusManifestCsv({entry}));
    EXPECT_EQ(rows.front()[2], "regime");
    EXPECT_EQ(rows.front()[3], "source");
}

TEST(TestCorpusManifest, TheCsvHeaderAndItsRowsAgreeColumnForColumn)
{
    // The failure this forbids is silent: a header that disagrees with its rows transposes two
    // features, and a model trains on the wrong ones without anything reporting an error.
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
    const auto column  = [&header](const std::string& name) {
        return static_cast<size_t>(
            std::find(header.begin(), header.end(), name) - header.begin());
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
    // Two manifests, one content (assemble.py:write). They are generated from separate
    // traversals, so agreement is asserted rather than assumed.
    const auto context = sdpaContext();
    const std::vector<ManifestEntry> entries{
        sdpaEntry("a", 1, 4096, "kernel", "pack.kdp.json", "decode_long"),
        sdpaEntry("b", 2048, 2048, "sweep", "draw 3", "prefill_short")};

    const auto manifest = corpusManifest(entries, context);
    const auto rows     = parseCsv(corpusManifestCsv(entries));
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
    // `origin` is free text by design -- a pack file and its kernel count, a model name, a draw
    // index -- so it is the one column an unquoted writer would silently break.
    auto entry   = sdpaEntry("a", 1, 4096, "kernel", "pack.kdp.json, 6 kernels", "decode_long");
    const auto rows = parseCsv(corpusManifestCsv({entry}));

    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[1].size(), rows[0].size());
    const auto& header = rows.front();
    const auto column  = static_cast<size_t>(
        std::find(header.begin(), header.end(), "origin") - header.begin());
    EXPECT_EQ(rows[1][column], "pack.kdp.json, 6 kernels");
}

TEST(TestCorpusManifest, TheTotalsReportWhatWasEmittedAndWhatWasAskedForSeparately)
{
    // Shortfall is reported, never filled (RFC 0019.13): a corpus of two problems from a
    // source that has two is complete, and inventing two more would be the defect.
    auto context      = sdpaContext();
    context.requested = 4;

    const auto manifest = corpusManifest(
        {sdpaEntry("a", 1, 4096, "kernel", "pack", "decode_long"),
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
    // One corpus may cover several operations -- the consumers read a single
    // `corpus/manifest.json` -- and their parameters do not agree. A row is not wrong about its
    // own columns just because another operation has some it does not, so the header is the
    // union and the cells an entry cannot fill are empty rather than absent.
    auto other = sdpaEntry("b", 2048, 2048, "sweep", "draw 3", "");
    other.operation  = "conv_fwd";
    other.regimeAxes = {};
    other.entry.point = ProblemPoint{{"channels", int64_t{64}}};

    const auto rows = parseCsv(
        corpusManifestCsv({sdpaEntry("a", 1, 4096, "kernel", "pack", "decode_long"), other}));

    ASSERT_EQ(rows.size(), 3u);
    const auto& header = rows.front();
    const auto column  = [&header](const std::string& name) {
        return static_cast<size_t>(std::find(header.begin(), header.end(), name)
                                   - header.begin());
    };
    ASSERT_LT(column("q.seqlen_k"), header.size());
    ASSERT_LT(column("q.channels"), header.size());

    // Each row carries its own operation's values and nothing of the other's.
    EXPECT_EQ(rows[1][column("op")], "sdpa_fwd");
    EXPECT_EQ(rows[1][column("q.seqlen_k")], "4096");
    EXPECT_EQ(rows[1][column("q.channels")], "");
    EXPECT_EQ(rows[2][column("op")], "conv_fwd");
    EXPECT_EQ(rows[2][column("q.seqlen_k")], "");
    EXPECT_EQ(rows[2][column("q.channels")], "64");

    // The facet columns are the first operation's alone, and the second leaves them empty
    // rather than borrowing a label from axes it never declared.
    ASSERT_LT(column("phase"), header.size());
    EXPECT_EQ(rows[1][column("phase")], "decode");
    EXPECT_EQ(rows[2][column("phase")], "");
}
