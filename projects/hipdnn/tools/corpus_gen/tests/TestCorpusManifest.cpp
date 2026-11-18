// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/CorpusManifest.hpp>
#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

/// @file TestCorpusManifest.cpp
/// @brief The record that makes a corpus joinable to its measurements.
///
/// `uhd_gen/reproduce/compare_engines.py` reads `benchmark`, `name` and `regime` from
/// `manifest["graphs"]`, so those keys are asserted by name.

using namespace hipdnn_corpus_gen;

namespace
{

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
    entry.benchmark = "key-" + name;
    entry.name = name;
    entry.file = "graphs/" + name + ".fb";
    entry.bytes = 4096;
    entry.operation = "sdpa_fwd";
    entry.regimeAxes = test::shippedOperation("sdpa_fwd").regimeLabel;
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

/// The index of column @p name in @p header, or header.size() when it is not there.
size_t columnOf(const std::vector<std::string>& header, const std::string& name)
{
    return static_cast<size_t>(std::find(header.begin(), header.end(), name) - header.begin());
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
    ASSERT_LT(columnOf(header, "q.seqlen_k"), header.size());
    EXPECT_EQ(rows[1][columnOf(header, "q.seqlen_k")], "4096");
    EXPECT_EQ(rows[2][columnOf(header, "q.seqlen_k")], "2048");
    EXPECT_EQ(rows[1][columnOf(header, "phase")], "decode");
    EXPECT_EQ(rows[2][columnOf(header, "phase")], "prefill");
    EXPECT_EQ(rows[1][columnOf(header, "op")], "sdpa_fwd");
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

/// Each shipped attention declaration, whose manifest carries one row per form it declares.
class TestCorpusManifestForms : public ::testing::TestWithParam<std::string>
{
};

TEST_P(TestCorpusManifestForms, AFormIsAFacetColumnAfterTheRegimeFacetsAndNotPartOfTheRegime)
{
    const auto metadata = test::shippedOperation(GetParam());
    const auto* declared = metadata.find("form");
    ASSERT_NE(declared, nullptr);
    const auto& forms = declared->values;
    ASSERT_FALSE(forms.empty());

    std::vector<ManifestEntry> entries;
    for(const auto& form : forms)
    {
        auto entry = sdpaEntry(form, 2048, 2048, "sweep", "draw " + form, "prefill_short");
        entry.operation = GetParam();
        entry.regimeAxes = metadata.regimeLabel;
        entry.formAxes = metadata.formLabel;
        entry.entry.point["form"] = form;
        entries.push_back(std::move(entry));
    }

    const auto manifest = corpusManifest(entries, sdpaContext());
    ASSERT_EQ(manifest["graphs"].size(), forms.size());
    for(size_t i = 0; i < forms.size(); ++i)
    {
        EXPECT_EQ(manifest["graphs"][i]["form"], forms[i]);
        // The regime is what it was: a form is a column of its own, not a fourth facet of the
        // label.
        EXPECT_EQ(manifest["graphs"][i]["regime"], "prefill_short") << forms[i];
        EXPECT_EQ(manifest["graphs"][i]["grouping"], "mha") << forms[i];
    }

    const auto rows = parseCsv(corpusManifestCsv(entries));
    ASSERT_EQ(rows.size(), forms.size() + 1);
    const auto& header = rows.front();
    ASSERT_LT(columnOf(header, "form"), header.size());
    EXPECT_EQ(columnOf(header, "form"), columnOf(header, "grouping") + 1)
        << "form follows the last regime facet";
    for(size_t i = 0; i < forms.size(); ++i)
    {
        EXPECT_EQ(rows[i + 1][columnOf(header, "form")], forms[i]);
        EXPECT_EQ(rows[i + 1][columnOf(header, "regime")], "prefill_short") << forms[i];
    }
    for(const auto& row : rows)
    {
        EXPECT_EQ(row.size(), header.size());
    }
}

INSTANTIATE_TEST_SUITE_P(SdpaDeclarations,
                         TestCorpusManifestForms,
                         ::testing::Values("sdpa_fwd", "sdpa_bwd"));

TEST(TestCorpusManifest, AnOriginCarryingACommaDoesNotShiftTheRow)
{
    // `origin` is free text, so it is the column an unquoted writer would break.
    auto entry = sdpaEntry("a", 1, 4096, "kernel", "pack.kdp.json, 6 kernels", "decode_long");
    const auto rows = parseCsv(corpusManifestCsv({entry}));

    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[1].size(), rows[0].size());
    const auto& header = rows.front();
    EXPECT_EQ(rows[1][columnOf(header, "origin")], "pack.kdp.json, 6 kernels");
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
    ASSERT_LT(columnOf(header, "q.seqlen_k"), header.size());
    ASSERT_LT(columnOf(header, "q.channels"), header.size());

    EXPECT_EQ(rows[1][columnOf(header, "op")], "sdpa_fwd");
    EXPECT_EQ(rows[1][columnOf(header, "q.seqlen_k")], "4096");
    EXPECT_EQ(rows[1][columnOf(header, "q.channels")], "");
    EXPECT_EQ(rows[2][columnOf(header, "op")], "conv_fwd");
    EXPECT_EQ(rows[2][columnOf(header, "q.seqlen_k")], "");
    EXPECT_EQ(rows[2][columnOf(header, "q.channels")], "64");

    // Facet columns are not borrowed by an operation that never declared them.
    ASSERT_LT(columnOf(header, "phase"), header.size());
    EXPECT_EQ(rows[1][columnOf(header, "phase")], "decode");
    EXPECT_EQ(rows[2][columnOf(header, "phase")], "");
}

namespace
{

/// A corpus directory of its own, with a graphs/ folder, removed when the test ends.
struct ScratchDirectory
{
    hipdnn_test_sdk::utilities::ScopedDirectory claimed;
    std::filesystem::path path;
    explicit ScratchDirectory(const std::string& name)
        : claimed(hipdnn_test_sdk::utilities::claimScratchDirectory("corpus_gen_" + name))
        , path(claimed.path())
    {
        std::filesystem::create_directories(path / "graphs");
    }
};

std::vector<uint8_t> builtSdpaGraph(int64_t seqlenQ)
{
    const auto metadata = test::shippedOperation("sdpa_fwd");
    auto point = sdpaPoint(seqlenQ, 4096);
    point["alignment"] = std::string("top_left");
    point["generate_stats"] = false;
    point["form"] = std::string("plain");
    const auto built = buildGraphFor(metadata, point);
    EXPECT_TRUE(built.ok()) << built.error;
    return built.bytes;
}

void writeJson(const std::filesystem::path& path, const nlohmann::json& document)
{
    std::ofstream(path) << document.dump();
}

} // namespace

TEST(TestCorpusManifest, AnExcludedCorpusIsHeldOutByTheContentKeysOfItsGraphFiles)
{
    // Whatever the manifest recorded as `benchmark`, the files say which graphs it holds; the
    // same graph under another name is still held out.
    const ScratchDirectory corpus("excluded");
    const auto graph = stampGraphIdentity(builtSdpaGraph(64), "an_old_name");
    std::ofstream(corpus.path / "graphs" / "old.fb", std::ios::binary)
        .write(reinterpret_cast<const char*>(graph.bytes.data()),
               static_cast<std::streamsize>(graph.bytes.size()));
    writeJson(corpus.path / "manifest.json",
              {{"graphs", {{{"benchmark", "anything"}, {"file", "graphs/old.fb"}}}}});

    const auto keys = excludedGraphKeys({corpus.path / "manifest.json"});
    EXPECT_EQ(keys, std::set<std::string>{graph.key});
    EXPECT_EQ(keys.count(stampGraphIdentity(builtSdpaGraph(64), "a_new_name").key), 1U)
        << "the same graph under a new name is still held out";
}

TEST(TestCorpusManifest, AnExcludedCorpusMissingAGraphFileIsAnErrorNotAnEmptySet)
{
    const ScratchDirectory corpus("excluded-missing");
    writeJson(corpus.path / "manifest.json",
              {{"graphs", {{{"benchmark", "an-id"}, {"file", "graphs/gone.fb"}}}}});
    EXPECT_THROW(excludedGraphKeys({corpus.path / "manifest.json"}), std::runtime_error);
}

TEST(TestCorpusManifest, AManifestThatCannotBeWrittenIsAnErrorNotASmallerCorpus)
{
    // A directory in the manifest's place makes the write fail; a run that went on would leave
    // graphs and commands that no manifest lists.
    const ScratchDirectory corpus("manifest-unwritable");
    std::filesystem::create_directory(corpus.path / "manifest.json");

    EXPECT_THROW(writeCorpusManifest(corpus.path, {}, ManifestContext{}), std::runtime_error);
}
