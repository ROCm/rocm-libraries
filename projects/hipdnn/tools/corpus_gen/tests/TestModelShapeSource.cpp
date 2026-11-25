// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/ModelShapeSource.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

/// @file TestModelShapeSource.cpp
/// @brief The model pool: the shapes somebody recorded a real workload running.
///
/// Rows for another operation and rows the declaration cannot express are counted apart, and
/// the first unusable row is explained.

using namespace hipdnn_corpus_gen;

namespace
{

/// A scratch tree of its own under the working directory: /tmp is not always writable in
/// containers.
class TempTree
{
public:
    explicit TempTree(const std::string& name)
        : _dir(hipdnn_test_sdk::utilities::claimScratchDirectoryUnder(
              std::filesystem::current_path(), "shapes_test_" + name))
        , _root(_dir.path())
    {
    }

    std::filesystem::path write(const std::string& name, const std::string& text) const
    {
        const auto path = _root / name;
        std::ofstream file(path);
        file << text;
        return path;
    }

private:
    hipdnn_test_sdk::utilities::ScopedDirectory _dir;
    std::filesystem::path _root;
};

OperationMetadata shippedSdpa()
{
    return test::shippedOperation("sdpa_fwd");
}

/// A complete header in the spelling the miner emits, and one complete row.
const char* kHeader = "name,q.batch,q.heads,q.heads_kv,q.seqlen_q,q.seqlen_k,q.head_dim,"
                      "q.is_causal,q.alignment,q.generate_stats,q.dtype,q.form\n";
const char* kRow = "llama3-70b prefill,1,64,8,4096,4096,128,true,top_left,false,bf16,plain\n";

/// kHeader and kRow with @p dropped columns (by header name) cut out of both.
std::string csvWithout(std::initializer_list<const char*> dropped)
{
    const auto fields = [](const std::string& line) {
        std::vector<std::string> parts;
        std::istringstream stream(line.substr(0, line.find('\n')));
        for(std::string part; std::getline(stream, part, ',');)
        {
            parts.push_back(part);
        }
        return parts;
    };
    const auto names = fields(kHeader);
    const auto values = fields(kRow);
    std::string header;
    std::string row;
    for(size_t i = 0; i < names.size(); ++i)
    {
        if(std::find(dropped.begin(), dropped.end(), names[i]) != dropped.end())
        {
            continue;
        }
        header += (header.empty() ? "" : ",") + names[i];
        row += (row.empty() ? "" : ",") + values[i];
    }
    return header + "\n" + row + "\n";
}

} // namespace

TEST(TestModelShapeSource, ARecordedShapeBecomesAPointCarryingItsNameAndRegime)
{
    const TempTree tree("named");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const auto entries
        = readModelShapes(metadata, tree.write("m.csv", kHeader + std::string(kRow)), report);

    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries.front().source, "model");
    EXPECT_EQ(entries.front().point.at("heads_kv"), ParameterValue{int64_t{8}});
    EXPECT_EQ(entries.front().point.at("is_causal"), ParameterValue{true});
    EXPECT_EQ(entries.front().point.at("dtype"), ParameterValue{std::string("bf16")});
    // The workload name is what explains the row in a later audit.
    EXPECT_NE(entries.front().origin.find("llama3-70b prefill"), std::string::npos)
        << entries.front().origin;
    EXPECT_FALSE(entries.front().regime.empty());
    EXPECT_EQ(report.rows, 1);
    EXPECT_EQ(report.unusable, 0);
}

TEST(TestModelShapeSource, AColumnWithoutTheQueryPrefixStillReads)
{
    const TempTree tree("bare");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const std::string bare = "batch,heads,heads_kv,seqlen_q,seqlen_k,head_dim,is_causal,"
                             "alignment,generate_stats,dtype,form\n"
                             "1,64,8,4096,4096,128,true,top_left,false,bf16,plain\n";

    const auto entries = readModelShapes(metadata, tree.write("m.csv", bare), report);
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_NE(entries.front().origin.find("row 1"), std::string::npos) << entries.front().origin;
}

// A file mined before `form` existed has no column for it and reads as the plain form, for the
// backward operation as for the forward one. A present column left empty is still a row that
// failed to record its form.
TEST(TestModelShapeSource, AFilePredatingTheFormColumnReadsAsPlain)
{
    const TempTree tree("predates");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const auto entries
        = readModelShapes(metadata, tree.write("old.csv", csvWithout({"q.form"})), report);
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries.front().point.at("form"), ParameterValue{std::string("plain")});

    // sdpa_bwd declares no kernel_catalog, so nothing but the form rule can answer for it.
    const auto backward = test::shippedOperation("sdpa_bwd");
    const auto backwardEntries = readModelShapes(
        backward, tree.write("old_bwd.csv", csvWithout({"q.form", "q.generate_stats"})), report);
    ASSERT_EQ(backwardEntries.size(), 1u) << report.firstProblem;
    EXPECT_EQ(backwardEntries.front().point.at("form"), ParameterValue{std::string("plain")});

    const std::string empty
        = kHeader
          + std::string("llama3-70b prefill,1,64,8,4096,4096,128,true,top_left,false,bf16,\n");
    EXPECT_TRUE(readModelShapes(metadata, tree.write("empty.csv", empty), report).empty());
    EXPECT_EQ(report.unusable, 1);
    EXPECT_NE(report.firstProblem.find("form"), std::string::npos) << report.firstProblem;
}

// Only `form` is exempt: a file with no (or a misspelled) `q.alignment` or `q.generate_stats`
// column would otherwise read as top_left / no stats, passing off an unrecorded shape as recorded.
TEST(TestModelShapeSource, AFileMissingAnyOtherParameterColumnIsRefused)
{
    const TempTree tree("missing_column");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    EXPECT_TRUE(readModelShapes(
                    metadata, tree.write("no_alignment.csv", csvWithout({"q.alignment"})), report)
                    .empty());
    EXPECT_EQ(report.unusable, 1);
    EXPECT_NE(report.firstProblem.find("alignment"), std::string::npos) << report.firstProblem;

    EXPECT_TRUE(readModelShapes(
                    metadata, tree.write("no_stats.csv", csvWithout({"q.generate_stats"})), report)
                    .empty());
    EXPECT_EQ(report.unusable, 1);
    EXPECT_NE(report.firstProblem.find("generate_stats"), std::string::npos) << report.firstProblem;
}

TEST(TestModelShapeSource, ARowNamingAnotherOperationIsSkippedRatherThanCountedUnusable)
{
    const TempTree tree("other_op");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const std::string mixed = "op," + std::string(kHeader).substr(std::string("name,").size())
                              + "gemm,1,64,8,4096,4096,128,true,top_left,false,bf16,plain\n"
                                "sdpa_fwd,1,64,8,4096,4096,128,true,top_left,false,bf16,plain\n";

    const auto entries = readModelShapes(metadata, tree.write("m.csv", mixed), report);

    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(report.rows, 2);
    // Several operations per mined file is normal, not `unusable`.
    EXPECT_EQ(report.otherOperation, 1);
    EXPECT_EQ(report.unusable, 0);
}

TEST(TestModelShapeSource, AMisspelledColumnLosesEveryRowAndSaysWhichName)
{
    const TempTree tree("misspelled");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    // `causal` for `is_causal`.
    const std::string wrong = "q.batch,q.heads,q.heads_kv,q.seqlen_q,q.seqlen_k,q.head_dim,"
                              "q.causal,q.alignment,q.generate_stats,q.dtype,q.form\n"
                              "1,64,8,4096,4096,128,true,top_left,false,bf16,plain\n"
                              "2,64,8,2048,2048,128,true,top_left,false,bf16,plain\n";

    const auto entries = readModelShapes(metadata, tree.write("m.csv", wrong), report);

    EXPECT_TRUE(entries.empty());
    EXPECT_EQ(report.rows, 2);
    EXPECT_EQ(report.unusable, 2);
    EXPECT_NE(report.firstProblem.find("is_causal"), std::string::npos) << report.firstProblem;
}

TEST(TestModelShapeSource, AValueTheDeclarationDoesNotAcceptIsRefusedRatherThanCoerced)
{
    const TempTree tree("bad_value");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const std::string bad
        = kHeader
          + std::string("mistyped,1,64,8,4096,4096,128,true,top_left,false,fp8_e4m3\n"
                        "not a number,1,64,8,4096,4096,many,true,top_left,false,bf16,plain\n");

    const auto entries = readModelShapes(metadata, tree.write("m.csv", bad), report);

    EXPECT_TRUE(entries.empty());
    EXPECT_EQ(report.unusable, 2);
    // The first row is complete, so it is refused for its dtype, not for a missing field.
    EXPECT_NE(report.firstProblem.find("'dtype' is not a declared value: 'fp8_e4m3'"),
              std::string::npos)
        << report.firstProblem;
}

TEST(TestModelShapeSource, AQuotedFieldMayCarryACommaWithoutSplittingTheRow)
{
    const TempTree tree("quoted");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const std::string quoted
        = kHeader
          + std::string(
              R"("llama3-70b, prefill",1,64,8,4096,4096,128,true,top_left,false,bf16,plain)")
          + "\n";

    const auto entries = readModelShapes(metadata, tree.write("m.csv", quoted), report);

    ASSERT_EQ(entries.size(), 1u);
    EXPECT_NE(entries.front().origin.find("llama3-70b, prefill"), std::string::npos)
        << entries.front().origin;
}

TEST(TestModelShapeSource, AFileThatCannotBeReadReportsItRatherThanLookingEmpty)
{
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const auto entries = readModelShapes(metadata, "no_such_shapes_file.csv", report);

    EXPECT_TRUE(entries.empty());
    EXPECT_EQ(report.rows, 0);
    EXPECT_FALSE(report.firstProblem.empty());
}

TEST(TestModelShapeSource, NoOracleIsAppliedSoAShapeNobodyServesStillArrives)
{
    const TempTree tree("unserved");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    // Far past any byte budget; admission is the caller's job.
    const std::string huge
        = kHeader
          + std::string("enormous,64,128,128,131072,131072,256,false,top_left,false,bf16,plain\n");

    const auto entries = readModelShapes(metadata, tree.write("m.csv", huge), report);

    EXPECT_EQ(entries.size(), 1u);
    EXPECT_EQ(report.unusable, 0);
}

namespace
{

/// Each shipped attention declaration (forward and backward), whose defaulted parameters differ.
class TestShippedModelShapeDefaults : public ::testing::TestWithParam<std::string>
{
};

} // namespace

TEST_P(TestShippedModelShapeDefaults,
       AFileThatPredatesAParameterReadsItAtItsDeclaredDefaultAndSaysSo)
{
    const TempTree tree("predates_" + GetParam());
    const bool forward = GetParam() == "sdpa_fwd";
    const auto metadata = test::shippedOperation(GetParam());
    ModelShapeReport report;

    // kHeader has no layout, scale or window column: a file mined before they were declared. The
    // backward declares no window and no generate_stats, and its file predates the form too.
    const auto text
        = forward ? kHeader + std::string(kRow) : csvWithout({"q.form", "q.generate_stats"});
    const auto entries = readModelShapes(metadata, tree.write("m.csv", text), report);

    ASSERT_EQ(entries.size(), 1u) << report.firstProblem;
    EXPECT_EQ(report.unusable, 0);
    EXPECT_EQ(entries.front().point.at("layout"), ParameterValue{std::string("bshd")});
    EXPECT_EQ(entries.front().point.at("scale"), ParameterValue{std::string("inv_sqrt_d")});
    EXPECT_EQ(entries.front().point.at("form"), ParameterValue{std::string("plain")});
    if(forward)
    {
        EXPECT_EQ(entries.front().point.at("window"), ParameterValue{std::string("none")});
    }
    else
    {
        EXPECT_EQ(entries.front().point.count("window"), 0U);
    }
    // The manifest can say which columns were filled in, and for how many rows.
    EXPECT_EQ(report.defaulted,
              forward ? (std::map<std::string, int64_t>{{"layout", 1}, {"scale", 1}, {"window", 1}})
                      : (std::map<std::string, int64_t>{{"form", 1}, {"layout", 1}, {"scale", 1}}));
}

INSTANTIATE_TEST_SUITE_P(SdpaDeclarations,
                         TestShippedModelShapeDefaults,
                         ::testing::Values("sdpa_fwd", "sdpa_bwd"));

TEST(TestModelShapeSource, AFileMissingAColumnWithNoDeclaredDefaultIsStillRefused)
{
    const TempTree tree("no_default");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    // No q.alignment, and sdpa_fwd's pack constants answer for alignment (top_left) and
    // generate_stats (false): a recorded KV-cache decode is bottom_right, so filling the pack's
    // answer in would pass a different problem off as the recorded one.
    const std::string noAlignment
        = "q.batch,q.heads,q.heads_kv,q.seqlen_q,q.seqlen_k,q.head_dim,q.is_causal,q.dtype\n"
          "4,32,8,1,4096,128,true,bf16\n";
    const auto entries = readModelShapes(metadata, tree.write("m.csv", noAlignment), report);

    EXPECT_TRUE(entries.empty());
    EXPECT_EQ(report.unusable, 1);
    EXPECT_NE(report.firstProblem.find("no value for 'alignment'"), std::string::npos)
        << report.firstProblem;
    EXPECT_TRUE(report.defaulted.empty());
}

TEST(TestModelShapeSource, AColumnThatIsThereIsReadNotReplacedByTheConstant)
{
    const TempTree tree("present");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const std::string withLayout
        = "q.batch,q.heads,q.heads_kv,q.seqlen_q,q.seqlen_k,q.head_dim,q.is_causal,q.alignment,"
          "q.generate_stats,q.dtype,q.layout\n"
          "1,64,8,4096,4096,128,true,top_left,false,bf16,kv_bhsd\n";

    const auto entries = readModelShapes(metadata, tree.write("m.csv", withLayout), report);

    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries.front().point.at("layout"), ParameterValue{std::string("kv_bhsd")});
}

TEST(TestModelShapeSource, AnEmptyCellInAColumnThatExistsIsStillRefused)
{
    const TempTree tree("empty_cell");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    // The constant answers for a file with no such column, not for a row that left it blank.
    const std::string blank
        = "q.batch,q.heads,q.heads_kv,q.seqlen_q,q.seqlen_k,q.head_dim,q.is_causal,q.alignment,"
          "q.generate_stats,q.dtype,q.layout\n"
          "1,64,8,4096,4096,128,true,top_left,false,bf16,\n";

    const auto entries = readModelShapes(metadata, tree.write("m.csv", blank), report);

    EXPECT_TRUE(entries.empty());
    EXPECT_EQ(report.unusable, 1);
    EXPECT_NE(report.firstProblem.find("no value for 'layout'"), std::string::npos)
        << report.firstProblem;
}
