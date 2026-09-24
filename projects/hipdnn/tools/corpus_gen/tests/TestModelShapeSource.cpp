// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/ModelShapeSource.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>

/// @file TestModelShapeSource.cpp
/// @brief The model pool: the shapes somebody recorded a real workload running.
///
/// The value of this pool is not that it adds points -- the sweep has more than anyone can
/// measure -- it is that it adds the *named* ones, and that a miner emitting a column spelling
/// the declaration does not know loses every row. So what is pinned here is the accounting: a
/// row for another operation and a row the declaration cannot express are different findings,
/// counted apart, and the first unusable row says why in words.

using namespace hipdnn_corpus_gen;

namespace
{

/// A scratch tree under the test's own working directory rather than under TMPDIR, for the same
/// reason `TestKernelCatalogSource` does: these runs happen inside containers where /tmp is not
/// always writable, and a test that cannot write is indistinguishable from one that failed.
class TempTree
{
public:
    explicit TempTree(const std::string& name)
        : _root(std::filesystem::current_path() / ("shapes_test_" + name))
    {
        std::error_code ignored;
        std::filesystem::remove_all(_root, ignored);
        std::filesystem::create_directories(_root, ignored);
    }
    TempTree(const TempTree&) = delete;
    TempTree& operator=(const TempTree&) = delete;
    ~TempTree()
    {
        std::error_code ignored;
        std::filesystem::remove_all(_root, ignored);
    }

    std::filesystem::path write(const std::string& name, const std::string& text) const
    {
        const auto path = _root / name;
        std::ofstream file(path);
        file << text;
        return path;
    }

private:
    std::filesystem::path _root;
};

OperationMetadata shippedSdpa()
{
    const std::string path
        = std::string(HIPDNN_CORPUS_GEN_OPERATIONS_DIR) + "/sdpa_fwd.opmeta.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());
    return parsed.ok() ? *parsed.metadata : OperationMetadata{};
}

/// A complete header in the spelling the miner emits, and one complete row.
const char* kHeader = "name,q.batch,q.heads,q.heads_kv,q.seqlen_q,q.seqlen_k,q.head_dim,"
                      "q.is_causal,q.alignment,q.dtype\n";
const char* kRow    = "llama3-70b prefill,1,64,8,4096,4096,128,true,top_left,bf16\n";

} // namespace

TEST(TestModelShapeSource, ARecordedShapeBecomesAPointCarryingItsNameAndRegime)
{
    const TempTree tree("named");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const auto entries = readModelShapes(metadata, tree.write("m.csv", kHeader + std::string(kRow)),
                                         report);

    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries.front().source, "model");
    EXPECT_EQ(entries.front().point.at("heads_kv"), ParameterValue{int64_t{8}});
    EXPECT_EQ(entries.front().point.at("is_causal"), ParameterValue{true});
    EXPECT_EQ(entries.front().point.at("dtype"), ParameterValue{std::string("bf16")});
    // The whole point of this pool to a later audit: "llama3-70b prefill" explains a regime
    // that a row of numbers does not.
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
                             "alignment,dtype\n1,64,8,4096,4096,128,true,top_left,bf16\n";

    const auto entries = readModelShapes(metadata, tree.write("m.csv", bare), report);
    ASSERT_EQ(entries.size(), 1u);
    // No name column, so the row identifies itself by position -- still an origin, and still
    // enough to find the line that produced a surprising graph.
    EXPECT_NE(entries.front().origin.find("row 1"), std::string::npos) << entries.front().origin;
}

TEST(TestModelShapeSource, ARowNamingAnotherOperationIsSkippedRatherThanCountedUnusable)
{
    const TempTree tree("other_op");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const std::string mixed
        = "op," + std::string(kHeader).substr(std::string("name,").size())
          + "gemm,1,64,8,4096,4096,128,true,top_left,bf16\n"
            "sdpa_fwd,1,64,8,4096,4096,128,true,top_left,bf16\n";

    const auto entries = readModelShapes(metadata, tree.write("m.csv", mixed), report);

    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(report.rows, 2);
    // One mined file may cover several operations; that is not a defect, and folding it into
    // `unusable` would report a working miner as a broken one.
    EXPECT_EQ(report.otherOperation, 1);
    EXPECT_EQ(report.unusable, 0);
}

TEST(TestModelShapeSource, AMisspelledColumnLosesEveryRowAndSaysWhichName)
{
    const TempTree tree("misspelled");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    // `causal` for `is_causal`. Every row goes; a count alone would not say why.
    const std::string wrong = "q.batch,q.heads,q.heads_kv,q.seqlen_q,q.seqlen_k,q.head_dim,"
                              "q.causal,q.alignment,q.dtype\n"
                              "1,64,8,4096,4096,128,true,top_left,bf16\n"
                              "2,64,8,2048,2048,128,true,top_left,bf16\n";

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
        = kHeader + std::string("mistyped,1,64,8,4096,4096,128,true,top_left,fp8_e4m3\n"
                                "not a number,1,64,8,4096,4096,many,true,top_left,bf16\n");

    const auto entries = readModelShapes(metadata, tree.write("m.csv", bad), report);

    EXPECT_TRUE(entries.empty());
    EXPECT_EQ(report.unusable, 2);
}

TEST(TestModelShapeSource, AQuotedFieldMayCarryACommaWithoutSplittingTheRow)
{
    const TempTree tree("quoted");
    const auto metadata = shippedSdpa();
    ModelShapeReport report;

    const std::string quoted
        = kHeader + std::string(R"("llama3-70b, prefill",1,64,8,4096,4096,128,true,top_left,bf16)")
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

    // Far past any benchmarking byte budget. Whether an engine serves a shape somebody runs is
    // a finding for the caller's admission to make, not a reason to drop the row here.
    const std::string huge
        = kHeader + std::string("enormous,64,128,128,131072,131072,256,false,top_left,bf16\n");

    const auto entries = readModelShapes(metadata, tree.write("m.csv", huge), report);

    EXPECT_EQ(entries.size(), 1u);
    EXPECT_EQ(report.unusable, 0);
}
