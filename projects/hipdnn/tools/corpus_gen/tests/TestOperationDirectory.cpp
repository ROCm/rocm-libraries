// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestOperationDirectory.cpp
 * @brief Covers loading a directory of declarations.
 *
 * The property under test is that nothing is dropped quietly. An operation whose declaration
 * fails to parse and is silently skipped leaves a hole in the corpus indistinguishable from an
 * operation nobody declared -- and the run still reports a total that looks whole.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/OperationDirectory.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace hipdnn_corpus_gen
{
namespace
{

/// A scratch directory that cleans up after itself, named per test.
class Scratch
{
public:
    explicit Scratch(const std::string& name)
        : _path(std::filesystem::temp_directory_path() / ("corpus_gen_" + name))
    {
        std::filesystem::remove_all(_path);
        std::filesystem::create_directories(_path);
    }
    ~Scratch()
    {
        std::filesystem::remove_all(_path);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    const std::filesystem::path& path() const
    {
        return _path;
    }

    void write(const std::string& name, const std::string& text) const
    {
        std::ofstream file(_path / name);
        file << text;
    }

private:
    std::filesystem::path _path;
};

std::string declaration(const std::string& operation)
{
    return R"({
      "schema_version": "1.0",
      "operation": ")" + operation + R"(",
      "parameters": { "M": { "type": "int64" } },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "matmul", "source": "x.hpp", "arguments": [] }
    })";
}

} // namespace

TEST(TestOperationDirectory, LoadsEveryDeclarationInTheDirectory)
{
    const Scratch scratch("loads_all");
    scratch.write("a.opmeta.json", declaration("alpha"));
    scratch.write("b.opmeta.json", declaration("beta"));

    const auto set = loadOperationDirectory(scratch.path());

    EXPECT_TRUE(set.errors.empty()) << (set.errors.empty() ? "" : set.errors.front());
    ASSERT_EQ(set.operations.size(), 2U);
}

TEST(TestOperationDirectory, VisitsFilesInAStableOrder)
{
    // Directory iteration order is unspecified. An unordered visit reshuffles which operations
    // a maxCombinations bound reaches, so two runs of the same command cover different ground
    // while both report success.
    const Scratch scratch("stable_order");
    scratch.write("zulu.opmeta.json", declaration("zulu"));
    scratch.write("alpha.opmeta.json", declaration("alpha"));
    scratch.write("mike.opmeta.json", declaration("mike"));

    const auto set = loadOperationDirectory(scratch.path());

    ASSERT_EQ(set.operations.size(), 3U);
    EXPECT_EQ(set.operations[0].second.operation, "alpha");
    EXPECT_EQ(set.operations[1].second.operation, "mike");
    EXPECT_EQ(set.operations[2].second.operation, "zulu");
}

TEST(TestOperationDirectory, AMalformedDeclarationIsReportedAndNamed)
{
    // Not skipped: a declaration that does not load is a hole in the corpus, and one malformed
    // file among twenty must say which one it was.
    const Scratch scratch("malformed");
    scratch.write("good.opmeta.json", declaration("good"));
    scratch.write("broken.opmeta.json", "{ this is not json");

    const auto set = loadOperationDirectory(scratch.path());

    EXPECT_EQ(set.operations.size(), 1U) << "the valid declaration should still load";
    ASSERT_FALSE(set.errors.empty()) << "a broken declaration was dropped silently";
    EXPECT_NE(set.errors.front().find("broken.opmeta.json"), std::string::npos)
        << "the error does not name the file: " << set.errors.front();
}

TEST(TestOperationDirectory, AnInvalidDeclarationIsReportedRatherThanLoaded)
{
    // Parses as JSON, fails §4.4 validation. The distinction matters: this is the case where a
    // file looks fine and means nothing.
    const Scratch scratch("invalid");
    scratch.write("bad.opmeta.json", R"({
      "schema_version": "1.0",
      "operation": "bad",
      "parameters": { "M": { "type": "int64" } },
      "stratification_axis": "not_a_permitted_axis",
      "regimes": {},
      "graph_builder": { "function": "matmul", "source": "x.hpp", "arguments": [] }
    })");

    const auto set = loadOperationDirectory(scratch.path());

    EXPECT_TRUE(set.operations.empty());
    ASSERT_FALSE(set.errors.empty());
    EXPECT_NE(set.errors.front().find("bad.opmeta.json"), std::string::npos);
}

TEST(TestOperationDirectory, IgnoresFilesThatAreNotDeclarations)
{
    // A corpus directory accumulates notes, generated CSVs and editor droppings. Reading them
    // as declarations would turn housekeeping into errors and hide the real ones.
    const Scratch scratch("ignores");
    scratch.write("real.opmeta.json", declaration("real"));
    scratch.write("notes.md", "not a declaration");
    scratch.write("results.json", R"({ "not": "a declaration" })");
    scratch.write("real.opmeta.json.bak", declaration("stale"));

    const auto set = loadOperationDirectory(scratch.path());

    EXPECT_TRUE(set.errors.empty()) << (set.errors.empty() ? "" : set.errors.front());
    ASSERT_EQ(set.operations.size(), 1U);
    EXPECT_EQ(set.operations.front().second.operation, "real");
}

TEST(TestOperationDirectory, AMissingDirectoryIsAnErrorRatherThanAnEmptyCorpus)
{
    // Silently returning nothing would read as "this engine serves no declared operation",
    // which is a statement about the engine rather than about a mistyped path.
    const auto set = loadOperationDirectory("/no/such/corpus/directory");

    EXPECT_TRUE(set.operations.empty());
    ASSERT_FALSE(set.errors.empty());
    EXPECT_NE(set.errors.front().find("not a directory"), std::string::npos);
}

TEST(TestOperationDirectory, AnEmptyDirectoryLoadsNothingAndSaysNothingIsWrong)
{
    const Scratch scratch("empty");
    const auto set = loadOperationDirectory(scratch.path());

    EXPECT_TRUE(set.operations.empty());
    EXPECT_TRUE(set.errors.empty()) << "an empty directory is not a malformed one";
}

} // namespace hipdnn_corpus_gen
