// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdGenArtifact.cpp
 * @brief Runs tools/uhd_gen and loads what it produces.
 *
 * Shells out to Python because the contract is cross-language: the runtime refuses an
 * artifact whose feature-signature hash disagrees with the one uhd_gen computed, and it
 * recovers the metric through the transform uhd_gen declares. The corpus is small but
 * noiseless, so the fit converges and the recovered scores can be checked against it.
 */

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/heuristics/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>
#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// The interpreter and the tools directory are the configuring tree's (see
// plugin_sdk/tests/CMakeLists.txt); a binary run elsewhere skips when they are absent, or
// fails when HIPDNN_REQUIRE_UHD_GEN_TESTS is set.
#if !defined(HIPDNN_UHD_GEN_PYTHON) || !defined(HIPDNN_UHD_GEN_TOOLS_DIR)
#error \
    "HIPDNN_UHD_GEN_PYTHON and HIPDNN_UHD_GEN_TOOLS_DIR must be defined; see plugin_sdk/tests/CMakeLists.txt"
#endif

namespace hipdnn_plugin_sdk::uhd
{
namespace
{

/// The tflops trainingCsv() records for one row: rising with tile_m, with M a mild second
/// signal.
double corpusTflops(int64_t m, int64_t tileM)
{
    return (static_cast<double>(tileM) / 4.0) + (static_cast<double>(m) / 4096.0);
}

/// A corpus where tflops rises with tile_m. Sized, with runUhdGen's round count, so the fit
/// converges onto the corpus: uhd_gen bags 80% of the rows per round and LightGBM needs 20
/// rows per leaf (min_data_in_leaf), so with fewer rows per tile many rounds cannot split on
/// tile_m, and too few rounds leave every prediction shrunk toward the corpus mean.
std::string trainingCsv()
{
    std::string csv = "q.M,kernel.tile_m,tflops\n";
    for(const int64_t tileM : {64, 128, 256})
    {
        for(int64_t row = 0; row < 100; ++row)
        {
            const int64_t m = 1024 + (row * 32);
            csv += std::to_string(m);
            csv += ',';
            csv += std::to_string(tileM);
            csv += ',';
            csv += std::to_string(corpusTflops(m, tileM));
            csv += '\n';
        }
    }
    return csv;
}

#ifdef _WIN32
constexpr char PATH_LIST_SEPARATOR = ';';
#else
constexpr char PATH_LIST_SEPARATOR = ':';
#endif

/// hipdnn_uhd_features, which the build and install trees both put beside this binary.
std::filesystem::path featureEvaluator()
{
    return hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
           / hipdnn_data_sdk::utilities::getExecutableName("hipdnn_uhd_features");
}

/// Why uhd_gen cannot run here, or empty when it can. The interpreter and the uhd_gen
/// sources are absolute paths into the configuring machine, so a binary run anywhere else,
/// such as an installed test on a runner without the source tree, finds neither.
std::string missingPrerequisite()
{
    const auto package = std::filesystem::path(HIPDNN_UHD_GEN_TOOLS_DIR) / "uhd_gen";
    if(!std::filesystem::is_directory(package))
    {
        return "the uhd_gen sources are not at " + package.string();
    }
    if(!std::filesystem::exists(HIPDNN_UHD_GEN_PYTHON))
    {
        return std::string("the configured Python interpreter is not at ") + HIPDNN_UHD_GEN_PYTHON;
    }
    if(!std::filesystem::exists(featureEvaluator()))
    {
        return "hipdnn_uhd_features is not beside the test binary at "
               + featureEvaluator().string();
    }
    return {};
}

/// Whether a missing prerequisite fails the test rather than skipping it. Set (to anything
/// but "0") by ctest when the build is configured with HIPDNN_REQUIRE_UHD_GEN_TESTS.
bool prerequisitesRequired()
{
    const auto value = hipdnn_data_sdk::utilities::getEnv("HIPDNN_REQUIRE_UHD_GEN_TESTS");
    return !value.empty() && value != "0";
}

/// Runs the configured Python with @p arguments and returns its exit status. PYTHONPATH, not
/// the working directory, makes uhd_gen importable, so the tools directory may be on a
/// different drive from the test.
int runPython(const std::string& arguments)
{
    std::string pythonPath = HIPDNN_UHD_GEN_TOOLS_DIR;
    const auto inherited = hipdnn_data_sdk::utilities::getEnv("PYTHONPATH");
    if(!inherited.empty())
    {
        pythonPath += PATH_LIST_SEPARATOR + inherited;
    }
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter pythonPathForChild(
        "PYTHONPATH", pythonPath);
    // LightGBM takes every core by default; beside a parallel ctest that stalls the fit for
    // many minutes. A caller's own setting stands.
    const auto threads = hipdnn_data_sdk::utilities::getEnv("OMP_NUM_THREADS");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter threadsForChild(
        "OMP_NUM_THREADS", threads.empty() ? std::string("4") : threads);

    const std::string invocation = std::string("\"") + HIPDNN_UHD_GEN_PYTHON + "\" "
                                   + arguments
                                   // Keep the tool's traceback in the test output.
                                   + " 1>&2";
#ifdef _WIN32
    // cmd.exe strips the first and last quote from a command line that starts with one.
    const std::string command = "\"" + invocation + "\"";
#else
    const std::string& command = invocation;
#endif
    return std::system(command.c_str());
}

/// Runs `python -m uhd_gen train <arguments>` against the evaluator beside this binary.
int runUhdGenTrain(const std::string& arguments)
{
    return runPython("-m uhd_gen train --feature-evaluator \"" + featureEvaluator().string() + "\" "
                     + arguments);
}

/// Writes, to argv[2], rows on every split threshold of the LightGBM model at argv[1] and rows
/// with one feature (or all) missing, each with LightGBM's own raw score: what the runtime's
/// tree walk must reproduce. A value exactly on a threshold is the only one that tells `<=`
/// from `<`, and a missing one the only one that reads a node's default direction. JSON has
/// no NaN, so a missing value is written as null.
constexpr const char* PROBE_SCRIPT = R"(import json, sys
import lightgbm as lgb
import numpy as np

booster = lgb.Booster(model_file=sys.argv[1])
thresholds = []


def walk(node):
    if "split_feature" in node:
        thresholds.append((node["split_feature"], node["threshold"]))
        walk(node["left_child"])
        walk(node["right_child"])


for tree in booster.dump_model()["tree_info"]:
    walk(tree["tree_structure"])
base = [2048.0, 128.0]
rows = []
for feature, threshold in thresholds:
    rows.append(list(base))
    rows[-1][feature] = threshold
for feature in range(len(base)):
    rows.append(list(base))
    rows[-1][feature] = float("nan")
rows.append([float("nan")] * len(base))
raw = booster.predict(np.array(rows), raw_score=True)
with open(sys.argv[2], "w") as out:
    json.dump({"thresholds": len(thresholds),
               "rows": [[None if v != v else v for v in row] for row in rows],
               "raw": [float(v) for v in raw]}, out)
)";

/// Trains on @p csv's single-layer corpus into @p outputDir.
int runUhdGen(const std::filesystem::path& csv, const std::filesystem::path& outputDir)
{
    return runUhdGenTrain("--input \"" + csv.string() + "\""
                          + " --features q.M kernel.tile_m --target tflops --provenance \""
                          + (csv.parent_path() / "provenance.json").string() + "\""
                          + " --output-dir \"" + outputDir.string() + "\""
                          + " --name \"uhd_gen artifact test\""
                          + " --num-boost-round 200 --early-stopping 20"
                          // The LightGBM model itself, the oracle the tree walk is checked by.
                          + " --keep-lgbm");
}

class TestUhdGenArtifact : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(const auto missing = missingPrerequisite(); !missing.empty())
        {
            if(prerequisitesRequired())
            {
                FAIL() << "uhd_gen cannot run from here and HIPDNN_REQUIRE_UHD_GEN_TESTS is set: "
                       << missing;
            }
            GTEST_SKIP() << "uhd_gen cannot run from here: " << missing;
        }

        _dir = std::make_unique<hipdnn_test_sdk::utilities::ScopedDirectory>(
            hipdnn_test_sdk::utilities::claimScratchDirectory("uhd_gen_artifact"));

        const auto csv = _dir->path() / "corpus.csv";
        std::ofstream(csv) << trainingCsv();
        std::ofstream(_dir->path() / "provenance.json") << nlohmann::json{
            {"ued", {{"id", "11000000-0000-0000-0000-000000000000"}, {"revision", "1.0"}}},
            {"kmd", {{"id", "12000000-0000-0000-0000-000000000000"}, {"revision", "1.0"}}},
            {"umd", nlohmann::json::array()}}.dump();

        _outputDir = _dir->path() / "out";

        // Not a skip: the configure step checked the tool's dependencies import, so a failure
        // here is a broken environment, not "nothing to run".
        ASSERT_EQ(runUhdGen(csv, _outputDir), 0)
            << "uhd_gen failed though configure found its dependencies importable; see its "
               "output above.";
    }

    /// The tool's descriptor, loaded through the same path the runtime uses.
    hipdnn_plugin_sdk::uhd::UhdConfig configFromTool() const
    {
        const auto path = _outputDir / "heuristic.uhd.json";
        std::ifstream file(path);
        const auto document = nlohmann::json::parse(file);
        return hipdnn_plugin_sdk::ingestor::UhdKernelHeuristic::configFrom(
            hipdnn_plugin_sdk::ingestor::detail::parseHeuristicDescriptor(document, path));
    }

    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedDirectory> _dir;
    std::filesystem::path _outputDir;
};

} // namespace

TEST_F(TestUhdGenArtifact, WritesTheArtifactsTheRuntimeLooksFor)
{
    // The artifact path is relative to the descriptor, and discovery only finds `*.uhd.json`.
    EXPECT_TRUE(std::filesystem::exists(_outputDir / "heuristic.uhd.json"));
    EXPECT_TRUE(std::filesystem::exists(_outputDir / "model.bin"));
}

TEST_F(TestUhdGenArtifact, TheRuntimeLoadsWhatTheToolWrote)
{
    uhd::UhdConfig config;
    ASSERT_NO_THROW(config = configFromTool())
        << "the descriptor loader rejected a descriptor uhd_gen produced";

    EXPECT_EQ(config.adapterType, "tree_data");
    // The runtime keys the model by metric (RFC 0019 §3.1).
    EXPECT_EQ(config.scoreMetric, "tflops");
    EXPECT_EQ(config.objective, "max");
    // uhd_gen trains on log(target): its inverse, exp, cannot yield a non-positive score,
    // which expm1 (log1p's inverse) could.
    EXPECT_EQ(config.scoreTransform, "log");
    EXPECT_EQ(config.featuresSignature.size(), 2U);
}

TEST_F(TestUhdGenArtifact, TheSignatureHashAgreesAcrossLanguages)
{
    // Python and C++ canonicalise and hash the signature independently.
    const auto config = configFromTool();

    EXPECT_EQ(FeatureExtractor::computeHash(config.featuresSignature), config.featuresHash);
}

TEST_F(TestUhdGenArtifact, TheModelScoresAndOrdersByTheFeatureItWasTrainedOn)
{
    const auto config = configFromTool();

    // makeUhdAdapter rejects a model whose baked-in hash disagrees with the descriptor.
    const auto adapter = makeUhdAdapter(config);
    ASSERT_NE(adapter, nullptr) << "the model artifact did not load against its descriptor";

    const FeatureExtractor extractor(config.featuresSignature, config.categoricalEncoding);

    constexpr int64_t QUERY_M = 2048;
    // What the runtime ranks by: the raw model output inverted through the declared transform.
    const auto scoreFor = [&](int64_t tileM) {
        FeatureExtractionContext ctx;
        ctx.bindQueryVars({{"q.M", QUERY_M}});
        ctx.bindKernelVars({{"tile_m", tileM}});
        return score_transform::applyInverse(adapter->score(extractor.extract(ctx)),
                                             config.scoreTransform);
    };

    EXPECT_GT(scoreFor(256), scoreFor(128));
    EXPECT_GT(scoreFor(128), scoreFor(64));

    // The fit converges onto the corpus (within 0.1% with LightGBM 4.7), so the recovered
    // score is the corpus's tflops. 1% leaves room for another LightGBM version fitting slightly
    // differently, and is tighter than the gap a mismatched transform opens: inverting a log
    // fit with expm1 reports one less, 1.5% below the largest value here.
    for(const int64_t tileM : {64, 128, 256})
    {
        const double expected = corpusTflops(QUERY_M, tileM);
        EXPECT_NEAR(scoreFor(tileM), expected, 0.01 * expected) << "tile_m=" << tileM;
    }
}

TEST_F(TestUhdGenArtifact, TheRuntimeWalksTheTreesAsLightGbmDoes)
{
    const auto config = configFromTool();
    const auto adapter = makeUhdAdapter(config);
    ASSERT_NE(adapter, nullptr);
    ASSERT_EQ(config.featuresSignature.size(), 2U) << "PROBE_SCRIPT's base row has two features";

    const auto script = _dir->path() / "probe.py";
    const auto probes = _dir->path() / "probes.json";
    std::ofstream(script) << PROBE_SCRIPT;
    ASSERT_EQ(runPython("\"" + script.string() + "\" \"" + (_outputDir / "model.lgbm").string()
                        + "\" \"" + probes.string() + "\""),
              0);
    std::ifstream file(probes);
    const auto document = nlohmann::json::parse(file);
    ASSERT_GT(document.at("thresholds").get<size_t>(), 0U) << "the model has no split to probe";

    const auto& rows = document.at("rows");
    const auto& raw = document.at("raw");
    ASSERT_EQ(rows.size(), raw.size());
    for(size_t i = 0; i < rows.size(); ++i)
    {
        std::vector<double> features;
        for(const auto& value : rows[i])
        {
            features.push_back(value.is_null() ? std::numeric_limits<double>::quiet_NaN()
                                               : value.get<double>());
        }
        const double expected = raw[i].get<double>();
        EXPECT_NEAR(adapter->score(features), expected, 1e-9 * std::max(1.0, std::abs(expected)))
            << "probe " << i << ": " << rows[i].dump();
    }
}

/// Trains a two-layer artifact and checks the runtime applies the grouping. Group 1 is
/// strictly better than group 0, and tflops rises with tile_m inside each group.
TEST(TestUhdGenArtifactGrouped, TheRuntimeGroupsWhatTheToolTrained)
{
    if(const auto missing = missingPrerequisite(); !missing.empty())
    {
        if(prerequisitesRequired())
        {
            FAIL() << "uhd_gen cannot run from here and HIPDNN_REQUIRE_UHD_GEN_TESTS is set: "
                   << missing;
        }
        GTEST_SKIP() << "uhd_gen cannot run from here: " << missing;
    }

    const auto dir = hipdnn_test_sdk::utilities::claimScratchDirectory("uhd_gen_grouped");

    const auto csv = dir.path() / "corpus.csv";
    {
        std::ofstream out(csv);
        out << "q.M,kernel.group,kernel.tile_m,tflops\n";
        for(const int64_t group : {0, 1})
        {
            for(const int64_t tileM : {64, 128, 256})
            {
                // Layer 1 fits one row per (problem, group); too few problems leave it a
                // single constant leaf that picks a group by position.
                for(int row = 0; row < 120; ++row)
                {
                    const double m = 1024.0 + (row * 128.0);
                    const double base = (group == 1) ? 400.0 : 10.0;
                    out << static_cast<int64_t>(m) << "," << group << "," << tileM << ","
                        << (base + static_cast<double>(tileM) / 4.0 + m / 4096.0) << "\n";
                }
            }
        }
    }

    std::ofstream(dir.path() / "provenance.json") << nlohmann::json{
        {"ued", {{"id", "11000000-0000-0000-0000-000000000000"}, {"revision", "1.0"}}},
        {"kmd", {{"id", "12000000-0000-0000-0000-000000000000"}, {"revision", "1.0"}}},
        {"umd", nlohmann::json::array()}}.dump();

    const auto outputDir = dir.path() / "out";
    ASSERT_EQ(runUhdGenTrain("--input \"" + csv.string() + "\""
                             + " --features q.M kernel.group kernel.tile_m"
                             + " --group-by-feature kernel.group"
                             // Layer 1 is fitted per (problem, group), so the problem key is
                             // required.
                             + " --group-by q.M --target tflops --provenance \""
                             + (dir.path() / "provenance.json").string() + "\"" + " --output-dir \""
                             + outputDir.string() + "\"" + " --name \"uhd_gen grouped test\""
                             + " --num-boost-round 40 --early-stopping 10"),
              0);

    const auto path = outputDir / "heuristic.uhd.json";
    std::ifstream file(path);
    const auto document = nlohmann::json::parse(file);
    const auto config = hipdnn_plugin_sdk::ingestor::UhdKernelHeuristic::configFrom(
        hipdnn_plugin_sdk::ingestor::detail::parseHeuristicDescriptor(document, path));

    const auto adapter = makeUhdAdapter(config);
    ASSERT_NE(adapter, nullptr) << "the grouped artifact did not load against its descriptor";

    const FeatureExtractor extractor(config.featuresSignature, config.categoricalEncoding);
    const auto row = [&](int64_t group, int64_t tileM) {
        FeatureExtractionContext ctx;
        ctx.bindQueryVars({{"q.M", int64_t{2048}}});
        ctx.bindKernelVars({{"group", group}, {"tile_m", tileM}});
        return extractor.extract(ctx);
    };

    const auto scores = adapter->scoreBatch({row(0, 256), row(1, 64), row(1, 256), row(0, 64)});
    ASSERT_EQ(scores.size(), 4U);

    // Group 0 is filtered out by layer 1, including its largest tile, which a flat model
    // would rank above group 1's smallest.
    EXPECT_EQ(scores[0], -std::numeric_limits<double>::infinity());
    EXPECT_EQ(scores[3], -std::numeric_limits<double>::infinity());

    // Layer 2 still orders within the chosen group.
    EXPECT_TRUE(std::isfinite(scores[1]));
    EXPECT_TRUE(std::isfinite(scores[2]));
    EXPECT_GT(scores[2], scores[1]);
}

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
