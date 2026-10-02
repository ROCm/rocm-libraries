// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdGenArtifact.cpp
 * @brief Runs tools/uhd_gen and loads what it produces.
 *
 * Shells out to Python because the contract is cross-language: the runtime refuses an
 * artifact whose feature-signature hash disagrees with the one uhd_gen computed. Training is
 * deliberately tiny; model quality is out of scope.
 */

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/heuristics/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>
#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// The interpreter and the tools directory are the configuring tree's (see
// plugin_sdk/tests/CMakeLists.txt); a binary run elsewhere skips when they are absent.
#if !defined(HIPDNN_UHD_GEN_PYTHON) || !defined(HIPDNN_UHD_GEN_TOOLS_DIR)
#error \
    "HIPDNN_UHD_GEN_PYTHON and HIPDNN_UHD_GEN_TOOLS_DIR must be defined; see plugin_sdk/tests/CMakeLists.txt"
#endif

namespace hipdnn_plugin_sdk::uhd
{
namespace
{

/// A corpus where tflops rises with tile_m. Sized for LightGBM's min_data_in_leaf (20): fewer
/// rows per tile yield a single constant leaf that scores every candidate identically.
std::string trainingCsv()
{
    std::string csv = "q.M,kernel.tile_m,tflops\n";
    for(const int64_t tileM : {64, 128, 256})
    {
        for(int row = 0; row < 25; ++row)
        {
            // Deterministic and monotonic in tile_m; M adds a mild second signal.
            const double m = 1024.0 + (row * 128.0);
            const double tflops = (static_cast<double>(tileM) / 4.0) + (m / 4096.0);
            csv += std::to_string(static_cast<int64_t>(m)) + "," + std::to_string(tileM) + ","
                   + std::to_string(tflops) + "\n";
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

/// Runs `python -m uhd_gen train <arguments>` against the evaluator beside this binary and
/// returns its exit status. PYTHONPATH, not the working directory, makes the package
/// importable, so the tools directory may be on a different drive from the test.
int runUhdGenTrain(const std::string& arguments)
{
    std::string pythonPath = HIPDNN_UHD_GEN_TOOLS_DIR;
    const auto inherited = hipdnn_data_sdk::utilities::getEnv("PYTHONPATH");
    if(!inherited.empty())
    {
        pythonPath += PATH_LIST_SEPARATOR + inherited;
    }
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter pythonPathForChild(
        "PYTHONPATH", pythonPath);

    std::string command = std::string("\"") + HIPDNN_UHD_GEN_PYTHON + "\" -m uhd_gen train"
                          + " --feature-evaluator \"" + featureEvaluator().string() + "\" "
                          + arguments
                          // Keep the tool's traceback in the test output.
                          + " 1>&2";
#ifdef _WIN32
    // cmd.exe strips the first and last quote from a command line that starts with one.
    command = "\"" + command + "\"";
#endif
    return std::system(command.c_str());
}

/// Trains on @p csv's single-layer corpus into @p outputDir.
int runUhdGen(const std::filesystem::path& csv, const std::filesystem::path& outputDir)
{
    return runUhdGenTrain("--input \"" + csv.string() + "\""
                          + " --features q.M kernel.tile_m --target tflops --provenance \""
                          + (csv.parent_path() / "provenance.json").string() + "\""
                          + " --output-dir \"" + outputDir.string() + "\""
                          + " --name \"uhd_gen artifact test\""
                          + " --num-boost-round 40 --early-stopping 10");
}

class TestUhdGenArtifact : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(const auto missing = missingPrerequisite(); !missing.empty())
        {
            GTEST_SKIP() << "uhd_gen cannot run from here: " << missing;
        }

        _dir = std::make_unique<hipdnn_test_sdk::utilities::ScopedDirectory>(
            std::filesystem::temp_directory_path() / "hipdnn_uhd_gen_artifact");

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
            << "uhd_gen failed. Its dependencies are in "
               "projects/hipdnn/tools/uhd_gen/requirements.txt, which the hipDNN dev image "
               "installs.";
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

    const auto scoreFor = [&](int64_t tileM) {
        FeatureExtractionContext ctx;
        ctx.bindQueryVars({{"q.M", int64_t{2048}}});
        ctx.bindKernelVars({{"tile_m", tileM}});
        return adapter->score(extractor.extract(ctx));
    };

    // Assert ordering, not values, so a different LightGBM version fitting slightly
    // differently still passes.
    EXPECT_GT(scoreFor(256), scoreFor(64));
}

/// Trains a two-layer artifact and checks the runtime applies the grouping. Group 1 is
/// strictly better than group 0, and tflops rises with tile_m inside each group.
TEST(TestUhdGenArtifactGrouped, TheRuntimeGroupsWhatTheToolTrained)
{
    if(const auto missing = missingPrerequisite(); !missing.empty())
    {
        GTEST_SKIP() << "uhd_gen cannot run from here: " << missing;
    }

    const hipdnn_test_sdk::utilities::ScopedDirectory dir(std::filesystem::temp_directory_path()
                                                          / "hipdnn_uhd_gen_grouped");

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
