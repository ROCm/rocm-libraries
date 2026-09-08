// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdGenArtifact.cpp
 * @brief Runs tools/uhd_gen and loads what it produces.
 *
 * uhd_gen writes a UHD and a model artifact; the runtime loads them and refuses the pair if
 * the feature signature it recomputes disagrees with the one the descriptor declares. Both
 * sides pin that agreement -- the same literal digest appears in test_features_hash.py and
 * TestFeatureExtractor.cpp -- but pinning a hash is not the same as loading a file, and
 * until this test nothing had ever fed a real uhd_gen artifact to the loader.
 *
 * A gtest that shells out to Python is unusual. It is the right shape here because the
 * contract under test is itself cross-language: the failure this catches is one side
 * changing how it spells or canonicalises a signature, which no test confined to one
 * language can see.
 *
 * The training is deliberately tiny -- a handful of rows and a few boosting rounds. What
 * matters is that the artifact loads, its hash agrees, and the model orders candidates by
 * the feature it was trained on; whether it is any good needs a benchmark corpus, which is
 * RFC 0019.13's subject and not this file's.
 */

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/FeatureExtractor.hpp>

#include <nlohmann/json.hpp>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <string>
#include <vector>

#if !defined(HIPDNN_UHD_GEN_PYTHON) || !defined(HIPDNN_UHD_GEN_TOOLS_DIR)
#error "HIPDNN_UHD_GEN_PYTHON and HIPDNN_UHD_GEN_TOOLS_DIR must be defined; see tools/CMakeLists.txt"
#endif

namespace hipdnn_plugin_sdk::ingestor::uhd
{
namespace
{

/// A corpus where tflops rises with tile_m, so a model that fit anything at all prefers the
/// larger tile.
///
/// Sized for LightGBM's defaults rather than for readability: min_data_in_leaf is 20, so a
/// handful of rows yields a single constant leaf and every candidate scores identically.
/// Twenty-five rows per tile leaves room to split on either side of any boundary.
std::string trainingCsv()
{
    std::string csv = "q.M,kernel.tile_m,tflops\n";
    for(const int64_t tileM : {64, 128, 256})
    {
        for(int row = 0; row < 25; ++row)
        {
            // Monotonic in tile_m, with a mild dependence on M so the second feature is not
            // pure noise. Deterministic: a flaky corpus would make a flaky assertion.
            const double m = 1024.0 + (row * 128.0);
            const double tflops = (static_cast<double>(tileM) / 4.0) + (m / 4096.0);
            csv += std::to_string(static_cast<int64_t>(m)) + "," + std::to_string(tileM) + ","
                   + std::to_string(tflops) + "\n";
        }
    }
    return csv;
}

/// Runs uhd_gen into @p outputDir. Returns the command's exit status.
///
/// Invoked from the tools directory so `python -m uhd_gen` resolves without the package
/// being installed, which is also how the README documents driving it.
int runUhdGen(const std::filesystem::path& csv, const std::filesystem::path& outputDir)
{
    const std::string command = std::string("cd ") + HIPDNN_UHD_GEN_TOOLS_DIR + " && "
                                + HIPDNN_UHD_GEN_PYTHON + " -m uhd_gen"
                                + " --input " + csv.string()
                                + " --features q.M kernel.tile_m"
                                + " --target tflops"
                                + " --output-dir " + outputDir.string()
                                + " --name 'uhd_gen artifact test'"
                                // A real run trains for hundreds of rounds; this corpus does
                                // not need them, and the wall-clock is charged to every run.
                                + " --num-boost-round 40 --early-stopping 10"
                                // Diagnostics deliberately not suppressed: when this
                                // fails, the tool's traceback is the only thing that says
                                // why, and it lands in the test's output.
                                + " 1>&2";
    return std::system(command.c_str());
}

class TestUhdGenArtifact : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _dir = std::make_unique<hipdnn_test_sdk::utilities::ScopedDirectory>(
            std::filesystem::temp_directory_path() / "hipdnn_uhd_gen_artifact");

        const auto csv = _dir->path() / "corpus.csv";
        std::ofstream(csv) << trainingCsv();

        _outputDir = _dir->path() / "out";

        // Not a skip. The tool's dependencies are part of the build environment
        // (projects/hipdnn/dockerfiles/Dockerfile.ubuntu24 installs them), and a skip here
        // would restore exactly the silence this test exists to end: uhd_gen's own suite
        // spent its life unregistered because a missing interpreter read as "nothing to
        // run" rather than as a broken environment.
        ASSERT_EQ(runUhdGen(csv, _outputDir), 0)
            << "uhd_gen failed. Its dependencies are in "
               "projects/hipdnn/tools/uhd_gen/requirements.txt, which the hipDNN dev image "
               "installs.";
    }

    /// The tool's descriptor, read back the way the runtime reads it: parsed by
    /// DescriptorLoader, then turned into a UhdConfig by the heuristic itself. There is no
    /// second file -- the descriptor IS the UHD -- so this is the whole load path.
    hipdnn_plugin_sdk::ingestor::uhd::UhdConfig configFromTool() const
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
    // The names are a contract, not an implementation detail: the artifact path is written
    // relative to the descriptor, so the loader resolves model.bin beside it. The `.uhd.json`
    // suffix is what descriptor discovery looks for -- a bare `uhd.json` is invisible to it.
    EXPECT_TRUE(std::filesystem::exists(_outputDir / "heuristic.uhd.json"));
    EXPECT_TRUE(std::filesystem::exists(_outputDir / "model.bin"));
}

TEST_F(TestUhdGenArtifact, TheRuntimeLoadsWhatTheToolWrote)
{
    uhd::UhdConfig config;
    ASSERT_NO_THROW(config = configFromTool())
        << "the descriptor loader rejected a descriptor uhd_gen produced";

    EXPECT_EQ(config.adapterType, "tree_data");
    EXPECT_EQ(config.objective, "max");
    // uhd_gen trains on log1p(target) and says so, which is what lets a consumer recover
    // the declared units.
    EXPECT_EQ(config.scoreTransform, "log1p");
    EXPECT_EQ(config.featuresSignature.size(), 2U);
}

TEST_F(TestUhdGenArtifact, TheSignatureHashAgreesAcrossLanguages)
{
    // The assertion this file exists for. Python canonicalises the signature and hashes it;
    // C++ does the same independently, and the runtime refuses the model on a mismatch. A
    // divergence in either spelling or canonicalisation shows up here and nowhere else.
    const auto config = configFromTool();

    EXPECT_EQ(FeatureExtractor::computeHash(config.featuresSignature), config.featuresHash);
}

TEST_F(TestUhdGenArtifact, TheModelScoresAndOrdersByTheFeatureItWasTrainedOn)
{
    const auto config = configFromTool();

    // makeUhdAdapter checks the descriptor's hash against the one baked into the model
    // artifact, so a non-null adapter is itself evidence the pair came from one run.
    const auto adapter = makeUhdAdapter(config);
    ASSERT_NE(adapter, nullptr) << "the model artifact did not load against its descriptor";

    const FeatureExtractor extractor(config.featuresSignature, config.derived);

    const auto scoreFor = [&](int64_t tileM) {
        FeatureExtractionContext ctx;
        ctx.bindQueryVars({{"M", int64_t{2048}}});
        ctx.bindKernelVars({{"tile_m", tileM}});
        return adapter->score(extractor.extract(ctx));
    };

    // The corpus makes tflops rise with tile_m, so the ordering is the one thing a model
    // trained on it must reproduce. Asserting an ordering rather than a value keeps this
    // from breaking on a LightGBM version that fits the same data slightly differently.
    EXPECT_GT(scoreFor(256), scoreFor(64));
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
