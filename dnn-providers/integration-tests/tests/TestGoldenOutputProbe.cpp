// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GoldenOutputProbe decides, before anything is loaded, which bundles the
// golden-data binary may drop. A wrong "no" silently removes a golden bundle from
// validation, so each way it can answer "no" is pinned here, as is each case it
// must let through for the loader to report.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/reference-validation/GoldenOutputProbe.hpp"

using hipdnn_integration_tests::bundle::DiscoveredBundle;
using hipdnn_integration_tests::bundle::SweepCase;
using hipdnn_integration_tests::bundle::detail::GoldenOutputProbe;
using hipdnn_test_sdk::utilities::claimScratchDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

class TestGoldenOutputProbe : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        _scopedDir.emplace(claimScratchDirectory("golden_output_probe"));
        _tempDir = _scopedDir->path();
    }

    static void touch(const std::filesystem::path& path, const std::string& content = "")
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path) << content;
    }

    static DiscoveredBundle singleBundle(const std::filesystem::path& jsonPath)
    {
        DiscoveredBundle disc;
        disc.jsonPath = jsonPath;
        return disc;
    }

    static DiscoveredBundle sweepCase(const std::filesystem::path& sweepJson,
                                      const std::string& caseId)
    {
        DiscoveredBundle disc;
        disc.jsonPath = sweepJson;
        disc.sweep = SweepCase{sweepJson.parent_path() / "graph.template.json", caseId};
        return disc;
    }

    // One case with a golden block pointing into golden/a/, laid out as a checkout
    // before `dvc pull`: the directory exists and holds only the committed pointer.
    std::filesystem::path writeSweepWithGoldenCase()
    {
        const auto sweepJson = _tempDir / "Op" / "Default" / "sweep.json";
        touch(sweepJson,
              R"({"cases": [{"id": "a", "golden": {"path": "golden/a/tensors.dvc"}},
                            {"id": "b"}]})");
        touch(sweepJson.parent_path() / "golden" / "a" / "tensors.dvc");
        return sweepJson;
    }
};

} // namespace

TEST_F(TestGoldenOutputProbe, SingleBundleCarriesGoldenOnlyWithItsOwnOutputBlob)
{
    const auto jsonPath = _tempDir / "Op" / "graph.json";
    touch(jsonPath);
    // Another bundle's blob in the same directory is not this bundle's golden data.
    touch(_tempDir / "Op" / "other.tensor5.bin");

    GoldenOutputProbe probe;
    EXPECT_FALSE(probe.mayCarryGoldenOutputs(singleBundle(jsonPath)));

    touch(_tempDir / "Op" / "graph.tensor5.bin");
    GoldenOutputProbe afterPull;
    EXPECT_TRUE(afterPull.mayCarryGoldenOutputs(singleBundle(jsonPath)));
}

// The pre-pull tree is the case that matters: `golden` is declared in sweep.json,
// which is in git, so declaration alone would keep every declaring case. Only the
// blobs themselves say golden data is present.
TEST_F(TestGoldenOutputProbe, SweepCaseCarriesGoldenOnlyOnceItsBlobsArePulled)
{
    const auto sweepJson = writeSweepWithGoldenCase();

    GoldenOutputProbe probe;
    EXPECT_FALSE(probe.mayCarryGoldenOutputs(sweepCase(sweepJson, "a")));

    touch(sweepJson.parent_path() / "golden" / "a" / "tensor5.bin");
    GoldenOutputProbe afterPull;
    EXPECT_TRUE(afterPull.mayCarryGoldenOutputs(sweepCase(sweepJson, "a")));
}

TEST_F(TestGoldenOutputProbe, SweepCaseWithoutAGoldenBlockCarriesNone)
{
    const auto sweepJson = writeSweepWithGoldenCase();

    GoldenOutputProbe probe;
    EXPECT_FALSE(probe.mayCarryGoldenOutputs(sweepCase(sweepJson, "b")));
}

// Each of these is an authoring error the loader reports. Dropping the bundle here
// would hide the report, so the probe must let it through.
TEST_F(TestGoldenOutputProbe, UndecidableSweepCasesAreLetThroughToTheLoader)
{
    const auto unparseable = _tempDir / "Unparseable" / "sweep.json";
    touch(unparseable, "{ not json");

    const auto noPath = _tempDir / "NoPath" / "sweep.json";
    touch(noPath, R"({"cases": [{"id": "a", "golden": {}}]})");

    const auto sweepJson = writeSweepWithGoldenCase();

    GoldenOutputProbe probe;
    EXPECT_TRUE(probe.mayCarryGoldenOutputs(sweepCase(unparseable, "a")));
    EXPECT_TRUE(probe.mayCarryGoldenOutputs(sweepCase(noPath, "a")));
    EXPECT_TRUE(probe.mayCarryGoldenOutputs(sweepCase(sweepJson, "missing")));
}

// NOLINTEND(readability-identifier-naming)
