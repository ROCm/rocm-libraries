// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>

#include "harness/golden/BundleDiscovery.hpp"
#include "harness/golden/BundleLoadCheck.hpp"
#include "harness/golden/IntegrationTestBundle.hpp"

using namespace hipdnn_integration_tests::golden;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

class TestBundleDiscoveryFixture : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        auto path
            = std::filesystem::temp_directory_path()
              / ("bundle_discovery_test_"
                 + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
        std::filesystem::remove_all(path);
        _scopedDir.emplace(path);
        _tempDir = _scopedDir->path();
    }

    // Writes a minimal but schema-valid batchnorm-inference graph (nchw, fp32).
    static void createMinimalBundle(const std::filesystem::path& dir, const std::string& name)
    {
        std::filesystem::create_directories(dir);
        std::ofstream ofs(dir / (name + ".json"));
        ofs << R"({"nodes": [{"inputs": {"x_tensor_uid": 0, "mean_tensor_uid": 1, )"
               R"("inv_variance_tensor_uid": 2, "scale_tensor_uid": 3, "bias_tensor_uid": 4}, )"
               R"("outputs": {"y_tensor_uid": 5}, "type": "BatchnormInferenceAttributes", )"
               R"("compute_data_type": "float", "name": ""}], "tensors": [)"
               R"({"name": "", "uid": 0, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 1, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 2, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 3, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 4, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
               R"("data_type": "float", "virtual": false}, )"
               R"({"name": "", "uid": 5, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
               R"("data_type": "float", "virtual": false}], "io_data_type": "float", )"
               R"("compute_data_type": "float", "intermediate_data_type": "float", "name": ""})";
    }

    // Writes a minimal bundle with JSON + .bin tensor data files so that
    // loadGraphAndTensors (and thus loadIntegrationTestBundle) can fully load it.
    // Tensor dims/strides match createMinimalBundle; each .bin is zero-filled
    // to the exact byte count the loader expects.
    static void createLoadableBundle(const std::filesystem::path& dir, const std::string& name)
    {
        createMinimalBundle(dir, name);
        const auto basePath = dir / name;

        // uid 0 (x):    dims [2,3,4,5], strides [60,20,5,1] -> 120 floats = 480 bytes
        // uid 1-4:      dims [1,3,1,1], strides [3,1,1,1]   ->   3 floats =  12 bytes
        // uid 5 (y):    dims [2,3,4,5], strides [60,20,5,1] -> 120 floats = 480 bytes
        auto writeBin = [&](int64_t uid, size_t byteCount) {
            std::vector<char> data(byteCount, 0);
            std::ofstream out(basePath.string() + ".tensor" + std::to_string(uid) + ".bin",
                              std::ios::binary);
            out.write(data.data(), static_cast<std::streamsize>(data.size()));
        };

        writeBin(0, 480);
        writeBin(1, 12);
        writeBin(2, 12);
        writeBin(3, 12);
        writeBin(4, 12);
        writeBin(5, 480);
    }

    // Finds a discovered bundle by its derived test name, or nullptr.
    static const DiscoveredBundle* findByTest(const std::vector<DiscoveredBundle>& bundles,
                                              const std::string& testName)
    {
        for(const auto& b : bundles)
        {
            if(b.testName == testName)
            {
                return &b;
            }
        }
        return nullptr;
    }
};

} // namespace

TEST_F(TestBundleDiscoveryFixture, FlatCustomerBundleDrop)
{
    // Case 2: a standalone customer folder dropped directly under the data root:
    // suite is the folder name, test is the .json stem. No tier/structure required.
    createMinimalBundle(_tempDir / "case_23421", "graph");

    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().suiteName, "case_23421");
    EXPECT_EQ(result.front().testName, "graph");
}

TEST_F(TestBundleDiscoveryFixture, TieredGoldenDataLayoutIsDiscovered)
{
    // Case 1: the structured golden_reference_data tier layout. Every directory
    // segment below the root joins into the suite with '_', the file stem is the
    // test.
    createMinimalBundle(_tempDir / "quick" / "BatchnormFwdInference" / "ncdhw" / "fp32" / "Small",
                        "Small");

    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().suiteName, "quick_BatchnormFwdInference_ncdhw_fp32_Small");
    EXPECT_EQ(result.front().testName, "Small");
}

TEST_F(TestBundleDiscoveryFixture, JsonAtRootThrows)
{
    // A .json directly at the data root has no folder to form a suite -> throw.
    std::ofstream(_tempDir / "graph.json") << R"({"tensors": []})";
    EXPECT_THROW(discoverBundles(_tempDir), std::runtime_error);
}

TEST_F(TestBundleDiscoveryFixture, EmptyLeafFolderThrows)
{
    // A leaf folder with no graph .json is an empty bundle folder -> throw. The
    // sibling bundle ensures the tree is otherwise valid.
    createMinimalBundle(_tempDir / "conv" / "good", "good");
    std::filesystem::create_directories(_tempDir / "conv" / "case_12312"); // empty leaf
    EXPECT_THROW(discoverBundles(_tempDir), std::runtime_error);
}

TEST_F(TestBundleDiscoveryFixture, LeafWithOnlyMetaJsonThrows)
{
    // A leaf folder holding only meta companions (no graph) is still empty.
    auto dir = _tempDir / "conv" / "meta_only";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "meta.json") << "{}";
    EXPECT_THROW(discoverBundles(_tempDir), std::runtime_error);
}

TEST_F(TestBundleDiscoveryFixture, EmptyRootThrows)
{
    // A completely empty data root is itself an empty leaf -> throw.
    EXPECT_THROW(discoverBundles(_tempDir), std::runtime_error);
}

TEST_F(TestBundleDiscoveryFixture, CollisionThrows)
{
    // Two bundles whose paths differ only by dash vs underscore both sanitize to
    // the same suite + test name -> collision.
    createMinimalBundle(_tempDir / "Op-A" / "case", "SameName");
    createMinimalBundle(_tempDir / "Op_A" / "case", "SameName");
    EXPECT_THROW(discoverBundles(_tempDir), std::runtime_error);
}

TEST_F(TestBundleDiscoveryFixture, CustomerDropAndTieredLayoutCoexistUnderOneRoot)
{
    // Cases 1 and 2 together: a flat customer drop and a deep tiered bundle live
    // under the same root and discover independently, each named purely from its
    // path. Depth is data, not a branch — the same leaf/recurse logic handles both.
    createMinimalBundle(_tempDir / "case_1", "graph");
    createMinimalBundle(_tempDir / "conv" / "nchw" / "fp16" / "resnet50", "resnet50");

    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 2u);

    const auto* flat = findByTest(result, "graph");
    ASSERT_NE(flat, nullptr);
    EXPECT_EQ(flat->suiteName, "case_1");

    const auto* deep = findByTest(result, "resnet50");
    ASSERT_NE(deep, nullptr);
    EXPECT_EQ(deep->suiteName, "conv_nchw_fp16_resnet50");
}

TEST_F(TestBundleDiscoveryFixture, SkipsMetaJson)
{
    // Both a bare meta.json and a {Name}.meta.json companion must be ignored.
    auto bundleDir = _tempDir / "conv" / "nchw" / "fp32" / "withmeta";
    createMinimalBundle(bundleDir, "withmeta");
    std::ofstream(bundleDir / "withmeta.meta.json") << "{}";
    std::ofstream(bundleDir / "meta.json") << "{}";

    auto result = discoverBundles(_tempDir);
    // Only the "withmeta" graph; neither meta file adds a bundle.
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().testName, "withmeta");
}

TEST_F(TestBundleDiscoveryFixture, ScanFilesByExtensionIsGenericAndSorted)
{
    // The generic scanner carries no bundle knowledge: it returns every matching
    // file (including meta files), recursively, in sorted order.
    auto root = _tempDir / "scan";
    std::filesystem::create_directories(root / "sub");
    std::ofstream(root / "b.json") << "{}";
    std::ofstream(root / "a.json") << "{}";
    std::ofstream(root / "sub" / "c.json") << "{}";
    std::ofstream(root / "sub" / "c.meta.json") << "{}";
    std::ofstream(root / "note.txt") << "ignore me";

    auto json = scanFilesByExtension(root, ".json");
    ASSERT_EQ(json.size(), 4u); // a, b, sub/c, sub/c.meta — .txt excluded
    EXPECT_TRUE(std::is_sorted(json.begin(), json.end()));
    EXPECT_EQ(json.front().filename(), "a.json");
}

TEST(TestMetaFile, IdentifiesCompanionMetadata)
{
    EXPECT_TRUE(isMetaFile("dir/meta.json"));
    EXPECT_TRUE(isMetaFile("dir/resnet50.meta.json"));
    EXPECT_TRUE(isMetaFile("resnet50.meta.json"));
    EXPECT_FALSE(isMetaFile("dir/resnet50.json"));
    EXPECT_FALSE(isMetaFile("dir/metadata.json"));
    EXPECT_FALSE(isMetaFile("dir/meta.bin"));
}

TEST(TestSanitizeForGtest, ReplacesInvalidChars)
{
    EXPECT_EQ(sanitizeForGtest("hello world!"), "hello_world_");
    EXPECT_EQ(sanitizeForGtest("Conv-Fprop.v2"), "Conv_Fprop_v2");
    EXPECT_EQ(sanitizeForGtest("already_valid_123"), "already_valid_123");
}

TEST_F(TestBundleDiscoveryFixture, UnparseableJsonIsDiscoveredButLoadThrows)
{
    // Discovery only scans paths, not content: a malformed .json is still
    // discovered (valid path) but throws when the loader tries to parse it.
    auto badDir = _tempDir / "BadOp" / "Malformed";
    std::filesystem::create_directories(badDir);
    std::ofstream(badDir / "Malformed.json") << "{{NOT VALID JSON AT ALL";

    auto bundles = discoverBundles(_tempDir);
    auto it = std::find_if(bundles.begin(), bundles.end(), [](const DiscoveredBundle& b) {
        return b.testName == "Malformed";
    });
    ASSERT_NE(it, bundles.end()) << "Malformed bundle should be discovered (valid .json path)";

    EXPECT_THROW(hipdnn_test_sdk::utilities::loadGraphAndTensors(it->jsonPath), std::exception);
}

// The harness uses checkBundlePreload().graphJsonParses to distinguish a
// malformed bundle (FAIL) from absent tensor data (SKIP). It must be false only
// for genuinely unparseable JSON, regardless of whether .bin files are present.
TEST_F(TestBundleDiscoveryFixture, PreloadCheckGraphJsonParsesRejectsMalformedAcceptsValid)
{
    auto dir = _tempDir / "bundle";
    std::filesystem::create_directories(dir);

    const auto goodPath = dir / "good.json";
    std::ofstream(goodPath) << R"({"tensors": []})";
    EXPECT_TRUE(checkBundlePreload(goodPath).graphJsonParses);

    const auto badPath = dir / "bad.json";
    std::ofstream(badPath) << "{{NOT VALID JSON AT ALL";
    EXPECT_FALSE(checkBundlePreload(badPath).graphJsonParses);

    EXPECT_FALSE(checkBundlePreload(dir / "does_not_exist.json").graphJsonParses);
}

// checkBundlePreload().tensorDataPresent drives the SKIP decision: true only
// when every tensor's companion .bin exists. A valid graph with no .bin files
// (DVC not pulled) must be false so the harness skips rather than fails.
TEST_F(TestBundleDiscoveryFixture, PreloadCheckTensorDataPresentDetectsMissingBinFiles)
{
    auto dir = _tempDir / "bundle";
    std::filesystem::create_directories(dir);
    const auto jsonPath = dir / "b.json";
    std::ofstream(jsonPath) << R"({"tensors": [{"uid": 0}, {"uid": 1}]})";

    // No .bin files yet -> data not present (but JSON still parses).
    {
        const auto check = checkBundlePreload(jsonPath);
        EXPECT_TRUE(check.graphJsonParses);
        EXPECT_FALSE(check.tensorDataPresent);
    }

    // Only one of two blobs present -> still not present.
    std::ofstream(dir / "b.tensor0.bin") << "x";
    EXPECT_FALSE(checkBundlePreload(jsonPath).tensorDataPresent);

    // Both blobs present -> data present.
    std::ofstream(dir / "b.tensor1.bin") << "y";
    EXPECT_TRUE(checkBundlePreload(jsonPath).tensorDataPresent);
}

// loadIntegrationTestBundle() must fully populate the bundle from disk: every
// tensor loaded, the single output tensor's data captured as golden reference,
// and metadata left empty when no .meta.json companion exists.
TEST_F(TestBundleDiscoveryFixture, LoadBundlePopulatesAllFields)
{
    auto dir = _tempDir / "op" / "loadtest";
    createLoadableBundle(dir, "loadtest");
    const auto jsonPath = dir / "loadtest.json";

    auto bundle = loadIntegrationTestBundle(jsonPath);

    // graphAndTensors should contain all 6 tensors (uids 0-5), but the output
    // tensor (uid 5) has been zeroed by extractAndClearOutputTensorData.
    EXPECT_EQ(bundle.graphAndTensors.tensorMap.size(), 6u);
    EXPECT_EQ(bundle.graphAndTensors.outputTensorUids.size(), 1u);
    EXPECT_EQ(bundle.graphAndTensors.outputTensorUids.front(), 5);

    // goldenOutputs is present (this bundle has an output tensor) and holds the
    // original data for that output tensor.
    ASSERT_TRUE(bundle.goldenOutputs.has_value());
    EXPECT_EQ(bundle.goldenOutputs->size(), 1u);
    EXPECT_NE(bundle.goldenOutputs->find(5), bundle.goldenOutputs->end());

    // metadata is optional — this bundle has no .meta.json so it should be empty.
    EXPECT_FALSE(bundle.metadata.has_value());
}

// When a .meta.json companion is present, loadIntegrationTestBundle() must
// surface it via the optional metadata field.
TEST_F(TestBundleDiscoveryFixture, LoadBundlePopulatesMetadataWhenPresent)
{
    auto dir = _tempDir / "op" / "withmeta";
    createLoadableBundle(dir, "withmeta");
    std::ofstream(dir / "withmeta.meta.json")
        << R"({"format_version": 1, "operation": "BatchnormInference", "seed": 42})";
    const auto jsonPath = dir / "withmeta.json";

    auto bundle = loadIntegrationTestBundle(jsonPath);

    ASSERT_TRUE(bundle.metadata.has_value());
    EXPECT_EQ(bundle.metadata->operation, "BatchnormInference");
    ASSERT_TRUE(bundle.metadata->seed.has_value());
    EXPECT_EQ(*bundle.metadata->seed, 42);
}

// loadIntegrationTestBundle() is the happy-path loader: it must throw when a
// tensor's .bin blob is absent rather than returning a partial bundle. The
// harness relies on this (guarded by checkBundlePreload) to turn missing DVC
// data into a SKIP.
TEST_F(TestBundleDiscoveryFixture, LoadBundleThrowsOnMissingBin)
{
    auto dir = _tempDir / "op" / "nobin";
    createMinimalBundle(dir, "nobin");
    const auto jsonPath = dir / "nobin.json";

    EXPECT_THROW(loadIntegrationTestBundle(jsonPath), std::exception);
}

// NOLINTEND(readability-identifier-naming)
