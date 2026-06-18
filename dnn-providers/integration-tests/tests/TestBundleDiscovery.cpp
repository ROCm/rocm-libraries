// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <variant>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>

#include "harness/golden/BundleDiscovery.hpp"
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

    // Writes a valid {name}.meta.json companion. Metadata is mandatory for any
    // bundle expected to load successfully (loadIntegrationTestBundle returns
    // LoadError::MISSING_METADATA without it).
    static void writeMetadata(const std::filesystem::path& dir, const std::string& name)
    {
        std::ofstream(dir / (name + ".meta.json"))
            << R"({"format_version": 1, "operation": "BatchnormInference"})";
    }

    // Writes a minimal bundle with JSON + .bin tensor data files + metadata so
    // that loadIntegrationTestBundle can fully load it. Tensor dims/strides match
    // createMinimalBundle; each .bin is zero-filled to the exact byte count the
    // loader expects.
    static void createLoadableBundle(const std::filesystem::path& dir, const std::string& name)
    {
        createMinimalBundle(dir, name);
        writeMetadata(dir, name);
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
    // Case 1: the structured integration_test_bundles tier layout. Every
    // directory segment below the root joins into the suite with '_', the file
    // stem is the test.
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

TEST_F(TestBundleDiscoveryFixture, EmptyLeafFolderWarnsAndSkips)
{
    // An empty leaf folder is warned and skipped; the valid sibling is
    // still discovered. The binary does not abort.
    createMinimalBundle(_tempDir / "conv" / "good", "good");
    std::filesystem::create_directories(_tempDir / "conv" / "case_12312"); // empty leaf
    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().testName, "good");
}

TEST_F(TestBundleDiscoveryFixture, LeafWithOnlyMetaJsonWarnsAndSkips)
{
    // A leaf holding only meta companions (no graph) is still "empty" —
    // warned and skipped, returns empty.
    auto dir = _tempDir / "conv" / "meta_only";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "meta.json") << "{}";
    auto result = discoverBundles(_tempDir);
    EXPECT_TRUE(result.empty());
}

TEST_F(TestBundleDiscoveryFixture, EmptyRootReturnsEmpty)
{
    // A completely empty data root is itself an empty leaf — warned and
    // skipped. Returns empty rather than aborting the binary.
    auto result = discoverBundles(_tempDir);
    EXPECT_TRUE(result.empty());
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

TEST(TestGraphFile, AllowlistsGraphsAndExcludesCompanions)
{
    // Plain graph .json files are graphs.
    EXPECT_TRUE(isGraphFile("dir/resnet50.json"));
    EXPECT_TRUE(isGraphFile("Small.json"));

    // Known companion kinds are excluded: "{Name}.meta.json" and bare "meta.json".
    EXPECT_FALSE(isGraphFile("dir/resnet50.meta.json"));
    EXPECT_FALSE(isGraphFile("dir/meta.json"));

    // A graph name that merely embeds dots is NOT a companion — only recognized
    // companion kinds are excluded. These must still be discovered as graphs so
    // ad-hoc "drop a folder, it runs" bundles are never silently dropped.
    EXPECT_TRUE(isGraphFile("dir/model.fp16.json"));
    EXPECT_TRUE(isGraphFile("dir/resnet50.v2.json"));
    // "claims" is not a companion kind yet, so {Name}.claims.json is a graph
    // today; it becomes a companion only once "claims" is added to companionKinds().
    EXPECT_TRUE(isGraphFile("dir/resnet50.claims.json"));

    // Non-.json files are never graphs.
    EXPECT_FALSE(isGraphFile("dir/resnet50.bin"));
    EXPECT_FALSE(isGraphFile("dir/resnet50.tensor0.bin"));
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

// loadIntegrationTestBundle() returns a fully-loaded bundle (the variant's
// IntegrationTestBundle alternative) when graph + all .bin tensor data +
// metadata are present: graph buffer captured, every tensor loaded, output UIDs
// derived, and metadata populated.
TEST_F(TestBundleDiscoveryFixture, LoadBundlePopulatesAllFields)
{
    auto dir = _tempDir / "op" / "loadtest";
    createLoadableBundle(dir, "loadtest");
    const auto jsonPath = dir / "loadtest.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    auto& bundle = std::get<IntegrationTestBundle>(result);

    // Output UIDs are derived from the graph (available regardless of tensor
    // data): this bundle has one output, uid 5.
    ASSERT_EQ(bundle.outputTensorUids.size(), 1u);
    EXPECT_EQ(bundle.outputTensorUids.front(), 5);

    // tensors present: all 6 tensors (uids 0-5) loaded with their data. Golden
    // extraction (saving + zeroing outputs) happens in the harness, not here, so
    // the output tensor still holds its loaded data at this point.
    ASSERT_TRUE(bundle.tensors.has_value());
    EXPECT_EQ(bundle.tensors->size(), 6u);
    EXPECT_NE(bundle.tensors->find(5), bundle.tensors->end());

    // metadata is mandatory and therefore populated.
    ASSERT_TRUE(bundle.metadata.operation.has_value());
    EXPECT_EQ(*bundle.metadata.operation, "BatchnormInference");
}

// The metadata companion's values are surfaced on the loaded bundle.
TEST_F(TestBundleDiscoveryFixture, LoadBundlePopulatesMetadataWhenPresent)
{
    auto dir = _tempDir / "op" / "withmeta";
    createMinimalBundle(dir, "withmeta");
    std::ofstream(dir / "withmeta.meta.json")
        << R"({"format_version": 1, "operation": "BatchnormInference", "seed": 42})";
    const auto jsonPath = dir / "withmeta.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    auto& bundle = std::get<IntegrationTestBundle>(result);

    ASSERT_TRUE(bundle.metadata.operation.has_value());
    EXPECT_EQ(*bundle.metadata.operation, "BatchnormInference");
    ASSERT_TRUE(bundle.metadata.seed.has_value());
    EXPECT_EQ(*bundle.metadata.seed, 42);
}

// A valid-graph bundle WITHOUT a .meta.json companion is a load error: metadata
// is mandatory.
TEST_F(TestBundleDiscoveryFixture, LoadBundleMissingMetadataIsError)
{
    auto dir = _tempDir / "op" / "nometa";
    createMinimalBundle(dir, "nometa"); // graph only, no .meta.json
    const auto jsonPath = dir / "nometa.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MISSING_METADATA);
}

// A graph whose tensors have no .bin blobs (but which has metadata) loads as a
// graph-only bundle: the tensors optional is absent (harness SKIPs), but it is
// NOT an error. Output UIDs are still derived from the graph.
TEST_F(TestBundleDiscoveryFixture, LoadBundleMissingBinIsGraphOnly)
{
    auto dir = _tempDir / "op" / "nobin";
    createMinimalBundle(dir, "nobin");
    writeMetadata(dir, "nobin"); // metadata is mandatory even for graph-only
    const auto jsonPath = dir / "nobin.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);

    EXPECT_FALSE(bundle.tensors.has_value());
    EXPECT_EQ(bundle.outputTensorUids.size(), 1u);
}

// A .bin that is present but the wrong size is present-but-broken data: the
// loader catches the underlying throw and classifies it as TENSOR_LOAD_FAILED
// (FAIL) rather than letting an exception escape or treating it as graph-only.
TEST_F(TestBundleDiscoveryFixture, LoadBundleWrongSizeBinIsTensorLoadError)
{
    auto dir = _tempDir / "op" / "badbin";
    createLoadableBundle(dir, "badbin"); // writes correct .bin + metadata
    // Overwrite one blob with the wrong byte count (uid 0 should be 480 bytes).
    std::ofstream(dir / "badbin.tensor0.bin", std::ios::binary) << "too short";
    const auto jsonPath = dir / "badbin.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::TENSOR_LOAD_FAILED);
}

// A "graph-only" bundle is a COMPLETE, valid graph whose tensor .bin blobs are
// simply not on disk (covered by LoadBundleMissingBinIsGraphOnly). A JSON that
// merely omits the required top-level graph fields (here, the "tensors" key) is
// NOT a graph-only bundle — it cannot build a flatbuffer graph and is reported
// as an InvalidGraphSchema error (FAIL), not a skip.
TEST_F(TestBundleDiscoveryFixture, LoadBundleMissingTensorsKeyIsSchemaError)
{
    auto dir = _tempDir / "op" / "notensorskey";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "notensorskey.json") << R"({"nodes": []})";

    auto result = loadIntegrationTestBundle(dir / "notensorskey.json");
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::INVALID_GRAPH_SCHEMA);
}

// Malformed JSON is a load error (the variant's LoadError alternative), which
// the harness turns into a FAIL.
TEST_F(TestBundleDiscoveryFixture, LoadBundleMalformedJsonIsError)
{
    auto dir = _tempDir / "op" / "badjson";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "badjson.json") << "{{NOT VALID";

    auto result = loadIntegrationTestBundle(dir / "badjson.json");
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MALFORMED_JSON);
}

// A nonexistent graph file is reported as a malformed-JSON load error (it cannot
// be opened or parsed), not a crash.
TEST_F(TestBundleDiscoveryFixture, LoadBundleMissingFileIsError)
{
    auto result = loadIntegrationTestBundle(_tempDir / "does_not_exist.json");
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MALFORMED_JSON);
}

// NOLINTEND(readability-identifier-naming)
