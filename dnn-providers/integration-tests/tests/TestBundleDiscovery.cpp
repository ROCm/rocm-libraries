// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

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
    struct SweepCaseSpec
    {
        std::string id;
        std::string ioDataType;
        std::vector<int64_t> xDims;
        std::vector<int64_t> xStrides;
        std::vector<int64_t> derivedDims;
        std::vector<int64_t> derivedStrides;
        bool includeGolden = true;
        bool goldenHasPath = true;
        bool includeMetadata = true;
    };

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

    // Writes a valid {name}.meta.json companion. Metadata is mandatory for a
    // golden bundle (one shipping output .bin blobs) and optional for graph-only
    // bundle cases.
    static void writeMetadata(const std::filesystem::path& dir, const std::string& name)
    {
        std::ofstream(dir / (name + ".meta.json"))
            << R"({"format_version": 1, "operation": "BatchnormInference"})";
    }

    static void createLoadableBundle(const std::filesystem::path& dir, const std::string& name)
    {
        createMinimalBundle(dir, name);
        writeMetadata(dir, name);
        const auto basePath = dir / name;

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

    static size_t elementSizeBytes(const std::string& dataType)
    {
        if(dataType == "float")
        {
            return sizeof(float);
        }
        if(dataType == "half" || dataType == "bfloat16")
        {
            return 2;
        }

        throw std::runtime_error("Unsupported test data type: " + dataType);
    }

    static void writeSweepTensorFile(const std::filesystem::path& dir,
                                     int64_t uid,
                                     const std::vector<int64_t>& dims,
                                     const std::string& dataType)
    {
        size_t elementCount = 1;
        for(const auto dim : dims)
        {
            elementCount *= static_cast<size_t>(dim);
        }

        std::vector<char> data(elementCount * elementSizeBytes(dataType), 0);
        std::ofstream out(dir / ("tensor" + std::to_string(uid) + ".bin"), std::ios::binary);
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }

    static nlohmann::json makeTemplateTensor(int64_t uid)
    {
        return nlohmann::json{{"name", ""},
                              {"uid", uid},
                              {"strides", "${case.strides}"},
                              {"dims", "${case.dims}"},
                              {"data_type", "${case.data_type}"},
                              {"virtual", false}};
    }

    static nlohmann::json makeSweepTensor(int64_t uid,
                                          const std::string& dataType,
                                          const std::vector<int64_t>& dims,
                                          const std::vector<int64_t>& strides)
    {
        return nlohmann::json{
            {"uid", uid}, {"data_type", dataType}, {"dims", dims}, {"strides", strides}};
    }

    static void createTemplateSweep(const std::filesystem::path& dir,
                                    const std::vector<SweepCaseSpec>& cases)
    {
        std::filesystem::create_directories(dir);

        const nlohmann::json templateJson
            = {{"nodes",
                nlohmann::json::array({{{"inputs",
                                         {{"x_tensor_uid", 0},
                                          {"mean_tensor_uid", 1},
                                          {"inv_variance_tensor_uid", 2},
                                          {"scale_tensor_uid", 3},
                                          {"bias_tensor_uid", 4}}},
                                        {"outputs", {{"y_tensor_uid", 5}}},
                                        {"type", "BatchnormInferenceAttributes"},
                                        {"compute_data_type", "float"},
                                        {"name", ""}}})},
               {"tensors",
                nlohmann::json::array({makeTemplateTensor(0),
                                       makeTemplateTensor(1),
                                       makeTemplateTensor(2),
                                       makeTemplateTensor(3),
                                       makeTemplateTensor(4),
                                       makeTemplateTensor(5)})},
               {"io_data_type", "${case.io_data_type}"},
               {"compute_data_type", "float"},
               {"intermediate_data_type", "float"},
               {"name", ""}};

        std::ofstream(dir / "graph.template.json") << templateJson.dump(2);

        nlohmann::json sweepJson = {{"version", 1}, {"cases", nlohmann::json::array()}};
        for(const auto& spec : cases)
        {
            nlohmann::json caseJson
                = {{"id", spec.id},
                   {"values",
                    {{"io_data_type", spec.ioDataType},
                     {"tensors",
                      nlohmann::json::array(
                          {makeSweepTensor(0, spec.ioDataType, spec.xDims, spec.xStrides),
                           makeSweepTensor(1, "float", spec.derivedDims, spec.derivedStrides),
                           makeSweepTensor(2, "float", spec.derivedDims, spec.derivedStrides),
                           makeSweepTensor(3, "float", spec.derivedDims, spec.derivedStrides),
                           makeSweepTensor(4, "float", spec.derivedDims, spec.derivedStrides),
                           makeSweepTensor(5, spec.ioDataType, spec.xDims, spec.xStrides)})}}}};

            if(spec.includeGolden)
            {
                caseJson["golden"]
                    = spec.goldenHasPath
                          ? nlohmann::json{{"path", "golden/" + spec.id + "/tensors.dvc"}}
                          : nlohmann::json::object();

                if(spec.goldenHasPath)
                {
                    const auto goldenDir = dir / "golden" / spec.id;
                    std::filesystem::create_directories(goldenDir);
                    std::ofstream(goldenDir / "tensors.dvc") << "outs:\n";
                    writeSweepTensorFile(goldenDir, 0, spec.xDims, spec.ioDataType);
                    writeSweepTensorFile(goldenDir, 1, spec.derivedDims, "float");
                    writeSweepTensorFile(goldenDir, 2, spec.derivedDims, "float");
                    writeSweepTensorFile(goldenDir, 3, spec.derivedDims, "float");
                    writeSweepTensorFile(goldenDir, 4, spec.derivedDims, "float");
                    writeSweepTensorFile(goldenDir, 5, spec.xDims, spec.ioDataType);
                }
            }

            if(spec.includeMetadata)
            {
                caseJson["metadata"] = nlohmann::json{
                    {"format_version", 1}, {"operation", "BatchnormInference"}, {"seed", 42}};
            }

            sweepJson["cases"].push_back(std::move(caseJson));
        }

        std::ofstream(dir / "sweep.json") << sweepJson.dump(2);
    }

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
    createMinimalBundle(_tempDir / "case_23421", "graph");

    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().suiteName, "case_23421");
    EXPECT_EQ(result.front().testName, "graph");
}

TEST_F(TestBundleDiscoveryFixture, TieredGoldenDataLayoutIsDiscovered)
{
    createMinimalBundle(_tempDir / "quick" / "BatchnormFwdInference" / "ncdhw" / "fp32" / "Small",
                        "Small");

    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().suiteName, "quick_BatchnormFwdInference_ncdhw_fp32_Small");
    EXPECT_EQ(result.front().testName, "Small");
}

TEST_F(TestBundleDiscoveryFixture, TemplateSweepCasesAreExpandedFromManifest)
{
    createTemplateSweep(
        _tempDir / "quick" / "BatchnormFwdInference" / "Inference",
        {{"small_fp32_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}},
         {"small_fp16_nchw", "half", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});

    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 2u);

    const auto* fp32 = findByTest(result, "small_fp32_nchw");
    ASSERT_NE(fp32, nullptr);
    EXPECT_TRUE(fp32->isTemplateSweepCase());
    EXPECT_EQ(fp32->suiteName, "quick_BatchnormFwdInference_Inference");
    EXPECT_EQ(fp32->jsonPath,
              _tempDir / "quick" / "BatchnormFwdInference" / "Inference" / "sweep.json");
    EXPECT_EQ(fp32->sweep->templatePath,
              _tempDir / "quick" / "BatchnormFwdInference" / "Inference" / "graph.template.json");
    EXPECT_EQ(fp32->sweep->caseId, "small_fp32_nchw");

    const auto* fp16 = findByTest(result, "small_fp16_nchw");
    ASSERT_NE(fp16, nullptr);
    EXPECT_TRUE(fp16->isTemplateSweepCase());
    EXPECT_EQ(fp16->suiteName, "quick_BatchnormFwdInference_Inference");
    EXPECT_EQ(fp16->sweep->caseId, "small_fp16_nchw");
}

TEST_F(TestBundleDiscoveryFixture, JsonAtRootUsesFolderNameAsSuite)
{
    // A .json directly at the data root uses the root folder name as suite.
    std::ofstream(_tempDir / "graph.json") << R"({"tensors": []})";
    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].suiteName, sanitizeForGtest(_tempDir.filename().string()));
    EXPECT_EQ(result[0].testName, "graph");
}

TEST_F(TestBundleDiscoveryFixture, EmptyLeafFolderWarnsAndSkips)
{
    createMinimalBundle(_tempDir / "conv" / "good", "good");
    std::filesystem::create_directories(_tempDir / "conv" / "case_12312");
    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().testName, "good");
}

TEST_F(TestBundleDiscoveryFixture, LeafWithOnlyMetaJsonWarnsAndSkips)
{
    auto dir = _tempDir / "conv" / "meta_only";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "meta.json") << "{}";
    auto result = discoverBundles(_tempDir);
    EXPECT_TRUE(result.empty());
}

TEST_F(TestBundleDiscoveryFixture, EmptyRootReturnsEmpty)
{
    auto result = discoverBundles(_tempDir);
    EXPECT_TRUE(result.empty());
}

TEST_F(TestBundleDiscoveryFixture, CollisionThrows)
{
    createMinimalBundle(_tempDir / "Op-A" / "case", "SameName");
    createMinimalBundle(_tempDir / "Op_A" / "case", "SameName");
    EXPECT_THROW(discoverBundles(_tempDir), std::runtime_error);
}

TEST_F(TestBundleDiscoveryFixture, CustomerDropAndTieredLayoutCoexistUnderOneRoot)
{
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
    auto bundleDir = _tempDir / "conv" / "nchw" / "fp32" / "withmeta";
    createMinimalBundle(bundleDir, "withmeta");
    std::ofstream(bundleDir / "withmeta.meta.json") << "{}";
    std::ofstream(bundleDir / "meta.json") << "{}";

    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().testName, "withmeta");
}

TEST_F(TestBundleDiscoveryFixture, ScanFilesByExtensionIsGenericAndSorted)
{
    std::filesystem::create_directories(_tempDir / "b");
    std::filesystem::create_directories(_tempDir / "a");
    std::ofstream(_tempDir / "b" / "z.json") << "{}";
    std::ofstream(_tempDir / "a" / "m.json") << "{}";
    std::ofstream(_tempDir / "a" / "ignore.txt") << "x";

    auto files = scanFilesByExtension(_tempDir, ".json");
    ASSERT_EQ(files.size(), 2u);
    EXPECT_EQ(files[0].filename(), "m.json");
    EXPECT_EQ(files[1].filename(), "z.json");
}

TEST(TestGraphFile, AllowlistsGraphsAndExcludesCompanions)
{
    EXPECT_TRUE(isGraphFile("dir/resnet50.json"));
    EXPECT_TRUE(isGraphFile("Small.json"));

    EXPECT_FALSE(isGraphFile("dir/resnet50.meta.json"));
    EXPECT_FALSE(isGraphFile("dir/meta.json"));
    EXPECT_FALSE(isGraphFile("dir/graph.template.json"));
    EXPECT_FALSE(isGraphFile("dir/sweep.json"));

    EXPECT_TRUE(isGraphFile("dir/model.fp16.json"));
    EXPECT_TRUE(isGraphFile("dir/resnet50.v2.json"));
    EXPECT_TRUE(isGraphFile("dir/resnet50.claims.json"));

    EXPECT_FALSE(isGraphFile("dir/resnet50.bin"));
    EXPECT_FALSE(isGraphFile("dir/resnet50.tensor0.bin"));
}

TEST(TestSanitizeForGtest, ReplacesInvalidChars)
{
    EXPECT_EQ(sanitizeForGtest("resnet50-layer3.v2"), "resnet50_layer3_v2");
    EXPECT_EQ(sanitizeForGtest("name with spaces"), "name_with_spaces");
    EXPECT_EQ(sanitizeForGtest("already_ok"), "already_ok");
}

TEST_F(TestBundleDiscoveryFixture, UnparseableJsonIsDiscoveredButLoadThrows)
{
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

TEST_F(TestBundleDiscoveryFixture, LoadBundlePopulatesAllFields)
{
    auto dir = _tempDir / "op" / "loadtest";
    createLoadableBundle(dir, "loadtest");
    const auto jsonPath = dir / "loadtest.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    auto& bundle = std::get<IntegrationTestBundle>(result);

    ASSERT_EQ(bundle.outputTensorUids.size(), 1u);
    EXPECT_EQ(bundle.outputTensorUids.front(), 5);

    ASSERT_TRUE(bundle.tensors.has_value());
    EXPECT_EQ(bundle.tensors->size(), 6u);
    EXPECT_NE(bundle.tensors->find(5), bundle.tensors->end());

    ASSERT_TRUE(bundle.metadata.operation.has_value());
    EXPECT_EQ(*bundle.metadata.operation, "BatchnormInference");
}

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

// A graph-only bundle (no .bin blobs, hence no golden data) without a .meta.json
// companion loads successfully: metadata validates golden data, and there is
// none here, so absent metadata is valid and default-constructed.
TEST_F(TestBundleDiscoveryFixture, LoadGraphOnlyBundleMissingMetadataLoads)
{
    auto dir = _tempDir / "op" / "nometa";
    createMinimalBundle(dir, "nometa"); // graph only, no .meta.json, no .bin
    const auto jsonPath = dir / "nometa.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);

    EXPECT_FALSE(bundle.tensors.has_value()); // graph-only: no tensor data
    EXPECT_FALSE(bundle.hasGoldenOutputs);
    EXPECT_FALSE(bundle.metadata.operation.has_value()); // default-constructed
}

// A GOLDEN bundle (output .bin blobs present) WITHOUT a .meta.json companion is
// a load error: metadata is mandatory whenever there is golden data to validate.
TEST_F(TestBundleDiscoveryFixture, LoadGoldenBundleMissingMetadataIsError)
{
    auto dir = _tempDir / "op" / "goldennometa";
    createLoadableBundle(dir, "goldennometa"); // writes .bin (inputs+outputs) + meta
    std::filesystem::remove(dir / "goldennometa.meta.json"); // drop the metadata
    const auto jsonPath = dir / "goldennometa.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MISSING_METADATA);
}

TEST_F(TestBundleDiscoveryFixture, LoadBundleMissingBinIsGraphOnly)
{
    auto dir = _tempDir / "op" / "nobin";
    createMinimalBundle(dir, "nobin");
    writeMetadata(dir, "nobin"); // metadata present (optional here, but exercised)
    const auto jsonPath = dir / "nobin.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);

    EXPECT_FALSE(bundle.tensors.has_value());
    EXPECT_EQ(bundle.outputTensorUids.size(), 1u);
}

TEST_F(TestBundleDiscoveryFixture, LoadTemplateSweepCasePopulatesExpandedGraphAndTensorData)
{
    createTemplateSweep(
        _tempDir / "quick" / "BatchnormFwdInference" / "Inference",
        {{"small_fp16_nchw", "half", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);
    ASSERT_TRUE(discovered.front().isTemplateSweepCase());

    auto result = loadIntegrationTestBundle(discovered.front());
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);

    ASSERT_TRUE(bundle.tensors.has_value());
    ASSERT_EQ(bundle.outputTensorUids.size(), 1u);
    EXPECT_EQ(bundle.outputTensorUids.front(), 5);
    EXPECT_EQ(bundle.tensors->at(0)->dims(), (std::vector<int64_t>{2, 3, 4, 5}));
    EXPECT_EQ(bundle.tensors->at(0)->strides(), (std::vector<int64_t>{60, 20, 5, 1}));
    EXPECT_EQ(bundle.tensors->at(5)->dims(), (std::vector<int64_t>{2, 3, 4, 5}));

    const auto tensorAttrMap = bundle.graphWrapper().getTensorMap();
    EXPECT_EQ(tensorAttrMap.at(0)->data_type(),
              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(tensorAttrMap.at(1)->data_type(),
              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    ASSERT_TRUE(bundle.metadata.operation.has_value());
    EXPECT_EQ(*bundle.metadata.operation, "BatchnormInference");
    ASSERT_TRUE(bundle.metadata.seed.has_value());
    EXPECT_EQ(*bundle.metadata.seed, 42);
}

TEST_F(TestBundleDiscoveryFixture, LoadTemplateSweepCaseWithoutGoldenIsGraphOnly)
{
    createTemplateSweep(_tempDir / "quick" / "BatchnormFwdInference" / "Inference",
                        {{"graph_only_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          false}});

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto result = loadIntegrationTestBundle(discovered.front());
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);
    EXPECT_FALSE(bundle.tensors.has_value());
}

TEST_F(TestBundleDiscoveryFixture, LoadTemplateSweepCaseMissingGoldenPathIsError)
{
    createTemplateSweep(_tempDir / "quick" / "BatchnormFwdInference" / "Inference",
                        {{"missing_path_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          true,
                          false}});

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto result = loadIntegrationTestBundle(discovered.front());
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::INVALID_SWEEP_CASE);
}

TEST_F(TestBundleDiscoveryFixture, LoadTemplateSweepCaseMissingTensorValueIsError)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"missing_tensor_value_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1}}});

    auto sweepJson = nlohmann::json::parse(std::ifstream(sweepDir / "sweep.json"));
    sweepJson["cases"][0]["values"]["tensors"].erase(
        sweepJson["cases"][0]["values"]["tensors"].begin());
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto result = loadIntegrationTestBundle(discovered.front());
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::INVALID_SWEEP_CASE);
}

TEST_F(TestBundleDiscoveryFixture, LoadBundleWrongSizeBinIsTensorLoadError)
{
    auto dir = _tempDir / "op" / "badbin";
    createLoadableBundle(dir, "badbin");
    std::ofstream(dir / "badbin.tensor0.bin", std::ios::binary) << "too short";
    const auto jsonPath = dir / "badbin.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::TENSOR_LOAD_FAILED);
}

TEST_F(TestBundleDiscoveryFixture, LoadBundleMissingTensorsKeyIsSchemaError)
{
    auto dir = _tempDir / "op" / "notensorskey";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "notensorskey.json") << R"({"nodes": []})";

    auto result = loadIntegrationTestBundle(dir / "notensorskey.json");
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::INVALID_GRAPH_SCHEMA);
}

TEST_F(TestBundleDiscoveryFixture, LoadBundleMalformedJsonIsError)
{
    auto dir = _tempDir / "op" / "badjson";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "badjson.json") << "{{NOT VALID";

    auto result = loadIntegrationTestBundle(dir / "badjson.json");
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MALFORMED_JSON);
}

TEST_F(TestBundleDiscoveryFixture, LoadBundleMissingFileIsError)
{
    auto result = loadIntegrationTestBundle(_tempDir / "does_not_exist.json");
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MALFORMED_JSON);
}

// A golden sweep case that omits its inline `metadata` block is rejected with
// MISSING_METADATA: metadata is what validates the golden data.
TEST_F(TestBundleDiscoveryFixture, LoadTemplateSweepCaseGoldenWithoutMetadataIsError)
{
    createTemplateSweep(_tempDir / "quick" / "BatchnormFwdInference" / "Inference",
                        {{"golden_no_meta_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          true, // includeGolden
                          true, // goldenHasPath
                          false}}); // includeMetadata

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto result = loadIntegrationTestBundle(discovered.front());
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MISSING_METADATA);
}

// Every sweep case must carry metadata, golden or not: a graph-only case that
// omits its metadata block is also MISSING_METADATA.
TEST_F(TestBundleDiscoveryFixture, LoadTemplateSweepCaseWithoutMetadataIsError)
{
    createTemplateSweep(_tempDir / "quick" / "BatchnormFwdInference" / "Inference",
                        {{"graph_only_no_meta_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          false, // includeGolden
                          false, // goldenHasPath
                          false}}); // includeMetadata

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto result = loadIntegrationTestBundle(discovered.front());
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::MISSING_METADATA);
}

// A sweep root whose manifest has two cases sharing the same id is rejected
// wholesale: readSweepCaseIds throws on the duplicate and discoverBundles lets
// it propagate, so a broken checked-in manifest fails the run instead of
// silently dropping sibling cases.
TEST_F(TestBundleDiscoveryFixture, DuplicateSweepCaseIdThrows)
{
    createTemplateSweep(
        _tempDir / "quick" / "BatchnormFwdInference" / "Inference",
        {{"dup_case_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}},
         {"dup_case_nchw", "half", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});

    EXPECT_THROW(discoverBundles(_tempDir), std::runtime_error);
}

// resolvePlaceholder falls back to the case-level `values` map when a
// placeholder key is absent from the per-tensor entry. A top-level template
// placeholder (no current tensor uid) resolves straight from the global value.
TEST_F(TestBundleDiscoveryFixture, ExpandTemplateResolvesPlaceholderFromGlobalCaseValue)
{
    const nlohmann::json templateJson
        = {{"io_data_type", "${case.io_data_type}"},
           {"tensors", nlohmann::json::array({{{"uid", 0}, {"data_type", "float"}}})}};

    // io_data_type is supplied only at the top level of `values`, not inside any
    // values.tensors[] entry.
    const nlohmann::json caseJson = {
        {"id", "global_fallback"},
        {"values", {{"io_data_type", "half"}, {"tensors", nlohmann::json::array({{{"uid", 0}}})}}}};

    DiscoveredBundle discovered;
    discovered.jsonPath = _tempDir / "sweep.json";

    const auto expanded = detail::expandTemplateGraph(templateJson, caseJson, discovered);
    EXPECT_EQ(expanded.at("io_data_type").get<std::string>(), "half");
}

// An unused top-level sweep value triggers a warning (warnUnusedSweepValues),
// not a load error. Assert the bundle still loads to keep the warning path
// distinct from the INVALID_SWEEP_CASE error path.
TEST_F(TestBundleDiscoveryFixture, UnusedSweepValueWarnsButLoadSucceeds)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"unused_value_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1}}});

    // Inject a top-level case value that no ${case.*} placeholder consumes.
    auto sweepJson = nlohmann::json::parse(std::ifstream(sweepDir / "sweep.json"));
    sweepJson["cases"][0]["values"]["extra_key"] = 99;
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto result = loadIntegrationTestBundle(discovered.front());
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
}

// NOLINTEND(readability-identifier-naming)
