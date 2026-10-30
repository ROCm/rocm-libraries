// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "BundleFixtureFiles.hpp"
#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

using namespace hipdnn_integration_tests::bundle;
using hipdnn_integration_tests::BundleMetadataError;
using hipdnn_test_sdk::utilities::claimScratchDirectory;

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
    // One load pass's sweep cache, as loadDiscoveredBundles() shares one.
    SweepManifestCache _sweeps;

    void SetUp() override
    {
        _scopedDir.emplace(claimScratchDirectory("bundle_discovery"));
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

    // Provenance-only metadata: valid JSON, but no `format_version`, so RFC 0011's
    // reader rejects it. This is the exact shape the SdpaFwd generator emitted for
    // 35 bundles, and the reason they vanished the moment their blobs were pulled.
    static void writeMetadataWithoutFormatVersion(const std::filesystem::path& dir,
                                                  const std::string& name)
    {
        std::ofstream(dir / (name + ".meta.json"))
            << R"({"generator": "generate_sdpa_fwd_golden.py", "seed": 42})";
    }

    // uid 5 is the only non-virtual output of createMinimalBundle's graph, so its
    // blob alone decides whether the bundle counts as golden.
    static void writeGoldenOutputBlob(const std::filesystem::path& dir, const std::string& name)
    {
        const std::vector<char> data(480, 0);
        std::ofstream out((dir / name).string() + ".tensor5.bin", std::ios::binary);
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }

    static void createLoadableBundle(const std::filesystem::path& dir, const std::string& name)
    {
        createMinimalBundle(dir, name);
        fixtures::writeMetadata(dir, name);
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

    // A one-node SDPA forward template shaped like quick/SdpaFwd/Default: the node keeps its
    // flags under "attributes" and its tensor uids under "inputs" and "outputs", and lists
    // every key, nulls included, because the JSON-to-flatbuffer converter reads each one.
    static nlohmann::json makeSdpaTemplate()
    {
        return nlohmann::json::parse(R"json({
  "compute_data_type": "float",
  "intermediate_data_type": "float",
  "io_data_type": "${case.io_data_type}",
  "name": "",
  "nodes": [
    {
      "attributes": {
        "alibi_mask": false,
        "attn_scale_value": 0.125,
        "causal_mask": false,
        "causal_mask_bottom_right": false,
        "diagonal_alignment": "BOTTOM_RIGHT",
        "dropout_probability": null,
        "generate_stats": null,
        "implementation": "AUTO",
        "left_bound": null,
        "max_seq_len_kv": null,
        "mma_core_mode": "unset",
        "padding_mask": false,
        "right_bound": 0
      },
      "compute_data_type": "float",
      "inputs": {
        "attn_mask_tensor_uid": null,
        "block_mask_tensor_uid": null,
        "descale_k_tensor_uid": null,
        "descale_q_tensor_uid": null,
        "descale_s_tensor_uid": null,
        "descale_v_tensor_uid": null,
        "dropout_mask_tensor_uid": null,
        "dropout_scale_tensor_uid": null,
        "k_tensor_uid": 0,
        "offset_tensor_uid": null,
        "page_table_k_tensor_uid": null,
        "page_table_v_tensor_uid": null,
        "q_tensor_uid": 1,
        "scale_o_tensor_uid": null,
        "scale_s_tensor_uid": null,
        "scale_tensor_uid": null,
        "seed_tensor_uid": null,
        "seq_len_kv_tensor_uid": null,
        "seq_len_q_tensor_uid": null,
        "sink_token_tensor_uid": null,
        "v_tensor_uid": 3
      },
      "name": "SdpaFwd_0",
      "outputs": {
        "amax_o_tensor_uid": null,
        "amax_s_tensor_uid": null,
        "max_tensor_uid": null,
        "o_tensor_uid": 2,
        "rng_dump_tensor_uid": null,
        "stats_tensor_uid": null,
        "sum_exp_tensor_uid": null
      },
      "type": "SdpaAttributes"
    }
  ],
  "tensors": [
    {"alignment": 16, "data_type": "${case.data_type}", "dims": "${case.dims}",
     "is_runtime_pass_by_value": false, "name": "K", "strides": "${case.strides}",
     "uid": 0, "virtual": false},
    {"alignment": 16, "data_type": "${case.data_type}", "dims": "${case.dims}",
     "is_runtime_pass_by_value": false, "name": "Q", "strides": "${case.strides}",
     "uid": 1, "virtual": false},
    {"alignment": 16, "data_type": "${case.data_type}", "dims": "${case.dims}",
     "is_runtime_pass_by_value": false, "name": "SdpaFwd_0::O", "strides": "${case.strides}",
     "uid": 2, "virtual": false},
    {"alignment": 16, "data_type": "${case.data_type}", "dims": "${case.dims}",
     "is_runtime_pass_by_value": false, "name": "V", "strides": "${case.strides}",
     "uid": 3, "virtual": false}
  ]
})json");
    }

    // A sweep case over makeSdpaTemplate() with no patch fields. Q, K, V and O are
    // [batch 1, heads 2, seq 16, head dim 8].
    static nlohmann::json makeSdpaCase(const std::string& id)
    {
        nlohmann::json tensors = nlohmann::json::array();
        for(int64_t uid = 0; uid < 4; ++uid)
        {
            tensors.push_back(makeSweepTensor(uid, "half", {1, 2, 16, 8}, {256, 128, 8, 1}));
        }
        return {{"id", id},
                {"values", {{"io_data_type", "half"}, {"tensors", tensors}}},
                {"metadata", {{"format_version", 1}, {"operation", "SdpaFwd"}, {"seed", 42}}}};
    }

    // The softmax statistics (log-sum-exp) tensor a space adds to switch stats on.
    static nlohmann::json makeStatsTensor(int64_t uid)
    {
        return {{"alignment", 16},
                {"data_type", "float"},
                {"dims", {1, 2, 16, 1}},
                {"is_runtime_pass_by_value", false},
                {"name", "SdpaFwd_0::Stats"},
                {"ragged_offset_multiplier", 1},
                {"strides", {32, 16, 1, 1}},
                {"uid", uid},
                {"virtual", false}};
    }

    static nlohmann::json expandSdpaCase(const nlohmann::json& caseJson)
    {
        DiscoveredBundle discovered;
        discovered.jsonPath = "sweep.json";
        return detail::expandTemplateGraph(makeSdpaTemplate(), caseJson, discovered);
    }

    // The graph a loaded case runs: expanded, then the patch fields in the harness's order.
    static nlohmann::json patchedSdpaGraph(const nlohmann::json& caseJson)
    {
        auto graph = expandSdpaCase(caseJson);
        detail::applyAddTensors(graph, caseJson);
        detail::applyNodePatches(graph, caseJson);
        detail::applyTensorPatches(graph, caseJson);
        return graph;
    }

    static const nlohmann::json& findTensor(const nlohmann::json& graph, int64_t uid)
    {
        for(const auto& tensor : graph.at("tensors"))
        {
            if(tensor.at("uid").get<int64_t>() == uid)
            {
                return tensor;
            }
        }
        throw std::out_of_range("no tensor with uid " + std::to_string(uid));
    }

    // Applies the patch fields of caseJson to graph and checks that the first failure names
    // the case and every expected part of the cause.
    static void expectPatchError(const nlohmann::json& graph,
                                 const nlohmann::json& caseJson,
                                 const std::vector<std::string>& expectedParts)
    {
        auto patched = graph;
        try
        {
            detail::applyAddTensors(patched, caseJson);
            detail::applyNodePatches(patched, caseJson);
            detail::applyTensorPatches(patched, caseJson);
            ADD_FAILURE() << "expected the patch fields to be rejected: " << caseJson.dump();
        }
        catch(const std::runtime_error& error)
        {
            const std::string message = error.what();
            const auto caseId = caseJson.at("id").get<std::string>();
            EXPECT_NE(message.find("sweep case '" + caseId + "'"), std::string::npos) << message;
            for(const auto& part : expectedParts)
            {
                EXPECT_NE(message.find(part), std::string::npos)
                    << "missing '" << part << "' in: " << message;
            }
        }
    }

    static void expectPatchError(const nlohmann::json& caseJson,
                                 const std::vector<std::string>& expectedParts)
    {
        expectPatchError(expandSdpaCase(caseJson), caseJson, expectedParts);
    }

    // makeSdpaCase(id) with one node_patches entry parsed from patchJson.
    static nlohmann::json makeSdpaCaseWithNodePatch(const std::string& id,
                                                    const std::string& patchJson)
    {
        auto caseJson = makeSdpaCase(id);
        caseJson["node_patches"] = nlohmann::json::array({nlohmann::json::parse(patchJson)});
        return caseJson;
    }

    // Writes a one-case sweep over makeSdpaTemplate() and loads that case.
    LoadResult loadSdpaCase(const nlohmann::json& caseJson) const
    {
        const auto sweepDir = _tempDir / "quick" / "SdpaFwd" / "Default";
        std::filesystem::create_directories(sweepDir);
        std::ofstream(sweepDir / "graph.template.json") << makeSdpaTemplate().dump(2);
        const nlohmann::json sweepJson
            = {{"version", 1}, {"cases", nlohmann::json::array({caseJson})}};
        std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

        const auto discovered = discoverBundles(_tempDir);
        return loadIntegrationTestBundle(discovered.at(0));
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

// A bundle root assembled from directory links (say quick/SdpaFwd linked to an
// installed tree) must discover what a copy of the same tree would, under the link's
// name, including a sweep that sits further down behind the link.
TEST_F(TestBundleDiscoveryFixture, DirectorySymlinksInsideTheRootAreFollowed)
{
    const auto root = _tempDir / "root";
    const auto outside = _tempDir / "outside";
    createMinimalBundle(outside / "Direct" / "nchw" / "Small", "Small");
    createTemplateSweep(
        outside / "Swept" / "Inference",
        {{"small_fp32_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});
    std::filesystem::create_directories(root / "quick");
    try
    {
        std::filesystem::create_directory_symlink(outside / "Direct", root / "quick" / "Direct");
        std::filesystem::create_directory_symlink(outside / "Swept", root / "quick" / "Swept");
    }
    catch(const std::filesystem::filesystem_error& e)
    {
        GTEST_SKIP() << "cannot create directory symlinks here: " << e.what();
    }

    const auto result = discoverBundles(root);
    ASSERT_EQ(result.size(), 2u);

    const auto* direct = findByTest(result, "Small");
    ASSERT_NE(direct, nullptr);
    EXPECT_EQ(direct->suiteName, "quick_Direct_nchw_Small");
    EXPECT_EQ(direct->jsonPath, root / "quick" / "Direct" / "nchw" / "Small" / "Small.json");

    const auto* swept = findByTest(result, "small_fp32_nchw");
    ASSERT_NE(swept, nullptr);
    EXPECT_EQ(swept->suiteName, "quick_Swept_Inference");
}

// A link back to its own ancestor would make the walk revisit the tree forever. It is
// not descended: discovery finishes and finds each bundle once.
TEST_F(TestBundleDiscoveryFixture, DirectorySymlinkToAnAncestorIsNotFollowed)
{
    createMinimalBundle(_tempDir / "conv" / "good", "good");
    try
    {
        std::filesystem::create_directory_symlink(_tempDir, _tempDir / "conv" / "loop");
    }
    catch(const std::filesystem::filesystem_error& e)
    {
        GTEST_SKIP() << "cannot create directory symlinks here: " << e.what();
    }

    const auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().suiteName, "conv_good");
}

// A cycle can span siblings: a/to_b -> b and b/to_a -> a. Neither link points at its
// own ancestor, but following both returns to a directory the walk is already
// inside. Discovery must stop there and still find what a copied tree holds: each
// bundle under its own folder and once more behind the link that reaches it.
TEST_F(TestBundleDiscoveryFixture, DirectorySymlinkCycleAcrossSiblingsIsNotFollowed)
{
    createMinimalBundle(_tempDir / "a" / "good", "good");
    createMinimalBundle(_tempDir / "b" / "fine", "fine");
    try
    {
        std::filesystem::create_directory_symlink(_tempDir / "b", _tempDir / "a" / "to_b");
        std::filesystem::create_directory_symlink(_tempDir / "a", _tempDir / "b" / "to_a");
    }
    catch(const std::filesystem::filesystem_error& e)
    {
        GTEST_SKIP() << "cannot create directory symlinks here: " << e.what();
    }

    const auto result = discoverBundles(_tempDir);
    std::vector<std::string> suites;
    suites.reserve(result.size());
    for(const auto& bundle : result)
    {
        suites.push_back(bundle.suiteName);
    }
    std::sort(suites.begin(), suites.end());
    EXPECT_EQ(suites, (std::vector<std::string>{"a_good", "a_to_b_fine", "b_fine", "b_to_a_good"}));
}

// A link out of the root to one of the root's own parents (think root/x -> $HOME)
// leads back to the root. It must not be followed: the walk would cover the whole
// parent tree first, here a bundle that sits beside the root, before it reached the
// root again.
TEST_F(TestBundleDiscoveryFixture, DirectorySymlinkToAParentOfTheRootIsNotFollowed)
{
    const auto root = _tempDir / "root";
    createMinimalBundle(root / "conv" / "good", "good");
    createMinimalBundle(_tempDir / "beside" / "other", "other");
    try
    {
        std::filesystem::create_directory_symlink(root.parent_path(), root / "conv" / "up");
    }
    catch(const std::filesystem::filesystem_error& e)
    {
        GTEST_SKIP() << "cannot create directory symlinks here: " << e.what();
    }

    const auto result = discoverBundles(root);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().suiteName, "conv_good");
}

// A directory the run may not list is skipped with a warning. Before, the walk threw
// on it and discovery failed, losing every bundle in the root.
TEST_F(TestBundleDiscoveryFixture, UnlistableDirectoryIsSkipped)
{
    namespace fs = std::filesystem;
    createMinimalBundle(_tempDir / "conv" / "good", "good");
    const auto locked = _tempDir / "locked";
    createMinimalBundle(locked / "hidden", "hidden");
    fs::permissions(locked, fs::perms::none);
    // Give the permissions back on every exit, so the scratch directory can be removed.
    struct RestorePermissions
    {
        fs::path path;
        ~RestorePermissions()
        {
            std::error_code error;
            fs::permissions(path, fs::perms::owner_all, error);
        }
    };
    const RestorePermissions restore{locked};

    std::error_code probeError;
    const fs::directory_iterator probe(locked, probeError);
    if(!probeError)
    {
        GTEST_SKIP() << "this process can still list a directory with no permissions";
    }

    std::vector<DiscoveredBundle> result;
    ASSERT_NO_THROW(result = discoverBundles(_tempDir));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().suiteName, "conv_good");
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

TEST_F(TestBundleDiscoveryFixture, SkipsSupportJson)
{
    auto bundleDir = _tempDir / "conv" / "nchw" / "fp32" / "withsupport";
    createMinimalBundle(bundleDir, "withsupport");
    std::ofstream(bundleDir / "withsupport.support.json") << R"({"version": 1, "claims": {}})";

    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().testName, "withsupport");
}

TEST_F(TestBundleDiscoveryFixture, SweepSupportJsonIsNotDiscoveredAsGraph)
{
    auto sweepDir = _tempDir / "conv" / "sweep";
    createTemplateSweep(sweepDir,
                        {SweepCaseSpec{"case0", "float", {2, 3}, {3, 1}, {2, 3}, {3, 1}}});
    std::ofstream(sweepDir / "support.json") << R"({"version": 1, "claims": {}})";

    auto result = discoverBundles(_tempDir);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result.front().isTemplateSweepCase());
    for(const auto& bundle : result)
    {
        EXPECT_NE(bundle.jsonPath.filename(), "support.json");
    }
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
    EXPECT_FALSE(isGraphFile("dir/resnet50.support.json"));
    EXPECT_FALSE(isGraphFile("dir/support.json"));

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

    ASSERT_TRUE(bundle.blobs.has_value());
    EXPECT_TRUE(bundle.hasGoldenOutputs);
    const auto tensors = bundle.loadTensors();
    EXPECT_EQ(tensors.size(), 6u);
    EXPECT_NE(tensors.find(5), tensors.end());

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

    EXPECT_FALSE(bundle.blobs.has_value()); // graph-only: no tensor data
    EXPECT_FALSE(bundle.hasGoldenOutputs);
    EXPECT_FALSE(bundle.metadata.operation.has_value()); // default-constructed
}

// A GOLDEN bundle (output .bin blobs present) WITHOUT a .meta.json companion is
// a load error: metadata is mandatory whenever there is golden data to validate.
// It is UNVALIDATABLE_GOLDEN_DATA rather than MISSING_METADATA because that
// distinction is what makes classifyBundle() fail instead of skip.
TEST_F(TestBundleDiscoveryFixture, LoadGoldenBundleMissingMetadataIsError)
{
    auto dir = _tempDir / "op" / "goldennometa";
    createLoadableBundle(dir, "goldennometa"); // writes .bin (inputs+outputs) + meta
    std::filesystem::remove(dir / "goldennometa.meta.json"); // drop the metadata
    const auto jsonPath = dir / "goldennometa.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::UNVALIDATABLE_GOLDEN_DATA);
}

// Present-but-malformed metadata is never softer than absent metadata: either way
// the golden blobs on disk have nothing usable describing how they were produced.
// A metadata file that parses as JSON but is rejected by the schema (here: no
// `format_version`) throws, so the parser's detail reaches the failure message.
TEST_F(TestBundleDiscoveryFixture, LoadGoldenBundleUnparseableMetadataIsError)
{
    auto dir = _tempDir / "op" / "goldenbadmeta";
    createLoadableBundle(dir, "goldenbadmeta");
    writeMetadataWithoutFormatVersion(dir, "goldenbadmeta");

    EXPECT_THROW(loadIntegrationTestBundle(dir / "goldenbadmeta.json"), BundleMetadataError);
}

// A graph-only bundle whose .meta.json is present but malformed does not fall back
// to default metadata: a typo'd enforcement_level would otherwise silently run at
// FULL. With no golden blobs it is still an authoring error, so it throws just as
// the golden case does.
TEST_F(TestBundleDiscoveryFixture, LoadGraphOnlyBundleMalformedMetadataIsError)
{
    auto dir = _tempDir / "op" / "graphbadmeta";
    createMinimalBundle(dir, "graphbadmeta"); // graph only, no .bin
    std::ofstream(dir / "graphbadmeta.meta.json")
        << R"({"format_version": 1, "enforcement_level": "buildible"})";

    EXPECT_THROW(loadIntegrationTestBundle(dir / "graphbadmeta.json"), BundleMetadataError);
}

// The invariant the whole change exists to protect: pulling golden data must never
// make a bundle quietly disappear. Each bundle is observed twice, before and after
// its output blob shows up. With no .meta.json the graph-only bundle loads and
// pulling the blob turns it red. With an unparseable .meta.json it is red in both
// states. Neither ever goes from a test to a silent skip.
TEST_F(TestBundleDiscoveryFixture, PullingGoldenDataNeverSilentlyDropsABundle)
{
    auto absentDir = _tempDir / "op" / "pullabsent";
    createLoadableBundle(absentDir, "pullabsent");
    std::filesystem::remove(absentDir / "pullabsent.meta.json");
    std::filesystem::remove(absentDir / "pullabsent.tensor5.bin"); // pre-`dvc pull` state

    auto badDir = _tempDir / "op" / "pullbad";
    createLoadableBundle(badDir, "pullbad");
    writeMetadataWithoutFormatVersion(badDir, "pullbad");
    std::filesystem::remove(badDir / "pullbad.tensor5.bin"); // pre-`dvc pull` state

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 2u);
    const auto* absentBundle = findByTest(discovered, "pullabsent");
    const auto* badBundle = findByTest(discovered, "pullbad");
    ASSERT_NE(absentBundle, nullptr);
    ASSERT_NE(badBundle, nullptr);

    EXPECT_TRUE(std::holds_alternative<detail::LoadedBundle>(
        detail::classifyBundle(*absentBundle, _sweeps)));
    EXPECT_TRUE(
        std::holds_alternative<detail::FailedLoad>(detail::classifyBundle(*badBundle, _sweeps)));

    writeGoldenOutputBlob(absentDir, "pullabsent"); // post-`dvc pull` state
    writeGoldenOutputBlob(badDir, "pullbad");

    auto afterPull = loadIntegrationTestBundle(absentDir / "pullabsent.json");
    ASSERT_TRUE(std::holds_alternative<LoadError>(afterPull));
    EXPECT_EQ(std::get<LoadError>(afterPull), LoadError::UNVALIDATABLE_GOLDEN_DATA);
    EXPECT_TRUE(
        std::holds_alternative<detail::FailedLoad>(detail::classifyBundle(*absentBundle, _sweeps)));
    EXPECT_TRUE(
        std::holds_alternative<detail::FailedLoad>(detail::classifyBundle(*badBundle, _sweeps)));
}

TEST_F(TestBundleDiscoveryFixture, LoadBundleMissingBinIsGraphOnly)
{
    auto dir = _tempDir / "op" / "nobin";
    createMinimalBundle(dir, "nobin");
    fixtures::writeMetadata(dir, "nobin"); // metadata present (optional here, but exercised)
    const auto jsonPath = dir / "nobin.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);

    EXPECT_FALSE(bundle.blobs.has_value());
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

    ASSERT_TRUE(bundle.blobs.has_value());
    ASSERT_EQ(bundle.outputTensorUids.size(), 1u);
    EXPECT_EQ(bundle.outputTensorUids.front(), 5);
    const auto tensors = bundle.loadTensors();
    EXPECT_EQ(tensors.at(0)->dims(), (std::vector<int64_t>{2, 3, 4, 5}));
    EXPECT_EQ(tensors.at(0)->strides(), (std::vector<int64_t>{60, 20, 5, 1}));
    EXPECT_EQ(tensors.at(5)->dims(), (std::vector<int64_t>{2, 3, 4, 5}));

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
    EXPECT_FALSE(bundle.blobs.has_value());
}

// Every case of a sweep shares one sweep.json, and a load pass must parse it once
// rather than once per case: re-parsing it per case cost 5.9 GB of JSON over the
// checked-in sweeps, and over half an hour before the first test ran on an MI300A
// host. Shown by breaking the manifest after the first case loads. The second case
// still loads through the pass's cache; a fresh load sees the broken file, which
// is the control that the file really is broken.
TEST_F(TestBundleDiscoveryFixture, SweepCasesInOneLoadPassParseTheManifestOnce)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(
        sweepDir,
        {{"case_a_fp32_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}},
         {"case_b_fp32_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 2u);

    SweepManifestCache sweeps;
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(
        loadIntegrationTestBundle(discovered[0], sweeps)));

    std::ofstream(sweepDir / "sweep.json", std::ios::trunc) << "{ not json";

    EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(
        loadIntegrationTestBundle(discovered[1], sweeps)));

    const auto fresh = loadIntegrationTestBundle(discovered[1]);
    ASSERT_TRUE(std::holds_alternative<LoadError>(fresh));
    EXPECT_EQ(std::get<LoadError>(fresh), LoadError::MALFORMED_JSON);
}

// The cache holds one sweep at a time. Moving to the next sweep must read that
// sweep's manifest, or its cases would be looked up in the previous sweep's and
// come back INVALID_SWEEP_CASE.
TEST_F(TestBundleDiscoveryFixture, SweepManifestCacheFollowsTheSweepBeingLoaded)
{
    createTemplateSweep(
        _tempDir / "quick" / "BatchnormFwdInference" / "First",
        {{"first_fp32_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});
    createTemplateSweep(
        _tempDir / "quick" / "BatchnormFwdInference" / "Second",
        {{"second_fp32_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 2u);

    SweepManifestCache sweeps;
    for(const auto& bundle : {discovered[0], discovered[1], discovered[0]})
    {
        EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(
            loadIntegrationTestBundle(bundle, sweeps)))
            << bundle.diagnosticPath();
    }
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

// A tensor_patches "set" that flips is_runtime_pass_by_value to true while the
// tensor still carries a baked value_type/value is rejected: buildGraphBuffer's
// validateRuntimePassByValueTensors check runs on the expanded+patched graph, so
// the contradiction can't slip through the template-sweep path. It throws
// RuntimePassByValueInvariantError rather than returning INVALID_GRAPH_SCHEMA,
// so this one contradiction can be routed to a hard failure (see
// classifyBundle() in BundleRegistration.hpp) instead of the quiet skip every
// other load failure gets.
TEST_F(TestBundleDiscoveryFixture, LoadTemplateSweepCaseWithBakedValueAndRuntimePassByValueIsError)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"baked_runtime_scale_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1}}});

    auto sweepJson = nlohmann::json::parse(std::ifstream(sweepDir / "sweep.json"));
    sweepJson["cases"][0]["tensor_patches"] = nlohmann::json::array(
        {{{"uid", 3},
          {"set",
           {{"is_runtime_pass_by_value", true}, {"value_type", "Float32Value"}, {"value", 2.0}}}}});
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    EXPECT_THROW(loadIntegrationTestBundle(discovered.front()),
                 detail::RuntimePassByValueInvariantError);
}

// A sweep case switches SDPA stats on by adding the statistics tensor and wiring it into the
// node. The built graph has the tensor, and its node names it in outputs and sets the flag.
TEST_F(TestBundleDiscoveryFixture, SweepCaseAddedTensorWiredByNodePatchesBuildsGraphWithItInTheNode)
{
    auto caseJson = makeSdpaCase("stats_on_case");
    caseJson["add_tensors"] = nlohmann::json::array({makeStatsTensor(4)});
    caseJson["node_patches"] = nlohmann::json::parse(R"([
        {"node": "SdpaFwd_0", "section": "outputs", "set": {"stats_tensor_uid": 4}},
        {"node": "SdpaFwd_0", "section": "attributes", "set": {"generate_stats": true}}])");

    auto result = loadSdpaCase(caseJson);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);

    const auto graph = bundle.graphWrapper();
    ASSERT_EQ(graph.nodeCount(), 1u);
    const auto* attributes = graph.getNode(0).attributes_as_SdpaAttributes();
    ASSERT_NE(attributes, nullptr);
    ASSERT_TRUE(attributes->stats_tensor_uid().has_value());
    EXPECT_EQ(*attributes->stats_tensor_uid(), 4);
    ASSERT_TRUE(attributes->generate_stats().has_value());
    EXPECT_TRUE(*attributes->generate_stats());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_EQ(tensorMap.count(4), 1u);
    EXPECT_EQ(tensorMap.at(4)->data_type(), hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    // The graph has a second output now, so the harness compares it.
    auto outputs = bundle.outputTensorUids;
    std::sort(outputs.begin(), outputs.end());
    EXPECT_EQ(outputs, (std::vector<int64_t>{2, 4}));
}

// "set" overwrites keys of the node's sections, and the change reaches the flatbuffer.
TEST_F(TestBundleDiscoveryFixture, SweepCaseNodePatchSetChangesTheBuiltGraph)
{
    auto caseJson = makeSdpaCase("scale_and_mask_case");
    caseJson["node_patches"] = nlohmann::json::parse(R"([
        {"node": "SdpaFwd_0", "section": "attributes",
         "set": {"attn_scale_value": 0.25, "causal_mask": true, "left_bound": 5}},
        {"node": 0, "section": "inputs", "set": {"k_tensor_uid": 3, "v_tensor_uid": 0}}])");

    auto result = loadSdpaCase(caseJson);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto graph = std::get<IntegrationTestBundle>(result).graphWrapper();
    const auto* attributes = graph.getNode(0).attributes_as_SdpaAttributes();
    ASSERT_NE(attributes, nullptr);
    ASSERT_TRUE(attributes->attn_scale_value().has_value());
    EXPECT_EQ(*attributes->attn_scale_value(), 0.25f);
    EXPECT_TRUE(attributes->causal_mask());
    ASSERT_TRUE(attributes->left_bound().has_value());
    EXPECT_EQ(*attributes->left_bound(), 5);
    EXPECT_EQ(attributes->k_tensor_uid(), 3);
    EXPECT_EQ(attributes->v_tensor_uid(), 0);
    // What the patch did not name keeps the template's value.
    EXPECT_FALSE(attributes->alibi_mask());
    EXPECT_EQ(attributes->q_tensor_uid(), 1);
}

// A case without the new fields expands and loads exactly as the template says.
TEST_F(TestBundleDiscoveryFixture, SweepCaseWithoutPatchFieldsLoadsAsTheTemplateExpands)
{
    const auto caseJson = makeSdpaCase("plain_case");
    EXPECT_EQ(patchedSdpaGraph(caseJson), expandSdpaCase(caseJson));

    auto result = loadSdpaCase(caseJson);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto graph = std::get<IntegrationTestBundle>(result).graphWrapper();
    EXPECT_EQ(graph.getTensorMap().size(), 4u);
    const auto* attributes = graph.getNode(0).attributes_as_SdpaAttributes();
    ASSERT_NE(attributes, nullptr);
    EXPECT_FALSE(attributes->stats_tensor_uid().has_value());
    EXPECT_FALSE(attributes->generate_stats().has_value());
}

// add_tensors runs before tensor_patches, so a tensor patch can edit a tensor the case added.
TEST_F(TestBundleDiscoveryFixture, SweepCaseTensorPatchAppliesToAnAddedTensor)
{
    auto caseJson = makeSdpaCase("patched_stats_case");
    caseJson["add_tensors"] = nlohmann::json::array({makeStatsTensor(4)});
    caseJson["tensor_patches"] = nlohmann::json::parse(
        R"([{"uid": 4, "set": {"alignment": 4}, "remove": ["ragged_offset_multiplier"]}])");

    const auto graph = patchedSdpaGraph(caseJson);
    const auto& stats = findTensor(graph, 4);
    EXPECT_EQ(stats.at("alignment").get<int64_t>(), 4);
    EXPECT_FALSE(stats.contains("ragged_offset_multiplier"));
    EXPECT_EQ(graph.at("tensors").size(), 5u);
}

// The same patch on a uid that no add_tensors entry provides is an error that names the case.
TEST_F(TestBundleDiscoveryFixture, SweepCaseTensorPatchOnAnUnaddedUidNamesTheCase)
{
    auto caseJson = makeSdpaCase("patch_before_add_case");
    caseJson["tensor_patches"] = nlohmann::json::parse(R"([{"uid": 4, "set": {"alignment": 4}}])");
    expectPatchError(caseJson,
                     {"sweep case 'patch_before_add_case'", "tensor_patch uid 4 not found"});
}

// A node is addressed by its name or by its index in nodes[], and only that node changes.
TEST_F(TestBundleDiscoveryFixture, SweepCaseNodePatchAddressesANodeByNameAndByIndex)
{
    auto graph = expandSdpaCase(makeSdpaCase("two_nodes_case"));
    graph["nodes"].push_back(graph.at("nodes").at(0));
    graph["nodes"][1]["name"] = "SdpaFwd_1";

    auto byName = makeSdpaCaseWithNodePatch(
        "two_nodes_case",
        R"({"node": "SdpaFwd_1", "section": "attributes", "set": {"causal_mask": true}})");
    auto byIndex = makeSdpaCaseWithNodePatch(
        "two_nodes_case", R"({"node": 1, "section": "attributes", "set": {"causal_mask": true}})");

    for(const auto& caseJson : {byName, byIndex})
    {
        auto patched = graph;
        detail::applyNodePatches(patched, caseJson);
        EXPECT_TRUE(patched.at("nodes").at(1).at("attributes").at("causal_mask").get<bool>());
        EXPECT_FALSE(patched.at("nodes").at(0).at("attributes").at("causal_mask").get<bool>());
    }
}

// "remove" erases keys after "set" has run.
TEST_F(TestBundleDiscoveryFixture, SweepCaseNodePatchRemoveErasesKeys)
{
    const auto caseJson
        = makeSdpaCaseWithNodePatch("remove_case",
                                    R"({"node": "SdpaFwd_0", "section": "attributes",
            "set": {"causal_mask": true}, "remove": ["dropout_probability", "max_seq_len_kv"]})");

    const auto graph = patchedSdpaGraph(caseJson);
    const auto& attributes = graph.at("nodes").at(0).at("attributes");
    EXPECT_FALSE(attributes.contains("dropout_probability"));
    EXPECT_FALSE(attributes.contains("max_seq_len_kv"));
    EXPECT_TRUE(attributes.at("causal_mask").get<bool>());
}

TEST_F(TestBundleDiscoveryFixture, SweepCaseNodePatchRejectsAnUnknownNodeAndABadIndex)
{
    const std::string tail = R"("section": "attributes", "set": {"causal_mask": true}})";

    expectPatchError(
        makeSdpaCaseWithNodePatch("unknown_node_case", R"({"node": "SdpaFwd_9", )" + tail),
        {"node_patches[0]", "node 'SdpaFwd_9' not found"});
    expectPatchError(makeSdpaCaseWithNodePatch("high_index_case", R"({"node": 1, )" + tail),
                     {"node_patches[0]", "node index 1 is out of range", "nodes[] has 1 entries"});
    expectPatchError(makeSdpaCaseWithNodePatch("negative_index_case", R"({"node": -1, )" + tail),
                     {"node_patches[0]", "node index -1 is out of range"});
}

// A name two nodes share does not pick one of them: the case must use the index.
TEST_F(TestBundleDiscoveryFixture, SweepCaseNodePatchRejectsAnAmbiguousNodeName)
{
    auto graph = expandSdpaCase(makeSdpaCase("ambiguous_case"));
    graph["nodes"].push_back(graph.at("nodes").at(0));

    const auto caseJson = makeSdpaCaseWithNodePatch(
        "ambiguous_case",
        R"({"node": "SdpaFwd_0", "section": "attributes", "set": {"causal_mask": true}})");
    expectPatchError(graph, caseJson, {"node_patches[0]", "node name 'SdpaFwd_0' is ambiguous"});

    // Neither node was edited.
    auto patched = graph;
    EXPECT_THROW(detail::applyNodePatches(patched, caseJson), std::runtime_error);
    EXPECT_EQ(patched, graph);
}

TEST_F(TestBundleDiscoveryFixture, SweepCaseNodePatchRejectsAnUnknownAndAMissingSection)
{
    // "attribute" is a typo of a section name.
    expectPatchError(
        makeSdpaCaseWithNodePatch(
            "unknown_section_case",
            R"({"node": "SdpaFwd_0", "section": "attribute", "set": {"causal_mask": true}})"),
        {"node_patches[0]", "unknown section \"attribute\""});
    expectPatchError(makeSdpaCaseWithNodePatch(
                         "numeric_section_case",
                         R"({"node": "SdpaFwd_0", "section": 7, "set": {"causal_mask": true}})"),
                     {"node_patches[0]", "unknown section 7"});

    // "parameters" is a real section name, but the forward node keeps its flags under
    // "attributes". The section is not created.
    const auto caseJson = makeSdpaCaseWithNodePatch(
        "missing_section_case",
        R"({"node": "SdpaFwd_0", "section": "parameters", "set": {"causal_mask": true}})");
    expectPatchError(caseJson, {"node_patches[0]", "node 'SdpaFwd_0' has no 'parameters' section"});
    auto graph = expandSdpaCase(caseJson);
    EXPECT_THROW(detail::applyNodePatches(graph, caseJson), std::runtime_error);
    EXPECT_FALSE(graph.at("nodes").at(0).contains("parameters"));
}

// A misspelled key must fail: the converters ignore keys they do not know, so inserting it
// would build the same graph and the case would pass without the feature it asked for.
TEST_F(TestBundleDiscoveryFixture, SweepCaseNodePatchSetRejectsAnUnknownKey)
{
    const auto caseJson = makeSdpaCaseWithNodePatch(
        "misspelled_key_case",
        R"({"node": "SdpaFwd_0", "section": "attributes", "set": {"generate_stat": true}})");
    expectPatchError(caseJson,
                     {"node_patches[0]",
                      "set key 'generate_stat' is not in the 'attributes' section of node "
                      "'SdpaFwd_0'"});

    auto graph = expandSdpaCase(caseJson);
    EXPECT_THROW(detail::applyNodePatches(graph, caseJson), std::runtime_error);
    EXPECT_FALSE(graph.at("nodes").at(0).at("attributes").contains("generate_stat"));

    // Inputs and outputs follow the same rule: a uid key the node does not list is rejected.
    expectPatchError(makeSdpaCaseWithNodePatch(
                         "bwd_only_key_case",
                         R"({"node": 0, "section": "inputs", "set": {"stats_tensor_uid": 4}})"),
                     {"node_patches[0]",
                      "set key 'stats_tensor_uid' is not in the 'inputs' section of node index 0"});
}

TEST_F(TestBundleDiscoveryFixture, SweepCaseNodePatchRemoveRejectsAMissingKey)
{
    expectPatchError(
        makeSdpaCaseWithNodePatch(
            "remove_missing_case",
            R"({"node": "SdpaFwd_0", "section": "attributes", "remove": ["no_such_key"]})"),
        {"node_patches[0]", "remove key \"no_such_key\" is not in the 'attributes' section"});
    // Removing the same key twice: the second finds it gone.
    expectPatchError(makeSdpaCaseWithNodePatch("remove_twice_case",
                                               R"({"node": "SdpaFwd_0", "section": "attributes",
                             "remove": ["left_bound", "left_bound"]})"),
                     {"node_patches[0]", "remove key \"left_bound\" is not in the 'attributes'"});
}

TEST_F(TestBundleDiscoveryFixture, SweepCaseAddTensorsRejectsADuplicateUid)
{
    // uid 0 is the template's K.
    auto inTemplate = makeSdpaCase("duplicate_template_uid_case");
    inTemplate["add_tensors"] = nlohmann::json::array({makeStatsTensor(0)});
    expectPatchError(inTemplate, {"add_tensors[0]", "uid 0 already exists in the graph"});

    auto twice = makeSdpaCase("duplicate_added_uid_case");
    twice["add_tensors"] = nlohmann::json::array({makeStatsTensor(4), makeStatsTensor(4)});
    expectPatchError(twice, {"add_tensors[1]", "uid 4 already exists in the graph"});
}

TEST_F(TestBundleDiscoveryFixture, SweepCaseAddTensorsRejectsMalformedEntries)
{
    auto notArray = makeSdpaCase("add_tensors_object_case");
    notArray["add_tensors"] = makeStatsTensor(4);
    expectPatchError(notArray, {"add_tensors must be an array"});

    const std::vector<std::pair<std::string, nlohmann::json>> badEntries{
        {"entry_not_object_case", nlohmann::json(4)},
        {"entry_without_uid_case", nlohmann::json{{"name", "stats"}, {"virtual", false}}},
        {"entry_string_uid_case", nlohmann::json{{"uid", "4"}, {"name", "stats"}}},
        {"entry_fractional_uid_case", nlohmann::json{{"uid", 4.5}, {"name", "stats"}}}};
    for(const auto& [id, entry] : badEntries)
    {
        auto caseJson = makeSdpaCase(id);
        caseJson["add_tensors"] = nlohmann::json::array({makeStatsTensor(4), entry});
        expectPatchError(caseJson, {"add_tensors[1] must be an object with an integer uid"});
    }
}

TEST_F(TestBundleDiscoveryFixture, SweepCaseNodePatchesRejectsMalformedEntries)
{
    auto notArray = makeSdpaCase("node_patches_object_case");
    notArray["node_patches"] = nlohmann::json::parse(R"({"node": 0})");
    expectPatchError(notArray, {"node_patches must be an array"});

    const std::vector<std::pair<std::string, std::string>> badEntries{
        {R"(3)", "node_patches[0] must be an object with node and section"},
        {R"({"section": "attributes", "set": {"causal_mask": true}})",
         "node_patches[0] must be an object with node and section"},
        {R"({"node": 0, "set": {"causal_mask": true}})",
         "node_patches[0] must be an object with node and section"},
        {R"({"node": 0, "section": "attributes", "sets": {"causal_mask": true}})",
         "node_patches[0]: unknown key 'sets'"},
        {R"({"node": 0, "section": "attributes"})", "node_patches[0] has neither set nor remove"},
        {R"({"node": 0, "section": "attributes", "set": [["causal_mask", true]]})",
         "node_patches[0]: set must be an object"},
        {R"({"node": 0, "section": "attributes", "remove": "causal_mask"})",
         "node_patches[0]: remove must be an array of key names"},
        {R"({"node": true, "section": "attributes", "set": {"causal_mask": true}})",
         "node_patches[0]: node must be a node name or an index"},
        {R"({"node": 0.5, "section": "attributes", "set": {"causal_mask": true}})",
         "node_patches[0]: node must be a node name or an index"}};
    int number = 0;
    for(const auto& [patch, message] : badEntries)
    {
        SCOPED_TRACE(patch);
        auto caseJson = makeSdpaCase("malformed_entry_case_" + std::to_string(number++));
        caseJson["node_patches"] = nlohmann::json::array({nlohmann::json::parse(patch)});
        expectPatchError(caseJson, {message});
    }
}

// At run time the detail is not reported: a bad case is INVALID_SWEEP_CASE, like any other
// authoring error in a sweep case.
TEST_F(TestBundleDiscoveryFixture, SweepCaseWithABadNodePatchFailsToLoadAsInvalidSweepCase)
{
    const auto caseJson = makeSdpaCaseWithNodePatch(
        "load_failure_case",
        R"({"node": "SdpaFwd_0", "section": "attributes", "set": {"generate_stat": true}})");

    const auto result = loadSdpaCase(caseJson);
    ASSERT_TRUE(std::holds_alternative<LoadError>(result));
    EXPECT_EQ(std::get<LoadError>(result), LoadError::INVALID_SWEEP_CASE);
}

// The same invariant must hold for a direct (non-sweep) bundle's graph.json,
// which never goes through applyTensorPatches. Regression test for the gap
// where validateRuntimePassByValueTensors lived only in applyTensorPatches and
// this path loaded a contradictory tensor without complaint.
TEST_F(TestBundleDiscoveryFixture, LoadDirectBundleWithBakedValueAndRuntimePassByValueIsError)
{
    auto dir = _tempDir / "op" / "bakedscale";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "bakedscale.json")
        << R"({"tensors": [{"uid": 0, "is_runtime_pass_by_value": true, )"
           R"("value_type": "Float32Value", "value": 2.0}]})";

    EXPECT_THROW(loadIntegrationTestBundle(dir / "bakedscale.json"),
                 detail::RuntimePassByValueInvariantError);
}

// Registration only records where a bundle's blobs are. A bad blob therefore fails the
// test that needs it, instead of dropping the bundle from the run when it is loaded.
TEST_F(TestBundleDiscoveryFixture, LoadBundleWrongSizeBinFailsOnlyWhenTheTensorsAreRead)
{
    auto dir = _tempDir / "op" / "badbin";
    createLoadableBundle(dir, "badbin");
    std::ofstream(dir / "badbin.tensor0.bin", std::ios::binary) << "too short";
    const auto jsonPath = dir / "badbin.json";

    auto result = loadIntegrationTestBundle(jsonPath);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    EXPECT_THROW(std::get<IntegrationTestBundle>(result).loadTensors(), std::exception);
}

TEST_F(TestBundleDiscoveryFixture, LoadRaggedBundleLoadsInputsAndGoldenOutputs)
{
    auto dir = _tempDir / "op" / "ragged";
    fixtures::writeRaggedBundleFiles(
        dir, "ragged", {0, 40, 60}, std::vector<float>(fixtures::K_RAGGED_OUTPUT_ELEMS, 0.0f));

    auto result = loadIntegrationTestBundle(dir / "ragged.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    auto& bundle = std::get<IntegrationTestBundle>(result);

    EXPECT_TRUE(bundle.hasGoldenOutputs);
    const auto tensors = bundle.loadTensors();
    ASSERT_EQ(tensors.size(), 7u);
    const auto* offsetTensor = tensors.at(6).get();
    for(const int64_t uid : {0, 5})
    {
        const auto* ragged
            = dynamic_cast<const hipdnn_data_sdk::utilities::RaggedTensorBase<float>*>(
                tensors.at(uid).get());
        ASSERT_NE(ragged, nullptr) << "uid " << uid;
        EXPECT_EQ(ragged->raggedOffset(), offsetTensor) << "uid " << uid;
        EXPECT_EQ(ragged->raggedIterationInfo()->rowOffsets, (std::vector<int64_t>{0, 40, 60}))
            << "uid " << uid;
    }
}

TEST_F(TestBundleDiscoveryFixture, LoadRaggedBundleWithInvalidOffsetsFailsOnlyWhenTheTensorsAreRead)
{
    auto dir = _tempDir / "op" / "badragged";
    fixtures::writeRaggedBundleFiles(
        dir, "badragged", {1, 40, 60}, std::vector<float>(fixtures::K_RAGGED_OUTPUT_ELEMS, 0.0f));

    auto result = loadIntegrationTestBundle(dir / "badragged.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    EXPECT_THROW(std::get<IntegrationTestBundle>(result).loadTensors(), std::exception);
}

TEST_F(TestBundleDiscoveryFixture,
       LoadRaggedBundleWithMissingOffsetTensorFailsOnlyWhenTheTensorsAreRead)
{
    auto dir = _tempDir / "op" / "nooffset";
    const int64_t undeclaredUid = fixtures::K_RAGGED_OFFSET_UID + 1;
    fixtures::writeRaggedBundleFiles(dir,
                                     "nooffset",
                                     {0, 40, 60},
                                     std::vector<float>(fixtures::K_RAGGED_OUTPUT_ELEMS, 0.0f),
                                     undeclaredUid);

    auto result = loadIntegrationTestBundle(dir / "nooffset.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    EXPECT_THROW(std::get<IntegrationTestBundle>(result).loadTensors(), std::exception);
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
// UNVALIDATABLE_GOLDEN_DATA: metadata is what validates the golden data, and the
// blobs are right there.
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
    EXPECT_EQ(std::get<LoadError>(result), LoadError::UNVALIDATABLE_GOLDEN_DATA);
}

// Same rule for sweeps as for direct bundles: a metadata block that is present but
// rejected by the schema throws BundleMetadataError rather than loading.
TEST_F(TestBundleDiscoveryFixture, LoadTemplateSweepCaseGoldenWithUnparseableMetadataIsError)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"golden_bad_meta_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          true, // includeGolden
                          true, // goldenHasPath
                          true}}); // includeMetadata

    // Strip the required format_version from the case's metadata block, leaving a
    // well-formed JSON object the metadata reader still refuses.
    nlohmann::json sweepJson;
    {
        std::ifstream in(sweepDir / "sweep.json");
        sweepJson = nlohmann::json::parse(in);
    }
    sweepJson["cases"][0]["metadata"].erase("format_version");
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    EXPECT_THROW(loadIntegrationTestBundle(discovered.front()), BundleMetadataError);
}

// A graph-only sweep case whose metadata block is rejected by the schema throws
// too: no golden blobs does not make a malformed block any less of an error.
TEST_F(TestBundleDiscoveryFixture, LoadTemplateSweepCaseGraphOnlyWithMalformedMetadataIsError)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"graph_only_bad_meta_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          false, // includeGolden
                          false, // goldenHasPath
                          true}}); // includeMetadata

    nlohmann::json sweepJson;
    {
        std::ifstream in(sweepDir / "sweep.json");
        sweepJson = nlohmann::json::parse(in);
    }
    sweepJson["cases"][0]["metadata"]["enforcement_level"] = "buildible";
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    EXPECT_THROW(loadIntegrationTestBundle(discovered.front()), BundleMetadataError);

    // And classification turns it red with the parser's detail, not a quiet skip.
    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome));
    EXPECT_NE(std::get<detail::FailedLoad>(outcome).message.find("buildible"), std::string::npos);
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

// classifyBundle() is the testable decision step behind registerBundleTests():
// it makes the same load-and-classify call that loop makes per discovered
// bundle, without touching ::testing::RegisterTest (which can only run before
// RUN_ALL_TESTS(), not from inside a running test body). These tests cover the
// routing itself — a good bundle becomes a LoadedBundle, a bad one becomes a
// FailedLoad — since nothing previously exercised that routing directly.
TEST_F(TestBundleDiscoveryFixture, ClassifyBundleReturnsLoadedBundleForGoodBundle)
{
    auto dir = _tempDir / "op" / "goodbundle";
    createLoadableBundle(dir, "goodbundle");

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::LoadedBundle>(outcome));
    auto& loaded = std::get<detail::LoadedBundle>(outcome);
    EXPECT_EQ(loaded.suiteName, discovered.front().suiteName);
    EXPECT_EQ(loaded.testName, discovered.front().testName);
    ASSERT_NE(loaded.bundle, nullptr);

    EXPECT_EQ(loaded.claimLocator.sidecarPath,
              supportJsonPath(discovered.front().diagnosticPath()));
    EXPECT_TRUE(loaded.claimLocator.caseId.empty());
    EXPECT_FALSE(loaded.claimLocator.isSweep());
}

TEST_F(TestBundleDiscoveryFixture, ClassifyBundleSetsLocatorForSweepCase)
{
    createTemplateSweep(
        _tempDir / "quick" / "BatchnormFwdInference" / "Inference",
        {{"fp32_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::LoadedBundle>(outcome));
    auto& loaded = std::get<detail::LoadedBundle>(outcome);

    EXPECT_EQ(loaded.claimLocator.sidecarPath,
              discovered.front().jsonPath.parent_path() / "support.json");
    EXPECT_EQ(loaded.claimLocator.caseId, "fp32_nchw");
    EXPECT_TRUE(loaded.claimLocator.isSweep());
}

// selectBundlesToLoad() is the registration-time filter step. Its counters are the
// denominators the support-claim summary divides by, so a drift here reattributes every
// gap line to the wrong cause without failing anything else. Three flat bundles:
// case_a is selected by the filter and has a claim, case_b is excluded and has a
// claim, case_c is excluded and has none.
TEST_F(TestBundleDiscoveryFixture, SelectBundlesToLoadCountsExcludedClaimsWhenObserving)
{
    for(const auto* suite : {"case_a", "case_b", "case_c"})
    {
        createMinimalBundle(_tempDir / suite, "graph");
    }
    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 3u);
    for(const auto& bundle : discovered)
    {
        if(bundle.suiteName != "case_c")
        {
            std::ofstream(supportJsonPath(bundle.diagnosticPath())) << "{}";
        }
    }

    BundleRegistrationStats stats;
    SupportClaimCoverage coverage;
    const auto selected = detail::selectBundlesToLoad(
        discovered, "case_a.*", /*writing=*/false, /*observing=*/true, stats, coverage);

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected.front().suiteName, "case_a");
    EXPECT_EQ(stats.discovered, 3u);
    EXPECT_EQ(stats.excludedByFilter, 2u);
    // Only the two excluded bundles are counted here: the selected one is counted as it
    // loads.
    EXPECT_EQ(coverage.graphsFound, 2u);
    EXPECT_EQ(coverage.graphsWithClaims, 1u);
}

// Without a named engine nothing is checkable, so the summary must not be handed
// denominators for claims no run was going to check.
TEST_F(TestBundleDiscoveryFixture, SelectBundlesToLoadLeavesCoverageAloneWhenNotObserving)
{
    createMinimalBundle(_tempDir / "case_a", "graph");
    createMinimalBundle(_tempDir / "case_b", "graph");
    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 2u);
    for(const auto& bundle : discovered)
    {
        std::ofstream(supportJsonPath(bundle.diagnosticPath())) << "{}";
    }

    BundleRegistrationStats stats;
    SupportClaimCoverage coverage;
    const auto selected = detail::selectBundlesToLoad(
        discovered, "case_a.*", /*writing=*/false, /*observing=*/false, stats, coverage);

    EXPECT_EQ(selected.size(), 1u);
    EXPECT_EQ(stats.excludedByFilter, 1u);
    EXPECT_EQ(coverage.graphsFound, 0u);
    EXPECT_EQ(coverage.graphsWithClaims, 0u);
}

// --write-support-claims needs every graph loaded: graphsFound is the denominator for
// the graphs the observer did not see, so a filter that dropped any would shrink it.
TEST_F(TestBundleDiscoveryFixture, SelectBundlesToLoadKeepsEveryBundleWhenWriting)
{
    createMinimalBundle(_tempDir / "case_a", "graph");
    createMinimalBundle(_tempDir / "case_b", "graph");
    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 2u);

    BundleRegistrationStats stats;
    SupportClaimCoverage coverage;
    const auto selected = detail::selectBundlesToLoad(
        discovered, "case_a.*", /*writing=*/true, /*observing=*/false, stats, coverage);

    EXPECT_EQ(selected.size(), 2u);
    EXPECT_EQ(stats.discovered, 2u);
    EXPECT_EQ(stats.excludedByFilter, 0u);
    EXPECT_EQ(coverage.graphsFound, 0u);
}

// Reuses the baked-value-plus-runtime-pass-by-value corruption from
// LoadTemplateSweepCaseWithBakedValueAndRuntimePassByValueIsError: that test
// confirms loadIntegrationTestBundle() throws RuntimePassByValueInvariantError
// for this case; this one confirms classifyBundle() catches that specific
// exception and turns it into a FailedLoad carrying the diagnostic path and
// reason, which is what lets registerBundleTests() surface it as a failing
// test instead of a log line.
TEST_F(TestBundleDiscoveryFixture, ClassifyBundleReturnsFailedLoadForRuntimePassByValueInvariant)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"baked_runtime_scale_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1}}});

    auto sweepJson = nlohmann::json::parse(std::ifstream(sweepDir / "sweep.json"));
    sweepJson["cases"][0]["tensor_patches"] = nlohmann::json::array(
        {{{"uid", 3},
          {"set",
           {{"is_runtime_pass_by_value", true}, {"value_type", "Float32Value"}, {"value", 2.0}}}}});
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome));
    auto& failed = std::get<detail::FailedLoad>(outcome);
    EXPECT_EQ(failed.suiteName, discovered.front().suiteName);
    EXPECT_EQ(failed.testName, discovered.front().testName);
    EXPECT_NE(failed.message.find(discovered.front().diagnosticPath().string()), std::string::npos);
    EXPECT_NE(failed.message.find("is_runtime_pass_by_value=true"), std::string::npos);
}

// Only the RuntimePassByValueInvariantError contradiction gets a FailedLoad.
// An ordinary invalid-sweep-case failure (here: a case referencing a golden
// path that was never created) must keep the original behavior — logged and
// classified as SkippedLoad, no failing test registered — so this fix doesn't
// widen every pre-existing load failure into a hard suite failure.
TEST_F(TestBundleDiscoveryFixture, ClassifyBundleReturnsSkippedLoadForOrdinaryInvalidSweepCase)
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

    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::SkippedLoad>(outcome));
    auto& skipped = std::get<detail::SkippedLoad>(outcome);
    EXPECT_NE(skipped.message.find(discovered.front().diagnosticPath().string()),
              std::string::npos);
    EXPECT_NE(skipped.message.find(toString(LoadError::INVALID_SWEEP_CASE)), std::string::npos);
}

// The counterpart to the SkippedLoad case above: golden blobs with no metadata
// is a failure that must turn the suite red. Pinning it here stops a future
// refactor from folding it back into the quiet path, which is precisely the
// regression that let 35 SdpaFwd bundles disappear.
TEST_F(TestBundleDiscoveryFixture, ClassifyBundleReturnsFailedLoadForUnvalidatableGoldenData)
{
    auto dir = _tempDir / "op" / "goldennometa";
    createLoadableBundle(dir, "goldennometa");
    std::filesystem::remove(dir / "goldennometa.meta.json");

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome));
    auto& failed = std::get<detail::FailedLoad>(outcome);
    EXPECT_EQ(failed.suiteName, discovered.front().suiteName);
    EXPECT_EQ(failed.testName, discovered.front().testName);
    EXPECT_NE(failed.message.find(discovered.front().diagnosticPath().string()), std::string::npos);
    EXPECT_NE(failed.message.find(toString(LoadError::UNVALIDATABLE_GOLDEN_DATA)),
              std::string::npos);
}

// Malformed metadata turns the suite red whether or not golden blobs are present,
// and the failure message carries the parser's detail so the author can see what
// to fix (HIPDNN_LOG_LEVEL is off by default, so a WARN log would be invisible).
TEST_F(TestBundleDiscoveryFixture, ClassifyBundleReturnsFailedLoadWithDetailForMalformedMetadata)
{
    auto dir = _tempDir / "op" / "graphbadmeta";
    createMinimalBundle(dir, "graphbadmeta"); // graph only, no .bin
    std::ofstream(dir / "graphbadmeta.meta.json")
        << R"({"format_version": 1, "enforcement_level": "buildible"})";

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome));
    auto& failed = std::get<detail::FailedLoad>(outcome);
    EXPECT_EQ(failed.suiteName, discovered.front().suiteName);
    EXPECT_EQ(failed.testName, discovered.front().testName);
    EXPECT_NE(failed.message.find(discovered.front().diagnosticPath().string()), std::string::npos);
    EXPECT_NE(failed.message.find("enforcement_level \"buildible\""), std::string::npos);
}

// Direct regression test for the original review repro: corrupting one sweep
// case must not affect classification of an unrelated, valid bundle
// discovered alongside it.
TEST_F(TestBundleDiscoveryFixture, ClassifyBundleIsolatesFailureAmongMultipleDiscoveredBundles)
{
    auto goodDir = _tempDir / "op" / "goodbundle";
    createLoadableBundle(goodDir, "goodbundle");

    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"baked_runtime_scale_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1}}});
    auto sweepJson = nlohmann::json::parse(std::ifstream(sweepDir / "sweep.json"));
    sweepJson["cases"][0]["tensor_patches"] = nlohmann::json::array(
        {{{"uid", 3},
          {"set",
           {{"is_runtime_pass_by_value", true}, {"value_type", "Float32Value"}, {"value", 2.0}}}}});
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 2u);

    const auto* goodBundle = findByTest(discovered, "goodbundle");
    const auto* badBundle = findByTest(discovered, "baked_runtime_scale_fp32_nchw");
    ASSERT_NE(goodBundle, nullptr);
    ASSERT_NE(badBundle, nullptr);

    EXPECT_TRUE(
        std::holds_alternative<detail::LoadedBundle>(detail::classifyBundle(*goodBundle, _sweeps)));
    EXPECT_TRUE(
        std::holds_alternative<detail::FailedLoad>(detail::classifyBundle(*badBundle, _sweeps)));
}

// Closes the loop between "classifyBundle() decided this bundle failed" and
// "the failing test GTest actually runs really fails": constructs the
// synthetic test body directly and confirms it records exactly the stored
// message as a non-fatal failure, the same way GTest would run it once
// registerSyntheticBundleTest() has registered it.
TEST_F(TestBundleDiscoveryFixture, FailedBundleLoadRecordsFailureWithMessage)
{
    detail::SyntheticBundleTest test(detail::SyntheticOutcome::FAIL, "boom");
    EXPECT_NONFATAL_FAILURE(test.TestBody(), "boom");
}

// The other outcome stands in for a declared coverage gap, such as a bundle too
// costly for the one lane that runs. It must skip naming the gap: a failure would
// turn every device-less run red for a deliberate trade, and a pass would claim a
// validation that never happened.
TEST_F(TestBundleDiscoveryFixture, SkippingSyntheticBodySkipsWithItsMessage)
{
    detail::SyntheticBundleTest test(detail::SyntheticOutcome::SKIP, "nothing validated this");

    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        test.TestBody();
    }

    ASSERT_EQ(results.size(), 1);
    EXPECT_TRUE(results.GetTestPartResult(0).skipped());
    EXPECT_NE(std::string(results.GetTestPartResult(0).message()).find("nothing validated this"),
              std::string::npos);
}

// NOLINTEND(readability-identifier-naming)
