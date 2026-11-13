// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "BundleFixtureFiles.hpp"
#include "SdpaFwdGraphTestUtils.hpp"
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

// Ops without an "outputs" sub-object name their output with a flat key. The flat
// inputs beside it, such as resample backward's dy and index, are not outputs.
TEST(TestExtractOutputUidsFromJson, ReadsFlatOutputKeys)
{
    const auto graphJson = nlohmann::json::parse(R"({"nodes": [
        {"type": "ReductionAttributes", "in_tensor_uid": 0, "out_tensor_uid": 1},
        {"type": "ResampleFwdAttributes", "x_tensor_uid": 2, "y_tensor_uid": 3},
        {"type": "ResampleBwdAttributes", "dy_tensor_uid": 4, "index_tensor_uid": 5,
         "dx_tensor_uid": 6}
    ]})");

    EXPECT_EQ(detail::extractOutputUidsFromJson(graphJson), (std::vector<int64_t>{1, 3, 6}));
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
        EXPECT_EQ(ragged->raggedIterationInfo()->origins, (std::vector<int64_t>{0, 40}))
            << "uid " << uid;
        EXPECT_EQ(ragged->raggedIterationInfo()->lengths, (std::vector<int64_t>{2, 1}))
            << "uid " << uid;
    }
}

// Every ragged tensor of a ragged SDPA bundle reads from a blob sized to its elementSpace
// and takes its per-batch origins and lengths from its offsets and, when the node has
// them, the seq_len tensors: Stats stored [H,T] (L7a, L7c, L7d), a head-major primary
// (L6), a shifted origin (L8), non-monotonic origins (L9) and a trailing gap (L14).
TEST_F(TestBundleDiscoveryFixture, LoadRaggedSdpaBundlesReadsBlobsSizedToElementSpace)
{
    for(const auto& layout : fixtures::raggedSdpaLoadableCases())
    {
        SCOPED_TRACE(layout.name);
        const auto dir = _tempDir / "op" / layout.name;
        fixtures::writeRaggedSdpaCase(dir, layout);

        auto result = loadIntegrationTestBundle(dir / (layout.name + ".json"));
        ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
        const auto tensors = std::get<IntegrationTestBundle>(result).loadTensors();

        for(const int64_t uid : {fixtures::K_SDPA_Q_UID,
                                 fixtures::K_SDPA_K_UID,
                                 fixtures::K_SDPA_V_UID,
                                 fixtures::K_SDPA_O_UID,
                                 fixtures::K_SDPA_STATS_UID})
        {
            SCOPED_TRACE("uid " + std::to_string(uid));
            const auto* ragged
                = dynamic_cast<const hipdnn_data_sdk::utilities::RaggedTensorBase<float>*>(
                    tensors.at(uid).get());
            ASSERT_NE(ragged, nullptr);
            EXPECT_EQ(ragged->elementSpace(),
                      uid == fixtures::K_SDPA_STATS_UID ? layout.statsSpace : layout.qkvoSpace);
            ASSERT_TRUE(ragged->raggedIterationInfo().has_value());
            EXPECT_EQ(ragged->raggedIterationInfo()->lengths, layout.lengths);
        }

        EXPECT_EQ(tensors.count(fixtures::K_SDPA_SEQ_LEN_Q_UID), layout.seqLen.empty() ? 0u : 1u);
        EXPECT_EQ(tensors.count(fixtures::K_SDPA_SEQ_LEN_KV_UID), layout.seqLen.empty() ? 0u : 1u);
    }
}

// A trailing gap leaves offset[B] past the last row, so a blob sized to offset[B] is larger
// than the elementSpace: it is rejected, naming the size rule, rather than read.
TEST_F(TestBundleDiscoveryFixture, LoadRaggedSdpaBundleWithBlobSizedToOffsetBIsRejected)
{
    auto layout = fixtures::raggedSdpaTrailingGapCase();
    ASSERT_EQ(layout.qkvo.offsets.back(), 12);
    ASSERT_EQ(layout.qkvoSpace, 10u);
    layout.qkvo.blobElems = static_cast<size_t>(layout.qkvo.offsets.back());

    const auto dir = _tempDir / "op" / "offsetb";
    fixtures::writeRaggedSdpaCase(dir, layout);
    auto result = loadIntegrationTestBundle(dir / (layout.name + ".json"));
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));

    try
    {
        std::get<IntegrationTestBundle>(result).loadTensors();
        FAIL() << "a blob sized to offset[B] must not load";
    }
    catch(const std::exception& e)
    {
        EXPECT_NE(std::string(e.what()).find("elementSpace"), std::string::npos) << e.what();
    }
}

// Without seq_len, a block longer than S_max is capped, so offsets written with the multiplier
// applied twice still build a layout. The blob, sized to the element space of the intended
// layout, no longer matches the element space of the one those offsets describe: rejected.
TEST_F(TestBundleDiscoveryFixture, LoadRaggedSdpaBundleWithDoubledOffsetsIsRejected)
{
    auto layout = fixtures::raggedSdpaHeadTokenStatsCase();
    ASSERT_TRUE(layout.seqLen.empty());
    ASSERT_EQ(layout.qkvo.offsets, (std::vector<int32_t>{0, 16, 40}));
    for(auto& offset : layout.qkvo.offsets)
    {
        offset *= 2;
    }

    const auto dir = _tempDir / "op" / "doubled";
    fixtures::writeRaggedSdpaCase(dir, layout);
    auto result = loadIntegrationTestBundle(dir / (layout.name + ".json"));
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));

    try
    {
        std::get<IntegrationTestBundle>(result).loadTensors();
        FAIL() << "offsets written with a doubled multiplier must not load";
    }
    catch(const std::exception& e)
    {
        EXPECT_NE(std::string(e.what()).find("elementSpace"), std::string::npos) << e.what();
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

// ---------------------------------------------------------------------------
// Metadata flags that contradict the golden data a bundle declares.
//
// "Declared" is read from the checked-in pointer (`{Name}.tensors.dvc`, or a sweep
// case's golden.path), not from the blobs: on a tree where `dvc pull` has not run the
// rules must give the same answer.
// ---------------------------------------------------------------------------

namespace
{

void writeFlagsMeta(const std::filesystem::path& dir,
                    const std::string& name,
                    const nlohmann::json& flags)
{
    nlohmann::json meta = {{"format_version", 1}};
    meta.update(flags);
    std::ofstream(dir / (name + ".meta.json")) << meta.dump(2);
}

// Merges `flags` into the first sweep case's metadata block.
void setFirstCaseFlags(const std::filesystem::path& sweepDir, const nlohmann::json& flags)
{
    nlohmann::json sweepJson;
    {
        std::ifstream in(sweepDir / "sweep.json");
        sweepJson = nlohmann::json::parse(in);
    }
    sweepJson["cases"][0]["metadata"].update(flags);
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);
}

const nlohmann::json BOTH_REFERENCES_OFF = {{"no_cpu_reference", true}, {"no_gpu_reference", true}};

} // namespace

TEST_F(TestBundleDiscoveryFixture, LoadBundleDisablingBothReferencesWithoutGoldenIsError)
{
    const auto dir = _tempDir / "op" / "bothoff";
    createMinimalBundle(dir, "bothoff");
    writeFlagsMeta(dir, "bothoff", BOTH_REFERENCES_OFF);

    EXPECT_THROW(loadIntegrationTestBundle(dir / "bothoff.json"), BundleMetadataError);

    // Red, naming the rule -- not a silent skip.
    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);
    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome));
    EXPECT_NE(std::get<detail::FailedLoad>(outcome).message.find("disables both references"),
              std::string::npos);
}

// The pointer is what declares the data. The blobs it names are absent here, as on a
// machine that has not pulled.
TEST_F(TestBundleDiscoveryFixture, LoadBundleDisablingBothReferencesWithGoldenPointerLoads)
{
    const auto dir = _tempDir / "op" / "bothoffdvc";
    createMinimalBundle(dir, "bothoffdvc");
    writeFlagsMeta(dir, "bothoffdvc", BOTH_REFERENCES_OFF);
    std::ofstream(dir / "bothoffdvc.tensors.dvc") << "outs:\n";

    auto result = loadIntegrationTestBundle(dir / "bothoffdvc.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    const auto& bundle = std::get<IntegrationTestBundle>(result);
    EXPECT_TRUE(bundle.metadata.noCpuReference);
    EXPECT_TRUE(bundle.metadata.noGpuReference);
    EXPECT_FALSE(bundle.hasGoldenOutputs);
}

TEST_F(TestBundleDiscoveryFixture, LoadBundleDisablingBothReferencesWithGoldenBlobsLoads)
{
    const auto dir = _tempDir / "op" / "bothoffbin";
    createLoadableBundle(dir, "bothoffbin");
    writeFlagsMeta(dir, "bothoffbin", BOTH_REFERENCES_OFF);

    auto result = loadIntegrationTestBundle(dir / "bothoffbin.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    EXPECT_TRUE(std::get<IntegrationTestBundle>(result).hasGoldenOutputs);
}

// One reference left is enough to verify the bundle, with or without golden data.
TEST_F(TestBundleDiscoveryFixture, LoadBundleDisablingOneReferenceWithoutGoldenLoads)
{
    for(const char* flag : {"no_cpu_reference", "no_gpu_reference"})
    {
        SCOPED_TRACE(flag);
        const auto dir = _tempDir / "op" / flag;
        createMinimalBundle(dir, "oneoff");
        writeFlagsMeta(dir, "oneoff", {{flag, true}});

        EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(
            loadIntegrationTestBundle(dir / "oneoff.json")));
    }
}

TEST_F(TestBundleDiscoveryFixture, LoadExpectedRejectionBundleWithoutGoldenLoads)
{
    const auto dir = _tempDir / "op" / "rejected";
    createMinimalBundle(dir, "rejected");
    writeFlagsMeta(dir, "rejected", {{"expected_rejection", true}});

    auto result = loadIntegrationTestBundle(dir / "rejected.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    EXPECT_TRUE(std::get<IntegrationTestBundle>(result).metadata.expectedRejection);
}

// The reference flags are ignored on an expected rejection, so setting both without any
// golden data is not the both-off error (which would make the combination unsatisfiable:
// the golden data it demands is the golden data an expected rejection must not have).
TEST_F(TestBundleDiscoveryFixture, LoadExpectedRejectionBundleIgnoresBothReferenceFlags)
{
    const auto dir = _tempDir / "op" / "rejectedbothoff";
    createMinimalBundle(dir, "rejectedbothoff");
    writeFlagsMeta(
        dir,
        "rejectedbothoff",
        {{"expected_rejection", true}, {"no_cpu_reference", true}, {"no_gpu_reference", true}});

    EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(
        loadIntegrationTestBundle(dir / "rejectedbothoff.json")));
}

// An expected rejection is never executed, so golden data beside it can only be a
// mistake -- and would drag the bundle into the golden-data binary.
TEST_F(TestBundleDiscoveryFixture, LoadExpectedRejectionBundleWithGoldenIsError)
{
    const auto pointerDir = _tempDir / "op" / "rejecteddvc";
    createMinimalBundle(pointerDir, "rejecteddvc");
    writeFlagsMeta(pointerDir, "rejecteddvc", {{"expected_rejection", true}});
    std::ofstream(pointerDir / "rejecteddvc.tensors.dvc") << "outs:\n";
    EXPECT_THROW(loadIntegrationTestBundle(pointerDir / "rejecteddvc.json"), BundleMetadataError);

    const auto blobDir = _tempDir / "op" / "rejectedbin";
    createLoadableBundle(blobDir, "rejectedbin");
    writeFlagsMeta(blobDir, "rejectedbin", {{"expected_rejection", true}});
    EXPECT_THROW(loadIntegrationTestBundle(blobDir / "rejectedbin.json"), BundleMetadataError);
}

TEST_F(TestBundleDiscoveryFixture, LoadSweepCaseDisablingBothReferencesWithoutGoldenIsError)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"bothoff_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          /*includeGolden=*/false}});
    setFirstCaseFlags(sweepDir, BOTH_REFERENCES_OFF);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);
    EXPECT_THROW(loadIntegrationTestBundle(discovered.front()), BundleMetadataError);
}

TEST_F(TestBundleDiscoveryFixture, LoadSweepCaseDisablingBothReferencesWithGoldenPathLoads)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"bothoff_golden_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1}}});
    setFirstCaseFlags(sweepDir, BOTH_REFERENCES_OFF);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(
        loadIntegrationTestBundle(discovered.front())));
}

TEST_F(TestBundleDiscoveryFixture, LoadSweepCaseExpectedRejectionWithGoldenIsError)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"rejected_golden_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1}}});
    setFirstCaseFlags(sweepDir, {{"expected_rejection", true}});

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);
    EXPECT_THROW(loadIntegrationTestBundle(discovered.front()), BundleMetadataError);
}

// ---------------------------------------------------------------------------
// An expected-rejection bundle has to reach an engine to be rejected by one. One that
// cannot even load must be red, not the quiet skip an unloadable bundle gets.
// ---------------------------------------------------------------------------

TEST_F(TestBundleDiscoveryFixture, ClassifyBundleFailsAnExpectedRejectionBundleThatCannotLoad)
{
    const auto flaggedDir = _tempDir / "op" / "flagged";
    createMinimalBundle(flaggedDir, "flagged");
    std::ofstream(flaggedDir / "flagged.json") << "{not json";
    writeFlagsMeta(flaggedDir, "flagged", {{"expected_rejection", true}});

    // Control: the same broken graph without the flag keeps the quiet skip.
    const auto plainDir = _tempDir / "op" / "plain";
    createMinimalBundle(plainDir, "plain");
    std::ofstream(plainDir / "plain.json") << "{not json";

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 2u);
    const auto* flagged = findByTest(discovered, "flagged");
    const auto* plain = findByTest(discovered, "plain");
    ASSERT_NE(flagged, nullptr);
    ASSERT_NE(plain, nullptr);

    auto flaggedOutcome = detail::classifyBundle(*flagged, _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(flaggedOutcome));
    EXPECT_NE(std::get<detail::FailedLoad>(flaggedOutcome).message.find("expected_rejection"),
              std::string::npos);
    EXPECT_TRUE(
        std::holds_alternative<detail::SkippedLoad>(detail::classifyBundle(*plain, _sweeps)));
}

TEST_F(TestBundleDiscoveryFixture, ClassifyBundleFailsAnExpectedRejectionSweepCaseThatCannotLoad)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"flagged_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          /*includeGolden=*/false}});
    setFirstCaseFlags(sweepDir, {{"expected_rejection", true}});

    // A case whose values no longer fill the template's placeholders is an
    // INVALID_SWEEP_CASE, which is a skip unless the case asked to be rejected.
    nlohmann::json sweepJson;
    {
        std::ifstream in(sweepDir / "sweep.json");
        sweepJson = nlohmann::json::parse(in);
    }
    sweepJson["cases"][0].erase("values");
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);
    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome));
    EXPECT_NE(std::get<detail::FailedLoad>(outcome).message.find("expected_rejection"),
              std::string::npos);
}

// Registration seeds graphsWithClaims from the same predicate the harness's
// shouldObserveClaims() applies. A sweep's cases share one support.json, so the file
// alone cannot exclude a flagged case: only the flag can, and a count the harness will
// never satisfy would make the summary blame the harness for the gap.
TEST_F(TestBundleDiscoveryFixture, RegistrationDoesNotCountAnExpectedRejectionCaseAsClaimed)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"claimed_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          /*includeGolden=*/false},
                         {"rejected_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          /*includeGolden=*/false}});

    nlohmann::json sweepJson;
    {
        std::ifstream in(sweepDir / "sweep.json");
        sweepJson = nlohmann::json::parse(in);
    }
    sweepJson["cases"][1]["metadata"]["expected_rejection"] = true;
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);
    std::ofstream(sweepDir / "support.json")
        << R"({"version": 1, "claims": {"E": [{"cases": ["claimed_fp32_nchw"],
                "support": {"gfx942": ["linux"]}}]}})";

    const detail::DiscoveredBundleSet set{_tempDir, discoverBundles(_tempDir)};
    ASSERT_EQ(set.bundles.size(), 2u);

    const auto before = supportClaimCoverage();
    const auto loadedBundles
        = detail::loadDiscoveredBundles(set, /*countFound=*/true, /*countClaims=*/true);
    ASSERT_TRUE(loadedBundles.has_value());
    EXPECT_EQ(loadedBundles->size(), 2u);

    EXPECT_EQ(supportClaimCoverage().graphsFound - before.graphsFound, 2u);
    EXPECT_EQ(supportClaimCoverage().graphsWithClaims - before.graphsWithClaims, 1u);
}

// An unknown fill kind is an authoring error, not a random fill: the bundle registers
// a failing test that names the input and the kind.
TEST_F(TestBundleDiscoveryFixture, ClassifyBundleReturnsFailedLoadForUnknownFillKind)
{
    auto dir = _tempDir / "op" / "unknownkind";
    createMinimalBundle(dir, "unknownkind");
    std::ofstream(dir / "unknownkind.meta.json")
        << R"({"format_version": 1, "inputs": {"3": {"kind": "vaules", "values": [1]}}})";

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome));
    const auto& message = std::get<detail::FailedLoad>(outcome).message;
    EXPECT_NE(message.find("uid 3"), std::string::npos) << message;
    EXPECT_NE(message.find("vaules"), std::string::npos) << message;
}

TEST_F(TestBundleDiscoveryFixture, ClassifySweepCaseReturnsFailedLoadForUnknownFillKind)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"unknown_kind_fp32_nchw",
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
    sweepJson["cases"][0]["metadata"]["inputs"] = {{"3", {{"kind", "vaules"}}}};
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome));
    const auto& message = std::get<detail::FailedLoad>(outcome).message;
    EXPECT_NE(message.find("uid 3"), std::string::npos) << message;
    EXPECT_NE(message.find("vaules"), std::string::npos) << message;
}

// A harness that knows format_version 2 refuses a VALUES fill in version 1 metadata,
// so a case cannot rely on a fill that an older harness would read as random.
TEST_F(TestBundleDiscoveryFixture, ClassifySweepCaseReturnsFailedLoadForValuesInFormatVersion1)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"values_v1_fp32_nchw",
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
    auto& metadata = sweepJson["cases"][0]["metadata"];
    metadata["format_version"] = 1;
    metadata["inputs"] = {{"3", {{"kind", "values"}, {"values", {1, 2}}}}};
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);

    auto outcome = detail::classifyBundle(discovered.front(), _sweeps);
    ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome));
    const auto& message = std::get<detail::FailedLoad>(outcome).message;
    EXPECT_NE(message.find("kind values needs format_version 2"), std::string::npos) << message;

    metadata["format_version"] = 2;
    std::ofstream(sweepDir / "sweep.json") << sweepJson.dump(2);
    const auto rediscovered = discoverBundles(_tempDir);
    ASSERT_EQ(rediscovered.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(
        loadIntegrationTestBundle(rediscovered.front())));
}

// ---------------------------------------------------------------------------
// aliases: the load-time rules
// ---------------------------------------------------------------------------

namespace
{

nlohmann::json readJsonFile(const std::filesystem::path& path)
{
    std::ifstream in(path);
    return nlohmann::json::parse(in);
}

void writeJsonFile(const std::filesystem::path& path, const nlohmann::json& json)
{
    std::ofstream(path) << json.dump(2);
}

// Sets the fields of each patch ([{"uid": u, "set": {...}}]) on that tensor of `graph`.
void setTensorFields(nlohmann::json& graph, const nlohmann::json& patches)
{
    for(const auto& patch : patches)
    {
        for(auto& tensor : graph.at("tensors"))
        {
            if(tensor.at("uid") != patch.at("uid"))
            {
                continue;
            }
            for(const auto& item : patch.at("set").items())
            {
                tensor[item.key()] = item.value();
            }
        }
    }
}

nlohmann::json aliasMetadata(const nlohmann::json& aliases)
{
    return nlohmann::json{{"format_version", 2}, {"aliases", aliases}};
}

// The FailedLoad message of the one bundle under `root`. Any other outcome is a
// problem with the test itself, not the thing it's checking, so it throws instead
// of returning a placeholder a caller's own assertion would then fail against a
// second time.
std::string failedLoadMessage(const std::filesystem::path& root, SweepManifestCache& sweeps)
{
    const auto discovered = discoverBundles(root);
    if(discovered.size() != 1)
    {
        throw std::runtime_error("expected one bundle under " + root.string() + ", found "
                                 + std::to_string(discovered.size()));
    }
    auto outcome = detail::classifyBundle(discovered.front(), sweeps);
    if(!std::holds_alternative<detail::FailedLoad>(outcome))
    {
        throw std::runtime_error("expected a FailedLoad for " + root.string());
    }
    return std::get<detail::FailedLoad>(outcome).message;
}

// One SDPA backward node: q=1, k=2, v=3, o=4, do=5, stats=6 in; dq=7, dk=8, dv=9 out.
nlohmann::json sdpaBackwardGraphJson()
{
    const auto tensor = [](int64_t uid, bool stats) {
        return nlohmann::json{
            {"name", ""},
            {"uid", uid},
            {"dims", stats ? std::vector<int64_t>{1, 1, 2, 1} : std::vector<int64_t>{1, 1, 2, 4}},
            {"strides",
             stats ? std::vector<int64_t>{2, 2, 1, 1} : std::vector<int64_t>{8, 8, 4, 1}},
            {"data_type", "float"},
            {"virtual", false}};
    };
    auto tensors = nlohmann::json::array();
    for(int64_t uid = 1; uid <= 9; ++uid)
    {
        tensors.push_back(tensor(uid, uid == 6));
    }
    auto graph = nlohmann::json::parse(R"({
        "name": "sdpa_backward", "compute_data_type": "float", "io_data_type": "float",
        "intermediate_data_type": "float",
        "nodes": [{
            "name": "sdpa_backward", "type": "SdpaBackwardAttributes",
            "compute_data_type": "float",
            "inputs": {
                "q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3, "o_tensor_uid": 4,
                "do_tensor_uid": 5, "stats_tensor_uid": 6, "scale_tensor_uid": null,
                "attn_mask_tensor_uid": null, "seq_len_q_tensor_uid": null,
                "seq_len_kv_tensor_uid": null, "seed_tensor_uid": null,
                "offset_tensor_uid": null, "dropout_mask_tensor_uid": null,
                "dropout_scale_tensor_uid": null, "dropout_scale_inv_tensor_uid": null},
            "outputs": {
                "dq_tensor_uid": 7, "dk_tensor_uid": 8, "dv_tensor_uid": 9,
                "dbias_tensor_uid": null},
            "parameters": {
                "alibi_mask": false, "padding_mask": false, "causal_mask": false,
                "causal_mask_bottom_right": false, "dropout_probability": null,
                "attn_scale_value": null, "left_bound": null, "right_bound": null,
                "diagonal_alignment": "TOP_LEFT"}}]
    })");
    graph["tensors"] = std::move(tensors);
    return graph;
}

struct AliasRuleCase
{
    std::string name;
    std::string aliases;
    std::string patches;
    std::string message;
};

// Each case breaks one rule on the graph of createMinimalBundle() or of a sweep case of
// createTemplateSweep(): x=0 [2,3,4,5]; uids 1-4 [1,3,1,1] with strides [3,1,1,1];
// y=5 is the output.
std::vector<AliasRuleCase> aliasRuleCases()
{
    const std::string view3Of2 = R"([{"uid": 3, "of": 2, "byte_offset": 0}])";
    return {
        {"view_not_in_graph",
         R"([{"uid": 9, "of": 2, "byte_offset": 0}])",
         "[]",
         "alias view 9 of owner 2: the view is not a tensor of the graph"},
        {"owner_not_in_graph",
         R"([{"uid": 3, "of": 9, "byte_offset": 0}])",
         "[]",
         "alias view 3 of owner 9: the owner is not a tensor of the graph"},
        {"view_virtual",
         view3Of2,
         R"([{"uid": 3, "set": {"virtual": true}}])",
         "alias view 3 of owner 2: the view is virtual"},
        {"view_output",
         R"([{"uid": 5, "of": 2, "byte_offset": 0}])",
         "[]",
         "alias view 5 of owner 2: the view is an output"},
        {"owner_output",
         R"([{"uid": 3, "of": 5, "byte_offset": 0}])",
         "[]",
         "alias view 3 of owner 5: the owner is an output"},
        {"view_runtime_pass_by_value",
         view3Of2,
         R"([{"uid": 3, "set": {"is_runtime_pass_by_value": true}}])",
         "alias view 3 of owner 2: the view is passed by value at run time"},
        {"owner_ragged",
         view3Of2,
         R"([{"uid": 2, "set": {"ragged_offset_tensor_uid": 4}}])",
         "alias view 3 of owner 2: the owner is ragged"},
        {"sub_byte",
         view3Of2,
         R"([{"uid": 2, "set": {"data_type": "fp4_e2m1"}},
             {"uid": 3, "set": {"data_type": "fp4_e2m1"}}])",
         "alias view 3 of owner 2: the view has the sub-byte data type FP4_E2M1"},
        {"data_types_differ",
         view3Of2,
         R"([{"uid": 3, "set": {"data_type": "half"}}])",
         "alias view 3 of owner 2: the view's data type HALF differs from the owner's data type "
         "FLOAT"},
        {"offset_not_a_multiple",
         R"([{"uid": 3, "of": 2, "byte_offset": 2}])",
         "[]",
         "alias view 3 of owner 2: byte_offset 2 is not a multiple of the view's element size 4"},
        {"view_negative_stride",
         view3Of2,
         R"([{"uid": 3, "set": {"strides": [3, -1, 1, 1]}}])",
         "alias view 3 of owner 2: the view has a negative stride"},
        {"owner_negative_stride",
         view3Of2,
         R"([{"uid": 2, "set": {"strides": [3, -1, 1, 1]}}])",
         "alias view 3 of owner 2: the owner has a negative stride"},
        {"owner_zero_dim",
         view3Of2,
         R"([{"uid": 2, "set": {"dims": [1, 0, 1, 1]}}])",
         "alias view 3 of owner 2: the owner has a zero dim"},
        {"owner_size_overflow",
         view3Of2,
         R"([{"uid": 2, "set": {"strides": [3, 4611686018427387904, 1, 1]}}])",
         "alias view 3 of owner 2: the owner's size does not fit in a 64-bit byte count"},
        {"offset_breaks_alignment",
         R"([{"uid": 3, "of": 2, "byte_offset": 4}])",
         "[]",
         "alias view 3 of owner 2: byte_offset 4 is not a multiple of the view's alignment 16"},
        {"owner_alignment_below_the_views",
         view3Of2,
         R"([{"uid": 3, "set": {"alignment": 32}}])",
         "alias view 3 of owner 2: the owner's alignment 16 is not a multiple of the view's "
         "alignment 32"},
        {"view_constant_value",
         view3Of2,
         R"([{"uid": 3, "set": {"value_type": "Float32Value", "value": 1.0}}])",
         "alias view 3 of owner 2: the view has a constant value"},
        {"owner_negative_dim",
         view3Of2,
         R"([{"uid": 2, "set": {"dims": [1, -3, 1, 1]}}])",
         "alias view 3 of owner 2: the owner has a negative dim"},
        {"view_reach_overflow",
         view3Of2,
         R"([{"uid": 3, "set": {"strides": [3, 4611686018427387904, 1, 1]}}])",
         "alias view 3 of owner 2: the view's reach does not fit in a 64-bit byte count"},
        {"zero_dim_view_offset_past_owner",
         R"([{"uid": 3, "of": 2, "byte_offset": 16}])",
         R"([{"uid": 3, "set": {"dims": [1, 0, 1, 1]}}])",
         "alias view 3 of owner 2: byte_offset 16 is past the end of the owner's 12-byte "
         "allocation"},
        {"out_of_bounds",
         R"([{"uid": 3, "of": 2, "byte_offset": 16}])",
         "[]",
         "alias view 3 of owner 2: the view reaches byte 28, past the end of the owner's "
         "12-byte allocation"},
    };
}

} // namespace

TEST_F(TestBundleDiscoveryFixture, AliasRuleViolationsFailTheLoadOfABundle)
{
    for(const auto& c : aliasRuleCases())
    {
        SCOPED_TRACE(c.name);
        const auto root = _tempDir / ("direct_" + c.name);
        const auto dir = root / "op" / "aliased";
        createMinimalBundle(dir, "aliased");
        auto graph = readJsonFile(dir / "aliased.json");
        setTensorFields(graph, nlohmann::json::parse(c.patches));
        writeJsonFile(dir / "aliased.json", graph);
        writeJsonFile(dir / "aliased.meta.json", aliasMetadata(nlohmann::json::parse(c.aliases)));

        EXPECT_THROW(loadIntegrationTestBundle(dir / "aliased.json"), BundleMetadataError);
        const auto message = failedLoadMessage(root, _sweeps);
        EXPECT_NE(message.find(c.message), std::string::npos) << message;
    }
}

TEST_F(TestBundleDiscoveryFixture, AliasRuleViolationsFailTheLoadOfASweepCase)
{
    for(const auto& c : aliasRuleCases())
    {
        SCOPED_TRACE(c.name);
        const auto root = _tempDir / ("sweep_" + c.name);
        const auto sweepDir = root / "quick" / "BatchnormFwdInference" / "Inference";
        createTemplateSweep(sweepDir,
                            {{"aliased_fp32_nchw",
                              "float",
                              {2, 3, 4, 5},
                              {60, 20, 5, 1},
                              {1, 3, 1, 1},
                              {3, 1, 1, 1},
                              false, // includeGolden
                              false, // goldenHasPath
                              true}}); // includeMetadata
        auto sweepJson = readJsonFile(sweepDir / "sweep.json");
        auto& caseJson = sweepJson["cases"][0];
        caseJson["metadata"] = aliasMetadata(nlohmann::json::parse(c.aliases));
        caseJson["tensor_patches"] = nlohmann::json::parse(c.patches);
        writeJsonFile(sweepDir / "sweep.json", sweepJson);

        const auto message = failedLoadMessage(root, _sweeps);
        EXPECT_NE(message.find(c.message), std::string::npos) << message;
    }
}

TEST_F(TestBundleDiscoveryFixture, ValidAliasLoadsInABundleAndASweepCase)
{
    const auto dir = _tempDir / "op" / "aliased";
    createMinimalBundle(dir, "aliased");
    writeJsonFile(
        dir / "aliased.meta.json",
        aliasMetadata(nlohmann::json::parse(R"([{"uid": 3, "of": 2, "byte_offset": 0}])")));
    const auto direct = loadIntegrationTestBundle(dir / "aliased.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(direct));
    const auto& directAliases = std::get<IntegrationTestBundle>(direct).metadata.aliases;
    ASSERT_EQ(directAliases.size(), 1u);
    EXPECT_EQ(directAliases.front(), (hipdnn_integration_tests::TensorAlias{3, 2, 0}));

    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"aliased_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          false, // includeGolden
                          false, // goldenHasPath
                          true}}); // includeMetadata
    auto sweepJson = readJsonFile(sweepDir / "sweep.json");
    sweepJson["cases"][0]["metadata"]
        = aliasMetadata(nlohmann::json::parse(R"([{"uid": 4, "of": 1, "byte_offset": 4}])"));
    // A view with a zero dim reaches no bytes; its offset stays inside the owner and
    // follows the view's alignment.
    sweepJson["cases"][0]["tensor_patches"]
        = nlohmann::json::parse(R"([{"uid": 4, "set": {"dims": [1, 0, 1, 1], "alignment": 4}}])");
    writeJsonFile(sweepDir / "sweep.json", sweepJson);

    const auto discovered = discoverBundles(_tempDir);
    const auto* sweepCase = findByTest(discovered, "aliased_fp32_nchw");
    ASSERT_NE(sweepCase, nullptr);
    const auto sweep = loadIntegrationTestBundle(*sweepCase, _sweeps);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(sweep));
    EXPECT_EQ(std::get<IntegrationTestBundle>(sweep).metadata.aliases.front(),
              (hipdnn_integration_tests::TensorAlias{4, 1, 4}));
}

TEST_F(TestBundleDiscoveryFixture, AliasInABundleWithGoldenDataFailsTheLoad)
{
    const std::string rule
        = "alias view 3 of owner 2: the bundle has golden data or tensor blobs, which store "
          "each tensor apart";

    const auto directRoot = _tempDir / "direct";
    const auto dir = directRoot / "op" / "aliased";
    createMinimalBundle(dir, "aliased");
    writeJsonFile(
        dir / "aliased.meta.json",
        aliasMetadata(nlohmann::json::parse(R"([{"uid": 3, "of": 2, "byte_offset": 0}])")));
    std::ofstream(dir / "aliased.tensors.dvc") << "outs:\n";
    const auto directMessage = failedLoadMessage(directRoot, _sweeps);
    EXPECT_NE(directMessage.find(rule), std::string::npos) << directMessage;

    const auto sweepRoot = _tempDir / "sweep";
    const auto sweepDir = sweepRoot / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(
        sweepDir,
        {{"aliased_fp32_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});
    auto sweepJson = readJsonFile(sweepDir / "sweep.json");
    sweepJson["cases"][0]["metadata"]
        = aliasMetadata(nlohmann::json::parse(R"([{"uid": 3, "of": 2, "byte_offset": 0}])"));
    writeJsonFile(sweepDir / "sweep.json", sweepJson);
    const auto sweepMessage = failedLoadMessage(sweepRoot, _sweeps);
    EXPECT_NE(sweepMessage.find(rule), std::string::npos) << sweepMessage;
}

// The input fill of an SDPA backward node replaces O and stats with the forward of Q, K
// and V, and reads each of them as its own tensor.
TEST_F(TestBundleDiscoveryFixture, AliasInAGraphWhoseFillDerivesInputsFailsTheLoad)
{
    const std::string rule = "alias view 3 of owner 2: the graph's input fill derives some inputs "
                             "from other inputs";
    const auto aliases
        = aliasMetadata(nlohmann::json::parse(R"([{"uid": 3, "of": 2, "byte_offset": 0}])"));

    const auto directRoot = _tempDir / "direct";
    const auto dir = directRoot / "op" / "aliased";
    std::filesystem::create_directories(dir);
    writeJsonFile(dir / "aliased.json", sdpaBackwardGraphJson());
    writeJsonFile(dir / "aliased.meta.json", aliases);
    const auto directMessage = failedLoadMessage(directRoot, _sweeps);
    EXPECT_NE(directMessage.find(rule), std::string::npos) << directMessage;

    const auto sweepRoot = _tempDir / "sweep";
    const auto sweepDir = sweepRoot / "quick" / "SdpaBackward" / "Aliased";
    std::filesystem::create_directories(sweepDir);
    writeJsonFile(sweepDir / "graph.template.json", sdpaBackwardGraphJson());
    writeJsonFile(
        sweepDir / "sweep.json",
        nlohmann::json{{"version", 1},
                       {"cases",
                        nlohmann::json::array({nlohmann::json{{"id", "aliased"},
                                                              {"values", nlohmann::json::object()},
                                                              {"metadata", aliases}}})}});
    const auto sweepMessage = failedLoadMessage(sweepRoot, _sweeps);
    EXPECT_NE(sweepMessage.find(rule), std::string::npos) << sweepMessage;
}

// Input blobs without golden outputs make a reference-verified bundle, which still
// reads each input from its own blob.
TEST_F(TestBundleDiscoveryFixture, AliasInABundleWithInputBlobsFailsTheLoad)
{
    const auto dir = _tempDir / "op" / "aliased";
    createLoadableBundle(dir, "aliased");
    std::filesystem::remove(dir / "aliased.tensor5.bin");
    writeJsonFile(
        dir / "aliased.meta.json",
        aliasMetadata(nlohmann::json::parse(R"([{"uid": 3, "of": 2, "byte_offset": 0}])")));

    const auto message = failedLoadMessage(_tempDir, _sweeps);
    EXPECT_NE(message.find("alias view 3 of owner 2: the bundle has golden data or tensor blobs"),
              std::string::npos)
        << message;
}

// The view has no blob, so not every input has one, but the other blobs still store
// each tensor apart.
TEST_F(TestBundleDiscoveryFixture, AliasInABundleWithSomeInputBlobsFailsTheLoad)
{
    const auto dir = _tempDir / "op" / "aliased";
    createLoadableBundle(dir, "aliased");
    std::filesystem::remove(dir / "aliased.tensor3.bin");
    std::filesystem::remove(dir / "aliased.tensor5.bin");
    writeJsonFile(
        dir / "aliased.meta.json",
        aliasMetadata(nlohmann::json::parse(R"([{"uid": 3, "of": 2, "byte_offset": 0}])")));

    const auto message = failedLoadMessage(_tempDir, _sweeps);
    EXPECT_NE(message.find("alias view 3 of owner 2: the bundle has golden data or tensor blobs"),
              std::string::npos)
        << message;
}

// Mirrors AliasInABundleWithSomeInputBlobsFailsTheLoad, for a sweep case: the view's
// own blob and the output's blob are both absent, but the owner's blob (and others)
// are still present, so the sweep case still stores some tensor apart from the alias.
TEST_F(TestBundleDiscoveryFixture, AliasInASweepCaseWithSomeInputBlobsFailsTheLoad)
{
    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"aliased_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          /*includeGolden=*/true,
                          /*goldenHasPath=*/true,
                          /*includeMetadata=*/false}});
    auto sweepJson = readJsonFile(sweepDir / "sweep.json");
    sweepJson["cases"][0]["metadata"]
        = aliasMetadata(nlohmann::json::parse(R"([{"uid": 3, "of": 2, "byte_offset": 0}])"));
    writeJsonFile(sweepDir / "sweep.json", sweepJson);
    std::filesystem::remove(sweepDir / "golden" / "aliased_fp32_nchw" / "tensor3.bin");
    std::filesystem::remove(sweepDir / "golden" / "aliased_fp32_nchw" / "tensor5.bin");

    const auto message = failedLoadMessage(_tempDir, _sweeps);
    EXPECT_NE(message.find("alias view 3 of owner 2: the bundle has golden data or tensor blobs"),
              std::string::npos)
        << message;
}

// ---------------------------------------------------------------------------
// start_offsets: the load-time rules
// ---------------------------------------------------------------------------

namespace
{

nlohmann::json startOffsetMetadata(const std::string& startOffsets,
                                   const std::string& aliases = std::string())
{
    auto metadata = nlohmann::json{{"format_version", 2},
                                   {"start_offsets", nlohmann::json::parse(startOffsets)}};
    if(!aliases.empty())
    {
        metadata["aliases"] = nlohmann::json::parse(aliases);
    }
    return metadata;
}

struct StartOffsetRuleCase
{
    std::string name;
    std::string startOffsets;
    std::string aliases;
    std::string patches;
    std::string message;
};

// Each case breaks one rule on the graph of createMinimalBundle() or of a sweep case of
// createTemplateSweep(): x=0 [2,3,4,5]; uids 1-4 [1,3,1,1]; y=5 is the output. Every
// tensor has the default alignment 16.
std::vector<StartOffsetRuleCase> startOffsetRuleCases()
{
    const std::string tensor3 = R"([{"uid": 3, "start_offset": 16}])";
    const std::string view3Of2 = R"([{"uid": 3, "of": 2, "byte_offset": 0}])";
    return {
        {"not_in_graph",
         R"([{"uid": 9, "start_offset": 16}])",
         "",
         "[]",
         "start offset of tensor 9: the tensor is not a tensor of the graph"},
        {"virtual",
         tensor3,
         "",
         R"([{"uid": 3, "set": {"virtual": true}}])",
         "start offset of tensor 3: the tensor is virtual"},
        {"runtime_pass_by_value",
         tensor3,
         "",
         R"([{"uid": 3, "set": {"is_runtime_pass_by_value": true}}])",
         "start offset of tensor 3: the tensor is passed by value at run time"},
        {"constant_value",
         tensor3,
         "",
         R"([{"uid": 3, "set": {"value_type": "Float32Value", "value": 1.0}}])",
         "start offset of tensor 3: the tensor has a constant value"},
        {"alias_view",
         tensor3,
         view3Of2,
         "[]",
         "start offset of tensor 3: the tensor is alias view 3 of owner 2; a view moves with "
         "its owner"},
        {"sub_byte",
         tensor3,
         "",
         R"([{"uid": 3, "set": {"data_type": "fp4_e2m1"}}])",
         "start offset of tensor 3: the tensor has the sub-byte data type FP4_E2M1"},
        {"default_alignment_does_not_divide_the_offset",
         R"([{"uid": 3, "start_offset": 2}])",
         "",
         "[]",
         "start offset of tensor 3: start_offset 2 is not a multiple of the tensor's "
         "alignment 16"},
        {"declared_alignment_does_not_divide_the_offset",
         R"([{"uid": 5, "start_offset": 6}])",
         "",
         R"([{"uid": 5, "set": {"alignment": 4}}])",
         "start offset of tensor 5: start_offset 6 is not a multiple of the tensor's "
         "alignment 4"},
    };
}

} // namespace

TEST_F(TestBundleDiscoveryFixture, StartOffsetRuleViolationsFailTheLoadOfABundle)
{
    for(const auto& c : startOffsetRuleCases())
    {
        SCOPED_TRACE(c.name);
        const auto root = _tempDir / ("direct_" + c.name);
        const auto dir = root / "op" / "shifted";
        createMinimalBundle(dir, "shifted");
        auto graph = readJsonFile(dir / "shifted.json");
        setTensorFields(graph, nlohmann::json::parse(c.patches));
        writeJsonFile(dir / "shifted.json", graph);
        writeJsonFile(dir / "shifted.meta.json", startOffsetMetadata(c.startOffsets, c.aliases));

        EXPECT_THROW(loadIntegrationTestBundle(dir / "shifted.json"), BundleMetadataError);
        const auto message = failedLoadMessage(root, _sweeps);
        EXPECT_NE(message.find(c.message), std::string::npos) << message;
    }
}

TEST_F(TestBundleDiscoveryFixture, StartOffsetRuleViolationsFailTheLoadOfASweepCase)
{
    for(const auto& c : startOffsetRuleCases())
    {
        SCOPED_TRACE(c.name);
        const auto root = _tempDir / ("sweep_" + c.name);
        const auto sweepDir = root / "quick" / "BatchnormFwdInference" / "Inference";
        createTemplateSweep(sweepDir,
                            {{"shifted_fp32_nchw",
                              "float",
                              {2, 3, 4, 5},
                              {60, 20, 5, 1},
                              {1, 3, 1, 1},
                              {3, 1, 1, 1},
                              false, // includeGolden
                              false, // goldenHasPath
                              true}}); // includeMetadata
        auto sweepJson = readJsonFile(sweepDir / "sweep.json");
        auto& caseJson = sweepJson["cases"][0];
        caseJson["metadata"] = startOffsetMetadata(c.startOffsets, c.aliases);
        caseJson["tensor_patches"] = nlohmann::json::parse(c.patches);
        writeJsonFile(sweepDir / "sweep.json", sweepJson);

        const auto message = failedLoadMessage(root, _sweeps);
        EXPECT_NE(message.find(c.message), std::string::npos) << message;
    }
}

// An input and an output, each with an alignment that divides its offset, and an alias
// view of a moved owner whose offset keeps the view's alignment.
TEST_F(TestBundleDiscoveryFixture, ValidStartOffsetsLoadInABundleAndASweepCase)
{
    using hipdnn_integration_tests::TensorStartOffset;
    const std::string startOffsets
        = R"([{"uid": 0, "start_offset": 2}, {"uid": 5, "start_offset": 1},
              {"uid": 2, "start_offset": 4}])";
    const std::string aliases = R"([{"uid": 3, "of": 2, "byte_offset": 8}])";
    const auto patches = nlohmann::json::parse(
        R"([{"uid": 0, "set": {"alignment": 2}},
            {"uid": 5, "set": {"alignment": 0}},
            {"uid": 2, "set": {"alignment": 4}},
            {"uid": 3, "set": {"alignment": 4, "dims": [1, 1, 1, 1]}}])");
    const std::vector<TensorStartOffset> expected = {{0, 2}, {5, 1}, {2, 4}};

    const auto dir = _tempDir / "op" / "shifted";
    createMinimalBundle(dir, "shifted");
    auto graph = readJsonFile(dir / "shifted.json");
    setTensorFields(graph, patches);
    writeJsonFile(dir / "shifted.json", graph);
    writeJsonFile(dir / "shifted.meta.json", startOffsetMetadata(startOffsets, aliases));
    const auto direct = loadIntegrationTestBundle(dir / "shifted.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(direct));
    EXPECT_EQ(std::get<IntegrationTestBundle>(direct).metadata.startOffsets, expected);

    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"shifted_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          false, // includeGolden
                          false, // goldenHasPath
                          true}}); // includeMetadata
    auto sweepJson = readJsonFile(sweepDir / "sweep.json");
    sweepJson["cases"][0]["metadata"] = startOffsetMetadata(startOffsets, aliases);
    sweepJson["cases"][0]["tensor_patches"] = patches;
    writeJsonFile(sweepDir / "sweep.json", sweepJson);

    const auto discovered = discoverBundles(_tempDir);
    const auto* sweepCase = findByTest(discovered, "shifted_fp32_nchw");
    ASSERT_NE(sweepCase, nullptr);
    const auto sweep = loadIntegrationTestBundle(*sweepCase, _sweeps);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(sweep));
    EXPECT_EQ(std::get<IntegrationTestBundle>(sweep).metadata.startOffsets, expected);
}

// The offset applies to the engine's copy only, so a bundle with golden data or tensor
// blobs keeps them.
TEST_F(TestBundleDiscoveryFixture, StartOffsetInABundleWithGoldenDataLoads)
{
    const auto startOffsets = startOffsetMetadata(R"([{"uid": 5, "start_offset": 16}])");

    const auto dir = _tempDir / "op" / "shifted";
    createLoadableBundle(dir, "shifted");
    writeJsonFile(dir / "shifted.meta.json", startOffsets);
    std::ofstream(dir / "shifted.tensors.dvc") << "outs:\n";
    const auto direct = loadIntegrationTestBundle(dir / "shifted.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(direct));
    const auto& directBundle = std::get<IntegrationTestBundle>(direct);
    EXPECT_TRUE(directBundle.hasGoldenOutputs);
    EXPECT_EQ(directBundle.metadata.startOffsets.size(), 1u);

    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(
        sweepDir,
        {{"shifted_fp32_nchw", "float", {2, 3, 4, 5}, {60, 20, 5, 1}, {1, 3, 1, 1}, {3, 1, 1, 1}}});
    auto sweepJson = readJsonFile(sweepDir / "sweep.json");
    sweepJson["cases"][0]["metadata"] = startOffsets;
    writeJsonFile(sweepDir / "sweep.json", sweepJson);

    const auto discovered = discoverBundles(_tempDir);
    const auto* sweepCase = findByTest(discovered, "shifted_fp32_nchw");
    ASSERT_NE(sweepCase, nullptr);
    const auto sweep = loadIntegrationTestBundle(*sweepCase, _sweeps);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(sweep));
    const auto& sweepBundle = std::get<IntegrationTestBundle>(sweep);
    EXPECT_TRUE(sweepBundle.hasGoldenOutputs);
    EXPECT_EQ(sweepBundle.metadata.startOffsets.size(), 1u);
}

TEST_F(TestBundleDiscoveryFixture, StartOffsetInFormatVersion1MetadataFailsTheLoad)
{
    const auto dir = _tempDir / "op" / "shifted";
    createMinimalBundle(dir, "shifted");
    writeJsonFile(dir / "shifted.meta.json",
                  nlohmann::json{{"format_version", 1},
                                 {"start_offsets",
                                  nlohmann::json::parse(R"([{"uid": 5, "start_offset": 16}])")}});
    const auto message = failedLoadMessage(_tempDir, _sweeps);
    EXPECT_NE(message.find("start_offsets needs format_version 2"), std::string::npos) << message;
}

// ---------------------------------------------------------------------------
// live_rows: the load-time rules
// ---------------------------------------------------------------------------

namespace
{

nlohmann::json liveRowMetadata(const std::string& liveRows,
                               const std::string& aliases = std::string())
{
    auto metadata
        = nlohmann::json{{"format_version", 2}, {"live_rows", nlohmann::json::parse(liveRows)}};
    if(!aliases.empty())
    {
        metadata["aliases"] = nlohmann::json::parse(aliases);
    }
    return metadata;
}

struct LiveRowRuleCase
{
    std::string name;
    std::string liveRows;
    std::string aliases;
    std::string patches;
    std::string message;
};

// Lengths for the two batches of y=5, in uid 1.
constexpr const char* K_INT32_LENGTHS_IN_UID_1
    = R"({"uid": 1, "set": {"data_type": "int32", "dims": [2, 1, 1, 1], "strides": [1, 1, 1, 1]}})";

// Each case breaks one rule on the graph of createMinimalBundle() or of a sweep case of
// createTemplateSweep(): x=0 [2,3,4,5]; uids 1-4 [1,3,1,1]; y=5 [2,3,4,5] is the output.
std::vector<LiveRowRuleCase> liveRowRuleCases()
{
    const std::string byLengths = R"([{"uid": 5, "axis": 2, "lengths": 1}])";
    const std::string lengthsPatch = std::string("[") + K_INT32_LENGTHS_IN_UID_1 + "]";
    return {
        {"output_not_in_graph",
         R"([{"uid": 9, "like": 0}])",
         "",
         "[]",
         "live rows of output 9: the output is not a tensor of the graph"},
        {"not_an_output",
         R"([{"uid": 0, "axis": 2, "lengths": 1}])",
         "",
         lengthsPatch,
         "live rows of output 0: the tensor is not a compared output"},
        {"ragged_output",
         byLengths,
         "",
         std::string(R"([{"uid": 5, "set": {"ragged_offset_tensor_uid": 4}}, )")
             + K_INT32_LENGTHS_IN_UID_1 + "]",
         "live rows of output 5: the output is ragged"},
        {"integer_output",
         byLengths,
         "",
         std::string(R"([{"uid": 5, "set": {"data_type": "int32"}}, )") + K_INT32_LENGTHS_IN_UID_1
             + "]",
         "live rows of output 5: the output has the data type INT32, which is not a "
         "floating-point type"},
        {"axis_zero",
         R"([{"uid": 5, "axis": 0, "lengths": 1}])",
         "",
         lengthsPatch,
         "live rows of output 5: axis 0 is not in the range 1 to 3"},
        {"axis_past_the_rank",
         R"([{"uid": 5, "axis": 4, "lengths": 1}])",
         "",
         lengthsPatch,
         "live rows of output 5: axis 4 is not in the range 1 to 3"},
        {"lengths_not_in_graph",
         R"([{"uid": 5, "axis": 2, "lengths": 9}])",
         "",
         "[]",
         "live rows of output 5: lengths tensor 9 is not a tensor of the graph"},
        {"lengths_is_an_output",
         R"([{"uid": 5, "axis": 2, "lengths": 5}])",
         "",
         "[]",
         "live rows of output 5: lengths tensor 5 is not a graph input"},
        {"lengths_virtual",
         byLengths,
         "",
         R"([{"uid": 1, "set": {"virtual": true}}])",
         "live rows of output 5: lengths tensor 1 is not a graph input"},
        {"lengths_runtime_pass_by_value",
         byLengths,
         "",
         R"([{"uid": 1, "set": {"is_runtime_pass_by_value": true}}])",
         "live rows of output 5: lengths tensor 1 is passed by value at run time"},
        {"lengths_alias_view",
         byLengths,
         R"([{"uid": 1, "of": 2, "byte_offset": 0}])",
         lengthsPatch,
         "live rows of output 5: lengths tensor 1 is alias view 1 of owner 2"},
        {"lengths_float",
         byLengths,
         "",
         R"([{"uid": 1, "set": {"dims": [2, 1, 1, 1], "strides": [1, 1, 1, 1]}}])",
         "live rows of output 5: lengths tensor 1 has the data type FLOAT, not INT32 or INT64"},
        {"lengths_count_differs_from_the_batches",
         byLengths,
         "",
         R"([{"uid": 1, "set": {"data_type": "int64"}}])",
         "live rows of output 5: lengths tensor 1 has 3 elements; the output has 2 batches"},
        {"like_not_in_graph",
         R"([{"uid": 5, "like": 9}])",
         "",
         "[]",
         "live rows of output 5: like tensor 9 is not a tensor of the graph"},
        {"like_virtual",
         R"([{"uid": 5, "like": 0}])",
         "",
         R"([{"uid": 0, "set": {"ragged_offset_tensor_uid": 4, "virtual": true}}])",
         "live rows of output 5: like tensor 0 is virtual"},
        {"like_not_ragged",
         R"([{"uid": 5, "like": 0}])",
         "",
         "[]",
         "live rows of output 5: like tensor 0 is not ragged"},
        {"like_batches_differ",
         R"([{"uid": 5, "like": 0}])",
         "",
         R"([{"uid": 0, "set": {"ragged_offset_tensor_uid": 4, "dims": [3, 3, 4, 5]}}])",
         "live rows of output 5: like tensor 0 does not have the same dims[0] as the output"},
    };
}

} // namespace

TEST_F(TestBundleDiscoveryFixture, LiveRowRuleViolationsFailTheLoadOfABundle)
{
    for(const auto& c : liveRowRuleCases())
    {
        SCOPED_TRACE(c.name);
        const auto root = _tempDir / ("direct_" + c.name);
        const auto dir = root / "op" / "rows";
        createMinimalBundle(dir, "rows");
        auto graph = readJsonFile(dir / "rows.json");
        setTensorFields(graph, nlohmann::json::parse(c.patches));
        writeJsonFile(dir / "rows.json", graph);
        writeJsonFile(dir / "rows.meta.json", liveRowMetadata(c.liveRows, c.aliases));

        EXPECT_THROW(loadIntegrationTestBundle(dir / "rows.json"), BundleMetadataError);
        const auto message = failedLoadMessage(root, _sweeps);
        EXPECT_NE(message.find(c.message), std::string::npos) << message;
    }
}

TEST_F(TestBundleDiscoveryFixture, LiveRowRuleViolationsFailTheLoadOfASweepCase)
{
    for(const auto& c : liveRowRuleCases())
    {
        SCOPED_TRACE(c.name);
        const auto root = _tempDir / ("sweep_" + c.name);
        const auto sweepDir = root / "quick" / "BatchnormFwdInference" / "Inference";
        createTemplateSweep(sweepDir,
                            {{"rows_fp32_nchw",
                              "float",
                              {2, 3, 4, 5},
                              {60, 20, 5, 1},
                              {1, 3, 1, 1},
                              {3, 1, 1, 1},
                              false, // includeGolden
                              false, // goldenHasPath
                              true}}); // includeMetadata
        auto sweepJson = readJsonFile(sweepDir / "sweep.json");
        auto& caseJson = sweepJson["cases"][0];
        caseJson["metadata"] = liveRowMetadata(c.liveRows, c.aliases);
        caseJson["tensor_patches"] = nlohmann::json::parse(c.patches);
        writeJsonFile(sweepDir / "sweep.json", sweepJson);

        const auto message = failedLoadMessage(root, _sweeps);
        EXPECT_NE(message.find(c.message), std::string::npos) << message;
    }
}

TEST_F(TestBundleDiscoveryFixture, ValidLiveRowsLoadInABundleAndASweepCase)
{
    using hipdnn_integration_tests::OutputLiveRows;
    const std::string liveRows = R"([{"uid": 5, "axis": 2, "lengths": 1}])";
    const auto patches = nlohmann::json::parse(std::string("[") + K_INT32_LENGTHS_IN_UID_1 + "]");
    const std::vector<OutputLiveRows> expected = {{5, 2, 1, std::nullopt}};

    const auto dir = _tempDir / "op" / "rows";
    createMinimalBundle(dir, "rows");
    auto graph = readJsonFile(dir / "rows.json");
    setTensorFields(graph, patches);
    writeJsonFile(dir / "rows.json", graph);
    writeJsonFile(dir / "rows.meta.json", liveRowMetadata(liveRows));
    const auto direct = loadIntegrationTestBundle(dir / "rows.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(direct));
    EXPECT_EQ(std::get<IntegrationTestBundle>(direct).metadata.liveRows, expected);

    const auto sweepDir = _tempDir / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"rows_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          false, // includeGolden
                          false, // goldenHasPath
                          true}}); // includeMetadata
    auto sweepJson = readJsonFile(sweepDir / "sweep.json");
    sweepJson["cases"][0]["metadata"] = liveRowMetadata(liveRows);
    sweepJson["cases"][0]["tensor_patches"] = patches;
    writeJsonFile(sweepDir / "sweep.json", sweepJson);

    const auto discovered = discoverBundles(_tempDir);
    const auto* sweepCase = findByTest(discovered, "rows_fp32_nchw");
    ASSERT_NE(sweepCase, nullptr);
    const auto sweep = loadIntegrationTestBundle(*sweepCase, _sweeps);
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(sweep));
    EXPECT_EQ(std::get<IntegrationTestBundle>(sweep).metadata.liveRows, expected);
}

TEST_F(TestBundleDiscoveryFixture, ValidLiveRowsLikeARaggedTensorLoad)
{
    using hipdnn_integration_tests::OutputLiveRows;
    const auto dir = _tempDir / "op" / "rows";
    createMinimalBundle(dir, "rows");
    auto graph = readJsonFile(dir / "rows.json");
    setTensorFields(
        graph, nlohmann::json::parse(R"([{"uid": 0, "set": {"ragged_offset_tensor_uid": 4}}])"));
    writeJsonFile(dir / "rows.json", graph);
    writeJsonFile(dir / "rows.meta.json", liveRowMetadata(R"([{"uid": 5, "like": 0}])"));

    const auto direct = loadIntegrationTestBundle(dir / "rows.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(direct));
    EXPECT_EQ(std::get<IntegrationTestBundle>(direct).metadata.liveRows,
              (std::vector<OutputLiveRows>{{5, std::nullopt, std::nullopt, 0}}));
}

// Live rows change only the comparison, so a bundle with golden data keeps it.
TEST_F(TestBundleDiscoveryFixture, LiveRowsInABundleWithGoldenDataLoad)
{
    const auto dir = _tempDir / "op" / "rows";
    createLoadableBundle(dir, "rows");
    auto graph = readJsonFile(dir / "rows.json");
    setTensorFields(graph,
                    nlohmann::json::parse(std::string("[") + K_INT32_LENGTHS_IN_UID_1 + "]"));
    writeJsonFile(dir / "rows.json", graph);
    writeJsonFile(dir / "rows.meta.json",
                  liveRowMetadata(R"([{"uid": 5, "axis": 2, "lengths": 1}])"));
    std::ofstream(dir / "rows.tensors.dvc") << "outs:\n";

    const auto direct = loadIntegrationTestBundle(dir / "rows.json");
    ASSERT_TRUE(std::holds_alternative<IntegrationTestBundle>(direct));
    EXPECT_EQ(std::get<IntegrationTestBundle>(direct).metadata.liveRows.size(), 1u);
}

TEST_F(TestBundleDiscoveryFixture, LiveRowsInFormatVersion1MetadataFailTheLoad)
{
    const auto dir = _tempDir / "op" / "rows";
    createMinimalBundle(dir, "rows");
    writeJsonFile(
        dir / "rows.meta.json",
        nlohmann::json{{"format_version", 1},
                       {"live_rows", nlohmann::json::parse(R"([{"uid": 5, "like": 0}])")}});
    const auto message = failedLoadMessage(_tempDir, _sweeps);
    EXPECT_NE(message.find("live_rows needs format_version 2"), std::string::npos) << message;
}

TEST_F(TestBundleDiscoveryFixture, AliasInFormatVersion1MetadataFailsTheLoad)
{
    const auto v1Aliases = nlohmann::json{
        {"format_version", 1},
        {"aliases", nlohmann::json::parse(R"([{"uid": 3, "of": 2, "byte_offset": 0}])")}};

    const auto directRoot = _tempDir / "direct";
    const auto dir = directRoot / "op" / "aliased";
    createMinimalBundle(dir, "aliased");
    writeJsonFile(dir / "aliased.meta.json", v1Aliases);
    const auto directMessage = failedLoadMessage(directRoot, _sweeps);
    EXPECT_NE(directMessage.find("aliases needs format_version 2"), std::string::npos)
        << directMessage;

    const auto sweepRoot = _tempDir / "sweep";
    const auto sweepDir = sweepRoot / "quick" / "BatchnormFwdInference" / "Inference";
    createTemplateSweep(sweepDir,
                        {{"aliased_fp32_nchw",
                          "float",
                          {2, 3, 4, 5},
                          {60, 20, 5, 1},
                          {1, 3, 1, 1},
                          {3, 1, 1, 1},
                          false, // includeGolden
                          false, // goldenHasPath
                          true}}); // includeMetadata
    auto sweepJson = readJsonFile(sweepDir / "sweep.json");
    sweepJson["cases"][0]["metadata"] = v1Aliases;
    writeJsonFile(sweepDir / "sweep.json", sweepJson);
    const auto sweepMessage = failedLoadMessage(sweepRoot, _sweeps);
    EXPECT_NE(sweepMessage.find("aliases needs format_version 2"), std::string::npos)
        << sweepMessage;
}

// ---------------------------------------------------------------------------
// absent_tensors: optional tensors in sweep templates
// ---------------------------------------------------------------------------

namespace
{

namespace fx = hipdnn_integration_tests::bundle::fixtures;
namespace sdpa_utils = hipdnn_integration_tests::test_utils;

constexpr int64_t SINK = sdpa_utils::K_OPTIONAL_SDPA_SINK_UID;
constexpr int64_t V_UID = sdpa_utils::K_OPTIONAL_SDPA_V_UID;
constexpr int64_t O_UID = sdpa_utils::K_OPTIONAL_SDPA_O_UID;
constexpr int64_t STATS = sdpa_utils::K_OPTIONAL_SDPA_STATS_UID;

// The case under test in the sweeps of the absent-tensor rules.
constexpr const char* ABSENT_CASE_ID = "absent_case";

nlohmann::json supersetGraph()
{
    return sdpa_utils::optionalSdpaFwdGraphJson(/*withStats=*/true, /*withSink=*/true);
}

nlohmann::json graphJsonOf(const IntegrationTestBundle& bundle)
{
    nlohmann::json graphJson;
    hipdnn_flatbuffers_sdk::data_objects::to_json(
        graphJson, *hipdnn_flatbuffers_sdk::data_objects::GetGraph(bundle.graphBuffer.data()));
    return graphJson;
}

// The loaded case `caseId` under `root`; nullopt, with a test failure, when it does not load.
std::optional<IntegrationTestBundle> loadSweepCase(const std::filesystem::path& root,
                                                   const std::string& caseId,
                                                   SweepManifestCache& sweeps)
{
    for(const auto& discovered : discoverBundles(root))
    {
        if(discovered.testName != caseId)
        {
            continue;
        }
        auto result = loadIntegrationTestBundle(discovered, sweeps);
        if(auto* loaded = std::get_if<IntegrationTestBundle>(&result))
        {
            return std::move(*loaded);
        }
        ADD_FAILURE() << caseId << " did not load: " << toString(std::get<LoadError>(result));
        return std::nullopt;
    }
    ADD_FAILURE() << "no case " << caseId << " under " << root;
    return std::nullopt;
}

struct AbsentRuleCase
{
    std::string name;
    // Changes the superset template and the sweep, whose one case marks the sink absent.
    std::function<void(nlohmann::json& templateJson, nlohmann::json& sweepJson)> edit;
    std::string message;
};

std::vector<AbsentRuleCase> absentRuleCases()
{
    const auto replaceCase = [](const std::vector<int64_t>& absentUids) {
        return [absentUids](nlohmann::json&, nlohmann::json& sweep) {
            sweep["cases"][0]
                = fx::optionalTensorSweepCase(ABSENT_CASE_ID, supersetGraph(), absentUids);
        };
    };
    const auto metadataField = [](const std::string& key, const nlohmann::json& value) {
        return [key, value](nlohmann::json&, nlohmann::json& sweep) {
            sweep["cases"][0]["metadata"][key] = value;
        };
    };
    const auto caseField = [](const std::string& key, const nlohmann::json& value) {
        return [key, value](nlohmann::json&, nlohmann::json& sweep) {
            sweep["cases"][0][key] = value;
        };
    };

    return {
        {"no_metadata_block",
         [](nlohmann::json&, nlohmann::json& sweep) { sweep["cases"][0].erase("metadata"); },
         "absent_tensors needs case metadata with format_version 2"},
        {"null_metadata_block",
         caseField("metadata", nullptr),
         "absent_tensors needs case metadata with format_version 2"},
        {"format_version_1",
         metadataField("format_version", 1),
         "absent_tensors needs format_version 2"},
        {"sweep_version_1",
         [](nlohmann::json&, nlohmann::json& sweep) { sweep["version"] = 1; },
         "absent_tensors needs sweep version 2, not 1"},
        {"sweep_without_version",
         [](nlohmann::json&, nlohmann::json& sweep) { sweep.erase("version"); },
         "absent_tensors needs sweep version 2, not 1"},
        {"object_in_place_of_a_list",
         caseField("absent_tensors", nlohmann::json{{"uid", SINK}}),
         "absent_tensors must be a non-empty array of integer tensor uids"},
        {"empty_list",
         caseField("absent_tensors", nlohmann::json::array()),
         "absent_tensors must be a non-empty array of integer tensor uids"},
        {"string_uid",
         caseField("absent_tensors", nlohmann::json::array({"0"})),
         "absent_tensors must be a non-empty array of integer tensor uids"},
        {"duplicate_uid",
         caseField("absent_tensors", nlohmann::json::array({SINK, SINK})),
         "absent tensor 0 is named twice in absent_tensors"},
        {"unknown_uid",
         caseField("absent_tensors", nlohmann::json::array({9})),
         "absent tensor 9 is not a tensor of the template"},
        {"values_tensors_entry",
         [](nlohmann::json&, nlohmann::json& sweep) {
             sweep["cases"][0]["values"]["tensors"].push_back(
                 nlohmann::json{{"uid", SINK}, {"dims", {1, 2, 1, 1}}, {"strides", {2, 1, 1, 1}}});
         },
         "absent tensor 0 has a values.tensors entry"},
        {"case_value_keyed_by_uid",
         [](nlohmann::json&, nlohmann::json& sweep) {
             sweep["cases"][0]["values"]["alignment"] = nlohmann::json{{"0", 16}};
         },
         "absent tensor 0 is a key of the case value values.alignment.0"},
        {"tensor_patches_entry",
         caseField("tensor_patches",
                   nlohmann::json::array({{{"uid", SINK}, {"set", {{"alignment", 8}}}}})),
         "absent tensor 0 has a tensor_patches entry"},
        {"inputs_entry",
         metadataField("inputs", nlohmann::json{{"0", {{"seed", 3}}}}),
         "absent tensor 0 has a metadata inputs entry"},
        {"alias_view",
         metadataField("aliases",
                       nlohmann::json::array({{{"uid", SINK}, {"of", 1}, {"byte_offset", 0}}})),
         "absent tensor 0 is the view of a metadata aliases entry"},
        {"alias_owner",
         metadataField("aliases",
                       nlohmann::json::array({{{"uid", 2}, {"of", SINK}, {"byte_offset", 0}}})),
         "absent tensor 0 is the owner of a metadata aliases entry"},
        {"start_offsets_entry",
         metadataField("start_offsets",
                       nlohmann::json::array({{{"uid", SINK}, {"start_offset", 16}}})),
         "absent tensor 0 has a metadata start_offsets entry"},
        {"live_rows_output",
         metadataField("live_rows", nlohmann::json::array({{{"uid", SINK}, {"like", O_UID}}})),
         "absent tensor 0 is named by a metadata live_rows entry"},
        {"live_rows_lengths",
         metadataField("live_rows",
                       nlohmann::json::array({{{"uid", STATS}, {"axis", 2}, {"lengths", SINK}}})),
         "absent tensor 0 is named by a metadata live_rows entry"},
        {"live_rows_like",
         metadataField("live_rows", nlohmann::json::array({{{"uid", STATS}, {"like", SINK}}})),
         "absent tensor 0 is named by a metadata live_rows entry"},
        {"golden_declared",
         caseField("golden", nlohmann::json{{"path", "golden/absent_case/tensors.dvc"}}),
         "absent_tensors and golden data do not mix"},
        {"virtual_tensor",
         [](nlohmann::json& templateJson, nlohmann::json&) {
             for(auto& tensor : templateJson["tensors"])
             {
                 if(tensor["uid"] == SINK)
                 {
                     tensor["virtual"] = true;
                 }
             }
         },
         "absent tensor 0 is virtual"},
        {"virtual_placeholder",
         [](nlohmann::json& templateJson, nlohmann::json&) {
             for(auto& tensor : templateJson["tensors"])
             {
                 if(tensor["uid"] == SINK)
                 {
                     tensor["virtual"] = "${case.virtual}";
                 }
             }
         },
         "absent tensor 0 has a virtual flag that is not a literal boolean in the template"},
        {"port_placeholder",
         [](nlohmann::json& templateJson, nlohmann::json&) {
             templateJson["nodes"][0]["inputs"]["sink_token_tensor_uid"] = "${case.sink_uid}";
         },
         "absent_tensors needs a uid or null in each node port, and "
         "nodes[0].inputs.sink_token_tensor_uid holds \"${case.sink_uid}\""},
        {"float_ports_of_two_nodes",
         [](nlohmann::json& templateJson, nlohmann::json&) {
             auto second = templateJson["nodes"][0];
             second["name"] = "second";
             second["inputs"]["sink_token_tensor_uid"] = 0.0;
             templateJson["nodes"].push_back(std::move(second));
         },
         "absent tensor 0 is held by node ports of more than one node"},
        {"ports_of_two_nodes",
         [](nlohmann::json& templateJson, nlohmann::json&) {
             auto second = templateJson["nodes"][0];
             second["name"] = "second";
             templateJson["nodes"].push_back(std::move(second));
         },
         "absent tensor 0 is held by node ports of more than one node"},
        {"flat_uid_list",
         [](nlohmann::json& templateJson, nlohmann::json&) {
             templateJson["nodes"][0]["output_tensor_uids"] = nlohmann::json::array({SINK});
         },
         "absent tensor 0 is held by a node outside a node port, at "
         "nodes[0].output_tensor_uids[0]"},
        {"uid_field_outside_the_ports",
         [](nlohmann::json& templateJson, nlohmann::json&) {
             templateJson["nodes"][0]["attributes"]["extra_tensor_uid"] = SINK;
         },
         "absent tensor 0 is held by a node outside a node port, at "
         "nodes[0].attributes.extra_tensor_uid"},
        {"tensor_reference",
         [](nlohmann::json& templateJson, nlohmann::json&) {
             templateJson["tensors"][0]["ragged_offset_tensor_uid"] = SINK;
         },
         "absent tensor 0 is referenced at tensors[0].ragged_offset_tensor_uid"},
        {"required_port",
         replaceCase({V_UID}),
         "the graph with absent tensors [3] does not build; a removed tensor fills a port the "
         "op requires"},
        {"last_compared_output",
         [replaceCase](nlohmann::json& templateJson, nlohmann::json& sweep) {
             replaceCase({STATS})(templateJson, sweep);
             for(auto& tensor : templateJson["tensors"])
             {
                 if(tensor["uid"] == O_UID)
                 {
                     tensor["virtual"] = true;
                 }
             }
         },
         "removing [5] leaves no output to compare"},
        {"malformed_tensor_patch",
         caseField("tensor_patches", nlohmann::json::array({{{"uid", "zero"}}})),
         "tensor_patch entry is missing an integer uid"},
    };
}

// Writes the superset sweep under `root` with its one case `ABSENT_CASE_ID` marking the
// sink absent, after `edit` changes it.
void writeAbsentRuleSweep(const std::filesystem::path& root, const AbsentRuleCase& c)
{
    auto templateJson = fx::templateOfGraph(supersetGraph());
    nlohmann::json sweep
        = {{"version", 2},
           {"cases",
            nlohmann::json::array(
                {fx::optionalTensorSweepCase(ABSENT_CASE_ID, supersetGraph(), {SINK})})}};
    c.edit(templateJson, sweep);
    fx::writeTemplateSweep(root / "quick" / "SdpaFwd" / "Optional",
                           templateJson,
                           sweep["cases"],
                           sweep.contains("version") ? sweep["version"] : nlohmann::json());
}

} // namespace

TEST_F(TestBundleDiscoveryFixture, AbsentTensorRuleViolationsFailTheLoadOfASweepCase)
{
    for(const auto& c : absentRuleCases())
    {
        SCOPED_TRACE(c.name);
        const auto root = _tempDir / c.name;
        writeAbsentRuleSweep(root, c);

        const auto discovered = discoverBundles(root);
        ASSERT_EQ(discovered.size(), 1u);
        EXPECT_THROW(loadIntegrationTestBundle(discovered.front(), _sweeps), BundleMetadataError);

        const auto message = failedLoadMessage(root, _sweeps);
        EXPECT_NE(message.find(std::string("#") + ABSENT_CASE_ID), std::string::npos) << message;
        EXPECT_NE(message.find(c.message), std::string::npos) << message;
    }
}

TEST_F(TestBundleDiscoveryFixture, UnknownSweepVersionFailsTheLoadOfEveryCase)
{
    const auto graph = sdpa_utils::optionalSdpaFwdGraphJson(/*withStats=*/true, /*withSink=*/false);
    const auto cases = nlohmann::json::array({fx::optionalTensorSweepCase("case_a", graph),
                                              fx::optionalTensorSweepCase("case_b", graph)});

    const std::vector<std::pair<std::string, nlohmann::json>> versions
        = {{"integer_3", 3}, {"string_2", "2"}, {"null", nullptr}};
    for(const auto& [name, version] : versions)
    {
        SCOPED_TRACE(name);
        const auto root = _tempDir / name;
        const auto sweepDir = root / "quick" / "SdpaFwd" / "Plain";
        fx::writeTemplateSweep(sweepDir, fx::templateOfGraph(graph), cases, 2);
        auto sweepJson = readJsonFile(sweepDir / "sweep.json");
        sweepJson["version"] = version;
        writeJsonFile(sweepDir / "sweep.json", sweepJson);

        const auto discovered = discoverBundles(root);
        ASSERT_EQ(discovered.size(), 2u);
        for(const auto& bundle : discovered)
        {
            auto outcome = detail::classifyBundle(bundle, _sweeps);
            ASSERT_TRUE(std::holds_alternative<detail::FailedLoad>(outcome)) << bundle.testName;
            const auto& message = std::get<detail::FailedLoad>(outcome).message;
            std::string expected = "#";
            expected += bundle.testName;
            expected += ": sweep version ";
            expected += version.dump();
            EXPECT_NE(message.find(expected), std::string::npos) << message;
            EXPECT_NE(message.find("sweep.json is not 1 or 2"), std::string::npos) << message;
        }
    }
}

TEST_F(TestBundleDiscoveryFixture, SweepVersion2WithoutAbsentTensorsLoads)
{
    const auto graph = sdpa_utils::optionalSdpaFwdGraphJson(/*withStats=*/true, /*withSink=*/false);
    fx::writeTemplateSweep(_tempDir / "quick" / "SdpaFwd" / "Plain",
                           fx::templateOfGraph(graph),
                           nlohmann::json::array({fx::optionalTensorSweepCase("plain", graph)}),
                           2);

    const auto bundle = loadSweepCase(_tempDir, "plain", _sweeps);
    ASSERT_TRUE(bundle.has_value());
    EXPECT_EQ(bundle->outputTensorUids, (std::vector<int64_t>{O_UID, STATS}));
}

// One superset template (Q, K, V, O, Stats and a sink, every tensor with placeholder dims
// and strides) gives the same graphs as one template per tensor set.
TEST_F(TestBundleDiscoveryFixture, AbsentTensorCasesMatchTheirOwnTemplates)
{
    const auto superset = supersetGraph();
    fx::writeTemplateSweep(
        _tempDir / "superset" / "quick" / "SdpaFwd" / "Superset",
        fx::templateOfGraph(superset),
        nlohmann::json::array({fx::optionalTensorSweepCase("all", superset),
                               fx::optionalTensorSweepCase("sink_absent", superset, {SINK}),
                               fx::optionalTensorSweepCase("stats_absent", superset, {STATS})}));

    struct Expected
    {
        std::string caseId;
        nlohmann::json graph;
        int64_t absentUid;
    };
    const std::vector<Expected> expected = {
        {"all", sdpa_utils::optionalSdpaFwdGraphJson(true, true), -1},
        {"sink_absent", sdpa_utils::optionalSdpaFwdGraphJson(true, false), SINK},
        {"stats_absent", sdpa_utils::optionalSdpaFwdGraphJson(false, true), STATS},
    };

    const auto supersetRoot = _tempDir / "superset";
    ASSERT_EQ(discoverBundles(supersetRoot).size(), 3u);
    for(const auto& e : expected)
    {
        SCOPED_TRACE(e.caseId);
        const auto separateRoot = _tempDir / ("separate_" + e.caseId);
        fx::writeTemplateSweep(
            separateRoot / "quick" / "SdpaFwd" / "Separate",
            fx::templateOfGraph(e.graph),
            nlohmann::json::array({fx::optionalTensorSweepCase(e.caseId, e.graph)}),
            1);

        const auto fromSuperset = loadSweepCase(supersetRoot, e.caseId, _sweeps);
        const auto fromSeparate = loadSweepCase(separateRoot, e.caseId, _sweeps);
        ASSERT_TRUE(fromSuperset.has_value());
        ASSERT_TRUE(fromSeparate.has_value());

        EXPECT_EQ(graphJsonOf(*fromSuperset), graphJsonOf(*fromSeparate));
        EXPECT_EQ(fromSuperset->outputTensorUids, fromSeparate->outputTensorUids);

        const auto packUids = fx::hostVariantPackUids(*fromSuperset);
        EXPECT_EQ(packUids, fx::hostVariantPackUids(*fromSeparate));
        if(e.absentUid >= 0)
        {
            EXPECT_EQ(packUids.count(e.absentUid), 0u);
            EXPECT_EQ(fromSuperset->graphWrapper().getTensorMap().count(e.absentUid), 0u);
        }
    }
}

TEST_F(TestBundleDiscoveryFixture, AbsentInputAndAbsentOptionalOutputLoad)
{
    const auto superset = supersetGraph();
    fx::writeTemplateSweep(
        _tempDir / "quick" / "SdpaFwd" / "Superset",
        fx::templateOfGraph(superset),
        nlohmann::json::array({fx::optionalTensorSweepCase("sink_absent", superset, {SINK}),
                               fx::optionalTensorSweepCase("stats_absent", superset, {STATS})}));

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 2u);
    for(const auto& bundle : discovered)
    {
        EXPECT_TRUE(
            std::holds_alternative<detail::LoadedBundle>(detail::classifyBundle(bundle, _sweeps)))
            << bundle.testName;
    }

    const auto sinkAbsent = loadSweepCase(_tempDir, "sink_absent", _sweeps);
    ASSERT_TRUE(sinkAbsent.has_value());
    const auto sinkJson = graphJsonOf(*sinkAbsent);
    EXPECT_TRUE(sinkJson["nodes"][0]["inputs"]["sink_token_tensor_uid"].is_null());
    EXPECT_EQ(sinkAbsent->metadata.formatVersion, 2);
    EXPECT_EQ(sinkAbsent->outputTensorUids, (std::vector<int64_t>{O_UID, STATS}));

    const auto statsAbsent = loadSweepCase(_tempDir, "stats_absent", _sweeps);
    ASSERT_TRUE(statsAbsent.has_value());
    EXPECT_TRUE(graphJsonOf(*statsAbsent)["nodes"][0]["outputs"]["stats_tensor_uid"].is_null());
    EXPECT_EQ(statsAbsent->outputTensorUids, (std::vector<int64_t>{O_UID}));
}

// The absent sink has no values entry, so expanding the template before pruning it asks
// for the sink's dims; the case would then be a quiet skip, not a loaded case.
TEST_F(TestBundleDiscoveryFixture, AbsentTensorsArePrunedBeforeTheTemplateIsExpanded)
{
    const auto superset = supersetGraph();
    const auto templateJson = fx::templateOfGraph(superset);
    const auto caseJson = fx::optionalTensorSweepCase("sink_absent", superset, {SINK});
    const auto sweepDir = _tempDir / "quick" / "SdpaFwd" / "Superset";
    fx::writeTemplateSweep(sweepDir, templateJson, nlohmann::json::array({caseJson}));

    const auto discovered = discoverBundles(_tempDir);
    ASSERT_EQ(discovered.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(
        loadIntegrationTestBundle(discovered.front(), _sweeps)));

    try
    {
        (void)detail::expandTemplateGraph(templateJson, caseJson, discovered.front());
        ADD_FAILURE() << "expanding the unpruned template did not throw";
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_STREQ(e.what(), "Missing per-tensor placeholder value");
    }
    EXPECT_NO_THROW((void)detail::expandTemplateGraph(
        detail::pruneAbsentTensors(templateJson, {SINK}), caseJson, discovered.front()));

    auto withoutAbsentKey = caseJson;
    withoutAbsentKey.erase("absent_tensors");
    fx::writeTemplateSweep(sweepDir, templateJson, nlohmann::json::array({withoutAbsentKey}));
    SweepManifestCache freshSweeps;
    EXPECT_TRUE(std::holds_alternative<detail::SkippedLoad>(
        detail::classifyBundle(discoverBundles(_tempDir).front(), freshSweeps)));
}

TEST_F(TestBundleDiscoveryFixture, AbsentTensorHeldAsAFloatInAPortIsPruned)
{
    const auto superset = supersetGraph();
    auto templateJson = fx::templateOfGraph(superset);
    templateJson["nodes"][0]["inputs"]["sink_token_tensor_uid"] = 0.0;
    fx::writeTemplateSweep(
        _tempDir / "quick" / "SdpaFwd" / "Superset",
        templateJson,
        nlohmann::json::array({fx::optionalTensorSweepCase("sink_absent", superset, {SINK})}));

    const auto sinkAbsent = loadSweepCase(_tempDir, "sink_absent", _sweeps);
    ASSERT_TRUE(sinkAbsent.has_value());
    EXPECT_TRUE(graphJsonOf(*sinkAbsent)["nodes"][0]["inputs"]["sink_token_tensor_uid"].is_null());
}

TEST(TestPruneAbsentTensors, RemovesTheTensorsAndNullsTheirPortsOnly)
{
    auto graph = fx::templateOfGraph(supersetGraph());
    graph["nodes"][0]["output_tensor_uids"] = nlohmann::json::array({SINK});
    graph["nodes"][0]["attributes"]["left_bound"] = SINK;

    auto expected = graph;
    auto keptTensors = nlohmann::json::array();
    for(const auto& tensor : graph.at("tensors"))
    {
        if(tensor.at("uid") != SINK && tensor.at("uid") != STATS)
        {
            keptTensors.push_back(tensor);
        }
    }
    expected["tensors"] = keptTensors;
    expected["nodes"][0]["inputs"]["sink_token_tensor_uid"] = nullptr;
    expected["nodes"][0]["outputs"]["stats_tensor_uid"] = nullptr;

    EXPECT_EQ(detail::pruneAbsentTensors(graph, {SINK, STATS}), expected);
    EXPECT_EQ(detail::pruneAbsentTensors(fx::templateOfGraph(supersetGraph()), {SINK}),
              fx::templateOfGraph(sdpa_utils::optionalSdpaFwdGraphJson(true, false)));
}

// NOLINTEND(readability-identifier-naming)
