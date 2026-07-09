// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/autotune/AutotuneFileWriter.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <set>
#endif

using namespace hipdnn_frontend;

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
using namespace hipdnn_frontend::autotune;
using namespace hipdnn_frontend::autotune::detail;
using namespace hipdnn_data_sdk::utilities;

// --- Test helpers ---

namespace
{
namespace config_criterion = hipdnn_data_sdk::detail::autotune_config::criterion;
namespace config_json = hipdnn_data_sdk::detail::autotune_config::json;
namespace config_op = hipdnn_data_sdk::detail::autotune_config::op;
namespace config_tensor = hipdnn_data_sdk::detail::autotune_config::tensor;
namespace config_version = hipdnn_data_sdk::detail::autotune_config::version;

inline std::filesystem::path makeUniqueTempDir()
{
    static std::atomic<int> s_counter{0};
    const auto unique = std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_"
                        + std::to_string(s_counter++);
    return std::filesystem::temp_directory_path() / ("hipdnn_test_" + unique);
}

struct TempFile
{
    hipdnn_test_sdk::utilities::ScopedDirectory dir;
    std::filesystem::path path;

    TempFile()
        : dir(makeUniqueTempDir())
        , path(dir.path() / "config.json")
    {
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

/// Create a simple AutotuneResult for testing
AutotuneResult makeResult(int64_t engineId,
                          const std::string& engineName,
                          float minTime = 1.0f,
                          bool succeeded = true,
                          int rank = 0)
{
    AutotuneResult r;
    r.engineId = engineId;
    r.engineName = engineName;
    r.minTimeMs = minTime;
    r.avgTimeMs = minTime + 0.5f;
    r.stddevMs = 0.1f;
    r.iterationsRun = 10;
    r.succeeded = succeeded;
    r.modeUsed = TuneMode::STANDARD;
    r.converged = true;
    r.workspaceSize = 1024;
    r.rank = rank;
    return r;
}
std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void writeTextFile(const std::filesystem::path& path, const std::string& contents)
{
    std::ofstream file(path);
    file << contents;
}

std::string writeJsonFile(const std::filesystem::path& path, const nlohmann::json& json)
{
    const auto contents = json.dump(2) + '\n';
    writeTextFile(path, contents);
    return contents;
}

// Collect all "<base>.corrupt-*" sibling files currently present next to `base`.
std::set<std::filesystem::path> collectCorruptSiblings(const std::filesystem::path& base)
{
    std::set<std::filesystem::path> out;
    const auto dir = base.parent_path();
    const auto prefix = base.filename().string() + ".corrupt-";
    std::error_code ec;
    if(!std::filesystem::exists(dir, ec))
    {
        return out;
    }
    for(const auto& entry : std::filesystem::directory_iterator(dir, ec))
    {
        if(entry.path().filename().string().rfind(prefix, 0) == 0)
        {
            out.insert(entry.path());
        }
    }
    return out;
}

nlohmann::json makeExistingVersionedRoot(int64_t version = config_version::CURRENT)
{
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    const std::vector<std::string> tensorIds = {config_tensor::X, config_tensor::W};
    auto oldEntry = hipdnn_frontend::autotune::detail::buildOverrideEntry(
        makeResult(1, "OLD"), config_op::CONV_FPROP, dims, {}, {}, tensorIds);

    nlohmann::json root;
    root[config_json::VERSION] = version;
    root[config_json::ENGINE_OVERRIDES] = nlohmann::json::array({oldEntry});
    return root;
}

std::vector<std::string> tensorIdsFor(std::string_view opName, size_t tensorCount)
{
    std::vector<std::string> ids;
    if(opName == config_op::CONV_DGRAD)
    {
        ids = {config_tensor::DY, config_tensor::W};
    }
    else if(opName == config_op::POINTWISE)
    {
        ids = {config_tensor::IN_0, config_tensor::IN_1, config_tensor::IN_2};
    }
    else
    {
        ids = {config_tensor::X, config_tensor::W, config_tensor::DY};
    }
    ids.resize(tensorCount);
    return ids;
}

Error writeVersionedAutotuneResults(const std::filesystem::path& filePath,
                                    const std::string& opName,
                                    const std::vector<AutotuneResult>& results,
                                    bool deleteAllExisting,
                                    const std::vector<std::vector<int64_t>>& tensorDims,
                                    const std::vector<std::vector<int64_t>>& tensorStrides,
                                    const Criteria& criteria = {})
{
    return hipdnn_frontend::autotune::detail::writeAutotuneResults(
        filePath,
        opName,
        results,
        deleteAllExisting,
        tensorDims,
        tensorStrides,
        criteria,
        tensorIdsFor(opName, tensorDims.size()));
}

Error writeReplacementConvFprop(const std::filesystem::path& path)
{
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(2, "NEW", 0.5f, true, 0));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    return writeVersionedAutotuneResults(path, config_op::CONV_FPROP, results, false, dims, {});
}

void expectReplacementRejectedAndUnchanged(const nlohmann::json& root,
                                           const std::string& expectedMessageSubstring)
{
    const TempFile tmpFile;
    const auto originalContents = writeJsonFile(tmpFile.path, root);
    const auto err = writeReplacementConvFprop(tmpFile.path);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE) << err.get_message();
    EXPECT_NE(err.get_message().find(expectedMessageSubstring), std::string::npos)
        << err.get_message();
    EXPECT_EQ(readTextFile(tmpFile.path), originalContents);
}

} // namespace

// --- KnobSetting to_json Tests ---

TEST(TestAutotuneFileWriter, KnobSettingToJsonInt)
{
    const KnobSetting setting("TILE_SIZE", int64_t{128});
    const nlohmann::json json(setting);

    EXPECT_EQ(json["knob_id"], "TILE_SIZE");
    EXPECT_EQ(json["type"], "int");
    EXPECT_EQ(json["value"], 128);
}

TEST(TestAutotuneFileWriter, KnobSettingToJsonDouble)
{
    const KnobSetting setting("LEARNING_RATE", 0.001);
    const nlohmann::json json(setting);

    EXPECT_EQ(json["knob_id"], "LEARNING_RATE");
    EXPECT_EQ(json["type"], "double");
    EXPECT_DOUBLE_EQ(json["value"].get<double>(), 0.001);
}

TEST(TestAutotuneFileWriter, KnobSettingToJsonString)
{
    const KnobSetting setting("ALGORITHM", std::string("gemm_v2"));
    const nlohmann::json json(setting);

    EXPECT_EQ(json["knob_id"], "ALGORITHM");
    EXPECT_EQ(json["type"], "string");
    EXPECT_EQ(json["value"], "gemm_v2");
}

TEST(TestAutotuneFileWriter, KnobSettingToJsonYieldsExactKeys)
{
    const KnobSetting setting("SPLIT_K", int64_t{4});
    const nlohmann::json json(setting);

    ASSERT_TRUE(json.is_object());
    EXPECT_EQ(json.size(), 3u);
    EXPECT_TRUE(json.contains("knob_id"));
    EXPECT_TRUE(json.contains("type"));
    EXPECT_TRUE(json.contains("value"));
}

// --- buildOverrideEntry Tests ---

TEST(TestAutotuneFileWriter, BuildOverrideEntryBasic)
{
    auto result = makeResult(1, "MIOPEN_ENGINE");
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    const std::vector<std::vector<int64_t>> tensorStrides
        = {{150528, 50176, 224, 1}, {147, 49, 7, 1}};

    auto entry = buildOverrideEntry(result, config_op::CONV_FPROP, tensorDims, tensorStrides);

    EXPECT_EQ(entry[config_json::OP], config_op::CONV_FPROP);
    EXPECT_EQ(entry[config_json::ENGINE_NAME], "MIOPEN_ENGINE");
    ASSERT_EQ(entry[config_json::TENSORS].size(), 2u);
    EXPECT_EQ(entry[config_json::TENSORS][0][config_json::DIM],
              std::vector<int64_t>({1, 3, 224, 224}));
    EXPECT_EQ(entry[config_json::TENSORS][1][config_json::DIM],
              std::vector<int64_t>({64, 3, 7, 7}));
    EXPECT_EQ(entry[config_json::TENSORS][0][config_json::STRIDE],
              std::vector<int64_t>({150528, 50176, 224, 1}));
    EXPECT_EQ(entry[config_json::TENSORS][1][config_json::STRIDE],
              std::vector<int64_t>({147, 49, 7, 1}));
    ASSERT_TRUE(entry.contains("autotune_metadata"));
    EXPECT_FALSE(entry["autotune_metadata"].contains("knobs")); // No knobs -> field absent
}

TEST(TestAutotuneFileWriter, BuildOverrideEntryWritesTensorIds)
{
    auto result = makeResult(1, "MIOPEN_ENGINE");
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    const std::vector<std::vector<int64_t>> tensorStrides
        = {{150528, 50176, 224, 1}, {147, 49, 7, 1}};
    const std::vector<std::string> tensorIds = {config_tensor::X, config_tensor::W};

    auto entry = hipdnn_frontend::autotune::detail::buildOverrideEntry(
        result, config_op::CONV_FPROP, tensorDims, tensorStrides, {}, tensorIds);

    ASSERT_EQ(entry[config_json::TENSORS].size(), 2u);
    EXPECT_EQ(entry[config_json::TENSORS][0][config_json::TENSOR_ID], config_tensor::X);
    EXPECT_EQ(entry[config_json::TENSORS][1][config_json::TENSOR_ID], config_tensor::W);
    EXPECT_EQ(entry[config_json::TENSORS][0][config_json::DIM],
              std::vector<int64_t>({1, 3, 224, 224}));
    EXPECT_EQ(entry[config_json::TENSORS][1][config_json::STRIDE],
              std::vector<int64_t>({147, 49, 7, 1}));
}

TEST(TestAutotuneFileWriter, NamedEntryRejectsLegacyEntryWithSamePositionalSignature)
{
    const TempFile tmpFile;
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    const std::vector<std::string> tensorIds = {config_tensor::X, config_tensor::W};

    nlohmann::json root;
    root[config_json::ENGINE_OVERRIDES] = nlohmann::json::array(
        {buildOverrideEntry(makeResult(1, "OLD"), config_op::CONV_FPROP, dims, {})});
    const auto originalContents = writeJsonFile(tmpFile.path, root);

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(2, "NEW", 0.5f, true, 0));
    auto err = hipdnn_frontend::autotune::detail::writeAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, dims, {}, {}, tensorIds);
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE) << err.get_message();
    EXPECT_NE(err.get_message().find("refusing to update legacy autotune config file"),
              std::string::npos)
        << err.get_message();
    EXPECT_EQ(readTextFile(tmpFile.path), originalContents);
}

TEST(TestAutotuneFileWriter, CurrentVersionExistingFileUpdatesSuccessfully)
{
    const TempFile tmpFile;
    writeJsonFile(tmpFile.path, makeExistingVersionedRoot());

    auto err = writeReplacementConvFprop(tmpFile.path);
    ASSERT_TRUE(err.is_good()) << err.get_message();

    std::ifstream file(tmpFile.path);
    const auto json = nlohmann::json::parse(file);
    ASSERT_EQ(json[config_json::ENGINE_OVERRIDES].size(), 1u);
    EXPECT_EQ(json[config_json::VERSION], config_version::CURRENT);
    EXPECT_EQ(json[config_json::ENGINE_OVERRIDES][0][config_json::ENGINE_NAME], "NEW");
}

TEST(TestAutotuneFileWriter, MissingVersionFileRejectsAndRemainsUnchanged)
{
    auto root = makeExistingVersionedRoot();
    root.erase(config_json::VERSION);
    expectReplacementRejectedAndUnchanged(root, "refusing to update legacy autotune config file");
}

TEST(TestAutotuneFileWriter, OlderVersionFileRejectsAndRemainsUnchanged)
{
    expectReplacementRejectedAndUnchanged(makeExistingVersionedRoot(config_version::CURRENT - 1),
                                          "refusing to update non-current autotune config file");
}

TEST(TestAutotuneFileWriter, NewerVersionFileRejectsAndRemainsUnchanged)
{
    expectReplacementRejectedAndUnchanged(makeExistingVersionedRoot(config_version::CURRENT + 1),
                                          "refusing to update non-current autotune config file");
}

TEST(TestAutotuneFileWriter, WrongTypeVersionFileRejectsAndRemainsUnchanged)
{
    auto root = makeExistingVersionedRoot();
    root[config_json::VERSION] = "2";
    expectReplacementRejectedAndUnchanged(root, "existing config version is not an integer");
}

TEST(TestAutotuneFileWriter, DuplicateTensorIdsRejectNewFileWrite)
{
    const TempFile tmpFile;
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(2, "NEW", 0.5f, true, 0));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    const std::vector<std::string> tensorIds = {config_tensor::X, config_tensor::X};

    auto err = hipdnn_frontend::autotune::detail::writeAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, dims, {}, {}, tensorIds);

    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE) << err.get_message();
    EXPECT_NE(err.get_message().find("tensor IDs must be unique"), std::string::npos)
        << err.get_message();
    EXPECT_FALSE(std::filesystem::exists(tmpFile.path));
}

TEST(TestAutotuneFileWriter, DuplicateTensorIdsRejectAndExistingFileRemainsUnchanged)
{
    const TempFile tmpFile;
    const auto originalContents = writeJsonFile(tmpFile.path, makeExistingVersionedRoot());
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(2, "NEW", 0.5f, true, 0));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    const std::vector<std::string> tensorIds = {config_tensor::X, config_tensor::X};

    auto err = hipdnn_frontend::autotune::detail::writeAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, dims, {}, {}, tensorIds);

    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE) << err.get_message();
    EXPECT_NE(err.get_message().find("tensor IDs must be unique"), std::string::npos)
        << err.get_message();
    EXPECT_EQ(readTextFile(tmpFile.path), originalContents);
}

TEST(TestAutotuneFileWriter, NamedEntryReplacesExistingEntryWithReorderedNamedTensors)
{
    const TempFile tmpFile;
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    const std::vector<std::string> tensorIds = {config_tensor::X, config_tensor::W};

    auto oldEntry = hipdnn_frontend::autotune::detail::buildOverrideEntry(
        makeResult(1, "OLD"), config_op::CONV_FPROP, dims, {}, {}, tensorIds);
    std::reverse(oldEntry[config_json::TENSORS].begin(), oldEntry[config_json::TENSORS].end());

    nlohmann::json root;
    root[config_json::VERSION] = config_version::CURRENT;
    root[config_json::ENGINE_OVERRIDES] = nlohmann::json::array({oldEntry});
    writeJsonFile(tmpFile.path, root);

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(2, "NEW", 0.5f, true, 0));
    auto err = hipdnn_frontend::autotune::detail::writeAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, dims, {}, {}, tensorIds);
    ASSERT_TRUE(err.is_good()) << err.get_message();

    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);
    ASSERT_EQ(json[config_json::ENGINE_OVERRIDES].size(), 1u);
    EXPECT_EQ(json[config_json::VERSION], config_version::CURRENT);
    EXPECT_EQ(json[config_json::ENGINE_OVERRIDES][0][config_json::ENGINE_NAME], "NEW");
    EXPECT_EQ(
        json[config_json::ENGINE_OVERRIDES][0][config_json::TENSORS][0][config_json::TENSOR_ID],
        config_tensor::X);
    EXPECT_EQ(
        json[config_json::ENGINE_OVERRIDES][0][config_json::TENSORS][1][config_json::TENSOR_ID],
        config_tensor::W);
}

TEST(TestAutotuneFileWriter, BuildOverrideEntryWithCriteria)
{
    auto result = makeResult(1, "MIOPEN_ENGINE");
    const std::vector<std::vector<int64_t>> tensorDims = {{2, 4, 16, 16}};
    const Criteria criteria = {{config_criterion::POINTWISE_MODE, 34}};

    auto entry = hipdnn_frontend::autotune::detail::buildOverrideEntry(
        result, config_op::POINTWISE, tensorDims, {}, criteria);

    ASSERT_TRUE(entry.contains(config_json::CRITERIA));
    EXPECT_EQ(entry[config_json::CRITERIA][config_criterion::POINTWISE_MODE], 34);
}

TEST(TestAutotuneFileWriter, BuildOverrideEntryWithKnobs)
{
    auto result = makeResult(1, "MIOPEN_ENGINE");
    result.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    result.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {{150528, 50176, 224, 1}};

    auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);

    ASSERT_TRUE(entry.contains("autotune_metadata"));
    ASSERT_TRUE(entry["autotune_metadata"].contains("knobs"));
    ASSERT_EQ(entry["autotune_metadata"]["knobs"].size(), 2u);

    EXPECT_EQ(entry["autotune_metadata"]["knobs"][0]["knob_id"], "TILE_SIZE");
    EXPECT_EQ(entry["autotune_metadata"]["knobs"][0]["value"], 128);
    EXPECT_EQ(entry["autotune_metadata"]["knobs"][1]["knob_id"], "SPLIT_K");
    EXPECT_EQ(entry["autotune_metadata"]["knobs"][1]["value"], 2);
}

TEST(TestAutotuneFileWriter, BuildOverrideEntryWithMetadata)
{
    auto result = makeResult(1, "MIOPEN_ENGINE", 1.5f, true, 0);
    result.modeUsed = TuneMode::EXHAUSTIVE;
    result.strategyUsed = AutotuneStrategy::FIXED_AVERAGE;
    result.ranExhaustive = true;
    result.iterationsRun = 20;

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {{150528, 50176, 224, 1}};

    auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);

    ASSERT_TRUE(entry.contains("autotune_metadata"));
    auto& meta = entry["autotune_metadata"];
    EXPECT_FLOAT_EQ(meta["min_time_ms"].get<float>(), 1.5f);
    EXPECT_EQ(meta["iterations_run"], 20);
    EXPECT_EQ(meta["mode"], "exhaustive");
    EXPECT_EQ(meta["strategy"], "fixed_average");
    EXPECT_EQ(meta["rank"], 0);
    EXPECT_FLOAT_EQ(meta["avg_time_ms"].get<float>(), 2.0f);
    EXPECT_FLOAT_EQ(meta["stddev_ms"].get<float>(), 0.1f);
    EXPECT_EQ(meta["workspace_size"], 1024);
    EXPECT_TRUE(meta.contains("timestamp"));
    // Timestamp should be ISO 8601 format (basic check for 'T' and 'Z')
    auto ts = meta["timestamp"].get<std::string>();
    EXPECT_NE(ts.find('T'), std::string::npos);
    EXPECT_NE(ts.find('Z'), std::string::npos);
    EXPECT_TRUE(meta["ran_exhaustive"].get<bool>());
}
// --- writeAutotuneResults Tests ---

TEST(TestAutotuneFileWriter, WriteToNewFile)
{
    const TempFile tmpFile;

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}, {64, 3, 7, 7}};

    auto err = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, tensorDims, {});
    ASSERT_TRUE(err.is_good()) << err.get_message();

    // Verify file exists and is valid JSON
    ASSERT_TRUE(std::filesystem::exists(tmpFile.path));
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    ASSERT_TRUE(json.contains(config_json::ENGINE_OVERRIDES));
    EXPECT_EQ(json[config_json::VERSION], config_version::CURRENT);
    EXPECT_EQ(json[config_json::ENGINE_OVERRIDES].size(), 1u);
    EXPECT_EQ(json[config_json::ENGINE_OVERRIDES][0][config_json::ENGINE_NAME], "MIOPEN_ENGINE");
}

TEST(TestAutotuneFileWriter, WriteRejectsLegacyExistingFile)
{
    const TempFile tmpFile;

    const std::string originalContents
        = R"({"engine_overrides":{"not":"an array"},"preserved":true})";
    writeTextFile(tmpFile.path, originalContents);

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};

    auto err = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, tensorDims, {});
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE) << err.get_message();
    EXPECT_NE(err.get_message().find("refusing to update legacy autotune config file"),
              std::string::npos)
        << err.get_message();
    EXPECT_EQ(readTextFile(tmpFile.path), originalContents);
}

TEST(TestAutotuneFileWriter, WriteSkipsFailedResults)
{
    const TempFile tmpFile;

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));
    results.push_back(makeResult(2, "HIPBLASLT_ENGINE", 0.0f, false, -1));
    results.push_back(makeResult(3, "FUSILLI_ENGINE", 3.0f, true, 1));

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};

    auto err = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, tensorDims, {});
    ASSERT_TRUE(err.is_good());

    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    // Only the rank-0 winner (first succeeded result) should be written
    EXPECT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "MIOPEN_ENGINE");
}

TEST(TestAutotuneFileWriter, AppendToExistingFile)
{
    const TempFile tmpFile;

    // Write initial results for conv_fprop
    std::vector<AutotuneResult> results1;
    results1.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));

    const std::vector<std::vector<int64_t>> dims1 = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    auto err1 = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results1, false, dims1, {});
    ASSERT_TRUE(err1.is_good());

    // Write new results for conv_dgrad (different op)
    std::vector<AutotuneResult> results2;
    results2.push_back(makeResult(2, "HIPBLASLT_ENGINE", 2.0f, true, 0));

    const std::vector<std::vector<int64_t>> dims2 = {{8, 64, 56, 56}};
    auto err2 = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_DGRAD, results2, false, dims2, {});
    ASSERT_TRUE(err2.is_good());

    // Both entries should be in the file
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 2u);
    EXPECT_EQ(json["engine_overrides"][0]["op"], "conv_fprop");
    EXPECT_EQ(json["engine_overrides"][1]["op"], "conv_dgrad");
}

TEST(TestAutotuneFileWriter, ReplaceMatchingEntryWithSameKnobs)
{
    const TempFile tmpFile;

    // Write initial result with no knobs
    std::vector<AutotuneResult> results1;
    results1.push_back(makeResult(1, "MIOPEN_ENGINE", 5.0f, true, 0));

    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    auto err1 = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results1, false, dims, {});
    ASSERT_TRUE(err1.is_good());

    // Write updated result for same op + tensors + same (empty) knobs
    std::vector<AutotuneResult> results2;
    results2.push_back(makeResult(2, "HIPBLASLT_ENGINE", 1.0f, true, 0));

    auto err2 = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results2, false, dims, {});
    ASSERT_TRUE(err2.is_good());

    // Should have replaced the matching entry (same op + same tensors + same knobs)
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "HIPBLASLT_ENGINE");
}

TEST(TestAutotuneFileWriter, ReplaceEntriesWithDifferentKnobs)
{
    const TempFile tmpFile;

    // Write initial result with SPLIT_K=2
    std::vector<AutotuneResult> results1;
    auto r1 = makeResult(1, "MIOPEN_ENGINE", 5.0f, true, 0);
    r1.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    results1.push_back(r1);

    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    auto err1 = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results1, false, dims, {});
    ASSERT_TRUE(err1.is_good());

    // Write new result for same op + tensors but DIFFERENT knobs (SPLIT_K=4)
    std::vector<AutotuneResult> results2;
    auto r2 = makeResult(2, "HIPBLASLT_ENGINE", 1.0f, true, 0);
    r2.knobSettings.emplace_back("SPLIT_K", int64_t{4});
    results2.push_back(r2);

    auto err2 = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results2, false, dims, {});
    ASSERT_TRUE(err2.is_good());

    // The old entry should be replaced (matching by op + tensors only)
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "HIPBLASLT_ENGINE");
}

TEST(TestAutotuneFileWriter, CriteriaDifferentiatesSameOperationAndTensors)
{
    const TempFile tmpFile;
    const std::vector<std::vector<int64_t>> dims = {{2, 4, 16, 16}, {2, 4, 16, 16}};

    std::vector<AutotuneResult> addResults;
    addResults.push_back(makeResult(1, "ADD_ENGINE", 1.0f, true, 0));
    auto err = hipdnn_frontend::autotune::detail::writeAutotuneResults(
        tmpFile.path.string(),
        config_op::POINTWISE,
        addResults,
        false,
        dims,
        {},
        {{config_criterion::POINTWISE_MODE, 2}},
        tensorIdsFor(config_op::POINTWISE, dims.size()));
    ASSERT_TRUE(err.is_good()) << err.get_message();

    std::vector<AutotuneResult> mulResults;
    mulResults.push_back(makeResult(2, "MUL_ENGINE", 1.0f, true, 0));
    err = hipdnn_frontend::autotune::detail::writeAutotuneResults(
        tmpFile.path.string(),
        config_op::POINTWISE,
        mulResults,
        false,
        dims,
        {},
        {{config_criterion::POINTWISE_MODE, 30}},
        tensorIdsFor(config_op::POINTWISE, dims.size()));
    ASSERT_TRUE(err.is_good()) << err.get_message();

    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);
    ASSERT_EQ(json["engine_overrides"].size(), 2u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "ADD_ENGINE");
    EXPECT_EQ(json["engine_overrides"][0]["criteria"]["pointwise_mode"], 2);
    EXPECT_EQ(json["engine_overrides"][1]["engine_name"], "MUL_ENGINE");
    EXPECT_EQ(json["engine_overrides"][1]["criteria"]["pointwise_mode"], 30);
}

TEST(TestAutotuneFileWriter, ReplaceOnlyExactOperationAndTensorSignature)
{
    const TempFile tmpFile;
    const std::vector<std::vector<int64_t>> matchingDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> otherDims = {{2, 3, 224, 224}};

    nlohmann::json root;
    const std::vector<std::string> tensorIds = {config_tensor::X};
    root[config_json::VERSION] = config_version::CURRENT;
    root[config_json::ENGINE_OVERRIDES] = nlohmann::json::array(
        {{{config_json::ENGINE_NAME, "MISSING_OP"},
          {config_json::TENSORS, nlohmann::json::array()}},
         {{config_json::OP, config_op::CONV_FPROP}, {config_json::ENGINE_NAME, "MISSING_TENSORS"}},
         hipdnn_frontend::autotune::detail::buildOverrideEntry(
             makeResult(7, "OTHER_TENSORS"), config_op::CONV_FPROP, otherDims, {}, {}, tensorIds),
         hipdnn_frontend::autotune::detail::buildOverrideEntry(
             makeResult(8, "OLD_MATCH"), config_op::CONV_FPROP, matchingDims, {}, {}, tensorIds)});
    {
        std::ofstream file(tmpFile.path);
        file << root.dump(2) << '\n';
    }

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(9, "NEW_MATCH", 0.5f, true, 0));
    auto err = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, matchingDims, {});
    ASSERT_TRUE(err.is_good()) << err.get_message();

    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);
    const auto& overrides = json["engine_overrides"];
    ASSERT_EQ(overrides.size(), 4u);
    EXPECT_EQ(overrides[0]["engine_name"], "MISSING_OP");
    EXPECT_EQ(overrides[1]["engine_name"], "MISSING_TENSORS");
    EXPECT_EQ(overrides[2]["engine_name"], "OTHER_TENSORS");
    EXPECT_EQ(overrides[3]["engine_name"], "NEW_MATCH");
}

TEST(TestAutotuneFileWriter, DeleteAllExistingContent)
{
    const TempFile tmpFile;

    // Write initial results
    std::vector<AutotuneResult> results1;
    results1.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));
    const std::vector<std::vector<int64_t>> dims1 = {{1, 3, 224, 224}};
    auto err1 = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results1, false, dims1, {});
    ASSERT_TRUE(err1.is_good());

    // Write new results with deleteAllExisting=true
    std::vector<AutotuneResult> results2;
    results2.push_back(makeResult(2, "HIPBLASLT_ENGINE", 2.0f, true, 0));
    const std::vector<std::vector<int64_t>> dims2 = {{8, 64, 56, 56}};
    auto err2 = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_DGRAD, results2, true, dims2, {});
    ASSERT_TRUE(err2.is_good());

    // Only the new results should be in the file
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["op"], "conv_dgrad");
}

// --- Round-trip Tests ---

TEST(TestAutotuneFileWriter, RoundTripWriteThenLoad)
{
    const TempFile tmpFile;

    // Write results with knobs
    std::vector<AutotuneResult> results;
    auto result = makeResult(MIOPEN_ENGINE_ID, "MIOPEN_ENGINE", 1.0f, true, 0);
    result.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    result.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    results.push_back(result);

    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    auto err = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, true, dims, {});
    ASSERT_TRUE(err.is_good());

    // Verify by parsing the JSON directly
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    ASSERT_TRUE(json.contains("engine_overrides"));
    ASSERT_EQ(json["engine_overrides"].size(), 1u);

    const auto& entry = json["engine_overrides"][0];
    EXPECT_EQ(entry["op"], "conv_fprop");
    EXPECT_EQ(entry["engine_name"], "MIOPEN_ENGINE");

    // Verify tensors
    ASSERT_EQ(entry["tensors"].size(), 2u);
    EXPECT_EQ(entry["tensors"][0]["dim"], std::vector<int64_t>({1, 3, 224, 224}));
    EXPECT_EQ(entry["tensors"][1]["dim"], std::vector<int64_t>({64, 3, 7, 7}));

    // Verify knobs round-tripped correctly
    ASSERT_TRUE(entry.contains("autotune_metadata"));
    ASSERT_TRUE(entry["autotune_metadata"].contains("knobs"));
    ASSERT_EQ(entry["autotune_metadata"]["knobs"].size(), 2u);

    bool foundTileSize = false;
    bool foundSplitK = false;
    for(const auto& knob : entry["autotune_metadata"]["knobs"])
    {
        const auto knobId = knob["knob_id"].get<std::string>();
        if(knobId == "TILE_SIZE")
        {
            EXPECT_EQ(knob["value"].get<int64_t>(), 128);
            foundTileSize = true;
        }
        else if(knobId == "SPLIT_K")
        {
            EXPECT_EQ(knob["value"].get<int64_t>(), 2);
            foundSplitK = true;
        }
    }
    EXPECT_TRUE(foundTileSize) << "TILE_SIZE knob not found in round-trip";
    EXPECT_TRUE(foundSplitK) << "SPLIT_K knob not found in round-trip";
}

TEST(TestAutotuneFileWriter, RoundTripNoKnobs)
{
    const TempFile tmpFile;

    // Write results without knobs
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(MIOPEN_ENGINE_ID, "MIOPEN_ENGINE", 1.0f, true, 0));

    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};
    auto err = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, true, dims, {});
    ASSERT_TRUE(err.is_good());

    // Verify by parsing the JSON directly
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    ASSERT_TRUE(json.contains("engine_overrides"));
    ASSERT_EQ(json["engine_overrides"].size(), 1u);

    const auto& entry = json["engine_overrides"][0];
    EXPECT_EQ(entry["op"], "conv_fprop");
    EXPECT_EQ(entry["engine_name"], "MIOPEN_ENGINE");
    ASSERT_EQ(entry["tensors"].size(), 1u);
    EXPECT_EQ(entry["tensors"][0]["dim"], std::vector<int64_t>({1, 3, 224, 224}));
    ASSERT_TRUE(entry.contains("autotune_metadata"));
    EXPECT_FALSE(entry["autotune_metadata"].contains("knobs"));
}

// --- Error handling Tests ---

TEST(TestAutotuneFileWriter, WriteToInvalidPathFails)
{
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    auto err = writeVersionedAutotuneResults("/nonexistent/deep/path/that/does/not/exist/file.json",
                                             config_op::CONV_FPROP,
                                             results,
                                             true,
                                             dims,
                                             {});
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE) << err.get_message();
    EXPECT_NE(err.get_message().find("cannot open temp file for writing"), std::string::npos)
        << err.get_message();
}

TEST(TestAutotuneFileWriter, WriteNoSucceededResultsIsOk)
{
    const TempFile tmpFile;

    // All results failed
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 0.0f, false, -1));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    auto err = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, true, dims, {});

    // Should succeed (no error) but write nothing
    EXPECT_TRUE(err.is_good());
}

TEST(TestAutotuneFileWriter, HandleCorruptExistingFile)
{
    const TempFile tmpFile;

    // Write corrupt JSON to the file
    {
        std::ofstream outFile(tmpFile.path);
        outFile << "{ this is not valid json ]}}";
    }

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    // During write, corrupt file is moved aside; writer starts fresh and returns OK.
    Error err;
    ASSERT_NO_THROW(err = writeVersionedAutotuneResults(
                        tmpFile.path, config_op::CONV_FPROP, results, false, dims, {}));
    EXPECT_TRUE(err.is_good()) << err.get_message();

    EXPECT_EQ(collectCorruptSiblings(tmpFile.path).size(), 1u);

    ASSERT_TRUE(std::filesystem::exists(tmpFile.path));
    std::ifstream freshFile(tmpFile.path);
    ASSERT_TRUE(freshFile.is_open());
    nlohmann::json json;
    ASSERT_NO_THROW(json = nlohmann::json::parse(freshFile));
    ASSERT_TRUE(json.is_object());
    EXPECT_EQ(json[config_json::ENGINE_OVERRIDES].size(), 1u);
}

// --- Non-object existing-file hardening tests ---

TEST(TestAutotuneFileWriter, HandleExistingTopLevelArray)
{
    const TempFile tmpFile;

    // A valid JSON file whose root is an array, not an object.
    {
        std::ofstream outFile(tmpFile.path);
        outFile << "[]";
    }

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    Error err;
    ASSERT_NO_THROW(err = writeVersionedAutotuneResults(
                        tmpFile.path, config_op::CONV_FPROP, results, false, dims, {}));
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE) << err.get_message();
    EXPECT_NE(err.get_message().find("is not a versioned object"), std::string::npos)
        << err.get_message();
}

TEST(TestAutotuneFileWriter, HandleExistingBareScalar)
{
    const TempFile tmpFile;

    // A valid JSON file whose root is a bare scalar, not an object.
    {
        std::ofstream outFile(tmpFile.path);
        outFile << "42";
    }

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    Error err;
    ASSERT_NO_THROW(err = writeVersionedAutotuneResults(
                        tmpFile.path, config_op::CONV_FPROP, results, false, dims, {}));
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE) << err.get_message();
    EXPECT_NE(err.get_message().find("is not a versioned object"), std::string::npos)
        << err.get_message();
}

TEST(TestAutotuneFileWriter, HandleInvalidJsonDoesNotThrow)
{
    const TempFile tmpFile;

    // Unparseable content.
    {
        std::ofstream outFile(tmpFile.path);
        outFile << "}{ not json at all";
    }

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    // Must not throw; corrupt file is moved aside; a fresh file is written; OK returned.
    Error err;
    ASSERT_NO_THROW(err = writeVersionedAutotuneResults(
                        tmpFile.path, config_op::CONV_FPROP, results, false, dims, {}));
    EXPECT_TRUE(err.is_good()) << err.get_message();

    EXPECT_EQ(collectCorruptSiblings(tmpFile.path).size(), 1u);

    ASSERT_TRUE(std::filesystem::exists(tmpFile.path));
    std::ifstream freshFile(tmpFile.path);
    ASSERT_TRUE(freshFile.is_open());
    nlohmann::json json;
    ASSERT_NO_THROW(json = nlohmann::json::parse(freshFile));
    ASSERT_TRUE(json.is_object());
    EXPECT_EQ(json[config_json::ENGINE_OVERRIDES].size(), 1u);
}

TEST(TestAutotuneFileWriter, HandleWellFormedObjectPreservesOtherKeys)
{
    const TempFile tmpFile;

    // A well-formed object with no engine_overrides and an unrelated key.
    {
        std::ofstream outFile(tmpFile.path);
        outFile << R"({"unrelated":true})";
    }

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    Error err;
    ASSERT_NO_THROW(err = writeVersionedAutotuneResults(
                        tmpFile.path, config_op::CONV_FPROP, results, false, dims, {}));
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE) << err.get_message();
    EXPECT_NE(err.get_message().find("refusing to update legacy autotune config file"),
              std::string::npos)
        << err.get_message();
}

// --- ran_exhaustive / converged metadata tests ---

TEST(TestAutotuneFileWriter, RanExhaustiveAlwaysWritten)
{
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {};

    // ranExhaustive = false should be written
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.ranExhaustive = false;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        ASSERT_TRUE(meta.contains("ran_exhaustive"));
        EXPECT_FALSE(meta["ran_exhaustive"].get<bool>());
    }

    // ranExhaustive = true should be written
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.ranExhaustive = true;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        ASSERT_TRUE(meta.contains("ran_exhaustive"));
        EXPECT_TRUE(meta["ran_exhaustive"].get<bool>());
    }
}

TEST(TestAutotuneFileWriter, ConvergedOnlyForRunUntilStable)
{
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {};

    // FIXED_AVERAGE: converged should NOT be present
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.strategyUsed = AutotuneStrategy::FIXED_AVERAGE;
        result.converged = true;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        EXPECT_FALSE(meta.contains("converged"));
    }

    // RUN_UNTIL_STABLE with converged=true: converged should be present
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.strategyUsed = AutotuneStrategy::RUN_UNTIL_STABLE;
        result.converged = true;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        ASSERT_TRUE(meta.contains("converged"));
        EXPECT_TRUE(meta["converged"].get<bool>());
    }

    // RUN_UNTIL_STABLE with converged=false: converged should be present
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.strategyUsed = AutotuneStrategy::RUN_UNTIL_STABLE;
        result.converged = false;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        ASSERT_TRUE(meta.contains("converged"));
        EXPECT_FALSE(meta["converged"].get<bool>());
    }
}

TEST(TestAutotuneFileWriter, BuildOverrideEntryWithEmptyStrides)
{
    auto result = makeResult(1, "MIOPEN_ENGINE");
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {}; // No strides

    auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);

    ASSERT_EQ(entry["tensors"].size(), 1u);
    EXPECT_EQ(entry["tensors"][0]["dim"], std::vector<int64_t>({1, 3, 224, 224}));
    EXPECT_FALSE(entry["tensors"][0].contains("stride")); // No strides provided
}

// --- Rank-0-only write behavior tests ---

TEST(TestAutotuneFileWriter, WritesOnlyRank0Winner)
{
    const TempFile tmpFile;

    // Create 3 succeeded results with different ranks (pre-sorted by rank)
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "FAST_ENGINE", 0.5f, true, 0));
    results.push_back(makeResult(2, "MEDIUM_ENGINE", 1.5f, true, 1));
    results.push_back(makeResult(3, "SLOW_ENGINE", 3.0f, true, 2));

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}, {64, 3, 7, 7}};

    auto err = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, tensorDims, {});
    ASSERT_TRUE(err.is_good()) << err.get_message();

    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    ASSERT_TRUE(json.contains("engine_overrides"));
    // Only the rank-0 winner should be written
    ASSERT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "FAST_ENGINE");
    EXPECT_EQ(json["engine_overrides"][0]["autotune_metadata"]["rank"], 0);
}

TEST(TestAutotuneFileWriter, WritesRank0WinnerSkippingLeadingFailures)
{
    const TempFile tmpFile;

    // Failed results appear first, then succeeded results
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(10, "BROKEN_ENGINE_A", 0.0f, false, -1));
    results.push_back(makeResult(11, "BROKEN_ENGINE_B", 0.0f, false, -1));
    results.push_back(makeResult(1, "WINNER_ENGINE", 1.0f, true, 0));
    results.push_back(makeResult(2, "RUNNER_UP_ENGINE", 2.0f, true, 1));

    const std::vector<std::vector<int64_t>> tensorDims = {{4, 64, 56, 56}};

    auto err = writeVersionedAutotuneResults(
        tmpFile.path, config_op::CONV_FPROP, results, false, tensorDims, {});
    ASSERT_TRUE(err.is_good()) << err.get_message();

    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    ASSERT_TRUE(json.contains("engine_overrides"));
    // Only the rank-0 winner should be written (failed results skipped)
    ASSERT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "WINNER_ENGINE");
    EXPECT_EQ(json["engine_overrides"][0]["autotune_metadata"]["rank"], 0);
}

TEST(TestAutotuneFileWriter, Rank0WinnerReplacesExistingEntry)
{
    const TempFile tmpFile;

    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};

    // Write initial rank-0 winner
    {
        std::vector<AutotuneResult> results;
        results.push_back(makeResult(1, "OLD_WINNER", 2.0f, true, 0));
        auto err = writeVersionedAutotuneResults(
            tmpFile.path, config_op::CONV_FPROP, results, false, dims, {});
        ASSERT_TRUE(err.is_good());
    }

    // Re-autotune with multiple results; rank-0 winner should replace old entry
    {
        std::vector<AutotuneResult> results;
        results.push_back(makeResult(5, "NEW_WINNER", 0.8f, true, 0));
        results.push_back(makeResult(6, "NEW_RUNNER_UP", 1.2f, true, 1));
        auto err = writeVersionedAutotuneResults(
            tmpFile.path, config_op::CONV_FPROP, results, false, dims, {});
        ASSERT_TRUE(err.is_good());
    }

    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    // Only 1 entry: the new rank-0 winner replaced the old one
    ASSERT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "NEW_WINNER");
    EXPECT_EQ(json["engine_overrides"][0]["autotune_metadata"]["rank"], 0);
}

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
