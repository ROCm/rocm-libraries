// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/autotune/AutotuneFileWriter.hpp>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#endif

using namespace hipdnn_frontend;

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
using namespace hipdnn_frontend::autotune;
using namespace hipdnn_data_sdk::utilities;

// ── Test helpers ────────────────────────────────────────────────────────────

namespace
{

/// Create a temporary file path for testing, cleaned up by destructor.
struct TempFile
{
    std::filesystem::path path;

    TempFile()
    {
        static std::atomic<int> s_counter{0};
        path = std::filesystem::temp_directory_path()
               / ("hipdnn_test_" + std::to_string(s_counter++) + ".json");
    }

    ~TempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        // Also remove any .tmp file
        std::filesystem::path tmp = path;
        tmp += ".tmp";
        std::filesystem::remove(tmp, ec);
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
    r.modeUsed = TuneMode::AUTO;
    r.converged = true;
    r.workspaceSize = 1024;
    r.rank = rank;
    return r;
}

} // namespace

// ── knobSettingToJson Tests ─────────────────────────────────────────────────

TEST(TestAutotuneFileWriter, KnobSettingToJsonInt)
{
    const KnobSetting setting("TILE_SIZE", int64_t{128});
    auto json = knobSettingToJson(setting);

    EXPECT_EQ(json["knob_id"], "TILE_SIZE");
    EXPECT_EQ(json["type"], "int");
    EXPECT_EQ(json["value"], 128);
}

TEST(TestAutotuneFileWriter, KnobSettingToJsonDouble)
{
    const KnobSetting setting("LEARNING_RATE", 0.001);
    auto json = knobSettingToJson(setting);

    EXPECT_EQ(json["knob_id"], "LEARNING_RATE");
    EXPECT_EQ(json["type"], "double");
    EXPECT_DOUBLE_EQ(json["value"].get<double>(), 0.001);
}

TEST(TestAutotuneFileWriter, KnobSettingToJsonString)
{
    const KnobSetting setting("ALGORITHM", std::string("gemm_v2"));
    auto json = knobSettingToJson(setting);

    EXPECT_EQ(json["knob_id"], "ALGORITHM");
    EXPECT_EQ(json["type"], "string");
    EXPECT_EQ(json["value"], "gemm_v2");
}

// ── buildOverrideEntry Tests ────────────────────────────────────────────────

TEST(TestAutotuneFileWriter, BuildOverrideEntryBasic)
{
    auto result = makeResult(1, "MIOPEN_ENGINE");
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    const std::vector<std::vector<int64_t>> tensorStrides
        = {{150528, 50176, 224, 1}, {147, 49, 7, 1}};

    auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);

    EXPECT_EQ(entry["op"], "conv_fprop");
    EXPECT_EQ(entry["engine_name"], "MIOPEN_ENGINE");
    ASSERT_EQ(entry["tensors"].size(), 2u);
    EXPECT_EQ(entry["tensors"][0]["dim"], std::vector<int64_t>({1, 3, 224, 224}));
    EXPECT_EQ(entry["tensors"][1]["dim"], std::vector<int64_t>({64, 3, 7, 7}));
    EXPECT_EQ(entry["tensors"][0]["stride"], std::vector<int64_t>({150528, 50176, 224, 1}));
    EXPECT_EQ(entry["tensors"][1]["stride"], std::vector<int64_t>({147, 49, 7, 1}));
    ASSERT_TRUE(entry.contains("autotune_metadata"));
    EXPECT_FALSE(entry["autotune_metadata"].contains("knobs")); // No knobs → field absent
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

// ── writeAutotuneResults Tests ──────────────────────────────────────────────

TEST(TestAutotuneFileWriter, WriteToNewFile)
{
    const TempFile tmpFile;

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}, {64, 3, 7, 7}};

    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, false, tensorDims, {});
    ASSERT_TRUE(err.is_good()) << err.get_message();

    // Verify file exists and is valid JSON
    ASSERT_TRUE(std::filesystem::exists(tmpFile.path));
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    ASSERT_TRUE(json.contains("engine_overrides"));
    EXPECT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "MIOPEN_ENGINE");
}

TEST(TestAutotuneFileWriter, WriteReplacesNonArrayEngineOverrides)
{
    const TempFile tmpFile;

    {
        std::ofstream file(tmpFile.path);
        file << R"({"engine_overrides":{"not":"an array"},"preserved":true})";
    }

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};

    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, false, tensorDims, {});
    ASSERT_TRUE(err.is_good()) << err.get_message();

    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);
    ASSERT_TRUE(json["engine_overrides"].is_array());
    ASSERT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "MIOPEN_ENGINE");
    EXPECT_TRUE(json["preserved"].get<bool>());
}

TEST(TestAutotuneFileWriter, WriteSkipsFailedResults)
{
    const TempFile tmpFile;

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));
    results.push_back(makeResult(2, "HIPBLASLT_ENGINE", 0.0f, false, -1));
    results.push_back(makeResult(3, "FUSILLI_ENGINE", 3.0f, true, 1));

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};

    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, false, tensorDims, {});
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
    auto err1 = writeAutotuneResults(tmpFile.path, "conv_fprop", results1, false, dims1, {});
    ASSERT_TRUE(err1.is_good());

    // Write new results for conv_dgrad (different op)
    std::vector<AutotuneResult> results2;
    results2.push_back(makeResult(2, "HIPBLASLT_ENGINE", 2.0f, true, 0));

    const std::vector<std::vector<int64_t>> dims2 = {{8, 64, 56, 56}};
    auto err2 = writeAutotuneResults(tmpFile.path, "conv_dgrad", results2, false, dims2, {});
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
    auto err1 = writeAutotuneResults(tmpFile.path, "conv_fprop", results1, false, dims, {});
    ASSERT_TRUE(err1.is_good());

    // Write updated result for same op + tensors + same (empty) knobs
    std::vector<AutotuneResult> results2;
    results2.push_back(makeResult(2, "HIPBLASLT_ENGINE", 1.0f, true, 0));

    auto err2 = writeAutotuneResults(tmpFile.path, "conv_fprop", results2, false, dims, {});
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
    auto err1 = writeAutotuneResults(tmpFile.path, "conv_fprop", results1, false, dims, {});
    ASSERT_TRUE(err1.is_good());

    // Write new result for same op + tensors but DIFFERENT knobs (SPLIT_K=4)
    std::vector<AutotuneResult> results2;
    auto r2 = makeResult(2, "HIPBLASLT_ENGINE", 1.0f, true, 0);
    r2.knobSettings.emplace_back("SPLIT_K", int64_t{4});
    results2.push_back(r2);

    auto err2 = writeAutotuneResults(tmpFile.path, "conv_fprop", results2, false, dims, {});
    ASSERT_TRUE(err2.is_good());

    // The old entry should be replaced (matching by op + tensors only)
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "HIPBLASLT_ENGINE");
}

TEST(TestAutotuneFileWriter, ReplaceOnlyExactOperationAndTensorSignature)
{
    const TempFile tmpFile;
    const std::vector<std::vector<int64_t>> matchingDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> otherDims = {{2, 3, 224, 224}};

    nlohmann::json root;
    root["engine_overrides"] = nlohmann::json::array(
        {{{"engine_name", "MISSING_OP"}, {"tensors", nlohmann::json::array()}},
         {{"op", "conv_fprop"}, {"engine_name", "MISSING_TENSORS"}},
         buildOverrideEntry(makeResult(7, "OTHER_TENSORS"), "conv_fprop", otherDims, {}),
         buildOverrideEntry(makeResult(8, "OLD_MATCH"), "conv_fprop", matchingDims, {})});
    {
        std::ofstream file(tmpFile.path);
        file << root.dump(2) << '\n';
    }

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(9, "NEW_MATCH", 0.5f, true, 0));
    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, false, matchingDims, {});
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
    auto err1 = writeAutotuneResults(tmpFile.path, "conv_fprop", results1, false, dims1, {});
    ASSERT_TRUE(err1.is_good());

    // Write new results with deleteAllExisting=true
    std::vector<AutotuneResult> results2;
    results2.push_back(makeResult(2, "HIPBLASLT_ENGINE", 2.0f, true, 0));
    const std::vector<std::vector<int64_t>> dims2 = {{8, 64, 56, 56}};
    auto err2 = writeAutotuneResults(tmpFile.path, "conv_dgrad", results2, true, dims2, {});
    ASSERT_TRUE(err2.is_good());

    // Only the new results should be in the file
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["op"], "conv_dgrad");
}

// ── Round-trip Tests ────────────────────────────────────────────────────────

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
    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, true, dims, {});
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
    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, true, dims, {});
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

// ── Strategy/Mode String Tests ──────────────────────────────────────────────

TEST(TestAutotuneFileWriter, StrategyToString)
{
    EXPECT_EQ(strategyToString(AutotuneStrategy::SINGLE_SHOT), "SINGLE_SHOT");
    EXPECT_EQ(strategyToString(AutotuneStrategy::FIXED_AVERAGE), "FIXED_AVERAGE");
    EXPECT_EQ(strategyToString(AutotuneStrategy::RUN_UNTIL_STABLE), "RUN_UNTIL_STABLE");
}

TEST(TestAutotuneFileWriter, StrategyToLowerString)
{
    EXPECT_EQ(strategyToLowerString(AutotuneStrategy::SINGLE_SHOT), "single_shot");
    EXPECT_EQ(strategyToLowerString(AutotuneStrategy::FIXED_AVERAGE), "fixed_average");
    EXPECT_EQ(strategyToLowerString(AutotuneStrategy::RUN_UNTIL_STABLE), "run_until_stable");
}

TEST(TestAutotuneFileWriter, TuneModeToString)
{
    EXPECT_EQ(tuneModeToString(TuneMode::AUTO), "AUTO");
    EXPECT_EQ(tuneModeToString(TuneMode::EXHAUSTIVE), "EXHAUSTIVE");
}

TEST(TestAutotuneFileWriter, TuneModeToLowerString)
{
    EXPECT_EQ(tuneModeToLowerString(TuneMode::AUTO), "auto");
    EXPECT_EQ(tuneModeToLowerString(TuneMode::EXHAUSTIVE), "exhaustive");
}

TEST(TestAutotuneFileWriter, UnknownEnumsUseUnknownLowercaseStrings)
{
    EXPECT_EQ(strategyToLowerString(static_cast<AutotuneStrategy>(999)), "unknown");
    EXPECT_EQ(tuneModeToLowerString(static_cast<TuneMode>(999)), "unknown");
}

// ── Error handling Tests ────────────────────────────────────────────────────

TEST(TestAutotuneFileWriter, WriteToInvalidPathFails)
{
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    auto err = writeAutotuneResults("/nonexistent/deep/path/that/does/not/exist/file.json",
                                    "conv_fprop",
                                    results,
                                    true,
                                    dims,
                                    {});

    EXPECT_TRUE(err.is_bad());
}

TEST(TestAutotuneFileWriter, WriteNoSucceededResultsIsOk)
{
    const TempFile tmpFile;

    // All results failed
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 0.0f, false, -1));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, true, dims, {});

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

    // Write valid results — should start fresh despite corrupt existing
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, false, dims, {});

    ASSERT_TRUE(err.is_good());

    // Verify valid JSON was written
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);
    EXPECT_EQ(json["engine_overrides"].size(), 1u);
}

// ── ran_exhaustive / converged metadata tests ───────────────────────────────

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

    // SINGLE_SHOT: converged should NOT be present
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.strategyUsed = AutotuneStrategy::SINGLE_SHOT;
        result.converged = false;
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

// ── Rank-0-only write behavior tests ───────────────────────────────────────

TEST(TestAutotuneFileWriter, WritesOnlyRank0Winner)
{
    const TempFile tmpFile;

    // Create 3 succeeded results with different ranks (pre-sorted by rank)
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "FAST_ENGINE", 0.5f, true, 0));
    results.push_back(makeResult(2, "MEDIUM_ENGINE", 1.5f, true, 1));
    results.push_back(makeResult(3, "SLOW_ENGINE", 3.0f, true, 2));

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}, {64, 3, 7, 7}};

    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, false, tensorDims, {});
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

    auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, false, tensorDims, {});
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
        auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, false, dims, {});
        ASSERT_TRUE(err.is_good());
    }

    // Re-autotune with multiple results; rank-0 winner should replace old entry
    {
        std::vector<AutotuneResult> results;
        results.push_back(makeResult(5, "NEW_WINNER", 0.8f, true, 0));
        results.push_back(makeResult(6, "NEW_RUNNER_UP", 1.2f, true, 1));
        auto err = writeAutotuneResults(tmpFile.path, "conv_fprop", results, false, dims, {});
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
