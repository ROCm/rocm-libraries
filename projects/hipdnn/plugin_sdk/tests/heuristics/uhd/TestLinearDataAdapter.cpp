// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestLinearDataAdapter.cpp
 * @brief Tests for LinearDataAdapter (per-config linear ranker) per RFC 0019 §7 "linear_data".
 */

#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/LinearDataAdapter.hpp>

#include <hipdnn_test_sdk/utilities/LinearModelTestBuilder.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_plugin_sdk::uhd;
using hipdnn_test_sdk::utilities::LinearModelTestBuilder;

namespace
{
constexpr const char* TEST_HASH = "sha256:0123456789abcdef";
constexpr double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();

// BEGIN SHARED FIXTURE: written by uhd_gen's linear_to_flatbuffer from the model in
// uhd_gen/tests/test_linear_model.py (`fixture_ranker`); the Python tests check these bytes
// are what the writer produces and score the same rows to the same values.
constexpr const char* FIXTURE_SHA256
    = "912ab5a04f6cc35dfc2f6a4ac6078c7244cd0a2a0517d2571e0981909b411453";
constexpr const char* FIXTURE_HEX
    = "2c000000484c494e0000000000001e0040003c003800340030002c002800240020001c0018000c00080004001e00"
      "0000"
      "3c0000004400000064000000000000000000000048000000640000006c0000009c0000001401000050020000cc02"
      "0000"
      "d0020000d80200000400000005000000312e302e3000000001000000040000000600000067667839343200001900"
      "0000"
      "323032362d30312d30315430303a30303a30302b30303a30300000000600000072656c6c696e000058feffff0800"
      "0000"
      "2400000003000000000000000000e03f000000000000c03f000000000000d03f0000000000000000000000000200"
      "0000"
      "400000000400000098feffff080000002400000003000000000000000000d43f000000000000e03f000000000000"
      "e03f"
      "00000000010000000000000000000040d0feffff080000002400000003000000000000000000c83f000000000000"
      "d03f"
      "000000000000b03f0000000001000000000000000000f03f04000000fc000000a4000000540000000400000072ff"
      "ffff"
      "000000000000d83f00000000080000002400000003000000000000000000c03f0000000000000000000000000000"
      "0000"
      "00000000020000000000000000000000000000000000f03f68ffffff080000002400000003000000000000000000"
      "e03f"
      "000000000000d0bf000000000000d83f00000000020000000000000000000040000000000000f87f00000a001800"
      "1400"
      "100004000a000000000000000000b03f000000000800000024000000030000000000000000000000000000000000"
      "e83f"
      "000000000000d03f0000000002000000000000000000f03f000000000000004008000c0008000400080000000800"
      "0000"
      "2400000003000000000000000000d03f000000000000e03f000000000000c0bf0000000002000000000000000000"
      "f03f"
      "000000000000f03f02000000500000001400000010002800240000001c0014000c00040010000000000000000000"
      "0040"
      "00000000000000c0000000000000d03f000000000000e03f0300000010002c00280027001c0014000c0004001000"
      "0000"
      "000000000000f83f000000000000f8bf000000000000004000000000000008400000000102000000010000000000"
      "0000"
      "020000000000000001000000170000007368613235363a3031323334353637383961626364656600";

// Rows over the fixture's 4 features, and the scores uhd_gen's scorer gives them.
const std::vector<std::vector<double>> FIXTURE_ROWS = {
    {1, 1, 20, 0.75}, // config (1,1)
    {1, 2, 1e9, -50.0}, // config (1,2), both terms clamped
    {2, NAN_VALUE, NAN_VALUE, 0.5}, // config (2,NaN); a missing size stands at the mean
    {1, 9, 100, 0.25}, // unseen config, family a=1
    {3, 1, 100, 0.25}, // unseen family, global
    {0, 1, 0, 0.5}, // config (0,1): the raw score is below its floor
    {-0.0, 1, -5, 0.5}, // -0.0 is the key 0.0; a negative size reads as 0
    {2, 7, 20, 2.0}, // unseen config, family a=2
};
const std::vector<double> FIXTURE_SCORES = {
    0.13613060943085575,
    0.625,
    0.5,
    0.32689006460515746,
    0.35094503230257873,
    0.375,
    0.375,
    1.3236306094308556,
};
// END SHARED FIXTURE

std::vector<uint8_t> fixtureBytes()
{
    const std::string hex = FIXTURE_HEX;
    std::vector<uint8_t> bytes;
    for(size_t i = 0; i + 1 < hex.size(); i += 2)
    {
        bytes.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return bytes;
}

/// Two configs keyed on slot 0, one term over slot 1, a family-free global.
LinearModelTestBuilder validModel()
{
    return LinearModelTestBuilder()
        .setNumFeatures(3)
        .setFeaturesHash(TEST_HASH)
        .setKeyFeatures({0})
        .addTerm({1, 0, 0.0, 1.0, -2.0, 2.0})
        .addConfig({{1.0}, {0.5, 2.0}, 0.0})
        .addConfig({{2.0}, {1.0, 1.0}, 0.0})
        .setGlobal({{}, {3.0, 0.0}, 0.0});
}

std::unique_ptr<LinearDataAdapter> load(LinearModelTestBuilder builder)
{
    const auto buffer = builder.build();
    return LinearDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
}

TEST(TestLinearDataAdapter, ScoresTheSharedFixtureAsUhdGenDoes)
{
    const auto bytes = fixtureBytes();
    ASSERT_EQ(sha256(bytes.data(), bytes.size()), FIXTURE_SHA256)
        << "the embedded fixture is not the writer's output";
    const auto adapter
        = LinearDataAdapter::loadFromBuffer(bytes.data(), bytes.size(), TEST_HASH, FIXTURE_SHA256);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->expectedFeatureCount(), 4U);
    EXPECT_EQ(adapter->getFeaturesHash(), TEST_HASH);

    std::vector<double> flat;
    for(const auto& row : FIXTURE_ROWS)
    {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    const auto batch = adapter->scoreBatch({flat.data(), FIXTURE_ROWS.size(), 4});
    ASSERT_EQ(batch.raw.size(), FIXTURE_SCORES.size());
    EXPECT_TRUE(batch.stage.empty()) << "a linear model has no early-exit schedule";
    for(size_t i = 0; i < FIXTURE_ROWS.size(); ++i)
    {
        EXPECT_NEAR(batch.raw[i], FIXTURE_SCORES[i], 1e-12) << "row " << i;
        EXPECT_DOUBLE_EQ(adapter->score(FIXTURE_ROWS[i]), batch.raw[i]) << "row " << i;
        EXPECT_TRUE(batch.fullyScored(i));
    }
}

TEST(TestLinearDataAdapter, ARowUsesItsConfigThenItsFamilyThenTheGlobalEntry)
{
    const auto adapter
        = load(validModel().setFamilyFeatures({2}).addFamily({{7.0}, {10.0, 0.0}, 0.0}));
    ASSERT_NE(adapter, nullptr);
    // config 1: 0.5 + 2.0 * 0.25; config 2: 1.0 + 1.0 * 0.25.
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 0.25, 7.0}), 1.0);
    EXPECT_DOUBLE_EQ(adapter->score({2.0, 0.25, 7.0}), 1.25);
    // Neither config: the family of slot 2, else the global entry.
    EXPECT_DOUBLE_EQ(adapter->score({5.0, 0.25, 7.0}), 10.0);
    EXPECT_DOUBLE_EQ(adapter->score({5.0, 0.25, 8.0}), 3.0);
}

TEST(TestLinearDataAdapter, ATermIsClampedToItsTrainingRangeAndAMissingValueIsZero)
{
    // Floors far below any score, so only the clamp bounds it.
    const auto adapter = load(LinearModelTestBuilder()
                                  .setNumFeatures(3)
                                  .setFeaturesHash(TEST_HASH)
                                  .setKeyFeatures({0})
                                  .addTerm({1, 0, 0.0, 1.0, -2.0, 2.0})
                                  .addConfig({{1.0}, {0.5, 2.0}, -100.0})
                                  .setGlobal({{}, {3.0, 0.0}, -100.0}));
    ASSERT_NE(adapter, nullptr);
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 100.0, 0.0}), 0.5 + (2.0 * 2.0));
    EXPECT_DOUBLE_EQ(adapter->score({1.0, -INF, 0.0}), 0.5 + (2.0 * -2.0));
    EXPECT_DOUBLE_EQ(adapter->score({1.0, INF, 0.0}), 0.5 + (2.0 * 2.0));
    EXPECT_DOUBLE_EQ(adapter->score({1.0, NAN_VALUE, 0.0}), 0.5);
}

TEST(TestLinearDataAdapter, Log1pReadsTheNonNegativePart)
{
    const auto adapter = load(LinearModelTestBuilder()
                                  .setNumFeatures(1)
                                  .setFeaturesHash(TEST_HASH)
                                  .addTerm({0, 1, 0.0, 1.0, -100.0, 100.0})
                                  .addConfig({{}, {0.0, 1.0}, 0.0})
                                  .setGlobal({{}, {0.0, 1.0}, 0.0}));
    ASSERT_NE(adapter, nullptr);
    EXPECT_NEAR(adapter->score({std::exp(1.0) - 1.0}), 1.0, 1e-15);
    EXPECT_DOUBLE_EQ(adapter->score({-7.0}), 0.0);
}

TEST(TestLinearDataAdapter, AConfigNeverBestInTrainingIsNotExtrapolatedBelowItsFloor)
{
    const auto adapter = load(LinearModelTestBuilder()
                                  .setNumFeatures(2)
                                  .setFeaturesHash(TEST_HASH)
                                  .setKeyFeatures({0})
                                  .addTerm({1, 0, 0.0, 1.0, -5.0, 5.0})
                                  .addConfig({{1.0}, {0.0, -1.0}, 0.25})
                                  .setGlobal({{}, {0.0, 0.0}, 0.0}));
    ASSERT_NE(adapter, nullptr);
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 4.0}), 0.25) << "raw -4 must be raised to the floor";
    EXPECT_DOUBLE_EQ(adapter->score({1.0, -4.0}), 4.0);
}

TEST(TestLinearDataAdapter, KeysCompareAsDoublesWithEveryNanEqualAndNegativeZeroAsZero)
{
    const auto adapter = load(LinearModelTestBuilder()
                                  .setNumFeatures(2)
                                  .setFeaturesHash(TEST_HASH)
                                  .setKeyFeatures({0, 1})
                                  .addConfig({{NAN_VALUE, 0.0}, {1.0}, 0.0})
                                  .addConfig({{4.0, NAN_VALUE}, {2.0}, 0.0})
                                  .setGlobal({{}, {9.0}, 0.0}));
    ASSERT_NE(adapter, nullptr);
    EXPECT_DOUBLE_EQ(adapter->score({NAN_VALUE, -0.0}), 1.0);
    EXPECT_DOUBLE_EQ(adapter->score({-std::numeric_limits<double>::quiet_NaN(), 0.0}), 1.0);
    EXPECT_DOUBLE_EQ(adapter->score({4.0, NAN_VALUE}), 2.0);
    EXPECT_DOUBLE_EQ(adapter->score({4.0, 0.0}), 9.0);
}

TEST(TestLinearDataAdapter, AModelWithoutKeyFeaturesScoresEveryRowWithItsOneConfig)
{
    const auto adapter = load(LinearModelTestBuilder()
                                  .setNumFeatures(1)
                                  .setFeaturesHash(TEST_HASH)
                                  .addConfig({{}, {2.0}, 0.0})
                                  .setGlobal({{}, {9.0}, 0.0}));
    ASSERT_NE(adapter, nullptr);
    EXPECT_DOUBLE_EQ(adapter->score({123.0}), 2.0);
}

TEST(TestLinearDataAdapter, ARowOfAnotherWidthIsDeclined)
{
    const auto adapter = load(validModel());
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->score({1.0, 0.0}), -INF);

    const std::vector<double> flat = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    const auto batch = adapter->scoreBatch({flat.data(), 3, 2});
    ASSERT_EQ(batch.raw.size(), 3U);
    for(const double raw : batch.raw)
    {
        EXPECT_EQ(raw, -INF);
    }
}

TEST(TestLinearDataAdapter, AnEmptyCatalogScoresNothing)
{
    const auto adapter = load(validModel());
    ASSERT_NE(adapter, nullptr);
    EXPECT_TRUE(adapter->scoreBatch({nullptr, 0, 3}).raw.empty());
}

TEST(TestLinearDataAdapter, ArchitecturesComeFromTheArtifact)
{
    const auto open = load(validModel());
    ASSERT_NE(open, nullptr);
    EXPECT_TRUE(open->isTrainedForArch("gfx1201"));

    const auto scoped = load(validModel().setTrainingArches({"gfx942"}));
    ASSERT_NE(scoped, nullptr);
    EXPECT_TRUE(scoped->isTrainedForArch("gfx942:sramecc+:xnack-"));
    EXPECT_FALSE(scoped->isTrainedForArch("gfx1201"));
}

TEST(TestLinearDataAdapter, TheDeclaredModelHashIsVerifiedBeforeTheBytesAreRead)
{
    const auto buffer = validModel().build();
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_EQ(LinearDataAdapter::loadFromBuffer(
                  buffer.data(), buffer.size(), TEST_HASH, sha256(std::string("other bytes"))),
              nullptr);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
        << recorder.getRecordedLogsAsString();

    EXPECT_NE(LinearDataAdapter::loadFromBuffer(
                  buffer.data(), buffer.size(), TEST_HASH, sha256(buffer.data(), buffer.size())),
              nullptr);
}

TEST(TestLinearDataAdapter, AFeaturesHashOtherThanTheDescriptorsIsRefused)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const auto buffer = validModel().setFeaturesHash("sha256:fedcba9876543210").build();
    EXPECT_EQ(LinearDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH), nullptr);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
        << recorder.getRecordedLogsAsString();
}

TEST(TestLinearDataAdapter, ABufferThatIsNotALinearModelIsRefused)
{
    EXPECT_EQ(LinearDataAdapter::loadFromBuffer(nullptr, 100, TEST_HASH), nullptr);
    const std::vector<uint8_t> tiny = {1, 2, 3};
    EXPECT_EQ(LinearDataAdapter::loadFromBuffer(tiny.data(), tiny.size(), TEST_HASH), nullptr);

    const auto unidentified = validModel().setIdentifier(false).build();
    EXPECT_EQ(
        LinearDataAdapter::loadFromBuffer(unidentified.data(), unidentified.size(), TEST_HASH),
        nullptr);

    // A table model's bytes carry another identifier.
    auto other = validModel().build();
    other[4] = 'T';
    EXPECT_EQ(LinearDataAdapter::loadFromBuffer(other.data(), other.size(), TEST_HASH), nullptr);

    EXPECT_EQ(LinearDataAdapter::load("no/such/linear_model.bin", TEST_HASH), nullptr);
}

struct Refusal
{
    const char* name;
    LinearModelTestBuilder builder;
};

TEST(TestLinearDataAdapter, AModelThatIsInconsistentWithItselfIsRefusedWithOneError)
{
    using Term = LinearModelTestBuilder::Term;
    const std::vector<Refusal> refusals = {
        {"key feature out of range", validModel().setKeyFeatures({3})},
        {"negative key feature", validModel().setKeyFeatures({-1})},
        {"family feature out of range", validModel().setFamilyFeatures({3})},
        {"basis feature out of range", validModel().addTerm(Term{3, 0, 0.0, 1.0, -1.0, 1.0})},
        {"unknown transform",
         LinearModelTestBuilder(validModel()).addTerm(Term{1, 2, 0, 1, -1, 1})},
        {"zero scale",
         LinearModelTestBuilder()
             .setNumFeatures(3)
             .setFeaturesHash(TEST_HASH)
             .addTerm(Term{1, 0, 0.0, 0.0, -1.0, 1.0})
             .setGlobal({{}, {0.0, 0.0}, 0.0})},
        {"negative scale",
         LinearModelTestBuilder()
             .setNumFeatures(3)
             .setFeaturesHash(TEST_HASH)
             .addTerm(Term{1, 0, 0.0, -1.0, -1.0, 1.0})
             .setGlobal({{}, {0.0, 0.0}, 0.0})},
        {"lo above hi",
         LinearModelTestBuilder()
             .setNumFeatures(3)
             .setFeaturesHash(TEST_HASH)
             .addTerm(Term{1, 0, 0.0, 1.0, 1.0, -1.0})
             .setGlobal({{}, {0.0, 0.0}, 0.0})},
        {"infinite hi",
         LinearModelTestBuilder()
             .setNumFeatures(3)
             .setFeaturesHash(TEST_HASH)
             .addTerm(Term{1, 0, 0.0, 1.0, -1.0, INF})
             .setGlobal({{}, {0.0, 0.0}, 0.0})},
        {"nan mean",
         LinearModelTestBuilder()
             .setNumFeatures(3)
             .setFeaturesHash(TEST_HASH)
             .addTerm(Term{1, 0, NAN_VALUE, 1.0, -1.0, 1.0})
             .setGlobal({{}, {0.0, 0.0}, 0.0})},
        {"short coefficients", validModel().addConfig({{3.0}, {1.0}, 0.0})},
        {"long coefficients", validModel().addConfig({{3.0}, {1.0, 2.0, 3.0}, 0.0})},
        {"nonfinite coefficient", validModel().addConfig({{3.0}, {1.0, INF}, 0.0})},
        {"nonfinite floor", validModel().addConfig({{3.0}, {1.0, 1.0}, NAN_VALUE})},
        {"short key", validModel().addConfig({{}, {1.0, 1.0}, 0.0})},
        {"long key", validModel().addConfig({{3.0, 4.0}, {1.0, 1.0}, 0.0})},
        {"duplicate config key", validModel().addConfig({{1.0}, {1.0, 1.0}, 0.0})},
        {"duplicate nan config key",
         LinearModelTestBuilder()
             .setNumFeatures(1)
             .setFeaturesHash(TEST_HASH)
             .setKeyFeatures({0})
             .addConfig({{NAN_VALUE}, {1.0}, 0.0})
             .addConfig({{-NAN_VALUE}, {2.0}, 0.0})
             .setGlobal({{}, {0.0}, 0.0})},
        {"duplicate zero config key",
         LinearModelTestBuilder()
             .setNumFeatures(1)
             .setFeaturesHash(TEST_HASH)
             .setKeyFeatures({0})
             .addConfig({{0.0}, {1.0}, 0.0})
             .addConfig({{-0.0}, {2.0}, 0.0})
             .setGlobal({{}, {0.0}, 0.0})},
        {"families without a family key", validModel().addFamily({{}, {1.0, 1.0}, 0.0})},
        {"duplicate family key",
         validModel()
             .setFamilyFeatures({2})
             .addFamily({{1.0}, {1.0, 1.0}, 0.0})
             .addFamily({{1.0}, {2.0, 2.0}, 0.0})},
        {"family key of the wrong width",
         validModel().setFamilyFeatures({2}).addFamily({{1.0, 2.0}, {1.0, 1.0}, 0.0})},
        {"no global entry", validModel().omitGlobal()},
        {"global with a key", validModel().setGlobal({{1.0}, {1.0, 1.0}, 0.0})},
        {"negative feature count", validModel().setNumFeatures(-1)},
    };

    for(const auto& refusal : refusals)
    {
        SCOPED_TRACE(refusal.name);
        auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        auto builder = refusal.builder;
        const auto buffer = builder.build();
        EXPECT_EQ(LinearDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH),
                  nullptr);
        EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
            << recorder.getRecordedLogsAsString();
    }
}

} // namespace
