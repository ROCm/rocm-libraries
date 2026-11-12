// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestMlpDataAdapter.cpp
 * @brief Tests for MlpDataAdapter (per-engine MLP ensemble, an L1 engine estimate) per RFC 0019
 *        §7 "mlp_data".
 */

#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/MlpDataAdapter.hpp>

#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace hipdnn_plugin_sdk::uhd;
namespace fb = hipdnn_flatbuffers_sdk::data_objects;

namespace
{
constexpr const char* TEST_HASH = "sha256:0123456789abcdef";
constexpr const char* TEST_METRIC = "tflops";
constexpr double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();

// BEGIN SHARED FIXTURE: written by uhd_gen's mlp_to_flatbuffer from the model in
// uhd_gen/tests/test_mlp_model.py (`fixture_ensemble`); the Python tests check these bytes
// are what the writer produces and score the same rows to the same values.
constexpr const char* FIXTURE_SHA256
    = "e5fb1c1b88f12bf47c3f71aef67d0780a1b47ffacf541a22ec9a564a9f309b76";
constexpr const char* FIXTURE_HEX
    = "2c000000484d4c500000000000001e00440040003c003800340030002c0024001c00180014000c00080004001e00"
      "0000"
      "40000000480000006400000000000000500000006c00000000000000000010400000000000000840680000002002"
      "0000"
      "2402000094020000ac0200000400000005000000312e302e30000000010000000400000006000000676678393432"
      "0000"
      "19000000323032362d30312d30315430303a30303a30302b30303a30300000000b0000007568645f67656e2e6d6c"
      "7000"
      "02000000e4000000040000002affffff0400000003000000700000003000000004000000f4feffff100000001400"
      "0000"
      "0100000002000000010000000000000002000000000000bf0000803e1cffffff1000000018000000020000000300"
      "0000"
      "02000000000000be0000403e060000000000403f000040bf000000000000803e0000803f000000bf58ffffff1000"
      "0000"
      "1c0000000300000004000000030000000000403e0000803d000080bd0c0000000000803e0000803f000000bf0000"
      "803e"
      "000080be0000003f000080bf000080be000040bf000000000000403f000040bf0000060008000400060000000400"
      "0000"
      "020000004800000004000000ccffffff10000000140000000100000005000000010000000000803d050000000000"
      "80be"
      "0000003f000080bf000080be0000003f0c00140010000c00080004000c0000001000000024000000050000000400"
      "0000"
      "05000000000080bd000040be0000003e00000000000000be14000000000040bf000000000000403f000040bf0000"
      "803f"
      "000000bf0000803e0000803f0000003f000080bf000080be0000003f000000000000403f000040bf000000000000"
      "00bf"
      "0000803e0000803f000000bf0100000001000000030000005c0000003000000004000000e2ffffff000000000000"
      "2040"
      "000000000000f03f000000000300000000000a001e0018000c0004000a0000000000000000001440000000000000"
      "f03f"
      "000000000100000000000a00140000000c0004000a00000000000000000018400000000000000040170000007368"
      "6132"
      "35363a30313233343536373839616263646566000600000074666c6f70730000";

// Rows over the fixture's 4 features, and the scores uhd_gen's scorer gives them.
const std::vector<std::vector<double>> FIXTURE_ROWS = {
    {10, 3, 0, 7}, // ordinary
    {0.5, 1.5, 5, 0.25}, // small values
    {100, NAN_VALUE, 1, 20}, // missing in an input that has a flag: z reads 0, flag 1
    {7, 2, NAN_VALUE, NAN_VALUE}, // missing in an input with no flag; slot 2 is not read
    {-5, -2, 0, -30}, // negatives: signed log
    {1e9, 1e7, 3, 1e12}, // large magnitudes
    {0, 0, 0, 0}, // zeros
    {-1e9, NAN_VALUE, 2, 1e-3}, // a large negative beside a missing value
};
const std::vector<double> FIXTURE_SCORES = {
    3.091429214924574,
    2.9953340888023376,
    2.7827204391360283,
    2.9268824011087418,
    2.5847426056861877,
    6.6569730043411255,
    2.8321138322353363,
    3.696769639849663,
};
// END SHARED FIXTURE

// Float32 members and a polynomial exp: the runtime agrees with uhd_gen to float rounding.
constexpr double TOLERANCE = 1e-5;

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

std::unique_ptr<MlpDataAdapter> loadFixture()
{
    const auto bytes = fixtureBytes();
    return MlpDataAdapter::loadFromBuffer(
        bytes.data(), bytes.size(), TEST_HASH, TEST_METRIC, FIXTURE_SHA256);
}

/// One feature, one input z = log1p(x), one network z -> 2 * z + 0.5; the score is
/// 1 + 2 * output = 2 + 4 * log1p(x). `edit` may reshape or break it.
std::vector<uint8_t> tinyModel(const std::function<void(fb::MlpModelT&)>& edit = {})
{
    fb::MlpModelT model;
    model.num_features = 1;
    model.metric = TEST_METRIC;
    model.features_hash = TEST_HASH;

    auto input = std::make_unique<fb::MlpInputT>();
    input->feature = 0;
    input->mean = 0.0;
    input->scale = 1.0;
    model.inputs.push_back(std::move(input));

    auto layer = std::make_unique<fb::MlpLayerT>();
    layer->inputs = 1;
    layer->outputs = 1;
    layer->weights = {2.0F};
    layer->biases = {0.5F};
    auto network = std::make_unique<fb::MlpNetworkT>();
    network->layers.push_back(std::move(layer));
    model.networks.push_back(std::move(network));

    model.target_mean = 1.0;
    model.target_scale = 2.0;
    model.framework = "uhd_gen.mlp";
    if(edit)
    {
        edit(model);
    }

    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(fb::CreateMlpModel(fbb, &model), fb::MlpModelIdentifier());
    return {fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize()};
}

std::unique_ptr<MlpDataAdapter> loadTiny(const std::function<void(fb::MlpModelT&)>& edit = {})
{
    const auto buffer = tinyModel(edit);
    return MlpDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH, TEST_METRIC);
}

TEST(TestMlpDataAdapter, ScoresTheSharedFixtureAsUhdGenDoes)
{
    const auto bytes = fixtureBytes();
    ASSERT_EQ(sha256(bytes.data(), bytes.size()), FIXTURE_SHA256)
        << "the embedded fixture is not the writer's output";
    const auto adapter = loadFixture();
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->expectedFeatureCount(), 4U);
    EXPECT_EQ(adapter->getFeaturesHash(), TEST_HASH);
    EXPECT_EQ(adapter->memberCount(), 2U);

    std::vector<double> flat;
    for(const auto& row : FIXTURE_ROWS)
    {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    const auto batch = adapter->scoreBatch({flat.data(), FIXTURE_ROWS.size(), 4});
    ASSERT_EQ(batch.raw.size(), FIXTURE_SCORES.size());
    for(size_t i = 0; i < FIXTURE_ROWS.size(); ++i)
    {
        EXPECT_NEAR(batch.raw[i], FIXTURE_SCORES[i], TOLERANCE) << "row " << i;
        EXPECT_DOUBLE_EQ(adapter->score(FIXTURE_ROWS[i]), batch.raw[i]) << "row " << i;
    }
}

TEST(TestMlpDataAdapter, ARowIsStandardizedFromItsSignedLogWithAMissingValueAtZero)
{
    // score = 1 + 2 * (2 * log1p(x) + 0.5) = 2 + 4 * log1p(x), by hand.
    const auto adapter = loadTiny();
    ASSERT_NE(adapter, nullptr);
    const double e = std::exp(1.0);
    EXPECT_NEAR(adapter->score({e - 1.0}), 6.0, TOLERANCE);
    EXPECT_NEAR(adapter->score({-(e - 1.0)}), -2.0, TOLERANCE);
    EXPECT_NEAR(adapter->score({0.0}), 2.0, TOLERANCE);
    // Missing is z = 0 before it is standardized, so it scores like zero.
    EXPECT_NEAR(adapter->score({NAN_VALUE}), 2.0, TOLERANCE);
}

TEST(TestMlpDataAdapter, AMissingFlagFollowsTheInputsAndIsOneWhereTheValueIsMissing)
{
    // Input [z, flag]: output = 2 * z + 3 * flag + 0.5.
    const auto adapter = loadTiny([](fb::MlpModelT& model) {
        model.missing_flags = {0};
        model.networks[0]->layers[0]->inputs = 2;
        model.networks[0]->layers[0]->weights = {2.0F, 3.0F};
    });
    ASSERT_NE(adapter, nullptr);
    EXPECT_NEAR(adapter->score({0.0}), 2.0, TOLERANCE);
    // flag 1, z 0: 1 + 2 * (3 + 0.5).
    EXPECT_NEAR(adapter->score({NAN_VALUE}), 8.0, TOLERANCE);
}

TEST(TestMlpDataAdapter, HiddenLayersAreFollowedBySilu)
{
    // z -> [z, -z] -> SiLU -> sum: output = silu(z) + silu(-z) = z * tanh(z / 2).
    const auto adapter = loadTiny([](fb::MlpModelT& model) {
        auto& network = *model.networks[0];
        auto& first = *network.layers[0];
        first.outputs = 2;
        first.weights = {1.0F, -1.0F};
        first.biases = {0.0F, 0.0F};
        auto second = std::make_unique<fb::MlpLayerT>();
        second->inputs = 2;
        second->outputs = 1;
        second->weights = {1.0F, 1.0F};
        second->biases = {0.0F};
        network.layers.push_back(std::move(second));
        model.target_mean = 0.0;
        model.target_scale = 1.0;
    });
    ASSERT_NE(adapter, nullptr);
    for(const double x : {0.5, 2.0, 10.0, -3.0})
    {
        const double z = std::copysign(std::log1p(std::fabs(x)), x);
        EXPECT_NEAR(adapter->score({x}), z * std::tanh(z / 2.0), TOLERANCE) << "x " << x;
    }
}

TEST(TestMlpDataAdapter, ARowTheModelCannotEncodeIsDeclined)
{
    const auto adapter = loadFixture();
    ASSERT_NE(adapter, nullptr);

    // Another width.
    EXPECT_TRUE(std::isnan(adapter->score({1.0, 2.0})));
    const std::vector<double> narrow = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    const auto batch = adapter->scoreBatch({narrow.data(), 3, 2});
    ASSERT_EQ(batch.raw.size(), 3U);
    for(const double raw : batch.raw)
    {
        EXPECT_TRUE(std::isnan(raw));
    }

    // An infinity in an input slot has no standardized value...
    EXPECT_TRUE(std::isnan(adapter->score({INF, 2.0, 3.0, 4.0})));
    EXPECT_TRUE(std::isnan(adapter->score({1.0, 2.0, 3.0, -INF})));
    // ...but one in a slot no input reads does not matter, and a NaN is only missing.
    EXPECT_TRUE(std::isfinite(adapter->score({1.0, 2.0, INF, 4.0})));
    EXPECT_TRUE(std::isfinite(adapter->score({NAN_VALUE, NAN_VALUE, NAN_VALUE, NAN_VALUE})));
}

TEST(TestMlpDataAdapter, AnEmptyBatchScoresNothing)
{
    const auto adapter = loadFixture();
    ASSERT_NE(adapter, nullptr);
    EXPECT_TRUE(adapter->scoreBatch({nullptr, 0, 4}).raw.empty());
}

TEST(TestMlpDataAdapter, ArchitecturesComeFromTheArtifact)
{
    // The fixture was trained for gfx942.
    const auto fixture = loadFixture();
    ASSERT_NE(fixture, nullptr);
    EXPECT_TRUE(fixture->isTrainedForArch("gfx942"));
    EXPECT_TRUE(fixture->isTrainedForArch("gfx942:sramecc+:xnack-"));
    EXPECT_FALSE(fixture->isTrainedForArch("gfx1201"));

    // A model that records none is open to every architecture.
    const auto open = loadTiny();
    ASSERT_NE(open, nullptr);
    EXPECT_TRUE(open->isTrainedForArch("gfx1201"));

    const auto scoped
        = loadTiny([](fb::MlpModelT& model) { model.training_arches = {"gfx950", "gfx1100"}; });
    ASSERT_NE(scoped, nullptr);
    EXPECT_TRUE(scoped->isTrainedForArch("gfx1100"));
    EXPECT_FALSE(scoped->isTrainedForArch("gfx942"));
}

TEST(TestMlpDataAdapter, TheDeclaredModelHashIsVerifiedBeforeTheBytesAreRead)
{
    const auto bytes = fixtureBytes();
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_EQ(
        MlpDataAdapter::loadFromBuffer(
            bytes.data(), bytes.size(), TEST_HASH, TEST_METRIC, sha256(std::string("other bytes"))),
        nullptr);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
        << recorder.getRecordedLogsAsString();

    EXPECT_NE(MlpDataAdapter::loadFromBuffer(
                  bytes.data(), bytes.size(), TEST_HASH, TEST_METRIC, FIXTURE_SHA256),
              nullptr);
    // An empty hash skips the check.
    EXPECT_NE(MlpDataAdapter::loadFromBuffer(bytes.data(), bytes.size(), TEST_HASH, TEST_METRIC),
              nullptr);
}

TEST(TestMlpDataAdapter, AFeaturesHashOtherThanTheDescriptorsIsRefused)
{
    const auto bytes = fixtureBytes();
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_EQ(MlpDataAdapter::loadFromBuffer(
                  bytes.data(), bytes.size(), "sha256:fedcba9876543210", TEST_METRIC),
              nullptr);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
        << recorder.getRecordedLogsAsString();
}

TEST(TestMlpDataAdapter, AMetricOtherThanTheDescriptorsIsRefused)
{
    const auto bytes = fixtureBytes();
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    // The fixture predicts the log of tflops; a UHD scoring time must not load it.
    EXPECT_EQ(MlpDataAdapter::loadFromBuffer(bytes.data(), bytes.size(), TEST_HASH, "time"),
              nullptr);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
        << recorder.getRecordedLogsAsString();

    EXPECT_NE(MlpDataAdapter::loadFromBuffer(bytes.data(), bytes.size(), TEST_HASH, TEST_METRIC),
              nullptr);
}

TEST(TestMlpDataAdapter, ABufferThatIsNotAnMlpModelIsRefused)
{
    EXPECT_EQ(MlpDataAdapter::loadFromBuffer(nullptr, 100, TEST_HASH, TEST_METRIC), nullptr);
    const std::vector<uint8_t> tiny = {1, 2, 3};
    EXPECT_EQ(MlpDataAdapter::loadFromBuffer(tiny.data(), tiny.size(), TEST_HASH, TEST_METRIC),
              nullptr);

    // Another model's identifier, then the right one on a half-written file.
    auto other = fixtureBytes();
    other[4] = 'T';
    EXPECT_EQ(MlpDataAdapter::loadFromBuffer(other.data(), other.size(), TEST_HASH, TEST_METRIC),
              nullptr);

    auto truncated = fixtureBytes();
    truncated.resize(truncated.size() / 2);
    EXPECT_EQ(
        MlpDataAdapter::loadFromBuffer(truncated.data(), truncated.size(), TEST_HASH, TEST_METRIC),
        nullptr);

    EXPECT_EQ(MlpDataAdapter::load("no/such/mlp_model.bin", TEST_HASH, TEST_METRIC), nullptr);
}

struct Refusal
{
    const char* name;
    std::function<void(fb::MlpModelT&)> edit;
};

TEST(TestMlpDataAdapter, AModelThatIsInconsistentWithItselfIsRefusedWithOneError)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<Refusal> refusals = {
        {"no metric", [](fb::MlpModelT& m) { m.metric.clear(); }},
        {"no features", [](fb::MlpModelT& m) { m.num_features = 0; }},
        {"too many features", [](fb::MlpModelT& m) { m.num_features = 4097; }},
        {"input slot outside the features", [](fb::MlpModelT& m) { m.inputs[0]->feature = 1; }},
        {"negative input slot", [](fb::MlpModelT& m) { m.inputs[0]->feature = -1; }},
        {"zero scale", [](fb::MlpModelT& m) { m.inputs[0]->scale = 0.0; }},
        {"nan mean", [](fb::MlpModelT& m) { m.inputs[0]->mean = std::nan(""); }},
        {"missing flag outside the features", [](fb::MlpModelT& m) { m.missing_flags = {1}; }},
        {"no network input", [](fb::MlpModelT& m) { m.inputs.clear(); }},
        {"no networks", [](fb::MlpModelT& m) { m.networks.clear(); }},
        {"network without layers", [](fb::MlpModelT& m) { m.networks[0]->layers.clear(); }},
        {"negative target scale", [](fb::MlpModelT& m) { m.target_scale = -1.0; }},
        {"infinite target mean",
         [](fb::MlpModelT& m) { m.target_mean = std::numeric_limits<double>::infinity(); }},
        {"first layer ignores the input width",
         [](fb::MlpModelT& m) {
             m.networks[0]->layers[0]->inputs = 2;
             m.networks[0]->layers[0]->weights = {1.0F, 1.0F};
         }},
        {"last layer of two outputs",
         [](fb::MlpModelT& m) {
             m.networks[0]->layers[0]->outputs = 2;
             m.networks[0]->layers[0]->weights = {1.0F, 1.0F};
             m.networks[0]->layers[0]->biases = {0.0F, 0.0F};
         }},
        {"too few weights", [](fb::MlpModelT& m) { m.networks[0]->layers[0]->weights.clear(); }},
        {"too many biases",
         [](fb::MlpModelT& m) { m.networks[0]->layers[0]->biases = {0.0F, 0.0F}; }},
        {"nan weight", [nan](fb::MlpModelT& m) { m.networks[0]->layers[0]->weights = {nan}; }},
        {"infinite bias",
         [](fb::MlpModelT& m) {
             m.networks[0]->layers[0]->biases = {std::numeric_limits<float>::infinity()};
         }},
    };

    for(const auto& refusal : refusals)
    {
        SCOPED_TRACE(refusal.name);
        auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        EXPECT_EQ(loadTiny(refusal.edit), nullptr);
        EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
            << recorder.getRecordedLogsAsString();
    }
}

} // namespace
