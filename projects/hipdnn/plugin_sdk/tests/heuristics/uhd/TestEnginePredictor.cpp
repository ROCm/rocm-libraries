// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/EnginePredictor.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>

namespace
{
using namespace hipdnn_plugin_sdk::uhd;
using hipdnn_flatbuffers_sdk::data_objects::PredictionStatus;
using hipdnn_test_sdk::utilities::GbdtModelTestBuilder;

std::atomic<size_t> scorerCalls{0};

double firstFeature(const double* values, size_t count)
{
    ++scorerCalls;
    return count == 0 ? 0.0 : values[0];
}

std::filesystem::path uniqueDirectory()
{
    static std::atomic<size_t> counter{0};
    static const auto session = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path()
           / ("engine_prediction_" + std::to_string(session) + "_" + std::to_string(counter++));
}

/// A UHD reaches an engine only through the UED role map the descriptor loader resolves
/// (RFC 0019 §3.1), so every case here hands predictEngine an already-resolved config.
class TestEnginePredictor : public ::testing::Test
{
protected:
    using Model = prediction_detail::Model;

    hipdnn_test_sdk::utilities::ScopedDirectory _directory{uniqueDirectory()};
    std::string _symbol = _directory.path().string();
    FeatureExtractionContext _features;

    void SetUp() override
    {
        NativeScorerRegistry::registerSymbol(_symbol, firstFeature);
        _features.bind("graph.work", std::log1p(42.0));
        scorerCalls = 0;
    }

    void TearDown() override
    {
        NativeScorerRegistry::unregisterSymbol(_symbol);
    }

    nlohmann::json document() const
    {
        const std::vector<nlohmann::json> signature = {"$graph.work"};
        return {{"version", "1.0"},
                {"id", "00112233-4455-6677-8899-aabbccddeeff"},
                {"name", "Engine throughput"},
                {"adapter", "native"},
                {"native", {{"symbol", _symbol}}},
                {"features_signature", signature},
                {"features_hash", FeatureExtractor::computeHash(signature)},
                {"objective", "max"},
                {"score", {{"metric", "tflops"}, {"calibrated", true}, {"transform", "log1p"}}},
                {"trained_against",
                 {{"ued", {{"id", "20112233-4455-6677-8899-aabbccddeeff"}, {"revision", "1.0"}}},
                  {"kmd", {{"id", "30112233-4455-6677-8899-aabbccddeeff"}, {"revision", "1.0"}}},
                  {"umd", nlohmann::json::array()}}}};
    }

    UhdConfig config(const nlohmann::json& doc) const
    {
        return parseUhdConfig(doc, _directory.path() / "model.uhd.json");
    }

    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT predict(const UhdConfig& cfg,
                                                                    bool evaluate = true,
                                                                    const std::string& arch
                                                                    = "gfx942") const
    {
        std::shared_ptr<const Model> compiled;
        if(evaluate)
        {
            compiled = prediction_detail::model(cfg);
        }
        return predictWith(cfg, compiled, evaluate, arch);
    }

    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT
        predictWith(const UhdConfig& cfg,
                    const std::shared_ptr<const Model>& compiled,
                    bool evaluate = true,
                    const std::string& arch = "gfx942",
                    const std::string& metric = "tflops") const
    {
        return predictEngine(
            17, "test:opaque", "selector-1", metric, arch, _features, evaluate, cfg, compiled);
    }

    /// A calibrated time model on the same scorer: objective `min` is what the metric fixes.
    UhdConfig timeConfig() const
    {
        auto doc = document();
        doc["id"] = "01112233-4455-6677-8899-aabbccddeeff";
        doc["objective"] = "min";
        doc["score"] = {{"metric", "time"}, {"calibrated", true}, {"transform", "identity"}};
        return config(doc);
    }
};

TEST_F(TestEnginePredictor, NativeCustomAndTreeRecoverTheSamePhysicalThroughput)
{
    auto native = config(document());
    const auto nativeResult = predict(native);
    ASSERT_EQ(nativeResult.status, PredictionStatus::AVAILABLE);
    EXPECT_EQ(nativeResult.metric, "tflops");
    EXPECT_NEAR(nativeResult.value, 42.0, 1e-12);

    auto custom = native;
    custom.adapterType = "custom_library";
    custom.modelArtifactPath
        = std::filesystem::absolute(
              std::filesystem::path(HIPDNN_TEST_PLUGIN_DIR)
              / hipdnn_data_sdk::utilities::getLibraryName("hipdnn_test_scorer_lib"))
              .string();
    custom.customLibrarySymbol = "test_linear_scorer";
    const auto customResult = predict(custom);
    ASSERT_EQ(customResult.status, PredictionStatus::AVAILABLE);
    EXPECT_NEAR(customResult.value, nativeResult.value, 1e-12);

    auto tree = native;
    tree.adapterType = "tree_data";
    tree.modelArtifactPath = (_directory.path() / "tree.fb").string();
    ASSERT_TRUE(GbdtModelTestBuilder()
                    .setNumFeatures(1)
                    .setFeaturesHash(tree.featuresHash)
                    .setBaseScore(std::log1p(42.0))
                    .buildToFile(tree.modelArtifactPath));
    const auto treeResult = predict(tree);
    ASSERT_EQ(treeResult.status, PredictionStatus::AVAILABLE);
    EXPECT_NEAR(treeResult.value, nativeResult.value, 1e-12);
}

/// RFC 0019 §7.2's digest is the adapter's to verify, and this is the L1 half of that.
///
/// The check used to live here, in the predictor, ahead of the factory call -- which is
/// precisely why the kernel-ranking role, which has no such preamble, dlopen'ed the same
/// library unverified. Moving it into CustomLibraryAdapter::load makes one implementation
/// serve both roles, and this pins the L1 side of that move: a mismatched digest must still
/// leave the engine without an estimate rather than quietly scoring through a substituted
/// library.
TEST_F(TestEnginePredictor, ACustomLibraryWhoseDeclaredHashIsNotItsBytesYieldsNoEstimate)
{
    auto custom = config(document());
    custom.adapterType = "custom_library";
    custom.modelArtifactPath
        = std::filesystem::absolute(
              std::filesystem::path(HIPDNN_TEST_PLUGIN_DIR)
              / hipdnn_data_sdk::utilities::getLibraryName("hipdnn_test_scorer_lib"))
              .string();
    custom.customLibrarySymbol = "test_linear_scorer";
    custom.modelHash = sha256(std::string("not this library"));

    const auto result = predict(custom);
    // INVALID, not UNAVAILABLE: §11.2 separates "I do not answer this question" from "I
    // answer, and the answer is bad". A library present under a digest it does not match is
    // the second, and reporting it as merely absent would hide a substituted artifact.
    EXPECT_EQ(result.status, PredictionStatus::INVALID);
    EXPECT_DOUBLE_EQ(result.value, 0.0);
    // Every answer names the metric it was asked in, a refusal included, so the host can
    // tell a wrong-metric answer from a right-metric refusal.
    EXPECT_EQ(result.metric, "tflops");
}

TEST_F(TestEnginePredictor, DescriptionPublishesBindingWithoutLoadingOrScoring)
{
    auto cfg = config(document());
    cfg.adapterType = "tree_data";
    cfg.modelArtifactPath = (_directory.path() / "not-deployed.fb").string();
    const auto description = predict(cfg, false);
    EXPECT_EQ(description.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(scorerCalls, 0U);
    const auto binding = nlohmann::json::parse(description.binding_json);
    EXPECT_EQ(binding.at("engine"), "test:opaque");
    EXPECT_EQ(binding.at("role"), "predict_engine");
    EXPECT_EQ(binding.at("metric"), "tflops");
    EXPECT_EQ(binding.at("selector_revision"), "selector-1");
    EXPECT_EQ(binding.at("uhd_id"), cfg.uhdId);
    // A description says what a model collected from it would be trained against. For an
    // engine with no descriptors that is the selector revision and nothing else (§4.1,
    // Open Question 7); a descriptor-backed engine adds its set on top in GenericEngine.
    // A description carrying none at all cannot be turned into a UHD, which is where every
    // opaque L1 collection stopped before this (run 67929509).
    EXPECT_EQ(binding.at("trained_against").at("selector_revision"), "selector-1");
    EXPECT_FALSE(binding.at("trained_against").contains("ued"));
    EXPECT_EQ(nlohmann::json::parse(description.features_json).at("graph.work"), std::log1p(42.0));
    EXPECT_EQ(predict(cfg).status, PredictionStatus::UNAVAILABLE);
}

/// RFC 0019 §11.2: an engine nothing binds a prediction model to -- no UED role map and
/// no UUID declared in provider code (Open Question 7) -- contributes no score, yet must
/// still describe the binding an author would train against: that description is how the
/// very first model gets collected.
TEST_F(TestEnginePredictor, EngineWithNoResolvedRoleDescribesItsBindingAndDeclinesToScore)
{
    const UhdConfig unbound;
    const auto evaluated = predictWith(unbound, nullptr);
    EXPECT_EQ(evaluated.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(evaluated.reason, "no predict_engine UHD for metric 'tflops' on arch 'gfx942'");
    EXPECT_TRUE(evaluated.binding_json.empty());
    EXPECT_EQ(scorerCalls, 0U);

    const auto description = predictWith(unbound, nullptr, false);
    const auto binding = nlohmann::json::parse(description.binding_json);
    EXPECT_EQ(binding.at("engine"), "test:opaque");
    EXPECT_EQ(binding.at("arch"), "gfx942");
    EXPECT_EQ(binding.at("selector_revision"), "selector-1");
    EXPECT_FALSE(binding.contains("uhd_id"));
    EXPECT_EQ(binding.at("trained_against").at("selector_revision"), "selector-1");
    EXPECT_EQ(nlohmann::json::parse(description.features_json).at("graph.work"), std::log1p(42.0));
}

/// The loader backfills engine/role/arch onto the config it resolved, so a model that
/// somehow names another engine or role must not be scored for this one.
TEST_F(TestEnginePredictor, LoaderBackfilledAttachmentMustAgreeWithTheAskingEngine)
{
    auto cfg = config(document());
    cfg.engineName = "test:opaque";
    cfg.role = "predict_engine";
    cfg.arch = "default";
    ASSERT_EQ(predict(cfg).status, PredictionStatus::AVAILABLE);

    cfg.engineName = "test:other";
    EXPECT_EQ(predict(cfg).status, PredictionStatus::INVALID);
    cfg.engineName = "test:opaque";
    cfg.role = "sort_kernel_catalog";
    EXPECT_EQ(predict(cfg).status, PredictionStatus::INVALID);
    cfg.role = "predict_engine";
    cfg.arch = "gfx950";
    EXPECT_EQ(predict(cfg).status, PredictionStatus::INVALID);
}

/// RFC 0019 §4.4: metrics never substitute for one another. A model of one metric asked a
/// question in another is a binding that does not match -- the number would be in the wrong
/// units -- so it is refused rather than reported, and the refusal still names the metric
/// that was asked.
TEST_F(TestEnginePredictor, AModelIsNeverAnsweredInAnotherMetric)
{
    const auto tflops = config(document());
    ASSERT_EQ(predict(tflops).status, PredictionStatus::AVAILABLE);
    const auto asTime
        = predictWith(tflops, prediction_detail::model(tflops), true, "gfx942", "time");
    EXPECT_EQ(asTime.status, PredictionStatus::INVALID);
    EXPECT_EQ(asTime.metric, "time");
    EXPECT_DOUBLE_EQ(asTime.value, 0.0);
}

/// One engine may bind one model per metric (RFC 0019 §3.1), and the arch fallback stays
/// inside the requested metric: (gfx950, time) falls back to (default, time) and never to
/// (gfx950, tflops), because that would report a throughput as a time.
TEST_F(TestEnginePredictor, BindingSelectsByMetricAndFallsBackWithinIt)
{
    EngineModelBinding binding;
    binding.bind("tflops", "default", config(document()));
    binding.bind("time", "gfx942", timeConfig());
    const auto ask = [&](const std::string& metric, const std::string& arch) {
        return binding.predict(17, "test:opaque", "selector-1", metric, arch, _features, true);
    };

    const auto tflops = ask("tflops", "gfx942");
    ASSERT_EQ(tflops.status, PredictionStatus::AVAILABLE);
    EXPECT_EQ(tflops.metric, "tflops");
    EXPECT_NEAR(tflops.value, 42.0, 1e-12);

    // The time model scores the same feature through an identity transform, so its value is
    // distinguishable from the throughput model's.
    const auto time = ask("time", "gfx942");
    ASSERT_EQ(time.status, PredictionStatus::AVAILABLE);
    EXPECT_EQ(time.metric, "time");
    EXPECT_NEAR(time.value, std::log1p(42.0), 1e-12);
    EXPECT_EQ(time.uhd_id, "01112233-4455-6677-8899-aabbccddeeff");

    const auto otherArch = ask("time", "gfx950");
    EXPECT_EQ(otherArch.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(otherArch.metric, "time");
    EXPECT_EQ(otherArch.reason, "no predict_engine UHD for metric 'time' on arch 'gfx950'");

    // A refusal is per metric too: refusing time on gfx942 leaves the tflops model answering.
    binding.markUnusable("time", "gfx942", PredictionStatus::INVALID, "refused");
    EXPECT_EQ(ask("time", "gfx942").status, PredictionStatus::INVALID);
    EXPECT_EQ(ask("tflops", "gfx942").status, PredictionStatus::AVAILABLE);
}

/// A time is valid only when strictly positive (RFC 0019 §11.4), where a throughput of 0 is
/// merely the worst one: validity is the metric's, not a single rule for every number.
TEST_F(TestEnginePredictor, ValidityIsTheRequestedMetrics)
{
    const auto cfg = timeConfig();
    _features.bind("graph.work", 0.0);
    const auto zero = predictWith(cfg, prediction_detail::model(cfg), true, "gfx942", "time");
    EXPECT_EQ(zero.status, PredictionStatus::INVALID);
    EXPECT_EQ(zero.metric, "time");

    _features.bind("graph.work", 0.5);
    const auto positive = predictWith(cfg, prediction_detail::model(cfg), true, "gfx942", "time");
    ASSERT_EQ(positive.status, PredictionStatus::AVAILABLE);
    EXPECT_DOUBLE_EQ(positive.value, 0.5);
}

/// RFC 0019 §4.1: `trained_against` names the descriptor set and only that. The engine
/// selector variant is gone, and the ued/kmd/umd triple is all-or-nothing.
TEST_F(TestEnginePredictor, DescriptorProvenanceIsRequiredWholeAndAdmitsNoEngineVariant)
{
    auto doc = document();
    doc["trained_against"].erase("kmd");
    EXPECT_THROW(config(doc), std::invalid_argument);

    doc = document();
    doc["trained_against"]["engine"] = {{"name", "test:opaque"}, {"version", "selector-1"}};
    EXPECT_THROW(config(doc), std::invalid_argument);

    doc = document();
    doc["engine"] = "test:opaque";
    EXPECT_THROW(config(doc), std::invalid_argument);
}

TEST_F(TestEnginePredictor, KernelReferenceInUnselectedBranchIsNotAnEngineModel)
{
    auto cfg = config(document());
    cfg.featuresSignature = {nlohmann::json{{"if", {true, "$graph.work", "$kernel.tile"}}}};
    cfg.featuresHash = FeatureExtractor::computeHash(cfg.featuresSignature);
    _features.bindKernelVars({{"tile", 128.0}});
    EXPECT_EQ(predict(cfg).status, PredictionStatus::INVALID);
    EXPECT_EQ(scorerCalls, 0U);
}

TEST_F(TestEnginePredictor, MissingFeatureDeclinesButLazyDefaultRetainsCoverage)
{
    auto cfg = config(document());
    _features.clear();
    EXPECT_EQ(predict(cfg).status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(scorerCalls, 0U);
    cfg.featuresSignature
        = {nlohmann::json{{"value_or_default", {"$graph.work", std::log1p(7.0)}}}};
    cfg.featuresHash = FeatureExtractor::computeHash(cfg.featuresSignature);
    const auto withDefault = predict(cfg);
    ASSERT_EQ(withDefault.status, PredictionStatus::AVAILABLE);
    EXPECT_NEAR(withDefault.value, 7.0, 1e-12);
}

TEST_F(TestEnginePredictor, InvalidScoreAndTransformNeverBecomeAvailable)
{
    auto cfg = config(document());
    _features.bind("graph.work", -1.0);
    EXPECT_EQ(predict(cfg).status, PredictionStatus::INVALID);
    _features.bind("graph.work", 1000.0);
    EXPECT_EQ(predict(cfg).status, PredictionStatus::INVALID);
    // An uninvertible transform, not merely an unusual one. `sqrt` stood here while
    // validateBinding kept its own {identity, log1p} list; it now asks
    // score_transform::isSupported, which accepts every transform this runtime can invert --
    // so pinning the gate needs a name no inverse exists for. Recovering the metric from a
    // z-score needs the training distribution's mean and variance, which no UHD carries.
    cfg.scoreTransform = "zscore";
    _features.bind("graph.work", 2.0);
    EXPECT_EQ(predict(cfg).status, PredictionStatus::INVALID);
    cfg.scoreTransform = "identity";
    cfg.scoreCalibrated = false;
    EXPECT_EQ(predict(cfg).status, PredictionStatus::INVALID);
}

/// The owning engine compiles a model once and reuses it, so a compiled model must not
/// reach back to its artifact -- and its trained-arch set still bounds coverage.
TEST_F(TestEnginePredictor, LoadedModelIsImmutableAndTrainingArchitectureLimitsCoverage)
{
    auto cfg = config(document());
    cfg.adapterType = "tree_data";
    cfg.modelArtifactPath = (_directory.path() / "cached.fb").string();
    ASSERT_TRUE(GbdtModelTestBuilder()
                    .setNumFeatures(1)
                    .setFeaturesHash(cfg.featuresHash)
                    .setBaseScore(std::log1p(9.0))
                    .setTrainingArches({"gfx942"})
                    .buildToFile(cfg.modelArtifactPath));
    const auto compiled = prediction_detail::model(cfg);
    ASSERT_EQ(predictWith(cfg, compiled, true, "gfx942:sramecc+:xnack-").status,
              PredictionStatus::AVAILABLE);
    ASSERT_TRUE(std::filesystem::remove(cfg.modelArtifactPath));
    const auto cached = predictWith(cfg, compiled);
    ASSERT_EQ(cached.status, PredictionStatus::AVAILABLE);
    EXPECT_NEAR(cached.value, 9.0, 1e-12);
    EXPECT_EQ(predictWith(cfg, compiled, true, "gfx950").status, PredictionStatus::UNAVAILABLE);
}

TEST_F(TestEnginePredictor, ParserRejectsDuplicateKeysAndOversizedNesting)
{
    const auto path = _directory.path() / "duplicate.uhd.json";
    {
        std::ofstream file(path);
        file << "{\"name\":\"first\",\"name\":\"second\"}";
    }
    EXPECT_THROW(readUhdDocument(path), std::invalid_argument);
    {
        std::ofstream file(path);
        file << std::string(200, '[') << "0" << std::string(200, ']');
    }
    EXPECT_THROW(readUhdDocument(path), std::invalid_argument);
}
} // namespace
