// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/heuristics/uhd/UhdParser.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

/**
 * @file TestUhdParser.cpp
 * @brief The common UHD format's admission rules, read by parseUhdConfig and readUhdDocument.
 *
 * Each refusal case is a valid document changed in one place, so the refusal can only come
 * from the rule that place breaks; AcceptsEveryBaseDocument proves each starting point
 * valid. A model path escaping its directory through a symlink needs one on disk and is
 * covered by TestEnginePredictor.ArtifactSymlinksMustStayInsideTheirDescriptorDirectory.
 */
namespace
{
using namespace hipdnn_plugin_sdk::uhd;
using hipdnn_plugin_sdk::uhd::parser_detail::MAX_DOCUMENT_BYTES;
using hipdnn_plugin_sdk::uhd::parser_detail::MAX_DOCUMENT_DEPTH;
using hipdnn_plugin_sdk::uhd::parser_detail::MAX_DOCUMENT_NODES;

const std::string DIGEST = std::string(64, 'a');
const std::string MATCHER_ID = "40112233-4455-6677-8899-aabbccddeeff";

std::filesystem::path uniqueDirectory()
{
    static std::atomic<size_t> s_counter{0};
    static const auto s_session = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path()
           / ("uhd_parser_" + std::to_string(s_session) + "_" + std::to_string(s_counter++));
}

/// Where an in-memory document claims to live. Never created: a model path resolves
/// against it lexically, and every model body declares its hash, so nothing is read.
std::filesystem::path descriptorPath()
{
    return std::filesystem::temp_directory_path() / "uhd_parser" / "model.uhd.json";
}

enum class Base
{
    NATIVE,
    TREE_DATA,
    CUSTOM_LIBRARY,
    STATIC_ORDER
};

/// A valid document of each body kind. The native one carries every optional block a
/// refusal case mutates: a feature contract, a calibrated score and a descriptor set
/// recording one matcher.
nlohmann::json baseDocument(Base base)
{
    if(base == Base::STATIC_ORDER)
    {
        return {{"version", "1.0"},
                {"id", "00112233-4455-6677-8899-aabbccddeeff"},
                {"name", "Declared order"},
                {"adapter", "static_order"},
                {"static_order", nlohmann::json::object()}};
    }
    const std::vector<nlohmann::json> signature = {"$graph.work"};
    nlohmann::json document
        = {{"version", "1.0"},
           {"id", "00112233-4455-6677-8899-aabbccddeeff"},
           {"name", "Engine throughput"},
           {"adapter", "native"},
           {"native", {{"symbol", "uhd_parser.scorer"}}},
           {"features_signature", signature},
           {"features_hash", FeatureExtractor::computeHash(signature)},
           {"objective", "max"},
           {"score", {{"metric", "tflops"}, {"calibrated", true}, {"transform", "log1p"}}},
           {"trained_against",
            {{"ued", {{"id", "20112233-4455-6677-8899-aabbccddeeff"}, {"revision", "1.0"}}},
             {"kmd", {{"id", "30112233-4455-6677-8899-aabbccddeeff"}, {"revision", "1.0"}}},
             {"umd", {{{"id", MATCHER_ID}, {"revision", "1.0"}}}}}}};
    if(base == Base::TREE_DATA)
    {
        document["adapter"] = "tree_data";
        document.erase("native");
        document["tree_data"] = {{"artifact", "models/model.fb"}, {"hash", DIGEST}};
    }
    else if(base == Base::CUSTOM_LIBRARY)
    {
        document["adapter"] = "custom_library";
        document.erase("native");
        document["custom_library"]
            = {{"library", "lib/scorer.so"}, {"symbol", "scorer"}, {"hash", DIGEST}};
    }
    return document;
}

/// Nested arrays reaching one level past what either bound admits.
nlohmann::json nestedBeyondDepthBound()
{
    nlohmann::json value = 0;
    for(size_t level = 0; level <= MAX_DOCUMENT_DEPTH; ++level)
    {
        value = nlohmann::json::array({std::move(value)});
    }
    return value;
}

struct Refusal
{
    /// @param textEdit Applied to the serialized document, which only readUhdDocument then
    ///        reads, so the refusal is the reader's own. When null, the edited document is
    ///        parsed in memory.
    Refusal(std::string caseName,
            Base caseBase,
            std::function<void(nlohmann::json&)> documentEdit,
            std::function<void(std::string&)> textEdit = nullptr)
        : name(std::move(caseName))
        , base(caseBase)
        , edit(std::move(documentEdit))
        , editText(std::move(textEdit))
    {
    }

    std::string name;
    Base base;
    std::function<void(nlohmann::json&)> edit;
    std::function<void(std::string&)> editText;
};

class TestUhdParserRefusal : public ::testing::TestWithParam<Refusal>
{
};

TEST_P(TestUhdParserRefusal, RefusesTheDocument)
{
    const auto& refusal = GetParam();
    auto document = baseDocument(refusal.base);
    if(refusal.edit)
    {
        refusal.edit(document);
    }
    if(!refusal.editText)
    {
        EXPECT_THROW(parseUhdConfig(document, descriptorPath()), std::invalid_argument);
        return;
    }
    const hipdnn_test_sdk::utilities::ScopedDirectory directory(uniqueDirectory());
    const auto path = directory.path() / "model.uhd.json";
    auto text = document.dump();
    refusal.editText(text);
    {
        std::ofstream file(path, std::ios::binary);
        file << text;
    }
    EXPECT_THROW(readUhdDocument(path), std::invalid_argument);
}

const std::vector<Refusal> REFUSALS = {
    // readUhdDocument
    {"EmptyFile", Base::NATIVE, nullptr, [](std::string& text) { text.clear(); }},
    {"FileOverTheSizeBound",
     Base::NATIVE,
     nullptr,
     [](std::string& text) { text.append(MAX_DOCUMENT_BYTES, ' '); }},
    {"FileNestedBeyondTheDepthBound",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["x-deep"] = nestedBeyondDepthBound(); },
     [](std::string&) {}},
    {"DuplicateKey",
     Base::NATIVE,
     nullptr,
     [](std::string& text) { text.insert(1, R"("name":"shadow",)"); }},
    // In-memory structural bounds
    {"DocumentNestedBeyondTheDepthBound",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["x-deep"] = nestedBeyondDepthBound(); }},
    {"DocumentOverTheNodeBound",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["x-wide"] = std::vector<int>(MAX_DOCUMENT_NODES, 0); }},
    {"DocumentOverTheSizeBound",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["x-long"] = std::string(MAX_DOCUMENT_BYTES, 'x'); }},
    // Root shape
    {"RootIsNotAnObject", Base::NATIVE, [](nlohmann::json& doc) { doc = nlohmann::json::array(); }},
    {"UnknownRootKey", Base::NATIVE, [](nlohmann::json& doc) { doc["featuresHash"] = "x"; }},
    {"MissingVersion", Base::NATIVE, [](nlohmann::json& doc) { doc.erase("version"); }},
    {"UnsupportedVersion", Base::NATIVE, [](nlohmann::json& doc) { doc["version"] = "1.1"; }},
    {"IdIsNotAUuid", Base::NATIVE, [](nlohmann::json& doc) { doc["id"] = "model-1"; }},
    {"EmptyName", Base::NATIVE, [](nlohmann::json& doc) { doc["name"] = ""; }},
    {"NameIsNotAString", Base::NATIVE, [](nlohmann::json& doc) { doc["name"] = 3; }},
    // Exactly one body, matching the adapter
    {"NoBody", Base::NATIVE, [](nlohmann::json& doc) { doc.erase("native"); }},
    {"TwoBodies",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["static_order"] = nlohmann::json::object(); }},
    {"AdapterNamesAnotherBody", Base::NATIVE, [](nlohmann::json& doc) { doc["adapter"] = "onnx"; }},
    {"AdapterIsUnknown", Base::NATIVE, [](nlohmann::json& doc) { doc["adapter"] = "xgboost"; }},
    // Feature contract
    {"EmptySignature",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["features_signature"] = nlohmann::json::array(); }},
    {"SignatureEntryIsABareName",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["features_signature"][0] = "graph.work"; }},
    {"SignatureEntryIsAnObjectWithTwoKeys",
     Base::NATIVE,
     [](nlohmann::json& doc) {
         doc["features_signature"][0] = {{"log1p", "$graph.work"}, {"sqrt", "$graph.work"}};
     }},
    {"FeaturesHashWithoutPrefix",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["features_hash"] = "0123456789abcdef"; }},
    {"FeaturesHashInUppercase",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["features_hash"] = "sha256:0123456789ABCDEF"; }},
    {"FeaturesHashTooShort",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["features_hash"] = "sha256:0123456789abcde"; }},
    {"SignatureWithoutFeaturesHash",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc.erase("features_hash"); }},
    {"SignatureWithoutTrainedAgainst",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc.erase("trained_against"); }},
    // Categorical encoding
    {"CategoricalEncodingIsNotAnObject",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["categorical_encoding"] = nlohmann::json::array(); }},
    {"CategoricalFieldIsNotAnObject",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["categorical_encoding"] = {{"$q.dtype", {"bf16"}}}; }},
    {"CategoricalFieldIsNotAReference",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["categorical_encoding"] = {{"q.dtype", {{"bf16", 1}}}}; }},
    {"CategoricalVocabularyIsEmpty",
     Base::NATIVE,
     [](nlohmann::json& doc) {
         doc["categorical_encoding"] = {{"$q.dtype", nlohmann::json::object()}};
     }},
    {"CategoricalCodeAboveInt32",
     Base::NATIVE,
     [](nlohmann::json& doc) {
         doc["categorical_encoding"]
             = {{"$q.dtype", {{"bf16", int64_t{std::numeric_limits<int32_t>::max()} + 1}}}};
     }},
    {"CategoricalCodeBelowInt32",
     Base::NATIVE,
     [](nlohmann::json& doc) {
         doc["categorical_encoding"]
             = {{"$q.dtype", {{"bf16", int64_t{std::numeric_limits<int32_t>::min()} - 1}}}};
     }},
    {"CategoricalCodeIsNotAnInteger",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["categorical_encoding"] = {{"$q.dtype", {{"bf16", 1.5}}}}; }},
    // Objective and score
    {"ScoringAdapterWithoutObjective",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc.erase("objective"); }},
    {"ObjectiveIsNeitherMaxNorMin",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["objective"] = "maximize"; }},
    {"ObjectiveContradictsTheMetric",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["objective"] = "min"; }},
    {"UnknownScoreKey",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["score"]["units"] = "TFLOPS"; }},
    {"UnregisteredMetric",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["score"]["metric"] = "gflops"; }},
    {"UnsupportedTransform",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["score"]["transform"] = "log2"; }},
    {"CalibratedIsNotABoolean",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["score"]["calibrated"] = "true"; }},
    {"CalibratedWithoutAMetric",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["score"].erase("metric"); }},
    // Provenance
    {"UnknownProvenanceKey",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"]["kdp"] = doc["trained_against"]["kmd"]; }},
    {"ProvenanceNamesNeitherForm",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"] = {{"feature_semantics_revision", 1}}; }},
    {"EmptySelectorRevision",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"]["selector_revision"] = ""; }},
    {"PartialDescriptorSet",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"].erase("kmd"); }},
    {"DependencyIdIsNotAUuid",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"]["ued"]["id"] = "engine-1"; }},
    {"DependencyRevisionIsNotMajorMinor",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"]["ued"]["revision"] = "1"; }},
    {"UmdIsNotAnArray",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"]["umd"] = doc["trained_against"]["umd"][0]; }},
    {"MatcherRecordedTwiceInAnotherCase",
     Base::NATIVE,
     [](nlohmann::json& doc) {
         doc["trained_against"]["umd"].push_back(
             {{"id", "40112233-4455-6677-8899-AABBCCDDEEFF"}, {"revision", "1.0"}});
     }},
    {"FeatureSemanticsRevisionIsZero",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"]["feature_semantics_revision"] = 0; }},
    {"FeatureSemanticsRevisionIsAFloat",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"]["feature_semantics_revision"] = 1.0; }},
    {"FeatureSemanticsRevisionIsABoolean",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["trained_against"]["feature_semantics_revision"] = true; }},
    // Bodies
    {"NativeWithoutASymbol",
     Base::NATIVE,
     [](nlohmann::json& doc) { doc["native"].erase("symbol"); }},
    {"UnknownBodyKey", Base::NATIVE, [](nlohmann::json& doc) { doc["native"]["library"] = "x"; }},
    {"StaticOrderDeclaresAnOrder",
     Base::STATIC_ORDER,
     [](nlohmann::json& doc) { doc["static_order"]["order"] = {"priority"}; }},
    {"StaticOrderBodyHasAnUnknownKey",
     Base::STATIC_ORDER,
     [](nlohmann::json& doc) { doc["static_order"]["priority"] = "desc"; }},
    {"ModelWithoutASignature",
     Base::TREE_DATA,
     [](nlohmann::json& doc) { doc.erase("features_signature"); }},
    {"ArtifactOutsideTheDescriptorDirectory",
     Base::TREE_DATA,
     [](nlohmann::json& doc) { doc["tree_data"]["artifact"] = "../model.fb"; }},
    {"ArtifactHashInUppercase",
     Base::TREE_DATA,
     [](nlohmann::json& doc) { doc["tree_data"]["hash"] = std::string(64, 'A'); }},
    {"CustomLibraryWithoutAHash",
     Base::CUSTOM_LIBRARY,
     [](nlohmann::json& doc) { doc["custom_library"].erase("hash"); }},
    {"CustomLibraryWithConfiguration",
     Base::CUSTOM_LIBRARY,
     [](nlohmann::json& doc) { doc["custom_library"]["config"] = {{"threads", 4}}; }},
};

INSTANTIATE_TEST_SUITE_P(Rule,
                         TestUhdParserRefusal,
                         ::testing::ValuesIn(REFUSALS),
                         [](const ::testing::TestParamInfo<Refusal>& info) {
                             return info.param.name;
                         });

TEST(TestUhdParser, AcceptsEveryBaseDocument)
{
    const auto native = parseUhdConfig(baseDocument(Base::NATIVE), descriptorPath());
    EXPECT_EQ(native.nativeSymbol, "uhd_parser.scorer");
    EXPECT_EQ(native.scoreMetric, "tflops");
    EXPECT_TRUE(native.scoreCalibrated);
    EXPECT_EQ(native.objective, "max");

    const auto tree = parseUhdConfig(baseDocument(Base::TREE_DATA), descriptorPath());
    EXPECT_EQ(tree.modelHash, DIGEST);
    EXPECT_EQ(std::filesystem::path(tree.modelArtifactPath).filename().string(), "model.fb");

    const auto custom = parseUhdConfig(baseDocument(Base::CUSTOM_LIBRARY), descriptorPath());
    EXPECT_EQ(custom.customLibrarySymbol, "scorer");
    EXPECT_EQ(custom.modelHash, DIGEST);

    // static_order scores nothing, so it is admitted without an objective.
    const auto ordered = parseUhdConfig(baseDocument(Base::STATIC_ORDER), descriptorPath());
    EXPECT_EQ(ordered.adapterType, "static_order");
}

/// The boundaries the int32 refusals sit next to are themselves accepted.
TEST(TestUhdParser, AcceptsCategoricalCodesAtTheInt32Limits)
{
    auto document = baseDocument(Base::NATIVE);
    document["categorical_encoding"] = {{"$q.dtype",
                                         {{"lowest", std::numeric_limits<int32_t>::min()},
                                          {"highest", std::numeric_limits<int32_t>::max()}}}};
    const auto config = parseUhdConfig(document, descriptorPath());
    const auto& codes = config.categoricalEncoding.at("$q.dtype");
    EXPECT_EQ(codes.at("lowest"), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(codes.at("highest"), std::numeric_limits<int32_t>::max());
}

/// The authored form: comments are stripped, and `x-`/`_` keys and the free-form root
/// `provenance` block are carried without being read.
TEST(TestUhdParser, ReadsTheAuthoredForm)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory directory(uniqueDirectory());
    const auto path = directory.path() / "model.uhd.json";
    auto document = baseDocument(Base::NATIVE);
    document["x-owner"] = "team";
    document["_comment"] = "tracking only";
    document["provenance"] = {{"collections", {"nightly"}}};
    document["score"]["x-note"] = "calibrated on gfx942";
    {
        std::ofstream file(path, std::ios::binary);
        file << "// authored by hand\n" << document.dump(2) << "\n/* trailing note */\n";
    }

    const auto config = parseUhdConfig(readUhdDocument(path), path);

    EXPECT_EQ(config.uhdId, document.at("id").get<std::string>());
    EXPECT_EQ(config.nativeSymbol, "uhd_parser.scorer");
    EXPECT_EQ(config.scoreMetric, "tflops");
}

TEST(TestUhdParser, RefusesADocumentThatCannotBeRead)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory directory(uniqueDirectory());
    EXPECT_THROW(readUhdDocument(directory.path() / "absent.uhd.json"), std::invalid_argument);
}

} // namespace
