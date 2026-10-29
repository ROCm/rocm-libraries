// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "GraphPatternTestSupport.hpp"
#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestDescriptors.cpp
 * @brief Tests for Descriptors.hpp: id formatting/hashing, the MetadataValue/MetadataType
 *        pairing, the descriptor spellings of the enums, and the KernelSource tagged union.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

TEST(TestIngestorDescriptors, ToStringFormatsAsCanonicalUuidText)
{
    const auto id = testId(0xAB);

    const auto text = toString(id);

    EXPECT_EQ(text.size(), 36U);
    EXPECT_EQ(text.find("ab"), 0U);
}

TEST(TestIngestorDescriptors, DescriptorIdHashIsConsistentForEqualIds)
{
    const DescriptorIdHash hash;
    const auto first = testId(0x11);
    const auto second = testId(0x11);

    EXPECT_EQ(hash(first), hash(second));
}

TEST(TestIngestorDescriptors, DescriptorIdHashDistinguishesDifferentIds)
{
    const DescriptorIdHash hash;

    EXPECT_NE(hash(testId(0x11)), hash(testId(0x22)));
}

struct MetadataTypeCase
{
    std::string name;
    MetadataValue value;
    MetadataType expectedType;
};

class TestIngestorDescriptorsMetadataTypeOf : public ::testing::TestWithParam<MetadataTypeCase>
{
};

TEST_P(TestIngestorDescriptorsMetadataTypeOf, ReportsTheVariantsAlternativeType)
{
    EXPECT_EQ(metadataTypeOf(GetParam().value), GetParam().expectedType);
}

INSTANTIATE_TEST_SUITE_P(
    EveryAlternative,
    TestIngestorDescriptorsMetadataTypeOf,
    ::testing::Values(
        MetadataTypeCase{"Bool", MetadataValue{true}, MetadataType::BOOL},
        MetadataTypeCase{"Int", MetadataValue{int64_t{42}}, MetadataType::INT},
        MetadataTypeCase{"Float", MetadataValue{1.5}, MetadataType::FLOAT},
        MetadataTypeCase{"String", MetadataValue{std::string{"FLOAT"}}, MetadataType::STRING},
        MetadataTypeCase{
            "IntList", MetadataValue{std::vector<int64_t>{1, 2, 3}}, MetadataType::INT_LIST}),
    [](const ::testing::TestParamInfo<MetadataTypeCase>& info) { return info.param.name; });

// KernelSourceKind / KernelSource: tagged union over the source kinds.

TEST(TestIngestorDescriptors, KernelSourceDefaultsToEmbeddedSource)
{
    const KernelSource source{};

    EXPECT_EQ(source.kind, KernelSourceKind::EMBEDDED_SOURCE);
    EXPECT_TRUE(source.sourceFile.empty());
    EXPECT_TRUE(source.entryPoint.empty());
    EXPECT_TRUE(source.library.empty());
    EXPECT_TRUE(source.tocKey.empty());
    EXPECT_TRUE(source.symbol.empty());
    EXPECT_TRUE(source.sha256.empty());
    EXPECT_TRUE(source.signature.empty());
}

TEST(TestIngestorDescriptors, KernelSourceCarriesEmbeddedSourceFileAndEntryPoint)
{
    KernelSource source;
    source.kind = KernelSourceKind::EMBEDDED_SOURCE;
    source.sourceFile = "PointwiseAdd.cpp";
    source.entryPoint = "pointwise_add_kernel";

    EXPECT_EQ(source.sourceFile, "PointwiseAdd.cpp");
    EXPECT_EQ(source.entryPoint, "pointwise_add_kernel");
}

TEST(TestIngestorDescriptors, CarriesTheKpackCoordinates)
{
    KernelSource source;
    source.kind = KernelSourceKind::KPACK;
    source.library = "kpack/hip_kernel_provider_gfx942.kpack";
    source.tocKey = "0f1e2d3c4b5a6978";
    source.symbol = "pointwise_add_kernel";
    source.sha256 = std::string(64, 'a');
    source.signature = {{"global_buffer", 8, 0, ""}, {"by_value", 4, 8, "count"}};

    EXPECT_EQ(source.kind, KernelSourceKind::KPACK);
    EXPECT_EQ(source.library, "kpack/hip_kernel_provider_gfx942.kpack");
    EXPECT_EQ(source.tocKey, "0f1e2d3c4b5a6978");
    EXPECT_EQ(source.symbol, "pointwise_add_kernel");
    EXPECT_EQ(source.sha256, std::string(64, 'a'));
    ASSERT_EQ(source.signature.size(), 2u);
    EXPECT_EQ(source.signature[1].kind, "by_value");
    EXPECT_EQ(source.signature[1].size, 4u);
    EXPECT_EQ(source.signature[1].offset, 8u);
    EXPECT_EQ(source.signature[1].name, "count");
    EXPECT_TRUE(source.sourceFile.empty());
    EXPECT_TRUE(source.entryPoint.empty());
}

TEST(TestIngestorDescriptors, KernelSourceKindNameParsesBackToTheSameKind)
{
    const std::vector<std::pair<KernelSourceKind, std::string>> cases{
        {KernelSourceKind::EMBEDDED_SOURCE, "embedded_source"},
        {KernelSourceKind::KPACK, "kpack"},
        {KernelSourceKind::HSACO_FILE, "hsaco_file"},
        {KernelSourceKind::ROCKE_BUILDER, "rocke_builder"},
    };

    for(const auto& [kind, name] : cases)
    {
        EXPECT_EQ(toString(kind), name);
        EXPECT_EQ(detail::kernelSourceKindFromString(toString(kind), "the test"), kind) << name;
    }
}

TEST(TestIngestorDescriptors, KernelSourceKindNameReportsAValueOutsideTheEnumAsUnknown)
{
    EXPECT_EQ(toString(static_cast<KernelSourceKind>(4)), "unknown (4)");
    EXPECT_EQ(toString(static_cast<KernelSourceKind>(-1)), "unknown (-1)");
}

TEST(TestIngestorDescriptors, MetadataTypeNameParsesBackToTheSameType)
{
    const std::vector<std::pair<MetadataType, std::string>> cases{
        {MetadataType::BOOL, "bool"},
        {MetadataType::INT, "int"},
        {MetadataType::FLOAT, "float"},
        {MetadataType::STRING, "string"},
        {MetadataType::INT_LIST, "int_list"},
    };

    for(const auto& [type, name] : cases)
    {
        EXPECT_EQ(toString(type), name);
        EXPECT_EQ(detail::metadataTypeFromString(toString(type), "the test"), type) << name;
    }
}

TEST(TestIngestorDescriptors, MetadataTypeNameReportsAValueOutsideTheEnumAsUnknown)
{
    EXPECT_EQ(toString(static_cast<MetadataType>(5)), "unknown (5)");
    EXPECT_EQ(toString(static_cast<MetadataType>(-1)), "unknown (-1)");
}

TEST(TestIngestorDescriptors, DescribesAKernelSignatureArgumentByKindWidthAndPosition)
{
    EXPECT_EQ(describeKernelSignature({}), "(no arguments)");
    EXPECT_EQ(describeKernelSignature({{"global_buffer", 8, 0, ""}, {"by_value", 4, 8, "count"}}),
              "global_buffer:8@0, by_value:4@8 'count'");
}

TEST(TestIngestorDescriptors, KernelDescriptorOriginDirectoryDefaultsToEmpty)
{
    const KernelDescriptor kernel{};

    EXPECT_TRUE(kernel.originDirectory.empty());
}

// HeuristicDescriptor: UhdAdapter as data; dispatch is tested in TestKernelHeuristic.cpp.

TEST(TestIngestorDescriptors, HeuristicDescriptorDefaultsToStaticOrder)
{
    const HeuristicDescriptor descriptor{};

    EXPECT_EQ(descriptor.adapter, UhdAdapter::STATIC_ORDER);
}

// operator== / operator!=: authored-content equality over every descriptor struct.

// Each binding names exactly as many members as its struct has, so adding a field makes the
// declaration ill-formed. Add the field to the type's operator== (or to its documented
// exclusion list), add its row to the table below, then extend the binding here.
[[maybe_unused]] void pinFieldCounts()
{
    {
        [[maybe_unused]] const auto& [f0, f1, f2] = MetadataField{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2, f3] = MetadataSchema{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2] = UhdScore{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1] = DescriptorDependency{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2] = HeuristicProvenance{};
    }
    {
        [[maybe_unused]] const auto& [f0,
                                      f1,
                                      f2,
                                      f3,
                                      f4,
                                      f5,
                                      f6,
                                      f7,
                                      f8,
                                      f9,
                                      f10,
                                      f11,
                                      f12,
                                      f13,
                                      f14,
                                      f15,
                                      f16,
                                      f17,
                                      f18,
                                      f19,
                                      f20]
            = HeuristicDescriptor{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
            = EngineDescriptor{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2, f3, f4] = MatchDescriptor{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2] = DispatchDescriptor{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2, f3] = KernelArgument{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2, f3, f4, f5, f6, f7] = KernelSource{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8] = KernelDescriptor{};
    }
    {
        [[maybe_unused]] const auto& [f0, f1, f2, f3, f4, f5, f6, f7] = KernelDescriptorPack{};
    }
}

struct EqualityOutcome
{
    bool equal;
    bool notEqual;
};

struct EqualityCase
{
    std::string name;
    /// False for a field the loader stamps or derives: changing it must leave the values equal.
    bool authored;
    std::function<EqualityOutcome()> compare;
};

/// A row that copies @p baseline, applies @p mutate to the copy, and compares the two.
template <typename T, typename Mutate>
EqualityCase fieldRow(const std::string& type,
                      const std::string& field,
                      bool authored,
                      const T& baseline,
                      Mutate mutate)
{
    return {type + field, authored, [baseline, mutate]() {
                T changed = baseline;
                mutate(changed);
                return EqualityOutcome{changed == baseline, changed != baseline};
            }};
}

const hipdnn_data_sdk::utilities::Version POPULATED_VERSION{2, 3, 4};
const hipdnn_data_sdk::utilities::Version OTHER_VERSION{5, 6, 7};

DescriptorDependency populatedDependency(uint8_t seed)
{
    return {testId(seed), POPULATED_VERSION};
}

MetadataField populatedMetadataField()
{
    MetadataField field{};
    field.name = "block";
    field.type = MetadataType::FLOAT;
    field.defaultValue = MetadataValue{2.5};
    return field;
}

HeuristicProvenance populatedProvenance()
{
    return {populatedDependency(0x21), populatedDependency(0x22), {populatedDependency(0x23)}};
}

KernelArgument populatedArgument()
{
    return {"by_value", 8U, 16U, "scale"};
}

KernelSource populatedSource()
{
    KernelSource source{};
    source.kind = KernelSourceKind::KPACK;
    source.sourceFile = "kernel.cpp";
    source.entryPoint = "entry";
    source.library = "kpack/shard.kpack";
    source.tocKey = "toc-key";
    source.symbol = "symbol";
    source.sha256 = "0123abcd";
    source.signature = {populatedArgument()};
    return source;
}

KernelDescriptor populatedKernel()
{
    KernelDescriptor kernel{};
    kernel.id = testId(0x31);
    kernel.name = "kernel";
    kernel.source = populatedSource();
    kernel.sources = {{"stage0", populatedSource()}};
    kernel.metadata = {{"block", MetadataValue{int64_t{64}}}};
    kernel.priority = 7;
    kernel.arch = {"gfx942"};
    kernel.originDirectory = "origin";
    kernel.treeRoot = "tree";
    return kernel;
}

HeuristicDescriptor populatedHeuristic()
{
    HeuristicDescriptor heuristic{};
    heuristic.id = testId(0x41);
    heuristic.name = "heuristic";
    heuristic.adapter = UhdAdapter::TREE_DATA;
    heuristic.featuresSignature = {nlohmann::json("$kernel.block"), nlohmann::json{{"op", "log"}}};
    heuristic.featuresHash = "features-hash";
    heuristic.categoricalEncoding = {{"$kernel.dtype", {{"BF16", 1}}}};
    heuristic.objective = "min";
    heuristic.score = {"latency", true, "log1p"};
    heuristic.nativeSymbol = "native.symbol";
    heuristic.modelArtifactPath = "model.bin";
    heuristic.authoredArtifactPath = "model.bin";
    heuristic.modelHash = "model-hash";
    heuristic.customLibrarySymbol = "scorer";
    heuristic.baseDir = "base";
    heuristic.treeRoot = "tree";
    heuristic.trainedAgainst = populatedProvenance();
    heuristic.trainedAgainstSelectorRevision = "selector-revision";
    heuristic.engineName = "engine";
    heuristic.role = "sort_kernel_catalog";
    heuristic.arch = "gfx942";
    heuristic.trainedAgainstJson = nlohmann::json{{"selector_revision", "selector-revision"}};
    return heuristic;
}

EngineDescriptor populatedEngine()
{
    EngineDescriptor engine{};
    engine.id = testId(0x51);
    engine.name = "vendor:engine";
    engine.heuristicId = testId(0x52);
    engine.sortKernelCatalog = {{"gfx942", {testId(0x53)}}};
    engine.predictEngine = {{"gfx942", {testId(0x54)}}};
    engine.predictApplicableKernels = {{"gfx942", testId(0x55)}};
    engine.metadataSchemaId = testId(0x56);
    engine.knobs = {"tile"};
    engine.behaviorNotes = {3};
    engine.sdkVersion = POPULATED_VERSION;
    engine.numericalNotes = {"fast-math"};
    engine.graphMatch = NativeGraphMatch{"graph.match"};
    engine.revision = POPULATED_VERSION;
    return engine;
}

std::vector<EqualityCase> allEqualityCases()
{
    std::vector<EqualityCase> rows;
    const auto append = [&rows](std::vector<EqualityCase> more) {
        rows.insert(rows.end(), more.begin(), more.end());
    };

    {
        const auto baseline = populatedMetadataField();
        const std::string type = "MetadataField";
        append(
            {fieldRow(type, "Name", true, baseline, [](auto& v) { v.name += "!"; }),
             fieldRow(type, "Type", true, baseline, [](auto& v) { v.type = MetadataType::BOOL; }),
             fieldRow(type, "DefaultValue", true, baseline, [](auto& v) {
                 v.defaultValue = MetadataValue{3.5};
             })});
    }
    {
        MetadataSchema baseline{};
        baseline.id = testId(0x01);
        baseline.name = "schema";
        baseline.fields = {populatedMetadataField()};
        baseline.revision = POPULATED_VERSION;
        const std::string type = "MetadataSchema";
        append({fieldRow(type, "Id", true, baseline, [](auto& v) { v.id = testId(0x02); }),
                fieldRow(type, "Name", true, baseline, [](auto& v) { v.name += "!"; }),
                fieldRow(type,
                         "Fields",
                         true,
                         baseline,
                         [](auto& v) { v.fields.push_back(MetadataField{}); }),
                fieldRow(type, "Revision", true, baseline, [](auto& v) {
                    v.revision = OTHER_VERSION;
                })});
    }
    {
        const UhdScore baseline{"latency", true, "log1p"};
        const std::string type = "UhdScore";
        append({fieldRow(type, "Metric", true, baseline, [](auto& v) { v.metric += "!"; }),
                fieldRow(type, "Calibrated", true, baseline, [](auto& v) { v.calibrated = false; }),
                fieldRow(
                    type, "Transform", true, baseline, [](auto& v) { v.transform = "identity"; })});
    }
    {
        const auto baseline = populatedDependency(0x11);
        const std::string type = "DescriptorDependency";
        append({fieldRow(type, "Id", true, baseline, [](auto& v) { v.id = testId(0x12); }),
                fieldRow(type, "Revision", true, baseline, [](auto& v) {
                    v.revision = OTHER_VERSION;
                })});
    }
    {
        const auto baseline = populatedProvenance();
        const std::string type = "HeuristicProvenance";
        append({fieldRow(type, "Ued", true, baseline, [](auto& v) { v.ued.id = testId(0x2A); }),
                fieldRow(type, "Kmd", true, baseline, [](auto& v) { v.kmd.id = testId(0x2B); }),
                fieldRow(type, "Umd", true, baseline, [](auto& v) {
                    v.umd.push_back(populatedDependency(0x2C));
                })});
    }
    {
        const auto baseline = populatedHeuristic();
        const std::string type = "HeuristicDescriptor";
        append({
            fieldRow(type, "Id", true, baseline, [](auto& v) { v.id = testId(0x4A); }),
            fieldRow(type, "Name", true, baseline, [](auto& v) { v.name += "!"; }),
            fieldRow(
                type, "Adapter", true, baseline, [](auto& v) { v.adapter = UhdAdapter::TABLE; }),
            fieldRow(type,
                     "FeaturesSignature",
                     true,
                     baseline,
                     [](auto& v) { v.featuresSignature.emplace_back("$kernel.extra"); }),
            fieldRow(type, "FeaturesHash", true, baseline, [](auto& v) { v.featuresHash += "!"; }),
            fieldRow(type,
                     "CategoricalEncoding",
                     true,
                     baseline,
                     [](auto& v) { v.categoricalEncoding["$kernel.dtype"]["FP32"] = 2; }),
            fieldRow(type, "Objective", true, baseline, [](auto& v) { v.objective = "max"; }),
            fieldRow(type, "Score", true, baseline, [](auto& v) { v.score.metric += "!"; }),
            fieldRow(type, "NativeSymbol", true, baseline, [](auto& v) { v.nativeSymbol += "!"; }),
            fieldRow(type,
                     "ModelArtifactPath",
                     false,
                     baseline,
                     [](auto& v) { v.modelArtifactPath += "!"; }),
            fieldRow(type,
                     "AuthoredArtifactPath",
                     true,
                     baseline,
                     [](auto& v) { v.authoredArtifactPath += "!"; }),
            fieldRow(type, "ModelHash", true, baseline, [](auto& v) { v.modelHash += "!"; }),
            fieldRow(type,
                     "CustomLibrarySymbol",
                     true,
                     baseline,
                     [](auto& v) { v.customLibrarySymbol += "!"; }),
            fieldRow(type, "BaseDir", false, baseline, [](auto& v) { v.baseDir = "other-base"; }),
            fieldRow(type, "TreeRoot", false, baseline, [](auto& v) { v.treeRoot = "other-tree"; }),
            fieldRow(type,
                     "TrainedAgainst",
                     true,
                     baseline,
                     [](auto& v) { v.trainedAgainst->ued.id = testId(0x4B); }),
            fieldRow(type,
                     "TrainedAgainstSelectorRevision",
                     true,
                     baseline,
                     [](auto& v) { v.trainedAgainstSelectorRevision += "!"; }),
            fieldRow(type, "EngineName", false, baseline, [](auto& v) { v.engineName += "!"; }),
            fieldRow(type, "Role", false, baseline, [](auto& v) { v.role = "predict_engine"; }),
            fieldRow(type, "Arch", false, baseline, [](auto& v) { v.arch = "gfx950"; }),
            fieldRow(type,
                     "TrainedAgainstJson",
                     true,
                     baseline,
                     [](auto& v) { v.trainedAgainstJson["selector_revision"] = "other"; }),
        });
    }
    {
        const auto baseline = populatedEngine();
        const std::string type = "EngineDescriptor";
        append({
            fieldRow(type, "Id", true, baseline, [](auto& v) { v.id = testId(0x5A); }),
            fieldRow(type, "Name", true, baseline, [](auto& v) { v.name += "!"; }),
            fieldRow(type,
                     "HeuristicId",
                     false,
                     baseline,
                     [](auto& v) { v.heuristicId = testId(0x5B); }),
            fieldRow(type,
                     "SortKernelCatalog",
                     true,
                     baseline,
                     [](auto& v) { v.sortKernelCatalog["gfx950"] = {testId(0x5C)}; }),
            fieldRow(type,
                     "PredictEngine",
                     true,
                     baseline,
                     [](auto& v) { v.predictEngine["gfx950"] = {testId(0x5D)}; }),
            fieldRow(type,
                     "PredictApplicableKernels",
                     true,
                     baseline,
                     [](auto& v) { v.predictApplicableKernels["gfx950"] = testId(0x5E); }),
            fieldRow(type,
                     "MetadataSchemaId",
                     true,
                     baseline,
                     [](auto& v) { v.metadataSchemaId = testId(0x5F); }),
            fieldRow(type, "Knobs", true, baseline, [](auto& v) { v.knobs.push_back("extra"); }),
            fieldRow(type,
                     "BehaviorNotes",
                     true,
                     baseline,
                     [](auto& v) { v.behaviorNotes.push_back(4); }),
            fieldRow(
                type, "SdkVersion", true, baseline, [](auto& v) { v.sdkVersion = OTHER_VERSION; }),
            fieldRow(type,
                     "NumericalNotes",
                     true,
                     baseline,
                     [](auto& v) { v.numericalNotes.push_back("extra"); }),
            fieldRow(type,
                     "GraphMatchNativeSymbol",
                     true,
                     baseline,
                     [](auto& v) { std::get<NativeGraphMatch>(v.graphMatch).symbol += "!"; }),
            fieldRow(type,
                     "GraphMatchArm",
                     true,
                     baseline,
                     [](auto& v) { v.graphMatch = compilePatternText(BINARY_POINTWISE_PATTERN); }),
            fieldRow(type, "Revision", true, baseline, [](auto& v) { v.revision = OTHER_VERSION; }),
        });
    }
    {
        const MatchDescriptor baseline{
            testId(0x61), "matcher", MatchScope::KERNEL, "match.symbol", POPULATED_VERSION};
        const std::string type = "MatchDescriptor";
        append(
            {fieldRow(type, "Id", true, baseline, [](auto& v) { v.id = testId(0x62); }),
             fieldRow(type, "Name", true, baseline, [](auto& v) { v.name += "!"; }),
             fieldRow(type, "Scope", true, baseline, [](auto& v) { v.scope = MatchScope::GRAPH; }),
             fieldRow(type, "MatchSymbol", true, baseline, [](auto& v) { v.matchSymbol += "!"; }),
             fieldRow(
                 type, "Revision", true, baseline, [](auto& v) { v.revision = OTHER_VERSION; })});
    }
    {
        const DispatchDescriptor baseline{testId(0x71), "dispatch", "dispatch.symbol"};
        const std::string type = "DispatchDescriptor";
        append({fieldRow(type, "Id", true, baseline, [](auto& v) { v.id = testId(0x72); }),
                fieldRow(type, "Name", true, baseline, [](auto& v) { v.name += "!"; }),
                fieldRow(type, "DispatchSymbol", true, baseline, [](auto& v) {
                    v.dispatchSymbol += "!";
                })});
    }
    {
        const auto baseline = populatedArgument();
        const std::string type = "KernelArgument";
        append({fieldRow(type, "Kind", true, baseline, [](auto& v) { v.kind = "global_buffer"; }),
                fieldRow(type, "Size", true, baseline, [](auto& v) { ++v.size; }),
                fieldRow(type, "Offset", true, baseline, [](auto& v) { ++v.offset; }),
                fieldRow(type, "Name", true, baseline, [](auto& v) { v.name += "!"; })});
    }
    {
        const auto baseline = populatedSource();
        const std::string type = "KernelSource";
        append({
            fieldRow(type,
                     "Kind",
                     true,
                     baseline,
                     [](auto& v) { v.kind = KernelSourceKind::HSACO_FILE; }),
            fieldRow(type, "SourceFile", true, baseline, [](auto& v) { v.sourceFile += "!"; }),
            fieldRow(type, "EntryPoint", true, baseline, [](auto& v) { v.entryPoint += "!"; }),
            fieldRow(type, "Library", true, baseline, [](auto& v) { v.library += "!"; }),
            fieldRow(type, "TocKey", true, baseline, [](auto& v) { v.tocKey += "!"; }),
            fieldRow(type, "Symbol", true, baseline, [](auto& v) { v.symbol += "!"; }),
            fieldRow(type, "Sha256", true, baseline, [](auto& v) { v.sha256 += "!"; }),
            fieldRow(type,
                     "Signature",
                     true,
                     baseline,
                     [](auto& v) { v.signature.push_back(KernelArgument{}); }),
        });
    }
    {
        const auto baseline = populatedKernel();
        const std::string type = "KernelDescriptor";
        append({
            fieldRow(type, "Id", true, baseline, [](auto& v) { v.id = testId(0x3A); }),
            fieldRow(type, "Name", true, baseline, [](auto& v) { v.name += "!"; }),
            fieldRow(type, "Source", true, baseline, [](auto& v) { v.source.symbol += "!"; }),
            fieldRow(type,
                     "Sources",
                     true,
                     baseline,
                     [](auto& v) { v.sources["stage1"] = populatedSource(); }),
            fieldRow(type,
                     "Metadata",
                     true,
                     baseline,
                     [](auto& v) { v.metadata["block"] = MetadataValue{int64_t{128}}; }),
            fieldRow(type, "Priority", true, baseline, [](auto& v) { ++v.priority; }),
            fieldRow(type, "Arch", true, baseline, [](auto& v) { v.arch.push_back("gfx950"); }),
            fieldRow(type,
                     "OriginDirectory",
                     false,
                     baseline,
                     [](auto& v) { v.originDirectory = "other-origin"; }),
            fieldRow(type, "TreeRoot", false, baseline, [](auto& v) { v.treeRoot = "other-tree"; }),
        });
    }
    {
        KernelDescriptorPack baseline{};
        baseline.id = testId(0x81);
        baseline.name = "pack";
        baseline.matcherIds = {testId(0x82)};
        baseline.engineId = testId(0x83);
        baseline.dispatchId = testId(0x84);
        baseline.arch = {"gfx942"};
        baseline.kernels = {populatedKernel()};
        baseline.kernelIds = {testId(0x85)};
        const std::string type = "KernelDescriptorPack";
        append({
            fieldRow(type, "Id", true, baseline, [](auto& v) { v.id = testId(0x8A); }),
            fieldRow(type, "Name", true, baseline, [](auto& v) { v.name += "!"; }),
            fieldRow(type,
                     "MatcherIds",
                     true,
                     baseline,
                     [](auto& v) { v.matcherIds.push_back(testId(0x8B)); }),
            fieldRow(type, "EngineId", true, baseline, [](auto& v) { v.engineId = testId(0x8C); }),
            fieldRow(
                type, "DispatchId", true, baseline, [](auto& v) { v.dispatchId = testId(0x8D); }),
            fieldRow(type, "Arch", true, baseline, [](auto& v) { v.arch.push_back("gfx950"); }),
            fieldRow(
                type, "Kernels", true, baseline, [](auto& v) { v.kernels.front().name += "!"; }),
            fieldRow(type,
                     "KernelIds",
                     true,
                     baseline,
                     [](auto& v) { v.kernelIds.push_back(testId(0x8E)); }),
        });
    }
    return rows;
}

class TestIngestorDescriptorEquality : public ::testing::TestWithParam<EqualityCase>
{
};

TEST_P(TestIngestorDescriptorEquality, ComparesAuthoredFieldsAndIgnoresStampedOnes)
{
    const auto& param = GetParam();

    const auto outcome = param.compare();

    EXPECT_EQ(outcome.equal, !param.authored);
    EXPECT_EQ(outcome.notEqual, param.authored);
}

INSTANTIATE_TEST_SUITE_P(EveryField,
                         TestIngestorDescriptorEquality,
                         ::testing::ValuesIn(allEqualityCases()),
                         [](const ::testing::TestParamInfo<EqualityCase>& info) {
                             return info.param.name;
                         });

// The packer copies a UED into every arch shard, and each parse compiles its own pattern.
// The copies must compare equal so the loader collapses them rather than dropping the id.
TEST(TestIngestorDescriptors, SeparatelyCompiledCopiesOfOnePatternAreEqual)
{
    auto lhs = populatedEngine();
    lhs.graphMatch = compileGraphPattern(nlohmann::json::parse(BINARY_POINTWISE_PATTERN),
                                         "gfx942/engine.ued.json");
    auto rhs = populatedEngine();
    rhs.graphMatch = compileGraphPattern(nlohmann::json::parse(BINARY_POINTWISE_PATTERN),
                                         "gfx950/engine.ued.json");

    EXPECT_TRUE(lhs == rhs);
    EXPECT_FALSE(lhs != rhs);
}

TEST(TestIngestorDescriptors, PatternsThatBindDifferentEdgesAreNotEqual)
{
    auto lhs = populatedEngine();
    lhs.graphMatch = compilePatternText(R"([{"kind": "op", "id": "pointwise", "op": "pointwise",
        "operands": {"in_0": "$a", "in_1": "$b"}, "results": {"out_0": "$y"}}])");
    auto rhs = populatedEngine();
    rhs.graphMatch = compilePatternText(R"([{"kind": "op", "id": "pointwise", "op": "pointwise",
        "operands": {"in_0": "$b", "in_1": "$a"}, "results": {"out_0": "$y"}}])");

    EXPECT_FALSE(lhs == rhs);
    EXPECT_TRUE(lhs != rhs);
}

// Kernel metadata: values are typed by the KMD only at resolve, so an integer and a
// floating-point spelling of one number are the same authored value.

struct MetadataEquivalenceCase
{
    std::string name;
    MetadataValues lhs;
    MetadataValues rhs;
    bool expectedEqual;
};

class TestIngestorKernelMetadataEquality : public ::testing::TestWithParam<MetadataEquivalenceCase>
{
};

TEST_P(TestIngestorKernelMetadataEquality, TreatsNumericSpellingsOfOneNumberAsEqual)
{
    KernelDescriptor lhs{};
    lhs.metadata = GetParam().lhs;
    KernelDescriptor rhs{};
    rhs.metadata = GetParam().rhs;

    EXPECT_EQ(lhs == rhs, GetParam().expectedEqual);
    EXPECT_EQ(rhs == lhs, GetParam().expectedEqual);
    EXPECT_EQ(lhs != rhs, !GetParam().expectedEqual);
}

INSTANTIATE_TEST_SUITE_P(
    Values,
    TestIngestorKernelMetadataEquality,
    ::testing::Values(
        MetadataEquivalenceCase{
            "IntEqualsFloatOfSameNumber", {{"x", int64_t{1}}}, {{"x", 1.0}}, true},
        MetadataEquivalenceCase{
            "IntDiffersFromFloatOfOtherNumber", {{"x", int64_t{1}}}, {{"x", 2.0}}, false},
        MetadataEquivalenceCase{
            "IntDiffersFromOtherInt", {{"x", int64_t{1}}}, {{"x", int64_t{2}}}, false},
        MetadataEquivalenceCase{"IntDiffersFromBool", {{"x", int64_t{1}}}, {{"x", true}}, false},
        MetadataEquivalenceCase{"FloatDiffersFromBool", {{"x", 1.0}}, {{"x", true}}, false},
        MetadataEquivalenceCase{
            "IntDiffersFromString", {{"x", int64_t{1}}}, {{"x", std::string{"1"}}}, false},
        MetadataEquivalenceCase{"IntListEqualsSameList",
                                {{"x", std::vector<int64_t>{1, 2}}},
                                {{"x", std::vector<int64_t>{1, 2}}},
                                true},
        MetadataEquivalenceCase{
            "IntListDiffersFromInt", {{"x", std::vector<int64_t>{1}}}, {{"x", int64_t{1}}}, false},
        MetadataEquivalenceCase{
            "DifferentKeysDiffer", {{"x", int64_t{1}}}, {{"y", int64_t{1}}}, false},
        MetadataEquivalenceCase{
            "ExtraKeyDiffers", {{"x", int64_t{1}}}, {{"x", int64_t{1}}, {"y", int64_t{1}}}, false},
        MetadataEquivalenceCase{"EveryEntryComparedNumerically",
                                {{"x", int64_t{1}}, {"y", 2.5}},
                                {{"x", 1.0}, {"y", 2.5}},
                                true}),
    [](const ::testing::TestParamInfo<MetadataEquivalenceCase>& info) { return info.param.name; });

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
