// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestDescriptors.cpp
 * @brief Tests for Descriptors.hpp: id formatting/hashing, the MetadataValue/MetadataType
 *        pairing, and the KernelSource tagged union.
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

    EXPECT_EQ(source.kind, KernelSourceKind::KPACK);
    EXPECT_EQ(source.library, "kpack/hip_kernel_provider_gfx942.kpack");
    EXPECT_EQ(source.tocKey, "0f1e2d3c4b5a6978");
    EXPECT_EQ(source.symbol, "pointwise_add_kernel");
    EXPECT_EQ(source.sha256, std::string(64, 'a'));
    EXPECT_TRUE(source.sourceFile.empty());
    EXPECT_TRUE(source.entryPoint.empty());
}

TEST(TestIngestorDescriptors, KernelDescriptorOriginDirectoryDefaultsToEmpty)
{
    const KernelDescriptor kernel{};

    EXPECT_TRUE(kernel.originDirectory.empty());
}

// HeuristicDescriptor: HeuristicKind as data; dispatch is tested in TestKernelHeuristic.cpp.

TEST(TestIngestorDescriptors, HeuristicDescriptorDefaultsToNativeKind)
{
    const HeuristicDescriptor descriptor{};

    EXPECT_EQ(descriptor.kind, HeuristicKind::NATIVE);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
