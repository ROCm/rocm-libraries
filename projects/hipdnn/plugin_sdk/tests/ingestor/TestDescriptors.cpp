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
 * @brief Unit tests for Descriptors.hpp: id formatting and hashing, the unified
 *        MetadataValue/MetadataType pairing over vector<int64_t>, and the KernelSource
 *        tagged union.
 *
 * makeKernelHeuristic() is declared in IKernelHeuristic.hpp, not here, so its tests live
 * in TestKernelHeuristic.cpp; this file covers HeuristicDescriptor purely as data.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

// ---------------------------------------------------------------------------
// toString(DescriptorId) / DescriptorIdHash
// ---------------------------------------------------------------------------

TEST(TestIngestorDescriptors, ToStringFormatsAsCanonicalUuidText)
{
    const auto id = testId(0xAB);

    const auto text = toString(id);

    // Canonical UUID text form: 32 hex digits plus 4 hyphens, 36 characters.
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

// ---------------------------------------------------------------------------
// metadataTypeOf: MetadataType must track MetadataValue's variant order exactly, one
// alternative at a time, so a mismatch here means the enum and the variant have drifted.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// KernelSourceKind / KernelSource: the tagged union over RFC 0017 §7's source kinds.
// ---------------------------------------------------------------------------

TEST(TestIngestorDescriptors, KernelSourceDefaultsToEmbeddedSource)
{
    // The only kind this POC implements, and the kind every existing fixture kernel
    // relies on defaulting to.
    const KernelSource source{};

    EXPECT_EQ(source.kind, KernelSourceKind::EMBEDDED_SOURCE);
    EXPECT_TRUE(source.sourceFile.empty());
    EXPECT_TRUE(source.entryPoint.empty());
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

// ---------------------------------------------------------------------------
// HeuristicDescriptor: HeuristicKind as data. The adapter dispatch on it is
// makeKernelHeuristic()'s behavior, tested in TestKernelHeuristic.cpp.
// ---------------------------------------------------------------------------

TEST(TestIngestorDescriptors, HeuristicDescriptorDefaultsToNativeKind)
{
    const HeuristicDescriptor descriptor{};

    EXPECT_EQ(descriptor.kind, HeuristicKind::NATIVE);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
