// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestKernelDefinition.cpp
 * @brief Unit tests for KernelDefinition.hpp: tryGetMetadata()'s optional contract, and
 *        the three typed getters (int, string, int-list) over both their happy path and
 *        their two throw branches (absent field, wrong alternative).
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

KernelDefinition makeKernelWithMetadata(MetadataValues metadata)
{
    return {testId(0x01),
            PACK_ID,
            DISPATCH_ID,
            KernelSource{KernelSourceKind::EMBEDDED_SOURCE, "Test.cpp", "TestKernel"},
            std::move(metadata),
            /*priority=*/0};
}

// ---------------------------------------------------------------------------
// tryGetMetadata
// ---------------------------------------------------------------------------

TEST(TestIngestorKernelDefinition, TryGetMetadataReturnsTheValueWhenPresent)
{
    const auto kernel = makeKernelWithMetadata({{BLOCK_SIZE, MetadataValue{int64_t{64}}}});

    const auto value = kernel.tryGetMetadata(BLOCK_SIZE);

    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(std::get<int64_t>(*value), 64);
}

TEST(TestIngestorKernelDefinition, TryGetMetadataReturnsNulloptWhenAbsent)
{
    // Nullopt rather than throwing, so a matcher or scorer written against a newer
    // schema can ask about a field an older kernel predates.
    const auto kernel = makeKernelWithMetadata({});

    EXPECT_EQ(kernel.tryGetMetadata(BLOCK_SIZE), std::nullopt);
}

// ---------------------------------------------------------------------------
// Typed accessor happy path: one per alternative getIntMetadata / getStringMetadata /
// getIntListMetadata reads.
// ---------------------------------------------------------------------------

TEST(TestIngestorKernelDefinition, GetIntMetadataReturnsTheIntegerValue)
{
    const auto kernel = makeKernelWithMetadata({{BLOCK_SIZE, MetadataValue{int64_t{256}}}});

    EXPECT_EQ(kernel.getIntMetadata(BLOCK_SIZE), 256);
}

TEST(TestIngestorKernelDefinition, GetStringMetadataReturnsTheStringValue)
{
    const auto kernel = makeKernelWithMetadata({{DTYPE, MetadataValue{std::string{"FLOAT"}}}});

    EXPECT_EQ(kernel.getStringMetadata(DTYPE), "FLOAT");
}

TEST(TestIngestorKernelDefinition, GetIntListMetadataReturnsTheListValue)
{
    constexpr const char* STRIDE_ORDER = "stride_order";
    const auto kernel
        = makeKernelWithMetadata({{STRIDE_ORDER, MetadataValue{std::vector<int64_t>{3, 1, 2, 0}}}});

    EXPECT_EQ(kernel.getIntListMetadata(STRIDE_ORDER), (std::vector<int64_t>{3, 1, 2, 0}));
}

// ---------------------------------------------------------------------------
// Throw matrix: each of the three typed getters, crossed with "field absent"
// (out_of_range) and "field holds a different alternative" (invalid_argument). Both are
// author errors a validating loader would have caught, so they throw rather than
// returning a default that would silently mis-rank or mis-launch a kernel.
// ---------------------------------------------------------------------------

struct KernelDefinitionThrowCase
{
    std::string name;
    /// Builds the kernel this case exercises. A function rather than a stored
    /// KernelDefinition so each case can supply metadata shaped for exactly the
    /// alternative it is testing.
    std::function<KernelDefinition()> makeKernel;
    /// Invokes the getter under test against the kernel. A function so the matrix can
    /// share one fixture shape across three differently-typed accessors.
    std::function<void(const KernelDefinition&)> callGetter;
    /// True when the getter must throw std::out_of_range (field absent); false when it
    /// must throw std::invalid_argument (wrong alternative). RTTI is disabled in this
    /// build, so the matrix distinguishes exception types via EXPECT_THROW's static
    /// type rather than dynamic_cast.
    bool expectsOutOfRange;
};

class TestIngestorKernelDefinitionThrowMatrix
    : public ::testing::TestWithParam<KernelDefinitionThrowCase>
{
};

TEST_P(TestIngestorKernelDefinitionThrowMatrix, ThrowsOnTheExpectedFailureMode)
{
    const auto& testCase = GetParam();
    const auto kernel = testCase.makeKernel();

    if(testCase.expectsOutOfRange)
    {
        EXPECT_THROW(testCase.callGetter(kernel), std::out_of_range);
    }
    else
    {
        EXPECT_THROW(testCase.callGetter(kernel), std::invalid_argument);
    }
}

INSTANTIATE_TEST_SUITE_P(
    EveryAccessorTimesEveryFailureMode,
    TestIngestorKernelDefinitionThrowMatrix,
    ::testing::Values(
        KernelDefinitionThrowCase{
            "GetIntMetadataAbsentFieldThrowsOutOfRange",
            [] { return makeKernelWithMetadata({}); },
            [](const KernelDefinition& kernel) { kernel.getIntMetadata(BLOCK_SIZE); },
            /*expectsOutOfRange=*/true},
        KernelDefinitionThrowCase{
            "GetIntMetadataWrongAlternativeThrowsInvalidArgument",
            [] { return makeKernelWithMetadata({{BLOCK_SIZE, MetadataValue{std::string{"64"}}}}); },
            [](const KernelDefinition& kernel) { kernel.getIntMetadata(BLOCK_SIZE); },
            /*expectsOutOfRange=*/false},
        KernelDefinitionThrowCase{
            "GetStringMetadataAbsentFieldThrowsOutOfRange",
            [] { return makeKernelWithMetadata({}); },
            [](const KernelDefinition& kernel) { kernel.getStringMetadata(DTYPE); },
            /*expectsOutOfRange=*/true},
        KernelDefinitionThrowCase{
            "GetStringMetadataWrongAlternativeThrowsInvalidArgument",
            [] { return makeKernelWithMetadata({{DTYPE, MetadataValue{int64_t{1}}}}); },
            [](const KernelDefinition& kernel) { kernel.getStringMetadata(DTYPE); },
            /*expectsOutOfRange=*/false},
        KernelDefinitionThrowCase{
            "GetIntListMetadataAbsentFieldThrowsOutOfRange",
            [] { return makeKernelWithMetadata({}); },
            [](const KernelDefinition& kernel) { kernel.getIntListMetadata("stride_order"); },
            /*expectsOutOfRange=*/true},
        KernelDefinitionThrowCase{
            "GetIntListMetadataWrongAlternativeThrowsInvalidArgument",
            [] { return makeKernelWithMetadata({{"stride_order", MetadataValue{int64_t{1}}}}); },
            [](const KernelDefinition& kernel) { kernel.getIntListMetadata("stride_order"); },
            /*expectsOutOfRange=*/false}),
    [](const ::testing::TestParamInfo<KernelDefinitionThrowCase>& info) {
        return info.param.name;
    });

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
