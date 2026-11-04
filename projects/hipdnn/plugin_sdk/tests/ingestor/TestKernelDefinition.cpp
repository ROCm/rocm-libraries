// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <filesystem>
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
 * @brief Tests for KernelDefinition.hpp: tryGetMetadata()'s optional contract, and the
 *        typed getters' happy path and throw branches.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

KernelDefinition makeKernelWithMetadata(MetadataValues metadata)
{
    KernelDefinition kernel;
    kernel.kernelId = testId(0x01);
    kernel.packId = PACK_ID;
    kernel.dispatchId = DISPATCH_ID;
    kernel.source = makeEmbeddedSource();
    kernel.metadata = std::move(metadata);
    return kernel;
}

TEST(TestIngestorKernelDefinition, TryGetMetadataReturnsTheValueWhenPresent)
{
    const auto kernel = makeKernelWithMetadata({{BLOCK_SIZE, MetadataValue{int64_t{64}}}});

    const auto value = kernel.tryGetMetadata(BLOCK_SIZE);

    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(std::get<int64_t>(*value), 64);
}

TEST(TestIngestorKernelDefinition, TryGetMetadataReturnsNulloptWhenAbsent)
{
    const auto kernel = makeKernelWithMetadata({});

    EXPECT_EQ(kernel.tryGetMetadata(BLOCK_SIZE), std::nullopt);
}

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

TEST(TestIngestorKernelDefinition, CarriesTheKpackCoordinates)
{
    const std::filesystem::path origin("/descriptors/gfx942");

    const auto kernel = makeKpackDefinition(testId(0x02), 64, origin);

    EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK);
    EXPECT_EQ(kernel.source.library, "kpack/hip_kernel_provider_gfx942.kpack");
    EXPECT_EQ(kernel.source.tocKey, "test-toc-key");
    EXPECT_EQ(kernel.source.symbol, "TestKernel");
    EXPECT_EQ(kernel.source.sha256, std::string(64, 'a'));
    EXPECT_EQ(kernel.originDirectory, origin);
    EXPECT_EQ(kernel.name, "kpack kernel");
}

// Throw matrix: each typed getter x field-absent/wrong-alternative.
struct KernelDefinitionThrowCase
{
    std::string name;
    std::function<KernelDefinition()> makeKernel;
    std::function<void(const KernelDefinition&)> callGetter;
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
