// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestMatchContext.cpp
 * @brief Unit tests for MatchContext.hpp: the catalog cache key's equality and hash.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing::ContentCarryingTestGraph;

CatalogKey keyFor(const ContentCarryingTestGraph::Spec& spec,
                  std::vector<int64_t> tensorUids,
                  DeviceId deviceId)
{
    const ContentCarryingTestGraph graph{spec};
    return CatalogKey{GraphContentKey{graph},
                      CatalogGraphFields::of(graph.getGraph()),
                      std::move(tensorUids),
                      deviceId};
}

/// Sets the optional graph-level fields so each can differ by presence and by value.
ContentCarryingTestGraph::Spec referenceSpec()
{
    ContentCarryingTestGraph::Spec spec;
    spec.preferredEngineId = 7;
    spec.minRequiredApiVersion = hipdnn_data_sdk::utilities::Version{1, 0, 0};
    return spec;
}

CatalogKey keyFor(const ContentCarryingTestGraph::Spec& spec)
{
    return keyFor(spec, {1, 2}, 0);
}

CatalogKey referenceKey()
{
    return keyFor(referenceSpec());
}

TEST(TestIngestorMatchContext, CatalogKeysWithEqualFieldsCompareEqual)
{
    // Built from separate graphs that differ only in name and id, which the key ignores.
    auto firstSpec = referenceSpec();
    firstSpec.graphId = makeGraphId(1);
    auto secondSpec = referenceSpec();
    secondSpec.graphId = makeGraphId(2);
    secondSpec.name = "renamed_graph";

    const CatalogKey first = keyFor(firstSpec);
    const CatalogKey second = keyFor(secondSpec);

    EXPECT_TRUE(first == second);
}

struct CatalogKeyInequalityCase
{
    std::string name;
    CatalogKey (*makeKey)();
};

class TestIngestorMatchContextCatalogKeyInequality
    : public ::testing::TestWithParam<CatalogKeyInequalityCase>
{
};

TEST_P(TestIngestorMatchContextCatalogKeyInequality, KeysDifferingInOneFieldCompareUnequal)
{
    EXPECT_FALSE(referenceKey() == GetParam().makeKey());
}

INSTANTIATE_TEST_SUITE_P(
    OneFieldAtATime,
    TestIngestorMatchContextCatalogKeyInequality,
    ::testing::Values(
        CatalogKeyInequalityCase{"DifferentGraphContent",
                                 [] {
                                     auto wider = referenceSpec();
                                     wider.tensors[0].dims = {4, 16};
                                     return keyFor(wider);
                                 }},
        CatalogKeyInequalityCase{"DifferentTensorUids",
                                 [] { return keyFor(referenceSpec(), {1000, 2000}, 0); }},
        CatalogKeyInequalityCase{"DifferentDeviceId",
                                 [] { return keyFor(referenceSpec(), {1, 2}, 1); }},
        CatalogKeyInequalityCase{"DifferentComputeDataType",
                                 [] {
                                     auto spec = referenceSpec();
                                     spec.computeDataType = DataType::HALF;
                                     return keyFor(spec);
                                 }},
        CatalogKeyInequalityCase{"DifferentIntermediateDataType",
                                 [] {
                                     auto spec = referenceSpec();
                                     spec.intermediateDataType = DataType::HALF;
                                     return keyFor(spec);
                                 }},
        CatalogKeyInequalityCase{"DifferentIoDataType",
                                 [] {
                                     auto spec = referenceSpec();
                                     spec.ioDataType = DataType::HALF;
                                     return keyFor(spec);
                                 }},
        CatalogKeyInequalityCase{"AbsentPreferredEngineId",
                                 [] {
                                     auto spec = referenceSpec();
                                     spec.preferredEngineId = std::nullopt;
                                     return keyFor(spec);
                                 }},
        CatalogKeyInequalityCase{"DifferentPreferredEngineId",
                                 [] {
                                     auto spec = referenceSpec();
                                     spec.preferredEngineId = 8;
                                     return keyFor(spec);
                                 }},
        CatalogKeyInequalityCase{"OverrideShapeEnabled",
                                 [] {
                                     auto spec = referenceSpec();
                                     spec.isOverrideShapeEnabled = true;
                                     return keyFor(spec);
                                 }},
        // The reference stamps the baseline 1.0.0 an absent field reads as, so this
        // fails if the key normalizes absence instead of keying the field as stamped.
        CatalogKeyInequalityCase{"AbsentMinRequiredEngineApiVersion",
                                 [] {
                                     auto spec = referenceSpec();
                                     spec.minRequiredApiVersion = std::nullopt;
                                     return keyFor(spec);
                                 }},
        CatalogKeyInequalityCase{"DifferentMinRequiredEngineApiVersion",
                                 [] {
                                     auto spec = referenceSpec();
                                     spec.minRequiredApiVersion
                                         = hipdnn_data_sdk::utilities::Version{1, 1, 0};
                                     return keyFor(spec);
                                 }}),
    [](const ::testing::TestParamInfo<CatalogKeyInequalityCase>& info) { return info.param.name; });

TEST(TestIngestorMatchContext, CatalogKeyHashIsConsistentForEqualKeys)
{
    const CatalogKeyHash hash;

    EXPECT_EQ(hash(referenceKey()), hash(referenceKey()));
}

TEST(TestIngestorMatchContext, CatalogKeyHashDistinguishesDifferentDeviceIds)
{
    const CatalogKeyHash hash;
    const CatalogKey onDeviceZero = keyFor(ContentCarryingTestGraph::Spec{}, {1, 2}, 0);
    const CatalogKey onDeviceOne = keyFor(ContentCarryingTestGraph::Spec{}, {1, 2}, 1);

    EXPECT_NE(hash(onDeviceZero), hash(onDeviceOne));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
