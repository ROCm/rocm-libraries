// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseAddPack.cpp
 * @brief The pack's descriptor data: its shape, its cross-references, and its engine id.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestPointwiseAddPack, ShipsThreeKernelsCoveringTwoBlockSizesAndTwoDataTypes)
{
    const auto set = buildPointwiseAddDescriptorSet();

    ASSERT_EQ(set.packs.size(), 1U);
    const auto& kernels = set.packs.front().kernels;
    ASSERT_EQ(kernels.size(), 3U);

    // The two FLOAT kernels differ only in block size; HALF is pruned on a FLOAT graph.
    const auto describes = [&kernels](int64_t blockSize, const std::string& dtype) {
        return std::any_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
            return std::get<int64_t>(kernel.metadata.at(std::string(BLOCK_SIZE_FIELD))) == blockSize
                   && std::get<std::string>(kernel.metadata.at(std::string(DTYPE_FIELD))) == dtype;
        });
    };

    EXPECT_TRUE(describes(64, "FLOAT"));
    EXPECT_TRUE(describes(256, "FLOAT"));
    EXPECT_TRUE(describes(64, "HALF"));
}

TEST(TestPointwiseAddPack, EveryKernelNamesTheEmbeddedSource)
{
    const auto set = buildPointwiseAddDescriptorSet();

    for(const auto& kernel : set.packs.front().kernels)
    {
        EXPECT_EQ(kernel.source.kind,
                  hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE);
        EXPECT_EQ(kernel.source.sourceFile, "PointwiseAdd.cpp");
        EXPECT_EQ(kernel.source.entryPoint, "PointwiseAdd");
    }
}

TEST(TestPointwiseAddPack, PackCrossReferencesResolveWithinTheDescriptorSet)
{
    const auto set = buildPointwiseAddDescriptorSet();
    const auto& pack = set.packs.front();

    // A pack names its engine, matchers, and dispatch by id; every one must resolve.
    EXPECT_EQ(pack.engineId, set.engine.id);

    ASSERT_EQ(pack.matcherIds.size(), 2U);
    for(const auto& matcherId : pack.matcherIds)
    {
        EXPECT_TRUE(std::any_of(set.matchers.begin(), set.matchers.end(), [&](const auto& matcher) {
            return matcher.id == matcherId;
        }));
    }

    EXPECT_TRUE(std::any_of(set.dispatches.begin(),
                            set.dispatches.end(),
                            [&](const auto& dispatch) { return dispatch.id == pack.dispatchId; }));
}

TEST(TestPointwiseAddPack, EngineNamesItsHeuristicAndMetadataSchema)
{
    const auto set = buildPointwiseAddDescriptorSet();

    // An engine carries exactly one heuristic and one metadata schema.
    EXPECT_EQ(set.engine.heuristicId, set.heuristic.id);
    EXPECT_EQ(set.engine.metadataSchemaId, set.schema.id);
}

TEST(TestPointwiseAddPack, ExposesBlockSizeAsAKnobAndDtypeAsInternal)
{
    const auto set = buildPointwiseAddDescriptorSet();

    // dtype is pinned by the graph rather than chosen.
    ASSERT_EQ(set.engine.knobs.size(), 1U);
    EXPECT_EQ(set.engine.knobs.front(), std::string(BLOCK_SIZE_FIELD));
}

TEST(TestPointwiseAddPack, EveryKnobNamesADeclaredMetadataField)
{
    const auto set = buildPointwiseAddDescriptorSet();

    // A knob naming a field the schema does not declare could never be filtered or defaulted.
    for(const auto& knob : set.engine.knobs)
    {
        EXPECT_TRUE(std::any_of(set.schema.fields.begin(),
                                set.schema.fields.end(),
                                [&](const auto& field) { return field.name == knob; }));
    }
}

TEST(TestPointwiseAddPack, MatchersCoverBothScopes)
{
    const auto set = buildPointwiseAddDescriptorSet();

    // Graph-scoped prunes the whole pack once per graph; kernel-scoped prunes per candidate.
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::GRAPH;
                            }),
              1);
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::KERNEL;
                            }),
              1);
}

TEST(TestPointwiseAddPack, EngineIdIsTheHashOfItsScopedName)
{
    // A descriptor-backed engine's id comes from its UED name, so the provider and
    // hipDNN derive the same id without sharing a table.
    EXPECT_EQ(registerEngineName(std::string(POINTWISE_ADD.engineName)),
              hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName));
}

TEST(TestPointwiseAddPack, RegisteringTheEngineNameTwiceIsIdempotent)
{
    // Called once per discovered set per process, but Containers come and go; a second
    // call must return the same id rather than throwing on a duplicate registration.
    const auto first = registerEngineName(std::string(POINTWISE_ADD.engineName));
    const auto second = registerEngineName(std::string(POINTWISE_ADD.engineName));

    EXPECT_EQ(first, second);
}

TEST(TestPointwiseAddPack, RegistersTheEngineNameForDiagnostics)
{
    // Registered at run time so a log line or collision report can name the engine.
    const auto id = registerEngineName(std::string(POINTWISE_ADD.engineName));

    EXPECT_EQ(hipdnn_data_sdk::utilities::getEngineNameFromId(id), POINTWISE_ADD.engineName);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
