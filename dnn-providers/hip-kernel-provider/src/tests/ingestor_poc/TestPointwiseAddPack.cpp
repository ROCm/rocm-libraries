// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include "ingestor_poc/NativeSymbolNames.hpp"
#include "ingestor_poc/PointwiseAddPack.hpp"

/**
 * @file TestPointwiseAddPack.cpp
 * @brief The POC's descriptor data: its shape, its cross-references, and its engine id.
 */
namespace
{

using namespace hip_kernel_provider::ingestor_poc;

TEST(TestPointwiseAddPack, ShipsThreeKernelsCoveringTwoBlockSizesAndTwoDataTypes)
{
    const auto set = buildPointwiseAddDescriptorSet();

    ASSERT_EQ(set.packs.size(), 1U);
    const auto& kernels = set.packs.front().kernels;
    ASSERT_EQ(kernels.size(), 3U);

    // The two FLOAT kernels differ only in block size, so ranking has an order to
    // produce and the block_size knob has a value set of two. The HALF kernel is what
    // the kernel-scoped matcher prunes on a FLOAT graph.
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
        EXPECT_EQ(kernel.sourceFile, "PointwiseAdd.cpp");
        EXPECT_EQ(kernel.entryPoint, "PointwiseAdd");
    }
}

TEST(TestPointwiseAddPack, PackCrossReferencesResolveWithinTheDescriptorSet)
{
    const auto set = buildPointwiseAddDescriptorSet();
    const auto& pack = set.packs.front();

    // A pack names its engine, matchers, and dispatch by id; every one must resolve or
    // the descriptor set cannot be evaluated.
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

    // An engine carries exactly one of each, because one selector ranks all of its
    // kernels over one feature space.
    EXPECT_EQ(set.engine.heuristicId, set.heuristic.id);
    EXPECT_EQ(set.engine.metadataSchemaId, set.schema.id);
}

TEST(TestPointwiseAddPack, ExposesBlockSizeAsAKnobAndDtypeAsInternal)
{
    const auto set = buildPointwiseAddDescriptorSet();

    // dtype is pinned by the graph rather than chosen, so exposing it would offer a
    // choice nothing can serve.
    ASSERT_EQ(set.engine.knobs.size(), 1U);
    EXPECT_EQ(set.engine.knobs.front(), std::string(BLOCK_SIZE_FIELD));
}

TEST(TestPointwiseAddPack, EveryKnobNamesADeclaredMetadataField)
{
    const auto set = buildPointwiseAddDescriptorSet();

    // A knob naming a field the schema does not declare is a load error in the real
    // system: it could never be filtered or defaulted.
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

    // Graph-scoped prunes the whole pack once per graph; kernel-scoped prunes per
    // candidate. Losing either collapses the pruning order the ingestor relies on.
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
    // A descriptor-backed engine's id comes from its UED name, the same derivation a
    // hand-written engine's registered name goes through.
    EXPECT_EQ(pointwiseAddEngineId(), hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME));
}

TEST(TestPointwiseAddPack, EngineIdIsStableAcrossCalls)
{
    // The id keys the engine in hipDNN's registry and in this provider's engine table,
    // so it must not vary between the two places that ask for it.
    EXPECT_EQ(pointwiseAddEngineId(), pointwiseAddEngineId());
}

TEST(TestPointwiseAddPack, RegistersTheEngineNameForDiagnostics)
{
    // Registered at run time rather than by a compile-time macro, so a log line and a
    // collision report can name the engine instead of printing a hex id.
    const auto id = pointwiseAddEngineId();

    EXPECT_EQ(hipdnn_data_sdk::utilities::getEngineNameFromId(id), ENGINE_NAME);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
