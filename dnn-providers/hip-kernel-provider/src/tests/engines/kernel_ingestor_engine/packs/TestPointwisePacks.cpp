// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "kernel_sources.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwisePacks.cpp
 * @brief The engine's descriptor data: its shape, its cross-references, and its engine id.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

/// @brief The key of the source that @p operation's kernels compile.
///
/// The staged descriptor names its source relative to the set root, and the embedded
/// source table is keyed on that same string.
std::string sourceKeyFor(const std::string& operation)
{
    return "kernels/Pointwise" + operation + ".cpp";
}

/// @brief The pack whose kernels compile @p operation, by source file.
const hipdnn_plugin_sdk::ingestor::KernelDescriptorPack&
    packFor(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set, const std::string& operation)
{
    const auto match = std::find_if(set.packs.begin(), set.packs.end(), [&](const auto& pack) {
        return !pack.kernels.empty()
               && pack.kernels.front().source.sourceFile == sourceKeyFor(operation);
    });
    if(match == set.packs.end())
    {
        throw std::runtime_error("no pack whose kernels compile " + sourceKeyFor(operation));
    }
    return *match;
}

TEST(TestPointwisePacks, EachPackShipsThreeKernelsCoveringTwoBlockSizesAndTwoDataTypes)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    ASSERT_EQ(distinctPackIdCount(set), 3U);
    for(const auto& pack : set.packs)
    {
        const auto& kernels = pack.kernels;
        ASSERT_EQ(kernels.size(), 3U) << pack.name;

        // The two FLOAT kernels differ only in block size; HALF is pruned on a FLOAT graph.
        const auto describes = [&kernels](int64_t blockSize, const std::string& dtype) {
            return std::any_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
                return std::get<int64_t>(kernel.metadata.at(std::string(BLOCK_SIZE_FIELD)))
                           == blockSize
                       && std::get<std::string>(kernel.metadata.at(std::string(DTYPE_FIELD)))
                              == dtype;
            });
        };

        EXPECT_TRUE(describes(64, "FLOAT")) << pack.name;
        EXPECT_TRUE(describes(256, "FLOAT")) << pack.name;
        EXPECT_TRUE(describes(64, "HALF")) << pack.name;
    }
}

TEST(TestPointwisePacks, EveryKernelNamesItsPacksEmbeddedSource)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    for(const std::string operation : {"Add", "Mul", "Sub"})
    {
        for(const auto& kernel : packFor(set, operation).kernels)
        {
            EXPECT_EQ(kernel.source.kind,
                      hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE);
            EXPECT_EQ(kernel.source.sourceFile, sourceKeyFor(operation));
            EXPECT_EQ(kernel.source.entryPoint, "Pointwise" + operation);
        }
    }
}

/// The pack tool writes each kernel's source key. The build generates the table that key
/// is looked up in. One computation over one descriptor set feeds both, so every staged
/// key must resolve here. An unresolved key reaches getKernelSrc() at plan-build time
/// instead, and throws far from the descriptor that carries it.
TEST(TestPointwisePacks, EveryEmbeddedSourceKeyResolvesInTheCompiledInTable)
{
    std::size_t checked = 0;

    for(const auto& set : discoverDescriptorSets())
    {
        for(const auto& pack : set.packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                if(kernel.source.kind
                   != hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE)
                {
                    continue;
                }

                std::string_view source;
                EXPECT_NO_THROW(source = hip_plugin::getKernelSrc(kernel.source.sourceFile.c_str()))
                    << kernel.name << " (id " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                    << ") names '" << kernel.source.sourceFile << "', staged in "
                    << kernel.originDirectory;
                EXPECT_FALSE(source.empty()) << kernel.source.sourceFile;
                ++checked;
            }
        }
    }

    RecordProperty("embeddedSourceKernelsChecked", static_cast<int>(checked));
    GTEST_LOG_(INFO) << "resolved " << checked << " embedded_source keys";
    EXPECT_GT(checked, 0U) << "no embedded_source kernel was discovered, so this case "
                              "proved nothing about the keys";
}

/// The authored packs claim every architecture. The packer stamps each emitted copy with
/// the architecture of the shard directory it writes it into, so a pack's stamp must name
/// a directory it actually sits under.
///
/// Asserted as "some component of the path is the stamped arch" rather than against a
/// directory at a fixed depth, because neither end is fixed. originDirectory is the
/// descriptor's OWN folder, which equals the shard only for a descriptor staged flat;
/// treeRoot is whatever root the catalog was loaded from, which is this binary's whole
/// discovery root here and one arch shard in the packer/loader seam tests. The arch
/// folder sits between them, at a depth that changes with the authored layout.
TEST(TestPointwisePacks, EveryPackNamesTheArchitectureItWasPackedFor)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    ASSERT_FALSE(set.packs.empty());
    for(const auto& pack : set.packs)
    {
        // One stamp per emitted copy: the packer writes a shard per arch and narrows each
        // copy to that one, so a pack carrying two would mean the narrowing was skipped.
        ASSERT_EQ(pack.arch.size(), 1U) << pack.name;
        ASSERT_FALSE(pack.kernels.empty()) << pack.name;

        const auto& origin = pack.kernels.front().originDirectory;
        const auto& stamped = pack.arch.front();
        EXPECT_TRUE(std::any_of(
            origin.begin(),
            origin.end(),
            [&](const std::filesystem::path& part) { return part.string() == stamped; }))
            << pack.name << ": stamped '" << stamped << "' but was staged at " << origin;
    }
}

/// The point of three packs under one engine: everything but the operation criterion and
/// the kernels is one descriptor referenced three times, not three copies.
TEST(TestPointwisePacks, EveryPackSharesTheEngineDispatchAndAllButOneMatcher)
{
    const auto& set = loadedSet("hipkernel:Pointwise");
    const auto& add = packFor(set, "Add");
    const auto& mul = packFor(set, "Mul");
    const auto& sub = packFor(set, "Sub");

    // Load-bearing form: packs land in set.packs by resolveDescriptorSets() selecting
    // engineId == set.engine.id, so comparing packs to each other proves nothing.
    EXPECT_EQ(add.engineId, set.engine.id);
    EXPECT_EQ(mul.engineId, set.engine.id);
    EXPECT_EQ(sub.engineId, set.engine.id);
    EXPECT_EQ(add.dispatchId, mul.dispatchId);
    EXPECT_EQ(add.dispatchId, sub.dispatchId);
    ASSERT_EQ(set.dispatches.size(), 1U);

    // Two criteria each: one operation check of their own, plus the shared kernel-scoped
    // dtype check. The expensive graph work is the engine's graph_match, which runs once
    // per graph for every pack rather than being listed by any of them.
    EXPECT_FALSE(set.engine.graphMatchNativeSymbol.empty());
    ASSERT_EQ(add.matcherIds.size(), 2U);
    ASSERT_EQ(mul.matcherIds.size(), 2U);
    ASSERT_EQ(sub.matcherIds.size(), 2U);

    // Counted rather than set_intersection'd: matcher ids are in the order the pack
    // authored them, not sorted, and a sorted-range algorithm would quietly under-count.
    const auto lists = {&mul.matcherIds, &sub.matcherIds};
    EXPECT_EQ(std::count_if(add.matcherIds.begin(),
                            add.matcherIds.end(),
                            [&lists](const auto& matcherId) {
                                return std::all_of(
                                    lists.begin(), lists.end(), [&matcherId](const auto* other) {
                                        return std::find(other->begin(), other->end(), matcherId)
                                               != other->end();
                                    });
                            }),
              1);
}

TEST(TestPointwisePacks, ExposesBlockSizeAsAKnobAndDtypeAsInternal)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    // dtype is pinned by the graph rather than chosen.
    ASSERT_EQ(set.engine.knobs.size(), 1U);
    EXPECT_EQ(set.engine.knobs.front(), std::string(BLOCK_SIZE_FIELD));
}

TEST(TestPointwisePacks, MatchersCoverBothScopes)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    // Three graph-scoped: one operation check per pack. Applicability is the engine's
    // graph_match, not a UMD. One kernel-scoped, shared, pruning per candidate.
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::GRAPH;
                            }),
              3);
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::KERNEL;
                            }),
              1);
}

/// The GPU suite (ExecutesASubtractGraphThroughItsOwnPack) proves the kernel computes
/// a - b, but add/mul are commutative so an operand swap there is invisible. Sub is
/// asymmetric, so this pins the fast, device-free half: binding never swaps
/// input_a/input_b before dispatch gets them.
TEST(TestPointwisePacks, SubtractsInTheRightDirection)
{
    const GraphFixture fixture(
        buildPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB));

    const auto bound = matchesGraph(POINTWISE_SUB, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, POINTWISE_SUB.inputAToken),
              INPUT_A_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, POINTWISE_SUB.inputBToken),
              INPUT_B_UID);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
