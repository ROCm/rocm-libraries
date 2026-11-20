// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "engines/kernel_ingestor_engine/packs/AttentionDenseCommon.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file AttentionDenseCatalogTileTestSupport.hpp
 * @brief The catalog-tile census body shared by the gfx942 and gfx950 dense-attention
 *        engines, which differ only in the engine name and the tile check they apply.
 *
 * The census suites stay one TEST per arch rather than a TEST_P: a census entry addresses
 * one suite by name, and a TEST_P suite and its case names carry the instantiation prefix
 * and parameter suffix that the EXPECTED_CASES pins do not.
 *
 * Three checks: the tile census below, the argument census
 * (expectEveryShippedAttentionDenseKernelTakesTheLaunchedArguments), and the served-set
 * census (expectEveryShippedAttentionDenseKernelReadsItsStridesAsRockeSaysItDoes). The second
 * is the only host check that compares a shipped kernel's recorded argument list with the list
 * the engine launches; without it a kernel/engine skew shows only when prepare() refuses the
 * kernel on a device. The third is the only one that compares the engine's rule for which
 * bodies read their strides with rocKE's.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The engine's own launchability check over a kernel's head_size, block_m and block_n.
using AttentionDenseTilePredicate = bool (*)(int64_t headSize, int64_t blockM, int64_t blockN);

/// A kernel's integer metadata field, or nullopt when it is absent or another type: the
/// cases candidateTile() declines.
inline std::optional<int64_t>
    attentionDenseIntegerField(const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel,
                               const char* field)
{
    const auto it = kernel.metadata.find(field);
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

/// Requires every kernel @p engineName loads to declare a tile @p isSupportedTile accepts.
inline void
    expectEveryShippedAttentionDenseTileIsLaunchable(std::string_view engineName,
                                                     AttentionDenseTilePredicate isSupportedTile)
{
    const auto& set = loadedSet(engineName);

    std::size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++checked;
            const auto headSize = attentionDenseIntegerField(kernel, "head_size");
            const auto blockM = attentionDenseIntegerField(kernel, "block_m");
            const auto blockN = attentionDenseIntegerField(kernel, "block_n");
            ASSERT_TRUE(headSize.has_value() && blockM.has_value() && blockN.has_value())
                << "kernel '" << kernel.name
                << "' does not carry integer head_size, block_m and block_n metadata";
            EXPECT_TRUE(isSupportedTile(*headSize, *blockM, *blockN))
                << "kernel '" << kernel.name << "' declares head_size " << *headSize << ", block_m "
                << *blockM << ", block_n " << *blockN << ", a tile the engine declines to launch";
        }
    }
    // A catalog that loaded no kernel would pass the loop above vacuously.
    EXPECT_GT(checked, 0U) << "the loaded set holds no kernels to check";
}

/**
 * @brief Requires every kernel @p engineName loads to record exactly the argument list
 *        @p arch's current contract launches it with: its bias contract, for a kernel that
 *        adds an additive bias.
 *
 * The expected list is attention_dense::kernelSignature for the contract's ABI at the
 * kernel's own launch grid and LSE form, read from its metadata through the arch's own hooks,
 * as prepare() and kernelMatches read them. Kind, size and name are compared, as
 * requireSignatureMatch compares them; offsets are not. The recorded list is the one the
 * pack lowered out of the compiled kernel, so the rocKE builder and this engine cannot drift
 * apart without this failing on a host.
 */
inline void expectEveryShippedAttentionDenseKernelTakesTheLaunchedArguments(
    std::string_view engineName, const attention_dense::AttentionDenseArch& arch)
{
    const auto& set = loadedSet(engineName);

    std::size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++checked;
            SCOPED_TRACE("kernel '" + kernel.name + "'");
            ASSERT_TRUE(kernel.sources.empty()) << "a dense kernel is one launch";

            hipdnn_plugin_sdk::ingestor::KernelDefinition definition;
            definition.name = kernel.name;
            definition.source = kernel.source;
            definition.metadata = kernel.metadata;

            attention_dense::AttentionDenseGrid grid;
            if(arch.servesGrids())
            {
                const auto candidateGrid = arch.candidateGrid(definition);
                ASSERT_TRUE(candidateGrid.has_value())
                    << "declares no launch grid the engine launches";
                grid = *candidateGrid;
            }
            bool writesLse = false;
            if(arch.servesStats())
            {
                const auto candidateWritesLse = arch.candidateWritesLse(definition);
                ASSERT_TRUE(candidateWritesLse.has_value()) << "declares no valid emit_lse";
                writesLse = *candidateWritesLse;
            }

            bool addsBias = false;
            if(arch.servesBias())
            {
                const auto candidateAddsBias = arch.candidateAddsBias(definition);
                ASSERT_TRUE(candidateAddsBias.has_value()) << "declares no valid use_additive_bias";
                addsBias = *candidateAddsBias;
            }

            const auto& contract = addsBias ? arch.biasContract : arch.currentContract;
            const auto expected = attention_dense::kernelSignature(contract.abi, grid, writesLse);
            const auto& recorded = kernel.source.signature;
            const std::string both
                = "recorded [" + hipdnn_plugin_sdk::ingestor::describeKernelSignature(recorded)
                  + "], launched [" + hipdnn_plugin_sdk::ingestor::describeKernelSignature(expected)
                  + "]";
            ASSERT_EQ(recorded.size(), expected.size()) << both;
            for(std::size_t index = 0; index < expected.size(); ++index)
            {
                EXPECT_EQ(recorded[index].kind, expected[index].kind) << index << ": " << both;
                EXPECT_EQ(recorded[index].size, expected[index].size) << index << ": " << both;
                EXPECT_EQ(recorded[index].name, expected[index].name) << index << ": " << both;
            }
        }
    }
    EXPECT_GT(checked, 0U) << "the loaded set holds no kernels to check";
}

/// One KMD field a served-set row sets: an integer, or the string @p text where it is not null.
struct AttentionDenseMetadataOverride
{
    const char* field;
    int64_t number;
    const char* text;
};

/// One row of the generated served-set table (gen_attention_dense_served_set.py): rocKE's
/// answer, `served`, for the shipped kernel @p kernel of arch @p arch (variant "") or for that
/// kernel with @p overrides applied to its KMD metadata (variant: the body class that flip
/// stands for).
struct AttentionDenseServedRow
{
    const char* arch;
    const char* kernel;
    const char* variant;
    std::vector<AttentionDenseMetadataOverride> overrides;
    bool served;
};

/// The generated table; empty when the build found no kernel descriptor pack to generate it from.
inline const std::vector<AttentionDenseServedRow>& attentionDenseServedRows()
{
    static const std::vector<AttentionDenseServedRow> s_rows{
#include "AttentionDenseServedSetTable.inc"
    };
    return s_rows;
}

/**
 * @brief Requires @p arch's candidateReadsStrides to answer, for every kernel @p engineName loads and
 *        for one flip of each body class, what rocKE's `body_serves_strides` answers.
 *
 * Serving a body that does not read its strides is a silent wrong result, and the engine's rule
 * reads KMD fields where rocKE reads the whole spec, so the two are compared where they can
 * disagree: on the real kernels (the rows of variant "", which name every kernel of the pack, no
 * more and no fewer) and on the shipped kernel with one body-class field flipped (persistent,
 * ragged and, on gfx942, the K double buffer, the V prefetch and the KV-segment split), which
 * holds the engine to rocKE for a class the catalog does not carry today.
 */
inline void expectEveryShippedAttentionDenseKernelReadsItsStridesAsRockeSaysItDoes(
    std::string_view engineName,
    std::string_view archLabel,
    const attention_dense::AttentionDenseArch& arch)
{
    ASSERT_NE(arch.candidateReadsStrides, nullptr) << "the arch declares no stride rule";
    const auto& set = loadedSet(engineName);

    std::map<std::string, const AttentionDenseServedRow*> shipped;
    std::vector<const AttentionDenseServedRow*> flips;
    for(const auto& row : attentionDenseServedRows())
    {
        if(row.arch != archLabel)
        {
            continue;
        }
        if(std::string_view(row.variant).empty())
        {
            ASSERT_TRUE(shipped.emplace(row.kernel, &row).second)
                << "the table names kernel '" << row.kernel << "' twice";
        }
        else
        {
            flips.push_back(&row);
        }
    }
    ASSERT_FALSE(shipped.empty())
        << "the generated served-set table has no " << archLabel
        << " rows: the build found no kernel descriptor pack to generate it from";

    std::map<std::string, hipdnn_plugin_sdk::ingestor::KernelDefinition> loaded;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            hipdnn_plugin_sdk::ingestor::KernelDefinition definition;
            definition.name = kernel.name;
            definition.source = kernel.source;
            definition.metadata = kernel.metadata;
            ASSERT_TRUE(loaded.emplace(kernel.name, definition).second)
                << "kernel '" << kernel.name << "' is loaded twice";
        }
    }
    // The table and the pack are the same catalog: a kernel on one side only means one of
    // them is stale.
    EXPECT_EQ(loaded.size(), shipped.size());

    std::size_t served = 0;
    for(const auto& [name, definition] : loaded)
    {
        const auto found = shipped.find(name);
        ASSERT_NE(found, shipped.end())
            << "kernel '" << name << "' is shipped but not in the table";
        const bool engineSays = arch.candidateReadsStrides(definition);
        EXPECT_EQ(engineSays, found->second->served)
            << "kernel '" << name << "': the engine says "
            << (engineSays ? "it reads its strides" : "it does not")
            << ", rocKE's body_serves_strides says "
            << (found->second->served ? "it does" : "it does not");
        served += engineSays ? 1U : 0U;
    }
    for(const auto* row : flips)
    {
        const auto found = loaded.find(row->kernel);
        ASSERT_NE(found, loaded.end()) << "flip '" << row->variant << "' names kernel '"
                                       << row->kernel << "', which the pack does not ship";
        auto definition = found->second;
        for(const auto& override : row->overrides)
        {
            if(override.text != nullptr)
            {
                definition.metadata[override.field] = std::string(override.text);
            }
            else
            {
                definition.metadata[override.field] = override.number;
            }
        }
        const bool engineSays = arch.candidateReadsStrides(definition);
        EXPECT_EQ(engineSays, row->served)
            << "kernel '" << row->kernel << "' flipped to " << row->variant << ": the engine says "
            << (engineSays ? "it reads its strides" : "it does not")
            << ", rocKE's body_serves_strides says " << (row->served ? "it does" : "it does not");
    }
    EXPECT_GT(served, 0U) << "no shipped kernel reads its strides: nothing would be served";
    EXPECT_FALSE(flips.empty()) << "the table has no body-class flips";
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
