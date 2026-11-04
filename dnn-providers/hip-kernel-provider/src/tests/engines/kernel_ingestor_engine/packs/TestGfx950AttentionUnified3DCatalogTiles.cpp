// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx950AttentionUnified3DCatalogTiles.cpp
 * @brief Census of what the shipped hipkernel:Gfx950AttentionUnified3D catalog declares,
 *        judged by the checks the engine itself applies.
 *
 * The engine launches a loaded kernel only when its candidateFor() accepts the record: a
 * buildable decode geometry launched as exactly `segment` and `reduce`. A kernel it declines
 * ships but never runs. This suite reads the kernels through the descriptor loader -- the
 * packed tree, where `kind: rocke` has become `kind: kpack` -- and requires every one to
 * pass, and every one to carry its two launches, each signed with the argument list its own
 * builder declares.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx950AttentionUnified3DPacks, and is registered with its own
 * hkp_register_census_tests() call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

constexpr std::size_t SEGMENT_ARGUMENTS = 19;
constexpr std::size_t REDUCE_ARGUMENTS = 5;

std::optional<int64_t> integerField(const ingestor::KernelDescriptor& kernel, const char* field)
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

TEST(TestGfx950AttentionUnified3DCatalogTiles, EveryShippedKernelsGeometryIsOneTheEngineLaunches)
{
    const auto& set = loadedSet(GFX950_ATTENTION_UNIFIED_3D_ENGINE_NAME);

    std::size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++checked;
            const auto headSize = integerField(kernel, "head_size");
            const auto blockSize = integerField(kernel, "block_size");
            const auto numQueryHeads = integerField(kernel, "num_query_heads");
            const auto numKvHeads = integerField(kernel, "num_kv_heads");
            const auto numSegments = integerField(kernel, "num_segments");
            const auto useDecodeGrid = integerField(kernel, "use_decode_grid");
            const auto maskType = integerField(kernel, "mask_type");
            ASSERT_TRUE(headSize && blockSize && numQueryHeads && numKvHeads && numSegments
                        && useDecodeGrid && maskType)
                << "kernel '" << kernel.name << "' does not carry the integer geometry metadata";
            EXPECT_TRUE(isSupportedGfx950Unified3DGeometry(
                *headSize, *blockSize, *numQueryHeads, *numKvHeads, *numSegments))
                << "kernel '" << kernel.name << "' declares a geometry the engine declines";
            EXPECT_EQ(*useDecodeGrid, 1) << "kernel '" << kernel.name << "' is not a decode kernel";
            EXPECT_TRUE(*maskType == 1 || *maskType == 2)
                << "kernel '" << kernel.name << "' is keyed mask_type " << *maskType
                << "; decode carries the two causal kinds only";
        }
    }
    EXPECT_GT(checked, 0U) << "the loaded set holds no kernels to check";
}

/// Each catalog entry is two launches of one module: `segment`, the segment kernel, and
/// `reduce`, the reduce kernel, both resolved out of the same packed code object (one library
/// and toc key) and each signed with its own builder's argument list. These are exactly the
/// launch names the engine's handler declares (launchNames(), pinned by
/// TestGfx950AttentionUnified3DDispatch), so the state manager keeps every entry. An entry
/// that lost its reduce launch would be dropped and never run; one whose reduce launch named
/// the segment kernel again would pass a count check and fail only on a device.
TEST(TestGfx950AttentionUnified3DCatalogTiles, EveryEntryLaunchesItsSegmentAndReduceKernels)
{
    const auto& set = loadedSet(GFX950_ATTENTION_UNIFIED_3D_ENGINE_NAME);

    std::size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++checked;
            SCOPED_TRACE("kernel '" + kernel.name + "'");
            ASSERT_EQ(kernel.sources.size(), 2U);
            ASSERT_EQ(kernel.sources.count("segment"), 1U);
            ASSERT_EQ(kernel.sources.count("reduce"), 1U);
            const auto& segment = kernel.sources.at("segment");
            const auto& reduce = kernel.sources.at("reduce");
            EXPECT_EQ(segment.kind, ingestor::KernelSourceKind::KPACK);
            EXPECT_EQ(reduce.kind, ingestor::KernelSourceKind::KPACK);
            EXPECT_EQ(reduce.library, segment.library);
            EXPECT_EQ(reduce.tocKey, segment.tocKey);
            EXPECT_NE(reduce.symbol, segment.symbol);
            EXPECT_EQ(segment.sha256.size(), 64U);
            EXPECT_EQ(reduce.sha256.size(), 64U);
            ASSERT_EQ(segment.signature.size(), SEGMENT_ARGUMENTS);
            ASSERT_EQ(reduce.signature.size(), REDUCE_ARGUMENTS);
            EXPECT_EQ(segment.signature.front().name, "segm_output_ptr");
            EXPECT_EQ(reduce.signature.front().name, "output_ptr");
            EXPECT_EQ(reduce.signature.back().name, "seq_lens_ptr");
        }
    }
    EXPECT_GT(checked, 0U) << "the loaded set holds no kernels to check";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
