// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1151WmmaAttentionCatalog.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx1151WmmaAttentionCatalog.cpp
 * @brief Census of the hipkernel:Gfx1151WmmaAttention catalog against the lists the
 *        engine and its GPU suites state in Gfx1151WmmaAttentionCatalog.hpp: the page
 *        sizes the graph matcher admits and the architectures the GPU suites run on.
 *
 * A page size added to the catalog alone would ship objects the matcher declines, and one
 * dropped from it would leave the matcher admitting graphs no object serves; an
 * architecture outside the list would ship objects the GPU suites skip on, and one listed
 * but dropped from the catalog would leave the GPU suites running where nothing serves.
 *
 * Each run reads one arch's shard. The census is registered at every wired arch the list
 * names (src/tests/CMakeLists.txt reads it from the header), not only where the engine
 * ships, so a dropped arch loads no set for the engine and both cases fail there.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx1151WmmaAttentionPacks, and is registered with its own hkp_register_census_tests()
 * call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestGfx1151WmmaAttentionCatalog, ShipsPagedObjectsForExactlyTheMatchersPageSizes)
{
    const auto& set = loadedSet(GFX1151_WMMA_ATTENTION_ENGINE_NAME);

    std::set<int64_t> shipped;
    std::size_t kernels = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++kernels;
            const auto it = kernel.metadata.find("page_block_size");
            if(it == kernel.metadata.end())
            {
                continue;
            }
            const auto* page = std::get_if<int64_t>(&it->second);
            ASSERT_NE(page, nullptr)
                << "kernel '" << kernel.name << "' carries a non-integer page_block_size";
            if(*page > 0)
            {
                shipped.insert(*page);
            }
        }
    }
    ASSERT_GT(kernels, 0U) << "the loaded set holds no kernels to check";
    const std::set<int64_t> stated(GFX1151_WMMA_ATTENTION_PAGE_SIZES.begin(),
                                   GFX1151_WMMA_ATTENTION_PAGE_SIZES.end());
    EXPECT_EQ(shipped, stated)
        << "the paged objects' page_block_size values and GFX1151_WMMA_ATTENTION_PAGE_SIZES "
           "differ";
}

TEST(TestGfx1151WmmaAttentionCatalog, ShipsForNoArchitectureOutsideTheStatedList)
{
    const auto& set = loadedSet(GFX1151_WMMA_ATTENTION_ENGINE_NAME);

    std::size_t kernels = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++kernels;
            EXPECT_FALSE(kernel.arch.empty())
                << "kernel '" << kernel.name << "' names no architecture";
            for(const auto& arch : kernel.arch)
            {
                EXPECT_NE(std::find(GFX1151_WMMA_ATTENTION_ARCHS.begin(),
                                    GFX1151_WMMA_ATTENTION_ARCHS.end(),
                                    arch),
                          GFX1151_WMMA_ATTENTION_ARCHS.end())
                    << "kernel '" << kernel.name << "' ships for " << arch
                    << ", which GFX1151_WMMA_ATTENTION_ARCHS does not list";
            }
        }
    }
    EXPECT_GT(kernels, 0U) << "the loaded set holds no kernels to check";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
