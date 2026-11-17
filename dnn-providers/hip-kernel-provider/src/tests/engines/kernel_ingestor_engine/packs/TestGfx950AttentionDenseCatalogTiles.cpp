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
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx950AttentionDenseCatalogTiles.cpp
 * @brief Census of the tiles the shipped hipkernel:Gfx950AttentionDense catalog declares,
 *        judged by the check the engine itself applies.
 *
 * The engine launches a loaded kernel only when its candidateTile() accepts the kernel's
 * head_size, block_m and block_n through isSupportedGfx950AttentionDenseTile(); a kernel
 * it declines ships but is never selected. This suite reads the kernels through the
 * descriptor loader and requires every one to pass that same check, so a catalog entry
 * the C++ engine cannot launch (an LDS-oversized or non-dividing tile, a missing field)
 * fails here instead of silently never running.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx950AttentionDensePacks, and is registered with its own hkp_register_census_tests()
 * call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

/// A kernel's integer metadata field, or nullopt when it is absent or another type: the
/// cases candidateTile() declines.
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

TEST(TestGfx950AttentionDenseCatalogTiles, EveryShippedKernelsTileIsOneTheEngineLaunches)
{
    const auto& set = loadedSet(GFX950_ATTENTION_DENSE_ENGINE_NAME);

    std::size_t checked = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ++checked;
            const auto headSize = integerField(kernel, "head_size");
            const auto blockM = integerField(kernel, "block_m");
            const auto blockN = integerField(kernel, "block_n");
            ASSERT_TRUE(headSize.has_value() && blockM.has_value() && blockN.has_value())
                << "kernel '" << kernel.name
                << "' does not carry integer head_size, block_m and block_n metadata";
            EXPECT_TRUE(isSupportedGfx950AttentionDenseTile(*headSize, *blockM, *blockN))
                << "kernel '" << kernel.name << "' declares head_size " << *headSize << ", block_m "
                << *blockM << ", block_n " << *blockN << ", a tile the engine declines to launch";
        }
    }
    // A catalog that loaded no kernel would pass the loop above vacuously.
    EXPECT_GT(checked, 0U) << "the loaded set holds no kernels to check";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
