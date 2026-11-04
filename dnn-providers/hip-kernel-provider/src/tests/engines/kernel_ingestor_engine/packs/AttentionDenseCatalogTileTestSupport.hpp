// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file AttentionDenseCatalogTileTestSupport.hpp
 * @brief The catalog-tile census body shared by the gfx942 and gfx950 dense-attention
 *        engines, which differ only in the engine name and the tile check they apply.
 *
 * The census suites stay one TEST per arch rather than a TEST_P: a census entry addresses
 * one suite by name, and a TEST_P suite and its case names carry the instantiation prefix
 * and parameter suffix that the EXPECTED_CASES pins do not.
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

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
