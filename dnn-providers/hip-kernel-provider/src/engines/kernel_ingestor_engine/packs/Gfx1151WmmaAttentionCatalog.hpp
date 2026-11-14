// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

/**
 * @file Gfx1151WmmaAttentionCatalog.hpp
 * @brief What the hipkernel:Gfx1151WmmaAttention catalog
 *        (configs/gfx1151_wmma_attention.yaml) ships, stated once for the engine and its
 *        tests: the census suite TestGfx1151WmmaAttentionCatalog checks both lists
 *        against the loaded objects.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The prefill engine's name, as its .ued.json declares it.
inline constexpr std::string_view GFX1151_WMMA_ATTENTION_ENGINE_NAME
    = "hipkernel:Gfx1151WmmaAttention";

/// The page sizes (page_block_size) the paged prefill objects ship for. The graph matcher
/// declines another page size with its cause logged rather than finding no object.
inline constexpr std::array<int64_t, 3> GFX1151_WMMA_ATTENTION_PAGE_SIZES{16, 32, 64};

/// The architectures the objects are built for.
inline constexpr std::array<std::string_view, 10> GFX1151_WMMA_ATTENTION_ARCHS{"gfx1100",
                                                                               "gfx1101",
                                                                               "gfx1102",
                                                                               "gfx1103",
                                                                               "gfx1150",
                                                                               "gfx1151",
                                                                               "gfx1152",
                                                                               "gfx1153",
                                                                               "gfx1200",
                                                                               "gfx1201"};

} // namespace hip_kernel_provider::kernel_ingestor_engine
