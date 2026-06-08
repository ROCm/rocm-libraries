/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/
#pragma once

#include <cstdint>
#include <string>

// Process-wide CLI knobs that hipblaslt-bench forwards into the matmul
// descriptor without going through the YAML-backed Arguments struct.
//
// sm_count_target maps to HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET and
// dyn_persistent_tile_mode maps to HIPBLASLT_MATMUL_DESC_DYN_PERSISTENT_TILE_EXT
// as a tri-state {0=OFF, 1=ON, 2=AUTO}. The resolved mode defaults to -1
// ("unset"): a negative value tells the bench to leave the attribute untouched
// so the library default applies. dyn_persistent_tile_mode_str() holds the raw
// CLI token (off|on|auto or 0|1|2) before client.cpp resolves it.
namespace hipblaslt_bench_options
{
    int32_t&     sm_count_target();
    int32_t&     dyn_persistent_tile_mode();
    std::string& dyn_persistent_tile_mode_str();
}
