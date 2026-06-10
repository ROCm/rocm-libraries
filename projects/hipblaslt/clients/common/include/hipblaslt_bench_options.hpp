/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/
#pragma once

#include <cstdint>

// Process-wide CLI knobs that hipblaslt-bench forwards into the matmul
// descriptor without going through the YAML-backed Arguments struct.
//
// sm_count_target maps to HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET and
// dyn_persistent_tile_enabled maps to HIPBLASLT_MATMUL_DESC_DYN_PERSISTENT_TILE_EXT.
// Defaults of 0 / false mean "no constraint; use the library defaults",
// matching the API defaults.
namespace hipblaslt_bench_options
{
    int32_t& sm_count_target();
    bool&    dyn_persistent_tile_enabled();
}
