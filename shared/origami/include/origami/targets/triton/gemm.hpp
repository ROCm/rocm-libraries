// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Compute Triton-specific StreamK grid size.
 *
 * Uses the shared streamk::pick_fractional_grid / streamk::pick_k_split
 * helpers for the heuristic core (no logic duplication with
 * streamk::grid_k_split_aware) and adds the Triton-only "last partial wave
 * -> prev_pow2(n_cu)" compensation.
 *
 * Tile count is batch-aware (uses streamk::compute_number_of_output_tiles).
 * Per-tile workspace bytes are derived sub-byte-safely from the C dtype, or
 * from `config.workspace_size_per_elem_c` when set (matches Tensile semantics).
 *
 * @param problem  Problem description (size, batch, dtypes).
 * @param config   Kernel configuration (uses `mt`, optionally `workspace_size_per_elem_c`).
 * @param hardware Hardware description (uses `N_CU`).
 * @return std::size_t StreamK grid size.
 */
std::size_t compute_triton_sk_grid(const problem_t&  problem,
                                   const config_t&   config,
                                   const hardware_t& hardware);

/**
 * @brief Default Triton tile candidate configs for the given problem and hardware.
 *
 * Returns the architecture-aware cross-product of candidate (block_m, block_n,
 * block_k) tile sizes as a flat list of `config_t`. The arch-specific gating
 * is driven by the narrower of `problem.a_dtype` and `problem.b_dtype`:
 *   - gfx950, narrow input <= 8 bits (F8/F4):
 *       MN restricted to {32, 64, 128, 256} -- no 16-MN MFMA support for those dtypes.
 *   - gfx942, narrow input == 8 bits (F8):
 *       MN extended with 512 -- the 512 tile is genuinely additive on this arch.
 *   - gfx942, narrow input <  8 bits (F4/F6):
 *       Falls through to the default range. F4/F6 are not natively supported
 *       on gfx942 MFMA; consumers should filter empirically downstream.
 *   - Everything else:
 *       MN in {16, 32, 64, 128, 256}, K in {16, 32, 64, 128, 256, 512}.
 *
 * Only `mt.{m,n,k}` is populated on each returned config. The caller is
 * expected to set `mi` (and any other config fields it needs, e.g. occupancy,
 * grid_selection, target) per its own selection policy.
 *
 * @param problem  Problem description (uses `a_dtype`, `b_dtype`).
 * @param hardware Hardware description (uses `arch`).
 * @return std::vector<config_t> Flat list of candidate tile configs.
 */
std::vector<config_t> get_triton_default_configs(const problem_t&  problem,
                                                 const hardware_t& hardware);

}  // namespace origami
