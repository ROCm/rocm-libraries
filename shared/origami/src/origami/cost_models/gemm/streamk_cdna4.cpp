// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/streamk.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// The SK3-vs-SK4 sub-path of a StreamK=5 hybrid kernel, fit to measured SK5 on(SK4)/off(SK3)
// sweeps on MI350X.  Gates, in order: grid size (tiles), then whether a cotenant currently holds
// any CU away from this kernel, then occupancy, falling back to tiles-per-CU only once occupancy
// alone isn't decisive.
hybrid_mode_t cdna4_model::tile_schedule(const problem_t& problem,
                                         const hardware_t& hardware,
                                         const config_t& config,
                                         size_t sm_count_target) {
  const cdna4_constants_t& c4 = constants(hardware);

  const size_t MT_M  = config.mt.m;
  const size_t MT_N  = config.mt.n;
  const size_t batch = std::max<size_t>(problem.batch, 1);
  const size_t tiles =
      streamk::compute_number_of_output_tiles(MT_M, MT_N, problem.size.m, problem.size.n, batch);

  // Too little work in the grid for dynamic rebalancing to be worth its
  // overhead.
  if (tiles <= c4.HYBRID_MIN_TILES_FOR_DYNAMIC) return hybrid_mode_t::static_;

  size_t available_cus =
      (sm_count_target > 0) ? std::min<size_t>(sm_count_target, hardware.N_CU) : hardware.N_CU;
  if (available_cus == 0) available_cus = hardware.N_CU;

  // No cotenant means static already matches the full CU count -- nothing
  // left for dynamic to rebalance.
  const bool has_cotenant = available_cus < hardware.N_CU;
  if (!has_cotenant) return hybrid_mode_t::static_;

  // Few wavefronts resident per CU means little overlap to absorb a
  // cotenant's CU-count mismatch on its own, so dynamic rebalancing helps
  // regardless of tiles_per_cu. (occupancy <= 0 means "unknown" -- fall
  // through to the tiles_per_cu check instead.)
  if (config.occupancy > 0 && config.occupancy <= c4.HYBRID_MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC)
    return hybrid_mode_t::dynamic;

  // At higher occupancy that overlap already absorbs small mismatches, so
  // dynamic only pays off once the grid is heavily overloaded relative to
  // the CUs actually available to it.
  const double tiles_per_cu = static_cast<double>(tiles) / static_cast<double>(available_cus);
  return (tiles_per_cu > c4.HYBRID_TILES_PER_CU_THRESHOLD_HIGH_OCCUPANCY) ? hybrid_mode_t::dynamic
                                                                          : hybrid_mode_t::static_;
}

}  // namespace gemm
}  // namespace origami
