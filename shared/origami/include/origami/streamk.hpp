/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "origami/hardware.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"
#include "origami/origami_export.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace origami {
namespace streamk {
/**
 * @brief Number of output tiles.
 *
 * @param mt_m Tile size in M-dimension.
 * @param mt_n Tile size in N-dimension.
 * @param m Matrix's m-dimension.
 * @param n Matrix's n-dimension.
 * @param batch Number of batches.
 * @return size_t Total number of output tiles.
 */
ORIGAMI_EXPORT size_t compute_number_of_output_tiles(size_t mt_m, size_t mt_n, size_t m, size_t n, size_t batch);

/**
 * @brief Select the best reduction strategy for StreamK.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param algorithm Grid selection algorithm
 * @return reduction_t Selected reduction strategy
 */
ORIGAMI_EXPORT reduction_t select_reduction(const problem_t& problem,
                                            const hardware_t& hardware,
                                            const config_t& config,
                                            grid_selection_t algorithm);

/**
 * @brief Based on the provided kernel config, select the best grid dimension.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_selection_t grid selection algorithm (@see origami::grid_selection_t)
 * @return size_t Dimensions of the grid launched.
 */
ORIGAMI_EXPORT size_t select_grid_size(const problem_t& problem,
                                       const hardware_t& hardware,
                                       const config_t& config,
                                       grid_selection_t algorithm);

enum class threshold_metrics : uint8_t 
{
  grid_efficiency,
  grid_waves,
  tiles,
  tiles_per_cu,
  occupancy
};

enum class comparison_type : uint8_t 
{ 
  less_then_or_equal, 
  greater_then 
};

struct threshold_rule 
{
  threshold_metrics feature;
  double            threshold;
  comparison_type   comparison;
  hybrid_mode_t     mode;
};

template <class Arch>
struct thresholds 
{
  static size_t output_tiles(const problem_t& problem, const config_t& config) {
    return compute_number_of_output_tiles(config.mt.m, config.mt.n, problem.size.m, problem.size.n,
                                          std::max<size_t>(problem.batch, 1));
  }

  static double feature_value(threshold_metrics metric, const problem_t& problem, const hardware_t& hardware,
                              const config_t& config, size_t sm_count_target) {
    switch (metric) 
    {
      case threshold_metrics::occupancy:
      {
        return (config.occupancy >= 0) ? static_cast<double>(config.occupancy) : std::numeric_limits<double>::quiet_NaN();
      }
      case threshold_metrics::tiles:
      {
        return static_cast<double>(output_tiles(problem, config)); 
      }
      case threshold_metrics::tiles_per_cu:
      {
        const size_t cus = (sm_count_target > 0) ? std::min<size_t>(sm_count_target, hardware.N_CU) : hardware.N_CU;
        const double available_cus = static_cast<double>(cus ? cus : hardware.N_CU);
        return static_cast<double>(output_tiles(problem, config)) / available_cus;
      }
      case threshold_metrics::grid_waves: 
      {
        const size_t grid = select_grid_size(problem, hardware, config, grid_selection_t::k_split_aware);
        return grid ? static_cast<double>(output_tiles(problem, config)) / static_cast<double>(grid)
                    : std::numeric_limits<double>::quiet_NaN();
      }
      case threshold_metrics::grid_efficiency: 
      {
        const size_t grid = select_grid_size(problem, hardware, config, grid_selection_t::k_split_aware);
        if (!grid) 
        {
          return std::numeric_limits<double>::quiet_NaN();
        }
        const size_t tiles      = output_tiles(problem, config);
        const size_t waves_ceil = math::safe_ceil_div(tiles, grid);
        return waves_ceil ? static_cast<double>(tiles) / static_cast<double>(waves_ceil * grid) : 0.0;
      }
      default:
        break;
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

  static hybrid_mode_t select_hybrid_mode(const problem_t& problem, const hardware_t& hardware,
                                          const config_t& config, size_t sm_count_target) {
    for (const threshold_rule& rule : Arch::decision_tree) {
      const double value = feature_value(rule.feature, problem, hardware, config, sm_count_target);
      const bool   fires = (rule.comparison == comparison_type::less_then_or_equal) ? (value <= rule.threshold)
                                                  : (value > rule.threshold);
      if (fires) 
      {
        return rule.mode;
      }
    }
    return hybrid_mode_t::static_;
  }
};

struct gfx942_values : thresholds<gfx942_values> 
{
  static constexpr double grid_waves_threshold = 1.17;

  static constexpr threshold_rule decision_tree[] = 
  {
      {threshold_metrics::grid_waves, grid_waves_threshold, comparison_type::greater_then, hybrid_mode_t::dynamic},
  };
};

struct gfx950_values : thresholds<gfx950_values> 
{
  static constexpr double grid_efficiency_threshold = 0.23;
  static constexpr double tiles_threshold           = 480;
  static constexpr double occupancy_threshold       = 2.5;

  static constexpr threshold_rule decision_tree[] = 
  {
      {threshold_metrics::tiles, tiles_threshold, comparison_type::less_then_or_equal, hybrid_mode_t::static_},
      {threshold_metrics::grid_efficiency, grid_efficiency_threshold, comparison_type::greater_then, hybrid_mode_t::dynamic},
      {threshold_metrics::occupancy, occupancy_threshold, comparison_type::less_then_or_equal, hybrid_mode_t::dynamic},
  };
};

/**
 * @brief Pick the SK3-vs-SK4 sub-path for a StreamK=5 hybrid kernel.
 *
 * Evaluates the per-architecture decision tree
 * (fit to measured SK5 on(SK4)/off(SK3) sweeps). Architectures without a
 * tuned list return hybrid_mode_t::static_.
 *
 * @param problem            Problem description (M, N, K, batch).
 * @param hardware           Hardware characteristics (@see origami::hardware_t).
 * @param config             Kernel configuration (provides MT shape and occupancy).
 * @param sm_count_target    Caller's effective CU budget (0 = use all
 *                           CUs the device exposes). When non-zero,
 *                           clamps hardware.N_CU from above.
 * @return hybrid_mode_t::static_ for SK3, hybrid_mode_t::dynamic for SK4.
 */
 ORIGAMI_EXPORT hybrid_mode_t select_hybrid_mode(const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config,
    size_t sm_count_target);

}  // namespace streamk
}  // namespace origami
