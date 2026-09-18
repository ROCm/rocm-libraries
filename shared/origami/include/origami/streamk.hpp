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

template <class Arch>
struct thresholds 
{
  enum class threshold_metrics : uint8_t 
  {
    grid_efficiency,
    grid_waves,
    tiles,
    tiles_per_cu,
    occupancy,
    m_dim,
    min_mn,
    active_cus,
    static_skgrid,
    iters_per_tile
  };

  static constexpr int static_result  = -1;
  static constexpr int dynamic_result = -2;

  struct decision_node 
  {
    threshold_metrics feature;
    double            threshold;
    int               if_lte;
    int               if_gt;
  };

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
      case threshold_metrics::m_dim:
      {
        return static_cast<double>(problem.size.m);
      }
      case threshold_metrics::min_mn:
      {
        return static_cast<double>(std::min<size_t>(problem.size.m, problem.size.n));
      }
      case threshold_metrics::active_cus:
      {
        const size_t cus = (sm_count_target > 0) ? std::min<size_t>(sm_count_target, hardware.N_CU) : hardware.N_CU;
        return static_cast<double>(cus ? cus : hardware.N_CU);
      }
      case threshold_metrics::static_skgrid:
      {
        return static_cast<double>(select_grid_size(problem, hardware, config, grid_selection_t::k_split_aware));
      }
      case threshold_metrics::iters_per_tile:
      {
        return config.mt.k ? static_cast<double>(math::safe_ceil_div(problem.size.k, config.mt.k))
                           : std::numeric_limits<double>::quiet_NaN();
      }
      default:
        break;
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

  static hybrid_mode_t select_hybrid_mode(const problem_t& problem, const hardware_t& hardware,
                                          const config_t& config, size_t sm_count_target) {
    int index = 0;
    for (;;) {
      const decision_node& node = Arch::decision_tree[index];
      const double         value = feature_value(node.feature, problem, hardware, config, sm_count_target);
      const int            next  = (value <= node.threshold) ? node.if_lte : node.if_gt;

      if (next  == dynamic_result) 
      {
        return hybrid_mode_t::dynamic;
      }
      else if(next == static_result)
      {
        return hybrid_mode_t::static_;
      }
      
      index = next;
    }
  }
};

struct gfx942_values : thresholds<gfx942_values> 
{
  static constexpr decision_node decision_tree[] = 
  {
      /* 0 */ {threshold_metrics::grid_waves, 1.17, static_result, dynamic_result},
  };
};

struct gfx950_values : thresholds<gfx950_values> 
{
  static constexpr int node_grid_efficiency = 0;
  static constexpr int node_min_mn          = 1;
  static constexpr int node_m               = 2;
  static constexpr int node_tiles           = 3;
  static constexpr int node_tiles_per_cu    = 4;
  static constexpr int node_static_skgrid   = 5;
  static constexpr int node_iters_per_tile  = 6;
  static constexpr int node_active_cus      = 7;
  static constexpr int node_occupancy       = 8;

  static constexpr decision_node decision_tree[] = 
  {
      // node                  feature                              thr      <= thr              > thr
      /* node_grid_efficiency */ {threshold_metrics::grid_efficiency, 0.23,   node_min_mn,        node_tiles_per_cu},
      /* node_min_mn          */ {threshold_metrics::min_mn,          1088.0, node_m,             node_tiles},
      /* node_m               */ {threshold_metrics::m_dim,           3277.0, static_result,        dynamic_result},
      /* node_tiles           */ {threshold_metrics::tiles,           34.0,   dynamic_result,       static_result},
      /* node_tiles_per_cu    */ {threshold_metrics::tiles_per_cu,    0.29,   node_static_skgrid, node_active_cus},
      /* node_static_skgrid   */ {threshold_metrics::static_skgrid,   68.0,   dynamic_result,       node_iters_per_tile},
      /* node_iters_per_tile  */ {threshold_metrics::iters_per_tile,  458.0,  static_result,        dynamic_result},
      /* node_active_cus      */ {threshold_metrics::active_cus,      240.0,  dynamic_result,       node_occupancy},
      /* node_occupancy       */ {threshold_metrics::occupancy,       2.5,    dynamic_result,       static_result},
  };
};

struct gfx1201_values : thresholds<gfx1201_values> 
{
  static constexpr int node_grid_waves = 0;
  static constexpr int node_m          = 1;

  static constexpr decision_node decision_tree[] = 
  {
      // node                 feature                          thr     <= thr         > thr
      /* node_grid_waves */ {threshold_metrics::grid_waves, 1.25,   node_m,        dynamic_result},
      /* node_m          */ {threshold_metrics::m_dim,      1289.0, static_result, dynamic_result},
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
