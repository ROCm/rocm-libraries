// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "origami/gemm.hpp"
#include "origami/simulator/tensilelite/formocast_simulator.hpp"
#include "origami/streamk.hpp"

namespace origami {

// Forward declaration for internal Formocast latency computation
static double compute_formocast_latency(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config);

/* ---------------------------------------------------------------------------------------- */
/* context_t constructor                                                                    */
/* ---------------------------------------------------------------------------------------- */
context_t::context_t(const problem_t& problem, const hardware_t& hardware, const config_t& config) {
  // Extract parameters
  const size_t NUM_XCD = hardware.NUM_XCD;
  const size_t N_CU    = hardware.N_CU;

  const size_t M     = problem.size.m;
  const size_t N     = problem.size.n;
  const size_t batch = problem.batch;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  // Heuristic parameters
  heuristic = get_heuristic_params(problem, hardware, config);

  // Element sizes
  a_bytes = data_type_to_bytes(problem.a_dtype);
  b_bytes = data_type_to_bytes(problem.b_dtype);
  d_bytes = data_type_to_bytes(problem.d_dtype);

  // Grid dimensions
  grid_m           = math::safe_ceil_div(M, MT_M);
  grid_n           = math::safe_ceil_div(N, MT_N);
  num_output_tiles = grid_m * grid_n * batch;

  // Launch parameters
  auto [reduction, wgs, cus, timesteps, split] =
      compute_launch_parameters(problem, hardware, config, config.grid_selection, 0);
  reduction_strategy = reduction;
  num_wgs            = wgs;
  num_timesteps      = timesteps;
  splitting_factor   = split;
  k_per_split        = math::safe_ceil_div(problem.size.k, splitting_factor);
  k_iters            = (config.mt.k > 0) ? math::safe_ceil_div(k_per_split, config.mt.k) : 1;

  // Hardware-derived values
  active_cus           = cus;
  mem_bw_limited       = compute_mem_bw_from_occupancy(hardware, active_cus);
  write_mem_bw_limited = compute_mem_bw_from_occupancy(hardware, num_output_tiles);
  real_occupancy       = std::min(
      std::max(config.occupancy, static_cast<int>(1)),
      static_cast<int>(math::safe_ceil_div(grid_m * grid_n * batch * splitting_factor, N_CU)));
  occupancy_factor = pow(heuristic.occupancy_decay_base, real_occupancy);

  // Tile-derived values
  tile_elements     = MT_M * MT_N;
  output_tile_bytes = tile_elements * d_bytes;

  // Workgroup mapping
  wgm = predict_workgroup_mapping(problem, hardware, config, grid_m, grid_n, splitting_factor);

  // Debug flag (cached to avoid repeated singleton lookups)
  debug = runtime_options::get().debug_enabled;

  // Cache tile dimensions
  if (debug) {
    const size_t K    = problem.size.k;
    const size_t MT_K = config.mt.k;
    const size_t MI_M = config.mi.m;
    const size_t MI_N = config.mi.n;
    const size_t MI_K = config.mi.k;
    const auto a_bits = datatype_to_bits(problem.a_dtype);
    const auto b_bits = datatype_to_bits(problem.b_dtype);

    OLOG_DEBUG("======== Origami Debug Info ========");
    OLOG_DEBUG("ProblemSize (MxNxBxK): " << int(M) << "x" << int(N) << "x" << int(batch) << "x"
                                         << int(K));
    OLOG_DEBUG("MacroTile: " << int(MT_M) << "x" << int(MT_N) << "x" << int(MT_K));
    OLOG_DEBUG("MatrixInstruction: " << int(MI_M) << "x" << int(MI_N) << "x" << int(MI_K));
    OLOG_DEBUG("ElementSizeA (bits): " << int(a_bits));
    OLOG_DEBUG("ElementSizeB (bits): " << int(b_bits));
    OLOG_DEBUG("CacheHintsA: " << int(config.cache_hints_a));
    OLOG_DEBUG("CacheHintsB: " << int(config.cache_hints_b));

    OLOG_DEBUG("Grid: " << int(grid_m) << "x" << int(grid_n));
    OLOG_DEBUG("NumOutputTiles: " << int(num_output_tiles));
    OLOG_DEBUG("NumWGs: " << int(num_wgs));
    OLOG_DEBUG("NumTimesteps: " << int(num_timesteps));
    OLOG_DEBUG("SplittingFactor: " << int(splitting_factor));
    OLOG_DEBUG("ReductionStrategy: " << int(reduction_strategy));

    OLOG_DEBUG("ActiveCUs: " << int(active_cus));
    OLOG_DEBUG("ReadMemBWFactor: " << mem_bw_limited);
    OLOG_DEBUG("WriteMemBWFactor: " << write_mem_bw_limited);
    OLOG_DEBUG("RealOccupancy: " << real_occupancy);
    OLOG_DEBUG("OccupancyFactor: " << occupancy_factor);

    OLOG_DEBUG("CHUNKxXCCxWGM: " << int(wgm.wgmxccchunk) << "x" << int(wgm.wgmxcc) << "x"
                                 << int(wgm.wgm));
  }
}

bool context_t::is_valid() const {
  return grid_m > 0 && grid_n > 0 && num_output_tiles > 0 && splitting_factor > 0 && num_wgs > 0 &&
         num_timesteps > 0 && active_cus > 0 && mem_bw_limited > 0.0 && tile_elements > 0 &&
         output_tile_bytes > 0 && a_bytes > 0 && b_bytes > 0 && d_bytes > 0;
}

/* ---------------------------------------------------------------------------------------- */
/* Helper functions                                                                         */
/* ---------------------------------------------------------------------------------------- */
// Calculate the work utilization which is the ratio of the useful problem volume to the
// total scheduled volume.
double calculate_work_utilization(const problem_t& problem, const config_t& config) {
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;
  const size_t K = problem.size.k;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  if (MT_M <= 0 || MT_N <= 0) return 1.0;

  // Calculate the full dimensions covered by the launched grid of tiles (spatial).
  const double launched_M =
      static_cast<double>(math::safe_ceil_div(M, MT_M)) * static_cast<double>(MT_M);
  const double launched_N =
      static_cast<double>(math::safe_ceil_div(N, MT_N)) * static_cast<double>(MT_N);

  // Calculate the full depth covered by the k-loop iterations (temporal).
  const double launched_K =
      static_cast<double>(math::safe_ceil_div(K, MT_K)) * static_cast<double>(MT_K);

  // The utilization is the ratio of the useful problem volume to the total scheduled volume.
  const double useful_volume   = static_cast<double>(M * N * K);
  const double launched_volume = launched_M * launched_N * launched_K;

  if (launched_volume < 1.0) return 1.0;  // Avoid division by zero for tiny/empty problems

  const double utilization = useful_volume / launched_volume;

  return utilization;
}

// Calculate the output utilization which is the ratio of the useful problem volume to the
// total scheduled volume.
double calculate_output_utilization(const problem_t& problem,
                                    const config_t& config,
                                    size_t vector_elems = 1) {
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  if (MT_M <= 0 || MT_N <= 0) return 1.0;

  // Tiled coverage in M/N
  const double launched_M =
      static_cast<double>(math::safe_ceil_div(M, MT_M)) * static_cast<double>(MT_M);
  const double launched_N =
      static_cast<double>(math::safe_ceil_div(N, MT_N)) * static_cast<double>(MT_N);

  // Optional: model vectorization/alignment remainders (e.g., ld/st width)
  // This assumes vectors must be fully inside bounds; tail elements are scalarized.
  const size_t M_vec = (vector_elems > 1) ? math::safe_ceil_div(M, vector_elems) * vector_elems : M;
  const size_t N_vec = (vector_elems > 1) ? math::safe_ceil_div(N, vector_elems) * vector_elems : N;

  const double useful   = static_cast<double>(M_vec) * static_cast<double>(N_vec);
  const double launched = launched_M * launched_N;

  if (launched < 1.0) return 1.0;
  return useful / launched;
}

// Round the number of elements to the nearest multiple of 128 bytes.
size_t round_elements_to_128B(size_t elements, size_t element_size_bits) {
  auto round_up_mul             = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
  const size_t transaction_bits = 128u * 8u;  // 1024
  const size_t g                = std::gcd(element_size_bits, transaction_bits);
  const size_t E_block          = transaction_bits / g;  // elements per 128B-aligned chunk
  return round_up_mul(elements, E_block);
}

/* ---------------------------------------------------------------------------------------- */
/* Misc. functions                                                                          */
/* ---------------------------------------------------------------------------------------- */
// Fast WGM prediction: mirrors select_workgroup_mapping's cheap paths, then
// evaluates L2 working set cost for the last XCD in the first timestep.
workgroup_mapping_t predict_workgroup_mapping(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const config_t& config,
                                              size_t grid_m,
                                              size_t grid_n,
                                              size_t splitting_factor) {
  // Extract parameters
  const size_t batch = problem.batch;

  const size_t N_CU    = hardware.N_CU;
  const size_t NUM_XCD = hardware.NUM_XCD;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);

  // Set up parameters
  const size_t numMTs      = grid_m * grid_n;
  const size_t cus_per_xcd = N_CU / NUM_XCD;

  // Batch case
  if (batch > 1) {
    auto numMTs_total = numMTs * batch;
    if (numMTs == 1 || numMTs_total <= NUM_XCD || numMTs % NUM_XCD == 0)
      return {0, 0, 1};
    else
      return {0, NUM_XCD, 1};
  }

  // Non-temporal
  const int nta = config.cache_hints_a;
  const int ntb = config.cache_hints_b;
  if (nta > 3 || ntb > 3) {
    bool use_wgmxcc   = (grid_m != 1 && grid_n != 1);
    size_t out_wgmxcc = use_wgmxcc ? NUM_XCD : 1;
    bool use_chunk =
        use_wgmxcc && ((numMTs < N_CU && numMTs % NUM_XCD == 0) || (numMTs % N_CU == 0));
    size_t out_chunk = use_chunk ? std::min(math::safe_ceil_div(numMTs, NUM_XCD), cus_per_xcd) : 0;

    if (nta > 3 && ntb < 4)
      return {out_chunk, out_wgmxcc, use_wgmxcc ? static_cast<int32_t>(grid_n) : 1};
    else if (nta < 4 && ntb > 3)
      return {out_chunk, out_wgmxcc, use_wgmxcc ? -static_cast<int32_t>(grid_m) : 1};
    else
      return {0, NUM_XCD, 1};
  }

  // WGMXCC
  size_t out_wgmxcc;
  if (splitting_factor % NUM_XCD == 0)
    out_wgmxcc = 0;
  else if (numMTs <= NUM_XCD)
    out_wgmxcc = 0;
  else
    out_wgmxcc = NUM_XCD;

  // WGM shortcuts
  if (out_wgmxcc == 0 || grid_m == 1 || grid_n == 1) return {0, out_wgmxcc, 1};

  // If the grid is large, use the square root of the number of CUs as the WGM.
  // Solution is not very sensitive to the WGM value in this case.
  const size_t grid_threshold = std::sqrt(N_CU);
  if (grid_m > grid_threshold && grid_n > grid_threshold)
    return {0, out_wgmxcc, static_cast<int32_t>(std::ceil(std::sqrt(N_CU / NUM_XCD)))};

  size_t numWGsPerXCD = std::min(math::safe_ceil_div(numMTs, NUM_XCD), cus_per_xcd);
  // If there is enough work per L2 and the grid_n is small, use the grid_n as the WGM.
  if (numWGsPerXCD >= cus_per_xcd / 2 && grid_n <= 8)
    return {0, out_wgmxcc, static_cast<int32_t>(grid_n)};

  // Build candidate list
  size_t wgm_cap = std::min(grid_n, numWGsPerXCD / 2);
  if (wgm_cap == 0) return {0, out_wgmxcc, 1};

  // Bitmask of candidates: bit i set means i is a WGM candidate.
  // Drawback: cannot handle values more than 64.
  uint64_t cmask = 0;
  for (size_t v : {1, 4, 6})
    if (v <= wgm_cap) cmask |= (1ULL << v);
  for (size_t i = 1; i * i <= wgm_cap; ++i) {
    if (wgm_cap % i == 0) {
      cmask |= (1ULL << i);
      cmask |= (1ULL << (wgm_cap / i));
    }
  }

  // Evaluate L2 cost for last XCD in the first timestep
  const size_t total          = numMTs;
  const size_t last_xcd       = NUM_XCD - 2;
  const size_t group_size     = total >= NUM_XCD ? total / NUM_XCD : total;
  const size_t tiles_this_xcd = std::min(cus_per_xcd, group_size);
  const size_t start          = last_xcd * group_size;
  const size_t count          = (start < total) ? std::min(tiles_this_xcd, total - start) : 0;

  const double a_cost = static_cast<double>(MT_M) * a_bytes;
  const double b_cost = static_cast<double>(MT_N) * b_bytes;

  size_t best_wgm  = 1;
  double best_cost = std::numeric_limits<double>::max();
  for (uint64_t m = cmask; m; m &= m - 1) {
    size_t wgm_candidate = static_cast<size_t>(__builtin_ctzll(m));
    size_t slab_tiles    = grid_m * wgm_candidate;
    size_t first_slab    = start / slab_tiles;
    size_t last_slab     = (start + count - 1) / slab_tiles;
    size_t first_row     = (start % slab_tiles) / wgm_candidate;
    size_t last_row      = ((start + count - 1) % slab_tiles) / wgm_candidate;

    size_t unique_rows, unique_cols;
    if (first_slab == last_slab) {
      unique_rows = last_row - first_row + 1;
      unique_cols = (unique_rows > 1) ? wgm_candidate : std::min(count, wgm_candidate);
    } else {
      unique_rows = (last_slab - first_slab > 1)
                        ? grid_m
                        : std::min(grid_m, (grid_m - first_row) + (last_row + 1));
      unique_cols = std::min((last_slab - first_slab + 1) * wgm_candidate, grid_n);
    }
    unique_rows = std::min(unique_rows, grid_m);
    unique_cols = std::min(unique_cols, grid_n);

    double cost = unique_rows * a_cost + unique_cols * b_cost;
    if (cost < best_cost) {
      best_cost = cost;
      best_wgm  = wgm_candidate;
    }
  }

  return {0, out_wgmxcc, static_cast<int32_t>(best_wgm)};
}

// Compute the launch parameters for the kernel.
std::tuple<reduction_t, size_t, size_t, size_t, size_t> compute_launch_parameters(
    const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config,
    grid_selection_t grid_selection,
    size_t max_cus) {
  const reduction_t reduction_strategy =
      streamk::select_reduction(problem, hardware, config, grid_selection);
  auto config_with_reduction               = config;
  config_with_reduction.reduction_strategy = reduction_strategy;
  const size_t num_wgs =
      streamk::select_grid_size(problem, hardware, config_with_reduction, grid_selection, max_cus);

  const size_t num_mts = streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
  // That means, we artifically observe one full timesteps, but that is not what actually happens
  // under the hood. From a theoretical point of view, these distributions change all of the
  // computations in Origami. With current implementation, it is hard to capture that
  // behaviour analytically. So for now, if the num_wgs is less than the num_mts, we calculate
  // num_timesteps based on the num_mts. Otherwise, we use num_wgs to compute num_timesteps.
  const size_t num_active_cus   = num_wgs < hardware.N_CU ? num_wgs : hardware.N_CU;
  const size_t num_timesteps    = num_wgs > num_mts ? math::safe_ceil_div(num_wgs, hardware.N_CU)
                                                    : math::safe_ceil_div(num_mts, hardware.N_CU);
  const size_t splitting_factor = math::safe_ceil_div(num_wgs, num_mts);

  return std::make_tuple(
      reduction_strategy, num_wgs, num_active_cus, num_timesteps, splitting_factor);
}

// Check if MT fits in LDS
bool check_lds_capacity(const hardware_t& hardware,
                        const dim3_t& mt,
                        const data_type_t& a_dtype,
                        const data_type_t& b_dtype) {
  const auto a_loads_in_bytes = mt.mk() * data_type_to_bytes(a_dtype);
  const auto b_loads_in_bytes = mt.nk() * data_type_to_bytes(b_dtype);
  const auto LDS_usage        = a_loads_in_bytes + b_loads_in_bytes;

  return LDS_usage <= hardware.lds_capacity;
}

// Compute limited achievable memory bandwidth based on active CUs
double compute_mem_bw_from_occupancy(const hardware_t& hardware, size_t num_active_cus) {
  const double CUs = static_cast<double>(num_active_cus);

  if (num_active_cus > hardware.N_CU) return 1.0;

  const double bw_limited = std::get<0>(hardware.mem_bw_per_wg_coefficients) * CUs * CUs +
                            std::get<1>(hardware.mem_bw_per_wg_coefficients) * CUs +
                            std::get<2>(hardware.mem_bw_per_wg_coefficients);
  return std::min(bw_limited, 1.0);
}

// Map a linear workgroup ID to 4D tile coordinates (k, m, n, b).
dim4_t wgm_to_grid(const dim4_t& grid, const workgroup_mapping_t& wgm_mapping, size_t id) {
  // Dispatch layout (outermost to innermost): batch -> MN slabs -> K splits.
  // WGM > 0 (row-major): slabs of WGM columns, M varies fastest within each slab.
  // WGM < 0 (col-major): slabs of |WGM| rows, N varies fastest within each slab.
  // Negative WGM is equivalent to transposing M/N, applying row-major, and swapping back.

  // Extract parameters
  const size_t wgmxcc     = wgm_mapping.wgmxcc;
  const size_t slab_width = static_cast<size_t>(std::abs(wgm_mapping.wgm));
  const bool col_major    = wgm_mapping.wgm < 0;

  // WGMXCC: remap dispatch ID so consecutive IDs land on the same XCD.
  if (wgmxcc > 1) {
    const size_t total      = grid.total();
    const size_t group_size = total / wgmxcc;
    id                      = (id / wgmxcc) + (id % wgmxcc) * group_size;
    if (id >= total) id = total - 1;
  }

  // For col-major, swap M and N so the same slab logic applies.
  const size_t g_m = col_major ? grid.n : grid.m;
  const size_t g_n = col_major ? grid.m : grid.n;

  dim4_t tile;
  const size_t tiles_per_batch = grid.mnk();
  tile.b                       = id / tiles_per_batch;
  const size_t within_batch    = id % tiles_per_batch;
  const size_t mn_linear       = within_batch / grid.k;
  tile.k                       = within_batch % grid.k;

  // Decode MN slab position from the linear MN index.
  if (slab_width == 0) {
    tile.m = 0;
    tile.n = 0;
    return tile;
  }
  const size_t tiles_per_slab      = g_m * slab_width;
  const size_t num_full_slabs      = g_n / slab_width;
  const size_t full_slabs_coverage = num_full_slabs * tiles_per_slab;

  size_t out_m, out_n;
  if (mn_linear < full_slabs_coverage) {
    const size_t slab_idx       = mn_linear / tiles_per_slab;
    const size_t offset_in_slab = mn_linear % tiles_per_slab;
    out_m                       = offset_in_slab / slab_width;
    out_n                       = slab_idx * slab_width + offset_in_slab % slab_width;
  } else {
    const size_t remainder_width = g_n - num_full_slabs * slab_width;
    if (remainder_width == 0) {
      out_m = g_m - 1;
      out_n = g_n - 1;
    } else {
      const size_t offset_in_remainder = mn_linear - full_slabs_coverage;
      out_m                            = offset_in_remainder / remainder_width;
      out_n = num_full_slabs * slab_width + offset_in_remainder % remainder_width;
    }
  }

  // Swap back for col-major.
  tile.m = col_major ? out_n : out_m;
  tile.n = col_major ? out_m : out_n;
  return tile;
}

// Count unique tile coordinates (k, m, n, b) touched by a contiguous range of
// workgroup IDs in raw dispatch order (no WGMXCC).
// Negative wgm means column-major (N varies fastest), handled by swapping M/N.
dim4_t count_unique_range(const dim4_t& grid, int wgm, size_t start, size_t count) {
  const bool col_major = wgm < 0;
  dim4_t unique;
  const size_t end             = start + count - 1;
  const size_t tiles_per_batch = grid.mnk();

  // First find the batch index.
  const size_t first_batch = start / tiles_per_batch;
  const size_t last_batch  = end / tiles_per_batch;
  unique.b                 = std::min(last_batch - first_batch + 1, grid.b);

  // Next find the K-split index. If the range stays within one MN tile and one batch,
  // only a subset of K-splits are touched; otherwise all K-splits are covered.
  const size_t first_within_batch = start % tiles_per_batch;
  const size_t last_within_batch  = end % tiles_per_batch;
  const size_t first_mn           = first_within_batch / grid.k;
  const size_t last_mn            = last_within_batch / grid.k;
  if (first_mn == last_mn && unique.b == 1) {
    unique.k = std::min((last_within_batch % grid.k) - (first_within_batch % grid.k) + 1, grid.k);
  } else {
    unique.k = grid.k;
  }

  // Early exit: if multiple batches or all MN tiles are covered:
  const size_t num_mn_tiles = math::safe_ceil_div(count, grid.k);
  if (unique.b > 1 || num_mn_tiles >= grid.mn()) {
    unique.m = grid.m;
    unique.n = grid.n;
    return unique;
  }

  // For col-major (negative WGM), swap M/N so the same slab logic applies.
  const size_t g_m     = col_major ? grid.n : grid.m;
  const size_t g_n     = col_major ? grid.m : grid.n;
  const size_t abs_wgm = static_cast<size_t>(std::abs(wgm));

  const size_t slab_width = std::min(abs_wgm, g_n);
  if (slab_width == 0) {
    unique.m = 0;
    unique.n = 0;
    return unique;
  }
  const size_t tiles_per_slab      = g_m * slab_width;
  const size_t num_full_slabs      = g_n / slab_width;
  const size_t full_slabs_coverage = num_full_slabs * tiles_per_slab;
  const size_t remainder_width     = g_n - num_full_slabs * slab_width;

  size_t first_fast, first_slow;
  if (first_mn < full_slabs_coverage) {
    size_t offset = first_mn % tiles_per_slab;
    first_fast    = offset / slab_width;
    first_slow    = (first_mn / tiles_per_slab) * slab_width + offset % slab_width;
  } else if (remainder_width > 0) {
    size_t offset = first_mn - full_slabs_coverage;
    first_fast    = offset / remainder_width;
    first_slow    = num_full_slabs * slab_width + offset % remainder_width;
  } else {
    first_fast = g_m - 1;
    first_slow = g_n - 1;
  }

  size_t last_fast, last_slow;
  if (last_mn < full_slabs_coverage) {
    size_t offset = last_mn % tiles_per_slab;
    last_fast     = offset / slab_width;
    last_slow     = (last_mn / tiles_per_slab) * slab_width + offset % slab_width;
  } else if (remainder_width > 0) {
    size_t offset = last_mn - full_slabs_coverage;
    last_fast     = offset / remainder_width;
    last_slow     = num_full_slabs * slab_width + offset % remainder_width;
  } else {
    last_fast = g_m - 1;
    last_slow = g_n - 1;
  }

  size_t unique_fast, unique_slow;
  const size_t first_slab = first_slow / slab_width;
  const size_t last_slab  = last_slow / slab_width;
  if (first_slab == last_slab) {
    unique_fast = last_fast - first_fast + 1;
    const size_t actual_slab_w =
        std::min((first_slab + 1) * slab_width, g_n) - first_slab * slab_width;
    unique_slow = (unique_fast > 1) ? actual_slab_w : (last_slow - first_slow + 1);
  } else {
    unique_fast =
        (last_slab - first_slab > 1) ? g_m : std::min(g_m, (g_m - first_fast) + (last_fast + 1));

    // Account for partial slab occupancy at boundaries instead of taking the
    // full bounding-box span from first_slab start to last_slab end.
    const size_t first_slab_w =
        std::min((first_slab + 1) * slab_width, g_n) - first_slab * slab_width;
    const size_t last_slab_w = std::min((last_slab + 1) * slab_width, g_n) - last_slab * slab_width;
    const size_t first_in_slab = first_slow - first_slab * slab_width;
    const size_t last_in_slab  = last_slow - last_slab * slab_width;

    // First slab: if >1 M-row remains, all N columns are visited; otherwise only the tail.
    const size_t n_first = (first_fast < g_m - 1) ? first_slab_w : (first_slab_w - first_in_slab);
    // Last slab: if >1 M-row is used, all N columns are visited; otherwise only the head.
    const size_t n_last = (last_fast > 0) ? last_slab_w : (last_in_slab + 1);
    // Middle slabs are fully covered.
    const size_t n_mid =
        (last_slab > first_slab + 1) ? (last_slab - first_slab - 1) * slab_width : 0;

    unique_slow = std::min(n_first + n_mid + n_last, g_n);
  }

  unique_fast = std::min(unique_fast, g_m);
  unique_slow = std::min(unique_slow, g_n);

  // Swap back for col-major.
  unique.m = col_major ? unique_slow : unique_fast;
  unique.n = col_major ? unique_fast : unique_slow;
  return unique;
}

// Count unique tiles for a specific XCD during a specific timestep.
// With wgmxcc: XCD x sees cus_per_xcd consecutive tiles in raw dispatch order.
// Without wgmxcc: XCD x gets every num_xcd-th tile (round-robin), so the tiles
// are strided — we compute unique k/m/n/b analytically for the strided set.
dim4_t count_unique_tiles(const dim4_t& grid,
                          const workgroup_mapping_t& wgm_mapping,
                          size_t N_CU,
                          size_t num_xcd,
                          size_t xcd_id,
                          size_t timestep_id) {
  // wgmxcc is either num_xcd (contiguous blocks) or 0 (round-robin).
  //
  // Contiguous (wgmxcc = num_xcd):
  // Each XCD gets a contiguous block of total/num_xcd tiles in dispatch order.
  // Round-robin (wgmxcc = 0):
  // Happens when splitting_factor % num_xcd == 0 (or tiny grids).
  // XCD x gets tiles x, x+num_xcd, x+2*num_xcd, ...
  // Since grid.k is a multiple of num_xcd, gcd(stride, grid.k) = num_xcd,
  // so mn_stride = 1: MN tiles are visited consecutively (only K is strided).
  // This is equivalent to a contiguous range in a grid with reduced K.

  if (N_CU == 0 || num_xcd == 0 || grid.m == 0 || grid.n == 0 || grid.k == 0) return {0, 0, 0, 0};

  const int signed_wgm       = wgm_mapping.wgm;
  const size_t total         = grid.total();
  const size_t cus_per_xcd   = N_CU / num_xcd;
  const size_t tiles_per_xcd = total / num_xcd;
  const size_t tiles_per_ts  = std::min(cus_per_xcd, tiles_per_xcd);

  if (wgm_mapping.wgmxcc > 1) {
    const size_t xcd_base = xcd_id * tiles_per_xcd;
    const size_t start    = xcd_base + timestep_id * tiles_per_ts;
    const size_t remaining =
        (start < xcd_base + tiles_per_xcd) ? xcd_base + tiles_per_xcd - start : 0;
    const size_t count = std::min(tiles_per_ts, remaining);
    if (count == 0) return {0, 0, 0, 0};
    return count_unique_range(grid, signed_wgm, start, count);
  }

  // Round-robin: XCD x gets tiles x, x+num_xcd, x+2*num_xcd, ...
  const size_t stride     = num_xcd;
  const size_t first_tile = timestep_id * N_CU + xcd_id;
  if (first_tile >= total) return {0, 0, 0, 0};
  const size_t count = std::min(cus_per_xcd, (total - first_tile + stride - 1) / stride);
  if (count == 0) return {0, 0, 0, 0};

  // When grid.k >= num_xcd (split-K with splitting_factor % num_xcd == 0),
  // gcd(stride, grid.k) = num_xcd, so mn_stride = 1: MN tiles are consecutive.
  // Treat as contiguous range in a reduced-K grid.
  if (grid.k >= num_xcd) {
    const size_t k_per_xcd     = math::safe_ceil_div(grid.k, num_xcd);
    const dim4_t reduced_grid  = {k_per_xcd, grid.m, grid.n, grid.b};
    const size_t reduced_total = reduced_grid.total();
    const size_t rs            = timestep_id * tiles_per_ts;
    const size_t rcount =
        std::min(tiles_per_ts, (rs < reduced_total) ? reduced_total - rs : static_cast<size_t>(0));
    if (rcount == 0) return {0, 0, 0, 0};
    dim4_t unique = count_unique_range(reduced_grid, signed_wgm, rs, rcount);
    unique.k      = std::min(unique.k, k_per_xcd);
    return unique;
  }

  // Small grid (numMTs <= num_xcd): stride jumps across batches.
  // Use GCD-based analysis since MN tiles are strided, not consecutive.
  const size_t tiles_per_batch = grid.mnk();
  const size_t mn              = grid.mn();
  dim4_t unique;

  const size_t gcd_k = std::gcd(stride, grid.k);
  unique.k           = std::min(count, grid.k / gcd_k);

  const size_t mn_stride = stride / gcd_k;
  const size_t gcd_mn    = std::gcd(mn_stride, mn);
  size_t unique_mn       = std::min(count / unique.k, mn / gcd_mn);
  if (unique_mn == 0 && count > 0) unique_mn = 1;

  // count unique batches from strided tile IDs.
  {
    size_t prev_b         = SIZE_MAX;
    size_t unique_batches = 0;
    for (size_t i = 0; i < count; ++i) {
      size_t b = (first_tile + i * stride) / tiles_per_batch;
      if (b != prev_b) {
        ++unique_batches;
        prev_b = b;
      }
    }
    unique.b = std::min(unique_batches, grid.b);
  }

  // With small grids (numMTs < num_xcd), unique_mn is tiny — direct computation is fast.
  if (unique_mn >= mn) {
    unique.m = grid.m;
    unique.n = grid.n;
  } else if (signed_wgm == 0) {
    unique.m = 0;
    unique.n = 0;
  } else {
    unique.m = std::min(unique_mn, grid.m);
    unique.n = std::min(unique_mn, grid.n);
  }
  return unique;
}

// Count unique tiles for an entire timestep (all XCDs combined).
dim4_t count_unique_tiles_timestep(const dim4_t& grid,
                                   const workgroup_mapping_t& wgm_mapping,
                                   size_t N_CU,
                                   size_t timestep_id) {
  const size_t total = grid.total();
  const size_t start = timestep_id * N_CU;
  const size_t count = std::min(N_CU, total > start ? total - start : static_cast<size_t>(0));

  return count_unique_range(grid, wgm_mapping.wgm, start, count);
}

/* ---------------------------------------------------------------------------------------- */
/* Compute-related functions                                                                */
/* ---------------------------------------------------------------------------------------- */
// Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K tile.
size_t compute_number_matrix_instructions(dim3_t mt, dim3_t mi) {
  // Compute the number of Matrix Instructions required in each dim.
  size_t num_m_instrs = math::safe_ceil_div(mt.m, mi.m);
  size_t num_n_instrs = math::safe_ceil_div(mt.n, mi.n);
  size_t num_k_instrs = math::safe_ceil_div(mt.k, mi.k);

  // Total number of matrix instructions.
  size_t num_matrix_instrs = num_m_instrs * num_n_instrs * num_k_instrs;

  return num_matrix_instrs;
}

// Compute arithmic intensity
double arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  // Numerator: 2.0 * m * n * k
  // Denominator: (m*n + n*k + m*k) * bytes_per_element
  double numerator   = 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;

  if (denominator == 0) return 0.0;
  return numerator / denominator;
}

// Computes Emulated arithmetic intensity for TF32 (assumes 3xBF16).
double emulated_tf32_arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  // Numerator: 3.0 * 2.0 * m * n * k
  // Denominator: (m*n + n*k + m*k) * bytes_per_element
  double numerator   = 3.0 * 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;

  if (denominator == 0) return 0.0;
  return numerator / denominator;
}

// Compute cvt overhead in x1 tf32 emulation
// TODO: We can generalize the same routine to cover more GEMMs that perform conversion
double compute_cvt_overhead_x1(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config) {
  // In X1 TF32 GEMMs, we do:
  // v_cvt_pk_bf16_f32  (convert/pack fp32 to bf16)
  // v_cvt_pk_bf16_f32  (convert/pack fp32 to bf16)
  // ds_write_b64
  // That is, the extra instructions that we need to account for are the two cvt_pk ops
  // per wavefront tile

  // However, these extra ops should not be added up to the overal tile latency becuase
  // they can be run in parallel to Matix and Memory operations (given they are not dependent).
  // So, We should ideally take L_tile = max{Mem, Comp, Vec (cvt latencies)}.
  // Since, Vec latency is not modeled yet, we somehow model that into the current logic
  // by scaling according to MFMA latencies and putting some heuristics to model the fact
  // that these vector operations can be hidden (read interleaved) with the other memory
  // or MFMA instructions.

  // --- Shorthands -----------------------------------------------------------
  const double MT_M = static_cast<double>(config.mt.m);
  const double MT_N = static_cast<double>(config.mt.n);
  const double MT_K = static_cast<double>(config.mt.k);

  const double MI_M = static_cast<double>(config.mi.m);
  const double MI_N = static_cast<double>(config.mi.n);
  const double MI_K = static_cast<double>(config.mi.k);

  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);

  // TODO: Use kernel's actual wavetiles (wavefront's tile size).
  const double wave_tile_m = MT_M / 2.0;
  const double wave_tile_n = MT_N / 2.0;
  const double wave_tile_k = MT_K / MI_K;

  // MFMA count
  const double N_MI     = (wave_tile_m / MI_M) * (wave_tile_n / MI_N) * wave_tile_k;
  const double num_mfma = 1.0 * N_MI;
  // Cycle scale per MI
  const double L_MI        = hardware.get_mi_latency(MI_M, MI_N, MI_K, problem.mi_dtype);
  const double mfma_cycles = num_mfma * L_MI;

  // 2) Bytes (per K-slice), using ceil-div to whole bytes
  const double bytesA = wave_tile_m * MT_K * static_cast<double>(a_bytes);
  const double bytesB = wave_tile_n * MT_K * static_cast<double>(b_bytes);

  // 3) Modeled transfer quanta (128B lines)
  //      dsA = bytesA / (128 * MI_M)
  //      dsB = bytesB / (128 * MI_N)
  //      GR  = dsA  (global->LDS modeled equal to A-side DS)
  const double dsA = (bytesA / 128.0) / MI_M;  // LDS->VGPR for A
  const double dsB = (bytesB / 128.0) / MI_N;  // LDS->VGPR for B
  const double GR  = dsA;                      // Global->LDS reads
  const double LR  = dsA + dsB;                // total DS->VGPR

  // 5) Exposed vs hidden CVT
  // spare MFMA
  const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
  // 2 cvt per each ds_write (this for SS_BSS -- should be revised for other datatypes)
  // Each cvt has a latency of four. It is scaled by the MI Latency
  // Note: change 16.0 based on mi_data_type if we want to generalize this for all
  // casting GEMMs.
  const double cvt = (2.0 * 4.0 / 16.0 * L_MI) * LR;
  // cvt ops are interleaved in main loop and don't stall matrix or memory units.
  // Heuristically, we set
  const double H        = (8.0 / 16.0 * L_MI) * spare_mfma + (4.0 / 16.0) * L_MI * (LR + GR);
  const double overhead = std::max(cvt - H, 0.0);

  return overhead;
}

// Compute cvt overhead in tf32 emulation
double compute_cvt_overhead(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config) {
  // Wavefront tile sizes
  // TODO: Use kernel's actual wavetiles (wavefront's tile size).
  const double wave_tile_m = config.mt.m / 2.0;
  const double wave_tile_n = config.mt.n / 2.0;
  const double wave_tile_k = config.mt.k / config.mi.k;

  // MFMA count and cycles
  const double N_MI = (wave_tile_m / config.mi.m) * (wave_tile_n / config.mi.n) * wave_tile_k;

  // TF32 emu: 3× BF16 MI issue slots
  const double num_mfma = 3.0 * static_cast<double>(N_MI);

  // Cycle scale per MI (use BF16 MI latency as the basic timing quantum)
  const double L_MI_bf16 =
      hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, data_type_t::BFloat16);
  // const double mfma_cycles = num_mfma * L_MI_bf16;

  // 2) Bytes (per K-slice), using ceil-div to whole bytes
  auto a_bytes = data_type_to_bytes(problem.a_dtype);
  auto b_bytes = data_type_to_bytes(problem.b_dtype);

  const double bytesA = static_cast<double>(wave_tile_m) * config.mt.k * a_bytes;
  const double bytesB = static_cast<double>(wave_tile_n) * config.mt.k * b_bytes;

  // const double mt_bytesA
  //     = static_cast<double>(MT_M) * MT_K * safe_ceil_div(element_size_A, 8);

  // 3) Modeled transfer quanta (128B lines)
  //      dsA = bytesA / (128 * MI_M)
  //      dsB = bytesB / (128 * MI_N)
  //      GR  = dsA  (global->LDS modeled equal to A-side DS)
  const double dsA = (bytesA / 128.0) / static_cast<double>(config.mi.m);  // LDS->VGPR for A
  const double dsB = (bytesB / 128.0) / static_cast<double>(config.mi.n);  // LDS->VGPR for B
  const double GR  = dsA;                                                  // Global->LDS reads
  const double LR  = dsA + dsB;                                            // total DS->VGPR

  // 4) Heuristic cycle weights (scaled to MI latency).
  //    Preserves your A=104, B=8, C=4 when L_MI_bf16 == 16.
  // 24 vector instructions per 2 ds_reads (16x16x32)
  // 24 vector instructions per 2 ds_reads for A and for B.
  // 3 instructions per fp32 value read; number ds_read * size
  const double A = (104.0 / 16.0) * L_MI_bf16;  // CVT per LR-sized chunk (DS->VGPR)
  const double B = (8.0 / 16.0) * L_MI_bf16;    // hidden per spare MFMA slot
  // MI16: 16 - 4 (12 cycles), for those 4 cycles, VGPRs are locked. 8 cycles to do anything.
  const double C = (4.0 / 16.0) * L_MI_bf16;  // hidden per (LR+GR) slot     // MI16
  // 32 cycles (mfma), 4 cycles, 28, 4 vgpr lock, 24 cycles left.
  // 24: 6 conv instructions, 3 ds_reads, ~6 grs

  // 5) Exposed vs hidden CVT
  const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
  const double cvt        = A * dsA;                         // only DS->VGPR contributes CVT
  const double H          = B * spare_mfma + C * (LR + GR);  // hidden cycles
  const double overhead   = std::max(cvt - H, 0.0);

  // 6) Efficiency
  // const double denom = mfma_cycles + overhead;
  // const double eff   = (denom > 0.0) ? (mfma_cycles / denom) : 1;

  return overhead;
}

// Determine the compute latency per MT_MxMT_NxMT_K Macro Tile (L_MT).
size_t compute_mt_compute_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config) {
  // Compute the number of matrix instructions
  size_t N_MI = compute_number_matrix_instructions(config.mt, config.mi);
  // Latency of a single MT_MxMT_NxMT_k tile is the latency of one MI multiplied by
  // number of MI per MT_MxMT_NxMT_k.
  size_t L_MI = hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, problem.mi_dtype);

  size_t L_MT = L_MI * N_MI;

  return L_MT;
}

/* ---------------------------------------------------------------------------------------- */
/* Memory-related functions                                                                 */
/* ---------------------------------------------------------------------------------------- */
// MALL tile dimensions: how many concurrent M/N tiles fit when all CUs share MALL.
// The MALL sees all CUs' traffic, so the tile footprint spans the full active_cus range.
std::pair<size_t, size_t> compute_mall_tiles(size_t grid_m,
                                             size_t grid_n,
                                             size_t active_cus,
                                             size_t wgm_value) {
  if (grid_m == 0 || grid_n == 0 || active_cus == 0) return {0, 0};

  const size_t W          = std::max(wgm_value, static_cast<size_t>(1));
  const size_t slab_tiles = grid_m * std::min(W, grid_n);
  const size_t full_slabs = std::min(active_cus / std::max(slab_tiles, static_cast<size_t>(1)),
                                     grid_n / std::min(W, grid_n));
  const size_t mall_n =
      std::min(std::max((full_slabs + 1) * std::min(W, grid_n), static_cast<size_t>(1)), grid_n);
  const size_t mall_m =
      std::min(math::safe_ceil_div(active_cus,
                                   std::max(mall_n / std::min(W, grid_n), static_cast<size_t>(1)) *
                                       std::min(W, grid_n)),
               grid_m);
  return {std::max(mall_m, static_cast<size_t>(1)), std::max(mall_n, static_cast<size_t>(1))};
}

// L2 tile dimensions: how many tiles share one XCD's L2, shrunk to fit capacity.
// Each XCD has its own L2; only CUs on that XCD contribute traffic.
std::pair<size_t, size_t> compute_l2_tiles(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           size_t grid_m,
                                           size_t grid_n,
                                           size_t active_cus,
                                           size_t splitting_factor,
                                           size_t wgm_value) {
  if (grid_m == 0 || grid_n == 0 || active_cus == 0) return {0, 0};

  const size_t num_xcd = std::max(hardware.NUM_XCD, static_cast<size_t>(1));
  // With splitting, total WGs = grid_m * grid_n * splitting_factor.
  // Each XCD sees its share of those WGs.
  const size_t total_wgs   = grid_m * grid_n * std::max(splitting_factor, static_cast<size_t>(1));
  const size_t wgs_per_xcd = std::min(active_cus / num_xcd, total_wgs / num_xcd);
  if (wgs_per_xcd == 0) return {1, 1};

  // Per-XCD footprint in MN space (splitting doesn't add new M/N tiles, but
  // each XCD may process fewer MN tiles when splitting increases total WGs)
  const size_t effective_mn_per_xcd =
      math::safe_ceil_div(wgs_per_xcd, std::max(splitting_factor, static_cast<size_t>(1)));
  auto [mall_m, mall_n] = compute_mall_tiles(grid_m, grid_n, effective_mn_per_xcd, wgm_value);

  // Capacity check: shrink if the working set exceeds L2
  const size_t split_K =
      math::safe_ceil_div(problem.size.k, std::max(splitting_factor, static_cast<size_t>(1)));
  const double a_bytes =
      static_cast<double>(config.mt.m) * split_K * data_type_to_bytes(problem.a_dtype);
  const double b_bytes =
      static_cast<double>(config.mt.n) * split_K * data_type_to_bytes(problem.b_dtype);
  const double l2_cap = static_cast<double>(hardware.L2_capacity);

  size_t l2_m = mall_m;
  size_t l2_n = mall_n;
  while (l2_m * a_bytes + l2_n * b_bytes > l2_cap && (l2_m > 1 || l2_n > 1)) {
    if (l2_m * a_bytes > l2_n * b_bytes && l2_m > 1)
      --l2_m;
    else if (l2_n > 1)
      --l2_n;
    else
      --l2_m;
  }
  return {std::max(l2_m, static_cast<size_t>(1)), std::max(l2_n, static_cast<size_t>(1))};
}

// Estimate L2 hit rate
double estimate_l2_hit(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       const context_t& context) {
  const size_t wgm_val = static_cast<size_t>(std::abs(context.wgm.wgm));
  auto [l2_m, l2_n]    = compute_l2_tiles(problem,
                                       hardware,
                                       config,
                                       context.grid_m,
                                       context.grid_n,
                                       context.active_cus,
                                       context.splitting_factor,
                                       wgm_val);

  const long long uA = static_cast<long long>(l2_m) * config.mt.mk();
  const long long uB = static_cast<long long>(l2_n) * config.mt.nk();
  const long long total =
      std::max(uA * static_cast<long long>(l2_n) + uB * static_cast<long long>(l2_m), 1LL);
  const long long cached = total - (uA + uB);

  return std::max(0.0, std::min(static_cast<double>(cached) / total, 1.0));
}

// Estimate MALL hit-rate
double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         const context_t& context) {
  const size_t wgm_val = static_cast<size_t>(std::abs(context.wgm.wgm));
  auto [mall_m, mall_n] =
      compute_mall_tiles(context.grid_m, context.grid_n, context.active_cus, wgm_val);

  const long long uA = static_cast<long long>(mall_m) * config.mt.mk();
  const long long uB = static_cast<long long>(mall_n) * config.mt.nk();
  const long long total =
      std::max(uA * static_cast<long long>(mall_n) + uB * static_cast<long long>(mall_m), 1LL);
  const long long cached = total - (uA + uB);

  return std::max(0.0, std::min(static_cast<double>(cached) / total, 1.0));
}

// Compute L2 hit rate from a global (problem-wide) perspective
double compute_l2_hit_rate_global(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  size_t l2_capacity_bytes) {
  if (l2_capacity_bytes == 0) throw std::runtime_error("L2 Capacity is zero");

  // 1. Calculate the grid dimensions in terms of macro-tiles
  const size_t grid_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  const size_t grid_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  if (grid_m == 0 || grid_n == 0)
    throw std::runtime_error("estimate_l2_hit grid dimensions can not be zero");

  // 2. Calculate the working set size for one full pass of global reuse
  // This is the data needed by one full column of CUs (for A) and one full row (for B).
  const double a_bytes = data_type_to_bytes(problem.a_dtype);
  const double b_bytes = data_type_to_bytes(problem.b_dtype);

  const double a_working_set           = static_cast<double>(grid_m * config.mt.mk()) * a_bytes;
  const double b_working_set           = static_cast<double>(grid_n * config.mt.nk()) * b_bytes;
  const double total_working_set_bytes = a_working_set + b_working_set;

  // 3. CRUCIAL: Check if the working set fits in the L2 cache.
  // If it doesn't, the global reuse pattern is broken by capacity misses,
  // and the hit rate will be very low.
  if (total_working_set_bytes > l2_capacity_bytes) {
    // Return a floor value for the hit rate. The exact value can be tuned,
    // but it should be low to indicate that the ideal reuse is not possible.
    return 0.1;  // 10% hit rate
  }

  // 4. If it fits, calculate the idealized global hit rate
  // Total reads if nothing was cached
  const double total_A_reads = static_cast<double>(grid_m * grid_n * config.mt.mk());
  const double total_B_reads = static_cast<double>(grid_m * grid_n * config.mt.nk());

  // Uncached reads are the first-time fetches for each row/column
  const double uncached_A_reads =
      static_cast<double>(grid_m * config.mt.mk());  // One full column fetches A
  const double uncached_B_reads =
      static_cast<double>(grid_n * config.mt.nk());  // One full row fetches B

  const double total_reads = total_A_reads + total_B_reads;
  if (total_reads == 0) return 1.0;  // No reads, perfect hit rate.

  const double cached_reads =
      (total_A_reads - uncached_A_reads) + (total_B_reads - uncached_B_reads);

  return cached_reads / total_reads;
}

// Estimate per-operand cache hit rates for L1, L2, and MALL.
cache_hit_rates_t estimate_cache_hit_rates(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           const context_t& context) {
  // Extract parameters
  const size_t num_xcd     = hardware.NUM_XCD;
  const size_t N_CU        = hardware.N_CU;
  const double l2_cap      = static_cast<double>(hardware.L2_capacity);
  const size_t k_per_split = context.k_per_split;
  const auto& wgm          = context.wgm;
  const bool debug         = context.debug;
  const double a_bytes     = context.a_bytes;
  const double b_bytes     = context.b_bytes;
  const double k_iters     = static_cast<double>(context.k_iters);
  const auto& heuristic    = context.heuristic;

  // Setup
  const dim4_t grid  = {context.splitting_factor, context.grid_m, context.grid_n, problem.batch};
  const size_t total = grid.total();
  const size_t cus_per_xcd   = N_CU / num_xcd;
  const size_t tiles_per_xcd = total / num_xcd;
  const double k_iters_sq    = k_iters * k_iters;

  // Helper function to clamp values between 0 and 1
  auto clamp01 = [](double v) { return std::max(0.0, std::min(v, 1.0)); };

  if (N_CU == 0 || total == 0 || grid.m == 0 || grid.n == 0) return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  // Per-tile data volumes (contiguous dimension rounded to 128B cache lines).
  constexpr double cl   = 128.0;
  const bool a_trans    = (problem.a_transpose == transpose_t::T);
  const bool b_trans    = (problem.b_transpose == transpose_t::T);
  const bool a_temporal = config.cache_hints_a <= 3;
  const bool b_temporal = config.cache_hints_b <= 3;

  const double a_contig = a_trans ? config.mt.k * a_bytes : config.mt.m * a_bytes;
  const double a_outer  = a_trans ? config.mt.m : config.mt.k;
  const double b_contig = b_trans ? config.mt.n * b_bytes : config.mt.k * b_bytes;
  const double b_outer  = b_trans ? config.mt.k : config.mt.n;

  const double a_iter = a_outer * std::ceil(a_contig / cl) * cl;
  const double b_iter = b_outer * std::ceil(b_contig / cl) * cl;
  const double a_row  = a_iter / static_cast<double>(config.mt.k);
  const double b_row  = b_iter / static_cast<double>(config.mt.k);
  const double a_tile = a_row * k_per_split;
  const double b_tile = b_row * k_per_split;

  // ----
  // MALL
  double mall_rate_a = 0.0;
  double mall_rate_b = 0.0;
  dim4_t mall_tiles;
  if (hardware.has_MALL()) {
    // Deep K-loops allow prefetch stagger. Shallow loops cause simultaneous loads with
    // minimal sharing across CUs.
    const double mall_warmup =
        heuristic.mall_cold_floor +
        (1.0 - heuristic.mall_cold_floor) * k_iters_sq / (k_iters_sq + heuristic.mall_depth_sq);

    // Count unique tiles for the first timestep
    mall_tiles = count_unique_tiles_timestep(grid, wgm, N_CU, 0);

    // Calculate the unique bytes loaded
    const double mall_total_a = a_temporal ? std::min(N_CU, total) * a_tile : 0.0;
    const double mall_total_b = b_temporal ? std::min(N_CU, total) * b_tile : 0.0;
    const double mall_total_pb_a =
        a_temporal ? static_cast<double>(mall_tiles.m) * mall_tiles.n * a_tile : 0.0;
    const double mall_total_pb_b =
        b_temporal ? static_cast<double>(mall_tiles.m) * mall_tiles.n * b_tile : 0.0;
    const double mall_unique_a = a_temporal ? mall_tiles.m * a_tile : 0.0;
    const double mall_unique_b = b_temporal ? mall_tiles.n * b_tile : 0.0;
    const double mall_reused_a = mall_total_pb_a - mall_unique_a;
    const double mall_reused_b = mall_total_pb_b - mall_unique_b;
    const double mall_cached_a =
        (mall_total_pb_a > 0) ? (mall_reused_a / mall_total_pb_a) * mall_total_a : 0.0;
    const double mall_cached_b =
        (mall_total_pb_b > 0) ? (mall_reused_b / mall_total_pb_b) * mall_total_b : 0.0;
    mall_rate_a = (mall_total_a > 0) ? clamp01((mall_cached_a / mall_total_a) * mall_warmup) : 0.0;
    mall_rate_b = (mall_total_b > 0) ? clamp01((mall_cached_b / mall_total_b) * mall_warmup) : 0.0;
  }

  // ----
  // L2
  // Pick only one XCD. Herein, we pick the second to last XCD.
  // First and last XCDs are not used because they are bounded by the grid dimensions.
  const size_t xcd_id = (num_xcd > 2) ? num_xcd - 2 : 0;
  // Count the unique tiles on the XCD
  const dim4_t l2_tiles = count_unique_tiles(grid, wgm, N_CU, num_xcd, xcd_id, 0);
  // Calculate the concurrent load on the XCD
  const double a_conc          = a_temporal ? static_cast<double>(l2_tiles.m) * a_iter : 0.0;
  const double b_conc          = b_temporal ? static_cast<double>(l2_tiles.n) * b_iter : 0.0;
  const double total_conc      = a_conc + b_conc;
  const double concurrent_load = total_conc * l2_tiles.k * l2_tiles.b;

  // Cache-line sharing:
  // When M or N is small, multiple tile rows/columns fit in the same 128B
  // cache lines, reducing the actual unique bytes loaded.
  double a_cl_factor = std::min(1.0, problem.size.m * a_bytes / cl);
  double b_cl_factor = std::min(1.0, problem.size.n * b_bytes / cl);

  // Spatial Reuse:
  const size_t rectangular_mn = l2_tiles.m * l2_tiles.n;
  const size_t tiles_on_xcd   = std::min(cus_per_xcd, tiles_per_xcd);
  const size_t kb             = std::max(l2_tiles.k * l2_tiles.b, static_cast<size_t>(1));
  const size_t actual_mn      = std::min(math::safe_ceil_div(tiles_on_xcd, kb), rectangular_mn);

  double l2_unique_a    = a_temporal ? l2_tiles.m * a_tile * a_cl_factor : 0.0;
  double l2_unique_b    = b_temporal ? l2_tiles.n * b_tile * b_cl_factor : 0.0;
  double l2_requested_a = a_temporal ? static_cast<double>(actual_mn) * a_tile : 0.0;
  double l2_requested_b = b_temporal ? static_cast<double>(actual_mn) * b_tile : 0.0;
  double spatial_reuse_a =
      (l2_requested_a > 0) ? std::max(1.0 - l2_unique_a / l2_requested_a, 0.0) : 0.0;
  double spatial_reuse_b =
      (l2_requested_b > 0) ? std::max(1.0 - l2_unique_b / l2_requested_b, 0.0) : 0.0;

  // K-split alignment penalty for round-robin dispatch:
  // In round-robin, different MN tiles land on different K-split offsets when
  // grid.k is not a multiple of num_xcd. Only MN tiles at the SAME K-split
  // share A/B data. If they're at different K-splits, no sharing occurs.
  if (wgm.wgmxcc <= 1 && grid.k >= num_xcd) {
    const size_t g                 = std::gcd(context.splitting_factor, num_xcd);
    const size_t mn_sharing_period = num_xcd / g;
    const size_t mn_on_xcd         = l2_tiles.m * l2_tiles.n;
    if (mn_sharing_period > 1 && mn_on_xcd > 1) {
      const size_t sharing_group = std::max(mn_on_xcd / mn_sharing_period, static_cast<size_t>(1));
      const double align_factor =
          static_cast<double>(sharing_group - 1) / static_cast<double>(mn_on_xcd - 1);
      spatial_reuse_a *= align_factor;
      spatial_reuse_b *= align_factor;
    }
  }

  // K-depth warmup:
  // Cold-start penalty for shallow K-loops.
  double l2_warmup = 1.0;
  if (l2_tiles.b == 1) {
    l2_warmup = heuristic.l2_cold_floor +
                (1.0 - heuristic.l2_cold_floor) * k_iters_sq / (k_iters_sq + heuristic.l2_depth_sq);
  }

  // L2 capacity / residency:
  // Estimate per-operand effective working sets so a small reused operand is
  // not penalized as strongly by a large competing stream. Each operand sees
  // its own load plus a share-weighted portion of the other's pressure.
  const double a_load           = a_conc * static_cast<double>(l2_tiles.k * l2_tiles.b);
  const double b_load           = b_conc * static_cast<double>(l2_tiles.k * l2_tiles.b);
  const double total_load       = a_load + b_load;
  const double a_share          = (total_load > 0.0) ? a_load / total_load : 0.0;
  const double b_share          = (total_load > 0.0) ? b_load / total_load : 0.0;
  const double a_interference   = b_load * a_share;
  const double b_interference   = a_load * b_share;
  const double effective_load_a = a_load + a_interference;
  const double effective_load_b = b_load + b_interference;
  const double l2_residency_a   = (a_load > 0.0) ? std::min(l2_cap / effective_load_a, 1.0) : 1.0;
  const double l2_residency_b   = (b_load > 0.0) ? std::min(l2_cap / effective_load_b, 1.0) : 1.0;

  // Pollution penalty:
  // When both operands are temporal, a competing stream can evict lines even if
  // the operand's own footprint would fit. Model that asymmetrically based on
  // how much of each operand's effective working set comes from the other stream.
  double pollution_rate_a = 1.0;
  double pollution_rate_b = 1.0;
  if (a_temporal && b_temporal) {
    if (l2_residency_a < 1.0 && effective_load_a > 0.0) {
      const double interference_frac_a = a_interference / effective_load_a;
      pollution_rate_a = 1.0 - (1.0 - heuristic.l2_pollution_penalty) * interference_frac_a;
    }
    if (l2_residency_b < 1.0 && effective_load_b > 0.0) {
      const double interference_frac_b = b_interference / effective_load_b;
      pollution_rate_b = 1.0 - (1.0 - heuristic.l2_pollution_penalty) * interference_frac_b;
    }
  }

  // Depth pressure (split-K only): larger MT_K loads more data per iteration,
  // increasing L2 pressure. Only matters with K-splits where multiple independent
  // streams compete for L2 space.
  double depth_penalty = 1.0;
  {
    const double depth_ref = cl / std::max(a_bytes, b_bytes);
    if (context.splitting_factor > 1 && config.mt.k > depth_ref)
      depth_penalty = heuristic.l2_depth_penalty;
  }

  // L2 hit rate
  double l2_rate_a =
      pollution_rate_a * l2_warmup * l2_residency_a * spatial_reuse_a * depth_penalty;
  double l2_rate_b =
      pollution_rate_b * l2_warmup * l2_residency_b * spatial_reuse_b * depth_penalty;

  // Request amplification (batched GEMMs only):
  // When the per-iteration working set fits in L2, intra-tile multi-wavefront
  // L1 misses all hit L2 (nothing evicts them), lifting the effective rate
  // toward ~0.9 regardless of spatial sharing.
  bool enable_batched_amp = (problem.batch > 1);
  if (enable_batched_amp && concurrent_load < l2_cap) {
    const double amp_ceiling = heuristic.l2_amp_ceiling_batched;
    const double headroom  = 1.0 - concurrent_load / l2_cap;
    const double amp_boost = headroom * headroom;
    l2_rate_a += amp_boost * std::max(amp_ceiling - l2_rate_a, 0.0);
    l2_rate_b += amp_boost * std::max(amp_ceiling - l2_rate_b, 0.0);
  }

  // Request amplification (small-tile split-K GEMMs only):
  // When the per-iteration working set fits in L2, intra-tile multi-wavefront
  // L1 misses all hit L2 (nothing evicts them), lifting the effective rate
  // toward ~0.4 regardless of spatial sharing.
  bool enable_split_k_amp = (l2_tiles.k > 1 && l2_tiles.m * l2_tiles.n < 5);
  if (enable_split_k_amp && concurrent_load < l2_cap) {
    const double amp_ceiling = heuristic.l2_amp_ceiling_k_split;
    const double headroom  = 1.0 - concurrent_load / l2_cap;
    const double amp_boost = headroom;
    l2_rate_a += amp_boost * std::max(amp_ceiling - l2_rate_a, 0.0);
    l2_rate_b += amp_boost * std::max(amp_ceiling - l2_rate_b, 0.0);
  }

  // Implicit L1 residency + request amplification (skinny-dimension GEMMs):
  // When one output dimension is tiny and the shared temporal operand fits in L1,
  // some of its traffic never reaches L2 at all. If it fits in L2 but not in L1,
  // intra-CU request amplification lifts its L2 hit rate.
  cache_hit_rates_t rates{};
  auto& [H_mem_l1_A, H_mem_l1_B, H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B] = rates;
  {
    const bool skinny_m          = (grid.m <= 2 && grid.n > grid.m * 8);
    const bool skinny_n          = (grid.n <= 2 && grid.m > grid.n * 8);
    const bool single_stream     = (grid.k == 1) && (grid.b == 1);
    constexpr double l1_capacity = 32.0 * 1024.0;

    if (single_stream && skinny_m && a_temporal && !b_temporal) {
      if (a_iter <= l1_capacity) {
        const double headroom = 1.0 - a_iter / l1_capacity;
        H_mem_l1_A            = heuristic.l1_hit_rate_ceiling_skinny * clamp01(headroom);
      } else if (concurrent_load < l2_cap) {
        const double headroom = 1.0 - concurrent_load / l2_cap;
        l2_rate_a += headroom * std::max(heuristic.l2_amp_ceiling_skinny - l2_rate_a, 0.0);
      }
    }

    if (single_stream && skinny_n && b_temporal && !a_temporal) {
      if (b_iter <= l1_capacity) {
        const double headroom = 1.0 - b_iter / l1_capacity;
        H_mem_l1_B            = heuristic.l1_hit_rate_ceiling_skinny * clamp01(headroom);
      } else if (concurrent_load < l2_cap) {
        const double headroom = 1.0 - concurrent_load / l2_cap;
        l2_rate_b += headroom * std::max(heuristic.l2_amp_ceiling_skinny - l2_rate_b, 0.0);
      }
    }
  }

  H_mem_l2_A   = a_temporal ? clamp01(l2_rate_a) : 0.0;
  H_mem_l2_B   = b_temporal ? clamp01(l2_rate_b) : 0.0;
  H_mem_mall_A = a_temporal ? clamp01(mall_rate_a) : 0.0;
  H_mem_mall_B = b_temporal ? clamp01(mall_rate_b) : 0.0;

  if (debug) {
    OLOG_DEBUG("MallTiles: " << mall_tiles.k << " " << mall_tiles.m << " " << mall_tiles.n << " "
                             << mall_tiles.b);
    OLOG_DEBUG("L2Tiles: " << l2_tiles.k << " " << l2_tiles.m << " " << l2_tiles.n << " "
                           << l2_tiles.b);
    OLOG_DEBUG("SpatialReuseA: " << spatial_reuse_a);
    OLOG_DEBUG("SpatialReuseB: " << spatial_reuse_b);
    OLOG_DEBUG("L2Warmup: " << l2_warmup);
    OLOG_DEBUG("PollutionRateA: " << pollution_rate_a);
    OLOG_DEBUG("PollutionRateB: " << pollution_rate_b);
    OLOG_DEBUG("L2ResidencyA: " << l2_residency_a);
    OLOG_DEBUG("L2ResidencyB: " << l2_residency_b);
    OLOG_DEBUG("H_mem_l1_A: " << H_mem_l1_A);
    OLOG_DEBUG("H_mem_l1_B: " << H_mem_l1_B);
    OLOG_DEBUG("H_mem_l2_A: " << H_mem_l2_A);
    OLOG_DEBUG("H_mem_l2_B: " << H_mem_l2_B);
    OLOG_DEBUG("H_mem_mall_A: " << H_mem_mall_A);
    OLOG_DEBUG("H_mem_mall_B: " << H_mem_mall_B);
  }

  return rates;
}

// Utility function to aggregate cache hit rates for debugging.
static std::tuple<double, double, double> aggregate_cache_hit_rates_for_debug(double H_mem_l1_A,
                                                                              double H_mem_l1_B,
                                                                              double H_mem_l2_A,
                                                                              double H_mem_l2_B,
                                                                              double H_mem_mall_A,
                                                                              double H_mem_mall_B,
                                                                              double Ld_A_total,
                                                                              double Ld_B_total,
                                                                              bool a_temporal,
                                                                              bool b_temporal) {
  const double temporal_total   = (a_temporal ? Ld_A_total : 0.0) + (b_temporal ? Ld_B_total : 0.0);
  const double Ld_A_to_l2       = a_temporal ? (1.0 - H_mem_l1_A) * Ld_A_total : 0.0;
  const double Ld_B_to_l2       = b_temporal ? (1.0 - H_mem_l1_B) * Ld_B_total : 0.0;
  const double l2_input_total   = Ld_A_to_l2 + Ld_B_to_l2;
  const double Ld_A_to_mall     = a_temporal ? (1.0 - H_mem_l2_A) * Ld_A_to_l2 : 0.0;
  const double Ld_B_to_mall     = b_temporal ? (1.0 - H_mem_l2_B) * Ld_B_to_l2 : 0.0;
  const double mall_input_total = Ld_A_to_mall + Ld_B_to_mall;

  const double H_mem_l1 = (temporal_total > 0.0)
                              ? ((H_mem_l1_A * (a_temporal ? Ld_A_total : 0.0)) +
                                 (H_mem_l1_B * (b_temporal ? Ld_B_total : 0.0))) /
                                    temporal_total
                              : 0.0;
  const double H_mem_l2 =
      (l2_input_total > 0.0)
          ? ((H_mem_l2_A * Ld_A_to_l2) + (H_mem_l2_B * Ld_B_to_l2)) / l2_input_total
          : 0.0;
  const double H_mem_mall =
      (mall_input_total > 0.0)
          ? ((H_mem_mall_A * Ld_A_to_mall) + (H_mem_mall_B * Ld_B_to_mall)) / mall_input_total
          : 0.0;

  return {H_mem_l1, H_mem_l2, H_mem_mall};
}

// Determine the memory latency
double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const context_t& context) {
  const bool debug = context.debug;

  // Extract parameters from structured types
  const auto a_bits  = datatype_to_bits(problem.a_dtype);
  const auto b_bits  = datatype_to_bits(problem.b_dtype);
  const bool a_trans = (problem.a_transpose == transpose_t::T);
  const bool b_trans = (problem.b_transpose == transpose_t::T);

  const auto a_bytes          = context.a_bytes;
  const auto b_bytes          = context.b_bytes;
  const size_t num_active_cus = context.active_cus;
  double bw_limited           = context.mem_bw_limited;
  auto heuristic              = context.heuristic;

  // 1) Estimate per-operand L1/MALL/L2 hit-rates using the analytical model
  const auto [H_mem_l1_A, H_mem_l1_B, H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B] =
      estimate_cache_hit_rates(problem, hardware, config, context);

  // 2) Total loads per CU (A + B, with 128B alignment and MX scales)
  size_t Ld_A      = a_trans ? config.mt.m * round_elements_to_128B(config.mt.k, a_bits)
                             : round_elements_to_128B(config.mt.m, a_bits) * config.mt.k;
  size_t Ld_B      = b_trans ? round_elements_to_128B(config.mt.n, b_bits) * config.mt.k
                             : config.mt.n * round_elements_to_128B(config.mt.k, b_bits);
  auto Ld_CU_bytes = (Ld_A * a_bytes) + (Ld_B * b_bytes);

  // Block scaled datatypes (MX): add scale bytes
  if (a_bits < 8 && problem.a_mx_block_size != 0)
    Ld_CU_bytes += math::safe_ceil_div(config.mt.mk(), problem.a_mx_block_size);
  if (b_bits < 8 && problem.b_mx_block_size != 0)
    Ld_CU_bytes += math::safe_ceil_div(config.mt.nk(), problem.b_mx_block_size);

  // 3) Total loads by all CUs, split by operand
  double Ld_A_total = static_cast<double>(Ld_A * a_bytes) * num_active_cus;
  double Ld_B_total = static_cast<double>(Ld_B * b_bytes) * num_active_cus;
  double total_Ld   = Ld_CU_bytes * static_cast<double>(num_active_cus);

  const bool a_nontemporal = config.cache_hints_a > 3;
  const bool b_nontemporal = config.cache_hints_b > 3;
  const bool a_temporal    = !a_nontemporal;
  const bool b_temporal    = !b_nontemporal;

  // 4) L2 latency (bandwidth-limited by CU occupancy ratio)
  double l2_bw = hardware.mem1_perf_ratio * static_cast<double>(num_active_cus) /
                 static_cast<double>(hardware.N_CU);

  // Temporal traffic first passes through an implicit L1 stage. Nontemporal
  // traffic bypasses L1/L2/MALL caching and is charged directly to DRAM.
  double Ld_A_to_l2 = a_temporal ? (1.0 - H_mem_l1_A) * Ld_A_total : 0.0;
  double Ld_B_to_l2 = b_temporal ? (1.0 - H_mem_l1_B) * Ld_B_total : 0.0;
  double Ld_l2      = Ld_A_to_l2 + Ld_B_to_l2;
  double L_mem_l2   = (l2_bw > 0) ? (Ld_l2 / l2_bw) : 0.0;

  double Ld_A_after_l2 = a_temporal ? (1.0 - H_mem_l2_A) * Ld_A_to_l2 : 0.0;
  double Ld_B_after_l2 = b_temporal ? (1.0 - H_mem_l2_B) * Ld_B_to_l2 : 0.0;

  double Ld_A_mall = hardware.has_MALL() ? Ld_A_after_l2 : 0.0;
  double Ld_B_mall = hardware.has_MALL() ? Ld_B_after_l2 : 0.0;
  double Ld_mall   = Ld_A_mall + Ld_B_mall;

  double Ld_A_dram = a_nontemporal
                         ? Ld_A_total
                         : (hardware.has_MALL() ? (1.0 - H_mem_mall_A) * Ld_A_mall : Ld_A_after_l2);
  double Ld_B_dram = b_nontemporal
                         ? Ld_B_total
                         : (hardware.has_MALL() ? (1.0 - H_mem_mall_B) * Ld_B_mall : Ld_B_after_l2);
  double Ld_dram   = Ld_A_dram + Ld_B_dram;

  // 7) MALL latency
  double mall_bw    = hardware.mem2_perf_ratio * bw_limited;
  double L_mem_mall = (mall_bw > 0) ? (Ld_mall / mall_bw) : 0.0;

  // 8) DRAM latency
  double dram_bw    = hardware.mem3_perf_ratio * bw_limited;
  double L_mem_dram = (dram_bw > 0) ? (Ld_dram / dram_bw) : 0.0;
  L_mem_dram += heuristic.main_memory_load_latency;

  // 9) Worst-case across all memory levels
  double L_mem = std::max({L_mem_l2 * heuristic.weight_mem_l2,
                           L_mem_mall * heuristic.weight_mem_mall,
                           L_mem_dram * heuristic.weight_mem_dram});

  if (debug) {
    const auto [H_mem_l1, H_mem_l2, H_mem_mall] = aggregate_cache_hit_rates_for_debug(H_mem_l1_A,
                                                                                      H_mem_l1_B,
                                                                                      H_mem_l2_A,
                                                                                      H_mem_l2_B,
                                                                                      H_mem_mall_A,
                                                                                      H_mem_mall_B,
                                                                                      Ld_A_total,
                                                                                      Ld_B_total,
                                                                                      a_temporal,
                                                                                      b_temporal);
    OLOG_DEBUG("H_mem_mall: " << H_mem_mall);
    OLOG_DEBUG("H_mem_l2: " << H_mem_l2);
    OLOG_DEBUG("H_mem_l1: " << H_mem_l1);
    OLOG_DEBUG("Ld_CU_bytes: " << Ld_CU_bytes);
    OLOG_DEBUG("total_Ld: " << total_Ld);
    OLOG_DEBUG("Ld_l2: " << Ld_l2);
    OLOG_DEBUG("Ld_dram: " << Ld_dram);
    OLOG_DEBUG("Ld_mall: " << Ld_mall);
    OLOG_DEBUG("L_mem_l2: " << L_mem_l2);
    OLOG_DEBUG("L_mem_mall: " << L_mem_mall);
    OLOG_DEBUG("L_mem_dram: " << L_mem_dram);
  }

  return L_mem;
}

/* ---------------------------------------------------------------------------------------- */
/* Tile-related functions                                                                   */
/* ---------------------------------------------------------------------------------------- */
// Determine the epilogue latency of a single tile.
double compute_epilogue_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const context_t& context) {
  // In epilogue:
  // 1. ACC -> VGPR
  // 2. Alpha/beta scaling
  // 3. Bias operations
  // 4. Activation functions
  // 5. Accumulator conversions
  // 6. Global memory stores

  // Items 2, 3, 4 are conditionally executed based on the problem.
  // For instance, if Beta=0, we skip a bunch of operations.
  // We skip bias and activation functions if they are not present.
  // Herein, we consider the simplest case for now: Alpha=1, Beta=0, and no bias/activation
  // functions. Skipping items 2, 3, 4, and 5 for now.

  // Extract parameters
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;

  const size_t N_CU = hardware.N_CU;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  const size_t num_active_cus          = context.active_cus;
  const size_t splitting_factor        = context.splitting_factor;
  const double d_bytes                 = context.d_bytes;
  const size_t grid_m                  = context.grid_m;
  const size_t grid_n                  = context.grid_n;
  const size_t num_output_tiles        = context.num_output_tiles;
  const double store_bw                = hardware.mem3_perf_ratio * context.mem_bw_limited;
  const double reduce_bw               = hardware.mem3_perf_ratio * context.write_mem_bw_limited;
  const bool debug                     = context.debug;
  const reduction_t reduction_strategy = context.reduction_strategy;
  const bool is_parallel_reduction     = (reduction_strategy == reduction_t::parallel);
  const auto& heuristic                = context.heuristic;

  if (d_bytes == 0.0) return 0.0;

  // Common setup
  const size_t total_mfmas =
      math::safe_ceil_div(MT_M, config.mi.m) * math::safe_ceil_div(MT_N, config.mi.n);
  const size_t elements_per_vectorized_store =
      static_cast<size_t>(std::ceil(heuristic.epilogue_bytes_per_vectorized_store / d_bytes));
  const size_t elements_per_cache_line =
      static_cast<size_t>(std::ceil(heuristic.epilogue_cache_line_bytes / d_bytes));
  const double alignment_penalty = (M % elements_per_cache_line != 0) ? 1.1 : 1.0;

  // Per-CU write bandwidth: total write BW shared among all writers
  // During epilogue store, ALL active WGs write simultaneously:
  // Non-finishing WGs write partials to workspace
  // Finishing WGs (or all WGs if no split) write final output
  const size_t num_writers     = num_active_cus;
  const double per_cu_store_bw = store_bw / static_cast<double>(num_writers);

  // Edge tile detection
  const bool has_interior  = (M >= MT_M && N >= MT_N);
  const bool has_m_edge    = (M % MT_M != 0);
  const bool has_n_edge    = (N % MT_N != 0);
  const size_t m_remainder = has_m_edge ? (M % MT_M) : MT_M;
  const size_t n_remainder = has_n_edge ? (N % MT_N) : MT_N;

  // Helper function to compute the epilogue cost for a given tile type
  auto compute_tile_epilogue = [&](size_t tile_m, size_t tile_n, bool is_scalar_path) -> double {
    // Scalar path is the m_edge tiles (including corner tiles)
    // 1) ACC -> VGPR
    size_t acc_reads      = is_scalar_path ? 2 * total_mfmas : total_mfmas;
    double L_acc_transfer = acc_reads * heuristic.epilogue_cycles_per_acc_read *
                            heuristic.epilogue_acc_read_parallelism;

    // 2) Bounds checking (edge tiles only)
    double L_edge_check   = 0.0;
    size_t total_elements = tile_m * tile_n;
    if (is_scalar_path) {
      double store_instr = math::safe_ceil_div(total_elements, heuristic.epilogue_threads_per_wave);
      L_edge_check       = store_instr * heuristic.epilogue_cycles_per_bounds_check;
    } else if (tile_m != MT_M || tile_n != MT_N) {
      size_t store_instr = math::safe_ceil_div(
          total_elements, heuristic.epilogue_threads_per_wave * elements_per_vectorized_store);
      L_edge_check = store_instr * heuristic.epilogue_cycles_per_bounds_check;
    }

    // 3) Store: per-tile bytes through per-CU bandwidth share
    // Split-K WGs write partials as f32 (4 bytes) to workspace, not d_dtype.
    double store_elem_bytes = (splitting_factor > 1 && !is_parallel_reduction)
                                  ? static_cast<double>(heuristic.epilogue_workspace_bytes_per_elem)
                                  : d_bytes;
    double store_bytes      = static_cast<double>(tile_m) * tile_n * store_elem_bytes;
    double store_scale      = is_scalar_path ? heuristic.epilogue_scalar_store_penalty : 1.0;
    double L_store          = (store_bytes * store_scale * alignment_penalty) / per_cu_store_bw;

    // 4) Per-tile K-split reduction (in-kernel: spinlock/tree/atomic)
    // After all WGs write partials, only the finishing WGs (one per output tile) are active.
    // They read partials from workspace, accumulate, then write final output.
    // Contention during reduction is much lower: only grid_m x grid_n finishing WGs are reading.
    double L_reduce = 0.0;
    if (splitting_factor > 1 && !is_parallel_reduction) {
      size_t n_partials = splitting_factor - 1;
      // Only finishing WGs (one per output tile) are active during reduction.
      double per_cu_reduce_bw = reduce_bw / static_cast<double>(num_output_tiles);
      // Partials are stored as f32 in workspace
      double partial_bytes = static_cast<double>(n_partials) * tile_m * tile_n *
                             heuristic.epilogue_workspace_bytes_per_elem;

      // Sync cost
      // Per partial: poll flag + barrier + reset flag + SRD setup + loop control
      // The finishing WG must wait for the first partner to finish storing its
      // partial before the poll succeeds. This wait ≈ the partner's store time.
      double L_poll_wait = store_bytes / per_cu_store_bw;
      double L_sync =
          L_poll_wait + static_cast<double>(n_partials) *
                            (heuristic.epilogue_salu_overhead + 2.0 * heuristic.epilogue_l_barrier +
                             heuristic.epilogue_l_smem);

      double L_partial_read = partial_bytes / per_cu_reduce_bw;
      double L_accumulate =
          static_cast<double>(n_partials * tile_m * tile_n) / heuristic.epilogue_threads_per_wave;
      double L_partial_write = static_cast<double>(tile_m) * tile_n * d_bytes / per_cu_reduce_bw;
      L_reduce               = L_sync + L_partial_read + L_accumulate + L_partial_write;

      // Small tiles don't benefit from split-K: the fixed sync overhead
      // per partial dominates the tiny per-WG compute, and workspace traffic is
      // proportionally large.
      if (tile_m * tile_n <= 2048) L_reduce *= 2.0;
    }

    return L_acc_transfer + L_edge_check + L_store + L_reduce;
  };

  // Evaluate all tile types
  double L_epilogue_interior = 0.0;
  double L_epilogue_n_edge   = 0.0;
  double L_epilogue_m_edge   = 0.0;
  double L_epilogue_corner   = 0.0;
  if (has_interior) L_epilogue_interior = compute_tile_epilogue(MT_M, MT_N, false);
  if (has_n_edge) L_epilogue_n_edge = compute_tile_epilogue(MT_M, n_remainder, false);
  if (has_m_edge) L_epilogue_m_edge = compute_tile_epilogue(m_remainder, MT_N, true);
  if (has_m_edge && has_n_edge)
    L_epilogue_corner = compute_tile_epilogue(m_remainder, n_remainder, true);

  // Take the worst case
  double L_epilogue =
      std::max({L_epilogue_interior, L_epilogue_n_edge, L_epilogue_m_edge, L_epilogue_corner});

  // if there are more output tiles than the number of CUs, we only consider
  // the dominant dimension.
  if (num_output_tiles >= 2 * N_CU) {
    if (grid_m > grid_n)
      L_epilogue = std::max(L_epilogue_interior, L_epilogue_n_edge);
    else
      L_epilogue = std::max(L_epilogue_interior, L_epilogue_m_edge);
  }

  if (debug) {
    OLOG_DEBUG("L_epilogue_interior: " << L_epilogue_interior);
    OLOG_DEBUG("L_epilogue_n_edge: " << L_epilogue_n_edge);
    OLOG_DEBUG("L_epilogue_m_edge: " << L_epilogue_m_edge);
    OLOG_DEBUG("L_epilogue_corner: " << L_epilogue_corner);
  }

  return L_epilogue;
}

// Compute the latency to compute a tile
double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            const context_t& context) {
  // Extract parameters from structured types
  const size_t K = problem.size.k;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  const auto a_bits = datatype_to_bits(problem.a_dtype);
  const auto b_bits = datatype_to_bits(problem.b_dtype);

  // Extract parameters from context
  const size_t splitting_factor = context.splitting_factor;
  const size_t k_per_split      = context.k_per_split;
  const auto& heuristic         = context.heuristic;
  const bool debug              = context.debug;

  // 1) Compute per-tile latencies
  double L_compute = compute_mt_compute_latency(problem, hardware, config);
  double L_mem     = compute_memory_latency(problem, hardware, config, context);

  double utilization            = calculate_work_utilization(problem, config);
  double effective_tile_penalty = (utilization > 1e-9) ? (1.0 / (utilization)) : 1.0;

  // 2) Work-group setup & iteration latencies
  double L_WG_setup = 1;

  // 3) Prologue and Epilogue latencies
  const size_t real_occupancy   = context.real_occupancy;
  const double occupancy_factor = context.occupancy_factor;

  // 3-1) Prologue
  double L_prologue = L_mem;
  L_prologue *= effective_tile_penalty;
  L_prologue *= occupancy_factor;

  // 3-2) Epilogue (per-tile store + optional in-kernel reduction)
  // Core epilogue (compute + stores + reduction) scaled by occupancy decay.
  // K-padding penalty added outside the decay (it's a structural mismatch, not a latency).
  double L_epilogue =
      (L_compute + compute_epilogue_latency(problem, hardware, config, context)) * occupancy_factor;

  double problem_k_quant = 0.0;
  if (K % MT_K != 0) {
    problem_k_quant = static_cast<double>(K % MT_K) / static_cast<double>(K);
    L_epilogue += problem_k_quant * heuristic.epilogue_k_padding_penalty;
  }

  // 4) Single-tile main-loop latency (pipelined: compute overlaps memory)
  double L_cvt = 0;
  // TODO: gfx90a also lacks native TF32 and should get CVT overhead,
  // but enabling it changes rankings — address in a separate PR.
  if (!hardware.has_native_TF32() && hardware.arch != hardware_t::architecture_t::gfx90a) {
    if (problem.mi_dtype == data_type_t::XFloat32) {
      L_cvt = compute_cvt_overhead(problem, hardware, config);
    } else if ((a_bits == 32) && (b_bits == 32) && (problem.mi_dtype == data_type_t::BFloat16)) {
      L_cvt = compute_cvt_overhead_x1(problem, hardware, config);
    }
  }
  double L_tile_single =
      std::max(L_compute * heuristic.weight_compute, L_mem * heuristic.weight_memory);
  L_tile_single *= (splitting_factor > 4) ? 1.0 : heuristic.main_loop_efficiency;
  L_tile_single *= effective_tile_penalty;
  L_tile_single += L_cvt;

  // 5) Number of K-iterations (excluding epilogue), at least 1
  long num_iter =
      std::max(static_cast<long>(math::safe_ceil_div(k_per_split, MT_K) - 1), static_cast<long>(1));

  // 6) Total tile latency
  double L_tile_total = L_tile_single * static_cast<double>(num_iter);
  L_tile_total += heuristic.weight_prologue * L_prologue;
  L_tile_total += heuristic.weight_epilogue * L_epilogue;
  L_tile_total += heuristic.weight_wg_setup * L_WG_setup;
  L_tile_total += heuristic.weight_loop_overhead * static_cast<double>(num_iter);

  // Apply final tile total weight
  L_tile_total *= heuristic.weight_tile_total;

  if (debug) {
    OLOG_DEBUG("utilization: " << utilization);
    OLOG_DEBUG("effective_tile_penalty: " << effective_tile_penalty);

    OLOG_DEBUG("L_mem: " << L_mem);
    OLOG_DEBUG("L_compute: " << L_compute);
    OLOG_DEBUG("L_cvt: " << L_cvt);
    OLOG_DEBUG("k_per_split: " << k_per_split);
    OLOG_DEBUG("num_iter: " << int(num_iter));
    OLOG_DEBUG("problem_k_quant: " << problem_k_quant);
    OLOG_DEBUG("L_prologue: " << L_prologue);
    OLOG_DEBUG("L_tile_single: " << L_tile_single);
    OLOG_DEBUG("L_epilogue: " << L_epilogue);
    OLOG_DEBUG("L_tile_total: " << L_tile_total);
  }

  return L_tile_total;
}

// Compute the latency of a timestep.
double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const context_t& context) {
  // Assume latency of a timestep is latency of a single K-complete output tile computed on one CU.
  double L_timestep = compute_tile_latency(problem, hardware, config, context);

  return L_timestep;
}

// Compute the latency of the PostGSU parallel reduction kernel.
double compute_parallel_reduction_latency(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const config_t& config,
                                          const context_t& context) {
  // Single kernel launch and flat reduction.
  // Each thread reads ALL splitting_factor partials from workspace (f32),
  // accumulates them sequentially, and writes one output element (d_dtype).
  // For small GSU (4/8/16) the reads are fully unrolled; for larger GSU a loop is used.

  // Only applies to parallel reduction with splitting
  if (context.splitting_factor <= 1 || context.reduction_strategy != reduction_t::parallel)
    return 0.0;

  const auto& heuristic = context.heuristic;

  // Extract parameters
  const size_t M               = problem.size.m;
  const size_t N               = problem.size.n;
  const size_t batch           = problem.batch;
  const size_t output_elements = M * N * batch;

  const size_t splitting_factor = context.splitting_factor;
  const double d_bytes          = context.d_bytes;

  if (d_bytes == 0.0) return 0.0;

  // Each thread processes VW output elements.
  const size_t VW = std::max(static_cast<size_t>(1), static_cast<size_t>(4.0 / d_bytes));
  const size_t total_wgs =
      math::safe_ceil_div(output_elements, heuristic.postgsu_threads_per_wg * VW);
  const size_t active_wgs = std::min(total_wgs, hardware.N_CU);
  const size_t timesteps  = math::safe_ceil_div(total_wgs, hardware.N_CU);

  // Bandwidth based on occupancy of the reduction kernel
  // Assuming data resides in MALL.
  double read_bw = hardware.mem2_perf_ratio * compute_mem_bw_from_occupancy(hardware, active_wgs);
  read_bw        = std::max(read_bw, 1e-12);

  // Total data movement per timestep:
  //   Read:  active_wgs × threads_per_wg × VW × splitting_factor × compute_bytes
  //   Write: active_wgs × threads_per_wg × VW × d_bytes
  double elements_per_ts = static_cast<double>(active_wgs) * heuristic.postgsu_threads_per_wg * VW;
  double read_bytes_per_ts  = elements_per_ts * splitting_factor * heuristic.postgsu_compute_bytes;
  double write_bytes_per_ts = elements_per_ts * d_bytes;

  // Per-timestep latency: read + accumulate + write
  double L_read  = read_bytes_per_ts / read_bw;
  double L_write = write_bytes_per_ts / read_bw;
  // Accumulate: each thread sequentially adds (splitting_factor-1) values.
  // All 64 lanes in a wavefront execute in parallel, but each WG processes
  // its own slice serially.
  double L_acc = static_cast<double>(splitting_factor - 1) * elements_per_ts /
                 (active_wgs * heuristic.postgsu_wavefront_size);

  double L_total =
      heuristic.postgsu_kernel_launch_overhead + (L_read + L_acc + L_write) * timesteps;

  if (context.debug) {
    OLOG_DEBUG("L_parallel_reduce_active_wgs: " << active_wgs);
    OLOG_DEBUG("L_parallel_reduce_timesteps: " << timesteps);
    OLOG_DEBUG("L_parallel_reduce_bw: " << read_bw);
    OLOG_DEBUG("L_parallel_reduce_reads: " << L_read);
    OLOG_DEBUG("L_parallel_reduce_accumulates: " << L_acc);
    OLOG_DEBUG("L_parallel_reduce_writes: " << L_write);
  }

  return L_total;
}

double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             size_t max_cus) {
  assert(config.is_valid());

  // Use Formocast simulation model if prediction_mode is set to simulation
  if (config.prediction_mode == prediction_modes_t::simulation) {
    return compute_formocast_latency(problem, hardware, config);
  }

  // Extract parameters from structured types
  size_t M     = problem.size.m;
  size_t N     = problem.size.n;
  size_t K     = problem.size.k;
  size_t batch = problem.batch;

  bool a_trans = problem.a_transpose == transpose_t::T;
  bool b_trans = problem.b_transpose == transpose_t::T;

  size_t MT_M = config.mt.m;
  size_t MT_N = config.mt.n;
  size_t MT_K = config.mt.k;
  size_t MI_M = config.mi.m;
  size_t MI_N = config.mi.n;
  size_t MI_K = config.mi.k;

  const int a_bits = datatype_to_bits(problem.a_dtype);
  const int b_bits = datatype_to_bits(problem.b_dtype);

  // 0) Short-circuit
  // We don't need to compute latency for all MTs. With this, we can shortcut.
  bool shortCircuit = true;
  if (shortCircuit) {
    // When problem dimensions are small enough that we can fit them in one tile, we should do
    // so. This short circuit condition also decreases selection latency when problems are very
    // small :)
    // TODO 256 and 256 here should be largest M and N tile dimensions in library
    if (M <= 256 && N <= 256 && K < 1024 && batch != 1 && (MT_M < M || MT_N < N))
      return std::numeric_limits<double>::max();

    // Use Dot2 only for M < 3
    if (MI_M == 1 && MI_N == 1 && MI_K == 64 && M > 2) return std::numeric_limits<double>::max();

    size_t K_mod_128bytes    = K * a_bits % 1024;
    size_t MT_K_mod_128bytes = MT_K * a_bits % 1024;
    if (K_mod_128bytes == 0 && MT_K_mod_128bytes == 0) {
      // avoid division by 0 if K == 0
      if (M <= MT_M * 2 && !b_trans && ((N * b_bits) / (M * a_bits) > 5)) {
        // Use nontemporal B
        if (!(config.cache_hints_b == 4)) { return std::numeric_limits<double>::max(); }
      } else if (N <= MT_N * 2 && a_trans && ((M * a_bits) / (N * b_bits) > 5)) {
        // Use Non Temporal A
        if (!(config.cache_hints_a == 4)) { return std::numeric_limits<double>::max(); }
      } else {
        // Never use Non Temporal
        if (config.cache_hints_a || config.cache_hints_b) {
          return std::numeric_limits<double>::max();
        }
      }
    } else if (config.cache_hints_a || config.cache_hints_b) {
      return std::numeric_limits<double>::max();
    }
  }

  // 1) Setup context (computes grid dims, launch params, WGM, etc.)
  context_t context(problem, hardware, config);

  // 2) Compute latency of a timestep
  double L_timestep = compute_timestep_latency(problem, hardware, config, context);

  // 3) Compute latency for all timesteps with linear scaling
  double total_latency = L_timestep * context.num_timesteps;

  //  4) Add parallel reduction kernel cost (separate kernel launch, 0 if not parallel)
  double L_parallel_reduce = compute_parallel_reduction_latency(problem, hardware, config, context);
  total_latency += L_parallel_reduce;

  if (context.debug) {
    OLOG_DEBUG("L_parallel_reduce: " << L_parallel_reduce);
    OLOG_DEBUG("total_latency: " << total_latency);
    OLOG_DEBUG("=================================");
  }

  return total_latency;
}

static double compute_formocast_latency(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config) {
  // Create Formocast simulator instance
  Formocast formocast;

  // Convert problem_t to Formocast::ProblemInfo
  Formocast::ProblemInfo prob_info;
  prob_info.M              = static_cast<double>(problem.size.m);
  prob_info.N              = static_cast<double>(problem.size.n);
  prob_info.K              = static_cast<double>(problem.size.k);
  prob_info.NumBatches     = static_cast<double>(problem.batch);
  prob_info.bpeA           = static_cast<uint32_t>(datatype_to_bits(problem.a_dtype) / 8);
  prob_info.bpeB           = static_cast<uint32_t>(datatype_to_bits(problem.b_dtype) / 8);
  prob_info.bpeD           = static_cast<uint32_t>(datatype_to_bits(problem.d_dtype) / 8);
  prob_info.bpeCompute     = static_cast<uint32_t>(datatype_to_bits(problem.mi_dtype) / 8);
  prob_info.transA         = (problem.a_transpose == transpose_t::T);
  prob_info.transB         = (problem.b_transpose == transpose_t::T);
  prob_info.swizzleTensorA = config.tensile().swizzle_a;
  prob_info.swizzleTensorB = config.tensile().swizzle_b;
  prob_info.dataType       = problem.mi_dtype;

  // Convert config_t to Formocast::SizeMapping
  Formocast::SizeMapping size_mapping;
  size_mapping.macroTile[0]         = static_cast<int>(config.mt.m);
  size_mapping.macroTile[1]         = static_cast<int>(config.mt.n);
  size_mapping.macroTile[2]         = static_cast<int>(config.mt.k);
  size_mapping.matrixInstruction[0] = static_cast<int>(config.mi.m);
  size_mapping.matrixInstruction[1] = static_cast<int>(config.mi.n);
  size_mapping.matrixInstruction[2] = static_cast<int>(config.mi.k);
  size_mapping.matrixInstruction[3] = 1;  // Default

  // Use depth_u if set, otherwise use mt.k
  size_mapping.depthU = (config.tensile().depth_u > 0) ? config.tensile().depth_u : config.mt.k;

  size_mapping.globalSplitU       = config.tensile().global_split_u;
  size_mapping.globalAccumulation = config.tensile().global_accumulation;
  size_mapping.LocalSplitU        = config.tensile().local_split_u;

  size_mapping.grvwA = config.grvw_a;
  size_mapping.grvwB = config.grvw_b;
  size_mapping.gwvwD = config.gwvw_d;
  size_mapping.gwvwC = config.gwvw_d;  // Typically same as D

  size_mapping.DirectToVgprA = config.tensile().direct_to_vgpr_a;
  size_mapping.DirectToVgprB = config.tensile().direct_to_vgpr_b;
  size_mapping.DirectToLdsA  = config.tensile().direct_to_lds_a;
  size_mapping.DirectToLdsB  = config.tensile().direct_to_lds_b;

  size_mapping.NumLoadsCoalescedA = config.tensile().num_loads_coalesced_a;
  size_mapping.NumLoadsCoalescedB = config.tensile().num_loads_coalesced_b;
  size_mapping.VectorWidthA       = config.vector_width_a;
  size_mapping.VectorWidthB       = config.vector_width_b;

  size_mapping.waveNum      = config.tensile().wave_num;
  size_mapping.waveGroup[0] = config.tensile().wave_group_m;
  size_mapping.waveGroup[1] = config.tensile().wave_group_n;

  size_mapping.workGroupMapping         = config.workgroup_mapping;
  size_mapping.workGroupMappingXCC      = config.tensile().workgroup_mapping_xcc;
  size_mapping.workGroupMappingXCCGroup = config.tensile().workgroup_mapping_xcc_group;
  size_mapping.globalSplitUCoalesced    = config.tensile().global_split_u_coalesced;
  size_mapping.globalSplitUWorkGroupMappingRoundRobin =
      config.tensile().global_split_u_wgm_round_robin;

  size_mapping.CUOccupancy            = config.occupancy;
  size_mapping.PrefetchGlobalRead     = config.tensile().prefetch_global_read;
  size_mapping.MathClocksUnrolledLoop = config.tensile().math_clocks_unrolled_loop;

  // Set problem, solution, and hardware in Formocast
  formocast.setProblem(prob_info);
  formocast.setSolution(size_mapping);
  formocast.setHardware(hardware.arch);

  // Get predicted performance
  Formocast::PredictedPerformance perf = formocast.predictedPerformance();

  // Return latency in microseconds
  return perf.microSeconds;
}

}  // namespace origami
