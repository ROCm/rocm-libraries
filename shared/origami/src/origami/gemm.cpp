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
#include <set>
#include <stdexcept>
#include <tuple>

#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "origami/gemm.hpp"
#include "origami/streamk.hpp"
#include "origami/simulator/tensilelite/formocast_simulator.hpp"

namespace origami {

// Forward declaration for internal Formocast latency computation
static double compute_formocast_latency(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config);


/* ---------------------------------------------------------------------------------------- */
/* context_t constructor                                                                    */
/* ---------------------------------------------------------------------------------------- */
context_t::context_t(const problem_t& problem,
                     const hardware_t& hardware,
                     const config_t& config) {
  const size_t M     = problem.size.m;
  const size_t N     = problem.size.n;
  const size_t batch = problem.batch;

  const size_t NUM_XCD = hardware.NUM_XCD;
  const size_t N_CU    = hardware.N_CU;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  // Grid dimensions
  grid_m    = math::safe_ceil_div(M, MT_M);
  grid_n    = math::safe_ceil_div(N, MT_N);
  num_tiles = grid_m * grid_n;

  // Launch parameters
  auto [reduction, wgs, cus, timesteps, split] =
      compute_launch_parameters(problem, hardware, config, config.grid_selection, N_CU);
  reduction_strategy = reduction;
  num_wgs            = wgs;
  num_timesteps      = timesteps;
  splitting_factor   = split;

  // Hardware-derived values
  active_cus     = cus;
  mem_bw_limited = compute_mem_bw_from_occupancy(hardware, active_cus);
  write_mem_bw_limited = compute_mem_bw_from_occupancy(hardware, num_tiles);

  // Tile-derived values
  tile_elements     = MT_M * MT_N;
  output_tile_bytes = tile_elements * data_type_to_bytes(problem.d_dtype);

  // Workgroup mapping
  wgm = predict_workgroup_mapping(problem, hardware, config, grid_m, grid_n, splitting_factor);

  // Cache tile dimensions
  const size_t wgm_val = static_cast<size_t>(std::abs(wgm.wgm));

  auto [mm, mn] = compute_mall_tiles(grid_m, grid_n, active_cus, wgm_val);
  mall_tile_m = mm;
  mall_tile_n = mn;

  auto [lm, ln] = compute_l2_tiles(problem, hardware, config,
                                    grid_m, grid_n, active_cus, splitting_factor, wgm_val);
  l2_tile_m = lm;
  l2_tile_n = ln;

  bool debug = runtime_options::get().debug_enabled;
  if(debug)
  {
    const size_t K     = problem.size.k;
    const size_t MT_K  = config.mt.k;
    const size_t MI_M  = config.mi.m;
    const size_t MI_N  = config.mi.n;
    const size_t MI_K  = config.mi.k;
    const auto a_bits  = datatype_to_bits(problem.a_dtype);
    const auto b_bits  = datatype_to_bits(problem.b_dtype);

    OLOG_DEBUG("======== Origami Debug Info ========");
    OLOG_DEBUG("Problem size: " << int(M) << "x" << int(N) << "x" << int(K));
    OLOG_DEBUG("batch: " << int(batch));
    OLOG_DEBUG("Macrotile: " << int(MT_M) << "x" << int(MT_N) << "x" << int(MT_K));
    OLOG_DEBUG("MatrixInstruction: " << int(MI_M) << "x" << int(MI_N) << "x" << int(MI_K));
    OLOG_DEBUG("Element size A (bits): " << int(a_bits));
    OLOG_DEBUG("Element size B (bits): " << int(b_bits));
    OLOG_DEBUG("Cache hints A: " << int(config.cache_hints_a));
    OLOG_DEBUG("Cache hints B: " << int(config.cache_hints_b));

    OLOG_DEBUG("Grid: " << int(grid_m) << "x" << int(grid_n));
    OLOG_DEBUG("Num tiles: " << int(num_tiles));
    OLOG_DEBUG("Num WGs: " << int(num_wgs));
    OLOG_DEBUG("Num timesteps: " << int(num_timesteps));
    OLOG_DEBUG("Splitting factor: " << int(splitting_factor));
    OLOG_DEBUG("Reduction strategy: " << int(reduction_strategy));
    
    OLOG_DEBUG("Active CUs: " << int(active_cus));
    OLOG_DEBUG("Read Mem BW limited: " << mem_bw_limited);
    OLOG_DEBUG("Write Mem BW limited: " << write_mem_bw_limited);

    OLOG_DEBUG("CHUNKxXCCxWGM: " << int(wgm.wgmxccchunk) << "x" << int(wgm.wgmxcc) << "x" << int(wgm.wgm));
    OLOG_DEBUG("Mall tile: " << int(mm) << "x" << int(mn));
    OLOG_DEBUG("L2 tile: " << int(lm) << "x" << int(ln));
  }
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
  auto round_up_mul = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
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
  const size_t N_CU    = hardware.N_CU;
  const size_t NUM_XCD = hardware.NUM_XCD;
  const size_t cus_per_xcd = N_CU / NUM_XCD;
  const size_t numMTs  = grid_m * grid_n;
  const size_t batch   = problem.batch;
  const size_t MT_M    = config.mt.m;
  const size_t MT_N    = config.mt.n;
  const auto a_bytes   = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes   = data_type_to_bytes(problem.b_dtype);

  // Batch case
  if (batch > 1) {
    size_t numTotalTiles = numMTs * batch;
    if (numMTs == 1 || numTotalTiles <= NUM_XCD || numMTs % NUM_XCD == 0)
      return {0, 0, 1};
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
  if (out_wgmxcc == 0 || grid_m == 1 || grid_n == 1)
    return {0, out_wgmxcc, 1};

  size_t numWGsPerXCD = std::min(math::safe_ceil_div(numMTs, NUM_XCD), cus_per_xcd);
  if (numWGsPerXCD >= cus_per_xcd / 2 && grid_n <= 8)
    return {0, out_wgmxcc, static_cast<int32_t>(grid_n)};

  // Build candidate list
  size_t wgm_cap = std::min(grid_n, numWGsPerXCD / 2);
  std::vector<size_t> candidates;
  std::set<size_t> cset;
  for (size_t v : {1, 4, 6})
    if (v <= wgm_cap) cset.insert(v);
  for (size_t i = 1; i * i <= wgm_cap; ++i) {
    if (wgm_cap % i == 0) {
      cset.insert(i);
      cset.insert(wgm_cap / i);
    }
  }
  candidates.assign(cset.begin(), cset.end());
  if (candidates.empty())
    return {0, out_wgmxcc, 1};

  // Evaluate L2 cost for last XCD in the first timestep
  const size_t total = numMTs;
  const size_t last_xcd = NUM_XCD - 1;
  const size_t group_size = total >= NUM_XCD ? total / NUM_XCD : total;
  const size_t tiles_this_xcd = std::min(cus_per_xcd, group_size);
  const size_t start = last_xcd * group_size;
  const size_t count = (start < total) ? std::min(tiles_this_xcd, total - start) : 0;

  const double a_cost = static_cast<double>(MT_M) * a_bytes;
  const double b_cost = static_cast<double>(MT_N) * b_bytes;

  auto l2_cost = [&](size_t W) -> double {
    size_t W_eff = std::min(W, grid_n);
    if (W_eff == 0 || count == 0) return std::numeric_limits<double>::max();

    size_t slab_tiles = grid_m * W_eff;
    size_t s0 = start / slab_tiles;
    size_t s1 = (start + count - 1) / slab_tiles;
    size_t m0 = (start % slab_tiles) / W_eff;
    size_t m1 = ((start + count - 1) % slab_tiles) / W_eff;

    size_t ur, uc;
    if (s0 == s1) {
      ur = m1 - m0 + 1;
      uc = (ur > 1) ? W_eff : std::min(count, W_eff);
    } else {
      ur = (s1 - s0 > 1) ? grid_m : std::min(grid_m, (grid_m - m0) + (m1 + 1));
      uc = std::min((s1 - s0 + 1) * W_eff, grid_n);
    }
    ur = std::min(ur, grid_m);
    uc = std::min(uc, grid_n);
    return ur * a_cost + uc * b_cost;
  };

  size_t best_wgm = 1;
  double best_cost = std::numeric_limits<double>::max();
  for (size_t w : candidates) {
    double cost = l2_cost(w);
    if (cost < best_cost) {
      best_cost = cost;
      best_wgm = w;
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
  auto config_with_reduction = config;
  config_with_reduction.reduction_strategy = reduction_strategy;
  const size_t num_wgs = streamk::select_grid_size(
    problem, hardware, config_with_reduction, grid_selection, max_cus);

  const size_t num_mts = streamk::compute_number_of_output_tiles(
    config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
  // That means, we artifically observe one full timesteps, but that is not what actually happens
  // under the hood. From a theoretical point of view, these distributions change all of the
  // computations in Origami. With current implementation, it is hard to capture that
  // behaviour analytically. So for now, if the num_wgs is less than the num_mts, we calculate
  // num_timesteps based on the num_mts. Otherwise, we use num_wgs to compute num_timesteps.
  const size_t num_active_cus = num_wgs < hardware.N_CU ? num_wgs : hardware.N_CU;
  const size_t num_timesteps = num_wgs > num_mts ? math::safe_ceil_div(num_wgs, hardware.N_CU)
                                         : math::safe_ceil_div(num_mts, hardware.N_CU);
  const size_t splitting_factor = math::safe_ceil_div(num_wgs, num_mts);

  return std::make_tuple(reduction_strategy, num_wgs, num_active_cus, num_timesteps,
                       splitting_factor);
}

// Check if MT fits in LDS
bool check_lds_capacity(const hardware_t& hardware,
                        const dim3_t& mt,
                        const data_type_t& a_dtype,
                        const data_type_t& b_dtype) {
  const auto a_loads_in_bytes = mt.mk() * data_type_to_bytes(a_dtype);
  const auto b_loads_in_bytes = mt.nk() * data_type_to_bytes(b_dtype);
  const auto LDS_usage = a_loads_in_bytes + b_loads_in_bytes;

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

// Compute MALL tile dimensions: how many concurrent workgroup tiles fit in MALL.
std::pair<size_t, size_t> compute_mall_tiles(size_t grid_m,
                                             size_t grid_n,
                                             size_t active_cus,
                                             size_t wgm_value) {
  size_t m = math::safe_ceil_div(active_cus, wgm_value);
  size_t n = std::min(wgm_value, grid_n);
  if (m > grid_m) {
    n += (m / grid_m) * wgm_value;
    m = grid_m;
  }
  m = std::max(std::min(grid_m, m), static_cast<size_t>(1));
  n = std::max(std::min(grid_n, n), static_cast<size_t>(1));
  return {m, n};
}

// Compute L2 tile dimensions: how many tiles share one XCD's L2, shrunk to fit capacity.
std::pair<size_t, size_t> compute_l2_tiles(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           size_t grid_m,
                                           size_t grid_n,
                                           size_t active_cus,
                                           size_t splitting_factor,
                                           size_t wgm_value) {
  const size_t concurrent = std::min(grid_m * grid_n, active_cus);
  const size_t effective_cus =
      math::safe_ceil_div(concurrent, splitting_factor * problem.batch);
  const size_t cu_per_xcd =
      std::max(math::safe_ceil_div(effective_cus, hardware.NUM_XCD), static_cast<size_t>(1));

  size_t n = std::min(wgm_value, grid_n);
  size_t m = math::safe_ceil_div(cu_per_xcd, n);
  if (m > grid_m) {
    n += (m / grid_m) * wgm_value;
    m = grid_m;
  }
  m = std::max(std::min(grid_m, m), static_cast<size_t>(1));
  n = std::max(std::min(grid_n, n), static_cast<size_t>(1));

  // Shrink to fit L2 capacity
  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);
  while (m * config.mt.mk() * a_bytes + n * config.mt.nk() * b_bytes > hardware.L2_capacity) {
    if (m > 1 && m >= n) m--;
    else if (n > 1) n--;
    else break;
  }
  return {m, n};
}

// Map a linear workgroup ID to 4D tile coordinates (k, m, n, b).
// Dispatch order: k (innermost) -> mn (WGM slab ordering) -> b (outermost).
dim4_t wgm_to_grid(const dim4_t& grid, const workgroup_mapping_t& wgm_mapping, size_t id) {
  const size_t wgmxcc = wgm_mapping.wgmxcc;
  const size_t wgm = static_cast<size_t>(std::abs(wgm_mapping.wgm));

  // Apply wgmxcc reordering so consecutive tiles in dispatch order land on the same XCD.
  if (wgmxcc > 1) {
    const size_t total = grid.total();
    const size_t group_size = total / wgmxcc;
    id = (id / wgmxcc) + (id % wgmxcc) * group_size;
    if (id >= total) id = total - 1;
  }

  const size_t mnk = grid.mnk();

  // Decompose: b (outermost) -> mn (WGM slab) -> k (innermost)
  dim4_t t;
  t.b = id / mnk;
  const size_t rem = id % mnk;
  const size_t mn_id = rem / grid.k;
  t.k = rem % grid.k;

  // MN slab ordering
  const size_t W = std::min(wgm, grid.n);
  if (W == 0) { t.m = 0; t.n = 0; return t; }

  const size_t slab_tiles  = grid.m * W;
  const size_t full_slabs  = grid.n / W;
  const size_t full_region = full_slabs * slab_tiles;

  if (mn_id < full_region) {
    const size_t s  = mn_id / slab_tiles;
    const size_t in = mn_id % slab_tiles;
    t.m = in / W;
    t.n = s * W + in % W;
  } else {
    const size_t last_w = grid.n - full_slabs * W;
    if (last_w == 0) { t.m = grid.m - 1; t.n = grid.n - 1; }
    else {
      const size_t in = mn_id - full_region;
      t.m = in / last_w;
      t.n = full_slabs * W + in % last_w;
    }
  }
  return t;
}

// Internal helper: count unique tiles for a contiguous range in raw dispatch order.
// Dispatch order: k (innermost) → mn (WGM slab) → b (outermost).
static dim4_t count_unique_range(const dim4_t& grid, size_t wgm, size_t start, size_t count) {
  if (count == 0 || grid.m == 0 || grid.n == 0)
    return {0, 0, 0, 0};

  // Use a no-wgmxcc mapping for raw dispatch order
  const workgroup_mapping_t raw_wgm = {0, 0, static_cast<int32_t>(wgm)};
  auto first = wgm_to_grid(grid, raw_wgm, start);
  auto last  = wgm_to_grid(grid, raw_wgm, start + count - 1);

  dim4_t u;

  // B: outermost dimension
  u.b = last.b - first.b + 1;
  u.b = std::min(u.b, grid.b);

  // K: innermost dimension — if we span more than one mn tile, we cover all k
  const size_t mnk = grid.mnk();
  const size_t mn_first = (start % mnk) / grid.k;
  const size_t mn_last  = ((start + count - 1) % mnk) / grid.k;
  if (mn_first == mn_last && u.b == 1) {
    u.k = last.k - first.k + 1;
  } else {
    u.k = grid.k;
  }
  u.k = std::min(u.k, grid.k);

  // M and N: WGM slab geometry
  const size_t W = std::min(wgm, grid.n);
  if (W == 0) { u.m = 0; u.n = 0; return u; }

  const size_t num_mn_tiles = (count + grid.k - 1) / grid.k;

  if (u.b > 1 || num_mn_tiles >= grid.mn()) {
    u.m = grid.m;
    u.n = grid.n;
  } else {
    const size_t slab0 = first.n / W;
    const size_t slab1 = last.n / W;

    if (slab0 == slab1) {
      u.m = last.m - first.m + 1;
      u.n = (u.m > 1) ? W : (last.n - first.n + 1);
    } else {
      if (slab1 - slab0 > 1) u.m = grid.m;
      else u.m = std::min(grid.m, (grid.m - first.m) + (last.m + 1));
      u.n = last.n - first.n + 1;
    }
  }

  u.m = std::min(u.m, grid.m);
  u.n = std::min(u.n, grid.n);
  return u;
}

// Count unique tiles for a specific XCD during a specific timestep.
// With wgmxcc: XCD x sees cus_per_xcd consecutive tiles in raw dispatch order.
// Without wgmxcc: XCD x gets every num_xcd-th tile (round-robin), so the tiles
// are strided — we compute unique k/m/n/b analytically for the strided set.
dim4_t count_unique_tiles(const dim4_t& grid, const workgroup_mapping_t& wgm_mapping,
                          size_t N_CU, size_t num_xcd,
                          size_t xcd_id, size_t timestep_id) {
  if (N_CU == 0 || num_xcd == 0 || grid.m == 0 || grid.n == 0 || grid.k == 0)
    return {0, 0, 0, 0};

  const size_t wgmxcc = wgm_mapping.wgmxcc;
  const size_t wgm = static_cast<size_t>(std::abs(wgm_mapping.wgm));
  const size_t total = grid.total();
  const size_t cus_per_xcd = N_CU / num_xcd;
  if (cus_per_xcd == 0) return {0, 0, 0, 0};

  if (wgmxcc > 1) {
    // With wgmxcc: XCD x gets a contiguous block of group_size tiles in raw dispatch order.
    // When total < N_CU, group_size < cus_per_xcd — cap tiles per timestep accordingly.
    const size_t group_size = total / wgmxcc;
    const size_t tiles_per_ts = std::min(cus_per_xcd, group_size);
    const size_t start = xcd_id * group_size + timestep_id * tiles_per_ts;
    const size_t remaining = (start < xcd_id * group_size + group_size)
                           ? xcd_id * group_size + group_size - start : 0;
    const size_t count = std::min(tiles_per_ts, remaining);
    return count_unique_range(grid, wgm, start, count);
  }

  // Without wgmxcc: round-robin assignment.
  // XCD x gets tile IDs: first, first+stride, first+2*stride, ...
  const size_t stride = num_xcd;
  const size_t first_tile = timestep_id * N_CU + xcd_id;
  if (first_tile >= total) return {0, 0, 0, 0};
  const size_t count = std::min(cus_per_xcd, (total - first_tile + stride - 1) / stride);
  if (count == 0) return {0, 0, 0, 0};

  const size_t mnk = grid.mnk();
  const size_t mn  = grid.mn();

  dim4_t u;

  // K: innermost — stride S samples grid.k / gcd(S, grid.k) distinct k values
  const size_t gcd_k = std::gcd(stride, grid.k);
  u.k = std::min(count, grid.k / gcd_k);

  // MN: middle — after k is consumed, effective mn-stride = S / gcd(S, grid.k)
  const size_t mn_stride = stride / gcd_k;
  const size_t gcd_mn = std::gcd(mn_stride, mn);
  size_t unique_mn = std::min(count / u.k, mn / gcd_mn);
  if (unique_mn == 0 && count > 0) unique_mn = 1;

  // B: outermost — after k and mn are consumed, effective b-stride = S / gcd(S, mnk)
  const size_t gcd_mnk = std::gcd(stride, mnk);
  const size_t b_stride = stride / gcd_mnk;
  const size_t gcd_b = std::gcd(b_stride, grid.b);
  u.b = std::min(count / (u.k * unique_mn), grid.b / gcd_b);
  if (u.b == 0 && count > 0) u.b = 1;

  // Split unique_mn into (m, n) by mapping strided mn_ids through WGM slab.
  // The mn_ids are strided with step mn_stride, NOT consecutive.
  const size_t W = std::min(wgm, grid.n);
  if (W == 0) { u.m = 0; u.n = 0; return u; }

  const size_t first_mn = (first_tile % mnk) / grid.k;
  const size_t slab_tiles = grid.m * W;

  size_t m_min = grid.m, m_max = 0;
  size_t n_min = grid.n, n_max = 0;
  for (size_t i = 0; i < unique_mn; ++i) {
    const size_t mn_id = (first_mn + i * mn_stride) % mn;
    const size_t full_slabs = grid.n / W;
    const size_t full_region = full_slabs * slab_tiles;
    size_t mi, ni;
    if (mn_id < full_region) {
      const size_t s  = mn_id / slab_tiles;
      const size_t in = mn_id % slab_tiles;
      mi = in / W;
      ni = s * W + in % W;
    } else {
      const size_t last_w = grid.n - full_slabs * W;
      if (last_w == 0) { mi = grid.m - 1; ni = grid.n - 1; }
      else {
        const size_t in = mn_id - full_region;
        mi = in / last_w;
        ni = full_slabs * W + in % last_w;
      }
    }
    m_min = std::min(m_min, mi); m_max = std::max(m_max, mi);
    n_min = std::min(n_min, ni); n_max = std::max(n_max, ni);
  }
  u.m = m_max - m_min + 1;
  u.n = n_max - n_min + 1;

  return u;
}

// Count unique tiles for an entire timestep (all XCDs combined).
dim4_t count_unique_tiles_timestep(const dim4_t& grid, const workgroup_mapping_t& wgm_mapping,
                                   size_t N_CU, size_t timestep_id) {
  const size_t wgm = static_cast<size_t>(std::abs(wgm_mapping.wgm));
  const size_t total = grid.total();
  const size_t start = timestep_id * N_CU;
  const size_t count = std::min(N_CU, total > start ? total - start : static_cast<size_t>(0));

  return count_unique_range(grid, wgm, start, count);
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

  // size_t mt_arith = arithmetic_intensity(MT_M, MT_N, MT_K, 2);
  // printf("MT_M:%d MT_N:%d MT_K:%d arith:%d\n", MT_M, MT_N, MT_K, mt_arith);
  // size_t arith = ((M * N * K * 2) / (M * K + N * K + M * N));
  size_t L_MT = L_MI * N_MI;

  return L_MT;
}

/* ---------------------------------------------------------------------------------------- */
/* Memory-related functions                                                                 */
/* ---------------------------------------------------------------------------------------- */
// Estimate L2 hit rate
double estimate_l2_hit(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       const context_t& context) {
  const size_t wgm_val = static_cast<size_t>(std::abs(context.wgm.wgm));
  auto [l2_m, l2_n] = compute_l2_tiles(problem, hardware, config,
                                       context.grid_m, context.grid_n,
                                       context.active_cus, context.splitting_factor, wgm_val);

  const long long uA    = static_cast<long long>(l2_m) * config.mt.mk();
  const long long uB    = static_cast<long long>(l2_n) * config.mt.nk();
  const long long total = std::max(uA * static_cast<long long>(l2_n)
                                 + uB * static_cast<long long>(l2_m), 1LL);
  const long long cached = total - (uA + uB);

  return std::max(0.0, std::min(static_cast<double>(cached) / total, 1.0));
}

// Estimate MALL hit-rate
double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         const context_t& context) {
  const size_t wgm_val = static_cast<size_t>(std::abs(context.wgm.wgm));
  auto [mall_m, mall_n] = compute_mall_tiles(context.grid_m, context.grid_n,
                                             context.active_cus, wgm_val);

  const long long uA    = static_cast<long long>(mall_m) * config.mt.mk();
  const long long uB    = static_cast<long long>(mall_n) * config.mt.nk();
  const long long total = std::max(uA * static_cast<long long>(mall_n)
                                 + uB * static_cast<long long>(mall_m), 1LL);
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

// Two-timestep cache hit rate estimation.
// T0 = first timestep (spatial only).
// T1 = second timestep (spatial + temporal reuse from T0).
// Extrapolate: overall = (T0_rate + (num_timesteps - 1) * T1_rate) / num_timesteps.
std::pair<double, double> estimate_cache_hit_rates(const problem_t& problem,
                                                   const hardware_t& hardware,
                                                   const config_t& config,
                                                   const context_t& context) {
  bool debug = runtime_options::get().debug_enabled;

  // Extract parameters
  const dim4_t grid = {context.splitting_factor, context.grid_m, context.grid_n, problem.batch};
  const size_t total   = grid.total();
  const size_t N_CU    = hardware.N_CU;
  const auto&  wgm     = context.wgm;
  const size_t num_ts  = context.num_timesteps;
  const size_t num_xcd = std::max(hardware.NUM_XCD, static_cast<size_t>(1));
  const size_t cus_per_xcd = N_CU / num_xcd;

  if (N_CU == 0 || total == 0 || grid.m == 0 || grid.n == 0)
    return {0.0, 0.0};

  // Helper function to clamp values between 0 and 1
  auto clamp01 = [](double v) { return std::max(0.0, std::min(v, 1.0)); };

  // Actual tiles per XCD per timestep: each XCD owns total/num_xcd tiles,
  // and processes min(cus_per_xcd, total/num_xcd) tiles per timestep.
  const size_t tiles_per_xcd = total / num_xcd;
  const size_t tiles_per_xcd_ts = std::min(cus_per_xcd, tiles_per_xcd);

  const size_t split_K = math::safe_ceil_div(problem.size.k, grid.k);
  const double a_tile  = static_cast<double>(config.mt.m) * split_K * data_type_to_bytes(problem.a_dtype);
  const double b_tile  = static_cast<double>(config.mt.n) * split_K * data_type_to_bytes(problem.b_dtype);  
  const double per_wg  = a_tile + b_tile;
  const double l2_cap  = 0.99 * static_cast<double>(hardware.L2_capacity);

  // Shared: timestep tile counts
  dim4_t t0, t1;
  const workgroup_mapping_t raw_wgm = {0, 0, wgm.wgm}; // no-wgmxcc for raw dispatch queries
  t0 = count_unique_tiles_timestep(grid, wgm, N_CU, 0);
  const size_t t0_count = std::min(N_CU, total);
  const double t0_uncached = (t0.m * a_tile + t0.n * b_tile) * t0.k * t0.b;
  const double t0_total    = t0_count * per_wg;
  const double t0_cached   = t0_total - t0_uncached; // spatial reuse only

  // MALL hit rate
  double mall_rate = 0.0;
  {
    double t1_mall_hits = t0_cached;
    double t1_total     = t0_total;

    if (num_ts > 1 && total > N_CU) {
      t1 = count_unique_tiles_timestep(grid, wgm, N_CU, 1);
      const size_t t1_count = std::min(N_CU, total - N_CU);
      const double t1_uncached = (t1.m * a_tile + t1.n * b_tile) * t1.k * t1.b;
      t1_total = t1_count * per_wg;

      // Temporal overlap: tiles reused from T0 -> T1
      auto t0_f = wgm_to_grid(grid, raw_wgm, 0);
      auto t0_l = wgm_to_grid(grid, raw_wgm, t0_count - 1);
      auto t1_f = wgm_to_grid(grid, raw_wgm, N_CU);
      auto t1_l = wgm_to_grid(grid, raw_wgm, N_CU + t1_count - 1);

      // N-range intersection (B-matrix reuse)
      size_t n_overlap = 0;
      if (t1_f.n <= t0_l.n && t1_l.n >= t0_f.n)
        n_overlap = std::min(t1_l.n, t0_l.n) - std::max(t1_f.n, t0_f.n) + 1;

      // A-matrix reuse: both timesteps cover ALL M rows -> fully cached
      size_t m_overlap = (t0.m == grid.m && t1.m == grid.m) ? grid.m : 0;

      // K overlap
      size_t k_overlap = std::min(t0.k, t1.k);

      double mall_temporal = m_overlap * k_overlap * a_tile + n_overlap * k_overlap * b_tile;
      t1_mall_hits = t1_total - std::max(t1_uncached - mall_temporal, 0.0);
    }

    // Extrapolate: T0 (cold) + (num_ts - 1) * T1 (warm)
    double mall_total_reads = t0_total + (num_ts > 1 ? static_cast<double>(num_ts - 1) * t1_total : 0.0);
    double mall_cached      = t0_cached + (num_ts > 1 ? static_cast<double>(num_ts - 1) * t1_mall_hits : 0.0);
    mall_rate = mall_total_reads > 0 ? clamp01(mall_cached / mall_total_reads) : 0.0;

    if(debug)
    {
      OLOG_DEBUG("MALL Tiles (T0): " << t0.k << " " << t0.m << " " << t0.n << " " << t0.b);
      OLOG_DEBUG("MALL Tiles (T1): " << t1.k << " " << t1.m << " " << t1.n << " " << t1.b);
      OLOG_DEBUG("MALL Hit-rate (T0): " << t0_cached / t0_total);
      OLOG_DEBUG("MALL Hit-rate (T1): " << t1_mall_hits / t1_total);
    }
  }

  // L2 hit rate
  double l2_rate = 0.0;
  {
    dim4_t xcd_t0,xcd_t1; 
    const size_t last_xcd = num_xcd - 1;
    const size_t per_xcd_count = std::max(std::min(t0_count / num_xcd, tiles_per_xcd_ts), static_cast<size_t>(1));

    // T0: spatial reuse only (cold start)
    xcd_t0 = count_unique_tiles(grid, wgm, N_CU, num_xcd, last_xcd, 0);
    double xcd_t0_ws  = (xcd_t0.m * a_tile + xcd_t0.n * b_tile) * xcd_t0.k * xcd_t0.b;
    double xcd_t0_req = per_xcd_count * per_wg;

    // L2 capacity check: use per-iteration working set (one MT_K chunk), not full K-loop.
    // Each main loop iteration loads MT_K worth of A and B — that's the live L2 footprint.
    const double a_iter = static_cast<double>(config.mt.m) * config.mt.k * data_type_to_bytes(problem.a_dtype);
    const double b_iter = static_cast<double>(config.mt.n) * config.mt.k * data_type_to_bytes(problem.b_dtype);
    double xcd_t0_iter_ws = (xcd_t0.m * a_iter + xcd_t0.n * b_iter) * xcd_t0.b;
    double l2_residency = (xcd_t0_iter_ws > 0) ? std::min(l2_cap / xcd_t0_iter_ws, 1.0) : 1.0;
    double effective_ws = xcd_t0_ws * l2_residency + xcd_t0_req * (1.0 - l2_residency);

    double t0_l2_hits = xcd_t0_req - effective_ws;

    double t1_l2_hits = t0_l2_hits;
    double xcd_t1_req = xcd_t0_req;

    if (num_ts > 1 && total > N_CU) {
      xcd_t1 = count_unique_tiles(grid, wgm, N_CU, num_xcd, last_xcd, 1);
      const size_t t1_count = std::min(N_CU, total - N_CU);
      xcd_t1_req = std::max(std::min(t1_count / num_xcd, tiles_per_xcd_ts), static_cast<size_t>(1)) * per_wg;

      // Temporal overlap: N-range intersection for this XCD across T0 -> T1
      size_t xcd_n_overlap = 0;
      if (tiles_per_xcd_ts > 0) {
        size_t xcd_start_t0, xcd_start_t1;
        if (wgm.wgmxcc > 1) {
          const size_t group_size = total / wgm.wgmxcc;
          xcd_start_t0 = last_xcd * group_size + 0 * tiles_per_xcd_ts;
          xcd_start_t1 = last_xcd * group_size + 1 * tiles_per_xcd_ts;
        } else {
          xcd_start_t0 = last_xcd;
          xcd_start_t1 = N_CU + last_xcd;
        }
        auto xf0 = wgm_to_grid(grid, raw_wgm, xcd_start_t0);
        auto xl0 = wgm_to_grid(grid, raw_wgm, xcd_start_t0 + tiles_per_xcd_ts - 1);
        auto xf1 = wgm_to_grid(grid, raw_wgm, xcd_start_t1);
        auto xl1 = wgm_to_grid(grid, raw_wgm, xcd_start_t1 + tiles_per_xcd_ts - 1);

        if (xf1.n <= xl0.n && xl1.n >= xf0.n)
          xcd_n_overlap = std::min(xl1.n, xl0.n) - std::max(xf1.n, xf0.n) + 1;
      }

      size_t xcd_m_overlap = (xcd_t0.m == grid.m && xcd_t1.m == grid.m) ? grid.m : 0;

      double l2_temporal = 0.0;
      if (xcd_t0_ws <= l2_cap)
        l2_temporal = xcd_m_overlap * a_tile + xcd_n_overlap * b_tile;

      double xcd_t1_ws = (xcd_t1.m * a_tile + xcd_t1.n * b_tile) * xcd_t1.k * xcd_t1.b;

      // L2 capacity scaling for T1 (same per-iteration footprint)
      double xcd_t1_iter_ws = (xcd_t1.m * a_iter + xcd_t1.n * b_iter) * xcd_t1.b;
      double l2_residency_t1 = (xcd_t1_iter_ws > 0) ? std::min(l2_cap / xcd_t1_iter_ws, 1.0) : 1.0;
      double effective_temporal = l2_temporal * l2_residency_t1;
      double effective_ws_t1 = xcd_t1_ws * l2_residency_t1 + xcd_t1_req * (1.0 - l2_residency_t1);

      t1_l2_hits = xcd_t1_req - std::max(effective_ws_t1 - effective_temporal, 0.0);
    }

    // Extrapolate: T0 (cold) + (num_ts - 1) * T1 (warm)
    double l2_total_reads = xcd_t0_req + (num_ts > 1 ? static_cast<double>(num_ts - 1) * xcd_t1_req : 0.0);
    double l2_cached      = t0_l2_hits + (num_ts > 1 ? static_cast<double>(num_ts - 1) * t1_l2_hits : 0.0);
    l2_rate = l2_total_reads > 0 ? clamp01(l2_cached / l2_total_reads) : 0.0;

    if(debug)
    {
      OLOG_DEBUG("Last XCD L2 Tiles (T0): " << xcd_t0.k << " " << xcd_t0.m << " " << xcd_t0.n << " " << xcd_t0.b);
      OLOG_DEBUG("Last XCD L2 Tiles (T1): " << xcd_t1.k << " " << xcd_t1.m << " " << xcd_t1.n << " " << xcd_t1.b);
      OLOG_DEBUG("L2 Hit-rate (T0): " << t0_l2_hits / xcd_t0_req);
      OLOG_DEBUG("L2 Hit-rate (T1): " << t1_l2_hits / xcd_t1_req);
    }
  }

  return {mall_rate, l2_rate};
}

// Determine the memory latency
double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const context_t& context) {
  bool debug = runtime_options::get().debug_enabled;

  // Extract parameters from structured types
  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);
  const auto a_bits  = datatype_to_bits(problem.a_dtype);
  const auto b_bits  = datatype_to_bits(problem.b_dtype);
  const bool a_trans = (problem.a_transpose == transpose_t::T);
  const bool b_trans = (problem.b_transpose == transpose_t::T);
  const size_t num_active_cus = context.active_cus;

  heuristic_params_t heuristic = get_heuristic_params(problem, hardware, config);

  // 1) Estimate MALL and L2 hit-rates using the two-timestep analytical model
  auto [H_mem_mall, H_mem_l2] = estimate_cache_hit_rates(problem, hardware, config, context);
  if (!hardware.has_MALL()) H_mem_mall = 0.0;
  // if (H_mem_l2 == 0) H_mem_l2 = heuristic.l2_min_hit_rate_default;

  // 2) Total loads per CU (A + B, with 128B alignment and MX scales)
  size_t Ld_A = a_trans ? config.mt.m * round_elements_to_128B(config.mt.k, a_bits)
                        : round_elements_to_128B(config.mt.m, a_bits) * config.mt.k;
  size_t Ld_B = b_trans ? round_elements_to_128B(config.mt.n, b_bits) * config.mt.k
                        : config.mt.n * round_elements_to_128B(config.mt.k, b_bits);
  auto Ld_CU_bytes = (Ld_A * a_bytes) + (Ld_B * b_bytes);

  // Block scaled datatypes (MX): add scale bytes
  if (a_bits < 8 && problem.a_mx_block_size != 0)
    Ld_CU_bytes += math::safe_ceil_div(config.mt.mk(), problem.a_mx_block_size);
  if (b_bits < 8 && problem.b_mx_block_size != 0)
    Ld_CU_bytes += math::safe_ceil_div(config.mt.nk(), problem.b_mx_block_size);

  // 3) Total loads by all CUs
  double total_Ld = Ld_CU_bytes * static_cast<double>(num_active_cus);

  // 4) L2 latency (bandwidth-limited by CU occupancy ratio)
  double l2_bw = hardware.mem1_perf_ratio *
                 static_cast<double>(num_active_cus) / static_cast<double>(hardware.N_CU);
  double L_mem_l2 = (l2_bw > 0) ? (total_Ld / l2_bw) : 0.0;

  // 5) MALL and DRAM bandwidth (occupancy-limited)
  double bw_limited = context.mem_bw_limited;

  // 6) Loads that reach each memory level
  double Ld_mall = hardware.has_MALL() ? (1.0 - H_mem_l2) * total_Ld : 0.0;
  double Ld_dram = (1.0 - H_mem_mall) * Ld_mall;

  // 7) MALL latency
  double mall_bw = hardware.mem2_perf_ratio * bw_limited;
  double L_mem_mall = (mall_bw > 0) ? (Ld_mall / mall_bw) : 0.0;

  // 8) DRAM latency
  double dram_bw = hardware.mem3_perf_ratio * bw_limited;
  double L_mem_dram = (dram_bw > 0) ? (Ld_dram / dram_bw) : 0.0;
  L_mem_dram += heuristic.main_memory_load_latency;
  
  // 9) Worst-case across all memory levels
  double L_mem = std::max({L_mem_l2   * heuristic.weight_mem_l2,
                           L_mem_mall * heuristic.weight_mem_mall,
                           L_mem_dram * heuristic.weight_mem_dram});

  if(debug)
  {
    OLOG_DEBUG("H_mem_mall: " << H_mem_mall);
    OLOG_DEBUG("H_mem_l2: " << H_mem_l2);
    OLOG_DEBUG("Ld_CU_bytes: " << Ld_CU_bytes);
    OLOG_DEBUG("total_Ld: " << total_Ld);
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
// Determine the epilogue latency of a single tile (per-tile model).
// The epilogue consists of:
//   1. Last MFMA iteration (NoLoadLoop): completes the final K-chunk compute
//   2. ACC -> VGPR transfer: moves accumulator results to VGPRs for store
//   3. Bounds checking: edge tiles need per-element or per-instruction checks
//   4. Global memory stores: write output tile through bandwidth-limited memory
//   5. K-split reduction: sync + read partials + accumulate (if splitting_factor > 1)
// The model evaluates each tile type (interior, N-edge, M-edge, corner) and takes the
// worst case (B), since the slowest tile determines when the next timestep can begin.
double compute_epilogue_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const context_t& context) {
  bool debug = runtime_options::get().debug_enabled;

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
  // Herein, we consider the simplest case for now: Alpha=1, Beta=0, and no bias/activation functions.
  // Skipping items 2, 3, 4, and 5 for now.

  // Extract parameters
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t d_bytes = data_type_to_bytes(problem.d_dtype);

  const reduction_t reduction_strategy = context.reduction_strategy;
  const size_t num_active_cus = context.active_cus;
  const size_t splitting_factor = context.splitting_factor;
  const int grid_m = context.grid_m;
  const int grid_n = context.grid_n;
  const size_t num_tiles = context.num_tiles;
  const bool is_parallel_reduction = (reduction_strategy == reduction_t::parallel);
  const double store_bw  = std::max(hardware.mem3_perf_ratio * context.mem_bw_limited, 1e-12);
  const double reduce_bw = std::max(hardware.mem3_perf_ratio * context.write_mem_bw_limited, 1e-12);

  // Early return if there is no output dtype
  if (d_bytes == 0) return 0.0;

  // Constants
  constexpr double cycles_per_acc_read = 8.0;
  constexpr double acc_read_parallelism = 0.9;
  constexpr double cycles_per_bounds_check = 6.0;
  constexpr double scalar_store_penalty = 2.0;
  constexpr size_t threads_per_wave = 64;
  constexpr size_t bytes_per_vectorized_store = 16; // buffer_store_dwordx4 = 16 bytes
  constexpr size_t cache_line_bytes = 128;
  constexpr double initial_sync_overhead = 5000.0;
  constexpr double cycles_per_flag_poll  = 5000.0;

  // Common setup
  const size_t total_mfmas = math::safe_ceil_div(MT_M, config.mi.m) *
                             math::safe_ceil_div(MT_N, config.mi.n);
  const size_t elements_per_vectorized_store = bytes_per_vectorized_store / d_bytes;
  const size_t elements_per_cache_line = math::safe_ceil_div(cache_line_bytes, d_bytes);
  const double alignment_penalty = (M % elements_per_cache_line != 0) ? 1.3 : 1.0;

  // Per-CU write bandwidth: total write BW shared among all writers
  // During epilogue store, ALL active WGs write simultaneously:
  // - Non-finishing WGs write partials to workspace
  // - Finishing WGs (or all WGs if no split) write final output
  const size_t num_writers = num_active_cus;
  const double per_cu_store_bw = store_bw / static_cast<double>(num_writers);

  // Edge tile detection
  const bool has_interior = (M >= MT_M && N >= MT_N);
  const bool has_m_edge = (M % MT_M != 0);
  const bool has_n_edge = (N % MT_N != 0);
  const size_t m_remainder = has_m_edge ? (M % MT_M) : MT_M;
  const size_t n_remainder = has_n_edge ? (N % MT_N) : MT_N;

  // Helper function to compute the epilogue cost for a given tile type
  auto compute_tile_epilogue = [&](size_t tile_m, size_t tile_n, bool is_scalar_path) -> double {
    // Scalar path is the m_edge tiles including corner tiles
    // 1) ACC -> VGPR
    size_t acc_reads = is_scalar_path ? 2 * total_mfmas : total_mfmas;
    double L_acc_transfer = acc_reads * cycles_per_acc_read * acc_read_parallelism;

    // 2) Bounds checking (edge tiles only)
    double L_edge_check = 0.0;
    size_t total_elements = tile_m * tile_n;
    if (is_scalar_path) {
      double store_instr = math::safe_ceil_div(total_elements, threads_per_wave);
      L_edge_check = store_instr * cycles_per_bounds_check;
    } else if (tile_m != MT_M || tile_n != MT_N) {
      size_t store_instr = math::safe_ceil_div(total_elements,
                                               threads_per_wave * elements_per_vectorized_store);
      L_edge_check = store_instr * cycles_per_bounds_check;
    }

    // 3) Store: per-tile bytes through per-CU bandwidth share
    double store_bytes = static_cast<double>(tile_m) * tile_n * d_bytes;
    double store_scale = is_scalar_path ? scalar_store_penalty : 1.0;
    double L_store = (store_bytes * store_scale * alignment_penalty) / per_cu_store_bw;

    // 4) Per-tile K-split reduction (in-kernel: spinlock/tree/atomic)
    // After all WGs write partials, only the finishing WGs (one per output tile) are active.
    // They read partials from workspace, accumulate, then write final output.
    // Contention during reduction is much lower: only grid_m x grid_n finishing WGs are reading.
    double L_reduce = 0.0;
    if (splitting_factor > 1 && !is_parallel_reduction) {
      size_t n_partials = splitting_factor - 1;
      // Only finishing WGs (one per output tile) are active during reduction.
      double per_cu_reduce_bw = reduce_bw / static_cast<double>(num_tiles);

      double partial_bytes = static_cast<double>(n_partials) * tile_m * tile_n * d_bytes;
      double L_sync = initial_sync_overhead + (n_partials - 1) * cycles_per_flag_poll;
      double L_partial_read = partial_bytes / per_cu_reduce_bw;
      double L_accumulate = static_cast<double>(n_partials * tile_m * tile_n) / threads_per_wave;
      L_reduce = L_sync + L_partial_read + L_accumulate;
    }

    return L_acc_transfer + L_edge_check + L_store + L_reduce;
  };

  // Evaluate all tile types
  double L_epilogue_interior = 0.0;
  double L_epilogue_n_edge = 0.0;
  double L_epilogue_m_edge = 0.0;
  double L_epilogue_corner = 0.0;
  if (has_interior)
    L_epilogue_interior = compute_tile_epilogue(MT_M, MT_N, false);
  if (has_n_edge)
    L_epilogue_n_edge = compute_tile_epilogue(MT_M, n_remainder, false);
  if (has_m_edge)
    L_epilogue_m_edge = compute_tile_epilogue(m_remainder, MT_N, true);
  if (has_m_edge && has_n_edge)
    L_epilogue_corner = compute_tile_epilogue(m_remainder, n_remainder, true);

  // Take the worst case
  double L_epilogue = std::max({L_epilogue_interior, L_epilogue_n_edge, L_epilogue_m_edge, L_epilogue_corner});

  if(context.num_timesteps > 1)
  {
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
  bool debug = runtime_options::get().debug_enabled;

  // Extract parameters from structured types
  const size_t K = problem.size.k;
  size_t batch   = problem.batch;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  const auto a_bits  = datatype_to_bits(problem.a_dtype);
  const auto b_bits  = datatype_to_bits(problem.b_dtype);
  const auto d_bytes = data_type_to_bytes(problem.d_dtype);

  // Extract parameters from context
  const size_t grid_m = context.grid_m;
  const size_t grid_n = context.grid_n;
  const size_t num_active_cus = context.active_cus;
  const size_t splitting_factor = context.splitting_factor;
  const double mem_bw_occ = context.mem_bw_limited;

  heuristic_params_t heuristic = get_heuristic_params(problem, hardware, config);

  // 1) Compute per-tile latencies
  double L_compute = compute_mt_compute_latency(problem, hardware, config);
  double L_mem = compute_memory_latency(problem, hardware, config, context);

  // TODO Does work utilization need to be 128-byte rounded for a cache line?
  double utilization        = calculate_work_utilization(problem, config);
  double output_utilization = calculate_output_utilization(problem, config, 1UL);
  // The effective latency per useful operation increases as utilization drops.
  // This penalty affects BOTH compute and memory bounds for the tile's core work.
  double effective_tile_penalty = (utilization > 1e-9) ? (1.0 / (utilization)) : 1.0;
  double output_utilization_penalty =
      (output_utilization > 1e-9) ? (1.0 / (output_utilization)) : 1.0;

  // 2) Work-group setup & iteration latencies
  double L_WG_setup = 1;  // WG_setup_Latency

  // 3) Prologue and Epilogue latencies
  // Prologue and Epilogue overhead are reduced with higher occupancy kernels.
  size_t real_occupancy =
      std::min(std::max(config.occupancy, static_cast<int>(1)),
               static_cast<int>(math::safe_ceil_div(grid_m * grid_n * batch * splitting_factor,
                                                    hardware.N_CU)));  // Number of WGs per CU.
  double occupancy_factor = pow(heuristic.occupancy_decay_base, real_occupancy);

  // 3-1) Prologue: set as memory latency
  double L_prologue = L_mem;
  L_prologue *= effective_tile_penalty;
  L_prologue *= occupancy_factor;

  // 3-2) Epilogue
  double L_epilogue = compute_epilogue_latency(problem, hardware, config, context);
  L_epilogue += L_compute;

  // Block 4: K-padding penalty (if applicable)
  double problem_k_quant = static_cast<double>(K % MT_K) / static_cast<double>(K);
  L_epilogue += problem_k_quant * heuristic.k_padding_penalty;

  // 4) Single-tile latency (apply penalty after finding the bottleneck)
  // tf32 emu has some more overhead
  double L_cvt = 0;
  if ((problem.mi_dtype == data_type_t::XFloat32) &&
      (hardware.arch == hardware_t::architecture_t::gfx950)) {
    L_cvt = compute_cvt_overhead(problem, hardware, config);
  } else if ((a_bits == 32) && (b_bits == 32) && (problem.mi_dtype == data_type_t::BFloat16) &&
             (hardware.arch == hardware_t::architecture_t::gfx950))  // SS_BSS on GFX950
  {
    L_cvt = compute_cvt_overhead_x1(problem, hardware, config);
  }
  double L_tile_single =
      std::max(L_compute * heuristic.weight_compute, L_mem * heuristic.weight_memory);
  L_tile_single *= heuristic.main_loop_efficiency;
  // L_tile_single *= effective_tile_penalty;
  L_tile_single += L_cvt;

  // 5) Number of K-iterations (excluding epilogue), at least 1
  const long k_per_split = static_cast<long>(math::safe_ceil_div(K, splitting_factor));
  long num_iter =
      std::max(static_cast<long>(math::safe_ceil_div(static_cast<size_t>(k_per_split), MT_K) - 1),
               static_cast<long>(1));

  // 6) Total tile latency
  double L_tile_total = L_tile_single * static_cast<double>(num_iter);
  L_tile_total += heuristic.weight_prologue * L_prologue;
  L_tile_total += heuristic.weight_epilogue * L_epilogue;
  L_tile_total += heuristic.weight_wg_setup * L_WG_setup;
  L_tile_total += heuristic.weight_loop_overhead * static_cast<double>(num_iter);

  // Apply final tile total weight
  L_tile_total *= heuristic.weight_tile_total;

  if(debug)
  {
    OLOG_DEBUG("utilization: " << utilization);
    OLOG_DEBUG("output_utilization: " << output_utilization);
    OLOG_DEBUG("effective_tile_penalty: " << effective_tile_penalty);
    OLOG_DEBUG("config.occupancy: " << config.occupancy);
    OLOG_DEBUG("real_occupancy: " << real_occupancy);

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

double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const context_t& context) {
  // Assume latency of a timestep is latency of a single K-complete output tile computed on one CU.
  double L_timestep =
      compute_tile_latency(problem, hardware, config, context);

  return L_timestep;
}

// Compute the latency of the parallel reduction kernel (separate kernel launch).
// This is a second kernel that reads all partials from workspace, accumulates, and writes output.
double compute_parallel_reduction_latency(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const config_t& config,
                                          const context_t& context) {
  // Only applies to parallel reduction with splitting
  if (context.splitting_factor <= 1 || context.reduction_strategy != reduction_t::parallel)
    return 0.0;

  // Extract parameters
  const size_t num_tiles = context.num_tiles;
  const size_t splitting_factor = context.splitting_factor;
  const size_t d_bytes = data_type_to_bytes(problem.d_dtype);
  const size_t tile_elements = config.mt.m * config.mt.n;

  // Constants
  constexpr double kernel_launch_overhead = 10000.0;
  constexpr double cycles_per_barrier = 500.0; // global memory fence + barrier sync
  constexpr size_t threads_per_wave = 64;

  // Log-based tree reduction (single kernel launch, sync barriers between rounds):
  // Round i: ceil(remaining/2) WGs per tile, each reads 2 partials, accumulates, writes 1.
  // A sync barrier separates each round (not a separate kernel launch).
  // Total rounds = ceil(log2(splitting_factor)).
  const size_t num_rounds = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(splitting_factor))));
  if (num_rounds == 0) return 0.0;

  // One kernel launch for the entire reduction
  double L_total = kernel_launch_overhead;
  size_t partials_remaining = splitting_factor;

  // Cache-aware bandwidth selection per round:
  // L2 is ~10× faster than DRAM, MALL is ~3× faster.
  // When the round's working set fits in cache, use the higher bandwidth.
  const double l2_bw_boost = 10.0;   // L2 is ~10× DRAM bandwidth
  const double mall_bw_boost = 3.0;  // MALL is ~3× DRAM bandwidth
  const double l2_capacity = static_cast<double>(hardware.L2_capacity);
  // MALL capacity: approximate as L2 × 64 (~256MB for MI300X)
  const double mall_capacity = l2_capacity * 64.0;

  for (size_t round = 0; round < num_rounds; ++round) {
    // This round: each WG reads 2 partials, accumulates, writes 1
    size_t wgs_per_tile = partials_remaining / 2;
    if (wgs_per_tile == 0) wgs_per_tile = 1;
    size_t total_wgs = num_tiles * wgs_per_tile;

    // Working set for this round: all partials being read + written
    double partial_size = static_cast<double>(tile_elements) * d_bytes;
    double round_working_set = total_wgs * 3.0 * partial_size; // 2 reads + 1 write per WG

    // Base DRAM bandwidth for this occupancy
    double dram_bw = std::max(
        hardware.mem3_perf_ratio * compute_mem_bw_from_occupancy(hardware, total_wgs), 1e-12);

    // Boost bandwidth if working set fits in cache
    double effective_bw = dram_bw;
    if (round_working_set <= l2_capacity)
      effective_bw = dram_bw * l2_bw_boost;
    else if (round_working_set <= mall_capacity)
      effective_bw = dram_bw * mall_bw_boost;

    double per_cu_bw = effective_bw / std::max(static_cast<double>(total_wgs), 1.0);

    // Each WG: read 2 partial results of the same output tile, accumulate, write 1 merged partial
    double L_read  = (2.0 * partial_size) / per_cu_bw;
    double L_write = partial_size / per_cu_bw;
    double L_acc   = static_cast<double>(tile_elements) / threads_per_wave;

    // Timesteps if total_wgs > N_CU
    size_t timesteps = math::safe_ceil_div(total_wgs, hardware.N_CU);

    // Round cost: barrier + per-tile work × timesteps
    double L_round = cycles_per_barrier + (L_read + L_acc + L_write) * timesteps;
    L_total += L_round;

    // Next round has half as many partials
    partials_remaining = (partials_remaining + 1) / 2;
  }

  // Penalty for high split-to-tile ratio: when splitting_factor >> num_tiles,
  // the reduction becomes pathologically inefficient — late rounds have very few WGs,
  // the last rounds are essentially serial, and workspace memory traffic is massive.
  // Split-ratio penalty for pathologically high split-to-tile ratios
  double split_ratio = static_cast<double>(splitting_factor) / std::max(static_cast<double>(num_tiles), 1.0);
  if (split_ratio > 1.0) {
    double ratio_penalty = 1.0 + 0.3 * std::sqrt(std::log2(split_ratio));
    L_total *= ratio_penalty;
  }

  // High split factor overhead: StreamK dispatch, workspace allocation,
  // partial tile assignment, and reduced pipeline efficiency per WG.
  constexpr double split_overhead_per_doubling = 0.15;
  if (splitting_factor > 1) {
    double split_penalty = 1.0 + split_overhead_per_doubling * std::log2(static_cast<double>(splitting_factor));
    L_total *= split_penalty;
  }

  if (runtime_options::get().debug_enabled) {
    OLOG_DEBUG("L_parallel_reduce_split_ratio: " << split_ratio);
    OLOG_DEBUG("L_parallel_reduce_rounds: " << num_rounds);
    OLOG_DEBUG("L_parallel_reduce_total: " << L_total);
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

  const int a_bits  = datatype_to_bits(problem.a_dtype);
  const int b_bits  = datatype_to_bits(problem.b_dtype);
  const int a_bytes = data_type_to_bytes(problem.a_dtype);

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

  // Compute latency for all timesteps.
  // Sublinear scaling: later timesteps benefit from warm caches and steady-state pipeline.
  // total = L_timestep × num_timesteps^alpha, where alpha < 1.
  constexpr double timestep_scaling_alpha = 0.98;
  double effective_timesteps = std::pow(static_cast<double>(context.num_timesteps), timestep_scaling_alpha);
  double total_latency = L_timestep * effective_timesteps;

  // Add parallel reduction kernel cost (separate kernel launch, 0 if not parallel)
  double L_parallel_reduce = compute_parallel_reduction_latency(problem, hardware, config, context);
  total_latency += L_parallel_reduce;

  bool debug = runtime_options::get().debug_enabled;
  if (debug)
  {
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
  prob_info.M = static_cast<double>(problem.size.m);
  prob_info.N = static_cast<double>(problem.size.n);
  prob_info.K = static_cast<double>(problem.size.k);
  prob_info.NumBatches = static_cast<double>(problem.batch);
  prob_info.bpeA = static_cast<uint32_t>(datatype_to_bits(problem.a_dtype) / 8);
  prob_info.bpeB = static_cast<uint32_t>(datatype_to_bits(problem.b_dtype) / 8);
  prob_info.bpeD = static_cast<uint32_t>(datatype_to_bits(problem.d_dtype) / 8);
  prob_info.bpeCompute = static_cast<uint32_t>(datatype_to_bits(problem.mi_dtype) / 8);
  prob_info.transA = (problem.a_transpose == transpose_t::T);
  prob_info.transB = (problem.b_transpose == transpose_t::T);
  prob_info.swizzleTensorA = config.tensile().swizzle_a;
  prob_info.swizzleTensorB = config.tensile().swizzle_b;
  prob_info.dataType = problem.mi_dtype;

  // Convert config_t to Formocast::SizeMapping
  Formocast::SizeMapping size_mapping;
  size_mapping.macroTile[0] = static_cast<int>(config.mt.m);
  size_mapping.macroTile[1] = static_cast<int>(config.mt.n);
  size_mapping.macroTile[2] = static_cast<int>(config.mt.k);
  size_mapping.matrixInstruction[0] = static_cast<int>(config.mi.m);
  size_mapping.matrixInstruction[1] = static_cast<int>(config.mi.n);
  size_mapping.matrixInstruction[2] = static_cast<int>(config.mi.k);
  size_mapping.matrixInstruction[3] = 1;  // Default

  // Use depth_u if set, otherwise use mt.k
  size_mapping.depthU = (config.tensile().depth_u > 0) ? config.tensile().depth_u : config.mt.k;

  size_mapping.globalSplitU = config.tensile().global_split_u;
  size_mapping.globalAccumulation = config.tensile().global_accumulation;
  size_mapping.LocalSplitU = config.tensile().local_split_u;

  size_mapping.grvwA = config.grvw_a;
  size_mapping.grvwB = config.grvw_b;
  size_mapping.gwvwD = config.gwvw_d;
  size_mapping.gwvwC = config.gwvw_d;  // Typically same as D

  size_mapping.DirectToVgprA = config.tensile().direct_to_vgpr_a;
  size_mapping.DirectToVgprB = config.tensile().direct_to_vgpr_b;
  size_mapping.DirectToLdsA = config.tensile().direct_to_lds_a;
  size_mapping.DirectToLdsB = config.tensile().direct_to_lds_b;

  size_mapping.NumLoadsCoalescedA = config.tensile().num_loads_coalesced_a;
  size_mapping.NumLoadsCoalescedB = config.tensile().num_loads_coalesced_b;
  size_mapping.VectorWidthA = config.vector_width_a;
  size_mapping.VectorWidthB = config.vector_width_b;

  size_mapping.waveNum = config.tensile().wave_num;
  size_mapping.waveGroup[0] = config.tensile().wave_group_m;
  size_mapping.waveGroup[1] = config.tensile().wave_group_n;

  size_mapping.workGroupMapping = config.workgroup_mapping;
  size_mapping.workGroupMappingXCC = config.tensile().workgroup_mapping_xcc;
  size_mapping.workGroupMappingXCCGroup = config.tensile().workgroup_mapping_xcc_group;
  size_mapping.globalSplitUCoalesced = config.tensile().global_split_u_coalesced;
  size_mapping.globalSplitUWorkGroupMappingRoundRobin = config.tensile().global_split_u_wgm_round_robin;

  size_mapping.CUOccupancy = config.occupancy;
  size_mapping.PrefetchGlobalRead = config.tensile().prefetch_global_read;
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
