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
namespace gemm {

// Forward declaration for internal Formocast latency computation
static double compute_formocast_latency(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config);

// Forward declarations (defined later; used by the context_t constructor).
operand_traffic_t compute_operand_traffic(
    const problem_t& problem,
    const config_t& config,
    const context_t& context,
    size_t transaction_bytes = heuristic_defaults_t::DRAM_SECTOR_BYTES);

// Safe Tensile-params accessor: returns the config's tensile params if present,
// else a default-constructed instance.  The prediction model must not throw for
// configs without a Tensile backend (tests, Python bindings, direct callers);
// only PredictionLibrary populates the backend.
static const tensile_params_t& tparams(const config_t& config) {
  static const tensile_params_t kDefaultTensile{};
  return config.has_tensile_params() ? config.tensile() : kDefaultTensile;
}

/* ---------------------------------------------------------------------------------------- */
/* context_t constructor                                                                    */
/* ---------------------------------------------------------------------------------------- */
context_t::context_t(const problem_t& problem, const hardware_t& hardware, const config_t& config) {
  // Effective usable CU count. Decided once here and consumed across the model
  // (launch params, occupancy, cache/epilogue/reduction estimates). A non-zero
  // problem.num_cus caps the count; 0 falls back to the full hardware count.
  n_cu              = resolve_num_cus(problem.num_cus, hardware.N_CU);
  const size_t N_CU = n_cu;

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

  auto [reduction, wgs, cus, timesteps, split] =
      compute_launch_parameters(problem, hardware, config, config.grid_selection);
  tile_schedule      = streamk::select_hybrid_mode(problem, hardware, config, problem.num_cus);
  reduction_strategy = reduction;
  num_wgs            = wgs;
  num_timesteps      = timesteps;
  splitting_factor   = split;
  k_per_split        = math::safe_ceil_div(problem.size.k, splitting_factor);
  k_iters            = (config.mt.k > 0) ? math::safe_ceil_div(k_per_split, config.mt.k) : 1;

  // Per-operand DRAM traffic: pure function of the values set above; computed
  // once here and shared by the cache-hit and memory-latency models.
  traffic = compute_operand_traffic(problem, config, *this);

  // Hardware-derived values
  active_cus           = cus;
  mem_bw_limited       = compute_mem_bw_from_occupancy(hardware, active_cus);
  write_mem_bw_limited = compute_mem_bw_from_occupancy(hardware, num_output_tiles);
  // Per-CU wave-pass count, capped so the 0.95^N amortization saturates
  // around the first few resident WGs.  Without the cap, batched problems
  // (real_occupancy in the tens or hundreds) drive occupancy_factor toward 0
  // and erase per-tile prologue/epilogue cost as a ranking signal.
  real_occupancy = static_cast<int>(math::safe_ceil_div(grid_m * grid_n * batch * splitting_factor, N_CU));
  const size_t real_occupancy_for_factor =
      std::min(static_cast<size_t>(real_occupancy), heuristic_defaults_t::OCCUPANCY_AMORT_CAP);
  occupancy_factor =
      pow(heuristic.occupancy_decay_base, real_occupancy_for_factor);

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

    OLOG_DEBUG("======== Origami Debug Info ========");  // This signature indicates the start of
                                                         // the debug information.
    OLOG_DEBUG("ProblemSize (MxNxBxK): " << int(M) << "x" << int(N) << "x" << int(batch) << "x"
                                         << int(K));
    OLOG_DEBUG("transpose: " << (problem.a_transpose == transpose_t::T ? "T" : "N") << (problem.b_transpose == transpose_t::T ? "T" : "N"));
    OLOG_DEBUG("MacroTile: " << int(MT_M) << "x" << int(MT_N) << "x" << int(MT_K));
    OLOG_DEBUG("MatrixInstruction: " << int(MI_M) << "x" << int(MI_N) << "x" << int(MI_K));
    OLOG_DEBUG("ElementSizeA (bits): " << int(a_bits));
    OLOG_DEBUG("ElementSizeB (bits): " << int(b_bits));
    OLOG_DEBUG("CacheHintsA: " << int(config.cache_hints_a));
    OLOG_DEBUG("CacheHintsB: " << int(config.cache_hints_b));
    OLOG_DEBUG("CacheHintsD: " << int(config.cache_hints_d));
    OLOG_DEBUG("DirectToLdsA: " << int(tparams(config).direct_to_lds_a));
    OLOG_DEBUG("DirectToLdsB: " << int(tparams(config).direct_to_lds_b));
    OLOG_DEBUG("CUOccupancy(cfg): " << int(config.occupancy));
    OLOG_DEBUG("LocalSplitU: " << int(tparams(config).local_split_u));
    OLOG_DEBUG("OneLDSBuffer: " << int(tparams(config).one_lds_buffer));
    OLOG_DEBUG("PrefetchGlobalRead: " << int(tparams(config).prefetch_global_read));
    OLOG_DEBUG("Wave: " << int(tparams(config).wave_group_m) << "x" << int(tparams(config).wave_group_n));
    OLOG_DEBUG("StreamK: " << int(config.stream_k));

    OLOG_DEBUG("Grid: " << int(grid_m) << "x" << int(grid_n));
    OLOG_DEBUG("NumOutputTiles: " << int(num_output_tiles));
    OLOG_DEBUG("NumWGs: " << int(num_wgs));
    OLOG_DEBUG("NumTimesteps: " << int(num_timesteps));
    OLOG_DEBUG("SplittingFactor: " << int(splitting_factor));
    OLOG_DEBUG("ReductionStrategy: " << int(reduction_strategy));
    OLOG_DEBUG("TileSchedule: " << hybrid_mode_to_string(tile_schedule));

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

// Round the number of elements to the nearest multiple of `transaction_bytes`.
// Used by the memory-latency model where the "natural" coalesced load width is
// not always a full 128-byte L1 line (e.g. bf16 DepthU=32 is a 64-byte load).
size_t round_elements_to_NB(size_t elements,
                            size_t element_size_bits,
                            size_t transaction_bytes) {
  if (element_size_bits == 0 || transaction_bytes == 0) return elements;
  auto round_up_mul             = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
  const size_t transaction_bits = transaction_bytes * 8u;
  const size_t g                = std::gcd(element_size_bits, transaction_bits);
  const size_t E_block          = transaction_bits / g;
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

  // Honor the caller's CU budget (problem.num_cus); 0 means use all CUs. Keeps
  // the predicted mapping consistent with the capped model (context.n_cu).
  const size_t N_CU    = resolve_num_cus(problem.num_cus, hardware.N_CU);
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
      return {0, 0, 0, 1};
    else
      return {0, 0, NUM_XCD, 1};
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
      return {0, out_chunk, out_wgmxcc, use_wgmxcc ? static_cast<int32_t>(grid_n) : 1};
    else if (nta < 4 && ntb > 3)
      return {0, out_chunk, out_wgmxcc, use_wgmxcc ? -static_cast<int32_t>(grid_m) : 1};
    else
      return {0, 0, NUM_XCD, 1};
  }

  // Batch case
  if (batch > 1) {
    auto numMTs_total = numMTs * batch;
    if (numMTs == 1 || numMTs_total <= NUM_XCD || numMTs % NUM_XCD == 0)
      return {0, 0, 0, 1};
    else
      return {0, 0, NUM_XCD, 1};
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
  if (out_wgmxcc == 0 || grid_m == 1 || grid_n == 1) return {0, 0, out_wgmxcc, 1};

  // If the grid is large, use the square root of the number of CUs as the WGM.
  // Solution is not very sensitive to the WGM value in this case.
  const size_t grid_threshold = std::sqrt(N_CU);
  if (grid_m > grid_threshold && grid_n > grid_threshold)
    return {0, 0, out_wgmxcc, static_cast<int32_t>(std::ceil(std::sqrt(N_CU / NUM_XCD)))};

  size_t numWGsPerXCD = std::min(math::safe_ceil_div(numMTs, NUM_XCD), cus_per_xcd);
  // If there is enough work per L2 and the grid_n is small, use the grid_n as the WGM.
  if (numWGsPerXCD >= cus_per_xcd / 2 && grid_n <= 8)
    return {0, 0, out_wgmxcc, static_cast<int32_t>(grid_n)};

  // Build candidate list
  size_t wgm_cap = std::min(grid_n, numWGsPerXCD / 2);
  if (wgm_cap == 0) return {0, 0, out_wgmxcc, 1};

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

  return {0, 0, out_wgmxcc, static_cast<int32_t>(best_wgm)};
}

// Compute the launch parameters for the kernel.
std::tuple<reduction_t, size_t, size_t, size_t, size_t> compute_launch_parameters(
    const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config,
    grid_selection_t grid_selection) {
  const size_t num_mts = streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  size_t num_wgs                 = num_mts;
  reduction_t reduction_strategy = reduction_t::none;

  if (config.stream_k > 0) {
    reduction_strategy = streamk::select_reduction(problem, hardware, config, grid_selection);
    auto config_with_reduction               = config;
    config_with_reduction.reduction_strategy = reduction_strategy;
    num_wgs = streamk::select_grid_size(problem, hardware, config_with_reduction, grid_selection);
  }

  // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
  // That means, we artifically observe one full timesteps, but that is not what actually happens
  // under the hood. From a theoretical point of view, these distributions change all of the
  // computations in Origami. With current implementation, it is hard to capture that
  // behaviour analytically. So for now, if the num_wgs is less than the num_mts, we calculate
  // num_timesteps based on the num_mts. Otherwise, we use num_wgs to compute num_timesteps.
  // Usable CU count: derived from the problem's CU budget (problem.num_cus).
  const size_t usable_cus       = resolve_num_cus(problem.num_cus, hardware.N_CU);
  const size_t num_active_cus   = num_wgs < usable_cus ? num_wgs : usable_cus;
  const size_t num_timesteps    = num_wgs > num_mts ? math::safe_ceil_div(num_wgs, usable_cus)
                                                    : math::safe_ceil_div(num_mts, usable_cus);
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

  // 2) Bytes (per K-slice), using ceil-div to whole bytes
  const double bytesA = wave_tile_m * MT_K * static_cast<double>(a_bytes);
  const double bytesB = wave_tile_n * MT_K * static_cast<double>(b_bytes);

  // 3) Modeled transfer quanta (LDS->VGPR transfer width)
  //      dsA = bytesA / (LDS_XFER * MI_M)
  //      dsB = bytesB / (LDS_XFER * MI_N)
  //      GR  = dsA  (global->LDS modeled equal to A-side DS)
  constexpr double lds_xfer = static_cast<double>(heuristic_defaults_t::LDS_XFER_BYTES);
  const double dsA = (bytesA / lds_xfer) / MI_M;  // LDS->VGPR for A
  const double dsB = (bytesB / lds_xfer) / MI_N;  // LDS->VGPR for B
  const double GR  = dsA;                         // Global->LDS reads
  const double LR  = dsA + dsB;                   // total DS->VGPR

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

  // 2) Bytes (per K-slice), using ceil-div to whole bytes
  auto a_bytes = data_type_to_bytes(problem.a_dtype);
  auto b_bytes = data_type_to_bytes(problem.b_dtype);

  const double bytesA = static_cast<double>(wave_tile_m) * config.mt.k * a_bytes;
  const double bytesB = static_cast<double>(wave_tile_n) * config.mt.k * b_bytes;

  // 3) Modeled transfer quanta (LDS->VGPR transfer width)
  //      dsA = bytesA / (LDS_XFER * MI_M)
  //      dsB = bytesB / (LDS_XFER * MI_N)
  //      GR  = dsA  (global->LDS modeled equal to A-side DS)
  constexpr double lds_xfer = static_cast<double>(heuristic_defaults_t::LDS_XFER_BYTES);
  const double dsA = (bytesA / lds_xfer) / static_cast<double>(config.mi.m);  // LDS->VGPR for A
  const double dsB = (bytesB / lds_xfer) / static_cast<double>(config.mi.n);  // LDS->VGPR for B
  const double GR  = dsA;                                                  // Global->LDS reads
  const double LR  = dsA + dsB;                                            // total DS->VGPR

  // 4) Heuristic cycle weights, scaled to MI latency (calibrated at
  //    L_MI_bf16 == 16 to A=104, B=8, C=4).
  const double A = (104.0 / 16.0) * L_MI_bf16;  // CVT per LR-sized chunk (DS->VGPR)
  const double B = (8.0 / 16.0) * L_MI_bf16;    // hidden per spare MFMA slot
  const double C = (4.0 / 16.0) * L_MI_bf16;    // hidden per (LR+GR) slot

  // 5) Exposed vs hidden CVT
  const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
  const double cvt        = A * dsA;                         // only DS->VGPR contributes CVT
  const double H          = B * spare_mfma + C * (LR + GR);  // hidden cycles
  const double overhead   = std::max(cvt - H, 0.0);

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
// Per-operand DRAM bytes one WG pulls for A/B over its K range (transaction-aligned rows).
operand_traffic_t compute_operand_traffic(const problem_t& problem,
                                          const config_t& config,
                                          const context_t& context,
                                          size_t transaction_bytes) {
  const double cl = static_cast<double>(transaction_bytes);
  const bool a_trans = (problem.a_transpose == transpose_t::T);
  const bool b_trans = (problem.b_transpose == transpose_t::T);

  const double mt_m_active = static_cast<double>(
      std::min(static_cast<size_t>(problem.size.m), config.mt.m));
  const double mt_n_active = static_cast<double>(
      std::min(static_cast<size_t>(problem.size.n), config.mt.n));
  const double k_per_wg = static_cast<double>(std::max(context.k_per_split, size_t{1}));
  const double mt_k     = static_cast<double>(std::max(config.mt.k, size_t{1}));

  operand_traffic_t traffic{};
  traffic.k_iters = std::max(k_per_wg / mt_k, 1.0);

  const double a_row_count = a_trans ? mt_m_active : k_per_wg;
  const double a_row_bytes = a_trans ? k_per_wg * context.a_bytes : mt_m_active * context.a_bytes;
  traffic.a_tile_bytes     = a_row_count * std::ceil(a_row_bytes / cl) * cl;

  const double b_row_count = b_trans ? k_per_wg : mt_n_active;
  const double b_row_bytes = b_trans ? mt_n_active * context.b_bytes : k_per_wg * context.b_bytes;
  traffic.b_tile_bytes     = b_row_count * std::ceil(b_row_bytes / cl) * cl;

  traffic.a_iter_bytes = traffic.a_tile_bytes / traffic.k_iters;
  traffic.b_iter_bytes = traffic.b_tile_bytes / traffic.k_iters;
  traffic.a_cl_share   = std::min(1.0, problem.size.m * context.a_bytes / cl);
  traffic.b_cl_share   = std::min(1.0, problem.size.n * context.b_bytes / cl);

  return traffic;
}

// Total A+B bytes loaded for a single K-window of MT_M x MT_N tile (transaction-aligned).
static double compute_operand_window_bytes(const problem_t& problem,
                                           const config_t& config,
                                           const context_t& context,
                                           size_t k_window,
                                           size_t transaction_bytes) {
  const int a_bits = datatype_to_bits(problem.a_dtype);
  const int b_bits = datatype_to_bits(problem.b_dtype);
  const bool a_trans = (problem.a_transpose == transpose_t::T);
  const bool b_trans = (problem.b_transpose == transpose_t::T);

  const size_t Ld_A = a_trans
      ? config.mt.m * round_elements_to_NB(k_window, a_bits, transaction_bytes)
      : round_elements_to_NB(config.mt.m, a_bits, transaction_bytes) * k_window;
  const size_t Ld_B = b_trans
      ? round_elements_to_NB(config.mt.n, b_bits, transaction_bytes) * k_window
      : config.mt.n * round_elements_to_NB(k_window, b_bits, transaction_bytes);

  return static_cast<double>(Ld_A) * context.a_bytes
       + static_cast<double>(Ld_B) * context.b_bytes;
}

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

// Estimate per-operand cache hit rates for L1, L2, and MALL.
cache_hit_rates_t estimate_cache_hit_rates(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           const context_t& context) {
  // Extract parameters
  const size_t num_xcd     = hardware.NUM_XCD;
  const size_t N_CU        = context.n_cu;
  const double l2_cap      = static_cast<double>(hardware.L2_capacity);
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

  // Per-tile data volumes.
  //
  // Each WG processes a 2-D block of (MT_M_active x k_per_split) elements of A
  // and (k_per_split x MT_N_active) elements of B.  We use min(M, MT_M) /
  // min(N, MT_N) so out-of-bounds rows/columns (Tensile's bounded
  // `buffer_load`s) don't count as DRAM traffic.
  //
  // The DRAM layout has one of K, M, or N as the contiguous axis; cache-line
  // round-up is applied ONCE on the full contig extent that one WG processes.
  // The previous formulation rounded per K-iter (slice of MT_K), which
  // over-counts cache lines when MT_K * a_bytes < cache_line: adjacent
  // K-iters' rows are physically contiguous in DRAM and share cache lines.
  //
  //   Layout       | contig axis | row count       | bytes per row
  //   ----------------------------------------------------------------
  //   TN (a_T=T)   | K           | MT_M_active     | k_per_split * a_bytes
  //   NN (a_T=N)   | M           | k_per_split     | MT_M_active * a_bytes
  //   NT (b_T=T)   | N           | k_per_split     | MT_N_active * b_bytes
  //   NN (b_T=N)   | K           | MT_N_active     | k_per_split * b_bytes
  //
  // Assumes natural leading dim (lda == K for TN, lda == M for NN, etc.) so
  // adjacent rows are contiguous in DRAM.  This holds for the bench problems
  // we model; if a user passes padded `lda > K`, this would under-count.
  constexpr double cl   = static_cast<double>(heuristic_defaults_t::DRAM_SECTOR_BYTES);
  const bool a_temporal = config.cache_hints_a < 4;
  const bool b_temporal = config.cache_hints_b < 4;
  const operand_traffic_t& traffic = context.traffic;
  const double a_tile = traffic.a_tile_bytes;
  const double b_tile = traffic.b_tile_bytes;
  const double a_iter = traffic.a_iter_bytes;
  const double b_iter = traffic.b_iter_bytes;

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
  // Calculate the concurrent load/store on the XCD
  const double a_conc          = a_temporal ? static_cast<double>(l2_tiles.m) * a_iter : 0.0;
  const double b_conc          = b_temporal ? static_cast<double>(l2_tiles.n) * b_iter : 0.0;
  const double total_conc_load = a_conc + b_conc;
  const double concurrent_load = total_conc_load * l2_tiles.k * l2_tiles.b;

  // Count each tile's full unique bytes here; sub-line cross-tile sharing is
  // applied to load volume in compute_memory_latency().

  // Spatial Reuse:
  const size_t rectangular_mn = l2_tiles.m * l2_tiles.n;
  const size_t tiles_on_xcd   = std::min(cus_per_xcd, tiles_per_xcd);
  const size_t kb             = std::max(l2_tiles.k * l2_tiles.b, static_cast<size_t>(1));
  const size_t actual_mn      = std::min(math::safe_ceil_div(tiles_on_xcd, kb), rectangular_mn);

  double l2_unique_a    = a_temporal ? l2_tiles.m * a_tile : 0.0;
  double l2_unique_b    = b_temporal ? l2_tiles.n * b_tile : 0.0;
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
  // When a temporal operand competes with another temporal stream, that
  // stream can evict lines even if the operand's own footprint would fit.
  // Model that asymmetrically based on how much of each operand's
  // effective working set comes from the other stream.
  double pollution_rate_a = 1.0;
  double pollution_rate_b = 1.0;
  // Pollution requires two competing temporal streams; the per-operand
  // asymmetry comes from interference_frac_{a,b} below.
  const bool both_temporal = a_temporal && b_temporal;
  if (both_temporal && l2_residency_a < 1.0 && effective_load_a > 0.0) {
    const double interference_frac_a = a_interference / effective_load_a;
    pollution_rate_a = 1.0 - (1.0 - heuristic.l2_pollution_penalty) * interference_frac_a;
  }
  if (both_temporal && l2_residency_b < 1.0 && effective_load_b > 0.0) {
    const double interference_frac_b = b_interference / effective_load_b;
    pollution_rate_b = 1.0 - (1.0 - heuristic.l2_pollution_penalty) * interference_frac_b;
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
    const double headroom    = 1.0 - concurrent_load / l2_cap;
    const double amp_boost   = headroom * headroom;
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
    const double headroom    = 1.0 - concurrent_load / l2_cap;
    const double amp_boost   = headroom;
    l2_rate_a += amp_boost * std::max(amp_ceiling - l2_rate_a, 0.0);
    l2_rate_b += amp_boost * std::max(amp_ceiling - l2_rate_b, 0.0);
  }

  // Implicit L1 residency + request amplification (skinny-dimension GEMMs):
  // When one output dimension is tiny, the operand along that dim is reused
  // across many WGs (and across many wavefronts within a WG), so some of its
  // traffic never reaches L2. Physically this happens whenever that reused
  // operand's per-iteration footprint fits in L1, regardless of whether
  // the OTHER operand is temporal or non-temporal.
  //
  // The only difference the other operand makes:
  //   * If it's non-temporal (NT), it bypasses L1 -> all 32 KB are dedicated
  //     to the reused stripe.
  //   * If it's also temporal, it steals some L1 capacity. Each WG on a CU
  //     sees a fresh slice of it, so that operand itself does NOT get an L1
  //     hit here (its working-set union overflows L1), but it does shrink
  //     the headroom available to the reused stripe.
  //
  // Combined L1 footprint only counts temporal operands; NT operands bypass
  // L1 entirely and contribute 0.
  cache_hit_rates_t rates{};
  auto& [H_mem_l1_A, H_mem_l1_B, H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B] = rates;
  {
    const bool skinny_m          = (grid.m <= 2 && grid.n > grid.m * 8);
    const bool skinny_n          = (grid.n <= 2 && grid.m > grid.n * 8);
    const bool single_stream     = (grid.k == 1) && (grid.b == 1);

    const double a_l1_ft  = a_temporal ? a_iter : 0.0;
    const double b_l1_ft  = b_temporal ? b_iter : 0.0;

    if (single_stream && skinny_m && a_temporal) {
      if (hardware.l1_capacity > 0 && a_l1_ft <= static_cast<double>(hardware.l1_capacity)) {
        const double headroom = 1.0 - a_l1_ft / static_cast<double>(hardware.l1_capacity);
        H_mem_l1_A            = heuristic.l1_hit_rate_ceiling_skinny * clamp01(headroom);
      } else if (concurrent_load < l2_cap) {
        const double headroom = 1.0 - concurrent_load / l2_cap;
        l2_rate_a += headroom *
                     std::max(heuristic.l2_amp_ceiling_skinny - l2_rate_a, 0.0);
      }
    }

    if (single_stream && skinny_n && b_temporal) {
      if (hardware.l1_capacity > 0 && b_l1_ft <= static_cast<double>(hardware.l1_capacity)) {
        const double headroom = 1.0 - b_l1_ft / static_cast<double>(hardware.l1_capacity);
        H_mem_l1_B            = heuristic.l1_hit_rate_ceiling_skinny * clamp01(headroom);
      } else if (concurrent_load < l2_cap) {
        const double headroom = 1.0 - concurrent_load / l2_cap;
        l2_rate_b += headroom *
                     std::max(heuristic.l2_amp_ceiling_skinny - l2_rate_b, 0.0);
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
    const double Ld_A_total =
        traffic.a_iter_bytes * static_cast<double>(context.active_cus) * traffic.a_cl_share;
    const double Ld_B_total =
        traffic.b_iter_bytes * static_cast<double>(context.active_cus) * traffic.b_cl_share;
    const auto [H_mem_l1, H_mem_l2, H_mem_mall] =
        aggregate_cache_hit_rates_for_debug(H_mem_l1_A,
                                            H_mem_l1_B,
                                            H_mem_l2_A,
                                            H_mem_l2_B,
                                            H_mem_mall_A,
                                            H_mem_mall_B,
                                            Ld_A_total,
                                            Ld_B_total,
                                            a_temporal,
                                            b_temporal);
    OLOG_DEBUG("H_mem_l1: " << H_mem_l1);
    OLOG_DEBUG("H_mem_l2: " << H_mem_l2);
    OLOG_DEBUG("H_mem_mall: " << H_mem_mall);
  }

  return rates;
}

// Determine the memory latency
double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const context_t& context) {
  const bool debug = context.debug;

  // Extract parameters from structured types
  const auto a_bits = datatype_to_bits(problem.a_dtype);
  const auto b_bits = datatype_to_bits(problem.b_dtype);

  const size_t num_active_cus = context.active_cus;
  double bw_limited           = context.mem_bw_limited;
  auto heuristic              = context.heuristic;

  // 1) Estimate per-operand L1/MALL/L2 hit-rates using the analytical model
  const auto [H_mem_l1_A, H_mem_l1_B, H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B] =
      estimate_cache_hit_rates(problem, hardware, config, context);

  // 2) Total loads per CU per K-iter (A + B, with MX scale bytes).
  //
  // Per-WG byte counts use the same physical formulation as
  // estimate_cache_hit_rates: round once on the full contig extent that one WG
  // processes (whole K span, with k_per_split for split-K), and use
  // min(M, MT_M) / min(N, MT_N) so out-of-bounds rows/columns (which Tensile's
  // bounded buffer descriptors return as zero without DRAM traffic) don't count
  // as real bytes.  Per-K-iter bytes are then whole-tile-bytes
  // divided by the number of K-iters the WG executes (k_per_split / MT_K),
  // not by k_per_split element count.  This keeps L_mem in per-K-iter cycles
  // so that L_mem_stream = L_mem * num_main_iters has consistent units.
  const operand_traffic_t& traffic = context.traffic;
  double Ld_CU_bytes = traffic.a_iter_bytes + traffic.b_iter_bytes;

  // Block scaled datatypes (MX): add scale bytes
  if (a_bits < 8 && problem.a_mx_block_size != 0)
    Ld_CU_bytes += math::safe_ceil_div(config.mt.mk(), problem.a_mx_block_size);
  if (b_bits < 8 && problem.b_mx_block_size != 0)
    Ld_CU_bytes += math::safe_ceil_div(config.mt.nk(), problem.b_mx_block_size);

  // 3) Total loads by all CUs, split by operand (per K-iter).
  //
  // Cross-tile cache-line sharing is modeled on load volume, not as a cache
  // hit-rate boost. When M (or N) is small, neighbouring tiles' rows/columns
  // pack into the same 64B cache line, reducing the unique bytes fetched.
  double Ld_A_total =
      traffic.a_iter_bytes * static_cast<double>(num_active_cus) * traffic.a_cl_share;
  double Ld_B_total =
      traffic.b_iter_bytes * static_cast<double>(num_active_cus) * traffic.b_cl_share;

  double total_Ld   = Ld_A_total + Ld_B_total;

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

  double Ld_l2 = Ld_A_to_l2 + Ld_B_to_l2;
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
  double L_mem = std::max({L_mem_l2, L_mem_mall, L_mem_dram});

  if (debug) {
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
                                const context_t& context,
                                double* scalar_store_fraction) {
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

  const size_t N_CU = context.n_cu;

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

  constexpr size_t WAVEFRONT_SIZE = heuristic_defaults_t::WAVEFRONT_SIZE;
  constexpr size_t STORE_PATTERN_IDEAL = 0;
  constexpr size_t STORE_PATTERN_NARROW = 1;
  constexpr size_t STORE_PATTERN_WIDE_SPLIT = 2;
  constexpr size_t STORE_PATTERN_SCALAR_EDGE = 3;
  constexpr size_t STORE_PATTERN_NONCONTIG = 4;

  const size_t wave_group_m_epi = std::max<size_t>(static_cast<size_t>(tparams(config).wave_group_m), 1);
  const size_t wave_group_n_epi = std::max<size_t>(static_cast<size_t>(tparams(config).wave_group_n), 1);
  const size_t wave_num_epi     = std::max<size_t>(wave_group_m_epi * wave_group_n_epi, 1);
  const size_t wave_issue_parallelism = std::min(wave_num_epi, hardware.simds_per_cu());
  const double wave_batches =
      std::ceil(static_cast<double>(wave_num_epi) / static_cast<double>(wave_issue_parallelism));

  // ACC->VGPR drain: keep this as a per-wave cost because waves can issue on
  // different SIMDs. Store issue below uses max(per-wave work, total work/SIMDs).
  const double acc_elems_per_thread =
      static_cast<double>(MT_M * MT_N) / (WAVEFRONT_SIZE * wave_num_epi);
  const double mi_reg_per_out = std::max(1.0, data_type_to_bytes(problem.mi_dtype) / 4.0);
  const double reads_per_wave = acc_elems_per_thread * mi_reg_per_out;
  const double L_acc_transfer = reads_per_wave * heuristic.epilogue_acc_read_parallelism;

  const size_t max_isa_store_elems = std::max(
      static_cast<size_t>(1),
      static_cast<size_t>(std::ceil(heuristic.epilogue_bytes_per_vectorized_store / d_bytes)));
  const size_t requested_svw = std::max(
      static_cast<size_t>(1),
      (config.gwvw_d > 0) ? static_cast<size_t>(config.gwvw_d)
                                      : max_isa_store_elems);
  const size_t sector_bytes = std::max<size_t>(heuristic.epilogue_cache_line_bytes, 1);
  const bool store_axis_m   = tparams(config).source_swap;

  // Natural contiguous accumulator run along the store axis.  SourceSwap stores
  // along M, the stride-1 D axis, and MFMA output contributes contiguous M rows
  // per lane.  Without SourceSwap, stores walk the non-contiguous N direction
  // unless a store-remap path exists, so model only scalar-contiguous lanes.
  const size_t natural_svw_base =
      store_axis_m ? std::max<size_t>(config.mi.m / 4, 1) : static_cast<size_t>(1);
  const size_t natural_svw = std::max<size_t>(1, std::min(natural_svw_base, max_isa_store_elems));

  // Per-CU write bandwidth: total write BW shared among all writers.
  const size_t num_writers = std::max(num_active_cus, static_cast<size_t>(1));
  const double per_cu_store_bw =
      (store_bw > 0.0) ? store_bw / static_cast<double>(num_writers) : 0.0;

  struct wave_store_plan_t {
    double active_elements = 0.0;
    double store_insts = 0.0;
    double issue_insts = 0.0;
    double sectors = 0.0;
    double useful_bytes = 0.0;
    double sector_bytes = 0.0;
    size_t pattern = STORE_PATTERN_IDEAL;
  };

  struct tile_epilogue_plan_t {
    // Per-wave store metrics summed over all waves of the tile (store_insts,
    // issue_insts, sectors, useful_bytes, sector_bytes, pattern).
    wave_store_plan_t store{};
    double acc_read = 0.0;
    double bounds = 0.0;
    double store_issue = 0.0;
    double store_memory = 0.0;
    double reduction = 0.0;
    double sector_efficiency = 1.0;
    double active_waves = 0.0;
    double max_wave_issue = 0.0;

    double total() const { return acc_read + bounds + store_issue + store_memory + reduction; }
  };

  auto wave_range = [](size_t extent, size_t parts, size_t idx) {
    const size_t begin = (extent * idx) / parts;
    const size_t end   = (extent * (idx + 1)) / parts;
    return std::pair<size_t, size_t>{begin, std::max(begin, end)};
  };

  auto active_extent = [](std::pair<size_t, size_t> r, size_t valid) {
    if (r.first >= valid) return static_cast<size_t>(0);
    return std::min(r.second, valid) - r.first;
  };

  auto compute_wave_store_plan = [&](size_t active_m,
                                     size_t active_n,
                                     bool scalar_path,
                                     double store_elem_bytes) -> wave_store_plan_t {
    wave_store_plan_t plan{};
    const size_t active_elements = active_m * active_n;
    if (active_elements == 0) return plan;

    const size_t axis_extent = store_axis_m ? active_m : active_n;
    const size_t axis_natural = std::max<size_t>(1, std::min(natural_svw, std::max(axis_extent, size_t{1})));
    const size_t logical_svw = scalar_path ? static_cast<size_t>(1) : requested_svw;
    const size_t contiguous_svw = scalar_path ? static_cast<size_t>(1) : std::min(logical_svw, axis_natural);
    const size_t split_count = scalar_path
        ? static_cast<size_t>(1)
        : std::max<size_t>(1, math::safe_ceil_div(logical_svw, contiguous_svw));

    if (scalar_path) {
      plan.pattern = STORE_PATTERN_SCALAR_EDGE;
    } else if (!store_axis_m) {
      plan.pattern = STORE_PATTERN_NONCONTIG;
    } else if (logical_svw < axis_natural) {
      plan.pattern = STORE_PATTERN_NARROW;
    } else if (logical_svw > axis_natural) {
      plan.pattern = STORE_PATTERN_WIDE_SPLIT;
    } else {
      plan.pattern = STORE_PATTERN_IDEAL;
    }

    const double logical_groups =
        std::ceil(static_cast<double>(active_elements) /
                  (static_cast<double>(WAVEFRONT_SIZE) * static_cast<double>(logical_svw)));
    plan.store_insts = std::max(1.0, logical_groups) * static_cast<double>(split_count);

    // Non-contiguous stores pay extra address/exec manipulation even when the
    // same number of buffer_store instructions is emitted.
    double address_issue = 0.0;
    if (plan.pattern == STORE_PATTERN_WIDE_SPLIT) address_issue = plan.store_insts;
    if (plan.pattern == STORE_PATTERN_NONCONTIG) address_issue = 2.0 * plan.store_insts;
    plan.issue_insts = plan.store_insts + address_issue;

    plan.useful_bytes = static_cast<double>(active_elements) * store_elem_bytes;

    // Count memory sectors per logical store group.  Ideal/narrow paths coalesce
    // the group payload.  Wide-split/non-contiguous paths touch one sector group
    // per contiguous sub-run, matching the SVW8-with-natural4 failure mode.
    const double useful_per_group = plan.useful_bytes / std::max(logical_groups, 1.0);
    const double subrun_bytes = useful_per_group / static_cast<double>(split_count);
    const double sectors_per_group =
        static_cast<double>(split_count) *
        std::max(1.0, std::ceil(subrun_bytes / static_cast<double>(sector_bytes)));
    plan.sectors = std::max(1.0, logical_groups) * sectors_per_group;

    // Streaming stores are more sensitive to partial sectors because the store
    // cannot rely on useful L2 write combining/reuse.  Keep this structural:
    // only under-filled sector groups expand the sector count.
    if (config.cache_hints_d > 3 && plan.useful_bytes > 0.0) {
      const double ideal_sectors =
          std::max(1.0, std::ceil(plan.useful_bytes / static_cast<double>(sector_bytes)));
      const double underfill = plan.sectors / ideal_sectors;
      plan.sectors *= std::min(std::max(underfill, 1.0), 2.0);
    }

    plan.sector_bytes = plan.sectors * static_cast<double>(sector_bytes);
    plan.active_elements = static_cast<double>(active_elements);
    return plan;
  };

  // Edge tile detection
  const bool has_interior  = (M >= MT_M && N >= MT_N);
  const bool has_m_edge    = (M % MT_M != 0);
  const bool has_n_edge    = (N % MT_N != 0);
  const size_t m_remainder = has_m_edge ? (M % MT_M) : MT_M;
  const size_t n_remainder = has_n_edge ? (N % MT_N) : MT_N;

  auto compute_tile_epilogue = [&](size_t tile_m, size_t tile_n, bool scalar_path) -> tile_epilogue_plan_t {
    tile_epilogue_plan_t tile{};
    tile.acc_read = L_acc_transfer;
    const bool edge_path = scalar_path || tile_m != MT_M || tile_n != MT_N;
    // Split-K main kernels write an f32 partial to workspace (not the final
    // output), for BOTH the in-kernel tree reduction and the separate parallel
    // (PostGSU) reduction.
    const double store_elem_bytes = (splitting_factor > 1)
                                        ? static_cast<double>(heuristic.epilogue_workspace_bytes_per_elem)
                                        : d_bytes;

    double max_wave_store_insts  = 0.0;
    double max_wave_sector_bytes = 0.0;
    for (size_t wg_m = 0; wg_m < wave_group_m_epi; ++wg_m) {
      const auto m_range = wave_range(MT_M, wave_group_m_epi, wg_m);
      const size_t active_m = active_extent(m_range, tile_m);
      for (size_t wg_n = 0; wg_n < wave_group_n_epi; ++wg_n) {
        const auto n_range = wave_range(MT_N, wave_group_n_epi, wg_n);
        const size_t active_n = active_extent(n_range, tile_n);
        wave_store_plan_t wave = compute_wave_store_plan(active_m, active_n, scalar_path, store_elem_bytes);
        if (wave.active_elements == 0.0) continue;

        tile.active_waves += 1.0;
        tile.store.store_insts += wave.store_insts;
        tile.store.issue_insts += wave.issue_insts;
        tile.store.sectors += wave.sectors;
        tile.store.useful_bytes += wave.useful_bytes;
        tile.store.sector_bytes += wave.sector_bytes;
        tile.max_wave_issue  = std::max(tile.max_wave_issue, wave.issue_insts);
        max_wave_store_insts  = std::max(max_wave_store_insts, wave.store_insts);
        max_wave_sector_bytes = std::max(max_wave_sector_bytes, wave.sector_bytes);
        tile.store.pattern = std::max(tile.store.pattern, wave.pattern);
      }
    }

    // Critical path over the SIMD-issue lanes.  With wave_num_epi > simds_per_cu
    // waves must run in ceil(wave_num/simds) serial batches: the slowest single
    // wave repeats for each batch.  Take the max of:
    //   (a) wave_batches × max_wave — serialized-batch bound (e.g. 6-wave WG on
    //       4-SIMD CU: 2 rounds × max_wave, not 1.5 × max_wave from total/4).
    //   (b) total / wave_issue_parallelism — throughput bound.
    auto critical_path = [&](double max_wave, double total) {
      return std::max(max_wave * wave_batches,
                      total / static_cast<double>(wave_issue_parallelism));
    };

    tile.bounds = edge_path ? heuristic.epilogue_cycles_per_bounds_check *
                                  critical_path(max_wave_store_insts, tile.store.store_insts)
                            : 0.0;
    const double store_issue_parallel = critical_path(tile.max_wave_issue, tile.store.issue_insts);
    tile.store_issue = scalar_path
        ? store_issue_parallel * heuristic.epilogue_scalar_store_penalty
        : store_issue_parallel;
    const double store_memory_bytes = critical_path(max_wave_sector_bytes, tile.store.sector_bytes);
    tile.store_memory = (per_cu_store_bw > 0.0)
        ? store_memory_bytes / per_cu_store_bw
        : 0.0;
    tile.sector_efficiency =
        (tile.store.sector_bytes > 0.0) ? tile.store.useful_bytes / tile.store.sector_bytes : 1.0;

    // Per-tile K-split reduction (in-kernel: spinlock/tree/atomic).  This still
    // uses byte movement plus fixed sync terms because the reduction path is
    // serialized by inter-WG handoff rather than by D-store coalescing alone.
    if (splitting_factor > 1 && !is_parallel_reduction) {
      size_t n_partials = splitting_factor - 1;
      double per_cu_reduce_bw = reduce_bw / static_cast<double>(num_output_tiles);
      double partial_bytes = static_cast<double>(n_partials) * tile_m * tile_n *
                             heuristic.epilogue_workspace_bytes_per_elem;
      double store_bytes = static_cast<double>(tile_m) * tile_n * store_elem_bytes;

      double L_poll_wait = store_bytes / per_cu_store_bw;
      double L_sync =
          L_poll_wait + static_cast<double>(n_partials) *
                            (heuristic.epilogue_salu_overhead + 2.0 * heuristic.epilogue_l_barrier +
                             heuristic.epilogue_l_smem);

      double L_partial_read = partial_bytes / per_cu_reduce_bw;
      double L_accumulate =
          static_cast<double>(n_partials * tile_m * tile_n) / WAVEFRONT_SIZE;
      double L_partial_write = static_cast<double>(tile_m) * tile_n * d_bytes / per_cu_reduce_bw;
      tile.reduction         = L_sync + L_partial_read + L_accumulate + L_partial_write;

      if (tile_m * tile_n <= 2048) tile.reduction *= 2.0;
    }

    return tile;
  };

  // Evaluate all tile types
  tile_epilogue_plan_t epilogue_interior;
  tile_epilogue_plan_t epilogue_n_edge;
  tile_epilogue_plan_t epilogue_m_edge;
  tile_epilogue_plan_t epilogue_corner;
  if (has_interior) epilogue_interior = compute_tile_epilogue(MT_M, MT_N, false);
  if (has_n_edge) epilogue_n_edge = compute_tile_epilogue(MT_M, n_remainder, false);
  if (has_m_edge) epilogue_m_edge = compute_tile_epilogue(m_remainder, MT_N, true);
  if (has_m_edge && has_n_edge)
    epilogue_corner = compute_tile_epilogue(m_remainder, n_remainder, true);

  // Aggregate the per-tile-type costs into a representative per-tile epilogue.
  //
  // The kernel launches grid_m x grid_n output tiles; at most one M-row and one
  // N-column are partial (edge/scalar) tiles.  How much those (often expensive,
  // scalar) edge tiles matter depends on HOW MANY tiles there are:
  //   * Few tiles (num_output_tiles <= N_CU): every tile runs concurrently in a
  //     single wave, so the slowest tile sets the wall-clock -> take the max.
  //   * Many tiles (> N_CU): tiles pipeline over timesteps and each CU drains a
  //     representative mix, so throughput is the tile-count-weighted average.
  // The edge tile-count fraction is 1/grid in that direction: a lone scalar
  // M-edge row among 1000 M-tiles is ~0.1% of the work, but one of two M-tiles
  // is 50%.  This is the key fix for skinny shapes: a single scalar edge tile
  // must not dominate a large grid (and must dominate a tiny one).
  const double f_m_edge =
      has_m_edge ? 1.0 / static_cast<double>(std::max<size_t>(grid_m, 1)) : 0.0;
  const double f_n_edge =
      has_n_edge ? 1.0 / static_cast<double>(std::max<size_t>(grid_n, 1)) : 0.0;
  const double w_interior = (1.0 - f_m_edge) * (1.0 - f_n_edge);
  const double w_m_edge   = f_m_edge * (1.0 - f_n_edge);
  const double w_n_edge   = (1.0 - f_m_edge) * f_n_edge;
  const double w_corner   = f_m_edge * f_n_edge;
  const double weighted_epilogue = w_interior * epilogue_interior.total() +
                                   w_m_edge * epilogue_m_edge.total() +
                                   w_n_edge * epilogue_n_edge.total() +
                                   w_corner * epilogue_corner.total();
  const double max_epilogue = std::max({epilogue_interior.total(), epilogue_n_edge.total(),
                                        epilogue_m_edge.total(), epilogue_corner.total()});

  const bool few_tiles = (num_output_tiles <= N_CU);
  double L_epilogue = few_tiles ? max_epilogue : weighted_epilogue;

  // selected_epilogue is only used for the debug term breakdown below.
  tile_epilogue_plan_t selected_epilogue = epilogue_interior;
  if (few_tiles) {
    for (const auto& c : {epilogue_interior, epilogue_m_edge, epilogue_n_edge, epilogue_corner})
      if (c.total() == max_epilogue) selected_epilogue = c;
  } else {
    double best_w = -1.0;
    const std::pair<const tile_epilogue_plan_t*, double> cands[4] = {
        {&epilogue_interior, w_interior}, {&epilogue_m_edge, w_m_edge},
        {&epilogue_n_edge, w_n_edge},     {&epilogue_corner, w_corner}};
    for (const auto& c : cands)
      if (c.second > best_w) { best_w = c.second; selected_epilogue = *c.first; }
  }

  // Fraction of the representative epilogue that is scalar-edge store work
  // (the M-edge and corner plans, scalar_path=true).  The scalar store path is a
  // serialized per-element predicated loop exposed only when store-bound;
  // reported here so compute_tile_latency can amplify it gated by store_exposure
  // rather than by occupancy.
  if (scalar_store_fraction != nullptr) {
    auto store_terms = [](const tile_epilogue_plan_t& p) {
      return p.bounds + p.store_issue + p.store_memory;
    };
    double scalar_store_cost = 0.0;
    if (few_tiles) {
      if (selected_epilogue.store.pattern == STORE_PATTERN_SCALAR_EDGE)
        scalar_store_cost = store_terms(selected_epilogue);
    } else {
      scalar_store_cost = w_m_edge * store_terms(epilogue_m_edge) +
                          w_corner * store_terms(epilogue_corner);
    }
    *scalar_store_fraction =
        (L_epilogue > 0.0) ? std::clamp(scalar_store_cost / L_epilogue, 0.0, 1.0) : 0.0;
  }

  if (debug) {
    OLOG_DEBUG("epi_wave_num: " << int(wave_num_epi));
    OLOG_DEBUG("epi_acc_elems_per_thread: " << acc_elems_per_thread);
    OLOG_DEBUG("epi_mi_reg_per_out: " << mi_reg_per_out);
    OLOG_DEBUG("epi_reads_per_wave: " << reads_per_wave);
    OLOG_DEBUG("L_epilogue_interior: " << epilogue_interior.total());
    OLOG_DEBUG("L_epilogue_n_edge: " << epilogue_n_edge.total());
    OLOG_DEBUG("L_epilogue_m_edge: " << epilogue_m_edge.total());
    OLOG_DEBUG("L_epilogue_corner: " << epilogue_corner.total());
    OLOG_DEBUG("epi_requested_svw: " << requested_svw);
    OLOG_DEBUG("epi_natural_svw: " << natural_svw);
    OLOG_DEBUG("epi_max_isa_svw: " << max_isa_store_elems);
    OLOG_DEBUG("epi_store_svw_meta: " << config.gwvw_d);
    OLOG_DEBUG("epi_source_swap: " << tparams(config).source_swap);
    OLOG_DEBUG("epi_wave_issue_parallelism: " << wave_issue_parallelism);
    OLOG_DEBUG("epi_wave_batches: " << wave_batches);
    OLOG_DEBUG("epi_store_pattern: " << selected_epilogue.store.pattern);
    OLOG_DEBUG("epi_store_insts: " << selected_epilogue.store.store_insts);
    OLOG_DEBUG("epi_store_issue_insts: " << selected_epilogue.store.issue_insts);
    OLOG_DEBUG("epi_store_sectors: " << selected_epilogue.store.sectors);
    OLOG_DEBUG("epi_store_useful_bytes: " << selected_epilogue.store.useful_bytes);
    OLOG_DEBUG("epi_store_sector_bytes: " << selected_epilogue.store.sector_bytes);
    OLOG_DEBUG("epi_store_sector_efficiency: " << selected_epilogue.sector_efficiency);
    OLOG_DEBUG("epi_active_waves: " << selected_epilogue.active_waves);
    OLOG_DEBUG("epi_L_acc_read: " << selected_epilogue.acc_read);
    OLOG_DEBUG("epi_L_bounds: " << selected_epilogue.bounds);
    OLOG_DEBUG("epi_L_store_issue: " << selected_epilogue.store_issue);
    OLOG_DEBUG("epi_L_store_memory: " << selected_epilogue.store_memory);
    OLOG_DEBUG("epi_L_reduce: " << selected_epilogue.reduction);
  }

  return L_epilogue;
}

// Apply D-store cache hint behavior to the HBM-baseline epilogue latency.
//
// compute_epilogue_latency() intentionally returns a store-bandwidth baseline.
// The cached-D L2 advantage depends on how exposed the epilogue is relative to
// the mainloop, so compute_tile_latency() supplies store_exposure =
// L_epilogue_hbm / (L_mainloop + L_epilogue_hbm).
static double apply_epilogue_store_cache_model(const problem_t& problem,
                                               const hardware_t& hardware,
                                               const config_t& config,
                                               const context_t& context,
                                               double L_epilogue_hbm,
                                               double store_exposure) {
  const bool debug = context.debug;

  double L_epilogue = L_epilogue_hbm;
  double store_rate_ratio = 0.0;
  double ntd_l2_help_factor = 0.0;

  // Per-wave M-store width (in DRAM sectors): the dominant NTD4 traffic driver,
  // counting the D output in 64 B sectors.
  const size_t MIWG_M_ntd = std::max<size_t>(tparams(config).wave_group_m, 1);
  const double per_wave_m_rows =
      static_cast<double>(config.mt.m) / static_cast<double>(MIWG_M_ntd);
  constexpr double sector_bytes = static_cast<double>(heuristic_defaults_t::DRAM_SECTOR_BYTES);
  const double per_wave_m_bytes = per_wave_m_rows * context.d_bytes;
  const double per_wave_m_lines =
      std::max(1.0, std::ceil(per_wave_m_bytes / sector_bytes));
  const double ntd4_traffic_factor =
      std::clamp(0.28 * per_wave_m_lines + 0.92, 1.0, 2.5);

  // Tiles/CU > 1 amplifies streaming-store contention.
  const size_t tiles_per_cu_signal = context.active_cus > 0
      ? std::max<size_t>(context.num_output_tiles
            / std::max<size_t>(context.active_cus, 1), 1)
      : 1;

  // Partial-M edge-tile signal.
  const size_t M_problem = problem.size.m;
  const size_t MT_M_pr   = std::max<size_t>(config.mt.m, 1);
  const bool partial_m = (M_problem % MT_M_pr) != 0;

  // Per-batch working set for L2-fit and cached-D thrash detection.
  const double a_total_bytes_pb = static_cast<double>(problem.size.m)
                                * static_cast<double>(problem.size.k)
                                * context.a_bytes;
  const double b_total_bytes_pb = static_cast<double>(problem.size.k)
                                * static_cast<double>(problem.size.n)
                                * context.b_bytes;
  const double d_total_bytes_pb = static_cast<double>(problem.size.m)
                                * static_cast<double>(problem.size.n)
                                * context.d_bytes;
  const double per_batch_ws = a_total_bytes_pb + b_total_bytes_pb + d_total_bytes_pb;
  const double l2_cap = static_cast<double>(hardware.L2_capacity);

  // How far the per-batch working set spills past L2, faded to [0,1] (1 = fits,
  // 0 = far larger).  Shared by both store paths.
  const double l2_overflow_fade = (l2_cap > 0.0 && per_batch_ws > l2_cap)
      ? std::max(0.0, 1.0 - (per_batch_ws / l2_cap - 1.0) / 6.0)
      : 1.0;
  // Cached path only fades its L2 benefit for deep K, where evicting A/B tiles
  // actually costs the mainloop.  The NTD penalty fade (d_l2_fit) is
  // K-independent: a huge output thrashes cached stores whether K is thin or deep.
  const double l2_fit_factor =
      (problem.size.k >= heuristic_defaults_t::L2_FIT_K_MIN) ? l2_overflow_fade : 1.0;
  const double d_l2_fit = l2_overflow_fade;

  if (config.cache_hints_d < 4) {
    // Cached-D path: L2 bandwidth advantage is useful only when store latency
    // is exposed on the tile critical path.
    store_rate_ratio = store_exposure;
    // Ramp the L2 store benefit in with store exposure, saturating at STORE_RATE_HIGH.
    ntd_l2_help_factor =
        std::clamp(store_rate_ratio / heuristic_defaults_t::STORE_RATE_HIGH, 0.0, 1.0);
    ntd_l2_help_factor *= l2_fit_factor;

    const double bw_speedup_max =
        hardware.mem3_perf_ratio > 0.0
            ? hardware.mem1_perf_ratio / hardware.mem3_perf_ratio
            : 1.0;
    const double bw_speedup = 1.0 + (bw_speedup_max - 1.0) * ntd_l2_help_factor;
    L_epilogue = L_epilogue_hbm / std::max(bw_speedup, 1.0);

    // Cached stores into a D >> L2 footprint pay write-allocate traffic and
    // can evict A/B tiles the mainloop still needs. This tax only matters
    // when stores are exposed and the working set is well beyond L2.
    if (l2_cap > 0.0 && per_batch_ws > 2.0 * l2_cap &&
        problem.size.k >= heuristic_defaults_t::L2_FIT_K_MIN && store_rate_ratio > 0.30) {
      const double overflow_ratio = per_batch_ws / l2_cap - 2.0;
      const double thrash_factor = std::clamp(overflow_ratio * 0.05, 0.0, 0.5);
      L_epilogue *= 1.0 + thrash_factor;
    }
  } else {
    // NTD=4 (streaming) path. Traffic penalties are relative to cached stores
    // landing in L2; when D is much larger than L2, cached stores would thrash
    // as well, so the streaming penalty fades for the regimes where streaming
    // measured best.
    const bool   d_is_16bit       = (context.d_bytes <= 2);
    const size_t m_tiles_ntd      = math::safe_ceil_div(
        problem.size.m, std::max<size_t>(config.mt.m, 1));
    const bool   single_partial_m = (m_tiles_ntd <= 1)
                                  && (problem.size.m < config.mt.m);
    // Fade the streaming penalty only for 16-bit output when D >> L2; extending
    // the fade to fp32/tf32 regressed (cached-D wins there), so keep the gate.
    const double ntd4_pen_gate =
        (d_is_16bit && !single_partial_m) ? d_l2_fit : 1.0;
    const double traf = 1.0 + (ntd4_traffic_factor - 1.0) * ntd4_pen_gate;
    L_epilogue = L_epilogue_hbm * traf;

    if (tiles_per_cu_signal >= 2) {
      L_epilogue *= 1.0 + 0.15 * ntd4_pen_gate;
    }

    if (partial_m) {
      const bool m_heavy_save = (MIWG_M_ntd >= 4) && (tiles_per_cu_signal <= 1);
      if (!m_heavy_save) {
        L_epilogue *= 1.0 + 0.18 * ntd4_pen_gate;
      }
    }
  }

  if (debug) {
    OLOG_DEBUG("per_wave_m_rows: " << per_wave_m_rows);
    OLOG_DEBUG("per_wave_m_lines: " << per_wave_m_lines);
    OLOG_DEBUG("ntd4_traffic_factor: " << ntd4_traffic_factor);
    OLOG_DEBUG("tiles_per_cu_signal: " << tiles_per_cu_signal);
    OLOG_DEBUG("partial_m: " << partial_m);
    OLOG_DEBUG("store_rate_ratio: " << store_rate_ratio);
    OLOG_DEBUG("d_l2_fit: " << d_l2_fit);
    OLOG_DEBUG("ntd_l2_help_factor: " << ntd_l2_help_factor);
    OLOG_DEBUG("L_epilogue: " << L_epilogue);
  }

  return L_epilogue;
}

// Compute the latency to compute a tile.
//
// Kernel structure (per WG):
//   Prologue  -- first-load stall (PGR prefetches)
//   MainLoop  -- L_main (steady-state) + L_ngll (drain) + L_nll (terminal)
//                + L_tail (partial K-tail iter) + K-loop penalties
//   Epilogue  -- per-tile output stores
//   Total     -- Prologue + MainLoop + Epilogue + tile_fixed_overhead
double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            const context_t& context) {
  assert(config.mt.m > 0 && config.mt.n > 0 && config.mt.k > 0);
  assert(config.mi.m > 0 && config.mi.n > 0 && config.mi.k > 0);
  assert(problem.size.m > 0 && problem.size.n > 0);
  assert(context.a_bytes > 0 && context.b_bytes > 0);
  assert(context.splitting_factor > 0);

  // ---------------------------------------------------------------------------
  // 0. Extract parameters and compute per-K-iter base costs
  // ---------------------------------------------------------------------------
  const size_t K    = problem.size.k;
  const size_t MT_K = config.mt.k;
  const int    a_bits = datatype_to_bits(problem.a_dtype);
  const int    b_bits = datatype_to_bits(problem.b_dtype);
  const long   pgr    = static_cast<long>(tparams(config).prefetch_global_read);

  const size_t splitting_factor = context.splitting_factor;
  const size_t k_per_split      = context.k_per_split;
  const double occupancy_factor = context.occupancy_factor;
  const auto&  heuristic        = context.heuristic;
  const bool   debug            = context.debug;

  // Per-K-iter cycle costs (simple MFMA-only compute model; bytes/BW memory).
  double L_compute = static_cast<double>(compute_mt_compute_latency(problem, hardware, config));
  double L_mem     = compute_memory_latency(problem, hardware, config, context);

  // ---------------------------------------------------------------------------
  // Per-wave aspect pressure.
  //
  // The MFMA-only L_compute above counts MFMAs and divides by parallel_mi_cu,
  // but ignores secondary effects of the wave/MIWT layout that gate effective
  // throughput:
  //   (a) occupancy must be high enough to hide exposed memory stalls
  //       (occupancy_score);
  //   (b) enough workgroups must be co-resident per CU to overlap work across
  //       workgroup boundaries  (wg_score).
  //
  // Each effect is a score in [0, 1]; their product is a throughput multiplier
  // applied to L_compute (the only term that captures MFMA throughput). The
  // multiplier (1 / score) is capped at 1.5x so that a
  // single mis-scored term cannot dominate the predicted latency.
  // ---------------------------------------------------------------------------
  const size_t MIWG_M_pw = std::max<size_t>(tparams(config).wave_group_m, 1);
  const size_t MIWG_N_pw = std::max<size_t>(tparams(config).wave_group_n, 1);
  const size_t waves_per_wg = MIWG_M_pw * MIWG_N_pw;

  // Occupancy.
  //
  // config.occupancy (Tensile CUOccupancy) is the resident workgroups per
  // CU: min over the LDS-, VGPR-, accVGPR-, and SGPR-limited occupancies,
  // each expressed in WGs/CU. Latency hiding, however, is driven by resident
  // waves per SIMD, so convert: waves/SIMD = WGs/CU * waves/WG / SIMD_per_CU.
  const double wgs_per_cu = static_cast<double>(std::max(config.occupancy, 1));
  const double waves_per_simd =
      wgs_per_cu * static_cast<double>(waves_per_wg) / static_cast<double>(hardware.simds_per_cu());
  const double occupancy_score = std::clamp(
      waves_per_simd / heuristic_defaults_t::TARGET_OCCUPANCY, 0.0, 1.0);

  // Occupancy only helps by hiding *exposed* memory stalls: the portion of
  // per-iter memory latency the prefetch pipeline cannot overlap behind
  // compute. For compute-bound tiles (L_compute >= L_mem) memory is fully
  // hidden, so the occupancy penalty must fade.
  const double exposed_mem_frac = std::clamp(
      (L_mem - L_compute) / std::max(L_mem, 1.0), 0.0, 1.0);
  const double occupancy_score_eff =
      1.0 - (1.0 - occupancy_score) * exposed_mem_frac;

  // Workgroup co-residency: how many workgroups fit on a CU to overlap work
  // across WG boundaries. config.occupancy is already resident WGs/CU, so it
  // is used directly, which (unlike a max-occupancy wave-slot ceiling) also
  // penalises register-starved kernels.
  const double wg_score = std::clamp(
      wgs_per_cu / heuristic_defaults_t::TARGET_WG_SLOTS_PER_CU, 0.0, 1.0);

  // Combined throughput score: occupancy * wg.
  double per_wave_score = occupancy_score_eff * wg_score;

  // Cap the multiplier. An uncapped product can fall to ~0.06 (single wave,
  // register spilling), implying a 16x compute inflation; real low-occupancy
  // kernels lose closer to 30-50%. Cap the penalty at 1.5x L_compute.
  constexpr double PER_WAVE_MIN_SCORE = 1.0 / 1.5;
  per_wave_score = std::max(per_wave_score, PER_WAVE_MIN_SCORE);

  L_compute /= per_wave_score;

  // XF32 / BF16-from-f32 conversion, charged per main-loop iter.
  // TODO: gfx90a also lacks native TF32 and should get CVT overhead, but
  // enabling it changes rankings — address in a separate PR.
  double L_cvt = 0.0;
  if (!hardware.has_native_TF32() &&
      hardware.arch != hardware_t::architecture_t::gfx90a) {
    if (problem.mi_dtype == data_type_t::XFloat32)
      L_cvt = compute_cvt_overhead(problem, hardware, config);
    else if (a_bits == 32 && b_bits == 32 &&
             problem.mi_dtype == data_type_t::BFloat16)
      L_cvt = compute_cvt_overhead_x1(problem, hardware, config);
  }

  // K-loop structure.  ISA splits K into floor(K/MT_K) full iters + a tail;
  // split-K then shards the full iters across WGs (ceiling-divided).  The
  // tail iter is charged to every WG: while StreamK only runs the tail on
  // the WG that completes a tile, attempts to model that have been worse
  // overall than the uniform-tail charge, so we keep the simple shape.
  const long total_full_iters = static_cast<long>(K / MT_K);
  const long k_iters = (splitting_factor > 1)
      ? static_cast<long>(math::safe_ceil_div(
            static_cast<size_t>(total_full_iters), splitting_factor))
      : total_full_iters;
  const long num_main_iters = std::max<long>(k_iters - pgr, 0);
  const long num_ngll_iters = (k_iters > 0 && pgr > 1)
      ? std::min<long>(pgr - 1, k_iters - 1) : 0;
  const size_t tail_k = K % MT_K;

  // Tail-iter shape, shared by L_tail and the PGR fill term below.  Both are
  // tail_k / MT_K; computing once keeps them in sync (zero when K % MT_K == 0).
  const double tail_fraction  = static_cast<double>(tail_k) / static_cast<double>(MT_K);
  const size_t tail_sub_iters = math::safe_ceil_div(tail_k, config.mi.k);

  // K-loop issue-efficiency scale.
  const double eff_scale =
      (splitting_factor > 4) ? 1.0 : heuristic.main_loop_efficiency;

  // Edge / spatial waste penalty.  utilization is the ratio
  // of useful problem volume to launched volume; the kernel still pays for
  // the launched volume, so per-iter latency scales by 1/utilization.
  const double utilization = calculate_work_utilization(problem, config);
  const double effective_tile_penalty =
      (utilization > 1e-9) ? (1.0 / utilization) : 1.0;

  // ---------------------------------------------------------------------------
  // 1. Prologue (PGR first-load stall)
  // ---------------------------------------------------------------------------
  // The prologue issues PGR global loads but the CU stalls at vmcnt only for
  // the oldest — roughly 1 x L_mem.  When k_iters == 0 the pipeline never
  // runs, so the prologue is skipped.
  //
  // No ETP scaling here: memory traffic is bounded to active lanes by
  // Tensile's buffer descriptors, so OOB lanes don't generate DRAM traffic
  // and shouldn't inflate L_mem.  The compute-only ETP is applied below.
  const double L_prologue = (k_iters > 0)
      ? (L_mem * occupancy_factor)
      : 0.0;

  // ---------------------------------------------------------------------------
  // 2. MainLoop = L_main + L_ngll + L_nll + L_tail + L_pgr_stall + bookkeeping
  //
  // ETP (= 1/utilization) is applied **only to compute terms** here.  MFMA
  // and LDS-read pipes spend cycles on every wave-wide lane regardless of
  // whether the output is in-bounds, so wasted lanes inflate compute wall
  // time.  Memory traffic is bounded to active lanes (Tensile bounded
  // buffer_load), so OOB lanes contribute zero DRAM bytes and shouldn't
  // be ETP-scaled.
  // ---------------------------------------------------------------------------

  // L_main — steady-state main iters with load + compute overlap.  Falls
  // back to additive (non-overlapped) mode when pgr <= 1.
  const double L_mem_stream         = L_mem * static_cast<double>(num_main_iters);
  const double L_compute_stream     = L_compute * static_cast<double>(num_main_iters);
  const double L_compute_stream_eff = L_compute_stream * effective_tile_penalty;
  double L_main = (pgr <= 1)
      ? (L_mem_stream + L_compute_stream_eff)
      : std::max(L_mem_stream, L_compute_stream_eff);
  L_main *= eff_scale;
  L_main += L_cvt * static_cast<double>(num_main_iters) * effective_tile_penalty;

  // L_ngll — drain iters (compute while consuming in-flight prefetches).
  const double L_ngll = static_cast<double>(num_ngll_iters) * L_compute
                      * eff_scale * effective_tile_penalty;

  // L_nll — terminal pure-compute iter from already-resident LDS.
  const double L_nll = (k_iters > 0)
      ? (L_compute * eff_scale * effective_tile_penalty)
      : 0.0;

  // Cost of one full DepthU (MT_K-wide) K-iteration.  Used by the residual tail
  // window below and the oversize/one-iter DepthU penalties further down.
  const double L_main_per_iter = (pgr <= 1)
      ? (L_mem + L_compute * effective_tile_penalty) * eff_scale
            + L_cvt * effective_tile_penalty
      : std::max(L_mem, L_compute * effective_tile_penalty) * eff_scale
            + L_cvt * effective_tile_penalty;

  // L_tail — the residual partial-K window (K % MT_K > 0).  Charges the work
  // executed over tail_k slices (memory share uses actual bytes so 128B
  // alignment doesn't make a half-size tail look free; memory is NOT ETP-scaled
  // since loads are bounded, while compute and per-sub-iter bookkeeping are),
  // plus, when this is the ONLY iteration (k_iters == 0), the unamortised
  // remainder of the MT_K-wide DepthU window (see end of block).
  double L_tail = 0.0;
  if (tail_k > 0) {
    constexpr size_t alignment_bytes = heuristic_defaults_t::DRAM_SECTOR_BYTES;
    // Sector-rounded A+B bytes for a K-window of `k` slices.  Rounding is
    // applied to whichever axis is contiguous in DRAM. The tail and full iters share this helper, so
    // the rounding is consistent between numerator and denominator.
    auto iter_bytes = [&](size_t k) -> double {
      return compute_operand_window_bytes(problem, config, context, k, alignment_bytes);
    };
    const double tail_mem_fraction = iter_bytes(tail_k) / iter_bytes(MT_K);

    const double L_tail_mem      = tail_mem_fraction * L_mem;
    const double L_tail_compute  = tail_fraction * L_compute * effective_tile_penalty;
    // Per-sub-iter bookkeeping (barrier / scalar branch / masked ds_read)
    // is wave-wide and does not have OOB-lane waste, so it is not
    // ETP-scaled.  When the kernel is heavily compute-bound (high ETP from
    // very low M*N utilization), the per-sub-iter branch/barrier pipelines
    // behind the dominant MFMA chain — empirically the overhead acts more
    // like a small constant than a per-sub-iter charge.  Fade overhead to
    // ~20% of nominal once ETP exceeds ~10.
    const double tail_overhead_scale = (effective_tile_penalty > heuristic_defaults_t::TAIL_OVERHEAD_COMPUTE_BOUND_ETP)
        ? heuristic_defaults_t::TAIL_OVERHEAD_COMPUTE_BOUND_SCALE
        : 1.0;
    const double L_tail_overhead = heuristic.tail_loop_overhead
                                 * static_cast<double>(tail_sub_iters)
                                 * tail_overhead_scale;
    L_tail = (L_tail_mem + L_tail_compute + L_tail_overhead) * eff_scale;

    // (The unamortised remainder of an oversized/partial DepthU window is now
    // charged uniformly by depth_waste_ratio in the L_du_waste block below,
    // covering both k_iters == 0 and the k_iters >= 1 tail consistently.)
  }

  // L_pgr_stall — PGR pipeline fill/drain exposure: PGR only pays off after
  // ~pgr+1 main iters; below that, expose a linearly fading number of L_mem
  // chunks scaled by the memory share of a K-iter so compute-heavy tiles
  // don't pay a full drain charge that HW hides behind NGLL/NLL.  Cap the
  // unamortized count at `k_iters`: with only k_iters K-iters of work, the
  // kernel cannot stall on more than k_iters prefetch fills (k_iters=1 with
  // PGR=2 issues exactly one prefetch, not three).  No ETP: pure memory.
  const double pgr_unamortized_iters =
      std::min(static_cast<double>(k_iters),
               static_cast<double>(pgr + 1) - static_cast<double>(num_main_iters));
  const double pgr_mem_exposure = L_mem / (L_mem + L_compute);
  const double L_pgr_stall = (pgr > 1 && pgr_unamortized_iters > 0.0)
      ? (pgr_unamortized_iters + ((num_main_iters == 0) ? tail_fraction : 0.0))
            * L_mem * pgr_mem_exposure * eff_scale
      : 0.0;

  // Per-K-iteration loop bookkeeping (branch / counter / barrier).  PGR3+
  // kernels keep more global reads in flight, and in hardware much of this
  // bookkeeping overlaps the prefetch pipeline.  Keep the full charge for
  // PGR1/2, but expose only a fraction for deeper PGR.
  const double pgr_loop_overlap =
      (pgr >= 3) ? (1.0 / static_cast<double>(pgr - 1)) : 1.0;
  const double L_loop_overhead =
      heuristic_defaults_t::K_ITER_LOOP_OVERHEAD * static_cast<double>(k_iters) * pgr_loop_overlap;

  // Sub-cache-line DepthU narrow-load penalty.  When K is the coalesced load
  // axis (transA=T for A, transB=N for B), the coalesced load spans MT_K*bpe
  // bytes; below the 128 B cache line the wave issues under-filled loads every
  // K-iter.  Modelled per-K-iter from MT_K and the count of K-coalesced operands
  // only (not byte volume / MT_M / MT_N, which would bias NN toward small MT_N).
  // TN weights both operands, NN only B, NT neither; scaled by k_iters, additive.
  constexpr double phys_cl = static_cast<double>(heuristic_defaults_t::CACHE_LINE_BYTES);
  const double a_bytes_du  = static_cast<double>(a_bits) / 8.0;
  const double b_bytes_du  = static_cast<double>(b_bits) / 8.0;
  const bool a_k_coalesced = (problem.a_transpose == transpose_t::T);   // TN / TT
  const bool b_k_coalesced = (problem.b_transpose == transpose_t::N);   // NN / TN
  const double mt_k_dd     = static_cast<double>(std::max<size_t>(MT_K, 1));
  const double a_underfill = a_k_coalesced
      ? std::max(0.0, phys_cl / std::max(mt_k_dd * a_bytes_du, 1.0) - 1.0) : 0.0;
  const double b_underfill = b_k_coalesced
      ? std::max(0.0, phys_cl / std::max(mt_k_dd * b_bytes_du, 1.0) - 1.0) : 0.0;
  const double narrow_load_factor = a_underfill + b_underfill;
  const double L_narrow_load = narrow_load_factor * static_cast<double>(k_iters)
                             * heuristic_defaults_t::NARROW_LOAD_ITER_PENALTY;

  // DepthU load waste: an MT_K that does not divide K rounds the K-loop up to a
  // full final window, loading ceil(K/MT_K)*MT_K deep but using only K.  The
  // extra (loaded - K) depth is wasted, measured identically whether it shows up
  // as an oversized single window (MT_K > K) or a partial tail iter, so equally
  // wasteful MT_K tie rather than biasing toward the deeper tile.  Zero when MT_K
  // divides K.
  const double K_problem = static_cast<double>(K);
  const double loaded_depth = (K_problem > 0.0)
      ? std::ceil(K_problem / mt_k_dd) * mt_k_dd : 0.0;
  // MT_K > K (a window wider than the whole problem) is strictly worse than a
  // tail, so weight its waste more heavily to keep a smaller tail-leaving MT_K
  // preferred over an oversized one when K has no clean divisor.
  const double oversize_weight = (mt_k_dd > K_problem)
      ? heuristic_defaults_t::OVERSIZE_WASTE_WEIGHT : 1.0;
  const double depth_waste_ratio = (K_problem > 0.0)
      ? (loaded_depth - K_problem) / K_problem * oversize_weight : 0.0;
  // Gate the single-iter penalty (below) to K large enough that a smaller MT_K
  // would give a real multi-iter K-loop; for tiny K, MT_K==K is the natural pick
  // and penalising it would flip the model to a costlier MT_K>K.
  const bool exact_one_iter_large_k = K >= heuristic_defaults_t::EXACT_ONE_ITER_K_MIN;

  // Batched few-iteration fill/drain penalty.  A batched GEMM pays the PGR
  // fill/drain once per tile; a short K-loop (deep MT_K vs K) amortizes it over
  // little work, so HW prefers shallower MT_K / more iters for batched micro-GEMMs.
  // Penalize by how far the K-loop length (main iters + partial tail) falls short
  // of a target.  Fires even with a tail present (unlike no_steady_state below),
  // and gated to batch > 1 so streaming batch==1 large-N shapes are untouched.
  const double k_loop_len = static_cast<double>(k_iters) + (tail_k > 0 ? 1.0 : 0.0);
  const double batched_fill_ratio =
      (problem.batch > 1 && k_loop_len >= 1.0)
          ? std::max(0.0, heuristic_defaults_t::BATCHED_FILL_ITER_TARGET - k_loop_len)
                * heuristic_defaults_t::BATCHED_FILL_PENALTY
          : 0.0;

  // Unbatched single fill+drain regime (batch==1, num_main_iters==0): the whole
  // K-loop lives in the PGR fill/drain window, so the kernel never reaches steady
  // state and the MT_K-deep fill is exposed unamortized.  Base tax for a genuine
  // single iter; deeper tiles are charged by DepthU fill excess so the model
  // can't escape a penalized k_iters==2 tile by hopping to a deeper one.
  const bool no_steady_state =
      (problem.batch == 1 && num_main_iters == 0 && k_iters >= 1 && tail_k == 0
       && exact_one_iter_large_k);
  const double fill_depth_excess =
      std::max(0.0, mt_k_dd * a_bytes_du / phys_cl - 1.0);
  const double no_steady_base = (k_iters == 1) ? 2.0 : 0.0;
  const double no_steady_state_ratio = no_steady_state
      ? no_steady_base + fill_depth_excess * heuristic_defaults_t::UNAMORTIZED_FILL_PENALTY
      : 0.0;

  // M-edge waste: when the whole M fits one tile (grid_M == 1, M <= MT_M), an
  // oversized MT_M computes MT_M rows but uses only M, with no other full M-tile
  // to amortize the waste.  ETP under-charges this single-tile case, so a modest
  // penalty tips it back to an M-clean tile.  Restricted to grid_M == 1: for
  // grid_M >= 2 ETP already accounts for the one partial tile, so charging here
  // would double-count.  Zero when MT_M divides M (or M >= MT_M).
  const double m_dd = static_cast<double>(std::max<size_t>(config.mt.m, 1));
  const double M_problem = static_cast<double>(problem.size.m);
  const double m_edge_ratio = (M_problem > 0.0 && M_problem <= m_dd)
      ? (m_dd - M_problem) / M_problem : 0.0;

  const double L_du_waste =
      (depth_waste_ratio * heuristic_defaults_t::TAIL_WASTE_PENALTY
       + m_edge_ratio * heuristic_defaults_t::M_EDGE_PENALTY
       + no_steady_state_ratio + batched_fill_ratio)
      * (L_main_per_iter + heuristic_defaults_t::K_ITER_LOOP_OVERHEAD);

  // MainLoop subtotal.
  const double L_mainloop =
      L_main + L_ngll + L_nll + L_tail + L_pgr_stall + L_loop_overhead + L_du_waste
      + L_narrow_load;

  // ---------------------------------------------------------------------------
  // 3. Epilogue (per-tile store; compute is already covered by NLL)
  // ---------------------------------------------------------------------------
  // Below the occupancy that saturates the store/return pipeline, un-overlapped
  // epilogue store latency is exposed and scales ~1/occupancy.  config.occupancy
  // is the register-limited resident waves/CU (CUOccupancy); saturation point is
  // calibrated to gfx950.  Only multiplies the epilogue term, so it is negligible
  // for mainloop-bound tiles and only bites store-bound shapes.
  const double cu_occ_epi = static_cast<double>(std::max(config.occupancy, 1));
  const double epi_occ_exposure = std::max(1.0, heuristic_defaults_t::EPILOGUE_OCC_SATURATION / cu_occ_epi);
  double scalar_store_fraction = 0.0;
  const double L_epilogue_hbm =
      compute_epilogue_latency(problem, hardware, config, context, &scalar_store_fraction)
      * occupancy_factor * epi_occ_exposure;
  const double store_exposure_den = L_mainloop + L_epilogue_hbm;
  const double store_exposure =
      (store_exposure_den > 0.0) ? L_epilogue_hbm / store_exposure_den : 0.0;
  double L_epilogue = apply_epilogue_store_cache_model(problem,
                                                       hardware,
                                                       config,
                                                       context,
                                                       L_epilogue_hbm,
                                                       store_exposure);

  // Scalar/edge stores are a serialized per-element predicated loop that is only
  // exposed when the kernel is store-bound; a long mainloop hides them.  Gate on
  // store-boundedness (not occupancy, which stays healthy even at CUOccupancy=1)
  // by amplifying the scalar store portion in proportion to store_exposure.
  const double scalar_store_mult =
      1.0 + (heuristic_defaults_t::SCALAR_STORE_EXPOSED_PENALTY - 1.0) * scalar_store_fraction * store_exposure;
  L_epilogue *= scalar_store_mult;

  // ---------------------------------------------------------------------------
  // 5. Total tile latency
  // ---------------------------------------------------------------------------
  const double L_tile_fixed = heuristic.tile_fixed_overhead;
  // Discount for hand-optimized kernels that beat the analytical model (set by
  // apply_tf32_heuristics); 1.0 otherwise.  The hand-tuned speedup only holds in
  // a clean XCD-aligned split-K regime, so suppress the discount when the split
  // factor and XCD count don't divide each other (uneven split-K across XCDs).
  double weight_tile_total = heuristic.weight_tile_total;
  if (weight_tile_total < 1.0) {  // a hand-opt discount is in effect
    const size_t sf  = std::max<size_t>(context.splitting_factor, 1);
    const size_t xcd = std::max<size_t>(hardware.NUM_XCD, 1);
    const bool xcd_aligned = (sf % xcd == 0) || (xcd % sf == 0);
    if (!xcd_aligned) weight_tile_total = 1.0;
  }
  const double L_tile_total =
      (L_prologue + L_mainloop + L_epilogue + L_tile_fixed) * weight_tile_total;

  if (debug) {
    OLOG_DEBUG("per_wave waves_per_wg: " << waves_per_wg);
    OLOG_DEBUG("per_wave wgs_per_cu: " << wgs_per_cu);
    OLOG_DEBUG("per_wave waves_per_simd: " << waves_per_simd);
    OLOG_DEBUG("per_wave occupancy_score: " << occupancy_score);
    OLOG_DEBUG("per_wave exposed_mem_frac: " << exposed_mem_frac);
    OLOG_DEBUG("per_wave occupancy_score_eff: " << occupancy_score_eff);
    OLOG_DEBUG("per_wave wg_score: " << wg_score);
    OLOG_DEBUG("per_wave_score (combined): " << per_wave_score);

    OLOG_DEBUG("utilization: " << utilization);
    OLOG_DEBUG("effective_tile_penalty: " << effective_tile_penalty);
    OLOG_DEBUG("L_mem: " << L_mem);
    OLOG_DEBUG("L_compute: " << L_compute);
    OLOG_DEBUG("L_cvt: " << L_cvt);
    OLOG_DEBUG("k_per_split: " << k_per_split);
    OLOG_DEBUG("k_iters: " << int(k_iters));
    OLOG_DEBUG("num_main_iters: " << int(num_main_iters));
    OLOG_DEBUG("num_ngll_iters: " << int(num_ngll_iters));
    OLOG_DEBUG("L_mem_stream: " << L_mem_stream);
    OLOG_DEBUG("narrow_load_factor: " << narrow_load_factor);
    OLOG_DEBUG("L_narrow_load: " << L_narrow_load);
    OLOG_DEBUG("L_compute_stream: " << L_compute_stream);

    OLOG_DEBUG("L_prologue: " << L_prologue);
    OLOG_DEBUG("L_main: " << L_main);
    OLOG_DEBUG("L_ngll: " << L_ngll);
    OLOG_DEBUG("L_nll: " << L_nll);
    OLOG_DEBUG("tail_k: " << tail_k);
    OLOG_DEBUG("L_tail: " << L_tail);
    OLOG_DEBUG("pgr_unamortized_iters: " << pgr_unamortized_iters);
    OLOG_DEBUG("pgr_mem_exposure: " << pgr_mem_exposure);
    OLOG_DEBUG("L_pgr_stall: " << L_pgr_stall);
    OLOG_DEBUG("L_loop_overhead: " << L_loop_overhead);
    OLOG_DEBUG("depth_waste_ratio: " << depth_waste_ratio);
    OLOG_DEBUG("no_steady_state_ratio: " << no_steady_state_ratio);
    OLOG_DEBUG("batched_fill_ratio: " << batched_fill_ratio);
    OLOG_DEBUG("fill_depth_excess: " << fill_depth_excess);
    OLOG_DEBUG("L_du_waste: " << L_du_waste);
    OLOG_DEBUG("L_mainloop: " << L_mainloop);
    
    OLOG_DEBUG("epi_cu_occupancy: " << cu_occ_epi);
    OLOG_DEBUG("epi_occ_exposure: " << epi_occ_exposure);
    OLOG_DEBUG("store_exposure: " << store_exposure);
    OLOG_DEBUG("scalar_store_fraction: " << scalar_store_fraction);
    OLOG_DEBUG("scalar_store_mult: " << scalar_store_mult);
    OLOG_DEBUG("L_epilogue_hbm: " << L_epilogue_hbm);
    OLOG_DEBUG("L_epilogue: " << L_epilogue);
    OLOG_DEBUG("L_tile_fixed: " << L_tile_fixed);
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
  const size_t active_wgs = std::min(total_wgs, context.n_cu);
  const size_t timesteps  = math::safe_ceil_div(total_wgs, context.n_cu);

  // Bandwidth based on occupancy of the reduction kernel.  Workspace
  // partials are served at MALL bandwidth (mem2) regardless of the
  // main-kernel cache_hints_d setting.
  const double bw_per_cu = compute_mem_bw_from_occupancy(hardware, active_wgs);
  double read_bw         = std::max(hardware.mem2_perf_ratio * bw_per_cu, 1e-12);

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
                             const config_t& config) {
  assert(config.is_valid());

  // ANALYTICAL_GEMM_PICK: force a specific MT size for solution selection.
  {
    const auto& pick = runtime_options::get().gemm_pick;
    if (pick.m > 0 && (config.mt.m != pick.m || config.mt.n != pick.n || config.mt.k != pick.k)) {
      return std::numeric_limits<double>::max();
    }
  }


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

    constexpr size_t cache_line_bits = heuristic_defaults_t::CACHE_LINE_BYTES * 8;  // 128 B in bits
    size_t K_mod_128bytes    = K * a_bits % cache_line_bits;
    size_t MT_K_mod_128bytes = MT_K * a_bits % cache_line_bits;
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

  // 3) Compute latency for all scheduling rounds.
  // `num_timesteps` is the number of waves of workgroups that must pass
  // through the available CUs.  Occupancy can hide stalls inside a tile, but
  // it does not reduce the number of scheduling rounds, so it belongs in the
  // per-tile latency model rather than as a divisor here.
  double total_latency             = L_timestep * context.num_timesteps;

  //  4) Kernel launch overhead
  total_latency += heuristic_defaults_t::KERNEL_LAUNCH_OVERHEAD;

  //  5) Add parallel reduction kernel cost (separate kernel launch, 0 if not parallel)
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
  prob_info.swizzleTensorA = tparams(config).swizzle_a;
  prob_info.swizzleTensorB = tparams(config).swizzle_b;
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
  size_mapping.depthU = (tparams(config).depth_u > 0) ? tparams(config).depth_u : config.mt.k;

  size_mapping.globalSplitU       = tparams(config).global_split_u;
  size_mapping.globalAccumulation = tparams(config).global_accumulation;
  size_mapping.LocalSplitU        = tparams(config).local_split_u;

  size_mapping.DirectToVgprA = tparams(config).direct_to_vgpr_a;
  size_mapping.DirectToVgprB = tparams(config).direct_to_vgpr_b;
  size_mapping.DirectToLdsA  = tparams(config).direct_to_lds_a;
  size_mapping.DirectToLdsB  = tparams(config).direct_to_lds_b;

  size_mapping.NumLoadsCoalescedA = tparams(config).num_loads_coalesced_a;
  size_mapping.NumLoadsCoalescedB = tparams(config).num_loads_coalesced_b;
  size_mapping.VectorWidthA       = config.vector_width_a;
  size_mapping.VectorWidthB       = config.vector_width_b;

  size_mapping.waveNum      = tparams(config).wave_num;
  size_mapping.waveGroup[0] = static_cast<int>(tparams(config).wave_group_m);
  size_mapping.waveGroup[1] = static_cast<int>(tparams(config).wave_group_n);

  size_mapping.workGroupMapping         = config.workgroup_mapping;
  size_mapping.workGroupMappingXCC      = tparams(config).workgroup_mapping_xcc;
  size_mapping.workGroupMappingXCCGroup = tparams(config).workgroup_mapping_xcc_group;
  size_mapping.globalSplitUCoalesced    = tparams(config).global_split_u_coalesced;
  size_mapping.globalSplitUWorkGroupMappingRoundRobin =
      tparams(config).global_split_u_wgm_round_robin;

  size_mapping.CUOccupancy            = config.occupancy;
  size_mapping.PrefetchGlobalRead     = tparams(config).prefetch_global_read;
  size_mapping.MathClocksUnrolledLoop = tparams(config).math_clocks_unrolled_loop;

  // Set problem, solution, and hardware in Formocast
  formocast.setProblem(prob_info);
  formocast.setSolution(size_mapping);
  formocast.setHardware(hardware.arch);

  // Get predicted performance
  Formocast::PredictedPerformance perf = formocast.predictedPerformance();

  // Return latency in microseconds
  return perf.microSeconds;
}

/* ---------------------------------------------------------------------------------------- */
/* Deprecated cache-hit helpers                                                             */
/* ---------------------------------------------------------------------------------------- */
// Kept for tests, Python bindings, and external callers. The active latency
// model uses estimate_cache_hit_rates(), which returns per-operand L1/L2/MALL rates.
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

// Deprecated compatibility wrapper; see estimate_cache_hit_rates().
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

// Deprecated compatibility helper; the active model uses per-operand cache rates.
double compute_l2_hit_rate_global(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  size_t l2_capacity_bytes) {
  if (l2_capacity_bytes == 0) throw std::runtime_error("L2 Capacity is zero");

  const size_t grid_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  const size_t grid_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  if (grid_m == 0 || grid_n == 0)
    throw std::runtime_error("estimate_l2_hit grid dimensions can not be zero");

  const double a_bytes = data_type_to_bytes(problem.a_dtype);
  const double b_bytes = data_type_to_bytes(problem.b_dtype);

  const double a_working_set           = static_cast<double>(grid_m * config.mt.mk()) * a_bytes;
  const double b_working_set           = static_cast<double>(grid_n * config.mt.nk()) * b_bytes;
  const double total_working_set_bytes = a_working_set + b_working_set;

  if (total_working_set_bytes > l2_capacity_bytes) return 0.1;

  const double total_A_reads = static_cast<double>(grid_m * grid_n * config.mt.mk());
  const double total_B_reads = static_cast<double>(grid_m * grid_n * config.mt.nk());
  const double uncached_A_reads = static_cast<double>(grid_m * config.mt.mk());
  const double uncached_B_reads = static_cast<double>(grid_n * config.mt.nk());

  const double total_reads = total_A_reads + total_B_reads;
  if (total_reads == 0) return 1.0;

  const double cached_reads =
      (total_A_reads - uncached_A_reads) + (total_B_reads - uncached_B_reads);

  return cached_reads / total_reads;
}

}  // namespace gemm
}  // namespace origami
