// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/streamk.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// The kernel a config describes
// ---------------------------------------------------------------------------

// Safe Tensile-params accessor: config's params if present, else a default.
// The model must not throw for configs without a Tensile backend (tests,
// Python bindings, direct callers); only PredictionLibrary populates it.
const tensile_params_t& tparams(const config_t& config) {
  static const tensile_params_t kDefaultTensile{};
  return config.has_tensile_params() ? config.tensile() : kDefaultTensile;
}

// TemporalHint of an operand's global reads as the kernel emits it (gfx1250 TH field).  With
// TDMFuse 0 one tensor_load_to_lds per wave moves both A and B, and TensileLite keeps the hint only
// when TemporalHintA == TemporalHintB (Components/NonTemporal.py tdmCacheModifiers), else the
// default TH_RT.  A hint the config does not carry comes from the cache_hints code (>= 4: NT).
int load_temporal_hint(const config_t& config, bool b_operand) {
  const tensile_params_t& tp = tparams(config);
  const int ta = tp.temporal_hint_a >= 0 ? tp.temporal_hint_a : (config.cache_hints_a >= 4 ? 1 : 0);
  const int tb = tp.temporal_hint_b >= 0 ? tp.temporal_hint_b : (config.cache_hints_b >= 4 ? 1 : 0);
  if ((tp.tdm_inst & 3) == 3 && tp.tdm_fuse == 0) return ta == tb ? ta : 0;
  return b_operand ? tb : ta;
}

// TemporalHint of the D stores (and of parallel-reduction partials, which take the D store code).
int store_temporal_hint(const config_t& config) {
  const int td = tparams(config).temporal_hint_d;
  return td >= 0 ? td : (config.cache_hints_d >= 4 ? 1 : 0);
}

// gfx1250 has no MALL, so its L2 takes the near-cache half of a hint: NT, LU, NT_RT and NT_HT loads
// keep their lines at the low retention priority, RT, HT and RT_NT at the regular one.
bool l2_low_priority_load(int th) { return th == 1 || th == 3 || th == 4 || th == 6; }

// Threads per wavefront of the kernel: the WavefrontSize it was built with when the config
// carries it, otherwise the architecture's default.
size_t kernel_wavefront_size(const hardware_t& hardware, const config_t& config) {
  const int built = tparams(config).wavefront_size;
  return (built > 0) ? static_cast<size_t>(built) : hardware.wavefront_size();
}

// Clocks one vector-ALU instruction of the kernel's wavefront occupies its SIMD.
size_t kernel_valu_cycles(const hardware_t& hardware, const config_t& config) {
  return std::max<size_t>(kernel_wavefront_size(hardware, config) / hardware.simd_lanes(), 1);
}

// The part of one DepthU iteration a single wave works on.  Each of the
// wave_group_m x wave_group_n waves of a LocalSplitU group owns
// MT_M/wave_group_m x MT_N/wave_group_n of the tile, and each of the
// local_split_u groups takes MT_K/local_split_u of the iteration's K.
dim3_t compute_wave_share(const config_t& config) {
  const auto& tp = tparams(config);
  return {config.mt.m / static_cast<size_t>(std::max(tp.wave_group_m, 1)),
          config.mt.n / static_cast<size_t>(std::max(tp.wave_group_n, 1)),
          config.mt.k / static_cast<size_t>(std::max(tp.local_split_u, 1))};
}

// WGs resident on one CU at a time: the kernel's occupancy caps them, and so does the
// grid when it hands an active CU fewer.  The dynamic stream-K sub-path launches its
// own grid, min(tiles, CUs x min(occupancy, DYNAMIC_MAX_WGS_PER_CU)), not num_wgs.
// W4A16 logic entries declare CUOccupancy per WGP (hardware_t::OCCUPANCY_SIMDS SIMDs), so a CU
// holds its share of it, which can be a fraction (one 16-wave workgroup per WGP is half a
// workgroup per CU).
double compute_resident_wgs(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            const context_t& context) {
  double occupancy = static_cast<double>(config.occupancy);
  if (is_w4a16(problem, hardware))
    occupancy *= static_cast<double>(hardware.simds_per_cu()) /
                 static_cast<double>(hardware_t::OCCUPANCY_SIMDS);
  if (context.tile_schedule == hybrid_mode_t::dynamic)
    return std::min(occupancy,
                    static_cast<double>(heuristic_defaults_t::STREAMK_DYNAMIC_MAX_WGS_PER_CU));
  const double wgs_per_cu = static_cast<double>(math::safe_ceil_div(
      std::max<size_t>(context.num_wgs, 1), std::max<size_t>(context.active_cus, 1)));
  return std::min(wgs_per_cu, occupancy);
}

// Whether the kernel's cluster peers share A and B tiles by multicast.  A config without
// TensileLite parameters and with a cluster is taken to.
bool is_multicast_cluster(const config_t& config) {
  return config.cluster_dim.m * config.cluster_dim.n > 1 &&
         (!config.has_tensile_params() || tparams(config).multicast);
}

// Bytes of A and B one DepthU iteration of a tile loads.
double operand_window_bytes(const config_t& config, const context_t& context) {
  return (static_cast<double>(config.mt.m) * context.a_bytes +
          static_cast<double>(config.mt.n) * context.b_bytes) *
         static_cast<double>(config.mt.k);
}

// ---------------------------------------------------------------------------
// Utilization and capacity
// ---------------------------------------------------------------------------

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

// calculate_work_utilization for the compute terms, which already charge whole MFMAs:
// an M or N narrower than one MFMA wastes nothing beyond what every kernel using that
// MFMA pays, so it counts as one MFMA wide.
double calculate_compute_utilization(const problem_t& problem, const config_t& config) {
  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  if (MT_M <= 0 || MT_N <= 0) return 1.0;

  auto launched = [](size_t x, size_t tile) {
    return static_cast<double>(math::safe_ceil_div(x, tile)) * static_cast<double>(tile);
  };
  const double launched_M      = launched(problem.size.m, MT_M);
  const double launched_N      = launched(problem.size.n, MT_N);
  const double launched_volume = launched_M * launched_N * launched(problem.size.k, MT_K);
  if (launched_volume < 1.0) return 1.0;

  const double useful_M =
      std::min(launched_M, static_cast<double>(std::max(problem.size.m, config.mi.m)));
  const double useful_N =
      std::min(launched_N, static_cast<double>(std::max(problem.size.n, config.mi.n)));
  return useful_M * useful_N * static_cast<double>(problem.size.k) / launched_volume;
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

// Round element count up to a multiple of `transaction_bytes`, since the
// coalesced load width is not always a full 128-byte L1 line.
size_t round_elements_to_NB(size_t elements, size_t element_size_bits, size_t transaction_bytes) {
  if (element_size_bits == 0 || transaction_bytes == 0) return elements;
  auto round_up_mul             = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
  const size_t transaction_bits = transaction_bytes * 8u;
  const size_t g                = std::gcd(element_size_bits, transaction_bits);
  const size_t E_block          = transaction_bits / g;
  return round_up_mul(elements, E_block);
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

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

context_t::context_t(const problem_t& problem, const hardware_t& hardware, const config_t& config)
    : context_t(problem, hardware, config, get_heuristic_params(problem, hardware, config)) {}

context_t::context_t(const problem_t& problem,
                     const hardware_t& hardware,
                     const config_t& config,
                     const heuristic_params_t& heuristic_params) {
  // Effective usable CU count. Decided once here and consumed across the model
  // (launch params, occupancy, cache/epilogue/reduction estimates). A non-zero
  // problem.num_cus caps the count; 0 falls back to the full hardware count.
  n_cu = resolve_num_cus(problem.num_cus, hardware.N_CU);

  const size_t M     = problem.size.m;
  const size_t N     = problem.size.n;
  const size_t batch = problem.batch;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  // Heuristic parameters
  heuristic = heuristic_params;

  // Element sizes and operand layout
  a_bytes      = data_type_to_bytes(problem.a_dtype);
  b_bytes      = data_type_to_bytes(problem.b_dtype);
  d_bytes      = data_type_to_bytes(problem.d_dtype);
  a_bits       = datatype_to_bits(problem.a_dtype);
  b_bits       = datatype_to_bits(problem.b_dtype);
  a_transposed = problem.a_transpose == transpose_t::T;
  b_transposed = problem.b_transpose == transpose_t::T;

  // The kernel's wavefront
  wavefront_size = kernel_wavefront_size(hardware, config);
  valu_cycles    = kernel_valu_cycles(hardware, config);

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
  k_iters            = math::safe_ceil_div(k_per_split, config.mt.k);

  // Per-operand DRAM traffic: pure function of the values set above; computed
  // once here and shared by the cache-hit and memory-latency models.
  const size_t dram_block_bytes =
      for_family(hardware, [&](auto family) { return family.dram_block_bytes(hardware); });
  traffic = compute_operand_traffic(problem, config, *this, dram_block_bytes);

  // Hardware-derived values
  active_cus           = cus;
  mem_bw_limited       = compute_mem_bw_from_occupancy(hardware, active_cus);
  write_mem_bw_limited = compute_mem_bw_from_occupancy(hardware, num_output_tiles);

  // Tile-derived values
  tile_elements     = MT_M * MT_N;
  output_tile_bytes = tile_elements * d_bytes;
  wave_group        = {static_cast<size_t>(std::max(tparams(config).wave_group_m, 1)),
                       static_cast<size_t>(std::max(tparams(config).wave_group_n, 1)),
                       static_cast<size_t>(std::max(tparams(config).local_split_u, 1))};

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

    OLOG_DEBUG("======== Origami Debug Info ========");  // This signature indicates the start of
                                                         // the debug information.
    OLOG_DEBUG("ProblemSize (MxNxBxK): " << int(M) << "x" << int(N) << "x" << int(batch) << "x"
                                         << int(K));
    OLOG_DEBUG("transpose: " << (a_transposed ? "T" : "N") << (b_transposed ? "T" : "N"));
    OLOG_DEBUG("MacroTile: " << int(MT_M) << "x" << int(MT_N) << "x" << int(MT_K));
    OLOG_DEBUG("MatrixInstruction: " << int(MI_M) << "x" << int(MI_N) << "x" << int(MI_K));
    OLOG_DEBUG("ClusterDimX: " << int(config.cluster_dim.m));
    OLOG_DEBUG("ClusterDimY: " << int(config.cluster_dim.n));
    OLOG_DEBUG("ClusterDimZ: " << int(config.cluster_dim.k));
    OLOG_DEBUG("ElementSizeA (bits): " << int(a_bits));
    OLOG_DEBUG("ElementSizeB (bits): " << int(b_bits));
    OLOG_DEBUG("CacheHintsA: " << int(config.cache_hints_a));
    OLOG_DEBUG("CacheHintsB: " << int(config.cache_hints_b));
    OLOG_DEBUG("CacheHintsD: " << int(config.cache_hints_d));
    OLOG_DEBUG("DirectToLdsA: " << int(tparams(config).direct_to_lds_a));
    OLOG_DEBUG("DirectToLdsB: " << int(tparams(config).direct_to_lds_b));
    OLOG_DEBUG("DirectToVgprA: " << int(tparams(config).direct_to_vgpr_a));
    OLOG_DEBUG("DirectToVgprB: " << int(tparams(config).direct_to_vgpr_b));
    OLOG_DEBUG("VectorWidth: " << config.vector_width_a << "x" << config.vector_width_b);
    OLOG_DEBUG("UnrollMajorLDS: " << tparams(config).unroll_major_lds_a << "x"
                                  << tparams(config).unroll_major_lds_b);
    OLOG_DEBUG("LDSTransposeRead: " << int(tparams(config).lds_transpose_read_a) << "x"
                                    << int(tparams(config).lds_transpose_read_b));
    OLOG_DEBUG("CUOccupancy(cfg): " << int(config.occupancy));
    OLOG_DEBUG("LocalSplitU: " << int(tparams(config).local_split_u));
    OLOG_DEBUG("OneLDSBuffer: " << int(tparams(config).one_lds_buffer));
    OLOG_DEBUG("PrefetchGlobalRead: " << int(tparams(config).prefetch_global_read));
    OLOG_DEBUG("Wave: " << int(tparams(config).wave_group_m) << "x"
                        << int(tparams(config).wave_group_n));
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

    OLOG_DEBUG("CHUNKxXCCxWGM: " << int(wgm.wgmxccchunk) << "x" << int(wgm.wgmxcc) << "x"
                                 << int(wgm.wgm));
  }
}

bool context_t::is_valid() const {
  return grid_m > 0 && grid_n > 0 && num_output_tiles > 0 && splitting_factor > 0 && num_wgs > 0 &&
         num_timesteps > 0 && active_cus > 0 && mem_bw_limited > 0.0 && tile_elements > 0 &&
         output_tile_bytes > 0 && a_bytes > 0 && b_bytes > 0 && d_bytes > 0;
}

// ---------------------------------------------------------------------------
// Shared stages
// ---------------------------------------------------------------------------

memory_terms_t base_model::tile_memory(const tile_args_t& args) {
  return memory(args.problem, args.hardware, args.config, args.context);
}

// Workgroup co-residency.  The MFMA-only compute time spreads a tile's MFMAs over
// every SIMD of the CU and ignores how many WGs overlap across WG boundaries.
//  - Only SIMDs that resident waves occupy run MFMAs; a WG brings
//    wave_group_m x wave_group_n x local_split_u waves.
//  - wg_score in [0,1] scales L_compute for cross-WG overlap; the 1/score
//    multiplier is capped at 1.5x so it can't dominate the prediction.
// Both count the WGs actually resident, which the kernel's occupancy caps, so
// register-starved kernels are still penalised.
void base_model::compute_iteration(const tile_args_t& args, double L_mi, tile_terms_t& t) {
  constexpr double PER_WAVE_MIN_SCORE = 1.0 / 1.5;
  const double simds_per_cu           = static_cast<double>(args.hardware.simds_per_cu());

  t.per_wave_score = std::max(t.wg_score, PER_WAVE_MIN_SCORE);
  t.L_compute      = L_mi / t.per_wave_score * (simds_per_cu / t.simds_busy);
  t.L_compute += t.L_local_read;
}

void base_model::load_stall(const tile_args_t&, tile_terms_t&) {}

// Steady-state main iters with load + compute overlap.  Falls back to additive
// (non-overlapped) mode when pgr <= 1.
double base_model::main_iterations(const tile_args_t& args, tile_terms_t& t) {
  const double L_compute_stream_eff = t.L_compute_stream * t.compute_tile_penalty;
  return (args.pgr <= 1) ? (t.L_mem_stream + L_compute_stream_eff)
                         : std::max(t.L_mem_stream, L_compute_stream_eff);
}

// The residual partial-K window (K % MT_K > 0).  Memory share uses actual bytes (so 128B
// alignment can't make a half-size tail look free) and is not ETP-scaled; compute is.
double base_model::tail_iteration(const tile_args_t& args, const tile_terms_t& t) {
  constexpr size_t alignment_bytes = heuristic_defaults_t::DRAM_SECTOR_BYTES;
  // Sector-rounded A+B bytes for a K-window of `k` slices, rounded on the
  // DRAM-contiguous axis; shared by the tail and full iters.
  auto iter_bytes = [&](size_t k) -> double {
    return compute_operand_window_bytes(
        args.problem, args.config, args.context, k, alignment_bytes);
  };
  const double tail_mem_fraction = iter_bytes(t.tail_k) / iter_bytes(args.config.mt.k);

  const double L_tail_mem     = tail_mem_fraction * t.memory.L_mem;
  const double L_tail_compute = t.tail_fraction * t.L_compute * t.compute_tile_penalty;
  return (L_tail_mem + L_tail_compute) * t.eff_scale;
}

// Per-K-iter loop bookkeeping (branch / counter / barrier).  Deeper PGR keeps
// more reads in flight and overlaps most of this, so expose only a fraction.
double base_model::loop_overhead(const tile_args_t& args, const tile_terms_t& t) {
  const double pgr_loop_overlap = (args.pgr >= 3) ? (1.0 / static_cast<double>(args.pgr - 1)) : 1.0;
  return heuristic_defaults_t::K_ITER_LOOP_OVERHEAD * static_cast<double>(t.k_iters) *
         pgr_loop_overlap;
}

double base_model::cluster_sync(const tile_args_t&, const tile_terms_t&) { return 0.0; }

double base_model::depth_waste(const tile_args_t& args, const tile_terms_t& t) {
  const problem_t& problem = args.problem;
  const config_t& config   = args.config;

  // Cost of one full DepthU (MT_K-wide) K-iteration.  The charge is calibrated against the
  // full edge penalty, sub-MFMA waste included.
  const double L_main_per_iter =
      (args.pgr <= 1)
          ? (t.memory.L_mem + t.L_compute * t.effective_tile_penalty) * t.eff_scale +
                t.L_cvt * t.effective_tile_penalty
          : std::max(t.memory.L_mem, t.L_compute * t.effective_tile_penalty) * t.eff_scale +
                t.L_cvt * t.effective_tile_penalty;

  // M-edge waste: when the whole M fits one tile (grid_M == 1, M <= MT_M), an
  // oversized MT_M computes rows it doesn't use with nothing to amortize against,
  // and ETP under-charges this single-tile case.  Restricted to grid_M == 1 (for
  // grid_M >= 2 ETP already covers the partial tile).  Zero when MT_M divides M.
  //
  // GEMV in M (M == 1) with a deep enough K-loop: the m_edge waste is unavoidable
  // (every candidate uses MT_M >= MI_M > 1) and identical across tiles, so applying
  // it only mis-ranks by per-iter cost -- penalizing the deep-K tiles these
  // memory-bound reductions actually want.
  const double m_dd      = static_cast<double>(config.mt.m);
  const double M_problem = static_cast<double>(problem.size.m);
  const bool submi_gemv =
      (problem.size.m == 1 && problem.size.k >= heuristic_defaults_t::SUBMI_GEMV_K_MIN);
  const double m_edge_ratio =
      (!submi_gemv && M_problem > 0.0 && M_problem <= m_dd) ? (m_dd - M_problem) / M_problem : 0.0;

  return (t.depth_waste_ratio * heuristic_defaults_t::TAIL_WASTE_PENALTY +
          m_edge_ratio * heuristic_defaults_t::M_EDGE_PENALTY + t.batched_fill_ratio) *
         (L_main_per_iter + heuristic_defaults_t::K_ITER_LOOP_OVERHEAD);
}

// Below the occupancy that saturates the store/return pipeline, un-overlapped
// epilogue store latency is exposed and scales ~1/(resident WGs per CU).
double base_model::epilogue_exposure(const hardware_t&, double wgs_per_cu) {
  return std::max(1.0, heuristic_defaults_t::EPILOGUE_OCC_SATURATION / wgs_per_cu);
}

double base_model::finish_tile(const tile_args_t&, const tile_terms_t& t) { return t.L_tile_total; }

// ---------------------------------------------------------------------------
// Tile model
// ---------------------------------------------------------------------------

namespace {

// The tile model, written once for every family.
//
// Kernel structure (per WG):
//   Prologue  -- first-load stall (PGR prefetches)
//   MainLoop  -- L_main (steady-state) + L_ngll (drain) + L_nll (terminal)
//                + L_tail (partial K-tail iter) + K-loop penalties
//   Epilogue  -- per-tile output stores
//   Total     -- Prologue + MainLoop + Epilogue
template <class Family>
tile_terms_t tile_terms(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config,
                        const context_t& context) {
  assert(config.mt.m > 0 && config.mt.n > 0 && config.mt.k > 0);
  assert(config.mi.m > 0 && config.mi.n > 0 && config.mi.k > 0);
  assert(problem.size.m > 0 && problem.size.n > 0);
  assert(context.a_bytes > 0 && context.b_bytes > 0);
  assert(context.splitting_factor > 0);

  const tensile_params_t& tp = tparams(config);
  const tile_args_t args{
      problem, hardware, config, context, tp, static_cast<long>(tp.prefetch_global_read)};
  const long pgr = args.pgr;

  const size_t K    = problem.size.k;
  const size_t MT_K = config.mt.k;
  const int a_bits  = context.a_bits;
  const int b_bits  = context.b_bits;

  const size_t splitting_factor = context.splitting_factor;
  const auto& heuristic         = context.heuristic;

  tile_terms_t t;
  t.k_per_split = context.k_per_split;

  // ---------------------------------------------------------------------------
  // 0. Per-K-iter base costs
  // ---------------------------------------------------------------------------

  // Per-K-iter cycle costs (simple MFMA-only compute model; bytes/BW memory).
  const bool w4a16 = is_w4a16(problem, hardware);
  double L_mi      = static_cast<double>(compute_mt_compute_latency(problem, hardware, config));
  if (w4a16) L_mi += compute_w4a16_dequant_latency(problem, hardware, config);
  t.memory = Family::tile_memory(args);

  // LDS read time that narrow M/N-contiguous reads (VectorWidth * bpe below a
  // bank) add over the full-width floor.  The LDS pipe feeds the MFMAs, so only
  // the part that outlasts both the floor and the MFMA work is exposed.
  t.L_lds_read   = compute_local_read_latency(problem, hardware, config);
  t.L_lds_floor  = compute_local_read_latency(problem, hardware, config, true);
  t.L_local_read = std::max(0.0, t.L_lds_read - std::max(L_mi, t.L_lds_floor));

  // Workgroups resident on a CU, and the SIMDs their waves occupy.  W4A16 counts the
  // co-residency target in waves (the workgroup slots on every SIMD): its one-WG/CU kernels are
  // the 16-wave tiles.
  const dim3_t& waves       = context.wave_group;
  const double simds_per_cu = static_cast<double>(hardware.simds_per_cu());
  const double waves_per_wg = static_cast<double>(waves.m * waves.n * waves.k);
  t.wgs_per_cu              = compute_resident_wgs(problem, hardware, config, context);
  t.wg_score =
      w4a16 ? std::clamp(t.wgs_per_cu * waves_per_wg /
                             (heuristic_defaults_t::TARGET_WG_SLOTS_PER_CU * simds_per_cu),
                         0.0,
                         1.0)
            : std::clamp(t.wgs_per_cu / heuristic_defaults_t::TARGET_WG_SLOTS_PER_CU, 0.0, 1.0);
  t.simds_busy = std::min(simds_per_cu, waves_per_wg * t.wgs_per_cu);

  Family::compute_iteration(args, L_mi, t);

  // XF32 / BF16-from-f32 conversion, charged per main-loop iter.
  if (Family::charges_conversion(hardware)) {
    if (problem.mi_dtype == data_type_t::XFloat32)
      t.L_cvt = compute_cvt_overhead(problem, hardware, config);
    else if (a_bits == 32 && b_bits == 32 && problem.mi_dtype == data_type_t::BFloat16)
      t.L_cvt = compute_cvt_overhead_x1(problem, hardware, config);
  }

  // K-loop structure.  ISA splits K into floor(K/MT_K) full iters + a tail;
  // split-K shards the full iters across WGs.  The tail iter is charged to
  // every WG (a uniform charge that models better than StreamK's per-tile tail).
  const long total_full_iters = static_cast<long>(K / MT_K);
  t.k_iters                   = (splitting_factor > 1) ? static_cast<long>(math::safe_ceil_div(
                                           static_cast<size_t>(total_full_iters), splitting_factor))
                                                       : total_full_iters;
  t.num_main_iters            = std::max<long>(t.k_iters - pgr, 0);
  t.num_ngll_iters = (t.k_iters > 0 && pgr > 1) ? std::min<long>(pgr - 1, t.k_iters - 1) : 0;
  t.tail_k         = K % MT_K;

  // Tail-iter shape, shared by L_tail and the PGR fill term below.  Both are
  // tail_k / MT_K; computing once keeps them in sync (zero when K % MT_K == 0).
  t.tail_fraction = static_cast<double>(t.tail_k) / static_cast<double>(MT_K);

  // K-loop issue-efficiency scale.
  t.eff_scale = (splitting_factor > 4) ? 1.0 : heuristic.main_loop_efficiency;

  // Edge / spatial waste penalty.  utilization is the ratio
  // of useful problem volume to launched volume; the kernel still pays for
  // the launched volume, so per-iter latency scales by 1/utilization.
  t.utilization = calculate_work_utilization(problem, config);
  t.effective_tile_penalty =
      (Family::charges_edge_waste && t.utilization > 1e-9) ? (1.0 / t.utilization) : 1.0;
  // The compute terms' penalty leaves out sub-MFMA waste, which L_compute already pays.
  const double compute_utilization = calculate_compute_utilization(problem, config);
  t.compute_tile_penalty           = (Family::charges_edge_waste && compute_utilization > 1e-9)
                                         ? (1.0 / compute_utilization)
                                         : 1.0;

  const double L_mem      = t.memory.L_mem;
  const double L_compute  = t.L_compute;
  const double main_iters = static_cast<double>(t.num_main_iters);

  // ---------------------------------------------------------------------------
  // 1. Prologue (PGR first-load stall)
  // ---------------------------------------------------------------------------
  // The prologue stalls at vmcnt only on the oldest PGR load (~1 x L_mem); skipped when
  // k_iters == 0.  No ETP: loads are bounded to active lanes, so OOB lanes generate no DRAM
  // traffic (compute-only ETP is applied below).
  Family::load_stall(args, t);
  t.L_prologue = (Family::charges_prologue && t.k_iters > 0) ? L_mem : 0.0;

  // ---------------------------------------------------------------------------
  // 2. MainLoop = L_main + L_ngll + L_nll + L_tail + L_pgr_stall + bookkeeping
  //
  // ETP (= 1/utilization) applies only to compute terms: MFMA/LDS pipes spend
  // cycles on every wave-wide lane, so OOB lanes inflate compute time.  Memory
  // is bounded to active lanes, so its bytes are not ETP-scaled.
  // ---------------------------------------------------------------------------

  // L_main — steady-state main iters with load + compute overlap.
  t.L_mem_stream     = L_mem * main_iters;
  t.L_compute_stream = L_compute * main_iters;
  t.L_main           = Family::main_iterations(args, t);
  t.L_main *= t.eff_scale;
  t.L_main += t.L_cvt * main_iters * t.compute_tile_penalty;

  // L_ngll — drain iters (compute while consuming in-flight prefetches).
  t.L_ngll =
      static_cast<double>(t.num_ngll_iters) * L_compute * t.eff_scale * t.compute_tile_penalty;

  // L_nll — terminal pure-compute iter from already-resident LDS.
  t.L_nll = (t.k_iters > 0) ? (L_compute * t.eff_scale * t.compute_tile_penalty) : 0.0;

  // L_tail — the residual partial-K window (K % MT_K > 0).  (The unamortised remainder of an
  // oversized/partial DepthU window is charged uniformly by depth_waste_ratio below, covering
  // both k_iters == 0 and the k_iters >= 1 tail consistently.)
  if (t.tail_k > 0) t.L_tail = Family::tail_iteration(args, t);

  // L_pgr_stall — PGR fill/drain exposure: PGR only pays off after ~pgr+1 main
  // iters; below that, expose a fading count of L_mem chunks scaled by a K-iter's
  // memory share.  Capped at k_iters (can't stall on more fills than iters of
  // work).  No ETP: pure memory.
  t.pgr_unamortized_iters =
      std::min(static_cast<double>(t.k_iters), static_cast<double>(pgr + 1) - main_iters);
  t.pgr_mem_exposure = L_mem / (L_mem + L_compute);
  t.L_pgr_stall =
      (pgr > 1 && t.pgr_unamortized_iters > 0.0)
          ? (t.pgr_unamortized_iters + ((t.num_main_iters == 0) ? t.tail_fraction : 0.0)) * L_mem *
                t.pgr_mem_exposure * t.eff_scale
          : 0.0;

  t.L_loop_overhead = Family::loop_overhead(args, t);
  t.L_cluster_sync  = Family::cluster_sync(args, t);

  // With one LDS buffer the next iteration's tile can only be written once every wave has read
  // the current one, so the writes fit only between the MFMAs of the last K-step.  When writing
  // the tile at LDS bandwidth outlasts that K-step, the writes and the barriers on either side
  // are exposed in every iteration that writes LDS, unless another WG on the CU fills them.
  if (tp.one_lds_buffer) {
    const dim3_t wave       = compute_wave_share(config);
    const double kstep_mfma = (static_cast<double>(wave.m) / static_cast<double>(config.mi.m)) *
                              (static_cast<double>(wave.n) / static_cast<double>(config.mi.n)) *
                              static_cast<double>(hardware.get_mi_latency(
                                  config.mi.m, config.mi.n, config.mi.k, problem.mi_dtype));
    const double tile_write =
        (static_cast<double>(config.mt.m) * data_type_to_bytes(lds_dtype_a(problem, hardware)) +
         static_cast<double>(config.mt.n) * context.b_bytes) *
        static_cast<double>(MT_K) / static_cast<double>(heuristic_defaults_t::LDS_XFER_BYTES);
    if (tile_write > kstep_mfma && t.wgs_per_cu <= 1.0)
      t.L_single_lds_buffer = (tile_write + 2.0 * heuristic_defaults_t::L_BARRIER) *
                              static_cast<double>(t.num_main_iters + t.num_ngll_iters);
  }

  // DepthU load waste: an MT_K that doesn't divide K loads ceil(K/MT_K)*MT_K deep
  // but uses only K; the extra depth is wasted, measured the same whether it's an
  // oversized window (MT_K > K) or a partial tail.  Zero when MT_K divides K.
  const double mt_k_dd      = static_cast<double>(MT_K);
  const double K_problem    = static_cast<double>(K);
  const double loaded_depth = (K_problem > 0.0) ? std::ceil(K_problem / mt_k_dd) * mt_k_dd : 0.0;
  // MT_K > K (a window wider than the whole problem) is strictly worse than a
  // tail, so weight its waste more heavily to keep a smaller tail-leaving MT_K
  // preferred over an oversized one when K has no clean divisor.
  const double oversize_weight =
      (mt_k_dd > K_problem) ? heuristic_defaults_t::OVERSIZE_WASTE_WEIGHT : 1.0;
  // Capped at one iteration: the unused depth is under one MT_K-deep iteration and
  // the charge is denominated in K-iterations.  Only binds when MT_K > K, where
  // normalising by K rather than by the loaded depth diverges as K shrinks.
  t.depth_waste_ratio =
      (K_problem > 0.0) ? std::min((loaded_depth - K_problem) / K_problem * oversize_weight, 1.0)
                        : 0.0;

  // Batched few-iteration fill/drain penalty.  A batched GEMM pays PGR fill/drain
  // once per tile; a short K-loop amortizes it poorly, so HW prefers shallower
  // MT_K.  Penalize by how far the K-loop length falls short of a target; gated
  // to batch > 1 so streaming batch==1 large-N shapes are untouched.
  const double k_loop_len = static_cast<double>(t.k_iters) + (t.tail_k > 0 ? 1.0 : 0.0);
  t.batched_fill_ratio =
      (problem.batch > 1 && k_loop_len >= 1.0)
          ? std::max(0.0, heuristic_defaults_t::BATCHED_FILL_ITER_TARGET - k_loop_len) *
                heuristic_defaults_t::BATCHED_FILL_PENALTY
          : 0.0;

  t.L_du_waste = Family::depth_waste(args, t);

  // MainLoop subtotal.
  t.L_mainloop = t.L_main + t.L_ngll + t.L_nll + t.L_tail + t.L_pgr_stall + t.L_loop_overhead +
                 t.L_single_lds_buffer + t.L_du_waste + t.L_cluster_sync;

  // ---------------------------------------------------------------------------
  // 3. Epilogue (per-tile store; compute is already covered by NLL)
  // ---------------------------------------------------------------------------
  // The occupancy exposure multiplies only the epilogue term, so it bites store-bound shapes
  // but not mainloop-bound ones.
  t.epi_cu_occupancy              = t.wgs_per_cu;
  t.epi_occ_exposure              = Family::epilogue_exposure(hardware, t.epi_cu_occupancy);
  const epilogue_terms_t epilogue = Family::epilogue(problem, hardware, config, context);
  t.scalar_store_fraction         = epilogue.scalar_store_fraction;
  t.L_epilogue                    = epilogue.L_epilogue * t.epi_occ_exposure;
  const double store_exposure_den = t.L_mainloop + t.L_epilogue;
  t.store_exposure = (store_exposure_den > 0.0) ? t.L_epilogue / store_exposure_den : 0.0;

  // Scalar/edge stores (a serialized per-element predicated loop) are exposed
  // only when store-bound; amplify them in proportion to store_exposure rather
  // than occupancy (which stays healthy even at CUOccupancy=1).
  t.scalar_store_mult = 1.0 + (heuristic_defaults_t::SCALAR_STORE_EXPOSED_PENALTY - 1.0) *
                                  t.scalar_store_fraction * t.store_exposure;
  t.L_epilogue *= t.scalar_store_mult;

  // ---------------------------------------------------------------------------
  // 4. Total tile latency
  // ---------------------------------------------------------------------------
  // Discount for hand-optimized kernels that beat the model (set by
  // apply_tf32_heuristics); 1.0 otherwise.  The speedup only holds for
  // XCD-aligned split-K, so suppress it when sf and NUM_XCD don't divide.
  t.weight_tile_total = heuristic.weight_tile_total;
  if (t.weight_tile_total < 1.0) {  // a hand-opt discount is in effect
    const size_t sf        = context.splitting_factor;
    const size_t xcd       = hardware.NUM_XCD;
    const bool xcd_aligned = (sf % xcd == 0) || (xcd % sf == 0);
    if (!xcd_aligned) t.weight_tile_total = 1.0;
  }
  // LocalSplitU LDS reduction, charged once per tile (0 when lsu <= 1).
  t.L_lsu_reduce =
      (context.wave_group.k <= 1) ? 0.0 : Family::lsu_reduction(problem, hardware, config, context);
  t.L_tile_total =
      (t.L_prologue + t.L_mainloop + t.L_epilogue + t.L_lsu_reduce) * t.weight_tile_total;
  t.L_tile_total = Family::finish_tile(args, t);

  return t;
}

// Prints the terms to the debug log, one "name: value" line each.  The log's CSV has one column
// per name, so the names and their order are part of the log format.
void log_tile_terms(const tile_terms_t& t) {
#define ORIGAMI_FIELD(type, name, description) OLOG_DEBUG(#name ": " << t.name);
#define ORIGAMI_FIELD_AS(type, name, log_name, description) \
  if (log_name[0] != '\0') OLOG_DEBUG(log_name << ": " << t.name);
#define ORIGAMI_LOGGED(log_name, expression) OLOG_DEBUG(log_name << ": " << expression);
  ORIGAMI_GEMM_TILE_TERMS(ORIGAMI_FIELD, ORIGAMI_FIELD_AS, ORIGAMI_LOGGED)
#undef ORIGAMI_FIELD
#undef ORIGAMI_FIELD_AS
#undef ORIGAMI_LOGGED
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

tile_terms_t compute_tile_terms(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const context_t& context) {
  const tile_terms_t terms = for_family(hardware, [&](auto family) {
    return tile_terms<decltype(family)>(problem, hardware, config, context);
  });
  if (context.debug) log_tile_terms(terms);
  return terms;
}

double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            const context_t& context) {
  return compute_tile_terms(problem, hardware, config, context).L_tile_total;
}

}  // namespace gemm
}  // namespace origami
