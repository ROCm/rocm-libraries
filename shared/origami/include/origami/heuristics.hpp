/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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

#include <functional>
#include <optional>
#include <string_view>
#include <unordered_map>

#include "origami/hardware.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

namespace origami {
/**
 * @brief Default values for heuristic parameters.
 * Centralized location for all default constants.
 */
struct heuristic_defaults_t {
  // Latency Component Weights
  static constexpr double WEIGHT_MEM_L2        = 1.0;
  static constexpr double WEIGHT_MEM_MALL      = 1.0;
  static constexpr double WEIGHT_MEM_DRAM      = 1.0;
  static constexpr double WEIGHT_COMPUTE       = 1.0;
  static constexpr double WEIGHT_MEMORY        = 1.0;
  static constexpr double WEIGHT_WG_SETUP      = 1.0;
  static constexpr double WEIGHT_PROLOGUE      = 1.0;
  static constexpr double WEIGHT_EPILOGUE      = 1.0;
  static constexpr double WEIGHT_LOOP_OVERHEAD = 100.0;
  static constexpr double WEIGHT_TILE_TOTAL    = 1.0;

  // Empirical Constants
  static constexpr double MAIN_MEMORY_LOAD_LATENCY         = 200.0;
  static constexpr double OCCUPANCY_DECAY_BASE             = 0.95;
  static constexpr double MALL_DEPTH_SQ                    = 2.0;
  static constexpr double MALL_COLD_FLOOR                  = 0.85;
  static constexpr double L2_DEPTH_SQ                      = 4.0;
  static constexpr double L2_COLD_FLOOR                    = 0.75;
  static constexpr double L2_POLLUTION_PENALTY             = 0.7;
  static constexpr double L2_AMP_CEILING_BATCHED           = 0.9;
  static constexpr double L2_AMP_CEILING_K_SPLIT           = 0.4;
  static constexpr double L2_AMP_CEILING_SKINNY            = 0.6;
  static constexpr double L2_DEPTH_PENALTY                 = 0.9;
  static constexpr double L1_HIT_RATE_CEILING_SKINNY       = 0.7;
  static constexpr double EPILOGUE_CYCLES_PER_ACC_READ     = 1.0;
  static constexpr double EPILOGUE_ACC_READ_PARALLELISM    = 0.9;
  // Each scalar-store wave-iter executes a short pipelined sequence of
  // 4-5 instructions: v_cmp_lt bounds check, s_and_saveexec mask, the
  // buffer_store itself, and s_mov_b64 exec restore (plus sometimes a
  // v_cndmask for the bias path).  At 1 cycle/issue throughput these
  // amortize to ~5 cycles per scalar wave-iter on gfx950.  Previously
  // reduced to 1.0 under-predicted edge-tile epilogues for large
  // scalar-path tiles (e.g. M=160 MT_M>=192 on large-N bf16 problems),
  // which combined with the K-alignment change to pick MI32x32 NTD4
  // kernels that are ~2x slower in HW.
  static constexpr double EPILOGUE_CYCLES_PER_BOUNDS_CHECK = 5.0;
  static constexpr double EPILOGUE_SCALAR_STORE_PENALTY    = 1.1;
  static constexpr size_t EPILOGUE_THREADS_PER_WAVE        = 64;
  static constexpr size_t EPILOGUE_BYTES_PER_VECTORIZED_STORE = 16;  // buffer_store_dwordx4
  static constexpr size_t EPILOGUE_CACHE_LINE_BYTES         = 128;
  static constexpr size_t EPILOGUE_WORKSPACE_BYTES_PER_ELEM = 4;
  static constexpr double EPILOGUE_SALU_OVERHEAD            = 35.0;
  static constexpr double EPILOGUE_L_BARRIER                = 100.0;
  static constexpr double EPILOGUE_L_SMEM = 1900.0;  // s_load_dword(glc) cross-XCD flag poll
  static constexpr double EPILOGUE_K_PADDING_PENALTY     = 50000.0;
  static constexpr size_t POSTGSU_COMPUTE_BYTES          = 4;  // workspace partials stored as f32
  static constexpr double POSTGSU_KERNEL_LAUNCH_OVERHEAD = 8000.0;
  static constexpr size_t POSTGSU_THREADS_PER_WG         = 256;
  static constexpr size_t POSTGSU_WAVEFRONT_SIZE         = 64;

  // Main Loop Efficiency
  static constexpr double MAIN_LOOP_EFFICIENCY = 1.0;

  // TF32 Emulation Constants
  static constexpr double TF32_ARITH_INTENSITY_THRESHOLD = 1000.0;

  // --- PGR (PrefetchGlobalRead) ---
  // Prologue = (setup + pgr) * L_mem.  Setup captures WG init overhead.
  static constexpr double PROLOGUE_SETUP_FRACTION  = 0.0;
  // Main loop: two overlapping streams —
  //   memory  stream = L_mem * (num_iter - pgr)
  //   compute stream = L_compute * num_iter
  //   duration = max(memory, compute)   (additive for PGR=1)

  // --- LSU (LocalSplitU) ---
  // Intra-WG LDS reduction overhead after the K-loop; scales as lsu * log2(lsu).
  static constexpr double LSU_REDUCTION_OVERHEAD = 1.0;   // cycles per element-group per log2(lsu) pass

  // --- NTD (NonTemporalD) ---
  // NTD=4 + K-split: partials bypass cache, so reduction reads hit DRAM.
  // Penalty per split factor (cycles).  0 = no penalty (non-split case).
  static constexpr double NTD_KSPLIT_PENALTY = 5000.0;

  // Cached-D (cache_hints_d < 4) L2 pollution of the input working set.
  // Fires only when ALL THREE conditions hold:
  //   (A) D output overflows L2 capacity, AND
  //   (B) per-tile input working set (k_iters x Ld_CU_bytes) exceeds the
  //       CU's fair L2 share, AND
  //   (C) the global working set (A + B + D) fits in MALL (last-level
  //       cache) so that cached D writes actually evict something that
  //       would otherwise be served from MALL.  When (A+B+D) >> MALL,
  //       inputs are streaming from HBM regardless and NTD0 vs NTD4
  //       behave identically at steady state -- no penalty applies.
  // Physically captures:
  //   * Small K / tiny tiles -> input WS fits in L2 -> cached D writes
  //     coexist with A+B, no eviction, no penalty.
  //   * Medium footprint (A+B+D <= MALL) with large K -> input loops
  //     thrash L2 and reuse through MALL; cached D evicts MALL-cached
  //     inputs -> penalty applies.
  //   * Large footprint (A+B+D >> MALL) -> MALL already cold for inputs;
  //     cached D neither hurts nor helps -> penalty attenuated to zero.
  // Multiplier on L_mem_dram:
  //   (1 + D_POLLUTION_PENALTY * d_overflow * input_overflow * mall_fit),
  // where mall_fit fades linearly from 1.0 (at WS = MALL) to 0.0 past
  // the fade point (default 1.5*MALL, controlled by
  // MALL_OVERFLOW_FADE_SLOPE).  The linear, narrow transition matches
  // the observed HW behavior: borderline overflows (WS just over MALL)
  // already lose all benefit of cached writes because some of A/B must
  // stream from HBM, so NTD0 and NTD4 converge quickly.
  static constexpr double D_POLLUTION_PENALTY = 1.0;

  // Cached-D (cache_hints_d < 4) write-allocate penalty in the epilogue.
  // On GPUs with write-allocate L2, a cached store that misses L2 must
  // first fetch the cache line from HBM, modify, and eventually write
  // back -- effectively doubling HBM traffic for large outputs.
  // NT writes (cache_hints_d >= 4) bypass this entirely.
  // Applied to effective epilogue store bandwidth:
  //   effective_store_bw =
  //       store_bw / (1 + D_WRITEALLOC_FACTOR * intensity * mall_fit)
  // when D overflows L2.  When D fits in L2, cached writes hit L2
  // bandwidth (mem1) instead, which is typically 2-3x HBM -- modeled by
  // switching the bandwidth tier, not by this factor.  The mall_fit
  // factor (linear fade-out, see D_POLLUTION_PENALTY and
  // MALL_OVERFLOW_FADE_SLOPE) attenuates the penalty when the GEMM's
  // total footprint exceeds MALL, since there is no input reuse to
  // protect.
  static constexpr double D_WRITEALLOC_FACTOR = 1.0;

  // Approximate MALL (Infinity Cache / last-level cache) capacity in
  // bytes for archs that carry a MALL stage (gfx942/gfx950 = 256 MiB).
  // Used only as a gate by the cached-D pollution penalty above -- it
  // does not affect the existing H_mem_mall_* hit-rate estimation.
  // Archs without MALL should not trigger the cached-D gate anyway since
  // hardware.has_MALL() governs the MALL stage in the latency model.
  static constexpr size_t MALL_CAPACITY_BYTES = 256u * 1024u * 1024u;

  // Slope of the mall_fit linear fade-out past MALL capacity.
  // mall_fit = clamp(1 + slope * (1 - WS/MALL), 0, 1).
  // With slope = 2.0:  mall_fit = 1 at WS <= MALL, 0.5 at WS=1.25*MALL,
  // 0 at WS >= 1.5*MALL.  A larger slope means the penalty disappears
  // faster past the fit boundary.  Empirically tuned so borderline
  // cases (WS just above MALL, where inputs are streaming from HBM)
  // do not incur a penalty despite the global D footprint overflowing
  // L2.
  static constexpr double MALL_OVERFLOW_FADE_SLOPE = 2.0;

  // --- LDS Occupancy ---
  // Per additional workgroup that can co-reside on a CU (beyond the first),
  // effective timestep count is multiplied by this factor.
  // < 1.0 means higher occupancy hides more inter-WG latency.
  static constexpr double OCC_TIMESTEP_BENEFIT = 0.95;

  // --- 1LDSBuffer ---
  // Per-iteration overhead (cycles) when using a single LDS buffer instead
  // of double-buffering.  Represents the read-sync-write barrier stall.
  static constexpr double ONE_LDS_BUFFER_OVERHEAD = 80.0;

  // --- LDSTrInst ---
  // Per-iteration VALU pack/transpose overhead (cycles) when NOT using
  // hardware LDS transpose loads (16-bit types only).  Without LDSTrInst,
  // each iteration needs explicit VPermB32/VSwapB32 after ds_load.
  static constexpr double PACK_TRANSPOSE_OVERHEAD = 15.0;

  // --- Tail Loop ---
  // Fixed overhead per tail-loop sub-iteration (cycles).
  // Covers K-masking, LDS address reset, barrier, ds_read/write, conditional branching.
  static constexpr double TAIL_LOOP_OVERHEAD = 1500.0;

  // --- Tile Fixed Overhead ---
  // Fixed per-tile cost (cycles) that does not scale with tile dimensions.
  // Covers WG dispatch, pipeline setup, address generation, and barriers.
  // Compresses the cost ratio between large and small tiles.
  static constexpr double TILE_FIXED_OVERHEAD = 1500.0;

  // --- DepthU Waste (tail-only kernels) ---
  // When k_iters==0 the full MT_K LDS allocation and register footprint are
  // paid but only tail_k < MT_K is actually used.  Larger MT_K means more
  // wasted LDS, higher register pressure, and heavier setup for zero benefit.
  // Penalty = DU_WASTE_OVERHEAD * (MT_K / tail_k).
  static constexpr double DU_WASTE_OVERHEAD = 500.0;

  // --- VGPR Pressure / ACC Occupancy ---
  // Larger MFMA tiles (especially MI32x32) allocate the full output
  // accumulator tile as persistent ACC VGPRs for the entire K-loop.
  // On gfx950 each SIMD has 512 ACC VGPRs; output tiles whose per-thread
  // ACC footprint (MT_M * MT_N / wave_threads) exceeds this budget lose
  // wave-level latency hiding because at most 1 wave can fit per SIMD.
  // Physically this shows up as (a) unmasked mem-load latency in the
  // main loop and (b) ACC spill/restore overhead in the epilogue.  The
  // penalty is smooth, bounded, and fires only when the per-thread ACC
  // footprint approaches or exceeds the budget (relative to a target
  // of 2 waves/SIMD for good latency hiding).
  // Multiplier on L_tile_total:
  //   (1 + VGPR_PRESSURE_PENALTY * max(0, 2 - acc_waves_per_simd)).
  // With budget 512 and strength 0.2, a MT192x256 MI32 tile
  // (acc_per_thread=768, acc_waves=0.67) incurs ~27% penalty, while
  // a MT160x192 MI16 tile (acc_per_thread=480, acc_waves=1.07) only
  // ~19% and MT80x256 (acc_per_thread=320, acc_waves=1.6) only ~8%.
  static constexpr double VGPR_PRESSURE_PENALTY = 0.2;
  static constexpr double ACC_VGPR_BUDGET_PER_SIMD = 512.0;
  static constexpr double ACC_WAVES_TARGET = 2.0;

  // --- MFMA Pipeline Hazard (DepthU too shallow for back-to-back MI reuse) ---
  // When MT_K <= MI_K each main-loop iteration contains exactly one (or fewer)
  // MI K-fold, so there is no opportunity to issue back-to-back MFMAs within
  // the iteration.  Every iteration restarts the MFMA pipeline, paying a
  // fixed bubble at iteration boundaries (typically ~dep_latency MFMA cycles
  // waiting on LDS + reg alloc).  On gfx950 MI16x16x32_bf16 (MI_K=32) this
  // manifests as a ~20-25% slowdown at MT_K=32 relative to MT_K=64, holding
  // all other tile dimensions constant.
  // Multiplier on L_mainloop:
  //   (1 + MI_PIPELINE_HAZARD_PENALTY * max(0, 2 - MT_K/MI_K)).
  // Smooth, bounded, and zero once MT_K/MI_K >= 2 (the first back-to-back
  // reuse opportunity kicks in).
  static constexpr double MI_PIPELINE_HAZARD_PENALTY = 0.15;

  // --- Spatial Waste ---
  // Additive per-tile penalty (cycles per wasted output element) when the
  // tile grid over-allocates in M or N.  Covers predicated-load overhead,
  // MFMA waste, cache pollution from shifted-pointer reads, and imperfect
  // overlap of wasted work.  Scales naturally with tile area so large tiles
  // get a big penalty and small tiles almost none.
  static constexpr double SPATIAL_WASTE_WEIGHT = 1.0;

  // --- Effective Tile Under-Utilization Penalty Strength ---
  // Softens the 1/utilization multiplier on L_tile_total.  The raw 1/util
  // formula assumes every wasted launched element costs as much as a useful
  // one, but HW pipelines and buffers hide a large fraction of the waste
  // (predicated loads, MFMA bubbles, and edge-masked stores overlap with
  // the surrounding useful work).  Empirically, on M=160 N-heavy problems,
  // MT192x160 / MT160x96 / MT64x160 kernels (util 0.66-0.83) are over-
  // predicted by +15-30% with alpha=1.0; dropping to alpha=0.5 roughly
  // halves the over-charge while leaving perfect-fit (util=1.0) kernels
  // untouched and still penalizing extreme dilution (util<<1) meaningfully.
  //
  //   effective_tile_penalty = 1 + alpha * (1/util - 1)
  //     util=1.00  -> 1.00   (no change)
  //     util=0.83  -> 1.10   (was 1.20 at alpha=1.0)
  //     util=0.66  -> 1.25   (was 1.50)
  //     util=0.50  -> 1.50   (was 2.00)
  static constexpr double TILE_PENALTY_ALPHA = 0.5;
};

/**
 * @brief Structure containing all trainable heuristic parameters.
 *
 * This structure consolidates all empirical constants and weights used in
 * the latency model, making them trainable and configuration-driven.
 */
struct heuristic_params_t {
  // === Latency Component Weights ===
  double weight_mem_l2        = heuristic_defaults_t::WEIGHT_MEM_L2;
  double weight_mem_mall      = heuristic_defaults_t::WEIGHT_MEM_MALL;
  double weight_mem_dram      = heuristic_defaults_t::WEIGHT_MEM_DRAM;
  double weight_compute       = heuristic_defaults_t::WEIGHT_COMPUTE;
  double weight_memory        = heuristic_defaults_t::WEIGHT_MEMORY;
  double weight_wg_setup      = heuristic_defaults_t::WEIGHT_WG_SETUP;
  double weight_prologue      = heuristic_defaults_t::WEIGHT_PROLOGUE;
  double weight_epilogue      = heuristic_defaults_t::WEIGHT_EPILOGUE;
  double weight_loop_overhead = heuristic_defaults_t::WEIGHT_LOOP_OVERHEAD;
  double weight_tile_total    = heuristic_defaults_t::WEIGHT_TILE_TOTAL;

  // === Empirical Constants ===
  double main_memory_load_latency         = heuristic_defaults_t::MAIN_MEMORY_LOAD_LATENCY;
  double occupancy_decay_base             = heuristic_defaults_t::OCCUPANCY_DECAY_BASE;
  double mall_depth_sq                    = heuristic_defaults_t::MALL_DEPTH_SQ;
  double mall_cold_floor                  = heuristic_defaults_t::MALL_COLD_FLOOR;
  double l2_depth_sq                      = heuristic_defaults_t::L2_DEPTH_SQ;
  double l2_cold_floor                    = heuristic_defaults_t::L2_COLD_FLOOR;
  double l2_pollution_penalty             = heuristic_defaults_t::L2_POLLUTION_PENALTY;
  double l2_amp_ceiling_batched           = heuristic_defaults_t::L2_AMP_CEILING_BATCHED;
  double l2_amp_ceiling_k_split           = heuristic_defaults_t::L2_AMP_CEILING_K_SPLIT;
  double l2_amp_ceiling_skinny            = heuristic_defaults_t::L2_AMP_CEILING_SKINNY;
  double l2_depth_penalty                 = heuristic_defaults_t::L2_DEPTH_PENALTY;
  double l1_hit_rate_ceiling_skinny       = heuristic_defaults_t::L1_HIT_RATE_CEILING_SKINNY;
  double epilogue_cycles_per_acc_read     = heuristic_defaults_t::EPILOGUE_CYCLES_PER_ACC_READ;
  double epilogue_acc_read_parallelism    = heuristic_defaults_t::EPILOGUE_ACC_READ_PARALLELISM;
  double epilogue_cycles_per_bounds_check = heuristic_defaults_t::EPILOGUE_CYCLES_PER_BOUNDS_CHECK;
  double epilogue_scalar_store_penalty    = heuristic_defaults_t::EPILOGUE_SCALAR_STORE_PENALTY;
  size_t epilogue_threads_per_wave        = heuristic_defaults_t::EPILOGUE_THREADS_PER_WAVE;
  size_t epilogue_bytes_per_vectorized_store =
      heuristic_defaults_t::EPILOGUE_BYTES_PER_VECTORIZED_STORE;
  size_t epilogue_cache_line_bytes = heuristic_defaults_t::EPILOGUE_CACHE_LINE_BYTES;
  size_t epilogue_workspace_bytes_per_elem =
      heuristic_defaults_t::EPILOGUE_WORKSPACE_BYTES_PER_ELEM;
  double epilogue_salu_overhead         = heuristic_defaults_t::EPILOGUE_SALU_OVERHEAD;
  double epilogue_l_barrier             = heuristic_defaults_t::EPILOGUE_L_BARRIER;
  double epilogue_l_smem                = heuristic_defaults_t::EPILOGUE_L_SMEM;
  double epilogue_k_padding_penalty     = heuristic_defaults_t::EPILOGUE_K_PADDING_PENALTY;
  size_t postgsu_compute_bytes          = heuristic_defaults_t::POSTGSU_COMPUTE_BYTES;
  double postgsu_kernel_launch_overhead = heuristic_defaults_t::POSTGSU_KERNEL_LAUNCH_OVERHEAD;
  size_t postgsu_threads_per_wg         = heuristic_defaults_t::POSTGSU_THREADS_PER_WG;
  size_t postgsu_wavefront_size         = heuristic_defaults_t::POSTGSU_WAVEFRONT_SIZE;

  // === Main Loop Efficiency ===
  double main_loop_efficiency = heuristic_defaults_t::MAIN_LOOP_EFFICIENCY;

  // === PGR / LSU / DTL / NTD parameters ===
  double prologue_setup_fraction  = heuristic_defaults_t::PROLOGUE_SETUP_FRACTION;
  double lsu_reduction_overhead = heuristic_defaults_t::LSU_REDUCTION_OVERHEAD;
  double ntd_ksplit_penalty     = heuristic_defaults_t::NTD_KSPLIT_PENALTY;
  double d_pollution_penalty    = heuristic_defaults_t::D_POLLUTION_PENALTY;
  double d_writealloc_factor    = heuristic_defaults_t::D_WRITEALLOC_FACTOR;
  size_t mall_capacity_bytes    = heuristic_defaults_t::MALL_CAPACITY_BYTES;
  double mall_overflow_fade_slope = heuristic_defaults_t::MALL_OVERFLOW_FADE_SLOPE;
  double occ_timestep_benefit    = heuristic_defaults_t::OCC_TIMESTEP_BENEFIT;
  double one_lds_buffer_overhead   = heuristic_defaults_t::ONE_LDS_BUFFER_OVERHEAD;
  double pack_transpose_overhead   = heuristic_defaults_t::PACK_TRANSPOSE_OVERHEAD;
  double tail_loop_overhead        = heuristic_defaults_t::TAIL_LOOP_OVERHEAD;
  double tile_fixed_overhead     = heuristic_defaults_t::TILE_FIXED_OVERHEAD;
  double du_waste_overhead       = heuristic_defaults_t::DU_WASTE_OVERHEAD;
  double spatial_waste_weight    = heuristic_defaults_t::SPATIAL_WASTE_WEIGHT;
  double mi_pipeline_hazard_penalty = heuristic_defaults_t::MI_PIPELINE_HAZARD_PENALTY;
  double vgpr_pressure_penalty       = heuristic_defaults_t::VGPR_PRESSURE_PENALTY;
  double acc_vgpr_budget_per_simd    = heuristic_defaults_t::ACC_VGPR_BUDGET_PER_SIMD;
  double acc_waves_target            = heuristic_defaults_t::ACC_WAVES_TARGET;
  double tile_penalty_alpha          = heuristic_defaults_t::TILE_PENALTY_ALPHA;

  /**
   * @brief Merge this parameter set with another (for hierarchical lookup).
   * Only non-default values from 'other' override values in 'this'.
   */
  void merge_with(const heuristic_params_t& other);
};

/**
 * @brief Key structure for looking up heuristics in the unified map.
 *
 * This key captures all relevant characteristics that might affect
 * heuristic parameter selection. Fields can be wildcards (using optional).
 */
struct heuristic_key_t {
  std::optional<hardware_t::architecture_t> arch;
  std::optional<data_type_t> a_dtype;
  std::optional<data_type_t> b_dtype;
  std::optional<data_type_t> mi_dtype;
  std::optional<transpose_t> a_transpose;
  std::optional<transpose_t> b_transpose;
  std::optional<size_t> mt_m;
  std::optional<size_t> mt_n;
  std::optional<size_t> mt_k;
  std::optional<bool> hand_optimized_main_loop;

  // For problem-size dependent heuristics
  std::optional<size_t> min_m;
  std::optional<size_t> max_m;
  std::optional<size_t> min_n;
  std::optional<size_t> max_n;
  std::optional<size_t> min_k;
  std::optional<size_t> max_k;

  /**
   * @brief Check if this key matches the given problem/hardware/config.
   * @return true if all specified fields match (wildcards match everything)
   */
  bool matches(const problem_t& problem, const hardware_t& hardware, const config_t& config) const;

  /**
   * @brief Get specificity score (number of non-wildcard fields).
   * Used for prioritizing more specific matches over general ones.
   */
  size_t specificity() const;
};

/**
 * @brief Key for hand-optimized kernels.
 *
 * Hand-optimized kernels such as CMS(Custom Main-loop Scheduling) kernels have fully specified
 * characteristics (arch, dtype, layout, MT sizes).
 */
struct hand_optimized_kernel_key_t {
  hardware_t::architecture_t arch;
  data_type_t mi_dtype;
  transpose_t a_transpose;
  transpose_t b_transpose;
  size_t mt_m;
  size_t mt_n;
  size_t mt_k;

  std::string to_string() const {
    return std::string(hardware_t::arch_enum_to_name(arch)) + "_" +
           origami::datatype_to_string(mi_dtype) + "_" +
           (a_transpose == transpose_t::T ? "T" : "N") +
           (b_transpose == transpose_t::T ? "T" : "N") + "_" + std::to_string(mt_m) + "x" +
           std::to_string(mt_n) + "x" + std::to_string(mt_k);
  }

  bool operator==(const hand_optimized_kernel_key_t& other) const {
    return arch == other.arch && mi_dtype == other.mi_dtype && a_transpose == other.a_transpose &&
           b_transpose == other.b_transpose && mt_m == other.mt_m && mt_n == other.mt_n &&
           mt_k == other.mt_k;
  }

  std::size_t hash() const {
    return math::hash_combine(static_cast<int>(arch),
                              static_cast<int>(mi_dtype),
                              static_cast<int>(a_transpose),
                              static_cast<int>(b_transpose),
                              mt_m,
                              mt_n,
                              mt_k);
  }
};

struct hand_optimized_kernel_key_hash {
  std::size_t operator()(const hand_optimized_kernel_key_t& k) const { return k.hash(); }
};

/**
 * @brief Unified heuristics database.
 *
 * This is the single source of truth for all heuristic parameters.
 * Maps from heuristic keys (problem characteristics) to parameter sets.
 *
 * @note Thread Safety:
 * - Singleton initialization is thread-safe (C++11 magic statics)
 * - lookup() is thread-safe for concurrent reads (const method)
 * - add_entry() is NOT thread-safe and should only be called during
 *   initialization (from constructor) before any concurrent access
 * - Do NOT call add_entry() after the singleton is initialized and
 *   multiple threads may be accessing the database
 */
class heuristics_database_t {
 public:
  /**
   * @brief Lookup heuristic parameters for given problem/hardware/config.
   *
   * Performs hierarchical lookup:
   * 1. For hand-optimized kernels: O(1) hash map lookup
   * 2. For general heuristics: Linear search with specificity ordering
   * 3. Falls back to default parameters
   *
   * @param problem Problem definition
   * @param hardware Hardware characteristics
   * @param config Kernel configuration
   * @return heuristic_params_t Merged parameter set
   */
  heuristic_params_t lookup(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config) const;

  /**
   * @brief Add or update a heuristic entry.
   */
  void add_entry(const heuristic_key_t& key, const heuristic_params_t& params);

  /**
   * @brief Return true if the database has a hand-optimized entry for the given (arch, dtype,
   * layout, MT).
   */
  bool has_hand_optimized_entry(hardware_t::architecture_t arch,
                                data_type_t mi_dtype,
                                transpose_t transA,
                                transpose_t transB,
                                size_t mt_m,
                                size_t mt_n,
                                size_t mt_k) const;

  /**
   * @brief Get the global heuristics database instance.
   */
  static heuristics_database_t& get_instance();

 private:
  heuristics_database_t();  // Private constructor for singleton

  // General heuristics storage (linear search)
  std::vector<std::pair<heuristic_key_t, heuristic_params_t>> entries_;

  // unordered_map for hand-optimized kernels
  std::
      unordered_map<hand_optimized_kernel_key_t, heuristic_params_t, hand_optimized_kernel_key_hash>
          hand_optimized_map_;

  heuristic_params_t default_params_;

  // Initialize with default heuristics
  void initialize_defaults();
};

/**
 * @brief Convenience function to get heuristic parameters.
 *
 * This is the main entry point that replaces compute_heuristic_weights().
 */
inline heuristic_params_t get_heuristic_params(const problem_t& problem,
                                               const hardware_t& hardware,
                                               const config_t& config) {
  return heuristics_database_t::get_instance().lookup(problem, hardware, config);
}

/**
 * @brief Helper to create a key for hand-optimized kernels
 */
heuristic_key_t make_hand_optimized_kernel_key(hardware_t::architecture_t arch,
                                               data_type_t mi_dtype,
                                               transpose_t transA,
                                               transpose_t transB,
                                               size_t MT_M,
                                               size_t MT_N,
                                               size_t MT_K);

/**
 * @brief Helper to create a key for tile configuration.
 */
heuristic_key_t make_tile_key(size_t MT_M,
                              size_t MT_N,
                              size_t MT_K,
                              std::optional<transpose_t> transA = std::nullopt,
                              std::optional<transpose_t> transB = std::nullopt);

/**
 * @brief Helper to create a key for architecture/datatype combination.
 */
heuristic_key_t make_arch_dtype_key(hardware_t::architecture_t arch, data_type_t mi_dtype);

}  // namespace origami
