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
#include "origami/origami_export.h"

namespace origami {
/**
 * @brief Default values for heuristic parameters.
 * Centralized location for all default constants.
 */
struct heuristic_defaults_t {
  // Latency Component Weights
  static constexpr double WEIGHT_TILE_TOTAL    = 1.0;
  static constexpr size_t CACHE_LINE_BYTES       = 128;
  static constexpr size_t DRAM_SECTOR_BYTES      = 64;
  // gfx1250: aligned block DRAM reads and L2 misses move.
  static constexpr size_t GFX1250_DRAM_BLOCK_BYTES = 256;
  static constexpr size_t LDS_XFER_BYTES         = 128;
  static constexpr double KERNEL_LAUNCH_OVERHEAD = 4.5 * 2200.0;
  static constexpr double POSTGSU_KERNEL_LAUNCH_OVERHEAD = 1.6 * 2200.0;
  static constexpr size_t POSTGSU_THREADS_PER_WG = 256;

  // Empirical Constants
  static constexpr double L2_AMP_CEILING_BATCHED = 0.9;
  static constexpr double L2_AMP_CEILING_K_SPLIT = 0.4;
  // Vector-ALU instructions charged per edge-store bounds check and per AccVGPR read;
  // fractional because they absorb overlap with other work.
  static constexpr double EPILOGUE_VALU_INSTS_PER_BOUNDS_CHECK = 1.25;
  static constexpr double EPILOGUE_VALU_INSTS_PER_ACC_READ     = 0.25;
  static constexpr size_t EPILOGUE_BYTES_PER_VECTORIZED_STORE  = 16;  // buffer_store_dwordx4
  static constexpr double EPILOGUE_NONSWAP_STORE_TRAFFIC       = 1.5;
  static constexpr size_t EPILOGUE_WORKSPACE_BYTES_PER_ELEM    = 4;
  static constexpr double EPILOGUE_SALU_OVERHEAD               = 35.0;
  static constexpr double L_BARRIER                            = 100.0;
  static constexpr double EPILOGUE_L_SMEM       = 1900.0;  // s_load_dword(glc) cross-XCD flag poll
  static constexpr size_t POSTGSU_COMPUTE_BYTES = 4;       // workspace partials stored as f32

  // Main Loop Efficiency
  static constexpr double MAIN_LOOP_EFFICIENCY = 1.0;

  // TF32 Emulation Constants
  static constexpr double TF32_ARITH_INTENSITY_THRESHOLD = 1000.0;

  // Model tuning constants (relocated from gemm.cpp)
  static constexpr double TARGET_WG_SLOTS_PER_CU       = 2.0;
  static constexpr double K_ITER_LOOP_OVERHEAD         = 500.0;
  static constexpr double BATCHED_FILL_ITER_TARGET     = 4.0;
  static constexpr double BATCHED_FILL_PENALTY         = 4.0;
  static constexpr double TAIL_WASTE_PENALTY           = 4.0;
  static constexpr double OVERSIZE_WASTE_WEIGHT        = 3.0;
  static constexpr double M_EDGE_PENALTY               = 2.0;
  static constexpr size_t SUBMI_GEMV_K_MIN             = 256;
  static constexpr double EPILOGUE_OCC_SATURATION      = 2.0;
  static constexpr double SCALAR_STORE_EXPOSED_PENALTY = 4.0;

  // gfx1250 (CDNA5) constants, read only where hardware.is_cdna5().  Each is a hardware fact or
  // comes from a microbenchmark of the one mechanism it describes (named with it).
  //
  // Units: the model works in engine-clock cycles, so every time is in cycles (_CLK) and every
  // rate in bytes per cycle (_B_PER_CLK), like the generic constants above.  The microbenchmarks
  // report wall time on a 2.4 GHz engine clock / 1.9 GHz memory clock part; each value is written
  // as <measured ns> * 2.4 or <measured B/ns> / 2.4 so the measured number stays visible, with the
  // cycle value in the comment.  Nothing here is rescaled by hardware_t::compute_clock_ghz.

  /// Tensor-DMA row service rate of one WGP from L2, B/clk, and the minimum bytes a row costs: a
  /// row narrower than the 256 B block L2 serves still costs a whole block.
  static constexpr double GFX1250_TDM_L2_ROW_B_PER_CLK = 557.0 / 2.4;  // 232.08 B/clk (557 B/ns)
  /// Tensor-DMA row service rate of one WGP from DRAM, B/clk; a row costs at least one 256 B DRAM
  /// block (fitted to a tensor-DMA bandwidth microbenchmark, q fixed at the block).
  static constexpr double GFX1250_TDM_DRAM_ROW_B_PER_CLK = 420.0 / 2.4;  // 175.0 B/clk (420 B/ns)
  /// Soft-minimum sharpness of the WGPs' tensor-DMA rate against the L2 peak (same fits; the DRAM peak is met
  /// in a hard minimum, which fits the sweep as well).
  static constexpr double GFX1250_TDM_L2_SHARPNESS   = 3.30;
  /// Fixed time of one wave's epilogue store instruction, cycles.
  static constexpr double GFX1250_STORE_INST_CLK = 14.31 * 2.4;  // 34.344 clk (14.31 ns)
  /// Store bandwidth of one wave's instruction stream, B/clk.
  static constexpr double GFX1250_STORE_WAVE_B_PER_CLK = 29.2 / 2.4;  // 12.167 B/clk (29.2 B/ns)
  /// Store bandwidth one WGP's waves share, B/clk.
  static constexpr double GFX1250_STORE_WGP_B_PER_CLK = 216.0 / 2.4;  // 90.0 B/clk (216 B/ns)
  /// Device store bandwidth ceiling, B/clk, and the soft-minimum sharpness it is met with.
  static constexpr double GFX1250_STORE_PEAK_B_PER_CLK = 20950.0 / 2.4;  // 8729.2 B/clk (20950 B/ns)
  static constexpr double GFX1250_STORE_SHARPNESS      = 2.17;
  /// Bytes landing in LDS with multicast and the cluster barrier over those without, two
  /// tensor loads in flight per wave: L2-resident 37.1 / 32.7 TB/s, DRAM-streaming 17.2 /
  /// 22.2 TB/s (multicast microbenchmark, 4x4 clusters of MT 128, DU 64).
  static constexpr double GFX1250_MULTICAST_L2_GAIN   = 37.1 / 32.7;
  static constexpr double GFX1250_MULTICAST_DRAM_GAIN = 17.2 / 22.2;
  /// WGPs of the part hardware_t::mem_bw_per_wg_coefficients was fit on; compute_mem_bw_from_
  /// occupancy rescales the active CU count by this over the part's N_CU.
  static constexpr double GFX1250_BW_FIT_WGPS = 256.0;
  /// Non-SourceSwap D store (fitted to a store-layout microbenchmark): a wave32 16x16 WMMA accumulator
  /// stored with 16 B lanes writes 32 B (COLUMN_BYTES) into each of 16 columns per instruction.  Extra time of such
  /// an instruction over the SourceSwap one, cycles, and the device ceiling it reaches, B/clk, for TemporalHintD
  /// RT / NT.
  static constexpr double GFX1250_NONSWAP_STORE_COLUMN_BYTES     = 32.0;
  static constexpr double GFX1250_NONSWAP_STORE_EXTRA_CLK_RT     = 61.4 * 2.4;    // 147.36 clk (61.4 ns)
  static constexpr double GFX1250_NONSWAP_STORE_EXTRA_CLK_NT     = 152.3 * 2.4;   // 365.52 clk (152.3 ns)
  static constexpr double GFX1250_NONSWAP_STORE_PEAK_B_PER_CLK_RT = 4674.0 / 2.4;  // 1947.5 B/clk (4674 B/ns)
  static constexpr double GFX1250_NONSWAP_STORE_PEAK_B_PER_CLK_NT = 1432.0 / 2.4;  // 596.67 B/clk (1432 B/ns)
  /// Sustained DRAM read bandwidth (B/clk) the GFX1250_*STORE_PEAK ceilings were measured at; the
  /// store model scales the ceilings by the part's DRAM peak (hardware_t::mem3_perf_ratio, B/clk)
  /// over this.
  static constexpr double GFX1250_STORE_REF_DRAM_B_PER_CLK = 17130.0 / 2.4;  // 7137.5 B/clk (17130 B/ns)
  /// Load-to-use latency of a DRAM access (single-wave pointer chase), cycles.
  static constexpr double GFX1250_DRAM_LATENCY_CLK = 180.0 * 2.4;  // 432 clk (180 ns)
  /// Load-to-use latency of an L2 hit, cycles.
  static constexpr double GFX1250_L2_LATENCY_CLK = 125.0 * 2.4;  // 300 clk (125 ns)
  /// Dependent LDS read latency in cycles the matrix pipe must cover.
  static constexpr double GFX1250_LDS_READ_LATENCY_CLK = 70.0;
  /// Per-lane vector loads (one 16 B load per lane in flight): one WGP's ceiling and the device's
  /// L2-resident peak, B/clk (bandwidth microbenchmark).
  static constexpr double GFX1250_LANE_LOAD_WGP_B_PER_CLK = 91.0 / 2.4;     // 37.917 B/clk (91 B/ns)
  static constexpr double GFX1250_LANE_LOAD_L2_B_PER_CLK  = 21100.0 / 2.4;  // 8791.7 B/clk (21100 B/ns)
  /// StreamK in-kernel linear fixup: time per partial that does not depend on its size (the owner's flag
  /// poll, flag reset, barrier and read set-up), cycles, with the partial's bytes read at
  /// GFX1250_LANE_LOAD_WGP_B_PER_CLK (StreamK fixup microbenchmark: 1.52 +- 0.07 us over 31 kernels,
  /// F = 2..4; no separate peer-release time was found).
  static constexpr double GFX1250_SK_FIXUP_HANDSHAKE_CLK = 1520.0 * 2.4;  // 3648 clk (1520 ns)
  /// Loaded tensor-DMA load latency, lat0 + c x device throughput: cycles and cycles per (B/clk) of device
  /// throughput, from L2 and from DRAM (fitted to a tensor-DMA bandwidth microbenchmark with the 256 B row minimum;
  /// measured as ns and ns per TB/s; 1 TB/s = 1000 B/ns, so ns per TB/s x 2.4^2 / 1000 = clk per B/clk).
  static constexpr double GFX1250_TDM_L2_LAT0_CLK        = 216.9 * 2.4;                // 520.56 clk (216.9 ns)
  static constexpr double GFX1250_TDM_L2_LAT_CLK_PER_BPC = 1.076 * 2.4 * 2.4 / 1000.0;  // 0.00619776 (1.076 ns per TB/s)
  static constexpr double GFX1250_TDM_DRAM_LAT0_CLK        = 630.6 * 2.4;                // 1513.44 clk (630.6 ns)
  static constexpr double GFX1250_TDM_DRAM_LAT_CLK_PER_BPC = 18.35 * 2.4 * 2.4 / 1000.0;  // 0.105696 (18.35 ns per TB/s)
  /// Interval between back-to-back kernel launches on one stream, cycles, for 1 to 1024 workgroups
  /// (kernel-launch microbenchmark).  Host / driver time, not clocked by the engine.
  static constexpr double GFX1250_KERNEL_LAUNCH_CLK = 2080.0 * 2.4;  // 4992 clk (2080 ns)
  /// TH_LOAD_NT tensor loads of rows narrower than the 256 B DRAM block: the block's other part is refetched with
  /// probability 1 / (1 + (D0 / D)^n) when D bytes of tensor-DMA traffic reach the device between the two uses
  /// (D0 in MiB; fitted to a tensor-DMA stride microbenchmark).
  static constexpr double GFX1250_TDM_NT_REFETCH_D0_MIB = 34.2;
  static constexpr double GFX1250_TDM_NT_REFETCH_N      = 1.6;
  /// Extra time of one wave's edge-path store instruction over the full-width store, measured inside real BF16 TN
  /// kernels (AdaptiveGemm twins on M-remainder grids), cycles: M-edge path by lane bytes
  /// (2, 4, 8, 16 B), and the full-width N-only edge path.
  static constexpr double GFX1250_EDGE_STORE_EXTRA_CLK_2B  = 94.8 * 2.4;  // 227.52 clk (94.8 ns)
  static constexpr double GFX1250_EDGE_STORE_EXTRA_CLK_4B  = 71.2 * 2.4;  // 170.88 clk (71.2 ns)
  static constexpr double GFX1250_EDGE_STORE_EXTRA_CLK_8B  = 47.6 * 2.4;  // 114.24 clk (47.6 ns)
  static constexpr double GFX1250_EDGE_STORE_EXTRA_CLK_16B = 33.7 * 2.4;  // 80.88 clk (33.7 ns)
  static constexpr double GFX1250_EDGE_STORE_EXTRA_CLK_N   = 38.6 * 2.4;  // 92.64 clk (38.6 ns)
  /// Soft-maximum sharpness of a TDM + LDS-read + WMMA K-loop's memory and math times, L2-resident and
  /// DRAM-streaming (fitted to a K-loop microbenchmark).
  static constexpr double GFX1250_KLOOP_L2_OVERLAP   = 1.93;
  static constexpr double GFX1250_KLOOP_DRAM_OVERLAP = 1.73;
  /// Fixed cost of one K-iteration (two barriers, tensor waits, TDM issue), cycles: intercept of a K-loop DepthU
  /// sweep for 4 waves, which nearly all gfx1250 library kernels use (8 waves measured 314 clk).
  /// The microbenchmark intercept alone is 207 clk; the value used is refit on the 145-size N-sweep (see
  /// GFX1250_FIT_* below).
  static constexpr double GFX1250_KITER_FIXED_CLK = 263.43924868273393;
  /// v_wmma_f32_16x16x32_bf16 issue interval per SIMD with 1, 2 and 4 waves on the SIMD, cycles at the engine
  /// clock (WMMA-rate microbenchmark, independent accumulators).
  static constexpr double GFX1250_WMMA_BF16_CLK = 10.1;  // 1 wave per SIMD; 2 / 4 waves measured 7.5 / 6.05
  /// Most workgroups per WGP the co-residency law was measured for (1..4 workgroups per WGP).
  static constexpr double GFX1250_MAX_MEASURED_RESIDENT_WGS = 4.0;
  /// K-loop slowdown of a multicast cluster over its exact 1x1 twin beyond the microbenchmarked multicast model:
  /// 1 + lambda x k / (k + K0) for k K-iterations, lambda by workgroups along the cluster's larger axis (2, 4, 8)
  /// (fitted to real BBS TN kernels on synthetic shapes).
  /// Model-relative: measured / predicted cluster-to-twin ratio of this model without the term, so it is refitted
  /// from the stored ratios after any change of the K-loop model (here: against the issue/wait-recurrence stall;
  /// rms log 0.140 over 768 cases; K0 = 64 is the last value of the fit's grid, 48 fits equally).
  static constexpr double GFX1250_CLUSTER_LOCKSTEP_2  = 0.132;
  static constexpr double GFX1250_CLUSTER_LOCKSTEP_8  = 0.624;
  static constexpr double GFX1250_CLUSTER_LOCKSTEP_K0 = 64.0;
  /// Rate at which a WGP's tail loop (K % DepthU != 0, including K < DepthU) moves the whole DepthU window of A and
  /// B, B/clk, with every WGP busy (measured as K = n DU + 8 / + 32 vs n DU).
  static constexpr double GFX1250_TAIL_WINDOW_B_PER_CLK = 66.8 / 2.4;  // 27.833 B/clk (66.8 B/ns)
  /// Odd K (A and B rows not 4 B aligned): extra time per tile, a constant plus a part per K-iteration, cycles per
  /// KiB of the DepthU window (measured as K = n DU + 1 vs n DU and K = 1).
  static constexpr double GFX1250_ODD_K_CLK_PER_KIB      = 92.0 * 2.4;  // 220.8 clk/KiB (92.0 ns/KiB)
  static constexpr double GFX1250_ODD_K_CLK_PER_KIB_ITER = 26.9 * 2.4;  // 64.56 clk/KiB (26.9 ns/KiB)
  /// Cost outside the K-loop that the tile model leaves out, in cycles, as a constant plus growth per KiB of the
  /// output tile and per KiB of one K-iteration of A and B.  Per tile round (slope of time vs tile
  /// rounds of real kernels; the constant fits ~0) and once per kernel (one-wave
  /// intercepts less one round).  The per-kernel cost also grows per doubling of CUOccupancy (measured with one WG
  /// per WGP; the mechanism is not isolated).  Model-relative (measured minus this model with every fixed term at
  /// zero): refitted against the issue/wait-recurrence stall, which moves the model's one-wave intercepts (per
  /// kernel rms 2.4 us, n 405); the per-round fit (K = 1 and K = DU slopes, rms 2.6 us, n 162) is unchanged by it.
  static constexpr double GFX1250_KERNEL_FIXED_CLK              = 569.0 * 2.4;  // 1365.6 clk (569 ns)
  static constexpr double GFX1250_KERNEL_FIXED_CLK_PER_KIB_D    = 28.1 * 2.4;   // 67.44 clk/KiB (28.1 ns/KiB)
  static constexpr double GFX1250_KERNEL_FIXED_CLK_PER_KIB_OP   = 6.7 * 2.4;    // 16.08 clk/KiB (6.7 ns/KiB)
  static constexpr double GFX1250_KERNEL_FIXED_CLK_PER_LOG2_OCC = 680.0 * 2.4;  // 1632 clk (680 ns)
  static constexpr double GFX1250_TILE_FIXED_CLK_PER_KIB_D    = 15.2 * 2.4;  // 36.48 clk/KiB (15.2 ns/KiB)
  static constexpr double GFX1250_TILE_FIXED_CLK_PER_KIB_OP   = 16.8 * 2.4;  // 40.32 clk/KiB (16.8 ns/KiB)

  /// gfx1250 refit of the model's relative-cost constants on the measured top-30 candidates of the 145-size
  /// (5 shapes x N = 64..512) bf16 TN N-sweep (centered objective, all data).  The fit
  /// was made on the base model without the gfx1250 terms, so the values that are multipliers on a model
  /// term are applied as such (1.0 = unchanged): the K-iteration fixed cost (GFX1250_KITER_FIXED_CLK
  /// above) and the 6-XCD peaks (hardware.hpp) are changed in place.
  /// Scale of the L2 / DRAM read times (L_mem, L2 cycles, DRAM cycles and the floor).
  static constexpr double GFX1250_FIT_MEM_SCALE = 1.1005829744282523;
  /// Scale of the per-K-iteration compute time L_compute (WMMA issue, LDS-read exposure).
  static constexpr double GFX1250_FIT_COMPUTE_SCALE = 0.7766234901999738;
  /// Scale of the load-to-use latencies of the K-loop (L2 and DRAM, flat and tensor-DMA loaded).
  static constexpr double GFX1250_FIT_LATENCY_SCALE = 0.821273602515649;
  /// Epilogue 1/occupancy exposure max(1, S / resident WGs per CU): fitted S, and the S (= EPILOGUE_OCC_SATURATION)
  /// of the base model it was fitted against.  The gfx1250 model has no such factor (it is 1), so the fit is
  /// applied as its ratio, max(1, S_fit / occ) / max(1, S_dev / occ).
  static constexpr double GFX1250_FIT_EPILOGUE_OCC_SATURATION     = 2.5326976116052373;
  static constexpr double GFX1250_FIT_EPILOGUE_OCC_SATURATION_DEV = 2.0;
};

/**
 * @brief StreamK=5 hybrid-mode (SK3 static vs SK4 dynamic) selection thresholds.
 *
 * Fit to measured SK5 on(SK4)/off(SK3) sweeps on MI350X (gfx950); see
 * origami::streamk::select_hybrid_mode() for what each threshold gates and
 * why. Other architectures keep the static (SK3) sub-path until they are
 * tuned in a follow-up PR.
 */
struct streamk_hybrid_defaults_t {
  static constexpr size_t MIN_TILES_FOR_DYNAMIC                 = 480;
  static constexpr int MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC  = 3;
  static constexpr double TILES_PER_CU_THRESHOLD_HIGH_OCCUPANCY = 8.41;
  static constexpr size_t DYNAMIC_MAX_WGS_PER_CU                = 3;
};

/**
 * @brief Structure containing all trainable heuristic parameters.
 *
 * This structure consolidates all empirical constants and weights used in
 * the latency model, making them trainable and configuration-driven.
 */
struct ORIGAMI_EXPORT heuristic_params_t {
  // === Latency Component Weights ===
  double weight_tile_total = heuristic_defaults_t::WEIGHT_TILE_TOTAL;

  // === Main Loop Efficiency ===
  double main_loop_efficiency = heuristic_defaults_t::MAIN_LOOP_EFFICIENCY;

  // === Kernel Rejection ===
  /// When true, the kernel is rejected: its predicted latency is forced to the
  /// maximum so that rank_configs() drops it from selection entirely.
  bool reject = false;

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
struct ORIGAMI_EXPORT heuristic_key_t {
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
  std::optional<bool> subtile;

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
class ORIGAMI_EXPORT heuristics_database_t {
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
   * @brief Add or update a hand-optimized kernel efficiency entry.
   *
   * Writes directly to the fast lookup map. Intended for initialization only.
   */
  void add_hand_optimized_efficiency(hand_optimized_kernel_key_t key, double main_loop_efficiency);

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
ORIGAMI_EXPORT heuristic_key_t make_hand_optimized_kernel_key(hardware_t::architecture_t arch,
                                               data_type_t mi_dtype,
                                               transpose_t transA,
                                               transpose_t transB,
                                               size_t MT_M,
                                               size_t MT_N,
                                               size_t MT_K);

/**
 * @brief Helper to create a key for tile configuration.
 */
ORIGAMI_EXPORT heuristic_key_t make_tile_key(size_t MT_M,
                              size_t MT_N,
                              size_t MT_K,
                              std::optional<transpose_t> transA = std::nullopt,
                              std::optional<transpose_t> transB = std::nullopt);

/**
 * @brief Helper to create a key for architecture/datatype combination.
 */
ORIGAMI_EXPORT heuristic_key_t make_arch_dtype_key(hardware_t::architecture_t arch, data_type_t mi_dtype);

}  // namespace origami
