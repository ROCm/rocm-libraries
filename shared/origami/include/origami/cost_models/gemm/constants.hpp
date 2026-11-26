// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "origami/hardware/variant.hpp"

namespace origami {

// ---------------------------------------------------------------------------
// Shared by every family
// ---------------------------------------------------------------------------

/**
 * @brief Constants of the GEMM model that every architecture family shares.
 *
 * A family with its own values keeps them in its own struct, one value set per chip variant:
 * cdna4_constants_t, cdna5_constants_t and cdna5_fit_t, below.  The W4A16 kernels of gfx1150 and
 * gfx1151 have theirs in w4a16_params_t.
 */
struct heuristic_defaults_t {
  // Latency Component Weights
  static constexpr double WEIGHT_TILE_TOTAL              = 1.0;
  static constexpr size_t CACHE_LINE_BYTES               = 128;
  static constexpr size_t DRAM_SECTOR_BYTES              = 64;
  static constexpr size_t LDS_XFER_BYTES                 = 128;
  static constexpr double KERNEL_LAUNCH_OVERHEAD         = 4.5 * 2200.0;
  static constexpr double POSTGSU_KERNEL_LAUNCH_OVERHEAD = 1.6 * 2200.0;
  static constexpr size_t POSTGSU_THREADS_PER_WG         = 256;
  // Workgroups per CU the dynamic StreamK sub-path (SK4) launches at most.
  static constexpr size_t STREAMK_DYNAMIC_MAX_WGS_PER_CU = 3;

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

  // Model tuning constants
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
};

// ---------------------------------------------------------------------------
// CDNA4 (gfx950)
// ---------------------------------------------------------------------------

/**
 * @brief Constants of the CDNA4 (gfx950) GEMM heuristics for one chip variant.
 *
 * The defaults are the values of the standard part (PCI device id 0x75a0); see
 * CDNA4_CONSTANTS_ID75A8 for the other part and cdna4_constants() for the lookup.
 */
struct cdna4_constants_t {
  /// Arithmetic intensity of an emulated-TF32 problem from which the hand-optimized 256x256x32
  /// kernels take their larger discount.
  double TF32_ARITH_INTENSITY_THRESHOLD = 1000.0;

  /// StreamK=5 hybrid mode (SK3 static vs SK4 dynamic), fit to measured SK5 on(SK4)/off(SK3)
  /// sweeps on MI350X; see cdna4_model::tile_schedule for what each threshold gates and why.
  size_t HYBRID_MIN_TILES_FOR_DYNAMIC                 = 480;
  int HYBRID_MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC  = 3;
  double HYBRID_TILES_PER_CU_THRESHOLD_HIGH_OCCUPANCY = 8.41;
};

// ---------------------------------------------------------------------------
// Value sets, one per chip variant
// ---------------------------------------------------------------------------

/// Constants of the standard part (PCI device id 0x75a0).
inline constexpr cdna4_constants_t CDNA4_CONSTANTS_ID75A0{};

/// Constants of the part with PCI device id 0x75a8.  It starts from the standard values; assign
/// here what the part needs differently.
inline constexpr cdna4_constants_t CDNA4_CONSTANTS_ID75A8 = [] {
  cdna4_constants_t constants = CDNA4_CONSTANTS_ID75A0;
  return constants;
}();

/**
 * @brief Constants of a gfx950 chip variant.
 */
constexpr const cdna4_constants_t& cdna4_constants(variant_t variant) noexcept {
  return variant == variant_t::gfx950_id75a8 ? CDNA4_CONSTANTS_ID75A8 : CDNA4_CONSTANTS_ID75A0;
}

// ---------------------------------------------------------------------------
// CDNA5 (gfx1250)
// ---------------------------------------------------------------------------

/**
 * @brief Measured constants of the CDNA5 (gfx1250) GEMM model for one chip variant.
 *
 * Each is a hardware fact or comes from a microbenchmark of the one mechanism it describes
 * (named with it).  The defaults are the values of the 8-XCD part (256 CUs); see
 * CDNA5_CONSTANTS_6XCD for the other part and cdna5_constants() for the lookup.
 *
 * Units: the model works in engine-clock cycles, so every time is in cycles (_CLK) and every
 * rate in bytes per cycle (_B_PER_CLK), like the constants of heuristic_defaults_t.  The
 * microbenchmarks report wall time on a 2.4 GHz engine clock / 1.9 GHz memory clock part; each
 * value is written as <measured ns> * 2.4 or <measured B/ns> / 2.4 so the measured number stays
 * visible, with the cycle value in the comment.  Nothing here is rescaled by
 * hardware_t::compute_clock_ghz.
 */
struct cdna5_constants_t {
  /// Aligned block DRAM reads and L2 misses move, in bytes.
  size_t DRAM_BLOCK_BYTES = 256;

  /// Tensor-DMA row service rate of one WGP from L2, B/clk, and the minimum bytes a row costs: a
  /// row narrower than the 256 B block L2 serves still costs a whole block.
  double TDM_L2_ROW_B_PER_CLK = 557.0 / 2.4;  // 232.08 B/clk (557 B/ns)

  /// Tensor-DMA row service rate of one WGP from DRAM, B/clk; a row costs at least one 256 B DRAM
  /// block (fitted to a tensor-DMA bandwidth microbenchmark, q fixed at the block).
  double TDM_DRAM_ROW_B_PER_CLK = 420.0 / 2.4;  // 175.0 B/clk (420 B/ns)

  /// Fixed time of one wave's epilogue store instruction, cycles.
  double STORE_INST_CLK = 14.31 * 2.4;  // 34.344 clk (14.31 ns)

  /// Store bandwidth of one wave's instruction stream, B/clk.
  double STORE_WAVE_B_PER_CLK = 29.2 / 2.4;  // 12.167 B/clk (29.2 B/ns)

  /// Store bandwidth one WGP's waves share, B/clk.
  double STORE_WGP_B_PER_CLK = 216.0 / 2.4;  // 90.0 B/clk (216 B/ns)

  /// Device store bandwidth ceiling, B/clk, and the soft-minimum sharpness it is met with.
  double STORE_PEAK_B_PER_CLK = 20950.0 / 2.4;  // 8729.2 B/clk (20950 B/ns)
  double STORE_SHARPNESS      = 2.17;

  /// Bytes landing in LDS with multicast and the cluster barrier over those without, two
  /// tensor loads in flight per wave: L2-resident 37.1 / 32.7 TB/s, DRAM-streaming 17.2 /
  /// 22.2 TB/s (multicast microbenchmark, 4x4 clusters of MT 128, DU 64).
  double MULTICAST_L2_GAIN   = 37.1 / 32.7;
  double MULTICAST_DRAM_GAIN = 17.2 / 22.2;

  /// WGPs of the part hardware_t::mem_bw_per_wg_coefficients was fit on; compute_mem_bw_from_
  /// occupancy rescales the active CU count by this over the part's N_CU.
  double BW_FIT_WGPS = 256.0;

  /// Non-SourceSwap D store (fitted to a store-layout microbenchmark): a wave32 16x16 WMMA
  /// accumulator stored with 16 B lanes writes 32 B (COLUMN_BYTES) into each of 16 columns per
  /// instruction.  Extra time of such an instruction over the SourceSwap one, cycles, and the
  /// device ceiling it reaches, B/clk, for TemporalHintD RT / NT.
  double NONSWAP_STORE_COLUMN_BYTES      = 32.0;
  double NONSWAP_STORE_EXTRA_CLK_RT      = 61.4 * 2.4;    // 147.36 clk
  double NONSWAP_STORE_EXTRA_CLK_NT      = 152.3 * 2.4;   // 365.52 clk
  double NONSWAP_STORE_PEAK_B_PER_CLK_RT = 4674.0 / 2.4;  // 1947.5 B/clk
  double NONSWAP_STORE_PEAK_B_PER_CLK_NT = 1432.0 / 2.4;  // 596.67 B/clk

  /// Sustained DRAM read bandwidth (B/clk) the *STORE_PEAK ceilings were measured at; the
  /// store model scales the ceilings by the part's DRAM peak (hardware_t::mem3_perf_ratio, B/clk)
  /// over this.
  double STORE_REF_DRAM_B_PER_CLK = 17130.0 / 2.4;  // 7137.5 B/clk (17130 B/ns)

  /// Per-lane vector loads (one 16 B load per lane in flight): one WGP's ceiling and the device's
  /// L2-resident peak, B/clk (bandwidth microbenchmark).
  double LANE_LOAD_WGP_B_PER_CLK = 91.0 / 2.4;     // 37.917 B/clk (91 B/ns)
  double LANE_LOAD_L2_B_PER_CLK  = 21100.0 / 2.4;  // 8791.7 B/clk (21100 B/ns)

  /// Interval between back-to-back kernel launches on one stream, cycles, for 1 to 1024 workgroups
  /// (kernel-launch microbenchmark).  Host / driver time, not clocked by the engine.
  double KERNEL_LAUNCH_CLK = 2080.0 * 2.4;  // 4992 clk (2080 ns)

  /// TH_LOAD_NT tensor loads of rows narrower than the 256 B DRAM block: the block's other part is
  /// refetched with probability 1 / (1 + (D0 / D)^n) when D bytes of tensor-DMA traffic reach the
  /// device between the two uses (D0 in MiB; fitted to a tensor-DMA stride microbenchmark).
  double TDM_NT_REFETCH_D0_MIB = 34.2;
  double TDM_NT_REFETCH_N      = 1.6;

  /// Extra time of one wave's edge-path store instruction over the full-width store, measured
  /// inside real BF16 TN kernels (AdaptiveGemm twins on M-remainder grids), cycles: M-edge path by
  /// lane bytes (2, 4, 8, 16 B), and the full-width N-only edge path.
  double EDGE_STORE_EXTRA_CLK_2B  = 94.8 * 2.4;  // 227.52 clk (94.8 ns)
  double EDGE_STORE_EXTRA_CLK_4B  = 71.2 * 2.4;  // 170.88 clk (71.2 ns)
  double EDGE_STORE_EXTRA_CLK_8B  = 47.6 * 2.4;  // 114.24 clk (47.6 ns)
  double EDGE_STORE_EXTRA_CLK_16B = 33.7 * 2.4;  // 80.88 clk (33.7 ns)
  double EDGE_STORE_EXTRA_CLK_N   = 38.6 * 2.4;  // 92.64 clk (38.6 ns)

  /// Soft-maximum sharpness of a TDM + LDS-read + WMMA K-loop's memory and math times, L2-resident
  /// and DRAM-streaming (fitted to a K-loop microbenchmark).
  double KLOOP_L2_OVERLAP   = 1.93;
  double KLOOP_DRAM_OVERLAP = 1.73;

  /// Fixed cost of one K-iteration (two barriers, tensor waits, TDM issue), cycles: intercept of a
  /// K-loop DepthU sweep for 4 waves, which nearly all gfx1250 library kernels use (8 waves
  /// measured 314 clk).  The microbenchmark intercept alone is 207 clk; the value used is refit:
  /// 263.4 clk on the 145-size N-sweep (see cdna5_fit_t below), then x3 on the ranking of
  /// measured bf16 TN candidates.
  double KITER_FIXED_CLK = 790.3177460482018;  // 3 x 263.4 clk

  /// v_wmma_f32_16x16x32_bf16 issue interval per SIMD with 1, 2 and 4 waves on the SIMD, cycles at
  /// the engine clock (WMMA-rate microbenchmark, independent accumulators).
  double WMMA_BF16_CLK = 10.1;  // 1 wave per SIMD; 2 / 4 waves measured 7.5 / 6.05

  /// K-loop slowdown of a multicast cluster over its exact 1x1 twin beyond the microbenchmarked
  /// multicast model: 1 + lambda x k / (k + K0) for k K-iterations, lambda by workgroups along the
  /// cluster's larger axis (2, 4, 8) (fitted to real BBS TN kernels on synthetic shapes).
  /// Model-relative: measured / predicted cluster-to-twin ratio of this model without the term, so
  /// it is refitted from the stored ratios after any change of the K-loop model (here: against the
  /// issue/wait-recurrence stall; rms log 0.140 over 768 cases; K0 = 64 is the last value of the
  /// fit's grid, 48 fits equally).
  double CLUSTER_LOCKSTEP_2  = 0.132;
  double CLUSTER_LOCKSTEP_8  = 0.624;
  double CLUSTER_LOCKSTEP_K0 = 64.0;

  /// Rate at which a WGP's tail loop (K % DepthU != 0, including K < DepthU) moves the whole DepthU
  /// window of A and B, B/clk, with every WGP busy (measured as K = n DU + 8 / + 32 vs n DU).
  double TAIL_WINDOW_B_PER_CLK = 66.8 / 2.4;  // 27.833 B/clk (66.8 B/ns)

  /// Odd K (A and B rows not 4 B aligned): extra time per tile, a constant plus a part per
  /// K-iteration, cycles per KiB of the DepthU window (measured as K = n DU + 1 vs n DU and K = 1).
  double ODD_K_CLK_PER_KIB      = 92.0 * 2.4;  // 220.8 clk/KiB (92.0 ns/KiB)
  double ODD_K_CLK_PER_KIB_ITER = 26.9 * 2.4;  // 64.56 clk/KiB (26.9 ns/KiB)

  /// Cost outside the K-loop that the tile model leaves out, in cycles, as a constant plus growth
  /// per KiB of the output tile and per KiB of one K-iteration of A and B.  Per tile round (slope
  /// of time vs tile rounds of real kernels; the constant fits ~0) and once per kernel (one-wave
  /// intercepts less one round).  The per-kernel cost also grows per doubling of CUOccupancy
  /// (measured with one WG per WGP; the mechanism is not isolated).  Model-relative (measured minus
  /// this model with every fixed term at zero): refitted against the issue/wait-recurrence stall,
  /// which moves the model's one-wave intercepts (per kernel rms 2.4 us, n 405); the per-round fit
  /// (K = 1 and K = DU slopes, rms 2.6 us, n 162) is unchanged by it.  The per-kernel constant and
  /// its two per-KiB parts are then x3 on the ranking of measured bf16 TN candidates: 3 x 569 ns,
  /// 3 x 28.1 ns/KiB of D and 3 x 6.7 ns/KiB of A and B.
  double KERNEL_FIXED_CLK              = 4096.8;
  double KERNEL_FIXED_CLK_PER_KIB_D    = 202.32000000000002;
  double KERNEL_FIXED_CLK_PER_KIB_OP   = 48.24;
  double KERNEL_FIXED_CLK_PER_LOG2_OCC = 680.0 * 2.4;  // 1632 clk
  double TILE_FIXED_CLK_PER_KIB_D      = 15.2 * 2.4;   // 36.48 clk/KiB
  double TILE_FIXED_CLK_PER_KIB_OP     = 16.8 * 2.4;   // 40.32 clk/KiB
};

/**
 * @brief Fitted constants of the CDNA5 (gfx1250) GEMM model for one chip variant.
 *
 * Unlike cdna5_constants_t, these come from GEMM measurements.  The defaults are the fit of the
 * 8-XCD part (256 CUs): a refit of the model's relative-cost constants on the measured top-30
 * candidates of the 145-size (5 shapes x N = 64..512) bf16 TN N-sweep (centered objective, all
 * data).  The fit was made on the base model without the gfx1250 terms, so the values that are
 * multipliers on a model term are applied as such (1.0 = unchanged).  The fit also changed two
 * constants in place: the K-iteration fixed cost (cdna5_constants_t::KITER_FIXED_CLK) and the
 * 6-XCD peaks (hardware/constants.hpp).  The last three scales are fitted to the ranking of
 * measured bf16 TN candidates instead.
 */
struct cdna5_fit_t {
  /// Scale of the L2 / DRAM read times (L_mem, L2 cycles, DRAM cycles and the floor).
  double MEM_SCALE = 1.1005829744282523;

  /// Scale of the per-K-iteration compute time L_compute (WMMA issue, LDS-read exposure).
  double COMPUTE_SCALE = 0.7766234901999738;

  /// Epilogue 1/occupancy exposure max(1, S / resident WGs per CU): fitted S, and the S (=
  /// EPILOGUE_OCC_SATURATION) of the base model it was fitted against.  The gfx1250 model has no
  /// such factor (it is 1), so the fit is applied as its ratio, max(1, S_fit / occ) / max(1, S_dev
  /// / occ).
  double EPILOGUE_OCC_SATURATION     = 2.5326976116052373;
  double EPILOGUE_OCC_SATURATION_DEV = 2.0;

  /// Scales fitted to the ranking of measured bf16 TN candidates (rank correlation within a shape
  /// and the pick's regret), applied on top of the constants and the fit above.  Co-residency:
  /// scale of the busy time each further resident workgroup adds, and the most resident workgroups
  /// per WGP the law is applied to (it was measured for up to 4).
  double RESIDENT_BUSY_SCALE = 2.0;
  double MAX_RESIDENT_WGS    = 2.0;

  /// Scale of the LocalSplitU LDS reduction.
  double LSU_REDUCTION_SCALE = 3.5;
};

// ---------------------------------------------------------------------------
// Value sets, one per chip variant
// ---------------------------------------------------------------------------

/// Measured constants of the 8-XCD part (256 CUs).
inline constexpr cdna5_constants_t CDNA5_CONSTANTS_8XCD{};

/// Measured constants of the 6-XCD part (96 CUs in a DPX partition, 192 unpartitioned).  It
/// starts from the 8-XCD values; assign here what the part measures differently.
inline constexpr cdna5_constants_t CDNA5_CONSTANTS_6XCD = [] {
  cdna5_constants_t constants = CDNA5_CONSTANTS_8XCD;
  return constants;
}();

/// Fitted constants of the 8-XCD part (256 CUs).
inline constexpr cdna5_fit_t CDNA5_FIT_8XCD{};

/// Fitted constants of the 6-XCD part (96 CUs in a DPX partition, 192 unpartitioned).  It
/// starts from the 8-XCD fit; assign here what the part's own fit gives.
inline constexpr cdna5_fit_t CDNA5_FIT_6XCD = [] {
  cdna5_fit_t fit = CDNA5_FIT_8XCD;
  return fit;
}();

/**
 * @brief Measured constants of a gfx1250 chip variant.
 */
constexpr const cdna5_constants_t& cdna5_constants(variant_t variant) noexcept {
  return variant == variant_t::gfx1250_6xcd ? CDNA5_CONSTANTS_6XCD : CDNA5_CONSTANTS_8XCD;
}

/**
 * @brief Fitted constants of a gfx1250 chip variant.
 */
constexpr const cdna5_fit_t& cdna5_fit(variant_t variant) noexcept {
  return variant == variant_t::gfx1250_6xcd ? CDNA5_FIT_6XCD : CDNA5_FIT_8XCD;
}

// ---------------------------------------------------------------------------
// W4A16 (gfx1150 and gfx1151)
// ---------------------------------------------------------------------------

/**
 * @brief W4A16 (int4 A dequantized to a bf16/fp16 MI type) on gfx1150 and gfx1151: constants of
 *        the latency model and of the Stream-K launch grid, fit on hipBLASLt kernel times.
 *
 * The two architectures run the same kernels on the same ISA, so every constant that describes
 * the kernels is the same on both.  What differs is read from hardware_t (CUs, SIMDs per CU,
 * MALL) and from the kernel's CUOccupancy, except the DRAM bandwidth of the decode model.
 */
struct w4a16_params_t {
  // Legend: the tag at the end of each constant says how its value was obtained.
  //   [EMPIRICAL] chosen by sweeping it against measured hipBLASLt kernel times;
  //   [KERNEL]    a property of the kernels of the set, not fitted;
  //   [DEFAULT]   a fallback for a config that does not carry the value, not fitted.
  //   [MEASURED]  measured on the device (the DRAM bandwidths of the decode model, in
  //               w4a16_decode_dram_bytes_per_us).

  // --- Tile latency model ---
  // Dequantize VALU per element, with its dependency stalls (bf16 rounds f32 products, fp16
  // uses packed halves).
  static constexpr double BF16_DEQUANT_VALU_PER_ELEM = 10.0;  // [EMPIRICAL]
  static constexpr double F16_DEQUANT_VALU_PER_ELEM  = 5.0;   // [EMPIRICAL]
  // Cycles per C element of a Stream-K partial tile on the critical path: the workgroup that
  // finishes a tile waits for, reads and adds the f32 partial of each other workgroup.
  // This one depends on the memory system, not only on the kernels: it was fitted on gfx1151,
  // which holds the partials in its MALL.  gfx1150 has no MALL and takes the same value because
  // its picks do not depend on it: from 0 to twice this value none of 632 measured problems
  // changes kernel.
  static constexpr double SK_FIXUP_CYCLES_PER_ELEM = 2.71;  // [EMPIRICAL]

  // --- Stream-K grid (grid_k_split_aware_w4a16) ---
  // Waves of a workgroup when the config does not carry its wave layout.
  static constexpr int DEFAULT_WAVES_PER_WORKGROUP = 4;  // [DEFAULT]
  // Workgroups with this many waves fill a CU (rules 1 and 4).
  static constexpr int FULL_CU_WAVES = 16;  // [KERNEL]
  // Rule 2: K iterations up to which a tile per CU or less stays DP, and up to which it does
  // when the tiles are more than half the CUs (each workgroup then gets over half a tile).
  static constexpr size_t SK_SPLIT_MIN_K_ITERS           = 8;   // [EMPIRICAL]
  static constexpr size_t SK_SPLIT_MIN_K_ITERS_OVER_HALF = 64;  // [EMPIRICAL]
  // Rules 4 and 6: with a MALL to hold the partials, tiles are shared below this many tiles
  // per CU (a single row of token tiles up to it).
  static constexpr size_t SHARED_MAX_TILES_PER_CU = 5;  // [EMPIRICAL]
  // Above the shared range (or without a MALL): DP for K up to this many iterations.
  static constexpr size_t MANY_TILES_DP_MAX_K_ITERS = 16;  // [EMPIRICAL]
  // Rule 4: largest ceil(t) / t that still runs DP: full-CU workgroups, 4-wave (non-token)
  // ones, and the rest.
  static constexpr double DP_MAX_WAVE_ROUNDING_FULL_CU = 1.5;   // [EMPIRICAL]
  static constexpr double DP_MAX_WAVE_ROUNDING_4_WAVES = 1.15;  // [EMPIRICAL]
  static constexpr double DP_MAX_WAVE_ROUNDING         = 1.06;  // [EMPIRICAL]
  // Rule 5: token tiles (MT_N at most this) get one workgroup per SIMD of the CU.
  static constexpr size_t TOKEN_TILE_MAX_N = 48;  // [EMPIRICAL]

  // --- Decode kernels (dot2 WaveSplitK: one wave per workgroup, MT_M weight rows x MT_N
  // tokens, K split over the wave's lanes; see compute_w4a16_decode_latency) ---
  // Cycles of a wave: fixed (start, prologue, reduction, store), and per K iteration (the
  // load-to-use latency of its global reads and the work on them).
  static constexpr double DECODE_WAVE_CYCLES = 5000.0;  // [EMPIRICAL]
  static constexpr double DECODE_ITER_CYCLES = 1500.0;  // [EMPIRICAL]
};

}  // namespace origami
