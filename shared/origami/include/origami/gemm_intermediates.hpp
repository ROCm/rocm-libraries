// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

/**
 * @brief Origami's intermediates of one (problem, hardware, config) GEMM prediction.
 *
 * The model state (context, tile and memory latency terms) as compute_total_latency computes it,
 * plus the problem, macro-tile and whole-problem quantities derived from it. Computed on request
 * (compute_gemm_intermediates) for the candidates a caller needs them for; rank_configs never
 * computes them.
 *
 * Doubles follow IEEE double arithmetic in the order each definition is written; integer
 * quantities are exact until converted. "ratio(a, b)" is a / b, NaN when b == 0. bpe_x =
 * datatype_to_bits(x_dtype) / 8. M, N, K, MT_*, MI_* are problem.size and config.mt / config.mi
 * as doubles; batch below is max(problem.batch, 1).
 */
struct ORIGAMI_EXPORT gemm_intermediates_t {
  /// @name Model state
  /// @{
  context_t            context{};  ///< context_t(problem, hardware, config)
  tile_latency_terms_t tile{};     ///< compute_tile_latency_terms(problem, hardware, config, context)
  double l2_hit_global   = 0.0;    ///< estimate_l2_hit
  double mall_hit_global = 0.0;    ///< estimate_mall_hit
  size_t num_matrix_instructions = 0;  ///< compute_number_matrix_instructions(config.mt, config.mi)
  /// @}

  /// @name Problem
  /// @{
  double batch           = 0.0;  ///< max(problem.batch, 1)
  double flops           = 0.0;  ///< 2 * M * N * K * batch
  double bytes           = 0.0;  ///< batch * (M*K*bpe_a + K*N*bpe_b + M*N*bpe_d)
  double arith_intensity = 0.0;  ///< flops / bytes (NaN when bytes == 0)
  double aspect          = 0.0;  ///< M / N (NaN when N == 0)
  /// (K * a_bits) mod 1024: K's misalignment to a 128-B cache line, in bits.
  size_t k_bits_mod_cache_line = 0;
  /// Regime of the cache-hint gate of is_config_feasible: 0 K or MT_K not 128-B aligned;
  /// 1 / 4 non-temporal B wanted (1 the library has one for this macro tile, 4 not);
  /// 2 / 5 non-temporal A wanted (likewise); 3 no non-temporal hint allowed.
  int cache_hint_gate = 0;
  /// @}

  /// @name Macro tile and kernel
  /// @{
  double quant_m     = 0.0;  ///< context.grid_m * MT_M / M: padded fraction of M
  double quant_n     = 0.0;  ///< context.grid_n * MT_N / N
  double quant_k     = 0.0;  ///< ceil(K / MT_K) * MT_K / K
  double quant_total = 0.0;  ///< quant_m * quant_n * quant_k
  double edge_frac_m = 0.0;  ///< 1 / grid_m if M % MT_M != 0 else 0: share of edge tiles in M
  double edge_frac_n = 0.0;  ///< 1 / grid_n if N % MT_N != 0 else 0
  double k_over_mt_k = 0.0;  ///< K / MT_K (fractional K-iterations of the whole problem)
  double mt_aspect   = 0.0;  ///< MT_M / MT_N
  /// 2*MT_M*MT_N*MT_K / (MT_M*MT_K + MT_N*MT_K + MT_M*MT_N), 0 if the denominator is 0: the
  /// arithmetic intensity rank_configs breaks latency ties on.
  double tile_arith_intensity = 0.0;
  size_t mt_k_bits_mod_cache_line = 0;  ///< (MT_K * a_bits) mod 1024
  double lds_bytes          = 0.0;  ///< (MT_M*bpe_a + MT_N*bpe_b) * MT_K: one A+B stage
  double lds_frac           = 0.0;  ///< lds_bytes / hardware.lds_capacity
  double lds_fit_count      = 0.0;  ///< floor(hardware.lds_capacity / lds_bytes)
  double lds_bytes_buffered = 0.0;  ///< (MT_M + MT_N) * MT_K * bpe_a * (1 if one_lds_buffer else 2)
  /// wave_group.m * wave_group.n * wave_group.k * wavefront, wavefront the kernel's
  /// tensile.wavefront_size when set, else hardware.wavefront_size().
  double threads_per_wg = 0.0;
  double grvw_bytes_a   = 0.0;  ///< config.grvw_a * bpe_a: global-read vector bytes of A
  double grvw_bytes_b   = 0.0;  ///< config.grvw_b * bpe_b
  double gwvw_bytes_d   = 0.0;  ///< config.gwvw_d * bpe_d: global-write vector bytes of D
  double occupancy      = 0.0;  ///< max(config.occupancy, 1)
  double waves_per_wg   = 0.0;  ///< wave_group.m * wave_group.n * wave_group.k
  double workgroup_m    = 0.0;  ///< MI_M * wave_group.m (Tensile WorkGroup[0] of MFMA kernels)
  /// floor(wave_group.m * wave_group.n * 64 / workgroup_m), 0 if workgroup_m == 0
  /// (Tensile WorkGroup[1] of MFMA kernels).
  double workgroup_n    = 0.0;
  double mi_wave_tile_m = 0.0;  ///< floor(MT_M / workgroup_m), 0 if workgroup_m == 0 (MIWaveTile[0])
  double mi_wave_tile_n = 0.0;  ///< floor(MT_N / (MI_N * wave_group.n)), 0 if that is 0 (MIWaveTile[1])
  double wave_tile_m    = 0.0;  ///< MT_M / max(workgroup_m, 1)
  double wave_tile_n    = 0.0;  ///< MT_N / max(workgroup_n, 1)
  /// (max(workgroup_m, 1) / max(workgroup_n, 1)) / max(MT_M / max(MT_N, 1), 1e-6): wave-grid
  /// aspect against the tile's.
  double wave_aspect    = 0.0;
  double waves_per_tile = 0.0;  ///< MT_M * MT_N / max(waves_per_wg * 64, 1)
  double pgr_over_k_iters = 0.0;  ///< PGR / max(K / MT_K, 1): can the prefetch depth be filled
  double pgr_unfilled     = 0.0;  ///< max(PGR - K / MT_K, 0)
  double k_per_lsu        = 0.0;  ///< K / wave_group.k
  /// max(DirectToLdsA, DirectToLdsB) * (M mod max(MT_M, 1)) / max(MT_M, 1).
  double dtl_edge_m       = 0.0;
  double tile_waves_304   = 0.0;  ///< num_output_tiles / 304
  double tile_tail_304    = 0.0;  ///< (num_output_tiles mod 304) / 304
  /// @}

  /// @name Whole problem (from total_latency)
  /// @{
  double total_latency = 0.0;  ///< the input: compute_total_latency(problem, hardware, config)
  double perf_gflops   = 0.0;  ///< compute_perf_gflops(hardware, problem, total_latency)
  /// total_latency - KERNEL_LAUNCH_OVERHEAD - tile.L_tile_total * num_timesteps when above
  /// 1e-9 * total_latency, else 0: the parallel-reduction kernel, as a residual of the total.
  double L_parallel_reduce_residual = 0.0;
  /// tile.L_tile_total / tile.weight_tile_total - (L_prologue + L_mainloop + L_lsu_reduce),
  /// without the division when weight_tile_total == 1: the epilogue as a residual of the tile
  /// total (tile.L_epilogue up to rounding).
  double L_epilogue_residual = 0.0;
  double cu_util          = 0.0;  ///< ratio(active_cus, context.n_cu)
  double tiles_per_cu     = 0.0;  ///< ratio(num_output_tiles, active_cus)
  /// ratio(ceil(num_output_tiles / active_cus) * active_cus, num_output_tiles), NaN when either
  /// is 0: last-wave padding of the CU sweep.
  double wave_quant_waste = 0.0;
  double bytes_per_tile   = 0.0;  ///< ratio(bytes, num_output_tiles)
  double cycles_per_tile  = 0.0;  ///< ratio(total_latency, num_output_tiles)
  double tile_share_of_total = 0.0;  ///< ratio(L_tile_total * tiles_per_cu, total_latency)
  double parallel_reduce_frac = 0.0;  ///< ratio(L_parallel_reduce_residual, total_latency)
  double mem_comp_ratio       = 0.0;  ///< ratio(L_mem, L_mt_compute)
  double mainloop_mem_frac    = 0.0;  ///< ratio(L_mem, L_mem + L_compute)
  double prologue_frac        = 0.0;  ///< ratio(L_prologue, L_tile_total)
  double mainloop_frac        = 0.0;  ///< ratio(L_mainloop, L_tile_total)
  double epilogue_frac        = 0.0;  ///< ratio(L_epilogue_residual, L_tile_total)
  double du_waste_frac        = 0.0;  ///< ratio(L_du_waste, L_tile_total)
  /// ratio(L_loop_overhead + L_pgr_stall + L_single_lds_buffer, L_tile_total).
  double overhead_frac        = 0.0;
  double mem_l2_share         = 0.0;  ///< ratio(L_mem_l2, L_mem)
  double mem_dram_share       = 0.0;  ///< ratio(L_mem_dram, L_mem)
  double lds_over_mfma        = 0.0;  ///< ratio(L_lds_read, L_mt_compute)
  /// @}
};

/**
 * @brief Origami's intermediates of one GEMM prediction (@see gemm_intermediates_t).
 *
 * One context_t and one compute_tile_latency_terms evaluation, plus estimate_l2_hit and
 * estimate_mall_hit. Not called by rank_configs or compute_total_latency.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration, as compute_total_latency was given it (including the
 *        non-temporal availability flags).
 * @param total_latency compute_total_latency(problem, hardware, config), e.g. the latency
 *        rank_configs returned for this config.
 * @return gemm_intermediates_t
 */
ORIGAMI_EXPORT gemm_intermediates_t compute_gemm_intermediates(const problem_t&  problem,
                                                               const hardware_t& hardware,
                                                               const config_t&   config,
                                                               double            total_latency);

}  // namespace gemm
}  // namespace origami
