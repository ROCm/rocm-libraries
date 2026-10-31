// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/gemm_intermediates.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "origami/heuristics.hpp"
#include "origami/origami.hpp"

namespace origami {
namespace gemm {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double ratio(double a, double b) { return b != 0.0 ? a / b : kNaN; }

}  // namespace

gemm_intermediates_t compute_gemm_intermediates(const problem_t&  problem,
                                                const hardware_t& hardware,
                                                const config_t&   config,
                                                double            total_latency) {
  static const tensile_params_t kDefaultTensile{};
  const tensile_params_t&       tp =
      config.has_tensile_params() ? config.tensile() : kDefaultTensile;

  gemm_intermediates_t r;
  r.context                 = context_t(problem, hardware, config);
  const context_t& ctx      = r.context;
  r.tile                    = compute_tile_latency_terms(problem, hardware, config, ctx);
  const auto& t             = r.tile;
  r.l2_hit_global           = estimate_l2_hit(problem, hardware, config, ctx);
  r.mall_hit_global         = estimate_mall_hit(problem, hardware, config, ctx);
  r.num_matrix_instructions = compute_number_matrix_instructions(config.mt, config.mi);

  // ---- problem ----------------------------------------------------------------------------------
  const double M = static_cast<double>(problem.size.m), N = static_cast<double>(problem.size.n),
               K = static_cast<double>(problem.size.k);
  const double batch = static_cast<double>(std::max<std::size_t>(problem.batch, 1));
  const int    a_bits = datatype_to_bits(problem.a_dtype), b_bits = datatype_to_bits(problem.b_dtype),
               d_bits = datatype_to_bits(problem.d_dtype);
  const double bpe_a = a_bits / 8.0, bpe_b = b_bits / 8.0, bpe_d = d_bits / 8.0;
  r.batch           = batch;
  r.flops           = 2.0 * M * N * K * batch;
  r.bytes           = batch * (M * K * bpe_a + K * N * bpe_b + M * N * bpe_d);
  r.arith_intensity = r.bytes != 0.0 ? 2.0 * M * N * K * batch / r.bytes : kNaN;
  r.aspect          = N != 0.0 ? M / N : kNaN;

  constexpr std::size_t cache_line_bits = heuristic_defaults_t::CACHE_LINE_BYTES * 8;
  r.k_bits_mod_cache_line    = (problem.size.k * a_bits) % cache_line_bits;
  r.mt_k_bits_mod_cache_line = (config.mt.k * a_bits) % cache_line_bits;
  {
    const std::size_t ma_bits = std::max(problem.size.m * static_cast<std::size_t>(a_bits),
                                         std::size_t{1});
    const std::size_t nb_bits = std::max(problem.size.n * static_cast<std::size_t>(b_bits),
                                         std::size_t{1});
    if (r.k_bits_mod_cache_line != 0 || r.mt_k_bits_mod_cache_line != 0)
      r.cache_hint_gate = 0;
    else if (problem.size.m <= config.mt.m * 2 && problem.b_transpose == transpose_t::N &&
             nb_bits / ma_bits > 5)
      r.cache_hint_gate = config.non_temporal_b_available ? 1 : 4;
    else if (problem.size.n <= config.mt.n * 2 && problem.a_transpose == transpose_t::T &&
             ma_bits / nb_bits > 5)
      r.cache_hint_gate = config.non_temporal_a_available ? 2 : 5;
    else
      r.cache_hint_gate = 3;
  }

  // ---- macro tile and kernel --------------------------------------------------------------------
  const double MT_M = static_cast<double>(config.mt.m), MT_N = static_cast<double>(config.mt.n),
               MT_K = static_cast<double>(config.mt.k);
  const double grid_m = static_cast<double>(ctx.grid_m), grid_n = static_cast<double>(ctx.grid_n);
  r.quant_m     = M != 0.0 ? grid_m * MT_M / M : kNaN;
  r.quant_n     = N != 0.0 ? grid_n * MT_N / N : kNaN;
  r.quant_k     = (K != 0.0 && MT_K != 0.0) ? std::ceil(K / MT_K) * MT_K / K : kNaN;
  r.quant_total = r.quant_m * r.quant_n * r.quant_k;
  r.edge_frac_m = (grid_m != 0.0 && std::fmod(M, MT_M) != 0.0) ? 1.0 / grid_m : 0.0;
  r.edge_frac_n = (grid_n != 0.0 && std::fmod(N, MT_N) != 0.0) ? 1.0 / grid_n : 0.0;
  r.k_over_mt_k = MT_K != 0.0 ? K / MT_K : kNaN;
  r.mt_aspect   = MT_N != 0.0 ? MT_M / MT_N : kNaN;
  {
    const double traffic   = MT_M * MT_K + MT_N * MT_K + MT_M * MT_N;
    r.tile_arith_intensity = traffic != 0.0 ? 2.0 * MT_M * MT_N * MT_K / traffic : 0.0;
  }
  const double lds_cap = static_cast<double>(hardware.lds_capacity);
  r.lds_bytes          = (MT_M * bpe_a + MT_N * bpe_b) * MT_K;
  r.lds_frac           = lds_cap != 0.0 ? r.lds_bytes / lds_cap : kNaN;
  r.lds_fit_count      = r.lds_bytes != 0.0 ? std::floor(lds_cap / r.lds_bytes) : kNaN;
  r.lds_bytes_buffered = (MT_M + MT_N) * MT_K * bpe_a * (tp.one_lds_buffer ? 1.0 : 2.0);

  const double wg_m = static_cast<double>(ctx.wave_group.m),
               wg_n = static_cast<double>(ctx.wave_group.n),
               lsu  = static_cast<double>(ctx.wave_group.k);
  const double wavefront = tp.wavefront_size > 0 ? static_cast<double>(tp.wavefront_size)
                                                 : static_cast<double>(hardware.wavefront_size());
  r.threads_per_wg = wg_m * wg_n * lsu * wavefront;
  r.grvw_bytes_a   = static_cast<double>(config.grvw_a) * bpe_a;
  r.grvw_bytes_b   = static_cast<double>(config.grvw_b) * bpe_b;
  r.gwvw_bytes_d   = static_cast<double>(config.gwvw_d) * bpe_d;
  r.occupancy      = static_cast<double>(std::max(config.occupancy, 1));

  const double MI_M = static_cast<double>(config.mi.m), MI_N = static_cast<double>(config.mi.n);
  r.waves_per_wg   = wg_m * wg_n * lsu;
  r.workgroup_m    = MI_M * wg_m;
  r.workgroup_n    = r.workgroup_m > 0.0 ? std::floor(wg_m * wg_n * 64.0 / r.workgroup_m) : 0.0;
  r.mi_wave_tile_m = MI_M * wg_m > 0.0 ? std::floor(MT_M / (MI_M * wg_m)) : 0.0;
  r.mi_wave_tile_n = MI_N * wg_n > 0.0 ? std::floor(MT_N / (MI_N * wg_n)) : 0.0;
  {
    const double wg0 = std::max(r.workgroup_m, 1.0), wg1 = std::max(r.workgroup_n, 1.0);
    r.wave_tile_m    = MT_M / wg0;
    r.wave_tile_n    = MT_N / wg1;
    r.wave_aspect    = (wg0 / wg1) / std::max(MT_M / std::max(MT_N, 1.0), 1e-6);
    r.waves_per_tile = (MT_M * MT_N) / std::max(r.waves_per_wg * 64.0, 1.0);
  }
  {
    const double pgr     = static_cast<double>(tp.prefetch_global_read);
    const double k_iters = r.k_over_mt_k;
    r.pgr_over_k_iters   = pgr / std::max(k_iters, 1.0);
    r.pgr_unfilled       = std::max(pgr - k_iters, 0.0);
    r.k_per_lsu          = K / lsu;
    const double dtl     = std::max(tp.direct_to_lds_a ? 1.0 : 0.0, tp.direct_to_lds_b ? 1.0 : 0.0);
    r.dtl_edge_m         = dtl * (std::fmod(M, std::max(MT_M, 1.0)) / std::max(MT_M, 1.0));
  }
  {
    const double tiles = static_cast<double>(ctx.num_output_tiles);
    r.tile_tail_304    = std::fmod(tiles, 304.0) / 304.0;
    r.tile_waves_304   = tiles / 304.0;
  }

  // ---- whole problem ----------------------------------------------------------------------------
  r.total_latency = total_latency;
  r.perf_gflops   = compute_perf_gflops(hardware, problem, total_latency);
  {
    const double reduce = total_latency - heuristic_defaults_t::KERNEL_LAUNCH_OVERHEAD -
                          t.L_tile_total * static_cast<double>(ctx.num_timesteps);
    r.L_parallel_reduce_residual = reduce > 1e-9 * total_latency ? reduce : 0.0;
  }
  r.L_epilogue_residual =
      (t.weight_tile_total != 1.0)
          ? t.L_tile_total / t.weight_tile_total - (t.L_prologue + t.L_mainloop + t.L_lsu_reduce)
          : t.L_tile_total - (t.L_prologue + t.L_mainloop + t.L_lsu_reduce);
  {
    const double tiles = static_cast<double>(ctx.num_output_tiles),
                 cus   = static_cast<double>(ctx.active_cus);
    r.cu_util          = ratio(cus, static_cast<double>(ctx.n_cu));
    r.tiles_per_cu     = ratio(tiles, cus);
    r.wave_quant_waste =
        (cus != 0.0 && tiles != 0.0) ? ratio(std::ceil(tiles / cus) * cus, tiles) : kNaN;
    r.bytes_per_tile      = ratio(r.bytes, tiles);
    r.cycles_per_tile     = ratio(total_latency, tiles);
    r.tile_share_of_total = ratio(t.L_tile_total * r.tiles_per_cu, total_latency);
  }
  const double L_mem       = t.memory.L_mem;
  r.parallel_reduce_frac   = ratio(r.L_parallel_reduce_residual, total_latency);
  r.mem_comp_ratio         = ratio(L_mem, t.L_mt_compute);
  r.mainloop_mem_frac      = ratio(L_mem, L_mem + t.L_compute);
  r.prologue_frac          = ratio(t.L_prologue, t.L_tile_total);
  r.mainloop_frac          = ratio(t.L_mainloop, t.L_tile_total);
  r.epilogue_frac          = ratio(r.L_epilogue_residual, t.L_tile_total);
  r.du_waste_frac          = ratio(t.L_du_waste, t.L_tile_total);
  r.overhead_frac =
      ratio(t.L_loop_overhead + t.L_pgr_stall + t.L_single_lds_buffer, t.L_tile_total);
  r.mem_l2_share   = ratio(t.memory.L_mem_l2, L_mem);
  r.mem_dram_share = ratio(t.memory.L_mem_dram, L_mem);
  r.lds_over_mfma  = ratio(t.L_lds_read, t.L_mt_compute);
  return r;
}

}  // namespace gemm
}  // namespace origami
