// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Feature extractor of Origami Boost: feature schema v4 (c6-4, 161 float64
// columns; f_nt_gate reads the per-macro-tile non-temporal availability on config_t; the
// streamk_launch columns sk_* read the caller's Stream-K launch, kernel_fields_t::streamk).
// Every column outside streamk_launch is a raw problem_t / config_t / kernel field or one of
// Origami's intermediates (gemm::compute_gemm_intermediates, one evaluation per candidate); the
// training feature builder reads the same intermediates through the Python bindings. The only
// arithmetic here is the five log columns (monotone transforms, each language's own log) and the
// enum / bool encodings. A model trained on other Origami sources is refused by the source-hash
// stamp.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "origami/nn/features/gemm_boost.hpp"

#include "origami/gemm.hpp"
#include "origami/gemm_intermediates.hpp"

namespace origami::nn::features::gemm_boost {

using boost::candidate_t;
using boost::feature_schema_t;
using boost::kernel_fields_t;
using boost::streamk_launch_t;
using boost::streamk_partials_t;

// Extractor slots, in schema order.
#define ORIGAMI_BOOST_FEATURES(X) \
  X(f_active_cus)                 \
  X(f_ai_problem)                 \
  X(f_ai_tile)                    \
  X(f_batch)                      \
  X(f_bd_prologue)                \
  X(f_bytes)                      \
  X(f_bytes_per_tile)             \
  X(f_cu_util)                    \
  X(f_cycles_per_tile)            \
  X(f_edge_frac_m)                \
  X(f_edge_frac_n)                \
  X(f_flops)                      \
  X(f_grid_m)                     \
  X(f_grid_n)                     \
  X(f_grvw_bytes_a)               \
  X(f_grvw_bytes_b)               \
  X(f_gwvw_bytes_d)               \
  X(f_k_bits_mod_128B)            \
  X(f_k_iters)                    \
  X(f_l2_hit_a)                   \
  X(f_l2_hit_b)                   \
  X(f_l2_hit_global)              \
  X(f_lds_bytes)                  \
  X(f_lds_frac)                   \
  X(f_lds_occupancy_proxy)        \
  X(f_log_k)                      \
  X(f_log_m)                      \
  X(f_log_n)                      \
  X(f_mainloop_mem_frac)          \
  X(f_mall_hit_a)                 \
  X(f_mall_hit_b)                 \
  X(f_mall_hit_global)            \
  X(f_mem_bw_limited)             \
  X(f_mem_comp_ratio)             \
  X(f_mem_latency)                \
  X(f_mfma_k)                     \
  X(f_mt_aspect)                  \
  X(f_mt_compute_latency)         \
  X(f_mt_k)                       \
  X(f_mt_m)                       \
  X(f_mt_n)                       \
  X(f_mtk_bits_mod_128B)          \
  X(f_nt_gate)                    \
  X(f_num_matrix_instructions)    \
  X(f_num_output_tiles)           \
  X(f_num_timesteps)              \
  X(f_num_wgs)                    \
  X(f_output_tile_bytes)          \
  X(f_perf_gflops)                \
  X(f_prob_aspect)                \
  X(f_prologue_frac)              \
  X(f_quant_k)                    \
  X(f_quant_m)                    \
  X(f_quant_n)                    \
  X(f_quant_total)                \
  X(f_reduction_strategy)         \
  X(f_splitting_factor)           \
  X(f_threads)                    \
  X(f_tile_elements)              \
  X(f_tile_lat_x_tiles_over_pred) \
  X(f_tile_latency)               \
  X(f_tile_schedule)              \
  X(f_tiles_m)                    \
  X(f_tiles_n)                    \
  X(f_tiles_per_cu)               \
  X(f_timestep_latency)           \
  X(f_total_tiles)                \
  X(f_wave_quant_waste)           \
  X(f_wgm_resolved)               \
  X(f_wgmxcc_resolved)            \
  X(f_wgmxccchunk_resolved)       \
  X(f_write_mem_bw_limited)       \
  X(f_local_read_latency)         \
  X(f_local_read_floor)           \
  X(f_local_read_exposed)         \
  X(f_resident_wgs)               \
  X(f_compute_eff_latency)        \
  X(f_work_utilization)           \
  X(f_model_k_iters)              \
  X(f_parallel_reduce)            \
  X(f_parallel_reduce_frac)       \
  X(g1_dtl_a)                     \
  X(g1_dtl_b)                     \
  X(g1_mi_m)                      \
  X(g1_mi_n)                      \
  X(g1_miwg_m)                    \
  X(g1_miwg_n)                    \
  X(g1_miwt_0)                    \
  X(g1_miwt_1)                    \
  X(g1_nt_a)                      \
  X(g1_nt_b)                      \
  X(g1_nt_d)                      \
  X(g1_pgr)                       \
  X(g1_source_swap)               \
  X(g1_vw_a)                      \
  X(g1_vw_b)                      \
  X(g1_waves)                     \
  X(g1_wg_0)                      \
  X(g1_wg_1)                      \
  X(g1_wg_2)                      \
  X(g1_nlc_a)                     \
  X(g1_nlc_b)                     \
  X(d_lds_bytes_eff)              \
  X(d_pgr_vs_kiter)               \
  X(d_pgr_unfilled)               \
  X(d_k_per_lsu)                  \
  X(d_dtl_x_edgem)                \
  X(d_wave_tile_m)                \
  X(d_wave_tile_n)                \
  X(d_wave_aspect)                \
  X(d_waves_per_tile)             \
  X(i_tile_tail)                  \
  X(i_tile_waves)                 \
  X(i_log_kiters)                 \
  X(i_kiter_tail)                 \
  X(i_stream_ratio)               \
  X(i_log_pred)                   \
  X(c_occupancy)                  \
  X(c_ulds_a)                     \
  X(c_ulds_b)                     \
  X(c_ldstr_a)                    \
  X(c_ldstr_b)                    \
  X(c_stream_k_mode)              \
  X(c_lsu)                        \
  X(c_one_lds_buffer)             \
  X(p_trans_a)                    \
  X(p_trans_b)                    \
  X(v2_main_loop_eff)             \
  X(v2_compute_util)              \
  X(v2_num_main_iters)            \
  X(v2_num_ngll_iters)            \
  X(v2_L_main)                    \
  X(v2_L_ngll)                    \
  X(v2_L_nll)                     \
  X(v2_L_tail)                    \
  X(v2_L_pgr_stall)               \
  X(v2_L_loop_overhead)           \
  X(v2_L_single_lds_buffer)       \
  X(v2_depth_waste_ratio)         \
  X(v2_batched_fill_ratio)        \
  X(v2_m_edge_ratio)              \
  X(v2_L_du_waste)                \
  X(v2_L_mainloop)                \
  X(v2_L_lsu_reduce)              \
  X(v2_L_epilogue)                \
  X(v2_epi_occ_exposure)          \
  X(v2_L_mem_l2)                  \
  X(v2_L_mem_mall)                \
  X(v2_L_mem_dram)                \
  X(v2_mainloop_frac)             \
  X(v2_epilogue_frac)             \
  X(v2_du_waste_frac)             \
  X(v2_overhead_frac)             \
  X(v2_mem_l2_share)              \
  X(v2_mem_dram_share)            \
  X(v2_lds_over_mfma)             \
  X(sk_grid_over_tiles)           \
  X(sk_split_tiles_frac)          \
  X(sk_fixup_bytes_over_output)   \
  X(sk_num_wgs_launched)          \
  X(sk_timesteps_launched)

namespace {

enum fid : int {
#define ORIGAMI_BOOST_ENUM(name) name,
  ORIGAMI_BOOST_FEATURES(ORIGAMI_BOOST_ENUM)
#undef ORIGAMI_BOOST_ENUM
      kNumFeatures
};
static_assert(static_cast<std::size_t>(kNumFeatures) == num_columns);

const char* const kNames[] = {
#define ORIGAMI_BOOST_NAME(name) #name,
    ORIGAMI_BOOST_FEATURES(ORIGAMI_BOOST_NAME)
#undef ORIGAMI_BOOST_NAME
};

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

const feature_schema_t& schema() {
  static const feature_schema_t v4{schema_id, feature_names_hash,
                                   std::vector<std::string>(kNames, kNames + kNumFeatures)};
  return v4;
}

const std::vector<std::string>& slot_names() {
  static const std::vector<std::string> names(kNames, kNames + kNumFeatures);
  return names;
}

int slot_index(const std::string& name) {
  static const std::unordered_map<std::string, int> index = [] {
    std::unordered_map<std::string, int> m;
    for (int i = 0; i < kNumFeatures; ++i) m.emplace(kNames[i], i);
    return m;
  }();
  const auto it = index.find(name);
  return it == index.end() ? -1 : it->second;
}

bool extract(const problem_t&   p,
             const hardware_t&  hw,
             const candidate_t& cand,
             double*            f,
             std::string*       why,
             std::uint8_t*      available) {
  std::fill(f, f + kNumFeatures, kNaN);
  if (available) std::fill(available, available + kNumFeatures, std::uint8_t{1});
  if (cand.config == nullptr) {
    if (why) *why = "no config";
    return false;
  }
  const config_t&        c    = *cand.config;
  const tensile_params_t tp   = c.has_tensile_params() ? std::get<tensile_params_t>(c.backend)
                                                       : tensile_params_t{};
  const kernel_fields_t& kf   = cand.fields;
  const double           pred = cand.origami_latency;
  if (!(pred > 0.0) || pred >= std::numeric_limits<double>::max()) {
    if (why) *why = "Origami rejected the candidate";
    return false;
  }

  const gemm::gemm_intermediates_t I   = gemm::compute_gemm_intermediates(p, hw, c, pred);
  const gemm::context_t&           ctx = I.context;
  const gemm::tile_latency_terms_t& t  = I.tile;
  auto num = [](auto v) { return static_cast<double>(v); };
  auto flag = [](bool b) { return b ? 1.0 : 0.0; };

  // ---- problem ----------------------------------------------------------------------------------
  f[f_ai_problem]      = I.arith_intensity;
  f[f_batch]           = I.batch;
  f[f_bytes]           = I.bytes;
  f[f_flops]           = I.flops;
  f[f_k_bits_mod_128B] = num(I.k_bits_mod_cache_line);
  f[f_log_k]           = std::log(std::max(num(p.size.k), 1.0));
  f[f_log_m]           = std::log(std::max(num(p.size.m), 1.0));
  f[f_log_n]           = std::log(std::max(num(p.size.n), 1.0));
  f[f_prob_aspect]     = I.aspect;
  f[f_nt_gate]         = I.cache_hint_gate;
  f[p_trans_a]         = flag(p.a_transpose == transpose_t::T);
  f[p_trans_b]         = flag(p.b_transpose == transpose_t::T);

  // ---- config -----------------------------------------------------------------------------------
  f[f_ai_tile]                 = I.tile_arith_intensity;
  f[f_grvw_bytes_a]            = I.grvw_bytes_a;
  f[f_grvw_bytes_b]            = I.grvw_bytes_b;
  f[f_gwvw_bytes_d]            = I.gwvw_bytes_d;
  f[f_lds_bytes]               = I.lds_bytes;
  f[f_mfma_k]                  = num(c.mi.k);
  f[f_mt_aspect]               = I.mt_aspect;
  f[f_mt_k]                    = num(c.mt.k);
  f[f_mt_m]                    = num(c.mt.m);
  f[f_mt_n]                    = num(c.mt.n);
  f[f_mtk_bits_mod_128B]       = num(I.mt_k_bits_mod_cache_line);
  f[f_num_matrix_instructions] = num(I.num_matrix_instructions);
  f[f_threads]                 = I.threads_per_wg;
  f[g1_dtl_a]                  = flag(tp.direct_to_lds_a);
  f[g1_dtl_b]                  = flag(tp.direct_to_lds_b);
  f[g1_mi_m]                   = num(c.mi.m);
  f[g1_mi_n]                   = num(c.mi.n);
  f[g1_miwg_m]                 = num(ctx.wave_group.m);
  f[g1_miwg_n]                 = num(ctx.wave_group.n);
  f[g1_miwt_0]                 = I.mi_wave_tile_m;
  f[g1_miwt_1]                 = I.mi_wave_tile_n;
  f[g1_nt_a]                   = num(c.cache_hints_a);
  f[g1_nt_b]                   = num(c.cache_hints_b);
  f[g1_nt_d]                   = num(c.cache_hints_d);
  f[g1_pgr]                    = num(tp.prefetch_global_read);
  f[g1_source_swap]            = flag(tp.source_swap);
  f[g1_vw_a]                   = num(c.vector_width_a);
  f[g1_vw_b]                   = num(c.vector_width_b);
  f[g1_waves]                  = I.waves_per_wg;
  f[g1_wg_0]                   = I.workgroup_m;
  f[g1_wg_1]                   = I.workgroup_n;
  f[g1_wg_2]                   = num(ctx.wave_group.k);
  if (kf.num_loads_coalesced_a >= 0) f[g1_nlc_a] = kf.num_loads_coalesced_a;
  else if (available) available[g1_nlc_a] = 0;
  if (kf.num_loads_coalesced_b >= 0) f[g1_nlc_b] = kf.num_loads_coalesced_b;
  else if (available) available[g1_nlc_b] = 0;
  f[c_occupancy]      = I.occupancy;
  f[c_ulds_a]         = num(tp.unroll_major_lds_a);
  f[c_ulds_b]         = num(tp.unroll_major_lds_b);
  f[c_ldstr_a]        = flag(tp.lds_transpose_read_a);
  f[c_ldstr_b]        = flag(tp.lds_transpose_read_b);
  f[c_stream_k_mode]  = num(c.stream_k);
  f[c_lsu]            = num(ctx.wave_group.k);
  f[c_one_lds_buffer] = flag(tp.one_lds_buffer);

  // ---- Origami model state ----------------------------------------------------------------------
  f[f_active_cus]           = num(ctx.active_cus);
  f[f_grid_m]               = num(ctx.grid_m);
  f[f_grid_n]               = num(ctx.grid_n);
  f[f_num_output_tiles]     = num(ctx.num_output_tiles);
  f[f_num_timesteps]        = num(ctx.num_timesteps);
  f[f_num_wgs]              = num(ctx.num_wgs);
  f[f_output_tile_bytes]    = ctx.output_tile_bytes;
  f[f_splitting_factor]     = num(ctx.splitting_factor);
  f[f_tile_elements]        = num(ctx.tile_elements);
  f[f_mem_bw_limited]       = ctx.mem_bw_limited;
  f[f_write_mem_bw_limited] = ctx.write_mem_bw_limited;
  f[f_tile_schedule]        = num(static_cast<std::uint32_t>(ctx.tile_schedule));
  f[f_reduction_strategy]   = ctx.reduction_strategy == reduction_t::none
                                  ? -1.0
                                  : num(static_cast<std::uint32_t>(ctx.reduction_strategy));
  f[f_wgm_resolved]         = num(ctx.wgm.wgm);
  f[f_wgmxcc_resolved]      = num(ctx.wgm.wgmxcc);
  f[f_wgmxccchunk_resolved] = num(ctx.wgm.wgmxccchunk);
  f[f_tiles_m]              = num(ctx.grid_m);
  f[f_tiles_n]              = num(ctx.grid_n);
  f[f_total_tiles]          = num(ctx.num_output_tiles);

  f[f_tile_latency]         = t.L_tile_total;
  f[f_timestep_latency]     = t.L_tile_total;
  f[f_mem_latency]          = t.memory.L_mem;
  f[f_mt_compute_latency]   = t.L_mt_compute;
  f[f_l2_hit_a]             = t.memory.H_mem_l2_A;
  f[f_l2_hit_b]             = t.memory.H_mem_l2_B;
  f[f_mall_hit_a]           = t.memory.H_mem_mall_A;
  f[f_mall_hit_b]           = t.memory.H_mem_mall_B;
  f[f_l2_hit_global]        = I.l2_hit_global;
  f[f_mall_hit_global]      = I.mall_hit_global;
  f[f_perf_gflops]          = I.perf_gflops;
  f[f_local_read_latency]   = t.L_lds_read;
  f[f_local_read_floor]     = t.L_lds_floor;
  f[f_local_read_exposed]   = t.L_local_read;
  f[f_resident_wgs]         = t.wgs_per_cu;
  f[f_compute_eff_latency]  = t.L_compute;
  f[f_work_utilization]     = t.utilization;
  f[f_model_k_iters]        = num(t.k_iters);
  f[f_bd_prologue]          = t.L_prologue;
  f[f_parallel_reduce]      = I.L_parallel_reduce_residual;

  f[v2_main_loop_eff]       = t.eff_scale;
  f[v2_compute_util]        = t.compute_utilization;
  f[v2_num_main_iters]      = num(t.num_main_iters);
  f[v2_num_ngll_iters]      = num(t.num_ngll_iters);
  f[v2_L_main]              = t.L_main;
  f[v2_L_ngll]              = t.L_ngll;
  f[v2_L_nll]               = t.L_nll;
  f[v2_L_tail]              = t.L_tail;
  f[v2_L_pgr_stall]         = t.L_pgr_stall;
  f[v2_L_loop_overhead]     = t.L_loop_overhead;
  f[v2_L_single_lds_buffer] = t.L_single_lds_buffer;
  f[v2_depth_waste_ratio]   = t.depth_waste_ratio;
  f[v2_batched_fill_ratio]  = t.batched_fill_ratio;
  f[v2_m_edge_ratio]        = t.m_edge_ratio;
  f[v2_L_du_waste]          = t.L_du_waste;
  f[v2_L_mainloop]          = t.L_mainloop;
  f[v2_L_lsu_reduce]        = t.L_lsu_reduce;
  f[v2_L_epilogue]          = I.L_epilogue_residual;
  f[v2_epi_occ_exposure]    = t.epi_occ_exposure;
  f[v2_L_mem_l2]            = t.memory.L_mem_l2;
  f[v2_L_mem_mall]          = t.memory.L_mem_mall;
  f[v2_L_mem_dram]          = t.memory.L_mem_dram;

  // ---- derived ----------------------------------------------------------------------------------
  f[f_bytes_per_tile]             = I.bytes_per_tile;
  f[f_cu_util]                    = I.cu_util;
  f[f_cycles_per_tile]            = I.cycles_per_tile;
  f[f_edge_frac_m]                = I.edge_frac_m;
  f[f_edge_frac_n]                = I.edge_frac_n;
  f[f_k_iters]                    = I.k_over_mt_k;
  f[f_lds_frac]                   = I.lds_frac;
  f[f_lds_occupancy_proxy]        = I.lds_fit_count;
  f[f_mainloop_mem_frac]          = I.mainloop_mem_frac;
  f[f_mem_comp_ratio]             = I.mem_comp_ratio;
  f[f_parallel_reduce_frac]       = I.parallel_reduce_frac;
  f[f_prologue_frac]              = I.prologue_frac;
  f[f_quant_k]                    = I.quant_k;
  f[f_quant_m]                    = I.quant_m;
  f[f_quant_n]                    = I.quant_n;
  f[f_quant_total]                = I.quant_total;
  f[f_tile_lat_x_tiles_over_pred] = I.tile_share_of_total;
  f[f_tiles_per_cu]               = I.tiles_per_cu;
  f[f_wave_quant_waste]           = I.wave_quant_waste;
  f[d_lds_bytes_eff]              = I.lds_bytes_buffered;
  f[d_pgr_vs_kiter]               = I.pgr_over_k_iters;
  f[d_pgr_unfilled]               = I.pgr_unfilled;
  f[d_k_per_lsu]                  = I.k_per_lsu;
  f[d_dtl_x_edgem]                = I.dtl_edge_m;
  f[d_wave_tile_m]                = I.wave_tile_m;
  f[d_wave_tile_n]                = I.wave_tile_n;
  f[d_wave_aspect]                = I.wave_aspect;
  f[d_waves_per_tile]             = I.waves_per_tile;
  f[i_tile_tail]                  = I.tile_tail_304;
  f[i_tile_waves]                 = I.tile_waves_304;
  f[i_log_kiters]                 = std::log10(std::max(I.k_over_mt_k, 1.0));
  f[i_kiter_tail]                 = t.tail_fraction;
  f[i_stream_ratio]               = I.k_over_mt_k;
  f[i_log_pred]                   = std::log10(std::max(pred, 1e-9));
  f[v2_mainloop_frac]             = I.mainloop_frac;
  f[v2_epilogue_frac]             = I.epilogue_frac;
  f[v2_du_waste_frac]             = I.du_waste_frac;
  f[v2_overhead_frac]             = I.overhead_frac;
  f[v2_mem_l2_share]              = I.mem_l2_share;
  f[v2_mem_dram_share]            = I.mem_dram_share;
  f[v2_lds_over_mfma]             = I.lds_over_mfma;

  // ---- v4: the Stream-K launch (schema v4 streamk_launch) -------------------------------------
  {
    const double m = num(p.size.m), n = num(p.size.n), batch = I.batch;
    const double bpe_d = datatype_to_bits(p.d_dtype) / 8.0;
    const streamk_launch_t& sk = kf.streamk;
    if (!sk.valid) {
      if (available)
        for (int j : {sk_grid_over_tiles, sk_split_tiles_frac, sk_fixup_bytes_over_output,
                      sk_num_wgs_launched, sk_timesteps_launched})
          available[j] = 0;
    } else if (!sk.is_streamk) {
      f[sk_grid_over_tiles]         = 1.0;
      f[sk_split_tiles_frac]        = 0.0;
      f[sk_fixup_bytes_over_output] = 0.0;
      f[sk_num_wgs_launched]        = 0.0;
      f[sk_timesteps_launched]      = 0.0;
    } else {
      const streamk_partials_t pt    = boost::streamk_partials(sk);
      const double             grid  = static_cast<double>(sk.final_grid);
      const double             tiles = static_cast<double>(sk.tiles);
      const std::uint64_t      cus   = (p.num_cus > 0 && p.num_cus < sk.cu_count)
                                           ? static_cast<std::uint64_t>(p.num_cus)
                                           : sk.cu_count;
      const std::uint64_t slots =
          cus * static_cast<std::uint64_t>(std::max(sk.cu_occupancy, 1));
      const double waves =
          slots ? static_cast<double>((sk.final_grid + slots - 1) / slots) : kNaN;
      const double mt = static_cast<double>(sk.mt_x) * static_cast<double>(sk.mt_y) *
                        static_cast<double>(sk.workspace_size_per_elem_c);
      f[sk_grid_over_tiles]         = grid / tiles;
      f[sk_split_tiles_frac]        = static_cast<double>(pt.split_tiles) / tiles;
      f[sk_fixup_bytes_over_output] =
          ((2.0 * static_cast<double>(pt.partials)) * mt) / (m * n * batch * bpe_d);
      f[sk_num_wgs_launched]   = grid;
      f[sk_timesteps_launched] = waves * tiles / grid;
    }
  }
  return true;
}

}  // namespace origami::nn::features::gemm_boost

namespace origami::nn::boost {

streamk_partials_t streamk_partials(const streamk_launch_t& d) {
  if (!d.is_streamk || d.is_dynamic) return {};
  if (d.reduction == reduction_t::parallel && d.sk_split >= 2)
    return {d.tiles * d.sk_split, d.tiles};
  const std::uint64_t sk = d.dp_only ? 0 : d.sk_tiles, iters = d.iters_per_tile, G = d.final_grid;
  if (G == 0 || sk == 0 || iters == 0) return {};
  const std::uint64_t T = sk * iters, W = T / G, E = T % G;
  // Every workgroup owns whole tiles: no start falls inside one.
  if (E == 0 && W % iters == 0) return {};
  // Walk s_g = g * W + min(g, E) as (tile, offset) = (s_g / iters, s_g % iters) without dividing:
  // s_{g+1} - s_g = W + (g < E).
  streamk_partials_t  out;
  const std::uint64_t Wq = W / iters, Wr = W % iters;
  std::uint64_t       tile = 0, off = 0, last = std::numeric_limits<std::uint64_t>::max();
  for (std::uint64_t g = 1; g < G; ++g) {
    tile += Wq;
    off += Wr + (g - 1 < E ? 1 : 0);
    if (off >= iters) {
      off -= iters;
      ++tile;
    }
    if (off == 0) continue;
    ++out.partials;
    if (tile != last) {
      ++out.split_tiles;
      last = tile;
    }
  }
  return out;
}

}  // namespace origami::nn::boost
