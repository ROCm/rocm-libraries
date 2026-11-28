// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// Bandwidth versus occupancy
// ---------------------------------------------------------------------------

double base_model::occupancy_curve_cus(const hardware_t&, size_t num_active_cus) {
  return static_cast<double>(num_active_cus);
}

// Compute limited achievable memory bandwidth based on active CUs
double compute_mem_bw_from_occupancy(const hardware_t& hardware, size_t num_active_cus) {
  if (num_active_cus > hardware.N_CU) return 1.0;

  const double CUs = for_family(
      hardware, [&](auto family) { return family.occupancy_curve_cus(hardware, num_active_cus); });
  const double bw_limited = std::get<0>(hardware.mem_bw_per_wg_coefficients) * CUs * CUs +
                            std::get<1>(hardware.mem_bw_per_wg_coefficients) * CUs +
                            std::get<2>(hardware.mem_bw_per_wg_coefficients);
  return std::min(bw_limited, 1.0);
}

// ---------------------------------------------------------------------------
// Memory latency
// ---------------------------------------------------------------------------

// Per-operand DRAM bytes one WG pulls for A/B over its K range (transaction-aligned rows).
operand_traffic_t compute_operand_traffic(const problem_t& problem,
                                          const config_t& config,
                                          const context_t& context,
                                          size_t transaction_bytes) {
  const double cl    = static_cast<double>(transaction_bytes);
  const bool a_trans = context.a_transposed;
  const bool b_trans = context.b_transposed;

  const double mt_m_active =
      static_cast<double>(std::min(static_cast<size_t>(problem.size.m), config.mt.m));
  const double mt_n_active =
      static_cast<double>(std::min(static_cast<size_t>(problem.size.n), config.mt.n));
  const double k_per_wg = static_cast<double>(std::max(context.k_per_split, size_t{1}));
  const double mt_k     = static_cast<double>(config.mt.k);

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
double compute_operand_window_bytes(const problem_t& problem,
                                    const config_t& config,
                                    const context_t& context,
                                    size_t k_window,
                                    size_t transaction_bytes) {
  const int a_bits   = context.a_bits;
  const int b_bits   = context.b_bits;
  const bool a_trans = context.a_transposed;
  const bool b_trans = context.b_transposed;

  const size_t Ld_A = a_trans
                          ? config.mt.m * round_elements_to_NB(k_window, a_bits, transaction_bytes)
                          : round_elements_to_NB(config.mt.m, a_bits, transaction_bytes) * k_window;
  const size_t Ld_B = b_trans
                          ? round_elements_to_NB(config.mt.n, b_bits, transaction_bytes) * k_window
                          : config.mt.n * round_elements_to_NB(k_window, b_bits, transaction_bytes);

  return static_cast<double>(Ld_A) * context.a_bytes + static_cast<double>(Ld_B) * context.b_bytes;
}

// Smooth minimum (x^-s + y^-s)^(-1/s), evaluated relative to the smaller value.
double soft_min(double x, double y, double sharpness) {
  const double lo = std::min(x, y);
  const double hi = std::max(x, y);
  if (!(lo > 0.0)) return std::max(lo, 0.0);
  return lo * std::pow(1.0 + std::pow(lo / hi, sharpness), -1.0 / sharpness);
}

double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const context_t& context) {
  return for_family(hardware, [&](auto family) {
    return family.memory(problem, hardware, config, context).L_mem;
  });
}

memory_terms_t base_model::memory(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  const context_t& context) {
  const bool debug = context.debug;

  // Extract parameters from structured types
  const int a_bits = context.a_bits;
  const int b_bits = context.b_bits;

  const size_t num_active_cus = context.active_cus;
  double bw_limited           = context.mem_bw_limited;

  // 1) Estimate per-operand MALL/L2 hit-rates using the analytical model
  const auto [H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B] =
      estimate_cache_hit_rates(problem, hardware, config, context);

  // 2) Total loads per CU per K-iter (A + B, with MX scale bytes).  Same
  // formulation as estimate_cache_hit_rates (whole-tile bytes / K-iters), keeping
  // L_mem in per-K-iter cycles so L_mem_stream = L_mem * num_main_iters is consistent.
  const operand_traffic_t& traffic = context.traffic;
  double Ld_CU_bytes               = traffic.a_iter_bytes + traffic.b_iter_bytes;

  // Block scaled datatypes (MX): add scale bytes
  if (a_bits < 8 && problem.a_mx_block_size != 0)
    Ld_CU_bytes += math::safe_ceil_div(config.mt.mk(), problem.a_mx_block_size);
  if (b_bits < 8 && problem.b_mx_block_size != 0)
    Ld_CU_bytes += math::safe_ceil_div(config.mt.nk(), problem.b_mx_block_size);

  // 3) Total loads by all CUs, split by operand (per K-iter).  Cross-tile
  // cache-line sharing is modeled on load volume (cl_share): when M/N is small,
  // neighbouring tiles pack into the same cache line, cutting unique bytes.
  double Ld_A_total =
      traffic.a_iter_bytes * static_cast<double>(num_active_cus) * traffic.a_cl_share;
  double Ld_B_total =
      traffic.b_iter_bytes * static_cast<double>(num_active_cus) * traffic.b_cl_share;

  double total_Ld = Ld_A_total + Ld_B_total;

  const bool a_nontemporal = config.cache_hints_a > 3;
  const bool b_nontemporal = config.cache_hints_b > 3;
  const bool a_temporal    = !a_nontemporal;
  const bool b_temporal    = !b_nontemporal;

  // 4) L2 latency (bandwidth-limited by CU occupancy ratio)
  double l2_bw = hardware.mem1_perf_ratio * static_cast<double>(num_active_cus) /
                 static_cast<double>(hardware.N_CU);

  // Nontemporal traffic bypasses L2/MALL caching and is charged directly to DRAM.
  double Ld_A_to_l2 = a_temporal ? Ld_A_total : 0.0;
  double Ld_B_to_l2 = b_temporal ? Ld_B_total : 0.0;

  double Ld_l2    = Ld_A_to_l2 + Ld_B_to_l2;
  double L_mem_l2 = (l2_bw > 0) ? (Ld_l2 / l2_bw) : 0.0;

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

  memory_terms_t terms;
  terms.L_mem = L_mem;
  return terms;
}

}  // namespace gemm
}  // namespace origami
