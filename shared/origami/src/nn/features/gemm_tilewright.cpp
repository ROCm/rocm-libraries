// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/features/gemm_tilewright.hpp"

#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace origami::nn::features::gemm_tilewright {
namespace {

double bpe_for_dtype(data_type_t dt) {
  switch (dt) {
    case data_type_t::Float:
    case data_type_t::XFloat32:
    case data_type_t::Int32:
      return 4.0;
    case data_type_t::Double:
      return 8.0;
    case data_type_t::Half:
    case data_type_t::BFloat16:
      return 2.0;
    case data_type_t::Float8:
    case data_type_t::Float8_fnuz:
    case data_type_t::BFloat8:
    case data_type_t::BFloat8_fnuz:
    case data_type_t::Int8:
    case data_type_t::Float8BFloat8:
    case data_type_t::BFloat8Float8:
    case data_type_t::Float8BFloat8_fnuz:
    case data_type_t::BFloat8Float8_fnuz:
      return 1.0;
    case data_type_t::Float6:
    case data_type_t::BFloat6:
      return 0.75;
    case data_type_t::Float4:
    case data_type_t::Int4:
      return 0.5;
    default:
      return 2.0;
  }
}

int dtype_id(data_type_t dt) {
  switch (dt) {
    case data_type_t::Float:
      return 0;
    case data_type_t::XFloat32:
      return 1;
    case data_type_t::Half:
      return 2;
    case data_type_t::BFloat16:
      return 3;
    case data_type_t::Float8:
    case data_type_t::Float8_fnuz:
      return 4;
    case data_type_t::BFloat8:
    case data_type_t::BFloat8_fnuz:
      return 5;
    case data_type_t::Float6:
      return 6;
    case data_type_t::BFloat6:
      return 7;
    case data_type_t::Float4:
      return 8;
    case data_type_t::Int8:
      return 9;
    case data_type_t::Int32:
      return 10;
    case data_type_t::Float8BFloat8:
    case data_type_t::Float8BFloat8_fnuz:
      return 11;
    case data_type_t::BFloat8Float8:
    case data_type_t::BFloat8Float8_fnuz:
      return 12;
    case data_type_t::Double:
      return 13;
    case data_type_t::Int4:
      return 14;
    default:
      return 0;
  }
}

struct hw_view_t {
  double N_CU;
  double LDS;
  double L2;
  double parallel_mi_cu;
  double c0, c1, c2;
};

hw_view_t hw_view(const hardware_t& h) {
  hw_view_t v;
  v.N_CU           = static_cast<double>(h.N_CU);
  v.LDS            = static_cast<double>(h.lds_capacity);
  v.L2             = static_cast<double>(h.L2_capacity);
  v.parallel_mi_cu = static_cast<double>(h.parallel_mi_cu);
  v.c0             = std::get<0>(h.mem_bw_per_wg_coefficients);
  v.c1             = std::get<1>(h.mem_bw_per_wg_coefficients);
  v.c2             = std::get<2>(h.mem_bw_per_wg_coefficients);
  return v;
}

inline double dlog2(double x) { return std::log2(x); }

inline double is_pow2_d(double x) {
  const long long xi = static_cast<long long>(x);
  return (xi > 0 && (xi & (xi - 1)) == 0) ? 1.0 : 0.0;
}

}  // namespace

void build_query(const problem_t& p, const hardware_t& hardware, float* out) {
  const hw_view_t hw = hw_view(hardware);

  const double m         = static_cast<double>(p.size.m);
  const double n         = static_cast<double>(p.size.n);
  const double k         = static_cast<double>(p.size.k);
  const double b         = static_cast<double>(p.batch);
  const double bpe_a     = bpe_for_dtype(p.a_dtype);
  const double bpe_b     = bpe_for_dtype(p.b_dtype);
  const double bpe_c     = bpe_for_dtype(p.c_dtype);
  const double bpe_d     = bpe_for_dtype(p.d_dtype);
  const double flop_mult = (p.mi_dtype == data_type_t::XFloat32) ? 3.0 : 1.0;

  const double mn          = m * n;
  const double mk          = m * k;
  const double nk          = n * k;
  const double total_flops = flop_mult * 2.0 * m * n * k * b;
  const double total_bytes = mk * bpe_a * b + nk * bpe_b * b + mn * bpe_c * b + mn * bpe_d * b;
  const double ai_prob     = total_flops / std::max(total_bytes, 1.0);

  std::size_t i = 0;
  out[i++]      = static_cast<float>(dlog2(std::max(m, 1.0)));
  out[i++]      = static_cast<float>(dlog2(std::max(n, 1.0)));
  out[i++]      = static_cast<float>(dlog2(std::max(k, 1.0)));
  out[i++]      = static_cast<float>(dlog2(std::max(b, 1.0)));
  out[i++]      = static_cast<float>(std::min(ai_prob, 1.0e6));
  out[i++]      = static_cast<float>(dlog2(std::max(ai_prob, 0.001)));
  out[i++]      = static_cast<float>(std::min(std::max(m / std::max(n, 1.0), 0.001), 10000.0));
  out[i++]      = static_cast<float>(std::min(std::max(m / std::max(k, 1.0), 0.001), 10000.0));
  out[i++]      = static_cast<float>(std::min(std::max(n / std::max(k, 1.0), 0.001), 10000.0));
  out[i++]      = static_cast<float>(is_pow2_d(m));
  out[i++]      = static_cast<float>(is_pow2_d(n));
  out[i++]      = static_cast<float>(is_pow2_d(k));
  out[i++]      = static_cast<float>(p.a_transpose == transpose_t::T ? 1 : 0);
  out[i++]      = static_cast<float>(p.b_transpose == transpose_t::T ? 1 : 0);
  out[i++]      = static_cast<float>(dtype_id(p.a_dtype));
  out[i++]      = static_cast<float>(dtype_id(p.b_dtype));
  out[i++]      = static_cast<float>(dtype_id(p.c_dtype));
  out[i++]      = static_cast<float>(dtype_id(p.d_dtype));
  out[i++]      = static_cast<float>(dtype_id(p.mi_dtype));
  out[i++]      = static_cast<float>(bpe_a);
  out[i++]      = static_cast<float>(bpe_b);

  const long long mi_ll = static_cast<long long>(p.size.m);
  const long long ni_ll = static_cast<long long>(p.size.n);
  for (int base : {256, 128, 64, 32, 16, 8}) out[i++] = static_cast<float>(mi_ll % base);
  for (int base : {256, 128, 64, 32, 16, 8}) out[i++] = static_cast<float>(ni_ll % base);
  for (int base : {64, 128, 256}) {
    const long long r = mi_ll % base;
    out[i++]          = static_cast<float>(std::min(r, base - r) / static_cast<double>(base));
  }
  for (int base : {64, 128, 256}) {
    const long long r = ni_ll % base;
    out[i++]          = static_cast<float>(std::min(r, base - r) / static_cast<double>(base));
  }

  for (int st : {32, 64, 128, 256}) {
    const double s_nt_m = std::ceil(m / st);
    const double s_nt_n = std::ceil(n / st);
    out[i++]            = static_cast<float>(dlog2(std::max(s_nt_m * s_nt_n, 1.0)));
  }
  for (int st : {32, 64, 128, 256}) {
    const double s_nt_m = std::ceil(m / st);
    const double s_nt_n = std::ceil(n / st);
    out[i++]            = static_cast<float>(mn / std::max(s_nt_m * st * s_nt_n * st, 1.0));
  }
  for (int kd : {32, 64, 128, 256})
    out[i++] = static_cast<float>(dlog2(std::max(std::ceil(k / kd), 1.0)));
  for (int st : {128, 256}) {
    const double s_nt = std::ceil(m / st) * std::ceil(n / st);
    const double s_w  = std::ceil(s_nt / hw.N_CU);
    out[i++]          = static_cast<float>(dlog2(std::max(s_w, 1.0)));
  }
  for (int st : {128, 256}) {
    const double s_nt = std::ceil(m / st) * std::ceil(n / st);
    const double s_w  = std::ceil(s_nt / hw.N_CU);
    out[i++]          = static_cast<float>(s_nt / std::max(s_w * hw.N_CU, 1.0));
  }
}

void build_item(const config_t& c, float* out) {
  const double mt_m   = static_cast<double>(std::max<std::size_t>(c.mt.m, 1));
  const double mt_n   = static_cast<double>(std::max<std::size_t>(c.mt.n, 1));
  const double mt_k   = static_cast<double>(std::max<std::size_t>(c.mt.k, 1));
  const double mi_m   = static_cast<double>(std::max<std::size_t>(c.mi.m, 1));
  const double mi_n   = static_cast<double>(std::max<std::size_t>(c.mi.n, 1));
  const double mi_k   = static_cast<double>(std::max<std::size_t>(c.mi.k, 1));
  const double occ    = static_cast<double>(std::max<int>(c.occupancy, 1));
  const double grvw_a = static_cast<double>(std::max<std::size_t>(c.grvw_a, 1));
  const double grvw_b = static_cast<double>(std::max<std::size_t>(c.grvw_b, 1));
  const double gwvw_d = static_cast<double>(std::max<std::size_t>(c.gwvw_d, 1));

  std::size_t i = 0;
  out[i++]      = static_cast<float>(dlog2(mt_m));
  out[i++]      = static_cast<float>(dlog2(mt_n));
  out[i++]      = static_cast<float>(dlog2(mt_k));
  out[i++]      = static_cast<float>(dlog2(mi_m));
  out[i++]      = static_cast<float>(dlog2(mi_n));
  out[i++]      = static_cast<float>(dlog2(mi_k));
  out[i++]      = static_cast<float>(c.cache_hints_a / 7.0);
  out[i++]      = static_cast<float>(c.cache_hints_b / 7.0);
  out[i++]      = static_cast<float>(occ / 9.0);
  out[i++]      = static_cast<float>(grvw_a / 8.0);
  out[i++]      = static_cast<float>(grvw_b / 8.0);
  out[i++]      = static_cast<float>(gwvw_d / 8.0);
}

void build_interaction(const problem_t& p,
                       const config_t& c,
                       const hardware_t& hardware,
                       float* out) {
  const hw_view_t hw = hw_view(hardware);

  const double m      = static_cast<double>(p.size.m);
  const double n      = static_cast<double>(p.size.n);
  const double k      = static_cast<double>(p.size.k);
  const double b      = static_cast<double>(p.batch);
  const double mt_m   = static_cast<double>(std::max<std::size_t>(c.mt.m, 1));
  const double mt_n   = static_cast<double>(std::max<std::size_t>(c.mt.n, 1));
  const double mt_k   = static_cast<double>(std::max<std::size_t>(c.mt.k, 1));
  const double mi_m   = static_cast<double>(std::max<std::size_t>(c.mi.m, 1));
  const double mi_n   = static_cast<double>(std::max<std::size_t>(c.mi.n, 1));
  const double mi_k   = static_cast<double>(std::max<std::size_t>(c.mi.k, 1));
  const double grvw_a = static_cast<double>(std::max<std::size_t>(c.grvw_a, 1));
  const double grvw_b = static_cast<double>(std::max<std::size_t>(c.grvw_b, 1));

  const double bpe_a     = bpe_for_dtype(p.a_dtype);
  const double bpe_b     = bpe_for_dtype(p.b_dtype);
  const double bpe_c     = bpe_for_dtype(p.c_dtype);
  const double bpe_d     = bpe_for_dtype(p.d_dtype);
  const double flop_mult = (p.mi_dtype == data_type_t::XFloat32) ? 3.0 : 1.0;
  const double N_CU      = hw.N_CU;

  const double mn          = m * n;
  const double mk          = m * k;
  const double nk          = n * k;
  const double total_flops = flop_mult * 2.0 * m * n * k * b;
  const double total_bytes = mk * bpe_a * b + nk * bpe_b * b + mn * bpe_c * b + mn * bpe_d * b;

  const double nt_m            = std::ceil(m / mt_m);
  const double nt_n            = std::ceil(n / mt_n);
  const double num_tiles       = nt_m * nt_n;
  const double num_tiles_total = num_tiles * b;
  const double k_iters         = std::ceil(k / mt_k);

  const double waves             = std::ceil(num_tiles / N_CU);
  const double wave_eff          = waves > 0 ? num_tiles / (waves * N_CU) : 1.0;
  const double rho               = num_tiles_total / N_CU;
  const double batch_tiles_ratio = b * num_tiles / N_CU;

  const double launched_m = nt_m * mt_m;
  const double launched_n = nt_n * mt_n;
  const double launched_k = k_iters * mt_k;
  const double util_out   = mn / std::max(launched_m * launched_n, 1.0);
  const double util_3d    = (m * n * k) / std::max(launched_m * launched_n * launched_k, 1.0);

  const double lds_bytes      = mt_m * mt_k * bpe_a + mt_n * mt_k * bpe_b;
  const double lds_ratio      = lds_bytes / hw.LDS;
  const double l2_fit_ratio   = total_bytes / hw.L2;
  const double bw_per_cu      = total_bytes / N_CU;
  const double l2_working_set = (nt_m * mt_m * mt_k * bpe_a) + (nt_n * mt_n * mt_k * bpe_b);
  const double l2_fit_ws      = std::min(l2_working_set / hw.L2, 2.0) / 2.0;

  const double L_MI =
      static_cast<double>(hardware.get_mi_latency(
          static_cast<std::size_t>(mi_m),
          static_cast<std::size_t>(mi_n),
          static_cast<std::size_t>(mi_k),
          p.mi_dtype)) /
      std::max(hw.parallel_mi_cu, 1.0);
  const double n_mi = std::ceil(mt_m / mi_m) * std::ceil(mt_n / mi_n) * std::ceil(mt_k / mi_k);
  const double L_MT = n_mi * L_MI;
  const double ai_tile =
      (flop_mult * 2.0 * mt_m * mt_n * mt_k) / (mt_m * mt_k + mt_n * mt_k + mt_m * mt_n);
  const double active_cus = std::min(num_tiles_total, N_CU);
  const double bw_occ     = std::min(1.0, hw.c0 * active_cus * active_cus + hw.c1 * active_cus + hw.c2);

  std::size_t i = 0;
  out[i++]      = static_cast<float>(dlog2(std::max(num_tiles, 1.0)));
  out[i++]      = static_cast<float>(dlog2(std::max(num_tiles_total, 1.0)));
  out[i++]      = static_cast<float>(dlog2(std::max(k_iters, 1.0)));
  out[i++]      = static_cast<float>(dlog2(std::max(waves, 1.0)));
  out[i++]      = static_cast<float>(wave_eff);
  out[i++]      = static_cast<float>(dlog2(std::max(rho, 0.001)));
  out[i++]      = static_cast<float>(dlog2(std::max(batch_tiles_ratio, 0.001)));
  out[i++]      = static_cast<float>(util_out);
  out[i++]      = static_cast<float>(util_3d);
  out[i++]      = static_cast<float>(std::min((mt_m * mt_n) / std::max(mn, 1.0), 1.0));
  out[i++]      = static_cast<float>(dlog2(std::max(lds_bytes, 1.0)));
  out[i++]      = static_cast<float>(lds_ratio);
  out[i++]      = static_cast<float>(std::min(l2_fit_ratio, 4.0) / 4.0);
  out[i++]      = static_cast<float>(l2_fit_ws);
  out[i++]      = static_cast<float>(dlog2(std::max(bw_per_cu, 1.0)));
  out[i++]      = static_cast<float>(dlog2(std::max(total_bytes, 1.0)));
  out[i++]      = static_cast<float>(dlog2(std::max(total_flops, 1.0)));
  out[i++]      = static_cast<float>(std::min(1.0, mt_m / std::max(m, 1.0)));
  out[i++]      = static_cast<float>(std::min(1.0, mt_n / std::max(n, 1.0)));
  out[i++]      = static_cast<float>((k - (k_iters - 1.0) * mt_k) / std::max(k, 1.0));
  out[i++]      = static_cast<float>(
      (std::fmod(k * bpe_a, 128.0) == 0.0 && std::fmod(mt_k * bpe_a, 128.0) == 0.0) ? 1.0 : 0.0);
  out[i++] = static_cast<float>((m <= 2.0 * mt_m) ? 1.0 : 0.0);
  out[i++] = static_cast<float>((n <= 2.0 * mt_n) ? 1.0 : 0.0);
  out[i++] = static_cast<float>((b > 1.0) ? 1.0 : 0.0);
  out[i++] = static_cast<float>(dlog2(std::max(mt_m * mt_k * bpe_a, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(mt_n * mt_k * bpe_b, 1.0)));
  out[i++] = static_cast<float>(
      dlog2(std::max((mt_m * mt_n * mt_k * 2.0 * flop_mult) /
                         std::max(mt_m * mt_k * bpe_a + mt_n * mt_k * bpe_b, 1.0),
                     0.001)));
  out[i++] = static_cast<float>(dlog2(std::max(k_iters * num_tiles * b, 1.0)));
  out[i++] = static_cast<float>(b * k_iters / std::max(num_tiles, 1.0));
  out[i++] = static_cast<float>(
      num_tiles_total > 0 ? num_tiles_total / (std::ceil(num_tiles_total / N_CU) * N_CU) : 1.0);
  out[i++] = static_cast<float>(dlog2(std::max(grvw_a * bpe_a, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(grvw_b * bpe_b, 1.0)));

  out[i++] = static_cast<float>(ai_tile);
  out[i++] = static_cast<float>(L_MI);
  out[i++] = static_cast<float>(dlog2(std::max(L_MT, 1.0)));
  out[i++] = static_cast<float>(bw_occ);
  out[i++] = static_cast<float>(active_cus / N_CU);
}

}  // namespace origami::nn::features::gemm_tilewright
