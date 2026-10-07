// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "features.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tilewright {
namespace detail {
namespace {

double bpe_for_dtype(DataType dt) {
  switch (dt) {
    case DataType::Float:
    case DataType::XFloat32:
    case DataType::Int32: return 4.0;
    case DataType::Double: return 8.0;
    case DataType::Half:
    case DataType::BFloat16: return 2.0;
    case DataType::Float8:
    case DataType::Float8_fnuz:
    case DataType::BFloat8:
    case DataType::BFloat8_fnuz:
    case DataType::Int8:
    case DataType::Float8BFloat8:
    case DataType::BFloat8Float8:
    case DataType::Float8BFloat8_fnuz:
    case DataType::BFloat8Float8_fnuz: return 1.0;
    case DataType::Float6:
    case DataType::BFloat6: return 0.75;
    case DataType::Float4:
    case DataType::Int4: return 0.5;
    default: return 2.0;
  }
}

int dtype_id(DataType dt) {
  switch (dt) {
    case DataType::Float: return 0;
    case DataType::XFloat32: return 1;
    case DataType::Half: return 2;
    case DataType::BFloat16: return 3;
    case DataType::Float8:
    case DataType::Float8_fnuz: return 4;
    case DataType::BFloat8:
    case DataType::BFloat8_fnuz: return 5;
    case DataType::Float6: return 6;
    case DataType::BFloat6: return 7;
    case DataType::Float4: return 8;
    case DataType::Int8: return 9;
    case DataType::Int32: return 10;
    case DataType::Float8BFloat8:
    case DataType::Float8BFloat8_fnuz: return 11;
    case DataType::BFloat8Float8:
    case DataType::BFloat8Float8_fnuz: return 12;
    case DataType::Double: return 13;
    case DataType::Int4: return 14;
    default: return 0;
  }
}

// Operand width used by the feasibility rules; unknown types count as 16 bits.
int dtype_bits_feasible(DataType dt) {
  switch (dt) {
    case DataType::Float:
    case DataType::XFloat32: return 32;
    case DataType::Half:
    case DataType::BFloat16: return 16;
    case DataType::Float8:
    case DataType::Float8_fnuz:
    case DataType::BFloat8:
    case DataType::BFloat8_fnuz: return 8;
    default: return 16;
  }
}

int datatype_to_bits(DataType type) {
  switch (type) {
    case DataType::Float: return 32;
    case DataType::Double: return 64;
    case DataType::ComplexFloat: return 64;
    case DataType::ComplexDouble: return 128;
    case DataType::Half: return 16;
    case DataType::Int8x4: return 32;
    case DataType::Int32: return 32;
    case DataType::BFloat16: return 16;
    case DataType::Int8: return 8;
    case DataType::Int4: return 4;
    case DataType::Int64: return 64;
    case DataType::XFloat32: return 32;
    case DataType::Float8_fnuz: return 8;
    case DataType::BFloat8_fnuz: return 8;
    case DataType::Float8BFloat8_fnuz: return 8;
    case DataType::BFloat8Float8_fnuz: return 8;
    case DataType::Float8: return 8;
    case DataType::BFloat8: return 8;
    case DataType::Float8BFloat8: return 8;
    case DataType::BFloat8Float8: return 8;
    case DataType::Float6: return 6;
    case DataType::BFloat6: return 6;
    case DataType::Float4: return 4;
    default: return -1;
  }
}

double data_type_to_bytes(DataType type) {
  return static_cast<double>(datatype_to_bits(type)) / 8.0;
}

DataType without_fnuz(DataType dt) {
  switch (dt) {
    case DataType::Float8_fnuz: return DataType::Float8;
    case DataType::BFloat8_fnuz: return DataType::BFloat8;
    case DataType::Float8BFloat8_fnuz: return DataType::Float8BFloat8;
    case DataType::BFloat8Float8_fnuz: return DataType::BFloat8Float8;
    default: return dt;
  }
}

double dlog2(double x) { return std::log2(x); }

// int(x) on a non-negative double, as the training pipeline evaluates it.
double is_pow2_d(double x) {
  if (x >= 18446744073709551616.0) return 1.0;
  const std::uint64_t xi = x >= 1.0 ? static_cast<std::uint64_t>(x) : 0;
  return (xi > 0 && (xi & (xi - 1)) == 0) ? 1.0 : 0.0;
}

std::uint64_t mul_saturated(std::uint64_t a, std::uint64_t b) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    return std::numeric_limits<std::uint64_t>::max();
  return a * b;
}

// floor(num_dim * num_bits / max(den_dim * den_bits, 1)) > 5
bool skinny(std::uint64_t num_dim,
            std::uint64_t num_bits,
            std::uint64_t den_dim,
            std::uint64_t den_bits) {
  const std::uint64_t num = mul_saturated(num_dim, num_bits);
  const std::uint64_t den = std::max<std::uint64_t>(mul_saturated(den_dim, den_bits), 1);
  return num / den > 5;
}

}  // namespace

HwView hw_view(const Model& model, const Hardware& h) {
  HwView v;
  v.N_CU           = static_cast<double>(h.N_CU);
  v.LDS            = static_cast<double>(h.lds_capacity);
  v.L2             = static_cast<double>(h.L2_capacity);
  v.parallel_mi_cu = model.parallel_mi_cu;
  v.c0             = model.bw_coef[0];
  v.c1             = model.bw_coef[1];
  v.c2             = model.bw_coef[2];
  return v;
}

double mi_cycles(const Model& model,
                 std::size_t mi_m,
                 std::size_t mi_n,
                 std::size_t mi_k,
                 DataType dt) {
  const std::int32_t key = static_cast<std::int32_t>(without_fnuz(dt));
  for (const MiEntry& e : model.mi_table)
    if (e.m == mi_m && e.n == mi_n && e.k == mi_k && e.dtype == key) return e.cycles;
  return model.mi_default_cycles;
}

KernelShape kernel_shape(const Config& c) {
  KernelShape k;
  k.mt_m          = c.mt.m;
  k.mt_n          = c.mt.n;
  k.mt_k          = c.mt.k;
  k.mk            = c.mt.mk();
  k.nk            = c.mt.nk();
  k.grvw_a        = c.grvw_a;
  k.grvw_b        = c.grvw_b;
  k.cache_hints_a = c.cache_hints_a;
  k.cache_hints_b = c.cache_hints_b;
  k.dot2          = c.mi.m == 1 && c.mi.n == 1 && c.mi.k == 64;
  return k;
}

Feasibility::Feasibility(const Problem& p,
                         const Hardware& hardware,
                         bool nt_a_available,
                         bool nt_b_available) {
  M_                         = p.size.m;
  N_                         = p.size.n;
  const std::uint64_t K      = p.size.k;
  const bool a_trans         = p.a_transpose == Transpose::T;
  const bool b_trans         = p.b_transpose == Transpose::T;
  a_bits_                    = static_cast<std::uint64_t>(dtype_bits_feasible(p.a_dtype));
  const std::uint64_t b_bits = static_cast<std::uint64_t>(dtype_bits_feasible(p.b_dtype));
  lds_valid_                 = datatype_to_bits(p.a_dtype) > 0 && datatype_to_bits(p.b_dtype) > 0;
  bytes_a_                   = data_type_to_bytes(p.a_dtype);
  bytes_b_                   = data_type_to_bytes(p.b_dtype);
  lds_capacity_              = static_cast<double>(hardware.lds_capacity);
  small_batched_             = M_ <= 256 && N_ <= 256 && K < 1024 && p.batch != 1;
  leading_free_a_            = p.a_dtype != DataType::Int8x4 && !a_trans;
  leading_free_b_            = p.a_dtype != DataType::Int8x4 && b_trans;
  k_aligned_                 = (K % 1024) * a_bits_ % 1024 == 0;
  skinny_b_                  = !b_trans && skinny(N_, b_bits, M_, a_bits_);
  skinny_a_                  = a_trans && skinny(M_, a_bits_, N_, b_bits);
  nt_a_                      = nt_a_available;
  nt_b_                      = nt_b_available;
}

bool check_lds_capacity(const Hardware& hardware,
                        const Dim3& mt,
                        DataType a_dtype,
                        DataType b_dtype) {
  Problem p;
  p.a_dtype = a_dtype;
  p.b_dtype = b_dtype;
  KernelShape k;
  k.mk = mt.mk();
  k.nk = mt.nk();
  return Feasibility(p, hardware, false, false).lds_fits(k);
}

namespace {

constexpr std::size_t kV2QueryDim   = 55;
constexpr std::size_t kV2ItemDim    = 12;
constexpr std::size_t kV2InterDim   = 37;
constexpr std::size_t kV2ViewKeyLen = 12;

void build_query_features(const Problem& p, const HwView& hw, float* out) {
  const double m         = static_cast<double>(p.size.m);
  const double n         = static_cast<double>(p.size.n);
  const double k         = static_cast<double>(p.size.k);
  const double b         = static_cast<double>(p.batch);
  const double bpe_a     = bpe_for_dtype(p.a_dtype);
  const double bpe_b     = bpe_for_dtype(p.b_dtype);
  const double bpe_c     = bpe_for_dtype(p.c_dtype);
  const double bpe_d     = bpe_for_dtype(p.d_dtype);
  const double flop_mult = (p.mi_dtype == DataType::XFloat32) ? 3.0 : 1.0;

  const double mn          = m * n;
  const double mk          = m * k;
  const double nk          = n * k;
  const double total_flops = flop_mult * 2.0 * m * n * k * b;
  const double total_bytes = mk * bpe_a * b + nk * bpe_b * b + mn * bpe_c * b + mn * bpe_d * b;
  const double ai_prob     = total_flops / std::max(total_bytes, 1.0);

  std::size_t i = 0;
  // problem (21)
  out[i++] = static_cast<float>(dlog2(std::max(m, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(n, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(k, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(b, 1.0)));
  out[i++] = static_cast<float>(std::min(ai_prob, 1.0e6));
  out[i++] = static_cast<float>(dlog2(std::max(ai_prob, 0.001)));
  out[i++] = static_cast<float>(std::min(std::max(m / std::max(n, 1.0), 0.001), 10000.0));
  out[i++] = static_cast<float>(std::min(std::max(m / std::max(k, 1.0), 0.001), 10000.0));
  out[i++] = static_cast<float>(std::min(std::max(n / std::max(k, 1.0), 0.001), 10000.0));
  out[i++] = static_cast<float>(is_pow2_d(m));
  out[i++] = static_cast<float>(is_pow2_d(n));
  out[i++] = static_cast<float>(is_pow2_d(k));
  out[i++] = static_cast<float>(p.a_transpose == Transpose::T ? 1 : 0);
  out[i++] = static_cast<float>(p.b_transpose == Transpose::T ? 1 : 0);
  out[i++] = static_cast<float>(dtype_id(p.a_dtype));
  out[i++] = static_cast<float>(dtype_id(p.b_dtype));
  out[i++] = static_cast<float>(dtype_id(p.c_dtype));
  out[i++] = static_cast<float>(dtype_id(p.d_dtype));
  out[i++] = static_cast<float>(dtype_id(p.mi_dtype));
  out[i++] = static_cast<float>(bpe_a);
  out[i++] = static_cast<float>(bpe_b);

  // modular (18): every m_mod, then n_mod, then m_align and n_align
  const std::uint64_t mi = p.size.m;
  const std::uint64_t ni = p.size.n;
  for (std::uint64_t base : {256, 128, 64, 32, 16, 8}) out[i++] = static_cast<float>(mi % base);
  for (std::uint64_t base : {256, 128, 64, 32, 16, 8}) out[i++] = static_cast<float>(ni % base);
  for (std::uint64_t base : {64, 128, 256}) {
    const std::uint64_t r = mi % base;
    out[i++] =
        static_cast<float>(static_cast<double>(std::min(r, base - r)) / static_cast<double>(base));
  }
  for (std::uint64_t base : {64, 128, 256}) {
    const std::uint64_t r = ni % base;
    out[i++] =
        static_cast<float>(static_cast<double>(std::min(r, base - r)) / static_cast<double>(base));
  }

  // enum_tiles (16)
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

void build_item_features(const Config& c, float* out) {
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
  // tile (12)
  out[i++] = static_cast<float>(dlog2(mt_m));
  out[i++] = static_cast<float>(dlog2(mt_n));
  out[i++] = static_cast<float>(dlog2(mt_k));
  out[i++] = static_cast<float>(dlog2(mi_m));
  out[i++] = static_cast<float>(dlog2(mi_n));
  out[i++] = static_cast<float>(dlog2(mi_k));
  out[i++] = static_cast<float>(c.cache_hints_a / 7.0);
  out[i++] = static_cast<float>(c.cache_hints_b / 7.0);
  out[i++] = static_cast<float>(occ / 9.0);
  out[i++] = static_cast<float>(grvw_a / 8.0);
  out[i++] = static_cast<float>(grvw_b / 8.0);
  out[i++] = static_cast<float>(gwvw_d / 8.0);
}

// Interaction features in three parts: the values that depend on the config,
// the problem's data types, the device and the model (inter_kernel), those
// that depend on the problem and the device (inter_problem), and the rest
// (inter_pair, which writes the full vector).
constexpr std::size_t kV2InterKernelDim  = 11;
constexpr std::size_t kV2InterProblemDim = 6;

void build_inter_kernel(const Model& model,
                        const Problem& p,
                        const Config& c,
                        const HwView& hw,
                        float* out) {
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
  const double flop_mult = (p.mi_dtype == DataType::XFloat32) ? 3.0 : 1.0;

  const double lds_bytes = mt_m * mt_k * bpe_a + mt_n * mt_k * bpe_b;
  const double lds_ratio = lds_bytes / hw.LDS;
  const double L_MI      = mi_cycles(model,
                                std::max<std::size_t>(c.mi.m, 1),
                                std::max<std::size_t>(c.mi.n, 1),
                                std::max<std::size_t>(c.mi.k, 1),
                                p.mi_dtype) /
                      std::max(hw.parallel_mi_cu, 1.0);
  const double n_mi = std::ceil(mt_m / mi_m) * std::ceil(mt_n / mi_n) * std::ceil(mt_k / mi_k);
  const double L_MT = n_mi * L_MI;
  const double ai_tile =
      (flop_mult * 2.0 * mt_m * mt_n * mt_k) / (mt_m * mt_k + mt_n * mt_k + mt_m * mt_n);

  std::size_t i = 0;
  out[i++]      = static_cast<float>(dlog2(std::max(lds_bytes, 1.0)));
  out[i++]      = static_cast<float>(lds_ratio);
  out[i++]      = static_cast<float>(dlog2(std::max(mt_m * mt_k * bpe_a, 1.0)));
  out[i++]      = static_cast<float>(dlog2(std::max(mt_n * mt_k * bpe_b, 1.0)));
  out[i++]      = static_cast<float>(
      dlog2(std::max((mt_m * mt_n * mt_k * 2.0 * flop_mult) /
                         std::max(mt_m * mt_k * bpe_a + mt_n * mt_k * bpe_b, 1.0),
                     0.001)));
  out[i++] = static_cast<float>(dlog2(std::max(grvw_a * bpe_a, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(grvw_b * bpe_b, 1.0)));
  out[i++] = static_cast<float>(ai_tile);
  out[i++] = static_cast<float>(L_MI);
  out[i++] = static_cast<float>(dlog2(std::max(L_MT, 1.0)));
  out[i++] = std::fmod(mt_k * bpe_a, 128.0) == 0.0 ? 1.0f : 0.0f;
}

void build_inter_problem(const Problem& p, const HwView& hw, float* out) {
  const double m         = static_cast<double>(p.size.m);
  const double n         = static_cast<double>(p.size.n);
  const double k         = static_cast<double>(p.size.k);
  const double b         = static_cast<double>(p.batch);
  const double bpe_a     = bpe_for_dtype(p.a_dtype);
  const double bpe_b     = bpe_for_dtype(p.b_dtype);
  const double bpe_c     = bpe_for_dtype(p.c_dtype);
  const double bpe_d     = bpe_for_dtype(p.d_dtype);
  const double flop_mult = (p.mi_dtype == DataType::XFloat32) ? 3.0 : 1.0;

  const double mn          = m * n;
  const double mk          = m * k;
  const double nk          = n * k;
  const double total_flops = flop_mult * 2.0 * m * n * k * b;
  const double total_bytes = mk * bpe_a * b + nk * bpe_b * b + mn * bpe_c * b + mn * bpe_d * b;

  out[0] = static_cast<float>(std::min(total_bytes / hw.L2, 4.0) / 4.0);
  out[1] = static_cast<float>(dlog2(std::max(total_bytes / hw.N_CU, 1.0)));
  out[2] = static_cast<float>(dlog2(std::max(total_bytes, 1.0)));
  out[3] = static_cast<float>(dlog2(std::max(total_flops, 1.0)));
  out[4] = static_cast<float>((b > 1.0) ? 1.0 : 0.0);
  out[5] = std::fmod(k * bpe_a, 128.0) == 0.0 ? 1.0f : 0.0f;
}

void build_inter_pair(const Problem& p,
                      const Config& c,
                      const HwView& hw,
                      const float* kernel,
                      const float* problem,
                      float* out) {
  const double m    = static_cast<double>(p.size.m);
  const double n    = static_cast<double>(p.size.n);
  const double k    = static_cast<double>(p.size.k);
  const double b    = static_cast<double>(p.batch);
  const double mt_m = static_cast<double>(std::max<std::size_t>(c.mt.m, 1));
  const double mt_n = static_cast<double>(std::max<std::size_t>(c.mt.n, 1));
  const double mt_k = static_cast<double>(std::max<std::size_t>(c.mt.k, 1));

  const double bpe_a = bpe_for_dtype(p.a_dtype);
  const double bpe_b = bpe_for_dtype(p.b_dtype);
  const double N_CU  = hw.N_CU;

  const double mn = m * n;

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

  const double l2_working_set = (nt_m * mt_m * mt_k * bpe_a) + (nt_n * mt_n * mt_k * bpe_b);
  const double l2_fit_ws      = std::min(l2_working_set / hw.L2, 2.0) / 2.0;

  const double active_cus = std::min(num_tiles_total, N_CU);
  const double bw_occ = std::min(1.0, hw.c0 * active_cus * active_cus + hw.c1 * active_cus + hw.c2);

  std::size_t i = 0;
  // interaction (32)
  out[i++] = static_cast<float>(dlog2(std::max(num_tiles, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(num_tiles_total, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(k_iters, 1.0)));
  out[i++] = static_cast<float>(dlog2(std::max(waves, 1.0)));
  out[i++] = static_cast<float>(wave_eff);
  out[i++] = static_cast<float>(dlog2(std::max(rho, 0.001)));
  out[i++] = static_cast<float>(dlog2(std::max(batch_tiles_ratio, 0.001)));
  out[i++] = static_cast<float>(util_out);
  out[i++] = static_cast<float>(util_3d);
  out[i++] = static_cast<float>(std::min((mt_m * mt_n) / std::max(mn, 1.0), 1.0));
  out[i++] = kernel[0];
  out[i++] = kernel[1];
  out[i++] = problem[0];
  out[i++] = static_cast<float>(l2_fit_ws);
  out[i++] = problem[1];
  out[i++] = problem[2];
  out[i++] = problem[3];
  out[i++] = static_cast<float>(std::min(1.0, mt_m / std::max(m, 1.0)));
  out[i++] = static_cast<float>(std::min(1.0, mt_n / std::max(n, 1.0)));
  out[i++] = static_cast<float>((k - (k_iters - 1.0) * mt_k) / std::max(k, 1.0));
  out[i++] = problem[5] != 0.0f && kernel[10] != 0.0f ? 1.0f : 0.0f;
  out[i++] = static_cast<float>((m <= 2.0 * mt_m) ? 1.0 : 0.0);
  out[i++] = static_cast<float>((n <= 2.0 * mt_n) ? 1.0 : 0.0);
  out[i++] = problem[4];
  out[i++] = kernel[2];
  out[i++] = kernel[3];
  out[i++] = kernel[4];
  out[i++] = static_cast<float>(dlog2(std::max(k_iters * num_tiles * b, 1.0)));
  out[i++] = static_cast<float>(b * k_iters / std::max(num_tiles, 1.0));
  out[i++] = static_cast<float>(
      num_tiles_total > 0 ? num_tiles_total / (std::ceil(num_tiles_total / N_CU) * N_CU) : 1.0);
  out[i++] = kernel[5];
  out[i++] = kernel[6];

  // hw_proxies (5)
  out[i++] = kernel[7];
  out[i++] = kernel[8];
  out[i++] = kernel[9];
  out[i++] = static_cast<float>(bw_occ);
  out[i++] = static_cast<float>(active_cus / N_CU);
}

void build_inter_features(const Model& model,
                          const Problem& p,
                          const Config& c,
                          const HwView& hw,
                          float* out) {
  float kernel[kV2InterKernelDim], problem[kV2InterProblemDim];
  build_inter_kernel(model, p, c, hw, kernel);
  build_inter_problem(p, hw, problem);
  build_inter_pair(p, c, hw, kernel, problem, out);
}

std::uint64_t key_value(std::size_t v) { return static_cast<std::uint64_t>(v); }
std::uint64_t key_value(int v) { return static_cast<std::uint64_t>(static_cast<std::int64_t>(v)); }

// Every numeric Config field; attributes and `index` do not enter v2 features.
void v2_view_key(const Config& c, std::uint64_t* out) {
  out[0]  = key_value(c.mt.m);
  out[1]  = key_value(c.mt.n);
  out[2]  = key_value(c.mt.k);
  out[3]  = key_value(c.mi.m);
  out[4]  = key_value(c.mi.n);
  out[5]  = key_value(c.mi.k);
  out[6]  = key_value(c.occupancy);
  out[7]  = key_value(c.cache_hints_a);
  out[8]  = key_value(c.cache_hints_b);
  out[9]  = key_value(c.grvw_a);
  out[10] = key_value(c.grvw_b);
  out[11] = key_value(c.gwvw_d);
}

// v2 models were trained on benchmarks that had the whole device to themselves
// and used the default tile scheduling.
bool v2_supports(const ExecutionContext& context, const Hardware& hardware) {
  return (context.cu_budget == 0 || context.cu_budget >= hardware.N_CU) &&
         context.schedule == Schedule::Default;
}

constexpr FeatureCatalog kCatalogV2 = {"e7fe4b524851e895",
                                       kV2QueryDim,
                                       kV2ItemDim,
                                       kV2InterDim,
                                       kV2ViewKeyLen,
                                       &v2_view_key,
                                       &v2_supports,
                                       nullptr,
                                       0,
                                       &build_query_features,
                                       &build_item_features,
                                       &build_inter_features,
                                       kV2InterKernelDim,
                                       kV2InterProblemDim,
                                       &build_inter_kernel,
                                       &build_inter_problem,
                                       &build_inter_pair};

static_assert(kCatalogV2.view_key_size <= kMaxViewKeySize);

constexpr const FeatureCatalog* kCatalogs[] = {&kCatalogV2};

}  // namespace

const FeatureCatalog* find_catalog(const std::string& hash) noexcept {
  for (const FeatureCatalog* c : kCatalogs)
    if (hash == c->hash) return c;
  return nullptr;
}

const FeatureCatalog& v2_catalog() noexcept { return kCatalogV2; }

bool is_kernel_feasible(const Problem& p,
                        const Config& c,
                        bool nt_a_available,
                        bool nt_b_available) {
  return Feasibility(p, Hardware{}, nt_a_available, nt_b_available).rules_pass(kernel_shape(c));
}

Signature signature_of(const Config& c) {
  return {static_cast<int>(c.mt.m),
          static_cast<int>(c.mt.n),
          static_cast<int>(c.mt.k),
          static_cast<int>(c.mi.m),
          static_cast<int>(c.mi.n),
          static_cast<int>(c.mi.k),
          c.cache_hints_a,
          c.cache_hints_b};
}

}  // namespace detail
}  // namespace tilewright
