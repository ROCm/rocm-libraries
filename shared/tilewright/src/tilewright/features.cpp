// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "features.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

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

// static_cast<float>(dlog2(n)) of the integers n below kCountLog2s; the
// interaction features' counts of tiles, iterations, waves and workgroups
// are mostly such integers.
constexpr std::size_t kCountLog2s = 2048;

const std::array<float, kCountLog2s>& count_log2s() {
  static const std::array<float, kCountLog2s> table = [] {
    std::array<float, kCountLog2s> t{};
    for (std::size_t n = 1; n < kCountLog2s; ++n)
      t[n] = static_cast<float>(dlog2(static_cast<double>(n)));
    return t;
  }();
  return table;
}

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

float count_log2(double x) {
  const double v = std::max(x, 1.0);
  if (v < static_cast<double>(kCountLog2s)) {
    const auto n = static_cast<std::size_t>(v);
    if (static_cast<double>(n) == v) return count_log2s()[n];
  }
  return static_cast<float>(dlog2(v));
}

std::size_t effective_cus(const ExecutionContext& context, const Hardware& hardware) noexcept {
  return context.cu_budget > 0 && context.cu_budget < hardware.N_CU ? context.cu_budget
                                                                    : hardware.N_CU;
}

HwView hw_view(const Model& model, const Hardware& h, const ExecutionContext& context) {
  HwView v;
  v.N_CU  = static_cast<double>(h.N_CU);
  v.N_eff = model.format_version >= 3 ? static_cast<double>(effective_cus(context, h)) : v.N_CU;
  v.LDS   = static_cast<double>(h.lds_capacity);
  v.L2    = static_cast<double>(h.L2_capacity);
  v.parallel_mi_cu = model.parallel_mi_cu;
  v.c0             = model.bw_coef[0];
  v.c1             = model.bw_coef[1];
  v.c2             = model.bw_coef[2];
  v.schedule       = context.schedule;
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
                         bool nt_b_available,
                         const SoftRules& soft) {
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
  small_batched_ =
      soft.small_batched_one_tile && M_ <= 256 && N_ <= 256 && K < 1024 && p.batch != 1;
  dot2_rule_         = soft.dot2_small_m;
  leading_free_a_    = soft.leading_free && p.a_dtype != DataType::Int8x4 && !a_trans;
  leading_free_b_    = soft.leading_free && p.a_dtype != DataType::Int8x4 && b_trans;
  non_temporal_rule_ = soft.non_temporal;
  k_aligned_         = (K % 1024) * a_bits_ % 1024 == 0;
  skinny_b_          = !b_trans && skinny(N_, b_bits, M_, a_bits_);
  skinny_a_          = a_trans && skinny(M_, a_bits_, N_, b_bits);
  nt_a_              = nt_a_available;
  nt_b_              = nt_b_available;
}

bool never_feasible(const CellBounds& b,
                    const KernelShape& k,
                    bool nt_a_available,
                    bool nt_b_available,
                    const SoftRules& soft) {
  if (!b.reached) return false;
  if (soft.dot2_small_m && k.dot2 && b.m_lo > 2) return true;
  if (soft.small_batched_one_tile && !b.batch_one && b.m_hi <= 256 && b.n_hi <= 256 &&
      b.k_hi < 1024 && (k.mt_m < b.m_lo || k.mt_n < b.n_lo))
    return true;
  if (!soft.non_temporal || (k.cache_hints_a == 0 && k.cache_hints_b == 0)) return false;
  // A hinted kernel passes only through the skinny branches of rules_pass.
  const auto half = [](std::uint64_t v) { return (v >> 1) + (v & 1); };
  const bool no_b = half(b.m_lo) > k.mt_m || (nt_b_available && k.cache_hints_b != 4);
  const bool no_a = half(b.n_lo) > k.mt_n || (nt_a_available && k.cache_hints_a != 4);
  return no_a && no_b;
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
  return Feasibility(p, hardware, false, false, SoftRules{}).lds_fits(k);
}

namespace {

constexpr std::size_t kV2QueryDim = 55;
constexpr std::size_t kV2ItemDim  = 12;
constexpr std::size_t kV2InterDim = 37;

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
    const double s_w  = std::ceil(s_nt / hw.N_eff);
    out[i++]          = static_cast<float>(dlog2(std::max(s_w, 1.0)));
  }
  for (int st : {128, 256}) {
    const double s_nt = std::ceil(m / st) * std::ceil(n / st);
    const double s_w  = std::ceil(s_nt / hw.N_eff);
    out[i++]          = static_cast<float>(s_nt / std::max(s_w * hw.N_eff, 1.0));
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
  out[1] = static_cast<float>(dlog2(std::max(total_bytes / hw.N_eff, 1.0)));
  out[2] = static_cast<float>(dlog2(std::max(total_bytes, 1.0)));
  out[3] = static_cast<float>(dlog2(std::max(total_flops, 1.0)));
  out[4] = static_cast<float>((b > 1.0) ? 1.0 : 0.0);
  out[5] = std::fmod(k * bpe_a, 128.0) == 0.0 ? 1.0f : 0.0f;
}

// Writes the interaction features before `bw_occ` (35 values) and returns
// min(tiles of the whole grid, N_eff).
double build_inter_pair_common(const Problem& p,
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
  const double N_CU  = hw.N_eff;

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

  std::size_t i = 0;
  // interaction (32)
  out[i++] = count_log2(num_tiles);
  out[i++] = count_log2(num_tiles_total);
  out[i++] = count_log2(k_iters);
  out[i++] = count_log2(waves);
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
  out[i++] = count_log2(k_iters * num_tiles * b);
  out[i++] = static_cast<float>(b * k_iters / std::max(num_tiles, 1.0));
  out[i++] = static_cast<float>(
      num_tiles_total > 0 ? num_tiles_total / (std::ceil(num_tiles_total / N_CU) * N_CU) : 1.0);
  out[i++] = kernel[5];
  out[i++] = kernel[6];

  // hw_proxies
  out[i++] = kernel[7];
  out[i++] = kernel[8];
  out[i++] = kernel[9];
  return std::min(num_tiles_total, N_CU);
}

constexpr std::size_t kV2PairCommon = 35;

void build_inter_pair(const Problem& p,
                      const Config& c,
                      const std::int64_t*,
                      const Model&,
                      const HwView& hw,
                      const float* kernel,
                      const float* problem,
                      const LaunchPlan&,
                      float* out) {
  const double active_cus = build_inter_pair_common(p, c, hw, kernel, problem, out);
  const double bw_occ = std::min(1.0, hw.c0 * active_cus * active_cus + hw.c1 * active_cus + hw.c2);
  out[kV2PairCommon]  = static_cast<float>(bw_occ);
  out[kV2PairCommon + 1] = static_cast<float>(active_cus / hw.N_eff);
}

void build_inter_features(const Model& model,
                          const Problem& p,
                          const Config& c,
                          const std::int64_t* attrs,
                          const HwView& hw,
                          const LaunchPlan& plan,
                          float* out) {
  float kernel[kV2InterKernelDim], problem[kV2InterProblemDim];
  build_inter_kernel(model, p, c, hw, kernel);
  build_inter_problem(p, hw, problem);
  build_inter_pair(p, c, attrs, model, hw, kernel, problem, plan, out);
}

void v2_item_features(const Model&, const Config& c, const std::int64_t*, float* out) {
  build_item_features(c, out);
}

std::uint64_t key_value(std::size_t v) { return static_cast<std::uint64_t>(v); }
std::uint64_t key_value(int v) { return static_cast<std::uint64_t>(static_cast<std::int64_t>(v)); }

void config_view_key(const Config& c, std::uint64_t* out) {
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

// Every numeric Config field; attributes and `index` do not enter v2 features.
void v2_view_key(const Model&, const Config& c, const std::int64_t*, std::uint64_t* out) {
  config_view_key(c, out);
}

// v2 models were trained on benchmarks that had the whole device to themselves
// and used the default tile scheduling.
bool v2_supports(const Model&, const ExecutionContext& context, const Hardware& hardware) {
  return (context.cu_budget == 0 || context.cu_budget >= hardware.N_CU) &&
         context.schedule == Schedule::Default;
}

constexpr FeatureCatalog kCatalogV2 = {"e7fe4b524851e895",
                                       2,
                                       kV2QueryDim,
                                       kV2ItemDim,
                                       kV2InterDim,
                                       false,
                                       &v2_view_key,
                                       &v2_supports,
                                       &build_query_features,
                                       &v2_item_features,
                                       &build_inter_features,
                                       kV2InterKernelDim,
                                       kV2InterProblemDim,
                                       &build_inter_kernel,
                                       &build_inter_problem,
                                       &build_inter_pair};

// ── MLREC_v3 ────────────────────────────────────────────────────────────────
// Features of the kernel signature (signature_of), the problem and the
// execution context, with the usable CUs (N_eff) in place of the device's,
// plus the kernel's launch (LaunchPlan) and the model's attribute features.
// Unlike v2 they read no occupancy or vector width, and have no bw_occ.

constexpr std::size_t kV3QueryDim = kV2QueryDim + 4;
constexpr std::size_t kV3ItemDim  = 8;
// v2's pair features before bw_occ, less log2_grvw_a_bytes and
// log2_grvw_b_bytes at kV2PairGrvw and kV2PairGrvw + 1.
constexpr std::size_t kV2PairGrvw   = 30;
constexpr std::size_t kV3PairCommon = kV2PairCommon - 2;
constexpr std::size_t kV3InterDim   = kV3PairCommon + 11;

void v3_query_features(const Problem& p, const HwView& hw, float* out) {
  build_query_features(p, hw, out);
  out[kV2QueryDim]     = static_cast<float>(hw.N_eff / hw.N_CU);
  out[kV2QueryDim + 1] = hw.schedule == Schedule::Default ? 1.0f : 0.0f;
  out[kV2QueryDim + 2] = hw.schedule == Schedule::Dynamic ? 1.0f : 0.0f;
  out[kV2QueryDim + 3] = hw.schedule == Schedule::Auto ? 1.0f : 0.0f;
}

void v3_item_features(const Model& m, const Config& c, const std::int64_t* attrs, float* out) {
  float tile[kV2ItemDim];
  build_item_features(c, tile);
  std::copy(tile, tile + kV3ItemDim, out);
  std::size_t i = kV3ItemDim;
  for (const AttributeFeature& f : m.item_attributes) {
    const std::int64_t x = attrs[f.read];
    switch (f.transform) {
      case AttributeTransform::Identity:
        out[i++] = static_cast<float>(static_cast<double>(x));
        break;
      case AttributeTransform::Log2p1:
        out[i++] = static_cast<float>(dlog2(1.0 + std::max(static_cast<double>(x), 0.0)));
        break;
      case AttributeTransform::Nonzero: out[i++] = x != 0 ? 1.0f : 0.0f; break;
      case AttributeTransform::OneHot:
        for (std::int64_t v : f.values) out[i++] = x == v ? 1.0f : 0.0f;
        break;
    }
  }
}

std::uint64_t ceil_div(std::uint64_t n, std::uint64_t d) {
  return d == 0 ? 0 : n / d + (n % d != 0 ? 1 : 0);
}

void v3_inter_pair(const Problem& p,
                   const Config& c,
                   const std::int64_t*,
                   const Model&,
                   const HwView& hw,
                   const float* kernel,
                   const float* problem,
                   const LaunchPlan& plan,
                   float* out) {
  float common[kV2PairCommon];
  const double active_cus = build_inter_pair_common(p, c, hw, kernel, problem, common);
  std::copy(common, common + kV2PairGrvw, out);
  std::copy(common + kV2PairGrvw + 2, common + kV2PairCommon, out + kV2PairGrvw);

  const std::uint64_t tiles = ceil_div(p.size.m, std::max<std::size_t>(c.mt.m, 1)) *
                              ceil_div(p.size.n, std::max<std::size_t>(c.mt.n, 1)) * p.batch;
  const std::uint64_t wgs  = plan.grid_wgs > 0 ? plan.grid_wgs : tiles;
  const double grid        = static_cast<double>(wgs);
  const double waves       = std::ceil(grid / hw.N_eff);
  const PartialReduction r = plan.partial_reduction;
  std::size_t i            = kV3PairCommon;
  out[i++]                 = static_cast<float>(active_cus / hw.N_eff);
  out[i++]                 = plan.dynamic_queue ? 1.0f : 0.0f;
  out[i++]                 = count_log2(grid);
  out[i++]                 = static_cast<float>(dlog2(std::max(grid / hw.N_eff, 0.001)));
  out[i++]                 = count_log2(waves);
  out[i++]                 = static_cast<float>(waves > 0 ? grid / (waves * hw.N_eff) : 1.0);
  out[i++] =
      static_cast<float>(dlog2(std::max(static_cast<double>(tiles) / std::max(grid, 1.0), 0.001)));
  out[i++] = wgs > 0 && tiles % wgs != 0 ? 1.0f : 0.0f;
  out[i++] = r == PartialReduction::Tree ? 1.0f : 0.0f;
  out[i++] = r == PartialReduction::Parallel ? 1.0f : 0.0f;
  out[i++] = r == PartialReduction::Atomic ? 1.0f : 0.0f;
}

void v3_inter_features(const Model& model,
                       const Problem& p,
                       const Config& c,
                       const std::int64_t* attrs,
                       const HwView& hw,
                       const LaunchPlan& plan,
                       float* out) {
  float kernel[kV2InterKernelDim], problem[kV2InterProblemDim];
  build_inter_kernel(model, p, c, hw, kernel);
  build_inter_problem(p, hw, problem);
  v3_inter_pair(p, c, attrs, model, hw, kernel, problem, plan, out);
}

// The signature, then every attribute the model reads.
void v3_view_key(const Model& m, const Config& c, const std::int64_t* attrs, std::uint64_t* out) {
  out[0] = key_value(c.mt.m);
  out[1] = key_value(c.mt.n);
  out[2] = key_value(c.mt.k);
  out[3] = key_value(c.mi.m);
  out[4] = key_value(c.mi.n);
  out[5] = key_value(c.mi.k);
  out[6] = key_value(c.cache_hints_a);
  out[7] = key_value(c.cache_hints_b);
  for (std::size_t j = 0; j < m.attribute_reads.size(); ++j)
    out[kV3ViewKeyFields + j] = static_cast<std::uint64_t>(attrs[j]);
}

bool v3_supports(const Model& m, const ExecutionContext& context, const Hardware& hardware) {
  if (hardware.N_CU == 0) return false;
  const unsigned s = static_cast<unsigned>(context.schedule);
  if (s > 2 || ((m.schedules >> s) & 1u) == 0) return false;
  const double fraction =
      static_cast<double>(effective_cus(context, hardware)) / static_cast<double>(hardware.N_CU);
  return fraction >= m.min_cu_fraction && fraction <= m.max_cu_fraction;
}

constexpr FeatureCatalog kCatalogV3 = {"b8b9f4a518add96f",
                                       3,
                                       kV3QueryDim,
                                       kV3ItemDim,
                                       kV3InterDim,
                                       true,
                                       &v3_view_key,
                                       &v3_supports,
                                       &v3_query_features,
                                       &v3_item_features,
                                       &v3_inter_features,
                                       kV2InterKernelDim,
                                       kV2InterProblemDim,
                                       &build_inter_kernel,
                                       &build_inter_problem,
                                       &v3_inter_pair};

constexpr const FeatureCatalog* kCatalogs[] = {&kCatalogV2, &kCatalogV3};

constexpr std::uint64_t kFnvOffset = 0xCBF29CE484222325ull;
constexpr std::uint64_t kFnvPrime  = 0x00000100000001B3ull;

std::uint64_t fnv1a(std::uint64_t h, const unsigned char* p, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * kFnvPrime;
  return h;
}

const char* transform_name(AttributeTransform t) {
  switch (t) {
    case AttributeTransform::Identity: return "identity";
    case AttributeTransform::Log2p1: return "log2p1";
    case AttributeTransform::Nonzero: return "nonzero";
    case AttributeTransform::OneHot: return "onehot";
  }
  return "";
}

// The names of the v3 features in vector order, as the training pipeline
// names them (lib/features.py); the attribute features follow the item names.
constexpr const char* kV3QueryNames[kV3QueryDim] = {"log2_m",
                                                    "log2_n",
                                                    "log2_k",
                                                    "log2_b",
                                                    "ai_prob",
                                                    "log2_ai_prob",
                                                    "aspect_mn",
                                                    "aspect_mk",
                                                    "aspect_nk",
                                                    "is_pow2_m",
                                                    "is_pow2_n",
                                                    "is_pow2_k",
                                                    "trans_a",
                                                    "trans_b",
                                                    "dtype_a_id",
                                                    "dtype_b_id",
                                                    "dtype_c_id",
                                                    "dtype_d_id",
                                                    "dtype_mi_id",
                                                    "bpe_a",
                                                    "bpe_b",
                                                    "m_mod_256",
                                                    "m_mod_128",
                                                    "m_mod_64",
                                                    "m_mod_32",
                                                    "m_mod_16",
                                                    "m_mod_8",
                                                    "n_mod_256",
                                                    "n_mod_128",
                                                    "n_mod_64",
                                                    "n_mod_32",
                                                    "n_mod_16",
                                                    "n_mod_8",
                                                    "m_align_64",
                                                    "m_align_128",
                                                    "m_align_256",
                                                    "n_align_64",
                                                    "n_align_128",
                                                    "n_align_256",
                                                    "log2_enum_tiles_32",
                                                    "log2_enum_tiles_64",
                                                    "log2_enum_tiles_128",
                                                    "log2_enum_tiles_256",
                                                    "enum_coverage_32",
                                                    "enum_coverage_64",
                                                    "enum_coverage_128",
                                                    "enum_coverage_256",
                                                    "log2_enum_k_iters_32",
                                                    "log2_enum_k_iters_64",
                                                    "log2_enum_k_iters_128",
                                                    "log2_enum_k_iters_256",
                                                    "log2_enum_waves_128",
                                                    "log2_enum_waves_256",
                                                    "enum_wave_eff_128",
                                                    "enum_wave_eff_256",
                                                    "cu_fraction",
                                                    "schedule_default",
                                                    "schedule_dynamic",
                                                    "schedule_auto"};
constexpr const char* kV3ItemNames[kV3ItemDim]   = {"log2_mt_m",
                                                    "log2_mt_n",
                                                    "log2_mt_k",
                                                    "log2_mi_m",
                                                    "log2_mi_n",
                                                    "log2_mi_k",
                                                    "nta_norm",
                                                    "ntb_norm"};
constexpr const char* kV3InterNames[kV3InterDim] = {"log2_num_tiles",
                                                    "log2_num_tiles_total",
                                                    "log2_k_iters",
                                                    "log2_waves",
                                                    "wave_eff",
                                                    "log2_rho",
                                                    "log2_batch_tiles_ratio",
                                                    "util_out",
                                                    "util_3d",
                                                    "tile_coverage",
                                                    "log2_lds_bytes",
                                                    "lds_ratio",
                                                    "l2_fit_ratio",
                                                    "l2_fit_ws",
                                                    "log2_bw_per_cu",
                                                    "log2_total_bytes",
                                                    "log2_total_flops",
                                                    "m_coverage",
                                                    "n_coverage",
                                                    "k_tail",
                                                    "k_aligned_128B",
                                                    "small_m_tile",
                                                    "small_n_tile",
                                                    "is_batched",
                                                    "log2_a_read_bytes",
                                                    "log2_b_read_bytes",
                                                    "log2_compute_density",
                                                    "log2_total_work",
                                                    "batch_k_iters_per_tile",
                                                    "batch_wave_eff",
                                                    "ai_tile",
                                                    "l_mi_cycles",
                                                    "log2_l_mt_cycles",
                                                    "active_cus_norm",
                                                    "is_dynamic",
                                                    "log2_grid",
                                                    "log2_grid_per_cu",
                                                    "log2_wg_waves",
                                                    "wg_wave_eff",
                                                    "log2_tiles_per_wg",
                                                    "uneven_work",
                                                    "reduction_tree",
                                                    "reduction_parallel",
                                                    "reduction_atomic"};

// Fixed inputs whose v3 features feature_spec_hash covers, with fixed arch
// constants, launch plans and execution contexts: a change to how any feature
// is computed changes the hash.
// lib/mlrec.py holds the same cases.
struct GoldenProblem {
  std::size_t m, n, k, batch;
  Transpose a_transpose, b_transpose;
  DataType ab, cd, mi;  // A and B, C and D, MI input
};

struct GoldenKernel {
  std::size_t mt_m, mt_n, mt_k, mi_m, mi_n, mi_k;
  int cache_hints_a, cache_hints_b;
};

struct GoldenCase {
  GoldenProblem problem;
  GoldenKernel kernel;
  Hardware hardware;
  ExecutionContext context;
  LaunchPlan plan;
};

constexpr Transpose kT = Transpose::T, kN = Transpose::N;
constexpr DataType kBf16 = DataType::BFloat16, kF8 = DataType::Float8, kF16 = DataType::Half,
                   kF32 = DataType::Float, kXf32 = DataType::XFloat32;

constexpr GoldenCase kGoldenCases[] = {
    {{4096, 4096, 4096, 1, kT, kN, kBf16, kBf16, kBf16},
     {256, 256, 64, 16, 16, 32, 0, 0},
     {256, 163840, 4194304},
     {0, Schedule::Default},
     {}},
    {{777, 12345, 1000, 3, kN, kT, kF8, kBf16, kF8},
     {128, 64, 128, 16, 16, 128, 4, 0},
     {256, 163840, 4194304},
     {64, Schedule::Dynamic},
     {true, 192, PartialReduction::Parallel}},
    {{1, 9000, 16, 1, kN, kN, kF16, kF16, kF16},
     {16, 64, 32, 1, 1, 64, 0, 4},
     {80, 65536, 8388608},
     {0, Schedule::Auto},
     {true, 0, PartialReduction::None}},
    {{300, 200, 70000, 8, kT, kT, kF32, kF32, kXf32},
     {64, 256, 16, 16, 16, 4, 1, 1},
     {304, 65536, 4194304},
     {100, Schedule::Default},
     {false, 304, PartialReduction::Atomic}},
};

void attribute_feature_names(const AttributeFeature& f, std::vector<std::string>* out) {
  switch (f.transform) {
    case AttributeTransform::Identity: out->push_back("attr_" + f.name); break;
    case AttributeTransform::Log2p1: out->push_back("attr_log2p1_" + f.name); break;
    case AttributeTransform::Nonzero: out->push_back("attr_nz_" + f.name); break;
    case AttributeTransform::OneHot:
      for (std::int64_t v : f.values) out->push_back("attr_" + f.name + "_eq_" + std::to_string(v));
      break;
  }
}

void append_names(const char* tag, const std::vector<std::string>& names, std::string* text) {
  *text += tag;
  for (std::size_t i = 0; i < names.size(); ++i) *text += (i > 0 ? "," : "\t") + names[i];
  *text += "\n";
}

// The bits of `values` as comma-separated 8-digit lowercase hex.
void append_bits(const std::vector<float>& values, std::string* text) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::uint32_t bits;
    std::memcpy(&bits, &values[i], sizeof bits);
    char hex[16];
    std::snprintf(hex, sizeof hex, "%s%08x", i > 0 ? "," : "", static_cast<unsigned>(bits));
    *text += hex;
  }
}

}  // namespace

const FeatureCatalog* find_catalog(const std::string& hash) noexcept {
  count_log2s();
  for (const FeatureCatalog* c : kCatalogs)
    if (hash == c->hash) return c;
  return nullptr;
}

const FeatureCatalog& v2_catalog() noexcept { return kCatalogV2; }

const FeatureCatalog& v3_catalog() noexcept { return kCatalogV3; }

std::size_t attribute_feature_count(const AttributeFeature& f) noexcept {
  return f.transform == AttributeTransform::OneHot ? f.values.size() : 1;
}

// The text it hashes:
//   <catalog hash>\n
//   version\t3\n
//   query\t<name>,<name>,...\n, then item (with the attribute features) and
//     interaction the same way
//   per spec entry <name>\t<default>\t<transform>\n, with \t<v1>,<v2>,...
//     before the newline of a one-hot entry
//   per golden case golden\t<case>\t<bits>\n, the bits of its query, item
//     and interaction features, each as 8 lowercase hex digits, joined by ','
// Golden case c gives spec entry j the value values[(j + c) % count] when it
// is one-hot, else (7 * j + 3 * c) % 17 - 4.
std::string feature_spec_text(const std::vector<AttributeFeature>& spec) {
  std::string text = std::string(kCatalogV3.hash) + "\nversion\t3\n";
  std::vector<std::string> names(std::begin(kV3QueryNames), std::end(kV3QueryNames));
  append_names("query", names, &text);
  names.assign(std::begin(kV3ItemNames), std::end(kV3ItemNames));
  for (const AttributeFeature& f : spec) attribute_feature_names(f, &names);
  append_names("item", names, &text);
  names.assign(std::begin(kV3InterNames), std::end(kV3InterNames));
  append_names("interaction", names, &text);
  for (const AttributeFeature& f : spec) {
    text += f.name + "\t" + std::to_string(f.default_value) + "\t" + transform_name(f.transform);
    if (f.transform == AttributeTransform::OneHot) {
      text += "\t";
      for (std::size_t i = 0; i < f.values.size(); ++i)
        text += (i > 0 ? "," : "") + std::to_string(f.values[i]);
    }
    text += "\n";
  }

  Model golden;
  golden.format_version    = 3;
  golden.catalog           = &kCatalogV3;
  golden.parallel_mi_cu    = 4.0;
  golden.mi_default_cycles = 64.0;
  golden.mi_table          = {{16, 16, 32, static_cast<std::int32_t>(DataType::BFloat16), 16.0},
                              {16, 16, 128, static_cast<std::int32_t>(DataType::Float8), 16.0},
                              {16, 16, 4, static_cast<std::int32_t>(DataType::XFloat32), 32.0}};
  golden.item_attributes   = spec;
  for (std::size_t j = 0; j < spec.size(); ++j) {
    golden.item_attributes[j].read = j;
    golden.attribute_reads.push_back(spec[j].name);
    golden.attribute_defaults.push_back(spec[j].default_value);
  }
  golden.query_dim = kV3QueryDim;
  golden.item_dim  = kV3ItemDim;
  for (const AttributeFeature& f : spec) golden.item_dim += attribute_feature_count(f);
  golden.inter_dim = kV3InterDim;
  std::vector<std::int64_t> attrs(golden.attribute_reads.size());
  for (std::size_t c = 0; c < std::size(kGoldenCases); ++c) {
    const GoldenProblem& g = kGoldenCases[c].problem;
    const GoldenKernel& gk = kGoldenCases[c].kernel;
    Problem p;
    p.size        = {g.m, g.n, g.k};
    p.batch       = g.batch;
    p.a_transpose = g.a_transpose;
    p.b_transpose = g.b_transpose;
    p.a_dtype = p.b_dtype = g.ab;
    p.c_dtype = p.d_dtype = g.cd;
    p.mi_dtype            = g.mi;
    Config k;
    k.mt            = {gk.mt_m, gk.mt_n, gk.mt_k};
    k.mi            = {gk.mi_m, gk.mi_n, gk.mi_k};
    k.cache_hints_a = gk.cache_hints_a;
    k.cache_hints_b = gk.cache_hints_b;
    for (std::size_t j = 0; j < spec.size(); ++j) {
      const AttributeFeature& f = spec[j];
      const std::int64_t v      = f.transform != AttributeTransform::OneHot
                                      ? static_cast<std::int64_t>((7 * j + 3 * c) % 17) - 4
                                  : f.values.empty() ? f.default_value
                                                     : f.values[(j + c) % f.values.size()];
      k.attributes.push_back({f.name, v});
    }
    read_attributes(golden, k, attrs.data());
    const HwView hw = hw_view(golden, kGoldenCases[c].hardware, kGoldenCases[c].context);
    std::vector<float> features(golden.query_dim + golden.item_dim + golden.inter_dim);
    kCatalogV3.query(p, hw, features.data());
    kCatalogV3.item(golden, k, attrs.data(), features.data() + golden.query_dim);
    kCatalogV3.inter(golden,
                     p,
                     k,
                     attrs.data(),
                     hw,
                     kGoldenCases[c].plan,
                     features.data() + golden.query_dim + golden.item_dim);
    text += "golden\t" + std::to_string(c) + "\t";
    append_bits(features, &text);
    text += "\n";
  }
  return text;
}

std::uint64_t feature_spec_hash(const std::vector<AttributeFeature>& spec) {
  const std::string text = feature_spec_text(spec);
  return fnv1a(kFnvOffset, reinterpret_cast<const unsigned char*>(text.data()), text.size());
}

std::uint64_t view_key_hash(const std::uint64_t* key, std::size_t n) noexcept {
  std::uint64_t h = kFnvOffset;
  for (std::size_t i = 0; i < n; ++i) {
    unsigned char bytes[8];
    for (std::size_t b = 0; b < 8; ++b) bytes[b] = static_cast<unsigned char>(key[i] >> (8 * b));
    h = fnv1a(h, bytes, 8);
  }
  return h;
}

void read_attributes(const Model& model, const Config& c, std::int64_t* out) {
  const std::size_t n = model.attribute_reads.size();
  for (std::size_t j = 0; j < n; ++j) {
    out[j] = model.attribute_defaults[j];
    for (const Attribute& a : c.attributes)
      if (a.name == model.attribute_reads[j]) {
        out[j] = a.value;
        break;
      }
  }
}

bool is_kernel_feasible(const Problem& p,
                        const Config& c,
                        bool nt_a_available,
                        bool nt_b_available,
                        bool non_temporal_rule) {
  SoftRules soft;
  soft.non_temporal = non_temporal_rule;
  return is_kernel_feasible(p, c, nt_a_available, nt_b_available, soft);
}

bool is_kernel_feasible(const Problem& p,
                        const Config& c,
                        bool nt_a_available,
                        bool nt_b_available,
                        const SoftRules& soft) {
  return Feasibility(p, Hardware{}, nt_a_available, nt_b_available, soft)
      .rules_pass(kernel_shape(c));
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
